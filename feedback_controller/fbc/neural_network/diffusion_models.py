import torch
import torch.nn as nn
import math
import logging
from typing import Union, Optional, Tuple
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines number of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())
        ))

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


class DiffusionPolicyModel(nn.Module):
    """
    Wrapper class for diffusion policy that integrates U-Net with scheduler.
    Provides both training and inference interfaces.
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 obs_horizon=2,
                 pred_horizon=16,
                 action_horizon=8,
                 num_diffusion_iters=100,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8,
                 diffusion_step_embed_dim=256):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        # Create the noise prediction network
        global_cond_dim = obs_dim * obs_horizon
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups
        )

        # Create the noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def forward(self, obs_seq, noisy_actions=None, timesteps=None):
        """
        Forward pass for training or inference.

        Training mode (noisy_actions and timesteps provided):
            obs_seq: (B, obs_horizon, obs_dim)
            noisy_actions: (B, pred_horizon, action_dim)
            timesteps: (B,) or scalar
            returns: predicted noise (B, pred_horizon, action_dim)

        Inference mode (noisy_actions and timesteps not provided):
            obs_seq: (B, obs_horizon, obs_dim)
            returns: denoised actions (B, pred_horizon, action_dim)
        """
        # Flatten observation for FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        if noisy_actions is not None and timesteps is not None:
            # Training mode: predict noise
            noise_pred = self.noise_pred_net(
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=obs_cond
            )
            return noise_pred
        else:
            # Inference mode: run full denoising
            return self.get_action(obs_seq)

    def get_action(self, obs_seq):
        """
        Inference method that runs full denoising loop.

        Args:
            obs_seq: (B, obs_horizon, obs_dim) observation sequence

        Returns:
            action_seq: (B, pred_horizon, action_dim) predicted action sequence
        """
        B = obs_seq.shape[0]
        device = obs_seq.device

        # Flatten observation for FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)

        # Initialize action from Gaussian noise
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=device
        )
        naction = noisy_action

        # Init scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        # Iterative denoising
        for k in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # Inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction


class TransformerForDiffusion(nn.Module):
    """
    Transformer-based noise prediction network for diffusion policy.
    Implements encoder-decoder architecture with cross-attention conditioning.

    Based on the diffusion_policy implementation but simplified for direct integration.
    """
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 8,
            n_head: int = 4,
            n_emb: int = 256,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.1,
            causal_attn: bool = True,
            time_as_cond: bool = True,
            obs_as_cond: bool = True,
            n_cond_layers: int = 0,
            ff_mult: int = 4
        ) -> None:
        """
        Args:
            input_dim: Dimension of input (action dimension)
            output_dim: Dimension of output (same as input for noise prediction)
            horizon: Prediction horizon (sequence length)
            n_obs_steps: Number of observation steps for conditioning
            cond_dim: Dimension of conditioning (obs_dim if obs_as_cond=True)
            n_layer: Number of transformer decoder layers
            n_head: Number of attention heads
            n_emb: Embedding dimension
            p_drop_emb: Dropout probability for embeddings
            p_drop_attn: Dropout probability for attention
            causal_attn: Whether to use causal attention mask
            time_as_cond: Whether to use diffusion timestep as conditioning token
            obs_as_cond: Whether to condition on observations via cross-attention
            n_cond_layers: Number of transformer encoder layers for condition (0 = MLP)
            ff_mult: Feedforward expansion multiplier (default 4, use 1-2 for smaller models)
        """
        super().__init__()

        # Compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1  # Start with timestep token
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond, "obs_as_cond requires time_as_cond=True"
            T_cond += n_obs_steps

        # Input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Condition encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=ff_mult * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                # Simple MLP for condition encoding
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, ff_mult * n_emb),
                    nn.Mish(),
                    nn.Linear(ff_mult * n_emb, n_emb)
                )

            # Decoder with cross-attention to condition
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=ff_mult * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # Encoder only BERT-style
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=ff_mult * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # Attention mask for causal attention
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1)  # Add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # Decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # Store constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        self.n_emb = n_emb

        # Initialize weights
        self.apply(self._init_weights)
        logger.info(
            "TransformerForDiffusion: number of parameters: %e",
            sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name, None)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name, None)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            # Don't raise error for other module types
            pass

    def forward(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass for noise prediction.

        Args:
            sample: (B, T, input_dim) noisy action sequence
            timestep: (B,) or int, diffusion timestep
            cond: (B, T_obs, cond_dim) observation conditioning

        Returns:
            output: (B, T, output_dim) predicted noise
        """
        # 1. Process timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # Broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # 2. Process input
        input_emb = self.input_emb(sample)  # (B, T, n_emb)

        if self.encoder_only:
            # BERT-style: concatenate time and input
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]  # Remove time token
        else:
            # Encoder-decoder with cross-attention
            # Build condition embeddings
            cond_embeddings = time_emb
            if self.obs_as_cond and cond is not None:
                cond_obs_emb = self.cond_obs_emb(cond)  # (B, T_obs, n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)

            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x  # (B, T_cond, n_emb)

            # Decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )

        # Head
        x = self.ln_f(x)
        x = self.head(x)  # (B, T, output_dim)
        return x

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        Separate parameters into decay/no-decay groups for AdamW.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Special case position embeddings
        no_decay.add("pos_emb")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        # Handle parameters not in either set
        remaining = param_dict.keys() - union_params
        no_decay.update(remaining)

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


class DiffusionTransformerPolicyModel(nn.Module):
    """
    Wrapper class for transformer-based diffusion policy.
    Integrates TransformerForDiffusion with DDPM scheduler.
    Provides both training and inference interfaces.
    """
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 obs_horizon: int = 2,
                 pred_horizon: int = 16,
                 action_horizon: int = 8,
                 num_diffusion_iters: int = 100,
                 n_layer: int = 8,
                 n_head: int = 4,
                 n_emb: int = 256,
                 p_drop_emb: float = 0.0,
                 p_drop_attn: float = 0.1,
                 causal_attn: bool = True,
                 n_cond_layers: int = 0,
                 ff_mult: int = 4):
        """
        Args:
            obs_dim: Observation dimension (state + target + onehot)
            action_dim: Action dimension (joint velocities)
            obs_horizon: Number of past observations to condition on
            pred_horizon: Number of future actions to predict
            action_horizon: Number of actions to execute before replanning
            num_diffusion_iters: Number of diffusion denoising steps
            n_layer: Number of transformer decoder layers
            n_head: Number of attention heads
            n_emb: Embedding dimension
            p_drop_emb: Dropout for embeddings
            p_drop_attn: Dropout for attention
            causal_attn: Whether to use causal attention
            n_cond_layers: Number of encoder layers for condition processing
            ff_mult: Feedforward expansion multiplier (default 4, use 1-2 for smaller models)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        # Create the transformer noise prediction network
        self.noise_pred_net = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=pred_horizon,
            n_obs_steps=obs_horizon,
            cond_dim=obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
            ff_mult=ff_mult
        )

        # Create the noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        print(f"DiffusionTransformerPolicyModel: number of parameters: {sum(p.numel() for p in self.parameters()):e}")

    def forward(self, obs_seq, noisy_actions=None, timesteps=None):
        """
        Forward pass for training or inference.

        Training mode (noisy_actions and timesteps provided):
            obs_seq: (B, obs_horizon, obs_dim)
            noisy_actions: (B, pred_horizon, action_dim)
            timesteps: (B,) or scalar
            returns: predicted noise (B, pred_horizon, action_dim)

        Inference mode (noisy_actions and timesteps not provided):
            obs_seq: (B, obs_horizon, obs_dim)
            returns: denoised actions (B, pred_horizon, action_dim)
        """
        if noisy_actions is not None and timesteps is not None:
            # Training mode: predict noise with observation conditioning
            noise_pred = self.noise_pred_net(
                sample=noisy_actions,
                timestep=timesteps,
                cond=obs_seq  # (B, obs_horizon, obs_dim)
            )
            return noise_pred
        else:
            # Inference mode: run full denoising
            return self.get_action(obs_seq)

    def get_action(self, obs_seq):
        """
        Inference method that runs full denoising loop.

        Args:
            obs_seq: (B, obs_horizon, obs_dim) observation sequence

        Returns:
            action_seq: (B, pred_horizon, action_dim) predicted action sequence
        """
        B = obs_seq.shape[0]
        device = obs_seq.device

        # Initialize action from Gaussian noise
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=device
        )
        naction = noisy_action

        # Init scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        # Iterative denoising
        for k in self.noise_scheduler.timesteps:
            # Predict noise with observation conditioning
            noise_pred = self.noise_pred_net(
                sample=naction,
                timestep=k,
                cond=obs_seq
            )

            # Inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction
