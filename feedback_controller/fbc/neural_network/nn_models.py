import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CustomLoss(nn.Module):
    def __init__(self, C):
        super(CustomLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.C = C

    def forward(self, action_predicted, action_ground_truth, x_des, x_t):
        loss_torques = self.criterion(action_predicted, action_ground_truth)
        
        x_des = x_des[:, :7]
        x_t = x_t[:, :7]
        # print(x_des.shape, x_t.shape)
        mse_latent = torch.mean((x_des - x_t).pow(2), dim=1) #(batch_size, 1)
        
        # exp = torch.exp(self.D * (t - self.E)) #(batch_size, 1)
        # scalar = exp
        scalar = 1.0

        scaled_mse_latent = scalar * mse_latent
        average_scaled_mse_latent = torch.mean(scaled_mse_latent) #(1, 1)

        return (loss_torques + (self.C * average_scaled_mse_latent)), loss_torques
    

class KLLoss(nn.Module):
    def __init__(self, C):
        super(KLLoss, self).__init__()
        self.torqs_criterion = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.C = C

    def forward(self, action_predicted, action_ground_truth, x_des, x_t):

        torques_loss = self.torqs_criterion(action_predicted, action_ground_truth)
        x_des_log = F.log_softmax(x_des, dim=1)
        x_t_dist = F.softmax(x_t, dim=1)

        latent_kl_loss = self.kl_loss(x_des_log, x_t_dist)

        return (torques_loss + (self.C * latent_kl_loss)), torques_loss
    

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        # super(Encoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0), #120x120 => 60x60
            nn.ReLU(True),
            nn.Conv2d(8, 32, kernel_size=2, stride=2, padding=0), #60x60 => 30x30
            nn.ReLU(True),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #30x30 => 30x30
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0), #30x30 => 15x15
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0), #15x15 => 7x7
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(7*7*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)

            # nn.Linear(15*15*16, encoded_space_dim),
            # nn.ReLU(True),
            # nn.Linear(256, encoded_space_dim)
        )
        
    def forward(self, x):
        # print(x.shape)
        x = self.encoder_cnn(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.encoder_lin(x)
        # print(x.shape)
        # print("-")
        return x
    

class TinyConvNet(nn.Module):
    def __init__(self, encoded_space_dim):
        super(TinyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                               kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 30 * 30, encoded_space_dim)  
        # Assuming input image size is 120x120 after pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc(x)
        return x
    

class AlexNetPT(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        # super(AlexNetPT, self).__init__()

        alexnet = models.alexnet(weights='DEFAULT')
        self.feature_extractor = alexnet.features
        print("featur extractor arch.:", self.feature_extractor)

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(256*5*9, 4),
            # nn.ReLU(True),
            # nn.Linear(1024, 128),
            # nn.ReLU(True),
            # nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        # print("Alexnet forward:")
        # print("input shape", x.shape)
        x = self.feature_extractor(x)
        # print("feature extractor shape", x.shape)
        x = self.flatten(x)
        # print("flatten shape", x.shape)
        x = self.encoder_lin(x)
        return x


class MLP_2L(nn.Module):
    def __init__(self, inp_dim, lat_dim_1, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp_dim, lat_dim_1),
            # nn.BatchNorm1d(lat_dim_1),  # Batch Normalization added
            nn.ReLU(True),
            nn.Linear(lat_dim_1, out_dim),
        )

    def forward(self, x):
        x = self.linear(x)
        return x
    

class MLP_3L(nn.Module):
    def __init__(self, inp_dim, lat_dim_1, lat_dim_2, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp_dim, lat_dim_1),
            # nn.BatchNorm1d(lat_dim_1),  # Batch Normalization added
            nn.ReLU(True),
            nn.Linear(lat_dim_1, lat_dim_2),
            # nn.BatchNorm1d(lat_dim_2),  # Batch Normalization added
            nn.ReLU(True),
            nn.Linear(lat_dim_2, out_dim)
        )

    def forward(self, x):
        x = self.linear(x)
        return x
        
    
class SingleImageEncoder(nn.Module):
    """CNN encoder for a single RGB image (front camera at t=0)."""
    def __init__(self, cnn_latent):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(128, cnn_latent)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class GeneralModel(nn.Module):
    """
    When use_image=True:
        - target_repr: single image at t=0 (front camera) -> CNN -> cnn_latent
        - Concatenate touch_history (one-hot, onehot_dim) after CNN
        - Map to state space (x_des)
        - state: robot joints + velocities (encoded_space_dim) used directly
        - diff = x_des - state
        - Controller predicts action from diff

    When use_image=False:
        - target_repr: coordinates + one-hot -> MLP encoder -> x_des
        - state: robot joints + velocities
        - diff = x_des - state
        - Controller predicts action from diff
    """
    def __init__(self, encoded_space_dim, target_dim, action_dim,
                 enc_hid, cont_hid, use_image, cnn_latent=128, onehot_dim=4):
        super().__init__()
        self.use_image = use_image

        if use_image:
            # CNN encoder for single target image (3 channels RGB)
            self.cnn_encoder = SingleImageEncoder(cnn_latent)
            # Map CNN latent + one_hot to state space
            combined_dim = cnn_latent + onehot_dim
            self.to_state_space = MLP_2L(combined_dim, 2 * combined_dim, encoded_space_dim)
        else:
            self.enc1 = MLP_2L(target_dim, enc_hid, encoded_space_dim)

        # Controller (shared for both modes)
        self.mlp_controller = MLP_3L(encoded_space_dim, cont_hid, cont_hid, action_dim)
        self.mlp_controller.linear[-1].bias.data.fill_(0.0)

    def forward(self, target_repr, state, touch_history):
        x = state

        if self.use_image:
            # target_repr is an image at t=0
            cnn_out = self.cnn_encoder(target_repr)
            # Concatenate with touch history (one-hot)
            combined = torch.cat((cnn_out, touch_history), dim=1)
            # Map to state space (x_des)
            x_des = self.to_state_space(combined)
        else:
            # Concatenate coords and touch_history for encoder input
            target_full = torch.cat((target_repr, touch_history), dim=1)
            x_des = self.enc1(target_full)

        diff = x_des - x

        acts_pred = self.mlp_controller(diff)
        # acts_pred = F.tanh(acts_pred)

        return acts_pred, x_des, diff
    

class MLPBaseline(nn.Module):
    def __init__(self, inp_dim, lin_hid, lin_out, out_dim):
        super().__init__()
        # print("########", inp_dim)
        self.linear_2l = MLP_2L(inp_dim, lin_hid, lin_out)
        # self.linear_2l2 = MLP_2L(96*2, 2*2*inp_dim, out_dim)
        self.linear = nn.Sequential(
            # nn.Linear(int(270*(0.75)+10), out_dim)
            nn.Linear(lin_out, out_dim)
        )
        self.linear[-1].bias.data.fill_(0.0)

    def forward(self, x):
        act_preds = self.linear_2l(x)
        # act_preds = self.linear_2l2(F.relu(act_preds))
        act_preds = self.linear(F.relu(act_preds))

        act_preds = F.tanh(act_preds)
        return act_preds


class ImageStackEncoder(nn.Module):
    """CNN encoder for a stack of consecutive RGB images."""
    def __init__(self, num_images, encoded_dim):
        super().__init__()
        in_channels = num_images * 3
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(128, encoded_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class TwoStreamBaseline(nn.Module):
    def __init__(self, target_dim, mlp_hidden_1, mlp_hidden_2, mlp_latent,
                 num_images, cnn_latent,
                 decoder_hidden_1, decoder_hidden_2, action_dim):
        super().__init__()
        self.mlp_encoder = MLP_3L(target_dim, mlp_hidden_1, mlp_hidden_2, mlp_latent)
        # Separate encoder for each camera view
        self.cnn_encoder_front = ImageStackEncoder(num_images, cnn_latent)
        self.cnn_encoder_side = ImageStackEncoder(num_images, cnn_latent)
        # Decoder: mlp_latent + 2 * cnn_latent (one per camera) -> action
        self.decoder = MLP_3L(mlp_latent + 2 * cnn_latent,
                              decoder_hidden_1, decoder_hidden_2, action_dim)
        self.decoder.linear[-1].bias.data.fill_(0.0)

    def forward(self, target_repr, image_stack_front, image_stack_side):
        mlp_out = self.mlp_encoder(target_repr)
        cnn_out_front = self.cnn_encoder_front(image_stack_front)
        cnn_out_side = self.cnn_encoder_side(image_stack_side)
        combined = torch.cat((mlp_out, cnn_out_front, cnn_out_side), dim=1)
        action_pred = self.decoder(combined)
        return action_pred