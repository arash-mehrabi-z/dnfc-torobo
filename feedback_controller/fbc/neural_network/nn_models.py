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
        
    
class GeneralModel(nn.Module):
    """
    Two-stream encoder model:
    - Stream 1: Encodes target_repr (goal representation)
    - Stream 2: Encodes ee_repr (N consecutive end-effector poses)
    - Difference between encoded streams → controller → action
    """
    def __init__(self, encoded_space_dim, target_dim, ee_dim, action_dim,
                 enc_hid, cont_hid, use_image):
        super().__init__()
        if use_image:
            self.target_enc = AlexNetPT(encoded_space_dim)
        else:
            # Target encoder: target_dim (13) → enc_hid → encoded_space_dim
            self.target_enc = MLP_3L(target_dim, enc_hid, enc_hid // 2, encoded_space_dim)

        # EE pose encoder: ee_dim (6*N, e.g., 24 for N=4) → enc_hid → encoded_space_dim
        self.ee_enc = MLP_3L(ee_dim, enc_hid, enc_hid // 2, encoded_space_dim)

        # Controller/decoder: takes difference and outputs action
        self.controller = MLP_3L(encoded_space_dim, cont_hid, cont_hid // 2, action_dim)
        self.controller.linear[-1].bias.data.fill_(0.0)

    def forward(self, target_repr, ee_repr):
        # Encode target (desired state representation)
        x_des = self.target_enc(target_repr)  # (batch_size, encoded_space_dim)

        # Encode current EE pose history
        x_curr = self.ee_enc(ee_repr)  # (batch_size, encoded_space_dim)

        # Compute difference: where we want to be - where we are
        diff = x_des - x_curr

        # Decode to action
        acts_pred = self.controller(diff)
        acts_pred = F.tanh(acts_pred)

        return acts_pred, x_des, x_curr, diff
    

class MLPBaseline(nn.Module):
    """
    Similar to GeneralModel but without encoding the state:
    - Stream 1: Encodes target_repr (goal representation) -> x_des
    - Stream 2: Uses joint_state directly (no encoding) as x_curr
    - Difference between x_des and x_curr → controller → action
    """
    def __init__(self, target_dim, state_dim, action_dim, enc_hid, cont_hid, use_image=False):
        super().__init__()
        if use_image:
            self.target_enc = AlexNetPT(state_dim)  # Output matches state_dim
        else:
            # Target encoder: target_dim → enc_hid → state_dim
            self.target_enc = MLP_3L(target_dim, enc_hid, enc_hid // 2, state_dim)

        # Controller/decoder: takes difference (state_dim) and outputs action
        self.controller = MLP_3L(state_dim, cont_hid, cont_hid // 2, action_dim)
        self.controller.linear[-1].bias.data.fill_(0.0)

    def forward(self, target_repr, joint_state):
        # Encode target (desired state representation)
        x_des = self.target_enc(target_repr)  # (batch_size, state_dim)

        # Use joint state directly as current state representation
        x_curr = joint_state  # (batch_size, state_dim)

        # Compute difference: where we want to be - where we are
        diff = x_des - x_curr

        # Decode to action
        acts_pred = self.controller(diff)
        acts_pred = F.tanh(acts_pred)

        return acts_pred, x_des, x_curr, diff