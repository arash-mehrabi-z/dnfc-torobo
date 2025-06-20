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


class MLP_3L(nn.Module):
    def __init__(self, inp_dim, lat_dim_1, lat_dim_2, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp_dim, lat_dim_1),
            nn.ReLU(True),
            nn.Linear(lat_dim_1, lat_dim_2),
            nn.ReLU(True),
            nn.Linear(lat_dim_2, out_dim)
        )

    def forward(self, x):
        x = self.linear(x)
        return x
        
    
class GeneralModel(nn.Module):
    def __init__(self, encoded_space_dim, target_dim, action_dim, use_image):
        super().__init__()
        if use_image:
            # self.enc = Encoder(encoded_space_dim)
            self.enc = AlexNetPT(encoded_space_dim)
        else:
            self.enc = MLP_3L(target_dim, 32, 32, encoded_space_dim)

        self.mlp_controller = MLP_3L(encoded_space_dim, 48, 48, action_dim)
        # self.linear = nn.Sequential(
        #     nn.Linear(256, action_dim)
        # )
        self.mlp_controller.linear[4].bias.data.fill_(0.0)

    def forward(self, target_repr, state):
        x = state
        # x_des = self.alexnet(img_tensor)
        x_des = self.enc(target_repr) # (batch_size, encoded_space_dim)

        diff = x_des - x
        inp_lat_mlp = diff
        # inp_lat_mlp = torch.cat((diff, state), dim=1)

        acts_pred = self.mlp_controller(inp_lat_mlp)
        # acts_pred = self.linear(F.relu(acts_pred))
        return acts_pred, x_des, diff
    

class MLPBaseline(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        # self.linear_3l = MLP_3L(inp_dim, 64, 248, 512)
        # self.linear_3l_2 = MLP_3L(512, 256, 64, out_dim)
        self.linear_3l = MLP_3L(inp_dim, 52, 64, 128)
        self.linear_3l_2 = MLP_3L(128, 64, 52, out_dim)
        self.linear_3l_2.linear[4].bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.linear_3l(x))
        act_preds = self.linear_3l_2(x)
        return act_preds