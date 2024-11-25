random_seed = 1

import os
import time
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader

np.random.seed(random_seed)
torch.manual_seed(random_seed)


class TrajectoryDataset(Dataset):
    def __init__(self, data_root_dir, file_adr, num_trajectories_in_dataset, use_image=False):
        self.trajectories = np.load(data_root_dir + file_adr)
        self.data_root_dir = data_root_dir
        self.num_trajectories_in_dataset = num_trajectories_in_dataset
        self.use_image = use_image
        if self.use_image:
            self.transform = transforms.Compose([
                transforms.Resize([60, 60]),
                # transforms.RandomHorizontalFlip(), # Flip the data horizontally
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])

    def __len__(self):
        return self.num_trajectories_in_dataset * 50
        # return 70*49

    def __getitem__(self, idx):
        traj_no = idx // 50
        step_no = idx % 50

        step = self.trajectories[traj_no, step_no, 0]
        state = self.trajectories[traj_no, step_no, 1:5]
        action = self.trajectories[traj_no, step_no, 7:]

        if self.use_image:
            image = Image.open(self.data_root_dir + 
                            f"/scene_images/goal_{traj_no}.png")
            image = self.transform(image)
            target_repr = image
        else:
            target_pos = self.trajectories[traj_no, step_no, 5:7]
            target_repr = target_pos

        return step, state, target_repr, action
    

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        # super(Encoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0), #60x60 => 30x30
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0), #30x30 => 15x15
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), #15x15 => 7x7
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(7*7*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
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
    

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 7*7*32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 7, 7))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, output_padding=1), #7x7 => 15x15
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, output_padding=0), #15x15 => 30x30
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2, output_padding=0) #30x30 => 60x60
        )
        
    def forward(self, x):
        # print(x.shape)
        x = self.decoder_lin(x)
        # print(x.shape)
        x = self.unflatten(x)
        # print(x.shape)
        x = self.decoder_conv(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        # print(x.shape)
        # print("---")
        return x


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    global test_results_save_root

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        val_loss = []

        for i, batch_data in enumerate(dataloader):
            batch_target_repr = batch_data[2].to(device).float()
            encoded_data = encoder(batch_target_repr)
            decoded_data = decoder(encoded_data)
            loss = loss_fn(decoded_data, batch_target_repr)
            val_loss.append(loss.detach().cpu().numpy())

            # Visualize original and reconstructed images
            for j in range(len(batch_target_repr)):
                original_image = batch_target_repr[j].cpu().numpy()
                reconstructed_image = decoded_data[j].cpu().numpy()

                # Reshape images if needed (e.g., for channels)
                original_image = original_image.transpose(1, 2, 0)  # Example for channels last
                reconstructed_image = reconstructed_image.transpose(1, 2, 0)  # Example for channels last

                # Plot the images
                plt.figure(figsize=(8, 4))
                plt.title(f"Original {i} {j}")
                plt.imshow(original_image, cmap='gray')  # Modify cmap if needed
                plt.axis('off')
                plt.savefig(test_results_save_root + f"/reconstructs/org_{i}_{j}.png")
                plt.close()

                # Plot the images
                plt.figure(figsize=(8, 4))
                plt.title(f"Reconstruct {i} {j}")
                plt.imshow(reconstructed_image, cmap='gray')  # Modify cmap if needed
                plt.axis('off')
                plt.savefig(test_results_save_root + f"/reconstructs/rec_{i}_{j}.png")
                plt.close()

    return np.mean(val_loss)


def visualize_losses(train_losses, val_losses, file_name, title, fig):
    global weights_save_root

    plt.figure(fig)
    plt.clf()
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    # Update the plot
    plt.draw()
    plt.pause(0.001)  # Pause for a short time to update the plot
    plt.savefig(weights_save_root + f"/{file_name}")
    # plt.close()


def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()


def log_loss(n, train_loss, val_loss):
    global weights_save_root

    with open(weights_save_root + "/loss.csv", 'a') as f:
        writer = csv.writer(f)
        row = [n, train_loss, val_loss]
        writer.writerow(row)

    print("===")
    print(f"Epoch: {n}:")
    print(f"train_loss: {train_loss}")
    print(f"val_loss torques: {val_loss}")


# General:
fig_1 = plt.figure(figsize=(12, 8))
current_dir_path = os.path.dirname(__file__)
num_trajectories_in_dataset = 100
trained_dataset_name = "10K_cartesian_w_step"
test_dataset_name = "100_cartesian_w_step"
use_model_saved_at_epoch = 40

# Training:
use_image = True
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_save_root = current_dir_path + "/weights" + f"/{trained_dataset_name}"
test_results_save_root = current_dir_path + "/test_results" + f"/{test_dataset_name}" + "_ae"

# Model:
encoded_space_dim = 4


### Creating objects ###
# General:
if not os.path.exists(test_results_save_root + f"/reconstructs"):
   os.makedirs(test_results_save_root + f"/reconstructs")

# Dataset:
test_dataset = TrajectoryDataset(current_dir_path + f'/data/{test_dataset_name}/0', 
                            '/trajectories_normalized.npy', num_trajectories_in_dataset, use_image)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Loss:
loss_fn = torch.nn.MSELoss()
losses_records = {'train_loss':[], 'val_loss':[]}

# Model:
encoder = Encoder(encoded_space_dim=encoded_space_dim)
e = encoder.to(device)
encoder.load_state_dict(torch.load(weights_save_root + f"/enc_{use_model_saved_at_epoch}.pth"))
encoder.eval()
print("Encoder:", sum(p.numel() for p in e.parameters())/1e6, 'M parameters')

decoder = Decoder(encoded_space_dim=encoded_space_dim)
d = decoder.to(device)
decoder.load_state_dict(torch.load(weights_save_root + f"/dec_{use_model_saved_at_epoch}.pth"))
decoder.eval()
print("Decoder:", sum(p.numel() for p in d.parameters())/1e6, 'M parameters')

with open(test_results_save_root + "/loss.csv", 'w') as f:
            writer = csv.writer(f)
            row = ["n", "train_loss", "val_loss"]
            writer.writerow(row)

# Printing hyperparameter:
print("device", device)

# Testing:
start_time = time.time()
val_loss = test_epoch(encoder, decoder, device, test_dataloader, loss_fn)
end_time = time.time()
log_loss(0, 0, val_loss)
epoch_time_minutes = end_time - start_time
print(f"Time taken: {epoch_time_minutes:.3f} seconds")