random_seed = 1

import os
import time
import csv
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


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []

    for i, batch_data in enumerate(dataloader):
        batch_target_repr = batch_data[2].to(device).float()
        encoded_data = encoder(batch_target_repr)
        decoded_data = decoder(encoded_data)
        loss = loss_fn(decoded_data, batch_target_repr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
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
num_trajectories_in_dataset = 10000

# Training:
use_image = True
num_epochs = 300 + 1
batch_size = 32
learning_rate = 3e-4
validation_interval = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = "10K_cartesian_w_step"
weights_save_root = current_dir_path + "/weights" + f"/{dataset_name}"

# Model:
encoded_space_dim = 4


### Creating objects ###
# General:
if not os.path.exists(weights_save_root):
   os.makedirs(weights_save_root)

# Dataset:
dataset = TrajectoryDataset(current_dir_path + f'/data/{dataset_name}/0', 
                            '/trajectories_normalized.npy', use_image)
train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Loss:
loss_fn = torch.nn.MSELoss()
losses_records = {'train_loss':[], 'val_loss':[]}

# Model:
encoder = Encoder(encoded_space_dim=encoded_space_dim)
decoder = Decoder(encoded_space_dim=encoded_space_dim)
encoder.to(device)
decoder.to(device)

# Optimizer:
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)

with open(weights_save_root + "/loss.csv", 'w') as f:
            writer = csv.writer(f)
            row = ["n", "train_loss", "val_loss"]
            writer.writerow(row)

# Printing hyperparameter:
print("device", device)

# Training:
for n_epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train_epoch(encoder, decoder, device, train_dataloader, loss_fn, optimizer)
    val_loss = test_epoch(encoder, decoder, device, val_dataloader, loss_fn)
    losses_records['train_loss'].append(train_loss)
    losses_records['val_loss'].append(val_loss)

    end_time = time.time()

    if n_epoch % validation_interval == 0:
        log_loss(n_epoch, train_loss, val_loss)
        
        epoch_time_minutes = end_time - start_time
        print(f"Epoch [{n_epoch}/{num_epochs}] - Time taken: {epoch_time_minutes:.3f} seconds")

        torch.save(encoder.state_dict(), weights_save_root+f"/enc_{n_epoch}.pth")
        torch.save(decoder.state_dict(), weights_save_root+f"/dec_{n_epoch}.pth")

        visualize_losses(losses_records['train_loss'], losses_records['val_loss'], 
                         f"loss_ae_{n_epoch}.png", "AE Loss", fig_1)
    # plot_ae_outputs(encoder, decoder, n=10)