random_seed = 1

import os
import time
import csv
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Subset

from nn_models import GeneralModel
from nn_models import MLPBaseline
from nn_models import TwoStreamBaseline
from nn_models import CustomLoss, KLLoss

from config import Config

# np.random.seed(random_seed)
# torch.manual_seed(random_seed)


class TrajectoryDataset(Dataset):
    def __init__(self, ds_root_dir, file_name, 
                 joint_dim, target_dim, use_image=False):
        self.joint_dim = joint_dim
        self.state_dim = 2 * joint_dim
        self.target_dim = target_dim
        self.ds_root_dir = ds_root_dir
        self.use_image = use_image

        data_file_path = os.path.join(ds_root_dir, file_name)
        self.trajectories = np.load(data_file_path)#[0:4]
        print("loaded data from", data_file_path)
        print("trajectories.shape", self.trajectories.shape)

        self.num_trajectories_in_dataset = self.trajectories.shape[0]
        self.num_steps = self.trajectories.shape[1]

        if self.use_image:
            self.transform = transforms.Compose([
                # transforms.Resize([108, 171]),
                # transforms.RandomHorizontalFlip(), # Flip the data horizontally
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])

    def __len__(self):
        return self.num_trajectories_in_dataset*self.num_steps

    def __getitem__(self, idx):
        traj_no = idx // self.num_steps
        step_no = idx % self.num_steps

        step = self.trajectories[traj_no, step_no, 0]
        state = self.trajectories[traj_no, step_no, 1:1+self.state_dim]

        action = self.trajectories[traj_no, step_no, -self.joint_dim:]

        if self.use_image:
            img_path = os.path.join(self.ds_root_dir,
                                    f"scenes_cropped/eps_{traj_no}/crpd_step_{step_no}.png")
            image = Image.open(img_path)
            # image = read_image(img_path)
            image = self.transform(image)
            target_repr = image
        else:
            target_start_idx = 1 + self.state_dim
            target_end_idx = target_start_idx + self.target_dim
            target_repr = self.trajectories[traj_no, step_no, 
                                           target_start_idx:target_end_idx]
            target_repr = target_repr

        return step, state, target_repr, action


class TwoStreamDataset(Dataset):
    def __init__(self, ds_root_dir, file_name, joint_dim, target_dim,
                 num_history_images=4, image_size=(128, 128),
                 traj_indices=None):
        """
        Args:
            ds_root_dir: Root directory containing data files and images
            file_name: Name of the .npy data file
            joint_dim: Number of joints
            target_dim: Dimension of target representation
            num_history_images: Number of consecutive images to stack
            image_size: Resize images to this size
            traj_indices: Array mapping dataset trajectory index to original
                         trajectory folder number. If None, assumes 1:1 mapping.
        """
        self.joint_dim = joint_dim
        self.state_dim = 2 * joint_dim
        self.target_dim = target_dim
        self.ds_root_dir = ds_root_dir
        self.num_history_images = num_history_images

        data_file_path = os.path.join(ds_root_dir, file_name)
        self.trajectories = np.load(data_file_path)
        print("loaded data from", data_file_path)
        print("trajectories.shape", self.trajectories.shape)

        self.num_trajectories = self.trajectories.shape[0]
        self.num_steps = self.trajectories.shape[1]

        # Trajectory index mapping: maps dataset row index -> original folder number
        if traj_indices is not None:
            self.traj_indices = traj_indices
            print(f"Using trajectory mapping: {len(traj_indices)} trajectories")
            print(f"First 5 original traj indices: {traj_indices[:5]}")
        else:
            # Default: assume 1:1 mapping (traj 0 -> folder 000, etc.)
            self.traj_indices = np.arange(self.num_trajectories)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.img_dir = os.path.join(ds_root_dir, "triangle_images_fixed_cam")

    def __len__(self):
        return self.num_trajectories * self.num_steps

    def __getitem__(self, idx):
        traj_no = idx // self.num_steps
        step_no = idx % self.num_steps

        step = self.trajectories[traj_no, step_no, 0]
        state = self.trajectories[traj_no, step_no, 1:1+self.state_dim]

        target_start = 1 + self.state_dim
        target_end = target_start + self.target_dim
        target_repr = self.trajectories[traj_no, step_no, target_start:target_end]

        action = self.trajectories[traj_no, step_no, -self.joint_dim:]

        # Get the original trajectory folder number from the mapping
        original_traj_no = self.traj_indices[traj_no]

        # Load last num_history_images images from FRONT camera
        images_front = []
        for i in range(self.num_history_images - 1, -1, -1):
            img_step = max(0, step_no - i)
            img_path = os.path.join(
                self.img_dir,
                f"traj_{original_traj_no:03d}/step_{img_step:03d}.jpg")
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            images_front.append(image)

        image_stack_front = torch.cat(images_front, dim=0)  # (num_images * 3, H, W)

        # Load last num_history_images images from SIDE camera
        images_side = []
        for i in range(self.num_history_images - 1, -1, -1):
            img_step = max(0, step_no - i)
            img_path = os.path.join(
                self.img_dir,
                f"traj_side_{original_traj_no:03d}/step_{img_step:03d}.jpg")
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            images_side.append(image)

        image_stack_side = torch.cat(images_side, dim=0)  # (num_images * 3, H, W)

        return step, state, target_repr, image_stack_front, image_stack_side, action


def log_loss(n, val_loss_cutsom, val_loss_torques):
    global weights_storage_root_dir
    global train_loss_custom, train_loss_torques

    loss_file_path = os.path.join(weights_storage_root_dir, "loss.csv")
    with open(loss_file_path, 'a') as f:
        writer = csv.writer(f)
        row = [n, train_loss_custom, train_loss_torques, val_loss_cutsom, 
               val_loss_torques]
        writer.writerow(row)

    print("===")
    print(f"Epoch: {n}:")
    print(f"train_loss_custom: {train_loss_custom}")
    print(f"train_loss torques: {train_loss_torques}")
    print(f"val_loss_cutsom: {val_loss_cutsom}")
    print(f"val_loss_torques: {val_loss_torques}")


def run_test(n):
    global val_dataloader, model, criterion, weights_storage_root_dir
    global val_losses_custom, val_losses_torques
    global use_custom_loss, use_baseline, use_two_stream
    global train_info

    val_loss_cutsom = 0
    val_loss_torques = 0
    for i, batch_data in enumerate(val_dataloader):
        batch_step = batch_data[0].to(device).float()
        batch_state = batch_data[1].to(device).float()
        batch_target_repr = batch_data[2].to(device).float()

        if use_two_stream:
            batch_images_front = batch_data[3].to(device).float()
            batch_images_side = batch_data[4].to(device).float()
            batch_action = batch_data[5].to(device).float()
        else:
            batch_action = batch_data[3].to(device).float()

        model.eval()
        with torch.no_grad():
            if use_two_stream:
                batch_action_pred = model(batch_target_repr, batch_images_front, batch_images_side)
            elif use_baseline:
                nn_input = torch.cat((batch_target_repr, batch_state), dim=1)
                batch_action_pred = model(nn_input)
            else:
                batch_action_pred, batch_x_des, batch_diff = model(batch_target_repr,
                                                                   batch_state)
            
        if use_custom_loss:
            # loss_custom, loss_torques = criterion(batch_action_pred, batch_action, 
            #                                     batch_x_des, batch_state, 
            #                                     batch_step)
            loss_custom, loss_torques = criterion(batch_action_pred, batch_action, 
                                                batch_x_des, batch_state)
        else:
            loss_torques = criterion(batch_action_pred, batch_action)
            loss_custom = loss_torques

        val_loss_cutsom += loss_custom.item() #* batch_state.size(0)
        val_loss_torques += loss_torques.item() #* batch_state.size(0)

    val_loss_cutsom /= len(val_dataloader.dataset)
    val_losses_custom.append(val_loss_cutsom**0.5)
    val_loss_torques /= len(val_dataloader.dataset)
    val_losses_torques.append(val_loss_torques**0.5)

    act_abs_diff = torch.abs(batch_action_pred - batch_action)
    mean_abs_diff = torch.mean(act_abs_diff, dim=0)
    for k in range(7):
        train_info[f'mae_joint_{k+1}_val'].append(mean_abs_diff[k].item())

    log_loss(n, val_loss_cutsom, val_loss_torques)
    model.train()


def visualize_losses(train_losses, val_losses, n, file_name, title, fig):
    global validation_interval, weights_storage_root_dir
    plt.figure(fig)
    plt.clf()
    
    # Plot validation loss
    plt.plot([-1] + list(range(0, n+1, validation_interval)), np.log(val_losses), 
             label='Validation Loss')
    
    # Plot training loss
    plt.plot(list(range(len(train_losses))), np.log(train_losses), 
             label='Training Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    
    # # Set y-axis limit
    # plt.ylim([0, 0.01])
    
    # Update the plot
    plt.draw()
    plt.pause(0.001)  # Pause for a short time to update the plot
    
    # Save the plot
    loss_fig_path = os.path.join(weights_storage_root_dir, file_name)
    plt.savefig(loss_fig_path)


def visualize_losses_second(train_losses1, train_losses2,  val_losses1, val_losses2, n, file_name, title, fig):
    global validation_interval, weights_storage_root_dir
    plt.figure(fig)
    plt.clf()
    real_train_losses1=[]
    real_train_losses2=[]

    for i in range(0,n+1,50):
        real_train_losses1.append(train_losses1[i])
        real_train_losses2.append(train_losses2[i])

    
    # Plot validation loss
    plt.plot( list(range(0, n+1, validation_interval)), val_losses1[1:], 
             label='Validation Loss of Our Baseline')
    plt.plot( list(range(0, n+1, validation_interval)), val_losses2[1:], 
             label='Validation Loss of Model ')
    
    plt.plot( list(range(0, n+1, validation_interval)), real_train_losses1, 
             label='Training Loss of Our Baseline')
    plt.plot(list(range(0, n+1, validation_interval)), real_train_losses2, 
             label='Training Loss of Model ')
        

def create_csv_files(weights_storage_root_dir, selected_val_ind):
    # file_path = os.path.join(weights_storage_root_dir, "input_independent_baseline.csv")
    # with open(file_path, 'w') as f:
    #     writer = csv.writer(f)
    #     row = ["n"]
    #     for j in range(2):
    #         row.append(f"act_pred_{j}")
    #         row.append(f"act_pred_zero_{j}")
    #         row.append(f"act_tru_{j}")
    #     writer.writerow(row)

    loss_file_path = os.path.join(weights_storage_root_dir, "loss.csv")
    with open(loss_file_path, 'w') as f:
        writer = csv.writer(f)
        row = ["n", "train_loss_custom", "train_loss_torques", "val_loss_cutsom", 
            "val_loss_torques"]
        writer.writerow(row)

    # file_path = os.path.join(weights_storage_root_dir, "prediction_dynamics.csv")
    # with open(file_path, 'w') as f:
    #     writer = csv.writer(f)
    #     row = ["n"]
    #     for val_idx in selected_val_ind:
    #         for j in range(2):
    #             row.append(f"act_pred_{val_idx}_{j}")
    #             row.append(f"act_tru_{val_idx}_{j}")
    #     writer.writerow(row)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = Config()

# Custom Loss:
C = config.C

# General:
current_dir_path = os.path.dirname(__file__)
current_dir_path = Path(current_dir_path)
fbc_root_dir_path = current_dir_path.parent.absolute()
dataset_name = config.dataset_name
ds_root_dir = os.path.join(fbc_root_dir_path, 
                    f'neural_network/data/torobo/{dataset_name}')
ds_file_name = config.ds_file_name

# Model:
joints_num = config.joints_num
encoded_space_dim = config.state_dim
target_dim = config.coords_dim + config.onehot_dim
action_dim = joints_num

# Training:
use_baseline = False
use_image = False
use_two_stream = True
use_custom_loss = config.use_custom_loss
num_epochs = 12000 + 1 
batch_size = 128
learning_rate = 3e-4
validation_interval = 100
num_trains = 3
noise_std = 0.004

if use_baseline or use_two_stream:
    use_custom_loss = False

train_info = dict()
for i in range(7):
    train_info[f'mae_joint_{i+1}_val'] = []
    train_info[f'mae_joint_{i+1}_train'] = []

for model_complexity in ['high']: #'low', 'medium', 'high', 'xhigh']:
    enc_hid, cont_hid, lin_hid, lin_out = config.get_model_dims(model_complexity)
    
    for i_train in range(num_trains):
        fig_1 = plt.figure(figsize=(12.8, 9.6))
        fig_2 = plt.figure(figsize=(12.8, 9.6))

        model_name = config.get_model_name(use_baseline, use_custom_loss, use_image,
                                           use_two_stream)
        
        if use_two_stream:
            # Load trajectory mapping from .npz file
            mapping_file = os.path.join(ds_root_dir, f'traj_mapping_{config.ds_ratio}.npz')
            mapping = np.load(mapping_file)
            train_traj_indices = mapping['train_traj_indices']
            print(f"Loaded trajectory mapping with {len(train_traj_indices)} trajectories")
            print(f"First five indices are:", train_traj_indices[:5])

            dataset = TwoStreamDataset(ds_root_dir, ds_file_name,
                                       joints_num, target_dim,
                                       num_history_images=config.num_history_images,
                                       image_size=config.image_size,
                                       traj_indices=train_traj_indices)
        else:
            dataset = TrajectoryDataset(ds_root_dir, ds_file_name,
                                        joints_num, target_dim, use_image)
        if use_two_stream:
            train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
        else:
            # train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
            split_file_path = os.path.join(ds_root_dir, config.train_val_file)
            split = torch.load(split_file_path)
            train_indices = split['train_indices']
            val_indices = split['val_indices']
            train_set = Subset(dataset, train_indices)
            val_set = Subset(dataset, val_indices)
            print(f"First five train_indices: {train_indices[:5]}")
            print(f"First five val_indices: {val_indices[:5]}")
        print("train_set:", len(train_set))
        print("val_set:", len(val_set))

        if use_two_stream:
            mlp_hidden_1, mlp_hidden_2, mlp_latent, cnn_latent, \
                decoder_hidden_1, decoder_hidden_2 = \
                config.get_two_stream_dims(model_complexity)
            model = TwoStreamBaseline(target_dim=target_dim,
                                      mlp_hidden_1=mlp_hidden_1,
                                      mlp_hidden_2=mlp_hidden_2,
                                      mlp_latent=mlp_latent,
                                      num_images=config.num_history_images,
                                      cnn_latent=cnn_latent,
                                      decoder_hidden_1=decoder_hidden_1,
                                      decoder_hidden_2=decoder_hidden_2,
                                      action_dim=action_dim)
        elif use_baseline:
            model = MLPBaseline(inp_dim=encoded_space_dim+target_dim,
                                lin_hid=lin_hid, lin_out=lin_out,
                                out_dim=action_dim)
        else:
            model = GeneralModel(encoded_space_dim=encoded_space_dim, target_dim=target_dim,
                                 enc_hid=enc_hid, cont_hid=cont_hid,
                                 action_dim=action_dim, use_image=use_image)
        m = model.to(device)
        num_params = sum(p.numel() for p in m.parameters())/1e3
        model_name += f"|{num_params}K_params"

        # selected_val_ind = random.sample(val_set.indices, k=3)
        # selected_train_idx = random.sample(train_set.indices, k=1)[0]

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Loss
        if use_custom_loss:
            criterion = CustomLoss(C)
            # criterion = KLLoss(C)
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)  # Adjust gamma as needed
        # data_tester(train_dataloader, model)
        weights_storage_root_dir = os.path.join(current_dir_path, 
                                                f"weights/{dataset_name}|{config.ds_ratio}|{model_name}/train_no_{i_train}")

        print(f"=== Train No: {i_train} ===")
        print(f"train num_samples:{len(train_dataloader.dataset)}")
        print(f"val num_samples:{len(val_dataloader.dataset)}")
        print("device:", device)
        print("dataset_name", dataset_name)
        print("model_name", model_name)
        print(num_params, 'K parameters')
        print("type_of_criterion", type(criterion))
        print("weights_storage_root_dir", weights_storage_root_dir)

        if os.path.exists(weights_storage_root_dir):
            print("The weight directory exists... Are you training one more time?")
            sys.exit(1)
        else:
            os.makedirs(weights_storage_root_dir)
        
        create_csv_files(weights_storage_root_dir, selected_val_ind=None)

        # while True:
        #     prediction_dynamics(dataset, selected_val_ind)

        train_loss_custom = 1e-10
        train_loss_torques = 1e-10
        train_losses_custom = [train_loss_custom]
        train_losses_torques = [train_loss_torques]
        val_losses_custom = []
        val_losses_torques = []

        for n in range(num_epochs):
            if n == 0:
                run_test(n)
                # pass

            start_time = time.time()
            train_loss_custom = 0
            train_loss_torques = 0

            model.train()
            for i, batch_data in enumerate(train_dataloader):
                batch_step = batch_data[0].to(device).float()
                batch_state = batch_data[1].to(device).float()
                batch_target_repr = batch_data[2].to(device).float()

                if use_two_stream:
                    batch_images_front = batch_data[3].to(device).float()
                    batch_images_side = batch_data[4].to(device).float()
                    batch_action = batch_data[5].to(device).float()
                else:
                    batch_action = batch_data[3].to(device).float()

                batch_noise = torch.normal(mean=0.0, std=noise_std, #std=0.001
                                        size=(batch_state.size()[0], encoded_space_dim)
                                        ).to(device).float()
                batch_state_noise = batch_state + batch_noise

                optimizer.zero_grad()
                if use_two_stream:
                    batch_action_pred_noise = model(batch_target_repr, batch_images_front, batch_images_side)
                    batch_action_noise = batch_action
                elif use_baseline:
                    nn_input = torch.cat((batch_target_repr, batch_state_noise), dim=1)
                    batch_action_pred_noise = model(nn_input)
                    batch_action_noise = batch_action - batch_noise[:, :joints_num]
                else:
                    batch_action_pred_noise , batch_x_des_noise, \
                        batch_diff_noise  = model(batch_target_repr, batch_state_noise)
                    # TODO: Think about it.
                    batch_action_noise = batch_action - batch_noise[:, :joints_num]

                if use_custom_loss:
                    # loss_custom, loss_torques = criterion(batch_action_pred_noise, batch_action_noise, 
                    #                                       batch_x_des_noise, batch_state, batch_step)
                    loss_custom, loss_torques = criterion(batch_action_pred_noise, batch_action_noise, 
                                                        batch_x_des_noise, batch_state)
                else:
                    loss_torques = criterion(batch_action_pred_noise, batch_action_noise)
                    loss_custom = loss_torques

                loss_custom.backward()
                optimizer.step()

                train_loss_custom += loss_custom.item() #* batch_action.size(0)
                train_loss_torques += loss_torques.item() #* batch_action.size(0)
            
            train_loss_custom /= len(train_dataloader.dataset)
            train_losses_custom.append(train_loss_custom**0.5)
            train_loss_torques /= len(train_dataloader.dataset)
            train_losses_torques.append(train_loss_torques**0.5)
            end_time = time.time()
            
            # scheduler.step()
            if n % validation_interval == 0:
                epoch_time = end_time - start_time
                print(f"Last epoch taken time: {epoch_time:.3f} seconds")

                act_abs_diff = torch.abs(batch_action_pred_noise - batch_action)
                mean_abs_diff = torch.mean(act_abs_diff, dim=0)
                for k in range(7):
                    train_info[f'mae_joint_{k+1}_train'].append(mean_abs_diff[k].item())

                run_test(n)

                # input_independent_baseline(dataset, selected_train_idx)
                # prediction_dynamics(dataset, selected_val_ind)

                weight_file_path = os.path.join(weights_storage_root_dir, f"fbc_{n}.pth")
                torch.save(model.state_dict(), weight_file_path)

                info_file_path = os.path.join(weights_storage_root_dir, f"info.pickle")
                with open(info_file_path, 'wb') as handle:
                    pickle.dump(train_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

                visualize_losses(train_losses_custom, val_losses_custom, n, 
                                f"loss_custom_{n//100}.png", "Custom Loss", fig_1)
                visualize_losses(train_losses_torques, val_losses_torques, n, 
                                f"loss_torques_{n//100}.png", "Training Loss", fig_2)
                
        plt.close('all')

