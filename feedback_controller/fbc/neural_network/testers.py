import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from torkin import TorKin
import torch
from nn_models import GeneralModel, MLPBaseline, TwoStreamBaseline
import torch.nn as nn
import math
import os
from config import Config
from PIL import Image
from torchvision import transforms


class Tester():
    def __init__(self) -> None:
        self.config = Config()

        self.joint_size = self.config.joints_num
        self.state_size = self.config.state_dim
        self.step_size = self.config.step_dim
        self.target_size = self.config.coords_dim
        self.onehot_size = self.config.onehot_dim

        self.cur_file_dir_path = os.path.dirname(__file__)
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

        # Load normalization parameters
        norm_params_path = os.path.join(
            self.cur_file_dir_path,
            f'data/torobo/{self.config.dataset_name}/normalization_params.npz'
        )
        norm_params = np.load(norm_params_path)
        self.xy_mean = norm_params['xy_mean'].flatten()  # (6,)
        self.xy_std = norm_params['xy_std'].flatten()    # (6,)
        self.action_std = torch.tensor(norm_params['action_std'].flatten()).to(self.device).float()
        self.state_mean = torch.tensor(norm_params['state_mean'].flatten()).to(self.device).float()
        self.state_std = torch.tensor(norm_params['state_std'].flatten()).to(self.device).float()
        # ee_pose normalization (only position is normalized)
        self.ee_pose_pos_mean = torch.tensor(norm_params['ee_pos_mean'].flatten()).to(self.device).float()
        self.ee_pose_pos_std = torch.tensor(norm_params['ee_pos_std'].flatten()).to(self.device).float()
        print(f"Loaded normalization params from {norm_params_path}")

        dataset_path = os.path.join(self.cur_file_dir_path,
                                    f'data/torobo/{self.config.dataset_name}/{self.config.ds_test_file}')
        self.dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1')
        print("Tester loaded dataset with shape:", self.dataset.shape)
        print("from this path", dataset_path)

        self.kin = TorKin()
        self.criterion = nn.L1Loss()
        self.criterion_mse = nn.MSELoss(reduction='sum')

        # Image preprocessing for use_image=True mode
        self.crop_params = {
            'top': 210,
            'left': 220,
            'height': 150,
            'width': 200,
        }
        transform_list = [
            transforms.Lambda(lambda img: transforms.functional.crop(
                img,
                top=self.crop_params['top'],
                left=self.crop_params['left'],
                height=self.crop_params['height'],
                width=self.crop_params['width']
            )),
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
        ]
        self.image_transform = transforms.Compose(transform_list)
        self.img_dir = os.path.join(
            self.cur_file_dir_path,
            f'data/torobo/{self.config.dataset_name}/triangle_images_fixed_cam'
        )

        # Load test trajectory mapping if available
        mapping_file = os.path.join(
            self.cur_file_dir_path,
            f'data/torobo/{self.config.dataset_name}/traj_mapping_{self.config.ds_ratio_test}.npz'
        )
        if os.path.exists(mapping_file):
            mapping = np.load(mapping_file)
            self.test_traj_indices = mapping['test_traj_indices']
            print(f"Loaded test trajectory mapping: {len(self.test_traj_indices)} trajectories")
        else:
            # Fallback: assume 1:1 mapping
            self.test_traj_indices = np.arange(self.dataset.shape[0])
            print(f"No trajectory mapping found, using 1:1 mapping")

    def compute_ee_pose(self, joint_positions, torso_values=[0, 0]):
        """Compute ee_pose (position + quaternion) from joint positions."""
        q_full = np.concatenate([torso_values, joint_positions])
        position, R_mat = self.kin.forwardkin(1, q_full)
        quat = Rotation.from_matrix(R_mat).as_quat()  # [x, y, z, w]
        return np.concatenate([position, quat])

    def normalize_ee_pose(self, ee_pose):
        """Normalize ee_pose: normalize position, keep quaternion as-is."""
        if ee_pose.dim() == 1:
            position = ee_pose[:3]
            quaternion = ee_pose[3:]
            position_normalized = (position - self.ee_pose_pos_mean) / self.ee_pose_pos_std
            return torch.cat([position_normalized, quaternion])
        else:
            position = ee_pose[:, :3]
            quaternion = ee_pose[:, 3:]
            position_normalized = (position - self.ee_pose_pos_mean) / self.ee_pose_pos_std
            return torch.cat([position_normalized, quaternion], dim=1)

    def load_model(self, train_no, epoch_no, use_custom_loss, model_complexity,
                   use_image=False):

        if use_image:
            # Load GeneralModel with image-based target representation
            cnn_latent, cont_hid, pose_enc_hid = self.config.get_image_model_dims(model_complexity)
            self.model = GeneralModel(
                encoded_space_dim=self.state_size,
                target_dim=self.target_size + self.onehot_size,
                action_dim=self.joint_size,
                enc_hid=0,
                cont_hid=cont_hid,
                use_image=True,
                cnn_latent=cnn_latent,
                onehot_dim=self.onehot_size,
                ee_pose_dim=self.config.ee_pose_dim,
                pose_enc_hid=pose_enc_hid
            )

            m = self.model.to(self.device)
            model_name = self.config.get_model_name(False, use_custom_loss, use_image=True)
            num_params = self.config.get_params_num(m)
            model_name += f"|{num_params}K_params"

            model_adr = f'weights/{self.config.dataset_name}|{self.config.ds_ratio}|{model_name}' + \
                f'/train_no_{train_no}/fbc_{epoch_no}.pth'
            model_path = os.path.join(self.cur_file_dir_path, model_adr)
            self.model.load_state_dict(torch.load(
                model_path,
                map_location=torch.device(self.device),
                weights_only=True))
            print("***\nLoaded GeneralModel (use_image=True) weights from", model_path)
        else:
            enc_hid, cont_hid, lin_hid, lin_out = self.config.get_model_dims(model_complexity)
            self.model = GeneralModel(self.state_size, self.target_size+self.onehot_size,
                                      self.joint_size,
                                      enc_hid, cont_hid,
                                      use_image=False)
            # self.baseline = MLPBaseline(self.state_size + (self.target_size+self.onehot_size),
            #                             lin_hid, lin_out,
            #                             self.joint_size)

            m = self.model.to(self.device)
            model_name_dnfc = self.config.get_model_name(False, use_custom_loss, False)
            num_params = self.config.get_params_num(m)
            model_name_dnfc += f"|{num_params}K_params"

            # m = self.baseline.to(self.device)
            # model_name_base = self.config.get_model_name(True, False, False)
            # num_params = self.config.get_params_num(m)
            # model_name_base += f"|{num_params}K_params"

            dnfc_adr = f'weights/{self.config.dataset_name}|{self.config.ds_ratio}|{model_name_dnfc}' + \
                f'/train_no_{train_no}/fbc_{epoch_no}.pth'
            # base_adr = f'weights/{self.config.dataset_name}|{self.config.ds_ratio}|{model_name_base}' + \
            #     f'/train_no_{train_no}/fbc_{epoch_no}.pth'

            dnfc_path = os.path.join(self.cur_file_dir_path, dnfc_adr)
            # base_path = os.path.join(self.cur_file_dir_path, base_adr)
            self.model.load_state_dict(torch.load(dnfc_path,
                                                  map_location=torch.device(self.device),
                                                  weights_only=True))
            # self.baseline.load_state_dict(torch.load(base_path,
            #                                          map_location=torch.device(self.device),
            #                                          weights_only=True))
            print("***\nLoaded model weights from", dnfc_path)
            # print("Loaded baseline weights from", base_path)
        

    def load_two_stream_model(self, train_no, epoch_no, model_complexity):
        mlp_hidden_1, mlp_hidden_2, mlp_latent, cnn_latent, \
            decoder_hidden_1, decoder_hidden_2 = \
            self.config.get_two_stream_dims(model_complexity)

        target_dim = self.target_size + self.onehot_size
        self.two_stream_model = TwoStreamBaseline(
            target_dim=target_dim,
            mlp_hidden_1=mlp_hidden_1,
            mlp_hidden_2=mlp_hidden_2,
            mlp_latent=mlp_latent,
            num_images=self.config.num_history_images,
            cnn_latent=cnn_latent,
            decoder_hidden_1=decoder_hidden_1,
            decoder_hidden_2=decoder_hidden_2,
            action_dim=self.joint_size
        )

        m = self.two_stream_model.to(self.device)
        model_name = self.config.get_model_name(False, False, False, use_two_stream=True)
        num_params = self.config.get_params_num(m)
        model_name += f"|{num_params}K_params"

        model_adr = f'weights/{self.config.dataset_name}|{self.config.ds_ratio}|{model_name}' + \
            f'/train_no_{train_no}/fbc_{epoch_no}.pth'

        model_path = os.path.join(self.cur_file_dir_path, model_adr)
        self.two_stream_model.load_state_dict(torch.load(
            model_path,
            map_location=torch.device(self.device),
            weights_only=True))
        print("***\nLoaded TwoStreamBaseline weights from", model_path)


    def get_delta_ang_offline(self,usebaseline,num):
        y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[]
        elem=self.dataset[num]

        if usebaseline:
            self.baseline.eval()
        else:
            self.model.eval()

        for i in range(self.dataset.shape[1]):
            input_tensor = torch.tensor(elem[i][self.step_size:self.step_size+self.state_size].tolist()).float()

            # input_tensor=torch.cat((joint_angles_tensor,velocities_tensor),dim=0)
            goal = elem[i][self.step_size+self.state_size:self.step_size+self.state_size+self.target_size].tolist()
            one_hot = elem[i][self.step_size+self.state_size+self.target_size:self.step_size+self.state_size+self.target_size+self.onehot_size].tolist()
            goal_tensor = torch.tensor(goal).float()
            goal_nn = torch.unsqueeze(goal_tensor, 0)
            state_nn = torch.unsqueeze(input_tensor, 0)
            touch_history = torch.tensor([one_hot]).float()
            if usebaseline:
                all = torch.cat((goal_nn, touch_history, state_nn), dim=1)
                velocities_tensor = self.baseline(all)
                velocities_tensor = torch.squeeze(velocities_tensor, 0)
            else:
                velocities_tensor = self.model(goal_nn, state_nn, touch_history)[0]
                velocities_tensor = torch.squeeze(velocities_tensor, 0)

            y1.append(float(velocities_tensor[0]))
            y2.append(float(velocities_tensor[1]))
            y3.append(float(velocities_tensor[2]))
            y4.append(float(velocities_tensor[3]))
            y5.append(float(velocities_tensor[4]))
            y6.append(float(velocities_tensor[5]))
            y7.append(float(velocities_tensor[6]))


        return y1,y2,y3,y4,y5,y6,y7
    

    def get_emulated(self, use_baseline, num, use_angle=False, return_path_point=False,
                      use_image=False):
        out = False
        y1, y2, y3, y4, y5, y6, y7 = [], [], [], [], [], [], []
        elem = self.dataset[num]

        # Initial state from dataset is NORMALIZED
        state_normalized = torch.tensor(elem[0][self.step_size :
                                     self.step_size+self.state_size].tolist()
                                     ).to(self.device).float()
        # Denormalize to get real state
        state_real = state_normalized * self.state_std + self.state_mean

        # Goal contains NORMALIZED coordinates (6 dims: x,y for 3 objects)
        goal_normalized = elem[0][self.step_size + self.state_size :
                       self.step_size + self.state_size + self.target_size].tolist()
        one_hot = elem[0][self.step_size + self.state_size + self.target_size :
                          self.step_size + self.state_size + self.target_size + self.onehot_size
                          ].tolist()

        # Denormalize coordinates for waypoint checking
        goal_real = (np.array(goal_normalized) * self.xy_std) + self.xy_mean

        # Load target image at step 0 if use_image=True
        target_image = None
        ee_pose_current = None
        ee_pose_prev = None
        if use_image:
            original_traj_no = self.test_traj_indices[num]
            img_path = os.path.join(
                self.img_dir,
                f"traj_{original_traj_no:03d}/step_000.jpg"
            )
            target_image = Image.open(img_path).convert('RGB')
            target_image = self.image_transform(target_image)
            target_image = target_image.unsqueeze(0).to(self.device).float()  # (1, 3, H, W)

            # Compute initial ee_pose from initial joint positions
            initial_joints = state_real[:7].cpu().numpy()
            ee_pose_current = torch.tensor(self.compute_ee_pose(initial_joints)).to(self.device).float()
            ee_pose_prev = ee_pose_current.clone()

        milestones = self.get_changes_indexes(num)

        # 6D coordinates: x,y for 3 objects (z is fixed at 0.865)
        z_fixed = 0.865
        obstA = torch.tensor([goal_real[0], goal_real[1], z_fixed])
        obstB = torch.tensor([goal_real[2], goal_real[3], z_fixed])
        obstC = torch.tensor([goal_real[4], goal_real[5], z_fixed])
        grepB = [obstB[0], obstB[1], 0.87]
        putB = [obstA[0], obstA[1], 0.9]
        grepC = [obstC[0], obstC[1], 0.87]
        putC = [obstA[0], obstA[1], 0.93]

        path_point = 0
        # print(milestones)
        milestone_js = [torch.tensor(elem[milestones[0]-1][self.step_size : self.step_size+self.joint_size].tolist()),
                        torch.tensor(elem[milestones[1]-1][self.step_size : self.step_size+self.joint_size].tolist()),
                        torch.tensor(elem[milestones[2]-1][self.step_size : self.step_size+self.joint_size].tolist()),
                        torch.tensor(elem[milestones[3]-1][self.step_size : self.step_size+self.joint_size].tolist())
                        ]
        if out:
            print(milestone_js)
            print('this is A')
            print(goal_real[:2])
            print('this is B')
            print(goal_real[2:4])
            print('this is C')
            print(goal_real[4:6])

        if use_baseline:
            self.baseline.eval()
        else:
            self.model.eval()

        for i in range(self.dataset.shape[1]):
            # Model receives NORMALIZED inputs
            touch_history = torch.tensor([one_hot]).to(self.device).float()

            with torch.no_grad():
                if use_baseline:
                    state_nn = torch.unsqueeze(state_normalized, 0).float()
                    goal_tensor = torch.tensor(goal_normalized).to(self.device).float()
                    goal_nn = torch.unsqueeze(goal_tensor, 0)
                    basel_input = torch.cat((goal_nn, touch_history, state_nn), dim=1)
                    velocities_tensor = self.baseline(basel_input)
                elif use_image:
                    # Use two consecutive ee_poses (normalized)
                    ee_pose_prev_norm = self.normalize_ee_pose(ee_pose_prev)
                    ee_pose_current_norm = self.normalize_ee_pose(ee_pose_current)
                    ee_poses = torch.cat([ee_pose_prev_norm, ee_pose_current_norm])
                    ee_poses_nn = torch.unsqueeze(ee_poses, 0).float()
                    velocities_tensor, x_des, _ = self.model(
                        target_image, ee_poses_nn, touch_history)
                else:
                    state_nn = torch.unsqueeze(state_normalized, 0).float()
                    goal_tensor = torch.tensor(goal_normalized).to(self.device).float()
                    goal_nn = torch.unsqueeze(goal_tensor, 0)
                    velocities_tensor, x_des, _ = self.model(goal_nn, state_nn, touch_history)

            velocities_tensor = torch.squeeze(velocities_tensor, 0)
            # Denormalize velocities
            velocities_real = velocities_tensor * self.action_std

            # Update real state
            state_real[:7] += velocities_real
            state_real[7:] = velocities_real
            # Re-normalize state for next iteration
            state_normalized = (state_real - self.state_mean) / self.state_std

            # Update ee_poses for next iteration (when use_image=True)
            if use_image:
                ee_pose_prev = ee_pose_current.clone()
                ee_pose_current = torch.tensor(
                    self.compute_ee_pose(state_real[:7].cpu().numpy())
                ).to(self.device).float()

            print_it_out = out

            # Use real state for waypoint checking
            if path_point == 0 and self.close_enough(state_real, grepB):
                # print(i)
                path_point = 1
                one_hot = [1, 0, 0, 0]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            elif path_point == 1 and self.close_enough(state_real, putB):
                path_point = 2
                one_hot = [0, 1, 0, 0]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            elif path_point == 2 and self.close_enough(state_real, grepC):
                path_point = 3
                one_hot = [0, 0, 1, 0]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            elif path_point == 3 and self.close_enough(state_real, putC):
                path_point = 4
                one_hot = [0, 0, 0, 1]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            # Return real (denormalized) values
            if use_angle:
                add = velocities_real
            else:
                add = state_real

            y1.append(float(add[0]))
            y2.append(float(add[1]))
            y3.append(float(add[2]))
            y4.append(float(add[3]))
            y5.append(float(add[4]))
            y6.append(float(add[5]))
            y7.append(float(add[6]))

        if return_path_point:
            return y1, y2, y3, y4, y5, y6, y7, path_point
        else:
            return y1, y2, y3, y4, y5, y6, y7
        

    def get_emulated_s(self,usebaseline,num,use_angle=False,return_path_point=False):

        y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[]
        elem=self.dataset[num]
        print(num)
        state=torch.tensor(elem[0][1:1+self.state_size].tolist())

        goal=elem[0][self.step_size+self.state_size+self.joint_size:self.step_size+self.state_size+self.joint_size+9].tolist()
        one_hot=elem[0][self.step_size+self.state_size+self.joint_size+9:].tolist()
        milestones=self.get_changes_indexes(num)
        path_point=0
        # print(milestones)
        points=[]
        points.append(goal[:3])
        points.append(goal[3:6])
        points.append(goal[6:9])

        milestone_js=[torch.tensor(elem[milestones[0]-1][1:8].tolist()),torch.tensor(elem[milestones[1]-1][1:8].tolist()),torch.tensor(elem[milestones[2]-1][1:8].tolist()),torch.tensor(elem[milestones[3]-1][1:8].tolist())]
        print(goal)
        if usebaseline:
            self.baseline.eval()
        else:
            self.model.eval()
        
        for i in range(self.dataset.shape[1]):
            goal_tensor = torch.tensor(goal).float()
            goal_nn = torch.unsqueeze(goal_tensor, 0)
            state_nn = torch.unsqueeze(state, 0).float()
            touch_history = torch.tensor([one_hot]).float()

            if usebaseline:
                all = torch.cat((goal_nn, touch_history, state_nn), dim=1)
                velocities_tensor = self.baseline(all)
                velocities_tensor = torch.squeeze(velocities_tensor, 0)
            else:
                velocities_tensor, x_des = self.model(goal_nn, state_nn, touch_history)[0:2]
                velocities_tensor = torch.squeeze(velocities_tensor, 0)

            state[:7] += velocities_tensor
            state[7:] = velocities_tensor
            if path_point==0 and self.close_enough(state[:7],milestone_js[0]):
                # print(i)
                path_point=1
                one_hot=[1,0,0,0]
                print(x_des)
                print(self.get_end_eff(x_des))
                points.append(self.get_end_eff(x_des))
            elif path_point==1 and self.close_enough(state[:7],milestone_js[1]):
                path_point=2
                one_hot=[0,1,0,0]
                print(x_des)
                print(self.get_end_eff(x_des))
                points.append(self.get_end_eff(x_des))
            elif path_point==2 and self.close_enough(state[:7],milestone_js[2]):
                path_point=3
                one_hot=[0,0,1,0]           
                print(x_des)
                print(self.get_end_eff(x_des))
                points.append(self.get_end_eff(x_des))
            elif path_point==3 and self.close_enough(state[:7],milestone_js[3]):
                path_point=4
                one_hot=[0,0,0,1]   
                print(x_des)
                print(self.get_end_eff(x_des))
                points.append(self.get_end_eff(x_des))
            if use_angle:
                add=velocities_tensor
            else:
                add=state
                
            y1.append(float(add[0]))
            y2.append(float(add[1]))
            y3.append(float(add[2]))
            y4.append(float(add[3]))
            y5.append(float(add[4]))
            y6.append(float(add[5]))
            y7.append(float(add[6]))
        return points


    
    def get_coordinats(self, num, use_baseline, use_image=False):
        y1, y2, y3, y4, y5, y6, y7 = self.get_emulated(use_baseline, num,
                                                       use_angle=False,
                                                       use_image=use_image)
        x,y,z=[],[],[]
        for i in range(len(y1)):
            my_l=[0,0]+[y1[i]]+[y2[i]]+[y3[i]]+[y4[i]]+[y5[i]]+[y6[i]]+[y7[i]]

            p, R = self.kin.forwardkin(1, np.array(my_l))
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        return x,y,z

    def get_js_in_rad(self,num,use_baseline):
        y1,y2,y3,y4,y5,y6,y7=self.get_emulated(use_baseline,num,True)* (180.0 / math.pi)
        return y1,y2,y3,y4,y5,y6,y7

    def get_real_delta_ang(self, num, use_angle):
        y1_real, y2_real, y3_real, y4_real = [], [], [], []
        y5_real, y6_real, y7_real = [], [], []
        # Get denormalization params for joints (first 7 dims of state)
        joint_mean = self.state_mean[:7].cpu().numpy()
        joint_std = self.state_std[:7].cpu().numpy()

        for j in self.dataset[num][0:]:
            # Joint positions are normalized in dataset at indices 1:8
            joints_normalized = j[1:8]
            # Denormalize: real = normalized * std + mean
            delta_pos = joints_normalized * joint_std + joint_mean

            y1_real.append((delta_pos[0]))
            y2_real.append((delta_pos[1]))
            y3_real.append((delta_pos[2]))
            y4_real.append((delta_pos[3]))
            y5_real.append((delta_pos[4]))
            y6_real.append((delta_pos[5]))
            y7_real.append((delta_pos[6]))
        return y1_real,y2_real,y3_real,y4_real,y5_real,y6_real,y7_real
    
    def get_real_coordinates(self, num):
        y1, y2, y3, y4, y5, y6, y7 = self.get_real_delta_ang(num, False)
        x, y, z = [], [], []
        for i in range(len(y1)):
            my_l = [0, 0] + [y1[i]] + [y2[i]] + [y3[i]] + \
                [y4[i]] + [y5[i]] + [y6[i]] + [y7[i]]
            p, R = self.kin.forwardkin(1, np.array(my_l))
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        return x, y, z

    def close_enough(self, js1, pos_2):
        # for i in range(7):
        #     if abs(js1[i]-js2[i])>0.05:
        #         return False
        # return True

        my_l = [0, 0]
        for j in js1:
            my_l.append(float(j))
        p1, R = self.kin.forwardkin(1, np.array(my_l))

        # my_l = [0, 0]
        # for j in js2:
        #     my_l.append(float(j))
        # p2, R = self.kin.forwardkin(1, np.array(my_l))
        p2 = pos_2

        # TODO: This shouldn't be mean, it should be sum.
        dist = np.linalg.norm(p1-p2)
        # print(p1, p2, dist)
        # if (self.criterion_mse(torch.tensor(p1), torch.tensor(p2))**0.5) <= 0.0001: 
        if dist <= 0.02: #0.015:
            return True
        return False


    def get_obs_coordinates(self, num):
        # 6D coordinates: x,y for 3 objects (NORMALIZED in dataset)
        elem = self.dataset[num]
        goal_normalized = elem[0][self.step_size + self.state_size :
                                  self.step_size + self.state_size + self.target_size]

        # Denormalize coordinates
        goal_real = (np.array(goal_normalized) * self.xy_std) + self.xy_mean

        # Add fixed z coordinate
        z_fixed = 0.865
        obstA = np.array([goal_real[0], goal_real[1], z_fixed])
        obstB = np.array([goal_real[2], goal_real[3], z_fixed])
        obstC = np.array([goal_real[4], goal_real[5], z_fixed])

        return obstA, obstB, obstC

    def get_loss(self,option,use_baseline):
        all_loss=0
        for num in range(self.dataset.shape[0]):
            loss=0
            real_output=self.get_real_delta_ang(num,True)
            if option=='emulated':    
                network_output=self.get_emulated(use_baseline,num,True)
            else:
                network_output=self.get_delta_ang_offline(use_baseline,num)
            for i in range(299):
                loss+=self.criterion(torch.tensor([real_output[j][i] for j in range(7)]),torch.tensor([network_output[j][i] for j in range(7)]))
            loss/=299
            all_loss+=loss
        all_loss/=self.dataset.shape[0]
        return all_loss

    def get_changes_indexes(self, num):
        indexes = []
        start = [0, 0, 0, 0]
        elem = self.dataset[num]
        for i in range(self.dataset.shape[1]):
            goal = elem[i][self.step_size+self.state_size :
                           self.step_size+self.state_size+self.target_size+self.onehot_size
                           ].tolist()
            for j in range(self.onehot_size):
                if goal[self.target_size + j] != start[j]:
                    indexes.append(i)
                    start = goal[self.target_size:]
        return indexes

    def calculate_cartesian_perform(self, use_baseline, use_image=False):
        point_reached = 0
        for num in range(self.dataset.shape[0]):
            point_reached += self.get_emulated(use_baseline, num,
                                               use_angle=False,
                                               return_path_point=True,
                                               use_image=use_image)[7]
        return point_reached/(4*self.dataset.shape[0])   

    def get_end_eff(self, js):
        my_l = [0, 0]
        for j in js:
            my_l.append(float(j))
        p1, R = self.kin.forwardkin(1, np.array(my_l))
        return p1
    
    def get_one_hot(self,num,i):
        return self.dataset[num][i][self.step_size + self.state_size + self.target_size : self.step_size + self.state_size + self.target_size+self.onehot_size].tolist()
    

    def get_goal(self,num,i):
        return self.dataset[num][i][self.step_size + self.state_size : self.step_size + self.state_size + self.target_size ].tolist()
    
    def get_state(self,num,i):
        return self.dataset[num][i][self.step_size : self.step_size + self.state_size ].tolist()

    def get_action(self,num,i):
        return self.dataset[num][i][self.step_size + self.state_size + self.target_size+self.onehot_size : self.step_size + self.state_size + self.target_size+self.onehot_size + self.joint_size ].tolist()
    def get_target(self,num, i):
        return self.dataset[num][i][self.step_size + self.state_size : self.step_size + self.state_size + self.target_size+self.onehot_size ].tolist()
   
    



