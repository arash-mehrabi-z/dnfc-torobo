import matplotlib.pyplot as plt
import numpy as np
from torkin import TorKin
import torch
from nn_models import  GeneralModel as GeneralModel
from nn_models import  MLPBaseline as MLPBaseline
import torch.nn as nn
import math
import matplotlib.cm as cm

import os

current_dir = os.getcwd()


class Tester():
    def __init__(self) -> None:
        self.treshold=0.015
        self.state_size=14
        self.step_size=1
        self.joint_size=7
        self.target_size=9
        self.onehot_size=4
        general_path='/Users/denizakkoc/Desktop/dnfc_arash/dnfc-torobo/feedback_controller/fbc/'

        self.model = GeneralModel(self.state_size,self.target_size+self.onehot_size,7,False)
        self.baseline = MLPBaseline(self.state_size+self.target_size+self.onehot_size,7)
        

        
        # self.model_path = general_path + '/neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse|91.541K_params' 
        # self.model_path = general_path + '/neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|mse_los|tar_cart|v_custl_mse|7.541K_params'

        # self.model_path = general_path + '/neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_smaller_encoder_32x64_0.05noise|7.541K_params'



        self.model_path = general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|mse_los|tar_cart|v_custl_mse_tanh|26.189K_params'

        # self.model_path= general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|mse_los|tar_cart|v_custl_mse_tanh|26.237K_params'



        
        self.model_path= general_path +'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_tanh_corrected|9.173K_params'
        self.model_path= general_path +'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_tanh_corrected|8.653K_params'
        
        # self.model_path= general_path +'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|mse_los|tar_cart|v_custl_mse_tanh|26.517K_params'
        self.model_path= general_path +'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_tanh_corrected|9.661K_params'

        self.model_path = general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_tanh_corrected_deneme_3|14.391K_params'
        self.model_path = general_path + 'neural_network/weights_neon/trajs:1900_blocks:3_random_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_tanh|14.715K_params'

        # self.model_path= general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_tanh|25.749K_params'


        # self.model_path = general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|cus_los_1e-05|tar_cart|v_custl_mse_tanh|25.301K_params'

        



        
        self.base_path = general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|mse_los|tar_cart|base|v_base_tanh|25.175K_params'

        
        # self.base_path = general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|mse_los|tar_cart|base|v_base|25.175K_params'
        self.base_path = general_path + 'neural_network/weights/trajs:360_blocks:3_triangle_v|0.8|mse_los|tar_cart|base|v_base_tanh_denem_3|14.695K_params'
        self.base_path = general_path + 'neural_network/weights_neon/trajs:1900_blocks:3_random_v|0.8|mse_los|tar_cart|base|v_base_tanh|14.907K_params'
        

        self.epoch1 = '/train_no_1/fbc_0.pth'
        self.epoch2 = '/train_no_1/fbc_0.pth'

        self.model.load_state_dict(torch.load(self.model_path+self.epoch1,map_location=torch.device('cpu'),weights_only=True))
        self.baseline.load_state_dict(torch.load(self.base_path+self.epoch2,map_location=torch.device('cpu'),weights_only=True))
        self.dataset_name='test'
        self.dataset = np.load(general_path+'neural_network/data/torobo/815_trajs_static/test_0.95.npy', allow_pickle=True, encoding='latin1')[:]
        print(self.dataset.shape)
        self.kin=TorKin()
        self.criterion= nn.L1Loss()
        self.criterion2= nn.MSELoss()


    def get_delta_ang_offline(self,usebaseline,num):

        y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[]
        elem=self.dataset[num]

        if usebaseline:
            self.baseline.eval()
        else:
            self.model.eval()


        for i in range(299):
            input_tensor=torch.tensor(elem[i][self.step_size:self.step_size+self.state_size].tolist())
            
            # input_tensor=torch.cat((joint_angles_tensor,velocities_tensor),dim=0)
            goal=elem[i][self.step_size+self.state_size:].tolist()
            goal_tensor=torch.tensor(goal)
            if usebaseline:
                all=torch.cat((goal_tensor, input_tensor),dim=0)
                velocities_tensor=self.baseline(all)
            else:
                velocities_tensor=self.model(goal_tensor, input_tensor)[0]
                
            y1.append(float(velocities_tensor[0]))
            y2.append(float(velocities_tensor[1]))
            y3.append(float(velocities_tensor[2]))
            y4.append(float(velocities_tensor[3]))
            y5.append(float(velocities_tensor[4]))
            y6.append(float(velocities_tensor[5]))
            y7.append(float(velocities_tensor[6]))


        return y1,y2,y3,y4,y5,y6,y7
    

    def get_emulated(self,usebaseline,num,use_angle=False,return_path_point=False,return_x_des=False):
        if return_x_des:
            x_des1,x_des2,x_des3,x_des4,x_des5,x_des6,x_des7=[],[],[],[],[],[],[]
        out=False

        y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[]
        elem=self.dataset[num]
        state=torch.tensor(elem[0][1:1+self.state_size].tolist())

        goal=elem[0][self.step_size+self.state_size: self.step_size+self.state_size+9].tolist()

        one_hot=elem[0][self.step_size+self.state_size+9 : self.step_size+self.state_size + 13].tolist()
        path_point=0

        if out:
            print(elem.shape)
            print(elem[15:])
            # print(milestone_js)
            print('this is A')
            print(goal[:3])
            print('this is B')
            print(goal[3:6])
            print('this is C')
            print(goal[6:])
        if usebaseline:
            self.baseline.eval()
        else:
            self.model.eval()
        
        for i in range(299):
            goal_tensor=torch.tensor(goal+one_hot)

            if usebaseline:
                all=torch.cat((goal_tensor, state),dim=0)
                velocities_tensor=self.baseline(all)
            else:
                velocities_tensor,x_des=self.model(goal_tensor, state)[0:2]

            print_it_out=out
            state[:7]+=velocities_tensor
            state[7:]=velocities_tensor
            if path_point==0 and self.close_enough(state[:7],[goal[3],goal[4],0.87]):
                # if not usebaseline:
                    # print('this is point B, ',[goal[3],goal[4],0.87])
                    # print('x_des end effector, ', self.get_end_eff(x_des[:7]))
                # print(i)
                path_point=1
                one_hot=[1,0,0,0]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))
            elif path_point==1 and self.close_enough(state[:7],[goal[0],goal[1],0.90]):
                path_point=2
                one_hot=[0,1,0,0]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))
            elif path_point==2 and self.close_enough(state[:7],[goal[6],goal[7],0.87]):
                path_point=3
                one_hot=[0,0,1,0]           
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))
            elif path_point==3 and self.close_enough(state[:7],[goal[0],goal[1],0.93]):
                path_point=4
                one_hot=[0,0,0,1]   
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))
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
            if return_x_des:    
                x_des1.append(x_des[0].detach().numpy())
                x_des2.append(x_des[1].detach().numpy())
                x_des3.append(x_des[2].detach().numpy())
                x_des4.append(x_des[3].detach().numpy())
                x_des5.append(x_des[4].detach().numpy())
                x_des6.append(x_des[5].detach().numpy())
                x_des7.append(x_des[6].detach().numpy())
        if return_path_point:
            if return_x_des:
                return y1,y2,y3,y4,y5,y6,y7,path_point,x_des1,x_des2,x_des3,x_des4,x_des5,x_des6,x_des7
            return y1,y2,y3,y4,y5,y6,y7,path_point
        else:
            if return_x_des:
                return y1,y2,y3,y4,y5,y6,y7,x_des1,x_des2,x_des3,x_des4,x_des5,x_des6,x_des7
            return y1,y2,y3,y4,y5,y6,y7
        


    
    def get_coordinats(self,num,use_baseline):
 
        y1,y2,y3,y4,y5,y6,y7=self.get_emulated(use_baseline,num,False)
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


    def get_real_delta_ang(self,num,use_angle):
        y1_real,y2_real,y3_real,y4_real,y5_real,y6_real,y7_real=[],[],[],[],[],[],[]
        for j in self.dataset[num][1:]:
            if use_angle:
                delta_pos=((j[1:8]))
            else:
                delta_pos=((j[1:8]))
            y1_real.append((delta_pos[0]))
            y2_real.append((delta_pos[1]))
            y3_real.append((delta_pos[2]))
            y4_real.append((delta_pos[3]))
            y5_real.append((delta_pos[4]))
            y6_real.append((delta_pos[5]))
            y7_real.append((delta_pos[6]))
        y1_real.append((delta_pos[0]))
        y2_real.append((delta_pos[1]))
        y3_real.append((delta_pos[2]))
        y4_real.append((delta_pos[3]))
        y5_real.append((delta_pos[4]))
        y6_real.append((delta_pos[5]))
        y7_real.append((delta_pos[6]))       
        return y1_real,y2_real,y3_real,y4_real,y5_real,y6_real,y7_real
    
    def get_real_coordinates(self, num):
        y1,y2,y3,y4,y5,y6,y7=self.get_real_delta_ang(num,False)
        x,y,z=[],[],[]
        for i in range(len(y1)):
            my_l=[0,0]+[y1[i]]+[y2[i]]+[y3[i]]+[y4[i]]+[y5[i]]+[y6[i]]+[y7[i]]
            p, R = self.kin.forwardkin(1, np.array(my_l))
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        return x,y,z

    def close_enough(self, js1,p2):
        # for i in range(7):
        #     if abs(js1[i]-js2[i])>0.05:
        #         return False
        # return True

        my_l=[0,0]
        for j in js1:
            my_l.append(float(j))
        p1, R = self.kin.forwardkin(1, np.array(my_l))
        # my_l=[0,0]
        # for j in js2:
        #     my_l.append(float(j))
        # p2, R = self.kin.forwardkin(1, np.array(my_l))

        if (3*(self.criterion2(torch.tensor(p1),torch.tensor(p2)))**(1/2))<self.treshold:
            # print('-----------')
            # print(3*(self.criterion2(torch.tensor(p1),torch.tensor(p2)))**(1/2))
            # print(js1)
            # print(js2)
            # print(p1)
            # print(p2)
            return True
        return False


    def get_obs_coordinates(self, num):
        elem=self.dataset[num]
        obstA=elem[1][1+self.state_size:4+self.state_size]
        obstB=elem[1][4+self.state_size:7+self.state_size]
        obstC=elem[1][7+self.state_size:10+self.state_size]

        return obstA,obstB,obstC

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

    # def get_changes_indexes(self,num):
    #     indecex=[]
    #     start=[0,0,0,0]
    #     elem=self.dataset[num]
    #     for i in range(299):
    #         goal=elem[i][self.step_size+self.state_size+self.joint_size:].tolist()
    #         for j in range(4):
    #             if goal[9+j]!=start[j]:
    #                 indecex.append(i)
    #                 start=goal[9:]
    #     return indecex
                

    def calculate_cartesian_perform(self,use_baseline):
        point_reached=0
        for num in range(self.dataset.shape[0]): 
            point_reached+=self.get_emulated(use_baseline,num,False,True)[7]
        return point_reached/(4*self.dataset.shape[0])  

    def calculate_cartesian_perform_2(self, use_baseline):
        point_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for num in range(self.dataset.shape[0]): 
            point = self.get_emulated(use_baseline, num, False, True)[7]
            if point in point_counts:
                point_counts[point] += 1
            else:
                print(f"Unexpected point value encountered: {point}")
        
        print("Point counts:", point_counts)
        return point_counts
 
    

    def get_end_eff(self,js):
        my_l=[0,0]
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
   
    



    # def get_trajectory_error(self,use_baseline):
    #     all_loss=0
    #     for i in range(len(self.dataset)):
    #         real_output=self.get_real_coordinates(i)
    #         network_output=self.get_coordinats(i,use_baseline)

    #         all_loss+=self.criterion(torch.tensor([real_output]),torch.tensor([network_output]))
    #     # print(self.criterion(torch.tensor([real_output]),torch.tensor([network_output])))
    #     all_loss/=self.dataset.shape[0]
    #     return all_loss.item()
    


    def get_trajectory_error(self,num,use_baseline):
        all_loss=0
        real_output=self.get_real_coordinates(num)
        network_output=self.get_coordinats(num,use_baseline)

        real_output=torch.tensor(real_output)
        network_output=torch.tensor(network_output)
        for i in range(299):
            all_loss+=self.criterion(real_output[:,i],network_output[:,i])
        # print(self.criterion(torch.tensor([real_output]),torch.tensor([network_output])))
        return all_loss.item()/299
    
    def get_all_trajectory_error(self,use_baseline):
        all_loss=0
        for num in range(self.dataset.shape[0]):
            all_loss+=self.get_trajectory_error(num,use_baseline)
        all_loss/=self.dataset.shape[0]
        return all_loss.item()



    def plot_x_des(self, num):
        # Retrieve data
        _, _, _, _, _, _, _, x_des1, x_des2, x_des3, x_des4, x_des5, x_des6, x_des7 = self.get_emulated(False, num, False, False, True)
        x1, x2, x3, x4, x5, x6, x7 = self.get_real_delta_ang(num, True)

        # Organize data for easier plotting
        x_des = [x_des1, x_des2, x_des3, x_des4, x_des5, x_des6, x_des7]
        x_real = [x1, x2, x3, x4, x5, x6, x7]
        time = np.arange(299)

        # Define colormaps to make each line distinct
        x_des_colormap = cm.hot
        x_real_colormap = cm.cool

        # Setting up the figure and subplots
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 18), sharex=True)
        fig.suptitle('X Desired vs Real Coordinates', fontsize=16)
        axes = axes.flatten()

        # Plot each x_des and x_real pair
        for i, (x_d, x_r, ax) in enumerate(zip(x_des, x_real, axes)):
            x_des_colors = x_des_colormap(np.linspace(0, 1, len(time)))  # Color gradient for x_des
            x_real_colors = x_real_colormap(np.linspace(0, 1, len(time)))  # Color gradient for x_real

            for j in range(len(time)-1):  # Plot with different color segments for gradient effect
                ax.plot(time[j:j+2], x_d[j:j+2], linestyle='--', color=x_des_colors[j], label=f'x_des{i+1}' if j == 0 else "")
                ax.plot(time[j:j+2], x_r[j:j+2], color=x_real_colors[j], alpha=0.7, label=f'x{i+1}' if j == 0 else "")
            
            ax.set_ylabel(f'X{i+1}')
            ax.legend(loc='upper right')

        # Remove the last unused subplot (8th position)
        fig.delaxes(axes[-1])

        axes[-2].set_xlabel('Time')  # Set x-axis label on second last to avoid overwriting
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
        plt.show()



    def plot_x_des_2(self, num):
        # Retrieve data
        _, _, _, _, _, _, _, x_des1, x_des2, x_des3, x_des4, x_des5, x_des6, x_des7 = self.get_emulated(False, num, False, False, True)
        x1, x2, x3, x4, x5, x6, x7 = self.get_real_delta_ang(num, True)

        # Organize data for easier plotting
        x_des = [x_des1, x_des2, x_des3, x_des4, x_des5, x_des6, x_des7]
        x_real = [x1, x2, x3, x4, x5, x6, x7]
        time = np.arange(299)

        # Define colormaps to make each line distinct
        x_des_colormap = cm.hot
        x_real_colormap = cm.cool

        # Setting up the figure and subplots
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 18), sharex=True, subplot_kw={'projection': None})
        fig.suptitle('X Desired vs Real Coordinates', fontsize=16)
        axes = axes.flatten()

        # Plot each x_des and x_real pair
        for i, (x_d, x_r, ax) in enumerate(zip(x_des, x_real, axes[:-1])):  # Exclude last axis for 3D plot
            x_des_colors = x_des_colormap(np.linspace(0, 1, len(time)))  # Color gradient for x_des
            x_real_colors = x_real_colormap(np.linspace(0, 1, len(time)))  # Color gradient for x_real

            for j in range(len(time)-1):  # Plot with different color segments for gradient effect
                ax.plot(time[j:j+2], x_d[j:j+2], linestyle='--', color=x_des_colors[j], label=f'x_des{i+1}' if j == 0 else "")
                ax.plot(time[j:j+2], x_r[j:j+2], color=x_real_colors[j], alpha=0.7, label=f'x{i+1}' if j == 0 else "")
            
            ax.set_ylabel(f'X{i+1} (rad)')
            ax.legend(loc='upper right')

        # Create a 3D plot in the last subplot position
        ax3d = fig.add_subplot(4, 2, 8, projection='3d')
        x_model, y_model, z_model = self.get_coordinats(num, False)
        # x_base, y_base, z_base = self.get_coordinats(num, True)
        x_real, y_real, z_real = self.get_real_coordinates(num)
        obstA, obstB, obstC = self.get_obs_coordinates(num)

        # Plot trajectories and obstacles in 3D
        ax3d.plot(x_real, y_real, z_real, c='r', label='ground truth')
        ax3d.plot(x_model, y_model, z_model, c='b', label='our model')
        # ax3d.plot(x_base, y_base, z_base, c='g', label='base')
        # ax3d.scatter(x_base, y_base, z_base, c='g', s=5)
        ax3d.scatter(x_real, y_real, z_real, c='r', s=5)
        ax3d.scatter(x_model, y_model, z_model, c='b', s=5)

        # Plot and annotate obstacles
        ax3d.scatter([obstA[0]], [obstA[1]], [obstA[2]], c='k', marker='o')
        ax3d.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='k', marker='o')
        ax3d.scatter([obstC[0]], [obstC[1]], [obstC[2]], c='k', marker='o')
        ax3d.text(obstA[0], obstA[1], obstA[2], 'point A', color='black', fontsize=10, ha='center')
        ax3d.text(obstB[0], obstB[1], obstB[2], 'point B', color='black', fontsize=10, ha='center')
        ax3d.text(obstC[0], obstC[1], obstC[2], 'point C', color='black', fontsize=10, ha='center')

        ax3d.set_xlabel('X Axis')
        ax3d.set_ylabel('Y Axis')
        ax3d.set_zlabel('Z Axis')
        ax3d.set_title('3D Trajectories')
        ax3d.legend()

        axes[-2].set_xlabel('Time')  # Set x-axis label on the second last subplot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
        plt.show()

    def plot_cartesian_performance(self, start_epoch=0, end_epoch=1000, step=100):
        """
        Plots the cartesian performance of the model across different epochs
        
        Args:
            start_epoch: Starting epoch number (default: 25)
            end_epoch: Ending epoch number (default: 15000)
            step: Epoch increment (default: 25)
        """
        epochs = list(range(start_epoch, end_epoch + 1, step))
        performances = []
        
        # Store original model path and epoch
        original_epoch = self.epoch1
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Evaluate performance for each epoch
        use_baseline=False
        for epoch in epochs:
            # Update epoch in model path
            self.epoch1 = f'/train_no_1/fbc_{epoch}.pth'
            
            try:
                # Load model weights for current epoch
                if(use_baseline):
                    print('this is baseline')
                    self.baseline.load_state_dict(torch.load(self.base_path + self.epoch1, 
                                                map_location=torch.device('cpu') ,weights_only=True))
                    performance = self.calculate_cartesian_perform(True)
                    performances.append(performance)

                else:
                    
                    print('this is model')
                    self.model.load_state_dict(torch.load(self.model_path + self.epoch1, 
                                map_location=torch.device('cpu') ,weights_only=True))
                
                
                # Calculate performance
                    performance = self.calculate_cartesian_perform(False)
                    performances.append(performance)
                    
                # Print progress
                print(f"Processed epoch {epoch}: Performance = {performance:.4f}")
                
            except FileNotFoundError:
                print(f"Skipping epoch {epoch}: File not found")
                continue
            except Exception as e:
                print(f"Error processing epoch {epoch}: {str(e)}")
                continue
        
        # Restore original epoch
        self.epoch1 = original_epoch
        
        # Plot results
        plt.plot(epochs[:len(performances)], performances, 'b-', linewidth=2)
        plt.fill_between(epochs[:len(performances)], performances, alpha=0.2)
        dummy=''
        if(use_baseline):
            dummy='baseline'
        else:
            dummy='dnfc'
        
        # Customize plot
        plt.title(f'Cartesian Performance Across Training Epochs {dummy} ', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Performance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add average line
        avg_performance = sum(performances) / len(performances)
        plt.axhline(y=avg_performance, color='r', linestyle='--', 
                    label=f'Average: {avg_performance:.4f}')
        
        # Add max performance point
        max_performance = max(performances)
        max_epoch = epochs[performances.index(max_performance)]
        plt.plot(max_epoch, max_performance, 'ro', 
                label=f'Max: {max_performance:.4f} (Epoch {max_epoch})')
        
        plt.legend()
        
        # Format axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        return epochs[:len(performances)], performances


    def plot_cartesian_performance_both(self, start_epoch=100, end_epoch=2000, step=100):
        """
        Plots the cartesian performance of both models across different epochs
        Args:
            start_epoch: Starting epoch number (default: 1000)
            end_epoch: Ending epoch number (default: 5000)
            step: Epoch increment (default: 25)
        """
        epochs = list(range(start_epoch, end_epoch + 1, step))
        model_performances = []
        baseline_performances = []
        
        # Store original model path and epoch
        original_epoch = self.epoch1
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Function to collect performance data for a model
        def collect_performance_data(use_baseline):
            performances = []
            for epoch in epochs:
                # Update epoch in model path
                
                try:
                    if use_baseline:
                        self.epoch1 = f'/train_no_1/fbc_{epoch}.pth'
                        self.baseline.load_state_dict(torch.load(self.base_path + self.epoch1,
                                                            map_location=torch.device('cpu')
                                                            ,weights_only=True))
                        performance = self.calculate_cartesian_perform(True)
                    else:
                        self.epoch1 = f'/train_no_1/fbc_{epoch}.pth'
                        self.model.load_state_dict(torch.load(self.model_path + self.epoch1,
                                                            map_location=torch.device('cpu')
                                                            ,weights_only=True))
                        performance = self.calculate_cartesian_perform(False)
                    
                    performances.append(performance)
                    print(f"Processed epoch {epoch} for {'baseline' if use_baseline else 'model'}: "
                        f"Performance = {performance:.4f}")
                    
                except FileNotFoundError:
                    print(f"Skipping epoch {epoch}: File not found")
                    continue
                except Exception as e:
                    print(f"Error processing epoch {epoch}: {str(e)}")
                    continue
            
            return performances
        
        # Collect data for both models
        print("Processing model performance...")
        model_performances = collect_performance_data(use_baseline=False)
        
        print("\nProcessing baseline performance...")
        baseline_performances = collect_performance_data(use_baseline=True)


        
        # Restore original epoch
        self.epoch1 = original_epoch
        
        # Plot results for both models
        valid_epochs = epochs[:len(model_performances)]
        
        # Plot model performance
        plt.plot(valid_epochs, model_performances, 'b-', linewidth=2, label='DNFC Model')
        plt.fill_between(valid_epochs, model_performances, alpha=0.2, color='blue')
        
        # Plot baseline performance
        plt.plot(valid_epochs, baseline_performances, 'g-', linewidth=2, label='Baseline Model')
        plt.fill_between(valid_epochs, baseline_performances, alpha=0.2, color='green')
        
        # Customize plot
        plt.title('Cartesian Performance Comparison Across Training Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Performance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add average lines
        model_avg = sum(model_performances) / len(model_performances)
        baseline_avg = sum(baseline_performances) / len(baseline_performances)
        
        plt.axhline(y=model_avg, color='blue', linestyle='--',
                    label=f'DNFC Avg: {model_avg:.4f}')
        plt.axhline(y=baseline_avg, color='green', linestyle='--',
                    label=f'Baseline Avg: {baseline_avg:.4f}')
        
        # Add max performance points
        model_max = max(model_performances)
        model_max_epoch = valid_epochs[model_performances.index(model_max)]
        plt.plot(model_max_epoch, model_max, 'bo',
                label=f'DNFC Max: {model_max:.4f} (Epoch {model_max_epoch})')
        
        baseline_max = max(baseline_performances)
        baseline_max_epoch = valid_epochs[baseline_performances.index(baseline_max)]
        plt.plot(baseline_max_epoch, baseline_max, 'go',
                label=f'Baseline Max: {baseline_max:.4f} (Epoch {baseline_max_epoch})')
        
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'epochs': valid_epochs,
            'model_performances': model_performances,
            'baseline_performances': baseline_performances
        }


 
