import matplotlib.pyplot as plt
import numpy as np
from torkin import TorKin
import torch
from nn_models import GeneralModel, MLPBaseline
import torch.nn as nn
import math
import os
from config import Config


class Tester():
    def __init__(self) -> None:
        self.config = Config()

        self.joint_size = self.config.joints_num
        self.state_size = self.config.state_dim
        self.step_size = self.config.step_dim
        self.target_size = self.config.coords_dim
        self.onehot_size = self.config.onehot_dim
        
        self.cur_file_dir_path = os.path.dirname(__file__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.load_model(train_no=0, epoch_no=0, use_custom_loss=False)
        
        dataset_path = os.path.join(self.cur_file_dir_path, 
                                    f'data/torobo/{self.config.dataset_name}/{self.config.ds_test_file}')
        self.dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1')
        print("Tester loaded dataset with shape:", self.dataset.shape)
        print("from this path", dataset_path)

        self.kin = TorKin()
        self.criterion = nn.L1Loss()
        self.criterion_mse = nn.MSELoss(reduction='sum')

    def load_model(self, train_no, epoch_no, use_custom_loss):
        model_name_dnfc = self.config.get_model_name(False, use_custom_loss, False)
        model_name_dnfc = self.config.add_params_to_name(model_name_dnfc, False)
        model_name_base = self.config.get_model_name(True, False, False)
        model_name_base = self.config.add_params_to_name(model_name_base, True)

        dnfc_adr = f'weights/{self.config.dataset_name}|{self.config.ds_ratio}|{model_name_dnfc}' + \
            f'/train_no_{train_no}/fbc_{epoch_no}.pth'
        base_adr = f'weights/{self.config.dataset_name}|{self.config.ds_ratio}|{model_name_base}' + \
            f'/train_no_{train_no}/fbc_{epoch_no}.pth'

        self.model = GeneralModel(self.state_size, self.target_size+self.onehot_size, 
                                  self.joint_size, use_image=False)
        # self.baseline = MLPBaseline(self.state_size + (self.target_size+self.onehot_size), 
        #                             self.joint_size)
        m = self.model.to(self.device)
        # m = self.baseline.to(self.device)

        dnfc_path = os.path.join(self.cur_file_dir_path, dnfc_adr)
        base_path = os.path.join(self.cur_file_dir_path, base_adr)
        self.model.load_state_dict(torch.load(dnfc_path, 
                                              map_location=torch.device(self.device),
                                              weights_only=True))
        # self.baseline.load_state_dict(torch.load(base_path, 
        #                                          map_location=torch.device(self.device),
        #                                          weights_only=True))
        print("***\nLoaded model weights from", dnfc_path)
        print("Loaded baseline weights from", base_path)


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
            goal=elem[i][self.step_size+self.state_size+self.joint_size:].tolist()
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
    

    def get_emulated(self, use_baseline, num, use_angle=False, return_path_point=False):
        out = False
        y1, y2, y3, y4, y5, y6, y7 = [], [], [], [], [], [], []
        elem = self.dataset[num]
        state = torch.tensor(elem[0][self.step_size : 
                                     self.step_size+self.state_size].tolist()
                                     ).to(self.device)

        goal = elem[0][self.step_size + self.state_size : 
                       self.step_size + self.state_size + self.target_size].tolist()
        one_hot = elem[0][self.step_size + self.state_size + self.target_size :
                          self.step_size + self.state_size + self.target_size + self.onehot_size
                          ].tolist()
        
        milestones = self.get_changes_indexes(num)

        obstA = torch.tensor(goal[0:3])
        obstB = torch.tensor(goal[3:6])
        obstC = torch.tensor(goal[6:9])
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
            print(goal[:3])
            print('this is B')
            print(goal[3:6])
            print('this is C')
            print(goal[6:])

        if use_baseline:
            self.baseline.eval()
        else:
            self.model.eval()
        
        for i in range(self.dataset.shape[1]):
            goal_tensor = torch.tensor(goal + one_hot).to(self.device)
            goal_nn = torch.unsqueeze(goal_tensor, 0)
            state_nn = torch.unsqueeze(state, 0)
            if use_baseline:
                basel_input = torch.cat((goal_nn, state_nn), dim=1)
                velocities_tensor = self.baseline(basel_input)
            else:
                velocities_tensor, x_des, _ = self.model(goal_nn, state_nn)

            velocities_tensor = torch.squeeze(velocities_tensor, 0)
            print_it_out = out

            state[:7] += velocities_tensor
            state[7:] = velocities_tensor

            if path_point==0 and self.close_enough(state, grepB):
                # print(i)
                path_point = 1
                one_hot = [1, 0, 0, 0]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            elif path_point==1 and self.close_enough(state, putB):
                path_point = 2
                one_hot = [0, 1, 0, 0]
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            elif path_point==2 and self.close_enough(state, grepC):
                path_point = 3
                one_hot = [0, 0, 1, 0]           
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            elif path_point==3 and self.close_enough(state, putC):
                path_point = 4
                one_hot=[0, 0, 0, 1]   
                if print_it_out:
                    print(x_des)
                    print(self.get_end_eff(x_des))

            if use_angle:
                add = velocities_tensor
            else:
                add = state
                
            y1.append(float(add[0]))
            y2.append(float(add[1]))
            y3.append(float(add[2]))
            y4.append(float(add[3]))
            y5.append(float(add[4]))
            y6.append(float(add[5]))
            y7.append(float(add[6]))

        if return_path_point:
            return y1,y2,y3,y4,y5,y6,y7,path_point
        else:
            return y1,y2,y3,y4,y5,y6,y7
        

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
        
        for i in range(299):
            goal_tensor=torch.tensor(goal+one_hot)

            if usebaseline:
                all=torch.cat((goal_tensor, state),dim=0)
                velocities_tensor=self.baseline(all)
            else:
                velocities_tensor,x_des=self.model(goal_tensor, state)[0:2]

            
            state[:7]+=velocities_tensor
            state[7:]=velocities_tensor
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


    
    def get_coordinats(self, num, use_baseline):
        y1, y2, y3, y4, y5, y6, y7 = self.get_emulated(use_baseline, num, 
                                                       use_angle=False)
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
        for j in self.dataset[num][0:]:
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
        if dist <= 0.015:#0.02:
            return True
        return False


    def get_obs_coordinates(self, num):
        elem = self.dataset[num]
        obstA = elem[1][1+self.state_size:1+self.state_size+(1*3)]
        obstB = elem[1][1+self.state_size+(1*3):1+self.state_size+(2*3)]
        obstC = elem[1][1+self.state_size+(2*3):1+self.state_size+(3*3)]

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
        for i in range(299):
            goal = elem[i][self.step_size+self.state_size :
                           self.step_size+self.state_size+self.target_size+self.onehot_size
                           ].tolist()
            for j in range(self.onehot_size):
                if goal[self.target_size + j] != start[j]:
                    indexes.append(i)
                    start = goal[self.target_size:]
        return indexes

    def calculate_cartesian_perform(self, use_baseline):
        point_reached = 0
        for num in range(self.dataset.shape[0]): 
            point_reached += self.get_emulated(use_baseline, num, 
                                               False, True)[7]
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
   
    



