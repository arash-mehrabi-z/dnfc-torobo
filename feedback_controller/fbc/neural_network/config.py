class Config:
    def __init__(self):
        # Custom Loss:
        self.use_custom_loss = False
        self.num_steps = 299
        self.C = 1e-5

        self.v_name = "just_state_no_enc" #"6l_linear" #"v_custl_mse"
        self.v_name_base = "3l_base" #"4l_base" 

        self.episodes_num_ds = 72 #500 #360 #2000
        self.dataset_name = f"trajs:{self.episodes_num_ds}_blocks:3" +\
            "_triangle_v_scarce" #"_random"
        self.ds_ratio = "interp_0.85" #"extrap_0.85" #"interp_0.95" #0.263
        self.ds_file_name = f'train_{self.ds_ratio}_with_2poses.npy'
        self.ds_ratio_test = self.ds_ratio #"interp_0.85"
        self.ds_test_file = f'test_{self.ds_ratio_test}.npy'
        self.train_val_file = f'split_indices_{self.ds_ratio}.pt'

        self.joints_num = 7
        self.state_dim = 2*self.joints_num
        self.coords_dim = 3*3
        self.action_dim = self.joints_num
        self.onehot_dim = 4
        self.step_dim = 1
        self.ee_dim = 12  # 2 consecutive end-effector poses: (pos 3 + euler 3) * 2
        self.encoded_space_dim = 14 #128 #32  # Latent space dimension for encoders

        # self.num_params = 24.085 #6.037 #36.117
        # self.num_params_base = 24.091 #6.039 #36.109

        
    def get_model_name(self, use_baseline, use_custom_loss, use_image):
        if use_custom_loss: 
            model_name = f"cus_los_{self.C}"
            # model_name = f"cus_los_const_mse_st"
        else: model_name = "mse_los"
        if use_image: model_name += "|tar_img"
        else: model_name += "|tar_cart"
        if use_baseline: model_name += f"|base|{self.v_name_base}"
        else: model_name += f"|{self.v_name}"
        return model_name
    
    
    # def add_params_to_name(self, name, model):
    #     num_params = sum(p.numel() for p in model.parameters())/1e3
    #     name += f"|{num_params}K_params"
    #     return name
    
        # if use_baseline:
        #     num_params = self.num_params_base
        # else:
        #     num_params = self.num_params
        # name += f"|{num_params}K_params"
        # return name

    
    def get_params_num(self, model):
        num_params = sum(p.numel() for p in model.parameters())/1e3
        return num_params
    

    def get_model_dims(self, model_complexity):
        if model_complexity == 'low':
            enc_hid = 128 // 2
            cont_hid = 384 // 2
            lin_hid = 60
            lin_out = 64
        elif model_complexity == 'medium':
            enc_hid = 128
            cont_hid = 384
            lin_hid = 74
            lin_out = 121
        elif model_complexity == 'high':
            enc_hid = 128 * 2
            cont_hid = 384 * 2
            lin_hid = 2*27-3
            lin_out = 192 * 2
        elif model_complexity == 'xhigh':
            enc_hid = 128 * 4
            cont_hid = 384 * 4
            lin_hid = 3*27
            lin_out = int(192 * 2.7)
        elif model_complexity == 'XXhigh':
            enc_hid = 128 * 8
            cont_hid = 384 * 8
            lin_hid = 3*27
            lin_out = int(192 * 2.7)
        else:
            raise Exception("Model complexity is not defined.")
        
        return enc_hid, cont_hid, lin_hid, lin_out
