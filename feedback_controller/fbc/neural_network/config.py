class Config:
    def __init__(self):
        # Custom Loss:
        self.use_custom_loss = True
        self.num_steps = 299
        self.C = 1e-5

        self.v_name = "2+2l_lat:sub-nvel" #"6l_linear" #"v_custl_mse" + #"v_klloss"
        self.v_name_base = "3l_base" #"4l_base" 

        self.episodes_num_ds = 500 #360
        self.dataset_name = f"trajs:{self.episodes_num_ds}_blocks:3" +\
            "_random" #+ "_v"
        self.ds_ratio = 0.263
        self.ds_file_name = f'train_{self.ds_ratio}.npy'
        self.ds_ratio_test = 0.95
        self.ds_test_file = f'test_{self.ds_ratio_test}.npy'

        self.joints_num = 7
        self.state_dim = 2*self.joints_num
        self.coords_dim = 3*3
        self.action_dim = self.joints_num

        self.onehot_dim = 4
        self.step_dim = 1

        self.num_params = 12.053 #36.117 #135.415 #13.333 #25.301
        self.num_params_base = 12.103 #36.023 #12.323 #13.371 #25.175

        
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
    
    
    def add_params_to_name(self, name, use_baseline):
        if use_baseline:
            num_params = self.num_params_base
        else:
            num_params = self.num_params
        name += f"|{num_params}K_params"
        return name

