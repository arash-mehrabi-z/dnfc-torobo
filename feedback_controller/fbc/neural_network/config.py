class Config:
    def __init__(self):
        # Custom Loss:
        self.use_custom_loss = True
        self.num_steps = 299
        self.C = 1e-5

        self.v_name = "v_custl_mse" #"v_base" #"v_klloss"
        self.v_name_base = "v_base"

        self.episodes_num_ds = 360
        self.dataset_name = f"trajs:{self.episodes_num_ds}_blocks:3" +\
            "_triangle_v"
        self.ds_ratio = 0.8
        self.ds_file_name = f'train_{self.ds_ratio}.npy'
        self.ds_test_file = f'test_{self.ds_ratio}.npy'

        self.joints_num = 7
        self.state_dim = 2*self.joints_num
        self.coords_dim = 3*3
        self.action_dim = self.joints_num

        self.onehot_dim = 4
        self.step_dim = 1

        self.num_params = 25.301 #7.541 #3.733 #25.301  #25.045 #25.301 # 294.037
        self.num_params_base = 25.175 #7.431 #3.753  #293.631

        
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

