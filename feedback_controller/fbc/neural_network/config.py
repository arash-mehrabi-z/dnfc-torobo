class Config:
    def __init__(self):
        # Custom Loss:
        self.use_custom_loss = False #True
        self.num_steps = 299
        self.C = 1e-5

        # If True, train/eval on raw (un-normalized) actions: the dataset
        # multiplies the stored normalized actions back by action_std, and the
        # testers skip the matching de-normalization. See TrajectoryDataset and
        # the `* action_std` sites in testers.py / online_tester.py.
        self.raw_actions = True

        self.v_name = "img_task_enc_rob_enc_raw_action" #"cart_task_enc_conc" #"2+2l_lat:sub-nvel"
        self.v_name_base = "3l_base" #"4l_base"
        self.v_name_two_stream = "two_stream_two_encs_concat_latent"
        self.num_history_images = 3
        self.image_size = (128, 128)

        self.episodes_num_ds = 360 #72 #500 #360 #2000
        self.dataset_name = f"trajs:{self.episodes_num_ds}_blocks:3" +\
            "_imgs_2cams" #"_tri_img" #"_triangle_v_scarce" #"_random"
        self.ds_ratio = "interp_0.85" #"ds" #"extrap_0.85" #"interp_0.95" #0.263
        self.ds_file_name = f'train_{self.ds_ratio}.npy'
        self.ds_ratio_test = self.ds_ratio #"interp_0.85"
        self.ds_test_file = f'test_{self.ds_ratio_test}.npy'
        self.train_val_file = f'split_indices_{self.ds_ratio}.pt'

        self.step_dim = 1
        self.joints_num = 7
        self.state_dim = 2*self.joints_num
        self.coords_dim = 3 * 2 #3*3
        self.action_dim = self.joints_num
        self.onehot_dim = 4
        self.ee_pose_dim = 7  # 3 position + 4 quaternion

        # self.num_params = 24.085 #6.037 #36.117
        # self.num_params_base = 24.091 #6.039 #36.109

        
    def get_model_name(self, use_baseline, use_custom_loss, use_image,
                        use_two_stream=False):
        if use_custom_loss:
            model_name = f"cus_los_{self.C}"
            # model_name = f"cus_los_const_mse_st"
        else: model_name = "mse_los"
        if use_two_stream: model_name += "|two_stream_basel"
        else: pass
        if use_image: model_name += "|tar_img_static"
        else: model_name += "|tar_cart"
        if use_two_stream: model_name += f"|{self.v_name_two_stream}"
        elif use_baseline: model_name += f"|base|{self.v_name_base}"
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
        else:
            raise Exception("Model complexity is not defined.")
        
        return enc_hid, cont_hid, lin_hid, lin_out

    def get_two_stream_dims(self, model_complexity):
        if model_complexity == 'low':
            mlp_hidden_1 = 32
            mlp_hidden_2 = 24
            mlp_latent = 16
            cnn_latent = 32
            decoder_hidden_1 = 32
            decoder_hidden_2 = 24
        elif model_complexity == 'medium':
            mlp_hidden_1 = 64
            mlp_hidden_2 = 48
            mlp_latent = 32
            cnn_latent = 64
            decoder_hidden_1 = 64
            decoder_hidden_2 = 48
        elif model_complexity == 'high':
            mlp_hidden_1 = 128
            mlp_hidden_2 = mlp_hidden_1
            mlp_latent = 32
            cnn_latent = 128
            decoder_hidden_1 = 512
            decoder_hidden_2 = 384
        elif model_complexity == 'xhigh':
            mlp_hidden_1 = 256
            mlp_hidden_2 = 192
            mlp_latent = 128
            cnn_latent = 256
            decoder_hidden_1 = 256
            decoder_hidden_2 = 192
        else:
            raise Exception("Model complexity is not defined.")

        return mlp_hidden_1, mlp_hidden_2, mlp_latent, cnn_latent, decoder_hidden_1, decoder_hidden_2

    def get_image_model_dims(self, model_complexity):
        """Get dimensions for GeneralModel with use_image=True."""
        if model_complexity == 'low':
            cnn_latent = 64
            cont_hid = 192
            pose_enc_hid = 32
        elif model_complexity == 'medium':
            cnn_latent = 128
            cont_hid = 384
            pose_enc_hid = 64
        elif model_complexity == 'high':
            cnn_latent = 128
            cont_hid = 384 * 2
            pose_enc_hid = 64
        elif model_complexity == 'xhigh':
            cnn_latent = 256
            cont_hid = 1536
            pose_enc_hid = 128
        else:
            raise Exception("Model complexity is not defined.")

        return cnn_latent, cont_hid, pose_enc_hid
