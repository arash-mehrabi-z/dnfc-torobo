% Head Forward Kinematics
% T_armlink_7 = Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45*T56*T67
% right_arm/link_7 = right_arm/link_tip  = right_gripper/gripper_base  (tip is the child in the tree)
% e.g. Arm link6 frame pos & ori is given by T_link6= Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45*T56 
% right_arm kinematic chain angles
%     TORSO     HEAD
q = [0.3 0.2   1.03 0.54]*0
syms t1 t2 h1 h2
q = [t1 t2 h1 h2];
TORSOix = [1 2];
HEADix = [3 4 ];

%Torso to World/Base/Foot
Rwt0 = [1 0 0;
        0 1 0;
        0 0 1];
    
Owt0 = [0,0, 0.74];

Twt0 = makehomeg(Rwt0, Owt0);

%Torso link1 frame to link 0 frame
t1 = q(TORSOix(1));  %Torso joint 1 dependent represention in "torso/link_0" frame  = "torso/link_0"
Rt0t1 = [  cos(t1)  -sin(t1)   0; 
           sin(t1)   cos(t1)   0;
           0         0         1];
      
Ot0t1 = [0 0 0.29];
Tt0t1 = makehomeg(Rt0t1, Ot0t1);

%Check0 = quat2rotm([1,0,0,0]) 
%Check_52 = quat2rotm([0.965, 0,0, -0.26]) 

%Torso link2 frame to link 1 frame
t2 = q(TORSOix(2)) ;  %torso joint 2 dependent represention in "torso/link_1" frame  = "torso/link_1"
Rt1t2 = [  cos(t2)  -sin(t2)   0; 
           0            0      1;
          -sin(t2)  -cos(t2)  0];
      
Ot1t2 = [0 0 0]; 
Tt1t2 = makehomeg(Rt1t2, Ot1t2);
%Check0 = quat2rotm([0.707, -0.707,0,0]) 
%Check45 = quat2rotm([0.654, -0.654,0.268,0.268]) 

%Torso link_tip (=head link0) to link 2 frame
Ot2t_tip = [0 -0.3945 0]
Rt2t_tip = [1   0    0
            0   0   -1
            0   1    0];  % 90deg rotation around X axis of torso link 2 frame
Tt2t_tip = makehomeg(Rt2t_tip, Ot2t_tip);       
Tt2h0    = Tt2t_tip; % torso/link_tip is identical to head/link0

%Head/link1 to head/link0 (joint h1 controls the xform)
Oh0h1 =[0 0 0];
Rh0h1 = [cos(h1) -sin(h1) 0
         sin(h1)  cos(h1) 0
            0       0     1];
Th0h1 = makehomeg(Rh0h1, Oh0h1);

%Head/link2 to head/link 1 (joint h2 controls the xform)
Oh1h2 =[-0.035 0 0];
Rh1h2 = rotX(-pi/2)*[cos(h2) -sin(h2) 0
         sin(h2)  cos(h2) 0
            0       0     1];
Th1h2 = makehomeg(Rh1h2, Oh1h2);

% Head/camera_link to Head/Link2 (or Linktip, they are identical)
% constant offset and rotation only
Oh2c =[0.076 -0.105 0];
Rh2c = rotX(pi/2);
Th2c = makehomeg(Rh2c, Oh2c);

% Head/camera_cam_col_fr to Head/camera_link
Occof =[0.02 0 0];
Rccof = rotX(0); % identity
Tccof = makehomeg(Rccof, Occof);

% Head/camera_cam_col_fr to Head/camera_link
Ocofcop =[0 0 0];
Rcofcop = [     0     0     1
               -1     0     0
                0    -1     0];
Tcofcop = makehomeg(Rcofcop, Ocofcop);

% Head/camera_cam_aligned_depth_to_color_fr to Head/camera_link
Odeptcam =[-0.0034 0.0257 0];
Rdeptcam = [     1     0     0
                0     1     0
                0     0     1];  %unit
Tdeptcam = makehomeg(Rdeptcam, Odeptcam);
% BTW. Head/camera_cam_aligned_depth_to_infra1_fr to Head/camera_link  IS
% IDENTITY!

% Head/camera_cam_color_optical_fr to Head/camera_cam_aligned_depth_to_color_fr
Occamopt = [0 0 0];
Rccamopt  = rotZ(-pi/2)*rotX(-pi/2);  % =[0 0 1; -1 0 0; 0 -1 0]
Tdeptcam = makehomeg(Rccamopt, Occamopt);

HEAD_T = Twt0*Tt0t1*Tt1t2*Tt2h0*Th0h1*Th1h2;  % head/link_2 (or link_tip) in World
CAML_T = HEAD_T*Th2c;                         % head/camera_link in World
CAMCOF_T = CAML_T*Tccof;                      % head/camera_color_frame
CAMCOP_T = CAMCOF_T*Tcofcop;                  % head/camera_color_optical_frame

COLOPT_T = CAML_T * Tdeptcam ;      % Head/camera_cam_color_optical_fr  in World

% Note the last two reference frames are simply shifted 2cm in the
% lefteye-to-right eye axis. If the robot is looking straight then this
% creates an world coordinate difference in the Y-axis. 
% Check which one is used by te PointCloud2 topic so that object locations
% can be inferre correctly.
 
 part = 'head';
 matdir =  './generated_code/';
 pydir  =  './generated_pycode/';
 fprintf('Generating HEAD kinematics PYTHON code...\n');
 % HEAD Kinematics Helper Code (Python)
 %fcode_py = gen_FKcode_py(HEAD_T, q, sprintf('%sFK_%s_HEAD.py',pydir,part));

 gfcode_py = gen_FKcode_py(HEAD_T, q,  sprintf('%sFK_%s_HEAD.py',pydir,part));
 [gJ_py,gjpycode]   = gen_JACcode_py(HEAD_T(1:3,4), q, sprintf('%sJpos_%s_HEAD.py',pydir,part), 'HEAD POINT POSITION JACOBIAN');
 [gzJ_py,gzjpycode] = gen_JACcode_py(HEAD_T(1:3,3), q, sprintf('%sJzaxis_%s_HEAD.py',pydir,part), 'HEAD POINT ZAXIS JACOBIAN');
 [gyJ_py,gyjpycode] = gen_JACcode_py(HEAD_T(1:3,2), q, sprintf('%sJyaxis_%s_HEAD.py',pydir,part),'HEAD POINT YAXIS JACOBIAN');
 
 

 % Now Matlab code
 fprintf('Generating HEAD  kinematics MATLAB code...\n');
 % HEAD TIP Kinematics Helper Code (Matlab)
 %afcode = gen_FKcode(HEAD_T, q,  sprintf('%sFK_%s_HEAD.m',matdir,part));
 
 %[aJ,ajcode]  = gen_JACcode(ARMTIP_T(1:3,4), q, sprintf('%sJpos_%s_ARMTIP.m',matdir,part),'ARM TIP POSITION JACOBIAN' );
 gfcode = gen_FKcode(HEAD_T, q, sprintf('%sFK_%s_HEAD.m',matdir,part));
 [gJ,gjcode]  = gen_JACcode(HEAD_T(1:3,4), q, sprintf('%sJpos_%s_HEAD.m',matdir,part),'HEAD POINT POSITION JACOBIAN' );
 [gzJ,gzjcode]  = gen_JACcode(HEAD_T(1:3,3), q, sprintf('%sJzaxis_%s_HEAD.m',matdir,part),'HEAD POINT Z-AXIS JACOBIAN');
 [gyJ,gyjcode]  = gen_JACcode(HEAD_T(1:3,2), q, sprintf('%sJyaxis_%s_HEAD.m',matdir,part), 'HEAD POINT Y-AXIS JACOBIAN');
 
