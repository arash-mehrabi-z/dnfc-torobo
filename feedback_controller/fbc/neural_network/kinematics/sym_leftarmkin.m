% Left Arm  Forward Kinematics
% T_armlink_7 = Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45*T56*T67
% Left_arm/link_7 = Left_arm/link_tip  = Left_gripper/gripper_base  (tip is the child in the tree)
% e.g. Arm link6 frame pos & ori is given by T_link6= Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45*T56 
% Left_arm kinematic chain angles
%     TORSO         UPPERARM                 WRIST      GRIPPER
q = [0.3 0.2   1.03 0.54 1.64 0.34    0.55 -0.37 -0.59   0]*0;

syms t1 t2 a1 a2 a3 a4 a5 a6 a7 g
q = [t1 t2 a1 a2 a3 a4 a5 a6 a7 g];
TORSOix = [1 2];
ARMix = [3 4 5 6 7 8 9];
GRIPix = [10];
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


% Translation only between link2 of the torso and the should_link which
% is the same as the link_0 of the arm
Rt20  = [ 1   0  0;
          0   1  0
          0   0  1];
    
Ot20 = [0 -0.285 +0.0175];
Tt20 = makehomeg(Rt20, Ot20);

%Arm link0 frame to link1 frame
 a1 = q(ARMix(1));  %Joint 1 dependent represention in "Left_arm/link_0" frame  = "torso/Left_shoulder_link"
 R01  = [ -sin(a1)     -cos(a1)    0;
          -cos(a1)      sin(a1)    0 ;
            0           0         -1 ];
 O01 = [0, 0, 0.1775];
 T01 = makehomeg(R01, O01);
 %Check0 = quat2rotm([0.707, 0,0, -0.707])   
 %Check90 = quat2rotm([1,0, 0, 0])
 
 %Arm link1 frame to link2 frame
 a2 = q(ARMix(2));  %Joint 2 dependent represention in "Left_arm/link_1" frame
 R12  = [ -cos(a2)    sin(a2)       0;
          0          0             1;
          sin(a2)   cos(a2)       0 ];
 O12 = [-0.02, 0, 0];
 T12 = makehomeg(R12, O12);
 %Check0 = quat2rotm([0.707, -0.707, 0, 0])   
 %Check90 = quat2rotm([0.5, -0.5, 0.5, 0.5])
 %Check_32 = quat2rotm([0.7, -0.7, -0.11, -0.11])
 
 %Arm link2 frame to link3 frame
 a3 = q(ARMix(3));  %Joint 3 dependent represention in "Left_arm/link_2" frame
 R23  = [ cos(a3)   -sin(a3)       0;
          0          0             1;
         -sin(a3)   -cos(a3)       0 ];
  
 O23 = [0, -0.25, 0];
 T23 = makehomeg(R23, O23);
 %Check0 = quat2rotm([0.707, 0.707, 0, 0])   
 %Check90 = quat2rotm([0.5, 0.5, -0.5, 0.5])
 %return
 
 %Arm link3 frame to link4 frame
 a4 = q(ARMix(4));  %Joint 4 dependent represention in "Left_arm/link_3" frame
 R34  = [ cos(a4)   -sin(a4)       0;
          0          0            -1;
          sin(a4)    cos(a4)       0 ];
 O34  = [0,0,0];
 T34 = makehomeg(R34, O34);
 %Check0 = quat2rotm([0.707, -0.707, 0, 0])
 %Check90 = quat2rotm([0.5, -0.5, 0.5, 0.5])
 %Check_45 = quat2rotm([0.654, -0.645, 0.268, 0.2683])-T34

%---------------- <>
%Arm link4 frame to link5 frame    
a5=q(ARMix(5));  %Joint 5 dependent represention in "Left_arm/link_4" frame
 %syms a5; T45=rotX(pi/2)*rotZ(-pi/2+a5)
 %vpa(expand(T45),5)
 
 R45  = [ sin(a5)   cos(a5)     0;
          0         0         1.0;
          cos(a5)  -sin(a5)     0] ; 
    
 O45 = [0, -0.25,0];
 T45 = makehomeg(R45, O45);
 
 %Check0 = quat2rotm([0.5, 0.5, 0.5, -0.5])
 %Check45= quat2rotm([0.6515, 0.6515, 0.275, -0.275])
 %Check90= quat2rotm([0.707, 0.707, 0, 0])
 %Check_90= quat2rotm([0, 0, -0.707, 0.707]) 
    
 %----------------
%Arm link5 frame to link6 frame      
a6=q(ARMix(6));  %Joint 6 dependent represention in "Left_arm/link_5" frame

 
 R56  = [ cos(a6)  -sin(a6)     0;
          0         0           -1.0;
          sin(a6)   cos(a6)     0];
    
 O56 = [0, 0, 0];
 T56 = makehomeg(R56, O56);  % WRIST)T*T56 should match left_arm/link_6 origin and orientation
 
 %Check0= quat2rotm([0.707, -0.707, 0, 0])
 %Check45= quat2rotm([0.6515, 0-.6515, 0.27, 0.27])
 
 %------------------ CHECK THIS!!
 %Arm link6 frame to link7 frame
 a7=q(ARMix(7));  %Joint 7 dependent represention in "Left_arm/link_6" frame

 
 R67  = [ 0         0       -1.0;
         -cos(a7)   sin(a7)  0;
          sin(a7)   cos(a7)  0] ;
    
 O67 = [0.063, -0.1, 0];
 T67 = makehomeg(R67,O67);
 %Check0= quat2rotm([0.5, 0.5, 0.5, 0.5])
 %Check_90 = quat2rotm([0.707, 0, 0.707, 0.0])   
 
 %----
 % Left_gripper/mimic_link to gripper_base=link_tip=link_7  grip aperture dependent 
 % transformation  (not used in the full base to grasp point chain)
 
 aper = q(GRIPix(1));
 T7mim = [1  0  0 0;
          0 -1  0 -aper/2;
          0  0 -1 0
          0  0  0 1];
 %----
 % Left_gripper/finger_link to mimic_link  grip aperture dependent transformation
 %  (not used in the full base to grasp point chain)
 % T7mim and Tmimfin (i.e. mimic_link and  finger_link frames) are actually frames attached to the 
 % base of the gripper pieces. They are aper amount distant from each other
 Tmimfin = [-1 0 0 0;
             0 -1 0 -aper;
             0  0 1 0             
             0 0 0 1];
 %---
 % Left_gripper/grasping_frame to gripper_base=link_tip=link_7  fixed
 % This frame is between T7min and Tmimfin  (i.e. mimic_link and
 % finger_link frames) but ofsetted (+z axis) towards the tip of the gripper pieces
  T7grasp = [1  0  0 0;
             0 -1  0 0;
             0  0 -1 -0.16;
             0  0  0 1];
 
 WRIST_T  = Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45;  %jnt 4 affects link 5 frame pos. but not ori.
 ARMTIP_T = WRIST_T*T56*T67;                            %jnt 5 affect link 5 frame ori but not pos.
 LAST3_T = T45*T56*T67; 
 GRASP_T  = ARMTIP_T*T7grasp;
 FINGmim_T  = ARMTIP_T*T7mim;
 FINGoth_T = FINGmim_T*Tmimfin;
  
 %GRASP_T = Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45*T56*T67*T7grasp;
 %ENDEFF_T = Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45*T56*T67*T7grasp
 
 %MID_T = Twt0*Tt0t1*Tt1t2*Tt20*T01*T12*T23*T34*T45*T56*T67*T7grasp
 %MID_pos      =  MID_T(1:3,4)'
 %MID_ori_quat = rotm2quat(MID_T(1:3,1:3))
 
 
 arm = 'left';
 matdir =  './generated_code/';
 pydir  =  './generated_pycode/';
 fprintf('Generating RIGHT arm kinematics PYTHON code...\n');
 % ARM TIP Kinematics Helper Code (Python)
 fcode_py = gen_FKcode_py(ARMTIP_T, q, sprintf('%sFK_%s_ARMTIP.py',pydir,arm));
 [gJ_py, gjpycode] = gen_JACcode_py(ARMTIP_T(1:3,4), q, sprintf('%sJpos_%s_ARMTIP.py',pydir,arm), 'ARMTIP POSITION JACOBIAN' );
 
 % WRIST Kinematics Helper Code (Python) 
 wfcode_py = gen_FKcode_py(WRIST_T, q,  sprintf('%sFK_%s_WRIST.py',pydir,arm));
 [wJ_py,wjpycode]  = gen_JACcode_py(WRIST_T(1:3,4), q, sprintf('%sJpos_%s_WRIST.py',pydir,arm), 'WRIST POSITION JACOBIAN');
 
 % GRASPING POINT Kinematics Helper Code (Python)
 gfcode_py = gen_FKcode_py(GRASP_T, q,  sprintf('%sFK_%s_GRASP.py',pydir,arm));
 [gJ_py,gjpycode]   = gen_JACcode_py(GRASP_T(1:3,4), q, sprintf('%sJpos_%s_GRASP.py',pydir,arm), 'GRASP POINT POSITION JACOBIAN');
 [gzJ_py,gzjpycode] = gen_JACcode_py(GRASP_T(1:3,3), q, sprintf('%sJzaxis_%s_GRASP.py',pydir,arm), 'GRASP POINT ZAXIS JACOBIAN');
 [gyJ_py,gyjpycode] = gen_JACcode_py(GRASP_T(1:3,2), q, sprintf('%sJyaxis_%s_GRASP.py',pydir,arm),'GRASP POINT YAXIS JACOBIAN');
 
 %return
 
 % LAST3 wrist angle Kinematics Helper Code (Matlab)
 %lfcode = gen_FKcode(LAST3_T, [a5,a6,a7], './generated_code/FK_right_LAST3.m');
 
 % Now Matlab code
 fprintf('Generating RIGHT arm kinematics MATLAB code...\n');
 % ARM TIP Kinematics Helper Code (Matlab)
 afcode = gen_FKcode(ARMTIP_T, q,  sprintf('%sFK_%s_ARMTIP.m',matdir,arm));
 [aJ,ajcode]  = gen_JACcode(ARMTIP_T(1:3,4), q, sprintf('%sJpos_%s_ARMTIP.m',matdir,arm),'ARM TIP POSITION JACOBIAN' );


 % WRIST Kinematics Helper Code (Matlab) 
 wfcode = gen_FKcode(WRIST_T, q, sprintf('%sFK_%s_WRIST.m',matdir,arm));
 [wJ,wjcode]  = gen_JACcode(WRIST_T(1:3,4), q, './generated_code/Jpos_right_WRIST.m', 'WRIST POSITION JACOBIAN');

 % GRASPING POINT Kinematics Helper Code (Matlab)
 gfcode = gen_FKcode(GRASP_T, q, sprintf('%sFK_%s_GRASP.m',matdir,arm));
 [gJ,gjcode]  = gen_JACcode(GRASP_T(1:3,4), q, sprintf('%sJpos_%s_GRASP.m',matdir,arm),'GRASP POINT POSITION JACOBIAN' );
 [gzJ,gzjcode]  = gen_JACcode(GRASP_T(1:3,3), q, sprintf('%sJzaxis_%s_GRASP.m',matdir,arm),'GRASP POINT Z-AXIS JACOBIAN');
 [gyJ,gyjcode]  = gen_JACcode(GRASP_T(1:3,2), q, sprintf('%sJyaxis_%s_GRASP.m',matdir,arm), 'GRASP POINT Y-AXIS JACOBIAN');
 return

 
 
% %  
% %  lfcode = gen_FKcode(LAST3_T, [a5,a6,a7], 'FK_left_LAST3.m');
% %  %[J,jcode]  = gen_oriJcode(LAST3_T, q, 'oriJ_left_LAST3.m');
% %  
% % %pyfcode = gen_FKcode_py(ARMTIP_T, q, 'FK_left_ARMTIP.py');
% %  
% % afcode = gen_FKcode(ARMTIP_T, q, './generated_code/FK_left_ARMTIP.m');
% % [aJ,ajcode]  = gen_posJcode(ARMTIP_T, q, './generated_code/posJ_left_ARMTIP.m');
% %  
% % wfcode = gen_FKcode(WRIST_T, q, './generated_code/FK_left_WRIST.m');
% % [wJ,jcode]  = gen_posJcode(WRIST_T, q, './generated_code/posJ_left_WRIST.m');
% % 
% % 
% % gfcode = gen_FKcode(GRASP_T, q, './generated_code/FK_left_GRASP.m');
% % [gJ,gjcode]  = gen_posJcode(GRASP_T, q, './generated_code/Jpos_left_GRASP.m');
% % [gzJ,gzjcode]  = gen_ZaxisJcode(GRASP_T, q, './generated_code/Jzaxis_left_GRASP.m');
% % [gyJ,gyjcode]  = gen_YaxisJcode(GRASP_T, q, './generated_code/Jyaxis_left_GRASP.m');
% % 
% % %fcode = gen_FKcode(GRASP_T, q, 'FK_left_GRASP.m');
% % %[J,jcode]  = gen_posJcode(GRASP_T, q, 'posJ_left_GRASP.m');

 