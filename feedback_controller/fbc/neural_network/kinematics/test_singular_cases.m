%Erhan Oztop, June 2021
%Checks for the singular wrist postures and IK behavior.
%Since it is based on WristPos and Endeffector (Link7) orientation, plus
%redundancy in arm kinematics, the solutions found does not seem to create
%the singular poses. Even if they do, the  analyticIK_*_LAST3 functions
%should handle these cases [not tested]


%Enter a joint configuration that has wrist singularity:
in_q = [0 0 0 0 0 90 0]*pi/180;
%Which arm to use 'left' or 'right' is allowed only.

arm = 'left'
fk_wrist_func   = sprintf('FK_%s_WRIST',arm)
fk_armtip_func   = sprintf('FK_%s_ARMTIP',arm)


if strcmp(arm,'right')
    cmd = sprintf('../Communication/ezpycom setq_rad rarm %f %f %f %f %f %f %f\n', in_q(1),in_q(2),in_q(3),in_q(4),in_q(5),in_q(6),in_q(7));
else
     cmd = sprintf('../Communication/ezpycom setq_rad larm %f %f %f %f %f %f %f\n', in_q(1),in_q(2),in_q(3),in_q(4),in_q(5),in_q(6),in_q(7));
end

system(cmd);  % if the controlserver.py is running it will send the configuration to the robot.
singTW1=feval(fk_wrist_func, 0,0,in_q(1),in_q(2),in_q(3),in_q(4),in_q(5),in_q(6),in_q(7),0); 
singP=singTW1(1:3,4);  % I got the wrist position
singTA1=feval(fk_armtip_func, 0,0,in_q(1),in_q(2),in_q(3),in_q(4),in_q(5),in_q(6),in_q(7),0); 
singR=singTA1(1:3,1:3) % I got the end effector orienatation

% So ask IK for the postion and and orientation realized above
[qfull,ee,R_last3] = simpleIKwOri(arm,[], singP, singR);

out_q = qfull(3:9)'; % extract the arm angles (IK reports also the torso joints but they are zero now)
sing_found_q_compare = [in_q; out_q]

T=feval(fk_armtip_func, 0,0,in_q(1),in_q(2),in_q(3),in_q(4),in_q(5),in_q(6),in_q(7),0); 
Tp=feval(fk_wrist_func, 0,0,in_q(1),in_q(2),in_q(3),in_q(4),in_q(5),in_q(6),in_q(7),0); 

foundR = T(1:3,1:3);
foundP = Tp(1:3,4);

singular_found_diff_R = foundR - singR
singular_found_diff_P = foundP - singP