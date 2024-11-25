function [q, pos_err,zax_err, TT] = IKwZAXIS(arm, qinit, qweight, p_des, R_des, errTH)
% Erhan Oztop June 2021. 
% Solves inverse kinematics for wrist poistion and end effector orientation. 
% IK with Jacobian transpose is used for the wrist. Last three angles are found analytically.
% INPUTS
% q: initial joint angles for iterative IK
% p_des: desired wrist position
% R_des: desired Transformation matrix (desire x, y, z axis in the columns)
% arm  : right  or left
% err_TH: the required accuracy for position IK
% q_weight: (Optinal) a real positive vector selecting/scaling the joints used in iterative IK.
% OUTPUTS
% q: found joint angles
% pos_error : positional error that could be achieved
% R_last3: is the local (wrt frame 4) rotation matrix realized by the found last 3 arm angles. 
% EXAMPLE:
%   z=[0  1  -1]; y =[1 0 0]; R_des = [cross(y,z)', y', z'];  [q,ee,R_last3] = IKwZAXIS('right',[],[], [0.4, 0, 1.0], R_des);
%   z=[0 -1  -1]; y =[1 0 0]; R_des = [cross(y,z)', y', z'];  [q,ee,R_last3] = IKwZAXIS('left',[],[], [0.4, 0, 1.0], R_des);
% Note: First two joint: Torsa; Last q value i.e. q[10] is the gripper translational position which is not used.

path('./generated_code',path);
DEF_err_TH = 0.5/1000.0;   %  0.5mm
DEF_qweight = [0 0 1 1 1 1 1 1 1 0]';  % set first two weight to zero for no torso. Last zero is for the gripper. 
DEF_qinit   = [0 0 90 90 0 90  0 0 0 0]'*pi/180;
STEP_RATE = 2;
POS_W         = .75;      % ORI_W = 1-POS_w   (so .75 means Position updates are 3 times stronger)
MAXIT = 3000;
USE_PINV = 0;
it = 0;

syms x
R_des(:,1) = R_des(:,1)/norm(R_des(:,1));
R_des(:,2) = R_des(:,2)/norm(R_des(:,2));
R_des(:,3) = R_des(:,3)/norm(R_des(:,3));
% Note this is probably not a sufficient check
if abs(det(R_des)-1)>1e-5 || norm(cross(R_des(:,1),R_des(:,2))-R_des(:,3))>1e-5
    fprintf('This doesnot seem to be a proper rotation matrix!\n');
    R_des = R_des
    return
end
%fk_wrist_func   = sprintf('FK_%s_WRIST',arm);
%Jpos_wrist_func = sprintf('Jpos_%s_WRIST',arm);
%fk_armtip_func   = sprintf('FK_%s_ARMTIP',arm);
%Jpos_armtip_func = sprintf('Jpos_%s_ARMTIP',arm);

fk_grasp_func   = sprintf('FK_%s_GRASP',arm);
Jpos_grasp_func = sprintf('Jpos_%s_GRASP',arm);
Jzaxis_grasp_func = sprintf('Jzaxis_%s_GRASP',arm);
Jyaxis_grasp_func = sprintf('Jyaxis_%s_GRASP',arm);


q_min = [-170 -60 -70 -35 -70 -45 -170 -105 -170]'*pi/180*0.95;
q_max = [170  80 250 105 250 120 170 90 170]'*pi/180*0.95;

q_min = [q_min; 0];     % gripper min in meters.
q_max = [q_max; 0.08];  % gripper max in meters.

if isempty(qinit)
    qinit = DEF_qinit;
end

if isempty(qweight)
    qweight = DEF_qweight;
end

if ~exist('err_TH','var')
    errTH = DEF_err_TH;
end
q = qinit;
q = reshape(q,length(q),1);

p_des = reshape(p_des, length(p_des),1);
z_des = R_des(1:3,3);
y_des = R_des(1:3,2);
q_at_start = q;



T = feval(fk_grasp_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));  
p_cur = T(1:3,4); z_cur = T(1:3,3); y_cur = T(1:3,2);
J = feval(Jpos_grasp_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
Jz = feval(Jzaxis_grasp_func,  q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
pos_err = norm(p_des - p_cur);
zax_err = norm(z_des - z_cur);
yax_err = norm(z_des - y_cur);

while (pos_err > errTH || zax_err > errTH) && it < MAXIT
    it = it + 1;
    T = feval(fk_grasp_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));    
    p_cur = T(1:3,4); z_cur = T(1:3,3);  y_cur = T(1:3,2);
    
    %R_cur = T(1:3,1:3);
    %R_a = R_cur'*R_des;
    %q_last = atan2(R_a(2,1),R_a(1,1))
    
    J = feval(Jpos_grasp_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
    Jz = feval(Jzaxis_grasp_func,  q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
    Jy = feval(Jyaxis_grasp_func,  q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
    
    % You can save computation time and decouple IKs by using smaller J's
    % This would be the result
    %J(:,7:end) = 0;
    %Jz(:,1:6)  = 0; 
    %Jy(:,1:6);
    
    dp = p_des - p_cur;
    dz = z_des - z_cur;
    dy = y_des - y_cur;
    pos_err = norm(p_des - p_cur);
    zax_err = norm(z_des - z_cur);
    yax_err = norm(y_des - y_cur);
    if (pos_err<=errTH && zax_err<=errTH)   % TODO: not checking yaxis!! Decouple pos and ori errors!
        fprintf("Found solution!\n");
        continue; 
    end
    
    if USE_PINV == 0
        dq_pos = STEP_RATE*J'*dp;
        dq_zax = STEP_RATE*Jz'*dz;
        dq_yax = STEP_RATE*Jy'*dy;
    else
        SC = 0.2;
        dq_pos = SC*STEP_RATE*pinv(J)*dp;
        dq_zax = SC*STEP_RATE*pinv(Jz)*dz;
        dq_yax = SC*STEP_RATE*pinv(Jy)*dy;
    end
    dq     = dq_pos*POS_W + (1-POS_W)*0.5*(dq_zax+0.2*dq_yax);
    q = q + dq.*qweight;
    
       
    
    lobo_viol = (q < q_min);
    hibo_viol = (q > q_max);
    q = max(q_min, q);
    q = min(q, q_max);
    if sum(lobo_viol)>0, fprintf('Joint LOwer limit hit at:%s\n', vec2str(lobo_viol,'%d')); end
    if sum(hibo_viol)>0, fprintf('Joint UPper limit hit at:%s\n', vec2str(hibo_viol,'%d')); end
     
    if mod(it,50)==5 send_to_controlserver(q,arm); end
    if mod(it,20)==0 fprintf('%d > pos, ori error:%2.4f, %2.4f\n',it, pos_err, zax_err); end
end

if it == MAXIT
    fprintf('IK did not succeed. The current pos, ori error:%2.4f, %2.4f\n',pos_err, zax_err);
    p_des_vs_p_cur_Del = [p_des,     p_cur, p_des - p_cur]
    z_des_vs_z_cur_Del = [z_des,     z_cur, z_des - z_cur]
    des_reached = 0;
else
    fprintf('Solved. \n Try: \n FK_right_WRIST(%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f)\n',q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
    des_reached = 1;
end

if (~des_reached),  
    fprintf('Warning desired position accuracy could not be reached (pos_err:%4.2fmm, zax_err:%4.2f) in %d iterations.\n',pos_err*1000,zax_err,it); 
else
    fprintf('Desired position accuracy reached (pos_err:%4.2fmm, zax_err:%4.2f) in %d iterations.\n',pos_err*1000,zax_err, it)
end
TT = feval(fk_grasp_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));

found_T = TT;
asked_T = [[R_des; 0 0 0],[p_des; 1]];
found_asked_T_diff = found_T - asked_T

%Tx = vpa(feval(fk_grasp_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), x, q(10)),4); 

send_to_controlserver(q,arm);



function send_to_controlserver(q, arm)
qq = q*180/pi;
if strcmp(arm,'right')
    cmd = sprintf('../communication/ezpycom setqT_deg rarm %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n', qq(1), qq(2), qq(3),qq(4),qq(5),qq(6),qq(7),qq(8),qq(9));
else
    cmd = sprintf('../communication/ezpycom setqT_deg larm %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n', qq(1), qq(2), qq(3),qq(4),qq(5),qq(6),qq(7),qq(8),qq(9));
end
fprintf('sending %s\n',cmd);
system(cmd);
