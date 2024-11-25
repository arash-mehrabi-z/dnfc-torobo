function [q, pos_err, R_last3] = simpleIKwOri(arm, qinit, p_des, R_des, errTH, q_weight)
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
%   z=[0  -1 0]; y =[0 0 1]; R_des = [cross(y,z)', y', z'];  [q,ee,R_last3] = simpleIKwOri('right',[], [0.30,-0.25,1.2], R_des);
%   R_des=[ [-1;0;0] [0;0;-1] [0;-1;0] ]; [q,ee,R_last3] = simpleIKwOri('left',[], [0.30,0.25,1.2], R_des);
% Note: First two joint: Torsa; Last q value i.e. q[10] is the gripper translational position which is not used.

path('./generated_code',path);
DEF_err_TH = 0.5/1000.0;   %  0.5mm
DEF_qenable = [0 0 1 1 1 1 0 0 0 0]';  % no torso 
DEF_qinit   = [0 0 60 60 0 60  0 0 0 0]'*pi/180;
STEP_RATE = 1;
MAXIT = 5000;
it = 0;

if abs(det(R_des)-1)>1e-5 || norm(cross(R_des(:,1),R_des(:,2))-R_des(:,3))>1e-5
    fprintf('This doesnot seem to be a proper rotation matrix!\n');
    R_des = R_des
    return
end
fk_wrist_func   = sprintf('FK_%s_WRIST',arm)
posJ_wrist_func = sprintf('Jpos_%s_WRIST',arm)
fk_armtip_func   = sprintf('FK_%s_ARMTIP',arm)
posJ_armtip_func = sprintf('Jpos_%s_ARMTIP',arm)
fk_last3_func   = sprintf('FK_%s_LAST3',arm)
analyticIK_last3_func   = sprintf('analyticIK_%s_LAST3',arm)

q_min = [-170 -60 -70 -35 -70 -45 -170 -105 -170]'*pi/180;
q_max = [170  80 250 105 250 120 170 90 170]'*pi/180;

q_min = [q_min; 0];     % gripper min in meters.
q_max = [q_max; 0.08];  % gripper max in meters.

if isempty(qinit)
    qinit = DEF_qinit;
end

if ~exist('qenable', 'var')
    q_weight = DEF_qenable;
end

if ~exist('err_TH','var')
    errTH = DEF_err_TH;
end
q = qinit;
q = reshape(q,length(q),1);
q = q.*q_weight;
p_des = reshape(p_des, length(p_des),1);
q_at_start = q;



T = feval(fk_wrist_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));    p_cur = T(1:3,4);
J = feval(posJ_wrist_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
pos_err = norm(p_des - p_cur);

while pos_err > errTH && it < MAXIT
    it = it + 1;
    T = feval(fk_wrist_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10)); p_cur = T(1:3,4);
    J = feval(posJ_wrist_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
    dp = p_des - p_cur;
    pos_err = norm(p_des - p_cur);
    if (pos_err<errTH), continue; end;
    dq = STEP_RATE*J'*dp;
    q = q + dq.*q_weight;
    lobo_viol = (q < q_min);
    hibo_viol = (q > q_max);
    q = max(q_min, q);
    q = min(q, q_max);
    if sum(lobo_viol)>0, fprintf('Joint LOwer limit hit at:%s\n', vec2str(lobo_viol,'%d')); end
     if sum(hibo_viol)>0, fprintf('Joint UPper limit hit at:%s\n', vec2str(hibo_viol,'%d')); end
     
    if mod(it,20==0) fprintf('%d > error:%2.4f\n',it, pos_err); end
end

if it == MAXIT
    fprintf('IK did not succeed. The current error:%2.4f\n',pos_err);
    p_des_vs_p_cur_Del = [p_des,     p_cur, p_des - p_cur]
    p_des_reached = 0;
else
    fprintf('Solved. \n Try: \n FK_right_WRIST(%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f)\n',q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
    p_des_reached = 1;
end

% The wrist IK includes R45 in it so we need to undo the orientation
% contribution of R45 to the endeffector, and apply analytical IK for
% T45*T56*T67
if strcmp(arm,'right')
    fixR45  = [ sin(q(7))   cos(q(7))     0;
                   0            0        -1.0;
               -cos(q(7))  sin(q(7))      0] ;   
    fixO45 = [0, -0.25,0];
    fixT45 = makehomeg(fixR45, fixO45);
else
    fixR45  = [ sin(q(7))   cos(q(7))     0;
                    0         0         1.0;
                cos(q(7))  -sin(q(7))     0] ; 
    
    fixO45 = [0, -0.25,0];
    fixT45 = makehomeg(fixR45, fixO45);
end

R_wrist = T(1:3,1:3)*fixR45';   % undo the orientation due to T45
% This is required:  R_wrist*R_last3 = R_des so solve R_last3 
R_last3 = R_wrist'*R_des
[q(7),q(8),q(9)] = feval(analyticIK_last3_func,R_last3);
last3_T = feval(fk_last3_func, q(7),q(8),q(9)); % double check the 3 angles we found
mustbezero_abc_ok = last3_T(1:3,1:3) - R_last3  % this must be zero!
%vpa(subs(T45*T56*T67,{'a5','a6','a7'},{a,b,c}),4)

%fprintf('With ORI \n Try: \n FK_right_ARMTIP(%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f)\n',q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
TT = feval(fk_armtip_func, q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
%R_found = TT(1:3,1:3)
mustbe_I = TT(1:3,1:3)'*R_des
if (~p_des_reached),  
    fprintf('Warning desired position accuracy could not be reached (pos_err:%4.2fmm) in %d iterations.\n',pos_err*1000,it); 
else
    fprintf('Desired position accuracy reached (pos_err:%4.2fmm) in %d iterations.\n',pos_err*1000, it)
end
found_T = TT;
asked_T = [[R_des; 0 0 0],[p_des; 1]];
found_asked_T_diff = found_T - asked_T

qq = q*180/pi;
if strcmp(arm,'right')
    cmd = sprintf('../Communication/ezpycom setq_deg rarm %f %f %f %f %f %f %f\n', qq(3),qq(4),qq(5),qq(6),qq(7),qq(8),qq(9));
else
    cmd = sprintf('../Communication/ezpycom setq_deg larm %f %f %f %f %f %f %f\n', qq(3),qq(4),qq(5),qq(6),qq(7),qq(8),qq(9));
end
fprintf('sending %s\n',cmd);
system(cmd);


