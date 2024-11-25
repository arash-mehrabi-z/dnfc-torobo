function [q, err] = simpleIK(q, p_des, err_TH)

qenable = [0 0 1 1 0 1 0 0 0 0]';
q = reshape(q,length(q),1);
q = q.*qenable;
p_des = reshape(p_des, length(p_des),1);
qinit = q;

if ~exist('err_TH','var')
    err_TH = 0.001;  % 0.5cm
end
    
T = FK_right_WRIST(q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));    p_cur = T(1:3,4);
J = posJ_right_WRIST(q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));

MAXIT = 5000;
it = 0;

err = norm(p_des - p_cur);

while err > err_TH & it < MAXIT
    it = it + 1;
    T = FK_right_WRIST(q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10)); p_cur = T(1:3,4);
    J = posJ_right_WRIST(q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
    dp = p_des - p_cur;
    err = norm(p_des - p_cur);
    if (err<err_TH), continue; end;
    dq = 1*J'*dp;
    q = q + dq.*qenable;
    fprintf('%d > error:%2.4f\n',it, err);
end

if it == MAXIT,
    fprintf('IK did not succeed. The current error:%2.4f\n',err);
    p_des_vs_p_cur_Del = [p_des,     p_cur, p_des - p_cur]
else
    fprintf('Solved. \n Try: \n FK_right_WRIST(%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f)\n',q(1),q(2),  q(3),q(4),q(5), q(6), q(7), q(8), q(9), q(10));
end



