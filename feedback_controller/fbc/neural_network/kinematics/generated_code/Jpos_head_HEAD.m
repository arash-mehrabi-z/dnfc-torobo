% HEAD POINT POSITION JACOBIAN
function J = Jpos_head_HEAD(t1, t2, h1, h2)
    J(1,1) = 0.035*cos(t1)*sin(h1) - 0.3945*sin(t1)*sin(t2) + 0.035*cos(h1)*cos(t2)*sin(t1);
    J(1,2) = 0.3945*cos(t1)*cos(t2) + 0.035*cos(h1)*cos(t1)*sin(t2);
    J(1,3) = 0.035*cos(h1)*sin(t1) + 0.035*cos(t1)*cos(t2)*sin(h1);
    J(1,4) = 0.0;
    J(2,1) = 0.3945*cos(t1)*sin(t2) + 0.035*sin(h1)*sin(t1) - 0.035*cos(h1)*cos(t1)*cos(t2);
    J(2,2) = 0.3945*cos(t2)*sin(t1) + 0.035*cos(h1)*sin(t1)*sin(t2);
    J(2,3) = 0.035*cos(t2)*sin(h1)*sin(t1) - 0.035*cos(h1)*cos(t1);
    J(2,4) = 0.0;
    J(3,1) = 0.0;
    J(3,2) = 0.035*cos(h1)*cos(t2) - 0.3945*sin(t2);
    J(3,3) = -0.035*sin(h1)*sin(t2);
    J(3,4) = 0.0;
%end of Jacobian computation

