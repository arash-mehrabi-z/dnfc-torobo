% FORWARD KINEMATICS 
% Try: T=FK_right_HEAD(0,0, 0,0,0,0,0,0,0, 0), rotm2quat(T(1:3,1:3))
function T = FK_right_HEAD(t1, t2, h1, h2)
    T(1,1) = - 0.00000000000000006123233995736766035868820147292*sin(h2)*(cos(h1)*sin(t1) + cos(t1)*cos(t2)*sin(h1)) - 1.0*cos(h2)*(sin(h1)*sin(t1) - 1.0*cos(h1)*cos(t1)*cos(t2)) - 1.0*cos(t1)*sin(h2)*sin(t2);
    T(1,2) = sin(h2)*(sin(h1)*sin(t1) - 1.0*cos(h1)*cos(t1)*cos(t2)) - 0.00000000000000006123233995736766035868820147292*cos(h2)*(cos(h1)*sin(t1) + cos(t1)*cos(t2)*sin(h1)) - 1.0*cos(h2)*cos(t1)*sin(t2);
    T(1,3) = 0.00000000000000006123233995736766035868820147292*cos(t1)*sin(t2) - 1.0*cos(h1)*sin(t1) - 1.0*cos(t1)*cos(t2)*sin(h1);
    T(1,4) = 0.3945*cos(t1)*sin(t2) + 0.035*sin(h1)*sin(t1) - 0.035*cos(h1)*cos(t1)*cos(t2);
    T(2,1) = cos(h2)*(cos(t1)*sin(h1) + cos(h1)*cos(t2)*sin(t1)) + 0.00000000000000006123233995736766035868820147292*sin(h2)*(cos(h1)*cos(t1) - 1.0*cos(t2)*sin(h1)*sin(t1)) - 1.0*sin(h2)*sin(t1)*sin(t2);
    T(2,2) = 0.00000000000000006123233995736766035868820147292*cos(h2)*(cos(h1)*cos(t1) - 1.0*cos(t2)*sin(h1)*sin(t1)) - 1.0*sin(h2)*(cos(t1)*sin(h1) + cos(h1)*cos(t2)*sin(t1)) - 1.0*cos(h2)*sin(t1)*sin(t2);
    T(2,3) = cos(h1)*cos(t1) + 0.00000000000000006123233995736766035868820147292*sin(t1)*sin(t2) - 1.0*cos(t2)*sin(h1)*sin(t1);
    T(2,4) = 0.3945*sin(t1)*sin(t2) - 0.035*cos(t1)*sin(h1) - 0.035*cos(h1)*cos(t2)*sin(t1);
    T(3,1) = 0.00000000000000006123233995736766035868820147292*sin(h1)*sin(h2)*sin(t2) - 1.0*cos(h1)*cos(h2)*sin(t2) - 1.0*cos(t2)*sin(h2);
    T(3,2) = cos(h1)*sin(h2)*sin(t2) - 1.0*cos(h2)*cos(t2) + 0.00000000000000006123233995736766035868820147292*cos(h2)*sin(h1)*sin(t2);
    T(3,3) = 0.00000000000000006123233995736766035868820147292*cos(t2) + sin(h1)*sin(t2);
    T(3,4) = 0.3945*cos(t2) + 0.035*cos(h1)*sin(t2) + 1.03;
    T(4,1) = 0.0;
    T(4,2) = 0.0;
    T(4,3) = 0.0;
    T(4,4) = 1.0;
%end of computeForKin

