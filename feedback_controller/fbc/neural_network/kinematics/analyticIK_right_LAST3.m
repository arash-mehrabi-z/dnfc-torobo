function [a,b,c] = analyticIK_right_LAST3(R)
% Need to find angles a=a5, b=a6, c=a7
% las t3 joints combined transformation is this (see rightarmkin.m). So R_last3 must be equl to this
% | Cos_a Sin_c - Cos_c Sin_a Sin_b    Cos_a Cos_c +  Sin_a Sin_b Sin_c     Cos_b Sin_a |
% |      Cos_b Cos_c                        -Cos_b Sin_c                      Sin _b    |
% | Sin_a Sin_c + Cos_a Cos_c Sin_b    Cos_c Sin_a - Cos_a Sin_b Sin_c     -Cos_a Cos_b |
% So we solve with these:

b = asin(R(2,3));    % robot joint 6 (b) range is (-pi/2  .. pi/2)*1.16 if |b|>pi/2 then asin() will not give the correct asnwer.
if (abs(cos(b))>1e-6)       % when joint 5 and 7 axis are not aligned easily find the a5 (a) and a7 (c) 
    c = atan2(-R(2,2), R(2,1));
    a = atan2(R(1,3),-R(3,3)) ;
else                % joint 5 and 7 axis are aligned let's use only c to solve 
    % When Cos_b=0, the desired eqaulity reduces to Rot(a)*Rot(pi/2-c)*[ [1 0] ; [0 -1] ] = [ [R11 R12 ]; [R31 R32] ]
    % Which  shows that a-c = atan2(R31,R11) - pi/2
    % NOTE:  this assumes b=pi/2. This was manually derived for when b=pi/2
    % a = -pi/2;
    % c = a+pi/2 - atan2(R(3,1),R(1,1));
    
    %NEW version derived by using:
    %simplify(subs(LAST3_T,{'a5','a6'},{x,+-pi/2})) where x is in
    %{0,pi,-pi/2, pi/2}
    if b<0   % can choose a freely
       a =0;
       c = atan2(R(1,1),R(1,2));
       % OR others..  
       %a=pi;
       %c=atan2(-R(1,1),-R(1,2))
       
       %a=pi/2;
       %c = atan2(-R(1,2),R(1,1)
       
       %a=-pi/2;
       %c= atan2(R(1,2),-R(1,1))
    else
       a = 0;
       c = atan2(R(1,1),R(1,2));
       % OR others..  
       %a=pi;
       %c=atan2(-R(1,1),-R(1,2))
       
       %a=pi/2;
       %c = atan2(R(1,2),-R(1,1)
       
       %a=-pi/2;
       %c= atan2(-R(1,2),R(1,1))
    end
end