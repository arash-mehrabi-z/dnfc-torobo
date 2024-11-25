function T = rotZ(q, homeg)

if ~exist('homeg','var')
    homeg = false;
end

if ~homeg
    T = [ cos(q) -sin(q) 0;
         sin(q)  cos(q) 0;
           0       0    1];
else
    T = [ cos(q) -sin(q) 0  0;
          sin(q)  cos(q) 0  0;
           0       0     1  0
           0       0     0  1]  ;
end
