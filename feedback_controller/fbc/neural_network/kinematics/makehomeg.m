function T =makehomeg(R,O)

offset = reshape(O,3,1);
T = [  [R, offset]; [0 0 0 1] ];
