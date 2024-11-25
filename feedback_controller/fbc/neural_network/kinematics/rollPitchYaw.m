function R=rollPitchYaw(roll, pitch, yaw)
% Performs the fixed axis rotations in the order below:
% yaw radians around X axis, then pitch radians around Y axins, and finally roll 
% radians around the Z axis.
R=rotZ(roll)*rotY(pitch)*rotX(yaw);