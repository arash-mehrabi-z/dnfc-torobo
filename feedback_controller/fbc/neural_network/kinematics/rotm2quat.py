#TODO  not implemented yet
import sys
from pyquaternion import Quaternion
#print('Number of arguments: {}'.format(len(sys.argv)))
#print('Argument(s) passed: {}'.format(str(sys.argv)))
#print sys.argv[1:]
qel = [0,0,0,0];
for i in range(len(sys.argv)-1):
	qel[i] = float(sys.argv[i+1])
q = Quaternion(qel)
R = q.rotation_matrix
#print q.rotation_matrix
print "   R_x:","[%+.4f,"%R[0,0],"%+.4f,"%R[1,0],"%+.4f]"%R[2,0]
print "   R_y:","[%+.4f,"%R[0,1],"%+.4f,"%R[1,1],"%+.4f]"%R[2,1]
print "   R_z:","[%+.4f,"%R[0,2],"%+.4f,"%R[1,2],"%+.4f]"%R[2,2]

