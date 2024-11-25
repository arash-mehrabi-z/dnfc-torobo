# FORWARD KINEMATICS 
# Try: T=FK_head_HEAD.forward_ik(0,0, 0,0,0,0,0,0,0, 0) # rotm2quat(T[1:3,1:3])
import numpy as np

class FK_head_HEAD:
	@staticmethod
	def forward_ik(t1, t2, h1, h2):
		T=np.zeros([4,4]);
		T[0,0] = - 6.123234e-17*np.sin(h2)*(np.cos(h1)*np.sin(t1) + np.cos(t1)*np.cos(t2)*np.sin(h1)) - 1.0*np.cos(h2)*(np.sin(h1)*np.sin(t1) - 1.0*np.cos(h1)*np.cos(t1)*np.cos(t2)) - 1.0*np.cos(t1)*np.sin(h2)*np.sin(t2)
		T[0,1] = np.sin(h2)*(np.sin(h1)*np.sin(t1) - 1.0*np.cos(h1)*np.cos(t1)*np.cos(t2)) - 6.123234e-17*np.cos(h2)*(np.cos(h1)*np.sin(t1) + np.cos(t1)*np.cos(t2)*np.sin(h1)) - 1.0*np.cos(h2)*np.cos(t1)*np.sin(t2)
		T[0,2] = 6.123234e-17*np.cos(t1)*np.sin(t2) - 1.0*np.cos(h1)*np.sin(t1) - 1.0*np.cos(t1)*np.cos(t2)*np.sin(h1)
		T[0,3] = 0.3945*np.cos(t1)*np.sin(t2) + 0.035*np.sin(h1)*np.sin(t1) - 0.035*np.cos(h1)*np.cos(t1)*np.cos(t2)
		T[1,0] = np.cos(h2)*(np.cos(t1)*np.sin(h1) + np.cos(h1)*np.cos(t2)*np.sin(t1)) + 6.123234e-17*np.sin(h2)*(np.cos(h1)*np.cos(t1) - 1.0*np.cos(t2)*np.sin(h1)*np.sin(t1)) - 1.0*np.sin(h2)*np.sin(t1)*np.sin(t2)
		T[1,1] = 6.123234e-17*np.cos(h2)*(np.cos(h1)*np.cos(t1) - 1.0*np.cos(t2)*np.sin(h1)*np.sin(t1)) - 1.0*np.sin(h2)*(np.cos(t1)*np.sin(h1) + np.cos(h1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(h2)*np.sin(t1)*np.sin(t2)
		T[1,2] = np.cos(h1)*np.cos(t1) + 6.123234e-17*np.sin(t1)*np.sin(t2) - 1.0*np.cos(t2)*np.sin(h1)*np.sin(t1)
		T[1,3] = 0.3945*np.sin(t1)*np.sin(t2) - 0.035*np.cos(t1)*np.sin(h1) - 0.035*np.cos(h1)*np.cos(t2)*np.sin(t1)
		T[2,0] = 6.123234e-17*np.sin(h1)*np.sin(h2)*np.sin(t2) - 1.0*np.cos(h1)*np.cos(h2)*np.sin(t2) - 1.0*np.cos(t2)*np.sin(h2)
		T[2,1] = np.cos(h1)*np.sin(h2)*np.sin(t2) - 1.0*np.cos(h2)*np.cos(t2) + 6.123234e-17*np.cos(h2)*np.sin(h1)*np.sin(t2)
		T[2,2] = 6.123234e-17*np.cos(t2) + np.sin(h1)*np.sin(t2)
		T[2,3] = 0.3945*np.cos(t2) + 0.035*np.cos(h1)*np.sin(t2) + 1.03
		T[3,0] = 0.0
		T[3,1] = 0.0
		T[3,2] = 0.0
		T[3,3] = 1.0
		return T
	#end of computeForKin
#end of class FK_head_HEAD
def main():
	print(FK_head_HEAD.forward_ik(0,0, 0,0,0,0, 0,0,0, 0))

if __name__ == '__main__':
	main()

