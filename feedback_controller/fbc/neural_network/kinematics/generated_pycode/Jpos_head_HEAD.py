# HEAD POINT POSITION JACOBIAN
# Try: [J,~]=Jpos_head_HEAD.compute_jac(0,0, 0,0,0,0,0,0,0, 0)
import numpy as np

class Jpos_head_HEAD:
	@staticmethod
	def compute_jac(t1, t2, h1, h2):
		J = np.zeros([3,4]);
		J[0,0] = 0.035*np.cos(t1)*np.sin(h1) - 0.3945*np.sin(t1)*np.sin(t2) + 0.035*np.cos(h1)*np.cos(t2)*np.sin(t1)
		J[0,1] = 0.3945*np.cos(t1)*np.cos(t2) + 0.035*np.cos(h1)*np.cos(t1)*np.sin(t2)
		J[0,2] = 0.035*np.cos(h1)*np.sin(t1) + 0.035*np.cos(t1)*np.cos(t2)*np.sin(h1)
		J[0,3] = 0.0
		J[1,0] = 0.3945*np.cos(t1)*np.sin(t2) + 0.035*np.sin(h1)*np.sin(t1) - 0.035*np.cos(h1)*np.cos(t1)*np.cos(t2)
		J[1,1] = 0.3945*np.cos(t2)*np.sin(t1) + 0.035*np.cos(h1)*np.sin(t1)*np.sin(t2)
		J[1,2] = 0.035*np.cos(t2)*np.sin(h1)*np.sin(t1) - 0.035*np.cos(h1)*np.cos(t1)
		J[1,3] = 0.0
		J[2,0] = 0.0
		J[2,1] = 0.035*np.cos(h1)*np.cos(t2) - 0.3945*np.sin(t2)
		J[2,2] = -0.035*np.sin(h1)*np.sin(t2)
		J[2,3] = 0.0
		return J
	#end of compute Jacobian
#end of class Jpos_head_HEAD
def main():
	print(Jpos_head_HEAD.compute_jac(0,0, 0,0,0,0, 0,0,0, 0))

if __name__ == '__main__':
	main()

