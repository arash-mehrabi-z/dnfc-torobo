# HEAD POINT YAXIS JACOBIAN
# Try: [J,~]=Jyaxis_head_HEAD.compute_jac(0,0, 0,0,0,0,0,0,0, 0)
import numpy as np

class Jyaxis_head_HEAD:
	@staticmethod
	def compute_jac(t1, t2, h1, h2):
		J = np.zeros([3,4]);
		J[0,0] = np.sin(h2)*(np.cos(t1)*np.sin(h1) + np.cos(h1)*np.cos(t2)*np.sin(t1)) - 6.123234e-17*np.cos(h2)*(np.cos(h1)*np.cos(t1) - 1.0*np.cos(t2)*np.sin(h1)*np.sin(t1)) + np.cos(h2)*np.sin(t1)*np.sin(t2)
		J[0,1] = np.cos(h1)*np.cos(t1)*np.sin(h2)*np.sin(t2) - 1.0*np.cos(h2)*np.cos(t1)*np.cos(t2) + 6.123234e-17*np.cos(h2)*np.cos(t1)*np.sin(h1)*np.sin(t2)
		J[0,2] = np.sin(h2)*(np.cos(h1)*np.sin(t1) + np.cos(t1)*np.cos(t2)*np.sin(h1)) + 6.123234e-17*np.cos(h2)*(np.sin(h1)*np.sin(t1) - 1.0*np.cos(h1)*np.cos(t1)*np.cos(t2))
		J[0,3] = 6.123234e-17*np.sin(h2)*(np.cos(h1)*np.sin(t1) + np.cos(t1)*np.cos(t2)*np.sin(h1)) + np.cos(h2)*(np.sin(h1)*np.sin(t1) - 1.0*np.cos(h1)*np.cos(t1)*np.cos(t2)) + np.cos(t1)*np.sin(h2)*np.sin(t2)
		J[1,0] = np.sin(h2)*(np.sin(h1)*np.sin(t1) - 1.0*np.cos(h1)*np.cos(t1)*np.cos(t2)) - 6.123234e-17*np.cos(h2)*(np.cos(h1)*np.sin(t1) + np.cos(t1)*np.cos(t2)*np.sin(h1)) - 1.0*np.cos(h2)*np.cos(t1)*np.sin(t2)
		J[1,1] = np.cos(h1)*np.sin(h2)*np.sin(t1)*np.sin(t2) - 1.0*np.cos(h2)*np.cos(t2)*np.sin(t1) + 6.123234e-17*np.cos(h2)*np.sin(h1)*np.sin(t1)*np.sin(t2)
		J[1,2] = - 6.123234e-17*np.cos(h2)*(np.cos(t1)*np.sin(h1) + np.cos(h1)*np.cos(t2)*np.sin(t1)) - 1.0*np.sin(h2)*(np.cos(h1)*np.cos(t1) - 1.0*np.cos(t2)*np.sin(h1)*np.sin(t1))
		J[1,3] = np.sin(h2)*np.sin(t1)*np.sin(t2) - 6.123234e-17*np.sin(h2)*(np.cos(h1)*np.cos(t1) - 1.0*np.cos(t2)*np.sin(h1)*np.sin(t1)) - 1.0*np.cos(h2)*(np.cos(t1)*np.sin(h1) + np.cos(h1)*np.cos(t2)*np.sin(t1))
		J[2,0] = 0.0
		J[2,1] = np.cos(h2)*np.sin(t2) + np.cos(h1)*np.cos(t2)*np.sin(h2) + 6.123234e-17*np.cos(h2)*np.cos(t2)*np.sin(h1)
		J[2,2] = 6.123234e-17*np.cos(h1)*np.cos(h2)*np.sin(t2) - 1.0*np.sin(h1)*np.sin(h2)*np.sin(t2)
		J[2,3] = np.cos(t2)*np.sin(h2) + np.cos(h1)*np.cos(h2)*np.sin(t2) - 6.123234e-17*np.sin(h1)*np.sin(h2)*np.sin(t2)
		return J
	#end of compute Jacobian
#end of class Jyaxis_head_HEAD
def main():
	print(Jyaxis_head_HEAD.compute_jac(0,0, 0,0,0,0, 0,0,0, 0))

if __name__ == '__main__':
	main()

