import numpy as np
from pyquaternion import Quaternion
class TOR:
    STATE_TOPIC_NAME_L = ['/torobo/torso_controller/torobo_joint_state',
                      '/torobo/right_arm_controller/torobo_joint_state',
                      '/torobo/right_gripper_controller/torobo_joint_state',
                      '/torobo/left_arm_controller/torobo_joint_state',
                      '/torobo/left_gripper_controller/torobo_joint_state',
                      '/torobo/head_controller/torobo_joint_state']

    TOPIC_NAME_L =  ['/torobo/torso_controller/command',
                 '/torobo/right_arm_controller/command',
                 '/torobo/right_gripper_controller/command',
                 '/torobo/left_arm_controller/command',
                 '/torobo/left_gripper_controller/command',
                 '/torobo/head_controller/command']


    RAD2DEG = 180.0/np.pi
    DEG2RAD = np.pi/180.0
    
    torso_JOINT_NAMES = ['torso/joint_' + str(i) for i in range(1, 3)] # from joint_1 to joint_8                 
    rightarm_JOINT_NAMES = ['right_arm/joint_' + str(i) for i in range(1, 8)] # from joint_1 to joint_8
    rightgripper_JOINT_NAMES = ['right_gripper/joint_' + str(i) for i in range(1, 2)] # from joint_1 to joint_8
    leftarm_JOINT_NAMES = ['left_arm/joint_' + str(i) for i in range(1, 8)] # from joint_1 to joint_8
    leftgripper_JOINT_NAMES = ['left_gripper/joint_' + str(i) for i in range(1, 2)] # from joint_1 to joint_8
    head_JOINT_NAMES = ['head/joint_' + str(i) for i in range(1, 3)] # from joint_1 to joint_8

    JOINT_NAMES_L = [torso_JOINT_NAMES,rightarm_JOINT_NAMES,rightgripper_JOINT_NAMES,
                 leftarm_JOINT_NAMES,leftgripper_JOINT_NAMES,head_JOINT_NAMES];


    _TORSO = 0
    _RARM  = 1
    _RGRIP = 2
    _LARM  = 3
    _LGRIP = 4
    _HEAD  = 5
    _ALL   = 6

    # min-max for joints as [torso arm gripper] vectors (10 element)
    ## seems not used ERH
    ## _Qmin10 = np.array([-170,-60,-70,-35,-70,-45,-170,-105,-170,0])*np.pi/180;
    ##_Qmax10 = np.array([170,80,250,105,250,120,170,90,170,0.08*180/np.pi])*np.pi/180;


    _TARGET_L2 =['torso', 'rarm ', 'rgrip', 'larm ','lgrip','head ','all  ']      # use lower in the command line
    _TARGET_L = ['torso', 'rarm', 'rgrip', 'larm','lgrip','head','all']      # use lower in the command line
    _NUMJNT_L = [   2   ,   7   ,    2   ,   7   ,   2   ,  2   , 20  ]

# min-max for joints as individual limbs as list
    _QminL =[ np.array([-170.0, -60])*np.pi/180,                                # torso
          np.array([-70, -35, -70, -60, -170, -105, -170])*np.pi/180,     # rightarm       #elbow joint is restricted to positive angles for IK
          np.array([0.0, -80.]),                                          # rightgripper
          np.array([-70, -35, -70, -60, -170, -105, -170])*np.pi/180,     # leftarm        #elbow joint is restricted to positive angles for IK
          np.array([0.0, -80.]),                                          # leftgripper
          np.array([-45,-45])*np.pi/180     #FIX#                         # head
         ]   # zero posture for ezcontroller
    _QmaxL= [ np.array([170, 80])*np.pi/180,                                  # torso
          np.array([250,105,250,120,170,90,170])*np.pi/180,               # rightarm
          np.array([0.08, 80]),                                               # rightgripper
          np.array([250,105,250,120,170,90,170])*np.pi/180,               # leftarm
          np.array([0.08, 80]),                                               # leftgripper
          np.array([45,45])*np.pi/180    #FIX#                            # head
         ]

    # These are nominal encolse and open parameters to be used. Tune according to the min max above.
    gripLENC_dist   = 12./1000.  # meters
    gripLENC_force  = 13.0     # newtons??
    gripLOPEN_dist  = 40./1000.  # meters
    gripLOPEN_force =-30.0     # newtons??
    
    gripRENC_dist   = 11./1000.  # meters
    gripRENC_force  = 19.0     # newtons??
    gripROPEN_dist  = 40./1000.  # meters
    gripROPEN_force =-20.0     # newtons??
    # TOROBO ARM STRUCTURE
    #  0     1     2      3       4        5        6
    #SHLD1 SHLD2 REDUNT ELBOW WRST_ROT WRST_BENT FINAL_ROT
 
    # zero posture for ezcontroller
    q_tuba = [ np.array([0.0,0.3491]),                #radian    # torso
             np.array([0.7855, 0.6679, 0.4112, 2.0740, 1.0322, -1.0649, -0.3227]),  # rightarm
             np.array([0.01, 10.0]),             # rightgripper
             np.array([10*np.pi/180, 60*np.pi/180., -40*np.pi/180, 0.0, 0.0, 0.0, 0.0]),   # leftarm
             np.array([gripLOPEN_dist, gripLOPEN_force]),             # leftgripper
             np.array([0.0,37.7])*np.pi/180                           # head
             ]

    q_tuba_2 = [ np.array([0.0,0.3491]),                #radian    # torso
             np.array([1.1243, 0.8624, 0.5702, 1.1793, 1.1193, -0.3502 ,1.3341]),  #radian # rightarm #second push
             np.array([0.01, 10.0]),             # rightgripper
             np.array([ 10*np.pi/180, 60*np.pi/180., -40*np.pi/180, 0.0, 0.0, 0.0, 0.0]),   # leftarm
             np.array([gripLOPEN_dist, gripLOPEN_force]),             # leftgripper
             np.array([0.0,37.7])*np.pi/180                           # head
             ]
    #correction_pre 1.0633 0.8040 0.5927 1.2948 1.1196 -0.3960 -0.2351
    q_tuba_pre_correction = [ np.array([0.0,0.3491]),                #radian    # torso
             np.array([1.1433, 0.9007, 0.5831, 1.1135, 1.0992, -0.3182, -0.2301]),  #radian # rightarm #second push
             np.array([0.01, 10.0]),             # rightgripper
             np.array([ 10*np.pi/180, 60*np.pi/180., -40*np.pi/180, 0.0, 0.0, 0.0, 0.0]),   # leftarm
             np.array([gripLOPEN_dist, gripLOPEN_force]),             # leftgripper
             np.array([0.0,37.7])*np.pi/180                           # head
             ]

    q_tuba_correction = [ np.array([0.0,0.3491]),                #radian    # torso
             np.array([1.4756, 1.5113, 0.5218, 0.2358, 1.0439, -0.1451, -0.3259]),  #radian # rightarm #second push
             np.array([0.01, 10.0]),             # rightgripper
             np.array([ 10*np.pi/180, 60*np.pi/180., -40*np.pi/180, 0.0, 0.0, 0.0, 0.0]),   # leftarm
             np.array([gripLOPEN_dist, gripLOPEN_force]),             # leftgripper
             np.array([0.0,37.7])*np.pi/180                           # head
             ]


    q_tuba_down = [ np.array([0.0,0.3491]),                #radian    # torso
             np.array([1.4608, 1.5522, 0.9453, 1.1267, -0.9280 ,0.4804, 2.1978]),  #radian # rightarm #second push
             np.array([gripROPEN_dist, gripROPEN_force]),             # rightgripper
             np.array([ 10*np.pi/180, 60*np.pi/180., -40*np.pi/180, 0.0, 0.0, 0.0, 0.0]),   # leftarm
             np.array([gripLOPEN_dist, gripLOPEN_force]),             # leftgripper
             np.array([0.0,37.7])*np.pi/180                           # head
             ]

    q_ezhome = [ np.array([0.0,0.0])*np.pi/180,                    # torso
             np.array([53., 13., 60, 80., 50, 10, 10 ])*np.pi/180,  # rightarm
             np.array([gripROPEN_dist, gripROPEN_force]),             # rightgripper
             np.array([53., 13., 60, 80., 50, 10, 10])*np.pi/180,   # leftarm
             np.array([gripLOPEN_dist, gripLOPEN_force]),             # leftgripper
             np.array([0.0,37.7])*np.pi/180                           # head
             ]
    
    q_prehome= [ np.array([0.0,0.0]),                                         # torso
             np.array([-20*np.pi/180, 60*np.pi/180.,-4*np.pi/180, 0.0, 0.0, 0.0, 0.0]),     # rightarm
             np.array([0.01, 10.0]),                                             # rightgripper
             np.array([-20*np.pi/180, 60*np.pi/180., -40*np.pi/180, 0.0, 0.0, 0.0, 0.0]),     # leftarm
             np.array([0.01, 10.0]),                                             # leftgripper
             np.array([0.0,0])                                            # head
             ] # zero posture for the Torobo native controller

    
    q_ezhome_good = [ np.array([0.0,0.0])*np.pi/180,                # torso
             np.array([53.0, 13, 3, 80, 70, 2, -45 ])*np.pi/180,    # rightarm
             np.array([0.0, 0.0]),                                  # rightgripper
             np.array([53.0, 13, 3, 80, 70, 2, -45])*np.pi/180,     # leftarm
             np.array([0.0, 0.0]),                                  # leftgripper
             np.array([0.0,35])*np.pi/180                           # head
             ]  # zero posture for ezcontroller
    
    xq_ezhome3 = [ np.array([0.0,0.0]),                                         # torso
             np.array([38.01, -5.38, 38.17, 105.58, 47.77, -21.23, -90.24])*np.pi/180,     # rightarm
             np.array([0.0, 0.0]),                                             # rightgripper
             np.array([38.01, -5.38, 38.17, 105.58, 47.77, -21.23, -90.24])*np.pi/180,     # leftarm
             np.array([0.0, 0.0]),                                             # leftgripper
             np.array([0.0,0])                                            # head
             ]  # zero posture for ezcontroller
    xq_ezhome2 = [ np.array([0.0,0.0]),                                         # torso
             np.array([50.24, 18.80, 38.77, 82.94, 33.06, -9.14, -83.50])*np.pi/180,     # rightarm
             np.array([0.0, 0.0]),                                             # rightgripper
             np.array([50.24, 18.80, 38.77, 82.94, 33.06, -9.14, -83.50])*np.pi/180,     # leftarm
             np.array([0.0, 0.0]),                                             # leftgripper
             np.array([0.0,0])                                            # head
             ]  # zero posture for ezcontroller
    
    xq_ezhome_natural = [ np.array([0.0,0.0]),                                         # torso
             np.array([48.18, 22.41, 37.87, 100.5, 26.65, -20.49, -70.23])*np.pi/180,     # rightarm
             np.array([0.0, 0.0]),                                             # rightgripper
             np.array([48.18, 22.41, 37.87, 100.5, 26.65, -20.49, -70.23])*np.pi/180,     # leftarm
             np.array([0.0, 0.0]),                                             # leftgripper
             np.array([0.0,0])                                            # head
             ]   # zero posture for ezcontroller
    
    xq_ezhome1 = [ np.array([0.0,0.2]),                                         # torso
             np.array([30.0, 45, 0.0, 30.0, 0, 0.0, 0.0])*np.pi/180.,     # rightarm
             np.array([0.05, 40.0]),                                             # rightgripper
             np.array([30.0, 45., 0.0, 30.0, 0, 0.0, 0.0])*np.pi/180.,     # leftarm
             np.array([0.05, 40.0]),                                             # leftgripper
             np.array([0.0,0])                                            # head
             ]   # zero posture for ezcontroller
    q_syshome= [ np.array([0.0,0.0]),                                         # torso
             np.array([0.0, 90*np.pi/180., 0.0, 0.0, 0.0, 0.0, 0.0]),     # rightarm
             np.array([0.01, 10.0]),                                             # rightgripper
             np.array([0.0, 90*np.pi/180., 0.0, 0.0, 0.0, 0.0, 0.0]),     # leftarm
             np.array([0.01, 10.0]),                                             # leftgripper
             np.array([0.0,0])                                            # head
             ] # zero posture for the Torobo native controller

    q_allzero = [ np.array([0.0,0.0])*np.pi/180,                                         # torso
             np.array([0.0, 0., 0.0, 0.0, 0.0, 0, 0])*np.pi/180,     # rightarm
             np.array([0.0, 0.0]),                                             # rightgripper
             np.array([0.0, 0., 0.0, 0.0, 0.0, 0, 0])*np.pi/180,     # leftarm
             np.array([0.0, 0.0]),                                             # leftgripper
             np.array([0.0,0])*np.pi/180                                    # head
             ] # zero posture for the Torobo native controller

    q_almostzero= [ np.array([0.0,20.0])*np.pi/180,                                         # torso
             np.array([10.0, 20., 10.0, 50.0, 5.0, -20, -40])*np.pi/180,     # rightarm
             np.array([0.01, -10.0]),                                             # rightgripper
             np.array([10.0, 20., 10.0, 50.0, 5.0, -20, -40])*np.pi/180,     # leftarm
             np.array([0.01, -10.0]),                                             # leftgripper
             np.array([0.0,35])*np.pi/180                                    # head
             ] # zero posture for the Torobo native controller
        

    ik_mode_POS = 1
    ik_mode_ORI = 2
    ik_mode_BOTH = 0



    
    """
    Converts list of floats to a single string (this prints the seperator at the beginning, check and replace with vec2str1)
    """
    @staticmethod
    def vec2str(L, form = "%3.2f", sep=' '):

        sexp = "%%s%c%s"%(sep,form)
        #print sexp
        s = ''
        for i in range(0, len(L)):
            s = str(sexp%('',L[i])) if i==0 else sexp%(s,L[i])
        return s
    
    @staticmethod    
    def vec2str1(L, form = "%3.2f", sep=' '):
        sexp  = "%%s%s%s"%(sep,form)
        sexp0 = "%%s%s%s"%('',form)
        s = ''
        for i in range(0, len(L)):
            s = str(sexp0%('',L[i])) if i==0 else sexp%(s,L[i])
        return s

    @staticmethod
    def rotZhom(a):
        TrotZ = np.array([[ np.cos(a), -np.sin(a), 0,  0], [np.sin(a),np.cos(a), 0, 0], [0,0,1,0],[0,0,0,1]]) 
        return TrotZ
    @staticmethod
    def rotYhom( a):
        TrotY = np.array([[np.cos(a), 0, np.sin(a), 0], [0,1,0,0], [-np.sin(a),0, np.cos(a),0],[0,0,0,1]])                                                                             
        return TrotY
    
    @staticmethod
    def rotZ(a):
        TrotZ = np.array([[ np.cos(a), -np.sin(a), 0], [np.sin(a),np.cos(a), 0], [0,0,1]]) 
        return TrotZ
    @staticmethod
    def rotY( a):
        TrotY = np.array([[np.cos(a), 0, np.sin(a)], [0,1,0], [-np.sin(a),0, np.cos(a)]])                                                                             
        return TrotY

    @staticmethod    
    def str2nums(s, sep=',', ignore_errors= True):
        args_s = s.split(sep)
        args_f = np.ones(len(args_s))*np.nan
        #args_i = np.ones(len(args_s))*np.nan
        for k in range(0, len(args_s)):
            try:
                args_f[k] = float(args_s[k])
            except:
                if not ignore_errors:
                    print("str2nums: "+args_s[k]+" is not a number!")
        return args_f            
    
    @staticmethod
    def quat2Rot( q):
        return Quaternion(q).rotation_matrix

    @staticmethod
    def quat2yaw_pitch_roll( q):
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        yaw   = np.arctan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
        pitch = np.arcsin(-2.0*(qx*qz - qw*qy))
        roll  = np.arctan2(2.0*(qx*qy + qw*qz), qw*qw + qx*qx - qy*qy - qz*qz)
        return yaw, pitch, roll


    @staticmethod
    def gen_ikpath(minpos,maxpos, step):
        xL = int(step[0])
        yL = int(step[1])
        zL = int(step[2])
        SIZE = xL*yL*zL


        xmin, xmax = minpos[0], maxpos[0]
        ymin, ymax = minpos[1], maxpos[1]
        zmin, zmax = minpos[2], maxpos[2]

        xst = (xmax - xmin)/(xL-1) if xL>1 else 0
        yst = (ymax - ymin)/(yL-1) if yL>1 else 0
        zst = (zmax - zmin)/(zL-1) if zL>1 else 0


        A = np.zeros([zL,yL,xL])
        data = np.zeros([SIZE,4])

        xi=yi=zi=0
        #x y z  counters
        dx = 1; dy = 1; dz=1
        k=0
        s=''
        done = False
        while not done:
            A[zi,yi,xi] = k
            x = xst*xi + xmin
            y = yst*yi + ymin
            z = zst*zi + zmin
            data[k,:] = np.array([x,y,z,k])
            k +=1
            if dx>0: s = s + '  %2d'%k
            else: s = '  %2d'%k + s
            xi += dx
            if xi==xL or xi<0:
                print (s)
                s = ''
                dx = -dx;  xi += dx
                yi += dy
                if yi>=yL or yi<0:
                    #print ("Layer "+str(zi)+" done")
                    dy = -dy; yi += dy
                    zi += dz
                    if zi>=zL or zi<0:
                        #print("Cube is done")
                        done = True

        #A is a 3D integer index array, that refers to data for actual x,y,z data.
        return A, data

    """
    Does limit checking for joint angles.
    If tarix os TOR. q is a list of list of joint angles (check for the full robot)
    Otherwise the check is done for the body part indicated bu tarix, and q is just a vector 
    """
    @staticmethod
    def check_range(tarix, q):
        gripRENC_dist   = 11./1000.  # meters
        gripRENC_force  = 19.0     # newtons??
        gripROPEN_dist  = 40./1000.  # meters
        gripROPEN_force =-20.0     # newtons??

        gripLENC_dist   = 12./1000.  # meters
        gripLENC_force  = 13.0     # newtons??
        gripLOPEN_dist  = 40./1000.  # meters
        gripLOPEN_force =-30.0     # newtons??
        q_ezhome = [ np.array([0.0,0.0])*np.pi/180,                    # torso
            np.array([53., 13., 60, 80., 50, 10, 10 ])*np.pi/180,  # rightarm
            np.array([gripROPEN_dist, gripROPEN_force]),             # rightgripper
            np.array([53., 13., 60, 80., 50, 10, 10])*np.pi/180,   # leftarm
            np.array([gripLOPEN_dist, gripLOPEN_force]),             # leftgripper
            np.array([0.0,37.7])*np.pi/180                           # head
            ]
        if tarix==TOR._ALL:  # q is a list!
            viollo =   [ 0*q_ezhome[i] for i in range(0, len(q_ezhome))  ]
            violup =   [ 0*q_ezhome[i] for i in range(0, len(q_ezhome))  ]

            for k in range(0,len(q)):
                viollo[k] = [q[k][i] <  TOR._QminL[k][i] for i in range(0, len(TOR._QminL[k]))]
                violup[k] = [q[k][i] >  TOR._QmaxL[k][i] for i in range(0, len(TOR._QminL[k]))]
                ok = not(any(viollo[k]) or any(violup[k]))
                if ~ok:
                    return ok,viollo, violup
                
        else:           # newdesq is a numpy array
            viollo = [q[i] <  TOR._QminL[tarix][i] for i in range(0, len(TOR._QminL[tarix]))]
            violup = [q[i] >  TOR._QmaxL[tarix][i] for i in range(0, len(TOR._QminL[tarix]))]
            #print "SINGLE viollo:",viollo
            #print "SINGLE viollup:",violup
            ok = not(any(viollo) or any(violup))

        return ok,viollo, violup



    class col:  #
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
            


