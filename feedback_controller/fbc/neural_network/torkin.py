from __future__ import print_function

import numpy as np
import sys, os
import roslibpy as rospy
from global_defines import TOR
from sklearn.neighbors import KDTree

from kinematics.generated_pycode.FK_left_GRASP import FK_left_GRASP as leftFK
from kinematics.generated_pycode.Jpos_left_GRASP import Jpos_left_GRASP as leftJpos
from kinematics.generated_pycode.Jzaxis_left_GRASP import Jzaxis_left_GRASP as leftJzaxis
from kinematics.generated_pycode.Jyaxis_left_GRASP import Jyaxis_left_GRASP as leftJyaxis

from kinematics.generated_pycode.FK_right_GRASP import FK_right_GRASP as rightFK
from kinematics.generated_pycode.Jpos_right_GRASP import Jpos_right_GRASP as rightJpos
from kinematics.generated_pycode.Jzaxis_right_GRASP import Jzaxis_right_GRASP as rightJzaxis
from kinematics.generated_pycode.Jyaxis_right_GRASP import Jyaxis_right_GRASP as rightJyaxis

from kinematics.generated_pycode.FK_head_HEAD import FK_head_HEAD as headFK
from kinematics.generated_pycode.Jpos_head_HEAD import Jpos_head_HEAD as headJpos
from kinematics.generated_pycode.Jzaxis_head_HEAD import Jzaxis_head_HEAD as headJzaxis
from kinematics.generated_pycode.Jyaxis_head_HEAD import Jyaxis_head_HEAD as headJyaxis

# function [q, pos_err,zax_err, TT] = IKwZAXIS(arm, qinit, qweight, p_des, R_des, errTH)
class bodypart:

    armKDposIX = range(1,10)  #[1,2,3,4,5,6,7,8,9,10]
    armKDsolIX = range(10,20)
    armKDstsIX = range(20,24) # 20,21,22,23
    
    def __init__(self,IK, torix, forward_cl, Jpos_cl, Jzaxis_cl, Jyaxis_cl, pos_errTH=0.001, ori_errTH=0.005):
        self.IK = IK  # the caller
        self.name = TOR._TARGET_L[torix];
        self.torix = torix
        self.pos_errTH = pos_errTH;
        self.ori_errTH = ori_errTH;
        self.forward_fn = forward_cl.forward_ik
        self.Jpos_fn   = Jpos_cl.compute_jac
        self.Jzaxis_fn  = Jzaxis_cl.compute_jac
        self.Jyaxis_fn  = Jyaxis_cl.compute_jac
        self.qinit  = None #np.array([0., 20, 53, 13, 60, 80,  50, 10, 10, 0])*np.pi/180  # defaults, should be overwritten
        self.qopt  =  np.array([0., 0, 52, 46, 8, 60, 25, 1, -10, 0])*np.pi/180 # nullspace optimization pulls to here
        #self.qopt = self.qopt.reshape([10,1])
        self.qweight = np.array([0.1, 0.1, 0.5, 0.25, 1, 1, 1, 1, 1, 0])               # defualts, should be overwritten  # last value was zero ERH
        self.q_min = None #np.array([-170, 0, -70, 5, -70, -45, -170, -105, -170, 0])*np.pi/180*0.99        # must be overwritten for body parts other than arms!
        self.q_max = None #np.array([170, 20, 250, 120, 250, 120, 170, 90, 170, 0.08*180/np.pi])*np.pi/180*0.99  # must be overwritten for body parts other than arms!
        self.kdtree= None
        self.kddata = None
        self.iktable_enable = True
        if torix==TOR._RARM: self.buildKD(IK.IKDATA_rarm)
        if torix==TOR._LARM: self.buildKD(IK.IKDATA_larm)
        #last element in the arrays correspond to gripper. Limits are in mm. (Currently not used in IK)
        print("bodypart structure for %s is constructed."%self.name)


    def buildKD(self, filename):
        data = []
        try:
            f = open(filename, 'r')
        except IOError: 
            print ("No data file named "+filename)
            return
        i = 0
        for line in f:
            #print(line)
            wordL = line.split(" ")
            #print(wordL)
            nums = np.ones([23+1])
            nums[0]=i   # add the index as the first entry
            i += 1
            for k in range(0, len(wordL)):
                    try:
                        nums[k+1] = float(wordL[k])
                    except:
                        print ("Warning there are entries that are not number")
                        print (line)
            data.append(nums)
        f.close()

        self.kddata = np.array(data)     # make everything array
        self.queries = self.kddata [:, bodypart.armKDposIX]  # these are x,y,z,Zx,Zy,Zz,Yx,Yy,Yz
        self.sols    = self.kddata [:, bodypart.armKDsolIX]  # there are the found solutions
        self.solstat = self.kddata [:, bodypart.armKDstsIX]  # [20] is a flag indicating ik success during table creation.
                                                             # [21]: poserr in meters; [22]:zaxis diff norm  [23]:yaxis diff norm

        self.kdtree = KDTree(self.queries[:,0:3], leaf_size=4)   # make the KDtree
        ## ix = np.array([all[:,0]]).transpose()     # this is order no. If the file is manually added this may not be consecqutive

        #print sols.shape
        #print ix.shape
        #print "sols:\n", sols      #np.hstack([ix,sols])
        #print "queries\n",queries   #np.hstack([ix,queries])
        #print "solstat:\n",solstat  #np.hstack([ix,solstat])


    def get_ikseed(self, p_des, R_des, mode='void', neig = 1 ):
        print("ikseed with mode="+mode)
        print(self.kdtree)
        if mode=='void':
            if self.kdtree == None or self.iktable_enable==False:
                return [self.qinit.copy(), -1]
        elif mode=='definit':
                return [self.qinit.copy(), -1]
        elif mode=='tableKD':
            print([p_des])
            dist, ix = self.kdtree.query(np.array([p_des]),neig)
            ix = ix[0];  dist = dist[0]
            clix = ix[0]  # closest point index in the table
            p = self.queries[clix] 
            print("nearest point to the one you asked:"+TOR.vec2str(p)+ "at index %d"%clix)
            return [self.sols[clix,:].copy(), clix]
        else:
            print("get_ikseed> No such mode:"+mode+" !! use definit or tableKD ")
            return [None, -1]       

    # TRY ./ezcom iktab rarm  /tmp/rarm60.txt 0.4 -0.4 0.80   0.6 0. 0.90  5 4 3     0 0 -1   1 0 0
    # TRY ./ezcom iktab rarm  /tmp/gg.txt     0.4 -0.4 0.85   0.6 0 0.85  3 3 1     0 0 -1   1 0 0/
    def make_iktab(self, ucom,args_s, args_f):
        tix = self.torix
        if tix != TOR._LARM and tix!=TOR._RARM:
            ucom.pprint("<> Sorry, no IK table construction for"+TOR._TARGET_L[tix])
            return [],[]
        if len(args_f)!= 16 and len(args_f)!= 10:
            print ("args_f len = %d"%len(args_f))
            print (args_f)
            ucom.pprint("<> Usage: iktab filename minx miny minz maxx maxy maxz gridx gridy gridz [z_axiz y_axis]")
            return [],[]
        ucom.request_setdesq(TOR._ALL, TOR.q_ezhome)
        rospy.sleep(2)
        save_state = self.iktable_enable
        self.iktable_enable = False
        
        #enter   X     Y    Z  specifications
        minpos=None #[0.4, -0.3, 0.8]
        maxpos=None #[0.8,  0.3, 0.9]
        step  = None #[ 4,    3,   2]
        defZaxis = [0.,0.,-1.]    # gripper pointing down
        defYaxis = [0.,1., 0.]    # open-close axis parallel to world-y
        filenm = args_s[0]
        lastsolOK = False
        lastq10 = None
        maxIKITERfirst = 1200 
        maxIKITER = 1000
        print("iktab> Will saving to "+filenm+" args_f:"+TOR.vec2str(args_f))
        
        minpos = args_f[1:4]
        maxpos = args_f[4:7]
        step   = args_f[7:10]
        print("iktab> minpos:"+TOR.vec2str(minpos)+" maxpos:"+TOR.vec2str(maxpos)+" step:"+TOR.vec2str(step))
        if len(args_s) == 16:    # filename, zaxis, yaxis
            des_z = args_f[10:13]
            des_y = args_f[13:16]
        else:
            des_z = np.array(defZaxis)
            des_y = np.array(defYaxis)

        qtorso = ucom.jcomm.getjointpos(TOR._TORSO)
        self.IK.set_torso_for_ik(qtorso)
        
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T    # normalization should be handled by torkin.ik 
        
        
    
        oldREPORTPROG = self.IK.set_REPORTPROG(False)           
        [A, points] = TOR.gen_ikpath(minpos,maxpos, step)   # create a minimum effector move parth over the grid
        print ("Point List:")
        print (points)
        TOR.printRot(R_des,"iktab> Will be using this orientation:")
        pntcnt = points.shape[0]
        sols     = np.ones([pntcnt,10])*np.nan
        queries    = np.ones([pntcnt,9])*np.nan
        solstat  = np.zeros([pntcnt,4])    # [0]: 0/1 solved or not the rest are errL contents 
        for k in range(0,points.shape[0]):
            p_des = points[k,0:3]
            if lastsolOK:
                q10 = lastq10
            else:
                q, q9, q10 = ucom.request_q(tix)

            ikITER = maxIKITER if lastsolOK else maxIKITERfirst
            
            print ("iktab> Solving "+TOR.vec2str(p_des)+" qinit:"+TOR.vec2str(q10*180./np.pi)+"deg")
            #cc=input("Go?")
            q10, solved, errL, itused = self.IK.ik(tix, p_des, R_des, qinit = q10, pos_errTH=None, ori_errTH=None, maxit=ikITER, IKstep_show = False)

            
            if q10 is None:
                ucom.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
                return []

            sols[k,:] = q10[:]
            solstat[k,0] = solved; solstat[k,1:4] = errL
            queries[k,0:3] = p_des
            queries[k,3:6] = R_des[:,2]
            queries[k,6:9] = R_des[:,1]

            
            if not solved:
                last10 = None
                lastsolOK = False
                ucom.pprint("<> iktab: Could not not reach the desired accuracy for point %d (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(k, errL[0]*1000,errL[1], errL[2]))          
            else:
                lastq10 = sols[k,:]  
                lastsolOK = True
                ucom.pprint("Solved.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                ucom.request_setdesq(tix,q10[2:9])
                ucom.request_setdesq(TOR._TORSO, q10[0:2])  # torso IK result must be sent if IK included it!!
                if k==0: rospy.sleep(5)
                else: rospy.sleep(0.5)

        self.IK.set_REPORTPROG(oldREPORTPROG)    # restore the original progress reporting mode
        self.iktable_enable = save_state         # restore the IKtable enable flag
        
        for k in range(0,points.shape[0]):
            print ("point %d"%k+("OK" if solstat[k,0]>0 else "XX")+ "   Errors:"+TOR.vec2str(solstat[k,1:4]))

        self.save_iktable(queries, sols, solstat, filenm)
        return  queries, sols, solstat

    def save_iktable(self,  queries, sols , solstat, fname):
        os.system("pwd")
        try:
            f = open(fname, 'w')
        except IOError: 
            print ("save_iktable> Canot open file for writing!! filename:"+fname)
            return
        for k in range(0,len(solstat)):
            #       pos, Zaxis, Yaxis (9)       torso, arm, gripper (10)      succ/fail, poserr, Zaxerr, Yaxerr (4) 
            s = TOR.vec2str(queries[k,:],"%+.2f")+" "+TOR.vec2str(sols[k,:],"%+.3f") +" "+TOR.vec2str(solstat[k,:],"%+.3f")
            print(str(k)+'> '+s)
            f.write(s+'\n')
        f.close()
        print("save_iktable> %d entries saved to"%len(solstat),fname)
            

                   

    def forwardkin(self, q):
        if self.torix == TOR._HEAD:
            T0 = self.forward_fn(q[0],q[1],q[2],q[3])   # This seems to have a bug. 
            locYrot90=TOR.rotYhom(np.pi/2)                # to get it fixed we apply a fixed rotation
            T  = np.matmul(T0,locYrot90)                # rotate in the T0 Y axis
            ##TOR.printXform(T,'\nHomeg. Xform:')

        else:    
            T = self.forward_fn(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9])
        return T
    def Jpos(self, q):
        if self.torix == TOR._HEAD:
            Jp = self.Jpos_fn(q[0],q[1],q[2],q[3])
        else:
            Jp = self.Jpos_fn(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9])
        return Jp
    def Jzaxis(self,q):
        if self.torix == TOR._HEAD:
            Jz = self.Jzaxis_fn(q[0],q[1],q[2],q[3])
        else:
            Jz = self.Jzaxis_fn(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9])
        return Jz
    def Jyaxis(self,q):
        if self.torix == TOR._HEAD:
            Jy = self.Jyaxis_fn(q[0],q[1],q[2],q[3])
        else:
            Jy = self.Jyaxis_fn(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9])
        return Jy
    
class TorKin:
    IKDATA_rarm = "/home/erhan/catkin_ws/src/erhtor_work/tormain/Resources/rarm60.txt"
    IKDATA_larm = "/home/erhan/catkin_ws/src/erhtor_work/tormain/Resources/larm60.txt"
    # min-max for IIK (teleoperation) joints as individual limbs as list
    QminL =[ np.array([-170/4.0, -60*0.2])*np.pi/180,                                # torso
          np.array([-70, -35, -70, -60*0+2, -170, -105, -170])*np.pi/180,     # rightarm       #elbow joint is restricted to positive angles for IK
          np.array([0.0, -80.]),                                          # rightgripper
          np.array([-70, -35, -70, -60*0+2, -170, -105, -170])*np.pi/180,     # leftarm        #elbow joint is restricted to positive angles for IK
          np.array([0.02, -30.]),                                          # leftgripper
          np.array([-45,-45])*np.pi/180     #FIX#                         # head
         ]   # zero posture for ezcontroller
    QmaxL= [ np.array([170, 80*0.4])*np.pi/180,                                  # torso
          np.array([250,105,250,120,170,90*.75,170])*np.pi/180,               # rightarm
          np.array([0.08, 80]),                                               # rightgripper
          np.array([250,105,250,120,170,90*.75,170])*np.pi/180,               # leftarm
          np.array([0.02, 30]),                                               # leftgripper
          np.array([45,45])*np.pi/180    #FIX#                            # head
         ]
    
    def __init__(self):
        self.ucom = None      # hook back to usercomm for being able to issue user commands. Use register_ucom()
        self.VERBOSE = False
        self.REPORTPROG = False
        self.bodypartL = [None]*TOR._ALL

        self.bodypartL[TOR._LARM] = self.leftarm  = bodypart(self, TOR._LARM,  leftFK,  leftJpos,  leftJzaxis, leftJyaxis)
        self.bodypartL[TOR._RARM] = self.rightarm = bodypart(self, TOR._RARM,rightFK, rightJpos, rightJzaxis, rightJyaxis)
        self.bodypartL[TOR._HEAD] = self.head     = bodypart(self, TOR._HEAD,headFK, headJpos, headJzaxis, headJyaxis)

        
        # USED by IIK i.e. teleop mode
        self.leftarm.q_min = 0*np.array([-170, -20, -70, -35, -70, -45, -170, -105, -170, 0])*np.pi/180*0.95        # must be overwritten for body parts other than arms!
        self.leftarm.q_max = 0*np.array([170,   20, 250,  104, 250, 120, 170, 90, 170, 0.08*180/np.pi])*np.pi/180*0.95  # must be overwritten for body parts other than arms!
        
        # USED by IK batch version
        self.leftarm.Q_min = 0*np.array([-170, -20, -70, -35, -70, -45, -170, -105, -170, 0])*np.pi/180        # must be overwritten for body parts other than arms!
        self.leftarm.Q_max = 0*np.array([170,   20, 250,  104, 250, 120, 170, 90, 170, 0.08*180/np.pi])*np.pi/180   # must be overwritten for body parts other than arms!

        self.leftarm.qinit  = np.array([0., 20, 53, 13, 60, 80,  50, 10, 10, 0])*np.pi/180
        self.rightarm.qinit = np.array([0., 20, 53, 13, 60, 80,  50, 10, 10, 0])*np.pi/180 
        eps = 0
        self.leftarm.q_min[2:9]  = TorKin.QminL[TOR._LARM][0:]+eps 
        self.leftarm.q_min[0:2] = TorKin.QminL[TOR._TORSO][0:2]+eps    #TODO: this is not general made for arm with 10 jnt IK.
        self.leftarm.Q_min[:]    = self.leftarm.q_min[:]
        self.leftarm.q_max[0:2] = TorKin.QmaxL[TOR._TORSO][0:2]-eps
        self.leftarm.q_max[2:9]  = TorKin.QmaxL[TOR._LARM][0:]-eps 
        self.leftarm.Q_max[:]    = self.leftarm.q_max[:]
        
        
        self.rightarm.q_min = self.leftarm.q_min[:]   # same limits for rightarm.
        self.rightarm.q_max = self.leftarm.q_max[:]   # note that these include also TORSO joints
        self.rightarm.Q_min = self.leftarm.Q_min[:]   # same limits for rightarm.
        self.rightarm.Q_max = self.leftarm.Q_max[:]   # note that these include also TORSO joints    

 
        self.qtorso_for_ik = np.array([0.0,0.0]);
        self.callc = 0   # a counter
        self.I     = np.eye(10)

    def set_REPORTPROG(self,state):
        oldst = self.REPORTPROG
        self.REPORTPROG = state
        return oldst

            
    def make_iktab(self, ucom, tix, args_s, args_f):
        print (tix)
        print(TOR._TARGET_L[tix])
        if tix == TOR._LARM:
            self.leftarm.make_iktab(ucom,args_s, args_f)
        elif tix == TOR._RARM: self.rightarm.make_iktab(ucom,args_s, args_f)
        else:  print("make_iktab: IK Table construction is supported for only rarm and larm")

    """
    Returns the pose of the indicated bodypart as p, R (R:rotation matrix)
    """
    def forwardkin(self, bodypartix, q):
        if bodypartix<TOR._TORSO or bodypartix>TOR._HEAD:
            print("Bad bodypartix %d ! Aborting forwardkin() ."%bodypartix)
            return None, None
        part = self.bodypartL[bodypartix]
        if part == None:
            print("No kinematic data is setup for %s. Sorry."%TOR._TARGET_L[bodypartix])
            return None, None
        
        q_padded = np.hstack([q,[0 for i in range(0, 10-len(q))]])   # add zeros to make it 10
        #print q_padded
        T = part.forwardkin(q_padded)  
        p_cur,  z_cur, y_cur = T[0:3,3], T[0:3,2], T[0:3,1]
        return p_cur, T[0:3,0:3]

    def circleTraj(self,bodypartix, p_des, R_des, r=0.15, qinit=None, qweight=None, pos_errTH=None, ori_errTH=None, maxit=1000):
         
        if bodypartix<TOR._TORSO or bodypartix>TOR._HEAD:
            print("Bad bodypartix %d ! Aborting IK solve request."%bodypartix)
            return None, False, [None, None, None]
        part = self.bodypartL[bodypartix]
        if part == None:
            print("No kinematic data is setup for %s. Sorry."%TOR._TARGET_L[bodypartix])
            return None, False, [None, None, None]
        
        if qinit is None: #len(qinit)==None:
            qinit = part.qinit.copy()
            ##qinit[0:2] = self.qtorso_for_ik
            print ("qinit updated:",qinit)
        if qweight==None:
            qweight = part.qweight
        if pos_errTH==None:
            pos_errTH = part.pos_errTH
        if ori_errTH==None:
            ori_errTH = part.ori_errTH

        a = 0.0
        N = 20
        i = 0
        ptraj = np.array([N+1,3])
        while a <= 20:
            x,y,z = r*np.cos(a), r*np.sin(a), 0
            u  = np.array([x,y,z])
            uu = p_des + np.multiply(R_des, u)
            a += 2*np.pi/N
            ptraj[i,:] = uu
            
            print (i, uu[0],uu[1], uu[2])
            #q, des_reached, [pos_err, zax_err, yax_err] = self._IKwZAXIS(part, qinit, qweight, p_des, R_des, pos_errTH, ori_errTH, maxit)
            #maxit = 5
            #qinit = q
            i += 1
            
        return ptraj


    

        
    # TODO: For body parts other than arms (TORSO, HEAD etc), set these in the bodypart class:
    # qinit, qweight, q_min, q_max. They are set according to arms by default.
    def ik_table(self,bodypartix, p_des, R_des, qinit=None, qweight=None, pos_errTH=None, ori_errTH=None, maxit=1000, IKstep_show=False):
        print ("in ik_Table, bodypartix:"+str(bodypartix))
        print (self.bodypartL[bodypartix])
        if self.ispartOK4IK(bodypartix)==False: return None, False, [None, None, None],0         
        part = self.bodypartL[bodypartix]
        print ("part name:"+part.name)
        if qinit is None:
            qinit, ign = part.get_ikseed(p_des, R_des,mode='tableKD')
            if self.VERBOSE: print("ik> --------This qinit (from ik_seed) will be used for ik:",TOR.vec2str(180/np.pi*qinit,'%2.2f'))
        else:
            if self.VERBOSE: print("ik> --------This qinit is *given* for ik:",TOR.vec2str(180/np.pi*qinit,'%2.2f')    )
        if qweight==None:   qweight = part.qweight
        if pos_errTH==None: pos_errTH = part.pos_errTH
        if ori_errTH==None: ori_errTH = part.ori_errTH
        self.print_ikpar(qinit, qweight, pos_errTH, ori_errTH, maxit)
        print ("Calling ikwzaxis...")
        return self._IKwZAXIS(part, qinit, qweight, p_des, R_des, pos_errTH, ori_errTH, maxit, IKstep_show)
         
        #print ("maxit     :"+str(maxit))
        #if self.VERBOSE: print ("----------------------------------")


    # TODO: For body parts other than arms (TORSO, HEAD etc), set these in the bodypart class:
    # qinit, qweight, q_min, q_max. They are set according to arms by default.
    def ik(self,bodypartix, p_des, R_des, qinit=None, qweight=None, pos_errTH=None, ori_errTH=None, maxit=1000, IKstep_show=False):       
        if self.ispartOK4IK(bodypartix)==False: return None, False, [None, None, None],0
        part = self.bodypartL[bodypartix]
        if qinit is None: 
            qinit, ign = part.get_ikseed(p_des, R_des, mode='definit')
            if self.VERBOSE: print("ik> --------qinit will be definit which is:",TOR.vec2str(180/np.pi*qinit,'%2.2f'))
        else:
            if self.VERBOSE: print("ik> --------qinit *given* as:",TOR.vec2str(180/np.pi*qinit,'%2.2f')    )
        if qweight==None:   qweight = part.qweight
        if pos_errTH==None: pos_errTH = part.pos_errTH
        if ori_errTH==None: ori_errTH = part.ori_errTH
            
        self.print_ikpar(qinit, qweight, pos_errTH, ori_errTH, maxit)
        return self._IKwZAXIS(part, qinit, qweight, p_des, R_des, pos_errTH, ori_errTH, maxit, IKstep_show)
    
    def _norm(self, x):
         return np.inner(x,x)**0.5

    def set_torso_for_ik(self, tq):
        self.qtorso_for_ik[0:2] = tq[0:2]
        #print "tq:", tq
        #print "self.qtorso_for_ik:",self.qtorso_for_ik
    def get_torso_for_ik(self):
        return self.qtorso_for_ik.copy()


    
    def print_ikpar(self, qinit, qweight, pos_errTH, ori_errTH, maxit):
        print ("--------IK PARAMS USED ------------")
        print ("qinit     :"+TOR.vec2str(qinit*180/np.pi)+"degrees")
        print ("qweight   :"+TOR.vec2str(qweight))
        print ("pos_errTH :"+str(pos_errTH))
        print ("ori_errTH :"+str(ori_errTH))


    def ispartOK4IK(self, bodypartix):
        if bodypartix<TOR._TORSO or bodypartix>TOR._HEAD:
            print("ispartOK4IK> Bad bodypartix %d ! Aborting IK solve request."%bodypartix)
            return False
        part = self.bodypartL[bodypartix]
        if part == None:
            print("ik> No kinematic data is setup for %s. Sorry."%TOR._TARGET_L[bodypartix])
            return False

        #print ("is part OK!")
        return True

    # TODO: For body parts other than arms (TORSO, HEAD etc), set these in the bodypart class:
    # qinit, qweight, q_min, q_max. They are set according to arms by default.
    def iik(self,bodypartix, p_des, R_des, qinit=None, pos_errTH=None, ori_errTH=None, maxit=1000):
        if self.ispartOK4IK(bodypartix)==False: return None, False, [None, None, None],0
        part = self.bodypartL[bodypartix]
        if qinit is None:  
            qinit = part.qinit.copy()
            #qinit, ign = part.get_ikseed(p_des, R_des, mode='definit')  #use this instead      
            if self.VERBOSE: print( "iik> qinit updated:",qinit)
        else:
            if self.VERBOSE: print( "iik> qinit provided as:"+TOR.vec2str(qinit*180/np.pi))

        if pos_errTH==None: pos_errTH = part.pos_errTH
        if ori_errTH==None: ori_errTH = part.ori_errTH
            
        return self._IIKwZAXIS(part, qinit, p_des, R_des, pos_errTH, ori_errTH, maxit)
    
     # This is started as a translation from the matlab version.
    def _IIKwZAXIS(self, part, qinit, in_p_des, in_R_des, pos_errTH, ori_errTH, MAXIT=1000):
        
        TORSO_RESTORE_gain = [0.05,0.05]*1   # when _IIK can use torso this will act as pull torso to qopt[0:2]
        USETORSO = False
        enforce_torso_angle = np.array([0.0,10.0])*np.pi/180
        PINV = True
        z_table = 0.55   # TODO: make this more user friendly, see the IK hack below
        ori_Gain = 3.0
        pos_Gain = 2.0
        #ori_Mask = np.array([0.0, 0.0, 0.05, 0.05  , 0.1, 0.5, 1, 1, 1, 0.0])
        Zax_Mask = np.array([0.0, 0.0,      0.05, 0.05, 0.1, 0.5, 1, 1, 0, 0.0])
        Yax_Mask = np.array([0.0, 0.0,      0.0,  0.0 , 0.0, 0.0, 0, 0, 1, 0.0])
        pos_Mask = np.array([0.05, 0.25,    0.5 , 1.0 , 0.2, 1.0, 0.05, 0.05, 0, 0.0])  # 5rd is the extra redundancy rotation
        #Zax_Mask = np.array([0.0, 0.0, 0.05, 0.05, 0.1, 0.5, 1, 1, 0, 0.0])
        #Yax_Mask = np.array([0.0, 0.0, 0.0,  0.0 , 0.0, 0.0, 0, 0, 1, 0.0])
        #pos_Mask = np.array([0.2, 0.2, 1.0 , 1.0   , 1.0, 1.0, 0., 0., 0, 0.0])

        
        #STEP_RATE = 2
        it = 0

        self.callc += 1
        p_des = in_p_des.copy()

        if p_des[2] < z_table: p_des[2] = z_table*1.045
        
        R_des = in_R_des.copy()       
        R_des[:,0] = R_des[:,0]/self._norm(R_des[:,0])
        R_des[:,1] = R_des[:,1]/self._norm(R_des[:,1])
        R_des[:,2] = R_des[:,2]/self._norm(R_des[:,2])
        #if self.VERBOSE: TOR.printRot(R_des, "_IIKwZAXIS> Normailzed R_des = ")
        # Note this is probably not a sufficient check
        normdiff =  self._norm(np.cross(R_des[:,0],R_des[:,1])-R_des[:,2])
        det = np.linalg.det(R_des)
        if abs(det-1)>1e-2 or normdiff>1e-2:
            print('_IIKwZAXIS>This does not seem to be a proper rotation matrix (even after normalization)!\n');
            print(in_R_des)
            print( "_IIKwZAXIS>determinant (normalzied):",det)
            print( "_IIKwZAXIS>X cross Y=",np.cross(R_des[:,0],R_des[:,1]), "and Z=",R_des[:,2], 'norm of diff: ', normdiff, '(normalized)')
            TOR.printRot(R_des,'_IIKwZAXIS>After normalization:')
            return None, None, None

        if self.VERBOSE:
            print( '_IIKwZAXIS> Solving IK for pos:', p_des)
            print( '_IIKwZAXIS> R_des:')
            print( R_des)

        q = qinit 
        if enforce_torso_angle is not None:
            q[0:2] = enforce_torso_angle
        
        z_des = R_des[0:3,2]
        y_des = R_des[0:3,1]
        q_at_start = q;
        if self.VERBOSE: print( "q_at_start [deg]:",TOR.vec2str(q*180.0/np.pi,"%2.2f"))
        T = part.forwardkin(q)  
        p_cur,  z_cur, y_cur = T[0:3,3], T[0:3,2], T[0:3,1]
        J, Jz, Jy = part.Jpos(q), part.Jzaxis(q), part.Jyaxis(q)
        pos_err, zax_err, yax_err = self._norm(p_des - p_cur), self._norm(z_des - z_cur), self._norm(z_des - y_cur)
        
        while (pos_err > pos_errTH or zax_err > ori_errTH or yax_err > ori_errTH) and it < MAXIT:
            it = it + 1
            T = part.forwardkin(q)  
            p_cur,  z_cur, y_cur = T[0:3,3], T[0:3,2], T[0:3,1]
            J, Jz, Jy = part.Jpos(q), part.Jzaxis(q), part.Jyaxis(q)

            dp = p_des - p_cur
            dz = z_des - z_cur
            dy = y_des - y_cur
          
            if p_cur[2] < z_table:
                if dp[2]>0:    # if des z position change is compatible with clearing the table
                    dp[2] = dp[2] + (z_table - p_cur[2])   
                else:           # if not compatible ignore the desired z position and clear the table 
                    dp[2] = (z_table - p_cur[2])
                dz = 0.02*dz   # suppress movements due to des_ori during table coll.
                dy = 0.02*dy   # suppress movements due to des_ori during table coll.
                print ("_IIKwZAXIS> Avoiding table collision! z:",p_cur[2]," z_des:",p_des[2], "z_table:",z_table)

            pos_err, zax_err, yax_err = self._norm(p_des - p_cur), self._norm(z_des - z_cur), self._norm(y_des - y_cur)
            if (pos_err <= pos_errTH and zax_err <= ori_errTH and yax_err <= ori_errTH):  # TODO: note not checking yaxis_err!
                if self.VERBOSE or True: print ("_IIKwZAXIS> Found solution!")
                continue

            if PINV:
                dq_pos = pos_Gain*pos_Mask*np.matmul( np.linalg.pinv(J),dp)
                dq_zax = ori_Gain*Zax_Mask*np.matmul( np.linalg.pinv(Jz),dz)
                dq_yax = ori_Gain*Yax_Mask*np.matmul( np.linalg.pinv(Jy),dy)
            else:
                dq_pos = pos_Gain*pos_Mask*np.matmul(J.T,dp)
                dq_zax = ori_Gain*Zax_Mask*np.matmul(Jz.T,dz)
                dq_yax = ori_Gain*Yax_Mask*np.matmul(Jy.T,dy)
            
            # use different weight for orientation and position ! (almost decoupling..)
            #dq = pos_Mask*dq_pos + ori_Mask*(0.5*dq_zax+0.5*dq_yax)
            dq = dq_pos + dq_zax + dq_yax
            if USETORSO == False: dq[0:2]=[0,0.0] 
            q  = q + dq #*qweight   # note component-wise product
 
            eps = 0.00
            lobo_viol = 1*(q < part.q_min-eps)
            hibo_viol = 1*(q > part.q_max+eps)
            #print ("qweight:"+TOR.vec2str(qweight))
            #print (lobo_viol)
            if sum(lobo_viol)>0:
                if self.VERBOSE or True:
                    print (self.callc, '> _IIKwZAXIS> Joint LOWER limit hit at:%s \n'%TOR.vec2str(lobo_viol,'%d'), "q:%s"%TOR.vec2str(180/np.pi*q,'%2.2f'), "(deg) will be rectified. [min:", TOR.vec2str(180/np.pi*part.q_min,'%2.2f'))
            if sum(hibo_viol)>0:
                if self.VERBOSE or True:
                    print (self.callc, '> _IIKwZAXIS>Joint UPper limit hit at:%s \n'%TOR.vec2str(hibo_viol,'%d'), "q:%s"%TOR.vec2str(180/np.pi*q,'%2.2f'), "(deg) will be rectified")

            q =lobo_viol*part.q_min + (1-lobo_viol)*q   # emulate matlab q = max(q_min, q)
            q = hibo_viol*part.q_max + (1-hibo_viol)*q  # emulate matlab q = min(q, q_max)
            
            if sum(lobo_viol)>0 or  sum(hibo_viol)>0:
                if self.VERBOSE: print (self.callc, '> _IIKwZAXIS> Rectified q:%s'%TOR.vec2str(180/np.pi*q,'%3.2f'))

            
            if USETORSO==True: # apply natural pose force for torso when IK is using torso
                q[0] = q[0] + (part.qopt[0] - q[0])*TORSO_RESTORE_gain[0]
                q[1] = q[1] + (part.qopt[1] - q[1])*TORSO_RESTORE_gain[1]
            
            #if it%50 == 5:
            #    send_to_controlserver(q,arm)
            if (self.VERBOSE or self.REPORTPROG) and it%20 == 0:
                print('_IIKwZAXIS> it=%d > pos, ori error:%2.4f, %2.4f\n'%(it, pos_err, zax_err))
         # END of while (IK iteration)        
            
        des_reached = False if it == MAXIT else True

        if self.VERBOSE:
            if it == MAXIT:
                print('_IIKwZAXIS> IK did not succeed. The current pos, ori error:%2.4f, %2.4f\n'%(pos_err, zax_err));
            else:
                print('_IIKwZAXIS> Solved. \n Try: \n FK_right_GRASP(%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f)\n'%(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9]))

        if (not des_reached): 
            if self.REPORTPROG and MAXIT>200:
                print('_IIKwZAXIS> Warning desired position accuracy could not be reached (pos_err:%2.4fmm, zax_err:%2.4f, yax_err:%2.4f) in %d iterations.\n'%(pos_err*1000,zax_err, yax_err, it)) 
        else:
            if self.VERBOSE or self.REPORTPROG:
                print('_IIKwZAXIS> Desired position accuracy reached (pos_err:%2.4fmm, zax_err:%2.4f, yax_err:%2.4f) in %d iterations.\n'%(pos_err*1000,zax_err, yax_err, it))
        

        if self.VERBOSE:
            found_T = part.forwardkin(q)
            asked_T = np.column_stack( [np.row_stack([R_des, [0.,0,0]]), np.hstack([p_des,1])]) # Emulating Matlab: [[R_des; 0 0 0],[p_des; 1]];
            TOR.printXform(found_T - asked_T, '_IIKwZAXIS> Difference between desired and found Transformation matrix:')

        return q, des_reached, [pos_err, zax_err, yax_err]
########### End of __IKwZAXIS()

   
    # This is started as a translation from the matlab version.
    # Must be used for batch IK
    def _IKwZAXIS(self, part, qinit, qweight, in_p_des, in_R_des, pos_errTH, ori_errTH, MAXIT=1000, IKstep_show=False):
        ori_Gain = 3.0/4
        pos_Gain = 2.0/2
        ori_IK_Mask = np.array([0.0, 0.0, 0.05, 0.05  , 0.1, 0.5, 1, 1, 1, 0.0])
        pos_IK_Mask = np.array([0.0, 0.0, 1.0 , 1.0   , 1.0, 1.0, 0, 0, 0, 0])

        Zax_Mask = np.array([0.0, 0.0, 0.05, 0.05, 0.1, 0.5, 1, 1, 0, 0.0])
        Yax_Mask = np.array([0.0, 0.0, 0.0,  0.0 , 0.0, 0.0, 0, 0, 1, 0.0])
        pos_Mask = np.array([0.2, 0.2, 1.0 , 1.0   , 1.0, 1.0, 0., 0., 0, 0.0])
        #STEP_RATE = 2
        it = 0
        IK_PINV = True
        IK_USETORSO = True
 
        p_des = in_p_des.copy()
        R_des = in_R_des.copy()
       
        R_des[:,0] = R_des[:,0]/self._norm(R_des[:,0])
        R_des[:,1] = R_des[:,1]/self._norm(R_des[:,1])
        R_des[:,2] = R_des[:,2]/self._norm(R_des[:,2])
        if self.VERBOSE:
            print( "_IKwZAXIS> Normailzed R_des = ",R_des)
        # Note this is probably not a sufficient check
        normdiff =  self._norm(np.cross(R_des[:,0],R_des[:,1])-R_des[:,2])
        det = np.linalg.det(R_des)
        if abs(det-1)>1e-2 or normdiff>1e-2:
            print('_IKwZAXIS> This does not seem to be a proper rotation matrix (even after normalization)!\n');
            print(in_R_des)
            print( "_IKwZAXIS> determinant (normalzied):",det)
            print( "_IKwZAXIS> X cross Y=",np.cross(R_des[:,0],R_des[:,1]), "and Z=",R_des[:,2], 'norm of diff: ', normdiff, '(normalized)')
            print('_IKwZAXIS> After normalization:')
            print(R_des)
            return None, None, None

        if self.VERBOSE:
            print( '_IKwZAXIS> Solving IK for pos:', p_des)
            print( '_IKwZAXIS> R_des')
            print( R_des)

        q = qinit 
        z_des = R_des[0:3,2]
        y_des = R_des[0:3,1]
        q_at_start = q;
        if self.VERBOSE: print ("_IKwZAXIS> q_at_start [deg]:",TOR.vec2str(q*180.0/np.pi,"%2.2f"))
        T = part.forwardkin(q)  
        p_cur,  z_cur, y_cur = T[0:3,3], T[0:3,2], T[0:3,1]
        J, Jz, Jy = part.Jpos(q), part.Jzaxis(q), part.Jyaxis(q)

        
        pos_err, zax_err, yax_err = self._norm(p_des - p_cur), self._norm(z_des - z_cur), self._norm(z_des - y_cur)


        while (pos_err > pos_errTH or zax_err > ori_errTH or yax_err > ori_errTH) and it < MAXIT:
            it = it + 1
            T = part.forwardkin(q)  
            p_cur,  z_cur, y_cur = T[0:3,3], T[0:3,2], T[0:3,1]
            J, Jz, Jy = part.Jpos(q), part.Jzaxis(q), part.Jyaxis(q)
            #print "IK",it," > q=", TOR.vec2str(q*180.0/np.pi,"%2.2f")
            #print '-----'
            #print J
            dp = p_des - p_cur
            dz = z_des - z_cur
            dy = y_des - y_cur

            pos_err, zax_err, yax_err = self._norm(p_des - p_cur), self._norm(z_des - z_cur), self._norm(y_des - y_cur)

            if (pos_err <= pos_errTH and zax_err <= ori_errTH and yax_err <= ori_errTH):  # TODO: note not checking yaxis_err!
                if self.VERBOSE: print ("_IKwZAXIS> Found solution!")
                continue


            if IK_PINV:
                dq_pos = pos_Gain*np.matmul( np.linalg.pinv(J),dp)
                dq_zax = ori_Gain*np.matmul( np.linalg.pinv(Jz),dz)
                dq_yax = ori_Gain*np.matmul( np.linalg.pinv(Jy),dy)
            else:
                dq_pos = pos_Gain*np.matmul(J.T,dp)
                dq_zax = ori_Gain*np.matmul(Jz.T,dz)
                dq_yax = ori_Gain*np.matmul(Jy.T,dy)
                           
            #use different weight for orientation and position ! (almost decoupling..)
            dq = pos_IK_Mask*dq_pos + ori_IK_Mask*(0.5*dq_zax+0.5*dq_yax)  # dq_yax had 0.2*  before
            q  = q + dq*qweight 
            
#            if IK_PINV:
#                dq_pos = pos_Gain*pos_Mask*np.matmul( np.linalg.pinv(J),dp)
#                dq_zax = ori_Gain*Zax_Mask*np.matmul( np.linalg.pinv(Jz),dz)
#                dq_yax = ori_Gain*Yax_Mask*np.matmul( np.linalg.pinv(Jy),dy)
#            else:
#                dq_pos = pos_Gain*pos_Mask*np.matmul(J.T,dp)
#                dq_zax = ori_Gain*Zax_Mask*np.matmul(Jz.T,dz)
#                dq_yax = ori_Gain*Yax_Mask*np.matmul(Jy.T,dy)
#            
#            dq = dq_pos + dq_zax + dq_yax
#            if IK_USETORSO == False: dq[0:2]=[0.0,0.0]
#            q  = q + dq #*qweight   # note component-wise product
 
            # note component-wise product
            # apply natural pose force for torso
            #q[0:2] = q[0:2] + (part.qopt[0:2] - q[0:2])*0.1/qweight[0:2]
            
            #bu ne? q[:] = q[:] + (part.qopt[:] - q[:])*0.1/qweight[:]
            ###print "dq:",dq*qweight
            eps = 0.00
            lobo_viol = 1*(q < part.Q_min-eps)
            hibo_viol = 1*(q > part.Q_max+eps)

            
            if sum(lobo_viol)>0:
                if self.REPORTPROG:
                    print( '_IKwZAXIS> Joint LOwer limit hit at:%s \n'%TOR.vec2str(lobo_viol,'%d'), "q:%s"%TOR.vec2str(180/np.pi*q,'%2.2f'), "(deg) will be rectified. [min:", TOR.vec2str(180/np.pi*part.q_min,'%2.2f'))
            if sum(hibo_viol)>0:
                if self.REPORTPROG:
                    print( '_IKwZAXIS> Joint UPper limit hit at:%s \n'%TOR.vec2str(hibo_viol,'%d'), "q:%s"%TOR.vec2str(180/np.pi*q,'%2.2f'), "(deg) will be rectified")

            q =lobo_viol*part.Q_min + (1-lobo_viol)*q   # emulate matlab q = max(q_min, q)
            q = hibo_viol*part.Q_max + (1-hibo_viol)*q  # emulate matlab q = min(q, q_max)

            if sum(lobo_viol)>0 or  sum(hibo_viol)>0:
                if self.REPORTPROG: print( '_IKwZAXIS>  Rectified q:%s'%TOR.vec2str(180/np.pi*q,'%3.2f')  )          
            if it%20 == 0 and self.REPORTPROG:
                print('_IKwZAXIS*> it = %d > pos, ori error:%2.4f, %2.4f'%(it, pos_err, zax_err), end='\r')
            if IKstep_show and it%10 == 1:
                self._send_to_controlserver(q,part.torix)
         # END of while (IK iteration)        
            
        des_reached = False if it == MAXIT else True

##        if self.VERBOSE or self.REPORTPROG:
##            if it == MAXIT:
##                print('_IKwZAXIS> IK did not succeed. The current pos, ori error:%2.4f, %2.4f\n'%(pos_err, zax_err));
##                #p_des_vs_p_cur_Del = [p_des,     p_cur, p_des - p_cur]
##                #z_des_vs_z_cur_Del = [z_des,     z_cur, z_des - z_cur]
##            else:
##                print('_IKwZAXIS> Solved. \n Try: \n FK_right_GRASP(%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f)\n'%(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9]))

        if (not des_reached): 
            if self.VERBOSE or self.REPORTPROG: print('_IKwZAXIS> Warning desired position accuracy could not be reached (pos_err:%2.4fmm, zax_err:%2.4f, yax_err:%2.4f) in %d iterations.\n'%(pos_err*1000,zax_err, yax_err, it)) 
        else:
            if self.VERBOSE or self.REPORTPROG:
                print('_IKwZAXIS> Desired position accuracy reached (pos_err:%2.4fmm, zax_err:%2.4f, yax_err:%2.4f) in %d iterations.\n'%(pos_err*1000,zax_err, yax_err, it))
            if IKstep_show: self._send_to_controlserver(q,part.torix)

        TT = part.forwardkin(q)
        found_T = TT
        asked_T = np.column_stack( [np.row_stack([R_des, [0,0,0]]), np.hstack([p_des,1]) ]  )
        # Emulating Matlab: asked_T = [[R_des; 0 0 0],[p_des; 1]];
        if self.VERBOSE:
            print( "_IKwZAXIS> Difference between desired and found Transformation matrix:")
            print( found_T - asked_T)

        return q, des_reached, [pos_err, zax_err, yax_err], it
## End of _IKwZAXIS()

    
    def _register_ucom(self, ucom):
        self.ucom = ucom
        
    def _send_to_controlserver(self,q, torix):
        qq = q*180/np.pi;
        cmd = ''
        #if strcmp(arm,'right')
        #    cmd = sprintf('../communication/ezpycom setqT_deg rarm %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n', qq(1), qq(2), qq(3),qq(4),qq(5),qq(6),qq(7),qq(8),qq(9));
        #else
        #    cmd = sprintf('../communication/ezpycom setqT_deg larm %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n', qq(1), qq(2), qq(3),qq(4),qq(5),qq(6),qq(7),qq(8),qq(9));
        #end
        self.ucom.request_setdesqT(torix, q[0:9])
        #system(cmd);
        


def torkin_main():
    q=[0,0.5,1,0,0,0,0,0,0,0]
    kin = TorKin()
    #leftarmT = kin.leftarm.forwardkin(q)
    #print "\nleftarm T:"
    #print leftarmT
    #leftarmJpos = kin.leftarm.Jpos(q)
    #print "\nleftarm Jpos:"
    #print leftarmJpos

    # Let's solve for Left Arm
    print("\n\n------ SOLVING for LEFT ARM ------------")
    des_z = np.array([0.0,-1,-1])
    des_y = np.array([1.0, 0, 0])
    des_x = np.cross(des_y,des_z)
    R_des = np.array([des_x, des_y, des_z]).T
    p_des = np.array([0.4, 0, 1.0])
    #print "R_des:"
    #print R_des
    kin.ik(TOR._LARM, p_des, R_des)

    # Let's solve for Right Arm
    print("\n\n------ SOLVING for RIGHT ARM ------------")
    des_z =np.array([0, 1.,-1])
    des_y =np.array([1, 0., 0])
    des_x = np.cross(des_y,des_z)
    R_des = np.array([des_x, des_y, des_z]).T
    p_des = np.array([0.4, 0, 1.0])
    kin.ik(TOR._RARM, p_des, R_des)
if __name__ == '__main__':
    print("**** Running [torkin] package as standalone test mode ****");
    torkin_main()
