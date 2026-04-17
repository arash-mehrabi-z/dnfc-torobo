#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy, socket
import numpy as np
from threading import Lock, Thread
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.msg import GripperCommand
from torobo_msgs.msg import ToroboJointState
from std_msgs.msg import String
#import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from numpy.linalg import norm
from sidecode.butterflyfun import trace_butterfly
#from sklearn.neighbors import KDTree
#import pickle
from scipy.interpolate import CubicSpline, LinearNDInterpolator,interp1d
from gazebo_msgs.srv import GetModelState
import learningcode.cnmpfa 
#from k_nearest_neighbor import KNearestNeighbor
import matplotlib.pyplot as plt
from lookuptraj import LOOKUPTRAJ
from deniz_recorder import Recorder
import time
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
try:
    touchEXISTS = True
    from tortouch import touchComm                   
except ImportError:
    touchEXISTS = False
    print "Your system does not seem to have Touch Device"
    

# Imports for TORMAIN system
from global_defines import TOR            # hook to global defined
from learner import LearningLoop
from torkin import TorKin                # hook to IK routines
if __name__ == '__main__':
    import controller                         # hook to P controller
#from torvis import simpleColor
from listenblob import blobLogger
from ar_tag_detect import ar_tag_detect
#From controller only ezController is used but "from controller import ezController"  fails (probably due to circular imports)

import time

class LatVecRec:
    LATVEC_TOPIC_NAME = '/cameraImageString'
    def __init__(self):
        self.latvec_subs = rospy.Subscriber(LatVecRec.LATVEC_TOPIC_NAME, String, self.latvec_callback, queue_size = 20)
        self.latveclock  = Lock()
        self.last_latvec  = np.zeros(128)
    def latvec_callback(self, data):
        s = data.data
        #print 'LATVEC_CALLBACK:' ,s
        sL = s.split(' ')
        fL = np.array([ float(i) for i in sL ])
        self.store_latvec(fL)
        #print "latvec:"+TOR.vec2str(fL)
        
    def store_latvec(self, fL):
        self.latveclock.acquire()
        self.last_latvec[:] = fL
        self.latveclock.release()
       
        
    def get_latvec(self):
        self.latveclock.acquire()
        x = self.last_latvec.copy()
        self.latveclock.release()
        return x
           
class rosTOR:

  
    def __init__(self):
        self._setup()
    
    def _setup(self):
        # Create a publishers
        self.R_pub = [None, None, None, None, None, None]
        for k in range(0,6):
            if k==TOR._RGRIP or k==TOR._LGRIP: # these nee different message type (TODO)
                print "[*] TOR._setup: creating publisher [grippers] ",TOR.TOPIC_NAME_L[k], "[",k,"]..."
                self.R_pub[k] = rospy.Publisher(TOR.TOPIC_NAME_L[k], GripperCommand, queue_size=1)
            else:
                print "[*] TOR._setup: creating publisher ",TOR.TOPIC_NAME_L[k], "[",k,"]..."
                self.R_pub[k] = rospy.Publisher(TOR.TOPIC_NAME_L[k], JointTrajectory, queue_size=1)
                # Wait until the publisher k gets ready.

            c=0;
            while self.R_pub[k].get_num_connections() == 0 :
                c +=1
                if c%10==0: print('TOR._setup: publisher %d still not ready....'%k);
                rospy.sleep(0.2)
                #print('TOR: publisher %d OK!'%k);



        #This moves the robot at start
        #self.goto(TOR.q_ezhome)
    
        
##    def goto_part(self, q, dt, publisher, JOINT_NAMES):
##        self.publish_joint_trajectory(
##            publisher = publisher,
##                joint_names = JOINT_NAMES,
##                positions = q,
##                time_from_start = dt
##        )
##        rospy.sleep(dt*1.1)
##
##    def goto(self, q_L):
##        for k in range(0,len(q_L)):
##            if k==TOR._RGRIP or k==TOR._LGRIP: continue
##            self.publish_joint_trajectory(
##                publisher = self.R_pub[k],
##                joint_names = TOR.JOINT_NAMES_L[k],
##                positions = q_L[k],
##                time_from_start = 1
##        )
##        rospy.sleep(1.1)
        
    def publish_joint_trajectory(self, publisher, joint_names, positions, time_from_start):
        """
        Function for publishing message to move the arm
    
        Parameters
        ----------
        publisher : rospy.Publisher
            publisher
        joint_names : list
            list of joint names
        positions : list
            list of joint's goal positions(radian)
        time_from_start : float
            transition time from start
    
        Returns
        -------
        None
    
        Throws
        ------
        None
        """
    
        # Creates a message.
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0 for i in range(len(joint_names))]
        point.accelerations = [0.0 for i in range(len(joint_names))]
        point.effort = [0.0 for i in range(len(joint_names))]
        point.time_from_start = rospy.Duration(time_from_start)
        trajectory.points.append(point)
    
        # Publish the message.
        publisher.publish(trajectory)        


    def gripper_command(self, publisher, position, max_effort):
        """
        Function for publishing message to move the gripper
            Note that position's unit is "meter" and max_effort's unit is "N".

        Parameters
        ----------
        publisher : rospy.Publisher
            publisher
        position : float
            position of finger.
        max_effort : float
            max effort for grasp

        Returns
        -------
        None

        Throws
        ------
        None
        """

        # Creates a message.
        command = GripperCommand()
        command.position = position
        command.max_effort = max_effort

        # Publish the message.
        publisher.publish(command)
    
    @staticmethod
    def _clone_jointlist(q_L, scale=1):
    #c_L = [None, None, None, None, None, None]
        #print "length q_L:", len(q_L), "    ", q_L
        c_L = [None]*len(q_L)
    
        for k in range(0,len(q_L)):
            c_L[k] = scale*q_L[k].copy()
        #print "length c_L:", len(c_L), "    ", c_L
        return c_L


    
# end of Class rosTOR

"""
JointComm is mainly for updating the joint variables by listening to appropriate
ROS/TOROBO topics. It has two user calls readgpos() and getjointpos(). The former
fills in a given p_L, the other returns a single limb list or the whole list of lists.
"""
class JointComm():
    def __init__(self):
        self.glock = Lock();
        self.gpos_L = rosTOR._clone_jointlist(TOR.q_allzero)
        #print "JointComm init:  ezhome:",TOR.q_ezhome
        #print "JointComm init:  gpos_L:",self.gpos_L
        self.gposvalid = 0
        self.listenerTh = Thread(target=self.jointStateListen, args=(1,))
        self.listenerTh.start()

    def readgpos(self, p_L):
        """ 
        Fills in p from the most recent joint update. If succesfull returns true;
        otherwise returns false.
        """
        valid = False
        self.glock.acquire()
        if self.gposvalid == TOR._BODY_ORED:
            for k in range(0, len(p_L)):
                #print "readgpos ", k,"  ",p_L[k]
                if len(p_L[k])==0: continue   # check this if it works ERH
                p  = p_L[k]
                p[:] = self.gpos_L[k][:]
                valid = True
        self.glock.release()
        return valid

    def getjointpos(self, k):
        """ 
        Returns  the most uptodate [joint_angles_LIST,True] if succesfull,  otherwise returns [[],False].
        """        
        q_L = rosTOR._clone_jointlist(TOR.q_ezhome)
        succ = self.readgpos(q_L)
        if succ:
            if k==TOR._ALL:
                return q_L, True
            else:
                return q_L[k]
        else:
            return [], False

    def _setgpos(self, p_L):
        self.glock.acquire()
        for k in range(0, len(p_L)):
            print "setgpos ", k,"  ",p_L[k]
            if p_L[k]==None: continue
            p  = p_L[k]
            gp = self.gpos_L[k]
            gp[:] = p[:]
        self.gposvalid = TOR._BODY_ORED
        self.glock.release()

    def _setgpos_single(self, p, k):
        self.glock.acquire()
        self.gpos_L[k][:] = p[:]
        self.gposvalid |= (1<<k)      # set specific part valid 
        self.glock.release()

    def callback_jointStateListen_RGRIP(self, data):        ## ERH x6?
        pos = data.position
        self._setgpos_single(pos, TOR._RGRIP)

    def callback_jointStateListen_LGRIP(self, data):        ## ERH x6?
        pos = data.position
        self._setgpos_single(pos, TOR._LGRIP)
        
    def callback_jointStateListen_RARM(self, data):        ## ERH x6?
        pos = data.position
        self._setgpos_single(pos, TOR._RARM)
        #print("jointstatelisten_RARM: I got : pos",pos)
        ##rospy.loginfo(rospy.get_caller_id() + "I heard pos %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f", pos[0],pos[1],pos[2],pos[3],pos[4],pos[5], pos[6])

    def callback_jointStateListen_LARM(self, data):        ## ERH x6?
        pos = data.position
        self._setgpos_single(pos, TOR._LARM)

    def callback_jointStateListen_TORSO(self, data):        ## ERH x6?
        pos = data.position
        self._setgpos_single(pos, TOR._TORSO)

    def callback_jointStateListen_HEAD(self, data):        ## ERH x6?
        pos = data.position
        self._setgpos_single(pos, TOR._HEAD)

    
     # ERH_CHECK: Does each of the subscribers have to have s seperate tread? Currently jointStateListen is started as a single thread.   
    def jointStateListen(self, name_id):
        print "example subscribed to:", TOR.STATE_TOPIC_NAME_L[TOR._RARM]
        rospy.Subscriber(TOR.STATE_TOPIC_NAME_L[TOR._RARM], ToroboJointState, self.callback_jointStateListen_RARM)
        rospy.Subscriber(TOR.STATE_TOPIC_NAME_L[TOR._LARM], ToroboJointState, self.callback_jointStateListen_LARM)
        rospy.Subscriber(TOR.STATE_TOPIC_NAME_L[TOR._TORSO], ToroboJointState, self.callback_jointStateListen_TORSO)
        rospy.Subscriber(TOR.STATE_TOPIC_NAME_L[TOR._HEAD],  ToroboJointState, self.callback_jointStateListen_HEAD)
        rospy.Subscriber(TOR.STATE_TOPIC_NAME_L[TOR._RGRIP], ToroboJointState, self.callback_jointStateListen_RGRIP)
        rospy.Subscriber(TOR.STATE_TOPIC_NAME_L[TOR._LGRIP], ToroboJointState, self.callback_jointStateListen_LGRIP)
        print('[*] JointComm.jointStateListen: subscribed to joint state topics of ROS/TOROBO)')   ## ERH x6
        #print '*****************'
        #print 'xxxxxxxxxxxxxxxxx'
        #rospy.spin()      # works without spin, why?

class Gazer():
    DEF_GAZE_HAND = TOR._RARM
    def __init__(self, ucomm):
        self.lookatTh = Thread(target=self.lookat_thread_func, args=(1,))
        self.ucomm = ucomm
        self.gazeEnabled = False
        self.lookatHand = None #TOR._LARM
        self.lookatBlob = None            # not implemented, not used
        self.lookatPoint = np.array([1,0,0.9])
        
        self.gazeMode = 'point'  # 'hand', 'blob'
        self.gazePar = self.lookatPoint 
        
        self.thc = 0
        self.quitgazer = False

        self.commandPERIOD = 0.1  #   head will be commanded with this period
        self.ucomm.register_gazer(self)
        self.lookatTh.start()
        
    def quit(self):
        self.quitgazer = True
        

    
    def set_gazemode(self, mode, par=None):
        if mode == "point":
            self.gazeMode ='point'
            if par is not None: 
                self.lookatPoint = self.gazePar = par   # np.array x,y,z
        elif mode == "blob":
            if par is not None: 
                self.lookatBlob = self.gazePar = par    # interger
            self.gazeMode ='blob'
        elif mode == "hand":
            if par is not None: 
                self.lookatHand = self.gazePar = par    # integer  TOR._LARM or _RARM
                print "Hand ix:", par
            self.gazeMode = 'hand'
        else: 
            print "ERROR:"+mode+" is not a valid gaze mode. Use point, blob or hand!"


    def disablegaze(self):
        self.gazeEnabled = False
        print "Stopped gazing."
        
        
    def enablegaze(self, hand=None,blob=None,point=None):
        self.gazeEnabled = True
        print "Enabled gazing. Mode=",self.gazeMode, "par:", self.gazePar
  
       
    def toggle_gaze(self):
        if self.gazeEnabled == True:
            print "Stopped gazing."
            self.gazeEnabled = False
        else:
            print "Enabled gazing. Mode=",self.gazeMode, "par:", self.gazePar
            self.gazeEnabled = True
            
             
    # I think not using Locks should be OK. https://superfastpython.com/thread-atomic-operations/
    def lookat_thread_func(self,mess):
        rospy.sleep(2)   # kick in 2 seconds later
        print "Look at threa function started!"
        while (not self.quitgazer):
            rospy.sleep(self.commandPERIOD)
            self.thc += 1
            #print self.thc," > gazeEnabled:", self.gazeEnabled
            if self.gazeEnabled == False: continue
            
            if self.gazeMode=='hand':
                tix = self.lookatHand
                if tix is not None:
                    hand, hadnori, ign = self.ucomm.request_pose_q(tix)
                    hand[2] += -0.12    # look 20cm below the hand
                    self.ucomm.lookat(hand)
                    #print self.thc, "> -------> Following tix:",tix," == ", TOR._TARGET_L[tix], "  mess:",mess
            elif self.gazeMode=='point':
                p = self.lookatPoint
                #print "looking at p:",p
                self.ucomm.lookat(p)
            elif self.gazeMode=='blob':
                dummy= 1 # not implemented yet
            else: print "ERROR 1978: This should not happen!"
            
            
    def get_pose_q(self, tix):
        #print "<> requested pose for body group tix:",TOR._TARGET_L[tix]
        if tix == TOR._LARM or tix==TOR._RARM:
            q, q9, q10 = self.request_q(tix)
            p, R = self.kin.forwardkin(tix, q9)
        if tix == TOR._HEAD:
            qtorso = self.jcomm.getjointpos(TOR._TORSO)
            q = self.jcomm.getjointpos(tix)
            q4 = np.hstack([qtorso,q])
            p, R = self.kin.forwardkin(tix, q4)
            q10 = None
            
        return p, R, q10
    
# Send commands from shell using
# rostopic pub -1  /ezcommand std_msgs/String "getjpos 4 5 6.6 7.7 2.3"    
class UserComm():
    TOPIC_NAME = "ezcommand"
    outTOPIC_NAME = "ezcommand_out"

    def __init__(self, jcomm, ezcon, kin, blob, latvec, lup):
        print 'debug -init'
        self.listenerTh = Thread(target=self.userInputListen, args=(1,))
        self.jcomm = jcomm
        self.listenerTh.start()
        self.ezcon = ezcon
        self.kin = kin
        self.ar_tag_detect=ar_tag_detect()
        self.blob = blob
        self.lup  = lup
        kin._register_ucom(self)
        self.ezcomout = rospy.Publisher(UserComm.outTOPIC_NAME, String, queue_size=10 )
        self.syshome_NEIG_deg = 5      # if too close to the syshome posture, home command creates self-collision. See 'home'
        self.canvas = None   # see viscon_on/off
        self.gazer = None    # needs to be registered after creation of UserComm
        self.ccnmp = None    # later make this a proper learning class
        self.latvec = latvec
        self.udpresponseServer = None
        self.udpClientSocket   = None
        self.udpresponseServer_open()   # sets up the above two
        self.UDPServerSocket = None
        self.udpserverTh = Thread(target=self.udpserverListen, args=(1,))
        self.udpserverTh.start()        # sets up the udpserver listener and fires the thread
        self.deniz_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=3)

        # Camera subscriber for capturing images
        self.cvbridge = CvBridge()
        # Fixed overhead camera
        self.current_image = None
        self.image_lock = Lock()
        # Use CAM_TOPIC_HW = '/torobo/head/camera/color/image_raw' for hardware
        # Use '/head/camera/color/image_raw' for Torobo eye camera in simulation
        CAM_TOPIC_SIM = '/fixed_camera/image_raw'  # Fixed overhead camera in Gazebo
        rospy.Subscriber(CAM_TOPIC_SIM, Image, self.image_callback, queue_size=1)
        # Side camera
        self.current_image_side = None
        self.image_lock_side = Lock()
        CAM_TOPIC_SIDE = '/side_camera/image_raw'  # Fixed side camera in Gazebo
        rospy.Subscriber(CAM_TOPIC_SIDE, Image, self.image_callback_side, queue_size=1)

        if touchEXISTS:
            self.touch = touchComm(self, activate=True)
        else:
            self.touch = None
        #rospy.init_node('node2pub');

    def image_callback(self, data):
        """Callback for camera image subscriber - stores the latest frame."""
        try:
            img = self.cvbridge.imgmsg_to_cv2(data, "bgr8")
            self.image_lock.acquire()
            self.current_image = img.copy()
            self.image_lock.release()
        except CvBridgeError as e:
            print("CvBridge Error: ", e)

    def get_current_image(self):
        """Thread-safe method to get the current camera image."""
        self.image_lock.acquire()
        img = self.current_image.copy() if self.current_image is not None else None
        self.image_lock.release()
        return img

    def image_callback_side(self, data):
        """Callback for side camera image subscriber - stores the latest frame."""
        try:
            img = self.cvbridge.imgmsg_to_cv2(data, "bgr8")
            self.image_lock_side.acquire()
            self.current_image_side = img.copy()
            self.image_lock_side.release()
        except CvBridgeError as e:
            print("CvBridge Error (side cam): ", e)

    def get_current_image_side(self):
        """Thread-safe method to get the current side camera image."""
        self.image_lock_side.acquire()
        img = self.current_image_side.copy() if self.current_image_side is not None else None
        self.image_lock_side.release()
        return img

    # Used by UDPserver or ROS node callback
    def parse_userInput(self, s, isfromUDP=0):
       
        print "Received:",s
        
        L = s.split(" ")
        command = L[0];
        if self._is_no_target_command(command):
            target = None
            argix  = 1
            #print "Command %s is no target command"%command
        else:
            if len(L)<2:
                self.pprint("ERROR: %s needs a body-part specification such as torso, larm, rarm, head etc."%command)
                return
            target = L[1];  # which body part is asked
            argix  = 2
        #print "----argix:",argix,"====",L
        args_s = L[argix:]
        args_f = np.ones(len(args_s))*np.nan
        args_i = np.ones(len(args_s))*np.nan
        for k in range(0, len(args_s)):
            try:
                args_f[k] = float(args_s[k])
                args_i[k] = int(args_f[k])
            except:
                if command!="play":
                    print "userInputListen ignoring nonstring to number conversion errors."

        #print("UserComm: command:",L)
        #print("args as string :",args_s)
        #print("args as integer:",TOR.vec2str(args_i))
        #print("args as float  :",TOR.vec2str(args_f))
                
        self.processCommand(command, target,args_s, np.array(args_i), np.array(args_f))
        
    def callback_userInputListen(self, mess):
        strdata = mess.data
        self.parse_userInput(strdata)
  
    def register_gazer(self,gazer):
        self.gazer = gazer
        
    """
    return if no target is required for a command
    """
    def _is_no_target_command(self, com):
        if com == "syshome" or com =="home" or com=="allzero" or com=="help" or com=="getblob":
            return 1
        if com=="touchcon_on" or com=="touchcon_off" or com=="quit" or com=="play" or com=="getrope":
            return 1
        if com=="lookat" or com == "gaze" or com=='touchmode' or com == 'makeknot':
            return 1
        return 0

    """
    Waits for user commands send to the topic TOPIC_NAME "ezcommand"
    """
    def userInputListen(self, name_id):
        #print('******* UserComm.listerner entered *********')
        rospy.Subscriber(UserComm.TOPIC_NAME, String, self.callback_userInputListen)
        print('[*] UserComm.userInputListen: subscribed to the ezcommand topic, send your commands there..')
        #rospy.spin()   # this will block (main spins so no need for individual threads)

    """
    Send the input string to topic outTOPIC_NAME (ezcommand_out)
    """
    def responseOutput(self, outstr):
        msg = String(data=outstr)
        self.ezcomout.publish(msg)

    def udpResponse(self, outstr):
        encmess = outstr.encode()        
        sentBytesCount    = self.udpClientSocket.sendto(encmess, self.udpresponseServer)
        #print '>',sentBytesCount,'bytes are/is just sent!'

    """
    print to console and call responseOutput()
    """
    def pprint(self, outstr):
        print(outstr)
        self.responseOutput(outstr)
        self.udpResponse(outstr)


    def closeUDP(self):
        if self.UDPServerSocket is not None:
            self.UDPServerSocket.close()
            print "UDP server closed"
        if self.UDPServerSocket is not None:
            self.UDPServerSocket.close()
            print "UDP response client closed"
    def udpserverListen(self, name_id):
        print('[+] UserComm.udpListen: setting up the server')

        localIP     = TOR.udpserverIP
        localPort   = TOR.udpserverPORT 
        bufferSize  = TOR.udpserverbufferSize 
        msgFromServer       = "Hello UDP Client"
        bytesToSend         = str.encode(msgFromServer)      
        # Create a datagram socket
        self.UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDPServerSocket.settimeout(None)
        # Bind to address and ip
        self.UDPServerSocket.bind((localIP, localPort))
        print "[+] UserComm.udpListen: UDP server up and listening"
        
        # Listen for incoming datagrams
        while(True):
            bytesAddressPair = self.UDPServerSocket.recvfrom(bufferSize)
            message = bytesAddressPair[0].decode()
            address = bytesAddressPair[1]
            if message=="server_selfkill":
                print "server_selfkill requested"
                break
            
            print 'Message:[',message,']', 'from [', address,']'
            self.parse_userInput(message, isfromUDP=1)
            # May send back an answer to the sender client
            #UDPServerSocket.sendto(bytesToSend, address)
        #--- Server loop ended. Close the server
        UDPServerSocket.close()
        print "Exited while loop, and the server should be now closed."

    def udpresponseServer_open(self):
       localIP      = TOR.udpserverIP
       localPort    = TOR.udpserverPORT 
       targetIP     = TOR.udpresponseServerIP   
       targetPort   = TOR.udpresponseServerPORT
       numBytes2Get = TOR.udpserverbufferSize
       # Create a UDP socket. UDP is datagram based.
       self.udpresponseServer   = (targetIP, targetPort)
       self.udpClientSocket     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       print "[+] udpResponse socket is opened: ", self.udpClientSocket   
       self.udpResponse("Hello from the client")



    

#-------
    def processCommand(self, com, target, args_s, args_i, args_f):
        try:
            tix = -1
            if target != None:
                if com == "pickupfree": tix = self.blob.resolve_blob(target)
                else:                   tix = TOR._TARGET_L.index(target)
        except ValueError:
            self.pprint("ERROR: Use a valid body part. i.e use: <Command> <all | torso | rarm | larm | lgrip | rgrip> <arglist...>")
            return

        if (com == 'getq'):
            self.pprint("Joints(rad): %s"%TOR.vec2str(self.jcomm.getjointpos(tix), "%2.4f"))
        elif (com == 'getq_deg'):
            self.pprint("Joints(deg): %s"%TOR.vec2str((180/np.pi)*self.jcomm.getjointpos(tix)))
        elif (com == 'getjnt_deg'):
            kk = args_i[0];
            self.pprint('Joint_%d(deg) : %f' % (kk, (180/np.pi)*self.jcomm.getjointpos(tix)[kk])   )
        elif (com == 'getjnt'):
            #kk = args_i[0];
            jp = self.jcomm.getjointpos(tix)
            print '---------->',jp
            if tix==TOR._LGRIP or tix == TOR._RGRIP:
                 self.pprint('Gripper aperture: %f mm' %(1000*self.jcomm.getjointpos(tix)[0])   )
            else:
                self.pprint('Joint_%d(rad) : %f' % (kk, self.jcomm.getjointpos(tix)[kk])   )
        elif (com == 'getdesq'):
            print('ezcon.qdes_L:', self.ezcon.qdes_L)
        elif (com == 'getpose'):
            p, R, q_rad = self.request_pose_q(tix)
            if len(p) > 0:
                self.pprint("\n Position: %s"%TOR.vec2str(p, "%+2.4f"))
                self.pprint(' Xaxis   : %s'%TOR.vec2str(R[0:3,0], "%+2.4f"))
                self.pprint(' Yaxis   : %s'%TOR.vec2str(R[0:3,1], "%+2.4f"))
                self.pprint(' Zaxis   : %s'%TOR.vec2str(R[0:3,2], "%+2.4f"))
        # Single joint_deg
        elif (com == 'setjnt_deg' or com == 'setjnt' ):
            if len(args_s)!=2:
                self.pprint('ERROR: Wrong number of arguments. Use: setjnt_deg <jount_group> <joint_ix> <value>')
                return
            if tix==TOR._ALL:
                self.pprint('ERROR: <all> is not supported as joint group for setjnt_deg  command')
                return
            kk = args_i[0];
            if kk<0 or kk>=TOR._NUMJNT_L[tix]:
                self.pprint('ERROR: joint group [%s] has no such joint index (%d)!'%(TOR._TARGET_L[tix],kk))
                return
            val_ = args_f[1];
            angles = self.jcomm.getjointpos(tix)
            if  com == 'setjnt_deg':
                angles[kk] = val_*np.pi/180
            else:
                angles[kk] = val_    
            # set the PD targets
            print('Setting joint ',kk,' to ', val_, '[',TOR._TARGET_L[tix],']')
            self.request_setdesq(tix, angles)   #make safety checks!
        # Full joint setting
        elif com == 'setqT_deg':
            # set the PD targets
            if len(args_s)<3:
                self.pprint('ERROR: Wrong number of arguments. Use: setqT_deg <joint_group> torsoj1 torsoj2 j1 j2 ...')
                return
            self.request_setdesqT(tix, angles*np.pi/180)

        elif com == 'setqT':
            # set the PD targets
            if len(args_s)<3:
                self.pprint('ERROR: Wrong number of arguments. Use: setqT_rad <joint_group> torsoj1 torsoj2 j1 j2 ...')
                return
            self.request_setdesqT(tix, angles)

        elif com == 'setq_deg':
            # set the PD targets
            self.request_setdesq(tix, args_f*np.pi/180)   #make safety checks!
        elif com == 'setq':
            # set the PD targets
            self.request_setdesq(tix, args_f)   #make safety checks!
            # self.wait_until_reach(tix, args_f, SEC=1)
        elif com == 'setpose':
            self.request_setpose(tix, args_f, live = 1, IK_qinit_current=False)
        elif com == 'setpose_deniz':
            self.request_setpose_deniz(tix, args_f, live = 1, IK_qinit_current=False)
        elif com == 'first_push':
            self.request_first_push(tix, args_f, live = 1, IK_qinit_current=False)
        elif com == 'second_push':
            self.request_second_push(tix, args_f, live = 1, IK_qinit_current=False)
        elif com == 'correction_tuba':
            self.request_correction(tix, args_f, live = 1, IK_qinit_current=False)
        elif com == 'setvialoc':
            self.request_3spline(tix, args_f) 
        elif com == 'setloc':
            self.request_setpose(tix, args_f, mode = TOR.ik_mode_POS)       
        elif com == 'setori':
            self.request_setpose(tix, args_f, mode = TOR.ik_mode_ORI )
        elif com == 'setposeL':
            self.request_setpose(tix, args_f, live = 1, IK_qinit_current=True)
        elif com == 'tgrasp' or com == 'treach':
            self.execute_reachgrasp(tix, args_f, live = 0, IK_qinit_current=True, grasp=(com=='tgrasp'))
        elif com == 'pickupfree':
            self.execute_pickupfree(tix, args_f, live = 0, IK_qinit_current=True)
        elif com == 'pickup_blob':
            self.execute_pickup_blob(tix, args_f, live = 0, IK_qinit_current=True)
        elif com == 'pickup_rope':
            self.execute_pickup_rope(tix, args_f, live = 0, IK_qinit_current=True)
        elif com == 'setDpose':
            self.request_setDpose(tix, args_f)
        elif com == 'setDloc':
            self.request_setDpose(tix, args_f, mode = TOR.ik_mode_POS)
        elif com == 'setlocL':
            self.request_setpose(tix, args_f, mode = TOR.ik_mode_POS, live=1)       
        elif com == 'setoriL':
            self.request_setpose(tix, args_f, mode = TOR.ik_mode_ORI, live=1)
        elif com == 'lookat':
            self.lookat(args_f)
        elif com == 'gaze':
            self.set_gaze(args_s)
        elif com =='linmot':
            self.linear_motion(tix, args_f)
        elif com =='measure':
            self.measure()
        elif com =='deniz':
            self.data_collector()

        elif com=='calibrate':
            self.calibrate_ar_tag()
        elif com =='deniz2':
            self.print_data()
        elif com=='reach_ar':
            self.reach_ar_tag(args_f)
        elif com=='get_ar':
            self.get_tag_pos(args_i[0])
        elif com =='move_point':
            self.move_point(args_s[0],args_f[1],args_f[2],args_f[3])
        elif com == 'allzero':
            self.request_setdesq(TOR._ALL, TOR.q_allzero)             
        elif com == 'home':
            q,q9,q10 = self.request_q(TOR._LARM)
            Ldist = norm(TOR.q_syshome[TOR._LARM] - q)
            q,q9,q10 = self.request_q(TOR._RARM)
            Rdist = norm(TOR.q_syshome[TOR._RARM] - q)
            if Rdist<self.syshome_NEIG_deg*np.pi/180.0 or Ldist<self.syshome_NEIG_deg*np.pi/180.0:   # if you are close to syshome posture
                self.request_setdesq(TOR._ALL, TOR.q_prehome)
                rospy.sleep(2.0)
            self.request_setdesq(TOR._ALL, TOR.q_ezhome)
        elif com == 'getblob':
            [cen,info] = self.blob.get_centers()
            #print cen,info
            for k in range(0,len(info)):
                print "Blob at ["+TOR.vec2str(cen[k],"%+4.3f")+"]  "+info[k]
                
        elif com == 'getrope':
            data, k = self.blob.get_special_blob_data()
            #print cen,info

            print "-->  rope pos:"+TOR.vec2str(data)
            print "---> data time: now - ("+str(k)+")"
            if k<0:
                print "no data in the buffer!"    
            else:
                p, R, q_rad = self.request_pose_q(TOR._LARM)
                print "Context:",TOR.vec2str(data - p,"%+2.5f")
                
        elif com == 'syshome':
            self.request_setdesq(TOR._ALL, TOR.q_syshome)   #make safety checks!
        elif com == 'iktab':
            self.kin.make_iktab(self,tix,args_s, args_f)
        elif com == 'viscon_on':
            if self.canvas is None:
                self.canvas = GameRunner(self)              # An attempt for teleoperation
            self.canvas.set_event_response(True)
        elif com == 'viscon_off':
            #self.canvas.set_event_response(False)
            if self.canvas is not None: self.canvas.closepygame()
            self.canvas = None
        elif com == 'touchcon_on':
            if self.touch is None:
                self.touch = touchComm(self, activate=True)              # Enable teleoperation
                self.pprint("touch control is turned on and activated")
            else:
                self.touch.set_event_response(True)
                self.pprint("touch control is reactivated")
        elif com == 'touchcon_off':
            if self.touch is None:
                self.touch.set_event_response(False)
                self.pprint("touch control is deactivated")
            else: self.pprint("touch control has not been started.")

        elif com == 'play':
            self.play_traj(args_s)

        elif com == 'playcart':
            self.playcart_traj(tix, args_s)

        elif com == 'deniz3':
            self.data_collector2()
        
        elif com == 'data_table':
            self.data_collector_table()
        elif com == 'data_triangle':
            self.data_collector_triangle()

            
        elif com == 'touchmode':
            print args_f
            if len(args_f)>0:
                self.pprint(self.touch.setcontrol_mode(int(args_f[0])))
            self.pprint("Now touch mode:"+self.touch.setcontrol_mode())

        elif com == 'correct':
            self.cnmp_correct(tix, args_f)

        elif com == 'lookcorr':
            self.play_lookupcorrection(tix, args_f)
        elif com == "makeknot":
             self.makeknot(args_s)
        elif com == "makeknot_file":
             self.makeknot_file(args_s)
        elif com == "butter" or com=="butterfly":
            self.makebutterfly(tix, args_f)
        elif com == 'help':
            self.pprint('GENERAL FORMAT:')
            self.pprint('   <Command> <all | torso | rarm | larm | lgrip | rgrip> <arglist...>')
            self.pprint('COMMAND LIST:')
            self.pprint('   getq: prints current joint angles in RADIANS')
            self.pprint('   getq_deg: prints current joint angles in DEGREES')
            self.pprint('   getpose jntgrp : prints current pose of the jntgroup')
            self.pprint('   setq_deg jntgrp j1 j2..jn : set the desired joint angles in DEGREES')
            self.pprint('   setq jnrgrp j1 j2..jn : set the desired joint angles in RADIANS')
            self.pprint('   setqT_deg jntgrp torsoj1 torsoj2 j1 j2..jn : set the desired joint angles in DEGREES')
            self.pprint('   setqT jnrgrp torsoj1 torsoj2 j1 j2..jn : set the desired joint angles in RADIANS')            
            self.pprint('   setjnt_deg jntgrp K val : set joint K to val DEGREES')
            self.pprint('   setjnt jntgrp K val : set joint K to val DEGREES')     
            self.pprint('   setpose jntgrp px py pz  Zaxis_x Zaxis_y Zaxis_z Yaxis_x Yaxis_y Yaxis_z [max_iter]')
            self.pprint('   setloc jntgrp px py pz [max_iter] ')
            self.pprint('   setori jntgrp  Zaxis_x Zaxis_y Zaxis_z Yaxis_x Yaxis_y Yaxis_z  [max_iter] ')
            self.pprint('   setposeL, setposL, setoriL : does the same but use current joint angles to seed IK')
            self.pprint('   tgrasp jntgrp px py pz  Zaxis_x Zaxis_y Zaxis_z Yaxis_x Yaxis_y Yaxis_z: perform a table grasp and come back')
            self.pprint('   treach jntgrp px py pz  Zaxis_x Zaxis_y Zaxis_z Yaxis_x Yaxis_y Yaxis_z: perform a table reach and come back')
            self.pprint('   getblob: show blob coordinates')
            self.pprint('   play filename [speed] [stopcom]:  plays filename with speed scale speed. stopcom indicates an early stop (see code)' )
            self.pprint('   touchmode [mode] : mode=1  control only position, mode=2 contols only orientation  mode=3 for both. No argument: prints mode')
            self.pprint('   pickup_blob [larm|rarm]  blob_index [beta_deg]  (use getblob to see the blob info)')
            self.pprint('   pickup_rope [larm|rarm]  beta_deg (closest rope end will be picked. Coloring is hardcoded!))')
            #self.pprint('   viscon_off: disable gui/based realtime IK')
            #self.pprint('   circle jntgrp px py pz  Zaxis_x Zaxis_y Zaxis_z Yaxis_x Yaxis_y Yaxis_z radius [max_iter]')
            self.pprint('   setvialoc p1 p2 p3 [ikiteration [sampled point count [ execution speed multiplier]]] : arms goes through p1, p2, p3 smoothly.' )
            self.pprint('   home: set the desired joint angles to ezcontrol home position')
            self.pprint('   syshome: set the desired joint angles to Torobo defined home position')      

        elif com == 'quit':
            print ('Trying to quit..')   # TODO: Doesn't quit. Find a better way
            quit()
            
        else:
            print("ezcommand: Unkown command received!  => ",com) 

    # # calls with TORSO is handled separetely
    def calibrate_ar_tag(self):
        sampled_list = self.spline_fit_viapoints_linear(np.asarray(self.ar_tag_detect.rob_arm_position_list), 50)
        self.request_setdesqT(1,sampled_list[0:9, 0])
        rospy.sleep(10)

        for i in range(1,sampled_list.shape[1]):
            
            self.request_setdesqT(1,sampled_list[0:9, i])
            rospy.sleep(1)
            print "Calibration point reached"

            print self.ar_tag_detect.get_position_by_id(6)

            self.ar_tag_detect.real_ar_tag_coor.append(self.ar_tag_detect.get_position_by_id(6))

            p, R, q_rad = self.request_pose_q(1)

            self.ar_tag_detect.real_rob_coor.append(p)

            print "Calibration point recorded"

        print "Calibration finished"
        self.ar_tag_detect.calibrated = True
        self.ar_tag_detect.update_homogenous()
        print "Homogenous matrix updated"
        print(self.ar_tag_detect.real_rob_coor)
        print(self.ar_tag_detect.real_ar_tag_coor)

    # def reach_tag(self, id):
    #     id=int(id)
    #     ar_pos_byid=self.ar_tag_detect.get_position_by_id(id)
    #     rob_pos_byid=(self.ar_tag_detect.R.apply(self.ar_tag_detect.marker_pos_in_external_frame[id]-self.ar_tag_detect.ar_mid.reshape(-1, 3)))[0]+self.ar_tag_detect.rob_mid.reshape(-1, 3)
    #     print self.ar_tag_detect.get_position_by_id(id)
    #     print (self.ar_tag_detect.R.apply(self.ar_tag_detect.marker_pos_in_external_frame[id]-self.ar_tag_detect.ar_mid.reshape(-1, 3)))[0]+self.ar_tag_detect.rob_mid.reshape(-1, 3)
    #     des_z = np.array([0., 0., -1.])
    #     des_y = np.array([0., 1., 0.])
    #     des_x = np.cross(des_y,des_z)
    #     R_des = np.array([des_x, des_y, des_z]).T
    #     q, solved, errL,itused = self.kin.ik(1, np.array(rob_pos_byid), R_des,  maxit=400, IKstep_show=0)
        
    #     self.request_setdesq(1, np.asarray(q))
        # rospy.sleep(5)

    def get_tag_pos(self,id):
        id=int(id)
        ar_pos_id=self.ar_tag_detect.get_position_by_id(id)
        robot_pos_id=self.ar_tag_detect.calculate_in_rob(id)
        print(ar_pos_id)
        print(robot_pos_id)
    
    def request_setpose_deniz(self, tix, args, mode=0, live=0, IK_qinit_current=False):

        p_des = np.array(args[0:3])
        p_des_sec=np.array(args[0:3])
        des_z = np.array(args[3:6])
        des_y = np.array(args[6:9])
        newargs = np.hstack([p_des, des_z, des_y])
        p_des_sec[2]+=0.1
        newargs_second = np.hstack([p_des_sec, des_z, des_y])
        self.execute_reachgrasp_deniz(1,newargs,newargs_second, 0,False, grasp=True)


    def reach_ar_tag(self, argsf):
            id=int(argsf[0])
            if len(argsf)==1:
                beta = 0*np.pi/180.
            else:
                beta = float(argsf[1])*np.pi/180.
            q_curr, q9, q10 = self.request_q(1)
            robot_pos_id=list(self.ar_tag_detect.calculate_in_rob(id))
            robot_pos_id[0]-=0.02#this is a hack i dont know why there is an offset like this
            robot_pos_id_sec=list(self.ar_tag_detect.calculate_in_rob(id))
            robot_pos_id_sec[0]-=0.02
            if robot_pos_id is None:
                print "No such tag detected"
                return
            if robot_pos_id[0]>2.5:
                print "Tag is too far away"
                return
            if robot_pos_id[0]<0.4:
                print "Tag is too close"
                return
            if robot_pos_id[1]>1.5:
                print "Tag is too high"
                return
            if robot_pos_id[2]>2.5:
                print "Tag is too far away"
                return
            if robot_pos_id[2]<0.4: 
                print "Tag is too close"

            
            pickZAXIS = np.array([0., 0., -1.])
            pickYAXIS = np.array([0., 1.,  0.])
            pickYAXIS /= norm(pickYAXIS)
            pickZAXIS /= norm(pickZAXIS)
            pickXAXIS = np.cross(pickYAXIS,pickZAXIS)
            R = np.vstack([pickXAXIS,pickYAXIS,pickZAXIS]).transpose()
            Rlocyrot = TOR.rotY(beta)
            Rnew = np.matmul(R,Rlocyrot)

            robot_pos_id_sec[1]=robot_pos_id_sec[1]-0.05
            robot_pos_id_sec[2]=robot_pos_id_sec[2]+0.1

            newargs = np.hstack([robot_pos_id,  Rnew[2,:], Rnew[1,:]])
            newargs_sec = np.hstack([robot_pos_id_sec,  Rnew[2,:], Rnew[1,:]])
            self.execute_reachgrasp_deniz(1,newargs, newargs_sec, 0,False, grasp=True)




    def request_ezdesq(self, tix, args):
        print "set_gaze args:",s
        if len(s)==0:
            self.gazer.toggle_gaze()
            print "Now gaze state:",self.gazer.gazeEnabled
            return
        if len(s)==3:    # it is a point
            x = float(s[0]); y = float(s[1]); z = float(s[2])
            self.gazer.set_gazemode('point', np.array([x,y,z]))
            self.gazer.enablegaze()
            return
        if len(s)==1:   # hand_ix
            print "pars:",s
            #if s[0]!='larm' and s[0]!='rarm': print "Invalid gaze parameter! use larm or rarm"; return
            hand_ix = TOR._LARM if s[0]=='larm' else TOR._RARM
            self.gazer.set_gazemode('hand', hand_ix)
            self.gazer.enablegaze()
            return
        if len(s)==2:  #  blob blobix
            if s[0]!='blob': print "Invalid gaze parameter! use blob blob_ix"; return
            blob_ix = int(s[1])
            self.gazer.set_gazemode('blob', blob_ix)
            self.gazer.enablegaze()
            
            
    def showpred(self, pred, times, comb):
        # comb=np.array([0.,0.7441 ,1.1903 ,0.9538 ,1.0474 ,0.4511 ,-0.4200, -1.7504])
        
        #plt.pause()
        pred = np.array(pred)
        fig, axs = plt.subplots(1,3,figsize=(20, 3))
        colors=['red','purple','purple','purple','purple','pink','purple']
        axisnames=['1','2','3','4','5','6','7']
        for anglex in range(pred.shape[1]):
            axs[anglex].plot(times,pred[:,anglex],color=colors[anglex],label='pred',zorder=5)
            #axs[anglex].errorbar(times,pred[:,anglex],yerr=predstd[:,anglex],color = 'black',alpha=0.4)
            axs[anglex].scatter(comb[0][0],comb[0][1+anglex],marker="X",color='black',zorder=10)
        
            axs[anglex].legend(loc='upper right',prop={'size': 7})
            axs[anglex].tick_params(axis='both', which='major', labelsize=11)
        
        
        plt.show() 
        #plt.close()


   

    def res2onehot(self, res):
        d = { "left":[1.,0.,0.], "right":[0.,0.,1.],"straddle":[0.,1.,0.]}
        return np.array(d[res],dtype='float')
    
    
    def cnmp_correct(self,tix, argf):

        numsamp = 20

        duration_in_sec = 6.0
        if len(argf)>=1:
            duration_in_sec = argf[0]

        cur_hand_pos, cur_R, cur_q_rad = self.request_pose_q(tix)
        cur_rope_pos, bufix = self.blob.get_special_blob_data()
        
        print "current rope pos:"+TOR.vec2str(cur_rope_pos)+"at (now - "+str(bufix)+" ticks ago)"
        
        if cur_rope_pos[0] > 80 or bufix<0:    # no blob data is available!
            print "******** --> No blob data is available, sorry cannot perform cnmp correction play  <--- *******\n"
            return

        if self.ccnmp == None:
            self.ccnmp = learningcode.cnmpfa.CNMPFA(1)     ## tix -> 1

        L = len(cur_hand_pos)
        context = cur_rope_pos - cur_hand_pos
        context = context[0:2]    # not using z as context
        
        np_observation = np.zeros((1,L+1), dtype='float')
        
        prediction_time=np.reshape(np.linspace(0.,1.,numsamp),(numsamp,1))


        print "observation:",np_observation
        print "context:", context

        pred  = self.ccnmp.cnmp_predict(1, np_observation, prediction_time,context)
        print pred
        
        
        pred = pred[0]
        
        print "pred shape:",pred.shape
        
        t_pred = np.concatenate((prediction_time, pred),axis=1)
        
        print "t_pred shape:",t_pred.shape
        
        print "time shape:",prediction_time.shape
        
        #self.showpred(pred, prediction_time, np_observation )
        
        self.pprint("Starting correction... context:"+TOR.vec2str(context)+" speed scale:"+  \
                    " action dur:"+str(duration_in_sec))        
        val = self.relcart2joint(tix, t_pred, duration_in_sec)
        self.pprint(TOR.col.WARNING+TOR.col.BOLD+"Correction done. The context was:"+TOR.vec2str(context)+" speed scale:"+  \
                    " action dur:"+str(duration_in_sec)+TOR.col.ENDC)
        return val
      
        
    def relcart2joint(self, tix, t_relpos, duration_in_sec=6.0):
          
        RELTRAJ_STEP = 1
        NUM_SAMPLE_FOR_PLAY = 10   # Was 20 before April 6, 2023
        cur_hand_pos, cur_R, cur_q_rad = self.request_pose_q(tix)
        print "current hand Pos:"+TOR.vec2str(cur_hand_pos)
        viapoint_list = []
        q, q9, q10 = self.request_q(tix)
        q_org = q10.copy()
        self.kin.set_torso_for_ik(q10[0:2])
        firstsoldone = False
        firstMAXIT = 1000
        laterMAXIT = 350
        MAXIT  = firstMAXIT
        numl = 0            
        
        print "current hand Pos:"+TOR.vec2str(cur_hand_pos)
        print "current q        :"+TOR.vec2str(cur_q_rad)        
        print "current q10      :"+TOR.vec2str(q10)   
        msec_st = t_relpos[0][0]
        for k in range(0,len(t_relpos), RELTRAJ_STEP):
            pos = t_relpos[k][1:4]+ cur_hand_pos
            msec = t_relpos[k][0] - msec_st
            R    = cur_R   #  NOTE this was R_end
            
            if k == 0:
                q, solved, errL,itused = self.kin.ik_corr(tix, pos, R, qinit=q10,  maxit=MAXIT)    # qinit may be omitted
                q10 = q_org  ## NOTE
            else:
                print q10
                if len(q10) != 0: 
                    print "qinit q10",TOR.vec2str(q10*180/np.pi), "degrees"
                else:
                    print "qinit q10 is []"
              
                q, solved, errL,itused = self.kin.ik_corr(tix, pos, R, qinit=q10, maxit=MAXIT)
         
            
            if q is None:
                self.pprint("   Line:"+str(numl)+ " > *********** Ooops! No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!")
                TOR.printRot(R)
                MAXIT = firstMAXIT
            elif not solved:
                if errL is None:
                    self.pprint("   Line:"+str(numl)+" > *********** Sorry, could not run IK (probably due to a bad R matrix)")    
                    TOR.printRot(R)
                    MAXIT = firstMAXIT
                else:
                    self.pprint("   Line:"+str(numl)+ " > Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(MAXIT,errL[0]*1000,errL[1], errL[2]))          
                print "IK FAILED for pos:"+TOR.vec2str(pos)+" and R:"
                TOR.printRot(R)
                print ("qinit(q10) used:",TOR.vec2str(q10,"%2.3f"))
                print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
                MAXIT = firstMAXIT
            else:
                q10 = q
                viapoint_list.append(q)
                self.pprint("Solved in %d iterations. (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                print "   Line:"+str(numl)+"> OK. IK found q:"+ TOR.vec2str(q)
                print "   Line:"+str(numl)+">       Ground_q:"+ TOR.vec2str(q)
                if firstsoldone == False:
                    firstsoldone = True
                    self.request_setdesqT(tix, q[0:9]) 
                MAXIT = laterMAXIT
            
            numl += 1
            
     
        print "num of lines IKed:", numl
 
             
        if len(viapoint_list) < 3:
            print viapoint_list
            print "Not sufficient number of via points can be solved (out of)",numl," tries. Not executing. Via point cnt:", len(viapoint_list)
            return False
        else:
            print TOR.col.WARNING+TOR.col.BOLD+"\n---> Number of viapoints/numberofattempted:"+str(len(viapoint_list))+\
            "/"+str(numl)+TOR.col.ENDC
            
            dt = 0.5*1.0*duration_in_sec / NUM_SAMPLE_FOR_PLAY # in seconds
            sampled_list = self.spline_fit_viapoints(np.asarray(viapoint_list), NUM_SAMPLE_FOR_PLAY)
            print('Executing the given xyz values')
            self.play_qlist(sampled_list, dt , tix)
            print("Relcart: Correction execution is done")
            return True
        

    def create_points(self,pointlist,q_start, NUM_SAMPLE_FOR_PLAY=50):
        des_z = np.array([0., 0., -1.])
        des_y = np.array([0., 1., 0.])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T

        #NUM_SAMPLE_FOR_PLAY = 50   # Was 20 before April 6, 2023
        q_curr=q_start
        viapoint_list = []
        for i in pointlist:
            q, solved, errL,itused = self.kin.ik(1, np.array(i), R_des, qinit=q_curr,  maxit=400, IKstep_show=0)
            
            if not solved:
                print pointlist
                print "IK failed for point:",i
                return None, None,None
            viapoint_list.append(q)

            q_curr = q
        viapoint_list.append(q_start)
        dt = 60.0 / NUM_SAMPLE_FOR_PLAY # in seconds
        sampled_list = self.spline_fit_viapoints_linear_2(np.asarray(viapoint_list), NUM_SAMPLE_FOR_PLAY)
        print('Executing the given xyz values')
        print("len viapoint_list", len(viapoint_list))
        return sampled_list, dt, viapoint_list
        
    def rotate_point(self,px, py, angle):
        """ Rotate a point (px, py) around the origin (0, 0) by the given angle (in degrees). """
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        x_new = px * cos_angle - py * sin_angle
        y_new = px * sin_angle + py * cos_angle
        return x_new, y_new
    def create_triang(self,center_x, center_y, distance, alpha):
        height = math.sqrt(3) / 2 * distance
        
        # Initial vertices relative to the center (before rotation)
        p1 = (distance, 0)
        p2 = (-distance / 2, height)
        p3 = (-distance / 2, -height)
        
        # Rotate each point around the center
        p1_rotated = self.rotate_point(p1[0], p1[1], alpha)
        p2_rotated = self.rotate_point(p2[0], p2[1], alpha)
        p3_rotated = self.rotate_point(p3[0], p3[1], alpha)
        
        # Translate the rotated points to the original center
        p1_final = (p1_rotated[0] + center_x, p1_rotated[1] + center_y)
        p2_final = (p2_rotated[0] + center_x, p2_rotated[1] + center_y)
        p3_final = (p3_rotated[0] + center_x, p3_rotated[1] + center_y)
        
        return p1_final[0],p1_final[1], p2_final[0],p2_final[1], p3_final[0],p3_final[1]

    def data_collector_triangle(self):
        des_z = np.array([0., 0., -1.])
        des_y = np.array([0., 1., 0.])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T
        center_x = 0.425 #+ 0.05
        center_y = -0.175 #+ 0.07
        dist = 0.07

        start_point = [center_x, center_y, 1.3]

        # Create base directory for saving images
        import os
        image_base_dir = "/home/arash/catkin_ws/src/erhtor3_work/triangle_images_fixed_cam"
        if not os.path.exists(image_base_dir):
            os.makedirs(image_base_dir)
            print("Created image base directory: " + image_base_dir)

        # # Tilt torso forward to help camera see the table
        # torso_tilt = 0.26  # radians (~28 degrees), adjust as needed (max ~1.4 rad / 80 deg)
        # qtorso = self.jcomm.getjointpos(TOR._TORSO)
        # qtorso[1] = torso_tilt
        # self.request_setdesq(TOR._TORSO, qtorso)
        # self.kin.set_torso_for_ik(qtorso)  # Tell IK solver to use tilted torso
        # print("Tilting torso forward by {:.1f} degrees...".format(torso_tilt * 180.0 / np.pi))
        # time.sleep(2.0)  # Wait for torso to reach position

        # # Look at the center of the workspace and wait for head to move
        # gaze_target = np.array([center_x, center_y, 0.865])
        # self.lookat(gaze_target)
        # time.sleep(1.0)  # Wait for head to reach target position

        # q, q9, q10 = self.request_q(TOR._RARM)
        # print("HI"*10)
        # print(qtorso)
        # print(q10)

        obs = []
        den = 0
        alpha = 0
        while den < 360:

            curretn_obs=[]
            # Create directory for this trajectory's images (overhead camera)
            traj_image_dir = os.path.join(image_base_dir, "traj_{:03d}".format(den))
            if not os.path.exists(traj_image_dir):
                os.makedirs(traj_image_dir)
            # Create directory for this trajectory's side camera images
            traj_image_dir_side = os.path.join(image_base_dir, "traj_side_{:03d}".format(den))
            if not os.path.exists(traj_image_dir_side):
                os.makedirs(traj_image_dir_side)

            # # Ensure head is looking at the workspace center
            # self.lookat(gaze_target)

            A_x,A_y,B_x,B_y,C_x,C_y=self.create_triang(center_x,center_y,dist,alpha)
            alpha+=1
            print("alpha", alpha)

            A=[A_x, A_y, 0.865]
            B=[B_x, B_y, 0.865]
            C=[C_x, C_y, 0.865]
            ABC=[A,B,C]
            # recorder.ABC=ABC

            q_start, solved, errL, itused = self.kin.ik(1, np.array(start_point), R_des, maxit=1000, IKstep_show=0)
            # q_start = q

            trajSTART_B=[start_point, [B_x, B_y, 0.9],[B_x, B_y,0.865],[B_x, B_y,0.98]]#8 POINT
            trajB_A=[[B_x, B_y,0.98], [A_x, A_y,0.98],[A_x, A_y,0.895]]#8 POINT
            trajA_C=[[A_x, A_y,0.895],[A_x, A_y,1.1],[C_x, C_y, 1.1], [C_x, C_y, 0.865]]#10 POINTT
            trajC_A=[ [C_x, C_y, 0.865], [C_x, C_y, 1.1], [A_x, A_y,1.1], [A_x, A_y,0.925]]#10 POINT
            trajA_FINISH=[ [A_x, A_y,0.925], [A_x, A_y,1.1]]

            all_traj=[start_point, [B_x, B_y, 0.95],[B_x, B_y,0.87],[B_x, B_y,0.98], [A_x, A_y,0.98],[A_x, A_y,0.9],[A_x, A_y,1.05],[C_x, C_y, 0.95], [C_x, C_y, 0.87], [C_x, C_y, 1.1], [A_x, A_y,1.0], [A_x, A_y,0.93]]
            # all_traj=[start_point, [B_x, B_y, 0.95],[B_x, B_y,0.87],[B_x, B_y,0.98], [A_x, A_y,0.98],[A_x, A_y,0.885],[A_x, A_y,1.05],[C_x, C_y, 0.95], [C_x, C_y, 0.87], [C_x, C_y, 1.1], [A_x, A_y,1.0], [A_x, A_y,0.90]]

            #all_traj=[start_point, [B_x, B_y,0.88],[A_x, A_y,0.91],[C_x, C_y, 0.88],  [A_x, A_y,0.94]]

            allsampled, delta_all, viapoints = self.create_points(all_traj, q_start, 150)
            print("allsampled.shape", allsampled.shape)
            if allsampled is None:
                continue
            # recorder.viapoints=viapoints

            self.move_point('point1', A_x, A_y, ABC[0][2])
            self.move_point('point2', ABC[1][0], ABC[1][1], ABC[1][2])
            self.move_point('point3', ABC[2][0], ABC[2][1], ABC[2][2])
            brakepoints=[viapoints[2],viapoints[5],viapoints[8],viapoints[11]]
            print("brakepoints", brakepoints)

            indexes = self.find_closest_indices(allsampled,brakepoints)
            print("indexes", indexes)

            if len(indexes) != 4:
                print("### Index length is not 4! ###")
                continue
            gripper_state=1
            one_hot=[0, 0, 0, 0]
            
            self.pprint("Going to initial pose...")
            # self.request_setdesqT(1, allsampled[0:9,0])   #command torso also  
            self.wait_until_reach(1, allsampled[0:10,0], SEC=12)

            curretn_obs.append([0,  allsampled[2:9, 0], ABC[0], ABC[1], ABC[2], one_hot])

            time.sleep(2.0)

            # Capture and save image for timestep 0 (overhead camera)
            img = self.get_current_image()
            if img is not None:
                img_filename = os.path.join(traj_image_dir, "step_{:03d}.jpg".format(0))
                cv2.imwrite(img_filename, img)
            # Capture and save image for timestep 0 (side camera)
            img_side = self.get_current_image_side()
            if img_side is not None:
                img_filename_side = os.path.join(traj_image_dir_side, "step_{:03d}.jpg".format(0))
                cv2.imwrite(img_filename_side, img_side)

            for i in range(1, allsampled.shape[1]):
                if i == indexes[0]:
                    gripper_state=0
                    one_hot = [1, 0, 0, 0]
                    print('true1 index:', i)
                    
                elif i == indexes[1]:
                    gripper_state=1
                    one_hot = [0, 1, 0, 0]
                    print('true2 index:', i)

                elif i == indexes[2]:
                    gripper_state=0
                    one_hot = [0, 0, 1, 0]
                    print('true3 index:', i)

                elif i == indexes[3]:
                    gripper_state=1
                    one_hot = [0, 0, 0, 1]
                    print('true4 index:', i)

                # self.request_setdesqT(1, allsampled[0:9,i]) 
                done = self.wait_until_reach(1, allsampled[0:10, i], SEC=delta_all)

                # Arash
                if one_hot[0]==1:
                    pos, _, _ = self.request_pose_q(TOR._RARM)
                    pos[2] -= 0.02
                    # comm.move('point2', pos)
                    self.move_point('point2', pos[0], pos[1], pos[2])
                elif one_hot[2]==1:
                    pos, _, _ = self.request_pose_q(TOR._RARM)
                    pos[2] -= 0.02
                    self.move_point('point3', pos[0], pos[1], pos[2])
                    # comm.move('point3', pos)
                  
                curretn_obs.append([i, allsampled[2:9, i], ABC[0], ABC[1], ABC[2], one_hot])
                # Capture and save image for this timestep (overhead camera)
                img = self.get_current_image()
                if img is not None:
                    img_filename = os.path.join(traj_image_dir, "step_{:03d}.jpg".format(i))
                    cv2.imwrite(img_filename, img)
                # Capture and save image for this timestep (side camera)
                img_side = self.get_current_image_side()
                if img_side is not None:
                    img_filename_side = os.path.join(traj_image_dir_side, "step_{:03d}.jpg".format(i))
                    cv2.imwrite(img_filename_side, img_side)

            # recorder.thread_record.join()

            # if done:
            print('Added to list')
            obs.append(curretn_obs)
            den += 1
            # print(recorder.true_num)
            print("den", den)
            print("len(obs)", len(obs))

            print "Episode finished!"
            # recorder.current_os=[]
            # if recorder.thread_record.isAlive:
            #     print('still alive ')

        
        name_str = "/home/arash/catkin_ws/src/erhtor3_work/triangle_images_fixed_cam/obs_triangle.npy"
        np.save(name_str, obs)  
        print("Triangle data collection is DONE")
        print(len(obs))
        print(len(obs[0]))
        print(len(obs[0][0]))
        
        
    def data_collector2(self):
        center = np.array([0.6, -0.1, 1.4])
        start_point = [0.6,-0.1, 1.7]
        ABC=np.array([[0.6, -0.3, 1.4],[0.6, -0.1, 1.4],[0.6, -0.1, 1.6]])
        des_z = np.array([1., 0., 0.])
        des_y = np.array([0., 1., 0.])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T
        q_start, solved, errL, itused = self.kin.ik(1, np.array(start_point), R_des, maxit=1000, IKstep_show=0)

        obs=[]
        alpha_list=np.linspace(-60,0,13)
        beta_list=np.linspace(-90,90,36)

        for i in alpha_list:
            alpha_radians = np.radians(i)

            rotation_matrix= np.array([[1,0,0],
                [0, np.cos(alpha_radians), -np.sin(alpha_radians)],
                [0, np.sin(alpha_radians),  np.cos(alpha_radians)]
                ])
            
            for j in beta_list:
                curretn_obs=[]
                pointlist=np.array([[0.0,-0.11, 0.11], [0.0,-0.11, 0.0],[0.0,0.0, -0.11],[0.0, 0.11, 0.0],[ 0.0, 0.0, 0.11], [0.0,-0.11, 0.11]])
                rotated_pointlist = [ [0.6, -0.1, 1.4],[0.6, -0.1, 1.4],[0.6, -0.1, 1.4],[0.6, -0.1, 1.4],[0.6, -0.1, 1.4],[0.6, -0.1, 1.4]]
                beta_radians   = np.radians(j)
                ABC=np.array([[0.6, -0.1, 1.4],[0.6, -0.1, 1.4],[0.6, -0.1, 1.4]])

                ABC_rotation=np.array([[0., -0.2, 0],[0, 0, 0],[0., 0, 0.2]])

                Ry = np.array([[np.cos(beta_radians), 0, np.sin(beta_radians)],
                                [0, 1, 0],
                                [-np.sin(beta_radians), 0, np.cos(beta_radians)]])


                all_rotation=np.dot(rotation_matrix, Ry)

                rotated_points = np.dot(all_rotation, pointlist.T).T
                ABC_rotated = np.dot(all_rotation, ABC_rotation.T).T
                for j in range(len(rotated_points)):
                    rotated_pointlist[j][0]+=rotated_points[j][0]
                    rotated_pointlist[j][1]+=rotated_points[j][1]
                    rotated_pointlist[j][2]+=rotated_points[j][2]
                for j in range(len(ABC_rotated)):
                    ABC[j][0]+=ABC_rotated[j][0]
                    ABC[j][1]+=ABC_rotated[j][1]
                    ABC[j][2]+=ABC_rotated[j][2]
                sample_list, delt = self.create_points(rotated_pointlist,q_start)
                if sample_list is None:
                    print(beta_radians)
                    print(alpha_radians)
                    print()
                    continue


                self.pprint("Going to initial pose...")
                self.request_setdesqT(1, sample_list[0:9,0])   #command torso also  
                self.wait_until_reach(1, sample_list[0:10,0], SEC=12)
            
                self.pprint("Starting play..dt="+str(delt))
                self.move_point('point1', ABC[0][0], ABC[0][1], ABC[0][2])

                self.move_point('point2', ABC[1][0], ABC[1][1], ABC[1][2])

                self.move_point('point3', ABC[2][0], ABC[2][1], ABC[2][2])

                for i in range(sample_list.shape[1]):
                    self.request_setdesqT(1,sample_list[0:9, i])   # torso IK result must be sent if IK included it!!,
                    time.sleep(delt)
                    curretn_obs.append([i, self.jcomm.getjointpos(1), ABC[0], ABC[1], ABC[2]])

                obs.append(curretn_obs)
                print "Play finished!"

        np.save("/home/deniz/catkin_ws/src/erhtor3_work/obs_task_v3_3d.npy", obs)  
        print("done")
        print(len(obs))
        print(len(obs[0]))
        print(len(obs[0][0]))


    def move_point(self, point_name, x, y, z):
        
        c = 0
        while self.deniz_pub.get_num_connections() == 0 :
            c +=1
            if c%10==0: print('Publisher still not ready');
            rospy.sleep(0.2)
            print('Publisher OK! c=%d'%c);
        try:
            state_msg = ModelState()
            state_msg.model_name = point_name
            state_msg.pose.position.x = x
            state_msg.pose.position.y = y
            state_msg.pose.position.z = z
            state_msg.pose.orientation.x = 0
            state_msg.pose.orientation.y = 0
            state_msg.pose.orientation.z = 0
            state_msg.pose.orientation.w = 1
            self.deniz_pub.publish(state_msg)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)



    def data_collector_table_stack2_copy(self):
        start_point = [0.4,-0.2, 1.3]
        recorder=Recorder([[5],[5],[],[5],[5],[5],[5],[5],[],[5],[5],[5],[5],[5],[],[5],[5],[5]],self.jcomm,[])

        des_z = np.array([0., 0., -1.])
        des_y = np.array([0., 1., 0.])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T
        recorder.count=0
        

        obs=[]
        den=0
        while den < 2:
            
            curretn_obs=[]

            A_x=random.uniform(0.35, 0.5)
            A_y=random.uniform(-0.35, 0.0)
            while True:
                B_x=random.uniform(0.3, 0.5)
                B_y=random.uniform(-0.4, 0.0)
                if ((A_x-B_x)**2. + (A_y-B_y)**2.)**(0.5)  >= 0.1:
                    break

            while True:
                C_x=random.uniform(0.3, 0.5)
                C_y=random.uniform(-0.4, 0.0)
                if ((A_x-C_x)**2. + (A_y-C_y)**2.)**(0.5) >= 0.1 and ((B_x-C_x)**2. + (B_y-C_y)**2.)**(0.5)  >= 0.1:
                    if abs(((A_x-C_x)/(A_y-C_y))-((B_x-C_x)/(B_y-C_y)))>0.2:
                        break

            A=[A_x, A_y, 0.865]
            B=[B_x, B_y, 0.865]
            C=[C_x, C_y, 0.865]
            ABC=[A,B,C]
            recorder.ABC=ABC
            
            

            q_start, solved, errL, itused = self.kin.ik(1, np.array(start_point), R_des, maxit=1000, IKstep_show=0)


            trajSTART_B=[start_point, [B_x, B_y, 0.9],[B_x, B_y,0.865],[B_x, B_y,0.98]]#8 POINT
            trajB_A=[[B_x, B_y,0.98], [A_x, A_y,0.98],[A_x, A_y,0.895]]#8 POINT
            trajA_C=[[A_x, A_y,0.895],[A_x, A_y,1.1],[C_x, C_y, 1.1], [C_x, C_y, 0.865]]#10 POINTT
            trajC_A=[ [C_x, C_y, 0.865], [C_x, C_y, 1.1], [A_x, A_y,1.1], [A_x, A_y,0.925]]#10 POINT
            trajA_FINISH=[ [A_x, A_y,0.925], [A_x, A_y,1.1]]
            all_traj=[start_point, [B_x, B_y, 0.95],[B_x, B_y,0.87],[B_x, B_y,0.98], [A_x, A_y,0.98],[A_x, A_y,0.9],[A_x, A_y,1.05],[C_x, C_y, 0.95], [C_x, C_y, 0.87], [C_x, C_y, 1.1], [A_x, A_y,1.0], [A_x, A_y,0.93]]

            allsampled, delta_all,viapoints=self.create_points(all_traj,q_start, 50)
            if allsampled is None:
                continue
            recorder.viapoints=viapoints

            self.move_point('point1', A_x, A_y, ABC[0][2])

            self.move_point('point2', ABC[1][0], ABC[1][1], ABC[1][2])

            self.move_point('point3', ABC[2][0], ABC[2][1], ABC[2][2])

            

            self.pprint("Going to initial pose...")
            print(viapoints[0])
            self.request_setdesqT(1, viapoints[0][0:9])   #command torso also  
            self.wait_until_reach(1, viapoints[0], SEC=12)
            recorder.true_num=0
            recorder.one_hot =[0,0,0,0]
            recorder.gripper_state=1
            # recorder.sleep_t=delta_all
            # recorder.thread_lock.acquire()
            # recorder.thread_active=True
            # recorder.thread_lock.release()
            
            recorder.thread_record=Thread(target=recorder.recorder_thread)
            recorder.thread_record.start()
            # curretn_obs.append([count, self.jcomm.getjointpos(1).tolist()+[recorder.gripper_state], ABC[0], ABC[1], ABC[2], recorder.one_hot ,delta_all ])
            # for i in range(1,allsampled.shape[1]):
            #     self.request_setdesqT(1, allsampled[0:9,i]) 
            #     self.wait_until_reach(1, allsampled[0:10,i], SEC=5)
            for i in range(1,len(viapoints)):
                self.request_setdesqT(1, viapoints[i][0:9]) 
                done=self.wait_until_reach(1, viapoints[i], SEC=6)
                if not done:
                    print('not addeed')
                    break
            recorder.thread_record.join()
            rospy.sleep(1)


            if recorder.true_num==4 and done:
                print('added to list')
                obs.append(recorder.current_os)
                den+=1
            print(recorder.true_num)
            print(den)
            print(len(obs))
            print "Play finished!"
            recorder.current_os=[]
            if recorder.thread_record.isAlive:
                print('still alive ')

        
        name_str="/home/deniz/catkin_ws/src/erhtor3_work/obs_10hz.npy"
        np.save(name_str, obs)  
        print("done")
        print(len(obs))
        print(len(obs[0]))
        print(len(obs[0][0]))


    def data_collector_table(self):
        start_point = [0.4,-0.2, 1.3]

        des_z = np.array([0., 0., -1.])
        des_y = np.array([0., 1., 0.])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T
        

        obs=[]
        den=0
        while den < 2500:
            
            curretn_obs=[]

            A_x=random.uniform(0.35, 0.5)
            A_y=random.uniform(-0.35, 0.0)
            while True:
                B_x=random.uniform(0.35, 0.5)
                B_y=random.uniform(-0.35, 0.0)
                if ((A_x-B_x)**2. + (A_y-B_y)**2.)**(0.5)  >= 0.1:
                    break

            # while True:
            #     C_x=random.uniform(0.35, 0.5)
            #     C_y=random.uniform(-0.35, 0.0)
            #     if ((A_x-C_x)**2. + (A_y-C_y)**2.)**(0.5) >= 0.1 and ((B_x-C_x)**2. + (B_y-C_y)**2.)**(0.5)  >= 0.1:
            #         if abs(((A_x-C_x)/(A_y-C_y))-((B_x-C_x)/(B_y-C_y)))>0.2:
            #             break

            A=[A_x, A_y, 0.865]
            B=[B_x, B_y, 0.865]
            # C=[C_x, C_y, 0.865]
            ABC=[A,B]
            # recorder.ABC=ABC
            
            

            q_start, solved, errL, itused = self.kin.ik(1, np.array(start_point), R_des, maxit=10000, IKstep_show=0)


            all_traj=[start_point, [B_x, B_y, 0.95],[B_x, B_y,0.87],[B_x, B_y,0.98], [A_x, A_y,0.98],[A_x, A_y,0.9]]#,[A_x, A_y,1.05],[C_x, C_y, 0.95], [C_x, C_y, 0.87], [C_x, C_y, 1.1], [A_x, A_y,1.0], [A_x, A_y,0.93]]
            #all_traj=[start_point, [B_x, B_y,0.88],[A_x, A_y,0.91],[C_x, C_y, 0.88],  [A_x, A_y,0.94]]

            allsampled, delta_all,viapoints=self.create_points(all_traj,q_start, 50)
            if allsampled is None:
                continue
            # recorder.viapoints=viapoints

            self.move_point('point1', A_x, A_y, ABC[0][2])

            self.move_point('point2', ABC[1][0], ABC[1][1], ABC[1][2])

            # self.move_point('point3', ABC[2][0], ABC[2][1], ABC[2][2])
            brakepoints=[viapoints[1],viapoints[2],viapoints[3],viapoints[4],viapoints[5]]

            indexes=self.find_closest_indices(allsampled,brakepoints)
            print(indexes)
            if len(indexes)!=len(brakepoints):
                continue
            one_hot=[0,0,0,0,0]
            

            self.pprint("Going to initial pose...")
            self.request_setdesqT(1, allsampled[0:9,0])   #command torso also  
            self.wait_until_reach(1, allsampled[0:10,0], SEC=12)

            # self.jcomm.getjointpos(1).tolist()
            # allsampled[2:9,i]
            curretn_obs.append([0,allsampled[2:9,0], all_traj[0],all_traj[1],all_traj[2],all_traj[3],all_traj[4],all_traj[5],all_traj[0], one_hot ])
            for i in range(1,allsampled.shape[1]):
                if i==indexes[0]:
                    one_hot=[1,0,0,0,0]
                    print 'true1'
                elif i==indexes[1]:
                    one_hot=[0,1,0,0,0]
                    print 'true2'
                elif i==indexes[2]:
                    one_hot=[0,0,1,0,0]
                    print 'true3'                 
                elif i==indexes[3]:
                    one_hot=[0,0,0,1,0]
                    print 'true4' 
                elif i==indexes[4]:
                    one_hot=[0,0,0,0,1]
                    print 'true5' 


                # elif i==indexes[2]:
                #     gripper_state=0
                #     one_hot=[0,0,1,0]
                #     print 'true3'
                # elif i==indexes[3]:
                #     gripper_state=1
                #     one_hot=[0,0,0,1]
                #     print 'true4'
                self.request_setdesqT(1, allsampled[0:9,i]) 
                done=self.wait_until_reach(1, allsampled[0:10,i], SEC=delta_all)
                # if not done:
                #     print('not addeed')
                #     break                    
                curretn_obs.append([i,allsampled[2:9,i] , all_traj[0],all_traj[1],all_traj[2],all_traj[3],all_traj[4],all_traj[5],all_traj[0], one_hot ])
            # recorder.thread_record.join()

            # if done and one_hot[4]==1:
            print('added to list')
            obs.append(curretn_obs)
            den+=1
            # print(recorder.true_num)
            print(den)
            print(len(obs))

            print "Play finished!"
            # recorder.current_os=[]
            # if recorder.thread_record.isAlive:
            #     print('still alive ')

        
        name_str="/home/deniz/catkin_ws/src/erhtor3_work/obs_sim_exp.npy"
        np.save(name_str, obs)  
        print("done")
        print(len(obs))
        print(len(obs[0]))
        print(len(obs[0][0]))

    def find_subsequence_indexes(self, main_list, sub_list):
        """
        Finds the indexes of the elements of sub_list in main_list in the same order.
        The elements of both lists are also lists.
        
        :param main_list: List of lists in which to search for the sub_list.
        :param sub_list: List of lists to find in the main_list.
        :return: List of indexes in main_list corresponding to the elements of sub_list.
        """
        indexes = []
        sub_list_index = 0

        for i in range(150):
            if sub_list_index < len(sub_list) and self.check_arrays( main_list[0:10,i], sub_list[sub_list_index]):
                indexes.append(i)
                sub_list_index += 1
                
                if sub_list_index == len(sub_list):
                    break


    # def find_closest_indices(self,list1, list2):
    #     closest_indices = []
    #     prev=0
        
    #     for lst2 in list2:
    #         min_distance = float('inf')
    #         closest_index = -1
    #         for i in range(prev+1, 300): #150? CHECK
    #             distance = np.linalg.norm(np.array(list1[0:10,i]) - np.array(lst2))
                
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 closest_index = i
            
    #         closest_indices.append(closest_index)
    #         prev=closest_index
        
    #     return closest_indices

    def find_closest_indices(self, list1, list2):
        closest_indices = []
        prev = 0
        for lst2 in list2:
            min_distance = float('inf')
            closest_index = -1
            for i in range(prev+1, 150): #150? CHECKAMZ
                distance = np.linalg.norm(np.array(list1[0:10, i]) - np.array(lst2))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i

            if closest_index == -1:
                continue
            else:
                closest_indices.append(closest_index)
                prev = closest_index
        return closest_indices


    def check_arrays(self,array1,array2):
        if len(array2)!=len(array1):
            return False

        for i in range(len(array1)):
            if abs(array1[i]-array2[i])>0.005:
                return False
        return True

    def cnmp_correct0(self,tix, argf):
        numsamp = 10
        T   = 1.0
        speed_scale = 1
        
        if len(argf)<4:
            print "You need to give the context! (4 numbers)"
            return
        
        print "argf ",argf
        context = np.array([argf[0],argf[1],argf[2],argf[3]])
        knn = KNearestNeighbor()
        
        
        
        lvec = self.latvec.get_latvec()
        print "latvec:"+TOR.vec2str(lvec)
        
        res  = knn.predict(np.reshape(lvec,(1,128)))[0]
        print ("Predicted class for the test image:", res)
        context = self.res2onehot(res)
        
        if len(argf)>=5:
            speed_scale = argf[4]
            
        if self.ccnmp == None:
            self.ccnmp = learningcode.cnmpfa.CNMPFA(tix)
        q, q9, q10 = self.request_q(tix)   
        prediction_time=np.reshape(np.linspace(0.,1.,numsamp),(numsamp,1))
        
        L = len(q)
        print "LEN:",L
        np_observation = np.zeros((1,L+1), dtype='float')
        np_observation[0, 1:]= q[0:L]
        print "observation:",np_observation
        print "context:", context
        
        pred  = self.ccnmp.cnmp_predict(1, np_observation, prediction_time,context)
        pred  = pred[0]
        print pred
        #self.showpred(pred, prediction_time, np_observation )
        self.pprint("Starting correction... context:"+TOR.vec2str(context)+" speed scale:"+str(speed_scale))
        
        dt = T/numsamp
        k = 0
        used_dt = dt/speed_scale
         
        while k<numsamp:
            if k%1==0:
               print "AIMING [",k,"]> ",tix,'  ',pred[k], 'used_sleep:',used_dt
               
            q10[0] = 0.0
            q10[1] = 10*np.pi/180
            q10[2:9] = pred[k] 
            self.request_setdesq(tix,q10[2:9])
            self.request_setdesq(TOR._TORSO,q10[0:2])
            #wait_until_reach(self, tix, q10, mask=1.0, SEC=12, angTH = 0.15*np.pi/180):
                
            #self.wait_until_reach(tix, q10, SEC=1, angTH=0.3*np.pi/180)
            k +=1
            rospy.sleep(used_dt)
        
        print "--- Waiting now..."
        rospy.sleep(1)
        print "Sleep finished. Going back"
        
        self.request_setdesq(tix,q)
        ##self.gazer.turnoff_gaze()
     
        

    def play_lookupcorrection(self, tix, args ):
        RELTRAJ_STEP = 5
        NUM_SAMPLE_FOR_PLAY = 20
        print args
        nDUR = 0
        if len(args)>0: nDUR = float(args[0])
            
        cur_hand_pos, cur_R, cur_q_rad = self.request_pose_q(tix)
        cur_rope_pos, bufix = self.blob.get_special_blob_data()
        print "current rope pos:"+TOR.vec2str(cur_rope_pos)+"at (now - "+str(bufix)+" ticks ago)"
        if cur_rope_pos[0] > 80 or bufix<0:    # no blob data is available!
            print "******** --> No blob data is available, sorry cannot perform correction play  <--- *******\n"
            return
        self.lup.readfolder()   # refresh the files
        t_relpos, rope_org, hand_org, R_org, Rend_org, q_org, file = self.lup.nearestTraj(cur_rope_pos, cur_hand_pos)
     
        print TOR.col.WARNING+TOR.col.BOLD+"\n---> Nearest File:"+file+TOR.col.ENDC
        viapoint_list = []
        q, q9, q10 = self.request_q(tix)
        #q10 = []
        firstsoldone = False
        firstMAXIT = 1000
        laterMAXIT = 350
        MAXIT  = firstMAXIT
        numl = 0            
        
        print "current hand Pos:"+TOR.vec2str(cur_hand_pos)
        print "current q        :"+TOR.vec2str(cur_q_rad)        
        msec_st = t_relpos[0][0]
        for k in range(0,len(t_relpos), RELTRAJ_STEP):
            posn = t_relpos[k][1:4] + hand_org  
            pos = t_relpos[k][1:4]+ cur_hand_pos
            print k," > pos differences:",TOR.vec2str(posn- pos)
            msec = t_relpos[k][0] - msec_st
            R    = Rend_org
            
            if k == 0:
                q, solved, errL,itused = self.kin.ik_corr(tix, pos, R, qinit=q10,  maxit=MAXIT)    # qinit may be omitted
                q10 = q_org
            else:
                print q10
                if len(q10) != 0: 
                    print "qinit q10",TOR.vec2str(q10*180/np.pi), "degrees"
                else:
                    print "qinit q10 is []"
              
                q, solved, errL,itused = self.kin.ik_corr(tix, pos, R, qinit=q10, maxit=MAXIT)
         
            
            if q is None:
                self.pprint("   Line:"+str(numl)+ " > *********** Ooops! No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!")
                TOR.printRot(R)
                MAXIT = firstMAXIT
            elif not solved:
                if errL is None:
                    self.pprint("   Line:"+str(numl)+" > *********** Sorry, could not run IK (probably due to a bad R matrix)")    
                    TOR.printRot(R)
                    MAXIT = firstMAXIT
                else:
                    self.pprint("   Line:"+str(numl)+ " > Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(MAXIT,errL[0]*1000,errL[1], errL[2]))          
                print "IK FAILED for pos:"+TOR.vec2str(pos)+" and R:"
                TOR.printRot(R)
                print ("qinit(q10) used:",TOR.vec2str(q10,"%2.3f"))
                print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
                MAXIT = firstMAXIT
            else:
                q10 = q
                viapoint_list.append(q)
                self.pprint("Solved in %d iterations. (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                print "   Line:"+str(numl)+"> OK. IK found q:"+ TOR.vec2str(q)
                print "   Line:"+str(numl)+">       Ground_q:"+ TOR.vec2str(q)
                if firstsoldone == False:
                    firstsoldone = True
                    self.request_setdesqT(tix, q[0:9]) 
                MAXIT = laterMAXIT
            
            numl += 1
            
     
        print "num of lines IKed:", numl
 
             
        if len(viapoint_list) < 3:
            print viapoint_list
            print "Not sufficient number of via points can be solved (out of)",numl," tries. Not executing. Via point cnt:", len(viapoint_list)
            return False
        else:
            print TOR.col.WARNING+TOR.col.BOLD+"\n---> Number of viapoints/numberofattempted:"+str(len(viapoint_list))+\
            "/"+str(numl)+"\n File was:"+file+TOR.col.ENDC
            DUR = msec - msec_st
            if nDUR>0: DUR = nDUR
            print "Trajectory was ",msec - msec_st,"msec. Will be played as ", DUR, "msecs.."
            dt = 0.5*1.0*DUR / NUM_SAMPLE_FOR_PLAY / 1000 # in seconds
            sampled_list = self.spline_fit_viapoints(np.asarray(viapoint_list), NUM_SAMPLE_FOR_PLAY)
            print('Executing the given xyz values')
            self.play_qlist(sampled_list, dt , tix)
            print("Correction execution is done")
            return True
       
    def makeknot(self, args):
        tix = 3
        speed1 = "1.0"
        dur_secs = 6.0
        speed_sc = float(speed1)
        knotfnm     = "/home/torcon/catkin_ws/src/erhtor3_work/tormain/DATA2GIT/OK_humop_20220908_132033_q"
        if len(args)>=1:
             speed_sc = float(args[0])   
        if len(args)>=2:
            dur_secs = float(args[1])
            
        p_t_args =  [knotfnm, speed1, "2"]               
        print "PLAYTRAJ Calling with args:",p_t_args
        self.play_traj(p_t_args)                      # play until the grasping point
        print "Sleeping a lot..."
        rospy.sleep(20)
        print "OK. Correction started..."
        corr_done = self.cnmp_correct(tix, [dur_secs])
        if corr_done: 
            print "No problem with the cnmp correction execution."
            p_t_args2 = [knotfnm, speed1, "3"]              # grasp and  play afterwards 
            print "Sleeping a bit..."
            rospy.sleep(4)
            self.play_traj(p_t_args2)
            print "All plays finished!"
        else:
            print "There was a problem in correction execution. Will stop here."
            
    # This seem to call the lookup table correction        
    def makeknot_file(self, args):
        tix = 3
        speed1 = "1.0"
        knotfnm     = "/home/torcon/catkin_ws/src/erhtor3_work/tormain/DATA2GIT/OK_humop_20220908_132033_q"
        if len(args)>=1:
            knotfnm=args[0]
        if len(args)>=2:
             speed1 = args[1]   
         
        p_t_args =  [knotfnm, speed1, "2"]               
        print "PLAYTRAJ Calling with args:",p_t_args
        self.play_traj(p_t_args)                      # play until the grasping point
        print "Sleeping a bit..."
        rospy.sleep(1.5)
        print "OK. Correction started..."
        corr_done = self.play_lookupcorrection(tix, ["5000"] )   # correcion time
        if corr_done: 
            print "No problem with the correction execution."
            p_t_args2 = [knotfnm, speed1, "3"]              # grasp and  play afterwards 
            print "Sleeping a bit..."
            rospy.sleep(1.5)
            self.play_traj(p_t_args2)
        else:
            print "There was a problem in correction execution. Will stop here."
            
    def playcart_traj(self, tix, args):
        FILE_LINE_STEP = 10
        nm = args[0]
        try:
            fp = open(nm,'r')
        except IOError:
            self.pprint("Cannot open "+nm+" for playing!")
            return
      
        viapoint_list=[]
        traj=[]
        numl = 0
        q, q9, q10 = self.request_q(tix)
        q10 = None
        firstsoldone = False
        firstMAXIT = 1000
        laterMAXIT = 250
        MAXIT  = firstMAXIT
        for line in fp:
            if numl%FILE_LINE_STEP != 0 : 
                numl += 1  
                continue    # jump over the lines 
                
            L = line.split(' ')
            L = [float(i) for i in L if i!='']
            #print "Line",numl,"[", line,']', "ARG_cnt :", len(L)
 
            fmno = int(L[0])
            msec = L[1]

            pos = np.array(L[2:5])    # for now assume it is setdesq
            Rrow1, Rrow2, Rrow3 = L[5:8],L[8:11],L[11:14]
            R = np.array([Rrow1,Rrow2,Rrow3])
            print R
            q_solution = L[14:24]
            
            if numl==0:
                #for k in range(0,len(L)):
                #    print "item k:",L[k]
                #print "ROT mat:",R
                #TOR.printRot(R)
                fmno_st = fmno
                msec_st = msec
            q10 = np.array(q10)
            
            if numl == 0:
                q, solved, errL,itused = self.kin.ik(tix, pos, R, qinit=q_solution, maxit=MAXIT)
            else:
                print q10
                if len(q10) != 0: 
                    print "qinit q10",TOR.vec2str(q10*180/np.pi), "degrees"
                else:
                    print "qinit q10 is []"
                q, solved, errL,itused = self.kin.ik(tix, pos, R, qinit=q10, maxit=MAXIT)
           
            
         
          
            if q is None:
                self.pprint("   Line:"+str(numl)+ " > *********** Ooops! No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!")
                TOR.printRot(R)
                print "   Line:"+str(numl)+"> This is the solution that should have been found:"+TOR.vec2str(q_solution)
            elif not solved:
                if errL is None:
                    self.pprint("   Line:"+str(numl)+" > *********** Sorry, could not run IK (probably due to a bad R matrix)")    
                    TOR.printRot(R)
                    print "   Line:"+str(numl)+"> This is the solution that should have been found:"+TOR.vec2str(q_solution)
                else:
                    self.pprint("   Line:"+str(numl)+ " > Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(MAXIT,errL[0]*1000,errL[1], errL[2]))          
                print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
            else:
                q10 = q
                viapoint_list.append(q)
                self.pprint("Solved in %d iterations. (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                print "   Line:"+str(numl)+"> OK. IK found q:"+ TOR.vec2str(q)
                print "   Line:"+str(numl)+">       Ground_q:"+ TOR.vec2str(q)
                if firstsoldone == False:
                    firstsoldone = True
                    self.request_setdesqT(tix, q[0:9]) 
            numl += 1
            MAXIT = laterMAXIT
            
        fp.close()
        print "num of lines:", numl
 
             
        NUM_SAMPLE = 20
        DUR = msec - msec_st
        print "Trajectory was ",DUR,"msec"
        dt = 0.5*1.0*DUR / NUM_SAMPLE / 1000 # in seconds
        sampled_list = self.spline_fit_viapoints(np.asarray(viapoint_list), NUM_SAMPLE)
        print('Executing the given xyz values')
        self.play_qlist(sampled_list, dt , tix)
        print("Execution is done")
        
        
  
        
    def play_traj(self, args):
        GRIPPER = TOR._LGRIP
        segmode_names=['full_play','stop_at_first_release','stop_at_first_grasp', 'from_grasp_to_end']
        segmode = 0
        nm = args[0]
        print "play args:"
        print args
        print "   lenargs:"+str(len(args))
        if len(args)>1:
            speed_scale = float(args[1])
            self.pprint("Playing wth speed scale:"+str(speed_scale))
        else:
            speed_scale = 1.0
            
        if len(args)>2:
            segmode = int(args[2])
            if segmode<0 or segmode>3:
                self.pprint("Requested segmode "+str(args[2])+" is invalid! Will not play. Valid modes: [0..3]")
                segmode = 0
            self.pprint("Using the segment mode:"+segmode_names[segmode])
        print "Requested to play"+nm+" with speed:"+str(speed_scale)+ " Segment mode:"+segmode_names[segmode]
        try:
            fp = open(nm,'r')
        except IOError:
            self.pprint("Cannot open "+nm+" for playing!")
            return
      
        traj=[]
        numl = 0
        if segmode==3: 
            add_points = False     # start playing after the 2nd grasp
        else:
            add_points = True      # play from the start up to a point
        for line in fp:
            #print "Line",numl,"[", line,']'
            L = line.split("/")
            fmno = int(L[0])
            msec = float(L[1])
            comm = L[2]    # for now assume it is setdesq
            if numl==0:
                fmno_st = fmno
                msec_st = msec
                
            bodyix  = int(L[3])

            f_args  = TOR.str2nums(L[4],',',False)
            if bodyix == GRIPPER and segmode !=0 :  # right gripper
                if segmode==1 and f_args[1]<0:  # release command 
                    self.pprint("Release command detected!")
                    print line
                    break
                if segmode==2 and f_args[1]>0:  # grasp command
                    self.pprint("Grasp command detected!")
                    print line
                    break 
                if segmode==3 and f_args[1]>0:  # grasp command
                    self.pprint("Grasp command detected!")
                    print line
                    add_points = True
            if add_points:        
                traj.append([fmno-fmno_st, msec-msec_st, comm, bodyix, f_args]) #last zero is for finger tip, unused
                numl += 1
        fp.close()
        #print traj
        
        ## self.gazer.turnon_gaze()
        k = 0
        d = traj[k]
        t, tix, myargs = float(d[1]), d[3], d[4]
 
        
        if tix == TOR._LGRIP or tix == TOR._RGRIP:
            print "GRASP command executing!"
            self.request_ezdesq(tix, myargs)  
            rospy.sleep(1.0)
            k +=1
            d = traj[k]
            t, tix, myargs = float(d[1]), d[3], d[4]
            
        self.pprint("Going to initial pose...")
        print "myargs:",myargs, "tix:",tix
        self.request_ezdesq(tix, myargs)  
        self.wait_until_reach(tix, myargs)
        self.pprint("Starting play..")
        while k<numl:
            if k%40==0:
                print k," PLAY >",tix,'  ',myargs
            self.request_ezdesq(tix, myargs)
            k +=1
            if k<numl:
                oldt = t
                d = traj[k]
                t, tix, myargs = float(d[1]), d[3], d[4]
                delt = (t-oldt)/1000  
                rospy.sleep(delt/speed_scale)
        
        ##self.gazer.turnoff_gaze()
         
    def lookat(self, args):
        pfix = args
        ##print "--> look at args :", args
        q = self.jcomm.getjointpos(TOR._HEAD)
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        q4 = np.hstack([qtorso,q])
        peye, Reye = self.kin.forwardkin(TOR._HEAD, q4)
        
        ##print "--> look at fixation origins matchetd:", pfix-peye
        pfix_loc = np.matmul(Reye.transpose(), pfix-peye)
        ##print "--> look at target in local coord:", TOR.vec2str(pfix_loc)
        dpan  = np.arctan2(pfix_loc[0], pfix_loc[2])
        dtilt = np.arctan2(pfix_loc[1], pfix_loc[2])
        ##print "--> look at del_pan del tilt:", dpan*180/np.pi, dtilt*180/np.pi
        ##print "--> current pan, tilt:", q[0]*180/np.pi, q[1]*180/np.pi
        self.request_setdesq(TOR._HEAD, [q[0]-dpan, q[1]+dtilt]) 
        
    def request_pose_q(self, tix):
        #print "<> requested pose for body group tix:",TOR._TARGET_L[tix]
        if tix == TOR._LARM or tix==TOR._RARM:
            q, q9, q10 = self.request_q(tix)
            p, R = self.kin.forwardkin(tix, q9)
        if tix == TOR._HEAD:
            qtorso = self.jcomm.getjointpos(TOR._TORSO)
            q = self.jcomm.getjointpos(tix)
            q4 = np.hstack([qtorso,q])
            p, R = self.kin.forwardkin(tix, q4)
            q10 = None
            
        return p, R, q10
    
    def request_q(self, tix):
        bodypart = TOR._TARGET_L[tix]
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        #self.pprint("\n qtorso:%s [degrees]"%TOR.vec2str(qtorso*180/np.pi, "%3.2f"))
        q = self.jcomm.getjointpos(tix)
        #self.pprint(" q_%s:%s [degrees]\n"%(bodypart,TOR.vec2str(q*180/np.pi,"%3.2f")))
        q9 = np.hstack([qtorso,q])
        q10 = np.hstack([q9,0])
        #print(q10)
        # print("Arash")
        # print("qtorso", qtorso)
        # print("q", q)
        # print("q9", q9)
        # print("q10", q10)
        # print("---")
        return q, q9, q10

   
    def makebutterfly(self, tix, argf):
        print "argf:",argf
        #qin = np.array([0., 0.35,  1.092,  0.76,  0.83,  1.76, 0.11, -0.258,  2.213,  0.])
        #print
        #self.wait_until_reach(tix, qin)
        if len(argf)>0:
            numpnt = int(argf[0])
            offset = np.array([argf[1],argf[2],argf[3]])
            scale  = argf[4]
            trace_butterfly(tix, self.kin, self, numpnt, offset,scale)
        else:
            trace_butterfly(tix, self.kin, self)
            
    # check if R_des normalization is OK in IK
    def makecircle(self, tix, args):

        K = len(args)
        DEF_itcnt = 1000
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        self.kin.set_torso_for_ik(qtorso)
        q10 = None

        if len(args)!=10 and len(args)!=11:
            self.pprint("ERROR: three vectors are needed as argument: position, Zaxis, Yaxis [maxit]")
            return
    
        itcnt = DEF_itcnt if K==9 else args[9]
        p_des = np.array(args[0:3])
        des_z = np.array(args[3:6])
        des_y = np.array(args[6:9])
        des_x = np.cross(des_y,des_z)
        radius = args[9]
        R_des = np.array([des_x, des_y, des_z]).T
       
        print "requested radius is:",radius
            
        q, solved, errL,itused = self.kin.ik(tix, p_des, R_des, maxit=itcnt)
    
        if q is None:
            self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
            return
        if not solved:
            self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
            print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
        else:
            self.pprint("Solved.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
            

        if solved:
            self.request_setdesq(tix,q[2:9])
            self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        return q, solved, errL
  
    # This function is called by tortouch.py  for teleoperation. tortouch may/should bypass torcomm.py
    # and directly request this from torkin.py
    # (check if R_des normalization is OK in IK)
    def request_setpose4IK(self, tix, args, mode=0):
        modestr=['fullIK', 'locationIK', 'orientationIK']
        K = len(args)
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        self.kin.set_torso_for_ik(qtorso)
        q10 = None
        
        if mode == TOR.ik_mode_BOTH:       # request full IK
            if len(args)!=10:
                self.pprint("ERROR [request_setpose4IK]: three vectors are needed as argument: position, Zaxis, Yaxis [maxit]")
                return

            itcnt =  args[9]
            p_des = np.array(args[0:3])
            des_z = np.array(args[3:6])
            des_y = np.array(args[6:9])
            des_x = np.cross(des_y,des_z)
            R_des = np.array([des_x, des_y, des_z]).T
            q, q9, q10 = self.request_q(tix)
        elif mode == TOR.ik_mode_POS:       # requesting location 
            if len(args)!=4:
                self.pprint("ERROR [request_setpose4IK]: one vector are needed as argument: position [maxit]")
                return
            itcnt = args[3]
            p, R_des, q10 = self.request_pose_q(tix)   # use current ori for ik
            p_des = np.array(args[0:3])   
        elif mode == TOR.ik_mode_ORI:       # requesting orientation
            if len(args)!=7:
                self.pprint("ERROR [request_setpose4IK]: two vectors are needed as argument: Z_axis, Y_axis [maxit]")
                return
            itcnt = args[6]
            p_des, R, q10 = self.request_pose(tix)   # use current pos for ik
            des_z = np.array(args[0:3])
            des_y = np.array(args[3:6])
            des_x = np.cross(des_y,des_z)
            R_des = np.array([des_x, des_y, des_z]).T
        else:
            self.pprint("ERROR [request_setpose4IK]: Use 0,1 or 2 for the IK mode (corresponding to fullIK, locationIK, orientationIK")

        q, solved, errL = self.kin.iik(tix, p_des, R_des, qinit = q10, maxit=itcnt)

    
        if q is None:
            self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
            return
        else:
            self.request_setdesq(tix,q[2:9])
            self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        return q, solved, errL
        # check if R_des normalization is OK in IK


        # check if R_des normalization is OK in IK

    def report_ikresult(self, q, solved, errL,itused, itcnt, live):
        if q is None:
            self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
            return
        if not solved:
            if live==0 or itcnt>=50:
                self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
            if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
        else:
            self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
          
            
    ''' Makes top to bottom pickup ortientation matrix
        with optional rotation around lateral -y- axis
    '''        
    def make_pickR(self, beta=0.0):
        pickZAXIS = np.array([0., 0., -1.])
        pickYAXIS = np.array([0., 1.,  0.])
        
        pickYAXIS /= norm(pickYAXIS)
        pickZAXIS /= norm(pickZAXIS)
        pickXAXIS = np.cross(pickYAXIS,pickZAXIS)
        R = np.vstack([pickXAXIS,pickYAXIS,pickZAXIS]).transpose()
        
        Rlocyrot = TOR.rotY(beta)
        Rnew = np.matmul(R,Rlocyrot)
        #TOR.printRot(Rnew,'   pickR=')
        return Rnew
    
    ''' makes a smooth path through 3 via points and executes.
        p1, p2, p3, [ikiteration [samled point count [ execution speed multiplier]]] 
    '''
    def request_3spline(self, tix, args):
        qtorso = np.array([0,20])*np.pi/180.0   # Forced torso
        #qtorso = self.jcomm.getjointpos(TOR._TORSO)  # If you want to use the current angles
        self.kin.set_torso_for_ik(qtorso)
        p, R, q_rad = self.request_pose_q(tix)
        
        R = self.make_pickR(-10*np.pi/180.0)
        print "You can try: ./ezcom setvialoc rarm 0.4 -0.55 0.9  0.4 0.15 0.9  0.4 0.0 0.9"
 
        q10=None
        baseT = 2.0  # seconds (use speed to change speed)
        via = [args[0:3], args[3:6], args[6:9]]
        #print "lenargs:",len(args), args
        
        if len(args)>9: 
            itcnt = int(args[9])
        else:
            itcnt = 500
           
        if len(args)>10:
            sample_N = int(args[10])
        else:
            sample_N = 8
            
        if len(args)>11: speed = float(args[11])
        else: speed = 1.0

        K = 3
        
        sol_q=[]
        issolved=[]
        errL=[]
        itused=[]
        indexL = []
        allsolved = True
        solvedc = 0
        for k in range(0,K):
            q, solved, errL_, itused_ = self.kin.ik_table(tix, via[k], R, maxit=itcnt, IKstep_show = False)    # can feed qinit from the previous?
            if (solved):
                sol_q.append(q)
                issolved.append(solved)
                errL.append(errL_) 
                itused.append(itused_)
                solvedc += 1
                indexL.append(k)
            else:
                print " x VIA ", k, " Canot be solved"
                self.report_ikresult(q, solved, errL_, itused_, itcnt, False)
            allsolved = allsolved and solved
        
        if solvedc<2:
            self.pprint('Sorry, cannot solve at least 2 points. Would not execute')
            return
        
        for k in range(0,len(sol_q)):
            print " Solved VIA ", indexL[k],":",TOR.vec2str(sol_q[k])
        sampled = self.spline_fit_viapoints(np.array(sol_q), sample_N )
        #print sampled
        self.play_qlist(sampled, baseT/sample_N/speed, tix)
        return sampled
            
    
    def make_spline(self, x,y):
        cs = CubicSpline(x, y)
        return cs    

    # def spline_fit_viapoints_linear(self, viapoint_array, sample_N):
    #     '''
    #     viapoint array shape: (x, 10) (numpy)
    #     return (10, sample_N)
    #     '''

    #     # Create a time array corresponding to the viapoints
    #     time = np.linspace(0, 1, num=viapoint_array.shape[0])
    #     cs_joint_list = []

    #     # Create linear interpolation for each joint dimension
    #     for i in range(viapoint_array.shape[1]):
    #         cs = interp1d(time, viapoint_array[:, i], kind='linear')
    #         cs_joint_list.append(cs)

    #     # Ensure that sampled_time includes viapoint times
    #     sampled_time = np.linspace(time[0], time[-1], num=sample_N)
    #     sampled_time = np.unique(np.concatenate((sampled_time, time)))

    #     sample_list = []
    #     for idx, cs in enumerate(cs_joint_list):
    #         # Evaluate the linear interpolant at the sampled times
    #         sampled_curve = cs(sampled_time)
    #         sample_list.append(np.asarray(sampled_curve))

    #     # Convert list of sampled curves to a numpy array
    #     sample_list = np.asarray(sample_list)
    #     return sample_list

    def spline_fit_viapoints_linear(self, viapoint_array, sample_N):
        '''
        viapoint array shape: (x, 10) (numpy)
        return (10, sample_N)
        '''

        time = np.linspace(0, 1, num=viapoint_array.shape[0])
        cs_joint_list = []

        for i in range(viapoint_array.shape[1]):
            cs = interp1d(time, viapoint_array[:, i], kind='linear')
            cs_joint_list.append(cs)

        sample_list=[]
        for idx, cs in enumerate(cs_joint_list):
            #each cs is for a spesific joint movement

            sampled_time = np.linspace(time[0], time[-1], num=sample_N) #sample from time
            sampled_curve = cs(sampled_time)
            sample_list.append(np.asarray(sampled_curve))

        sample_list= np.asarray(sample_list)
        return sample_list
    
    def spline_fit_viapoints_linear_2(self,viapoint_array, sample_N):
        '''
        viapoint_array shape: (n, 10) (numpy)
        return (10, sample_N)
        '''
        # Calculate cumulative distances along the via points
        distances = np.zeros(viapoint_array.shape[0])
        for i in range(1, viapoint_array.shape[0]):
            distances[i] = distances[i-1] + np.linalg.norm(viapoint_array[i] - viapoint_array[i-1])

        # Normalize distances to range [0, 1]
        distances /= distances[-1]

        cs_joint_list = []
        for i in range(viapoint_array.shape[1]):
            cs = CubicSpline(distances, viapoint_array[:, i], bc_type='natural', extrapolate=False )
            cs_joint_list.append(cs)

        # Sample from the cumulative distance
        sampled_distances = np.linspace(0, 1, sample_N)
        sample_list = []
        for cs in cs_joint_list:
            sampled_curve = cs(sampled_distances)
            sample_list.append(sampled_curve)

        sample_list = np.asarray(sample_list)
        return sample_list
    # def spline_fit_viapoints_mixed(self,viapoint_array, sample_N):
    #     '''
    #     viapoint_array shape: (n, 10) (numpy)
    #     return (10, sample_N)
    #     '''
    #     # Calculate cumulative distances along the via points
    #     distances = np.zeros(viapoint_array.shape[0])
    #     for i in range(1, viapoint_array.shape[0]):
    #         distances[i] = distances[i-1] + np.linalg.norm(viapoint_array[i] - viapoint_array[i-1])

    #     # Normalize distances to range [0, 1]
    #     distances /= distances[-1]

    #     cs_joint_list = []
    #     for i in range(viapoint_array.shape[1]):
    #         if i==1 or i==4 or i==5 or i==7 or i==10:
    #             cs = interp1d(distances, viapoint_array[:, i], kind='linear')
    #         else:
    #             cs = interp1d(distances, viapoint_array[:, i], kind='linear')
    #         cs_joint_list.append(cs)

    #     # Sample from the cumulative distance
    #     sampled_distances = np.linspace(0, 1, sample_N)
    #     sample_list = []
    #     for cs in cs_joint_list:
    #         sampled_curve = cs(sampled_distances)
    #         sample_list.append(sampled_curve)

    #     sample_list = np.asarray(sample_list)
    #     return sample_list
    
    def spline_fit_viapoints(self, viapoint_array, sample_N):
        '''
        viapoint array shape: (x, 10) (numpy)
        return (10, sample_N)
        '''

        time = np.linspace(0,1, num=viapoint_array.shape[0])
        cs_joint_list=[]

        for i in range(viapoint_array.shape[1]):
            cs = CubicSpline(time, viapoint_array[:,i])
            cs_joint_list.append(cs)

        #we did spline fit. there is no need for x anymore.

        sample_list=[]
        for idx, cs in enumerate(cs_joint_list):
            #each cs is for a spesific joint movement

            sampled_time = np.linspace(time[0], time[-1], num=sample_N) #sample from time
            sampled_curve = cs(sampled_time)
            sample_list.append(np.asarray(sampled_curve))

        sample_list= np.asarray(sample_list)
        return sample_list

        
   
    ''' Given a list of [torso.arm.gripper] joint angles, it executes them
        with a delay of delt in between
    '''
    def play_qlist(self, sample_list, delt, tix):
        self.pprint("Going to initial pose...")
        self.request_setdesqT(tix, sample_list[0:9,0])   #command torso also  
        self.wait_until_reach(tix, sample_list[0:10,0], SEC=12)
        
        self.pprint("Starting play..dt="+str(delt))
        
        for i in range(sample_list.shape[1]):
            self.request_setdesqT(tix,sample_list[0:9, i])   # torso IK result must be sent if IK included it!!
            #print "Sent spline sample ",i," : ",sample_list[0:9, i]
            p, R, q_rad = self.request_pose_q(tix)
            time.sleep(delt)
        print "Play finished!"
        
    def request_setposeG(self, tix, args, live=0):
        K = len(args)
        DEF_itcnt = 1000
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        self.kin.set_torso_for_ik(qtorso)
        q10 = None
        print ("in setposeG")
        if len(args)!=9 and len(args)!=10:
            self.pprint("ERROR: three vectors are needed as argument: position, Zaxis, Yaxis [maxit]")
            return

        itcnt = DEF_itcnt if K==9 else args[9]
        p_des = np.array(args[0:3])
        des_z = np.array(args[3:6])
        des_y = np.array(args[6:9])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T
  
        q, solved, errL, itused = self.kin.ik_table(tix, p_des, R_des, maxit=itcnt, IKstep_show = (live==1))
        #q, solved, errL, itused =       self.kin.ik(tix, p_des, R_des, maxit=itcnt, IKstep_show = (live==1))
        self.report_ikresult(q, solved, errL, itused, itcnt, live)
        
        
        if solved:
            self.request_setdesq(tix,q[2:9])
            self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
            print "request_setposeG> sending the robot q:",q
        return q, solved, errL
       
    """
    Pick with either hand using a heuristic
    """
    def execute_pickupfree(self, tix, args, live=0, IK_qinit_current=False):
        dummy=1
        

    """
    Pick with left or right hand  pickup <larms|rarm>  <blob1 no> <blob2no> 
    Align the opposition axis perpenducuklar to blob1-blob2 and pick from the center
    """           
    def execute_pickup_rope(self, tix, args, live=0, IK_qinit_current=False):
        beta = 0*np.pi/180.
        pickZAXIS = np.array([0., 0., -1.])
        pickYAXIS = np.array([0., 1.,  0.])
        
        beta = float(args[0])*np.pi/180.   # Top pick is beta=0. This rotates this orientation around the open-close axis of the gripper      
        
        res = self.blob.get_rope_ends('red','blue')
        if len(res)==0:
            return 
        A = res[0]
        B = res[1]
       
       
        if len(B)==0:    # assume the same end is returned
            B = A
        if len(A)==0 or len(B)==0:
            print ("execute_pickup_rope:something wrong here!")
            return
           
        endW = 0.9  # I think 1 means hold from the tip color. 0    
        posA = endW*A[1]+(1-endW)*A[0]   #end point of the rope A
        posB = endW*B[1]+(1-endW)*B[0]   #end point of the rope B
       
        side = 1-2*(tix == TOR._LARM)    # left arm:-1, right arm=1  right arm should pick Y posiiton towards the - side
           
        print "posA:",TOR.vec2str(posA)
        print "posB:",TOR.vec2str(posB)
        
        if side*posA[1] <= side*posB[1]: 
            print "Selecting posA"
            pos, pickXAXIS = posA, (A[0]-A[1])*side
        else:
            print "Selecting posB"
            pos, pickXAXIS = posB, (B[0]-B[1])*side     

        pos[2] += 0.001/2          # a bit margin for avoid ground
        
        #----
        
        pickXAXIS /= norm(pickXAXIS)
        pickZAXIS /= norm(pickZAXIS)
        pickYAXIS = np.cross(pickXAXIS, pickZAXIS)
        pickXAXIS = np.cross(pickYAXIS,pickZAXIS)
        R = np.vstack([pickXAXIS,pickYAXIS,pickZAXIS]).transpose()
        TOR.printRot(R,'   R=')
        Rlocyrot = TOR.rotY(beta)
        Rnew = np.matmul(R,Rlocyrot)
        newargs = np.hstack([pos, Rnew[2,:], Rnew[1,:]])
        TOR.printRot(Rnew,'Rnew=')
        #------
        
        #newargs = np.hstack([pos, pickZAXIS, pickYAXIS])
        print newargs
        self.execute_reachgrasp(tix,newargs, live,IK_qinit_current, grasp=True)
        
    """
    Pick with left or right hand  pickup <larms|rarm>  <blob no>
    """    
    def execute_pickup_blob(self, tix, args, live=0, IK_qinit_current=False):
        beta = 0*np.pi/180.
        pickZAXIS = np.array([0., 0., -1.])
        pickYAXIS = np.array([0., 1.,  0.])
        
        if len(args)>1: beta = float(args[1])*np.pi/180.   # default pick is from top. This rotates this orientation around the open-close axis of the gripper
        bix = int(args[0])
        [cen,info] = self.blob.get_centers()
        if bix<0 or bix>= len(info):
           self.pprint("There is no such Blob index:"+bix+"!")
           return
        
        pos = np.array(cen[bix])
        if any(np.isnan(pos)):
            self.pprint("Blob "+bix+info[bix]+" is not visible")
            return
        
        pos[2] += 0.01
        
        #----
        pickYAXIS /= norm(pickYAXIS)
        pickZAXIS /= norm(pickZAXIS)
        pickXAXIS = np.cross(pickYAXIS,pickZAXIS)
        R = np.vstack([pickXAXIS,pickYAXIS,pickZAXIS]).transpose()
        TOR.printRot(R,'   R=')
        Rlocyrot = TOR.rotY(beta)
        Rnew = np.matmul(R,Rlocyrot)
        newargs = np.hstack([pos, Rnew[2,:], Rnew[1,:]])
        TOR.printRot(Rnew,'Rnew=')
        #------
        
        #newargs = np.hstack([pos, pickZAXIS, pickYAXIS])
        print newargs
        self.execute_reachgrasp(tix,newargs, live,IK_qinit_current, grasp=True)
        
    def execute_reachgrasp(self, tix, args, live=0, IK_qinit_current=False, grasp=False):
        if tix == TOR._RARM:   gtix =TOR._RGRIP
        elif tix == TOR._LARM: gtix =TOR._LGRIP
        else: self.pprint("execute_grasp: can not grasp with body part "+TOR._TARGET_L[tix]); return

        if grasp==True: 
            self.request_setgripper(gtix, openit=True)
            q_in, q9_in, q10_in = self.request_q(tix)
             
        q10, solved, errL = self.request_setposeG(tix, args, live)
        if not solved:
            self.pprint("IK cannot converge, aborting grasp/reach request")
            self.pprint("ERRORS: (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(errL[0]*1000,errL[1], errL[2], TOR.vec2str(q10*180/np.pi,"%3.2f")))
            return
        else:  # if solved
            if grasp==True: 
                self.wait_until_reach(tix, q10)
                self.request_setgripper(gtix, openit=False);    rospy.sleep(2)   # wait for close
                self.request_setdesq(tix,q_in)                                   # go back to where started
                self.request_setdesq(TOR._TORSO, q10_in[0:2])  

    
    def execute_reachgrasp_deniz(self, tix, args,args_second, live=0, IK_qinit_current=False, grasp=False):
        if tix == TOR._RARM:   gtix =TOR._RGRIP
        elif tix == TOR._LARM: gtix =TOR._LGRIP
        else: self.pprint("execute_grasp: can not grasp with body part "+TOR._TARGET_L[tix]); return

        if grasp==True: 
            self.request_setgripper(gtix, openit=True)
            q_in, q9_in, q10_in = self.request_q(tix)
             
        q10, solved, errL = self.request_setposeG(tix, args, live)
        q10_second, solved_second, errL_second = self.request_setposeG(tix, args_second, live)
        if not solved:
            self.pprint("IK cannot converge, aborting grasp/reach request")
            self.pprint("ERRORS: (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(errL[0]*1000,errL[1], errL[2], TOR.vec2str(q10*180/np.pi,"%3.2f")))
            return
        else:  # if solved
            if grasp==True: 
                self.wait_until_reach(tix, q10_second)
                self.wait_until_reach(tix, q10)
                self.request_setgripper(gtix, openit=False);    rospy.sleep(2)   # wait for close
                self.request_setdesq(tix,q_in)                                   # go back to where started
                self.request_setdesq(TOR._TORSO, q10_in[0:2])  
             
    """
    Wait for joints to reach a certain configuration. Use mask to indicate which joints to care. [1.,1,1,1,0,0,0..] etc.
    """
    def wait_until_reach(self, tix, q10, mask=1.0, SEC=12, angTH = 0.5*np.pi/180):
        reached = True
        
        sleepT = 0.001
        k = int(SEC/sleepT)
        angERR = 1; 
        while angERR > angTH and k > 0:

            aq, aq9, aq10 = self.request_q(tix)

            #print(aq10.shape, q10.shape)
            angERR = norm((aq10 - q10)*mask)/10.0

            self.request_setdesqT(1, q10[0:9]) 
            rospy.sleep(0.001)
            # rospy.sleep(0.0005)


            k -= 1
        #endwhile
        
        if angERR > angTH: 
            print TOR.col.WARNING+"Joint targets cannot be reached avg angle error (deg):"+str(angERR*180/np.pi)+TOR.col.ENDC
            reached = False
        
        return reached
            
    def gripper_open(self, tix):
        pars = np.array([TOR.gripLOPEN_dist, TOR.gripLOPEN_force]) if tix == TOR._LGRIP else np.array([TOR.gripROPEN_dist, TOR.gripROPEN_force])   
        self.request_setdesq(tix, pars)
    
    def gripper_close(self, tix):
        pars =  np.array([TOR.gripLENC_dist, TOR.gripLENC_force]) if tix == TOR._LGRIP else np.array([TOR.gripRENC_dist, TOR.gripRENC_force]) 
        self.request_setdesq(tix, pars)
        
        
    """
    Open or close the gripper with fixed parameter from global_defines.py (TOR)
    """        
    def request_setgripper(self, gtix, openit=True):
        if gtix!=TOR._RGRIP and gtix!=TOR._LGRIP:
            self.pprint("setgripper: bad gripper index given:"+str(gtix))
            return
        
        if openit:
            self.gripper_open(gtix) #self.request_setdesq(gtix, np.array([TOR.gripOPEN_dist, TOR.gripOPEN_force]))
        else:
            self.gripper_close(gtix) #self.request_setdesq(gtix, np.array([TOR.gripENC_dist, TOR.gripENC_force])) 
            
    # check if R_des normalization is OK in IK

    def request_setpose(self, tix, args, mode=0, live=0, IK_qinit_current=False):
        modestr=['fullIK', 'locationIK', 'orientationIK']
        K = len(args)
        DEF_itcnt = 1000
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        self.kin.set_torso_for_ik(qtorso)
        q10 = None
        if mode == TOR.ik_mode_BOTH:       # request full IK
            if len(args)!=9 and len(args)!=10:
                self.pprint("ERROR: three vectors are needed as argument: position, Zaxis, Yaxis [maxit]")
                return
            #itcnt = DEF_itcnt*(K==9) + (K==10)*args[9]
            itcnt = DEF_itcnt if K==9 else args[9]
            p_des = np.array(args[0:3])
            des_z = np.array(args[3:6])
            des_y = np.array(args[6:9])
            des_x = np.cross(des_y,des_z)
            R_des = np.array([des_x, des_y, des_z]).T
        elif mode == TOR.ik_mode_POS:       # requesting location 
            if len(args)!=3 and len(args)!=4:
                self.pprint("ERROR: one vector are needed as argument: position [maxit]")
                return
            itcnt = DEF_itcnt if K==3 else args[3]
            p, R_des, q10 = self.request_pose_q(tix)   # use current ori for ik
            p_des = np.array(args[0:3])   
        elif mode == TOR.ik_mode_ORI:       # requesting orientation
            if len(args)!=6 and len(args)!=7:
                self.pprint("ERROR: two vectors are needed as argument: Z_axis, Y_axis [maxit]")
                return
            itcnt = DEF_itcnt if K==6 else args[6]
            p_des, R, q10 = self.request_pose(tix)   # use current pos for ik
            des_z = np.array(args[0:3])
            des_y = np.array(args[3:6])
            des_x = np.cross(des_y,des_z)
            R_des = np.array([des_x, des_y, des_z]).T
        else:
            self.pprint("ERROR: Use 0,1 or 2 for the IK mode (corresponding to fullIK, locationIK, orientationIK")
        #0-----------------------------------------------------------------------
        #self.pprint("Solving %s with max %d iterations"%(modestr[mode], itcnt))
        if IK_qinit_current:
            if q10 is None:
                q, q9, q10 = self.request_q(tix)

            #print 'q10 is ',q10
            #if live==0: print "setpose R_des:\n",R_des
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        else:
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, maxit=itcnt, IKstep_show = (live==1))
    
        if q is None:
            self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
            return
        if not solved:
            if live==0 or itcnt>=50:
                self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
            if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
        else:
            self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
            #print q

        if solved or live==1:
            self.request_setdesq(tix,q[2:9])
            self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        return q, solved, errL
        # check if R_des normalization is OK in IK
    def sample_points_in_circle(self, center, radii, num_points_per_circle):
        """
        Sample points in circular patterns with varying radii.

        Args:
        - center (tuple): Center coordinates of the circles (x, y).
        - radii (list or array): Range of radii of the circles.
        - num_points_per_circle (int): Number of points to sample per circle.

        Returns:
        - points (np.array): Array of sampled points of shape (num_points, 2).
        """
        points = []
        for radius in radii:
            theta = np.linspace(-2*np.pi/6, 2*np.pi/6, num_points_per_circle)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            circle_points = np.column_stack((x, y))
            points.append(circle_points)
        return np.vstack(points)
    

    def measure(self):
        des_z = np.array([1., 0., 0.])
        des_y = np.array([0., 1., 0.])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T
        p_start=np.array([+0.5, -0.15, +1.0])#burası değişecek
        q, solved, errL, itused = self.kin.ik(1, p_start, R_des, maxit=1000, IKstep_show=0)
        self.request_setdesq(1,q[2:9])
        p_end=np.array([+0.5, 0.0, +1.8])
        q2, solved, errL, itused = self.kin.ik(1, p_end, R_des, maxit=1000, IKstep_show=0,qinit=q)
        rospy.sleep(5)
        self.request_setdesq(1,q2[2:9])
        start=time.time()
        p, R, q_rad = self.request_pose_q(1)
        print("in")
        while abs(p[0]-p_end[0])>0.05 or abs(p[1]-p_end[1])>0.05 or abs(p[2]-p_end[2])>0.05:
            p, R, q_rad = self.request_pose_q(1)
            continue
        print("out")
        end=time.time()
        print(end-start)
        return


    def data_collector(self):
        des_z = np.array([1., 0., 0.])
        des_y = np.array([0., 1., 0.])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T
        p_start=np.array([+0.5, -0.15, +1.0])#burası değişecek
        q, solved, errL, itused = self.kin.ik(1, p_start, R_des, maxit=1000, IKstep_show=0)
        self.request_setdesq(1,q[2:9])

        z_range = np.linspace(1.3, 1.8, 51)
        y_range = np.linspace(-0.3, 0.0, 16)
        y_range_inverse = np.linspace(0.0,-0.3, 16)
        print(y_range_inverse)
        #points = self.sample_points_in_circle(mid, radii, num_points_per_circle)
        obs=[]
        print("started")
        start_point=True
        count=0
        for i in z_range:
            if count%2==0:
                for j in y_range:
                    if start_point:
                        start_point=False
                        q_dest=q
                        continue
                    self.request_setdesq(1,q[2:9])
                    p_dest=np.array([0.5, j, i])
                    q_dest, solved, errL, itused = self.kin.ik(1, p_dest, R_des, maxit=1000, IKstep_show=0,qinit=q_dest)
                    rospy.sleep(5)

                    if solved:
                        current=[]
                        current.append([0, self.jcomm.getjointpos(1), p_dest])
                        self.ezcon.setdesq(1,q_dest[2:9])
                        for k in range(1,50):     
                            rospy.sleep(0.1)        
                            current.append([k, self.jcomm.getjointpos(1), p_dest])
                        obs.append(current)
                    else:
                        q_dest=None   
                        print(i)
                        print(j)
                    
            else:
                for j in y_range_inverse:
                    self.request_setdesq(1,q[2:9])
                    p_dest=np.array([0.5, j, i])
                    q_dest, solved, errL, itused = self.kin.ik(1, p_dest, R_des, maxit=1000, IKstep_show=0,qinit=q_dest)
                    rospy.sleep(5)

                    if solved:
                        current=[]
                        current.append([0, self.jcomm.getjointpos(1), p_dest])
                        self.ezcon.setdesq(1,q_dest[2:9])
                        for k in range(1,50):  
                            rospy.sleep(0.1)           
                            current.append([k, self.jcomm.getjointpos(1), p_dest])
                        obs.append(current)
                    else:
                        q_dest=None
                        print(i)
                        print(j)

            count+=1

        np.save("/home/deniz/catkin_ws/src/erhtor3_work/obs_final_degree.npy", obs)  
        print(count)
        print("done")
        print(len(obs))

    def print_data(self):
        x=[]
        y=[]
        z=[]
        data=np.load('/home/deniz/catkin_ws/src/erhtor3_work/obs_final.npy',allow_pickle=True)
        print(data.shape)
        print(data[0][0])
        for i in data[800]:
            p_cur, R, q_rad = self.request_pose_q(1)
            p_cur,_ = self.kin.forwardkin(1,i[1])
            x.append(p_cur[0])
            y.append(p_cur[1])
            z.append(p_cur[2])


            print(i[1])
            print(p_cur)
            #self.ezcon.setdesq(1,i[1])
            #rospy.sleep(5)
        # Create figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        ax.scatter(x, y, z, c='r', marker='o')

        # Set labels and title
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Scatter Plot')

        # Adjust viewing angle
        ax.view_init(elev=20, azim=30)

        # Display plot
        plt.show()





    def request_correction(self, tix, args, mode=0, live=0, IK_qinit_current=False):
        K = len(args)
        DEF_itcnt = 1000
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        self.kin.set_torso_for_ik(qtorso)
        q10 = None

        gms = self.gms_client()

        
        red_object = gms('wood_cube_7_5cm_clone_1','ground_plane')
        obj_center_x = red_object.pose.position.x
        obj_center_y = red_object.pose.position.y
        

        bottom_obj = gms('wood_cube_7_5cm','ground_plane')
        bottom_obj_center_x = bottom_obj.pose.position.x
        bottom_obj_center_y = bottom_obj.pose.position.y 

        bottom_right_obj = gms('wood_cube_7_5cm_clone','ground_plane')
        bottom_right_obj_center_x = bottom_right_obj.pose.position.x
        bottom_right_obj_center_y = bottom_right_obj.pose.position.y 

        if abs(bottom_right_obj_center_x- obj_center_x)< 0.08:


        
            #from up
            self.request_setdesq(TOR._ALL, TOR.q_tuba_pre_correction)
            time.sleep(4)
            self.request_setdesq(TOR._ALL, TOR.q_tuba_correction)
            time.sleep(4)

            # we have our desired position, we need current position, and object center position. 
            # after that, we will apply cubic spline fit and get sampled points. then we will apply ik to each one of them.

            #current pose of the rarm
            p, R, q_rad = self.request_pose_q(tix)

            end_effector_x = p[0]
            end_effector_y = p[1]

            #get pose of the object and the corridor
        

            #x and y is defined according to the robots coordinates
            y= np.array([end_effector_y, end_effector_y ]) #it must be in increasing order
            x= np.array([ bottom_obj_center_x+0.14, end_effector_x,])
            cs = self.create_curve(x, y)
            xs = np.linspace(x[0], x[-1], num=4) #sample from y
            
            sampled_curve = cs(xs)

            sampled_curve = np.flipud(sampled_curve)
            xs = np.flipud(xs)
        elif abs(bottom_obj_center_x- obj_center_x)< 0.08:
            p, R, q_rad = self.request_pose_q(tix)

            end_effector_x = p[0]
            end_effector_y = p[1]

            #get pose of the object and the corridor
        

            #x and y is defined according to the robots coordinates
            y= np.array([end_effector_y, end_effector_y ]) #it must be in increasing order
            x= np.array([ end_effector_x, bottom_obj_center_x+0.025])
            cs = self.create_curve(x, y)
            xs = np.linspace(x[0], x[-1], num=4) #sample from y
            
            sampled_curve = cs(xs)
        else:
            print('No need for correction')
            return


        for idx, sampled_y in enumerate(sampled_curve):
 
            q, q9, q10 = self.request_q(tix)

            itcnt = DEF_itcnt 
            p_des = np.array([ xs[idx],sampled_y, p[-1]])

            R_des = R


            #print ('q10 is ',q10)
            #if live==0: print "setpose R_des:\n",R_des
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        
            if q is None:
                self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
                return
            if not solved:
                if live==0 or itcnt>=50:
                    self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
                if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
            else:
                self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                #print q

            if solved or live==1:
                self.request_setdesq(tix,q[2:9])
                self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        time.sleep(4)

        sampled_curve = np.flipud(sampled_curve)
        xs = np.flipud(xs)


        for idx, sampled_y in enumerate(sampled_curve):
            print(sampled_y, xs[idx])
 
            q, q9, q10 = self.request_q(tix)

            itcnt = DEF_itcnt 
            p_des = np.array([ xs[idx],sampled_y, p[-1]])

            R_des = R


            #print ('q10 is ',q10)
            #if live==0: print "setpose R_des:\n",R_des
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        
            if q is None:
                self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
                return
            if not solved:
                if live==0 or itcnt>=50:
                    self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
                if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
            else:
                self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                #print q

            if solved or live==1:
                self.request_setdesq(tix,q[2:9])
                self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        time.sleep(4)

        

    def request_first_push(self, tix, args, mode=0, live=0, IK_qinit_current=False):
        
        K = len(args)
        DEF_itcnt = 1000
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        self.kin.set_torso_for_ik(qtorso)
        q10 = None
        

        # we have our desired position, we need current position, and object center position. 
        # after that, we will apply cubic spline fit and get sampled points. then we will apply ik to each one of them.

        #current pose of the rarm
        p, R, q_rad = self.request_pose_q(tix)

        #get pose of the object and the corridor
        gms = self.gms_client()

        end_effector_x = p[0]
        end_effector_y = p[1]

        '''
        red_object = gms('wood_cube_7_5cm_clone_1','ground_plane')
        obj_center_x = red_object.pose.position.x
        obj_center_y = red_object.pose.position.y
        '''

        bottom_obj = gms('wood_cube_7_5cm','ground_plane')
        bottom_obj_center_x = bottom_obj.pose.position.x
        bottom_obj_center_y = bottom_obj.pose.position.y 

        #x and y is defined according to the robots coordinates
        y= np.array([end_effector_y, end_effector_y ]) #it must be in increasing order
        x= np.array([end_effector_x, 0.525 ])
        cs = self.create_curve(x, y)
        xs = np.linspace(x[0], x[-1], num=4) #sample from y
        
        sampled_curve = cs(xs)


        for idx, sampled_y in enumerate(sampled_curve):
            print(sampled_y, xs[idx])
 
            q, q9, q10 = self.request_q(tix)

            itcnt = DEF_itcnt 
            p_des = np.array([ xs[idx],sampled_y, p[-1]])

            R_des = R


            #print ('q10 is ',q10)
            #if live==0: print "setpose R_des:\n",R_des
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        
            if q is None:
                self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
                return
            if not solved:
                if live==0 or itcnt>=50:
                    self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
                if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
            else:
                self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                #print q

            if solved or live==1:
                self.request_setdesq(tix,q[2:9])
                self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        time.sleep(4)

        sampled_curve = np.flipud(sampled_curve)
        xs = np.flipud(xs)

        for idx, sampled_y in enumerate(sampled_curve):
 
            q, q9, q10 = self.request_q(tix)

            itcnt = DEF_itcnt 
            p_des = np.array([ xs[idx],sampled_y, p[-1]])

            R_des = R


            #print ('q10 is ',q10)
            #if live==0: print "setpose R_des:\n",R_des
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        
            if q is None:
                self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
                return
            if not solved:
                if live==0 or itcnt>=50:
                    self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
                if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
            else:
                self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                #print q

            if solved or live==1:
                self.request_setdesq(tix,q[2:9])
                self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        time.sleep(4)

    def request_second_push(self, tix, args, mode=0, live=0, IK_qinit_current=False):
        #self.request_setdesq(TOR._ALL, TOR.q_tuba)
        #time.sleep(2)

        K = len(args)
        DEF_itcnt = 1000
        qtorso = self.jcomm.getjointpos(TOR._TORSO)
        self.kin.set_torso_for_ik(qtorso)
        q10 = None
        itcnt = DEF_itcnt 
        #self.request_setdesq(TOR._ALL, TOR.q_tuba_2) instead, ik
        #time.sleep(4)

        gms = self.gms_client()

        bottom_obj = gms('wood_cube_7_5cm','ground_plane')
        bottom_obj_center_x = bottom_obj.pose.position.x
        bottom_obj_center_y = bottom_obj.pose.position.y

        red_object = gms('wood_cube_7_5cm_clone_1','ground_plane')
        obj_center_x = red_object.pose.position.x
        obj_center_y = red_object.pose.position.y

        q, q9, q10 = self.request_q(tix)

 
        p_des = np.array([ obj_center_x,-0.05, 1.07])

        des_z = np.array([0,1,0])
        des_y = np.array([1,0,0])
        des_x = np.cross(des_y,des_z)
        R_des = np.array([des_x, des_y, des_z]).T 

        q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        
        if q is None:
            self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
            return
        if not solved:
            if live==0 or itcnt>=50:
                self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
            if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
        else:
            self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
            #print q

        if solved or live==1:
            self.request_setdesq(tix,q[2:9])
            self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        time.sleep(4)

        #current pose of the rarm
        p, R, q_rad = self.request_pose_q(tix)

        end_effector_x = p[0]
        end_effector_y = p[1]

        #x and y is defined according to the robots coordinates
        y= np.array([end_effector_y, bottom_obj_center_y+0.02 ]) #it must be in increasing order
        x= np.array([end_effector_x, end_effector_x ])
        cs = self.create_curve(y, x)
        ys = np.linspace(y[0], y[-1], num=4) #sample from y
        sampled_curve = cs(ys)

        for idx, sampled_x in enumerate(sampled_curve):
            
 
            q, q9, q10 = self.request_q(tix)

            itcnt = DEF_itcnt 
            p_des = np.array([ sampled_x, ys[idx], p[-1]])

            R_des = R


            #print ('q10 is ',q10)
            #if live==0: print "setpose R_des:\n",R_des
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        
            if q is None:
                self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
                return
            if not solved:
                if live==0 or itcnt>=50:
                    self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
                if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
            else:
                self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                #print q

            if solved or live==1:
                self.request_setdesq(tix,q[2:9])
                self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        time.sleep(4)

        sampled_curve = np.flipud(sampled_curve)
        ys = np.flipud(ys)

        for idx, sampled_x in enumerate(sampled_curve):
 
            q, q9, q10 = self.request_q(tix)

            itcnt = DEF_itcnt 
            p_des = np.array([ sampled_x,ys[idx], p[-1]])

            R_des = R


            #print ('q10 is ',q10)
            #if live==0: print "setpose R_des:\n",R_des
            q, solved, errL, itused = self.kin.ik(tix, p_des, R_des, qinit = q10, maxit=itcnt, IKstep_show = (live==1))
        
            if q is None:
                self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
                return
            if not solved:
                if live==0 or itcnt>=50:
                    self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
                if live==0: print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
            else:
                self.pprint("Solved in %d iters.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(itused, errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
                #print q

            if solved or live==1:
                self.request_setdesq(tix,q[2:9])
                self.request_setdesq(TOR._TORSO, q[0:2])  # torso IK result must be sent if IK included it!!
        time.sleep(4)



        self.request_setdesq(TOR._ALL, TOR.q_tuba)
        time.sleep(4)

        
        # check if R_des normalization is OK in IK

    def gms_client(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            #resp1 = gms(model_name,relative_entity_name) #wood_cube_7_5cm' pose
            return gms
        except rospy.ServiceException as e:
            print "Service call failed: %s"%e

    def create_curve(self, x,y):
        cs = CubicSpline(x, y)
        return cs

    # Before chunked continuous IK is started this must be called 
    def request_init_realtime(self, tix):
        self.ik_p_init, self.ik_R_init, self.ik_q10_init = self.request_pose_q(tix)
        print "Initializing chunked IK for arm:%s"%TOR._TARGET_L[tix]

    # check if R_des normalization is OK in IK
    def request_setDposFori(self, tix, del_pos, R_des=[]):
        modestr=['fullIK', 'locationIK', 'orientationIK']
        K = len(del_pos)
        itcnt = 1
        p_cur, R_cur, q10_cur = self.request_pose_q(tix)
        self.kin.set_torso_for_ik(q10_cur[0:2])          # let IK system now the torso

        if len(R_des)==0:
            R_des = R_cur
        else:
            p_des = p_cur + np.array(del_pos)   


        #self.pprint("Solving %s with max %d iterations"%(modestr[mode], itcnt))
        q, solved, errL,itused = self.kin.ik(tix, p_des, R_des, qinit = q10_cur, maxit=itcnt)

        if q is None:
            self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
            return
        if not solved:
            self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
            print ("unsolved q:",TOR.vec2str(q,"%2.3f"))
        else:
            self.pprint("Solved.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
            #print q

        self.request_setdesq(tix,q[2:9])
        return q, solved, errL


    # This function is not used   (used by the game canvas mouse IK control)
    def request_setpose_realtime(self, tix, args, mode=0):
        modestr=['fullIK', 'locationIK', 'orientationIK']
        K = len(args)
        DEF_itcnt = 10
        #qtorso = self.jcomm.getjointpos(TOR._TORSO)
        #self.kin.set_torso_for_ik(qtorso)    

        if mode == TOR.ik_mode_BOTH:       # request full IK
            if len(args)!=9 and len(args)!=10:
                self.pprint("ERROR: three vectors are needed as argument: position, Zaxis, Yaxis [maxit]")
                return
            #itcnt = DEF_itcnt*(K==9) + (K==10)*args[9]
            itcnt = DEF_itcnt if K==9 else args[9]
            p_des = np.array(args[0:3])
            des_z = np.array(args[3:6])
            des_y = np.array(args[6:9])
            des_x = np.cross(des_y,des_z)
            R_des = np.array([des_x, des_y, des_z]).T
        elif mode == TOR.ik_mode_POS:       # requesting location 
            if len(args)!=3 and len(args)!=4:
                self.pprint("ERROR: one vector are needed as argument: position [maxit]")
                return
            itcnt = DEF_itcnt if K==3 else args[3]
            R_des = self.ik_R_init              # use ori at the start of chunked ik
            p_des = np.array(args[0:3])   
        elif mode == TOR.ik_mode_ORI:           # requesting orientation
            if len(args)!=6 and len(args)!=7:
                self.pprint("ERROR: two vectors are needed as argument: Z_axis, Y_axis [maxit]")
                return
            itcnt = DEF_itcnt if K==6 else args[6]
            p_des = self.ik_p_init              # use current pos  at the start of chunked ik
            des_z = np.array(args[0:3])
            des_y = np.array(args[3:6])
            des_x = np.cross(des_y,des_z)
            R_des = np.array([des_x, des_y, des_z]).T
        else:
            self.pprint("ERROR: Use 0,1 or 2 for the IK mode (corresponding to fullIK, locationIK, orientationIK")
            return 


        # make a few ik steps    
        q, solved, errL ,itused = self.kin.ik(tix, p_des, R_des, self.ik_q10_init, maxit=itcnt)
        
         
        if q is None:
            self.pprint("No kinematics setup for this body part or bad arguments (is R_des a valid rotation matrix?)!");
            return
        if not solved:  # we are not expecting to solve but for progress indicator text output can be used
             #self.pprint("Sorry, could not reach the desired accuracy in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(itcnt,errL[0]*1000,errL[1], errL[2]))          
             #self.pprint("unsolved q:%s"%TOR.vec2str(q,"%2.3f"))
            dummy = 1
        else:
            #self.pprint("Solved.  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)  [q = %s degrees]\n"%(errL[0]*1000,errL[1], errL[2], TOR.vec2str(q*180/np.pi,"%3.2f")))
            dummy = 0
        #print "updating self.ik_q10_init which was:",self.ik_q10_init, "with q:", q
        self.ik_q10_init[:] = q[:]
        self.request_setdesq(tix,q[2:9])       # send the solution so far
    
        
    def linear_motion(self, tix, args):
        if tix==TOR._ALL:
            self.pprint('ERROR: No linear motion for ALL body parts\n');
            return
        if len(args)!=3:
            self.pprint('ERROR: linear motion requires 3 parameters (target location)\n');
            return
        p, R_des, q10_init = self.request_pose_q(tix)
        p_tar = np.array(args[0:3])
        STEPS = 200
        p_des = p.copy()
        ik_itcnt = 10
        print p, p_tar
        delta = (p_tar - p)/STEPS
        #q_init, q9_init, q10_init = self.request_q(tix)   # should not be needed
        for k in range(1,STEPS):
            
            p_des = p_des + delta
            
            q, solved, errL,itused = self.kin.ik(tix, p_des, R_des, qinit=q10_init, maxit=ik_itcnt)
            q10_init[:] = q[:]   # for next IK step
            if not solved:
                nop=0
                #self.pprint("torcomm: accuracy not met in %d iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(ik_itcnt,errL[0]*1000,errL[1], errL[2]))          
                #self.pprint ("torcomm: unsolved q:%s (deg)"%TOR.vec2str(q*180/np.pi,"%3.2f"))
            else:
                self.pprint("** torcomm: Solved  (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(errL[0]*1000,errL[1], errL[2]))
            #print q
            self.request_setdesq(tix,q[2:9])
        # correct for the remaining error with possibly not linear motion    
        if not solved:
            self.pprint("torcomm: Accumulated accuracy %d (steps) x %d (Ik_iter) iterations: [pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f]\n"%(STEPS, ik_itcnt,errL[0]*1000,errL[1], errL[2]))
            self.pprint("torcomm: Trying to correct for the final error...")
            cor_itcnt = 1000
            q, corsolved, errL,itused = self.kin.ik(tix, p_des, R_des, qinit=q10_init, maxit=cor_itcnt)
            if corsolved:
                self.pprint("torcomm: Correction OK   (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(errL[0]*1000,errL[1], errL[2]))
            else:
                self.pprint("torcomm: Correction cannot be fully done. Final accuracy: in %d IK iterations (pos_err:%2.3fmm, zax_err:%2.3f, yax_err:%2.3f)\n"%(cor_itcnt,errL[0]*1000,errL[1], errL[2]))                

            self.request_setdesq(tix,q[2:9])
       
    def request_setdesq(self, tix, args):
        #print "REQUEST_SETDESQ"
        #print tix,'(',TOR._TARGET_L[tix],')', args
        if tix==TOR._ALL:
            if len(args)!= len(TOR.q_syshome):
                self.pprint("ERROR: [probably be not implemented!] All group index requires %d arguments"%len(TOR.q_syshome))
                return
            for k in range(0,len(args)):
                if TOR._NUMJNT_L[k] != len(args[k]):
                    self.pprint("ERROR: Joint group [%s] equires %d arguments (in setqT commands 2 more!)"%(TOR._TARGET_L[tix],TOR._NUMJNT_L[k]) )
                    return
        else:
         if TOR._NUMJNT_L[tix] != len(args):
                self.pprint("ERROR: Joint group [%s] requires %d arguments (in setqT commands 2 more!)"%(TOR._TARGET_L[tix],TOR._NUMJNT_L[tix]) )
                return

        self.ezcon.setdesq(tix, args)    # now ask the ezcon to update the reference angles 
 
    def request_setdesqT(self, tix, args):
        ##print "REQUEST_SETDESQT"
        ##print tix, args
        if tix==TOR._ALL:
            if len(args)!= len(TOR.q_syshome):
                self.pprint("ERROR: [probably be not implemented!] All group index requires %d arguments."%len(TOR.q_syshome))
                return
            for k in range(0,len(args)):
                if TOR._NUMJNT_L[k] != len(args[k]):
                    self.pprint("ERROR: Joint group [%s] equires %d arguments, sent %d. "%(TOR._TARGET_L[tix],TOR._NUMJNT_L[k]+2,len(args[k])) )
                    return
        else:
         if TOR._NUMJNT_L[tix]+2 != len(args):
                self.pprint("ERROR: Joint group [%s] requires %d arguments, sent %d "%(TOR._TARGET_L[tix],TOR._NUMJNT_L[tix]+2, len(args)) )
                return

        self.ezcon.setdesq(TOR._TORSO, args[0:2])
        self.ezcon.setdesq(tix, args[2:])  


    

def shutdown_func():
    global gazerhook
    global ucom
    gazer.quit()
    ucom.closeUDP()
    print TOR.col.WARNING+TOR.col.BOLD+"\n*** Shutdown requested ***"+TOR.col.ENDC+" [May take time to complete. Check with ps -ef | grep torcomm.py]\n"

def collect_data_corridor_move():
    ucom.set_gaze([0.45, 0.2, 1.07])
    rospy.sleep(1.5)
    ucom.request_setdesq(TOR._ALL, TOR.q_tuba)
    
    if learner_data_collection:
        datacollectorTh = Thread(target=learner.data_collection)
        datacollectorTh.start()
        learner.learner_loop()
        
def test_learning_corridor_move(ucom):
    ucom.set_gaze([0.45, 0.2, 1.07])
    rospy.sleep(1.5)
    ucom.request_setdesq(TOR._ALL, TOR.q_tuba)
    
    learner.ae_cartesian_loop(p1=2, p2=3, p3=3) #pn is the prediction size 2: xy 3:xyz, choose according to the model used
        #learner.cartesian_loop(p1=2, p2=3, p3=3)
        
def com_main():
    global gazer
    global ucom
    lup  = LOOKUPTRAJ(TOR.CORRECTION_FOLDER)       # Create a lookup table
    rospy.init_node('erhcommNODE', anonymous=False)   # Only one controller allowed
    rospy.on_shutdown(shutdown_func)
    latvec = LatVecRec()    
    rostor  = rosTOR()                # Encapsulate robot trajectory commands
    jcom  = JointComm()             # Start the joint angle reading trhread
    rospy.sleep(0.25)               # Give some time to jcom to receive joint angles from ros (controller should be ok without this too but...)
    ezcon = controller.ezController(jcom, rostor) # Run a basic controller   
    kin  = TorKin()           # Create the inverse kinematics solver
    blob = blobLogger()       # Node: torvis.py must be running as a rosnode
    
    ucom = UserComm(jcom, ezcon, kin, blob, latvec, lup)   # Listen user commands
    gazer = Gazer(ucom)
    #learner = LearningLoop(ucom)
  
    #print (TOR.col.WARNING+"TODO: setq from cli make the gripper momentariliy close and thus get them stuck for a release request!")
    #print ("Does this happen during teleop release? ")
    #print ("Implement enclose and release commands from CLI"+TOR.col.ENDC)
    #print ("Write your steps to make the haptic device work. Remember the -lncurses problem. A simple symlink solved the problem. see ls -lrt /lib/x86*/")

    # If need to start from a fixed posture:
    #rospy.sleep(0.25) 
    #ucom.request_setdesq(TOR._ALL, TOR.q_ezhome)
    #ucom.request_setdesq(TOR._HEAD, np.array([-10.0, 37])*np.pi/180)
   
    ucom.touch.setcontrol_mode(3)     # all and not relative (teleop mode)
        
    rospy.spin()              # this will keep all the threads running (main() will not exit.)

if __name__ == '__main__':
    #print("**** PLEASE RUN THIS VIA torstart.py so that parallel module definitions can be imported ****");
   
    com_main()




    
