import rospy
from torobo_msgs.msg import ToroboJointState
from std_msgs.msg import String
import socket
from threading import Lock,Thread

class Comm():
    def __init__(self):
        self.joint_state = None
        self.jsLock = Lock()
        self.state_topic = '/torobo/right_arm_controller/torobo_joint_state'
        self.counter = 0

        # Head and torso state
        self.head_state = None
        self.torso_state = None
        self.headLock = Lock()
        self.torsoLock = Lock()
        self.head_topic = '/torobo/head_controller/torobo_joint_state'
        self.torso_topic = '/torobo/torso_controller/torobo_joint_state'

        rospy.Subscriber(self.state_topic, ToroboJointState, callback=self.call_func)
        rospy.Subscriber(self.head_topic, ToroboJointState, callback=self.head_call_func)
        rospy.Subscriber(self.torso_topic, ToroboJointState, callback=self.torso_call_func)

        targetIP = "localhost"
        targetPort = 50000
        self.server = (targetIP, targetPort)
        self.udpClientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def call_func(self, data):
        self.jsLock.acquire()
        self.joint_state = data.position
        # print(self.joint_state)
        self.counter += 1
        self.jsLock.release()

    def head_call_func(self, data):
        self.headLock.acquire()
        self.head_state = data.position
        self.headLock.release()

    def torso_call_func(self, data):
        self.torsoLock.acquire()
        self.torso_state = data.position
        self.torsoLock.release()

    def get_head_state(self):
        """Get current head joint positions."""
        self.headLock.acquire()
        state = list(self.head_state) if self.head_state is not None else None
        self.headLock.release()
        return state

    def get_torso_state(self):
        """Get current torso joint positions."""
        self.torsoLock.acquire()
        state = list(self.torso_state) if self.torso_state is not None else None
        self.torsoLock.release()
        return state



    def create_and_pub_msg(self, positions):

        if type(positions) != list:
            positions = positions.tolist()
        command = 'setq rarm'
        for i in positions:
            command = command + ' ' + str(i)
        encmess = command.encode()
        self.udpClientSocket.sendto(encmess, self.server)
        # print(command)

    def move(self, point_name, positions):

        if type(positions) != list:
            positions = positions.tolist()
        command = 'move_point rarm ' + point_name
        for i in positions:
            command = command + ' ' + str(i)
        encmess = command.encode()
        self.udpClientSocket.sendto(encmess, self.server)

    def which(self,com):

        command = ' ' +com

        encmess = command.encode()
        self.udpClientSocket.sendto(encmess, self.server)

    def set_torso(self, positions):
        """Set torso joint positions."""
        if type(positions) != list:
            positions = positions.tolist()
        command = 'setq torso'
        for i in positions:
            command = command + ' ' + str(i)
        encmess = command.encode()
        self.udpClientSocket.sendto(encmess, self.server)

    def set_head(self, positions):
        """Set head joint positions (pan, tilt)."""
        if type(positions) != list:
            positions = positions.tolist()
        command = 'setq head'
        for i in positions:
            command = command + ' ' + str(i)
        encmess = command.encode()
        self.udpClientSocket.sendto(encmess, self.server)