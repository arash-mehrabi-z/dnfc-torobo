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

        rospy.Subscriber(self.state_topic, ToroboJointState, callback=self.call_func)

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