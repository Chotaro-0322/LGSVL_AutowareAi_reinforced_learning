from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class RL_processing(Node):
    def __init__(self):
        super().__init__("rl_processing")

        self.current_pose_sub = self.create_subscription(PoseStamped, "current_pose", self.current_poseCallback, 1)

        GAMMA = 0.99
        MAX_STEP = 200
        NUM_EPISODES = 500

    def current_poseCallback(self, msg):
        current_pose = msg.data
