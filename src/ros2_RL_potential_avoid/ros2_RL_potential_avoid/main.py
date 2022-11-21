import rclpy
from rclpy.node import Node
import threading

from .environment import Environment

def main(args=None):
    rclpy.init(args=args)
    Env = Environment()

    rl_running_thread = threading.Thread(target=Env.run)
    rl_running_thread.start()
    
    rclpy.spin(Env)
    

if __name__ == "__main__":
    main()