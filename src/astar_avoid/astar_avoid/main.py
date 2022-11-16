import rclpy
from rclpy.node import Node
import threading

from .astar import Astar_avoid

def main(args=None):
    rclpy.init(args=args)
    Astar = Astar_avoid()

    running_thread = threading.Thread(target=Astar.run)
    running_thread.start()
    
    rclpy.spin(Astar)
    

if __name__ == "__main__":
    main()