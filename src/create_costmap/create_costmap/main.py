import rclpy
from rclpy.node import Node
import threading

from .create_costmap import Create_costmap

def main(args=None):
    rclpy.init(args=args)
    Cost_G = Create_costmap()
    
    rclpy.spin(Cost_G)
    

if __name__ == "__main__":
    main()