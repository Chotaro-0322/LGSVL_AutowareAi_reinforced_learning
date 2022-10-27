from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ros1_bridge",
            node_executable="dynamic_bridge",
            output = "screen",
        ),
        Node(
            package="ros2_RL_calculator",
            node_executable="rl_calculator",
            output="screen",
        )
    ])