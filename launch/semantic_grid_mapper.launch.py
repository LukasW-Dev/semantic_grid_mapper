from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='semantic_grid_mapper',
            executable='semantic_grid_mapper',
            name='semantic_grid_mapper_node',
            output='screen',
            parameters=['/home/robolab/ros2_ws/src/semantic_grid_mapper/config/grid_mapper_params.yaml']
        )
    ])
