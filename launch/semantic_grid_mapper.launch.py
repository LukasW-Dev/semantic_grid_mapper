from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    
    config_path = os.path.expanduser('~/ros2_ws/src/semantic_grid_mapper/config/grid_mapper_params.yaml')
    filter_path = os.path.expanduser('~/ros2_ws/src/semantic_grid_mapper/config/filter.yaml')

    return LaunchDescription([
        Node(
            package='semantic_grid_mapper',
            executable='semantic_grid_mapper',
            name='semantic_grid_mapper_node',
            output='screen',
            parameters=[config_path],
        )
    ])
