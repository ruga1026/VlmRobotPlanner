
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='map_maker',
            executable='map_maker',
            name='map_maker',
            output='screen'
        ),
    ])
