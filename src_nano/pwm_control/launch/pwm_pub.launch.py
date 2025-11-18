# pwm_pub.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pwm_control',     # ROS2 패키지 이름
            executable='pwm_pub',   # 설치된 실행 파일 또는 스크립트 이름
            name='pwm_control',
            output='screen'
            # parameters=[{'enable_pca9685': True, 'pwm_hz': 50}]  # 필요 시 파라미터 지정
        )
    ])
