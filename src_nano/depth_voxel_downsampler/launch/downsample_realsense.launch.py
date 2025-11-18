#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    node = Node(
        package='depth_voxel_downsampler',
        executable='voxel_downsampler_node',
        name='voxel_downsampler',
        output='screen',
        parameters=[{
            'input_topic': '/camera/depth/color/points',
            'output_topic': '/depth/points_downsampled',                 # Best-Effort
            'output_topic_reliable': '/depth/points_downsampled_reliable', # Reliable 복제
            'publish_reliable_duplicate': True,   # RViz가 Reliable만 쓴다면 True 유지
            'leaf_size': 0.05,
            'z_min': 0.10,
            'z_max': 4.50,
            'drop_rgb': True,
            'remove_nan': True,
            'rate_limit_hz': 15.0,
            'qos_depth': 5,
        }]
    )
    return LaunchDescription([node])
