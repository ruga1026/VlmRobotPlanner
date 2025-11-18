from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['package.xml', 'yoloe-11s-seg.pt']),
        ('share/' + package_name, ['package.xml', 'yoloe-11m-seg.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nano2',
    maintainer_email='nano2@todo.todo',
    description='YOLO with RealSense ROS2 node',
    license='Oh',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image2yolo = yolo.image2yolo:main',
            'marker = yolo.marker:main',
            'map2pgm = yolo.map2pgm:main',
            'similar_word = yolo.similar_word:main',
            'marker_json_pub = yolo.marker_json_pub:main',
        ],
    },
)