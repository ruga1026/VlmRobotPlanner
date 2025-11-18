from setuptools import setup
from glob import glob
import os

package_name = 'vlm_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools', 'rclpy', 'ollama'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='VLM node packaged for ROS 2 Humble (Python)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # executable name  =  python_module.function
            'vlm_node = vlm_node.vlm_node:main',
            'test = test.test:main',
        ],
    },
)

