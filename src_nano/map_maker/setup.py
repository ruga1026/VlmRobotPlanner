import os
from glob import glob
from setuptools import setup

package_name = 'map_maker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    entry_points={
        'console_scripts': [
            'map_maker = map_maker.map_maker:main',
        ],
    },
)