# ~/ros2_ws/src/pwm_control/setup.py
from setuptools import setup

package_name = 'pwm_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # 폴더 pwm_control 를 패키지로 설치
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pwm_pub.launch.py', 'launch/cmdvel.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nano2',
    maintainer_email='nano2@todo.todo',
    description='PWM control node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # entry point 이름 = pwm_pub, 실제 함수 = pwm_control/pwm_pub.py 의 main()
            'pwm_pub = pwm_control.pwm_pub:main',
            'cmdvel = pwm_control.cmdvel:main'
        ],
    },
)
