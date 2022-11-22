from setuptools import setup

package_name = 'ros2_RL_potential_avoid'
subpackage = 'ros2_RL_potential_avoid.ros2_numpy.ros2_numpy'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, subpackage],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chohome',
    maintainer_email='mf21131@shibaura-it.ac.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'potential_avoid = ros2_RL_potential_avoid.main:main'
        ],
    },
)
