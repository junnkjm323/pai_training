"""Spawn SO-101 (follower) in Ignition Gazebo Fortress with ros2_control.

Mirrors crane_plus_gazebo: ign_ros2_control + JointTrajectoryController for arm/gripper
+ joint_state_broadcaster. The official URDF in SO-ARM100 is left untouched; it is
patched in memory (absolute mesh paths, base fixed to world, ros2_control tags).

Usage (inside the container):
    ros2 launch sim/so101/so101_gazebo.launch.py              # with GUI (run in noVNC)
    ros2 launch sim/so101/so101_gazebo.launch.py headless:=true
"""

import os
import xml.etree.ElementTree as ET

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
URDF_DIR = os.path.join(REPO_ROOT, 'SO-ARM100', 'Simulation', 'SO101')
URDF_PATH = os.path.join(URDF_DIR, 'so101_new_calib.urdf')
CONTROLLERS_YAML = os.path.join(HERE, 'so101_controllers.yaml')

ARM_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
GRIPPER_JOINT = 'gripper'


def build_robot_description():
    robot = ET.parse(URDF_PATH).getroot()

    # Relative "assets/xxx.stl" -> absolute file:// URI so Ignition can find the meshes.
    for mesh in robot.iter('mesh'):
        fn = mesh.get('filename')
        if not fn.startswith(('file://', 'package://')):
            mesh.set('filename', 'file://' + os.path.join(URDF_DIR, fn))

    # Fix base_link to the world.
    ET.SubElement(robot, 'link', name='world')
    fixed = ET.SubElement(robot, 'joint', name='world_joint', type='fixed')
    ET.SubElement(fixed, 'parent', link='world')
    ET.SubElement(fixed, 'child', link='base_link')
    ET.SubElement(fixed, 'origin', xyz='0 0 0', rpy='0 0 0')

    limits = {j.get('name'): j.find('limit') for j in robot.findall('joint')}
    ros2_control = ET.SubElement(robot, 'ros2_control', name='IgnitionSystem', type='system')
    hw = ET.SubElement(ros2_control, 'hardware')
    ET.SubElement(hw, 'plugin').text = 'ign_ros2_control/IgnitionSystem'
    for name in ARM_JOINTS + [GRIPPER_JOINT]:
        joint = ET.SubElement(ros2_control, 'joint', name=name)
        cmd = ET.SubElement(joint, 'command_interface', name='position')
        ET.SubElement(cmd, 'param', name='min').text = limits[name].get('lower')
        ET.SubElement(cmd, 'param', name='max').text = limits[name].get('upper')
        for iface in ('position', 'velocity', 'effort'):
            ET.SubElement(joint, 'state_interface', name=iface)

    gazebo = ET.SubElement(robot, 'gazebo')
    plugin = ET.SubElement(gazebo, 'plugin', filename='libign_ros2_control-system.so',
                           name='ign_ros2_control::IgnitionROS2ControlPlugin')
    ET.SubElement(plugin, 'parameters').text = CONTROLLERS_YAML

    return ET.tostring(robot, encoding='unicode')


def generate_launch_description():
    headless = LaunchConfiguration('headless')
    world = LaunchConfiguration('world')
    # Without this the ign_ros2_control system plugin is not found (same as crane_plus_gazebo).
    env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ.get('LD_LIBRARY_PATH', '')}

    # -r: start the simulation immediately, otherwise the controllers never come up.
    gazebo_gui = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world], output='screen',
        additional_env=env, condition=UnlessCondition(headless))
    gazebo_server = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', world], output='screen',
        additional_env=env, condition=IfCondition(headless))

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen', parameters=[{'robot_description': build_robot_description()}])

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', '/robot_description', '-name', 'so101', '-allow_renaming', 'true'])

    spawners = [
        Node(package='controller_manager', executable='spawner', output='screen',
             arguments=[name, '--controller-manager-timeout', '60'])
        for name in ('joint_state_broadcaster', 'so101_arm_controller', 'so101_gripper_controller')
    ]

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'])

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('world', default_value='empty.sdf'),
        SetParameter(name='use_sim_time', value=True),
        gazebo_gui,
        gazebo_server,
        robot_state_publisher,
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=spawners)),
        bridge,
    ])
