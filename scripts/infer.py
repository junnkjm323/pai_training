#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import argparse
import importlib.util
import numpy as np

from lerobot.common.policies.act.modeling_act import ACTPolicy

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

class InferenceNode(Node):
    def __init__(self, model_path, config=None):
        super().__init__('crane_plus_act_infer')
        # Robot-specific names default to CRANE+; a features config can override them (see so101_features.py).
        self.arm_joint_names = getattr(config, 'arm_joint_names', ARM_JOINT_NAMES)
        self.gripper_joint_name = getattr(config, 'gripper_joint_name', GRIPPER_JOINT_NAME)
        self.horizon = getattr(config, 'command_horizon', 0.1)
        n_arm = len(self.arm_joint_names)
        self.arm_pub = self.create_publisher(JointTrajectory, getattr(config, 'arm_command_topic', '/crane_plus_arm_controller/joint_trajectory'), 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, getattr(config, 'gripper_command_topic', '/crane_plus_gripper_controller/joint_trajectory'), 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        # Optional: environment_state from a PointStamped topic (reach target) instead of a joint state copy.
        self.latest_env_state = None
        env_topic = getattr(config, 'environment_state_topic', None)
        self.env_required = bool(env_topic)
        if env_topic:
            from geometry_msgs.msg import PointStamped
            self.create_subscription(PointStamped, env_topic, self.env_state_callback, 10)
        # Optional: only act while an episode is active, and reset the action queue at its start.
        self.active = True
        active_topic = getattr(config, 'active_topic', None)
        if active_topic:
            from std_msgs.msg import Bool
            self.active = False
            self.create_subscription(Bool, active_topic, self.active_callback, 10)
        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.eval()
        self.device = torch.device('cpu')
        self.policy.to(self.device)
        self.last_action = np.zeros(n_arm + 1, dtype=np.float32)
        self.timer = self.create_timer(0.05, self.publish_action)  # 20Hz
        self.latest_joint_state = None

    def joint_state_callback(self, msg):
        # Extract joint positions in the order of arm joint names + [gripper joint name]
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_pos = [name_to_pos.get(j, 0.0) for j in self.arm_joint_names]
        gripper_pos = name_to_pos.get(self.gripper_joint_name, 0.0)
        self.latest_joint_state = np.array(arm_pos + [gripper_pos], dtype=np.float32)

    def env_state_callback(self, msg):
        self.latest_env_state = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float32)

    def active_callback(self, msg):
        if msg.data and not self.active:
            self.policy.reset()
        self.active = msg.data

    def publish_action(self):
        if self.latest_joint_state is None or not self.active:
            return
        env_state = self.latest_joint_state
        if self.env_required:
            if self.latest_env_state is None:
                return
            env_state = self.latest_env_state
        # Prepare observation dict for ACTPolicy
        obs = {
            'observation.state': torch.tensor(self.latest_joint_state, dtype=torch.float32).unsqueeze(0),  # (1, D)
            'observation.environment_state': torch.tensor(env_state, dtype=torch.float32).unsqueeze(0),  # (1, E)
        }
        with torch.no_grad():
            action = self.policy.select_action(obs).cpu().numpy().flatten()  # (D,)
        self.last_action = action
        n_arm = len(self.arm_joint_names)
        # Publish to arm and gripper
        jt = JointTrajectory()
        jt.joint_names = self.arm_joint_names
        pt = JointTrajectoryPoint()
        pt.positions = action[:n_arm].tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(self.horizon * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)
        jt_g = JointTrajectory()
        jt_g.joint_names = [self.gripper_joint_name]
        pt_g = JointTrajectoryPoint()
        pt_g.positions = [float(action[n_arm])]
        pt_g.time_from_start.sec = 0
        pt_g.time_from_start.nanosec = int(self.horizon * 1e9)
        jt_g.points.append(pt_g)
        self.gripper_pub.publish(jt_g)
        print("action:", action[:n_arm], "gripper:", action[n_arm])

def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location("features_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    parser = argparse.ArgumentParser(description='ACTPolicy inference node for CRANE+ V2 (ROS2)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained ACTPolicy model directory')
    parser.add_argument('--config', type=str, default=None, help='Features config with robot topics/joints (e.g. configs/so101_features.py); CRANE+ if omitted')
    args = parser.parse_args()
    rclpy.init()
    node = InferenceNode(args.model_path, load_config_py(args.config) if args.config else None)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
