JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": JOINT_NAMES,
    },
    # Reach target position (from /reach_target), not a copy of the joint state.
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["target_x", "target_y", "target_z"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": JOINT_NAMES,
    },
}
# Below is used for the dataset conversion
robot_type = "so101"
repo_id = "so101_reach"
root = "/home/ubuntu/pai2026-final-assignment/data/lerobot_dataset/so101_reach"
task_name = "reach"

# Topics and joint order (read by rosbag2lerobot.py and infer.py; CRANE+ defaults apply when absent)
state_joint_names = JOINT_NAMES  # reorder /joint_states by name
arm_command_topic = "/so101_arm_controller/joint_trajectory"
gripper_command_topic = "/so101_gripper_controller/joint_trajectory"
arm_joint_names = JOINT_NAMES[:5]
gripper_joint_name = "gripper"
environment_state_topic = "/reach_target"  # geometry_msgs/PointStamped
active_topic = "/reach_active"  # std_msgs/Bool from so101_reach.py; the policy only acts while true
command_horizon = 0.0  # time_from_start [s]; see so101_leader_bridge.py
