#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import yaml
import importlib.util

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class RosbagToLeRobot:
    def __init__(self, bag_dir, output_dir, config_path, target_freq=None):
        self.bag_path = Path(bag_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_freq = target_freq

        # Find all episode directories (containing .db3 and metadata.yaml)
        self.episode_dirs = sorted(p.parent for p in self.bag_path.rglob('*.db3'))
        if not self.episode_dirs:
            raise FileNotFoundError(f"No rosbag2 episode directories found in {bag_dir}")

        print(f"Found {len(self.episode_dirs)} episode directories")
        for ep_dir in self.episode_dirs:
            print(f"  - {ep_dir}")

        if config_path.endswith('.py'):
            config = load_config_py(config_path)
            features = config.features
            robot_type = getattr(config, "robot_type", "crane_plus")
            repo_id = getattr(config, "repo_id", "teleop_rosbag_dataset")
            task_name = getattr(config, "task_name", "teleop")
            topics = config
        else:
            # YAML fallback
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            features = config['features']
            robot_type = config.get('robot_type', 'crane_plus')
            repo_id = config.get('repo_id', 'teleop_rosbag_dataset')
            task_name = config.get('task_name', 'teleop')
            topics = argparse.Namespace(**config)

        self.task_name = task_name
        # Topic names default to CRANE+; a features config can override them (see so101_features.py).
        self.state_topic = getattr(topics, 'state_topic', '/joint_states')
        self.state_joint_names = getattr(topics, 'state_joint_names', None)  # None: keep /joint_states order
        self.arm_topic = getattr(topics, 'arm_command_topic', '/crane_plus_arm_controller/joint_trajectory')
        self.gripper_topic = getattr(topics, 'gripper_command_topic', '/crane_plus_gripper_controller/joint_trajectory')
        self.env_topic = getattr(topics, 'environment_state_topic', None)  # None: copy of the joint state

        # --- Dataset creation with user prompt if dir exists ---
        while True:
            try:
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=target_freq or 20,
                    root=self.output_dir,
                    robot_type=robot_type,
                    features=features,
                    use_videos=True,
                )
                break
            except FileExistsError:
                print(f"[Warning] Directory {self.output_dir} already exists.")
                ans = input("Overwrite? (y: overwrite, n: specify new directory name): ").strip().lower()
                if ans == 'y':
                    # Overwrite: Delete the existing directory and recreate it
                    import shutil
                    shutil.rmtree(self.output_dir)
                    print(f"{self.output_dir} deleted. Recreating...")
                    continue
                else:
                    new_dir = input("Enter new directory name: ").strip()
                    self.output_dir = self.output_dir.parent / new_dir
                    print(f"New output directory: {self.output_dir}")
                    continue

        if "teleop" not in self.dataset.meta.task_to_task_index:
            self.dataset.meta.add_task("teleop")

    def _get_messages(self, episode_dir):
        """Get messages from all required topics in the rosbag."""
        messages = {self.state_topic: [], self.arm_topic: [], self.gripper_topic: []}
        if self.env_topic:
            messages[self.env_topic] = []
        with Reader(episode_dir) as reader:
            connections = [x for x in reader.connections if x.topic in messages.keys()]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                messages[connection.topic].append((msg, timestamp))
        return messages

    def _sample_and_hold(self, times, values, target_times):
        values = np.asarray(values)
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
            new_values = np.zeros((len(target_times), values.shape[1]))
        elif len(values.shape) >= 3:
            new_values = np.zeros((len(target_times), *values.shape[1:]), dtype=values.dtype)
        else:
            new_values = np.zeros((len(target_times), values.shape[1]))
        for i, target_time in enumerate(target_times):
            idx = np.searchsorted(times, target_time, side='right') - 1
            if idx < 0:
                idx = 0
            new_values[i] = values[idx]
        if len(values.shape) == 1:
            new_values = new_values.flatten()
        return new_values

    def convert(self):
        max_frames = 0
        episode_lengths = []
        print("\nFirst pass: calculating episode lengths...")
        for episode_dir in self.episode_dirs:
            try:
                messages = self._get_messages(episode_dir)
                j_times = np.array([ts for _, ts in messages[self.state_topic]])
                a_times = np.array([ts for _, ts in messages[self.arm_topic]])
                g_times = np.array([ts for _, ts in messages[self.gripper_topic]])
                # Use the intersection of available times
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                target_freq = self.target_freq or 20
                num_frames = int((end_time - start_time) * target_freq / 1e9)
                max_frames = max(max_frames, num_frames)
                episode_lengths.append(num_frames)
                print(f"Episode {episode_dir}: {num_frames} frames")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                episode_lengths.append(0)
                continue

        print(f"\nLongest episode: {max_frames} frames")

        for idx, episode_dir in enumerate(self.episode_dirs):
            print(f"\nProcessing episode {idx} from {episode_dir}...")
            try:
                messages = self._get_messages(episode_dir)
                joint_states = messages[self.state_topic]
                arm_cmds = messages[self.arm_topic]
                gripper_cmds = messages[self.gripper_topic]
                j_times = np.array([ts for _, ts in joint_states])
                if self.state_joint_names:
                    j_pos = np.array([[dict(zip(msg.name, msg.position))[n] for n in self.state_joint_names]
                                      for msg, _ in joint_states])
                else:
                    j_pos = np.array([msg.position for msg, _ in joint_states])
                a_times = np.array([ts for _, ts in arm_cmds])
                a_pos = np.array([msg.points[0].positions if msg.points else [0,0,0,0] for msg, _ in arm_cmds])
                g_times = np.array([ts for _, ts in gripper_cmds])
                g_pos = np.array([msg.points[0].positions[0] if (msg.points and len(msg.points[0].positions)>0) else 0.0 for msg, _ in gripper_cmds])
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                target_freq = self.target_freq or 20
                num_frames = episode_lengths[idx]
                target_times = np.linspace(start_time, end_time, num_frames)
                # Sample and hold for each
                j_pos_sampled = self._sample_and_hold(j_times, j_pos, target_times)  # (num_frames, 5)
                a_pos_sampled = self._sample_and_hold(a_times, a_pos, target_times)  # (num_frames, 4)
                g_pos_sampled = self._sample_and_hold(g_times, g_pos, target_times)  # (num_frames,)
                if self.env_topic:
                    e_msgs = messages[self.env_topic]
                    e_times = np.array([ts for _, ts in e_msgs])
                    e_pos = np.array([[msg.point.x, msg.point.y, msg.point.z] for msg, _ in e_msgs])
                    env_sampled = self._sample_and_hold(e_times, e_pos, target_times)
                else:
                    env_sampled = j_pos_sampled
                # Concatenate after sampling
                a_pos_full = np.hstack([a_pos_sampled, g_pos_sampled.reshape(-1,1)])  # (num_frames, 5)
                for i in range(max_frames):
                    idx_to_use = min(i, num_frames - 1)
                    self.dataset.add_frame({
                        "observation.state": torch.tensor(j_pos_sampled[idx_to_use], dtype=torch.float32),
                        "observation.environment_state": torch.tensor(env_sampled[idx_to_use], dtype=torch.float32),
                        "action": torch.tensor(a_pos_full[idx_to_use], dtype=torch.float32),
                        "task": self.task_name,
                    })
                print(f"Saving episode {idx}...")
                self.dataset.save_episode()
                print(f"Episode {idx} saved successfully")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return str(self.dataset.root)

def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/conversion_config.yaml')
    args = parser.parse_args()
    # Load parameters from conversion_config.yaml
    with open(args.config, "r") as f:
        conv_cfg = yaml.safe_load(f)
    bag_dir = conv_cfg["bag_dir"]
    output_dir = conv_cfg["output_dir"]
    config = conv_cfg["config"]
    target_freq = conv_cfg.get("target_freq", 20)

    converter = RosbagToLeRobot(bag_dir, output_dir, config, target_freq)
    output_path = converter.convert()
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    main() 