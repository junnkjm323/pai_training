# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Tatsuya Kamijo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tatsuya Kamijo (based on HuggingFace LeRobot lerobot/examples/3_train_policy.py)

"""
Train ACT (Action Chunking with Transformer) policy on the collected dataset.

This script trains an ACT policy using configuration parameters specified in:
    - configs/train_config.yaml: Training hyperparameters like batch size, learning rate, etc.
    - configs/crane_features.py: Dataset and feature configuration

Usage:
    pixi run train
    pixi run train --config configs/so101_train_config.yaml

"""


from pathlib import Path
import argparse
import importlib.util
import wandb
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

def load_config_py(config_path: Path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def load_train_config(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# =====================
# Load parameters from YAML
# =====================
_parser = argparse.ArgumentParser()
_parser.add_argument("--config", default="configs/train_config.yaml")
train_cfg = load_train_config(_parser.parse_args().config)
DATASET_ROOT = Path(train_cfg["dataset_root"])
REPO_ID = train_cfg["repo_id"]
OUTPUT_DIRECTORY = Path(train_cfg["output_directory"])
CONFIG_PATH = Path(train_cfg["config_path"])
DEVICE = torch.device(train_cfg["device"])
TRAINING_STEPS = train_cfg["training_steps"]
LOG_FREQ = train_cfg["log_freq"]
NUM_WORKERS = train_cfg["num_workers"]
BATCH_SIZE = train_cfg["batch_size"]
CHUNK_SIZE = train_cfg["chunk_size"]
N_ACTION_STEPS = train_cfg["n_action_steps"]
LEARNING_RATE = train_cfg["learning_rate"]
DELTA_TIMESTAMPS = train_cfg["delta_timestamps"]
WANDB_PROJECT = train_cfg.get("wandb_project", "pai_crane_gazebo")
# =====================

def main():
    # Create a directory to store the training checkpoint.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    config = load_config_py(CONFIG_PATH)
    features = config.features

    wandb.init(project=WANDB_PROJECT, config={
        "repo_id": REPO_ID,
        "dataset_root": DATASET_ROOT,
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "device": str(DEVICE),
    })

    dataset_metadata = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features: {input_features}")
    print(f"output_features: {output_features}")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS
    )
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(DEVICE)

    # Instantiate the dataset
    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, delta_timestamps=DELTA_TIMESTAMPS)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    step = 0
    done = False
    losses = []
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # Remove the sequence_length dimension to adapt to the ACT model implementation
            batch["observation.state"] = batch["observation.state"][:, -1, :]  # (B, D)
            batch["observation.environment_state"] = batch["observation.environment_state"][:, -1, :]  # (B, D)
            # action is left as is
            loss, _ = policy.forward(batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
                wandb.log({"loss": loss.item(), "step": step})
            step += 1
            if step >= TRAINING_STEPS:
                done = True
                break

    policy.save_pretrained(OUTPUT_DIRECTORY)
    print(f"policy saved to {OUTPUT_DIRECTORY}")
    save_loss_curve(losses, OUTPUT_DIRECTORY)
    wandb.finish()

def save_loss_curve(losses, output_directory: Path):
    """Write loss.csv and loss.png (raw loss and moving average, log scale) next to the saved policy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(output_directory / "loss.csv", "w") as f:
        f.write("step,loss\n")
        f.writelines(f"{i},{v}\n" for i, v in enumerate(losses))
    window = max(1, len(losses) // 50)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses, alpha=0.3, linewidth=0.8, label="loss")
    ax.plot(np.arange(window - 1, len(losses)), smooth, linewidth=1.5, label=f"moving average ({window} steps)")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "loss.png", dpi=150)
    print(f"loss curve saved to {output_directory / 'loss.png'}")

if __name__ == "__main__":
    main()
