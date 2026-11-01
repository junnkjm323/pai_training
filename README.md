# Physical AI Imitation Learning excercise

## Features
- rosbag2lerobot: Dataset conversion from rosbag2 to LeRobotDataset v2.1 format
- training: Imitation Learning using LeRobot

## Installation
Clone the repo
```bash
cd && git clone --recursive https://github.com/matsuolab/pai_training
cd pai_training
```
Get [pixi](https://pixi.sh/latest/) and install all the dependencies
```bash
curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell bash)"' >> ~/.bashrc
source ~/.bashrc
```
```bash
pixi install
```

## Usage
Convert rosbag2 to LeRobotDataset v2.1 format
```bash
pixi run convert
```
Visualize the dataset
```bash
pixi run python scripts/visualize_dataset.py --root /home/ubuntu/dataset/lerobot_dataset/crane_plus_pekori --repo-id crane_plus_pekori --episode-index 1 --mode distant
# Copy the output from above command and paste it in your *local* web browser (it may take a second to load)
```
Train the policy
```bash
pixi run wandb login # login to wandb, access https://wandb.ai/authorize to get your API key
# Invoke training with the config in [configs/train_config.yaml](configs/train_config.yaml)
pixi run train
```
Run inference.  
It takes the model path as an argument.
```bash
pixi run infer --model-path /home/ubuntu/checkpoints/train/crane_plus_pekori_act
```
Debug policy.  
This script evaluates the policy on the specified dataset offline.  
This is useful to check if the policy is working as expected.  
You can separate the dataset into training and evaluation sets to evaluate the policy in more detail.
```bash
pixi run debug-policy --dataset-root /home/ubuntu/dataset/lerobot_dataset/crane_plus_pekori --repo-id crane_plus_pekori --policy-path /home/ubuntu/checkpoints/train/crane_plus_pekori_act --episode-index 1
```

## Pixi help
Visit [pixi](https://pixi.sh/latest/) for more details.

### Environment Management
```bash
# Cleanup environment
pixi clean

# Add dependencies
pixi list  # show all installed packages

# Package Management
# PyPI (pip) packages
pixi add --pypi package  # install
pixi remove --pypi package  # uninstall

# conda packages
pixi add package  # install
pixi remove package  # uninstall

# Activate environment
pixi shell  # enter pixi's virtual environment shell (like source env/bin/activate in venv)
# or
pixi run "..."  # run command in default shell
```

### Task Management
Tasks can be defined in `pyproject.toml` under `[tool.pixi.tasks]`:
```toml
[tool.pixi.tasks]
my-task = "python my_script.py"  # simple command
```

You can then run these tasks using:
```bash
pixi run my-task  #= pixi shell & python my_script.py
pixi run python my_script.py  # run a command in the pixi environment
```

## SO-101 (Gazebo) teleop, data collection, ACT
Teleoperate a simulated SO-101 in Ignition Gazebo with an SO-101 leader arm, record reach demonstrations with rosbag2,
replay them, convert them with `rosbag2lerobot.py`, train ACT with `train.py` and run the policy with `infer.py`.
Without a leader arm, a scripted demonstrator (`--demo`) produces the same leader packets.

```
leader arm --USB--> so101_leader_bridge.py send --UDP:5005--> so101_leader_bridge.py ros --> JointTrajectoryController (Gazebo)
                                                                                             ^ so101_reach.py: target, success, rosbag2
```

| Path | What |
|---|---|
| `sim/so101/so101_gazebo.launch.py`, `so101_controllers.yaml` | SO-101 (`SO-ARM100` submodule URDF) in Ignition Gazebo + ros2_control |
| `sim/so101/so101_leader_bridge.py` | `send` (LeRobot leader -> UDP), `fake` (sine waves), `ros` (UDP -> controllers) |
| `sim/so101/so101_reach.py` | reach task: `record [--demo]`, `replay`, `eval`, `fk` |
| `sim/so101/leader/pixi.toml` | separate env with LeRobot 0.4.1 for the leader (the default env keeps the old lerobot submodule) |
| `configs/so101_*` | features (state 6, target xyz 3, action 6) and conversion / training configs |
| `docker/Dockerfile` | course image (pinned by digest) + serial port access (`dialout` group) only |

### Environment
Linux host with Docker (the USB pass-through below does not work with Docker Desktop on macOS/Windows).
```bash
git clone --recursive https://github.com/junnkjm323/pai_training
cd pai_training
docker build -t pai2026-so101 docker/
# The parent directory is mounted so that data/ (bags, datasets, checkpoints) sits next to the repo.
docker run -d --name pai2026-so101 --shm-size=512m \
  -v "$(dirname "$PWD")":/home/ubuntu/pai2026-final-assignment \
  -p 127.0.0.1:6080:80 -p 127.0.0.1:5005:5005/udp -p 127.0.0.1:9090:9090 -p 127.0.0.1:9876:9876 \
  --device-cgroup-rule='c 166:* rmw' --device-cgroup-rule='c 188:* rmw' \
  -v /dev:/hostdev:ro \
  pai2026-so101
```
- Desktop: http://127.0.0.1:6080 (noVNC, password `ubuntu`). The leader shows up as `/hostdev/ttyACM*`, also when plugged in later;
  only ttyACM (166) / ttyUSB (188) character devices are allowed.
- Run every ROS node as the `ubuntu` user (e.g. the noVNC terminal, or `su - ubuntu`). With Fast DDS shared memory,
  messages from `ubuntu` nodes do not reach nodes started as `root`.

Inside the container, as `ubuntu`:
```bash
curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.83.0 bash   # the version this setup was tested with
source ~/.bashrc
cd ~/pai2026-final-assignment/pai_training
pixi install --frozen                        # training/inference env (old lerobot submodule)
(cd sim/so101/leader && pixi install --frozen)  # leader env (LeRobot 0.4.1)
source /opt/ros/humble/setup.bash
```

### Run
Each step in its own terminal, from `~/pai2026-final-assignment/pai_training` with ROS sourced.
```bash
ros2 launch sim/so101/so101_gazebo.launch.py          # Gazebo GUI (add headless:=true for no GUI)
python3 sim/so101/so101_leader_bridge.py ros          # UDP -> controllers
```
Leader arm (one-time setup, then streaming). The calibration is stored in `sim/so101/leader/calibration/`.
```bash
cd sim/so101/leader
pixi run lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/hostdev/ttyACM0
pixi run lerobot-calibrate --teleop.type=so101_leader --teleop.port=/hostdev/ttyACM0 --teleop.id=so101_leader
pixi run send --port /hostdev/ttyACM0                 # --signs/--offsets map LeRobot angles onto the URDF joints
# Direction check: run with --show-raw, move one leader joint at a time and compare with the Gazebo arm;
# flip that joint's entry in --signs if it moves the other way (both use "zero = middle of the range").
```
Record, replay, train, evaluate:
```bash
python3 sim/so101/so101_reach.py record --wall --episodes 30  # human with the leader: the console says which joint
                                                              # is off home; the bag starts when the leader starts moving
python3 sim/so101/so101_reach.py record --demo --episodes 50 --seed 0   # scripted demonstrator instead
python3 sim/so101/so101_reach.py replay ../data/rosbags/so101_reach/episode_000   # metrics -> episodes.jsonl
pixi run convert --config configs/so101_conversion_config.yaml
WANDB_MODE=disabled pixi run train --config configs/so101_train_config.yaml   # or `pixi run wandb login` first
pixi run infer --model-path ../data/checkpoints/so101_reach_act --config configs/so101_features.py
python3 sim/so101/so101_reach.py eval --episodes 20 --seed 100   # the policy only acts while /reach_active is true
```
Bags, datasets and checkpoints go to `../data/` (outside the repo).

Reach over a wall (`--wall`): a wall stands between the tucked home pose and most targets, so the arm has to go over
it; touching it (checked kinematically along the links) fails the episode. Same pipeline with the `so101_wall` configs:
```bash
python3 sim/so101/so101_reach.py record --wall --demo --episodes 100 --seed 0   # planner inserts via-points over the wall
pixi run convert --config configs/so101_wall_conversion_config.yaml
WANDB_MODE=disabled pixi run train --config configs/so101_wall_train_config.yaml
pixi run infer --model-path ../data/checkpoints/so101_wall_act --config configs/so101_wall_features.py
python3 sim/so101/so101_reach.py eval --wall --episodes 20 --seed 100
python3 sim/so101/so101_reach.py eval --wall --baseline direct --episodes 20 --seed 100    # no policy: straight move
python3 sim/so101/so101_reach.py eval --wall --baseline planner --episodes 20 --seed 100   # no policy: the planner
```

### Notes
- Gazebo has no GPU in the container and runs below real time (about 40 % on the development machine), while
  recording, the bridge and the policy run at 20 Hz wall-clock time. Tracking lag and results depend on the machine.
- `--demo` and `eval` targets are reproducible with `--seed`; human teleoperation is not.
- Commands use `time_from_start = 0`: a non-zero horizon makes JointTrajectoryController restart a spline on every
  packet, which shrinks the motion a lot when Gazebo runs slower than real time.
