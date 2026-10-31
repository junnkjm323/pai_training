# Course image + serial port access for the SO-101 leader arm. Nothing else is changed.
# The base is pinned by digest (the :latest tag of 2025-02-12) so a rebuild gets the same ROS 2 Humble /
# Ignition Gazebo Fortress / ros2_control packages.
#
#   docker build -t pai2026-so101 docker/
#   (run command and the rest of the setup: see README "SO-101 (Gazebo) teleop, data collection, ACT")
FROM airobotbook/ros2-desktop-ai-robot-book-humble@sha256:b58ec74c019aacd9ce4818fa25fc5a9e0a1f22f8c9cd1f5260994bf15cd9eb63

# Read the SO-101 leader's USB serial port (/dev/ttyACM*, group dialout) as the ubuntu user.
RUN usermod -aG dialout ubuntu
