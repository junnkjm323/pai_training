"""SO-101 leader -> UDP -> ROS 2 bridge for teleoperating the Gazebo SO-101.

Modes:
    send  (host)      read the leader arm with LeRobot and send joint angles over UDP
    fake  (anywhere)  send sine-wave joint angles over UDP instead of a real leader
    ros   (container) receive UDP and publish JointTrajectory commands to the
                      ros2_control controllers started by so101_gazebo.launch.py

Packets are JSON: {"seq": int, "t": sender wall time [s], "pos": [6 joint angles, rad]}
in URDF (so101_new_calib) joint order and units, so `ros` mode does no conversion.
The container shares the host kernel clock, so `t` can be compared with receive time.

Examples:
    python3 sim/so101/so101_leader_bridge.py fake                      # host or container
    python3 sim/so101/so101_leader_bridge.py send --port /dev/ttyACM0  # host, LeRobot env
    python3 sim/so101/so101_leader_bridge.py ros                       # container, ROS sourced
"""

import argparse
import json
import math
import socket
import time

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
ARM_JOINTS = JOINT_NAMES[:5]
# From SO-ARM100/Simulation/SO101/so101_new_calib.urdf (rad).
LIMITS = {
    'shoulder_pan': (-1.91986, 1.91986),
    'shoulder_lift': (-1.74533, 1.74533),
    'elbow_flex': (-1.69, 1.69),
    'wrist_flex': (-1.65806, 1.65806),
    'wrist_roll': (-2.74385, 2.84121),
    'gripper': (-0.174533, 1.74533),
}


def clamp(name, value):
    lo, hi = LIMITS[name]
    return min(max(value, lo), hi)


def udp_sender(host, port, rate, sample):
    """Call sample() -> 6 angles [rad] at `rate` Hz and send each as a packet."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / rate
    seq = 0
    next_t = time.monotonic()
    print(f'sending to udp://{host}:{port} at {rate} Hz (Ctrl-C to stop)')
    try:
        while True:
            pos = [clamp(n, p) for n, p in zip(JOINT_NAMES, sample())]
            sock.sendto(json.dumps({'seq': seq, 't': time.time(), 'pos': pos}).encode(), (host, port))
            if seq % rate == 0:
                print(f'seq={seq} pos=' + ' '.join(f'{p:+.2f}' for p in pos))
            seq += 1
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass


def run_fake(args):
    t0 = time.monotonic()
    # Per joint: (amplitude as a fraction of the half range, period [s], phase [rad]).
    waves = [(0.6, 8.0, 0.0), (0.4, 6.0, 1.0), (0.5, 5.0, 2.0), (0.5, 7.0, 0.5), (0.5, 9.0, 0.0), (0.5, 4.0, 0.0)]

    def sample():
        t = time.monotonic() - t0
        out = []
        for name, (amp, per, ph) in zip(JOINT_NAMES, waves):
            lo, hi = LIMITS[name]
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            out.append(mid + amp * half * math.sin(2 * math.pi * t / per + ph))
        return out

    udp_sender(args.host, args.udp_port, args.rate, sample)


def run_send(args):
    # LeRobot moved the SO-101 leader between releases; accept both layouts.
    try:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    except ImportError:
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    leader = SO101Leader(SO101LeaderConfig(port=args.port, id=args.id, use_degrees=True))
    leader.connect()
    signs = [float(s) for s in args.signs.split(',')]
    offsets = [float(o) for o in args.offsets.split(',')]
    g_lo, g_hi = LIMITS['gripper']

    def sample():
        action = leader.get_action()
        out = []
        for i, name in enumerate(ARM_JOINTS):
            # LeRobot body joints are in degrees (use_degrees=True).
            out.append(signs[i] * math.radians(action[f'{name}.pos']) + offsets[i])
        # LeRobot gripper is 0..100 (closed..open); map onto the URDF gripper range.
        out.append(signs[5] * (g_lo + (g_hi - g_lo) * action['gripper.pos'] / 100.0) + offsets[5])
        return out

    try:
        udp_sender(args.host, args.udp_port, args.rate, sample)
    finally:
        leader.disconnect()


def run_ros(args):
    import rclpy
    from builtin_interfaces.msg import Duration
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class LeaderBridge(Node):
        def __init__(self):
            super().__init__('so101_leader_bridge')
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('0.0.0.0', args.udp_port))
            self.sock.setblocking(False)
            self.arm_pub = self.create_publisher(JointTrajectory, '/so101_arm_controller/joint_trajectory', 10)
            self.gripper_pub = self.create_publisher(JointTrajectory, '/so101_gripper_controller/joint_trajectory', 10)
            # Raw leader angles, stamped with the sender's wall time, for latency/jitter analysis.
            self.leader_pub = self.create_publisher(JointState, '/leader/joint_states', 10)
            self.latest = None
            self.latest_recv = 0.0
            self.create_timer(0.001, self.poll_udp)
            self.create_timer(1.0 / args.rate, self.publish_command)
            self.get_logger().info(f'listening on udp://0.0.0.0:{args.udp_port}, commanding at {args.rate} Hz')

        def poll_udp(self):
            while True:
                try:
                    data, _ = self.sock.recvfrom(4096)
                except BlockingIOError:
                    return
                pkt = json.loads(data)
                self.latest, self.latest_recv = pkt, time.time()
                msg = JointState()
                msg.header.stamp.sec = int(pkt['t'])
                msg.header.stamp.nanosec = int((pkt['t'] % 1) * 1e9)
                msg.header.frame_id = str(pkt['seq'])
                msg.name = JOINT_NAMES
                msg.position = [float(p) for p in pkt['pos']]
                self.leader_pub.publish(msg)

        def trajectory(self, names, positions):
            msg = JointTrajectory()
            msg.joint_names = names
            # Stamp 0 = start now. A nonzero horizon makes JTC restart a spline from rest on
            # every packet, which attenuates the motion badly when Gazebo runs below real time.
            msg.points = [JointTrajectoryPoint(
                positions=positions, time_from_start=Duration(nanosec=int(args.horizon * 1e9)))]
            return msg

        def publish_command(self):
            # Stop commanding if the leader went silent, so the arm holds its last pose.
            if self.latest is None or time.time() - self.latest_recv > args.timeout:
                return
            pos = [clamp(n, float(p)) for n, p in zip(JOINT_NAMES, self.latest['pos'])]
            self.arm_pub.publish(self.trajectory(ARM_JOINTS, pos[:5]))
            self.gripper_pub.publish(self.trajectory(['gripper'], pos[5:]))

    rclpy.init()
    node = LeaderBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['send', 'fake', 'ros'])
    parser.add_argument('--host', default='127.0.0.1', help='UDP destination (send/fake)')
    parser.add_argument('--udp-port', type=int, default=5005)
    parser.add_argument('--rate', type=int, default=20, help='send rate / command rate [Hz]')
    parser.add_argument('--port', default='/dev/ttyACM0', help='leader serial port (send)')
    parser.add_argument('--id', default='so101_leader', help='LeRobot calibration id (send)')
    parser.add_argument('--signs', default='1,1,1,1,1,1', help='per-joint sign, LeRobot -> URDF (send)')
    parser.add_argument('--offsets', default='0,0,0,0,0,0', help='per-joint offset [rad] after sign (send)')
    parser.add_argument('--horizon', type=float, default=0.0, help='time_from_start of each command [s] (ros)')
    parser.add_argument('--timeout', type=float, default=0.5, help='stop commanding after this silence [s] (ros)')
    args = parser.parse_args()
    {'send': run_send, 'fake': run_fake, 'ros': run_ros}[args.mode](args)


if __name__ == '__main__':
    main()
