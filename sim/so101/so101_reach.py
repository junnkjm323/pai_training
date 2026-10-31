"""Reach task for the Gazebo SO-101: targets, success check, rosbag2 recording, replay, eval.

A green sphere marks the target. An episode succeeds when the gripper tip
(gripper_frame_link, by forward kinematics of /joint_states) stays within --tol of the
target for --hold seconds. The target is published on /reach_target so it ends up in
the bag and can be given to the policy as observation.environment_state.

--wall adds a thin wall in front of the arm. Targets behind it can only be reached by
going over it; touching the wall (checked kinematically along the links, the wall is
visual-only in Gazebo) fails the episode. The scripted demonstrator inserts a via-point
above the wall when the direct joint-space path would hit it.

Modes (run inside the container with ROS sourced, Gazebo + so101_leader_bridge.py ros running):
    record --demo     scripted demonstrator streams leader packets over UDP (no real leader needed)
    record            a human drives the leader; each episode starts once the arm is back home
    replay BAG...     re-publish the recorded commands and compare with the recorded motion
    eval              run episodes for a policy (scripts/infer.py with an SO-101 config)
    eval --baseline direct|planner   same episodes, driven by a straight joint-space move or by the
                      demonstrator's planner over UDP instead of a policy (bridge required)
    fk                print the tip position for the given joints (debugging)

Bags go to <out>/episode_XXX (sqlite3); one JSON line per episode is appended to <out>/episodes.jsonl.
"""

import argparse
import json
import math
import os
import random
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                         'SO-ARM100', 'Simulation', 'SO101', 'so101_new_calib.urdf')
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
ARM_JOINTS = JOINT_NAMES[:5]
TIP_LINK = 'gripper_frame_link'
HOME = [0.0] * 6
WORLD = 'empty'
MARKER = 'reach_target'

# Wall task. The wall stands between the tucked home pose and the far targets.
WALL = 'reach_wall'
WALL_CENTER = np.array([0.245, 0.0, 0.04])
WALL_SIZE = np.array([0.02, 0.24, 0.08])
WALL_MARGIN = 0.015  # rough link radius added around the wall (contact check)
PLAN_MARGIN = WALL_MARGIN + 0.01  # extra buffer when planning, so tracking error does not cause contact
HOME_WALL = [0.0, -1.2, 1.3, 0.9, 0.0, 0.0]  # tip at about (0.19, 0, 0.04), in front of the wall
IK_NOMINAL = [0.0, -0.6, 1.2, 0.9, 0.0, 0.0]
IK_JOINTS = [0, 1, 2, 3]  # wrist_roll and gripper stay at 0

ARM_TOPIC = '/so101_arm_controller/joint_trajectory'
GRIPPER_TOPIC = '/so101_gripper_controller/joint_trajectory'
TARGET_TOPIC = '/reach_target'
ACTIVE_TOPIC = '/reach_active'
RECORD_TOPICS = {
    '/joint_states': 'sensor_msgs/msg/JointState',
    ARM_TOPIC: 'trajectory_msgs/msg/JointTrajectory',
    GRIPPER_TOPIC: 'trajectory_msgs/msg/JointTrajectory',
    '/leader/joint_states': 'sensor_msgs/msg/JointState',
    TARGET_TOPIC: 'geometry_msgs/msg/PointStamped',
}


# ---------------------------------------------------------------- kinematics

def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_matrix(axis, q):
    x, y, z = axis
    c, s, t = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def load_chain(tip=TIP_LINK):
    """Joints from base_link to `tip` as (name, type, 4x4 origin, axis)."""
    robot = ET.parse(URDF_PATH).getroot()
    by_child = {j.find('child').get('link'): j for j in robot.findall('joint')}
    chain, link = [], tip
    while link in by_child:
        j = by_child[link]
        o = j.find('origin')
        T = np.eye(4)
        T[:3, :3] = _rpy_matrix(*map(float, o.get('rpy', '0 0 0').split()))
        T[:3, 3] = list(map(float, o.get('xyz', '0 0 0').split()))
        axis = j.find('axis')
        chain.append((j.get('name'), j.get('type'), T,
                      np.array(list(map(float, axis.get('xyz').split()))) if axis is not None else None))
        link = j.find('parent').get('link')
    return chain[::-1]


CHAIN = load_chain()


def fk_frames(q):
    """Positions of every joint frame from shoulder_pan to the tip, in base_link (= world, the base is
    fixed at the origin), for joints in JOINT_NAMES order."""
    qd = dict(zip(JOINT_NAMES, q))
    T = np.eye(4)
    frames = []
    for name, jtype, origin, axis in CHAIN:
        T = T @ origin
        if jtype == 'revolute':
            R = np.eye(4)
            R[:3, :3] = _axis_matrix(axis, qd[name])
            T = T @ R
        frames.append(T[:3, 3].copy())
    return frames


def fk(q):
    """Tip position."""
    return fk_frames(q)[-1]


def ik(p, seed=IK_NOMINAL, iters=300):
    """Damped least squares on IK_JOINTS for the tip position, pulled towards IK_NOMINAL in the null space
    so that a target maps to one posture. Returns (joints, position error)."""
    from so101_leader_bridge import JOINT_NAMES as _names, LIMITS
    q = np.array(seed, dtype=float)
    p = np.asarray(p, dtype=float)
    nominal = np.array(IK_NOMINAL)[IK_JOINTS]
    for _ in range(iters):
        x = fk(q)
        e = p - x
        if np.linalg.norm(e) < 1e-4:
            break
        J = np.zeros((3, len(IK_JOINTS)))
        for k, i in enumerate(IK_JOINTS):
            dq = q.copy()
            dq[i] += 1e-5
            J[:, k] = (fk(dq) - x) / 1e-5
        JJ = J @ J.T + 1e-3 * np.eye(3)
        step = J.T @ np.linalg.solve(JJ, e)
        null = (np.eye(len(IK_JOINTS)) - J.T @ np.linalg.solve(JJ, J)) @ (nominal - q[IK_JOINTS])
        q[IK_JOINTS] += step + 0.1 * null
        for i in IK_JOINTS:
            q[i] = min(max(q[i], LIMITS[_names[i]][0]), LIMITS[_names[i]][1])
    return q.tolist(), float(np.linalg.norm(p - fk(q)))


def wall_collision(q, margin=WALL_MARGIN):
    """True if any link segment (joint frame to joint frame) comes within `margin` of the wall."""
    lo = WALL_CENTER - WALL_SIZE / 2 - margin
    hi = WALL_CENTER + WALL_SIZE / 2 + margin
    frames = fk_frames(q)
    for a, b in zip(frames[:-1], frames[1:]):
        for s in np.linspace(0.0, 1.0, 8):
            x = a + (b - a) * s
            if np.all(x > lo) and np.all(x < hi):
                return True
    return False


def path_collides(q0, q1, steps=30):
    """Joint-space straight line (what min_jerk follows) hits the wall."""
    return any(wall_collision([a + (b - a) * s for a, b in zip(q0, q1)], PLAN_MARGIN)
               for s in np.linspace(0, 1, steps))


def plan_path(q0, q1):
    """Waypoints from q0 to q1: [q1] if the direct move is clear, else via-points above the wall
    ([over, q1], or [rise in front, over, q1] / [over, descend in front, q1]), or None."""
    if not path_collides(q0, q1):
        return [list(q1)]
    p0, p1 = fk(q0), fk(q1)
    far, near = (p1, p0) if p1[0] > p0[0] else (p0, p1)
    top = WALL_CENTER[2] + WALL_SIZE[2] / 2
    for clearance in (0.06, 0.09, 0.12):
        q_over, err = ik([WALL_CENTER[0], far[1], top + clearance], seed=q1)
        if err > 0.005:
            continue
        q_front, err_front = ik([WALL_CENTER[0] - 0.07, near[1], top + clearance], seed=q_over)
        candidates = [[q_over]]
        if err_front < 0.005:
            # Rise in front of the wall first when going out, come down in front of it when coming back.
            candidates.append([q_front, q_over] if p1[0] > p0[0] else [q_over, q_front])
        for vias in candidates:
            points = [list(q0)] + vias + [list(q1)]
            if not any(path_collides(a, b) for a, b in zip(points[:-1], points[1:])):
                return vias + [list(q1)]
    return None


def sample_target(rng):
    """Random reachable goal: joints with a fixed wrist/gripper, so target -> joints is (nearly) unique."""
    while True:
        q = [rng.uniform(-1.2, 1.2), rng.uniform(-1.0, 0.8), rng.uniform(-0.8, 1.2), 0.6, 0.0, 0.0]
        p = fk(q)
        if 0.03 < p[2] < 0.30 and 0.15 < math.hypot(p[0], p[1]) < 0.38:
            return q, p


def sample_wall_target(rng):
    """Goal for the wall task: 60 % behind the wall (go over), 20 % beyond its ends (go around/direct),
    20 % in front of it. Returns (joints, position, kind, waypoints from HOME_WALL)."""
    while True:
        kind = rng.choices(['over', 'around', 'front'], weights=[0.6, 0.2, 0.2])[0]
        if kind == 'over':
            p = [rng.uniform(0.29, 0.36), rng.uniform(-0.08, 0.08), rng.uniform(0.02, 0.07)]
        elif kind == 'around':
            p = [rng.uniform(0.26, 0.32), rng.choice([-1, 1]) * rng.uniform(0.17, 0.24), rng.uniform(0.02, 0.08)]
        else:
            p = [rng.uniform(0.10, 0.19), rng.uniform(-0.20, 0.20), rng.uniform(0.02, 0.08)]
            if np.linalg.norm(np.array(p) - fk(HOME_WALL)) < 0.06:
                continue
        q, err = ik(p)
        if err > 0.003 or wall_collision(q, PLAN_MARGIN):
            continue
        path = plan_path(HOME_WALL, q)
        if path is not None:
            return q, np.array(p), kind, path


def min_jerk(q0, q1, s):
    s = min(max(s, 0.0), 1.0)
    w = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return [a + (b - a) * w for a, b in zip(q0, q1)]


def waypoint_trajectory(q0, waypoints, durations):
    """f(t) moving through the waypoints with one min-jerk segment each (stopping at each waypoint)."""
    starts = np.concatenate([[0.0], np.cumsum(durations)])
    points = [list(q0)] + [list(w) for w in waypoints]

    def f(t):
        k = int(np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(durations) - 1))
        return min_jerk(points[k], points[k + 1], (t - starts[k]) / durations[k])
    return f


# ---------------------------------------------------------------- gazebo marker

def ign_service(service, reqtype, req):
    subprocess.run(['ign', 'service', '-s', f'/world/{WORLD}/{service}', '--reqtype', reqtype,
                    '--reptype', 'ignition.msgs.Boolean', '--timeout', '3000', '--req', req],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def spawn_marker(radius):
    sdf = (f'<sdf version="1.6"><model name="{MARKER}"><static>true</static><link name="link">'
           f'<visual name="v"><geometry><sphere><radius>{radius}</radius></sphere></geometry>'
           '<material><ambient>0 1 0 0.6</ambient><diffuse>0 1 0 0.6</diffuse></material></visual>'
           '</link></model></sdf>')
    ign_service('create', 'ignition.msgs.EntityFactory', 'sdf: ' + json.dumps(sdf))


def move_marker(p):
    ign_service('set_pose', 'ignition.msgs.Pose',
                f'name: "{MARKER}", position: {{x: {p[0]}, y: {p[1]}, z: {p[2]}}}')


def spawn_wall():
    sx, sy, sz = WALL_SIZE
    x, y, z = WALL_CENTER
    sdf = (f'<sdf version="1.6"><model name="{WALL}"><static>true</static><pose>{x} {y} {z} 0 0 0</pose>'
           f'<link name="link"><visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
           '<material><ambient>0.6 0.35 0.15 1</ambient><diffuse>0.6 0.35 0.15 1</diffuse></material></visual>'
           '</link></model></sdf>')
    ign_service('create', 'ignition.msgs.EntityFactory', 'sdf: ' + json.dumps(sdf))


# ---------------------------------------------------------------- ROS node

def run(args):
    import rclpy
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import PointStamped
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class Reach(Node):
        def __init__(self):
            super().__init__('so101_reach')
            self.q = None
            self.writer = None
            self.target = None
            self.target_pub = self.create_publisher(PointStamped, TARGET_TOPIC, 10)
            self.active_pub = self.create_publisher(Bool, ACTIVE_TOPIC, 10)
            self.arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
            self.gripper_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
            # Raw subscriptions: bag the bytes exactly as received. Re-serializing in rclpy can leave
            # trailing padding that the rosbags reader used by rosbag2lerobot.py rejects.
            for topic, type_name in RECORD_TOPICS.items():
                msg_type = {'sensor_msgs/msg/JointState': JointState,
                            'trajectory_msgs/msg/JointTrajectory': JointTrajectory,
                            'geometry_msgs/msg/PointStamped': PointStamped}[type_name]
                self.create_subscription(msg_type, topic, lambda raw, t=topic: self.on_raw(t, raw), 50, raw=True)
            self.create_timer(0.05, self.publish_target)
            self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.seq = 0

        def on_raw(self, topic, raw):
            if topic == '/joint_states':
                msg = deserialize_message(raw, JointState)
                pos = dict(zip(msg.name, msg.position))
                self.q = [pos.get(n, 0.0) for n in JOINT_NAMES]
            if self.writer is not None:
                self.writer.write(topic, raw, time.time_ns())

        def publish_target(self):
            active = Bool(data=self.target is not None)
            self.active_pub.publish(active)
            if self.target is not None:
                msg = PointStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'world'
                msg.point.x, msg.point.y, msg.point.z = map(float, self.target)
                self.target_pub.publish(msg)

        # -- helpers
        def spin_for(self, seconds):
            end = time.time() + seconds
            while time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.01)

        def wait_joint_states(self):
            while self.q is None:
                rclpy.spin_once(self, timeout_sec=0.1)

        def send_leader(self, q):
            """Pretend to be the leader arm (same packets as so101_leader_bridge.py send)."""
            pkt = {'seq': self.seq, 't': time.time(), 'pos': list(map(float, q))}
            self.udp.sendto(json.dumps(pkt).encode(), (args.udp_host, args.udp_port))
            self.seq += 1

        def stream_leader(self, waypoints, durations, hold=0.0):
            traj = waypoint_trajectory(self.q, waypoints, durations)
            t0 = time.time()
            while (t := time.time() - t0) < sum(durations) + hold:
                self.send_leader(traj(t))
                self.spin_for(0.05)

        def command_direct(self, waypoints, durations):
            """Command the controllers directly (replay/eval resets, when no leader is streaming)."""
            def point(q, t):
                return JointTrajectoryPoint(positions=list(q), time_from_start=Duration(
                    sec=int(t), nanosec=int((t % 1) * 1e9)))
            times = np.cumsum(durations)
            self.arm_pub.publish(JointTrajectory(
                joint_names=ARM_JOINTS, points=[point(q[:5], t) for q, t in zip(waypoints, times)]))
            self.gripper_pub.publish(JointTrajectory(
                joint_names=['gripper'], points=[point(q[5:], t) for q, t in zip(waypoints, times)]))

        def tip_error(self):
            return float(np.linalg.norm(fk(self.q) - np.asarray(self.target)))

        def wait_success(self, timeout, on_tick=None):
            """Return (success, time to reach, final error, collided). Reached = within tol for `hold` s;
            success = reached without ever touching the wall (--wall)."""
            t0 = time.time()
            inside_since = None
            collided = False
            while time.time() - t0 < timeout:
                if on_tick:
                    on_tick(time.time() - t0)
                self.spin_for(0.05)
                collided = collided or (args.wall and wall_collision(self.q))
                err = self.tip_error()
                if err < args.tol:
                    inside_since = inside_since or time.time()
                    if time.time() - inside_since >= args.hold:
                        return not collided, inside_since - t0, err, collided
                else:
                    inside_since = None
            return False, None, self.tip_error(), collided

        def open_bag(self, path):
            import rosbag2_py
            self.writer = rosbag2_py.SequentialWriter()
            self.writer.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                             rosbag2_py.ConverterOptions('cdr', 'cdr'))
            for topic, type_name in RECORD_TOPICS.items():
                self.writer.create_topic(rosbag2_py.TopicMetadata(
                    name=topic, type=type_name, serialization_format='cdr'))

        def close_bag(self):
            writer, self.writer = self.writer, None
            del writer  # the sqlite3 file is finalized when the writer is destroyed

    rclpy.init()
    node = Reach()
    node.wait_joint_states()
    spawn_marker(args.tol)
    if args.wall:
        spawn_wall()
    try:
        {'record': record, 'eval': evaluate, 'replay': replay}[args.mode](node, args)
    except KeyboardInterrupt:
        pass
    finally:
        node.target = None
        node.publish_target()
        node.close_bag()
        node.destroy_node()
        rclpy.try_shutdown()


def log_episode(out, entry):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'episodes.jsonl'), 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(json.dumps(entry))


def next_episode_index(out):
    existing = [d for d in os.listdir(out) if d.startswith('episode_')] if os.path.isdir(out) else []
    return max([int(d.split('_')[1]) for d in existing], default=-1) + 1


def home_pose(args):
    return HOME_WALL if args.wall else HOME


def pick_target(rng, args):
    """(joints, position, kind, waypoints from home) for the next episode."""
    if args.wall:
        return sample_wall_target(rng)
    q, p = sample_target(rng)
    return q, p, 'reach', [q]


def segment_durations(rng, path):
    # Human-like timing: one smooth move, or two shorter ones through the via-point.
    return [rng.uniform(2.0, 3.5)] if len(path) == 1 else [rng.uniform(1.3, 2.0) for _ in path]


def go_home(node, args):
    home = home_pose(args)
    # Plan around the wall on the way back too, so resets never cut through it.
    path = (plan_path(node.q, home) if args.wall else None) or [home]
    def home_error():
        return max(abs(a - b) for a, b in zip(node.q[:5], home[:5]))

    if args.mode == 'record' and args.demo or getattr(args, 'baseline', None):
        node.stream_leader(path, [1.5] * len(path), hold=0.5)
        # Gazebo may run well below real time (e.g. with the GUI): keep holding home until the arm is
        # really there, since the next episode's path is planned from home.
        deadline = time.time() + 10.0
        while home_error() > 0.02 and time.time() < deadline:
            node.send_leader(home)
            node.spin_for(0.05)
    elif args.mode == 'record':
        print('move the leader back to the home pose...')
        while home_error() > args.home_tol:
            node.spin_for(0.1)
        node.spin_for(0.5)
    else:
        node.command_direct(path, [1.5] * len(path))
        node.spin_for(1.5 * len(path) + 1.0)
        deadline = time.time() + 10.0
        while home_error() > 0.02 and time.time() < deadline:
            node.spin_for(0.1)


def record(node, args):
    rng = random.Random(args.seed)
    start = next_episode_index(args.out)
    for ep in range(start, start + args.episodes):
        go_home(node, args)
        q_goal, target, kind, waypoints = pick_target(rng, args)
        move_marker(target)
        node.target = target
        path = os.path.join(args.out, f'episode_{ep:03d}')
        node.open_bag(path)
        on_tick = None
        if args.demo:
            delay = rng.uniform(0.3, 0.8)
            traj = waypoint_trajectory(node.q, waypoints, segment_durations(rng, waypoints))
            on_tick = lambda t: node.send_leader(traj(t - delay))
        ok, t_reach, err, collided = node.wait_success(args.timeout, on_tick)
        node.spin_for(0.5)  # keep a little of the settled pose in the bag
        node.close_bag()
        node.target = None
        log_episode(args.out, {'episode': ep, 'bag': path, 'demo': args.demo, 'kind': kind,
                               'via_point': len(waypoints) > 1, 'target': list(map(float, target)),
                               'success': ok, 'collided': collided, 'time_to_reach': t_reach, 'final_error': err})


def evaluate(node, args):
    rng = random.Random(args.seed)
    results = []
    for ep in range(args.episodes):
        go_home(node, args)
        q_goal, target, kind, waypoints = pick_target(rng, args)
        move_marker(target)
        on_tick = None
        if args.baseline:
            # Same targets, no policy: a straight joint-space move, or the demonstrator's planner.
            path = [q_goal] if args.baseline == 'direct' else waypoints
            traj = waypoint_trajectory(node.q, path, [2.5] if len(path) == 1 else [1.6] * len(path))
            on_tick = lambda t: node.send_leader(traj(t))
        node.target = target  # /reach_active goes true: the policy starts acting
        ok, t_reach, err, collided = node.wait_success(args.timeout, on_tick)
        node.target = None
        node.spin_for(0.2)
        results.append((ok, collided))
        log_episode(args.out, {'eval_episode': ep, 'baseline': args.baseline, 'kind': kind,
                               'target': list(map(float, target)), 'success': ok, 'collided': collided,
                               'time_to_reach': t_reach, 'final_error': err})
    print(f'success rate: {sum(ok for ok, _ in results)}/{len(results)}, '
          f'wall contact: {sum(c for _, c in results)}/{len(results)}')


def replay(node, args):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory

    for bag in args.bags:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
        msgs = []
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic in (ARM_TOPIC, GRIPPER_TOPIC, '/joint_states', TARGET_TOPIC):
                cls = {ARM_TOPIC: JointTrajectory, GRIPPER_TOPIC: JointTrajectory,
                       '/joint_states': JointState, TARGET_TOPIC: PointStamped}[topic]
                msgs.append((t, topic, deserialize_message(data, cls)))
        targets = [m for _, tp, m in msgs if tp == TARGET_TOPIC]
        target = [targets[0].point.x, targets[0].point.y, targets[0].point.z]
        cmds = [(t, tp, m) for t, tp, m in msgs if tp in (ARM_TOPIC, GRIPPER_TOPIC)]
        orig = [(t, [dict(zip(m.name, m.position))[n] for n in JOINT_NAMES])
                for t, tp, m in msgs if tp == '/joint_states']

        go_home(node, args)
        move_marker(target)
        node.target = target
        replayed, sent = [], []
        t_bag0, t_wall0 = cmds[0][0], time.time()
        for t, topic, m in cmds:
            while time.time() - t_wall0 < (t - t_bag0) / 1e9:
                rclpy_spin_once(node)
                replayed.append((time.time() - t_wall0, list(node.q)))
            (node.arm_pub if topic == ARM_TOPIC else node.gripper_pub).publish(m)
            sent.append(time.time() - t_wall0)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            rclpy_spin_once(node)
            replayed.append((time.time() - t_wall0, list(node.q)))
        err = node.tip_error()
        collided = bool(args.wall and any(wall_collision(q) for _, q in replayed))
        node.target = None

        # Compare with the recorded joint states on the same (command-relative) time axis.
        ot = np.array([(t - t_bag0) / 1e9 for t, _ in orig])
        oq = np.array([q for _, q in orig])
        rt = np.array([t for t, _ in replayed])
        rq = np.array([q for _, q in replayed])
        mask = (rt >= ot[0]) & (rt <= ot[-1])
        interp = np.stack([np.interp(rt[mask], ot, oq[:, j]) for j in range(6)], 1)
        rmse = np.sqrt(((interp - rq[mask]) ** 2).mean(0))
        lag_err = {}
        for lag in np.arange(-0.5, 0.51, 0.05):
            m2 = (rt - lag >= ot[0]) & (rt - lag <= ot[-1])
            i2 = np.stack([np.interp(rt[m2] - lag, ot, oq[:, j]) for j in range(5)], 1)
            lag_err[round(float(lag), 2)] = float(np.sqrt(((i2 - rq[m2][:, :5]) ** 2).mean()))
        best_lag = min(lag_err, key=lag_err.get)
        orig_dt = np.diff([t for t, tp, _ in cmds if tp == ARM_TOPIC]) / 1e9
        sent_dt = np.diff([s for s, (_, tp, _) in zip(sent, cmds) if tp == ARM_TOPIC])
        log_episode(args.out, {
            'replay_of': bag, 'target': target, 'final_error': err, 'success': err < args.tol and not collided,
            'collided': collided,
            'joint_rmse': dict(zip(JOINT_NAMES, rmse.round(4).tolist())),
            'best_lag_s': best_lag, 'rmse_at_best_lag': round(lag_err[best_lag], 4),
            'cmd_period_ms': {'recorded_mean': float(orig_dt.mean() * 1e3), 'recorded_std': float(orig_dt.std() * 1e3),
                              'replayed_mean': float(sent_dt.mean() * 1e3), 'replayed_std': float(sent_dt.std() * 1e3)},
        })


def rclpy_spin_once(node):
    import rclpy
    rclpy.spin_once(node, timeout_sec=0.002)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['record', 'replay', 'eval', 'fk'])
    parser.add_argument('bags', nargs='*', help='episode bag dirs (replay) / joint values (fk)')
    parser.add_argument('--demo', action='store_true', help='scripted demonstrator instead of a human (record)')
    parser.add_argument('--wall', action='store_true', help='reach-over-the-wall task')
    parser.add_argument('--baseline', choices=['direct', 'planner'], default=None,
                        help='eval without a policy: straight joint move or the demonstrator planner (eval)')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--out', default=None,
                        help='default: /home/ubuntu/pai2026-final-assignment/data/rosbags/so101_reach[_wall]')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tol', type=float, default=0.02, help='success radius [m] (also the marker size)')
    parser.add_argument('--hold', type=float, default=0.5, help='time to stay within tol [s]')
    parser.add_argument('--timeout', type=float, default=15.0, help='episode timeout [s]')
    parser.add_argument('--home-tol', type=float, default=0.15, help='max joint error to count as home [rad]')
    parser.add_argument('--udp-host', default='127.0.0.1')
    parser.add_argument('--udp-port', type=int, default=5005)
    args = parser.parse_intermixed_args()  # allows `replay --wall BAG...`
    if args.out is None:
        args.out = '/home/ubuntu/pai2026-final-assignment/data/rosbags/' + ('so101_wall' if args.wall else 'so101_reach')
    if args.mode == 'fk':
        q = [float(v) for v in args.bags] + [0.0] * (6 - len(args.bags))
        print(np.round(fk(q), 4))
        return
    run(args)


if __name__ == '__main__':
    main()
