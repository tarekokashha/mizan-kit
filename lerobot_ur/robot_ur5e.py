"""
lerobot_ur.robot_ur5e  (M-03 KEYSTONE, skeleton v0)
===================================================

A LeRobot-style follower robot for Universal Robots e-Series arms over RTDE,
with the built-in force-torque sensor and a gripper in the observation.

STATUS: skeleton. The control logic below is real ur_rtde code and runs
against URSim; the LeRobot base-class wiring must be aligned with the
LeRobot version you install (the plugin contract is documented at
https://huggingface.co/docs/lerobot/integrate_hardware and has changed
across releases). Do the alignment as the first overnight task:

    1. pip install lerobot ur_rtde
    2. from lerobot.robots import Robot, RobotConfig   # check exact names
    3. subclass, keep the method bodies below, satisfy the abstract API
    4. register the plugin entry point so `lerobot-record --robot.type=ur5e`
       discovers it

Existing community plugins to credit and consolidate (see the third-party
page on the LeRobot docs): F-Fer/lerobot_ur5e_gello, yechen056/UR5e-LeRobot,
scy-v/lerobot_ur5e_auto.

Safety: this class never commands motion unless `send_action` is called by
a teleoperator or a policy through LeRobot, and it clamps every command to
the configured joint-velocity and force envelopes. Test against URSim first.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

try:
    import rtde_control
    import rtde_receive
    import dashboard_client
except Exception:  # pragma: no cover
    rtde_control = rtde_receive = dashboard_client = None


@dataclass
class UR5eConfig:
    ip: str = "127.0.0.1"            # URSim in Docker, or the real controller
    control_hz: float = 500.0        # e-Series RTDE rate
    max_joint_velocity: float = 0.8  # rad/s, clamp on every command
    max_force_norm: float = 60.0     # N, stop if the flange F/T exceeds this
    servo_lookahead: float = 0.1     # servoJ lookahead time (s)
    servo_gain: float = 300.0        # servoJ gain
    gripper: str = "none"            # "robotiq_2f85" | "none"
    cameras: dict = field(default_factory=dict)  # name -> LeRobot camera config
    id: str = "ur5e"


class UR5eFollower:
    """Follower robot: read joints + F/T + gripper, write joint targets."""

    name = "ur5e"

    def __init__(self, config: UR5eConfig):
        self.config = config
        self._rtde_c = None
        self._rtde_r = None
        self._dash = None
        self._gripper = None
        self._connected = False

    # ---- LeRobot feature contracts -------------------------------------- #
    @property
    def observation_features(self) -> dict:
        feats = {
            "joint_position": (6,),
            "joint_velocity": (6,),
            "tcp_pose": (6,),
            "tcp_force_torque": (6,),
            "gripper_position": (1,),
        }
        for cam in self.config.cameras:
            feats[cam] = "image"
        return feats

    @property
    def action_features(self) -> dict:
        return {"joint_position": (6,), "gripper_position": (1,)}

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---- lifecycle -------------------------------------------------------- #
    def connect(self) -> None:
        if rtde_receive is None:
            raise ImportError("pip install ur_rtde")
        self._dash = dashboard_client.DashboardClient(self.config.ip)
        self._dash.connect()
        # URSim and real arms both need: power on, brake release, remote control
        if not self._dash.isInRemoteControl():
            pass  # on a real arm, switch the pendant to Remote Control by hand
        self._dash.powerOn()
        self._dash.brakeRelease()
        time.sleep(1.0)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self.config.ip, frequency=self.config.control_hz)
        self._rtde_c = rtde_control.RTDEControlInterface(self.config.ip, frequency=self.config.control_hz)
        if self.config.gripper == "robotiq_2f85":
            self._gripper = _RobotiqPlaceholder()
        self._connected = True

    def disconnect(self) -> None:
        if self._rtde_c is not None:
            try:
                self._rtde_c.servoStop()
                self._rtde_c.stopScript()
            finally:
                self._rtde_c.disconnect()
        if self._rtde_r is not None:
            self._rtde_r.disconnect()
        if self._dash is not None:
            self._dash.disconnect()
        self._connected = False

    def calibrate(self) -> None:
        """UR arms are factory calibrated; LeRobot expects the hook to exist."""
        return None

    # ---- I/O ------------------------------------------------------------- #
    def get_observation(self) -> dict:
        t = time.perf_counter()
        obs = {
            "joint_position": np.asarray(self._rtde_r.getActualQ(), dtype=np.float32),
            "joint_velocity": np.asarray(self._rtde_r.getActualQd(), dtype=np.float32),
            "tcp_pose": np.asarray(self._rtde_r.getActualTCPPose(), dtype=np.float32),
            "tcp_force_torque": np.asarray(self._rtde_r.getActualTCPForce(), dtype=np.float32),
            "gripper_position": np.asarray([self._gripper.position() if self._gripper else 0.0], dtype=np.float32),
            "timestamp_monotonic": t,  # written so M-02's audit can check sync
        }
        for name, cam in self.config.cameras.items():
            obs[name] = cam.async_read() if hasattr(cam, "async_read") else None
        return obs

    def send_action(self, action: dict) -> dict:
        """Clamp, safety-check, then servoJ one 2 ms step. Returns the action sent."""
        q_target = np.asarray(action["joint_position"], dtype=np.float64)
        q_now = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
        dt = 1.0 / self.config.control_hz
        max_step = self.config.max_joint_velocity * dt
        q_cmd = q_now + np.clip(q_target - q_now, -max_step, max_step)
        ft = np.asarray(self._rtde_r.getActualTCPForce(), dtype=np.float64)
        if np.linalg.norm(ft[:3]) > self.config.max_force_norm:
            self._rtde_c.servoStop()
            raise RuntimeError(f"force envelope exceeded: {np.linalg.norm(ft[:3]):.1f} N")
        t0 = self._rtde_c.initPeriod()
        self._rtde_c.servoJ(q_cmd.tolist(), 0.0, 0.0, dt, self.config.servo_lookahead, self.config.servo_gain)
        self._rtde_c.waitPeriod(t0)
        if self._gripper is not None and "gripper_position" in action:
            self._gripper.move(float(np.asarray(action["gripper_position"]).ravel()[0]))
        return {"joint_position": q_cmd.astype(np.float32), "gripper_position": action.get("gripper_position")}


class _RobotiqPlaceholder:
    """Replace with the Robotiq URCap socket driver or the ur_rtde gripper helper."""

    def __init__(self):
        self._pos = 0.0

    def position(self) -> float:
        return self._pos

    def move(self, pos: float) -> None:
        self._pos = float(np.clip(pos, 0.0, 1.0))
