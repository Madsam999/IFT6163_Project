"""AprilTag navigation environment for the Pupper quadruped.

A tag from the tag36h11 family is placed on a vertical billboard in the scene.
The robot observes a low-resolution camera image from a front-facing virtual camera,
runs AprilTag detection each step, and receives goal-relative observations derived
from the detected pose rather than ground-truth XY coordinates.

This makes the task vision-based and Sim2Real-compatible: the same detector runs
on real hardware using an onboard camera.

Observation (23-dim):
    [0:18]  base pupper obs (PMTG-wrapped: 12 motor + 4 IMU + 2 phase)
    [18]    tag visible (1.0) or not (0.0)
    [19]    bearing to tag in robot frame [-pi, pi] (0 when not visible)
    [20]    estimated distance to tag in metres (0 when not visible)
    [21]    dx to tag in world frame (ground truth, kept for reward only — not given to policy)
    [22]    dy to tag in world frame (ground truth, kept for reward only — not given to policy)

    Note: [21] and [22] are used internally for the reward and are NOT part of the
    policy's observation space (obs_dim=21 for the policy).

Reward:
    Potential-based on ground-truth distance (same formula as PupperNavGymEnv).
    +50 success bonus when within GOAL_RADIUS metres.
    -1 per step tag is not visible (encourages the robot to keep the tag in view).

Episode ends when:
    - Robot reaches goal (success)
    - Robot falls
    - Max steps reached (gymnasium TimeLimit wrapper)
"""

import os
from typing import Optional, Tuple

import cv2
import gin
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

import puppersim
import puppersim.data as pd
from pybullet_envs.minitaur.envs_v2 import env_loader
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils

try:
    import pupil_apriltags as apriltag
    _APRILTAG_AVAILABLE = True
except ImportError:
    _APRILTAG_AVAILABLE = False

# Camera intrinsics for the virtual PyBullet camera (approximate, tuned for 84x84)
_CAM_WIDTH = 84
_CAM_HEIGHT = 84
_CAM_FOV = 60.0  # degrees
_CAM_FX = _CAM_WIDTH / (2.0 * np.tan(np.deg2rad(_CAM_FOV / 2.0)))
_CAM_FY = _CAM_FX
_CAM_CX = _CAM_WIDTH / 2.0
_CAM_CY = _CAM_HEIGHT / 2.0

# Physical size of the tag in metres (the billboard will be this wide)
_TAG_PHYSICAL_SIZE = 0.5  # metres


def _create_inner_env(render: bool = False):
    CONFIG_DIR = puppersim.getPupperSimPath()
    config_file = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(config_file)
    gin.bind_parameter("SimulationParameters.enable_rendering", render)
    return env_loader.load()


class PupperAprilTagNavEnv(gym.Env):
    """Pupper navigation task using AprilTag visual detection as goal signal."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    GOAL_RADIUS = 0.35       # metres — success threshold
    GOAL_MIN_DIST = 1.5      # metres — minimum spawn distance
    GOAL_MAX_DIST = 3.0      # metres — maximum spawn distance
    TAG_HEIGHT = 0.5         # metres — billboard centre height above ground

    def __init__(self, render_mode: Optional[str] = None,
                 tag_id: int = 0,
                 tag_dir: str = "apriltag-imgs/tag36h11"):
        if not _APRILTAG_AVAILABLE:
            raise ImportError(
                "pupil-apriltags is required: pip install pupil-apriltags opencv-python"
            )

        self._inner = _create_inner_env(render=render_mode == "human")
        self.render_mode = render_mode

        self._tag_id = tag_id
        self._tag_dir = tag_dir
        self._tag_texture_id: Optional[int] = None
        self._tag_body_id: Optional[int] = None
        self._goal_pos: Optional[np.ndarray] = None
        self._prev_dist: float = 0.0
        self._rng = np.random.default_rng()

        # AprilTag detector — reused across steps for efficiency
        self._detector = apriltag.Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
        )

        # Policy observation: 18 base + 3 vision features (visible, bearing, distance)
        base_low = self._inner.observation_space.low
        base_high = self._inner.observation_space.high
        vision_low = np.array([0.0, -np.pi, 0.0], dtype=np.float32)
        vision_high = np.array([1.0, np.pi, 10.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, vision_low]),
            high=np.concatenate([base_high, vision_high]),
            dtype=np.float32,
        )
        self.action_space = self._inner.action_space

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _robot_xyyaw(self) -> Tuple[np.ndarray, float]:
        pos = env_utils.get_robot_base_position(self._inner.robot)
        _, _, yaw = self._inner.robot.base_roll_pitch_yaw
        return np.array(pos[:2], dtype=np.float32), yaw

    def _spawn_tag(self):
        """Place a textured AprilTag billboard at a random position."""
        rng = self._rng
        angle = rng.uniform(0.0, 2.0 * np.pi)
        dist = rng.uniform(self.GOAL_MIN_DIST, self.GOAL_MAX_DIST)
        self._goal_pos = np.array(
            [dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32
        )

        client = self._inner._pybullet_client

        # Remove previous billboard
        if self._tag_body_id is not None:
            try:
                client.removeBody(self._tag_body_id)
            except Exception:
                pass
            self._tag_body_id = None

        # Load the tag PNG as a pybullet texture
        tag_filename = f"tag36_11_{self._tag_id:05d}.png"
        tag_path = os.path.join(self._tag_dir, tag_filename)
        if not os.path.exists(tag_path):
            raise FileNotFoundError(f"Tag image not found: {tag_path}")

        self._tag_texture_id = client.loadTexture(tag_path)

        # Billboard: thin box with the tag texture on the front face
        half = _TAG_PHYSICAL_SIZE / 2.0
        col = client.createCollisionShape(p.GEOM_BOX, halfExtents=[half, 0.01, half])
        vis = client.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half, 0.01, half],
            rgbaColor=[1, 1, 1, 1],
        )
        self._tag_body_id = client.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[
                float(self._goal_pos[0]),
                float(self._goal_pos[1]),
                self.TAG_HEIGHT,
            ],
        )
        client.changeVisualShape(
            self._tag_body_id, -1,
            textureUniqueId=self._tag_texture_id,
        )

    def _get_camera_image(self) -> np.ndarray:
        """Capture a front-facing grayscale image from the robot's head."""
        client = self._inner._pybullet_client
        robot_pos, yaw = self._robot_xyyaw()

        cam_pos = [float(robot_pos[0]), float(robot_pos[1]), 0.20]  # ~head height
        target_x = cam_pos[0] + np.cos(yaw)
        target_y = cam_pos[1] + np.sin(yaw)
        target_pos = [target_x, target_y, 0.20]

        view_matrix = client.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1],
        )
        proj_matrix = client.computeProjectionMatrixFOV(
            fov=_CAM_FOV,
            aspect=1.0,
            nearVal=0.05,
            farVal=10.0,
        )
        _, _, rgb_px, _, _ = client.getCameraImage(
            width=_CAM_WIDTH,
            height=_CAM_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,  # fast CPU renderer
        )
        rgb = np.array(rgb_px, dtype=np.uint8).reshape(_CAM_HEIGHT, _CAM_WIDTH, 4)
        gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
        return gray

    def _detect_tag(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """Run AprilTag detection. Returns (visible, bearing, distance)."""
        detections = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(_CAM_FX, _CAM_FY, _CAM_CX, _CAM_CY),
            tag_size=_TAG_PHYSICAL_SIZE,
        )

        # Find the detection matching our tag ID
        for det in detections:
            if det.tag_id == self._tag_id and det.pose_t is not None:
                tx, ty, tz = det.pose_t.flatten()
                # tz is depth (forward), tx is lateral
                distance = float(np.sqrt(tx**2 + tz**2))
                bearing = float(np.arctan2(tx, tz))  # positive = tag is to the right
                return 1.0, bearing, min(distance, 10.0)

        return 0.0, 0.0, 0.0

    def _ground_truth_dist(self) -> float:
        robot_pos, _ = self._robot_xyyaw()
        return float(np.linalg.norm(self._goal_pos - robot_pos))

    def _build_obs(self, base_obs, visible, bearing, tag_dist):
        vision = np.array([visible, bearing, tag_dist], dtype=np.float32)
        return np.concatenate([base_obs, vision]).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        base_obs = self._inner.reset()
        self._spawn_tag()
        self._prev_dist = self._ground_truth_dist()

        gray = self._get_camera_image()
        visible, bearing, tag_dist = self._detect_tag(gray)
        return self._build_obs(base_obs, visible, bearing, tag_dist), {}

    def step(self, action):
        base_obs, _task_reward, done, info = self._inner.step(action)

        gray = self._get_camera_image()
        visible, bearing, tag_dist = self._detect_tag(gray)

        curr_dist = self._ground_truth_dist()
        reward = (self._prev_dist - curr_dist) / self._inner.env_time_step
        self._prev_dist = curr_dist

        # Penalise steps where the tag is not visible
        if not visible:
            reward -= 1.0

        success = curr_dist < self.GOAL_RADIUS
        if success:
            reward += 50.0
            done = True

        info = dict(info) if info else {}
        info["success"] = success
        info["dist_to_goal"] = curr_dist
        info["tag_visible"] = bool(visible)

        obs = self._build_obs(base_obs, visible, bearing, tag_dist)
        return obs, reward, bool(done), False, info

    def render(self):
        return self._inner.render("rgb_array")

    def close(self):
        self._inner.close()
