import pybullet
import puppersim.data as pd
from pybullet_utils import bullet_client
from puppersim.reacher import reacher_kinematics
from puppersim.reacher import reacher_robot_utils
import time
import math
import gymnasium as gym
import numpy as np
import random
from pupper_hardware_interface import interface
from serial.tools import list_ports
import os
from typing import Optional
from pybullet_utils import bullet_client as bc

KP = 6.0
KD = 1.0
MAX_CURRENT = 4.0
RENDER_HEIGHT = 360
RENDER_WIDTH = 480


class ReacherEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            run_on_robot=False,
            render=False,
            render_meshes=False,
            leg_index=3,
    ):

        self.action_space = gym.spaces.Box(
            np.array([-2 * math.pi, -1.5 * math.pi, -1.0 * math.pi]),
            np.array([2 * math.pi, 1.5 * math.pi, 1.0 * math.pi]),
            dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            # np.array([-1, -1, -1, -1, -1, -1, 0.05, 0.05, 0.05, -0.3, -0.3, -0.3]),
            # np.array([1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3]),
            # observation space range for target
            np.array([-0.1, -0.1, 0.05]),
            np.array([0.1, 0.1, 0.15]),
            dtype=np.float32)
        self._leg_index = leg_index
        self._is_render = render
        self.render_mode = render_mode

        self.target = np.array([0, 0, 0.1])
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._pybullet_client = None
        # if self._is_render:
        #   self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        # else:
        #   self._pybullet_client = bc.BulletClient()

        self._run_on_robot = run_on_robot
        if self._run_on_robot:
            serial_port = reacher_robot_utils.get_serial_port()
            self._hardware_interface = interface.Interface(serial_port)
            time.sleep(0.25)
            self._hardware_interface.set_joint_space_parameters(
                kp=KP, kd=KD, max_current=MAX_CURRENT)
        else:
            if self._is_render:
                self._pybullet_client = bullet_client.BulletClient(
                    connection_mode=pybullet.GUI)
                self._pybullet_client.configureDebugVisualizer(
                    self._pybullet_client.COV_ENABLE_GUI, 0)
                self._pybullet_client.resetDebugVisualizerCamera(
                    cameraDistance=0.3,
                    cameraYaw=-134,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 0.1])
            else:
                self._pybullet_client = bullet_client.BulletClient(
                    connection_mode=pybullet.DIRECT)

        if render_meshes:
            self.urdf_filename = "pupper_arm.urdf"
        else:
            self.urdf_filename = "pupper_arm_no_mesh.urdf"

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = np.array([0, 0, 0.1])
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=float(RENDER_WIDTH) /
                                                                              RENDER_HEIGHT,
                                                                       nearVal=0.1,
                                                                       farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def reset(self, target=None, seed=None, options=None):
        self.target = target if target is not None else np.array([0.00, 0.00, 0.1])

        if self._run_on_robot:
            reacher_robot_utils.blocking_move(self._hardware_interface,
                                              goal=np.zeros(3),
                                              traverse_time=2.0)
            obs = self._get_obs_on_robot()
        else:
            self._pybullet_client.resetSimulation()
            URDF_PATH = os.path.join(pd.getDataPath(), self.urdf_filename)
            self.robot_id = self._pybullet_client.loadURDF(URDF_PATH, useFixedBase=True)
            self._pybullet_client.setGravity(0, 0, -9.8)
            self.num_joints = self._pybullet_client.getNumJoints(self.robot_id)
            for joint_id in range(self.num_joints):
                # Disables the default motors in PyBullet.
                self._pybullet_client.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=self._pybullet_client.POSITION_CONTROL,
                    targetVelocity=0,
                    force=0)

            # self.target = np.random.uniform(0.05, 0.10, 3)
            # Target position: xy range of -0.1 to 0.1. z range of 0.05 to 0.15.
            # self.target = np.concatenate([np.random.uniform(-0.1, 0.1, 2), np.random.uniform(0.05, 0.15, 1)])

            # possible_targets = []
            # possible_targets.append(np.array([-0.07, -0.07, 0.07]))
            # possible_targets.append(np.array([0.07, 0.07, 0.07]))
            # possible_targets.append(np.array([-0.07, 0.07, 0.07]))
            # possible_targets.append(np.array([0.07, -0.07, 0.07]))
            # self.target = random.choice(possible_targets)

            # target_angles = np.random.uniform(-0.5*math.pi, 0.5*math.pi, 3)
            # self.target = reacher_kinematics.calculate_forward_kinematics_robot(target_angles)

            self._target_visual_shape = self._pybullet_client.createVisualShape(
                self._pybullet_client.GEOM_SPHERE, radius=0.015)
            self._target_visualization = self._pybullet_client.createMultiBody(
                baseVisualShapeIndex=self._target_visual_shape,
                basePosition=self.target)

            obs = self._get_obs()

        return obs, {}

    def setTarget(self, target):
        self.target = target

    def calculateInverseKinematics(self, target_pos):
        # compute end effector pos in cartesian cords given angles
        end_effector_link_id = self._get_end_effector_link_id()
        inverse_kinematics = self._pybullet_client.calculateInverseKinematics(
            self.robot_id, end_effector_link_id, target_pos)

        return inverse_kinematics

    def _apply_actions(self, actions):
        for joint_id, action in zip(range(self.num_joints), actions):
            # Disables the default motors in PyBullet.
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=action,
                maxVelocity=1000,
                positionGain=0.3)

    def _apply_actions_on_robot(self, actions):
        full_actions = np.zeros([3, 4])
        full_actions[:, self._leg_index] = np.reshape(actions, 3)

        self._hardware_interface.set_joint_space_parameters(kp=KP,
                                                            kd=KD,
                                                            max_current=MAX_CURRENT)
        self._hardware_interface.set_actuator_postions(np.array(full_actions))

    def _get_obs(self):
        joint_states = self._pybullet_client.getJointStates(
            self.robot_id, list(range(self.num_joints)))
        joint_angles = [joint_data[0] for joint_data in joint_states][0:3]
        joint_velocities = [joint_data[1] for joint_data in joint_states][0:3]
        return np.concatenate([
            # np.cos(joint_angles),
            # np.sin(joint_angles),
            self.target,
            # joint_velocities,
            # self._get_vector_from_end_effector_to_goal(),
        ])

    def _get_obs_on_robot(self):
        self._hardware_interface.read_incoming_data()
        self._robot_state = self._hardware_interface.robot_state
        joint_angles = self._robot_state.position[self._leg_index *
                                                  3:self._leg_index * 3 + 3]
        joint_velocities = self._robot_state.velocity[self._leg_index *
                                                      3:self._leg_index * 3 + 3]
        np.set_printoptions(precision=2)
        return np.concatenate([
            # np.cos(joint_angles),
            # np.sin(joint_angles),
            self.target,
            # joint_velocities,
            # self._get_vector_from_end_effector_to_goal(),
        ])

    def step(self, actions):
        if self._run_on_robot:
            self._apply_actions_on_robot(actions)
            ob = self._get_obs_on_robot()
        else:
            self._apply_actions(actions)
            ob = self._get_obs()
            self._pybullet_client.stepSimulation()

        reward_dist = -np.linalg.norm(
            self._get_vector_from_end_effector_to_goal()) ** 2
        reward_ctrl = 0
        reward = reward_dist + reward_ctrl

        done = False

        truncated = False

        return ob, reward, done, truncated, {}

    def _get_end_effector_link_id(self):
        for joint_id in range(self.num_joints):
            joint_name = self._pybullet_client.getJointInfo(self.robot_id, joint_id)[1]
            if joint_name.decode("UTF-8") == "leftFrontToe":
                return joint_id
        raise ValueError("leftFrontToe not found")

    def _get_vector_from_end_effector_to_goal(self):
        if self._run_on_robot:
            joint_angles = self._robot_state.position[self._leg_index * 3: self._leg_index * 3 + 3]
            end_effector_pos = reacher_kinematics.calculate_forward_kinematics_robot(
                joint_angles)
        else:
            end_effector_link_id = self._get_end_effector_link_id()
            end_effector_pos = self._pybullet_client.getLinkState(
                bodyUniqueId=self.robot_id,
                linkIndex=end_effector_link_id,
                computeForwardKinematics=1)[0]
            # print("end effector: ", end_effector_pos)
        return np.array(end_effector_pos) - np.array(self.target)

    def shutdown(self):
        # TODO: Added this function to attempt to gracefully close
        # the serial connection to the Teensy so that the robot
        # does not jerk, but it doesn't actually work
        self._hardware_interface.serial_handle.close()
