import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import numpy as np
from common.paths import MODELS_PATH
import time


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class ANYmalHistoryNC(gym.Env):
    """
    An env
    NC = No Constraints?
    """
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 66), np.array([np.inf] * 66), dtype=np.float32)

        self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
        # actions_low = np.asarray([-0.09, -0.2, -1.4, -0.15, -0.2, -1.4, -0.09, -1.0, 0.2, -0.15, -1.0, 0.2])
        # actions_high = np.asarray([0.15, 1.0, -0.2, 0.09, 1.0, -0.2, 0.15, 0.2, 1.4, 0.09, 0.2, 1.4])
        actions_low = np.asarray([-0.12, -0.873, -0.873, -0.20, -0.873, -0.873, -0.12, -0.873, -0.873, -0.20, -0.873, -0.873])
        actions_high = np.asarray([0.20, 0.873, 0.873, 0.12, 0.873, 0.873, 0.20, 0.873, 0.873, 0.12, 0.873, 0.873])

        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        self.timestep = 0.01
        self.render_mode = render
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)

        self.plane_id = p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        self.quadruped_start_pos = [0, 0, 0.5]
        self.quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.prev_joint_states = np.zeros((4, 12))
        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = self.quadruped_joint_angles

        self.prev_action = self.quadruped_joint_angles

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       self.quadruped_start_pos, self.quadruped_start_orientation)

        p.setPhysicsEngineParameter(numSolverIterations=100)
        self.quadruped_joint_ids = []

        active_joint = 0
        for j in range(p.getNumJoints(self.quadruped_id)):
            p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.quadruped_id, j)
            joint_type = info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                self.quadruped_joint_ids.append(j)
                p.resetJointState(self.quadruped_id, j, self.quadruped_joint_angles[active_joint])
                active_joint += 1

        self.feet_ids = {'LF':5, 'RF':10, 'LH':15, 'RH':20}
        self.env_step_counter = 0
        self.quadruped_pos = self.quadruped_start_pos
        self.quadruped_orientation = self.quadruped_start_orientation

        joint_torques = []

        for j in self.quadruped_joint_ids:
            joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

        self.prev_torques = np.asarray(joint_torques)

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

        action = np.clip(np.asarray(action[:]), self.action_space.low, self.action_space.high)
        self._perform_action(action)
        p.stepSimulation()

        self.prev_action = action
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self.env_step_counter += 1

        return np.array(self._observation), reward, done, {}

    def reset(self):
        self.env_step_counter = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        quadruped_start_pos = [0, 0, 0.5]
        quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       quadruped_start_pos, quadruped_start_orientation)

        active_joint = 0
        for j in self.quadruped_joint_ids:
            p.resetJointState(self.quadruped_id, j, self.quadruped_joint_angles[active_joint])
            active_joint += 1

        self._observation = self._compute_observation()

        return np.array(self._observation)

    def _perform_action(self, action):
        i = 0
        for j in self.quadruped_joint_ids:
            p.setJointMotorControl2(self.quadruped_id, j, p.POSITION_CONTROL, action[i], force=40)
            i += 1

    def _compute_observation(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = joint_states

        observations = np.concatenate([quadruped_pos, quadruped_orientation,
                                       self.prev_joint_states.flatten(), self.prev_action])

        return observations

    def _compute_reward(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_linear_vel, quadruped_angular_vel = p.getBaseVelocity(self.quadruped_id)

        joint_torques = []
        for j in self.quadruped_joint_ids:
            joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

        vel_x = quadruped_linear_vel[0]
        vel_y = quadruped_linear_vel[1]
        vel_yaw = quadruped_angular_vel[2]

        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        if vel_x < 0.7:
            rew_vel_x = vel_x
        else:
            rew_vel_x = 1.4 - vel_x

        reward = 1 * rew_vel_x - 0.01 * np.abs(vel_y) \
                 - 0.01 * np.abs(vel_yaw) \
                 - 0.01 * np.abs(quadruped_orientation[0]) - 0.01 * np.abs(quadruped_orientation[1]) \
                 - 0.005 * np.abs(0.5 - quadruped_pos[2]) \
                 - 0.00001 * np.linalg.norm(np.asarray(joint_torques)) \
                 - 0.0001 * np.linalg.norm(self.prev_torques - joint_torques)

        self.prev_torques = np.asarray(joint_torques)

        return reward

    def _get_velocity(self):
        quadruped_linear_vel, quadruped_angular_vel = p.getBaseVelocity(self.quadruped_id)
        vel_x = quadruped_linear_vel[0]
        vel_y = quadruped_linear_vel[1]
        return np.sqrt(vel_x**2+vel_y**2)

    def _get_foot_contacts(self):
        LF = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['LF']) == () else 1
        RF = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['RF']) == () else 1
        LH = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['LH']) == () else 1
        RH = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['RH']) == () else 1
        return LF, RF, LH, RH

    def _compute_done(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_linear_vel, _ = p.getBaseVelocity(self.quadruped_id)
        vel_x = quadruped_linear_vel[0]
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        done = bool(quadruped_pos[2] < 0.3)
        done = bool(done or np.abs(quadruped_orientation[0]) >= np.pi / 4)
        done = bool(done or np.abs(quadruped_orientation[1]) >= np.pi / 4)
        done = bool(done or np.abs(quadruped_orientation[2]) >= np.pi / 4)
        done = bool(done or vel_x > 1)
        done = bool(done or self.env_step_counter >= 4096)

        return done

    def render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(0.01)
            p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

            self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                           self.quadruped_start_pos, self.quadruped_start_orientation)

            p.setPhysicsEngineParameter(numSolverIterations=100)
            self.quadruped_joint_ids = []

            for j in range(p.getNumJoints(self.quadruped_id)):
                p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
                info = p.getJointInfo(self.quadruped_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                    self.quadruped_joint_ids.append(j)
            p.setRealTimeSimulation(1)

        time.sleep(0.01)


class ANYmalHistory3(ANYmalHistoryNC):
    def __init__(self, *args, **kwargs):
        super(ANYmalHistory3, self).__init__(*args, **kwargs)

        self.prev_joint_states = np.zeros((3, 12))
        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 42), np.array([np.inf] * 42), dtype=np.float32)

    def _perform_action(self, action):
        i = 0
        for j in self.quadruped_joint_ids:
            p.setJointMotorControl2(self.quadruped_id, j, p.POSITION_CONTROL, action[i], force=40)
            i += 1

    def _compute_observation(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = joint_states

        observations = np.concatenate([quadruped_pos, quadruped_orientation,
                                       self.prev_joint_states.flatten()])

        return observations

    def _compute_reward(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_linear_vel, quadruped_angular_vel = p.getBaseVelocity(self.quadruped_id)

        joint_torques = []
        for j in self.quadruped_joint_ids:
            joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

        vel_x = quadruped_linear_vel[0]
        vel_y = quadruped_linear_vel[1]
        vel_yaw = quadruped_angular_vel[2]

        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        if vel_x < 0.7:
            rew_vel_x = vel_x
        else:
            rew_vel_x = 1.4 - vel_x

        reward = 1 * rew_vel_x - 0.01 * np.abs(vel_y) \
                 - 0.01 * np.abs(vel_yaw) \
                 - 0.01 * np.abs(quadruped_orientation[0]) - 0.01 * np.abs(quadruped_orientation[1]) \
                 - 0.0001 * np.linalg.norm(self.prev_torques - joint_torques)

        self.prev_torques = np.asarray(joint_torques)

        return reward

class ANYmalHistory3Steer(ANYmalHistory3):
    def __init__(self, *args, **kwargs):
        super(ANYmalHistory3, self).__init__(*args, **kwargs)

        self.goal_velocity_low = 0.2
        self.goal_velocity_high = 0.9
        self.target_velocity = np.random.uniform(self.goal_velocity_low, self.goal_velocity_high)

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 55), np.array([np.inf] * 55), dtype=np.float32)
        self.prev_joint_states = np.zeros((3, 18))


    def _compute_observation(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, 0:12] = joint_states
        self.prev_joint_states[-1, 12:15] = quadruped_pos
        self.prev_joint_states[-1, 15:18] = quadruped_orientation

        observations = np.concatenate([self.prev_joint_states.flatten(), np.array(self.target_velocity).reshape(1)])

        return observations

    def reset(self):
        self.env_step_counter = 0
        self.target_velocity = np.random.uniform(self.goal_velocity_low, self.goal_velocity_high)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        quadruped_start_pos = [0, 0, 0.5]
        quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       quadruped_start_pos, quadruped_start_orientation)

        active_joint = 0
        for j in self.quadruped_joint_ids:
            p.resetJointState(self.quadruped_id, j, self.quadruped_joint_angles[active_joint])
            active_joint += 1

        self._observation = self._compute_observation()

        return np.array(self._observation)


    def _compute_reward(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_linear_vel, quadruped_angular_vel = p.getBaseVelocity(self.quadruped_id)

        joint_torques = []
        for j in self.quadruped_joint_ids:
            joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

        vel_x = quadruped_linear_vel[0]
        vel_y = quadruped_linear_vel[1]
        vel_yaw = quadruped_angular_vel[2]

        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        if vel_x < self.target_velocity:
            rew_vel_x = vel_x
        else:
            rew_vel_x = (2 * self.target_velocity) - vel_x

        reward = 1 * rew_vel_x - 0.01 * np.abs(vel_y) \
                 - 0.01 * np.abs(vel_yaw) \
                 - 0.01 * np.abs(quadruped_orientation[0]) - 0.01 * np.abs(quadruped_orientation[1]) \
                 - 0.0001 * np.linalg.norm(self.prev_torques - joint_torques)

        self.prev_torques = np.asarray(joint_torques)

        return reward