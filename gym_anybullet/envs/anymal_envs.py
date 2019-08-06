import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import numpy as np
from common.paths import MODELS_PATH
import time


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class ANYmalSimple(gym.Env):
    def __init__(self, render=False):
        self._observation = []
        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 24), np.array([np.inf] * 24), dtype=np.float32)
        self.action_space = spaces.Box(-3.14 * np.ones(12), 3.14 * np.ones(12), dtype=np.float32)
        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.02)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        self.quadruped_start_pos = [0, 0, 0.58]
        self.quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       self.quadruped_start_pos, self.quadruped_start_orientation,
                                       flags=p.URDF_USE_SELF_COLLISION)

        p.setPhysicsEngineParameter(numSolverIterations=100)
        self.quadruped_joint_ids = []

        active_joint = 0
        for j in range(p.getNumJoints(self.quadruped_id)):
            p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.quadruped_id, jointIndex=j)
            joint_type = info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                self.quadruped_joint_ids.append(j)
                p.resetJointState(self.quadruped_id, j, self.quadruped_joint_angles[active_joint])
                active_joint += 1

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        self._perform_action(action)
        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self.env_step_counter += 1

        # return np.array(self._observation), reward, done, self.env_step_counter
        return np.array(self._observation), reward, done, {}


    def reset(self):
        # self.env_step_counter = 0
        #
        # p.resetSimulation()
        # p.setGravity(0, 0, -9.81)
        # p.setTimeStep(0.02)
        # p.loadURDF(MODELS_PATH + 'plane/plane.urdf')
        #
        # quadruped_start_pos = [0, 0, 0.6]
        # quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        #
        # self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
        #                                quadruped_start_pos, quadruped_start_orientation,
        #                                flags=p.URDF_USE_SELF_COLLISION)

        self.env_step_counter = 0
        p.resetBasePositionAndOrientation(self.quadruped_id, self.quadruped_start_pos, self.quadruped_start_orientation)
        p.resetBaseVelocity(self.quadruped_id, [0, 0, 0], [0, 0, 0])

        active_joint = 0
        for j in self.quadruped_joint_ids:
            p.resetJointState(self.quadruped_id, j, self.quadruped_joint_angles[active_joint])
            active_joint += 1

        self._observation = self._compute_observation()

        return np.array(self._observation)

    def _perform_action(self, action):
        i = 0
        for j in self.quadruped_joint_ids:
            p.setJointMotorControl2(self.quadruped_id, j, p.POSITION_CONTROL, action[i], force=30)
            i += 1

    def _compute_observation(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)
        quadruped_linear_vel, quadruped_angular_vel = p.getBaseVelocity(self.quadruped_id)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        return np.concatenate([quadruped_pos, quadruped_orientation,
                               quadruped_linear_vel, quadruped_angular_vel, joint_states])

    def _compute_reward(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_linear_vel, _ = p.getBaseVelocity(self.quadruped_id)

        joint_torques = []

        for j in self.quadruped_joint_ids:
            joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

        vel_x = quadruped_linear_vel[0]
        vel_y = quadruped_linear_vel[1]

        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        return 1 + 0.5 * vel_x - 0.008 * np.square(vel_y) \
               - 0.08 * np.square(quadruped_orientation[0]) - 0.1 * np.square(quadruped_orientation[1]) \
               - 0.04 * np.square(quadruped_orientation[2]) - 0.04 * np.square(0.58 - quadruped_pos[2]) \
               - 0.04 * np.linalg.norm(np.asarray(joint_torques))

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
        done = bool(done or self.env_step_counter >= 8192)

        return done

    def render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True
            self.physics_client = p.connect(p.GUI)
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(0.02)
            p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

            quadruped_start_pos = [0, 0, 0.6]
            quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

            self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                           quadruped_start_pos, quadruped_start_orientation,
                                           flags=p.URDF_USE_SELF_COLLISION)

            p.setPhysicsEngineParameter(numSolverIterations=100)
            self.quadruped_joint_ids = []

            for j in range(p.getNumJoints(self.quadruped_id)):
                p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
                info = p.getJointInfo(self.quadruped_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                    self.quadruped_joint_ids.append(j)
            p.setRealTimeSimulation(1)
        time.sleep(0.02)


class ANYmalHistory(gym.Env):
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 69), np.array([np.inf] * 69), dtype=np.float32)

        self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
        actions_low = np.asarray(self.quadruped_joint_angles) - 1
        actions_high = actions_low + 2
        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.02)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        self.quadruped_start_pos = [0, 0, 0.5]
        self.quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.prev_joint_states = np.zeros((4, 12))
        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = self.quadruped_joint_angles

        self.prev_action = self.quadruped_joint_angles

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       self.quadruped_start_pos, self.quadruped_start_orientation,
                                       flags=p.URDF_USE_SELF_COLLISION)

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

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
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
        p.setTimeStep(0.02)
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
            p.setJointMotorControl2(self.quadruped_id, j, p.POSITION_CONTROL, action[i], force=30)
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
                                       self.prev_joint_states.flatten(), self.prev_action, [0.8, 0, 0]])

        return observations

    def _compute_reward(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_linear_vel, _ = p.getBaseVelocity(self.quadruped_id)

        joint_torques = []

        for j in self.quadruped_joint_ids:
            joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

        vel_x = quadruped_linear_vel[0]
        vel_y = quadruped_linear_vel[1]

        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        return 0.5 - 0.1 * np.square(0.8 - vel_x) - 0.0008 * np.square(vel_y) \
               - 0.004 * np.square(quadruped_orientation[0]) - 0.004 * np.square(quadruped_orientation[1]) \
               - 0.004 * np.square(quadruped_orientation[2]) - 0.0004 * np.square(0.5 - quadruped_pos[2]) \
               - 0.0004 * np.linalg.norm(np.asarray(joint_torques))

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
        done = bool(done or vel_x < -0.2)
        done = False
        done = bool(done or self.env_step_counter >= 8192)

        return done

    def render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True
            self.physics_client = p.connect(p.GUI)
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(0.02)
            p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

            quadruped_start_pos = [0, 0, 0.6]
            quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

            self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                           quadruped_start_pos, quadruped_start_orientation,
                                           flags=p.URDF_USE_SELF_COLLISION)

            p.setPhysicsEngineParameter(numSolverIterations=100)
            self.quadruped_joint_ids = []

            for j in range(p.getNumJoints(self.quadruped_id)):
                p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
                info = p.getJointInfo(self.quadruped_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                    self.quadruped_joint_ids.append(j)
            p.setRealTimeSimulation(1)
        time.sleep(0.02)


class ANYmalHistoryND(gym.Env):
    """
    ND = Not done (i.e. no early termination
    """
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 69), np.array([np.inf] * 69), dtype=np.float32)

        self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
        actions_low = np.asarray([-0.06, 0.0, -1.2, -0.12, 0.0, -1.2, -0.06, -0.8, 0.4, -0.12, -0.8, 0.4])
        actions_high = np.asarray([0.12, 0.8, -0.4, 0.06, 0.8, -0.4, 0.12, -0.0, 1.2, 0.09, 0.0, 1.2])

        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        self.command = np.random.rand(3) * 2 - 1
        self.command[2] = self.command[2] * 10

        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

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

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._perform_action(action)
        p.stepSimulation()

        if self.env_step_counter % 2048 == 0:
            self.command = np.random.rand(3) * 2 - 1
            self.command[2] = self.command[2] * 10
            print(self.command)

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
        p.setTimeStep(0.01)
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
                                       self.prev_joint_states.flatten(), self.prev_action, self.command])

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

        return -0.1 * np.abs(self.command[0] - vel_x) - 0.1 * np.abs(self.command[1] - vel_y) \
               - 0.1 * np.abs(self.command[2] - vel_yaw) \
               - 0.005 * np.abs(quadruped_orientation[0]) - 0.005 * np.abs(quadruped_orientation[1]) \
               - 0.01 * np.abs(0.5 - quadruped_pos[2]) \
               - 0.005 * np.linalg.norm(np.asarray(joint_torques))

    def _compute_done(self):
        done = bool(self.env_step_counter >= 8392)

        return done

    def _render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True
            self.physics_client = p.connect(p.GUI)
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


class ANYmalHistoryStable(gym.Env):
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 69), np.array([np.inf] * 69), dtype=np.float32)

        self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
        actions_low = np.ones(12) * -0.1
        actions_high = np.ones(12) * 0.1
        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.02)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

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

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

        for _ in range(1):
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
        p.resetBasePositionAndOrientation(self.quadruped_id, self.quadruped_start_pos, self.quadruped_start_orientation)
        p.resetBaseVelocity(self.quadruped_id, [0, 0, 0], [0, 0, 0])

        active_joint = 0
        for j in self.quadruped_joint_ids:
            p.resetJointState(self.quadruped_id, j, self.quadruped_joint_angles[active_joint])
            active_joint += 1

        self._observation = self._compute_observation()

        return np.array(self._observation)

    def _perform_action(self, action):
        i = 0

        for j in self.quadruped_joint_ids:
            joint_state = p.getJointState(self.quadruped_id, j)[0]
            p.setJointMotorControl2(self.quadruped_id, j, p.POSITION_CONTROL, targetPosition=joint_state + action[i],
                                    force=30)
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
                                       self.prev_joint_states.flatten(), self.prev_action, [0.8, 0, 0]])

        return observations

    def _compute_reward(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_linear_vel, _ = p.getBaseVelocity(self.quadruped_id)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        return (-0.01 * np.linalg.norm(np.asarray(joint_states) - np.asarray(self.quadruped_joint_angles)) -
                1 * np.linalg.norm(np.asarray(quadruped_pos) - np.asarray(self.quadruped_start_pos)) -
                1 * np.linalg.norm(np.asarray(quadruped_orientation) - np.asarray(self.quadruped_orientation)))

    def _compute_done(self):
        done = bool(self.env_step_counter >= 16384)

        return done

    def render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True
            self.physics_client = p.connect(p.GUI)
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(0.02)
            p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

            quadruped_start_pos = [0, 0, 0.5]
            quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

            self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                           quadruped_start_pos, quadruped_start_orientation,
                                           flags=p.URDF_USE_SELF_COLLISION)

            p.setPhysicsEngineParameter(numSolverIterations=100)
            self.quadruped_joint_ids = []

            for j in range(p.getNumJoints(self.quadruped_id)):
                p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
                info = p.getJointInfo(self.quadruped_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                    self.quadruped_joint_ids.append(j)
            p.setRealTimeSimulation(1)
        time.sleep(0.02)


class ANYmalHistoryNC(gym.Env):
    """
    An env
    NC = No Constraints?
    """
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 66), np.array([np.inf] * 66), dtype=np.float32)

        self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
        actions_low = np.asarray([-0.09, -0.2, -1.4, -0.15, -0.2, -1.4, -0.09, -1.0, 0.2, -0.15, -1.0, 0.2])
        actions_high = np.asarray([0.15, 1.0, -0.2, 0.09, 1.0, -0.2, 0.15, 0.2, 1.4, 0.09, 0.2, 1.4])

        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
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
        # TODO do I need to set timestep again here? make it class variable
        self.plane_id = p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

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
        # print("Torque cost: "+str(- 0.0001 * np.linalg.norm(self.prev_torques - joint_torques)))

        self.prev_torques = np.asarray(joint_torques)
        # print("Vel rew: "+str(1 * rew_vel_x - 0.01 * np.abs(vel_y)) )

        return reward

    def _get_velocity(self):
        quadruped_linear_vel, quadruped_angular_vel = p.getBaseVelocity(self.quadruped_id)
        vel_x = quadruped_linear_vel[0]
        vel_y = quadruped_linear_vel[1]
        return np.sqrt(vel_x**2+vel_y**2)

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
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(0.01)
            self.plane_id = p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

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


class ANYmalHistoryNCPhaseX(ANYmalHistoryNC):
    def __init__(self, *args, **kwargs):
        super(ANYmalHistoryNCPhaseX, self).__init__(*args, **kwargs)

        self.target_vel = 0.7
        self._gait_period = 1/(1.06+(0.34*self.target_vel)) # seconds, from 25kg dog in Heglund et al 1988
        self.sim_timestep = p.getPhysicsEngineParameters()['fixedTimeStep'] # this is set in parent class
        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 67), np.array([np.inf] * 67), dtype=np.float32)
        self.feet_ids = {'LF':5, 'RF':10, 'LH':15, 'RH':20}

    def reset(self):
        self.env_step_counter = 0
        self.phase = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

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

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

        action = np.clip(np.asarray(action[:]), self.action_space.low, self.action_space.high)
        self._perform_action(action)
        p.stepSimulation()

        self.prev_action = action

        self.phase = (self.phase + (self.sim_timestep/self._gait_period))%1
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self.env_step_counter += 1

        return np.array(self._observation), reward, done, {}


    def _compute_observation(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = joint_states


        # observations = np.concatenate([quadruped_pos, quadruped_orientation,
        #                                self.prev_joint_states.flatten(), self.prev_action, np.reshape(self.phase,1)])
        observations = np.concatenate([quadruped_pos, quadruped_orientation,
                                       self.prev_joint_states.flatten(), self.prev_action, np.reshape([0],1)])

        return observations

    def _get_foot_contacts(self):
        LF = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['LF']) == () else 1
        RF = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['RF']) == () else 1
        LH = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['LH']) == () else 1
        RH = 0 if p.getContactPoints(self.quadruped_id, self.plane_id, linkIndexA=self.feet_ids['RH']) == () else 1
        return LF, RF, LH, RH

class ANYmalHistoryNCPhaseI(ANYmalHistoryNC):
    def __init__(self, *args, **kwargs):
        super(ANYmalHistoryNCPhaseI, self).__init__(*args, **kwargs)

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

    def intrinsic_phase(self, obs):
        """
        Computes phase based on intrinsic, heuristic parameters of the agent's movement
        Might require quite careful thought and tuning to implement succesfully
        """
        raise NotImplementedError
     # _compute_observation() returns:
     #    quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
     #    quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)
     #
     #    joint_states = []
     #
     #    for j in self.quadruped_joint_ids:
     #        joint_states.append(p.getJointState(self.quadruped_id, j)[0])
     #
     #    self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
     #    self.prev_joint_states[-1, :] = joint_states
     #
     #    observations = np.concatenate([quadruped_pos, quadruped_orientation,
     #                                   self.prev_joint_states.flatten(), self.prev_action])
     #
     #    return observations


class ANYmalHistoryNCMod(gym.Env):
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 66), np.array([np.inf] * 66), dtype=np.float32)

        self.quadruped_joint_angles = [-0.05, 0.8, -1.0, 0.05, 0.8, -1.0, -0.05, -0.8, 1.0, 0.05, -0.8, 1.0]

        self._actions_low = np.asarray([-0.08, -0.2, -2.0, 0.02, -0.2, -2.0, -0.08, -1.8, 0, 0.02, -1.8, 0])
        self._actions_high = np.asarray([-0.02, 1.8, 0, 0.08, 1.8, 0, 0.35, -0.02, 2.0, 0.08, 0.2, 2.0])

        self._hack_actions_low = -10 * np.ones(12)
        self._hack_actions_high = 10 * np.ones(12)

        self.action_space = spaces.Box(self._hack_actions_low, self._hack_actions_high, dtype=np.float32)

        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.05)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        self.quadruped_start_pos = [0, 0, 0.5]
        self.quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.prev_quadruped_pos = np.array([0, 0, 0])

        self.prev_joint_states = np.zeros((4, 12))
        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = self.quadruped_joint_angles

        self.prev_action = self.quadruped_joint_angles

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       self.quadruped_start_pos, self.quadruped_start_orientation)

        p.setPhysicsEngineParameter(numSolverIterations=25)
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

        self.env_step_counter = 0
        self.quadruped_pos = self.quadruped_start_pos
        self.quadruped_orientation = self.quadruped_start_orientation

        joint_torques = []

        for j in self.quadruped_joint_ids:
            joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

        self.prev_torques = np.asarray(joint_torques)

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

        action = np.clip(action, self._hack_actions_low, self._hack_actions_high)
        action = ((np.asarray(action) - self._hack_actions_low) / (self._hack_actions_high - self._hack_actions_low) *
                  (self._actions_high - self._actions_low)) + self._actions_low

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

        p.resetBasePositionAndOrientation(self.quadruped_id, self.quadruped_start_pos, self.quadruped_start_orientation)
        p.resetBaseVelocity(self.quadruped_id, [0, 0, 0], [0, 0, 0])

        active_joint = 0
        joint_angles = np.random.normal(self.quadruped_joint_angles, 0.1)

        for j in self.quadruped_joint_ids:
            p.resetJointState(self.quadruped_id, j, joint_angles[active_joint])
            active_joint += 1

        self._observation = self._compute_observation()

        return np.array(self._observation)

    def _perform_action(self, action):
        action = np.random.normal(action, 0.1)

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

        quadruped_pos = np.random.normal(quadruped_pos, 0.1)
        quadruped_orientation = np.random.normal(quadruped_orientation, 0.1)
        joint_states = np.random.normal(joint_states, 0.1)

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
        # vel_y = quadruped_linear_vel[1]
        # vel_yaw = quadruped_angular_vel[2]

        # quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        if vel_x < 0.7:
            rew_vel_x = vel_x
        else:
            rew_vel_x = 1.4 - vel_x

        reward = 0.2 + rew_vel_x + 0.5 * (quadruped_pos[0] - self.prev_quadruped_pos[0])

        self.prev_torques = np.asarray(joint_torques)
        self.prev_quadruped_pos = np.asarray(quadruped_pos)

        return reward

    def _compute_done(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

        done = bool(abs(quadruped_pos[1]) > 5)
        done = bool(done or abs(quadruped_orientation[0]) > np.pi / 4)
        done = bool(done or abs(quadruped_orientation[1]) > np.pi / 4)
        done = bool(done or abs(quadruped_orientation[2]) > np.pi / 2)
        done = bool(done or abs(quadruped_pos[2]) < 0.3)

        return bool(done or self.env_step_counter >= 4096)

    def render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True

            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(0.05)
            p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

            self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                           self.quadruped_start_pos, self.quadruped_start_orientation)

            p.setPhysicsEngineParameter(numSolverIterations=25)
            self.quadruped_joint_ids = []

            for j in range(p.getNumJoints(self.quadruped_id)):
                p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
                info = p.getJointInfo(self.quadruped_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                    self.quadruped_joint_ids.append(j)
            p.setRealTimeSimulation(1)

        time.sleep(0.05)


class ANYmalHistoryNCDiff(gym.Env):
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 66), np.array([np.inf] * 66), dtype=np.float32)

        self.quadruped_joint_angles = [-0.05, 0.8, -1.0, 0.05, 0.8, -1.0, -0.05, -0.8, 1.0, 0.05, -0.8, 1.0]

        self.joints_low = np.asarray([-0.08, 0.2, -1.6, 0.02, 0.2, -1.6, -0.08, -1.4, 0.4, 0.02, -1.4, 0.4])
        self.joints_high = np.asarray([-0.02, 1.4, -0.4, 0.08, 1.4, -0.4, -0.02, -0.2, 1.6, 0.08, -0.2, 1.6])

        actions_low = np.ones(12) * -0.2
        actions_high = np.ones(12) * 0.2

        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.02)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        self.quadruped_start_pos = [0, 0, 0.5]
        self.quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.prev_joint_states = np.zeros((4, 12))
        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = self.quadruped_joint_angles

        self.prev_action = self.quadruped_joint_angles

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       self.quadruped_start_pos, self.quadruped_start_orientation)

        p.setPhysicsEngineParameter(numSolverIterations=25)
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
        p.setTimeStep(0.02)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

        quadruped_start_pos = [0, 0, 0.5]
        quadruped_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                       quadruped_start_pos, quadruped_start_orientation)

        active_joint = 0

        joint_angles = np.random.normal(self.quadruped_joint_angles, 0.1)

        for j in self.quadruped_joint_ids:
            p.resetJointState(self.quadruped_id, j, joint_angles[active_joint])
            active_joint += 1

        self._observation = self._compute_observation()

        return np.array(self._observation)

    def _perform_action(self, action):
        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        action = np.array(action) + np.array(joint_states)
        action = np.clip(action, self.joints_low, self.joints_high)
        action = np.random.normal(action, 0.05)

        i = 0
        for j in self.quadruped_joint_ids:
            p.setJointMotorControl2(self.quadruped_id, j, p.POSITION_CONTROL, action[i], force=30)
            i += 1

    def _compute_observation(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        quadruped_pos = np.random.normal(quadruped_pos, 0.1)
        quadruped_orientation = np.random.normal(quadruped_orientation, 0.1)
        joint_states = np.random.normal(joint_states, 0.1)

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

        reward = 5 * rew_vel_x - 0.05 * np.abs(vel_y) \
                 - 0.05 * np.abs(vel_yaw) \
                 - 0.05 * np.abs(quadruped_orientation[0]) - 0.05 * np.abs(quadruped_orientation[1]) \
                 - 0.25 * np.abs(0.5 - quadruped_pos[2]) - 0.05 * np.abs(quadruped_pos[1]) \
                 - 0.00001 * np.linalg.norm(np.asarray(joint_torques)) \
                 - 0.00005 * np.linalg.norm(self.prev_torques - joint_torques)

        self.prev_torques = np.asarray(joint_torques)

        return reward

    def _compute_done(self):
        return bool(self.env_step_counter >= 4096)

    def render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True

            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(0.02)
            p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

            self.quadruped_id = p.loadURDF(MODELS_PATH + 'anymal_boxy/anymal_boxy.urdf',
                                           self.quadruped_start_pos, self.quadruped_start_orientation)

            p.setPhysicsEngineParameter(numSolverIterations=25)
            self.quadruped_joint_ids = []

            for j in range(p.getNumJoints(self.quadruped_id)):
                p.changeDynamics(self.quadruped_id, j, linearDamping=0, angularDamping=0)
                info = p.getJointInfo(self.quadruped_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                    self.quadruped_joint_ids.append(j)
            p.setRealTimeSimulation(1)

        time.sleep(0.02)


class ANYmalHistoryNDNoise(gym.Env):
    def __init__(self, render=False):
        self._observation = []

        self.observation_space = spaces.Box(-1 * np.array([np.inf] * 69), np.array([np.inf] * 69), dtype=np.float32)

        self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
        self._actions_low = np.asarray([-0.06, 0.0, -1.2, -0.12, 0.0, -1.2, -0.06, -0.8, 0.4, -0.12, -0.8, 0.4])
        self._actions_high = np.asarray([0.12, 0.8, -0.4, 0.06, 0.8, -0.4, 0.12, -0.0, 1.2, 0.09, 0.0, 1.2])

        self._hack_actions_low = -100 * np.ones(12)
        self._hack_actions_high = 100 * np.ones(12)

        self.action_space = spaces.Box(self._hack_actions_low, self._hack_actions_high, dtype=np.float32)

        self.noise = 0
        self.current_step = 0
        self.total_steps = 100000000
        self.env_step_counter = 0

        self.command = np.random.rand(3) * 2 - 1
        self.render_mode = render

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.loadURDF(MODELS_PATH + 'plane/plane.urdf')

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

    def step(self, action):
        self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

        action = ((np.asarray(action) - self._hack_actions_low) / (self._hack_actions_high - self._hack_actions_low) *
                  (self._actions_high - self._actions_low)) + self._actions_low

        self._perform_action(action)
        p.stepSimulation()

        if self.env_step_counter % 2048 == 0:
            if self.current_step < self.total_steps / 500:
                multiplier = 500 * self.current_step / self.total_steps
            else:
                multiplier = 1

            self.command = (np.random.rand(3) * 2 - 1) * multiplier

        self.prev_action = action
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self.env_step_counter += 1
        self.current_step += 1

        self.noise = 0.15 * (self.current_step / self.total_steps)

        return np.array(self._observation), reward, done, {}

    def reset(self):
        self.env_step_counter = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
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

        action = np.random.normal(action, self.noise)

        for j in self.quadruped_joint_ids:
            p.setJointMotorControl2(self.quadruped_id, j, p.POSITION_CONTROL, action[i], force=30)
            i += 1

    def _compute_observation(self):
        quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
        quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

        joint_states = []

        for j in self.quadruped_joint_ids:
            joint_states.append(p.getJointState(self.quadruped_id, j)[0])

        quadruped_pos = np.random.normal(quadruped_pos, self.noise)
        quadruped_orientation = np.random.normal(quadruped_orientation, self.noise)
        joint_states = np.random.normal(joint_states, self.noise)

        self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
        self.prev_joint_states[-1, :] = joint_states

        observations = np.concatenate([quadruped_pos, quadruped_orientation,
                                       self.prev_joint_states.flatten(), self.prev_action, self.command])

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

        rew_vel_x = 0.2 * (gaussian(vel_x, self.command[0], 0.05) - 0.5)
        rew_vel_y = 0.2 * (gaussian(vel_y, self.command[1], 0.05) - 0.5)
        rew_vel_yaw = 0.2 * (gaussian(vel_yaw, self.command[2], 0.05) - 0.5)

        return rew_vel_x + rew_vel_y + rew_vel_yaw \
               - 0.005 * np.abs(quadruped_orientation[0]) - 0.005 * np.abs(quadruped_orientation[1]) \
               - 0.01 * np.abs(0.5 - quadruped_pos[2]) \
               - 0.005 * np.linalg.norm(np.asarray(joint_torques))

    def _compute_done(self):
        done = bool(self.env_step_counter >= 8392)

        return done

    def render(self, mode='human', close=False):
        if not self.render_mode:
            p.disconnect()
            self.render_mode = True
            self.physics_client = p.connect(p.GUI)
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
