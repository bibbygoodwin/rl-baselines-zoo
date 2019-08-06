import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import numpy as np
from common.paths import MODELS_PATH
import time


class ANYmalHistoryRecover(gym.Env):
	def __init__(self, render=False):
		self._observation = []

		self.observation_space = spaces.Box(-1 * np.array([np.inf] * 66), np.array([np.inf] * 66), dtype=np.float32)

		self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
		actions_low = np.ones(12) * -3.14
		actions_high = np.ones(12) * 3.14

		self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

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

		self.env_step_counter = 0
		self.quadruped_pos = self.quadruped_start_pos
		self.quadruped_orientation = self.quadruped_start_orientation

		joint_torques = []

		for j in self.quadruped_joint_ids:
			joint_torques.append(p.getJointState(self.quadruped_id, j)[3])

		self.prev_torques = np.asarray(joint_torques)

		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		self.quadruped_pos, self.quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)

		action = np.clip(np.asarray(action[:]), self.action_space.low, self.action_space.high)
		self._perform_action(action)

		p.stepSimulation()

		self.prev_action = action

		if self.env_step_counter % 200 == 0:
			external_force = ((np.random.rand(3) * 2) - 1) * 15000
			external_force[2] = external_force[2] / 4
			p.applyExternalForce(self.quadruped_id, -1, external_force, [0, 0, 0], p.LINK_FRAME)
			p.stepSimulation()

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

		joint_torques = []
		joint_states = []

		for j in self.quadruped_joint_ids:
			joint_torques.append(p.getJointState(self.quadruped_id, j)[3])
			joint_states.append(p.getJointState(self.quadruped_id, j)[0])

		quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

		if quadruped_pos[2] > 0.35 and quadruped_pos[2] < 0.6:
			pos_rew = np.abs(0.5 - quadruped_pos[2])
		else:
			pos_rew = 0

		reward = 2 * pos_rew - 0.1 * np.abs(quadruped_orientation[0]) - 0.1 * np.abs(quadruped_orientation[1]) \
				- 0.3 * np.abs(0.5 - quadruped_pos[2]) \
				- 0.0001 * np.linalg.norm(np.asarray(joint_torques)) \
				- 0.0001 * np.linalg.norm(self.prev_torques - joint_torques) \
				- 0.01 * np.linalg.norm(np.asarray(self.quadruped_joint_angles) - np.asarray(joint_states))

		self.prev_torques = np.asarray(joint_torques)

		return reward

	def _compute_done(self):
		quadruped_pos, quadruped_orientation = p.getBasePositionAndOrientation(self.quadruped_id)
		quadruped_orientation = p.getEulerFromQuaternion(quadruped_orientation)

		done = bool(quadruped_pos[2] < 0.35)
		done = bool(done or np.abs(quadruped_orientation[0]) >= np.pi / 2)
		done = bool(done or np.abs(quadruped_orientation[1]) >= np.pi / 2)

		return bool(done or self.env_step_counter >= 2048)

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
			p.setRealTimeSimulation(0)

		time.sleep(0.01)
