import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import numpy as np
from common.paths import MODELS_PATH
import time


class CassieSimple(gym.Env):
	def __init__(self, render=False):
		self._observation = []
		self.observation_space = spaces.Box(-1 * np.array([np.inf] * 26), np.array([np.inf] * 26), dtype=np.float32)
		self.action_space = spaces.Box(-3.14 * np.ones(14), 3.14 * np.ones(14), dtype=np.float32)
		self.render_mode = render

		if render:
			self.physics_client = p.connect(p.GUI)
		else:
			self.physics_client = p.connect(p.DIRECT)

		p.resetSimulation()
		p.setGravity(0, 0, -9.81)
		p.setTimeStep(0.02)

		p.loadURDF(MODELS_PATH + 'plane/plane.urdf')
		self.humanoid_start_pos = [0, 0, 0.8]
		self.humanoid_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self.humanoid_id = p.loadURDF(MODELS_PATH + "cassie/urdf/cassie_collide.urdf", [0, 0, 0.8], useFixedBase=False)
		self.joint_ids = []

		p.setPhysicsEngineParameter(numSolverIterations=100)
		p.changeDynamics(self.humanoid_id, -1, linearDamping=0, angularDamping=0)

		self.joint_angles = [0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0]

		active_joint = 0
		for j in range(p.getNumJoints(self.humanoid_id)):
			p.changeDynamics(self.humanoid_id, j, linearDamping=0, angularDamping=0)
			info = p.getJointInfo(self.humanoid_id, j)

			joint_type = info[2]
			if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
				self.joint_ids.append(j)
				p.resetJointState(self.humanoid_id, j, self.joint_angles[active_joint])
				active_joint += 1

		self.env_step_counter = 0
		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		self.humanoid_pos, self.humanoid_orientation = p.getBasePositionAndOrientation(self.humanoid_id)
		self._perform_action(action)
		p.stepSimulation()
		self._observation = self._compute_observation()
		reward = self._compute_reward()
		done = self._compute_done()

		self.env_step_counter += 1

		return np.array(self._observation), reward, done, {}

	def reset(self):
		self.env_step_counter = 0

		p.resetBasePositionAndOrientation(self.humanoid_id, self.humanoid_start_pos, self.humanoid_start_orientation)
		p.resetBaseVelocity(self.humanoid_id, [0, 0, 0], [0, 0, 0])

		active_joint = 0
		for j in self.joint_ids:
			p.resetJointState(self.humanoid_id, j, self.joint_angles[active_joint])
			active_joint += 1

		self._observation = self._compute_observation()

		return np.array(self._observation)

	def _perform_action(self, action):
		i = 0
		for j in self.joint_ids:
			p.setJointMotorControl2(self.humanoid_id, j, p.POSITION_CONTROL, action[i], force=140)
			i += 1

	def _compute_observation(self):
		humanoid_pos, humanoid_orientation = p.getBasePositionAndOrientation(self.humanoid_id)
		humanoid_orientation = p.getEulerFromQuaternion(humanoid_orientation)
		humanoid_linear_vel, humanoid_angular_vel = p.getBaseVelocity(self.humanoid_id)

		joint_states = []

		for j in self.joint_ids:
			joint_states.append(p.getJointState(self.humanoid_id, j)[0])

		return np.concatenate([humanoid_pos, humanoid_orientation,
		                       humanoid_linear_vel, humanoid_angular_vel, joint_states])

	def _compute_reward(self):
		humanoid_pos, humanoid_orientation = p.getBasePositionAndOrientation(self.humanoid_id)
		humanoid_linear_vel, _ = p.getBaseVelocity(self.humanoid_id)

		joint_torques = []

		for j in self.joint_ids:
			joint_torques.append(p.getJointState(self.humanoid_id, j)[3])

		vel_x = humanoid_linear_vel[0]
		vel_y = humanoid_linear_vel[1]

		humanoid_orientation = p.getEulerFromQuaternion(humanoid_orientation)

		return 0.4 + 0.1 * vel_x - 0.002 * np.square(vel_y) \
				- 0.0001 * np.square(humanoid_orientation[0]) - 0.0001 * np.square(humanoid_orientation[1]) \
				- 0.0001 * np.square(humanoid_orientation[2]) - 0.01 * np.square(0.8 - humanoid_pos[2]) \
				- 0.001 * np.linalg.norm(np.asarray(joint_torques))

	def _compute_done(self):
		humanoid_pos, humanoid_orientation = p.getBasePositionAndOrientation(self.humanoid_id)
		humanoid_linear_vel, _ = p.getBaseVelocity(self.humanoid_id)
		vel_x = humanoid_linear_vel[0]
		humanoid_orientation = p.getEulerFromQuaternion(humanoid_orientation)

		done = bool(humanoid_pos[2] < 0.3)
		done = bool(done or np.abs(humanoid_orientation[0]) >= np.pi / 2)
		done = bool(done or np.abs(humanoid_orientation[1]) >= np.pi / 2)
		done = bool(done or np.abs(humanoid_orientation[2]) >= np.pi / 2)
		done = bool(done or vel_x > 2)
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
			self.humanoid_start_pos = [0, 0, 0.8]
			self.humanoid_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
			self.humanoid_id = p.loadURDF(MODELS_PATH + "cassie/urdf/cassie_collide.urdf", [0, 0, 0.8],
			                              useFixedBase=False)
			self.joint_ids = []

			p.setPhysicsEngineParameter(numSolverIterations=100)
			p.changeDynamics(self.humanoid_id, -1, linearDamping=0, angularDamping=0)

			self.joint_angles = [0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0]

			active_joint = 0
			for j in range(p.getNumJoints(self.humanoid_id)):
				p.changeDynamics(self.humanoid_id, j, linearDamping=0, angularDamping=0)
				info = p.getJointInfo(self.humanoid_id, j)

				joint_type = info[2]
				if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
					self.joint_ids.append(j)
					p.resetJointState(self.humanoid_id, j, self.joint_angles[active_joint])
					active_joint += 1

			p.setRealTimeSimulation(1)
		time.sleep(0.02)
