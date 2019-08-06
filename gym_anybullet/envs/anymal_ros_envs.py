import gym
from gym import spaces
import numpy as np
import zmq


class ANYmalHistoryNC(gym.Env):
	def __init__(self):
		self._observation = []

		self.observation_space = spaces.Box(-1 * np.array([np.inf] * 66), np.array([np.inf] * 66), dtype=np.float32)

		self.quadruped_joint_angles = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]
		actions_low = np.asarray([-0.09, -0.2, -1.4, -0.15, -0.2, -1.4, -0.09, -1.0, 0.2, -0.15, -1.0, 0.2])
		actions_high = np.asarray([0.15, 1.0, -0.2, 0.09, 1.0, -0.2, 0.15, 0.2, 1.4, 0.09, 0.2, 1.4])

		self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

		context = zmq.Context()
		self.socket = context.socket(zmq.REQ)
		self.socket.connect("tcp://localhost:5555")

		self.quadruped_start_pos = [0, 0, 0.48]
		self.quadruped_start_orientation = [0, 0, 0]

		self.prev_joint_states = np.zeros((4, 12))
		self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
		self.prev_joint_states[-1, :] = self.quadruped_joint_angles

		self.prev_action = self.quadruped_joint_angles

		self.quadruped_pos = np.array(self.quadruped_start_pos)
		self.quadruped_orientation = np.array(self.quadruped_start_orientation)
		self.quadruped_joint_states = np.array(self.quadruped_joint_angles)

		self.env_step_counter = 0

	def step(self, action):
		action = np.clip(np.asarray(action[:]), self.action_space.low, self.action_space.high)
		self._perform_action(action)

		self.prev_action = action
		self._observation = self._compute_observation()
		reward = self._compute_reward()
		done = self._compute_done()

		self.env_step_counter += 1

		return np.array(self._observation), reward, done, {}

	def reset(self):
		self.env_step_counter = 0
		self._observation = self._compute_observation()

		return np.array(self._observation)

	def _perform_action(self, action):
		for i in range(len(action)):
			if (i - 2) % 3 == 0:
				action[i] -= 0.4
		self.socket.send_string(np.array2string(action, formatter={'float_kind': lambda action: "%.4f" % action})[1:-1])
		incoming_message = self.socket.recv_string()
		incoming_message = np.fromstring(incoming_message, dtype=np.float, sep=' ')

		self.quadruped_pos = np.array(incoming_message[0:3]).flatten()
		self.quadruped_orientation = np.array(incoming_message[3:6]).flatten()
		self.quadruped_joint_states = np.array(incoming_message[6:]).flatten()

	def _compute_observation(self):
		self.prev_joint_states[:-1, :] = self.prev_joint_states[1:, :]
		self.prev_joint_states[-1, :] = self.quadruped_joint_states

		observations = np.concatenate([self.quadruped_pos, self.quadruped_orientation,
										self.prev_joint_states.flatten(), self.prev_action])

		return observations

	def _compute_reward(self):
		return 0

	def _compute_done(self):
		return False

	def render(self, mode='human', close=False):
		pass
