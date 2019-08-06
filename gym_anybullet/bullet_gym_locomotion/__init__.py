# import gym
# from gym.envs.registration import registry, make, spec
# def register(id,*args,**kvargs):
# 	if id in registry.env_specs:
# 		return
# 	else:
# 		return gym.envs.registration.register(id,*args,**kvargs)
#
# # ------------bullet-------------
#
# register(
# 	id='HopperBulletEnv-v0',
# 	entry_point='gym_locomotion_envs:HopperBulletEnv',
# 	max_episode_steps=1000,
# 	reward_threshold=2500.0
# 	)
