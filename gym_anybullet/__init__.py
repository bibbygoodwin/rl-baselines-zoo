from gym.envs.registration import register

register(
	id='ANYmalSimple-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalSimple',
)

register(
	id='ANYmalHistory-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistory',
)

register(
	id='ANYmalHistoryND-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistoryND',
)

register(
	id='ANYmalHistoryNC-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistoryNC',
)

register(
	id='ANYmalHistoryRecover-v0',
	entry_point='gym_anybullet.envs.anymal_recover_envs:ANYmalHistoryRecover',
)

register(
	id='ANYmalHistoryStable-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistoryStable',
)

register(
	id='CassieSimple-v0',
	entry_point='gym_anybullet.envs.cassie_envs:CassieSimple',
)

register(
	id='ANYmalHistoryNCROS-v0',
	entry_point='gym_anybullet.envs.anymal_ros_envs:ANYmalHistoryNC',
)

register(
	id='ANYmalHistoryNCMod-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistoryNCMod',
)

register(
	id='ANYmalHistoryNCDiff-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistoryNCDiff',
)

register(
	id='ANYmalHistoryNDNoise-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistoryNDNoise',
)

register(
	id='ANYmalPhaseX-v0',
	entry_point='gym_anybullet.envs.anymal_envs:ANYmalHistoryNCPhaseX',
)

# --- STEERABLE ENVIRONMENTS ---
register(
	id='ANYmalSteerableVel-v0',
	entry_point='gym_anybullet.envs.anymal_steerable_envs:ANYmalHistoryNC',
)

register(
	id='ANYmalHistory3-v0',
	entry_point='gym_anybullet.envs.anymal_steerable_envs:ANYmalHistory3',
)

register(
	id='ANYmalHistory3Steer-v0',
	entry_point='gym_anybullet.envs.anymal_steerable_envs:ANYmalHistory3Steer',
)

register(
	id='HopperBulletPhase-v0',
	entry_point='gym_anybullet.bullet_gym_locomotion.gym_locomotion_envs:HopperBulletPhaseEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HopperBullet-v0',
	entry_point='gym_anybullet.bullet_gym_locomotion.gym_locomotion_envs:HopperBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HopperBulletPhaseSanity-v0',
	entry_point='gym_anybullet.bullet_gym_locomotion.gym_locomotion_envs:HopperBulletPhaseSanityEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HopperBulletPhase-v2',
	entry_point='gym_anybullet.bullet_gym_locomotion.gym_locomotion_envs:HopperBulletPhaseEnv2',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)