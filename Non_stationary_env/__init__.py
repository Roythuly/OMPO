from gym.envs import register

register(
    id='HopperRandom-v0',
    entry_point='Non_stationary_env.hopper_random:HopperRandomEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperTransfer-v0',
    entry_point='Non_stationary_env.hopper_random:HopperTransferEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Walker2dRandom-v0',
    entry_point='Non_stationary_env.walker2d_random:Walker2dRandomEnv',
    max_episode_steps=1000
)

register(
    id='Walker2dTransfer-v0',
    entry_point='Non_stationary_env.walker2d_random:Walker2dTransferEnv',
    max_episode_steps=1000
)

register(
    id='AntTransfer-v0',
    entry_point='Non_stationary_env.ant_random:AntTransferEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='AntRandom-v0',
    entry_point='Non_stationary_env.ant_random:AntRandomEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HumanoidTransfer-v0',
    entry_point='Non_stationary_env.humanoid_random:HumanoidTransferEnv',
    max_episode_steps=1000
)

register(
    id='HumanoidRandom-v0',
    entry_point='Non_stationary_env.humanoid_random:HumanoidRandomEnv',
    max_episode_steps=1000
)