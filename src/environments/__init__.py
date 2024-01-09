from gym.envs.registration import register

register(
    id="Chain-v0",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=10,
    kwargs={"length": 10},
)

register(
    id="Chain-v1",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=25,
    kwargs={"length": 25},
)

register(
    id="Chain-v2",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=50,
    kwargs={"length": 50},
)

register(
    id="Chain-v3",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=100,
    kwargs={"length": 100},
)

register(
    id="ChainLoop-v0",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=400,
    kwargs={"loop": True,
            "length": 10},
)

register(
    id="ChainLoop-v1",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=400,
    kwargs={"loop": True,
            "length": 25},
)

register(
    id="ChainLoop-v2",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=400,
    kwargs={"loop": True,
            "length": 50},
)

register(
    id="ChainLoop-v3",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=400,
    kwargs={"loop": True,
            "length": 100},
)

register(
    id="OrderedChain-v0",
    entry_point="src.environments.chain:Chain",
    max_episode_steps=10,
    kwargs={"ordered": True}
)

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

register(
    id='FrozenLakeNotSlippery-v1',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
    max_episode_steps=400,
    reward_threshold=0.78,  # optimum = .8196
)
