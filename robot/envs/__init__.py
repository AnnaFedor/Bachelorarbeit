import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------


register(
    id='KukaBulletEnv-v0',
    entry_point='pybullet_envs.bullet:KukaGymEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='KukaCamBulletEnv-v0',
    entry_point='pybullet_envs.bullet:KukaCamGymEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='KukaDiverseObjectGrasping-v0',
    entry_point='pybullet_envs.bullet:KukaDiverseObjectEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)


def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
    return btenvs
