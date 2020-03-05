# add parent dir to find package. Only needed for source code build, pip install doesn't fbklneed it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from kukaCamGymEnv import KukaCamGymEnv

from stable_baselines import DQN


def main():
    env = KukaCamGymEnv(renders=True, isDiscrete=False)
    model = DDPG.load("./savepoints/kuka_cam_model_tr.pkl")
    obs = env.reset()
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
          action, _states = model.predict(obs)
          print()
          print("action ", action)
          print()
          obs, reward, done, info = env.step(action)
          env.render()
        print("Episode reward", reward)

if __name__ == '__main__':
    main()
