# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from kukaCamGymEnv import KukaCamGymEnv

from stable_baselines.ddpg.policies import CnnPolicy
from stable_baselines import DDPG

import datetime


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    # print("totalt")
    # print(totalt)
    is_solved = totalt > 2000 and total >= 10
    return is_solved


def main():
    env = KukaCamGymEnv(renders=True, isDiscrete=True)
    model = DDPG(CnnPolicy, env, verbose=1, tensorboard_log='./tensorboard/')
    model.learn(total_timesteps=100, log_interval=1)
    print("Saving model to kuka_cam_model_tr.pkl")
    model.save("./savepoints/kuka_cam_model_tr.pkl")



if __name__ == '__main__':
    main()
