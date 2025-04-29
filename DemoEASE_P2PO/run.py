import torch
import kinova_sim
from ppo import ppo
from ddpg import ddpg
import torch.nn as nn
import gym
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def main(timestamp, repeat, param):
    net_arch = [256, 1024, 256]
    ep_len = 500
    num_epoch = 500
    env_fn = lambda: gym.make('kinova4dof-v0', gui=False, ep_len=ep_len)

    # logger_kwargs = dict(output_dir='data/' + timestamp +
    #                      '/num_demos_' + str(param) + '/iter' + str(repeat), exp_name='kinova')

    # ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
    #      steps_per_epoch=5*ep_len, epochs=10, max_ep_len=ep_len, start_steps=2*ep_len,
    #      update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=param,
    #      logger_kwargs=logger_kwargs, save_freq=100, lambda_bc=2)

    logger_kwargs = dict(output_dir='data/' + timestamp +
                                    '/lambda_' + str(param) + '/iter' + str(repeat), exp_name='kinova')

    ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
         steps_per_epoch=5 * ep_len, epochs=num_epoch, max_ep_len=ep_len, start_steps=2 * ep_len,
         update_after=ep_len, update_every=int(0.05 * ep_len), demo_start_episodes=200,
         logger_kwargs=logger_kwargs, save_freq=100, lambda_bc=param)


if __name__ == '__main__':
    main(timestamp=sys.argv[1], repeat=int(sys.argv[2]), param=float(sys.argv[3]))
    
