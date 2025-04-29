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
import gc


def main():
    net_arch = [256, 1024, 256]  # [128, 512, 128] or [64, 128, 256, 512, 256, 128, 64]
    ep_len = 500
    repeat = 5

    # env_fn = lambda: gym.make('kinova4dof-v0', gui=True, ep_len=ep_len)
    # ppo(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU), steps_per_epoch=5*ep_len,
    #     epochs=200, max_ep_len=ep_len, logger_kwargs=dict(output_dir='data/trialx', exp_name='kinova'), save_freq=50)

    # env_fn = lambda: gym.make('kinova4dof-v0', gui=False, ep_len=ep_len)
    # ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
    #      steps_per_epoch=5*ep_len, epochs=1000, max_ep_len=ep_len, start_steps=2*ep_len,
    #      update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=250,
    #      logger_kwargs=dict(output_dir='data/trial_'+time.strftime("%b%d_%H:%M"), exp_name='kinova'),
    #      save_freq=50, lambda_bc=2)

    lambdas = [0.5]
    timestamp = time.strftime("%m%d_%H:%M")
    for lambda_bc in lambdas:
        for iteration in range(repeat):
            logger_kwargs = dict(output_dir='data/' + timestamp +
                                            '/lambda_' + str(lambda_bc) + '/iter' + str(iteration), exp_name='kinova')

            env_fn = lambda: gym.make('kinova4dof-v0', gui=False, ep_len=ep_len)

            ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
                 steps_per_epoch=5*ep_len, epochs=500, max_ep_len=ep_len, start_steps=2*ep_len,
                 update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=200,
                 logger_kwargs=logger_kwargs, save_freq=100, lambda_bc=lambda_bc)

            gc.collect()

    # demos = [100, 200, 400]
    # demos = [400]
    # timestamp = time.strftime("%b_%d_%H:%M")
    # for demo in demos:
    #     for iteration in range(repeat):
    #         logger_kwargs = dict(output_dir='data/' + timestamp +
    #                                         '/num_demos_' + str(demo) + '/iter' + str(iteration), exp_name='kinova')
    #
    #         env_fn = lambda: gym.make('kinova4dof-v0', gui=False, ep_len=ep_len)
    #
    #         ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
    #              steps_per_epoch=5*ep_len, epochs=2000, max_ep_len=ep_len, start_steps=2*ep_len,
    #              update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=demo,
    #              logger_kwargs=logger_kwargs, save_freq=200, lambda_bc=2)


def test():
    ac = torch.load('data/trial0_local/ddpg_model.pt')
    ac.pi.eval()
    env = gym.make('kinova4dof-v0', gui=True, ep_len=600)

    verify_reset = False
    if verify_reset:
        for _ in range(5):
            o = env.reset()
            print("Resetting...")
            d = False
            while not d:
                o, _, d, _ = env.step(np.array([0, -0.1, 0, 0]))
                print("Joint Speeds: ", o[12:16])

    runs = 5
    action_seq = dict()
    torque_seq = dict()
    success = 0
    for i in range(runs):
        o = env.reset()
        d = False
        action_seq[i] = []
        torque_seq[i] = []

        while not d:
            action = ac.act(torch.tensor(o, dtype=torch.float32))
            o, r, d, info = env.step(action)
            t = env.get_torque()

            success += 1 if d and r > 0 else 0
            action_seq[i].append(list(o[12:16]))
            torque_seq[i].append(list(t))

    print("Success rate = ", 100*success/runs, "%")

    verify_plot = False
    if verify_plot:
        _, axs = plt.subplots(runs, 2)
        for i in range(runs):
            axs[i, 0].plot([x[0] for x in action_seq[i]])
            axs[i, 0].plot([x[1] for x in action_seq[i]])
            axs[i, 0].plot([x[2] for x in action_seq[i]])
            axs[i, 0].plot([x[3] for x in action_seq[i]])
            axs[i, 0].legend(['1', '2', '3', '4'])
            axs[i, 0].set_ylabel('Run #' + str(i) + ' Velocities')

            axs[i, 1].plot([x[0] for x in torque_seq[i]])
            axs[i, 1].plot([x[1] for x in torque_seq[i]])
            axs[i, 1].plot([x[2] for x in torque_seq[i]])
            axs[i, 1].plot([x[3] for x in torque_seq[i]])
            axs[i, 1].legend(['1', '2', '3', '4'])
            axs[i, 1].set_ylabel('Run #' + str(i) + ' Torques')

        plt.show()


def implement():
    for iter in [2]:  # range(5):
        # ac = torch.load('data/06_01_16:13/num_demos_100/iter' + str(iter) + '/ddpg_model.pt')
        # ac = torch.load('data/Old Trials/trial_Apr14_12:56/lambda_1/iter' + str(iter) + '/ddpg_model.pt')
        ac = torch.load('data/06_10_10:02/lambda_2/iter' + str(iter) + '/ddpg_model.pt')
        ac.pi.eval()
        env = gym.make('kinova4dof-v0', gui=False, ep_len=900)

        runs = 100
        action_seq = dict()
        torque_seq = dict()
        success = 0
        for i in range(runs):
            o = env.reset()
            d = False
            action_seq[i] = []
            torque_seq[i] = []

            while not d:
                # action = ac.act(torch.tensor(o, dtype=torch.float32))
                feedback = env.get_feedback()
                err_p = feedback['e_pos']
                err_v = feedback['e_vel']
                action = np.array(-0.4 * err_p - 0.2 * err_v)
                o, r, d, info = env.step(action)
                t = env.get_torque()
                t = np.linalg.norm([abs(ti) for ti in t])

                success += 1 if d and r > 0 else 0
                action_seq[i].append(list(o[12:16]))
                torque_seq[i].append(t)

            torque_seq[i] = sum(torque_seq[i]) / len(torque_seq[i])
        print("Average torque utility = ", 100 * (sum(torque_seq.values()) / len(torque_seq.values())) / 2, '%')
        print("Success rate = ", 100 * success / runs, "%")
        env.close()


if __name__ == '__main__':
    # main()
    # test()
    implement()
