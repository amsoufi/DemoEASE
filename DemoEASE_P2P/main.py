import torch
import kinova_sim
from ppo import ppo
from ddpg import ddpg
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt


def main():
    net_arch = [128, 512, 128]  # [64, 128, 256, 512, 256, 128, 64]
    ep_len = 400

    env_fn = lambda: gym.make('kinova4dof-v0', gui=True, ep_len=ep_len)
    logger_kwargs = dict(output_dir='data/trial', exp_name='kinova')
    ppo(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU), steps_per_epoch=5*ep_len,
        epochs=200, max_ep_len=ep_len, logger_kwargs=logger_kwargs, save_freq=50)

    repeat = 5
    # lambdas = [0, 0.6, 1.2, 1.8]
    lambdas = [0.1, 0.3]
    # demos = [0, 60, 180]

    # for demo in demos:
    # for lambda_bc in lambdas:
    #     for iteration in range(repeat):
            # logger_kwargs = dict(output_dir='data/run11_log/lambda_' + str(lambda_bc) + '/iter' + str(iteration),
            #                      exp_name='kinova')

            # logger_kwargs = dict(output_dir='data/run12_log/num_demos_' + str(demo) + '/iter' + str(iteration),
            #                      exp_name='kinova')

            # env_fn = lambda: gym.make('kinova4dof-v0', gui=False)
            #
            # ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
            #      steps_per_epoch=5*ep_len, epochs=250, max_ep_len=ep_len, start_steps=2*ep_len,
            #      update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=100,  # 100
            #      logger_kwargs=logger_kwargs, save_freq=50, lambda_bc=lambda_bc)

            # ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
            #      steps_per_epoch=5 * ep_len, epochs=2000, max_ep_len=ep_len, start_steps=2 * ep_len,
            #      update_after=ep_len, update_every=int(0.05 * ep_len), demo_start_episodes=demo,
            #      logger_kwargs=logger_kwargs, save_freq=200, lambda_bc=1)

    # _, get_action = load_policy_and_env('path/to/output_dir')
    # env = gym.make('kinova-v0')
    #
    # run_policy(env, get_action)


def test():
    ac = torch.load('data/run6_log/num_demos_150/iter0/ddpg_model.pt')
    ac.pi.eval()
    env = gym.make('kinova4dof-v0', gui=True, ep_len=400)

    verify_reset = False
    if verify_reset:
        for _ in range(100):
            o = env.reset()
            print("Resetting...")
            d = False
            while not d:
                o, _, d, _ = env.step(np.array([0, 0, 0, 0]))
                # print("Joint Speeds: ", o[12:16])

    runs = 10
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

    verify_plot = True
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
        # ac = torch.load('data/run10_log/lambda_1.8/iter' + str(iter) + '/ddpg_model.pt')
        ac = torch.load('data/run6_log/num_demos_150/iter' + str(iter) + '/ddpg_model.pt')
        ac.pi.eval()
        env = gym.make('kinova4dof-v0', gui=False, ep_len=600)
        # env = gym.make('kinova4dof-v0', gui=True, ep_len=600)

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
