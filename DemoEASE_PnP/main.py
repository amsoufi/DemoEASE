import math
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
import pybullet as p
import pybullet_data


def main():
    net_arch = [128, 512, 128]  # [64, 128, 256, 512, 256, 128, 64]
    ep_len = 1200
    repeat = 5

    env_fn = lambda: gym.make('kinova4dof-v0', gui=True, ep_len=ep_len)
    ppo(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU), steps_per_epoch=5 * ep_len,
        epochs=200, max_ep_len=ep_len, logger_kwargs=dict(output_dir='data/trial4', exp_name='kinova'), save_freq=100)

    # env_fn = lambda: gym.make('kinova4dof-v0', gui=False, ep_len=ep_len)
    # ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
    #      steps_per_epoch=5*ep_len, epochs=1000, max_ep_len=ep_len, start_steps=2*ep_len,
    #      update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=250,
    #      logger_kwargs=dict(output_dir='data/trial_'+time.strftime("%b%d_%H:%M"), exp_name='kinova'),
    #      save_freq=100, lambda_bc=2)

    # lambdas = [1, 2, 5]
    # for lambda_bc in lambdas:
    #     for iteration in range(repeat):
    #         logger_kwargs = dict(output_dir='data/trial_' + time.strftime("%b%d_%H:%M") +
    #                                         '/lambda_' + str(lambda_bc) + '/iter' + str(iteration), exp_name='kinova')
    #
    #         env_fn = lambda: gym.make('kinova4dof-v0', gui=False, ep_len=ep_len)
    #
    #         ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
    #              steps_per_epoch=5*ep_len, epochs=500, max_ep_len=ep_len, start_steps=2*ep_len,
    #              update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=200,
    #              logger_kwargs=logger_kwargs, save_freq=100, lambda_bc=lambda_bc)

    # demos = [100, 200, 400]
    # for demo in demos:
    #     for iteration in range(repeat):
    #         logger_kwargs = dict(output_dir='data/trial_' + time.strftime("%b%d_%H:%M") +
    #                                         '/num_demos_' + str(demo) + '/iter' + str(iteration), exp_name='kinova')
    #
    #         env_fn = lambda: gym.make('kinova4dof-v0', gui=False, ep_len=ep_len)
    #
    #         ddpg(env_fn=env_fn, ac_kwargs=dict(hidden_sizes=net_arch, activation=nn.ReLU),
    #              steps_per_epoch=5*ep_len, epochs=500, max_ep_len=ep_len, start_steps=2*ep_len,
    #              update_after=ep_len, update_every=int(0.05*ep_len), demo_start_episodes=demo,
    #              logger_kwargs=logger_kwargs, save_freq=100, lambda_bc=2)


def test():
    ac = torch.load('data/trial2/ddpg_model.pt')
    ac.pi.eval()
    env = gym.make('kinova4dof-v0', gui=True, ep_len=6000)

    verify_reset = True
    if verify_reset:
        for _ in range(5):
            o = env.reset()
            print("Resetting...")
            d = False
            while not d:
                o, _, d, _ = env.step(np.array([]))
                print("Joint Speeds: ", o[12:16])

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

    print("Success rate = ", 100 * success / runs, "%")

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


def load_gripper():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(fileName="plane.urdf", basePosition=[0, 0, 0])

    f_name_gripper = os.path.join(os.path.dirname(__file__), 'kinova_sim/resources/kinovaGen3_4DOF.urdf')
    gripper = p.loadURDF(fileName=f_name_gripper, basePosition=[0, 0, 0], useFixedBase=1)
    base_id = 8

    for i in range(p.getNumJoints(gripper)):
        info = p.getJointInfo(gripper, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = info[2]
        linkName = info[12]
        ls = p.getLinkState(gripper, i)
        linkOrig = [x * 100 for x in list(ls[4])]
        print((jointID, jointName, jointType, linkName, linkOrig))

    p.setJointMotorControlArray(bodyUniqueId=gripper, jointIndices=[i for i in range(p.getNumJoints(gripper))],
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=[0 for _ in range(p.getNumJoints(gripper))],
                                forces=[0 for _ in range(p.getNumJoints(gripper))])

    cid = p.createConstraint(parentBodyUniqueId=gripper, parentLinkIndex=base_id + 5, childBodyUniqueId=gripper,
                             childLinkIndex=base_id + 3, jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                             parentFramePosition=[7.776548502369529e-15, 0.03760000318288803, 0.0429999977350235],
                             childFramePosition=[0.0, -0.017901099286973476, 0.006516002118587494])
    p.changeConstraint(cid, maxForce=1e6)

    cid = p.createConstraint(parentBodyUniqueId=gripper, parentLinkIndex=base_id + 10, childBodyUniqueId=gripper,
                             childLinkIndex=base_id + 8, jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                             parentFramePosition=[7.776548502369529e-15, 0.03760000318288803, 0.0429999977350235],
                             childFramePosition=[0.0, -0.017901099286973476, 0.006516002118587494])
    p.changeConstraint(cid, maxForce=1e6)

    cid = p.createConstraint(parentBodyUniqueId=gripper, parentLinkIndex=base_id + 1, childBodyUniqueId=gripper,
                             childLinkIndex=base_id + 6, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                             parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
    p.changeConstraint(cid, gearRatio=-1, maxForce=1e6, erp=1)

    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    t = 0

    arm_lower_limits = [-np.inf, -2.41, -2.66, -2.23]
    arm_upper_limits = [-theta for theta in arm_lower_limits]
    arm_joint_ranges = [arm_upper_limits[i] - arm_lower_limits[i] for i in range(len(arm_lower_limits))]
    arm_rest_poses = [0 / 180 * np.pi, 15 / 180 * np.pi, -130 / 180 * np.pi, 55 / 180 * np.pi]

    p.resetJointState(gripper, 1, arm_rest_poses[0])
    p.resetJointState(gripper, 2, arm_rest_poses[1])
    p.resetJointState(gripper, 3, arm_rest_poses[2])
    p.resetJointState(gripper, 5, arm_rest_poses[3])

    orn = p.getQuaternionFromEuler((np.pi * (0.5 + 90/180), 0, np.pi * (0.5 + 45 / 180)))
    j = p.calculateInverseKinematics(gripper, 19, targetPosition=(0.3, 0.3, 0.5), targetOrientation=orn,
                                     lowerLimits=arm_lower_limits, upperLimits=arm_upper_limits,
                                     jointRanges=arm_joint_ranges, restPoses=arm_rest_poses,
                                     maxNumIterations=1000, residualThreshold=0.001)

    while True:
        p.stepSimulation()
        t += 0.01
        theta = 0.5 * abs(math.cos(t))
        p.setJointMotorControl2(bodyIndex=gripper, jointIndex=base_id + 1, controlMode=p.POSITION_CONTROL,
                                targetPosition=theta, force=100)

        p.setJointMotorControlArray(bodyIndex=gripper, jointIndices=[1, 2, 3, 5], controlMode=p.POSITION_CONTROL,
                                    targetPositions=[j[0], j[1], j[2], j[3]], forces=[39, 39, 39, 9])

        # p.setJointMotorControlArray(bodyIndex=gripper, jointIndices=[1, 2, 3, 5], controlMode=p.POSITION_CONTROL,
        #                             targetPositions=arm_rest_poses, forces=[39, 39, 39, 9])
        # ls = p.getLinkState(gripper, 19)
        # print(ls[4], p.getEulerFromQuaternion(ls[5]))

        time.sleep(0.001)


if __name__ == '__main__':
    main()
    # test()
    # load_gripper()
