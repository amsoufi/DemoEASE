import gym
import numpy as np
import math
import operator
import pybullet as p
from pybullet_utils import bullet_client
from kinova_sim.resources.robot_4DOF import Robot
from kinova_sim.resources.goal import Goal
from kinova_sim.resources.block import Block
from kinova_sim.resources.plane import Plane
import matplotlib.pyplot as plt


class KinovaEnv4DOF(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, ep_len=1000):

        sim_mode = p.GUI if gui else p.DIRECT
        self.client = bullet_client.BulletClient(connection_mode=sim_mode)  # ddpg: DIRECT, ppo: DIRECT or GUI
        self.client.resetSimulation()
        self.client.setGravity(0, 0, -9.8)
        self.np_random, _ = gym.utils.seeding.np_random()

        # Reduce length of episodes for RL algorithms
        self.sim_freq = 50
        self.act_freq = 50
        self.client.setTimeStep(1 / self.sim_freq)
        self.ep_len = ep_len
        self.rew_shape = {"sparse": 0, "dist": 1, "ctrl": 1}

        if gui:
            p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=30, cameraPitch=-70,
                                         cameraTargetPosition=[0, 0, 0])
            p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0)
            Plane(self.client)

        self.i = None
        self.ori = None
        self.tool_init_ori = None
        self.joint_goal = None
        self.goalv = None
        self.goalp = None
        self.collision_flag = False
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None

        self.robot = Robot(self.client, (0, 15 / 180 * np.pi, -130 / 180 * np.pi, 55 / 180 * np.pi))
        self.goal = Goal(self.client, (-10, 0, 0), (0, 0, 0))
        self.block = Block(self.client, base=(10, 0, 0), scale=1)

        self.action_space = gym.spaces.box.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))
        self.observation_space = gym.spaces.box.Box(-np.inf, np.inf, np.shape(self.reset()), dtype="float32")

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.robot.apply_action(action)

        robot_id, _ = self.robot.get_ids()
        block_id = self.block.get_ids()

        if len(self.client.getContactPoints(robot_id, block_id)) > 0:
            self.collision_flag = True

        robot_ob = self.robot.get_observation()
        goal_ob = self.goal.get_observation()
        block_ob = self.block.get_observation()
        # self.render("human")  # Render while using pybullet.DIRECT

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((robot_ob[0] - goal_ob[0]) ** 2 +
                                  (robot_ob[1] - goal_ob[1]) ** 2 +
                                  (robot_ob[2] - goal_ob[2]) ** 2))

        # _, misalignment = p.getAxisAngleFromQuaternion(p.getDifferenceQuaternion(self.robot.get_tool_ori(),
        #                                                                          self.tool_init_ori))
        # misalignment = 180/np.pi * np.arcsin(np.sin(misalignment))

        effort = self.robot.get_joint_torque()
        effort = np.linalg.norm(np.array(effort, dtype=np.float32))

        # reward_dist = - dist_to_goal * 0.001 - misalignment/100. * 0.001  # dist (cm), misalignment (deg)
        reward_dist = - dist_to_goal * 0.002
        reward_ctrl = - effort * 0.001

        # Calculating negative reward
        if self.rew_shape["sparse"]:
            reward = -0.001
        else:
            reward = self.rew_shape["dist"] * reward_dist + self.rew_shape["ctrl"] * reward_ctrl

        # Done by reaching goal
        # if dist_to_goal < 0.05 and misalignment < 5.0:
        if dist_to_goal < 0.05:
            self.done = True
            reward = 10

        # Done if number of steps is exceeded
        self.i += 1
        if self.i >= self.ep_len:
            self.done = True

        if self.collision_flag:
            reward = -2
            self.done = True

        for i in range(int(self.sim_freq / self.act_freq)):
            p.stepSimulation()

        ob = np.array(robot_ob[-16:] + goal_ob + block_ob + tuple(map(operator.sub, robot_ob[:3], goal_ob[:3])),
                      dtype=np.float32)

        return ob, reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, local=False):
        self.i = 0
        self.collision_flag = False

        # Set the goal to a random target
        if local:
            region = np.random.randint(4) - 1
            # region = 0
            pho = self.np_random.uniform(low=0.4, high=0.6, size=1)
            theta = self.np_random.uniform(45 / 180 * np.pi, 90 / 180 * np.pi) + region * np.pi/2
            z = self.np_random.uniform(low=0.35, high=0.55, size=1)
            thetab = theta - self.np_random.uniform(15 / 180 * np.pi, 30 / 180 * np.pi)
        else:
            region = 0
            pho = self.np_random.uniform(low=0.3, high=0.7, size=1)
            theta = self.np_random.uniform(-180 / 180 * np.pi, 180 / 180 * np.pi)
            z = self.np_random.uniform(low=0.25, high=0.65, size=1)
            thetab = theta + 2 * (self.np_random.randint(2) - 0.5) * self.np_random.uniform(15 / 180 * np.pi,
                                                                                            30 / 180 * np.pi)

        x = pho * math.cos(theta)
        y = pho * math.sin(theta)
        zb = z + self.np_random.uniform(low=-0.1, high=0.1, size=1)

        delta_ori = np.pi * 0
        ori1 = self.np_random.uniform(-delta_ori, delta_ori) - 0 / 180 * np.pi - region * np.pi/2
        ori2 = self.np_random.uniform(-delta_ori, delta_ori) + 15 / 180 * np.pi
        ori3 = self.np_random.uniform(-delta_ori, delta_ori) - 130 / 180 * np.pi
        ori5 = self.np_random.uniform(-delta_ori, delta_ori) + 55 / 180 * np.pi

        self.ori = (ori1, ori2, ori3, ori5)
        self.robot.reset(self.ori)

        goal_dynamic = False
        vdir = 2 * np.random.randint(0, 1) - 1
        v = (-y * vdir * self.np_random.uniform(0, 0.001),
             x * vdir * self.np_random.uniform(0, 0.001),
             self.np_random.uniform(-0.001, 0.0005))

        self.goalp = (x, y, z)
        self.goalv = v if goal_dynamic else (0, 0, 0)
        self.done = False

        # Visual element of the goal
        self.goal.reset(self.goalp, self.goalv)
        self.block.reset((pho*math.cos(thetab), pho*math.sin(thetab), zb))

        # self.record = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "record.mp4")

        # Get observation to return
        robot_ob = self.robot.get_observation()
        goal_ob = self.goal.get_observation()
        block_ob = self.block.get_observation()

        self.tool_init_ori = self.robot.get_tool_ori()
        self.joint_goal = self.robot.inv_kin([x, y, z], block_ob)

        self.client.removeAllUserDebugItems()

        ob = np.array(robot_ob[-16:] + goal_ob + block_ob + tuple(map(operator.sub, robot_ob[:3], goal_ob[:3])),
                      dtype=np.float32)
        return ob

    def render(self, mode='human'):

        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((1000, 1000, 4)))

        # Base information
        robot_id, client_id = self.robot.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = self.robot.get_pos_and_ori()
        newpos = [1, 2.5, 1]
        pos1 = []
        for j in range(3):
            pos1.append(pos[j] + newpos[j])

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [0, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos1, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(1000, 1000, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (1000, 1000, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

        return

    def get_feedback(self):
        robot_id, _ = self.robot.get_ids()
        block_id = self.block.get_ids()

        closest_points = self.client.getClosestPoints(robot_id, block_id, 1)
        if len(closest_points) > 0:
            dist_to_block = [point[8] for point in closest_points]
            dist_to_block = min(dist_to_block)
        else:
            dist_to_block = 1

        fb = self.robot.get_control_feedback()
        fb['block'] = dist_to_block

        return fb

    def get_torque(self):
        effort = self.robot.get_joint_torque()
        return effort

    def close(self):
        self.client.disconnect()
