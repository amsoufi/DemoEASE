import pybullet as r
import os
import numpy as np
import operator
import math
from math import inf
from collections import namedtuple


class Robot:
    def __init__(self, client, rori):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'kinovaGen3_4DOF.urdf')
        self.robot = self.client.loadURDF(fileName=f_name,
                                          basePosition=[0, 0, 0],
                                          useFixedBase=1,
                                          flags=r.URDF_USE_SELF_COLLISION | r.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        self.eef_id = 19
        self.active_joints = [1, 2, 3, 5]
        self.arm_num_dofs = len(self.active_joints)
        self.joint_goal = None
        self.task_goal = None
        self.goal_num = 0

        self.reset(rori)

        # Unlock joints limits
        self.client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.active_joints,
            controlMode=r.VELOCITY_CONTROL,
            forces=[0, 0, 0, 0])

        obs0 = self.get_observation()
        self.PrevPos = obs0[:3]

    def reset(self, rori):
        self.goal_num = 0
        self.client.resetJointState(self.robot, self.active_joints[0], rori[0])
        self.client.resetJointState(self.robot, self.active_joints[1], rori[1])
        self.client.resetJointState(self.robot, self.active_joints[2], rori[2])
        self.client.resetJointState(self.robot, self.active_joints[3], rori[3])

    def get_ids(self):
        return self.robot, self.client

    def apply_action(self, action):
        # Expects action to be 4 dimensional
        action = np.clip(action, -1, 1)
        t1, t2, t3, t5 = action

        # Affine velocity values to reasonable values
        vel_gain = 1  # 0.5
        t1 *= 1.39 * vel_gain
        t2 *= 1.39 * vel_gain
        t3 *= 1.39 * vel_gain
        t5 *= 1.22 * vel_gain

        # Set the velocity of the joints directly
        self.client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.active_joints,
            controlMode=r.VELOCITY_CONTROL,
            targetVelocities=[t1, t2, t3, t5],
            forces=[39., 39., 39., 9.])

    def get_observation(self):
        ls = self.client.getLinkState(self.robot, linkIndex=self.eef_id, computeLinkVelocity=1)
        js = self.client.getJointStates(self.robot, jointIndices=self.active_joints)

        js_angle = (js[0][0], js[1][0], js[2][0], js[3][0])
        js_omega = (js[0][1], js[1][1], js[2][1], js[3][1])

        # Tip Position, Orientation + Joint Angles, Velocity
        observation = ls[4] + ls[5] + js_angle + tuple(np.cos(np.array(js_angle))) + tuple(
            np.sin(np.array(js_angle))) + js_omega

        try:
            self.client.addUserDebugLine(self.PrevPos, ls[4], [1, 0, 1], 2, 0)
        except AttributeError:
            pass
        self.PrevPos = ls[4]

        return observation

    def get_joint_torque(self):
        js = self.client.getJointStates(self.robot, jointIndices=self.active_joints)
        js_torque = (js[0][3] / 39, js[1][3] / 39, js[2][3] / 39, js[3][3] / 9)
        return js_torque

    def get_tool_ori(self):
        return tuple(self.client.getLinkState(self.robot, linkIndex=self.eef_id, computeLinkVelocity=1)[1])

    def get_pos_and_ori(self):
        base_pos, base_ori = self.client.getBasePositionAndOrientation(self.robot)
        return base_pos, base_ori

    def inv_kin(self, goal_pos, block_ob):
        arm_lower_limits = [-inf, -2.41, -2.66, -2.23]
        arm_upper_limits = [-theta for theta in arm_lower_limits]
        arm_joint_ranges = [arm_upper_limits[i] - arm_lower_limits[i] for i in range(len(arm_lower_limits))]
        arm_rest_poses = [45 / 180 * np.pi, 0, 0, 0]

        ls = self.client.getLinkState(self.robot, linkIndex=self.eef_id, computeLinkVelocity=1)
        tool_pos = ls[4]
        tool_pos = list(tool_pos)
        block_pos = list(block_ob)
        block_size = block_pos[-1]
        block_pos = [block_pos[0], block_pos[1], block_pos[2]]

        g_0 = [tool_pos[0], tool_pos[1], block_pos[2] + block_size + 0.3]
        j_0 = self.client.calculateInverseKinematics(self.robot, self.eef_id,
                                                     targetPosition=g_0,
                                                     lowerLimits=arm_lower_limits, upperLimits=arm_upper_limits,
                                                     jointRanges=arm_joint_ranges, restPoses=arm_rest_poses,
                                                     maxNumIterations=1000, residualThreshold=0.0001)

        j_0 = list(j_0)
        for i in range(len(arm_lower_limits)):
            if j_0[i] > 0:
                temp = j_0[i] % (2 * np.pi)
                temp = np.clip(temp, 0, arm_upper_limits[i])
            else:
                temp = (-j_0[i]) % (2 * np.pi)
                temp = -np.clip(temp, 0, arm_upper_limits[i])
            j_0[i] = temp

        j_0 = tuple(j_0)

        g_1 = [goal_pos[0], goal_pos[1], block_pos[2] + block_size + 0.3]
        j_1 = self.client.calculateInverseKinematics(self.robot, self.eef_id,
                                                     targetPosition=g_1,
                                                     lowerLimits=arm_lower_limits, upperLimits=arm_upper_limits,
                                                     jointRanges=arm_joint_ranges, restPoses=arm_rest_poses,
                                                     maxNumIterations=1000, residualThreshold=0.0001)

        j_1 = list(j_1)
        for i in range(len(arm_lower_limits)):
            if j_1[i] > 0:
                temp = j_1[i] % (2 * np.pi)
                temp = np.clip(temp, 0, arm_upper_limits[i])
            else:
                temp = (-j_1[i]) % (2 * np.pi)
                temp = -np.clip(temp, 0, arm_upper_limits[i])
            j_1[i] = temp

        j_1 = tuple(j_1)

        g_des = goal_pos
        j_des = self.client.calculateInverseKinematics(self.robot, self.eef_id,
                                                       targetPosition=g_des,
                                                       lowerLimits=arm_lower_limits, upperLimits=arm_upper_limits,
                                                       jointRanges=arm_joint_ranges, restPoses=arm_rest_poses,
                                                       maxNumIterations=1000, residualThreshold=0.0001)

        j_des = list(j_des)
        for i in range(len(arm_lower_limits)):
            if j_des[i] > 0:
                temp = j_des[i] % (2 * np.pi)
                temp = np.clip(temp, 0, arm_upper_limits[i])
            else:
                temp = (-j_des[i]) % (2 * np.pi)
                temp = -np.clip(temp, 0, arm_upper_limits[i])
            j_des[i] = temp

        j_des = tuple(j_des)

        self.joint_goal = (j_0, j_1, j_des)
        self.task_goal = [g_0, g_1, g_des]

        return self.joint_goal

    def get_control_feedback(self):
        js = self.client.getJointStates(self.robot, jointIndices=self.active_joints)

        js_angle = (js[0][0], js[1][0], js[2][0], js[3][0])
        js_omega = (js[0][1], js[1][1], js[2][1], js[3][1])

        ls = self.client.getLinkState(self.robot, linkIndex=self.eef_id, computeLinkVelocity=1)
        tool_pos = ls[4]
        tool_pos = list(tool_pos)
        goal_dist = [tool_pos[i] - self.task_goal[self.goal_num][i] for i in range(len(tool_pos))]
        goal_dist = abs(np.linalg.norm(goal_dist))

        if goal_dist < 0.2 and self.goal_num < 2:
            self.goal_num += 1
        setpoint = self.joint_goal[self.goal_num]

        fb = dict(e_pos=np.array(tuple(map(operator.sub, js_angle, setpoint)), dtype=np.float32),
                  e_vel=np.array(js_omega, dtype=np.float32))

        return fb
