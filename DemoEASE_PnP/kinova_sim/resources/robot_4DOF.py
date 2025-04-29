import time
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
                                          flags=r.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | r.URDF_USE_SELF_COLLISION)

        self.eef_id = 8
        self.tcp_id = 19
        self.active_joints = [1, 2, 3, 5]
        self.arm_num_dofs = len(self.active_joints)
        self.joint_goal = None
        self.task_goal = None
        self.goal_num = 0
        self.grip = -1

        numJoints = self.client.getNumJoints(self.robot)

        # for i in range(1, self.eef_id):
        #     self.client.setCollisionFilterPair(self.robot, self.robot, i, i+1, 1)
        #     for j in range(self.eef_id+1, numJoints):
        #         self.client.setCollisionFilterPair(self.robot, self.robot, i, j, 1)

        for i in range(self.eef_id+1, numJoints):
            for j in range(i+1, numJoints):
                self.client.setCollisionFilterPair(self.robot, self.robot, i, j, 0)

        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce',
                                'maxVelocity'])
        self.joints = []
        self.links = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = self.client.getJointInfo(self.robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            linkName = info[12]
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity)
            self.joints.append(info)
            self.links.append((jointID, jointName, jointType, linkName))

        # for link in self.links:
        #     print(link)
        #     ls = self.client.getLinkState(self.robot, linkIndex=link[0], computeLinkVelocity=1)
        #     print('Position = {}, Orientation = {}'.format(ls[4], ls[5]))

        for i in range(self.eef_id, numJoints):
            self.client.changeDynamics(self.robot, i, lateralFriction=1.0, spinningFriction=1.0,
                                       rollingFriction=0.0001, frictionAnchor=True)

        self.client.resetJointState(self.robot, self.active_joints[0], rori[0])
        self.client.resetJointState(self.robot, self.active_joints[1], rori[1])
        self.client.resetJointState(self.robot, self.active_joints[2], rori[2])
        self.client.resetJointState(self.robot, self.active_joints[3], rori[3])

        # Unlock joints limits
        self.client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[i for i in range(numJoints)],
            controlMode=r.VELOCITY_CONTROL,
            targetVelocities=[0 for _ in range(numJoints)],
            forces=[0 for _ in range(numJoints)])

        self.gripper_range = [0.02, 0.085]
        self.setup_constraints()

        obs0 = self.get_observation()
        self.PrevPos = obs0[:3]

    def setup_constraints(self):
        cid = self.client.createConstraint(parentBodyUniqueId=self.robot, parentLinkIndex=self.eef_id + 5,
                                           childBodyUniqueId=self.robot, childLinkIndex=self.eef_id + 3,
                                           jointType=r.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                                           parentFramePosition=[7.776548502369529e-15, 0.03760000318288803,
                                                                0.0429999977350235],
                                           childFramePosition=[0.0, -0.017901099286973476, 0.006516002118587494])
        self.client.changeConstraint(cid, maxForce=1e6)

        cid = self.client.createConstraint(parentBodyUniqueId=self.robot, parentLinkIndex=self.eef_id + 10,
                                           childBodyUniqueId=self.robot, childLinkIndex=self.eef_id + 8,
                                           jointType=r.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                                           parentFramePosition=[7.776548502369529e-15, 0.03760000318288803,
                                                                0.0429999977350235],
                                           childFramePosition=[0.0, -0.017901099286973476, 0.006516002118587494])
        self.client.changeConstraint(cid, maxForce=1e6)

        cid = self.client.createConstraint(parentBodyUniqueId=self.robot, parentLinkIndex=self.eef_id + 1,
                                           childBodyUniqueId=self.robot, childLinkIndex=self.eef_id + 6,
                                           jointType=r.JOINT_GEAR, jointAxis=[0, 1, 0],
                                           parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self.client.changeConstraint(cid, gearRatio=-1, maxForce=1e6, erp=1)

    def move_gripper(self, open_length):
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        self.client.setJointMotorControl2(self.robot, self.eef_id + 1, r.POSITION_CONTROL,
                                          targetPosition=open_angle, force=5, maxVelocity=2)

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def reset(self, rori):
        self.goal_num = 0
        self.client.resetJointState(self.robot, self.active_joints[0], rori[0])
        self.client.resetJointState(self.robot, self.active_joints[1], rori[1])
        self.client.resetJointState(self.robot, self.active_joints[2], rori[2])
        self.client.resetJointState(self.robot, self.active_joints[3], rori[3])
        self.open_gripper()
        self.client.removeAllUserDebugItems()

    def get_ids(self):
        return self.robot, self.client

    def apply_action(self, action):
        # Expects action to be 4 dimensional
        action = np.clip(action, -1, 1)
        t1, t2, t3, t5, gripper = action

        # Affine velocity values to reasonable values
        vel_gain = 1
        t1 *= 1.39 * vel_gain
        t2 *= 1.39 * vel_gain
        t3 *= 1.39 * vel_gain
        t5 *= 1.22 * vel_gain

        if gripper >= 0:
            self.close_gripper()
            self.grip = 1
        else:
            self.open_gripper()
            self.grip = -1

        # Set the velocity of the joints directly
        self.client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.active_joints,
            controlMode=r.VELOCITY_CONTROL,
            targetVelocities=[t1, t2, t3, t5],
            forces=[39., 39., 39., 9.])

    def get_observation(self):
        # ls = self.client.getLinkState(self.robot, linkIndex=self.eef_id, computeLinkVelocity=1)
        ls = self.client.getLinkState(self.robot, linkIndex=self.tcp_id, computeLinkVelocity=1)

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
        # ls = self.client.getLinkState(self.robot, linkIndex=self.eef_id, computeLinkVelocity=1)
        ls = self.client.getLinkState(self.robot, linkIndex=self.tcp_id, computeLinkVelocity=1)

        return ls[1]

    def get_pos_and_ori(self):
        base_pos, base_ori = self.client.getBasePositionAndOrientation(self.robot)

        return base_pos, base_ori

    def inv_kin(self, goal_pos, block_ob, theta, thetab):
        arm_lower_limits = [-inf, -2.41, -2.66, -2.23]
        arm_upper_limits = [-theta for theta in arm_lower_limits]
        arm_joint_ranges = [arm_upper_limits[i] - arm_lower_limits[i] for i in range(len(arm_lower_limits))]
        arm_rest_poses = [0, 15 / 180 * np.pi, -130 / 180 * np.pi, 55 / 180 * np.pi]

        block_ob = list(block_ob)
        block_size = block_ob[-1]
        block_pos = [block_ob[0], block_ob[1], block_ob[2]]

        orn = r.getQuaternionFromEuler((np.pi, 0, np.pi/2 + theta))
        ornb = r.getQuaternionFromEuler((np.pi, 0, np.pi/2 + thetab))

        offset = [block_pos[0]/math.sqrt(block_pos[0]**2 + block_pos[1]**2),
                  block_pos[1]/math.sqrt(block_pos[0]**2 + block_pos[1]**2)]

        g_0 = [block_pos[0]-0.2*offset[0], block_pos[1]-0.2*offset[1], 0.2 + 2*block_size]
        j_0 = self.client.calculateInverseKinematics(self.robot, self.tcp_id,  # self.eef_id,
                                                     targetPosition=g_0,
                                                     targetOrientation=ornb,
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

        g_1 = [block_pos[0]+block_size*offset[0], block_pos[1]+block_size*offset[1], 0.2 + block_size]
        j_1 = self.client.calculateInverseKinematics(self.robot, self.tcp_id,  # self.eef_id,
                                                     targetPosition=g_1,
                                                     targetOrientation=ornb,
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

        g_2 = [block_pos[0] + block_size * offset[0], block_pos[1] + block_size * offset[1], 0.3 + block_size]
        j_2 = self.client.calculateInverseKinematics(self.robot, self.tcp_id,  # self.eef_id,
                                                     targetPosition=g_2,
                                                     targetOrientation=ornb,
                                                     lowerLimits=arm_lower_limits, upperLimits=arm_upper_limits,
                                                     jointRanges=arm_joint_ranges, restPoses=arm_rest_poses,
                                                     maxNumIterations=1000, residualThreshold=0.0001)

        j_2 = list(j_2)
        for i in range(len(arm_lower_limits)):
            if j_2[i] > 0:
                temp = j_2[i] % (2 * np.pi)
                temp = np.clip(temp, 0, arm_upper_limits[i])
            else:
                temp = (-j_2[i]) % (2 * np.pi)
                temp = -np.clip(temp, 0, arm_upper_limits[i])
            j_2[i] = temp

        j_2 = tuple(j_2)

        g_des = [goal_pos[0], goal_pos[1], 0.1+block_size]
        j_des = self.client.calculateInverseKinematics(self.robot, self.tcp_id,  # self.eef_id,
                                                       targetPosition=g_des,
                                                       targetOrientation=orn,
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

        self.joint_goal = (j_0, j_1, j_2, j_des)
        self.task_goal = [g_0, g_1, g_2, g_des]
        self.goal_num = 0
        self.grip = -1

        return self.joint_goal

    def get_control_feedback(self):
        js = self.client.getJointStates(self.robot, jointIndices=self.active_joints)

        js_angle = (js[0][0], js[1][0], js[2][0], js[3][0])
        js_omega = (js[0][1], js[1][1], js[2][1], js[3][1])

        ls = self.client.getLinkState(self.robot, linkIndex=self.tcp_id, computeLinkVelocity=1)
        tool_pos = ls[4]
        tool_pos = list(tool_pos)
        goal_dist = [tool_pos[i] - self.task_goal[self.goal_num][i] for i in range(len(tool_pos))]
        goal_dist = abs(np.linalg.norm(goal_dist))

        if goal_dist < 0.02 and self.goal_num < 3:
            self.goal_num += 1

        temp = -1
        if self.goal_num == 2:
            temp = 1
        if self.goal_num == 3:
            temp = -1 if goal_dist < 0.02 else 1

        setpoint = self.joint_goal[self.goal_num]

        fb = dict(e_pos=np.array(tuple(map(operator.sub, js_angle, setpoint)), dtype=np.float32),
                  e_vel=np.array(js_omega, dtype=np.float32),
                  grip=temp)  # grip := gripper command

        return fb
