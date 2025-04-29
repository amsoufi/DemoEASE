import pybullet as r
import os
import numpy as np
import operator
from math import inf


class Robot:
    def __init__(self, client, rori):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'kinovaGen3.urdf')
        self.robot = self.client.loadURDF(fileName=f_name,
                                          basePosition=[0, 0, 0],
                                          useFixedBase=1,
                                          flags=r.URDF_USE_SELF_COLLISION)

        # Joint indices as found by p.getJointInfo()
        self.NumJoints = self.client.getNumJoints(self.robot) - 1
        self.joints = [i for i in range(self.NumJoints)]

        self.client.resetJointState(self.robot, 0, rori[0])
        self.client.resetJointState(self.robot, 1, rori[1])
        self.client.resetJointState(self.robot, 2, rori[2])
        self.client.resetJointState(self.robot, 3, rori[3])
        self.client.resetJointState(self.robot, 4, rori[4])
        self.client.resetJointState(self.robot, 5, rori[5])

        # Unlock joints limits
        self.client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=r.VELOCITY_CONTROL,
            forces=[0, 0, 0, 0, 0, 0])

        self.joint_goal = None

    def get_ids(self):
        return self.robot, self.client

    def apply_action(self, action):
        # Expects action to be 6 dimensional
        action = np.clip(action, -1, 1)
        t1, t2, t3, t4, t5, t6 = action

        # Affine velocity values to reasonable values
        vel_gain = 0.3333
        t1 *= 1.39 * vel_gain
        t2 *= 1.39 * vel_gain
        t3 *= 1.39 * vel_gain
        t4 *= 1.22 * vel_gain
        t5 *= 1.22 * vel_gain
        t6 *= 1.22 * vel_gain

        # Set the velocity of the joints directly
        self.client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=r.VELOCITY_CONTROL,
            targetVelocities=[t1, t2, t3, t4, t5, t6],
            forces=[39., 39., 39., 9., 9., 9.])

    def get_observation(self):
        ls = self.client.getLinkState(self.robot, linkIndex=self.NumJoints, computeLinkVelocity=1)

        js = self.client.getJointStates(self.robot, jointIndices=self.joints)

        js_angle = (js[0][0], js[1][0], js[2][0], js[3][0], js[4][0], js[5][0])
        js_omega = (js[0][1], js[1][1], js[2][1], js[3][1], js[4][1], js[5][1])

        # Tip Position, Orientation + Joint Angles, Velocity
        observation = ls[4] + ls[5] + js_angle + tuple(np.cos(np.array(js_angle))) + tuple(
            np.sin(np.array(js_angle))) + js_omega

        return observation

    def get_joint_torque(self):
        js = self.client.getJointStates(self.robot, jointIndices=self.joints)

        js_torque = (js[0][3] / 39, js[1][3] / 39, js[2][3] / 39, js[3][3] / 9, js[4][3] / 9, js[5][3] / 9)

        return js_torque

    def get_tool_ori(self):
        ls = self.client.getLinkState(self.robot, linkIndex=self.NumJoints, computeLinkVelocity=1)

        return ls[1]

    def get_pos_and_ori(self):
        base_pos, base_ori = self.client.getBasePositionAndOrientation(self.robot)

        return base_pos, base_ori

    def inv_kin(self, goal_pos, goal_orn):
        arm_lower_limits = [-inf, -2.41, -2.66, -inf, -2.23, -inf]
        arm_upper_limits = [-theta for theta in arm_lower_limits]
        arm_joint_ranges = [arm_upper_limits[i] - arm_lower_limits[i] for i in range(len(arm_lower_limits))]
        arm_rest_poses = [45/180 * np.pi, 0, 0, 0, 0, 0]

        j_des = self.client.calculateInverseKinematics(self.robot, self.NumJoints,
                                                       targetPosition=goal_pos,
                                                       targetOrientation=goal_orn,
                                                       lowerLimits=arm_lower_limits, upperLimits=arm_upper_limits,
                                                       jointRanges=arm_joint_ranges, restPoses=arm_rest_poses,
                                                       maxNumIterations=1000, residualThreshold=0.0001)

        j_des = list(j_des)
        for i in range(len(arm_lower_limits)):
            j_des[i] = j_des[i] % (2 * np.pi)
            if j_des[i] > np.pi:
                j_des[i] = j_des[i] - 2 * np.pi
            j_des[i] = np.clip(j_des[i], arm_lower_limits[i], arm_upper_limits[i])

        self.joint_goal = tuple(j_des)

        return self.joint_goal

    def get_control_feedback(self):
        js = self.client.getJointStates(self.robot, jointIndices=self.joints)

        js_angle = (js[0][0], js[1][0], js[2][0], js[3][0], js[4][0], js[5][0])
        print(js_angle)
        js_omega = (js[0][1], js[1][1], js[2][1], js[3][1], js[4][1], js[5][1])

        fb = dict(e_pos=np.array(tuple(map(operator.sub, js_angle, self.joint_goal)), dtype=np.float32),
                  e_vel=np.array(js_omega, dtype=np.float32))
        return fb
