import pybullet as g
import os


class Goal:
    def __init__(self, client, base, vel):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        self.goal = self.client.loadURDF(fileName=f_name,
                                         basePosition=[base[0], base[1], base[2]],
                                         useFixedBase=1,
                                         flags=g.URDF_IGNORE_COLLISION_SHAPES)

        self.vel = vel
        self.reset(base, vel)

    def reset(self, base, vel):
        self.client.resetBasePositionAndOrientation(self.goal, [base[0], base[1], base[2]], (0, 0, 0, 1))
        self.client.resetBaseVelocity(self.goal, linearVelocity=vel)

    def get_observation(self):
        ls = self.client.getBasePositionAndOrientation(self.goal)

        observation = ls[0] + self.vel

        return observation
