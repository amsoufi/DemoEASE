import pybullet as p
import os


class Block:
    def __init__(self, client, base, scale):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        self.block = self.client.loadURDF(fileName=f_name,
                                          basePosition=[base[0], base[1], base[2]],
                                          useFixedBase=1,
                                          globalScaling=scale)  # mass is also increased from 0.05 to 0.5 in the urdf
        self.base = base
        self.scale = scale

    def reset(self, base):
        self.client.resetBasePositionAndOrientation(self.block, [base[0], base[1], base[2]], (0, 0, 0, 1))

    def get_observation(self):
        ls = self.client.getBasePositionAndOrientation(self.block)
        observation = list(ls[0])
        observation.append(self.scale * 0.02)
        observation = tuple(observation)

        return observation

    def get_ids(self):
        return self.block
