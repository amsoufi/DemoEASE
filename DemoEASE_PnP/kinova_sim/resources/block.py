import pybullet as p
import os


class Block:
    def __init__(self, client, base, orn, scale):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        self.block = self.client.loadURDF(fileName=f_name,
                                          basePosition=[base[0], base[1], base[2]],
                                          baseOrientation=list(orn),
                                          useFixedBase=0,
                                          globalScaling=scale)  # mass is also increased from 0.05 to 0.5 in the urdf

        self.scale = scale

    def reset(self, base, orn):
        self.client.resetBasePositionAndOrientation(self.block, base, orn)

    def get_observation(self):
        ls = self.client.getBasePositionAndOrientation(self.block)

        return ls[0]

    def get_ids(self):
        return self.block
