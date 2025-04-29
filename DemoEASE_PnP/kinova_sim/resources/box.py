import pybullet as g
import os


class Box:
    def __init__(self, client, base, orn):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'box.urdf')
        self.box = self.client.loadURDF(fileName=f_name,
                                        basePosition=[base[0], base[1], base[2]],
                                        baseOrientation=list(orn),
                                        useFixedBase=1,
                                        flags=g.URDF_USE_SELF_COLLISION)

    def reset(self, base, orn):
        self.client.resetBasePositionAndOrientation(self.box, base, orn)
