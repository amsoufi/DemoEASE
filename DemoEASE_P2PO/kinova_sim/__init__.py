from gym.envs.registration import registry, register, make, spec
register(
    id='kinova-v0',
    entry_point='kinova_sim.envs:KinovaEnv'
)

register(
    id='kinova4dof-v0',
    entry_point='kinova_sim.envs:KinovaEnv4DOF'
)
