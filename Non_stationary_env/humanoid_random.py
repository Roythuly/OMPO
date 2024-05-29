import mujoco_py
import gym
import numpy as np
import random

from gym.envs.mujoco.humanoid_v3 import HumanoidEnv

class HumanoidTransferEnv(HumanoidEnv):
    def __init__(self, gravity = 9.81, wind = 0, **kwargs):
        super().__init__(**kwargs)
        # self.reset(gravity = gravity, wind = wind)
        self.model.opt.viscosity = 0.00002 
        self.model.opt.density = 1.2
        self.model.opt.gravity[:] = np.array([0., 0., -gravity])
        self.model.opt.wind[:] = np.array([-wind, 0., 0.])

class HumanoidRandomEnv(HumanoidEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.opt.viscosity = 0.00002 
        self.model.opt.density = 1.2

    
    def step_with_random(self, action, gravity = 9.81, wind = 0):
        # print("Step with new gravity = ", gravity, " wind = ", wind)
        
        self.model.opt.gravity[:] = np.array([0., 0., -gravity])
        self.model.opt.wind[:] = np.array([-wind, 0., 0.])
        
        return self.step(action)

    def reset(self, gravity = 9.81, wind = 0):
        '''Must called like env.reset(body_len = XXX)'''
        # print("Reset with new gravity = ", gravity, " wind = ", wind)
        
        self.model.opt.gravity[:] = np.array([0., 0., -gravity])
        self.model.opt.wind[:] = np.array([-wind, 0., 0.])

        return super().reset()
