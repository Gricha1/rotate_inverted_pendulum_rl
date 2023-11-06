import os

import numpy as np
import cv2
import gym
from gym import utils
from gym.envs.mujoco import MujocoEnv, MuJocoPyEnv
from gym.spaces import Box


#class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
class InvertedPendulumEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        #self.render_mode == "rgb_array"
        model_path = os.path.join(os.path.dirname(__file__), 'custom_model.xml')
        #MujocoEnv.__init__(
        MuJocoPyEnv.__init__(
            self,
            model_path,
            2,
            observation_space=observation_space,
            **kwargs
        )
        #self.last_ob = None

    def step(self, a):

        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all())

        if self.render_mode == "human":
            self.render()

        # add transform obs angle
        ob[1] = np.mod(ob[1], 2*np.pi) # [0, 2pi]
        ob[1] = (ob[1] - 2*np.pi) if ob[1] > np.pi else ob[1] # [-pi, pi]

        # add reward
        reward = 0
        stable_coef = 10 # 1
        swinged_up_2_coef = 5
        swinged_up_1_coef = 1 # 1
        x_bound_coef = 200
        x_change_coef = 0.3 # 0.3
        ctrl_coef = 0 # 0.1
        test_coef = 0

        pendulum_angle = ob[1]
        assert abs(pendulum_angle) <= np.pi, f"{pendulum_angle}"

        eps = 0.2
        stable_reward = 0
        if abs(pendulum_angle) <= eps:
          stable_reward = 1
        reward += stable_coef * stable_reward

        swinged_up_1_reward = 1 - abs(pendulum_angle) / np.pi
        reward += swinged_up_1_coef * swinged_up_1_reward

        x = abs(ob[0]) # ob[0] in [-1.1, 1.1]
        x_change_reward = -x ** 2
        reward += x_change_coef * x_change_reward

        test_reward = 1
        reward += test_coef * test_reward

        swinged_up_2_reward = 0
        if abs(pendulum_angle) <= np.pi:
          swinged_up_2_reward = (1 - abs(pendulum_angle) / np.pi)
        reward += swinged_up_2_coef * swinged_up_2_reward

        x_bound_reward = 0
        if x > 0.8:
          x_bound_reward = -1
        reward += x_bound_coef * x_bound_reward

        ctrl_reward = -abs(a[0]) / 3
        reward += ctrl_coef * ctrl_reward

        # add terminal condition
        #if abs(pendulum_angle) > eps:
        #  terminated = True
        if x_bound_coef > 0 and x > 0.8:
          terminated = True

        return ob, reward, terminated, {}


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        qpos[1] = 3.14 # Set the pole to be facing down
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    # render without additional window
    def custom_render(self, shape=(600, 600)):
        # camera_name = ('fixednear', 'fixedfar', 'vision', 'track')
        #image = self.sim.render(shape[0], shape[1], camera_name="fixedfar", mode='offscreen')
        image = self.sim.render(shape[0], shape[1], mode='offscreen')
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        return rotated_image
       