from typing import SupportsFloat, Any
import os
import mujoco as mj
import mujoco_viewer
from gymnasium import Env
from gymnasium.core import RenderFrame, ActType, ObsType
import numpy as np


class CarWallEnv(Env):
    def __init__(self, render: bool = False, off_screen: bool = False):
        xml_path = 'models/car_wall.xml'
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # boundary values
        self.starting_x = -2
        self.left_bound_xpos = -3
        self.wall_pos = 1
        self.observation_clipping = 2
        self.time_limit = 1200
        self.t = 0

        # reward function parameters
        self.base_reward = 1600
        self.time_penalty = -1
        self.impact_vel_penalty_coeff = 200
        self.not_succeeded_penalty = -300
        self.distance_reward_coeff = 300

        self.mu = 0.0
        self.sigma = 0.2
        self.random_starts = True
        self.render = render
        self.offscreen = off_screen

        if render:
            if off_screen:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
            else:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if action == 0:
            action = -1
        self.data.ctrl = action
        mj.mj_step(self.model, self.data)
        self.t += 1

        if self.render and not self.offscreen:
            self.viewer.applied_a = action
            self.viewer.render()

        return self.observation, self.reward, self.done, False, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        if self.random_starts:
            starting_qpos = np.random.normal(self.starting_x, self.sigma/5, 1)
            starting_qvel = np.random.normal(0.0, self.sigma*3, 1)
        else:
            starting_qpos = np.random.normal(self.starting_x, 0.0, 1)
            starting_qvel = np.random.normal(0.0, 0.0, 1)
        self.data.qpos = starting_qpos
        self.data.qvel = starting_qvel
        self.data.ctrl = [0.0]
        self.t = 0

        return self.observation, {}

    @property
    def reward(self) -> float:
        reward = self.time_penalty
        if self.done:
            if self._impact_with_wall():
                reward += self.base_reward
                reward -= self.impact_vel_penalty_coeff*self.data.qvel[0]
            else:  # means not succeeded in this run
                reward += self.not_succeeded_penalty

            reward += self.distance_reward_coeff*self.data.qpos[0]
        return reward

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    @property
    def observation(self):
        return np.clip(np.concatenate([self.data.qpos, self.data.qvel]), -self.observation_clipping,
                       self.observation_clipping)

    @property
    def done(self):
        x_pos = self.data.qpos
        if x_pos <= self.left_bound_xpos:
            return True
        if self._impact_with_wall():
            return True
        if self.t >= self.time_limit:
            return True
        return False

    def _impact_with_wall(self) -> bool:
        contact_geom1 = self.data.contact.geom1
        contact_geom2 = self.data.contact.geom2
        if len(contact_geom1) > 1:
            for geo1, geo2 in zip(contact_geom1, contact_geom2):
                if geo1 == 1 and geo2 == 2:
                    return True
                if geo1 == 2 and geo2 == 1:
                    return True

        return False
        #
        # if self.data.qpos >= self.wall_pos:
        #     return True
