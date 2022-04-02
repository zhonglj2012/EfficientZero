import numpy as np
from core.game import Game
from core.utils import arr_to_str


class DefaultWrapper(Game):
    def __init__(self, env, discount: float):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)

        return observation

    def close(self):
        self.env.close()
