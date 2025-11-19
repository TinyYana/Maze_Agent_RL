import gymnasium
from envs.maze_env_backup import MazeMasterEnv


class MazeEnv(gymnasium.Env):
    def __init__(self, render_mode=None):
        super(MazeMasterEnv, self).__init__()

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def render(self):
        return super().render()

    def close(self):
        return super().close()
