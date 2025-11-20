import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    monsters = []

    def __init__(self, size=15, k_step=5, t_min=50, t_max=150, render_mode=None):
        super(MazeEnv, self).__init__()

        # 基礎參數設定
        self.grid_size = size
        self.k_step = k_step
        self.t_min = t_min
        self.t_max = t_max
        self.rng = np.random.default_rng(114514)

        self.ID_EMPTY = 0
        self.ID_WALL = 1
        self.ID_PLAYER = 2
        self.ID_EXIT = 3
        self.ID_MONSTER = 4

        # 定義動作空間
        self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 5])
        # 定義觀測空間
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        self.render_mode = render_mode

    def step(self, action):
        x, y, action_type = action
        # if action_type == 0:
        #     return self.maze.copy(), 0, False, False, {}
        if action_type == 1:
            self.maze[x, y] = self.ID_WALL
        elif action_type == 2:
            self.maze[x, y] = self.ID_EMPTY
        elif action_type == 3:
            self.maze[x, y] = self.ID_MONSTER
        elif action_type == 4:
            self.maze[x, y] = self.ID_EXIT
        obs = self.maze.copy()
        reward = self._compute_reward()
        terminated = self._check_terminal()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # 1. 建立空地
        self.maze[:, :] = self.ID_EMPTY

        # 2. 放玩家
        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.maze[0, 0] = self.ID_PLAYER

        # 3. 放出口
        self.exit_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1], dtype=np.int32
        )
        self.maze[-1, -1] = self.ID_EXIT

        # 4. 回傳觀測值
        return self.maze.copy(), {}

    def _move_exit_to(self, x, y):
        self.maze[self.exit_pos[0], self.exit_pos[1]] = self.ID_EMPTY
        self.exit_pos = np.array([x, y], dtype=np.int32)
        self.maze[x, y] = self.ID_EXIT

    def _place_wall(self, x, y):
        if self.maze[x, y] == self.ID_EMPTY:
            self.maze[x, y] = self.ID_WALL

    def _remove_wall(self, x, y):
        if self.maze[x, y] == self.ID_WALL:
            self.maze[x, y] = self.ID_EMPTY

    def _spawn_monster(self, x, y):
        if self.maze[x, y] == self.ID_EMPTY:
            self.maze[x, y] = self.ID_MONSTER
            self.monsters.append((x, y))

    def render(self):
        return super().render()

    def close(self):
        return super().close()
