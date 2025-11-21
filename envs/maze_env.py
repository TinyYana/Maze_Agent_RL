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

        # 初始化 maze 矩陣 (在 init 先定義好形狀，雖然 reset 會覆蓋)
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

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
        # 處理隨機種子
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. 初始化並生成隨機迷宮
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._generate_random_maze()

        # 2. 放玩家 (確保起點是空地)
        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.maze[0, 0] = self.ID_PLAYER

        # 3. 放出口 (確保終點是空地)
        self.exit_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1], dtype=np.int32
        )
        # 如果終點剛好被牆堵住，強制變成出口 (雖然 DFS 演算法通常會覆蓋全圖，但保險起見)
        self.maze[-1, -1] = self.ID_EXIT

        # 清空怪物列表
        self.monsters = []

        # 4. 回傳觀測值
        return self.maze.copy(), {}

    def _generate_random_maze(self):
        """
        使用 DFS (Recursive Backtracking) 生成隨機迷宮
        注意：為了生成漂亮的迷宮，grid_size 最好是奇數 (例如 15, 17, 21)
        """
        # 先將所有地方填滿牆壁
        self.maze.fill(self.ID_WALL)

        # 起點設為 (0, 0) 並標記為空地
        start_x, start_y = 0, 0
        self.maze[start_x, start_y] = self.ID_EMPTY

        # 使用堆疊 (Stack) 進行 DFS，避免遞迴深度限制
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]

            # 尋找未訪問的鄰居 (距離為 2，因為中間要留牆)
            # 方向：上、下、左、右
            neighbors = []
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 檢查邊界
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # 如果該鄰居是牆壁 (代表未訪問過)
                    if self.maze[nx, ny] == self.ID_WALL:
                        neighbors.append((nx, ny, dx, dy))

            if neighbors:
                # 隨機選擇一個鄰居
                # 使用 self.rng.choice 需要將 list 轉為 index 選擇，或者手動選擇
                idx = self.rng.integers(0, len(neighbors))
                nx, ny, dx, dy = neighbors[idx]

                # 打通當前格與鄰居中間的牆
                self.maze[x + dx // 2, y + dy // 2] = self.ID_EMPTY

                # 將鄰居設為空地
                self.maze[nx, ny] = self.ID_EMPTY

                # 將鄰居加入堆疊
                stack.append((nx, ny))
            else:
                # 如果沒有未訪問的鄰居，回溯
                stack.pop()

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
        if len(self.monsters) >= 2:
            return
        if self.maze[x, y] == self.ID_EMPTY:
            self.maze[x, y] = self.ID_MONSTER
            self.monsters.append((x, y))

    def render(self):
        return super().render()

    def close(self):
        return super().close()
