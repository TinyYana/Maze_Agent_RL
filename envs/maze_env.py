import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import pygame  # 新增：匯入 pygame

from agents.player_bot import astar_path

# 加入父目錄以匯入 agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}  # 修改：設定 FPS
    monsters = []

    def __init__(self, size=15, k_step=5, t_min=50, t_max=150, render_mode=None):
        super(MazeEnv, self).__init__()

        # 基礎參數設定
        self.grid_size = size
        self.k_step = k_step
        self.t_min = t_min
        self.t_max = t_max
        self.rng = np.random.default_rng(114514)

        # 玩家狀態追蹤
        self.current_time = 0

        self.ID_EMPTY = 0
        self.ID_WALL = 1
        self.ID_PLAYER = 2
        self.ID_EXIT = 3
        self.ID_MONSTER = 4

        # 初始化 maze 矩陣
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # 定義動作空間
        self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 5])
        # 定義觀測空間
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        self.render_mode = render_mode

        # 新增：渲染相關變數
        self.window_size = 600  # 視窗大小
        self.window = None
        self.clock = None

    def step(self, action):
        x, y, action_type = action

        # 1. Maze Master 修改迷宮
        if action_type == 1:
            self.maze[x, y] = self.ID_WALL
        elif action_type == 2:
            self.maze[x, y] = self.ID_EMPTY
        elif action_type == 3:
            self.maze[x, y] = self.ID_MONSTER
        elif action_type == 4:
            # 移動出口
            self.maze[self.exit_pos[0], self.exit_pos[1]] = self.ID_EMPTY
            self.exit_pos = np.array([x, y], dtype=np.int32)
            self.maze[x, y] = self.ID_EXIT

        # 保護機制
        if np.array_equal([x, y], self.player_pos):
            self.maze[x, y] = self.ID_PLAYER

        # 2. 模擬玩家移動
        path = astar_path(self.maze, self.player_pos, self.exit_pos)

        reward = 0
        terminated = False
        truncated = False
        info = {}

        if path is None:
            reward = -50
            terminated = True
            info["result"] = "blocked"
        else:
            steps_can_move = len(path) - 1
            steps_this_turn = min(steps_can_move, self.k_step)

            if steps_this_turn > 0:
                new_pos = path[steps_this_turn]
                if not np.array_equal(self.player_pos, self.exit_pos):
                    self.maze[self.player_pos[0], self.player_pos[1]] = self.ID_EMPTY
                self.player_pos = np.array(new_pos)
                self.maze[self.player_pos[0], self.player_pos[1]] = self.ID_PLAYER
                self.current_time += steps_this_turn

            # 3. 計算獎勵機制
            if np.array_equal(self.player_pos, self.exit_pos):
                terminated = True
                if self.t_min <= self.current_time <= self.t_max:
                    reward = 100
                    info["result"] = "flow_success"
                elif self.current_time < self.t_min:
                    reward = -100
                    info["result"] = "too_fast"
                else:
                    reward = -100
                    info["result"] = "too_slow"
            else:
                if self.current_time > self.t_max * 1.5:
                    terminated = True
                    reward = -100
                    info["result"] = "timeout"
                if self.current_time > 30:
                    reward = -1

        obs = self.maze.copy()

        # 新增：如果是 human 模式，執行渲染
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._generate_random_maze()

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.maze[0, 0] = self.ID_PLAYER
        self.current_time = 0

        self.exit_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1], dtype=np.int32
        )
        self.maze[-1, -1] = self.ID_EXIT
        self.monsters = []

        # 新增：重置時也要渲染第一幀
        if self.render_mode == "human":
            self._render_frame()

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

    # 新增：渲染邏輯實作
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size / self.grid_size

        # 繪製迷宮
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * pix_square_size,
                    x * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                )
                cell_value = self.maze[x, y]

                if cell_value == self.ID_WALL:
                    pygame.draw.rect(canvas, (0, 0, 0), rect)  # 黑色牆壁
                elif cell_value == self.ID_PLAYER:
                    pygame.draw.rect(canvas, (0, 0, 255), rect)  # 藍色玩家
                elif cell_value == self.ID_EXIT:
                    pygame.draw.rect(canvas, (0, 255, 0), rect)  # 綠色出口
                elif cell_value == self.ID_MONSTER:
                    pygame.draw.rect(canvas, (255, 0, 0), rect)  # 紅色怪物
                # 空地為白色 (背景)

        # 繪製格線
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
