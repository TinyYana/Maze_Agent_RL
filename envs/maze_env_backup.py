import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from agents.astar_bot import astar_path


# --- 環境類別 (你的程式碼 + Bot 整合) ---
class MazeMasterEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MazeMasterEnv, self).__init__()

        self.grid_size = 15
        self.cell_size = 40

        self.ID_EMPTY = 0
        self.ID_WALL = 1
        self.ID_PLAYER = 2
        self.ID_EXIT = 3
        self.ID_MONSTER = 4

        # --- 關鍵修改 1: Action Space ---
        # 我們使用 MultiDiscrete 讓 AI 同時輸出三個數值：
        # 1. X 座標 (0 ~ 14)
        # 2. Y 座標 (0 ~ 14)
        # 3. 動作類型 (0: 無動作, 1: 加牆, 2: 刪牆, 3: 放怪)
        self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 4])

        # 觀測空間維持不變 (或是你可以考慮加入 Bot 的路徑資訊進去，但目前先保持簡單)
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None  # 控制 FPS

        self.state = None
        self.player_pos = None
        self.exit_pos = None
        self.current_path = []
        self.prev_path_len = 0  # 記錄上一步的路徑長度，用於計算進步幅度

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # 圍牆
        self.state[0, :] = self.ID_WALL
        self.state[-1, :] = self.ID_WALL
        self.state[:, 0] = self.ID_WALL
        self.state[:, -1] = self.ID_WALL

        self.player_pos = [1, 1]
        self.exit_pos = [self.grid_size - 2, self.grid_size - 2]

        self.state[self.player_pos[0], self.player_pos[1]] = self.ID_PLAYER
        self.state[self.exit_pos[0], self.exit_pos[1]] = self.ID_EXIT

        # 初始路徑計算
        path = astar_path(self.state, self.player_pos, self.exit_pos)
        self.current_path = path if path else []
        self.prev_path_len = len(self.current_path)

        return self.state, {}

    def step(self, action):
        # 解析 AI 的動作: [Row, Col, Type]
        r, c, act_type = action

        reward = 0
        terminated = False

        # --- 1. 執行 AI 的佈局動作 ---
        action_success = False

        # 檢查座標是否合法 (不能改玩家、出口、邊界)
        if (r, c) != tuple(self.player_pos) and (r, c) != tuple(self.exit_pos):
            # 邊界檢查 (雖然 MultiDiscrete 保證了範圍，但我們有圍牆)
            if 0 < r < self.grid_size - 1 and 0 < c < self.grid_size - 1:

                current_cell = self.state[r, c]

                if act_type == 1:  # 加牆
                    if current_cell == self.ID_EMPTY:
                        self.state[r, c] = self.ID_WALL
                        action_success = True
                    else:
                        reward -= 0.1  # 懲罰：嘗試在非空地上蓋牆 (浪費動作)

                elif act_type == 2:  # 刪牆
                    if current_cell == self.ID_WALL:
                        self.state[r, c] = self.ID_EMPTY
                        action_success = True
                    else:
                        reward -= 0.1  # 懲罰：嘗試刪除空氣

                elif act_type == 3:  # 放怪
                    # 限制：如果該位置已經是怪，或不是空地，就不能放
                    if current_cell == self.ID_EMPTY:
                        self.state[r, c] = self.ID_MONSTER
                        reward -= 0.5  # --- 關鍵修改 3: 增加放怪成本 ---
                        # 這樣 AI 只有在「放怪能大幅增加路徑長度」時才會放，不會亂放
                        action_success = True
                    else:
                        reward -= 0.1

        # --- 2. 玩家 (Bot) 移動與獎勵計算 ---

        # 重新計算路徑
        path = astar_path(self.state, self.player_pos, self.exit_pos)
        self.current_path = path

        if path is None:
            # 情況 A: 路被堵死了
            reward -= 5.0  # 重罰
            # 為了讓 AI 學會「不要堵死」，我們必須還原這個動作
            if action_success:
                # 簡單還原：如果是加牆/放怪導致堵死，就變回空地
                if act_type in [1, 3]:
                    self.state[r, c] = self.ID_EMPTY
                # 如果是刪牆導致堵死(不太可能)，還原牆
                elif act_type == 2:
                    self.state[r, c] = self.ID_WALL

            # 重新計算路徑以確保狀態一致
            path = astar_path(self.state, self.player_pos, self.exit_pos)
            self.current_path = path
            current_len = len(path) if path else 0  # 應該要有路了
        else:
            current_len = len(path)

            # --- 關鍵修改 2: 差分獎勵 (Differential Reward) ---
            # 我們獎勵「路徑變長了多少」，而不是「路徑有多長」
            # 這會驅使 AI 主動去改變地形來延長路徑
            diff = current_len - self.prev_path_len

            if diff > 0:
                reward += diff * 1.0  # 路徑變長，給予正向獎勵
            elif diff < 0:
                reward -= 0.2  # 路徑變短 (Bot 往前走了)，給予輕微懲罰

            # 額外獎勵：保持路徑複雜度
            reward += current_len * 0.01

            # 移動 Bot
            if current_len > 1:
                next_step = path[1]
                self.state[self.player_pos[0], self.player_pos[1]] = self.ID_EMPTY
                self.player_pos = list(next_step)
                self.state[self.player_pos[0], self.player_pos[1]] = self.ID_PLAYER

            # 遊戲結束判定
            if self.player_pos == self.exit_pos:
                terminated = True
                reward -= 10  # 被通關了，大懲罰

        # 更新歷史長度
        self.prev_path_len = current_len if path else 0

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, False, {}

    def _get_random_pos_near_player(self):
        # 這個函式已經不需要了，因為 AI 現在自己決定座標
        pass

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Maze Master: RL Agent vs A* Bot")
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(
            (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        )
        canvas.fill((240, 240, 240))

        # 繪製路徑預覽
        if self.current_path and len(self.current_path) > 1:
            for i in range(len(self.current_path) - 1):
                start = (
                    self.current_path[i][1] * self.cell_size + self.cell_size // 2,
                    self.current_path[i][0] * self.cell_size + self.cell_size // 2,
                )
                end = (
                    self.current_path[i + 1][1] * self.cell_size + self.cell_size // 2,
                    self.current_path[i + 1][0] * self.cell_size + self.cell_size // 2,
                )
                pygame.draw.line(canvas, (255, 200, 0), start, end, 5)

        # 繪製格子
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_value = self.state[r, c]
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                if cell_value == self.ID_WALL:
                    pygame.draw.rect(canvas, (50, 50, 50), rect)
                    pygame.draw.rect(canvas, (100, 100, 100), rect, 2)
                elif cell_value == self.ID_PLAYER:
                    pygame.draw.circle(
                        canvas, (0, 100, 255), rect.center, self.cell_size // 3
                    )
                elif cell_value == self.ID_EXIT:
                    pygame.draw.rect(canvas, (0, 200, 0), rect)
                elif cell_value == self.ID_MONSTER:
                    pygame.draw.circle(
                        canvas, (200, 0, 0), rect.center, self.cell_size // 4
                    )

                pygame.draw.rect(canvas, (220, 220, 220), rect, 1)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(10)  # 稍微加快一點顯示速度

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
