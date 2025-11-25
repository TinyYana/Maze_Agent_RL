import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import config  # 匯入設定檔

from agents.astar_bot import astar_path


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": config.FPS}

    def __init__(self, render_mode=None):
        super(MazeEnv, self).__init__()

        # 載入設定
        self.grid_size = config.GRID_SIZE
        self.render_mode = render_mode
        self.rng = np.random.default_rng(114514)

        # 狀態變數
        self.current_time = 0
        self.player_hp = config.PLAYER_MAX_HP
        self.monsters = []  # 儲存怪物座標 list of [x, y]

        # 初始化 maze 矩陣
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # 定義動作空間：一次執行 ACTIONS_PER_TURN 個動作
        # 每個動作包含 (x, y, type)
        # 結構：[x1, y1, type1, x2, y2, type2, ...]
        n_actions = config.ACTIONS_PER_TURN
        action_dims = []
        for _ in range(n_actions):
            action_dims.extend([self.grid_size, self.grid_size, 5])

        self.action_space = spaces.MultiDiscrete(action_dims)

        # 定義觀測空間
        # 修改：為了兼容 NatureCNN，我們需要：
        # 1. 增加 Channel 維度 (1, H, W)
        # 2. 使用 uint8 類型
        # 3. 放大圖像尺寸 (因為 15x15 對 NatureCNN 來說太小，會導致卷積層報錯)
        self.scale_factor = 4  # 放大 4 倍 -> 60x60
        obs_size = self.grid_size * self.scale_factor

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, obs_size, obs_size), dtype=np.uint8
        )

        # 渲染相關
        self.window_size = config.WINDOW_SIZE
        self.window = None
        self.clock = None
        self.font = None

    def step(self, action):
        # 0. 計算動作前的路徑長度 (作為基準)
        old_path = astar_path(self.maze, self.player_pos, self.exit_pos)
        old_path_len = len(old_path) if old_path else 0

        # 1. 解析並執行 Maze Master 的多重動作
        # action array 形狀為 (ACTIONS_PER_TURN * 3,)
        reshaped_actions = action.reshape(config.ACTIONS_PER_TURN, 3)

        for act in reshaped_actions:
            x, y, action_type = act
            self._apply_action(x, y, action_type)

        # 確保玩家和出口標記正確
        self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_PLAYER
        self.maze[self.exit_pos[0], self.exit_pos[1]] = config.ID_EXIT

        # 2. 模擬玩家移動 (A*)
        path = astar_path(self.maze, self.player_pos, self.exit_pos)

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # --- 新增邏輯：計算路徑長度變化獎勵 ---
        if path is not None and old_path is not None:
            new_path_len = len(path)
            # 如果路徑變長了，且沒有被堵死，給予獎勵
            if new_path_len > old_path_len:
                diff = new_path_len - old_path_len
                # 限制最大獎勵，避免因為迷宮生成演算法的劇烈變化導致獎勵爆炸
                reward += min(diff, 10) * config.REWARD_PATH_EXTEND
                # print(f"Good job! Path extended by {diff}")
        # -------------------------------------

        if path is None:
            reward += config.REWARD_BLOCKED  # 使用 += 累加
            terminated = (
                True  # 堵死路通常視為回合結束，或者你可以選擇不結束但給予懲罰並還原地圖
            )
            info["result"] = "blocked"

            # 選擇性：如果堵死路，還原上一步操作 (讓 Agent 繼續嘗試而不是直接重置)
            # 這裡保持 terminated = True 比較簡單，讓它學會 "Game Over"
        else:
            steps_can_move = len(path) - 1
            steps_this_turn = min(steps_can_move, config.K_STEP)

            if steps_this_turn > 0:
                # 清除舊位置
                if not np.array_equal(self.player_pos, self.exit_pos):
                    self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_EMPTY

                # 檢查移動路徑上是否碰撞怪物 (新增邏輯)
                path_segment = path[1 : steps_this_turn + 1]
                monsters_to_keep = []
                hit_count = 0

                for m_pos in self.monsters:
                    # 檢查此怪物是否在玩家的路徑上
                    is_hit = False
                    for p_pos in path_segment:
                        if m_pos[0] == p_pos[0] and m_pos[1] == p_pos[1]:
                            is_hit = True
                            break

                    if is_hit:
                        hit_count += 1
                    else:
                        monsters_to_keep.append(m_pos)

                if hit_count > 0:
                    self.player_hp -= hit_count
                    reward += config.REWARD_HIT * hit_count  # 玩家受傷給予獎勵
                    self.monsters = monsters_to_keep

                # 更新位置
                new_pos = path[steps_this_turn]
                self.player_pos = np.array(new_pos)
                self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_PLAYER
                self.current_time += steps_this_turn

        # 3. 怪物 AI 移動 (追擊玩家)
        self._move_monsters()

        # 4. 碰撞檢測 (玩家 vs 怪物)
        self._check_collisions()
        # 注意：_check_collisions 內部有扣血邏輯，這裡也要加上獎勵
        # 為了簡化，建議把 _check_collisions 的扣血邏輯移出來，或者在裡面加 reward
        # 這裡暫時假設 _check_collisions 只負責更新狀態，我們在外面檢查 HP 變化
        # (由於 _check_collisions 比較簡單，我們假設它已經執行，如果 HP 變少就是被打到了)

        # 5. 判定遊戲狀態與獎勵
        if self.player_hp <= 0:
            terminated = True
            reward += config.REWARD_DEATH  # 使用 +=
            info["result"] = "died"
        elif np.array_equal(self.player_pos, self.exit_pos):
            terminated = True
            if config.TIME_MIN <= self.current_time <= config.TIME_MAX:
                reward += config.REWARD_GOAL
                info["result"] = "flow_success"
            elif self.current_time < config.TIME_MIN:
                reward += config.REWARD_TOO_FAST
                info["result"] = "too_fast"
            else:
                reward += config.REWARD_TOO_SLOW
                info["result"] = "too_slow"
        else:
            if self.current_time > config.TIME_MAX * 1.5:
                terminated = True
                reward += config.REWARD_TIMEOUT
                info["result"] = "timeout"
            if self.current_time > 30:  # 這裡原本是 > 30 才給分，建議直接給
                reward += config.REWARD_STEP

        # 修改：使用 _get_obs() 取得處理過的觀測值
        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        將 maze 轉換為 NatureCNN 可接受的格式：
        1. 數值映射 (增加對比度)
        2. 放大尺寸 (Nearest Neighbor)
        3. 增加 Channel 維度 (1, H, W)
        """
        # 轉為 uint8
        obs = self.maze.astype(np.uint8)

        # 數值映射：0->0, 1->50, 2->100, 3->150, 4->200
        # 這樣 CNN 更容易區分不同物件 (原本 0-4 數值太接近，視為像素時幾乎都是黑色)
        obs = obs * 50

        # 放大圖像 (使用 numpy repeat 模擬 nearest neighbor scaling)
        # axis=0 (垂直放大), axis=1 (水平放大)
        obs = np.repeat(
            np.repeat(obs, self.scale_factor, axis=0), self.scale_factor, axis=1
        )

        # 增加 Channel 維度 -> (1, 60, 60)
        obs = np.expand_dims(obs, axis=0)

        return obs

    def _apply_action(self, x, y, action_type):
        """執行單一編輯動作"""
        # 保護機制：不能修改玩家或出口當前位置
        if np.array_equal([x, y], self.player_pos) or np.array_equal(
            [x, y], self.exit_pos
        ):
            return

        if action_type == 1:  # Wall
            self.maze[x, y] = config.ID_WALL
            # 如果該位置原本有怪物，從列表中移除
            self._remove_monster_at(x, y)

        elif action_type == 2:  # Empty
            self.maze[x, y] = config.ID_EMPTY
            self._remove_monster_at(x, y)

        elif action_type == 3:  # Monster
            # 只有在空地才能放怪物
            if self.maze[x, y] == config.ID_EMPTY:
                self.maze[x, y] = config.ID_MONSTER
                self.monsters.append([x, y])

        elif action_type == 4:  # Move Exit
            self.maze[self.exit_pos[0], self.exit_pos[1]] = config.ID_EMPTY
            self.exit_pos = np.array([x, y], dtype=np.int32)
            self.maze[x, y] = config.ID_EXIT
            self._remove_monster_at(x, y)

    def _move_monsters(self):
        """所有怪物使用 A* 向玩家移動"""
        new_monster_positions = []

        for m_pos in self.monsters:
            mx, my = m_pos

            # 暫時將自己設為空，以便計算路徑 (避免自己擋住自己)
            self.maze[mx, my] = config.ID_EMPTY

            # 計算到玩家的路徑
            # 注意：astar_path 會避開 ID_WALL (1)，但會穿過 ID_PLAYER (2) 和其他 ID_MONSTER (4)
            # 這是我們想要的，怪物應該要能追到玩家
            path = astar_path(self.maze, m_pos, self.player_pos)

            if path and len(path) > 1:
                # 移動一步
                next_step = path[1]  # path[0] 是起點
                nx, ny = next_step

                # 檢查目標點是否是牆壁 (astar 應該已經過濾，但雙重確認)
                if self.maze[nx, ny] != config.ID_WALL:
                    new_pos = [nx, ny]
                else:
                    new_pos = [mx, my]
            else:
                new_pos = [mx, my]

            new_monster_positions.append(new_pos)

            # 更新地圖上的怪物位置 (如果是玩家位置，暫時不覆蓋 ID_PLAYER，碰撞檢測會處理)
            if not np.array_equal(new_pos, self.player_pos) and not np.array_equal(
                new_pos, self.exit_pos
            ):
                self.maze[new_pos[0], new_pos[1]] = config.ID_MONSTER

        self.monsters = new_monster_positions

    def _check_collisions(self):
        """檢查玩家是否碰到怪物"""
        # 檢查是否有怪物與玩家座標重疊
        monsters_to_keep = []
        hit = False

        for m_pos in self.monsters:
            if np.array_equal(m_pos, self.player_pos):
                hit = True
                # 碰到怪物，該怪物消失 (避免連續扣血)
                # 不將其加入 monsters_to_keep
            else:
                monsters_to_keep.append(m_pos)

        if hit:
            self.player_hp -= 1
            # print(f"Ouch! HP: {self.player_hp}")

        self.monsters = monsters_to_keep

    def _remove_monster_at(self, x, y):
        """移除特定座標的怪物"""
        self.monsters = [m for m in self.monsters if not (m[0] == x and m[1] == y)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._generate_random_maze()

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.maze[0, 0] = config.ID_PLAYER
        self.current_time = 0
        self.player_hp = config.PLAYER_MAX_HP

        self.exit_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1], dtype=np.int32
        )
        self.maze[-1, -1] = config.ID_EXIT
        self.monsters = []

        if self.render_mode == "human":
            self._render_frame()

        # 修改：回傳處理過的觀測值
        return self._get_obs(), {}

    def _generate_random_maze(self):
        """
        使用 DFS (Recursive Backtracking) 生成隨機迷宮
        注意：為了生成漂亮的迷宮，grid_size 最好是奇數 (例如 15, 17, 21)
        """
        # 先將所有地方填滿牆壁
        self.maze.fill(config.ID_WALL)

        # 起點設為 (0, 0) 並標記為空地
        start_x, start_y = 0, 0
        self.maze[start_x, start_y] = config.ID_EMPTY

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
                    if self.maze[nx, ny] == config.ID_WALL:
                        neighbors.append((nx, ny, dx, dy))

            if neighbors:
                # 隨機選擇一個鄰居
                # 使用 self.rng.choice 需要將 list 轉為 index 選擇，或者手動選擇
                idx = self.rng.integers(0, len(neighbors))
                nx, ny, dx, dy = neighbors[idx]

                # 打通當前格與鄰居中間的牆
                self.maze[x + dx // 2, y + dy // 2] = config.ID_EMPTY

                # 將鄰居設為空地
                self.maze[nx, ny] = config.ID_EMPTY

                # 將鄰居加入堆疊
                stack.append((nx, ny))
            else:
                # 如果沒有未訪問的鄰居，回溯
                stack.pop()

        # 隨機移除一些牆壁以製造迴路 (Braiding)，增加迷宮的強健性
        # 避免 Agent 放一個牆壁就直接把路堵死
        braid_ratio = 0.15  # 15% 的機率移除剩餘的牆壁
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if self.maze[x, y] == config.ID_WALL:
                    # 確保不會破壞邊界，且隨機打通
                    if self.rng.random() < braid_ratio:
                        self.maze[x, y] = config.ID_EMPTY

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + 50)
            )  # 增加高度給 UI
            self.font = pygame.font.SysFont("Arial", 24)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size + 50))
        canvas.fill(config.COLOR_WHITE)

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

                if cell_value == config.ID_WALL:
                    pygame.draw.rect(canvas, config.COLOR_BLACK, rect)
                elif cell_value == config.ID_PLAYER:
                    pygame.draw.rect(canvas, config.COLOR_BLUE, rect)
                elif cell_value == config.ID_EXIT:
                    pygame.draw.rect(canvas, config.COLOR_GREEN, rect)
                elif cell_value == config.ID_MONSTER:
                    pygame.draw.rect(canvas, config.COLOR_RED, rect)

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

        # 繪製 UI (血量)
        hp_text = f"HP: {self.player_hp} / {config.PLAYER_MAX_HP}"
        text_surface = self.font.render(hp_text, True, (0, 0, 0))
        canvas.blit(text_surface, (10, self.window_size + 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
