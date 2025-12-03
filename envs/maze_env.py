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

        # 新增：儲存手動操作的下一步方向 (dx, dy)
        self.manual_move = (0, 0)

    def set_player_move(self, dx, dy):
        """外部呼叫此函式來設定玩家下一步的移動方向"""
        self.manual_move = (dx, dy)

    def step(self, action):
        # 0. 計算動作前的路徑長度 (作為基準，計算獎勵用，即使是人類玩，AI 仍需知道路徑狀況)
        old_path = astar_path(self.maze, self.player_pos, self.exit_pos)
        old_path_len = len(old_path) if old_path else 0

        # 初始化本回合獎勵
        reward = 0

        # 1. 解析並執行 Maze Master 的多重動作
        # action array 形狀為 (ACTIONS_PER_TURN * 3,)
        reshaped_actions = action.reshape(config.ACTIONS_PER_TURN, 3)

        for act in reshaped_actions:
            x, y, action_type = act
            # 累加動作產生的獎勵 (例如移動出口的懲罰)
            reward += self._apply_action(x, y, action_type)

        # 確保玩家和出口標記正確
        self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_PLAYER
        self.maze[self.exit_pos[0], self.exit_pos[1]] = config.ID_EXIT

        # 2. 玩家移動邏輯 (區分 AI 與 HUMAN)
        path = astar_path(
            self.maze, self.player_pos, self.exit_pos
        )  # 重新計算路徑用於獎勵判定

        # reward = 0  <-- 移除這行，避免覆蓋掉 _apply_action 產生的獎勵
        terminated = False
        truncated = False
        info = {}

        # --- 計算路徑長度變化獎勵 (保持不變，讓 Maze Master 繼續學習) ---
        if path is not None and old_path is not None:
            new_path_len = len(path)
            if new_path_len > old_path_len:
                diff = new_path_len - old_path_len
                reward += min(diff, 10) * config.REWARD_PATH_EXTEND

        # --- 移動處理 ---
        if config.PLAYER_MODE == "AI":
            # === 原有的 A* 秬動邏輯 ===
            if path is None:
                reward += config.REWARD_BLOCKED
                terminated = True
                info["result"] = "blocked"
            else:
                steps_can_move = len(path) - 1
                steps_this_turn = min(steps_can_move, config.K_STEP)

                if steps_this_turn > 0:
                    # 清除舊位置
                    if not np.array_equal(self.player_pos, self.exit_pos):
                        self.maze[self.player_pos[0], self.player_pos[1]] = (
                            config.ID_EMPTY
                        )

                    # 檢查移動路徑上是否碰撞怪物
                    path_segment = path[1 : steps_this_turn + 1]
                    monsters_to_keep = []
                    hit_count = 0

                    for m_pos in self.monsters:
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
                        reward += config.REWARD_HIT * hit_count
                        self.monsters = monsters_to_keep

                    # 更新位置
                    new_pos = path[steps_this_turn]
                    self.player_pos = np.array(new_pos)
                    self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_PLAYER
                    self.current_time += steps_this_turn

        elif config.PLAYER_MODE == "HUMAN":
            # === 新增的手動移動邏輯 ===
            dx, dy = self.manual_move

            # 如果路被完全堵死，還是要判斷輸贏 (可選)
            if path is None:
                pass

            if dx != 0 or dy != 0:
                new_x = self.player_pos[0] + dx
                new_y = self.player_pos[1] + dy

                # 檢查邊界
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                    target_cell = self.maze[new_x, new_y]
                    moved = False

                    # 情況 1: 一般移動 (空地、出口、怪物)
                    if target_cell != config.ID_WALL:
                        moved = True

                    # 情況 2: 破牆 (是牆壁 且 有工具)
                    elif target_cell == config.ID_WALL and self.player_hammers > 0:
                        self.player_hammers -= 1
                        self.maze[new_x, new_y] = config.ID_EMPTY  # 破壞牆壁
                        moved = True
                        print(f"使用了破牆工具！剩餘次數: {self.player_hammers}")

                    # 執行移動
                    if moved:
                        # 還原舊位置 (如果是出口要保留出口 ID)
                        if not np.array_equal(self.player_pos, self.exit_pos):
                            self.maze[self.player_pos[0], self.player_pos[1]] = (
                                config.ID_EMPTY
                            )
                        else:
                            self.maze[self.player_pos[0], self.player_pos[1]] = (
                                config.ID_EXIT
                            )

                        # 更新玩家座標
                        self.player_pos = np.array([new_x, new_y])
                        self.maze[new_x, new_y] = config.ID_PLAYER

            # 重置手動指令，避免下一幀自動移動
            self.manual_move = (0, 0)

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
        action_reward = 0

        # 保護機制：不能修改玩家或出口當前位置
        if np.array_equal([x, y], self.player_pos) or np.array_equal(
            [x, y], self.exit_pos
        ):
            return action_reward

        # 取得當前格子的狀態
        current_cell = self.maze[x, y]

        # 保護機制：如果該格子已經有東西 (牆壁、怪物)，則不允許覆蓋
        # 這樣可以防止牆壁把怪物吃掉，或是怪物重疊
        if current_cell != config.ID_EMPTY:
            # 唯一的例外：如果是「移除」動作 (action_type == 2)，允許移除牆壁或怪物
            if action_type != 2:
                return action_reward

        if action_type == 1:  # 放置牆壁
            self.maze[x, y] = config.ID_WALL

        elif action_type == 2:  # 移除 (變成空地)
            # 如果原本是怪物，要從怪物列表中移除
            if current_cell == config.ID_MONSTER:
                self._remove_monster_at(x, y)
            self.maze[x, y] = config.ID_EMPTY

        elif action_type == 3:  # 放置出口
            # 修改：允許 AI 移動出口
            # 1. 將舊的出口位置還原為空地
            old_x, old_y = self.exit_pos
            self.maze[old_x, old_y] = config.ID_EMPTY

            # 2. 更新出口座標變數
            self.exit_pos = np.array([x, y], dtype=np.int32)

            # 3. 在地圖矩陣上標記新出口
            self.maze[x, y] = config.ID_EXIT

            # 懲罰移動出口
            action_reward = config.REWARD_MOVE_EXIT

        elif action_type == 4:  # 放置怪物
            if len(self.monsters) < config.MAX_MONSTERS:
                self.maze[x, y] = config.ID_MONSTER
                self.monsters.append([x, y])
            else:
                # (可選) 如果你想讓 Agent 知道「不能再放了」，可以在這裡做個標記
                # 但通常只要動作無效，Agent 慢慢就會學到
                pass

        return action_reward

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
        self.player_hammers = config.PLAYER_INITIAL_HAMMERS  # 新增：初始化破牆工具

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
        使用 DFS 生成基礎迷宮，然後隨機打通牆壁以製造多條路徑 (Braid Maze)
        """
        # 1. 初始化：全填滿牆壁
        self.maze.fill(config.ID_WALL)

        # 起點設為 (0, 0)
        start_x, start_y = 0, 0
        self.maze[start_x, start_y] = config.ID_EMPTY

        # 2. DFS 生成完美迷宮 (確保連通性)
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]

            # 尋找周圍距離為 2 的未訪問鄰居 (跨過一面牆)
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.maze[nx, ny] == config.ID_WALL:
                        neighbors.append((nx, ny, dx // 2, dy // 2))

            if neighbors:
                # 隨機選一個鄰居
                nx, ny, wx, wy = self.rng.choice(neighbors)
                # 打通中間的牆
                self.maze[x + wx, y + wy] = config.ID_EMPTY
                # 標記鄰居為通路
                self.maze[nx, ny] = config.ID_EMPTY
                stack.append((nx, ny))
            else:
                stack.pop()

        # 3. [新增] 隨機移除牆壁以製造多條路徑 (Loops)
        # loop_probability: 每個牆壁被移除的機率
        # 0.0 = 完美迷宮 (死路多), 1.0 = 空地
        loop_probability = 0.15  # 建議 0.1 ~ 0.2

        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if self.maze[x, y] == config.ID_WALL:
                    # 檢查是否為內部牆壁 (不破壞邊界)
                    # 且隨機骰子命中
                    if self.rng.random() < loop_probability:
                        # 額外檢查：避免產生過於空曠的區域 (可選)
                        # 這裡直接移除，讓迷宮更開放
                        self.maze[x, y] = config.ID_EMPTY

        # 確保出口附近是空的 (雖然 DFS 通常會處理，但保險起見)
        self.maze[self.grid_size - 1, self.grid_size - 1] = config.ID_EMPTY
        self.maze[self.grid_size - 2, self.grid_size - 1] = config.ID_EMPTY
        self.maze[self.grid_size - 1, self.grid_size - 2] = config.ID_EMPTY

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

        # 新增：繪製破牆工具數量
        hammer_text = f"Hammer: {self.player_hammers}"
        hammer_surface = self.font.render(hammer_text, True, (0, 0, 0))
        canvas.blit(hammer_surface, (150, self.window_size + 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
