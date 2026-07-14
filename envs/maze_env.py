import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config
from agents.astar_bot import astar_path

# 匯入拆分出去的模組
from envs.maze_generator import MazeGenerator
from envs.rendering import MazeRenderer


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": config.FPS}

    def __init__(self, render_mode=None):
        super(MazeEnv, self).__init__()

        # 載入設定
        self.grid_size = config.GRID_SIZE
        self.render_mode = render_mode
        self.rng = np.random.default_rng(114514)

        # 初始化渲染器
        self.renderer = MazeRenderer(config.WINDOW_SIZE, self.grid_size, config.FPS)

        # 狀態變數
        self.current_time = 0
        self.player_hp = config.PLAYER_MAX_HP
        self.monsters = []
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.player_pos = np.array([0, 0])
        self.exit_pos = np.array([0, 0])
        self.player_steps = 0
        self.player_hammers = 0

        # 定義動作空間
        n_actions = config.ACTIONS_PER_TURN
        action_dims = []
        for _ in range(n_actions):
            action_dims.extend([self.grid_size, self.grid_size, 5])
        self.action_space = spaces.MultiDiscrete(action_dims)

        # 定義觀測空間
        self.scale_factor = 4
        obs_size = self.grid_size * self.scale_factor
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, obs_size, obs_size), dtype=np.uint8
        )

        # 手動操作變數
        self.manual_move = (0, 0)

        # 給渲染器顯示的 AI 決策描述
        self.last_ai_action = "等待第一步"

        # 玩家個性與行為統計 (實際值於 reset 設定)
        self.player_randomness = config.AI_RANDOMNESS
        self.player_hesitate_prob = 0.0
        self.action_counts = {"wall": 0, "monster": 0, "exit": 0, "remove": 0, "skip": 0}

        # === 新增：允許的動作列表 (預設全部允許) ===
        # 0: Skip, 1: Wall, 2: Remove, 3: Exit, 4: Monster
        self.allowed_actions = {0, 1, 2, 3, 4}

    def set_allowed_actions(self, actions_set):
        """設定當前環境允許執行的動作類型"""
        self.allowed_actions = set(actions_set)
        # 0 (Skip) 永遠必須被允許，否則模型會崩潰
        self.allowed_actions.add(0)

    def set_player_move(self, dx, dy):
        self.manual_move = (dx, dy)

    def step(self, action):
        """
        環境的主要步進函式，現在被拆解為多個階段
        """
        # 1. 執行 Maze Master 的編輯動作
        edit_reward = self._execute_maze_master_actions(action)

        # 2. 計算路徑並處理玩家移動 (AI 或 Human)
        player_reward, player_done, player_info = self._handle_player_turn()

        # 3. 怪物移動
        self._move_monsters()

        # 4. 處理碰撞 (玩家 vs 怪物)
        collision_reward = self._handle_collisions()

        # 5. 判定遊戲結束狀態 (勝利/死亡/超時)
        status_reward, status_done, status_info = self._check_game_status()

        # 總結
        total_reward = edit_reward + player_reward + collision_reward + status_reward
        terminated = player_done or status_done
        truncated = False

        # 合併 info
        info = {}
        if player_info:
            info.update(player_info)
        if status_info:
            info.update(status_info)

        # 回合結束時顯示結果橫幅
        if terminated and self.render_mode == "human":
            banner_map = {
                "died": ("你被怪物擊倒了", (192, 57, 43)),
                "blocked": ("路徑被堵死", (192, 57, 43)),
                "flow_success": ("完美通關！落在 AI 目標區間", (39, 174, 96)),
                "too_fast": ("通關 — 但比 AI 預期更快", (211, 158, 15)),
                "too_slow": ("通關 — 但比 AI 預期更慢", (211, 158, 15)),
                "timeout": ("超時了", (127, 140, 141)),
            }
            text, color = banner_map.get(info.get("result"), ("回合結束", (127, 140, 141)))
            self.renderer.set_banner(text, color)

        # 觀測與渲染
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, total_reward, terminated, truncated, info

    # =========================================================================
    # 第一階段：Maze Master 編輯邏輯
    # =========================================================================

    def _execute_maze_master_actions(self, action):
        """解析並執行編輯動作"""
        reward = 0
        reshaped_actions = action.reshape(config.ACTIONS_PER_TURN, 3)
        for act in reshaped_actions:
            x, y, action_type = act
            reward += self._apply_single_action(x, y, action_type)
        return reward

    def _fx(self, kind, x=0, y=0):
        """轉送視覺特效事件給渲染器 (僅人眼模式，訓練時不累積)"""
        if self.render_mode == "human":
            self.renderer.add_effect(kind, x, y)

    def _action_place_wall(self, x, y):
        """改進版放牆邏輯：增加更細緻的獎勵"""
        # 先計算放牆前的路徑
        old_path = astar_path(self.maze, self.player_pos, self.exit_pos)
        old_len = len(old_path) if old_path else 0

        # 嘗試放牆
        self.maze[x, y] = config.ID_WALL

        # 檢查是否堵死
        new_path = astar_path(self.maze, self.player_pos, self.exit_pos)

        if new_path is None:
            self.maze[x, y] = config.ID_EMPTY  # 撤銷
            self.last_ai_action = f"想堵死路徑 ({x}, {y}) → 被規則撤銷"
            return config.REWARD_BLOCKED

        self._fx("wall", x, y)
        self.last_ai_action = f"蓋牆 ({x}, {y})"
        self.action_counts["wall"] += 1
        new_len = len(new_path)
        reward = config.REWARD_BUILD_WALL

        # 獎勵放牆後路徑變長
        if new_len > old_len:
            path_bonus = (new_len - old_len) * config.REWARD_PATH_EXTEND
            reward += min(path_bonus, 15.0)  # 設上限避免過度

        # 額外獎勵：牆放在原路徑上（迫使玩家繞路）
        if old_path and (x, y) in old_path:
            reward += config.REWARD_WALL_NEAR_PATH

        return reward

    def _apply_single_action(self, x, y, action_type):
        """分發單一編輯動作（增加跳過動作的懲罰）"""

        # === 新增：檢查動作是否被允許 ===
        if action_type not in self.allowed_actions:
            return config.REWARD_SKIP_ACTION  # 如果動作被禁用，視為 Skip

        # 保護機制：不能修改玩家或出口當前位置
        if np.array_equal([x, y], self.player_pos) or np.array_equal(
            [x, y], self.exit_pos
        ):
            return config.REWARD_SKIP_ACTION  # 小懲罰

        # action_type == 0 表示不做事
        if action_type == 0:
            self.last_ai_action = "按兵不動"
            self.action_counts["skip"] += 1
            return config.REWARD_SKIP_ACTION

        # 保護機制：除非是移除動作，否則不能覆蓋非空格子 (牆或怪物)
        occupied = self.maze[x, y] != config.ID_EMPTY or any(
            m[0] == x and m[1] == y for m in self.monsters
        )
        if occupied and action_type != 2:
            return config.REWARD_SKIP_ACTION

        if action_type == 1:
            return self._action_place_wall(x, y)
        elif action_type == 2:
            return self._action_remove(x, y)
        elif action_type == 3:
            return self._action_move_exit(x, y)
        elif action_type == 4:
            return self._action_place_monster(x, y)
        return 0

    def _action_remove(self, x, y):
        self._remove_monster_at(x, y)
        self.maze[x, y] = config.ID_EMPTY
        self._fx("remove", x, y)
        self.last_ai_action = f"清除格子 ({x}, {y})"
        self.action_counts["remove"] += 1
        return 0

    def _action_move_exit(self, x, y):
        old_exit_pos = self.exit_pos.copy()
        self.exit_pos = np.array([x, y], dtype=np.int32)

        path = astar_path(self.maze, self.player_pos, self.exit_pos)
        if path is None:
            self.exit_pos = old_exit_pos  # 撤銷
            return config.REWARD_BLOCKED

        # 反掛機：出口不能搬到玩家附近，否則玩家可原地來回走等出口送上門
        if len(path) - 1 < config.EXIT_MIN_PLAYER_DIST:
            self.exit_pos = old_exit_pos  # 撤銷
            self.last_ai_action = f"想把出口搬到玩家附近 ({x}, {y}) → 被規則撤銷"
            return config.REWARD_BLOCKED

        self._fx("exit", x, y)
        self.last_ai_action = f"搬移出口 → ({x}, {y})"
        self.action_counts["exit"] += 1
        return config.REWARD_MOVE_EXIT

    def _action_place_monster(self, x, y):
        if len(self.monsters) < config.MAX_MONSTERS:
            self.monsters.append([x, y])
            self._fx("monster", x, y)
            self.last_ai_action = f"放出怪物 ({x}, {y})"
            self.action_counts["monster"] += 1
        return 0

    # =========================================================================
    # 第二階段：玩家移動邏輯
    # =========================================================================

    def _handle_player_turn(self):
        """處理玩家的回合 (AI 或 Human)"""
        path = []

        # 如果是 AI 模式，或者雖然是 Human 模式但需要導航輔助
        if config.PLAYER_MODE == "AI":
            # 猶豫：模擬真實玩家停下來思考 (該回合不移動，Maze Master 仍可編輯)
            if self.rng.random() < self.player_hesitate_prob:
                return 0, False, {}

            path = astar_path(
                self.maze,
                self.player_pos,
                self.exit_pos,
                randomness=self.player_randomness,
            )
            return self._move_ai_player(path)

        elif config.PLAYER_MODE == "HUMAN":
            # 人類操作邏輯...
            return self._move_human_player()

        return 0, False, {}

    def _move_ai_player(self, path):
        reward = 0
        terminated = False
        info = {}

        if path is None:
            reward += config.REWARD_BLOCKED
            terminated = True
            info["result"] = "blocked"
            return reward, terminated, info

        steps_can_move = len(path) - 1
        steps_this_turn = min(steps_can_move, config.K_STEP)

        if steps_this_turn > 0:
            # 模擬移動過程中的碰撞 (AI 移動時可能會穿過怪物)
            path_segment = path[1 : steps_this_turn + 1]
            hit_count = self._check_path_collision(path_segment)

            if hit_count > 0:
                self.player_hp -= hit_count
                reward += config.REWARD_HIT * hit_count
                self._fx("hit")

            # 更新位置
            self._update_player_position(path[steps_this_turn])
            self.current_time += steps_this_turn
            self.player_steps += steps_this_turn

        return reward, terminated, info

    def _move_human_player(self):
        dx, dy = self.manual_move
        self.manual_move = (0, 0)  # 立即重置指令

        if dx == 0 and dy == 0:
            return 0, False, {}

        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy

        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return 0, False, {}

        target_cell = self.maze[new_x, new_y]
        moved = False

        if target_cell != config.ID_WALL:
            moved = True
        elif self.player_hammers > 0:
            self.player_hammers -= 1
            self.maze[new_x, new_y] = config.ID_EMPTY
            self._fx("hammer", new_x, new_y)
            moved = True
            print(f"使用了破牆工具！剩餘次數: {self.player_hammers}")
        else:
            self._fx("bump", new_x, new_y)  # 撞牆且無鎚可用，至少給個視覺回饋

        if moved:
            self.current_time += 1
            self.player_steps += 1
            self._update_player_position([new_x, new_y])

        return 0, False, {}

    def _update_player_position(self, new_pos):
        """更新玩家座標 (實體與地形分離，不再寫入 maze 格子)"""
        self.player_pos = np.array(new_pos)

    def _check_path_collision(self, path_segment):
        """檢查 AI 移動路徑上是否會撞到怪物"""
        hit_count = 0
        monsters_to_keep = []

        for m_pos in self.monsters:
            is_hit = False
            for p_pos in path_segment:
                if np.array_equal(m_pos, p_pos):
                    is_hit = True
                    break

            if is_hit:
                hit_count += 1
            else:
                monsters_to_keep.append(m_pos)

        self.monsters = monsters_to_keep
        return hit_count

    # =========================================================================
    # 第三、四階段：怪物與碰撞
    # =========================================================================

    def _move_monsters(self):
        """所有怪物使用 A* 向玩家移動 (maze 只含地形，可直接尋路)"""
        new_monster_positions = []

        for m_pos in self.monsters:
            path = astar_path(self.maze, m_pos, self.player_pos)
            if path and len(path) > 1:
                new_monster_positions.append(list(path[1]))
            else:
                new_monster_positions.append(list(m_pos))

        self.monsters = new_monster_positions

    def _handle_collisions(self):
        """檢查玩家當前位置是否與怪物重疊"""
        monsters_to_keep = []
        hit = False

        for m_pos in self.monsters:
            if np.array_equal(m_pos, self.player_pos):
                hit = True
            else:
                monsters_to_keep.append(m_pos)

        self.monsters = monsters_to_keep

        if hit:
            self.player_hp -= 1
            self._fx("hit")
            # 這裡不回傳獎勵，因為通常是 AI 移動時主動撞怪才給獎勵，
            # 或是怪物移動後撞到玩家。這裡可以視需求增加被動受傷的獎勵。
            return config.REWARD_HIT
        return 0

    def _remove_monster_at(self, x, y):
        self.monsters = [m for m in self.monsters if not (m[0] == x and m[1] == y)]

    # =========================================================================
    # 第五階段：狀態判定
    # =========================================================================

    def _check_game_status(self):
        """判定遊戲是否結束及對應獎勵"""
        reward = 0
        terminated = False
        info = {}

        if self.player_hp <= 0:
            terminated = True
            reward += config.REWARD_DEATH
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
            # 超時判定
            if self.current_time > config.TIME_MAX * 1.5:
                terminated = True
                reward += config.REWARD_TIMEOUT
                info["result"] = "timeout"
            # 存活獎勵 (鼓勵遊戲持續進行)
            if self.current_time > 30:
                reward += config.REWARD_STEP

        return reward, terminated, info

    # =========================================================================
    # 輔助函式
    # =========================================================================

    def _entity_grid(self):
        """將實體 (怪物/出口/玩家) 疊合到地形上，優先序：玩家 > 出口 > 怪物"""
        grid = self.maze.copy()
        for m in self.monsters:
            grid[m[0], m[1]] = config.ID_MONSTER
        grid[self.exit_pos[0], self.exit_pos[1]] = config.ID_EXIT
        grid[self.player_pos[0], self.player_pos[1]] = config.ID_PLAYER
        return grid

    def _get_obs(self):
        """產生觀測值"""
        obs = self._entity_grid().astype(np.uint8)
        obs = obs * 50
        obs = np.repeat(
            np.repeat(obs, self.scale_factor, axis=0), self.scale_factor, axis=1
        )
        obs = np.expand_dims(obs, axis=0)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # maze 只保存地形 (牆/空地)，實體另外用座標追蹤
        self.maze = MazeGenerator.generate(self.grid_size, self.rng)

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.current_time = 0
        self.player_steps = 0
        self.player_hp = config.PLAYER_MAX_HP
        self.player_hammers = config.PLAYER_INITIAL_HAMMERS

        self.exit_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1], dtype=np.int32
        )
        self.monsters = []
        self.last_ai_action = "等待第一步"

        # 玩家個性：固定值 (demo/評估) 或每回合隨機抽樣 (訓練泛化)
        if config.PLAYER_PROFILE_RANDOMIZE:
            self.player_randomness = self.rng.uniform(*config.PLAYER_RANDOMNESS_RANGE)
            self.player_hesitate_prob = self.rng.uniform(*config.PLAYER_HESITATE_RANGE)
        else:
            self.player_randomness = config.AI_RANDOMNESS
            self.player_hesitate_prob = 0.0

        # Maze Master 行為統計 (供動態調整分析用)
        self.action_counts = {"wall": 0, "monster": 0, "exit": 0, "remove": 0, "skip": 0}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            self.renderer.render(
                self.maze,
                self.player_pos,
                self.exit_pos,
                self.monsters,
                self.player_hp,
                self.player_hammers,
                self.player_steps,
                self.current_time,
                self.last_ai_action,
            )

    def close(self):
        self.renderer.close()
