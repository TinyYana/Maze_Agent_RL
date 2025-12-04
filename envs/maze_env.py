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

        # 觀測與渲染
        obs = self._get_obs()
        if self.render_mode == "human":
            self.renderer.render(
                self.maze, self.player_hp, self.player_hammers, self.player_steps
            )

        return obs, total_reward, terminated, truncated, info

    # =========================================================================
    # 第一階段：Maze Master 編輯邏輯
    # =========================================================================

    def _execute_maze_master_actions(self, action):
        """解析並執行編輯動作"""
        reward = 0
        # 計算動作前的路徑長度 (作為基準)
        old_path = astar_path(self.maze, self.player_pos, self.exit_pos)
        self.old_path_len = len(old_path) if old_path else 0

        reshaped_actions = action.reshape(config.ACTIONS_PER_TURN, 3)
        for act in reshaped_actions:
            x, y, action_type = act
            reward += self._apply_single_action(x, y, action_type)

        # 確保標記正確 (防止被覆蓋)
        self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_PLAYER
        self.maze[self.exit_pos[0], self.exit_pos[1]] = config.ID_EXIT

        return reward

    def _apply_single_action(self, x, y, action_type):
        """分發單一編輯動作"""
        # 保護機制：不能修改玩家或出口當前位置
        if np.array_equal([x, y], self.player_pos) or np.array_equal(
            [x, y], self.exit_pos
        ):
            return 0

        # 保護機制：除非是移除動作，否則不能覆蓋非空格子
        current_cell = self.maze[x, y]
        if current_cell != config.ID_EMPTY and action_type != 2:
            return 0

        if action_type == 1:
            return self._action_place_wall(x, y)
        elif action_type == 2:
            return self._action_remove(x, y)
        elif action_type == 3:
            return self._action_move_exit(x, y)
        elif action_type == 4:
            return self._action_place_monster(x, y)
        return 0

    def _is_path_blocked(self):
        """檢查當前迷宮是否還有路"""
        return astar_path(self.maze, self.player_pos, self.exit_pos) is None

    def _action_place_wall(self, x, y):
        self.maze[x, y] = config.ID_WALL
        if self._is_path_blocked():
            self.maze[x, y] = config.ID_EMPTY  # 撤銷
            return config.REWARD_BLOCKED
        return 0

    def _action_remove(self, x, y):
        if self.maze[x, y] == config.ID_MONSTER:
            self._remove_monster_at(x, y)
        self.maze[x, y] = config.ID_EMPTY
        return 0

    def _action_move_exit(self, x, y):
        old_exit_pos = self.exit_pos.copy()
        old_cell_value = self.maze[x, y]

        # 移動出口
        self.maze[old_exit_pos[0], old_exit_pos[1]] = config.ID_EMPTY
        self.exit_pos = np.array([x, y], dtype=np.int32)
        self.maze[x, y] = config.ID_EXIT

        if self._is_path_blocked():
            # 撤銷
            self.maze[x, y] = old_cell_value
            self.exit_pos = old_exit_pos
            self.maze[old_exit_pos[0], old_exit_pos[1]] = config.ID_EXIT
            return config.REWARD_BLOCKED
        return config.REWARD_MOVE_EXIT

    def _action_place_monster(self, x, y):
        if len(self.monsters) < config.MAX_MONSTERS:
            self.maze[x, y] = config.ID_MONSTER
            self.monsters.append([x, y])
        return 0

    # =========================================================================
    # 第二階段：玩家移動邏輯
    # =========================================================================

    def _handle_player_turn(self):
        """處理玩家移動與路徑獎勵"""
        reward = 0
        terminated = False
        info = {}

        # 重新計算路徑
        path = astar_path(self.maze, self.player_pos, self.exit_pos)

        # 計算路徑長度變化獎勵 (鼓勵 Maze Master 延長路徑)
        if path is not None and self.old_path_len > 0:
            new_path_len = len(path)
            if new_path_len > self.old_path_len:
                diff = new_path_len - self.old_path_len
                reward += min(diff, 10) * config.REWARD_PATH_EXTEND

        # 根據模式執行移動
        if config.PLAYER_MODE == "AI":
            r, t, i = self._move_ai_player(path)
        elif config.PLAYER_MODE == "HUMAN":
            r, t, i = self._move_human_player()
        else:
            r, t, i = 0, False, {}

        reward += r
        terminated = terminated or t
        info.update(i)

        return reward, terminated, info

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
        elif target_cell == config.ID_WALL and self.player_hammers > 0:
            self.player_hammers -= 1
            self.maze[new_x, new_y] = config.ID_EMPTY
            moved = True
            print(f"使用了破牆工具！剩餘次數: {self.player_hammers}")

        if moved:
            self.player_steps += 1
            self._update_player_position([new_x, new_y])

        return 0, False, {}

    def _update_player_position(self, new_pos):
        """更新玩家座標並處理地圖標記"""
        # 清除舊位置 (如果是出口則保留出口標記)
        if not np.array_equal(self.player_pos, self.exit_pos):
            self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_EMPTY
        else:
            self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_EXIT

        self.player_pos = np.array(new_pos)
        self.maze[self.player_pos[0], self.player_pos[1]] = config.ID_PLAYER

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
        """所有怪物使用 A* 向玩家移動"""
        new_monster_positions = []

        for m_pos in self.monsters:
            mx, my = m_pos
            self.maze[mx, my] = config.ID_EMPTY  # 暫時清除以便尋路

            path = astar_path(self.maze, m_pos, self.player_pos)

            if path and len(path) > 1:
                next_step = path[1]
                nx, ny = next_step
                # 簡單檢查：不走進牆壁
                if self.maze[nx, ny] != config.ID_WALL:
                    new_pos = [nx, ny]
                else:
                    new_pos = [mx, my]
            else:
                new_pos = [mx, my]

            new_monster_positions.append(new_pos)

            # 更新地圖標記 (不覆蓋玩家或出口)
            if not np.array_equal(new_pos, self.player_pos) and not np.array_equal(
                new_pos, self.exit_pos
            ):
                self.maze[new_pos[0], new_pos[1]] = config.ID_MONSTER

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
            # 這裡不回傳獎勵，因為通常是 AI 移動時主動撞怪才給獎勵，
            # 或是怪物移動後撞到玩家。這裡可以視需求增加被動受傷的獎勵。
            return 0
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

    def _get_obs(self):
        """產生觀測值"""
        obs = self.maze.astype(np.uint8)
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

        self.maze = MazeGenerator.generate(self.grid_size, self.rng)

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.maze[0, 0] = config.ID_PLAYER
        self.current_time = 0
        self.player_steps = 0
        self.player_hp = config.PLAYER_MAX_HP
        self.player_hammers = config.PLAYER_INITIAL_HAMMERS

        self.exit_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1], dtype=np.int32
        )
        self.maze[-1, -1] = config.ID_EXIT
        self.monsters = []

        if self.render_mode == "human":
            self.renderer.render(
                self.maze, self.player_hp, self.player_hammers, self.player_steps
            )

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            self.renderer.render(
                self.maze, self.player_hp, self.player_hammers, self.player_steps
            )

    def close(self):
        self.renderer.close()
