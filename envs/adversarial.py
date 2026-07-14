"""對抗式訓練環境：玩家視角 (PlayerEnvV2) 與 Maze Master 視角 (MasterEnv)。

與 v1 (player_env.py) 的差異：
- 觀測統一為 8x15x15 平面 (雙方同構)：牆/玩家/出口/怪物 + HP/鎚子/時間/怪物節奏相位廣播面
- 雙方都提供 action_masks() 給 MaskablePPO：非法動作機率直接歸零
  (玩家的「無鎚撞牆」被遮罩 => 卡死問題從規則層根治)
- 回合時序修正：Master 先對局面 S 編輯，對手在編輯後的 S' 上決策
  (v1 是雙方都看編輯前的 S，有一步資訊滯後)
- Master 是零和「拖延者」：玩家活得越久分越高，殺死玩家/太快通關都是失敗

動作編碼 (Master)：Discrete(5*225)，index = 編輯類型*225 + x*15 + y
編輯類型沿用 maze_env：0 skip / 1 蓋牆 / 2 清除 / 3 搬出口 / 4 放怪
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import config
from agents.astar_bot import astar_path
from envs.maze_env import MazeEnv

N_CHANNELS = 8
N_EDIT_TYPES = 5
ACTION_TO_MOVE = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def build_obs(game: MazeEnv) -> np.ndarray:
    """雙方共用的 8 通道觀測 (完全資訊，兩邊看到同一份)"""
    n = game.grid_size
    obs = np.zeros((N_CHANNELS, n, n), dtype=np.float32)
    obs[0] = (game.maze == config.ID_WALL).astype(np.float32)
    obs[1, game.player_pos[0], game.player_pos[1]] = 1.0
    obs[2, game.exit_pos[0], game.exit_pos[1]] = 1.0
    for m in game.monsters:
        obs[3, m[0], m[1]] = 1.0
    obs[4] = game.player_hp / config.PLAYER_MAX_HP
    obs[5] = game.player_hammers / max(config.PLAYER_INITIAL_HAMMERS, 1)
    obs[6] = np.clip(1.0 - game.current_time / config.TIME_MAX, -1.0, 1.0)
    pattern = config.MONSTER_MOVE_PATTERN
    obs[7] = float(pattern[game.turn_count % len(pattern)])  # 怪物本回合會不會動
    return obs


def player_action_masks(game: MazeEnv) -> np.ndarray:
    """玩家 4 個方向的合法性：出界不可走；無鎚時牆不可走"""
    mask = np.zeros(4, dtype=bool)
    for a, (dx, dy) in ACTION_TO_MOVE.items():
        tx, ty = game.player_pos[0] + dx, game.player_pos[1] + dy
        if not (0 <= tx < game.grid_size and 0 <= ty < game.grid_size):
            continue
        if game.maze[tx, ty] == config.ID_WALL and game.player_hammers == 0:
            continue
        mask[a] = True
    if not mask.any():  # 理論上不會發生 (至少來路可走)，防呆全開
        mask[:] = True
    return mask


def master_action_masks(game: MazeEnv) -> np.ndarray:
    """Master 5*225 個動作的合法性 (skip 只保留 (0,0) 一格，消除重複動作)"""
    n = game.grid_size
    wall = game.maze == config.ID_WALL
    monster = np.zeros((n, n), dtype=bool)
    for m in game.monsters:
        monster[m[0], m[1]] = True
    protected = np.zeros((n, n), dtype=bool)
    protected[game.player_pos[0], game.player_pos[1]] = True
    protected[game.exit_pos[0], game.exit_pos[1]] = True
    empty = ~wall & ~monster & ~protected

    mask = np.zeros((N_EDIT_TYPES, n, n), dtype=bool)
    mask[0, 0, 0] = True  # skip
    mask[1] = empty  # 蓋牆
    mask[2] = (wall | monster) & ~protected  # 清除
    mask[3] = empty  # 搬出口
    if len(game.monsters) < config.MAX_MONSTERS:
        mask[4] = empty  # 放怪
    return mask.flatten()


def decode_master_action(action: int):
    """Discrete index -> maze_env 的 (x, y, type) 陣列"""
    edit_type, cell = divmod(int(action), 225)
    x, y = divmod(cell, 15)
    return np.array([x, y, edit_type], dtype=np.int64)


class _AdversarialBase(gym.Env):
    """共用：內部 MazeEnv 的分階段推進 (修正決策時序)"""

    metadata = {"render_modes": ["human"], "render_fps": config.FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = MazeEnv(render_mode=render_mode)
        self.episode_steps = 0
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_CHANNELS, 15, 15), dtype=np.float32)

    def _finish_turn(self):
        """怪物移動 + 碰撞 + 勝負判定，回傳 (terminated, info, hp_lost_here)"""
        hp_before = self.game.player_hp
        self.game._move_monsters()
        self.game._handle_collisions()
        _, terminated, info = self.game._check_game_status()
        self.episode_steps += 1
        return terminated, info, hp_before - self.game.player_hp

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()


class PlayerEnvV2(_AdversarialBase):
    """玩家視角：agent 是玩家，凍結的 Master 是環境動態的一部分"""

    def __init__(self, master_model_path=None, render_mode=None, randomize_hammers=False):
        super().__init__(render_mode)
        config.PLAYER_MODE = "NN"
        self.randomize_hammers = randomize_hammers
        self.action_space = spaces.Discrete(4)
        self.master = None
        if master_model_path is not None:
            from sb3_contrib import MaskablePPO

            self.master = MaskablePPO.load(master_model_path, device="cpu")

    def action_masks(self):
        return player_action_masks(self.game)

    def _exit_distance(self):
        path = astar_path(self.game.maze, self.game.player_pos, self.game.exit_pos)
        return len(path) - 1 if path else None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        if self.randomize_hammers:
            self.game.player_hammers = int(self.game.rng.integers(0, config.PLAYER_INITIAL_HAMMERS + 1))
        self.episode_steps = 0
        return build_obs(self.game), {}

    def step(self, action):
        # 1. 凍結 Master 先編輯 (它看的是玩家行動前的局面)
        if self.master is not None:
            m_obs, m_mask = build_obs(self.game), master_action_masks(self.game)
            m_act, _ = self.master.predict(m_obs, action_masks=m_mask, deterministic=False)
            self.game._execute_maze_master_actions(decode_master_action(m_act))

        # 2. 玩家在編輯後的局面上行動
        hp_before = self.game.player_hp
        dist_before = self._exit_distance()
        dx, dy = ACTION_TO_MOVE[int(action)]
        self.game.set_player_move(dx, dy)
        _, player_done, player_info = self.game._handle_player_turn()

        # 3. 怪物 + 碰撞 + 判定
        terminated, info, _ = self._finish_turn()
        terminated = terminated or player_done
        if player_info:
            info.update(player_info)

        # 4. 玩家獎勵 (沿用 v1 的速通配方；遮罩已消滅撞牆，不再需要 bump 懲罰)
        reward = config.PLAYER_REWARD_STEP
        hp_lost = hp_before - self.game.player_hp
        if hp_lost > 0:
            reward += config.PLAYER_REWARD_HIT * hp_lost
        result = info.get("result")
        if result in ("flow_success", "too_fast", "too_slow"):
            reward += config.PLAYER_REWARD_GOAL
            time_left = config.TIME_MAX * 1.5 - self.game.current_time
            reward += config.PLAYER_REWARD_SPEED_BONUS * max(0.0, time_left)
        elif result == "died":
            reward += config.PLAYER_REWARD_DEATH
        elif result == "timeout":
            reward += config.PLAYER_REWARD_TIMEOUT
        elif not terminated and dist_before is not None:
            dist_after = self._exit_distance()
            if dist_after is not None:
                reward += float(np.clip(config.PLAYER_REWARD_DIST * (dist_before - dist_after), -0.5, 0.5))

        truncated = self.episode_steps >= config.PLAYER_MAX_EPISODE_STEPS
        return build_obs(self.game), reward, terminated, truncated, info


class MasterEnv(_AdversarialBase):
    """Master 視角：agent 是 Maze Master，對手玩家凍結。

    opponent:
      ("astar", None)  A* bot (帶隨機個性，同原版 train.py)
      ("v2", path)     MaskablePPO 玩家 (本框架訓練的)
      ("nn4", path)    v1 的 4 通道 PPO 玩家 (速通版舊模型)
    """

    def __init__(self, opponent=("astar", None), render_mode=None):
        super().__init__(render_mode)
        self.action_space = spaces.Discrete(N_EDIT_TYPES * 225)
        self.opp_kind, opp_path = opponent
        self.opp_model = None
        if self.opp_kind == "v2":
            from sb3_contrib import MaskablePPO

            self.opp_model = MaskablePPO.load(opp_path, device="cpu")
        elif self.opp_kind == "nn4":
            from stable_baselines3 import PPO

            self.opp_model = PPO.load(opp_path, device="cpu")
        config.PLAYER_MODE = "AI" if self.opp_kind == "astar" else "NN"

    def action_masks(self):
        return master_action_masks(self.game)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self.episode_steps = 0
        return build_obs(self.game), {}

    def _opponent_move(self):
        """對手玩家在 Master 編輯後的局面上走一步"""
        if self.opp_kind == "astar":
            return self.game._handle_player_turn()  # AI 模式內部自己算 A*
        if self.opp_kind == "v2":
            obs, mask = build_obs(self.game), player_action_masks(self.game)
            act, _ = self.opp_model.predict(obs, action_masks=mask, deterministic=False)
        else:  # nn4
            from envs.player_env import build_player_obs

            act, _ = self.opp_model.predict(build_player_obs(self.game), deterministic=False)
        self.game.set_player_move(*ACTION_TO_MOVE[int(act)])
        return self.game._handle_player_turn()

    def step(self, action):
        # 1. Master 編輯 (先清空訊息，避免上一步的「撤銷」殘留造成誤判)
        self.game.last_ai_action = ""
        self.game._execute_maze_master_actions(decode_master_action(action))
        blocked_try = "撤銷" in self.game.last_ai_action

        # 2. 對手玩家行動 -> 3. 怪物/碰撞/判定
        _, player_done, player_info = self._opponent_move()
        terminated, info, _ = self._finish_turn()
        terminated = terminated or player_done
        if player_info:
            info.update(player_info)

        # 4. Master 零和獎勵：拖延得分，殺人/被速通失分
        reward = 0.0
        if blocked_try:
            reward += config.MASTER_REWARD_BLOCKED_TRY
        if not terminated:
            reward += config.MASTER_REWARD_PER_TURN
        else:
            result = info.get("result")
            if result in ("died", "blocked"):
                reward += config.MASTER_REWARD_PLAYER_DIED
            elif result == "too_fast":
                reward += config.MASTER_REWARD_TOO_FAST

        truncated = self.episode_steps >= config.PLAYER_MAX_EPISODE_STEPS
        return build_obs(self.game), reward, terminated, truncated, info
