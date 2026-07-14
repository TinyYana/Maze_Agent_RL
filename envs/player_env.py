"""玩家視角的 Gymnasium 環境：NN 玩家在裡面學「走迷宮 + 躲怪」。

包一層 MazeEnv 來重用全部遊戲規則 (撞牆耗鎚、怪物 A* 追蹤、心流判定)，
角色對調：這裡的 agent 是「玩家」，Maze Master 變成環境動態的一部分——
- 第一階段 (master_model=None)：Maze Master 每回合按兵不動，純靜態迷宮。
- 第二階段 (傳入已訓練的 maze_master_ppo)：凍結的 Master 每步照常編輯迷宮。

觀測 (Dict)：
- grid:  4x15x15 one-hot (牆 / 玩家 / 出口 / 怪物)
- state: [HP, 鎚子, 剩餘時間] 各自正規化

動作 Discrete(4)：上/下/左/右，與 main.py 的 MOVE_KEYS 同順序。

獎勵與 Maze Master 那套完全分開，定義在 config.PLAYER_REWARD_*。
其中距離塑形 (potential-based shaping) 用 A* 最短距離的變化當即時回饋，
解決「只有終點才有分數」的稀疏獎勵問題。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import config
from agents.astar_bot import astar_path
from envs.maze_env import MazeEnv

# 動作編號 -> (dx, dy)，dx 是列 (往下為正)。順序同 main.py 的上/下/左/右
ACTION_TO_MOVE = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def build_player_obs(env: MazeEnv):
    """從 MazeEnv 當前狀態組出 NN 玩家的觀測 (訓練與 main.py 演示共用)

    註：曾實驗過第 5 通道「出口 BFS 距離場」當導航提示 (模型存檔
    models/player_ppo_5ch_distfield.zip)，同訓練量下通關率持平但平均步數
    變慢 (39 vs 32)，故維持 4 通道。
    """
    grid = np.zeros((4, env.grid_size, env.grid_size), dtype=np.float32)
    grid[0] = (env.maze == config.ID_WALL).astype(np.float32)
    grid[1, env.player_pos[0], env.player_pos[1]] = 1.0
    grid[2, env.exit_pos[0], env.exit_pos[1]] = 1.0
    for m in env.monsters:
        grid[3, m[0], m[1]] = 1.0

    state = np.array(
        [
            env.player_hp / config.PLAYER_MAX_HP,
            env.player_hammers / max(config.PLAYER_INITIAL_HAMMERS, 1),
            np.clip(1.0 - env.current_time / config.TIME_MAX, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return {"grid": grid, "state": state}


class PlayerMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": config.FPS}

    def __init__(self, master_model_path=None, render_mode=None, randomize_hammers=False):
        super().__init__()

        # 訓練時隨機化開局鎚子數 (0~2)：固定 2 支會讓「無鎚導航」練習量不足，
        # 玩家學成「只會鑿牆直線衝」，沒鎚子撞上牆就卡死 (實測卡死回合鎚子全是 0)
        self.randomize_hammers = randomize_hammers

        # 內部 MazeEnv 走 manual_move 路徑 (與人類同規則：撞牆自動耗鎚)
        config.PLAYER_MODE = "NN"
        self.game = MazeEnv(render_mode=render_mode)

        # 凍結的 Maze Master (只推論不更新)；延遲 import 避免循環相依
        self.master = None
        if master_model_path is not None:
            from stable_baselines3 import PPO

            self.master = PPO.load(master_model_path, device="cpu")
        self._skip_action = np.zeros(3 * config.ACTIONS_PER_TURN, dtype=np.int64)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    0.0, 1.0, shape=(4, self.game.grid_size, self.game.grid_size), dtype=np.float32
                ),
                "state": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            }
        )

        self.episode_steps = 0

    def _exit_distance(self):
        """玩家到出口的 A* 最短步數；被堵死時回傳 None (塑形跳過該步)"""
        path = astar_path(self.game.maze, self.game.player_pos, self.game.exit_pos)
        return len(path) - 1 if path else None

    def _master_action(self):
        if self.master is None:
            return self._skip_action
        # 凍結的 Master 看它原本的 60x60 觀測；非確定性讓干擾更多樣
        action, _ = self.master.predict(self.game._get_obs(), deterministic=False)
        return action

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        if self.randomize_hammers:
            self.game.player_hammers = int(self.game.rng.integers(0, config.PLAYER_INITIAL_HAMMERS + 1))
        self.episode_steps = 0
        return build_player_obs(self.game), {}

    def step(self, action):
        dx, dy = ACTION_TO_MOVE[int(action)]

        # 記錄步前狀態，供獎勵計算
        hp_before = self.game.player_hp
        dist_before = self._exit_distance()
        tx = self.game.player_pos[0] + dx
        ty = self.game.player_pos[1] + dy
        out_of_bounds = not (0 <= tx < self.game.grid_size and 0 <= ty < self.game.grid_size)
        bumped = out_of_bounds or (
            self.game.maze[tx, ty] == config.ID_WALL and self.game.player_hammers == 0
        )

        # 玩家出招 + Maze Master 編輯 + 怪物移動，全部在內部 step 完成
        self.game.set_player_move(dx, dy)
        _, _, terminated, _, info = self.game.step(self._master_action())

        # --- 玩家獎勵 (與 Master 的獎勵無關) ---
        reward = config.PLAYER_REWARD_STEP
        if bumped:
            reward += config.PLAYER_REWARD_BUMP

        hp_lost = hp_before - self.game.player_hp
        if hp_lost > 0:
            reward += config.PLAYER_REWARD_HIT * hp_lost

        result = info.get("result")
        if result in ("flow_success", "too_fast", "too_slow"):
            reward += config.PLAYER_REWARD_GOAL
            # 速度紅利：越早通關拿越多 (30 步通關約 +12，100 步約 +5)
            time_left = config.TIME_MAX * 1.5 - self.game.current_time
            reward += config.PLAYER_REWARD_SPEED_BONUS * max(0.0, time_left)
        elif result == "died":
            reward += config.PLAYER_REWARD_DEATH
        elif result == "timeout":
            reward += config.PLAYER_REWARD_TIMEOUT
        elif not terminated and dist_before is not None:
            # 距離塑形：只在回合未結束時計，離出口變近給正分、變遠給負分
            dist_after = self._exit_distance()
            if dist_after is not None:
                delta = dist_before - dist_after
                reward += float(
                    np.clip(config.PLAYER_REWARD_DIST * delta, -0.5, 0.5)
                )

        # 撞牆不會推進 current_time，內部超時判定可能永遠不觸發，這裡自己截斷
        self.episode_steps += 1
        truncated = self.episode_steps >= config.PLAYER_MAX_EPISODE_STEPS

        return build_player_obs(self.game), reward, terminated, truncated, info

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
