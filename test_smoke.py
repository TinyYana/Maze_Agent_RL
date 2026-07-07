"""最小煙霧測試：python test_smoke.py

驗證地形/實體分離重構的不變式：
- 地形層只含空地與牆（實體不會污染、也不會被擦掉）
- 出口與玩家永遠存在於疊合後的觀測格
- 觀測形狀與 dtype 不變（與已訓練的 PPO 模型相容）
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import config
from envs.maze_env import MazeEnv

config.PLAYER_MODE = "AI"
env = MazeEnv(render_mode=None)
obs, _ = env.reset(seed=42)
assert obs.shape == (1, 60, 60) and obs.dtype == np.uint8

episodes = 0
for _ in range(2000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert set(np.unique(env.maze)) <= {config.ID_EMPTY, config.ID_WALL}, "地形層被實體污染"
    grid = env._entity_grid()
    assert grid[env.exit_pos[0], env.exit_pos[1]] in (config.ID_EXIT, config.ID_PLAYER), "出口消失"
    assert grid[env.player_pos[0], env.player_pos[1]] == config.ID_PLAYER, "玩家消失"
    assert len(env.monsters) <= config.MAX_MONSTERS
    for m in env.monsters:
        assert env.maze[m[0], m[1]] != config.ID_WALL, "怪物卡在牆裡"

    if terminated:
        episodes += 1
        obs, _ = env.reset()

env.close()
assert episodes > 0
print(f"OK ({episodes} 回合)")
