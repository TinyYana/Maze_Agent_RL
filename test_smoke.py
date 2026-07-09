"""最小煙霧測試：python test_smoke.py

驗證地形/實體分離重構的不變式：
- 地形層只含空地與牆（實體不會污染、也不會被擦掉）
- 出口與玩家永遠存在於疊合後的觀測格
- 觀測形狀與 dtype 不變（與已訓練的 PPO 模型相容）

以及操作手感的兩個前提：
- HUMAN 模式下按鍵當下那一個 step 就會移動玩家
- 渲染迴圈跑在 config.FPS，不會被額外的 sleep 拖慢事件輪詢

還有字型載入：本機找得到中文字型檔，且 SysFont 爆掉時不會連帶讓遊戲開不起來。
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import config
import envs.rendering as rendering
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

# --- 手感：按鍵當幀就移動 ---
config.PLAYER_MODE = "HUMAN"
env = MazeEnv(render_mode="human")
env.reset(seed=7)
skip_action = np.zeros(3 * config.ACTIONS_PER_TURN, dtype=np.int64)  # Maze Master 按兵不動

start = env.player_pos.copy()
env.set_player_move(1, 0)
env.step(skip_action)
assert not np.array_equal(env.player_pos, start), "按下方向鍵後，同一個 step 內玩家沒有移動"

# --- 手感：渲染迴圈維持在 config.FPS (舊版有 sleep(0.05) 會拖到約 8fps) ---
frames = 30
env.render()  # 先暖機一幀，避免把視窗初始化時間算進去
start_t = time.perf_counter()
for _ in range(frames):
    env.render()
elapsed = time.perf_counter() - start_t
expected = frames / config.FPS
assert elapsed < expected * 2.0, f"渲染迴圈太慢：{elapsed:.2f}s，預期約 {expected:.2f}s"

# --- 字型：本機找得到中文字型檔 ---
candidates = rendering.FONT_FILES.get(sys.platform, rendering.LINUX_FONT_FILES)
found = [p for p in candidates if os.path.exists(p)]
assert found, f"{sys.platform} 上找不到任何中文字型檔：{candidates}"


# pygame 2.6.1 的 initsysfonts_win32() 會在某些 Windows 上拋 TypeError，
# 讓 SysFont 整個不能用 (打包成 exe 後就是一開就閃退)。模擬這個情況。
def _boom(*args, **kwargs):
    raise TypeError("模擬 initsysfonts_win32 掃字型登錄檔爆炸")


original_sysfont = pygame.font.SysFont
pygame.font.SysFont = _boom
try:
    # 有字型檔可用時，根本不該走到 SysFont
    assert rendering.load_cjk_font(16) is not None, "有中文字型檔時仍去呼叫了 SysFont"
    # 連字型檔都找不到時，最差也只能退回內建字型，不能讓遊戲開不起來
    rendering.FONT_FILES[sys.platform] = ["/does/not/exist.ttc"]
    assert rendering.load_cjk_font(16) is not None, "SysFont 失敗時沒有退回內建字型"
finally:
    pygame.font.SysFont = original_sysfont
    rendering.FONT_FILES[sys.platform] = candidates

env.close()
print(f"OK ({episodes} 回合，{frames} 幀耗時 {elapsed:.2f}s / 預期 {expected:.2f}s，字型 {found[0]})")
