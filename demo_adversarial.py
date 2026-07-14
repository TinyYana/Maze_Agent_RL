"""對抗模型即時對局 Demo：玩家 AI vs Maze Master，遊戲中可切換雙方版本。

兩個網路在同一個 Pygame 視窗裡真實對局 (非錄影回放)，每步各做一次前向推論。
時序與訓練時 (envs/adversarial.py) 完全一致：
    Master 編輯 → 玩家在編輯後局面行動 → 怪物移動 → 碰撞/判定

操作：
    1 2 3 4    切換玩家 AI 版本 (r1~r4)
    Q W E R    切換 Maze Master 版本 (r1~r4)
    N          換一座新迷宮
    S          同一座迷宮重開
    Space      暫停 / 繼續
    →          暫停時單步執行
    Esc        離開

切換版本會用同一座迷宮重開，方便直接對比不同世代的打法差異。
右上角 HUD 顯示當前版本組合與該組合的累積戰績。
"""
import os
import random
import sys
import time

# 推論執行緒上限 = 全機邏輯處理器的一半，另一半永遠留給系統散熱。
# 放任 torch 用滿所有核心會讓 CPU 過熱，也會拖垮 pygame 事件迴圈 (按鍵沒反應)。
# 必須在 import torch 之前設 OMP_NUM_THREADS 才生效。
MAX_THREADS = int(os.getenv("DEMO_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
os.environ.setdefault("OMP_NUM_THREADS", str(MAX_THREADS))

import pygame
import torch

torch.set_num_threads(MAX_THREADS)

import config

config.PLAYER_MODE = "NN"

from sb3_contrib import MaskablePPO

from envs.maze_env import MazeEnv
from envs.adversarial import (
    ACTION_TO_MOVE,
    build_obs,
    decode_master_action,
    master_action_masks,
    player_action_masks,
)
from envs.rendering import load_cjk_font

MODELS_DIR = os.getenv("ADV_MODELS", "models_adv_final/maze_rl")
STEP_FPS = float(os.getenv("DEMO_STEP_FPS", "3"))  # 每秒推進幾步 (推論吃 CPU，別調太高)
RENDER_FPS = int(os.getenv("DEMO_RENDER_FPS", "30"))  # 畫面更新率 (降低以省 CPU)
DEBUG_KEYS = os.getenv("DEMO_DEBUG_KEYS", "1") == "1"  # 把收到的按鍵印到 console
END_PAUSE = 2.0  # 回合結束後停留幾秒再自動重開

PLAYER_KEYS = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4}
MASTER_KEYS = {pygame.K_q: 1, pygame.K_w: 2, pygame.K_e: 3, pygame.K_r: 4}

RESULT_BANNER = {
    "flow_success": ("玩家通關！落在心流區間", (39, 174, 96)),
    "too_fast": ("玩家通關 — 比 Master 預期更快", (39, 174, 96)),
    "too_slow": ("玩家通關 — 但被拖慢了", (211, 158, 15)),
    "died": ("玩家被怪物擊倒 — Master 得手", (192, 57, 43)),
    "blocked": ("路徑被堵死", (192, 57, 43)),
    "timeout": ("超時 — Master 成功拖垮玩家", (127, 140, 141)),
}
RESULT_BUCKET = {
    "flow_success": "win", "too_fast": "win", "too_slow": "win",
    "died": "died", "blocked": "died",
    "timeout": "timeout", "truncated": "timeout",
}


class ModelCache:
    """惰性載入 + 快取，切換版本時不必重複讀檔"""

    def __init__(self):
        self._cache = {}

    def get(self, side, rnd):
        key = (side, rnd)
        if key not in self._cache:
            path = os.path.join(MODELS_DIR, f"{side}_r{rnd}")
            print(f"載入 {side}_r{rnd} ...")
            self._cache[key] = MaskablePPO.load(path, device="cpu")
        return self._cache[key]


def step_once(game, player, master):
    """推進一個完整回合，回傳 (是否結束, info)"""
    # 1. Master 編輯 (遮罩擋掉所有非法格)
    m_act, _ = master.predict(
        build_obs(game), action_masks=master_action_masks(game), deterministic=False
    )
    game.last_ai_action = ""
    game._execute_maze_master_actions(decode_master_action(m_act))

    # 2. 玩家在編輯後的局面上行動
    p_act, _ = player.predict(
        build_obs(game), action_masks=player_action_masks(game), deterministic=False
    )
    game.set_player_move(*ACTION_TO_MOVE[int(p_act)])
    _, player_done, player_info = game._handle_player_turn()

    # 3. 怪物移動 + 碰撞 + 勝負判定
    game._move_monsters()
    game._handle_collisions()
    _, status_done, status_info = game._check_game_status()

    info = {}
    if player_info:
        info.update(player_info)
    if status_info:
        info.update(status_info)
    return (player_done or status_done), info


def draw_hud(canvas, top_height, p_ver, m_ver, seed, stats, font, font_tiny):
    """版本/戰績面板。由 renderer 的 overlay 掛鉤呼叫，與畫面同一幀送出 (不會閃)"""
    w, h = 214, 152
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((17, 21, 32, 228))
    pygame.draw.rect(panel, config.COLOR_ACCENT, panel.get_rect(), 1)

    rows = [
        (f"玩家 AI          r{p_ver}", config.COLOR_PLAYER_BODY),
        (f"Maze Master   r{m_ver}", config.COLOR_MONSTER_BODY),
        (f"迷宮 #{seed}", (150, 160, 178)),
    ]
    y = 8
    for text, color in rows:
        panel.blit(font.render(text, True, color), (10, y))
        y += 21

    s = stats.get((p_ver, m_ver), {})
    rec = f"通關 {s.get('win', 0)}  死亡 {s.get('died', 0)}  超時 {s.get('timeout', 0)}"
    panel.blit(font.render(rec, True, config.COLOR_TEXT), (10, y + 2))

    y += 28
    pygame.draw.line(panel, (58, 66, 84), (10, y), (w - 10, y), 1)
    y += 6
    for tip in ("1~4  切玩家版本    QWER  切 Master", "N 新迷宮   S 重開   空白 暫停"):
        panel.blit(font_tiny.render(tip, True, (120, 132, 152)), (10, y))
        y += 17

    canvas.blit(panel, (config.WINDOW_SIZE - w - 8, top_height + 8))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.path.isdir(MODELS_DIR):
        print(f"找不到模型目錄 {MODELS_DIR}，請確認對抗模型已下載。")
        return

    cache = ModelCache()
    p_ver, m_ver = 4, 4  # 預設看最終版對決
    player = cache.get("player", p_ver)
    master = cache.get("master", m_ver)

    game = MazeEnv(render_mode="human")
    game.renderer.fps = RENDER_FPS  # 省 CPU：不需要 60fps 來看一秒三步的棋
    seed = 91001
    game.reset(seed=seed)
    font = load_cjk_font(15)
    font_tiny = load_cjk_font(12)

    stats = {}  # (p_ver, m_ver) -> {"win":n, "died":n, "timeout":n}
    paused = False
    ended_at = None  # 回合結束的時間戳
    next_step = time.monotonic()

    print(__doc__)
    print(f"推論執行緒：{MAX_THREADS} / {os.cpu_count()} (一半核心留給系統散熱)")
    print(f"推進速度：{STEP_FPS} 步/秒，畫面 {RENDER_FPS} fps")
    print(f"當前組合：玩家 r{p_ver} vs Master r{m_ver}，迷宮 #{seed}")

    def restart(new_seed=None):
        nonlocal seed, ended_at, next_step
        if new_seed is not None:
            seed = new_seed
        game.reset(seed=seed)
        game.renderer.banner = None  # 清掉上一局的結果橫幅
        ended_at = None
        next_step = time.monotonic()

    running = True
    while running:
        single_step = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if DEBUG_KEYS:
                    print(f"[鍵盤] {pygame.key.name(event.key)} (焦點={pygame.key.get_focused()})", flush=True)
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in PLAYER_KEYS:
                    p_ver = PLAYER_KEYS[event.key]
                    player = cache.get("player", p_ver)
                    print(f"→ 玩家 AI 切換為 r{p_ver}（同一迷宮重開）")
                    restart()
                elif event.key in MASTER_KEYS:
                    m_ver = MASTER_KEYS[event.key]
                    master = cache.get("master", m_ver)
                    print(f"→ Maze Master 切換為 r{m_ver}（同一迷宮重開）")
                    restart()
                elif event.key == pygame.K_n:
                    restart(random.randint(1, 999999))
                    print(f"→ 新迷宮 #{seed}")
                elif event.key == pygame.K_s:
                    restart()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    single_step = True

        # 回合結束：停留幾秒讓人看清結果，再用同一迷宮自動重開
        if ended_at is not None:
            if time.monotonic() - ended_at >= END_PAUSE:
                restart()
        elif single_step or (not paused and time.monotonic() >= next_step):
            next_step = time.monotonic() + 1.0 / STEP_FPS
            done, info = step_once(game, player, master)
            if done:
                result = info.get("result", "truncated")
                text, color = RESULT_BANNER.get(result, ("回合結束", (127, 140, 141)))
                game.renderer.set_banner(text, color)
                bucket = RESULT_BUCKET.get(result, "timeout")
                s = stats.setdefault((p_ver, m_ver), {"win": 0, "died": 0, "timeout": 0})
                s[bucket] += 1
                print(f"[r{p_ver} vs r{m_ver}] {text}（{game.current_time} 步）")
                ended_at = time.monotonic()

        # HUD 透過 overlay 掛鉤併入同一幀 (每幀重綁以取得最新版本/戰績)
        game.renderer.overlay = lambda canvas: draw_hud(
            canvas, game.renderer.top_height, p_ver, m_ver, seed, stats, font, font_tiny
        )
        game.render()

    game.close()


if __name__ == "__main__":
    main()
