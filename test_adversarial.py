"""對抗環境的最小測試：python test_adversarial.py

驗證 BatchedFrozenOpponent 依賴的外部推論路徑與原本 env 內建路徑
「機制完全等價」：同種子 + 同凍結模型 (deterministic) + 同 learner 動作序列
=> 每一步的遊戲狀態必須一模一樣。

需要 models_adv_final/maze_rl/ 的 checkpoints (dev 分支自帶)。
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

import config
from envs.adversarial import (
    ACTION_TO_MOVE,
    MasterEnv,
    PlayerEnvV2,
    build_obs,
    master_action_masks,
    player_action_masks,
)

CKPT = os.path.join("models_adv_final", "maze_rl")
MASTER = os.path.join(CKPT, "master_r4")
PLAYER = os.path.join(CKPT, "player_r4")
N_STEPS = 120


def state_of(game):
    return (
        game.maze.tobytes(),
        tuple(game.player_pos),
        tuple(game.exit_pos),
        tuple(map(tuple, game.monsters)),
        game.player_hp,
        game.player_hammers,
        game.current_time,
        game.turn_count,
    )


def force_deterministic(model):
    orig = model.predict
    model.predict = lambda obs, action_masks=None, deterministic=False: orig(
        obs, action_masks=action_masks, deterministic=True
    )
    return model


# --- PlayerEnvV2：env 內建 Master vs 外部 [master_act, player_act] 打包 ---
env_a = PlayerEnvV2(master_model_path=MASTER)
force_deterministic(env_a.master)
env_b = PlayerEnvV2(master_model_path=None)
master = force_deterministic(type(env_a.master).load(MASTER, device="cpu"))

env_a.reset(seed=123)
env_b.reset(seed=123)
rng = np.random.default_rng(7)
for i in range(N_STEPS):
    legal = np.flatnonzero(player_action_masks(env_a.game))
    p_act = int(rng.choice(legal))

    m_obs, m_mask = env_b.master_io()
    m_act, _ = master.predict(m_obs, action_masks=m_mask)

    _, _, term_a, trunc_a, _ = env_a.step(p_act)
    _, _, term_b, trunc_b, _ = env_b.step(np.array([int(m_act), p_act], dtype=np.int64))

    assert state_of(env_a.game) == state_of(env_b.game), f"PlayerEnvV2 第 {i} 步狀態分歧"
    assert (term_a, trunc_a) == (term_b, trunc_b)
    if term_a or trunc_a:
        env_a.reset(seed=456 + i)
        env_b.reset(seed=456 + i)
env_a.close()
env_b.close()
print(f"OK PlayerEnvV2 外部路徑等價 ({N_STEPS} 步)")

# --- MasterEnv：env 內建 v2 對手 vs 外部 apply_edit 兩段式 ---
env_a = MasterEnv(opponent=("v2", PLAYER))
force_deterministic(env_a.opp_model)
env_b = MasterEnv(opponent=("external", None))
player = force_deterministic(type(env_a.opp_model).load(PLAYER, device="cpu"))

env_a.reset(seed=321)
env_b.reset(seed=321)
rng = np.random.default_rng(9)
for i in range(N_STEPS):
    legal = np.flatnonzero(master_action_masks(env_a.game))
    m_act = int(rng.choice(legal))

    p_obs, p_mask = env_b.apply_edit(m_act)
    p_act, _ = player.predict(p_obs, action_masks=p_mask)

    _, r_a, term_a, trunc_a, _ = env_a.step(m_act)
    _, r_b, term_b, trunc_b, _ = env_b.step(int(p_act))

    assert state_of(env_a.game) == state_of(env_b.game), f"MasterEnv 第 {i} 步狀態分歧"
    assert (r_a, term_a, trunc_a) == (r_b, term_b, trunc_b), f"MasterEnv 第 {i} 步獎勵/結束分歧"
    if term_a or trunc_a:
        env_a.reset(seed=654 + i)
        env_b.reset(seed=654 + i)
env_a.close()
env_b.close()
print(f"OK MasterEnv 外部路徑等價 ({N_STEPS} 步)")

# --- 反掛機遮罩：搬出口的合法格永遠離玩家夠遠 ---
env = MasterEnv(opponent=("external", None))
for seed in (1, 42, 777):
    env.reset(seed=seed)
    n = env.game.grid_size
    exit_mask = master_action_masks(env.game).reshape(5, n, n)[3]
    px, py = env.game.player_pos
    for x in range(n):
        for y in range(n):
            if exit_mask[x, y]:
                assert abs(x - px) + abs(y - py) >= config.EXIT_MIN_PLAYER_DIST, \
                    f"出口可以被搬到玩家附近 ({x},{y})，距離 {abs(x - px) + abs(y - py)}"
env.close()
print("OK 出口距離遮罩")
