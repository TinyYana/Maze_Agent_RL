"""對抗訓練檢查點的系統性體檢：python evaluate_adversarial.py [輪數]

單行自評 (player_rN vs master_rN) 看不到的問題，這裡全都抓：
  1. 交叉矩陣：每代玩家 vs 每代 Master (+無 Master 基準) —— 抓假進步/循環退化
  2. 無 Master 靜態迷宮 —— 抓災難性遺忘 (r4 玩家實測 0/20 全超時)
  3. 掛機探測 bot —— 已知漏洞的回歸測試 (來回走 + 出口靠近就撲)
headline 指標是 flow rate (心流通關佔比)，不是通關率。

環境變數：ADV_MODELS (預設 models_adv_final/maze_rl)、EVAL_EPISODES (預設 20)
"""
import os
import sys
from collections import Counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from agents.astar_bot import astar_path
from envs.adversarial import ACTION_TO_MOVE, PlayerEnvV2, player_action_masks

MODELS_DIR = os.getenv("ADV_MODELS", os.path.join("models_adv_final", "maze_rl"))
N_EP = int(os.getenv("EVAL_EPISODES", "20"))
GRAB_DIST = 3  # 掛機 bot 的撲擊距離
REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}
WIN_RESULTS = ("flow_success", "too_fast", "too_slow")


def ckpt(name):
    path = os.path.join(MODELS_DIR, name)
    return path if os.path.exists(path + ".zip") else None


def run_episodes(env, act_fn):
    """act_fn(env, 上一步動作) -> 動作。回傳 (結局計數, 贏時步數列表)"""
    results, win_times = Counter(), []
    for ep in range(N_EP):
        env.reset(seed=90000 + ep)
        done, last_a, info = False, None, {}
        while not done:
            last_a = act_fn(env, last_a)
            _, _, term, trunc, info = env.step(last_a)
            done = term or trunc
        res = info.get("result", "truncated")
        results[res] += 1
        if res in WIN_RESULTS:
            win_times.append(env.game.current_time)
    return results, win_times


def nn_player(model):
    def act(env, _last):
        a, _ = model.predict(build_obs_cached(env), action_masks=env.action_masks(), deterministic=False)
        return int(a)

    return act


def build_obs_cached(env):
    from envs.adversarial import build_obs

    return build_obs(env.game)


def camper(env, last_a):
    """來回走；出口進撲擊範圍就直奔 —— 玩家實際抓到的漏洞"""
    game = env.game
    path = astar_path(game.maze, game.player_pos, game.exit_pos)
    if path and 1 < len(path) <= GRAB_DIST + 1:
        step = path[1]
        for a, (dx, dy) in ACTION_TO_MOVE.items():
            if (game.player_pos[0] + dx, game.player_pos[1] + dy) == tuple(step):
                return a
    mask = player_action_masks(game)
    prefer = REVERSE.get(last_a)
    if prefer is not None and mask[prefer]:
        return prefer
    return int(np.flatnonzero(mask)[0])


def fmt(results, win_times):
    wins = sum(results[k] for k in WIN_RESULTS)
    flow = results["flow_success"]
    avg_t = f"{np.mean(win_times):5.1f}" if win_times else "    —"
    return (f"flow {flow / N_EP:4.0%} 通關 {wins:2d}/{N_EP} 贏時均步 {avg_t} | "
            f"timeout {results['timeout'] + results['truncated']:2d} 死亡 {results['died']:2d}")


def main():
    from sb3_contrib import MaskablePPO

    max_round = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    players = [(f"player_r{r}", ckpt(f"player_r{r}")) for r in range(max_round + 1)]
    masters = [("no_master", None)] + [(f"master_r{r}", ckpt(f"master_r{r}")) for r in range(max_round + 1)]

    print(f"=== 交叉矩陣 + 靜態基準 ({N_EP} 回合/格，種子 90000+) ===")
    for p_name, p_path in players:
        if p_path is None:
            continue
        model = MaskablePPO.load(p_path, device="cpu")
        for m_name, m_path in masters:
            if m_name != "no_master" and m_path is None:
                continue
            env = PlayerEnvV2(master_model_path=m_path)
            results, win_times = run_episodes(env, nn_player(model))
            env.close()
            print(f"{p_name} vs {m_name:10s} | {fmt(results, win_times)}", flush=True)

    print(f"=== 掛機探測 (通關數 > 0 = 漏洞存在) ===")
    for m_name, m_path in masters[1:]:
        if m_path is None:
            continue
        env = PlayerEnvV2(master_model_path=m_path)
        results, win_times = run_episodes(env, camper)
        env.close()
        print(f"camper vs {m_name:10s} | {fmt(results, win_times)}", flush=True)


if __name__ == "__main__":
    main()
