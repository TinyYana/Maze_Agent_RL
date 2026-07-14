"""對抗式訓練調度：玩家 vs Maze Master 輪流訓練 (MaskablePPO + ConvGRU 規劃網路)。

流程：
  Phase 1  玩家預熱：靜態迷宮 (無 Master) 學導航
  Phase 2  Master 預熱：對凍結的速通玩家 (v1 4通道模型，若無則 A* bot) 學干擾
  Phase 3  輪流對抗 N 輪：每輪凍結一方訓另一方，對手從最近 POOL_SIZE 個
           歷史檢查點抽樣 (每個並行環境抽一個)，防止剪刀石頭布循環退化
  每輪結束跑固定種子評估，結果寫 CSV

環境變數 (遠端執行時由 shell 指定)：
  ADV_OUT      輸出目錄 (checkpoint/CSV，預設 ./models_adv)
  N_ENVS       並行環境數 (預設 min(12, cpu-2))
  DEVICE       cuda / cpu / auto (預設 auto)
  PREWARM_PLAYER_STEPS / PREWARM_MASTER_STEPS  (預設各 1000000)
  ROUND_STEPS  每輪每方步數 (預設 500000)
  N_ROUNDS     對抗輪數 (預設 4)
  SKIP_PREWARM 設 1 時跳過預熱 (需 ADV_OUT 裡已有 player_r0 / master_r0)
"""
import csv
import os
import random

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import config
from agents.planner_policy import MasterPolicy, PlayerFeaturesV2
from envs.adversarial import PlayerEnvV2, MasterEnv
from envs.batched_opponent import BatchedFrozenOpponent

if torch.cuda.is_available():
    # ConvGRU 全是卷積且輸入形狀固定：TF32 + cudnn autotune 是免費加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

OUT = os.getenv("ADV_OUT", "./models_adv")
N_ENVS = int(os.getenv("N_ENVS", str(max(min(12, (os.cpu_count() or 8) - 2), 2))))
DEVICE = os.getenv("DEVICE", "auto")
PREWARM_PLAYER_STEPS = int(os.getenv("PREWARM_PLAYER_STEPS", "1000000"))
PREWARM_MASTER_STEPS = int(os.getenv("PREWARM_MASTER_STEPS", "1000000"))
ROUND_STEPS = int(os.getenv("ROUND_STEPS", "500000"))
N_ROUNDS = int(os.getenv("N_ROUNDS", "4"))
POOL_SIZE = 5
# spawn 最穩：forkserver/fork 會與 torch 的執行緒狀態衝突死鎖 (新 pod 實測卡死)
START_METHOD = os.getenv("SUBPROC_START", "spawn")
LEGACY_PLAYER = "player_ppo"  # v1 速通模型，當 Master 預熱對手
# 凍結對手收到主進程批次推論 (GPU)，worker 只跑遊戲邏輯；BATCH_OPP=0 退回 env 內逐筆 CPU
BATCH_OPP = os.getenv("BATCH_OPP", "1") == "1"
# 玩家對抗輪次中保留的靜態迷宮 (無 Master) 環境比例，防災難性遺忘
STATIC_FRAC = float(os.getenv("STATIC_FRAC", "0.25"))


def ppo_kwargs():
    return dict(
        verbose=1,
        learning_rate=2.5e-4,
        batch_size=1024,
        n_steps=max(8192 // N_ENVS, 128),
        ent_coef=0.01,
        gamma=0.99,
        clip_range=0.2,
        gae_lambda=0.95,
        device=DEVICE,
    )


def _limit_worker_threads():
    """子進程的 torch 推論限制單執行緒。

    12 個 worker 各開滿執行緒會在 16 核上互相踩踏 (實測 fps 165 -> 4)，
    凍結對手的單筆推論本來就吃不滿多執行緒。
    """
    import torch

    torch.set_num_threads(1)


def make_player_env(rank, master_path):
    def _init():
        _limit_worker_threads()
        config.PLAYER_MODE = "NN"
        env = PlayerEnvV2(master_model_path=master_path, randomize_hammers=True)
        env.game.rng = np.random.default_rng(42000 + rank)
        return Monitor(env)

    return _init


def player_vecenv(master_pool):
    """玩家側向量環境。BATCH_OPP 時凍結 Master 由主進程批次推論 (GPU)。

    保留 STATIC_FRAC 比例的 env 不放 Master：對抗輪次全是 Master 環境時，
    導航基本功會被災難性遺忘 (交叉評估實測 player_r4 在無 Master 的靜態
    迷宮 0/20 全超時，見 docs/ADVERSARIAL_REVIEW.md)。
    """
    n_static = int(N_ENVS * STATIC_FRAC) if master_pool else N_ENVS
    paths = [None if i < n_static else random.Random(i).choice(master_pool) for i in range(N_ENVS)]
    worker_paths = [None] * N_ENVS if BATCH_OPP else paths
    env = SubprocVecEnv([make_player_env(i, worker_paths[i]) for i in range(N_ENVS)], start_method=START_METHOD)
    if BATCH_OPP and master_pool:
        env = BatchedFrozenOpponent(env, paths, "master", device=DEVICE)
    return env


def make_master_env(rank, opp):
    def _init():
        _limit_worker_threads()
        config.PLAYER_MODE = "NN"
        config.PLAYER_PROFILE_RANDOMIZE = True  # astar 對手時隨機化個性
        env = MasterEnv(opponent=opp)
        env.game.rng = np.random.default_rng(43000 + rank)
        return Monitor(env)

    return _init


def master_vecenv(opponent_pool):
    """Master 側向量環境。v2 對手且 BATCH_OPP 時同樣收到主進程批次推論"""
    opps = [random.Random(i).choice(opponent_pool) for i in range(N_ENVS)]
    batchable = BATCH_OPP and all(kind == "v2" for kind, _ in opps)
    worker_opps = [("external", None)] * N_ENVS if batchable else opps
    env = SubprocVecEnv([make_master_env(i, worker_opps[i]) for i in range(N_ENVS)], start_method=START_METHOD)
    if batchable:
        env = BatchedFrozenOpponent(env, [path for _, path in opps], "player", device=DEVICE)
    return env


def evaluate(player_path, master_path, n_episodes=50):
    """固定種子評估：回傳 (通關率, 贏時平均步數, 死亡率)"""
    player = MaskablePPO.load(player_path, device="cpu")
    env = PlayerEnvV2(master_model_path=master_path)
    wins, died, win_times = 0, 0, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=90000 + ep)
        done = False
        while not done:
            act, _ = player.predict(obs, action_masks=env.action_masks(), deterministic=False)
            obs, r, term, trunc, info = env.step(int(act))
            done = term or trunc
        res = info.get("result", "truncated")
        if res in ("flow_success", "too_fast", "too_slow"):
            wins += 1
            win_times.append(env.game.current_time)
        elif res == "died":
            died += 1
    env.close()
    avg_t = float(np.mean(win_times)) if win_times else -1.0
    return wins / n_episodes, avg_t, died / n_episodes


def train_side(model_path_or_none, env, policy, policy_kwargs, steps, run_name, resume_path=None):
    """建立或載入模型，訓練 steps 步後回傳模型"""
    if resume_path:
        model = MaskablePPO.load(resume_path, env=env, device=DEVICE)
    else:
        model = MaskablePPO(policy, env, policy_kwargs=policy_kwargs, **ppo_kwargs())
        n_params = sum(p.numel() for p in model.policy.parameters())
        print(f"[{run_name}] 新建網路，參數量 {n_params:,}，裝置 {model.device}")
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    return model


def pool_paths(prefix, upto_round):
    """最近 POOL_SIZE 個檢查點路徑 (r0..upto_round)"""
    paths = [os.path.join(OUT, f"{prefix}_r{r}") for r in range(max(0, upto_round - POOL_SIZE + 1), upto_round + 1)]
    return [p for p in paths if os.path.exists(p + ".zip")]


def main():
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "adversarial_curve.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "player_winrate", "avg_clear_time", "death_rate"])

    player_kwargs = dict(
        features_extractor_class=PlayerFeaturesV2,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[],
    )

    skip_prewarm = os.getenv("SKIP_PREWARM") == "1"
    p0 = os.path.join(OUT, "player_r0")
    m0 = os.path.join(OUT, "master_r0")

    # ---- Phase 1: 玩家預熱 (靜態迷宮) ----
    if not skip_prewarm:
        print(f"=== Phase 1: 玩家預熱 {PREWARM_PLAYER_STEPS} 步 ({N_ENVS} envs) ===")
        env = player_vecenv([])
        model = train_side(None, env, "MlpPolicy", player_kwargs, PREWARM_PLAYER_STEPS, "player_prewarm")
        model.save(p0)
        env.close()

        # ---- Phase 2: Master 預熱 (對凍結的速通玩家或 A*) ----
        opp = ("nn4", LEGACY_PLAYER) if os.path.exists(LEGACY_PLAYER + ".zip") else ("astar", None)
        print(f"=== Phase 2: Master 預熱 {PREWARM_MASTER_STEPS} 步，對手 {opp[0]} ===")
        env = master_vecenv([opp])
        model = train_side(None, env, MasterPolicy, {}, PREWARM_MASTER_STEPS, "master_prewarm")
        model.save(m0)
        env.close()

    # ---- Phase 3: 輪流對抗 (斷點續跑：已存檔的半輪跳過、已入 CSV 的評估不重跑) ----
    with open(csv_path) as f:
        done_evals = {row.split(",")[0] for row in f.read().splitlines()[1:] if row}

    for r in range(1, N_ROUNDS + 1):
        p_ckpt = os.path.join(OUT, f"player_r{r}")
        if os.path.exists(p_ckpt + ".zip"):
            print(f"=== Round {r}/{N_ROUNDS}: player_r{r} 已存在，跳過 ===")
        else:
            print(f"=== Round {r}/{N_ROUNDS}: 訓練玩家 (對手池 {len(pool_paths('master', r - 1))} 個 Master) ===")
            env = player_vecenv(pool_paths("master", r - 1))
            model = train_side(None, env, None, None, ROUND_STEPS, f"player_r{r}",
                               resume_path=os.path.join(OUT, f"player_r{r - 1}"))
            model.save(p_ckpt)
            env.close()

        m_ckpt = os.path.join(OUT, f"master_r{r}")
        if os.path.exists(m_ckpt + ".zip"):
            print(f"=== Round {r}/{N_ROUNDS}: master_r{r} 已存在，跳過 ===")
        else:
            print(f"=== Round {r}/{N_ROUNDS}: 訓練 Master (對手池 {len(pool_paths('player', r))} 個玩家) ===")
            opp_pool = [("v2", p) for p in pool_paths("player", r)]
            env = master_vecenv(opp_pool)
            model = train_side(None, env, None, None, ROUND_STEPS, f"master_r{r}",
                               resume_path=os.path.join(OUT, f"master_r{r - 1}"))
            model.save(m_ckpt)
            env.close()

        if str(r) in done_evals:
            continue
        winrate, avg_t, death = evaluate(p_ckpt, m_ckpt)
        print(f"[Round {r} 評估] 玩家通關率 {winrate:.0%}，贏時平均 {avg_t:.0f} 步，死亡率 {death:.0%}")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([r, winrate, avg_t, death])

    print("對抗訓練完成。輸出目錄:", OUT)


if __name__ == "__main__":
    main()
