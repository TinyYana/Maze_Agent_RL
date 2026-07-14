"""NN 玩家訓練腳本 (PPO + 自訂小 CNN)，兩階段課程：

第一階段：Maze Master 不出手，在 DFS 隨機靜態迷宮學「走迷宮」基本功。
第二階段：載入凍結的 maze_master_ppo.zip 當干擾者，學「應付動態迷宮 + 躲怪」。

第二階段用「混合環境」：一半進程有 Master、一半仍是靜態迷宮。
全部環境都放 Master 會發生災難性遺忘——梯度被「怎麼不被怪打死」主導，
把第一階段學會的導航能力洗掉 (實測靜態通關率 66% → 0%)。

用法:
    python train_player.py
環境變數:
    PLAYER_TIMESTEPS_P1  第一階段步數 (預設 300000；0 = 跳過)
    PLAYER_TIMESTEPS_P2  第二階段步數 (預設 300000；找不到 Master 模型時自動跳過)
    PLAYER_RESUME        從既有模型續訓 (路徑，不含 .zip)，搭配 P1=0 可只跑第二階段
    PLAYER_LR            覆蓋學習率 (續訓微調時建議降到 3e-5 防災難性遺忘)
"""
import os
import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import config
from agents.player_policy import PlayerFeaturesExtractor
from envs.player_env import PlayerMazeEnv

MASTER_MODEL_PATH = "maze_master_ppo"
PLAYER_MODEL_PATH = "player_ppo"


def make_env(rank, master_model_path=None):
    """建立單一玩家環境 (供 SubprocVecEnv 子進程呼叫)"""

    def _init():
        # 子進程有自己的 config 模組，需在此設定
        config.PLAYER_MODE = "NN"
        env = PlayerMazeEnv(
            master_model_path=master_model_path, render_mode=None, randomize_hammers=True
        )
        # 每個環境用不同種子，避免所有進程產生一模一樣的迷宮序列
        env.game.rng = np.random.default_rng(1919810 + rank)
        return Monitor(env)

    return _init


def train():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"player_run_{timestamp}"

    log_dir = "./tensorboard_logs/"
    models_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    n_envs = min(8, os.cpu_count() or 4)
    timesteps_p1 = int(os.getenv("PLAYER_TIMESTEPS_P1", "300000"))
    timesteps_p2 = int(os.getenv("PLAYER_TIMESTEPS_P2", "300000"))
    has_master = os.path.exists(MASTER_MODEL_PATH + ".zip")

    import torch

    print(f"=== NN 玩家訓練: {run_name} ===")
    print(f"並行環境數: {n_envs} (SubprocVecEnv)")
    print(f"第一階段 (靜態迷宮): {timesteps_p1} 步")
    if has_master:
        print(f"第二階段 (凍結 Maze Master 干擾): {timesteps_p2} 步")
    else:
        print(f"⚠️ 找不到 {MASTER_MODEL_PATH}.zip，將只跑第一階段。")

    # === 第一階段：靜態迷宮 ===
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    policy_kwargs = dict(
        features_extractor_class=PlayerFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[],  # 策略/價值頭直接接在 256 維特徵上
    )

    resume_path = os.getenv("PLAYER_RESUME")
    lr_override = os.getenv("PLAYER_LR")
    if resume_path:
        custom = {"learning_rate": float(lr_override)} if lr_override else {}
        model = PPO.load(resume_path, env=env, tensorboard_log=log_dir, custom_objects=custom)
        print(f"從既有模型續訓: {resume_path} (lr={model.learning_rate})")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            batch_size=256,
            # 只有 4 個動作，不需要 Maze Master 那麼強的探索壓力 (它是 0.1)
            ent_coef=0.01,
            gamma=0.99,
            n_steps=max(4096 // n_envs, 64),
            clip_range=0.2,
            gae_lambda=0.95,
            device="auto",
            tensorboard_log=log_dir,
        )
    print(f"運算裝置: {model.device} (torch {torch.__version__}, CUDA 可用: {torch.cuda.is_available()})")
    print(f"網路參數量: {sum(p.numel() for p in model.policy.parameters()):,}")

    if timesteps_p1 > 0:
        model.learn(total_timesteps=timesteps_p1, tb_log_name=f"player_p1_{run_name}")
        model.save(os.path.join(models_dir, f"{PLAYER_MODEL_PATH}_p1_{run_name}"))
    env.close()

    # === 第二階段：混合環境 (偶數進程有凍結 Master，奇數進程維持靜態) ===
    if has_master and timesteps_p2 > 0:
        env = SubprocVecEnv(
            [
                make_env(i, master_model_path=MASTER_MODEL_PATH if i % 2 == 0 else None)
                for i in range(n_envs)
            ]
        )
        model.set_env(env)
        # reset_num_timesteps=False 讓 TensorBoard 曲線接在第一階段後面
        model.learn(
            total_timesteps=timesteps_p2,
            tb_log_name=f"player_p2_{run_name}",
            reset_num_timesteps=False,
        )
        env.close()

    # 存成預設名稱 (main.py 讀取) + 歷史備份
    model.save(PLAYER_MODEL_PATH)
    backup_path = os.path.join(models_dir, f"{PLAYER_MODEL_PATH}_{run_name}")
    model.save(backup_path)

    print("訓練完成！")
    print(f"主要模型已更新: {PLAYER_MODEL_PATH}.zip")
    print(f"歷史備份已存檔: {backup_path}.zip")


if __name__ == "__main__":
    train()
