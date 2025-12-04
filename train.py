from stable_baselines3 import PPO
from envs.maze_env import MazeEnv
import os


def train():
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 1. 建立環境
    env = MazeEnv(render_mode=None)

    # 2. 定義模型 (加入修正參數)
    model = PPO(
        "CnnPolicy",  # 再次確認：如果你的輸入不是圖片，請改用 "MlpPolicy"
        env,
        verbose=1,  # 設為 1 比較乾淨，只顯示重要 Log
        learning_rate=0.0001,  # 稍微調低一點點
        batch_size=128,
        ent_coef=0.05,  # 強制探索，數值可嘗試 0.01 ~ 0.05
        gamma=0.99,  # 折扣因子，確保它重視長期獎勵 (預設 0.99)
        n_steps=4096,
        clip_range=0.1,  # 限制每次更新幅度，防止毀滅性更新 (原本 0.2)
        gae_lambda=0.95,  # 這是 PPO 的標準設定，有助於平滑優勢估計
        device="auto",
        tensorboard_log=log_dir,
    )

    print(f"開始訓練... (Entropy Coef: {model.ent_coef})")

    # 3. 開始訓練
    # 建議增加訓練步數，因為增加探索後收斂會變慢，但結果會更好
    model.learn(total_timesteps=200000, tb_log_name="maze_ppo_fixed")

    # 4. 儲存模型
    model_path = "maze_master_ppo"
    model.save(model_path)
    print(f"模型已儲存至 {model_path}.zip")


if __name__ == "__main__":
    train()
