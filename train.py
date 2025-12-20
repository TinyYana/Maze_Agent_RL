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
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,  # 稍微提高學習率
        batch_size=256,  # 加大 batch
        ent_coef=0.1,  # 提高探索係數，強迫嘗試放牆
        gamma=0.99,
        n_steps=4096,
        clip_range=0.2,
        gae_lambda=0.95,
        device="auto",
        tensorboard_log=log_dir,
    )

    print(f"開始訓練... (Entropy Coef: {model.ent_coef})")

    # 3. 開始訓練
    # 增加訓練步數
    model.learn(total_timesteps=500000, tb_log_name="maze_ppo_v2")

    # 4. 儲存模型
    model_path = "maze_master_ppo"
    model.save(model_path)
    print(f"模型已儲存至 {model_path}.zip")


if __name__ == "__main__":
    train()
