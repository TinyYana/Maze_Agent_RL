from stable_baselines3 import PPO
from envs.maze_env import MazeEnv
import os


def train():
    # 定義 TensorBoard log 儲存目錄
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 1. 建立環境
    # 強烈建議：若要觀察 TensorBoard 數據，請將 render_mode 設為 None 以加速訓練
    env = MazeEnv(render_mode=None)
    # env = MazeEnv(render_mode='human')

    # 2. 定義模型
    # 新增 tensorboard_log 參數指定 Log 路徑
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        device="auto",
        tensorboard_log=log_dir,
    )

    print("開始訓練...")
    # 3. 開始訓練
    # 新增 tb_log_name 參數，這會建立 ./tensorboard_logs/maze_ppo_run_1/ 的資料夾
    model.learn(total_timesteps=100000, tb_log_name="maze_ppo_run")

    # 4. 儲存模型
    model_path = "maze_master_ppo"
    model.save(model_path)
    print(f"模型已儲存至 {model_path}.zip")


if __name__ == "__main__":
    train()
