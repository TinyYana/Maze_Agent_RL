from stable_baselines3 import PPO
from envs.maze_env import MazeEnv


def train():
    # 1. 建立環境 (訓練時不需要 render_mode="human"，會拖慢速度)
    env = MazeEnv(render_mode='human')

    # 2. 定義模型
    # verbose=1: 顯示訓練進度
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="auto")

    print("開始訓練...")
    # 3. 開始訓練
    # total_timesteps: 訓練總步數，越多通常越聰明，但也越久
    # 建議先設 10000 跑跑看，正式訓練可能需要 100000+
    model.learn(total_timesteps=100000)

    # 4. 儲存模型
    model_path = "maze_master_ppo"
    model.save(model_path)
    print(f"模型已儲存至 {model_path}.zip")


if __name__ == "__main__":
    train()
