import numpy as np
import pygame
import time
from stable_baselines3 import PPO  # 引入 PPO
from envs.maze_env import MazeMasterEnv


if __name__ == "__main__":
    # 建立環境，開啟人眼渲染模式
    env = MazeMasterEnv(render_mode="human")
    obs, info = env.reset()

    print("載入模型中...")
    try:
        # 嘗試載入訓練好的模型
        model = PPO.load("maze_master_ppo")
        print("模型載入成功！")
        use_model = True
    except FileNotFoundError:
        print("找不到模型檔案 'maze_master_ppo.zip'，將使用隨機動作。")
        print("請先執行 'python train.py' 進行訓練。")
        use_model = False

    print("遊戲開始！")

    running = True
    while running:
        if use_model:
            # 使用模型預測動作
            # deterministic=False 讓 AI 保持一點隨機性 (探索)，True 則完全依照最高機率
            action, _states = model.predict(obs, deterministic=False)
        else:
            # 隨機動作 (Fallback)
            action = np.random.choice([0, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.1])

        # 執行一步
        obs, reward, terminated, truncated, info = env.step(action)

        # 稍微延遲一下，不然人類眼睛跟不上 AI 的速度
        # 訓練時不需要這個，但展示時需要
        # time.sleep(0.1)

        if terminated:
            print("回合結束 (玩家到達出口或路徑被堵死)。重置環境。")
            obs, info = env.reset()

        # 處理 Pygame 關閉視窗事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()
