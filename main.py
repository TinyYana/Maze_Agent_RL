import numpy as np
import pygame
import time
from stable_baselines3 import PPO  # 引入 PPO
from envs.maze_env import MazeEnv  # 修改：從 maze_env 匯入 MazeEnv


if __name__ == "__main__":
    # 建立環境，開啟人眼渲染模式
    env = MazeEnv(render_mode="human")  # 修改：使用 MazeEnv 類別
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
    print("--- 操作說明 ---")
    print("空白鍵 (Space): 暫停 / 繼續")
    print("右方向鍵 (Right): 暫停時單步執行")
    print("----------------")

    running = True
    paused = False  # 新增：暫停狀態標記

    while running:
        step_this_frame = False  # 新增：單步執行標記

        # 處理 Pygame 關閉視窗事件與按鍵輸入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"遊戲{'暫停' if paused else '繼續'}")
                elif event.key == pygame.K_RIGHT:
                    if paused:
                        step_this_frame = True
                        print("單步執行")

        # 只有在 (非暫停) 或 (暫停且觸發單步) 時才執行環境更新
        if not paused or step_this_frame:
            if use_model:
                # 使用模型預測動作
                # deterministic=False 讓 AI 保持一點隨機性 (探索)，True 則完全依照最高機率
                action, _states = model.predict(obs, deterministic=False)
            else:
                # 隨機動作 (Fallback)
                # 修改：MazeEnv 的 action_space 是 MultiDiscrete，直接使用 sample() 產生合法動作
                action = env.action_space.sample()

            # 執行一步
            obs, reward, terminated, truncated, info = env.step(action)

            # 稍微延遲一下，不然人類眼睛跟不上 AI 的速度
            # 訓練時不需要這個，但展示時需要
            # time.sleep(0.1)

            if terminated:
                reason = info.get("result", "未知")
                reason_map = {
                    "blocked": "路徑被堵死",
                    "died": "生命值耗盡",
                    "flow_success": "到達出口 (成功)",
                    "too_fast": "到達出口 (太快)",
                    "too_slow": "到達出口 (太慢)",
                    "timeout": "超時",
                }
                print(f"回合結束 -> {reason_map.get(reason, reason)}")
                obs, info = env.reset()
        else:
            # 暫停時持續渲染畫面，避免視窗卡死
            env.render()
            # 降低 CPU 使用率
            time.sleep(0.1)

    env.close()
