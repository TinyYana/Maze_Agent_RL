import numpy as np
import pygame
import time
import sys  # 新增
import os  # 新增
from stable_baselines3 import PPO
from envs.maze_env import MazeEnv
import config


# --- 新增：資源路徑處理函式 ---
def resource_path(relative_path):
    """取得資源的絕對路徑，用於 PyInstaller 打包後的暫存路徑處理"""
    try:
        # PyInstaller 會建立一個暫存資料夾並將路徑存於 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ---------------------------

if __name__ == "__main__":
    # 建立環境，開啟人眼渲染模式
    env = MazeEnv(render_mode="human")  # 使用 MazeEnv 類別
    obs, info = env.reset()

    # 按住方向鍵可連續移動 (手感)
    pygame.key.set_repeat(180, 110)

    print("載入模型中...")
    try:
        # 修改：使用 resource_path 來取得模型路徑
        model_path = resource_path("maze_master_ppo")
        model = PPO.load(model_path)
        print(f"模型載入成功！(路徑: {model_path})")
        use_model = True
    except FileNotFoundError:
        print("找不到模型檔案 'maze_master_ppo.zip'，將使用隨機動作。")
        print("請先執行 'python train.py' 進行訓練。")
        use_model = False

    print("遊戲開始！")
    print("--- 操作說明 ---")
    print("空白鍵 (Space): 暫停 / 繼續")
    print("右方向鍵 (Right): 暫停時單步執行")
    print("方向鍵 (Up/Down/Left/Right): 移動角色 (HUMAN 模式)")
    print("H 鍵或畫面右下角 ? 按鈕: 開關遊戲說明")
    print("----------------")

    running = True
    paused = False

    while running:
        step_this_frame = False
        human_input_received = False  # 標記本幀是否有玩家輸入

        # 處理 Pygame 關閉視窗事件與按鍵輸入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 點擊 "?" 按鈕開關遊戲說明
                btn = env.renderer.help_button_rect
                if btn and btn.collidepoint(event.pos):
                    env.renderer.show_help = not env.renderer.show_help
            elif event.type == pygame.KEYDOWN:
                # H 或 ? 開關說明；說明開啟時 ESC 也可關閉，其餘按鍵忽略
                if event.key in (pygame.K_h, pygame.K_QUESTION, pygame.K_SLASH):
                    env.renderer.show_help = not env.renderer.show_help
                    continue
                if env.renderer.show_help:
                    if event.key == pygame.K_ESCAPE:
                        env.renderer.show_help = False
                    continue

                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"遊戲{'暫停' if paused else '繼續'}")
                elif event.key == pygame.K_RIGHT and paused:  # 只有暫停時右鍵才是單步
                    step_this_frame = True
                    print("單步執行")

                # --- 新增：手動操作監聽 ---
                # 只有在 HUMAN 模式且沒有暫停時才接收移動指令
                if config.PLAYER_MODE == "HUMAN" and not paused:
                    move_x, move_y = 0, 0
                    if event.key == pygame.K_UP:
                        move_x = -1
                    elif event.key == pygame.K_DOWN:
                        move_x = 1
                    elif event.key == pygame.K_LEFT:
                        move_y = -1
                    elif event.key == pygame.K_RIGHT:
                        move_y = 1

                    if move_x != 0 or move_y != 0:
                        env.set_player_move(move_x, move_y)
                        human_input_received = True
                # ------------------------

        # 決定是否執行環境更新 (說明面板開啟時遊戲暫停)
        should_step = False

        if env.renderer.show_help:
            pass
        elif config.PLAYER_MODE == "HUMAN":
            # 手動模式：只有在收到輸入 (或暫停時的單步除錯) 時才執行
            if human_input_received or (paused and step_this_frame):
                should_step = True
        else:
            # AI 模式：非暫停狀態，或觸發單步執行
            if not paused or step_this_frame:
                should_step = True

        # 只有在需要執行時才呼叫 env.step
        if should_step:
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
            # 暫停或等待輸入時持續渲染畫面，避免視窗卡死
            env.render()
            # 降低 CPU 使用率
            time.sleep(0.05)

    env.close()
