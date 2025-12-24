import numpy as np
import pandas as pd
import os
import time
import datetime
from stable_baselines3 import PPO
from envs.maze_env import MazeEnv
import config


def run_experiments(model_path="maze_master_ppo", n_episodes_per_scenario=50):
    """
    執行多組實驗並比較結果
    """
    # 1. 準備實驗資料夾
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 確保只取檔名部分，避免路徑問題
    model_name = os.path.basename(model_path)
    # 建立路徑: experiment_log/模型名稱_時間/
    experiment_dir = os.path.join("experiment_log", f"{model_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"--- 開始實驗 (每組測試 {n_episodes_per_scenario} 回合) ---")
    print(f"實驗結果將儲存於: {experiment_dir}")

    # 強制設定為 AI 玩家模式
    config.PLAYER_MODE = "AI"

    # 載入模型
    if os.path.exists(f"{model_path}.zip"):
        model = PPO.load(model_path)
        print("已載入訓練好的模型。")
    else:
        model = None
        print("警告：未找到模型，將使用隨機動作 (Random Agent)。")

    env = MazeEnv(render_mode=None)

    # 定義實驗場景 (Scenarios)
    # Key: 場景名稱
    # Value: 允許的動作 ID (0:Skip, 1:Wall, 2:Remove, 3:Exit, 4:Monster)
    scenarios = {
        "1. Static Maze (No Agent)": [0],  # 基準線：完全不干預
        "2. Walls Only": [0, 1, 2],  # 只能放牆/拆牆
        "3. Walls + Monsters": [0, 1, 2, 4],  # 加入怪物
        "4. Full Features": [0, 1, 2, 3, 4],  # 全部功能
    }

    all_results = []

    for scenario_name, allowed_actions in scenarios.items():
        print(f"\n正在執行場景: {scenario_name} ...")

        # 設定環境允許的動作
        env.set_allowed_actions(allowed_actions)

        for i in range(n_episodes_per_scenario):
            obs, info = env.reset()
            terminated = False
            truncated = False

            start_time = time.time()

            while not (terminated or truncated):
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)

            duration = time.time() - start_time

            # 記錄數據
            result_type = info.get("result", "unknown")
            is_success = (
                1 if result_type in ["flow_success", "too_fast", "too_slow"] else 0
            )
            is_flow = 1 if result_type == "flow_success" else 0

            record = {
                "Scenario": scenario_name,
                "Episode": i + 1,
                "Result": result_type,
                "Steps": env.player_steps,
                "Final_HP": env.player_hp,
                "Is_Success": is_success,  # 是否到達出口
                "Is_Flow": is_flow,  # 是否在心流區間到達
                "Real_Time_Sec": round(duration, 4),
            }
            all_results.append(record)

            if (i + 1) % 10 == 0:
                print(f"  進度: {i+1}/{n_episodes_per_scenario}", end="\r")

    env.close()

    # --- 數據分析與儲存 ---
    df = pd.DataFrame(all_results)

    # 儲存原始數據 (加入路徑)
    csv_filename = os.path.join(experiment_dir, "experiment_results_raw.csv")
    df.to_csv(csv_filename, index=False)
    print(f"\n\n原始數據已儲存至: {csv_filename}")

    # 計算摘要統計
    summary = df.groupby("Scenario").agg(
        {
            "Steps": ["mean", "std"],
            "Is_Success": "mean",  # 成功率
            "Is_Flow": "mean",  # 心流率
            "Final_HP": "mean",
        }
    )

    # 重新命名欄位以利閱讀
    summary.columns = ["Avg_Steps", "Std_Steps", "Success_Rate", "Flow_Rate", "Avg_HP"]
    summary["Success_Rate"] *= 100
    summary["Flow_Rate"] *= 100

    print("\n=== 實驗結果摘要 ===")
    print(summary)

    # 儲存摘要數據 (加入路徑)
    summary_csv = os.path.join(experiment_dir, "experiment_summary.csv")
    summary.to_csv(summary_csv)
    print(f"摘要數據已儲存至: {summary_csv}")


if __name__ == "__main__":
    # 確保你有安裝 pandas: pip install pandas
    run_experiments(n_episodes_per_scenario=100)
