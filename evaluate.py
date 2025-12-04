import numpy as np
import time
import os
from stable_baselines3 import PPO
from envs.maze_env import MazeEnv
import config
from collections import Counter


def evaluate_model(model_path="maze_master_ppo", n_episodes=100):
    """
    評估模型效能
    :param model_path: 模型路徑 (不含 .zip)
    :param n_episodes: 測試回合數
    """

    # 強制設定為 AI 模式，確保 A* Bot 自動運作
    original_mode = config.PLAYER_MODE
    config.PLAYER_MODE = "AI"

    print(f"--- 開始評估模型: {model_path} ---")
    print(f"測試回合數: {n_episodes}")
    print(f"目標步數範圍 (Flow): {config.TIME_MIN} ~ {config.TIME_MAX}")

    # 檢查模型是否存在
    if not os.path.exists(f"{model_path}.zip"):
        print(f"錯誤: 找不到模型檔案 {model_path}.zip")
        return

    # 建立無渲染環境以加速測試
    env = MazeEnv(render_mode=None)
    model = PPO.load(model_path)

    # 數據收集容器
    stats = {
        "rewards": [],
        "steps": [],
        "hps": [],
        "results": [],
    }

    start_time = time.time()

    for i in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            # 評估時通常使用 deterministic=True (確定性策略) 以獲得穩定結果
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        # 記錄本回合數據
        result_type = info.get("result", "unknown")

        # --- 新增除錯資訊 ---
        if result_type == "too_fast":
            # 隨機印出一些太快的案例，看看是多少步
            if np.random.rand() < 0.1:
                print(
                    f"\n[Debug] Too Fast Case: Steps={env.player_steps}, Threshold={config.TIME_MIN}"
                )
        # ------------------

        stats["results"].append(result_type)
        stats["steps"].append(env.player_steps)
        stats["hps"].append(env.player_hp)
        stats["rewards"].append(episode_reward)

        # 簡易進度條
        if (i + 1) % 10 == 0:
            print(f"進度: {i + 1}/{n_episodes}...", end="\r")

    total_time = time.time() - start_time
    print(f"\n評估完成！耗時: {total_time:.2f} 秒\n")

    # --- 計算統計指標 ---
    results_count = Counter(stats["results"])
    total_valid = len(stats["results"])

    # 1. 成功率指標
    flow_success = results_count.get("flow_success", 0)
    too_fast = results_count.get("too_fast", 0)
    too_slow = results_count.get("too_slow", 0)
    died = results_count.get("died", 0)
    timeout = results_count.get("timeout", 0)
    blocked = results_count.get("blocked", 0)

    flow_rate = (flow_success / total_valid) * 100
    survival_rate = ((total_valid - died - blocked) / total_valid) * 100

    # 2. 數值指標
    avg_steps = np.mean(stats["steps"])
    std_steps = np.std(stats["steps"])
    avg_reward = np.mean(stats["rewards"])
    avg_hp = np.mean(stats["hps"])

    # --- 輸出報表 ---
    print("=" * 40)
    print("       MODEL EVALUATION REPORT       ")
    print("=" * 40)
    print(f"平均獎勵 (Avg Reward): {avg_reward:.2f}")
    print(f"平均步數 (Avg Steps) : {avg_steps:.2f} (±{std_steps:.2f})")
    print(f"平均剩餘 HP          : {avg_hp:.2f} / {config.PLAYER_MAX_HP}")
    print("-" * 40)
    print(f"心流成功率 (Flow Rate) : {flow_rate:.1f}%  <-- 關鍵指標")
    print(f"存活率 (Survival Rate) : {survival_rate:.1f}%")
    print("-" * 40)
    print("結果分佈 (Result Distribution):")
    print(f"  [O] 完美區間 (Flow Success): {flow_success} ({flow_rate:.1f}%)")
    print(
        f"  [-] 太快通關 (Too Fast)    : {too_fast} ({(too_fast/total_valid)*100:.1f}%)"
    )
    print(
        f"  [+] 太慢通關 (Too Slow)    : {too_slow} ({(too_slow/total_valid)*100:.1f}%)"
    )
    print(f"  [X] 玩家死亡 (Died)        : {died} ({(died/total_valid)*100:.1f}%)")
    print(
        f"  [!] 路徑堵死 (Blocked)     : {blocked} ({(blocked/total_valid)*100:.1f}%)"
    )
    print(
        f"  [T] 超時 (Timeout)         : {timeout} ({(timeout/total_valid)*100:.1f}%)"
    )
    print("=" * 40)

    # 還原設定
    config.PLAYER_MODE = original_mode
    env.close()


if __name__ == "__main__":
    # 可以在這裡修改要測試的模型名稱
    evaluate_model(model_path="maze_master_ppo", n_episodes=1000)
