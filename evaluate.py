import numpy as np
import time
import os
import datetime
from stable_baselines3 import PPO
from envs.maze_env import MazeEnv
import config
from collections import Counter


def evaluate_model(model_path="maze_master_ppo", n_episodes=100):
    """
    評估模型效能並輸出報告至檔案
    :param model_path: 模型路徑 (不含 .zip)
    :param n_episodes: 測試回合數
    """

    # 建立 logs 資料夾
    log_dir = "evaluation_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 產生報告檔名 (包含時間戳記)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{log_dir}/eval_report_{model_path}_{timestamp}.txt"

    # 強制設定為 AI 模式
    original_mode = config.PLAYER_MODE
    config.PLAYER_MODE = "AI"

    # 準備緩衝輸出的字串列表
    output_buffer = []

    def log(message):
        """同時印出到螢幕並存入緩衝區"""
        print(message)
        output_buffer.append(message)

    log(f"--- 開始評估模型: {model_path} ---")
    log(f"測試回合數: {n_episodes}")
    log(f"目標步數範圍 (Flow): {config.TIME_MIN} ~ {config.TIME_MAX}")
    log(f"地圖尺寸: {config.GRID_SIZE}x{config.GRID_SIZE}")

    if not os.path.exists(f"{model_path}.zip"):
        log(f"錯誤: 找不到模型檔案 {model_path}.zip")
        return

    env = MazeEnv(render_mode=None)
    model = PPO.load(model_path)

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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        result_type = info.get("result", "unknown")

        # 除錯資訊 (只印到螢幕，不一定要寫入報告，視需求而定)
        if result_type == "too_fast" and np.random.rand() < 0.05:
            print(f"[Debug] Too Fast: Steps={env.player_steps}")

        stats["results"].append(result_type)
        stats["steps"].append(env.player_steps)
        stats["hps"].append(env.player_hp)
        stats["rewards"].append(episode_reward)

        if (i + 1) % 10 == 0:
            print(f"進度: {i + 1}/{n_episodes}...", end="\r")

    total_time = time.time() - start_time
    log(f"\n評估完成！耗時: {total_time:.2f} 秒\n")

    # --- 計算統計指標 ---
    results_count = Counter(stats["results"])
    total_valid = len(stats["results"])

    flow_success = results_count.get("flow_success", 0)
    too_fast = results_count.get("too_fast", 0)
    too_slow = results_count.get("too_slow", 0)
    died = results_count.get("died", 0)
    timeout = results_count.get("timeout", 0)
    blocked = results_count.get("blocked", 0)

    flow_rate = (flow_success / total_valid) * 100
    survival_rate = ((total_valid - died - blocked) / total_valid) * 100

    avg_steps = np.mean(stats["steps"])
    std_steps = np.std(stats["steps"])
    avg_reward = np.mean(stats["rewards"])
    avg_hp = np.mean(stats["hps"])

    # --- 輸出報表 ---
    log("=" * 40)
    log("       MODEL EVALUATION REPORT       ")
    log(f"       Date: {timestamp}             ")
    log("=" * 40)
    log(f"平均獎勵 (Avg Reward): {avg_reward:.2f}")
    log(f"平均步數 (Avg Steps) : {avg_steps:.2f} (±{std_steps:.2f})")
    log(f"平均剩餘 HP          : {avg_hp:.2f} / {config.PLAYER_MAX_HP}")
    log("-" * 40)
    log(f"心流成功率 (Flow Rate) : {flow_rate:.1f}%")
    log(f"存活率 (Survival Rate) : {survival_rate:.1f}%")
    log("-" * 40)
    log("結果分佈 (Result Distribution):")
    log(f"  [O] 完美區間 (Flow Success): {flow_success} ({flow_rate:.1f}%)")
    log(
        f"  [-] 太快通關 (Too Fast)    : {too_fast} ({(too_fast/total_valid)*100:.1f}%)"
    )
    log(
        f"  [+] 太慢通關 (Too Slow)    : {too_slow} ({(too_slow/total_valid)*100:.1f}%)"
    )
    log(f"  [X] 玩家死亡 (Died)        : {died} ({(died/total_valid)*100:.1f}%)")
    log(f"  [!] 路徑堵死 (Blocked)     : {blocked} ({(blocked/total_valid)*100:.1f}%)")
    log(f"  [T] 超時 (Timeout)         : {timeout} ({(timeout/total_valid)*100:.1f}%)")
    log("=" * 40)

    # 寫入檔案
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_buffer))

    print(f"\n報告已儲存至: {report_filename}")

    config.PLAYER_MODE = original_mode
    env.close()


if __name__ == "__main__":
    # 記得修改這裡的模型名稱以符合您最新的訓練結果
    evaluate_model(model_path="maze_master_ppo", n_episodes=1000)
