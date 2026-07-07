"""檢驗 Maze Master 是否隨玩家表現動態調整行為。

用法: python adaptation_check.py [模型路徑] [每組回合數]

兩個檢驗層次：
1. 跨玩家個性：對「高手 / 普通 / 手殘」玩家，AI 的動作分布與結果是否不同
2. 回合內節奏：當玩家預估抵達時間「太快 / 剛好 / 太慢」時，AI 選擇的動作是否改變
   (太快時應該多蓋牆拖慢；太慢時應該少干預)
"""
import sys
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

import config
from agents.astar_bot import astar_path
from envs.maze_env import MazeEnv

PROFILES = {
    "高手 (完美A*)": {"randomness": 0.0, "hesitate": 0.0},
    "普通 (偶爾繞路)": {"randomness": 0.4, "hesitate": 0.1},
    "手殘 (常迷路+猶豫)": {"randomness": 0.8, "hesitate": 0.3},
}
ACTION_NAMES = {0: "skip", 1: "wall", 2: "remove", 3: "exit", 4: "monster"}


def pace_bucket(env):
    """依目前步數 + 剩餘最短路徑，預估玩家會太快/剛好/太慢抵達"""
    path = astar_path(env.maze, env.player_pos, env.exit_pos)
    if path is None:
        return None
    eta = env.current_time + len(path) - 1
    if eta < config.TIME_MIN:
        return "預估太快"
    if eta > config.TIME_MAX:
        return "預估太慢"
    return "預估剛好"


def main(model_path="maze_master_ppo", n_episodes=100):
    config.PLAYER_MODE = "AI"
    config.PLAYER_PROFILE_RANDOMIZE = False

    model = PPO.load(model_path)
    env = MazeEnv(render_mode=None)

    print(f"模型: {model_path} | 每組 {n_episodes} 回合 | 心流區間 {config.TIME_MIN}-{config.TIME_MAX} 步\n")

    pace_action = {}  # (bucket, action_name) -> count，跨所有 profile 彙整

    print("=== 檢驗 1：對不同水準玩家的行為差異 ===")
    header = f"{'玩家個性':<14}{'心流率':>7}{'均步數':>8}{'蓋牆/百步':>10}{'放怪/百步':>10}{'搬出口/百步':>11}{'按兵/百步':>10}"
    print(header)

    for name, profile in PROFILES.items():
        results, steps_list = [], []
        counts_sum = Counter()

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=1000 + ep)  # 各組共用同批迷宮，控制變因
            env.player_randomness = profile["randomness"]
            env.player_hesitate_prob = profile["hesitate"]

            terminated = False
            turns = 0
            while not terminated:
                bucket = pace_bucket(env)
                action, _ = model.predict(obs, deterministic=True)
                if bucket:
                    act_name = ACTION_NAMES.get(int(np.asarray(action).reshape(-1)[2]), "?")
                    pace_action[(bucket, act_name)] = pace_action.get((bucket, act_name), 0) + 1
                obs, reward, terminated, truncated, info = env.step(action)
                turns += 1
                if truncated:
                    break

            results.append(info.get("result", "unknown"))
            steps_list.append(turns)
            counts_sum.update(env.action_counts)

        total_turns = sum(steps_list)
        flow = results.count("flow_success") / n_episodes * 100
        per100 = {k: counts_sum[k] / total_turns * 100 for k in ("wall", "monster", "exit", "skip")}
        print(
            f"{name:<14}{flow:>6.1f}%{np.mean(steps_list):>8.1f}"
            f"{per100['wall']:>10.1f}{per100['monster']:>10.1f}{per100['exit']:>11.1f}{per100['skip']:>10.1f}"
        )
        print(f"    結果分佈: {dict(Counter(results))}")

    print("\n=== 檢驗 2：回合內依玩家節奏的動作選擇 (所有回合彙整) ===")
    buckets = ["預估太快", "預估剛好", "預估太慢"]
    print(f"{'節奏狀態':<10}" + "".join(f"{a:>9}" for a in ACTION_NAMES.values()) + f"{'樣本數':>9}")
    rates = {}  # bucket -> {action: 比例}
    for b in buckets:
        total = sum(pace_action.get((b, a), 0) for a in ACTION_NAMES.values())
        if total == 0:
            continue
        row = "".join(f"{pace_action.get((b, a), 0) / total * 100:>8.1f}%" for a in ACTION_NAMES.values())
        print(f"{b:<10}{row}{total:>9}")
        rates[b] = {a: pace_action.get((b, a), 0) / total for a in ACTION_NAMES.values()}

    print("\n=== 結論 ===")
    total_samples = sum(pace_action.values())
    slow_samples = sum(pace_action.get(("預估太慢", a), 0) for a in ACTION_NAMES.values())
    if total_samples and slow_samples / total_samples < 0.02:
        print(f"⚠️ 「預估太慢」樣本僅 {slow_samples}/{total_samples} ({slow_samples/total_samples*100:.1f}%)，該列統計參考性低。")
    if "預估太快" in rates and "預估太慢" in rates:
        # 主要指標：干預率 (非 skip 的比例) 是否隨玩家落後而上升
        act_fast = 1 - rates["預估太快"]["skip"]
        act_ok = 1 - rates.get("預估剛好", {}).get("skip", 1)
        act_slow = 1 - rates["預估太慢"]["skip"]
        print(f"干預率 (非按兵不動)：太快 {act_fast*100:.1f}% / 剛好 {act_ok*100:.1f}% / 太慢 {act_slow*100:.1f}%")
        print(f"蓋牆比例：太快 {rates['預估太快']['wall']*100:.1f}% vs 太慢 {rates['預估太慢']['wall']*100:.1f}%")
        print(f"拆除比例：太快 {rates['預估太快']['remove']*100:.1f}% vs 太慢 {rates['預估太慢']['remove']*100:.1f}%")

        if abs(act_slow - act_fast) * 100 > 10:
            direction = "玩家落後時 AI 明顯更積極出手" if act_slow > act_fast else "玩家越快 AI 越積極出手"
            print(f"→ 有動態調整跡象：{direction} (干預率差 {abs(act_slow-act_fast)*100:.1f} 百分點)。")
            if rates["預估太慢"]["remove"] > rates["預估太快"]["remove"] + 0.05:
                print("→ 且玩家落後時拆除 (開路) 比例上升，方向符合「幫慢玩家、擋快玩家」的預期。")
        else:
            print("→ 未見明顯動態調整：AI 的干預程度與玩家節奏幾乎無關。")
        print("(注意：此為觀察性相關，非因果——AI 蓋牆多的回合玩家自然變慢。)")

    env.close()


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "maze_master_ppo"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(model_path, n)
