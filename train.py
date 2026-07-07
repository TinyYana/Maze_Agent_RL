import os
import numpy as np
import datetime  # 新增：用於產生時間戳記
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.maze_env import MazeEnv

# 記得確保 config 有被引入
import config


def make_env(rank):
    """建立單一訓練環境 (供 SubprocVecEnv 子進程呼叫)"""

    def _init():
        # 子進程有自己的 config 模組，需在此設定
        config.PLAYER_MODE = "AI"
        config.PLAYER_PROFILE_RANDOMIZE = True
        env = MazeEnv(render_mode=None)
        # 每個環境用不同種子，避免所有進程產生一模一樣的迷宮序列
        env.rng = np.random.default_rng(114514 + rank)
        # Monitor 包裝讓 TensorBoard 記錄 ep_rew_mean / ep_len_mean
        return Monitor(env)

    return _init


# --- 新增：自定義 Callback 用來存 AI 看到的圖 ---
class SaveObservationCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        import cv2

        # 每隔 save_freq 步執行一次
        if self.n_calls % self.save_freq == 0:
            # 取得目前的觀測值 (這就是 AI 看到的 raw data)
            # obs 通常是 (Batch_Size, Channels, Height, Width) 或 (B, H, W, C)
            obs = self.locals["new_obs"]

            # 取第一筆資料 (因為可能有 batch)
            img_data = obs[0]

            # 處理 Channel 維度 (如果是 Channel-First)
            if img_data.shape[0] < img_data.shape[1]:
                img_data = np.moveaxis(img_data, 0, -1)

            # 去除 channel 維度變成純 2D 陣列 (H, W)
            # 例如從 (15, 15, 1) 變成 (15, 15)
            heatmap = img_data.squeeze()

            # --- 建立彩色對照表 (RGB) ---
            # 建立一個與 heatmap 相同大小的彩色畫布 (H, W, 3)
            h, w = heatmap.shape
            color_img = np.zeros((h, w, 3), dtype=np.uint8)

            # 填色邏輯 (BGR 格式，因為 OpenCV 預設是 BGR)
            # 因為 Env 裡把 ID 乘以了 50，所以我們要偵測的是這些數值：

            # 空地 0: 米白色
            color_img[heatmap == 0] = [240, 240, 240]

            # 牆壁 1 * 50 = 50: 深灰色
            color_img[heatmap == 50] = [60, 60, 60]

            # 玩家 2 * 50 = 100: 藍色
            color_img[heatmap == 100] = [255, 100, 50]

            # 出口 3 * 50 = 150: 金黃色
            color_img[heatmap == 150] = [0, 215, 255]

            # 怪物 4 * 50 = 200: 紅色
            color_img[heatmap == 200] = [50, 50, 255]

            # 放大圖片以便觀察 (放大 20 倍)
            scale = 20
            large_img = cv2.resize(
                color_img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
            )

            # 加上格線 (選擇性)
            for i in range(0, large_img.shape[0], scale):
                cv2.line(large_img, (0, i), (large_img.shape[1], i), (200, 200, 200), 1)
            for j in range(0, large_img.shape[1], scale):
                cv2.line(large_img, (j, 0), (j, large_img.shape[0]), (200, 200, 200), 1)

            # 存檔
            filename = os.path.join(self.save_path, f"obs_step_{self.n_calls}.png")
            cv2.imwrite(filename, large_img)

        return True


def train():
    # 1. 產生實驗 ID (時間戳記)
    # 例如：run_20260106_153025
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    # 2. 設定路徑
    log_dir = "./tensorboard_logs/"
    debug_img_dir = os.path.join("./debug_vision/", run_name)  # 依照時間分類圖片
    models_dir = "./models/"  # 依照時間備份模型

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(debug_img_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 3. 建立環境 (多進程並行收集 + 每回合隨機抽玩家個性，避免只會對付完美 A*)
    # 環境步進是純 Python A* 運算 (CPU 瓶頸)，SubprocVecEnv 可線性加速資料收集
    n_envs = min(8, os.cpu_count() or 4)
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # 4. 定義模型 (n_steps 除以環境數，維持每次更新的總 rollout 量 4096 不變)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        batch_size=256,
        ent_coef=0.1,
        gamma=0.99,
        n_steps=max(4096 // n_envs, 64),
        clip_range=0.2,
        gae_lambda=0.95,
        device="auto",
        tensorboard_log=log_dir,
    )

    # 建立 Callback 實例 (傳入這次專屬的資料夾)
    vision_callback = None
    try:
        import cv2  # noqa: F401

        vision_callback = SaveObservationCallback(save_freq=1000, save_path=debug_img_dir)
    except ImportError:
        print("⚠️ cv2 無法載入，將跳過 AI 視野圖儲存 callback。")

    total_timesteps = int(os.getenv("TRAIN_TIMESTEPS", "500000"))

    import torch

    print(f"=== 開始訓練: {run_name} ===")
    print(f"運算裝置: {model.device} (torch {torch.__version__}, CUDA 可用: {torch.cuda.is_available()})")
    if not torch.cuda.is_available():
        print("⚠️ 未偵測到 CUDA！將使用 CPU 訓練 (慢)。請安裝 GPU 版 torch:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps")
    print(f"並行環境數: {n_envs} (SubprocVecEnv)")
    print(f"玩家個性隨機化: True (繞路 {config.PLAYER_RANDOMNESS_RANGE}, 猶豫 {config.PLAYER_HESITATE_RANGE})")
    print(f"Entropy Coef: {model.ent_coef}")
    print(f"AI 視野圖將儲存於: {debug_img_dir}")
    print(f"訓練步數: {total_timesteps}")

    # 5. 開始訓練
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"maze_ppo_{run_name}",  # TensorBoard 也加上名稱區分
        callback=vision_callback,
    )

    # 6. 儲存模型
    # (A) 存成預設名稱，方便 main.py 直接讀取
    default_model_path = "maze_master_ppo"
    model.save(default_model_path)

    # (B) 存成備份名稱，保留歷史紀錄
    backup_model_path = os.path.join(models_dir, f"maze_master_ppo_{run_name}")
    model.save(backup_model_path)

    env.close()
    print("訓練完成！")
    print(f"主要模型已更新: {default_model_path}.zip")
    print(f"歷史備份已存檔: {backup_model_path}.zip")


if __name__ == "__main__":
    train()
