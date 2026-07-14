# 對抗訓練中斷點盤點與復原手冊 (2026-07-14)

GPUtw pod 的 /vault NFS 在 Round 3 Master 側訓練中途故障，pod 已由使用者關機止血。
本文件記錄中斷當下的完整狀態，供伺服器修復 (可能換 pod) 後一鍵復原。

## 本機資產 (完整，G:\Maze_Agent_RL 為正本)

- **程式碼**：全部最新，包含
  - `train_adversarial.py`（含 worker 單執行緒修復 `_limit_worker_threads`）
  - `envs/adversarial.py`（8 通道觀測、雙方遮罩、PlayerEnvV2/MasterEnv）
  - `agents/planner_policy.py`（ConvGRU 規劃核心 128ch/K=8）
  - `config.py`（MONSTER_MOVE_PATTERN=(0,1,1)、MASTER_REWARD_*）
- **上傳包**：scratchpad `maze_rl/`（含 smoke_remote.py），可直接重傳
- **v1 系譜模型**：player_ppo.zip（4ch 速通版）、models/ 下全部歷史備份
- **已知評估數據**：models_adv/adversarial_curve_partial.csv（R1、R2）

## 只存在 /vault 的資產 (待確認存活)

路徑 `/vault/ai-gpu/outputs/maze_rl/`：
- player_r0.zip (11.7MB, 玩家預熱 1M 步, ep_rew 22)
- master_r0.zip (12.3MB, Master 預熱 1M 步 vs nn4)
- player_r1.zip、master_r1.zip（Round 1 完成）
- player_r2.zip、master_r2.zip（Round 2 完成）
- player_r3.zip（Round 3 玩家側完成）
- adversarial_curve.csv（R1、R2 兩行，內容已抄錄到本機 partial 檔）
- 日誌：/vault/ai-gpu/logs/maze_adv_run1~3.log

中斷點：Round 3 Master 側跑到約 124k/150k 步（master_r3 未存檔）。

## 復原步驟 (新連線建立後)

1. `nvidia-smi` + `ls /vault/ai-gpu/outputs/maze_rl/` 確認 GPU 與資料存活。
2. **第一優先**：gpu-download 把 outputs/maze_rl 全部 zip 下載回本機 models_adv/（保底）。
3. 若換了 pod：重裝套件 `pip install stable_baselines3==2.7.0 sb3-contrib==2.7.0 gymnasium pygame`，
   重傳 scratchpad 的 maze_rl 上傳包。
4. 續跑（從 Round 3 Master 側重來）：
   ```
   cd /vault/ai-gpu/projects/maze_rl && \
   SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy OMP_NUM_THREADS=1 \
   ADV_OUT=/vault/ai-gpu/outputs/maze_rl N_ENVS=12 DEVICE=cuda \
   SKIP_PREWARM=1 ROUND_STEPS=150000 N_ROUNDS=4 START_ROUND=3 \
   nohup python3 -u train_adversarial.py > /vault/ai-gpu/logs/maze_adv_run4.log 2>&1 &
   ```
   注意：train_adversarial.py 需先加 START_ROUND 支援（round 迴圈起點改為
   `range(int(os.getenv("START_ROUND", "1")), N_ROUNDS + 1)`，並讓玩家側
   在 player_r{r} 已存在時跳過）——復原時實作。
5. 若 /vault 資料全損：重跑全流程（無 SKIP_PREWARM），總計約 7 小時。

## 評估基準 (供最終報告對照)

- Round 1: 通關 34%、贏時 45 步、死亡 38%
- Round 2: 通關 32%、贏時 69 步、死亡 22%（均衡點進入心流區間 50-100）
- 對照舊系統：通關 21%、死亡 75-86%
