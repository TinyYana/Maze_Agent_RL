# 對抗式訓練實驗評估與修改方向 (2026-07-14)

對 dev 分支對抗式訓練框架 (fb74bce, Shen900921) 的獨立實驗評估，
以及讓專案更符合強化學習宗旨的修改規劃。

實驗腳本：交叉評估 / Master 行為統計 / 掛機探測，各 20 回合固定種子 (90000+ep)。

## TL;DR

框架本身是對的：雙方都是學習中的 agent、MaskablePPO 遮罩非法動作、
決策時序正確、對手池防循環退化——這是專案從「單 agent 打腳本」升級成
「真.多智能體對抗」的正確一步。但實驗暴露三個問題，都有明確解法：

1. **Master 的獎勵和專案宗旨(心流)脫鉤**——它學成了「超時磨王」
2. **玩家側災難性遺忘**——r4 玩家在靜態迷宮 0/20 全超時
3. **RL 玩家自己重新發現了人類玩家的掛機漏洞**——證明漏洞防治必須進遮罩層

## 實驗數據

### 交叉評估矩陣 (玩家通關數 / 20 回合，括號為贏時平均步數)

|            | 無 Master   | master_r0  | master_r4 |
|------------|-------------|------------|-----------|
| player_r0  | **18** (39) | 7 (55)     | 0 (—)     |
| player_r2  | 4 (98)      | 12 (9.1)   | 3 (71)    |
| player_r4  | **0** (—)   | 12 (**8.5**)| 7 (92)   |

- Master 側軍備競賽真實發生：master_r4 對舊玩家封零 (0/20，17 局超時)。
- 玩家側的「進步」有一半是假的：player_r4 打 master_r0 贏時只要 8.5 步
  ——15x15 迷宮不可能 8 步到對角，是「等出口被洗到附近再撲」的掛機打法。
- player_r4 在無 Master 的靜態迷宮 20 局全超時：導航基本功被洗掉了。

### master_r4 行為統計 (20 回合 vs player_r4，2587 次決策)

搬出口 **57%** / 清除 19% / 蓋牆 13% / 放怪 9% / skip 2%。
結局：**timeout 15/20 (75%)**，心流通關僅 2/20。

原因在獎勵表：`MASTER_REWARD_PER_TURN=0.05` 且超時無懲罰，
超時一局賺 ~150×0.05=7.5 分 > 心流通關 ~3.75 分。
Master 的理性最優解就是拿著出口玩你追我跑、把每局拖到超時。
訓練 log 佐證：Master 側 ep_len 從 107 一路爬到 132 (逼近 150 超時上限)。

### 掛機探測 (腳本化「來回走 + 出口靠近就撲」bot，重現玩家回報的漏洞)

| 對手 | 掛機通關 | 死亡 | 出口曾到的最近距離(中位) |
|---|---|---|---|
| 舊 maze_master_ppo (main) | 5/20 | 13 | 9 步 |
| 對抗 master_r4 | 2/20 | 15 | 3 步 |

對抗訓練天然壓制掛機 (25%→10%，主要回應是放怪擊殺)，但沒有根治：
master_r4 高頻搬出口的過程中，出口仍會晃進撲擊範圍被白撿。

## 修改方向 (按優先級)

### P0-1 Master 獎勵對齊心流宗旨

Master 該是「節奏導演」不是「純拖延者」。對抗性保留 (它仍然阻止速通、
仍然被玩家的進步逼著變強)，但終局獎勵改成以心流區間為中心：

```python
MASTER_REWARD_FLOW = +10.0      # 玩家在 50-100 步通關：Master 的本職成功
MASTER_REWARD_TIMEOUT = -8.0    # 磨到超時 = 失職 (新增，根治磨王)
MASTER_REWARD_TOO_SLOW = -3.0   # 拖過頭 (新增)
MASTER_REWARD_TOO_FAST = -5.0   # 沿用
MASTER_REWARD_PLAYER_DIED = -10.0  # 沿用
# PER_TURN 改為只在 current_time <= TIME_MAX 時累積，超過心流窗就停止計分
```

這樣「玩家 (速通) vs Master (控節奏)」的張力仍是零和式的對抗，
但均衡點會被拉進心流區間，而不是拉到超時上限。

### P0-2 玩家輪次混合環境，根治災難性遺忘

train_player.py 的教訓 (「全放 Master 會把導航能力洗掉」) 沒有帶進
train_adversarial.py。修法一行：`player_vecenv` 的對手抽樣改成
「每輪保留 1/4 的 env 無 Master (靜態迷宮)」，並把「無 Master 通關率」
納入每輪評估，跌破門檻就警報。

### P0-3 出口距離下限進 master_action_masks

main 分支已上的反掛機規則 (fix/exit-anti-camping：出口新位置與玩家的
路徑距離 < 8 即撤銷) 要移植到 dev，而且該做在**遮罩層**：
`mask[3] &= 曼哈頓距離 >= EXIT_MIN_PLAYER_DIST 的格子`
(曼哈頓是路徑距離的下界，夠擋掛機；精確 A* 逐格算太貴)。
遮罩層防治對 RL 是更好的做法：非法動作機率直接歸零，
不浪費探索預算在「試了會被撤銷」的動作上。
注意：P0-1 給了 Master 心流獎勵後，它會重新有動機把出口送到玩家腳邊
收割通關——**沒有這條遮罩，掛機漏洞會在新獎勵下復活**。三個 P0 是一組的。

### P1-1 評估體系升級：從單點自評到系統性體檢

這次抓到的三個問題，全都是協作者的單行評估 (player_rN vs master_rN)
看不到的。每輪訓練後固定跑：

1. **交叉矩陣**：新 checkpoint vs 全部歷史 checkpoint (抓循環退化/假進步)
2. **無 Master 基準**：抓災難性遺忘
3. **掛機探測 bot**：已知漏洞的回歸測試 (之後每發現一種漏洞就加一個探測 bot)
4. headline 指標從「通關率」改成 **flow rate** (flow_success 佔比)
   ——通關率把 too_fast/too_slow 混進來，掩蓋了節奏控制的好壞

CSV 欄位建議：round, flow_rate, winrate, timeout_rate, death_rate,
avg_clear_time, static_winrate (無 Master), camper_winrate。

以上 1-3 已落地成 `evaluate_adversarial.py` (交叉矩陣 + 靜態基準 +
掛機探測一鍵跑)，之後每輪訓練完直接執行即可。

### P1-2 對手池抽樣搬到 reset 層

目前每個 worker 整輪固定打同一個對手 (rank 決定)。改成每次 reset
重抽，單一 worker 內就有對手多樣性，梯度更平滑 (模型都已在主進程
批次推論，換對手零成本，見下方 GPU 優化)。

### P2 選配

- 觀測第 9 通道：心流窗相位 (current_time 相對 TIME_MIN/TIME_MAX)，
  讓 Master 不用從 time 通道自己推算窗界。
- 輪數加深 (8-12 輪 × 較短步數)，評估回合數 50→200 (±7% 信賴區間)。
- PlayerEnvV2 的距離塑形從「每步兩次 A*」改為「地形變動時算一次
  BFS 距離場」——玩家側採樣的下一個吞吐瓶頸。
- 長期：把 flow 從「終局判定」改成「過程獎勵」(每回合依玩家進度與
  理想節奏曲線的偏差給分)，讓 Master 學會全程調速而非終點衝刺。

## GPU 吞吐優化 (已實作，見 envs/batched_opponent.py)

瓶頸不在學習端而在採樣端：凍結對手 (ConvGRU K=8/128ch，每次推論
~1G MACs) 原本在 12 個 worker 裡各自用單執行緒 CPU 逐筆跑 (pod 實測
fps 88，本機桌機只有 fps 15)。改為 `BatchedFrozenOpponent` VecEnv
包裝器：worker 只跑遊戲邏輯，對手觀測收回主進程按 checkpoint 分組、
批次過 GPU。

避免「GPU 跑完換 CPU 跑、互相乾等」的串行乾等：
- Master 側：apply_edit 編輯指令一次扇出全部 worker 再統一收回，
  worker 的遊戲邏輯真正並行 (原本 N 次串行 IPC 往返)。
- 玩家側：master_io 用 env_method 一次廣播 (送全部→收全部，worker 並行)。
- 教訓：跨步預取 (step_wait 先送、step_async 再收) 不可行——
  MaskablePPO 每步的 get_action_masks 走同一條 pipe，回覆會錯位。

其他要點：
- 決策時序與原設計完全一致 (Master 先編輯、對手看編輯後局面)；
  MasterEnv 拆成 apply_edit + step 兩段式跨進程完成。
- 等價性驗證：test_adversarial.py，同種子同動作序列下外部路徑與
  env 內建路徑逐步狀態完全相同 (120 步 × 雙環境)。
- learner 端加 TF32 + cudnn benchmark (固定形狀卷積網路的免費加速)。
- `BATCH_OPP=0` 可退回原路徑 (無 GPU 的機器)。
- 實測吞吐 (RTX 4070 + 20 執行緒桌機, 12 envs, learner 皆在 GPU)：

| 側 | 原路徑 (worker CPU) | 批次 GPU | 加速 |
|---|---|---|---|
| Master 側 (v2 對手, 16368 步) | 15 fps (1092s) | **131 fps** (125s) | **8.7x** |
| 玩家側 (Master 對手, 8184 步) | 61 fps | **96 fps** | 1.6x |

玩家側加速較小的原因：1/4 env 是靜態迷宮 (本來就沒對手推論)，且
PlayerEnvV2 每步跑兩次 A* 做距離塑形，環境邏輯本身佔比高——
下一個吞吐瓶頸在這裡 (可改成單次 BFS 距離場，見 P2)。

批次版在桌機上已超過 3090 pod 原路徑的 88 fps；同樣的改動搬回
pod 預期還會更快 (worker 不再載 torch 模型，記憶體也省 12×660MB)。

## 下一步

1. 三個 P0 一起上 (獎勵 + 混合環境 + 遮罩)，用 GPU 優化後的管線重跑
   8 輪對抗，每輪跑 P1-1 的完整體檢。
2. 目標驗收：flow rate 顯著上升、timeout rate 下降、靜態通關率不退化、
   掛機探測通關 0。
3. 把最終 Master 接回 main.py 的 HUMAN 模式，讓真人玩家對上
   「以把你保持在心流區間為目標」的導演——這才是本專案宗旨的完全體。
