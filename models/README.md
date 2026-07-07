# models/ — 模型檔案說明

> 目前**正式使用中**的模型是專案根目錄的 `maze_master_ppo.zip`
> （`main.py` 演示與 PyInstaller 打包都載入它），不在這個資料夾裡。

## 本層檔案

| 檔案 | 說明 |
|---|---|
| `maze_master_ppo_run_<時間戳>.zip` | `train.py` 每次訓練結束的自動歷史備份（與當次根目錄模型相同） |
| `maze_master_ppo_baseline_perfect_astar.zip` | 舊基準模型：只對「完美 A* 玩家」訓練的版本，供與玩家個性隨機化版本對照（見 docs/PROJECT_OVERVIEW.md §8.5） |

## archive/ — 歷史實驗模型

開發過程（2025-11 ~ 2026-01）各階段的實驗模型，按日期歸檔，檔名多為當時的
實驗註記（例如 `success_71_30avg`、`放怪把玩家弄死`）。僅供回溯開發歷程，
不保證與目前的環境程式碼相容（觀測格式歷經多次改版）。

## 使用方式

想用某個歷史模型跑演示或評估：

```bash
python adaptation_check.py models/archive/202601/maze_master_ppo_flow_45 100
# 或把該 zip 複製到根目錄改名為 maze_master_ppo.zip 後執行 main.py
```

（Stable-Baselines3 的 `PPO.load()` 路徑不需要 `.zip` 副檔名。）
