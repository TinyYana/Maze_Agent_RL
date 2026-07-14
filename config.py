import numpy as np

# --- 環境設定 ---
GRID_SIZE = 15
WINDOW_SIZE = 600
FPS = 60  # 畫面更新率，同時決定事件輪詢頻率 (直接影響按鍵手感)
AI_STEP_FPS = 12  # AI 模式每秒推進的步數，與畫面更新率脫鉤

# --- 遊戲機制 ---
PLAYER_MAX_HP = 3
PLAYER_INITIAL_HAMMERS = 2
ACTIONS_PER_TURN = 1
K_STEP = 1
MONSTER_SPEED = 1  # (已棄用，由 MONSTER_MOVE_PATTERN 取代)
# 怪物移動節奏：每回合依序取值循環，1=走一步 0=不動。
# (0,1,1) = 「兩回合一步 → 一回合一步」交替，給玩家可學習的逃脫節奏窗
MONSTER_MOVE_PATTERN = (0, 1, 1)
MAX_MONSTERS = 3

# "AI" = A* Bot / "HUMAN" = 手動遊玩 / "NN" = 神經網路玩家 (player_ppo.zip)
PLAYER_MODE = "NN"

# --- AI 行為設定 ---
# 0.0 = 完美 A* (最短路徑)
# 0.3 = 輕微隨機 (偶爾繞路)
# 1.0 = 高度隨機 (容易迷路)
AI_RANDOMNESS = 0.0

# 訓練泛化：每回合隨機抽玩家個性 (繞路程度 + 猶豫機率)，
# 模擬真實玩家而非完美 A*。train.py 會開啟；demo/評估預設關閉。
PLAYER_PROFILE_RANDOMIZE = False
PLAYER_RANDOMNESS_RANGE = (0.0, 0.8)  # 每回合抽樣的繞路程度
PLAYER_HESITATE_RANGE = (0.0, 0.3)  # 每回合抽樣的猶豫機率 (該回合不移動)

# --- 時間限制 ---
TIME_MIN = 50
TIME_MAX = 100

# --- ID 定義 ---
ID_EMPTY = 0
ID_WALL = 1
ID_PLAYER = 2
ID_EXIT = 3
ID_MONSTER = 4

# --- 獎勵設定 (重新調整) ---
REWARD_GOAL = 50  # 提高成功獎勵
REWARD_TIMEOUT = -15
REWARD_BLOCKED = -5.0  # 堵死懲罰加重
REWARD_TOO_FAST = -25  # 太快懲罰加重（鼓勵延長遊戲）
REWARD_TOO_SLOW = -15
REWARD_STEP = 0.1
REWARD_DEATH = -8

REWARD_HIT = 1.0
REWARD_PATH_EXTEND = 0.5  # 每延長一步的獎勵（累積效果更明顯）
REWARD_MOVE_EXIT = -0.3

# === 新增：放牆相關獎勵 ===
REWARD_BUILD_WALL = 3.0  # 放牆成功基礎獎勵（大幅提高）
REWARD_WALL_NEAR_PATH = 2.0  # 放牆靠近玩家路徑額外獎勵
REWARD_SKIP_ACTION = -0.1  # 不做任何事的小懲罰（鼓勵行動）

# === NN 玩家獎勵設定 (train_player.py / envs/player_env.py) ===
# 與 Maze Master 的 REWARD_* 完全獨立：玩家的目標是「盡快通關 + 活著」
PLAYER_REWARD_GOAL = 10.0  # 抵達出口 (不分快慢，心流區間是 Master 的事)
PLAYER_REWARD_SPEED_BONUS = 0.1  # 速度紅利：通關時每比超時上限 (TIME_MAX*1.5) 早一步再加的獎勵
PLAYER_REWARD_DEATH = -5.0  # HP 歸零
PLAYER_REWARD_TIMEOUT = -2.0  # 超時
PLAYER_REWARD_STEP = -0.01  # 每步時間壓力
PLAYER_REWARD_HIT = -1.0  # 每損失 1 HP
# 注意：不要調重。-0.3 會讓從零訓練的策略「怕牆」不敢進窄走廊，探索直接癱瘓 (實測 0/100)
PLAYER_REWARD_BUMP = -0.05  # 撞牆且無鎚 (無效動作)
PLAYER_REWARD_DIST = 0.1  # 距離塑形：A* 距離每縮短 1 格的獎勵
PLAYER_MAX_EPISODE_STEPS = 300  # 玩家環境的回合步數上限 (防呆截斷)

# === 對抗式訓練：Maze Master 零和獎勵 (train_adversarial.py / envs/adversarial.py) ===
# Master 的目標是「拖延」不是「殺人」：存活每回合加分 (玩家拖越久 Master 賺越多)，
# 但殺死玩家或試圖堵死路徑都重罰，維持「可解但難走」的設計哲學
MASTER_REWARD_PER_TURN = 0.05  # 玩家還沒通關的每一回合
MASTER_REWARD_PLAYER_DIED = -10.0  # 玩家死亡 (Master 失職)
MASTER_REWARD_BLOCKED_TRY = -1.0  # 嘗試堵死路徑被規則撤銷
MASTER_REWARD_TOO_FAST = -5.0  # 玩家在 TIME_MIN 之前就通關 (拖延失敗)

# --- 視覺美化設定 (Visuals) ---
# 深色科技感主題 (RGB)

# 背景與格線
COLOR_BG = (24, 29, 43)  # 深藍夜色背景
COLOR_GRID = (35, 42, 60)  # 微亮格線

# 牆壁 (石板浮雕風格)
COLOR_WALL_TOP = (74, 86, 112)  # 牆壁頂部
COLOR_WALL_SIDE = (42, 50, 68)  # 牆壁陰影面

# 玩家 (亮青色/英雄色)
COLOR_PLAYER_BODY = (64, 169, 240)
COLOR_PLAYER_BORDER = (36, 120, 190)

# 出口 (金色/傳送門)
COLOR_EXIT_OUTER = (241, 196, 15)
COLOR_EXIT_INNER = (250, 220, 90)

# 怪物 (警示紅)
COLOR_MONSTER_BODY = (235, 87, 75)
COLOR_MONSTER_EYE = (30, 34, 46)

# UI 介面
COLOR_UI_BG = (17, 21, 32)  # 更深的儀表板背景
COLOR_TEXT = (228, 234, 242)  # 白色文字
COLOR_ACCENT = (155, 89, 182)  # 紫色點綴 (代表 Maze Master)

# 舊的顏色定義 (為了相容性保留，或直接替換)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
