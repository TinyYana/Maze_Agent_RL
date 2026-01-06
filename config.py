import numpy as np

# --- 環境設定 ---
GRID_SIZE = 15
WINDOW_SIZE = 600
FPS = 15

# --- 遊戲機制 ---
PLAYER_MAX_HP = 3
PLAYER_INITIAL_HAMMERS = 2
ACTIONS_PER_TURN = 1
K_STEP = 1
MONSTER_SPEED = 1
MAX_MONSTERS = 3

PLAYER_MODE = "HUMAN"

# --- AI 行為設定 ---
# 0.0 = 完美 A* (最短路徑)
# 0.3 = 輕微隨機 (偶爾繞路)
# 1.0 = 高度隨機 (容易迷路)
AI_RANDOMNESS = 0.0

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

# --- 視覺美化設定 (Visuals) ---
# 使用柔和的現代配色 (RGB)

# 背景與格線
COLOR_BG = (241, 242, 246)  # 淺灰白背景
COLOR_GRID = (223, 228, 234)  # 非常淡的格線

# 牆壁 (深色石板風格)
COLOR_WALL_TOP = (87, 96, 111)  # 牆壁頂部
COLOR_WALL_SIDE = (47, 53, 66)  # 牆壁陰影面

# 玩家 (亮青色/英雄色)
COLOR_PLAYER_BODY = (52, 152, 219)
COLOR_PLAYER_BORDER = (41, 128, 185)

# 出口 (金色/傳送門)
COLOR_EXIT_OUTER = (241, 196, 15)
COLOR_EXIT_INNER = (243, 156, 18)

# 怪物 (警示紅)
COLOR_MONSTER_BODY = (231, 76, 60)
COLOR_MONSTER_EYE = (44, 62, 80)

# UI 介面
COLOR_UI_BG = (44, 62, 80)  # 深藍色儀表板背景
COLOR_TEXT = (236, 240, 241)  # 白色文字

# 舊的顏色定義 (為了相容性保留，或直接替換)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
