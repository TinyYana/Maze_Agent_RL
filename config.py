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

# --- 顏色定義 (R, G, B) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
