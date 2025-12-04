import numpy as np

# --- 環境設定 ---
GRID_SIZE = 15
WINDOW_SIZE = 600
FPS = 15

# --- 遊戲機制 ---
PLAYER_MAX_HP = 3
PLAYER_INITIAL_HAMMERS = 2  # 新玩家初始破牆工具數量
ACTIONS_PER_TURN = 1
K_STEP = 1
MONSTER_SPEED = 1
MAX_MONSTERS = 3  # 場上最多同時存在怪物

PLAYER_MODE = "HUMAN"  # 'AI' (A*自動) 或 'HUMAN' (手動)

# --- 時間限制 ---
TIME_MIN = 30
TIME_MAX = 100

# --- ID 定義 ---
ID_EMPTY = 0
ID_WALL = 1
ID_PLAYER = 2
ID_EXIT = 3
ID_MONSTER = 4

# --- 獎勵設定 ---
REWARD_GOAL = 50
REWARD_TIMEOUT = -20
REWARD_BLOCKED = -0.1
REWARD_TOO_FAST = -30
REWARD_TOO_SLOW = -20
REWARD_STEP = 1.5
REWARD_DEATH = -20

REWARD_HIT = 1.5  # 保持正向獎勵，鼓勵「蹭血」增加難度，但不能殺死
REWARD_PATH_EXTEND = 3.0
REWARD_INVALID_ACTION = -0.1  # 無效動作的小懲罰 (例如怪物滿了還想放)
REWARD_MOVE_EXIT = -1.0  # 移動出口的懲罰

# --- 顏色定義 (R, G, B) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
