import numpy as np

# --- 環境設定 ---
GRID_SIZE = 15
WINDOW_SIZE = 600
FPS = 5

# --- 遊戲機制 ---
PLAYER_MAX_HP = 3
ACTIONS_PER_TURN = (
    1  # 重要修改：從 5 改為 1。一次只做一個動作，讓因果關係更明確，Agent 更容易學。
)
K_STEP = 1
MONSTER_SPEED = 1

# --- 時間限制 ---
TIME_MIN = 150
TIME_MAX = 200

# --- ID 定義 ---
ID_EMPTY = 0
ID_WALL = 1
ID_PLAYER = 2
ID_EXIT = 3
ID_MONSTER = 4

# --- 獎勵設定 ---
REWARD_GOAL = 50  # 稍微降低，避免 Agent 為了這個大獎勵而不敢冒險
REWARD_TIMEOUT = -20  # 超時懲罰
REWARD_BLOCKED = -2  # 再次降低堵路懲罰，讓 Agent 敢於嘗試放牆壁 (原本 -5)
REWARD_TOO_FAST = -100  # 加重！如果玩家太快通關，這是 Agent 的嚴重失職
REWARD_TOO_SLOW = -20
REWARD_STEP = 0.5  # 增加！每拖住玩家一步，給予顯著獎勵 (原本 0.1)
REWARD_DEATH = -30  # 降低死亡懲罰，允許偶爾失手，這樣 Agent 才敢放怪物
REWARD_HIT = 10  # 增加！鼓勵讓玩家受傷（增加緊張感）
REWARD_PATH_EXTEND = 2.0  # 新增：如果 Agent 的動作成功讓路徑變長，給予額外獎勵

# --- 顏色定義 (R, G, B) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
