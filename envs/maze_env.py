class MazeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MazeMasterEnv, self).__init__()

        self.grid_size = 15
        self.cell_size = 40

        self.ID_EMPTY = 0
        self.ID_WALL = 1
        self.ID_PLAYER = 2
        self.ID_EXIT = 3
        self.ID_MONSTER = 4

        # --- 關鍵修改 1: Action Space ---
        # 我們使用 MultiDiscrete 讓 AI 同時輸出三個數值：
        # 1. X 座標 (0 ~ 14)
        # 2. Y 座標 (0 ~ 14)
        # 3. 動作類型 (0: 無動作, 1: 加牆, 2: 刪牆, 3: 放怪)
        self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 4])

        # 觀測空間維持不變 (或是你可以考慮加入 Bot 的路徑資訊進去，但目前先保持簡單)
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None  # 控制 FPS

        self.state = None
        self.player_pos = None
        self.exit_pos = None
        self.current_path = []
        self.prev_path_len = 0  # 記錄上一步的路徑長度，用於計算進步幅度
