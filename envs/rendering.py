import pygame
import config


class MazeRenderer:
    def __init__(self, window_size, grid_size, fps):
        self.window_size = window_size
        self.grid_size = grid_size
        self.fps = fps
        self.window = None
        self.clock = None
        self.font = None
        self.ui_height = 60  # 增加 UI 高度

    def init_window(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Maze Agent RL")

            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + self.ui_height)
            )
            # 使用更現代的字體，如果沒有則回退到預設
            try:
                self.font = pygame.font.Font(
                    None, 32
                )  # None 使用系統預設好看的無襯線字體
            except:
                self.font = pygame.font.SysFont("Arial", 24)

            self.clock = pygame.time.Clock()

    def render(self, maze, player_hp, player_hammers, player_steps):
        self.init_window()

        # 1. 繪製背景
        canvas = pygame.Surface((self.window_size, self.window_size + self.ui_height))
        canvas.fill(config.COLOR_BG)

        cell_size = self.window_size / self.grid_size

        # 2. 繪製格線 (淡淡的)
        for x in range(self.grid_size + 1):
            pos = x * cell_size
            pygame.draw.line(
                canvas, config.COLOR_GRID, (0, pos), (self.window_size, pos), 1
            )
            pygame.draw.line(
                canvas, config.COLOR_GRID, (pos, 0), (pos, self.window_size), 1
            )

        # 3. 繪製迷宮物件
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_value = maze[r, c]
                # 注意：pygame 座標是 (x, y) 對應 (col, row)
                x_pos = c * cell_size
                y_pos = r * cell_size

                if cell_value == config.ID_WALL:
                    self._draw_wall(canvas, x_pos, y_pos, cell_size)
                elif cell_value == config.ID_EXIT:
                    self._draw_exit(canvas, x_pos, y_pos, cell_size)
                elif cell_value == config.ID_MONSTER:
                    self._draw_monster(canvas, x_pos, y_pos, cell_size)
                # 玩家最後畫，確保他在最上層

        # 4. 繪製玩家 (獨立迴圈確保不被遮擋)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if maze[r, c] == config.ID_PLAYER:
                    x_pos = c * cell_size
                    y_pos = r * cell_size
                    self._draw_player(canvas, x_pos, y_pos, cell_size)

        # 5. 繪製 UI
        self._draw_ui(canvas, player_hp, player_hammers, player_steps)

        # 更新畫面
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)

    def _draw_wall(self, canvas, x, y, size):
        """繪製有立體感的牆壁"""
        # 底部陰影
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(canvas, config.COLOR_WALL_SIDE, rect)

        # 頂部亮面 (稍微縮小並上移一點點，製造厚度感)
        offset = 4
        top_rect = pygame.Rect(x + 2, y + 2, size - 4, size - 4)
        pygame.draw.rect(canvas, config.COLOR_WALL_TOP, top_rect, border_radius=3)

    def _draw_player(self, canvas, x, y, size):
        """繪製圓形玩家"""
        center = (x + size / 2, y + size / 2)
        radius = size / 2 - 4

        # 外框
        pygame.draw.circle(canvas, config.COLOR_PLAYER_BORDER, center, radius)
        # 內體
        pygame.draw.circle(canvas, config.COLOR_PLAYER_BODY, center, radius - 3)

    def _draw_monster(self, canvas, x, y, size):
        """繪製怪物 (菱形或帶眼睛的圓)"""
        center = (x + size / 2, y + size / 2)
        radius = size / 2 - 4

        # 身體
        pygame.draw.circle(canvas, config.COLOR_MONSTER_BODY, center, radius)

        # 眼睛 (兇狠感)
        eye_offset_x = radius / 2.5
        eye_offset_y = radius / 4
        pygame.draw.circle(
            canvas,
            config.COLOR_MONSTER_EYE,
            (center[0] - eye_offset_x, center[1] - eye_offset_y),
            3,
        )
        pygame.draw.circle(
            canvas,
            config.COLOR_MONSTER_EYE,
            (center[0] + eye_offset_x, center[1] - eye_offset_y),
            3,
        )

    def _draw_exit(self, canvas, x, y, size):
        """繪製出口 (同心正方形)"""
        center_x = x + size / 2
        center_y = y + size / 2

        # 外圈
        rect_outer = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
        pygame.draw.rect(canvas, config.COLOR_EXIT_OUTER, rect_outer, border_radius=4)

        # 內圈
        rect_inner = pygame.Rect(x + 10, y + 10, size - 20, size - 20)
        pygame.draw.rect(canvas, config.COLOR_EXIT_INNER, rect_inner, border_radius=2)

    def _draw_ui(self, canvas, hp, hammers, steps):
        # 繪製底部深色面板
        ui_rect = pygame.Rect(0, self.window_size, self.window_size, self.ui_height)
        pygame.draw.rect(canvas, config.COLOR_UI_BG, ui_rect)

        # 準備文字
        # 使用 Emoji 或簡單符號來增加視覺趣味
        hp_text = f"♥ HP: {hp}/{config.PLAYER_MAX_HP}"
        hammer_text = f"⚒ Hammer: {hammers}"
        step_text = f"👣 Steps: {steps}"

        # 繪製文字 (加上陰影效果)
        padding = 20
        section_width = self.window_size / 3

        labels = [hp_text, hammer_text, step_text]

        for i, text in enumerate(labels):
            text_surf = self.font.render(text, True, config.COLOR_TEXT)
            # 簡單的置中計算
            x_pos = i * section_width + (section_width - text_surf.get_width()) / 2
            y_pos = self.window_size + (self.ui_height - text_surf.get_height()) / 2
            canvas.blit(text_surf, (x_pos, y_pos))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
