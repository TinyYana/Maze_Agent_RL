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

    def init_window(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            # 增加高度給 UI
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + 50)
            )
            self.font = pygame.font.SysFont("Arial", 24)
            self.clock = pygame.time.Clock()

    def render(self, maze, player_hp, player_hammers, player_steps):
        self.init_window()

        canvas = pygame.Surface((self.window_size, self.window_size + 50))
        canvas.fill(config.COLOR_WHITE)

        pix_square_size = self.window_size / self.grid_size

        # 繪製迷宮
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * pix_square_size,
                    x * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                )
                cell_value = maze[x, y]

                if cell_value == config.ID_WALL:
                    pygame.draw.rect(canvas, config.COLOR_BLACK, rect)
                elif cell_value == config.ID_PLAYER:
                    pygame.draw.rect(canvas, config.COLOR_BLUE, rect)
                elif cell_value == config.ID_EXIT:
                    pygame.draw.rect(canvas, config.COLOR_GREEN, rect)
                elif cell_value == config.ID_MONSTER:
                    pygame.draw.rect(canvas, config.COLOR_RED, rect)

        # 繪製格線
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        # 繪製 UI
        self._draw_ui(canvas, player_hp, player_hammers, player_steps)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)

    def _draw_ui(self, canvas, hp, hammers, steps):
        # 血量
        hp_text = f"HP: {hp} / {config.PLAYER_MAX_HP}"
        text_surface = self.font.render(hp_text, True, (0, 0, 0))
        canvas.blit(text_surface, (10, self.window_size + 10))

        # 破牆工具
        hammer_text = f"Hammer: {hammers}"
        hammer_surface = self.font.render(hammer_text, True, (0, 0, 0))
        canvas.blit(hammer_surface, (150, self.window_size + 10))

        # 步數
        step_text = f"Steps: {steps}"
        step_surface = self.font.render(step_text, True, (0, 0, 0))
        canvas.blit(step_surface, (300, self.window_size + 10))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
