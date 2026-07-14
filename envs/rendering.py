import math
import os
import sys

import pygame
import config

# 中文字型直接指定檔案路徑，不走 pygame.font.SysFont：
# SysFont 在 Windows 會呼叫 initsysfonts_win32() 掃描字型登錄檔，
# 一旦 Fonts 這個 key 底下出現非字串的值 (例如 REG_DWORD)，
# pygame 2.6.1 會把 int 丟給 os.path.splitext() 直接拋 TypeError。
# 指定路徑同時也讓兩個平台的字型選擇是確定的。
_WINDIR = os.environ.get("WINDIR", "C:\\Windows")
FONT_FILES = {
    "win32": [
        os.path.join(_WINDIR, "Fonts", "msjh.ttc"),  # 微軟正黑體
        os.path.join(_WINDIR, "Fonts", "msjh.ttf"),
        os.path.join(_WINDIR, "Fonts", "msyh.ttc"),  # 微軟雅黑
        os.path.join(_WINDIR, "Fonts", "mingliu.ttc"),  # 細明體
        os.path.join(_WINDIR, "Fonts", "simsun.ttc"),
    ],
    "darwin": [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ],
}
LINUX_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

# 找不到上面任何一個檔案時才用得到的字型名稱 (SysFont 走登錄檔/字型目錄掃描)
FONT_NAMES = "Microsoft JhengHei,Microsoft YaHei,Hiragino Sans GB,Noto Sans CJK TC"


def load_cjk_font(size, bold=False):
    """依序嘗試各平台的中文字型檔，全部失敗才退回 pygame 內建字型 (不會崩潰，但中文變豆腐框)"""
    for path in FONT_FILES.get(sys.platform, LINUX_FONT_FILES):
        if not os.path.exists(path):
            continue
        try:
            font = pygame.font.Font(path, size)
        except OSError:
            continue  # 字型檔存在但載入失敗 (損毀或格式不支援)
        font.set_bold(bold)
        return font

    try:
        return pygame.font.SysFont(FONT_NAMES, size, bold=bold)
    except Exception:  # noqa: BLE001 - SysFont 在某些 Windows 環境會掃登錄檔掃到炸
        return pygame.font.Font(None, size)

# 特效顏色 (依 AI 動作類型)
FX_COLORS = {
    "wall": (155, 89, 182),  # 紫：蓋牆
    "monster": (231, 76, 60),  # 紅：放怪
    "exit": (241, 196, 15),  # 金：搬出口
    "remove": (46, 204, 113),  # 綠：清除
    "hammer": (46, 204, 113),  # 綠：玩家破牆
    "bump": (120, 130, 150),  # 灰：撞牆但沒有破牆鎚
}


class MazeRenderer:
    def __init__(self, window_size, grid_size, fps):
        self.window_size = window_size
        self.grid_size = grid_size
        self.fps = fps
        self.window = None
        self.clock = None
        self.ui_height = 64
        self.top_height = 40

        # 互動狀態
        self.show_help = False
        self.help_button_rect = None  # 給 main.py 做滑鼠點擊判定

        # 特效時長以秒為單位換算成幀數，改 FPS 時不會讓動畫變快或變慢
        self._fx_frames = self._frames(0.45)
        self._flash_frames = self._frames(0.35)
        self._pulse_frames = self._frames(0.35)
        self._banner_frames = self._frames(2.2)

        # 視覺回饋狀態
        self.effects = []  # [{r, c, ttl, ttl0, color}]
        self.hit_flash = 0  # 受擊紅閃剩餘幀數
        self.banner = None  # {text, color, ttl, ttl0}
        self.ai_pulse = 0  # AI 行動指示燈亮度
        self._last_action_seen = None
        self._player_px = None  # 玩家平滑移動的目前繪製座標
        self._trail = []  # 玩家移動殘影 [(x, y)]

    def _frames(self, seconds):
        return max(1, int(self.fps * seconds))

    def init_window(self):
        if self.window is not None:
            return
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Maze Agent RL - PPO Maze Master")

        self.total_height = self.top_height + self.window_size + self.ui_height
        self.window = pygame.display.set_mode((self.window_size, self.total_height))
        self.font = load_cjk_font(17)
        self.font_small = load_cjk_font(14)
        self.font_big = load_cjk_font(30, bold=True)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # 外部事件 API (由環境呼叫)
    # ------------------------------------------------------------------

    def add_effect(self, kind, r=0, c=0):
        """在格子 (r, c) 加入一個擴散光圈特效；kind='hit' 則為全畫面紅閃"""
        if kind == "hit":
            self.hit_flash = self._flash_frames
            return
        ttl = self._fx_frames
        self.effects.append(
            {"r": r, "c": c, "ttl": ttl, "ttl0": ttl, "color": FX_COLORS.get(kind, (255, 255, 255))}
        )

    def set_banner(self, text, color):
        """顯示回合結束橫幅，會自動淡出"""
        self.banner = {"text": text, "color": color, "ttl": self._banner_frames, "ttl0": self._banner_frames}

    # ------------------------------------------------------------------
    # 主渲染
    # ------------------------------------------------------------------

    def render(
        self,
        maze,
        player_pos,
        exit_pos,
        monsters,
        player_hp,
        player_hammers,
        player_steps,
        current_time,
        last_ai_action,
    ):
        self.init_window()

        canvas = pygame.Surface((self.window_size, self.total_height))
        canvas.fill(config.COLOR_BG)

        cell = self.window_size / self.grid_size
        oy = self.top_height  # 迷宮區域的 y 偏移

        # 1. 格線
        for i in range(self.grid_size + 1):
            pos = i * cell
            pygame.draw.line(canvas, config.COLOR_GRID, (0, oy + pos), (self.window_size, oy + pos), 1)
            pygame.draw.line(canvas, config.COLOR_GRID, (pos, oy), (pos, oy + self.window_size), 1)

        # 2. 牆壁 (maze 只含靜態地形)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if maze[r, c] == config.ID_WALL:
                    self._draw_wall(canvas, c * cell, oy + r * cell, cell)

        # 3. 實體分層繪製：出口 -> 怪物 -> 玩家 (重疊時全部可見)
        self._draw_exit(canvas, exit_pos[1] * cell, oy + exit_pos[0] * cell, cell)
        for i, m in enumerate(monsters):
            self._draw_monster(canvas, m[1] * cell, oy + m[0] * cell, cell, idx=i)
        self._draw_player_smooth(canvas, player_pos, cell, oy)

        # 4. 特效層 (半透明)
        self._draw_effects(canvas, cell, oy)

        # 5. 頂部 AI 狀態列 + 底部資訊列
        self._draw_top_bar(canvas, last_ai_action)
        self._draw_ui(canvas, player_hp, player_hammers, player_steps, current_time)

        # 6. 回合結束橫幅與教學面板
        self._draw_banner(canvas, oy)
        if self.show_help:
            self._draw_help(canvas)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    # 各元件
    # ------------------------------------------------------------------

    def _draw_wall(self, canvas, x, y, size):
        pygame.draw.rect(canvas, config.COLOR_WALL_SIDE, pygame.Rect(x, y, size, size))
        top_rect = pygame.Rect(x + 2, y + 2, size - 4, size - 4)
        pygame.draw.rect(canvas, config.COLOR_WALL_TOP, top_rect, border_radius=3)

    def _draw_shadow(self, canvas, cx, cy, size):
        """實體腳下的柔和橢圓陰影"""
        shadow = pygame.Surface((size, size // 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 70), shadow.get_rect())
        canvas.blit(shadow, (cx - size / 2, cy + size / 4))

    def _draw_player_smooth(self, canvas, player_pos, cell, oy):
        """玩家位置用插值平滑移動 + 殘影，增加手感"""
        target = (player_pos[1] * cell + cell / 2, oy + player_pos[0] * cell + cell / 2)

        if self._player_px is None:
            self._player_px = list(target)
        dx = target[0] - self._player_px[0]
        dy = target[1] - self._player_px[1]
        # 距離太遠 (重生/瞬移) 直接貼齊，避免滑過整張地圖
        if abs(dx) > cell * 3 or abs(dy) > cell * 3:
            self._player_px = list(target)
            self._trail.clear()
        else:
            self._player_px[0] += dx * 0.45
            self._player_px[1] += dy * 0.45

        center = (self._player_px[0], self._player_px[1])
        radius = cell / 2 - 4

        # 移動殘影：移動時累積，停下後逐幀排空 (否則殘影會永遠留在原地)
        moving = abs(dx) + abs(dy) > 0.5
        if moving:
            if not self._trail or (abs(center[0] - self._trail[-1][0]) + abs(center[1] - self._trail[-1][1])) > 1.5:
                self._trail.append(tuple(center))
                if len(self._trail) > 8:
                    self._trail.pop(0)
        elif self._trail:
            self._trail.pop(0)
        if len(self._trail) > 1:
            ghost = pygame.Surface((self.window_size, self.total_height), pygame.SRCALPHA)
            n = len(self._trail)
            for i, (tx, ty) in enumerate(self._trail[:-1]):
                alpha = int(60 * (i + 1) / n)
                r = radius * (0.35 + 0.4 * (i + 1) / n)
                pygame.draw.circle(ghost, (*config.COLOR_PLAYER_BODY, alpha), (tx, ty), r)
            canvas.blit(ghost, (0, 0))

        self._draw_shadow(canvas, center[0], center[1], cell * 0.8)
        pygame.draw.circle(canvas, config.COLOR_PLAYER_BORDER, center, radius)
        pygame.draw.circle(canvas, config.COLOR_PLAYER_BODY, center, radius - 3)
        # 光澤高光
        pygame.draw.circle(
            canvas, (170, 220, 250),
            (center[0] - radius * 0.35, center[1] - radius * 0.35), radius * 0.22,
        )

    def _draw_monster(self, canvas, x, y, size, idx=0):
        # 上下浮動動畫 (每隻相位錯開)
        bob = math.sin(pygame.time.get_ticks() / 250 + idx * 1.7) * size * 0.06
        center = (x + size / 2, y + size / 2 + bob)
        radius = size / 2 - 4

        self._draw_shadow(canvas, x + size / 2, y + size / 2, size * 0.8)
        pygame.draw.circle(canvas, config.COLOR_MONSTER_BODY, center, radius)
        # 頂部亮面
        pygame.draw.circle(canvas, (250, 130, 115), (center[0], center[1] - radius * 0.3), radius * 0.55)
        pygame.draw.circle(canvas, config.COLOR_MONSTER_BODY, (center[0], center[1] - radius * 0.15), radius * 0.6)
        eye_dx = radius / 2.5
        eye_dy = radius / 4
        pygame.draw.circle(canvas, config.COLOR_MONSTER_EYE, (center[0] - eye_dx, center[1] - eye_dy), 3.5)
        pygame.draw.circle(canvas, config.COLOR_MONSTER_EYE, (center[0] + eye_dx, center[1] - eye_dy), 3.5)

    def _draw_exit(self, canvas, x, y, size):
        # 脈動光暈 (傳送門感)
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 320)
        cx, cy = x + size / 2, y + size / 2
        glow = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        for ratio, alpha in ((1.0, 22), (0.8, 34), (0.62, 48)):
            a = int(alpha * (0.6 + 0.4 * pulse))
            pygame.draw.circle(glow, (*config.COLOR_EXIT_OUTER, a), (size, size), size * ratio)
        canvas.blit(glow, (cx - size, cy - size))

        rect_outer = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
        pygame.draw.rect(canvas, config.COLOR_EXIT_OUTER, rect_outer, border_radius=6)
        inset = 10 - 2 * pulse
        rect_inner = pygame.Rect(x + inset, y + inset, size - inset * 2, size - inset * 2)
        pygame.draw.rect(canvas, config.COLOR_EXIT_INNER, rect_inner, border_radius=4)

    def _draw_effects(self, canvas, cell, oy):
        if not self.effects and self.hit_flash <= 0:
            return
        overlay = pygame.Surface((self.window_size, self.total_height), pygame.SRCALPHA)

        for e in self.effects:
            t = 1 - e["ttl"] / e["ttl0"]  # 0 -> 1
            cx = e["c"] * cell + cell / 2
            cy = oy + e["r"] * cell + cell / 2
            radius = cell * (0.3 + 0.55 * t)
            alpha = int(200 * (1 - t))
            pygame.draw.circle(overlay, (*e["color"], alpha), (cx, cy), radius, 3)
            e["ttl"] -= 1
        self.effects = [e for e in self.effects if e["ttl"] > 0]

        if self.hit_flash > 0:
            alpha = int(110 * self.hit_flash / self._flash_frames)
            pygame.draw.rect(
                overlay, (231, 76, 60, alpha),
                pygame.Rect(0, oy, self.window_size, self.window_size), 12,
            )
            self.hit_flash -= 1

        canvas.blit(overlay, (0, 0))

    def _draw_top_bar(self, canvas, last_ai_action):
        pygame.draw.rect(canvas, config.COLOR_UI_BG, pygame.Rect(0, 0, self.window_size, self.top_height))

        # AI 行動指示燈：動作改變時閃亮
        if last_ai_action != self._last_action_seen:
            self._last_action_seen = last_ai_action
            self.ai_pulse = self._pulse_frames
        pulse = self.ai_pulse / self._pulse_frames
        self.ai_pulse = max(0, self.ai_pulse - 1)
        glow = (int(155 + 100 * pulse), int(89 + 100 * pulse), int(182 + 60 * pulse))
        pygame.draw.circle(canvas, glow, (20, self.top_height // 2), 6 + int(3 * pulse))

        title = self.font.render("Maze Master · PPO 強化學習", True, config.COLOR_TEXT)
        canvas.blit(title, (36, (self.top_height - title.get_height()) // 2))

        action = self.font_small.render(f"AI 決策：{last_ai_action}", True, (150, 160, 178))
        canvas.blit(action, (self.window_size - action.get_width() - 14, (self.top_height - action.get_height()) // 2))

        # 紫色點綴分隔線 (Maze Master 主題色)
        pygame.draw.line(canvas, config.COLOR_ACCENT, (0, self.top_height - 2), (self.window_size, self.top_height - 2), 2)

    def _draw_heart(self, canvas, cx, cy, r, color):
        pygame.draw.circle(canvas, color, (int(cx - r / 2), int(cy - r / 4)), int(r / 2) + 1)
        pygame.draw.circle(canvas, color, (int(cx + r / 2), int(cy - r / 4)), int(r / 2) + 1)
        pygame.draw.polygon(
            canvas, color,
            [(cx - r, cy - r / 5), (cx + r, cy - r / 5), (cx, cy + r)],
        )

    def _draw_hammer(self, canvas, x, y, color):
        pygame.draw.rect(canvas, color, pygame.Rect(x, y, 16, 7), border_radius=2)  # 鎚頭
        pygame.draw.rect(canvas, color, pygame.Rect(x + 6, y + 6, 4, 12), border_radius=1)  # 握柄

    def _draw_ui(self, canvas, hp, hammers, steps, current_time):
        y0 = self.top_height + self.window_size
        pygame.draw.rect(canvas, config.COLOR_UI_BG, pygame.Rect(0, y0, self.window_size, self.ui_height))
        pygame.draw.line(canvas, config.COLOR_ACCENT, (0, y0), (self.window_size, y0), 2)
        cy = y0 + self.ui_height // 2

        # HP 愛心
        for i in range(config.PLAYER_MAX_HP):
            color = config.COLOR_MONSTER_BODY if i < hp else (58, 66, 84)
            self._draw_heart(canvas, 26 + i * 28, cy, 10, color)

        # 破牆鎚
        self._draw_hammer(canvas, 120, cy - 9, (241, 196, 15))
        hammer_text = self.font.render(f"× {hammers}", True, config.COLOR_TEXT)
        canvas.blit(hammer_text, (142, cy - hammer_text.get_height() // 2))

        # 心流節奏條：顯示 AI 的目標區間 (TIME_MIN ~ TIME_MAX 步)
        bar_x, bar_w, bar_h = 210, 240, 10
        bar_y = cy + 2
        total = config.TIME_MAX * 1.5  # 超時上限
        label = self.font_small.render(
            f"節奏 {current_time} 步｜AI 目標 {config.TIME_MIN}–{config.TIME_MAX} 步", True, (189, 195, 199)
        )
        canvas.blit(label, (bar_x, bar_y - label.get_height() - 4))

        pygame.draw.rect(canvas, (75, 90, 105), pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=5)
        zone_x = bar_x + bar_w * config.TIME_MIN / total
        zone_w = bar_w * (config.TIME_MAX - config.TIME_MIN) / total
        pygame.draw.rect(canvas, (39, 174, 96), pygame.Rect(zone_x, bar_y, zone_w, bar_h), border_radius=5)
        marker_x = bar_x + bar_w * min(current_time, total) / total
        pygame.draw.rect(canvas, config.COLOR_TEXT, pygame.Rect(marker_x - 2, bar_y - 3, 4, bar_h + 6), border_radius=2)

        # 說明按鈕 "?"
        btn_cx, btn_r = self.window_size - 30, 15
        self.help_button_rect = pygame.Rect(btn_cx - btn_r, cy - btn_r, btn_r * 2, btn_r * 2)
        pygame.draw.circle(canvas, (52, 152, 219), (btn_cx, cy), btn_r)
        qmark = self.font.render("?", True, config.COLOR_TEXT)
        canvas.blit(qmark, (btn_cx - qmark.get_width() // 2, cy - qmark.get_height() // 2))

    def _draw_banner(self, canvas, oy):
        if not self.banner:
            return
        t = self.banner["ttl"] / self.banner["ttl0"]
        alpha = int(230 * min(1.0, t * 3))  # 最後 1/3 淡出

        text = self.font_big.render(self.banner["text"], True, (255, 255, 255))
        pad = 24
        w, h = text.get_width() + pad * 2, text.get_height() + pad
        x = (self.window_size - w) // 2
        y = oy + self.window_size // 2 - h // 2

        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (*self.banner["color"], alpha), panel.get_rect(), border_radius=12)
        panel.blit(text, (pad, pad // 2))
        canvas.blit(panel, (x, y))

        self.banner["ttl"] -= 1
        if self.banner["ttl"] <= 0:
            self.banner = None

    HELP_LINES = [
        "你（藍色圓點）要走到金色出口。",
        "對手「Maze Master」是一個 PPO 強化學習模型，",
        "牠每回合會即時改造迷宮：蓋牆、搬移出口、放出怪物。",
        "（出口不會被搬到你附近——原地等出口送上門是沒用的）",
        "",
        "AI 的訓練目標不是困死你，而是控制你的節奏：",
        "讓你在 50～100 步之間抵達出口（心流區間），",
        "太快或太慢，AI 在訓練時都會被扣分。",
        "",
        "操作方式",
        "   方向鍵　移動（撞牆時自動消耗破牆鎚 ×1）",
        "   空白鍵　暫停 / 繼續　　（暫停時 → 鍵單步）",
        "   H 或 ?　開關本說明",
        "",
        "下方節奏條顯示你的步數是否落在 AI 的綠色目標區間。",
    ]

    def _draw_help(self, canvas):
        overlay = pygame.Surface((self.window_size, self.total_height), pygame.SRCALPHA)
        overlay.fill((20, 25, 35, 200))

        line_h = 26
        pad = 28
        panel_w = self.window_size - 90
        panel_h = pad * 2 + 44 + len(self.HELP_LINES) * line_h
        px = (self.window_size - panel_w) // 2
        py = (self.total_height - panel_h) // 2

        pygame.draw.rect(overlay, (44, 62, 80, 245), pygame.Rect(px, py, panel_w, panel_h), border_radius=14)

        title = self.font_big.render("遊戲說明", True, (241, 196, 15))
        overlay.blit(title, (px + pad, py + pad - 6))

        for i, line in enumerate(self.HELP_LINES):
            surf = self.font.render(line, True, config.COLOR_TEXT)
            overlay.blit(surf, (px + pad, py + pad + 44 + i * line_h))

        canvas.blit(overlay, (0, 0))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
