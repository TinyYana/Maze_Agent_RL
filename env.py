import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import heapq

# --- A* 演算法實作區 (Bot 的大腦) ---
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f

def astar_path(maze, start, end):
    """
    輸入: maze(二維陣列, 0是路, 1是牆), start(座標), end(座標)
    輸出: 路徑座標列表 [(x,y), (x,y)...] 或 None (若無路)
    """
    # 檢查起點或終點是否被堵住
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return None

    start_node = Node(None, tuple(start))
    end_node = Node(None, tuple(end))

    open_list = []
    closed_list = set() # 使用 set 加速搜尋

    heapq.heappush(open_list, start_node)

    # 設定最大搜尋次數防止卡死 (Optional)
    max_iterations = 5000
    count = 0

    while open_list:
        count += 1
        if count > max_iterations:
            return None # 搜尋超時

        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # 找到終點
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # 反轉路徑，從起點開始

        # 產生子節點 (上下左右)
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # 左右上下
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # 檢查邊界
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[0]) -1) or node_position[1] < 0:
                continue

            # 檢查是否為牆壁 (1: Wall, 4: Monster 視為障礙物? 論文中玩家碰到怪會扣血但可能穿過，這裡暫時視為障礙以簡化)
            # 這裡假設 Bot 聰明到會避開怪和牆 (1 和 4)
            cell_value = maze[node_position[0]][node_position[1]]
            if cell_value == 1: 
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:
            if child.position in closed_list:
                continue

            child.g = current_node.g + 1
            # H: 曼哈頓距離
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h

            # 檢查是否已在 open_list 中且有更好的路徑 (簡化版略過此檢查，直接推入 heap 效能通常可接受)
            heapq.heappush(open_list, child)

    return None


# --- 環境類別 (你的程式碼 + Bot 整合) ---
class MazeMasterEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MazeMasterEnv, self).__init__()

        self.grid_size = 15
        self.cell_size = 40
        
        self.ID_EMPTY = 0
        self.ID_WALL = 1
        self.ID_PLAYER = 2
        self.ID_EXIT = 3
        self.ID_MONSTER = 4

        # 動作: 0:No-op, 1:改出口(略), 2:加牆, 3:刪牆, 4:放怪
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int8)

        self.render_mode = render_mode
        self.window = None
        self.clock = None # 控制 FPS

        self.state = None
        self.player_pos = None
        self.exit_pos = None
        
        # 記錄 Bot 的路徑用於視覺化 (Optional)
        self.current_path = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # 圍牆
        self.state[0, :] = self.ID_WALL
        self.state[-1, :] = self.ID_WALL
        self.state[:, 0] = self.ID_WALL
        self.state[:, -1] = self.ID_WALL

        self.player_pos = [1, 1]
        self.exit_pos = [self.grid_size - 2, self.grid_size - 2]

        self.state[self.player_pos[0], self.player_pos[1]] = self.ID_PLAYER
        self.state[self.exit_pos[0], self.exit_pos[1]] = self.ID_EXIT
        
        return self.state, {}

    def step(self, action):
        """ 
        1. Agent (迷宮之主) 動作 
        2. Bot (玩家) 思考並移動
        """
        # --- 1. 迷宮之主的回合 ---
        # 簡化：隨機找一個空位執行動作 (或是針對玩家附近)
        target_pos = self._get_random_pos_near_player()
        
        if target_pos:
            r, c = target_pos
            # 確保不要改到玩家、出口或邊界
            if (r, c) != tuple(self.player_pos) and (r, c) != tuple(self.exit_pos):
                if action == 2:   # 加牆
                    self.state[r, c] = self.ID_WALL
                elif action == 3: # 刪牆
                    if self.state[r, c] == self.ID_WALL:
                        self.state[r, c] = self.ID_EMPTY
                elif action == 4: # 放怪
                    self.state[r, c] = self.ID_MONSTER

        # --- 2. 玩家 (Bot) 的回合 ---
        # 傳入目前的地圖狀態給 A*
        # 注意：A* 需要知道哪裡是牆壁。我們把 Monster 也視為障礙物傳入
        path = astar_path(self.state, self.player_pos, self.exit_pos)
        self.current_path = path # 存起來畫圖用

        reward = 0
        terminated = False
        
        if path and len(path) > 1:
            # path[0] 是目前位置，path[1] 是下一步
            next_step = path[1]
            
            # 更新地圖上的舊位置
            self.state[self.player_pos[0], self.player_pos[1]] = self.ID_EMPTY
            
            # 移動玩家
            self.player_pos = list(next_step)
            self.state[self.player_pos[0], self.player_pos[1]] = self.ID_PLAYER
            
            # 獎勵設計 (參考論文): 
            # 如果玩家能動，給 Agent 一點點負獎勵 (因為 Agent 目標可能是難度調整，這比較複雜，暫時設為 0)
            # 這裡假設 Agent 的目標是「阻擋玩家但不要死路」
            pass
        else:
            # 無路可走！這對 Agent 來說可能是好事 (難度高) 也可能是壞事 (死局)
            # 這裡暫時不懲罰
            pass

        # --- 3. 判定 ---
        if self.player_pos == self.exit_pos:
            terminated = True
            reward = -10 # 玩家贏了，迷宮之主輸了 (假設這是對抗遊戲)

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, False, {}

    def _get_random_pos_near_player(self):
        # 簡單實作：全圖隨機選一點，讓遊戲變化大一點
        r = np.random.randint(1, self.grid_size - 1)
        c = np.random.randint(1, self.grid_size - 1)
        return (r, c)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Maze Master: RL Agent vs A* Bot")
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill((240, 240, 240))

        # 繪製路徑預覽 (黃色線條) - 這是 Debug 好幫手
        if self.current_path and len(self.current_path) > 1:
            for i in range(len(self.current_path) - 1):
                start = (self.current_path[i][1] * self.cell_size + self.cell_size//2, 
                         self.current_path[i][0] * self.cell_size + self.cell_size//2)
                end = (self.current_path[i+1][1] * self.cell_size + self.cell_size//2, 
                       self.current_path[i+1][0] * self.cell_size + self.cell_size//2)
                pygame.draw.line(canvas, (255, 200, 0), start, end, 5)

        # 繪製格子
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_value = self.state[r, c]
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                
                if cell_value == self.ID_WALL:
                    pygame.draw.rect(canvas, (50, 50, 50), rect) # 深灰牆
                    pygame.draw.rect(canvas, (100, 100, 100), rect, 2) # 邊框
                elif cell_value == self.ID_PLAYER:
                    pygame.draw.circle(canvas, (0, 100, 255), rect.center, self.cell_size // 3) # 藍色圓形玩家
                elif cell_value == self.ID_EXIT:
                    pygame.draw.rect(canvas, (0, 200, 0), rect) # 綠色出口
                elif cell_value == self.ID_MONSTER:
                    pygame.draw.circle(canvas, (200, 0, 0), rect.center, self.cell_size // 4) # 紅色小怪
                
                # 畫淡色網格線
                pygame.draw.rect(canvas, (220, 220, 220), rect, 1)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        
        # 限制 FPS，不然跑太快看不清楚
        self.clock.tick(5) # 每秒 5 幀

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# --- 主程式入口 (Main Loop) ---
if __name__ == "__main__":
    # 建立環境，開啟人眼渲染模式
    env = MazeMasterEnv(render_mode="human")
    obs, info = env.reset()

    print("遊戲開始！紅色牆壁會隨機出現，藍點(玩家)會試圖避開並前往綠區(出口)。")
    
    running = True
    while running:
        # 這裡模擬強化學習 Agent 的決策
        # 目前我們先用「隨機動作」來測試環境是否運作正常
        # action = 0 (不動), 2 (加牆), 3 (刪牆), 4 (放怪)
        
        # 為了不讓牆壁太多導致無法走，我們增加 No-op (0) 的機率
        action = np.random.choice([0, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.1])
        
        # 執行一步
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print("玩家到達出口！回合結束。")
            obs, info = env.reset() # 重置遊戲

        # 處理 Pygame 關閉視窗事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()