"""BFS 距離場：取代熱路徑上的逐次 A*。

採樣端的 CPU 熱點不是神經網路，是每步 5~8 次的純 Python A*
(怪物追蹤 x3、距離塑形 x2、堵路檢查 x1~2)，而且舊 A* 每個節點都
配置一個 AStarBot 物件。這裡改成「一次 BFS、多次 O(1) 查表」：

    field = env.dist_from(target)   # (地形版本, 目標) 沒變就直接回快取
    d = field[x, y]                 # 到 target 的最短步數；UNREACHABLE = 不通

單位步長網格上 BFS 與 A* 的最短距離完全一致；唯一的行為差異是
並列最短路的 tie-break (怪物在等長路徑間的選擇順序)。
帶隨機個性的 A* bot (astar_path randomness>0) 不在此範圍，維持原樣。
"""
from collections import deque

import numpy as np

import config

UNREACHABLE = 1 << 20  # 遠大於任何路徑長，且 +1 不會溢位 int32


def dist_field(maze, target):
    """從 target 出發、繞開牆的 BFS 距離場 (int32 陣列，不可達 = UNREACHABLE)"""
    n = len(maze)
    field = np.full((n, n), UNREACHABLE, dtype=np.int32)
    tx, ty = int(target[0]), int(target[1])
    if maze[tx][ty] == config.ID_WALL:
        return field
    field[tx, ty] = 0
    queue = deque([(tx, ty)])
    while queue:
        x, y = queue.popleft()
        d = field[x, y] + 1
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = x + dx, y + dy
            if 0 <= a < n and 0 <= b < n and field[a, b] == UNREACHABLE and maze[a][b] != config.ID_WALL:
                field[a, b] = d
                queue.append((a, b))
    return field
