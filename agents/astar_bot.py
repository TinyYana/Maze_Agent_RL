import heapq
import random  # 新增 random 模組


class AStarBot:
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


def astar_path(maze, start, end, randomness=0.0):
    """
    輸入: maze, start, end
    參數: randomness (float) - 0.0 為完美 A*, 數值越大(如 0.5, 1.0) 越容易走歪
    輸出: 路徑座標列表
    """
    # 檢查起點或終點是否被堵住
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return None

    start_node = AStarBot(None, tuple(start))
    end_node = AStarBot(None, tuple(end))

    open_list = []
    closed_list = set()

    heapq.heappush(open_list, start_node)

    max_iterations = 5000
    count = 0

    while open_list:
        count += 1
        if count > max_iterations:
            return None

        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # 找到終點
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1],
            )

            if (
                node_position[0] > (len(maze) - 1)
                or node_position[0] < 0
                or node_position[1] > (len(maze[0]) - 1)
                or node_position[1] < 0
            ):
                continue

            cell_value = maze[node_position[0]][node_position[1]]
            if cell_value == 1:
                continue

            new_node = AStarBot(current_node, node_position)
            children.append(new_node)

        for child in children:
            if child.position in closed_list:
                continue

            child.g = current_node.g + 1

            # H: 曼哈頓距離
            child.h = abs(child.position[0] - end_node.position[0]) + abs(
                child.position[1] - end_node.position[1]
            )

            # --- 修改重點：加入隨機雜訊 ---
            # 如果 randomness > 0，我們會隨機干擾 H 值
            # 這會讓 Bot 誤以為某些較遠的路徑其實比較近
            noise = 0
            if randomness > 0:
                # 產生 -randomness 到 +randomness 之間的雜訊比例
                noise = child.h * random.uniform(-randomness, randomness)

            child.f = child.g + child.h + noise
            # ---------------------------

            heapq.heappush(open_list, child)

    return None
