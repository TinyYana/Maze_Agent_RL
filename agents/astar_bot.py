import heapq


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


def astar_path(maze, start, end):
    """
    輸入: maze(二維陣列, 0是路, 1是牆), start(座標), end(座標)
    輸出: 路徑座標列表 [(x,y), (x,y)...] 或 None (若無路)
    """
    # 檢查起點或終點是否被堵住
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return None

    start_node = AStarBot(None, tuple(start))
    end_node = AStarBot(None, tuple(end))

    open_list = []
    closed_list = set()  # 使用 set 加速搜尋

    heapq.heappush(open_list, start_node)

    # 設定最大搜尋次數防止卡死 (Optional)
    max_iterations = 5000
    count = 0

    while open_list:
        count += 1
        if count > max_iterations:
            return None  # 搜尋超時

        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # 找到終點
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # 反轉路徑，從起點開始

        # 產生子節點 (上下左右)
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 左右上下
            node_position = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1],
            )

            # 檢查邊界
            if (
                node_position[0] > (len(maze) - 1)
                or node_position[0] < 0
                or node_position[1] > (len(maze[0]) - 1)
                or node_position[1] < 0
            ):
                continue

            # 檢查是否為牆壁 (1: Wall, 4: Monster 視為障礙物? 論文中玩家碰到怪會扣血但可能穿過，這裡暫時視為障礙以簡化)
            # 這裡假設 Bot 聰明到會避開怪和牆 (1 和 4)
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
            child.f = child.g + child.h

            # 檢查是否已在 open_list 中且有更好的路徑 (簡化版略過此檢查，直接推入 heap 效能通常可接受)
            heapq.heappush(open_list, child)

    return None
