import numpy as np
import config


class MazeGenerator:
    @staticmethod
    def generate(grid_size, rng):
        """
        生成迷宮並返回 numpy 陣列
        :param grid_size: 迷宮大小
        :param rng: numpy 的 random generator 實例
        :return: maze (numpy array)
        """
        maze = np.zeros((grid_size, grid_size), dtype=np.int8)

        # 1. 初始化：全填滿牆壁
        maze.fill(config.ID_WALL)

        # 起點設為 (0, 0)
        start_x, start_y = 0, 0
        maze[start_x, start_y] = config.ID_EMPTY

        # 2. DFS 生成完美迷宮 (確保連通性)
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]

            # 尋找周圍距離為 2 的未訪問鄰居 (跨過一面牆)
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if maze[nx, ny] == config.ID_WALL:
                        neighbors.append((nx, ny, dx // 2, dy // 2))

            if neighbors:
                # 隨機選一個鄰居
                nx, ny, wx, wy = rng.choice(neighbors)
                # 打通中間的牆
                maze[x + wx, y + wy] = config.ID_EMPTY
                # 標記鄰居為通路
                maze[nx, ny] = config.ID_EMPTY
                stack.append((nx, ny))
            else:
                stack.pop()

        # 3. 隨機移除牆壁以製造多條路徑 (Loops)
        loop_probability = 0.15

        for x in range(1, grid_size - 1):
            for y in range(1, grid_size - 1):
                if maze[x, y] == config.ID_WALL:
                    if rng.random() < loop_probability:
                        maze[x, y] = config.ID_EMPTY

        # 確保出口附近是空的
        maze[grid_size - 1, grid_size - 1] = config.ID_EMPTY
        maze[grid_size - 2, grid_size - 1] = config.ID_EMPTY
        maze[grid_size - 1, grid_size - 2] = config.ID_EMPTY

        return maze
