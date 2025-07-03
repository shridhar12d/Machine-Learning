import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq


def create_grid_and_obstacles():
    grid = np.zeros((10, 10), dtype=int)
    grid[1:9, 4] = 1  
    grid[5, 5:9] = 1  
    return grid

# A* algorithm for pathfinding
def astar(grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue
            else:
                continue

            tentative_g_score = gscore[current] + 1
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + np.sqrt((neighbor[0] - goal[0]) ** 2 + (neighbor[1] - goal[1]) ** 2)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False


start, goal = (0, 0), (9, 9)
grid = create_grid_and_obstacles()
path = astar(grid, start, goal)


fig, ax = plt.subplots()
ax.imshow(grid, cmap='gray', interpolation='none', alpha=0.8)
robot_dot, = ax.plot([], [], 'ro', label='Robot')
path_line, = ax.plot([], [], 'c-', linewidth=2, label='Path')
ax.legend()


def init():
    path_x, path_y = zip(*path)
    path_line.set_data(path_y, path_x)
    return path_line, robot_dot


def animate(i):
    if i < len(path):
        x, y = path[i]
        robot_dot.set_data(y, x)
    return robot_dot, path_line


ani = FuncAnimation(fig, animate, init_func=init, frames=len(path), interval=200, blit=True)


gif_path = 'robot_navigation.gif'
ani.save(gif_path, writer='imagemagick', fps=5)

plt.show()
