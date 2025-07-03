import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def heuristic(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def a_star_search(grid, start, goal):
    """Perform the A* search algorithm."""
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            data.reverse()
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor[0]][neighbor[1]] == 0:
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False

def animate_path(grid, path):
   
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.imshow(grid, cmap='Greys', origin='upper')
    plot, = ax.plot([], [], 'ro-', linewidth=2, markersize=10)  

    def update(i):
        plot.set_data([p[1] for p in path[:i+1]], [p[0] for p in path[:i+1]])
        return plot,

    ani = FuncAnimation(fig, update, frames=len(path), interval=300, blit=True)
    plt.grid()
    
    ani.save('/content/path_animation.mp4', writer='ffmpeg', fps=2)
    plt.close()
    return ani


grid = np.zeros((5, 10))
obstacles = [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
for obs in obstacles:
    grid[obs] = 1  #

start = (2, 0)
goal = (2, 9)  

path = a_star_search(grid, start, goal)
animate_path(grid, path)


display(HTML("<video controls><source src='/content/path_animation.mp4' type='video/mp4'></video>"))