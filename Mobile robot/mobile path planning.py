import matplotlib.pyplot as plt
import numpy as np

def plot_graph(nodes, edges, path=None):
    fig, ax = plt.subplots()
    for (x, y) in nodes:
        ax.plot(x, y, 'o', color='black')
    for (start, end) in edges:
        line = plt.Line2D((nodes[start][0], nodes[end][0]), (nodes[start][1], nodes[end][1]), lw=2)
        ax.add_line(line)
    if path:
        for i in range(len(path)-1):
            line = plt.Line2D((nodes[path[i]][0], nodes[path[i+1]][0]), (nodes[path[i]][1], nodes[path[i+1]][1]), lw=2, color='red')
            ax.add_line(line)
    plt.show()


nodes = [(1, 2), (4, 4), (5, 6), (7, 8)] 
edges = [(0, 1), (1, 2), (2, 3)]  


path = [0, 1, 2, 3]


plot_graph(nodes, edges, path)
