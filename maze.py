from collections import deque
import numpy as np 
import matplotlib.pyplot as plt
import heapq


maze = np.array([
    [ 1,  3,  0,  1,  5,  1,  0,  3,  1,  1],
    [ 1,  0,  3,  0,  1,  0,  1,  0,  5,  1],
    [ 1,  5,  1, 10,  3,  5,  1,  1,  0,  1],
    [ 0,  1,  0,  5,  1, 10,  3,  1,  1,  1],
    [ 1,  1,  3,  1,  0,  5, 10,  3,  0,  1],
])

start = (0, 0)
goal = (4, 9)

def neighbors(maze, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] #up, down, left, right
    rows,cols = maze.shape 
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1] and maze[r, c] != 0:
            neighbors.append((r, c))
    return neighbors

#print(neighbors(maze, 0, 0))
#print(neighbors(maze, 1, 1))

def bfs(maze, start, goal):
    queue = deque([start])
    came_from = {start: None}
    while queue: 
        curr = queue.popleft()
        if curr == goal:
            return new_path(came_from, start, goal)
        for neighbor in neighbors(maze, curr[0], curr[1]):
            if neighbor not in came_from: 
                queue.append(neighbor)
                came_from[neighbor] = curr
    return None

def new_path(came_from, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

#visualizing maze
def visualize(maze, path, start, goal, title="Maze"):
    rows, cols = maze.shape
    fig, axis = plt.subplots()
    colors = {
        0: [0.0, 0.0, 0.0], # wall -black
        1: [0.9, 0.9, 0.9], # cost 1 - grey
        3: [0.7, 0.85, 0.7],  # cost 3 - light green
        5: [0.6, 0.6, 0.85],   # cost 5 - light blue
        10: [0.85, 0.6, 0.6] # cost 10 - light red
    }
    display = np.zeros((rows, cols, 3))
    for x in range(rows):
        for y in range(cols):
            display[x, y] = colors.get(maze[x, y], [0.0, 0.0, 0.0])
    for x,y in path:
        display[x, y] = [0.0, 0.6, 0.6] #path color 
    axis.imshow(display)

    for x in range(rows):
        for y in range(cols):
            if maze[x,y] != 0:
                axis.text(y, x, str(maze[x,y]), ha='center', va='center', color='black')
        axis.text(start[1], start[0], 'S', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
        axis.text(goal[1], goal[0], 'G', ha='center', va='center', color='red', fontsize=12, fontweight='bold')

    axis.grid(which='minor', color='gray', linestyle='-', linewidth=1.5)
    axis.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axis.set_title(title)
    plt.tight_layout()
    plt.show()



def ucs(maze, start, goal):
    # priority queue of (cost, node)
    queue = []
    heapq.heappush(queue, (0, start))
    came_from = {start: None}
    visited_cost = {start: 0}

    while queue:
        curr_cost, curr = heapq.heappop(queue)
        if curr == goal:
            return new_path(came_from, start, goal)

        for neighbor in neighbors(maze, curr[0], curr[1]):
            new_cost = curr_cost + maze[neighbor[0], neighbor[1]]

            if neighbor not in visited_cost or new_cost < visited_cost[neighbor]:
                visited_cost[neighbor] = new_cost
                came_from[neighbor] = curr
                heapq.heappush(queue, (new_cost, neighbor))
    return None

bfs_path = bfs(maze, start, goal)
ucs_path = ucs(maze, start, goal)

visualize(maze, bfs_path, start, goal, title="BFS Path")
visualize(maze, ucs_path, start, goal, title="UCS Path")