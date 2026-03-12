from collections import deque
import numpy as np 
import matplotlib.pyplot as plt
import math
import random
import heapq


maze = np.array([
    [ 1,  3,  0,  1,  1,  1,  0,  1,  1,  1],
    [ 1,  0,  1,  0, 10, 10, 10,  0,  1,  1],
    [ 1,  1,  1,  0, 10, 10, 10,  1,  0,  1],
    [ 0,  1,  1,  1, 10, 10,  1,  1,  1,  1],
    [ 1,  1,  0,  1,  1,  1,  1,  1,  0,  1],
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

def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def greedy(maze, start, goal):
    queue = [(heuristic(start, goal), start)]
    came_from = {start: None}

    while queue:
        _, curr = heapq.heappop(queue)
        if curr == goal:
            return  new_path(came_from, start, goal)
        for neighbor in neighbors(maze, curr[0], curr[1]):
            if neighbor not in came_from:
                came_from[neighbor] = curr
                heapq.heappush(queue, (heuristic(neighbor, goal), neighbor))
    return None

#priority = new_cost + heuristic(neighbor, goal)
def astar(maze, start, goal):
    queue = [(heuristic(start, goal), start)]
    came_from = {start: None}
    visit_cost = {start: 0}

    while queue:
        _, curr = heapq.heappop(queue)
        if curr == goal:
            return new_path(came_from, start, goal)
        for neighbor in neighbors(maze, curr[0], curr[1]):
            new_cost = visit_cost[curr] + maze[neighbor]

            if neighbor not in visit_cost or new_cost < visit_cost[neighbor]:
                visit_cost[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                came_from[neighbor] = curr
                heapq.heappush(queue, (priority, neighbor))
    return None

class Node:
    def __init__(self, cell, parent=None):
        self.cell = cell
        self.parent = parent
        self.wins = 0
        self.visits = 0
        self.children = []
#ucb1 function - (wins / visits) + C * sqrt(log(parent_visits) / visits)

def ucb1(child, parent):
    if child.visits == 0:
        return float('inf')
    c = math.sqrt(2)
    return child.wins / child.visits + c * math.sqrt(math.log(parent.visits) / child.visits)

def rollout(maze, node, goal, max_steps=100):
    current = node.cell
    visited = {current}
    steps = 0
    while current != goal and steps < max_steps:
        options = []
        for n in neighbors(maze, current[0], current[1]):
            if n not in visited:
                options.append(n)
        if not options: 
            return 0
        current = random.choice(options)
        visited.add(current)
        steps += 1
    if current == goal: 
        return 1
    else: 
        return 0
def backprop(node, result):
    while node: 
        node.visits += 1
        node.wins += result
        node = node.parent

def mcts(maze, start, goal, iterations):
    root_node = Node(start)
    for i in range(iterations):
        node = root_node
        while node.cell != goal:
            not_visited = [n for n in neighbors(maze, node.cell[0], node.cell[1]) 
                          if n not in [child.cell for child in node.children]]
            if not_visited:
                neighbor = random.choice(not_visited)
                node.children.append(Node(neighbor, parent=node))
                node = node.children[-1]
            else:
                node = max(node.children, key=lambda c: ucb1(c, node))                   

        result = rollout(maze, node, goal)
        backprop(node, result)
        
    path = []
    node = root_node
    while node: 
        path.append(node.cell)
        if node.cell == goal:
            break
        if not node.children: 
            break
        node = max(node.children, key=lambda c: c.visits)
    return path

bfs_path = bfs(maze, start, goal)
ucs_path = ucs(maze, start, goal)
greedy_path = greedy(maze, start, goal)
astar_path = astar(maze, start, goal)
mcts_path = mcts(maze, start, goal, iterations=2000)

visualize(maze, bfs_path, start, goal, title="BFS Path")
visualize(maze, ucs_path, start, goal, title="UCS Path")
visualize(maze, greedy_path, start, goal, title="Greedy Path")
visualize(maze, astar_path, start, goal, title="A* Path")
visualize(maze, mcts_path, start, goal, title="MCTS Path")