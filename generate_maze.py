import numpy
import random
from maze import bfs, ucs, greedy, astar, mcts
import csv
import time

def generate_maze(size, density, weight):
    grid = numpy.ones((size,size), dtype=int)
    for i in range(size): #rows
        for j in range(size): #cols 
            if random.random() < density:
                grid[i, j] = 0 #wall
            else:
                grid[i, j] = random.choice(weight) #random weight for path
    grid[0, 0] = 1 #start
    grid[size-1, size-1] = 1 #end
    return grid

def solvable(maze, start, goal):
    path = bfs(maze, start, goal)
    return path is not None

def make_dataset(num_mazes):
    dataset = []
    while len(dataset) < num_mazes:
        size = random.choice([10, 20, 40])
        density = random.uniform(0.2, 0.5)
        weight =  [1, 3, 5, 10]
        maze = generate_maze(size, density, weight)
        start = (0, 0)
        goal = (size-1, size-1)

        if not solvable(maze, (0, 0), (size-1, size-1)):
            continue

        results = {}
        algos = {'bfs': bfs, 'ucs': ucs, 'greedy': greedy, 'astar': astar}
        for name, algo in algos.items():
            start_time = time.time()
            path = algo(maze, start, goal)
            end_time = time.time()
            results[name] = {
                'path_cost': sum(maze[x, y] for x, y in path) if path else None,
                'time': end_time - start_time}

        #mcts, separate due to iterations
        start_time = time.time()
        mcts_path = mcts(maze, start, goal, iterations=2000)
        end_time = time.time()
        results['mcts'] = {
            'path_cost': sum(maze[x, y] for x, y in mcts_path) if mcts_path else None,
            'time': end_time - start_time}
        dataset.append(results)
    return dataset

dataset = make_dataset(5)
for row in dataset: 
    print(row)