import numpy
import random
from maze import bfs, ucs, greedy, astar, mcts
import csv
import time

def generate_maze(size, density, weight):
    '''
    generates a random weighted terrain maze of a given size.
    each cell is randomly assigned a wall based on density or a terrain cost from a weight list. 
    '''
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
    '''
    checks if a maze is solvable using bfs.
    maze is solvable means is there a valid path from start to goal.
    '''
    path = bfs(maze, start, goal)
    return path is not None

def make_dataset(num_mazes):
    '''creates dataset of num_mazes randomly generated mazes.
    for each maze that is solvable, run all 5 algorithms and records path cost and runtime. 
    best algorithm is determined by lowest path cost; runtime is used as a tiebreaker
    '''
    dataset = []
    while len(dataset) < num_mazes:
        size = random.choice([10, 20])
        density = random.uniform(0.2, 0.5)
        weight =  [1, 3, 5, 10]
        maze = generate_maze(size, density, weight)
        start = (0, 0)
        goal = (size-1, size-1)

        if not solvable(maze, (0, 0), (size-1, size-1)):
            continue
        #terrain_variance = float(numpy.var(maze[maze != 0]))

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
        mcts_result, mcts_found = mcts(maze, start, goal, iterations=1000)
        end_time = time.time()
        results['mcts'] = {
            'path_cost': sum(maze[x,y] for x,y in mcts_result) if mcts_found and mcts_result else None,
            'time': end_time - start_time
        }
        best_algo = min(results, key=lambda x: (
        results[x]['path_cost'] if results[x]['path_cost'] is not None else float('inf'),
        results[x]['time']
    ))

        dataset.append({
            'size': size,
            'density': density,
            #'terrain_variance': terrain_variance,
            'results': results,
            'best algorithm': best_algo
        })
    return dataset

dataset = make_dataset(500)
with open ('dataset.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=[ 'size', 'density','bfs_cost', 'bfs_time', 'ucs_cost', 'ucs_time',
'greedy_cost', 'greedy_time', 'astar_cost', 'astar_time',
'mcts_cost', 'mcts_time', 'best_algorithm'])
    writer.writeheader()
    for data in dataset:
        writer.writerow({
            'size': data['size'],
            'density': data['density'],
            #'terrain_variance': data['terrain_variance'],
            'bfs_cost': data['results']['bfs']['path_cost'],
            'bfs_time': data['results']['bfs']['time'],
            'ucs_cost': data['results']['ucs']['path_cost'],
            'ucs_time': data['results']['ucs']['time'],
            'greedy_cost': data['results']['greedy']['path_cost'],
            'greedy_time': data['results']['greedy']['time'],
            'astar_cost': data['results']['astar']['path_cost'],
            'astar_time': data['results']['astar']['time'],
            'mcts_cost': data['results']['mcts']['path_cost'],
            'mcts_time': data['results']['mcts']['time'],
            'best_algorithm': data['best algorithm']
        })
for row in dataset: 
    print(row)