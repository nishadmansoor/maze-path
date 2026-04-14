## Choosing the Path: Maze Path Algorithm Selection
This project predicts which search algorithm performs best on a given weighted maze configuration by learning from a dataset of 500 randomly generated mazes. 

Algorithms: 
- **BFS** - Explores level by level, guarantees fewest step path, but ignores terrain costs
- **UCS** - Uses priority queue ordered by a cumulative cost, guarantees lowest cost
- **Greedy** - Moves towards cell closest to goal, uses Manhattan distance heuristic 
- **A\*** - Combines cumulative path cost & heuristic, guarantees optimal path efficiently
- **MCTS** - Repeated random simulations to estimate which directions are good, explores without systematic search

## Project Structure
```
├── maze.py              # Maze environment, visualizes all 5 algorithms 
├── generate_maze.py     # Generates random mazes & dataset
├── model.ipynb          # Random Forest classifier, evaluation
└── dataset.csv          # Training dataset of 500 mazes
```
 ## How to Run
 
** Run search algorithms on **
```bash
python maze.py
```
 
**Generate a dataset:**
```bash
python generate_maze.py
```
 
**Train and evaluate the model:**

Open `model.ipynb` in Jupyter and run all cells.
 
## Evaluation
The model is evaluated on a 20% test set using:
 
* Overall accuracy
* Per-class precision, recall, and F1-score
* Confusion matrix
* Feature importance scores

## Results
* **Accuracy:** 72% (vs. ~53% majority-class baseline)
* **Best predicted classes:** UCS (F1: 0.80), A* (F1: 0.66)
* **Most predictive feature:** `astar_time` (importance: 0.185)
* **Least predictive feature:** `size` (importance: 0.002)
