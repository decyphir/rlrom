import minigrid
import numpy as np

def _door_pos(env):
        grid = env.unwrapped.grid
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell is not None and cell.type == 'door':
                    return np.array([i, j])
        return None
    
def _ball_pos(env):
    grid = env.unwrapped.grid
    for i in range(grid.width):
        for j in range(grid.height):
            cell = grid.get(i, j)
            if cell is not None and cell.type == 'ball':
                return np.array([i, j])
    return env.unwrapped.agent_pos +np.array([1, 0])

def distance(env):
    return np.linalg.norm(_door_pos(env) - _ball_pos(env))

def _door_open(env):
    grid = env.unwrapped.grid
    for i in range(grid.width):
        for j in range(grid.height):
            cell = grid.get(i, j)
            if cell is not None and cell.type == 'door':
                if cell.is_open:
                    return 1
                else: return 0
    return None