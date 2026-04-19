import numpy as np

def method(grid):
    out = grid.copy()
    # Find the 5s
    fives = np.argwhere(grid == 5)
    # Find the inner shape color (non-zero, non-5)
    mask = (grid != 0) & (grid != 5)
    coords = np.argwhere(mask)
    color = grid[coords[0, 0], coords[0, 1]]
    
    # Determine outer rectangle bounds from the 5s
    top = fives[:, 0].min() + 1
    bottom = fives[:, 0].max() - 1
    left = fives[:, 1].min() + 1
    right = fives[:, 1].max() - 1
    
    # Draw hollow rectangle
    out[top, left:right+1] = color
    out[bottom, left:right+1] = color
    out[top:bottom+1, left] = color
    out[top:bottom+1, right] = color
    
    return out
