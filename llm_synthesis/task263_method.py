def method(grid):
    import numpy as np
    h, w = grid.shape
    if h > w:
        blocks = [grid[i*3:(i+1)*3, :] for i in range(h//3)]
    else:
        blocks = [grid[:, i*3:(i+1)*3] for i in range(w//3)]
    
    patterns = [(b != 0).astype(int) for b in blocks]
    
    for i, p in enumerate(patterns):
        matches = sum(1 for q in patterns if np.array_equal(p, q))
        if matches == 1:
            return blocks[i]
    
    return blocks[0]
