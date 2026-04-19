def method(grid):
    import numpy as np
    out = np.zeros_like(grid)
    # Find the 3x3 meta-block containing non-zero cells
    pattern = None
    for br in range(3):
        for bc in range(3):
            block = grid[br*3:(br+1)*3, bc*3:(bc+1)*3]
            if np.any(block != 0):
                pattern = block.copy()
                break
        if pattern is not None:
            break
    if pattern is None:
        return out
    # Place pattern at meta-block positions corresponding to non-zero cells
    for r in range(3):
        for c in range(3):
            if pattern[r, c] != 0:
                out[r*3:(r+1)*3, c*3:(c+1)*3] = pattern
    return out
