def method(grid):
    import numpy as np
    g = np.array(grid)
    # Find the 8
    pos = np.argwhere(g == 8)
    r, c = pos[0]
    # Extract 3x3 around it
    block = g[r-1:r+2, c-1:c+2].copy()
    # Find the non-zero, non-8 color
    colors = np.unique(block)
    color = [x for x in colors if x != 0 and x != 8][0]
    block[block == 8] = color
    return block
