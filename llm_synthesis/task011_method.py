import numpy as np

def method(grid):
    g = np.array(grid)
    out = g.copy()
    
    # Block top-left positions
    positions = [0, 4, 8]
    
    # Find special block (the one missing digit 8, i.e., 4 non-zero cells)
    special = None
    for bi, r in enumerate(positions):
        for bj, c in enumerate(positions):
            block = g[r:r+3, c:c+3]
            nonzero = block[block != 0]
            if len(nonzero) == 4 and 8 not in nonzero:
                special = (bi, bj)
                special_block = block.copy()
    
    # Clear all blocks to 0 (keep separator 5s)
    for r in positions:
        for c in positions:
            out[r:r+3, c:c+3] = 0
    
    # Fill blocks based on special block's non-zero cells
    for i in range(3):
        for j in range(3):
            v = special_block[i, j]
            if v != 0:
                r = positions[i]
                c = positions[j]
                out[r:r+3, c:c+3] = v
    
    return out
