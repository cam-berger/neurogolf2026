import numpy as np

def method(grid):
    H, W = grid.shape
    out = np.tile(grid, (2, 2)).copy()
    nH, nW = out.shape
    # For each non-zero cell, mark diagonal neighbors as 8 (if currently 0)
    nonzero_positions = list(zip(*np.nonzero(out)))
    for r, c in nonzero_positions:
        for dr in (-1, 1):
            for dc in (-1, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < nH and 0 <= nc < nW:
                    if out[nr, nc] == 0:
                        out[nr, nc] = 8
    return out
