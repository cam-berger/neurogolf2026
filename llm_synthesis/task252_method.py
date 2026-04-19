import numpy as np

def method(grid):
    out = grid.copy()
    cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] != 0]
    diagonals = {}
    for r, c in cells:
        diagonals.setdefault(r - c, []).append((r, c))
    for d, pts in diagonals.items():
        pts.sort()
        for i, (r, c) in enumerate(pts):
            if i % 2 == 1:
                out[r, c] = 4
    return out
