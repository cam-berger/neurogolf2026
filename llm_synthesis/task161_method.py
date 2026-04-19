import numpy as np
from collections import Counter

def method(grid):
    h, w = grid.shape
    vals = grid[grid != 0]
    counts = Counter(vals.tolist())
    # The rare/marker color is the least frequent non-zero color
    rare = min(counts, key=counts.get)
    
    out = np.zeros_like(grid)
    rows = set()
    cols = set()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == rare:
                if 0 < r < h - 1:
                    rows.add(r)
                if 0 < c < w - 1:
                    cols.add(c)
    for r in rows:
        out[r, :] = rare
    for c in cols:
        out[:, c] = rare
    return out
