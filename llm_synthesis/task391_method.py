import numpy as np
from collections import Counter

def method(grid):
    grid = np.asarray(grid)
    H, W = grid.shape
    colors = []
    for i in range(1, H, 2):
        for j in range(1, W, 3):
            colors.append(int(grid[i, j]))
    cnt = Counter(colors)
    bg = cnt.most_common(1)[0][0]
    others = [(c, n) for c, n in cnt.items() if c != bg]
    others.sort(key=lambda x: -x[1])
    result = np.array([[c] for c, _ in others], dtype=int)
    return result
