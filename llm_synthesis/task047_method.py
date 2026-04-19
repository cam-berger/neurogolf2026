import numpy as np

def method(grid):
    g = np.asarray(grid)
    H, W = g.shape
    markers = [(r, c, g[r, c]) for r in range(H) for c in range(W) if g[r, c] != 0]
    out = np.zeros_like(g)
    for i in range(H):
        for j in range(W):
            colors = set()
            for r, c, v in markers:
                if i == r or j == c:
                    colors.add(v)
            if len(colors) == 1:
                out[i, j] = colors.pop()
            elif len(colors) > 1:
                out[i, j] = 2
    return out
