import numpy as np

def method(grid):
    g = np.array(grid)
    h, w = g.shape
    out = g.copy()
    layers = min(h, w) // 2
    colors = [g[i, i] for i in range(layers)]
    reversed_colors = colors[::-1]
    for i in range(layers):
        c = reversed_colors[i]
        out[i, i:w-i] = c
        out[h-1-i, i:w-i] = c
        out[i:h-i, i] = c
        out[i:h-i, w-1-i] = c
    # handle center if odd
    if h % 2 == 1 and w % 2 == 1:
        # center keeps innermost reversed (already handled if layers covers)
        pass
    return out
