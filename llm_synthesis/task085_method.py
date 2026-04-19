import numpy as np

def method(grid):
    g = grid.copy()
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not visited[i, j]:
                color = g[i, j]
                # find bounding box of this rectangle
                # expand right
                c1 = j
                while c1 + 1 < w and g[i, c1 + 1] == color:
                    c1 += 1
                r1 = i
                while r1 + 1 < h and g[r1 + 1, j] == color:
                    r1 += 1
                visited[i:r1+1, j:c1+1] = True
                # middle row alternates
                if r1 - i == 2:  # 3 rows
                    mid = i + 1
                    for cc in range(j, c1 + 1):
                        if (cc - j) % 2 == 1:
                            g[mid, cc] = 0
    return g
