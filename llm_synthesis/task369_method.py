import numpy as np

def method(grid):
    g = np.asarray(grid)
    out = g.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    for i in range(H):
        for j in range(W):
            if g[i, j] == 0 and not visited[i, j]:
                stack = [(i, j)]
                comp = []
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and g[ny, nx] == 0:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(comp) == 1:
                    color = 3
                elif len(comp) == 2:
                    color = 2
                else:
                    color = 1
                for y, x in comp:
                    out[y, x] = color
    return out
