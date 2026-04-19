import numpy as np

def method(grid):
    g = np.array(grid)
    out = np.zeros_like(g)
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    for i in range(H):
        for j in range(W):
            if g[i, j] == 5 and not visited[i, j]:
                # BFS to find connected component (4-connectivity)
                stack = [(i, j)]
                cells = []
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == 5 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                color = 2 if len(cells) == 6 else 1
                for y, x in cells:
                    out[y, x] = color
    return out
