import numpy as np

def method(grid):
    g = np.asarray(grid)
    out = np.zeros_like(g)
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    for i in range(H):
        for j in range(W):
            if g[i, j] != 0 and not visited[i, j]:
                color = g[i, j]
                # Find connected rectangle via BFS (4-connectivity on same color)
                stack = [(i, j)]
                cells = []
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if visited[r, c] or g[r, c] != color:
                        continue
                    visited[r, c] = True
                    cells.append((r, c))
                    stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                height = r1 - r0 + 1
                new_r0 = r0 - height
                new_r1 = r1 - height
                for r, c in cells:
                    nr = r - height
                    if 0 <= nr < H:
                        out[nr, c] = color
    return out
