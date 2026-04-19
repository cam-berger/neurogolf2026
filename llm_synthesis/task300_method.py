import numpy as np

def method(grid):
    from collections import deque
    g = np.asarray(grid)
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    best = None
    best_size = -1
    for i in range(H):
        for j in range(W):
            if g[i, j] != 0 and not visited[i, j]:
                color = g[i, j]
                q = deque([(i, j)])
                visited[i, j] = True
                cells = []
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr,nc] and g[nr,nc] == color:
                            visited[nr,nc] = True
                            q.append((nr,nc))
                if len(cells) > best_size:
                    best_size = len(cells)
                    rs = [x[0] for x in cells]
                    cs = [x[1] for x in cells]
                    r0, r1 = min(rs), max(rs)
                    c0, c1 = min(cs), max(cs)
                    best = g[r0:r1+1, c0:c1+1].copy()
    return best
