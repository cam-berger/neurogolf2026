import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    for i in range(H):
        for j in range(W):
            if g[i, j] == 0 and not visited[i, j]:
                # BFS to get connected component of 0s
                stack = [(i, j)]
                cells = []
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if visited[r, c]:
                        continue
                    if g[r, c] != 0:
                        continue
                    visited[r, c] = True
                    cells.append((r, c))
                    stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
                
                if not cells:
                    continue
                
                rs = [x[0] for x in cells]
                cs = [x[1] for x in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                
                # Check the CC fills its bounding box exactly
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if len(cells) != area:
                    continue
                
                # Check border: one cell outside the bbox must be 5 or outside grid
                ok = True
                # top row
                if r0 - 1 >= 0:
                    for c in range(c0, c1 + 1):
                        if g[r0 - 1, c] != 5:
                            ok = False
                            break
                if ok and r1 + 1 < H:
                    for c in range(c0, c1 + 1):
                        if g[r1 + 1, c] != 5:
                            ok = False
                            break
                if ok and c0 - 1 >= 0:
                    for r in range(r0, r1 + 1):
                        if g[r, c0 - 1] != 5:
                            ok = False
                            break
                if ok and c1 + 1 < W:
                    for r in range(r0, r1 + 1):
                        if g[r, c1 + 1] != 5:
                            ok = False
                            break
                
                if ok:
                    for (r, c) in cells:
                        g[r, c] = 4
    
    return g
