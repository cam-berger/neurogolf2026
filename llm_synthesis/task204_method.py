import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                yield nr, nc
    
    for r in range(H):
        for c in range(W):
            if g[r, c] == 1 and not visited[r, c]:
                # BFS to get component
                stack = [(r, c)]
                comp = []
                while stack:
                    y, x = stack.pop()
                    if visited[y, x]:
                        continue
                    visited[y, x] = True
                    comp.append((y, x))
                    for ny, nx in neighbors(y, x):
                        if g[ny, nx] == 1 and not visited[ny, nx]:
                            stack.append((ny, nx))
                
                ys = [p[0] for p in comp]
                xs = [p[1] for p in comp]
                r0, r1 = min(ys), max(ys)
                c0, c1 = min(xs), max(xs)
                side = r1 - r0 + 1
                fill = 7 if side % 2 == 1 else 2
                for i in range(r0+1, r1):
                    for j in range(c0+1, c1):
                        if g[i, j] == 0:
                            g[i, j] = fill
    return g
