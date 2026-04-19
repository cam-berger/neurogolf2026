import numpy as np

def method(grid):
    g = grid.copy()
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    shapes = []
    for i in range(h):
        for j in range(w):
            if g[i, j] == 4 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= h or c < 0 or c >= w: continue
                    if visited[r, c] or g[r, c] != 4: continue
                    visited[r, c] = True
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((r+dr, c+dc))
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                shapes.append((min(rs), max(rs), min(cs), max(cs), len(cells)))
    
    shapes.sort(key=lambda s: s[4])
    colors = [1, 2]
    for idx, (r0, r1, c0, c1, _) in enumerate(shapes):
        color = colors[idx] if idx < 2 else 2
        for r in range(r0+1, r1):
            for c in range(c0+1, c1):
                g[r, c] = color
    return g
