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
    
    components = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == 2 and not visited[r, c]:
                # BFS
                stack = [(r, c)]
                visited[r, c] = True
                comp = []
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for nr, nc in neighbors(cr, cc):
                        if g[nr, nc] == 2 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append(comp)
    
    out = g.copy()
    for comp in components:
        if len(comp) < 2:
            continue
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1 = min(rows)-1, max(rows)+1
        c0, c1 = min(cols)-1, max(cols)+1
        for rr in range(r0, r1+1):
            for cc in range(c0, c1+1):
                if 0 <= rr < H and 0 <= cc < W:
                    if out[rr, cc] == 0:
                        out[rr, cc] = 3
    return out
