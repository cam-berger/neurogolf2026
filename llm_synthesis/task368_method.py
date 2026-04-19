import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    
    # Find template: cells that are not 0 and not 5
    mask_tmpl = (g != 0) & (g != 5)
    rs, cs = np.where(mask_tmpl)
    r0, r1 = rs.min(), rs.max()
    c0, c1 = cs.min(), cs.max()
    template = g[r0:r1+1, c0:c1+1].copy()
    th, tw = template.shape
    
    # Find connected regions of 5
    visited = np.zeros_like(g, dtype=bool)
    for i in range(H):
        for j in range(W):
            if g[i, j] == 5 and not visited[i, j]:
                # BFS
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if visited[y, x] or g[y, x] != 5:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                y0, y1 = min(ys), max(ys)
                x0, x1 = min(xs), max(xs)
                h, w = y1 - y0 + 1, x1 - x0 + 1
                if h == th and w == tw:
                    g[y0:y1+1, x0:x1+1] = template
    
    return g
