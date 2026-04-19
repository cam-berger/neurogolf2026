def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    
    visited = np.zeros_like(g, dtype=bool)
    rects = []
    for i in range(H):
        for j in range(W):
            if g[i, j] == 2 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if visited[y, x] or g[y, x] != 2:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                rects.append((min(ys), max(ys), min(xs), max(xs)))
    
    out = g.copy()
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            y1a, y1b, x1a, x1b = rects[i]
            y2a, y2b, x2a, x2b = rects[j]
            ro_a = max(y1a, y2a)
            ro_b = min(y1b, y2b)
            if ro_a > ro_b:
                continue
            if x1b < x2a:
                gap_a, gap_b = x1b + 1, x2a - 1
            elif x2b < x1a:
                gap_a, gap_b = x2b + 1, x1a - 1
            else:
                continue
            if gap_a > gap_b:
                continue
            region = g[ro_a:ro_b+1, gap_a:gap_b+1]
            if np.all(region == 0):
                out[ro_a:ro_b+1, gap_a:gap_b+1] = 9
    return out
