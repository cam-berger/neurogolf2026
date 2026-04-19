def method(grid):
    import numpy as np
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    best_count = -1
    best_rect = None
    
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= h or x < 0 or x >= w:
                        continue
                    if visited[y, x] or grid[y, x] == 0:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((y + dy, x + dx))
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                r0, r1 = min(ys), max(ys)
                c0, c1 = min(xs), max(xs)
                sub = grid[r0:r1+1, c0:c1+1]
                count2 = int(np.sum(sub == 2))
                if count2 > best_count:
                    best_count = count2
                    best_rect = sub.copy()
    
    return best_rect
