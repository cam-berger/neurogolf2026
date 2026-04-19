def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    for i in range(h):
        for j in range(w):
            if g[i, j] == 1 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= h or x < 0 or x >= w:
                        continue
                    if visited[y, x] or g[y, x] != 1:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((y + dy, x + dx))
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                r0, r1 = min(ys), max(ys)
                c0, c1 = min(xs), max(xs)
                color = None
                for y in range(r0, r1 + 1):
                    for x in range(c0, c1 + 1):
                        v = grid[y, x]
                        if v != 0 and v != 1:
                            color = v
                if color is None:
                    continue
                for y in range(r0, r1 + 1):
                    for x in range(c0, c1 + 1):
                        if g[y, x] != 1:
                            g[y, x] = color
                if r0 > 0:
                    for x in range(c0, c1 + 1):
                        g[r0 - 1, x] = color
    return g
