def method(grid):
    import numpy as np
    H, W = grid.shape
    out = np.zeros_like(grid)
    visited = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 2 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if visited[y, x] or grid[y, x] != 2:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                r0, r1 = min(ys), max(ys)
                c0, c1 = min(xs), max(xs)
                h, w = r1 - r0 + 1, c1 - c0 + 1
                if h >= 3 and w >= 3:
                    sub = grid[r0:r1+1, c0:c1+1]
                    border_ok = ((sub[0, :] == 2).all() and (sub[-1, :] == 2).all()
                                 and (sub[:, 0] == 2).all() and (sub[:, -1] == 2).all())
                    interior = sub[1:-1, 1:-1]
                    if border_ok and (interior == 0).all():
                        out[r0+1:r1, c0+1:c1] = 3
    return out
