def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    visited = np.zeros((H, W), dtype=bool)
    
    shapes = []  # (cells, sub_matrix)
    for i in range(H):
        for j in range(W):
            if g[i, j] == 8 and not visited[i, j]:
                # BFS with 8-connectivity
                stack = [(i, j)]
                cells = []
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and g[ny, nx] == 8:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                y0, y1 = min(ys), max(ys)
                x0, x1 = min(xs), max(xs)
                sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=int)
                for (y, x) in cells:
                    sub[y - y0, x - x0] = 1
                shapes.append((cells, sub))
    
    for cells, sub in shapes:
        count = 0
        for cells2, sub2 in shapes:
            if sub.shape == sub2.shape and np.array_equal(sub, sub2):
                count += 1
        color = 1 if count > 1 else 2
        for (y, x) in cells:
            out[y, x] = color
    
    return out
