def method(grid):
    import numpy as np
    from collections import deque
    g = grid.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    shapes = []
    for i in range(H):
        for j in range(W):
            if g[i, j] != 0 and not visited[i, j]:
                color = g[i, j]
                cells = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and g[ny, nx] == color:
                                visited[ny, nx] = True
                                q.append((ny, nx))
                shapes.append((color, cells))
    
    for color, cells in shapes:
        if len(cells) != 3:
            continue
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        y0, y1 = min(ys), max(ys)
        x0, x1 = min(xs), max(xs)
        if (y1 - y0) != 1 or (x1 - x0) != 1:
            continue
        cell_set = set(cells)
        corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
        missing_list = [c for c in corners if c not in cell_set]
        if len(missing_list) != 1:
            continue
        missing = missing_list[0]
        opposite = (y0 + y1 - missing[0], x0 + x1 - missing[1])
        dy = 1 if missing[0] - opposite[0] > 0 else -1
        dx = 1 if missing[1] - opposite[1] > 0 else -1
        y, x = missing[0] + dy, missing[1] + dx
        while 0 <= y < H and 0 <= x < W:
            g[y, x] = color
            y += dy
            x += dx
    return g
