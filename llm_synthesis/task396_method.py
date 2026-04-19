def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    H, W = grid.shape
    colors = [int(c) for c in np.unique(grid) if c != 0]
    
    best = None  # (size, t, b, l, r, color)
    
    for c in colors:
        mask = (grid == c)
        visited = np.zeros_like(mask)
        for i in range(H):
            for j in range(W):
                if mask[i, j] and not visited[i, j]:
                    stack = [(i, j)]
                    cells = []
                    visited[i, j] = True
                    while stack:
                        y, x = stack.pop()
                        cells.append((y, x))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                    ys = [y for y, _ in cells]
                    xs = [x for _, x in cells]
                    t, b, l, r = min(ys), max(ys), min(xs), max(xs)
                    h, w = b - t + 1, r - l + 1
                    if h < 3 or w < 3:
                        continue
                    # Check border of bounding box is all c
                    ok = True
                    for y in range(t, b + 1):
                        if not ok:
                            break
                        for x in range(l, r + 1):
                            on_border = (y == t or y == b or x == l or x == r)
                            if on_border and grid[y, x] != c:
                                ok = False
                                break
                    if not ok:
                        continue
                    size = h * w
                    if best is None or size > best[0]:
                        best = (size, t, b, l, r, c)
    
    if best is None:
        return grid.copy()
    
    _, t, b, l, r, c = best
    others = [co for co in colors if co != c]
    other = others[0] if others else c
    out = grid[t:b+1, l:r+1].copy()
    out[out == c] = other
    return out
