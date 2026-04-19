import numpy as np

def method(grid):
    g = np.asarray(grid)
    # find 5
    r5, c5 = np.argwhere(g == 5)[0]
    H, W = g.shape
    # find a seed cell adjacent to 5 that's non-zero and non-5
    seed = None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r5 + dr, c5 + dc
            if 0 <= nr < H and 0 <= nc < W and g[nr, nc] not in (0, 5):
                seed = (nr, nc)
                break
        if seed:
            break
    color = g[seed]
    # BFS over 8-connected cells of this color
    visited = {seed}
    stack = [seed]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
    rs = [p[0] for p in visited]
    cs = [p[1] for p in visited]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=g.dtype)
    for r, c in visited:
        out[r - r0, c - c0] = color
    return out
