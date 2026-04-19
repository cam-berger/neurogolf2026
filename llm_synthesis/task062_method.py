def method(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    colors = set(int(x) for x in g.flatten()) - {0, 2}
    C = colors.pop()
    twos = [tuple(p) for p in np.argwhere(g == 2).tolist()]
    cs = [tuple(p) for p in np.argwhere(g == C).tolist()]
    c_set = set(cs)

    # Determine axis: find a 2-cell adjacent to a C-cell; axis is midway between them
    axis_r = None
    axis_c = None
    for r, c in twos:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in c_set:
                if dr != 0:
                    axis_r = r + dr / 2.0
                else:
                    axis_c = c + dc / 2.0
                break
        if axis_r is not None or axis_c is not None:
            break

    new_cells = set(cs)
    for r, c in cs:
        if axis_r is not None:
            nr = int(2 * axis_r - r)
            new_cells.add((nr, c))
        elif axis_c is not None:
            nc = int(2 * axis_c - c)
            new_cells.add((r, nc))

    out = np.full((h, w), 3)
    for r, c in new_cells:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = C
    return out
