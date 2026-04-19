def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    colors = np.unique(grid)
    for c in colors:
        ys, xs = np.where(grid == c)
        if len(ys) == 0:
            continue
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        expected = set()
        for i in range(r0, r1 + 1):
            expected.add((i, c0))
            expected.add((i, c1))
        for j in range(c0, c1 + 1):
            expected.add((r0, j))
            expected.add((r1, j))
        actual = set(zip(ys.tolist(), xs.tolist()))
        if actual == expected:
            return grid[r0:r1 + 1, c0:c1 + 1].copy()
    return grid.copy()
