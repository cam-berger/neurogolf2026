def method(grid):
    import numpy as np
    cells = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                cells.append((c, r, int(grid[r, c])))
    cells.sort(key=lambda x: (x[0], x[1]))
    values = [v for _, _, v in cells]
    # pad to 9
    while len(values) < 9:
        values.append(0)
    out = np.zeros((3, 3), dtype=int)
    for i in range(3):
        row_vals = values[i*3:(i+1)*3]
        if i % 2 == 1:
            row_vals = row_vals[::-1]
        out[i] = row_vals
    return out
