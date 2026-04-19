def method(grid):
    import numpy as np
    # Check if all rows are identical
    if all(np.array_equal(grid[0], grid[i]) for i in range(grid.shape[0])):
        row = grid[0]
        out = [row[0]]
        for v in row[1:]:
            if v != out[-1]:
                out.append(v)
        return np.array(out).reshape(1, -1)
    # Else all columns identical
    col = grid[:, 0]
    out = [col[0]]
    for v in col[1:]:
        if v != out[-1]:
            out.append(v)
    return np.array(out).reshape(-1, 1)
