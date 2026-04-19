def method(grid):
    out = grid.copy()
    rows, cols = np.where(grid == 8)
    if len(rows) == 0:
        return out
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    region = out[r0:r1+1, c0:c1+1]
    region[region == 0] = 2
    return out
