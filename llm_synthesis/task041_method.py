def method(grid):
    out = grid.copy()
    colors = set(int(x) for x in np.unique(grid)) - {0}
    for c in colors:
        rs, cs = np.where(grid == c)
        rows_cells = {}
        for r, col in zip(rs, cs):
            rows_cells.setdefault(int(r), []).append(int(col))
        for r, cols in rows_cells.items():
            lo, hi = min(cols), max(cols)
            out[r, lo:hi+1] = c
    return out
