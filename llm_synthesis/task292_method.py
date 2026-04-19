def method(grid):
    out = grid.copy()
    for c in range(out.shape[1]):
        if c % 3 == 0:
            out[:, c] = np.where(out[:, c] == 4, 6, out[:, c])
    return out
