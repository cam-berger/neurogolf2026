def method(grid):
    left = grid[:, :3]
    right = grid[:, 4:]
    out = np.where((left == 0) & (right == 0), 8, 0)
    return out
