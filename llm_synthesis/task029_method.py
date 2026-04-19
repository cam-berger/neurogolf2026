def method(grid):
    grid = np.asarray(grid)
    H, W = grid.shape
    for c in np.unique(grid):
        positions = np.argwhere(grid == c)
        if len(positions) < 8:
            continue
        r1, c1 = positions.min(axis=0)
        r2, c2 = positions.max(axis=0)
        if r2 - r1 < 2 or c2 - c1 < 2:
            continue
        # Check if all border cells equal c
        top = grid[r1, c1:c2+1]
        bot = grid[r2, c1:c2+1]
        left = grid[r1:r2+1, c1]
        right = grid[r1:r2+1, c2]
        if (np.all(top == c) and np.all(bot == c) and
                np.all(left == c) and np.all(right == c)):
            # Count of c should equal perimeter (no extras inside or outside would form same rect)
            perim = 2*(r2-r1) + 2*(c2-c1)
            # Extract interior
            return grid[r1+1:r2, c1+1:c2].copy()
    return grid.copy()
