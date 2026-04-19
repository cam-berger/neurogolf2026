def method(grid):
    import numpy as np
    g = np.asarray(grid)
    ones = np.where(g == 1)
    r0, r1 = ones[0].min(), ones[0].max()
    c0, c1 = ones[1].min(), ones[1].max()
    flipped = g[::-1, :]
    result = flipped[r0:r1+1, c0:c1+1].copy()
    # Safety: if the flipped region still contains 1s (shouldn't if vertical symmetry holds),
    # try horizontal flip instead.
    if np.any(result == 1):
        flipped2 = g[:, ::-1]
        result = flipped2[r0:r1+1, c0:c1+1].copy()
        if np.any(result == 1):
            flipped3 = g[::-1, ::-1]
            result = flipped3[r0:r1+1, c0:c1+1].copy()
    return result
