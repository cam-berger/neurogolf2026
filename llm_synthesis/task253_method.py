def method(grid):
    import numpy as np
    out = np.zeros((4, 4), dtype=grid.dtype)
    colors = np.unique(grid)
    for c in colors:
        if c == 0:
            continue
        rs, cs = np.where(grid == c)
        r0, c0 = rs.min(), cs.min()
        cells = set(zip(rs.tolist(), cs.tolist()))
        # Find missing cell in 2x2 bbox
        missing = None
        for dr in range(2):
            for dc in range(2):
                if (r0 + dr, c0 + dc) not in cells:
                    missing = (dr, dc)
        # Quadrant is opposite corner of missing
        qr = 1 - missing[0]
        qc = 1 - missing[1]
        # Place shape (preserving orientation) in the quadrant
        for dr in range(2):
            for dc in range(2):
                if (dr, dc) != missing:
                    out[qr * 2 + dr, qc * 2 + dc] = c
    return out
