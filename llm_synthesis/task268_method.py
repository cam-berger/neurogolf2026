def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    mask = g != 0
    if not mask.any():
        return g
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    
    top = g[r0, c0:c1+1]
    bot = g[r1, c0:c1+1]
    left = g[r0:r1+1, c0]
    right = g[r0:r1+1, c1]
    
    for rr in range(r0+1, r1):
        for cc in range(c0+1, c1):
            if g[rr, cc] == 0:
                g[rr, cc] = 4
    
    def gap_indices(arr):
        return [i for i, v in enumerate(arr) if v == 0]
    
    direction = None
    gap_cells = []
    
    if (top == 0).any():
        direction = (-1, 0)
        gap_cells = [(r0, c0 + i) for i in gap_indices(top)]
    elif (bot == 0).any():
        direction = (1, 0)
        gap_cells = [(r1, c0 + i) for i in gap_indices(bot)]
    elif (left == 0).any():
        direction = (0, -1)
        gap_cells = [(r0 + i, c0) for i in gap_indices(left)]
    elif (right == 0).any():
        direction = (0, 1)
        gap_cells = [(r0 + i, c1) for i in gap_indices(right)]
    
    if direction is None:
        return g
    
    dr, dc = direction
    
    for (r, c) in gap_cells:
        rr, cc = r, c
        while 0 <= rr < H and 0 <= cc < W:
            if g[rr, cc] == 0:
                g[rr, cc] = 4
            rr += dr
            cc += dc
    
    if gap_cells:
        first = gap_cells[0]
        last = gap_cells[-1]
        if dr == 0:
            perps = [(-1, 0), (1, 0)]
        else:
            perps = [(0, -1), (0, 1)]
        
        for start, perp in [(first, perps[0]), (last, perps[1])]:
            sdr = dr + perp[0]
            sdc = dc + perp[1]
            rr = start[0] + sdr
            cc = start[1] + sdc
            while 0 <= rr < H and 0 <= cc < W:
                if g[rr, cc] == 0:
                    g[rr, cc] = 4
                rr += sdr
                cc += sdc
    
    return g
