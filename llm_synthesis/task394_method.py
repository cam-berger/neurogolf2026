def method(grid):
    import numpy as np
    H, W = grid.shape
    zeros = np.argwhere(grid == 0)
    if len(zeros) == 0:
        return grid.copy()
    r0, c0 = zeros.min(axis=0)
    r1, c1 = zeros.max(axis=0)
    
    # find row period
    pr = H
    for p in range(1, H):
        ok = True
        for r in range(H - p):
            for c in range(W):
                a, b = grid[r, c], grid[r + p, c]
                if a != 0 and b != 0 and a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            pr = p
            break
    
    # find col period
    pc = W
    for p in range(1, W):
        ok = True
        for r in range(H):
            for c in range(W - p):
                a, b = grid[r, c], grid[r, c + p]
                if a != 0 and b != 0 and a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            pc = p
            break
    
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=grid.dtype)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            val = 0
            for rr in range(r % pr, H, pr):
                for cc in range(c % pc, W, pc):
                    if grid[rr, cc] != 0:
                        val = grid[rr, cc]
                        break
                if val != 0:
                    break
            out[r - r0, c - c0] = val
    return out
