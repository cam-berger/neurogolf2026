def method(grid):
    import numpy as np
    H, W = grid.shape
    H_out = W  # output is square WxW = 10x10
    
    def check(p, s):
        for r in range(p, H):
            for c in range(W):
                c2 = c - s
                if 0 <= c2 < W:
                    if grid[r, c] != grid[r-p, c2]:
                        return False
                else:
                    if grid[r, c] != 0:
                        return False
        return True
    
    best = None
    for p in range(1, H):
        # try s=0 first, then other shifts
        shifts = [0] + [x for x in range(-(W-1), W) if x != 0]
        for s in shifts:
            if check(p, s):
                best = (p, s)
                break
        if best is not None:
            break
    
    out = np.zeros((max(H_out, H), W), dtype=grid.dtype)
    out[:H] = grid
    
    if best is not None:
        p, s = best
        for r in range(H, H_out):
            for c in range(W):
                c2 = c - s
                if 0 <= c2 < W:
                    out[r, c] = out[r-p, c2]
    
    return out[:H_out]
