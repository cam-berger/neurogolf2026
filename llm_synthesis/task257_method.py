def method(grid):
    import numpy as np
    TL = grid[0:4, 0:4]
    TR = grid[0:4, 5:9]
    BL = grid[5:9, 0:4]
    BR = grid[5:9, 5:9]
    priority = [7, 4, 8, 6]
    out = np.zeros((4, 4), dtype=grid.dtype)
    for r in range(4):
        for c in range(4):
            vals = {TL[r, c], TR[r, c], BL[r, c], BR[r, c]}
            vals.discard(0)
            vals.discard(1)
            chosen = 0
            for p in priority:
                if p in vals:
                    chosen = p
                    break
            out[r, c] = chosen
    return out
