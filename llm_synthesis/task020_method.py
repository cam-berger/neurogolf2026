import numpy as np

def method(grid):
    g = grid.copy()
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return g
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    out = g.copy()
    H, W = g.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r, c] != 0:
                continue
            dr = r - cr
            dc = c - cc
            # Try 4-fold rotational symmetry: 0, 90, 180, 270
            candidates = [(dr, dc), (-dc, dr), (-dr, -dc), (dc, -dr)]
            for ddr, ddc in candidates:
                rr = cr + ddr
                cc2 = cc + ddc
                if 0 <= rr < H and 0 <= cc2 < W:
                    if g[rr, cc2] != 0:
                        out[r, c] = g[rr, cc2]
                        break
    return out
