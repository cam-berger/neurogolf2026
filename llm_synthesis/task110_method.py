def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    nz = (g != 0)
    
    # find vertical period
    py = None
    for p in range(1, H):
        ok = True
        for r in range(H - p):
            if not ok: break
            for c in range(W):
                if nz[r, c] and nz[r+p, c] and g[r, c] != g[r+p, c]:
                    ok = False
                    break
        if ok:
            py = p
            break
    if py is None:
        py = H
    
    # find horizontal period
    px = None
    for p in range(1, W):
        ok = True
        for r in range(H):
            if not ok: break
            for c in range(W - p):
                if nz[r, c] and nz[r, c+p] and g[r, c] != g[r, c+p]:
                    ok = False
                    break
        if ok:
            px = p
            break
    if px is None:
        px = W
    
    out = g.copy()
    kr_range = H // py + 2
    kc_range = W // px + 2
    for r in range(H):
        for c in range(W):
            if not nz[r, c]:
                found = False
                for k in range(-kr_range, kr_range + 1):
                    if found: break
                    for l in range(-kc_range, kc_range + 1):
                        nr = r + k * py
                        nc = c + l * px
                        if 0 <= nr < H and 0 <= nc < W and nz[nr, nc]:
                            out[r, c] = g[nr, nc]
                            found = True
                            break
    return out
