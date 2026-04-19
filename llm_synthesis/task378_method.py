def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    colors = [c for c in np.unique(g) if c != 0]
    
    def is_filled_rect(color):
        mask = (g == color)
        if not mask.any():
            return False
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]
        return bool(mask[r0:r1+1, c0:c1+1].all())
    
    # Content is the color that forms a filled rectangle; wall is the other
    content_color = None
    for c in colors:
        if is_filled_rect(c):
            content_color = c
            break
    if content_color is None:
        return g
    wall_candidates = [c for c in colors if c != content_color]
    if not wall_candidates:
        return g
    wall_color = wall_candidates[0]
    
    for r in range(H):
        for col in range(W):
            if g[r, col] != wall_color:
                continue
            dirs_h = []
            dirs_v = []
            if col > 0 and g[r, col-1] == wall_color:
                dirs_h.append(-1)
            if col < W-1 and g[r, col+1] == wall_color:
                dirs_h.append(1)
            if r > 0 and g[r-1, col] == wall_color:
                dirs_v.append(-1)
            if r < H-1 and g[r+1, col] == wall_color:
                dirs_v.append(1)
            if len(dirs_h) != 1 or len(dirs_v) != 1:
                continue
            dh = -dirs_h[0]
            dv = -dirs_v[0]
            rr, cc = r + dv, col + dh
            while 0 <= rr < H and 0 <= cc < W:
                if g[rr, cc] == 0:
                    g[rr, cc] = content_color
                rr += dv
                cc += dh
    
    return g
