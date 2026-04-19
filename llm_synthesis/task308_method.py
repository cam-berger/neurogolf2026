def method(grid):
    import numpy as np
    from collections import Counter
    
    H, W = grid.shape
    flat = grid.flatten().tolist()
    bg = Counter(flat).most_common(1)[0][0]
    
    colors = [c for c in np.unique(grid) if c != bg]
    
    shapes = []
    for c in colors:
        pts = np.argwhere(grid == c)
        rows = sorted(set(pts[:, 0].tolist()))
        cols = sorted(set(pts[:, 1].tolist()))
        if len(rows) == 2 and len(cols) == 2 and len(pts) == 4:
            h = rows[1] - rows[0] + 1
            w = cols[1] - cols[0] + 1
            shapes.append((c, 'rect', h, w))
        else:
            r_min, r_max = int(pts[:, 0].min()), int(pts[:, 0].max())
            c_min, c_max = int(pts[:, 1].min()), int(pts[:, 1].max())
            diameter = max(r_max - r_min, c_max - c_min) + 1
            shapes.append((c, 'diamond', diameter, diameter))
    
    N = max(max(s[2], s[3]) for s in shapes)
    out = np.full((N, N), bg, dtype=grid.dtype)
    
    for c, kind, h, w in shapes:
        if kind == 'diamond':
            mid = N // 2
            out[0, mid] = c
            out[N - 1, mid] = c
            out[mid, 0] = c
            out[mid, N - 1] = c
        else:
            if h == N and w == N:
                out[0, 0] = c
                out[0, N - 1] = c
                out[N - 1, 0] = c
                out[N - 1, N - 1] = c
            elif h == N and w < N:
                off = (N - w) // 2
                out[0, off] = c
                out[0, N - 1 - off] = c
                out[N - 1, off] = c
                out[N - 1, N - 1 - off] = c
            elif w == N and h < N:
                off = (N - h) // 2
                out[off, 0] = c
                out[N - 1 - off, 0] = c
                out[off, N - 1] = c
                out[N - 1 - off, N - 1] = c
            else:
                # fallback: place at corners scaled
                off_r = (N - h) // 2
                off_c = (N - w) // 2
                out[off_r, off_c] = c
                out[off_r, N - 1 - off_c] = c
                out[N - 1 - off_r, off_c] = c
                out[N - 1 - off_r, N - 1 - off_c] = c
    
    return out
