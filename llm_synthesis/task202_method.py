def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    
    row0 = g[0]
    vals = row0[row0 != 0]
    horizontal = len(set(vals.tolist())) == 1
    
    if horizontal:
        row_color = []
        for i in range(h):
            v = g[i][g[i] != 0]
            row_color.append(int(v[0]) if len(v) else 0)
        bands = []
        start = 0
        for i in range(1, h):
            if row_color[i] != row_color[start]:
                bands.append((start, i))
                start = i
        bands.append((start, h))
        for (s, e) in bands:
            sub = g[s:e]
            cols = np.where(np.any(sub == 0, axis=0))[0]
            for c in cols:
                g[s:e, c] = 0
    else:
        col_color = []
        for j in range(w):
            v = g[:, j][g[:, j] != 0]
            col_color.append(int(v[0]) if len(v) else 0)
        bands = []
        start = 0
        for j in range(1, w):
            if col_color[j] != col_color[start]:
                bands.append((start, j))
                start = j
        bands.append((start, w))
        for (s, e) in bands:
            sub = g[:, s:e]
            rows = np.where(np.any(sub == 0, axis=1))[0]
            for r in rows:
                g[r, s:e] = 0
    return g
