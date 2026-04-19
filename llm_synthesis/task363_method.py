def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    
    # Find 2 positions
    twos = np.argwhere(g == 2)
    if len(twos) == 0:
        return g
    r0, c0 = twos.min(axis=0)
    r1, c1 = twos.max(axis=0)
    th, tw = r1 - r0 + 1, c1 - c0 + 1
    
    # Build template
    tmpl = g[r0:r1+1, c0:c1+1]
    two_positions = [(i, j) for i in range(th) for j in range(tw) if tmpl[i, j] == 2]
    five_positions = [(i, j) for i in range(th) for j in range(tw) if tmpl[i, j] == 5]
    
    # Search
    new_twos = []
    for i in range(H - th + 1):
        for j in range(W - tw + 1):
            if i == r0 and j == c0:
                continue  # skip original
            ok = True
            for di, dj in two_positions:
                if g[i+di, j+dj] != 0:
                    ok = False
                    break
            if not ok:
                continue
            for di, dj in five_positions:
                if g[i+di, j+dj] != 5:
                    ok = False
                    break
            if not ok:
                continue
            for di, dj in two_positions:
                new_twos.append((i+di, j+dj))
    
    for r, c in new_twos:
        g[r, c] = 2
    
    return g
