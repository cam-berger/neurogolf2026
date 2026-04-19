import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    
    # Find vertical bars of 2s on left and right edges
    bars = []
    for c in [0, W-1]:
        rows = [r for r in range(H) if g[r, c] == 2]
        if rows:
            bars.append((c, rows))
    
    if len(bars) < 2:
        return g
    
    eights = list(zip(*np.where(g == 8)))
    
    bar_with_8s = None
    other_bar = None
    for c, rows in bars:
        rset = set(rows)
        if any(er in rset for er, ec in eights):
            bar_with_8s = (c, rows)
        else:
            other_bar = (c, rows)
    
    if bar_with_8s is None or other_bar is None:
        return g
    
    c1, rows1 = bar_with_8s
    rmin1 = min(rows1)
    rset1 = set(rows1)
    
    relative_positions = []
    for er, ec in eights:
        if er in rset1:
            g[er, ec] = 4
            if c1 == 0:
                for cc in range(1, ec):
                    g[er, cc] = 8
            else:
                for cc in range(ec+1, c1):
                    g[er, cc] = 8
            relative_positions.append(er - rmin1)
    
    c2, rows2 = other_bar
    rmin2 = min(rows2)
    for rp in relative_positions:
        tr = rmin2 + rp
        if 0 <= tr < H:
            if c2 == 0:
                for cc in range(1, W):
                    g[tr, cc] = 8
            else:
                for cc in range(0, c2):
                    g[tr, cc] = 8
    
    return g
