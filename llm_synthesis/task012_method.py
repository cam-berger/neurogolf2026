import numpy as np

def method(grid):
    g = np.array(grid)
    h, w = g.shape
    out = np.zeros_like(g)
    
    for r in range(1, h-1):
        for c in range(1, w-1):
            center = g[r, c]
            if center == 0:
                continue
            u, d, l, rt = g[r-1,c], g[r+1,c], g[r,c-1], g[r,c+1]
            if u != 0 and u == d == l == rt and u != center:
                c2 = center  # inner/diagonal color
                c1 = u       # arm color
                # stamp 5x5 pattern
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        rr, cc = r+dr, c+dc
                        if rr < 0 or rr >= h or cc < 0 or cc >= w:
                            continue
                        if dr == 0 and dc == 0:
                            v = c2
                        elif dr == 0 or dc == 0:
                            v = c1
                        elif abs(dr) == abs(dc):
                            v = c2
                        else:
                            v = 0
                        if v != 0:
                            out[rr, cc] = v
    return out
