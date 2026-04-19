import numpy as np

def method(grid):
    out = grid.copy()
    h, w = grid.shape
    fours = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == 4]
    for i in range(len(fours)):
        for j in range(i+1, len(fours)):
            for k in range(j+1, len(fours)):
                for l in range(k+1, len(fours)):
                    pts = [fours[i], fours[j], fours[k], fours[l]]
                    rs = sorted(set(p[0] for p in pts))
                    cs = sorted(set(p[1] for p in pts))
                    if len(rs) == 2 and len(cs) == 2:
                        r1, r2 = rs
                        c1, c2 = cs
                        expected = {(r1,c1),(r1,c2),(r2,c1),(r2,c2)}
                        if set(pts) == expected:
                            out[r1+1:r2, c1+1:c2] = 2
    return out
