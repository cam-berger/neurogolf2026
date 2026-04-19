import numpy as np

def method(grid):
    g = np.asarray(grid)
    # Find the marker color: one that appears exactly 4 times forming a rectangle
    marker_color = None
    corners = None
    for c in np.unique(g):
        if c == 0:
            continue
        pts = np.argwhere(g == c)
        if len(pts) == 4:
            rs = sorted(set(pts[:,0].tolist()))
            cs = sorted(set(pts[:,1].tolist()))
            if len(rs) == 2 and len(cs) == 2:
                # confirm all 4 combos present
                combos = {(r, cc) for r in rs for cc in cs}
                if combos == {tuple(p) for p in pts.tolist()}:
                    marker_color = c
                    corners = (rs[0], rs[1], cs[0], cs[1])
                    break
    r0, r1, c0, c1 = corners
    interior = g[r0+1:r1, c0+1:c1].copy()
    out = np.where(interior != 0, marker_color, 0)
    return out
