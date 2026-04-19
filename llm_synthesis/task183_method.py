def method(grid):
    import numpy as np
    g = np.asarray(grid)
    n = g.shape[0]
    TL, TR = g[0, 0], g[0, -1]
    BL, BR = g[-1, 0], g[-1, -1]
    inner = g[2:-2, 2:-2].copy()
    h, w = inner.shape
    hh, hw = h // 2, w // 2
    out = np.zeros_like(inner)
    quads = [((0, hh, 0, hw), TL),
             ((0, hh, hw, w), TR),
             ((hh, h, 0, hw), BL),
             ((hh, h, hw, w), BR)]
    for (r0, r1, c0, c1), color in quads:
        sub = inner[r0:r1, c0:c1]
        out[r0:r1, c0:c1] = np.where(sub == 8, color, 0)
    return out
