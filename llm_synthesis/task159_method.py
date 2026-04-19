import numpy as np

def method(grid):
    g = np.asarray(grid)
    # Find box of 2s
    ys, xs = np.where(g == 2)
    r0, r1 = ys.min(), ys.max()
    c0, c1 = xs.min(), xs.max()
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    interior_h = box_h - 2
    interior_w = box_w - 2
    
    # Find pattern (non-zero, non-2)
    mask = (g != 0) & (g != 2)
    pys, pxs = np.where(mask)
    pr0, pr1 = pys.min(), pys.max()
    pc0, pc1 = pxs.min(), pxs.max()
    pattern = g[pr0:pr1+1, pc0:pc1+1]
    color = g[pys[0], pxs[0]]
    
    ph, pw = pattern.shape
    sh = interior_h // ph
    sw = interior_w // pw
    
    # Scale pattern
    scaled = np.kron((pattern == color).astype(int), np.ones((sh, sw), dtype=int)) * color
    
    # Build output
    out = np.full((box_h, box_w), 0, dtype=g.dtype)
    out[0, :] = 2
    out[-1, :] = 2
    out[:, 0] = 2
    out[:, -1] = 2
    out[1:1+scaled.shape[0], 1:1+scaled.shape[1]] = scaled
    return out
