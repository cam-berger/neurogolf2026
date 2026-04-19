import numpy as np
from collections import Counter

def method(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find frame color F and inner color C
    nz = np.argwhere(g != 0)
    fr0, fc0 = nz.min(0)
    fr1, fc1 = nz.max(0)
    
    # Frame color: the color at corners of non-zero bbox
    F = g[fr0, fc0]
    colors = [c for c in np.unique(g) if c != 0 and c != F]
    C = colors[0]
    
    # Determine standard inner rectangle using mode of ranges per row/col
    rows_with_c = {}
    for r in range(H):
        cs = np.where(g[r] == C)[0]
        if len(cs) > 0:
            rows_with_c[r] = (int(cs.min()), int(cs.max()))
    
    cols_with_c = {}
    for c in range(W):
        rs = np.where(g[:, c] == C)[0]
        if len(rs) > 0:
            cols_with_c[c] = (int(rs.min()), int(rs.max()))
    
    row_ranges = Counter(rows_with_c.values())
    std_c_range = row_ranges.most_common(1)[0][0]
    col_ranges = Counter(cols_with_c.values())
    std_r_range = col_ranges.most_common(1)[0][0]
    
    ir0, ir1 = std_r_range
    ic0, ic1 = std_c_range
    
    # Find bumps
    v_cuts = set()
    h_cuts = set()
    for r in range(H):
        for c in range(W):
            if g[r, c] == C:
                if not (ir0 <= r <= ir1 and ic0 <= c <= ic1):
                    if r < ir0 and ic0 <= c <= ic1:
                        v_cuts.add(c)
                    elif r > ir1 and ic0 <= c <= ic1:
                        v_cuts.add(c)
                    elif c < ic0 and ir0 <= r <= ir1:
                        h_cuts.add(r)
                    elif c > ic1 and ir0 <= r <= ir1:
                        h_cuts.add(r)
    
    # Build output
    out = np.zeros_like(g)
    out[fr0:fr1+1, fc0:fc1+1] = F
    out[ir0:ir1+1, ic0:ic1+1] = C
    
    for c in v_cuts:
        out[ir0:ir1+1, c] = F
        out[0:fr0, c] = C
        out[fr1+1:H, c] = C
    
    for r in h_cuts:
        out[r, ic0:ic1+1] = F
        out[r, 0:fc0] = C
        out[r, fc1+1:W] = C
    
    return out
