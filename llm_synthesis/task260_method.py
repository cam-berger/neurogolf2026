def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    
    # find the main (non-0, non-5) color
    colors = set(g.flatten().tolist()) - {0, 5}
    C = list(colors)[0]
    
    # find main diagonal offset k such that c - r = k for C cells
    C_pos = list(zip(*np.where(g == C)))
    k = C_pos[0][1] - C_pos[0][0]
    
    # draw main diagonal
    for r in range(H):
        c = r + k
        if 0 <= c < W:
            out[r, c] = C
    
    # compute offsets of 5 cells from main diagonal
    five_pos = list(zip(*np.where(g == 5)))
    offs = [((c - r) - k) for r, c in five_pos]
    
    pos_list = [o for o in offs if o > 0]
    neg_list = [o for o in offs if o < 0]
    
    if pos_list:
        w = max(pos_list)
        new_k = k + (w + 2)
        for r in range(H):
            c = r + new_k
            if 0 <= c < W:
                out[r, c] = C
    
    if neg_list:
        w = -min(neg_list)
        new_k = k - (w + 2)
        for r in range(H):
            c = r + new_k
            if 0 <= c < W:
                out[r, c] = C
    
    return out
