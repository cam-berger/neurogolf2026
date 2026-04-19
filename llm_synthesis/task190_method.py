def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    
    # Find 2x2 block
    color = None
    br, bc = None, None
    for r in range(h-1):
        for c in range(w-1):
            if g[r,c] != 0 and g[r,c+1] == g[r,c] and g[r+1,c] == g[r,c] and g[r+1,c+1] == g[r,c]:
                color = g[r,c]
                br, bc = r, c
                break
        if color is not None:
            break
    
    cr = br + 0.5
    cc = bc + 0.5
    block_cells = {(br,bc),(br,bc+1),(br+1,bc),(br+1,bc+1)}
    
    # Find single cells (non-zero, not in block)
    singles = []
    for r in range(h):
        for c in range(w):
            if g[r,c] != 0 and (r,c) not in block_cells:
                singles.append((r,c))
    
    for (r,c) in singles:
        dr = 1 if r > cr else -1
        dc = 1 if c > cc else -1
        rr, cc2 = r+dr, c+dc
        while 0 <= rr < h and 0 <= cc2 < w:
            g[rr,cc2] = color
            rr += dr
            cc2 += dc
    
    return g
