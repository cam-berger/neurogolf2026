def method(grid):
    g = grid.copy()
    H, W = g.shape
    # find 2x2 non-zero block
    r0 = c0 = None
    for r in range(H-1):
        for c in range(W-1):
            if g[r,c] != 0 and g[r,c+1] != 0 and g[r+1,c] != 0 and g[r+1,c+1] != 0:
                r0, c0 = r, c
                break
        if r0 is not None:
            break
    TL, TR = g[r0,c0], g[r0,c0+1]
    BL, BR = g[r0+1,c0], g[r0+1,c0+1]
    # place diagonal blocks with opposite-corner color
    placements = [(-2,-2,BR), (-2,2,BL), (2,-2,TR), (2,2,TL)]
    for dr, dc, col in placements:
        for i in range(2):
            for j in range(2):
                rr, cc = r0+dr+i, c0+dc+j
                if 0 <= rr < H and 0 <= cc < W:
                    g[rr,cc] = col
    return g
