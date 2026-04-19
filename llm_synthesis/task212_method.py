def method(grid):
    g = grid.copy()
    H, W = g.shape
    # Find the row of 5s
    line_row = None
    for r in range(H):
        if np.all(g[r] == 5):
            line_row = r
            break
    
    for c in range(W):
        for r in range(H):
            if r == line_row:
                continue
            v = g[r, c]
            if v == 2:
                # extend toward the line
                if r < line_row:
                    # fill from r+1 down to line_row-1
                    for rr in range(r+1, line_row):
                        if g[rr, c] == 0:
                            g[rr, c] = 2
                else:
                    # fill from line_row+1 up to r-1
                    for rr in range(line_row+1, r):
                        if g[rr, c] == 0:
                            g[rr, c] = 2
            elif v == 1:
                # extend away from the line
                if r < line_row:
                    # fill from 0 up to r-1
                    for rr in range(0, r):
                        if g[rr, c] == 0:
                            g[rr, c] = 1
                else:
                    # fill from r+1 to H-1
                    for rr in range(r+1, H):
                        if g[rr, c] == 0:
                            g[rr, c] = 1
    return g
