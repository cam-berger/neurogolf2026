def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    
    # Find 8-lines
    h_lines = [r for r in range(H) if np.all(grid[r, :] == 8)]
    v_lines = [c for c in range(W) if np.all(grid[:, c] == 8)]
    
    # Find 2-markers (from original)
    markers = list(zip(*np.where(grid == 2)))
    
    for (mr, mc) in markers:
        if h_lines:
            above = [r for r in h_lines if r < mr]
            below = [r for r in h_lines if r > mr]
            targets = []
            if above:
                targets.append(max(above))
            if below:
                targets.append(min(below))
            for tr in targets:
                # 3x3 box of 8s at (tr, mc)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        rr, cc = tr+dr, mc+dc
                        if 0 <= rr < H and 0 <= cc < W:
                            g[rr, cc] = 8
                g[tr, mc] = 2
                # Fill 2s between marker and box
                if tr < mr:
                    for rr in range(tr+2, mr):
                        if 0 <= rr < H:
                            g[rr, mc] = 2
                else:
                    for rr in range(mr+1, tr-1):
                        if 0 <= rr < H:
                            g[rr, mc] = 2
        elif v_lines:
            left = [c for c in v_lines if c < mc]
            right = [c for c in v_lines if c > mc]
            targets = []
            if left:
                targets.append(max(left))
            if right:
                targets.append(min(right))
            for tc in targets:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        rr, cc = mr+dr, tc+dc
                        if 0 <= rr < H and 0 <= cc < W:
                            g[rr, cc] = 8
                g[mr, tc] = 2
                if tc < mc:
                    for cc in range(tc+2, mc):
                        if 0 <= cc < W:
                            g[mr, cc] = 2
                else:
                    for cc in range(mc+1, tc-1):
                        if 0 <= cc < W:
                            g[mr, cc] = 2
    
    return g
