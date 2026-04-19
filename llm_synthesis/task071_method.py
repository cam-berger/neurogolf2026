def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    colors = [int(c) for c in np.unique(g) if c != 0]
    if len(colors) < 2:
        return g.copy()
    
    B = None
    A = None
    for c in colors:
        ys, xs = np.where(g == c)
        if len(ys) == 0:
            continue
        r0, r1 = ys.min(), ys.max()
        c0, c1 = xs.min(), xs.max()
        if (r1 - r0 + 1) * (c1 - c0 + 1) == len(ys):
            B = c
        else:
            A = c
    if A is None:
        for c in colors:
            if c != B:
                A = c
                break
    if B is None:
        for c in colors:
            if c != A:
                B = c
                break
    
    A_mask = (g == A)
    B_mask = (g == B)
    A_cells = list(zip(*np.where(A_mask)))
    B_cells = list(zip(*np.where(B_mask)))
    
    def try_axis_v(a2):
        for r, j in A_cells:
            jm = a2 - j
            if jm < 0 or jm >= W:
                return None
            if not (A_mask[r, jm] or B_mask[r, jm]):
                return None
        out_mask = A_mask.copy()
        for r, j in A_cells:
            jm = a2 - j
            out_mask[r, jm] = True
        return out_mask
    
    def try_axis_h(a2):
        for r, j in A_cells:
            rm = a2 - r
            if rm < 0 or rm >= H:
                return None
            if not (A_mask[rm, j] or B_mask[rm, j]):
                return None
        out_mask = A_mask.copy()
        for r, j in A_cells:
            rm = a2 - r
            out_mask[rm, j] = True
        return out_mask
    
    best = None
    best_score = -1
    
    for a2 in range(0, 2 * W):
        res = try_axis_v(a2)
        if res is not None:
            score = int(np.sum(res & B_mask))
            if score > best_score:
                best_score = score
                best = res
    
    for a2 in range(0, 2 * H):
        res = try_axis_h(a2)
        if res is not None:
            score = int(np.sum(res & B_mask))
            if score > best_score:
                best_score = score
                best = res
    
    out = np.zeros_like(g)
    if best is not None:
        out[best] = A
    else:
        out[A_mask] = A
    return out
