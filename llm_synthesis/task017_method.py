def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    
    candidates = []
    for pr in range(1, 12):
        for pc in range(1, 12):
            candidates.append((pr*pc, pr, pc))
    candidates.sort()
    
    best = None
    for _, pr, pc in candidates:
        ok = True
        reps = {}
        for i in range(H):
            if not ok: break
            for j in range(W):
                if g[i, j] == 0:
                    continue
                k = (i % pr, j % pc)
                if k in reps:
                    if reps[k] != g[i, j]:
                        ok = False
                        break
                else:
                    reps[k] = g[i, j]
        if ok and len(reps) == pr * pc:
            best = (pr, pc, reps)
            break
    
    if best is None:
        return g
    
    pr, pc, reps = best
    out = g.copy()
    for i in range(H):
        for j in range(W):
            if out[i, j] == 0:
                out[i, j] = reps[(i % pr, j % pc)]
    return out
