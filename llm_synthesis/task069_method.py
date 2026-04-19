import numpy as np

def method(grid):
    g = np.asarray(grid)
    H, W = g.shape
    out = g.copy()
    
    # Find template (non-0, non-8 cells)
    tmask = (g != 0) & (g != 8)
    if not tmask.any():
        return out
    ys, xs = np.where(tmask)
    r0, r1 = ys.min(), ys.max()
    c0, c1 = xs.min(), xs.max()
    template = g[r0:r1+1, c0:c1+1].copy()
    th, tw = template.shape
    tmp_mask = (template != 0)
    
    # Clear template from output
    for i in range(th):
        for j in range(tw):
            if template[i, j] != 0:
                out[r0+i, c0+j] = 0
    
    # Find connected components of 8s
    emask = (g == 8)
    visited = np.zeros_like(emask, dtype=bool)
    
    for r in range(H):
        for c in range(W):
            if emask[r, c] and not visited[r, c]:
                comp = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if emask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                
                ys_c = [p[0] for p in comp]
                xs_c = [p[1] for p in comp]
                br0, br1 = min(ys_c), max(ys_c)
                bc0, bc1 = min(xs_c), max(xs_c)
                bh = br1 - br0 + 1
                bw = bc1 - bc0 + 1
                
                if bh == th and bw == tw:
                    comp_mask = np.zeros((th, tw), dtype=bool)
                    for y, x in comp:
                        comp_mask[y-br0, x-bc0] = True
                    if np.array_equal(comp_mask, tmp_mask):
                        for y, x in comp:
                            out[y, x] = 0
                        for i in range(th):
                            for j in range(tw):
                                if template[i, j] != 0:
                                    out[br0+i, bc0+j] = template[i, j]
    
    return out
