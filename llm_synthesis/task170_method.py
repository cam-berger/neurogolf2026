def method(grid):
    import numpy as np
    H, W = grid.shape
    mask = grid != 0
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                stack = [(i, j)]
                comp = []
                while stack:
                    r, c = stack.pop()
                    if 0 <= r < H and 0 <= c < W and mask[r, c] and not visited[r, c]:
                        visited[r, c] = True
                        comp.append((r, c))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr or dc:
                                    stack.append((r + dr, c + dc))
                components.append(comp)
    
    key_comp = None
    shape_comp = None
    for comp in components:
        colors = set(int(grid[r, c]) for r, c in comp)
        if len(colors) > 1:
            key_comp = comp
        else:
            shape_comp = comp
    
    # Key bbox
    rs = [r for r, c in key_comp]
    cs = [c for r, c in key_comp]
    kr0, kr1 = min(rs), max(rs)
    kc0, kc1 = min(cs), max(cs)
    key = grid[kr0:kr1+1, kc0:kc1+1]
    N = key.shape[0]
    
    # Shape mask
    shape_mask = np.zeros_like(grid, dtype=bool)
    for r, c in shape_comp:
        shape_mask[r, c] = True
    
    # Shape bbox
    rs = [r for r, c in shape_comp]
    cs = [c for r, c in shape_comp]
    sr0, sr1 = min(rs), max(rs)
    sc0, sc1 = min(cs), max(cs)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    bh = sh // N
    bw = sw // N
    
    pattern = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            r = sr0 + i * bh
            c = sc0 + j * bw
            if np.any(shape_mask[r:r+bh, c:c+bw]):
                pattern[i, j] = True
    
    out = np.where(pattern, key, 0)
    return out
