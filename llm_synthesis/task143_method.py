def method(grid):
    import numpy as np
    from collections import defaultdict
    g = grid.copy()
    H, W = g.shape
    
    # Find the L-shape boundary of 5s
    r5 = None
    for r in range(H):
        if (g[r] == 5).sum() >= 2:
            r5 = r
            break
    c5 = None
    for c in range(W):
        if (g[:, c] == 5).sum() >= 2:
            c5 = c
            break
    
    # Extract template
    template_coords = [(i, j) for i in range(r5) for j in range(c5) if g[i, j] != 0]
    if not template_coords:
        return g
    
    def norm_shape(coords):
        minr = min(c[0] for c in coords)
        minc = min(c[1] for c in coords)
        return frozenset((r - minr, c - minc) for r, c in coords)
    
    template_norm = norm_shape(template_coords)
    
    # Group non-template, non-background, non-5 cells by color via BFS
    visited = np.zeros_like(g, dtype=bool)
    for i in range(r5 + 1):
        for j in range(c5 + 1):
            visited[i, j] = True
    
    from collections import deque
    for i in range(H):
        for j in range(W):
            if visited[i, j] or g[i, j] == 0 or g[i, j] == 5:
                continue
            color = g[i, j]
            q = deque([(i, j)])
            visited[i, j] = True
            cells = [(i, j)]
            while q:
                r, c = q.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and g[nr, nc] == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            if norm_shape(cells) == template_norm:
                for r, c in cells:
                    g[r, c] = 5
    
    return g
