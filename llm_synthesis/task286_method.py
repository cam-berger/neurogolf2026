def method(grid):
    import numpy as np
    from collections import deque
    
    g = grid.copy()
    H, W = g.shape
    
    # Find seed cells (non-0, non-8 colors)
    seeds = []
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0 and g[r, c] != 8:
                seeds.append((r, c))
    
    # Determine parity -> color mapping from seeds
    parity_color = {}
    for (r, c) in seeds:
        parity_color[(r + c) % 2] = int(g[r, c])
    
    # BFS through 0-cells (and seed cells) to find connected component
    visited = np.zeros((H, W), dtype=bool)
    q = deque()
    for s in seeds:
        visited[s] = True
        q.append(s)
    
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                if g[nr, nc] == 0:
                    visited[nr, nc] = True
                    q.append((nr, nc))
    
    # Fill 0-cells in the component based on parity
    for r in range(H):
        for c in range(W):
            if visited[r, c] and g[r, c] == 0:
                p = (r + c) % 2
                if p in parity_color:
                    g[r, c] = parity_color[p]
    
    return g
