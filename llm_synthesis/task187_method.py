import numpy as np
from collections import deque

def method(grid):
    g = np.array(grid)
    H, W = g.shape
    # find marker color (non-zero)
    nz = g[g != 0]
    if len(nz) == 0:
        return np.full_like(g, 3)
    marker = nz[0]
    
    out = np.where(g == 0, 3, g).astype(int)
    
    # Flood fill from all border cells that are not marker (through non-marker cells)
    # Cells reachable from border -> stay 3 (outside)
    # Cells not reachable -> become 2 (inside)
    reachable = np.zeros((H, W), dtype=bool)
    q = deque()
    for r in range(H):
        for c in range(W):
            if (r == 0 or r == H-1 or c == 0 or c == W-1) and g[r, c] != marker:
                if not reachable[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
    
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not reachable[nr, nc] and g[nr, nc] != marker:
                reachable[nr, nc] = True
                q.append((nr, nc))
    
    # Cells that are 0 in input, not reachable from border -> 2
    mask = (g == 0) & (~reachable)
    out[mask] = 2
    
    return out
