def method(grid):
    import numpy as np
    from collections import deque
    
    H, W = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    best_count = -1
    best_box = None
    
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0 and not visited[i, j]:
                # BFS to find connected component
                q = deque([(i, j)])
                visited[i, j] = True
                cells = []
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] != 0:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                sub = grid[r0:r1+1, c0:c1+1]
                count2 = int(np.sum(sub == 2))
                if count2 > best_count:
                    best_count = count2
                    best_box = (r0, r1, c0, c1)
    
    r0, r1, c0, c1 = best_box
    return grid[r0:r1+1, c0:c1+1].copy()
