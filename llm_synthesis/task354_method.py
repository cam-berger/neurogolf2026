import numpy as np

def method(grid):
    out = grid.copy()
    H, W = grid.shape
    
    # Get markers from row 0: (col, color)
    markers = [(c, grid[0, c]) for c in range(W) if grid[0, c] != 0]
    
    # Find connected components of 5s
    visited = np.zeros_like(grid, dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 5 and not visited[r, c]:
                # BFS
                stack = [(r, c)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W: continue
                    if visited[y, x] or grid[y, x] != 5: continue
                    visited[y, x] = True
                    cells.append((y, x))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((y+dy, x+dx))
                
                cols = [x for _, x in cells]
                cmin, cmax = min(cols), max(cols)
                
                # Find matching marker
                for mc, mcolor in markers:
                    if cmin <= mc <= cmax:
                        for y, x in cells:
                            out[y, x] = mcolor
                        break
    
    return out
