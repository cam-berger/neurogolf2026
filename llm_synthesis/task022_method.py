def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    out = np.zeros((3, 3), dtype=grid.dtype)
    
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= h or x < 0 or x >= w:
                        continue
                    if visited[y, x] or grid[y, x] == 0:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            stack.append((y + dy, x + dx))
                
                fives = [(y, x) for y, x in cells if grid[y, x] == 5]
                if not fives:
                    continue
                fy, fx = fives[0]
                
                for y, x in cells:
                    dy, dx = y - fy, x - fx
                    if -1 <= dy <= 1 and -1 <= dx <= 1:
                        out[dy + 1, dx + 1] = grid[y, x]
    
    return out
