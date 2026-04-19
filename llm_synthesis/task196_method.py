def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    for i in range(H):
        for j in range(W):
            if g[i, j] == 1 and not visited[i, j]:
                # BFS to find connected component
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if visited[y, x] or g[y, x] != 1:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
                
                if not cells:
                    continue
                
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                y0, y1 = min(ys), max(ys)
                x0, x1 = min(xs), max(xs)
                h = y1 - y0 + 1
                w = x1 - x0 + 1
                
                # Must be at least 3x3 to have interior
                if h < 3 or w < 3:
                    continue
                
                # Expected border cells of bounding box
                expected = set()
                for x in range(x0, x1 + 1):
                    expected.add((y0, x))
                    expected.add((y1, x))
                for y in range(y0, y1 + 1):
                    expected.add((y, x0))
                    expected.add((y, x1))
                
                if set(cells) == expected:
                    # Also verify interior is all 0
                    interior_ok = True
                    for y in range(y0 + 1, y1):
                        for x in range(x0 + 1, x1):
                            if g[y, x] != 0:
                                interior_ok = False
                                break
                        if not interior_ok:
                            break
                    if interior_ok:
                        for y, x in cells:
                            g[y, x] = 3
    
    return g
