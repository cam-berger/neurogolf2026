def method(grid):
    g = grid.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    rects = []
    for i in range(H):
        for j in range(W):
            if g[i, j] == 6 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if visited[y, x] or g[y, x] != 6:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((y+dy, x+dx))
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                rects.append((min(ys), max(ys), min(xs), max(xs)))
    
    out = g.copy()
    # Fill interior 8s with 4
    for r0, r1, c0, c1 in rects:
        for y in range(r0, r1+1):
            for x in range(c0, c1+1):
                if out[y, x] == 8:
                    out[y, x] = 4
    # Draw 3 borders around each bounding box
    for r0, r1, c0, c1 in rects:
        for y in range(r0-1, r1+2):
            for x in range(c0-1, c1+2):
                if 0 <= y < H and 0 <= x < W:
                    if y == r0-1 or y == r1+1 or x == c0-1 or x == c1+1:
                        if out[y, x] == 8:
                            out[y, x] = 3
    return out
