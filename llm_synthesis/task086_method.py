def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    visited = np.zeros_like(g, dtype=bool)
    
    for i in range(H):
        for j in range(W):
            if g[i, j] != 0 and not visited[i, j]:
                stack = [(i, j)]
                cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if visited[y, x] or g[y, x] == 0:
                        continue
                    visited[y, x] = True
                    cells.append((y, x))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((y + dy, x + dx))
                
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                r0, r1 = min(ys), max(ys)
                c0, c1 = min(xs), max(xs)
                N = r1 - r0 + 1
                k = N - 2
                
                A = g[r0, c0]
                B = g[r0 + 1, c0 + 1]
                
                nr0, nr1 = r0 - k, r1 + k
                nc0, nc1 = c0 - k, c1 + k
                
                # Inner N x N: swap A and B
                for y in range(r0, r1 + 1):
                    for x in range(c0, c1 + 1):
                        if g[y, x] == A:
                            out[y, x] = B
                        else:
                            out[y, x] = A
                
                # Top strip
                for y in range(nr0, r0):
                    for x in range(c0, c1 + 1):
                        if 0 <= y < H and 0 <= x < W:
                            out[y, x] = A
                # Bottom strip
                for y in range(r1 + 1, nr1 + 1):
                    for x in range(c0, c1 + 1):
                        if 0 <= y < H and 0 <= x < W:
                            out[y, x] = A
                # Left strip
                for y in range(r0, r1 + 1):
                    for x in range(nc0, c0):
                        if 0 <= y < H and 0 <= x < W:
                            out[y, x] = A
                # Right strip
                for y in range(r0, r1 + 1):
                    for x in range(c1 + 1, nc1 + 1):
                        if 0 <= y < H and 0 <= x < W:
                            out[y, x] = A
    
    return out
