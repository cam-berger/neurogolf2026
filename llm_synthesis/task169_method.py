def method(grid):
    g = grid.copy()
    out = g.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    mapping = {2: 3, 3: 2, 4: 1}
    
    for i in range(H):
        for j in range(W):
            if g[i, j] == 5 and not visited[i, j]:
                # BFS
                stack = [(i, j)]
                cells = []
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and g[ny,nx] == 5 and not visited[ny,nx]:
                            visited[ny,nx] = True
                            stack.append((ny,nx))
                c = mapping.get(len(cells), 5)
                for y, x in cells:
                    out[y, x] = c
    return out
