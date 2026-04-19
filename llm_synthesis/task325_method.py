import numpy as np

def method(grid):
    g = (np.asarray(grid) != 0).astype(np.int8)
    H, W = g.shape
    seen = np.zeros_like(g)
    count = 0
    for i in range(H):
        for j in range(W):
            if g[i, j] and not seen[i, j]:
                count += 1
                stack = [(i, j)]
                seen[i, j] = 1
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if g[ny, nx] and not seen[ny, nx]:
                                    seen[ny, nx] = 1
                                    stack.append((ny, nx))
    out = np.zeros((count, count), dtype=np.asarray(grid).dtype)
    for k in range(count):
        out[k, k] = 8
    return out
