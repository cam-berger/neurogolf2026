import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    rects = []
    
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
                    stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                r0, r1 = min(ys), max(ys)
                c0, c1 = min(xs), max(xs)
                two = None
                for (y, x) in cells:
                    if g[y, x] == 2:
                        two = (y, x)
                        break
                if two is not None:
                    rects.append((r0, r1, c0, c1, two))
    
    out = g.copy()
    
    for r0, r1, c0, c1, (ty, tx) in rects:
        on_top = (ty == r0)
        on_bottom = (ty == r1)
        on_left = (tx == c0)
        on_right = (tx == c1)
        
        # Determine which edge and extension direction
        if on_top and not on_bottom:
            # extend upward
            d = r1 - r0  # distance to opposite (bottom) edge
            half = d
            for y in range(0, r0):
                for dx in range(-half, half + 1):
                    x = tx + dx
                    if 0 <= x < W:
                        out[y, x] = 2 if dx == 0 else 3
        elif on_bottom and not on_top:
            d = r1 - r0
            half = d
            for y in range(r1 + 1, H):
                for dx in range(-half, half + 1):
                    x = tx + dx
                    if 0 <= x < W:
                        out[y, x] = 2 if dx == 0 else 3
        elif on_left and not on_right:
            d = c1 - c0
            half = d
            for x in range(0, c0):
                for dy in range(-half, half + 1):
                    y = ty + dy
                    if 0 <= y < H:
                        out[y, x] = 2 if dy == 0 else 3
        elif on_right and not on_left:
            d = c1 - c0
            half = d
            for x in range(c1 + 1, W):
                for dy in range(-half, half + 1):
                    y = ty + dy
                    if 0 <= y < H:
                        out[y, x] = 2 if dy == 0 else 3
    
    return out
