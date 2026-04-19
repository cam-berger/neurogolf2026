import numpy as np

def method(grid):
    out = grid.copy()
    H, W = grid.shape
    # Find two non-zero markers
    pts = [(r, c, int(grid[r, c])) for r in range(H) for c in range(W) if grid[r, c] != 0]
    (r1, c1, v1), (r2, c2, v2) = pts[0], pts[1]
    
    if r1 == r2:
        # Horizontal alignment
        row = r1
        if c1 > c2:
            c1, c2, v1, v2 = c2, c1, v2, v1
        D = c2 - c1
        k = (D - 3) // 2
        bar1 = c1 + k
        bar2 = c2 - k
        # Stem for v1: row, cols c1..bar1
        for c in range(c1, bar1 + 1):
            out[row, c] = v1
        # Bar for v1: col bar1, rows row-2..row+2
        for dr in range(-2, 3):
            rr = row + dr
            if 0 <= rr < H:
                out[rr, bar1] = v1
        # Hooks for v1: at (row±2, bar1+1)
        for dr in [-2, 2]:
            rr = row + dr
            cc = bar1 + 1
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = v1
        # Stem for v2
        for c in range(bar2, c2 + 1):
            out[row, c] = v2
        for dr in range(-2, 3):
            rr = row + dr
            if 0 <= rr < H:
                out[rr, bar2] = v2
        for dr in [-2, 2]:
            rr = row + dr
            cc = bar2 - 1
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = v2
    else:
        # Vertical alignment
        col = c1
        if r1 > r2:
            r1, r2, v1, v2 = r2, r1, v2, v1
        D = r2 - r1
        k = (D - 3) // 2
        bar1 = r1 + k
        bar2 = r2 - k
        # Stem for v1
        for r in range(r1, bar1 + 1):
            out[r, col] = v1
        for dc in range(-2, 3):
            cc = col + dc
            if 0 <= cc < W:
                out[bar1, cc] = v1
        for dc in [-2, 2]:
            cc = col + dc
            rr = bar1 + 1
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = v1
        # Stem for v2
        for r in range(bar2, r2 + 1):
            out[r, col] = v2
        for dc in range(-2, 3):
            cc = col + dc
            if 0 <= cc < W:
                out[bar2, cc] = v2
        for dc in [-2, 2]:
            cc = col + dc
            rr = bar2 - 1
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = v2
    
    return out
