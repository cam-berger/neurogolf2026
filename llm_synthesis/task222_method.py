import numpy as np

def method(grid):
    g = np.asarray(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    best = (0, None)  # area, (r0,r1,c0,c1,color)
    for color in range(1, 10):
        mask = (g == color).astype(int)
        # find largest rectangle of all-1s
        heights = np.zeros(W, dtype=int)
        for r in range(H):
            for c in range(W):
                heights[c] = heights[c] + 1 if mask[r, c] else 0
            # largest rectangle in histogram
            stack = []
            for c in range(W + 1):
                cur = heights[c] if c < W else 0
                start = c
                while stack and stack[-1][1] > cur:
                    idx, h = stack.pop()
                    area = h * (c - idx)
                    if area > best[0] and h >= 2 and (c - idx) >= 2:
                        best = (area, (r - h + 1, r, idx, c - 1, color))
                    start = idx
                stack.append((start, cur))
    if best[1] is not None:
        r0, r1, c0, c1, color = best[1]
        out[r0:r1+1, c0:c1+1] = color
    return out
