import numpy as np

def method(grid):
    out = grid.copy()
    rows, cols = out.shape
    for r in range(rows):
        left = out[r, 0]
        right = out[r, cols-1]
        if left != 0 and right != 0:
            mid = cols // 2
            for c in range(cols):
                if c < mid:
                    out[r, c] = left
                elif c == mid:
                    out[r, c] = 5
                else:
                    out[r, c] = right
    return out
