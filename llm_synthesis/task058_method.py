import numpy as np

def method(grid):
    n = grid.shape[0]
    out = np.zeros_like(grid)
    
    # Build sequence of segment lengths: n, n-1, n-1, n-3, n-3, n-5, n-5, ...
    lengths = [n]
    k = 1
    while True:
        v = n - (2 * k - 1)
        if v <= 0:
            break
        lengths.append(v)
        lengths.append(v)
        k += 1
    
    # Directions cycle: R, D, L, U
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Start at (0,0), direction R.
    # Each segment has 'length' cells filled.
    # After each segment, turn CW and the next segment starts at (cur + dir_new).
    r, c = 0, 0
    for i, L in enumerate(lengths):
        dr, dc = dirs[i % 4]
        for _ in range(L):
            if 0 <= r < n and 0 <= c < n:
                out[r, c] = 3
            r += dr
            c += dc
        # Back up one step (we overshot by one)
        r -= dr
        c -= dc
        # Turn CW and step once to start next segment
        ndr, ndc = dirs[(i + 1) % 4]
        r += ndr
        c += ndc
    
    return out
