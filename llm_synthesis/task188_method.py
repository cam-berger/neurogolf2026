import numpy as np

def method(grid):
    h, w = grid.shape
    
    def can_halve_v():
        return h % 2 == 0 and np.array_equal(grid[:h//2], grid[h//2:])
    
    def can_halve_h():
        return w % 2 == 0 and np.array_equal(grid[:, :w//2], grid[:, w//2:])
    
    v_ok = can_halve_v()
    h_ok = can_halve_h()
    
    if v_ok and h_ok:
        # Both work; prefer halving the larger dimension
        if w >= h:
            return grid[:, :w//2].copy()
        else:
            return grid[:h//2].copy()
    if v_ok:
        return grid[:h//2].copy()
    if h_ok:
        return grid[:, :w//2].copy()
    return grid.copy()
