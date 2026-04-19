import numpy as np

def method(grid):
    g = grid.copy()
    h, w = g.shape
    
    # Find 2-line (full row or full column)
    two_rows = [r for r in range(h) if np.all(g[r] == 2)]
    two_cols = [c for c in range(w) if np.all(g[:, c] == 2)]
    
    # Find bbox of 3s
    r3, c3 = np.where(g == 3)
    r_min, r_max = r3.min(), r3.max()
    c_min, c_max = c3.min(), c3.max()
    
    out = np.zeros_like(g)
    
    if two_cols:
        tc = two_cols[0]
        out[:, tc] = 2
        shape_w = c_max - c_min + 1
        shape = g[r_min:r_max+1, c_min:c_max+1]
        if c_max < tc:  # shape is to the left of the 2-line
            new_c_max = tc - 1
            new_c_min = new_c_max - shape_w + 1
            out[r_min:r_max+1, new_c_min:new_c_max+1] = shape
            out[:, new_c_min - 1] = 8
        else:  # shape is to the right of the 2-line
            new_c_min = tc + 1
            new_c_max = new_c_min + shape_w - 1
            out[r_min:r_max+1, new_c_min:new_c_max+1] = shape
            out[:, new_c_max + 1] = 8
    else:
        tr = two_rows[0]
        out[tr, :] = 2
        shape_h = r_max - r_min + 1
        shape = g[r_min:r_max+1, c_min:c_max+1]
        if r_max < tr:  # shape above
            new_r_max = tr - 1
            new_r_min = new_r_max - shape_h + 1
            out[new_r_min:new_r_max+1, c_min:c_max+1] = shape
            out[new_r_min - 1, :] = 8
        else:  # shape below
            new_r_min = tr + 1
            new_r_max = new_r_min + shape_h - 1
            out[new_r_min:new_r_max+1, c_min:c_max+1] = shape
            out[new_r_max + 1, :] = 8
    
    return out
