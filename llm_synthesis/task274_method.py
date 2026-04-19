def method(grid):
    import numpy as np
    rows_with_5 = np.any(grid == 5, axis=1)
    cols_with_5 = np.any(grid == 5, axis=0)
    five_rows = np.where(rows_with_5)[0]
    five_cols = np.where(cols_with_5)[0]
    r_top, r_bot = five_rows[0], five_rows[-1]
    c_left, c_right = five_cols[0], five_cols[-1]
    # interior region (excluding the 5 walls and bottom)
    inside = grid[r_top:r_bot, c_left+1:c_right]
    empty_rows = int(np.sum(np.all(inside == 0, axis=1)))
    
    out = np.zeros((3, 3), dtype=int)
    positions = [(0,0),(0,1),(0,2),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2)]
    for i in range(min(empty_rows, 9)):
        out[positions[i]] = 8
    return out
