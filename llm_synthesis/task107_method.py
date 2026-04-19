def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    # Collect unique non-zero colors in last row and last column
    border_colors = set()
    for v in grid[:, -1]:
        if v != 0:
            border_colors.add(int(v))
    for v in grid[-1, :]:
        if v != 0:
            border_colors.add(int(v))
    scale = len(border_colors) + 1
    
    N = 5 * scale
    out = np.zeros((N, N), dtype=grid.dtype)
    
    # Fill blocks: last row, last col as-is; inner non-zero cells as shape
    for i in range(5):
        for j in range(5):
            val = grid[i, j]
            if i == 4 or j == 4:
                out[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = val
            elif val != 0:
                out[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = val
    
    # Find shape bounding box from inner 4x4
    inner = grid[:4, :4]
    nz = np.argwhere(inner != 0)
    if len(nz) > 0:
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        R0 = int(r0) * scale
        C0 = int(c0) * scale
        R1 = (int(r1) + 1) * scale - 1
        C1 = (int(c1) + 1) * scale - 1
        
        zero_r_min, zero_r_max = 0, 4*scale - 1
        zero_c_min, zero_c_max = 0, 4*scale - 1
        
        corners = [
            (R0, C0, -1, -1),
            (R0, C1, -1, 1),
            (R1, C0, 1, -1),
            (R1, C1, 1, 1),
        ]
        for sr, sc, dr, dc in corners:
            r, c = sr + dr, sc + dc
            while zero_r_min <= r <= zero_r_max and zero_c_min <= c <= zero_c_max:
                out[r, c] = 2
                r += dr
                c += dc
    
    return out
