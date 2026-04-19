def method(grid):
    import numpy as np
    colors = [c for c in np.unique(grid) if c != 0]
    
    def adj_count(c):
        mask = (grid == c).astype(int)
        return int(np.sum(mask[:-1,:] * mask[1:,:]) + np.sum(mask[:,:-1] * mask[:,1:]))
    
    scores = sorted(colors, key=adj_count, reverse=True)
    block_color = scores[0]
    noise_color = scores[1]
    
    rows, cols = np.where(grid == block_color)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    
    H = r_max - r_min + 1
    W = c_max - c_min + 1
    
    out = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            r1 = r_min + i * H // 3
            r2 = r_min + (i+1) * H // 3
            c1 = c_min + j * W // 3
            c2 = c_min + (j+1) * W // 3
            region = grid[r1:r2, c1:c2]
            if np.any(region == block_color):
                out[i, j] = noise_color
    return out
