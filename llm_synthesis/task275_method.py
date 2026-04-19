def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    h, w = grid.shape
    
    # Split along the longer dimension
    if w >= h:
        a = grid[:, :w//2]
        b = grid[:, w//2:]
    else:
        a = grid[:h//2, :]
        b = grid[h//2:, :]
    
    def unique_nonzero(g):
        vals = g[g != 0]
        return set(vals.tolist())
    
    # Template is the half with only one non-zero color
    if len(unique_nonzero(a)) == 1:
        template, mapping = a, b
    else:
        template, mapping = b, a
    
    H, W = template.shape
    mask = (template != 0).astype(grid.dtype)
    
    output = np.zeros((H * H, W * W), dtype=grid.dtype)
    
    for i in range(H):
        for j in range(W):
            color = mapping[i, j]
            if color != 0:
                output[i*H:(i+1)*H, j*W:(j+1)*W] = mask * color
    
    return output
