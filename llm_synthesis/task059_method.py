def method(grid):
    g = np.array(grid)
    out = g.copy()
    # Find non-background, non-separator color
    color = 0
    for v in np.unique(g):
        if v != 0 and v != 5:
            color = v
            break
    
    # 3x3 cells located at row/col blocks 0-2, 4-6, 8-10
    starts = [0, 4, 8]
    counts = {}
    for i, r in enumerate(starts):
        for j, c in enumerate(starts):
            block = g[r:r+3, c:c+3]
            counts[(i,j)] = int(np.sum(block == color))
    
    max_count = max(counts.values())
    
    # Clear all cells, fill max cells
    for i, r in enumerate(starts):
        for j, c in enumerate(starts):
            if counts[(i,j)] == max_count and max_count > 0:
                out[r:r+3, c:c+3] = color
            else:
                out[r:r+3, c:c+3] = 0
    
    return out
