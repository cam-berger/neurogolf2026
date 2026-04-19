def method(grid):
    import numpy as np
    from collections import Counter
    
    h, w = grid.shape
    zero_rows = [i for i in range(h) if np.all(grid[i] == 0)]
    zero_cols = [j for j in range(w) if np.all(grid[:, j] == 0)]
    
    def get_ranges(zeros, total):
        ranges = []
        start = 0
        for z in zeros:
            if z > start:
                ranges.append((start, z))
            start = z + 1
        if start < total:
            ranges.append((start, total))
        return ranges
    
    row_ranges = get_ranges(zero_rows, h)
    col_ranges = get_ranges(zero_cols, w)
    
    quads = []
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            sub = grid[r0:r1, c0:c1]
            colors = set(sub.flatten().tolist()) - {0}
            if len(colors) == 1:
                color = next(iter(colors))
                quads.append((color, sub))
    
    cnt = Counter(c for c, s in quads)
    for c, s in quads:
        if cnt[c] == 1:
            return s
    return quads[0][1]
