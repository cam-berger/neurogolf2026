def method(grid):
    g = grid.copy()
    h, w = g.shape
    
    def find_block(color):
        coords = np.argwhere(g == color)
        if len(coords) == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return r0, c0, r1, c1
    
    b1 = find_block(1)
    if b1 is not None:
        r0, c0, _, _ = b1
        r, c = r0 - 1, c0 - 1
        while r >= 0 and c >= 0:
            g[r, c] = 1
            r -= 1
            c -= 1
    
    b2 = find_block(2)
    if b2 is not None:
        _, _, r1, c1 = b2
        r, c = r1 + 1, c1 + 1
        while r < h and c < w:
            g[r, c] = 2
            r += 1
            c += 1
    
    return g
