def method(grid):
    g = grid.copy()
    h, w = g.shape
    # Find each U-shape: pattern is X X X / X 0 X where middle has the gap
    for r in range(h - 1):
        for c in range(w - 2):
            top = g[r, c:c+3]
            bot = g[r+1, c:c+3]
            if (top[0] == top[1] == top[2] != 0 and
                bot[0] == bot[2] == top[0] and bot[1] == 0):
                g[h-1, c+1] = 4
    return g
