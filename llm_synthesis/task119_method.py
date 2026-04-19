def method(grid):
    grid = grid.copy()
    H, W = grid.shape
    eights = list(zip(*np.where(grid == 8)))
    
    pt1 = min(eights)
    pt2 = max(eights)
    
    def edge_dist(p):
        r, c = p
        return min(r, H-1-r, c, W-1-c)
    
    if edge_dist(pt1) >= edge_dist(pt2):
        head, tail = pt1, pt2
    else:
        head, tail = pt2, pt1
    
    dr = 1 if head[0] > tail[0] else -1
    dc = 1 if head[1] > tail[1] else -1
    
    r, c = head
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if grid[nr, nc] == 2:
            nr1, nc1 = r + dr, c - dc
            nr2, nc2 = r - dr, c + dc
            if 0 <= nr1 < H and 0 <= nc1 < W and grid[nr1, nc1] == 0:
                dc = -dc
                nr, nc = nr1, nc1
            elif 0 <= nr2 < H and 0 <= nc2 < W and grid[nr2, nc2] == 0:
                dr = -dr
                nr, nc = nr2, nc2
            else:
                break
        if grid[nr, nc] != 0:
            break
        grid[nr, nc] = 3
        r, c = nr, nc
    
    return grid
