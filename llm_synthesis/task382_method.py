def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    
    eights = list(zip(*np.where(grid == 8)))
    twos = list(zip(*np.where(grid == 2)))
    
    def edge_of(pts):
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        if all(r == 0 for r in rs): return 'top'
        if all(r == h-1 for r in rs): return 'bottom'
        if all(c == 0 for c in cs): return 'left'
        if all(c == w-1 for c in cs): return 'right'
        return None
    
    e8 = edge_of(eights)
    e2 = edge_of(twos)
    
    # direction of travel from 8-edge into the grid
    travel = {'top': (1, 0), 'bottom': (-1, 0), 'left': (0, 1), 'right': (0, -1)}
    td = travel[e8]
    # direction to shift away from the 2-edge
    shift_away = {'top': (1, 0), 'bottom': (-1, 0), 'left': (0, 1), 'right': (0, -1)}
    sd = shift_away[e2]
    
    # markers' coordinate along the travel axis
    if td[0] != 0:
        marker_coords = set(r for r, c in twos)
    else:
        marker_coords = set(c for r, c in twos)
    
    for r0, c0 in eights:
        r, c = r0, c0
        while 0 <= r < h and 0 <= c < w:
            if g[r, c] == 0:
                g[r, c] = 8
            nr = r + td[0]
            nc = c + td[1]
            na = nr if td[0] != 0 else nc
            if na in marker_coords:
                nr += sd[0]
                nc += sd[1]
            r, c = nr, nc
    
    return g
