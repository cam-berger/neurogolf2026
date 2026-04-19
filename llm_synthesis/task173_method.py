def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    
    cells = [(r, c, int(g[r,c])) for r in range(H) for c in range(W) if g[r,c] != 0]
    n = len(cells)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in range(i+1, n):
            if abs(cells[i][0]-cells[j][0]) <= 2 and abs(cells[i][1]-cells[j][1]) <= 2:
                union(i, j)
    clusters = {}
    for i, c in enumerate(cells):
        clusters.setdefault(find(i), []).append(c)
    shapes = list(clusters.values())
    
    def classify(shape):
        rs = [s[0] for s in shape]
        cs = [s[1] for s in shape]
        r_min, r_max = min(rs), max(rs)
        c_min, c_max = min(cs), max(cs)
        positions = {(s[0], s[1]): s[2] for s in shape}
        nc = len(shape)
        if nc == 5 and r_max-r_min == 2 and c_max-c_min == 2:
            cr = (r_min+r_max)//2
            cc = (c_min+c_max)//2
            plus_p = {(cr,cc),(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)}
            if set(positions.keys()) == plus_p:
                center = positions[(cr,cc)]
                outer = [positions[p] for p in plus_p if p != (cr,cc)]
                if len(set(outer)) == 1:
                    return ('plus', center, outer[0], (cr,cc))
            x_p = {(cr,cc),(cr-1,cc-1),(cr-1,cc+1),(cr+1,cc-1),(cr+1,cc+1)}
            if set(positions.keys()) == x_p:
                center = positions[(cr,cc)]
                outer = [positions[p] for p in x_p if p != (cr,cc)]
                if len(set(outer)) == 1:
                    return ('X', center, outer[0], (cr,cc))
        if nc == 3:
            if r_min == r_max and c_max-c_min == 2:
                cr, cc = r_min, (c_min+c_max)//2
                hp = {(cr,cc-1),(cr,cc),(cr,cc+1)}
                if set(positions.keys()) == hp:
                    center = positions[(cr,cc)]
                    o = [positions[(cr,cc-1)], positions[(cr,cc+1)]]
                    if o[0] == o[1]:
                        return ('H', center, o[0], (cr,cc))
            if c_min == c_max and r_max-r_min == 2:
                cr, cc = (r_min+r_max)//2, c_min
                vp = {(cr-1,cc),(cr,cc),(cr+1,cc)}
                if set(positions.keys()) == vp:
                    center = positions[(cr,cc)]
                    o = [positions[(cr-1,cc)], positions[(cr+1,cc)]]
                    if o[0] == o[1]:
                        return ('V', center, o[0], (cr,cc))
        return None
    
    templates = []
    fragments = []
    for sh in shapes:
        cls = classify(sh)
        if cls is not None:
            templates.append(cls)
        else:
            fragments.append(sh)
    
    offsets_map = {
        'plus': [(-1,0),(1,0),(0,-1),(0,1)],
        'X': [(-1,-1),(-1,1),(1,-1),(1,1)],
        'H': [(0,-1),(0,1)],
        'V': [(-1,0),(1,0)],
    }
    
    for frag in fragments:
        frag_colors = set(s[2] for s in frag)
        match = None
        for (ttype, cc_color, oc_color, _) in templates:
            t_colors = {cc_color, oc_color}
            if frag_colors.issubset(t_colors):
                match = (ttype, cc_color, oc_color)
                if frag_colors == {cc_color} or frag_colors == {oc_color} or frag_colors == t_colors:
                    break
        if match is None:
            continue
        ttype, cc_color, oc_color = match
        offs = offsets_map[ttype]
        
        if frag_colors == {cc_color}:
            cr, cc = frag[0][0], frag[0][1]
            for dr, dc in offs:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < H and 0 <= nc < W and g[nr,nc] == 0:
                    g[nr, nc] = oc_color
        elif frag_colors == {oc_color}:
            rs = [s[0] for s in frag]
            cs = [s[1] for s in frag]
            cr = (min(rs)+max(rs))//2
            cc = (min(cs)+max(cs))//2
            g[cr, cc] = cc_color
        else:
            cr, cc = None, None
            for r, c, v in frag:
                if v == cc_color:
                    cr, cc = r, c
                    break
            if cr is not None:
                for dr, dc in offs:
                    nr, nc = cr+dr, cc+dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr,nc] == 0:
                        g[nr, nc] = oc_color
    
    return g
