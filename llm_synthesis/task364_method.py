def method(grid):
    import numpy as np
    g = np.array(grid)
    out = g.copy()
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    
    def neighbors(r, c):
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and g[nr,nc] == 3:
                result.append((nr,nc))
        return result
    
    def classify(comp):
        deg = {cell: len(neighbors(*cell)) for cell in comp}
        has_junction = any(d >= 3 for d in deg.values())
        endpoints = [c for c,d in deg.items() if d == 1]
        
        if has_junction:
            return 2
        if len(endpoints) != 2:
            # closed loop: treat as "same direction" turns (rectangle-like)
            return 6
        
        # Walk from one endpoint to the other, counting turns
        start = endpoints[0]
        end = endpoints[1]
        path = [start]
        prev = None
        cur = start
        while cur != end:
            nxts = [n for n in neighbors(*cur) if n != prev]
            if not nxts:
                break
            prev = cur
            cur = nxts[0]
            path.append(cur)
        
        # Directions between consecutive cells
        dirs = []
        for k in range(len(path)-1):
            r1,c1 = path[k]
            r2,c2 = path[k+1]
            dirs.append((r2-r1, c2-c1))
        
        # Find turns
        turns = []  # +1 for left, -1 for right (or similar)
        for k in range(len(dirs)-1):
            d1 = dirs[k]
            d2 = dirs[k+1]
            if d1 == d2:
                continue
            # cross product: d1 x d2 = d1[0]*d2[1] - d1[1]*d2[0]
            cross = d1[0]*d2[1] - d1[1]*d2[0]
            turns.append(1 if cross > 0 else -1)
        
        if len(turns) == 0:
            return 1  # straight line, treat as simple
        if len(turns) == 1:
            return 1
        # 2 or more turns
        if all(t == turns[0] for t in turns):
            return 6
        return 2
    
    for i in range(H):
        for j in range(W):
            if g[i,j] == 3 and not visited[i,j]:
                comp = []
                stack = [(i,j)]
                visited[i,j] = True
                while stack:
                    r,c = stack.pop()
                    comp.append((r,c))
                    for nr,nc in neighbors(r,c):
                        if not visited[nr,nc]:
                            visited[nr,nc] = True
                            stack.append((nr,nc))
                
                color = classify(comp)
                for (r,c) in comp:
                    out[r,c] = color
    
    return out
