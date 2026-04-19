import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    
    def cc(mask):
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
        for i in range(H):
            for j in range(W):
                if mask[i,j] and not visited[i,j]:
                    stack = [(i,j)]
                    cells = []
                    while stack:
                        y,x = stack.pop()
                        if y<0 or y>=H or x<0 or x>=W: continue
                        if visited[y,x] or not mask[y,x]: continue
                        visited[y,x]=True
                        cells.append((y,x))
                        stack += [(y+1,x),(y-1,x),(y,x+1),(y,x-1)]
                    comps.append(cells)
        return comps
    
    # Find 5-components (boxes)
    boxes = []
    for cells in cc(g==5):
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        r0,r1 = min(ys),max(ys)
        c0,c1 = min(xs),max(xs)
        if r1-r0 < 2 or c1-c0 < 2:
            continue
        holes = []
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                if g[r,c]==0:
                    holes.append((r,c))
        if holes:
            hrs = [h[0] for h in holes]
            hcs = [h[1] for h in holes]
            hr0,hc0 = min(hrs),min(hcs)
            norm = tuple(sorted((r-hr0,c-hc0) for r,c in holes))
            boxes.append((holes, norm))
    
    # Non-5, non-0 components. Only colors with exactly ONE component are fill candidates.
    shapes = []
    for color in range(1,10):
        if color == 5: continue
        comps = cc(g==color)
        if len(comps) != 1:
            continue
        cells = comps[0]
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        r0,c0 = min(ys),min(xs)
        norm = tuple(sorted((y-r0,x-c0) for y,x in cells))
        shapes.append([cells, color, norm])
    
    used = set()
    for holes, hnorm in boxes:
        for i,s in enumerate(shapes):
            if i in used: continue
            cells, color, snorm = s
            if snorm == hnorm:
                for (r,c) in holes:
                    g[r,c] = color
                for (y,x) in cells:
                    g[y,x] = 0
                used.add(i)
                break
    
    return g
