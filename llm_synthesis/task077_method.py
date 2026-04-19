import numpy as np

def method(grid):
    out = grid.copy()
    H, W = grid.shape
    twos = list(zip(*np.where(grid == 2)))
    n = len(twos)
    
    # Union-find to cluster 2-cells by Chebyshev distance <= 2
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
            r1, c1 = twos[i]
            r2, c2 = twos[j]
            if max(abs(r1-r2), abs(c1-c2)) <= 2:
                union(i, j)
    
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(twos[i])
    
    for cells in groups.values():
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                if out[r, c] != 2:
                    out[r, c] = 4
    
    return out
