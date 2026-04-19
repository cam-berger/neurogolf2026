def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    
    # Flood fill 9s reachable from boundary
    reached = np.zeros_like(g, dtype=bool)
    stack = []
    for i in range(h):
        for j in range(w):
            if (i==0 or i==h-1 or j==0 or j==w-1) and g[i,j]==9:
                if not reached[i,j]:
                    reached[i,j]=True; stack.append((i,j))
    while stack:
        i,j = stack.pop()
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni,nj = i+di,j+dj
            if 0<=ni<h and 0<=nj<w and g[ni,nj]==9 and not reached[ni,nj]:
                reached[ni,nj]=True
                stack.append((ni,nj))
    
    # Label connected components of 1s (4-connectivity)
    labels = np.zeros_like(g, dtype=int)
    cur = 0
    for i in range(h):
        for j in range(w):
            if g[i,j]==1 and labels[i,j]==0:
                cur += 1
                st = [(i,j)]; labels[i,j]=cur
                while st:
                    y,x = st.pop()
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny,nx = y+dy,x+dx
                        if 0<=ny<h and 0<=nx<w and g[ny,nx]==1 and labels[ny,nx]==0:
                            labels[ny,nx]=cur
                            st.append((ny,nx))
    
    # Components adjacent to enclosed (unreached) 9 cells should become 8
    marked = set()
    for i in range(h):
        for j in range(w):
            if g[i,j]==9 and not reached[i,j]:
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj = i+di,j+dj
                    if 0<=ni<h and 0<=nj<w and labels[ni,nj]>0:
                        marked.add(labels[ni,nj])
    
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if labels[i,j] in marked:
                out[i,j] = 8
    return out
