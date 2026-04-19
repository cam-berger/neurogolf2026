def method(grid):
    import numpy as np
    g = grid
    H, W = g.shape
    
    # Group pixels by color
    colors = {}
    for c in np.unique(g):
        if c == 0:
            continue
        pixels = list(zip(*np.where(g == c)))
        colors[int(c)] = pixels
    
    # Template: color with most pixels
    template_color = max(colors, key=lambda c: len(colors[c]))
    tpix = colors[template_color]
    rs = [p[0] for p in tpix]
    cs = [p[1] for p in tpix]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    
    tpat = np.zeros((h, w), dtype=bool)
    for r, c in tpix:
        tpat[r - r0, c - c0] = True
    
    out = g.copy()
    
    steps_r = [-(h+1), 0, h+1]
    steps_c = [-(w+1), 0, w+1]
    
    for color, pixels in colors.items():
        if color == template_color:
            continue
        
        best = None
        for dr in steps_r:
            for dc in steps_c:
                if dr == 0 and dc == 0:
                    continue
                found_k = None
                valid = True
                for pr, pc in pixels:
                    matched_k = None
                    for k in range(1, max(H, W) + 1):
                        tr = pr - r0 - k * dr
                        tc = pc - c0 - k * dc
                        if 0 <= tr < h and 0 <= tc < w and tpat[tr, tc]:
                            matched_k = k
                            break
                    if matched_k is None:
                        valid = False
                        break
                    if found_k is None:
                        found_k = matched_k
                    elif found_k != matched_k:
                        valid = False
                        break
                if valid and found_k is not None:
                    best = (dr, dc, found_k)
                    break
            if best is not None:
                break
        
        if best is None:
            continue
        
        dr, dc, _ = best
        k = 1
        while True:
            rr = r0 + k * dr
            cc = c0 + k * dc
            if rr + h <= 0 or rr >= H or cc + w <= 0 or cc >= W:
                break
            for dy in range(h):
                for dx in range(w):
                    if tpat[dy, dx]:
                        ry = rr + dy
                        cx = cc + dx
                        if 0 <= ry < H and 0 <= cx < W:
                            out[ry, cx] = color
            k += 1
    
    return out
