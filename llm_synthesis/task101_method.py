import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    
    def label_components(mask):
        lab = np.zeros(mask.shape, dtype=int)
        cur = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and lab[i, j] == 0:
                    cur += 1
                    stack = [(i, j)]
                    while stack:
                        y, x = stack.pop()
                        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] and lab[y, x] == 0:
                            lab[y, x] = cur
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    if dy == 0 and dx == 0:
                                        continue
                                    stack.append((y + dy, x + dx))
        return lab, cur
    
    mask = g != 0
    lab, n = label_components(mask)
    
    template_id = None
    for i in range(1, n + 1):
        vals = set(g[lab == i].tolist())
        if 1 in vals and 2 in vals:
            template_id = i
            break
    
    ys, xs = np.where(lab == template_id)
    r0, c0 = ys.min(), xs.min()
    r1, c1 = ys.max(), xs.max()
    template = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for y, x in zip(ys, xs):
        template[y - r0, x - c0] = g[y, x]
    
    markers = [(y, x) for y in range(template.shape[0])
               for x in range(template.shape[1]) if template[y, x] == 2]
    
    mask2 = (g == 2) & (lab != template_id)
    lab2, n2 = label_components(mask2)
    blocks = []
    for i in range(1, n2 + 1):
        ys, xs = np.where(lab2 == i)
        y0, x0 = ys.min(), xs.min()
        y1, x1 = ys.max(), xs.max()
        size = max(y1 - y0 + 1, x1 - x0 + 1)
        blocks.append([y0, x0, size, False])
    
    out = g.copy()
    
    for i in range(len(blocks)):
        blk = blocks[i]
        if blk[3]:
            continue
        scale = blk[2]
        for my, mx in markers:
            top_y = blk[0] - my * scale
            top_x = blk[1] - mx * scale
            ok = True
            matched = []
            for my2, mx2 in markers:
                by = top_y + my2 * scale
                bx = top_x + mx2 * scale
                found = None
                for j, b2 in enumerate(blocks):
                    if not b2[3] and b2[0] == by and b2[1] == bx and b2[2] == scale:
                        found = j
                        break
                if found is None:
                    ok = False
                    break
                matched.append(found)
            if ok:
                th, tw = template.shape
                for ty in range(th):
                    for tx in range(tw):
                        v = template[ty, tx]
                        if v == 0:
                            continue
                        for dy in range(scale):
                            for dx in range(scale):
                                yy = top_y + ty * scale + dy
                                xx = top_x + tx * scale + dx
                                if 0 <= yy < H and 0 <= xx < W:
                                    out[yy, xx] = v
                for j in matched:
                    blocks[j][3] = True
                break
    
    return out
