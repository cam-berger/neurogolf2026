import numpy as np

def method(grid):
    boxes = [grid[:, 0:4], grid[:, 5:9], grid[:, 10:14]]
    result = []
    for box in boxes:
        zeros = [(r, c) for r in range(4) for c in range(4) if box[r, c] == 0]
        if len(zeros) == 0:
            color = 2
        else:
            rows = set(r for r, c in zeros)
            cols = set(c for r, c in zeros)
            if rows == {1, 2} and cols == {1, 2}:
                color = 8  # center
            elif rows == {1, 2} and cols == {0, 3}:
                color = 3  # sides
            elif rows == {2, 3} and cols == {1, 2}:
                color = 4  # bottom
            elif rows == {0, 1} and cols == {1, 2}:
                color = 4  # top (symmetric guess)
            else:
                color = 2
        result.append([color, color, color])
    return np.array(result)
