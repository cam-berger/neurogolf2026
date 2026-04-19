def method(grid):
    import numpy as np
    B = np.hstack([np.fliplr(grid), grid])
    return np.vstack([np.flipud(B), B, np.flipud(B)])
