"""Auto-generated rule for task 283."""

from custom_rules.helpers import make_rule, Window

def rule_283(w):
    # 5s get recolored based on how many of their 8 neighbors are also 5:
    #   3 -> 1 (outer corner of block)
    #   5 -> 4 (outer edge)
    #   8 -> 2 (interior)
    # 0s stay 0.
    if w.center == 5:
        count = w.neighbor_count(5)
        if count == 3:
            return 1
        if count == 5:
            return 4
        if count == 8:
            return 2
    return w.center


generate = make_rule(283, rule_283, kernel=3)
