"""Auto-generated rule for task 352."""

from custom_rules.helpers import make_rule, Window

def rule_352(w):
    # Each 2 gets a 3x3 halo of 1s. Other colors (6, 8, 3, ...) are inert.
    if w.center == 0 and 2 in w.neighbors:
        return 1
    return w.center


generate = make_rule(352, rule_352, kernel=3)
