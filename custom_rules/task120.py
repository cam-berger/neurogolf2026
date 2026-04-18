"""Auto-generated rule for task 120."""

from custom_rules.helpers import make_rule, Window

def rule_120(w):
    # Blocks of solid color get hollowed out:
    #   - 0 background stays 0
    #   - block borders keep their color
    #   - interior cells (all 8 neighbors match center) become 8
    if w.center == 0:
        return 0
    if all(n == w.center for n in w.neighbors):
        return 8
    return w.center


generate = make_rule(120, rule_120, kernel=3)
