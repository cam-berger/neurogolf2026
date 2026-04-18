"""Auto-generated rule for task 344."""

from custom_rules.helpers import make_rule, Window

def rule_344(w):
    # 3 cardinally adjacent to a 2 -> 8
    # 2 cardinally adjacent to a 3 -> 0 (consumed)
    # 5 is background/inert. Diagonal adjacency does NOT trigger.
    if w.center == 3 and 2 in w.cardinal:
        return 8
    if w.center == 2 and 3 in w.cardinal:
        return 0
    return w.center


generate = make_rule(344, rule_344, kernel=3)
