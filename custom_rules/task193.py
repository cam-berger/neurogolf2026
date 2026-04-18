"""Auto-generated rule for task 193."""

from custom_rules.helpers import make_rule, Window

def rule_193(w):
    # Keep a cell only if it's part of a solid 2x2 block of same color.
    # (Scattered cells and L-protrusions get removed.)
    c = w.center
    if c == 0:
        return 0
    if (w.top_left == c and w.top == c and w.left == c):
        return c
    if (w.top == c and w.top_right == c and w.right == c):
        return c
    if (w.left == c and w.bottom == c and w.bottom_left == c):
        return c
    if (w.right == c and w.bottom == c and w.bottom_right == c):
        return c
    return 0


generate = make_rule(193, rule_193, kernel=3)
