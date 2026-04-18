"""Auto-generated rule for task 81."""

from custom_rules.helpers import make_rule, Window

def rule_81(w):
    # if 3 sides or diagonals are 8, change the center to 1
    if w.center == 0:
        if w.top == 8 and w.right == 8 and w.top_right == 8:
            return 1
        elif w.top == 8 and w.left == 8 and w.top_left== 8:
            return 1
        elif w.left == 8 and w.bottom_left == 8 and w.bottom == 8:
            return 1
        elif w.right == 8 and w.bottom == 8 and w.bottom_right == 8:
            return 1
    return w.center  


generate = make_rule(81, rule_81, kernel=3)
