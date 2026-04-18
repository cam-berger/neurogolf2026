"""Auto-generated rule for task 97."""

from custom_rules.helpers import make_rule, Window

def rule_97(w):
    # Isolated colored cells (0 or 1 colored neighbors) get erased to 0                                                                                                               
    # Connected colored cells (2+ colored neighbors) stay                                                                                                                             
    if w.center != 0:                                                                                                                                                                 
        count = sum(c for c in w.neighbors if c not in (-1, 0))                                                                                                                       
        if count >= w.center:                                    
            return w.center                                                                                                                                                           
        return 0  # erase isolated cells                  
    return w.center                                                                                                                                                                   


generate = make_rule(97, rule_97, kernel=3)
