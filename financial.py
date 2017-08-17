import numpy as np

def get_coupon(rm_in, freq=1):

    rm = rm_in.copy()
    
    rm_t = (1.0 + rm) ** (1.0 / float(freq)) - 1.0
    q_star = rm_t / (1 - (1.0 / ((1.0 + rm_t) ** (30 * freq))))
    
    return q_star

