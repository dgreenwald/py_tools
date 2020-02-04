import numpy as np

def get_coupon(rm, freq=1, years=30):

    rm_t = (1.0 + rm) ** (1.0 / float(freq)) - 1.0
    q_star = rm_t / (1 - (1.0 / ((1.0 + rm_t) ** (years * freq))))
    
    return q_star
