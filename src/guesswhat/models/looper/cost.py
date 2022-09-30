import numpy as np
#assuming the questions are tokenized as lists of tokens

def IOU_dist(q1,q2):
    sim = len(set(q1) & set(q_2)) / len(set(q1).union(set(q_2)))
    return 1 - sim

def


def cost_fn(q1,q2,max_length):
    exponent = (max_length/max_length-1)*(indexes)/(max_length-1)
    return np.power(2,exponent) -1