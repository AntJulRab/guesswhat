import numpy as np
#assuming the questions are tokenized as lists of tokens

def IOU_dist(q1,q2):
    sim = len(set(q1) & set(q2)) / len(set(q1).union(set(q2)))
    return 1 - sim


def cost_fn(q1,q2,max_length,a,b):
    exponent = (max_length/max_length-1)*(np.absolute(a-b))/(max_length-1)
    return np.power(2,exponent) -1
