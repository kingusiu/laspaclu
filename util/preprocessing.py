import numpy as np

def min_max_normalize(data):
    return (data - np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))