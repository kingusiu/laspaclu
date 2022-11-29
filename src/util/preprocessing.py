import numpy as np

def min_max_normalize(data):
    return (data - np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))

def min_max_normalize_all_data(data_qcd, data_sig):
    all_data = np.vstack([data_qcd, data_sig])
    min_val = np.min(all_data, axis=0)
    max_val = np.max(all_data, axis=0)
    range_val = max_val - min_val
    return (data_qcd - min_val)/range_val, (data_sig - min_val)/range_val
      