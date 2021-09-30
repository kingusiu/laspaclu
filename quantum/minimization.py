from collections import OrderedDict
import math
import numpy as np
from qiskit import Aer, execute

import oracles as ora
import grover as gro


def measure_by_prob(counts, n):
    # sort indices by prob
    sorted_counts = OrderedDict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    # take most probable value (if it is in the limits of array)
    for k in sorted_counts.keys():
        if int(k,2) < n:
            return int(k,2)
    print('no valid index found')
    return -1


def duerr_hoyer_minimization(distances):
    
    nn = int(math.floor(math.log2(len(distances)) + 1))
    iter_n = math.ceil(np.sqrt(2**nn)) # suggested in duerr & hoyer 
    
    threshold_oracles = ora.create_threshold_oracle_set(distances)
    
    # init first random threshold
    idx = np.random.randint(0, len(distances)-1)
    
    # iterate
    for _ in range(iter_n):
        
        # pick next threshold
        threshold = distances[idx]
        print('next minimum guess: dist[{}] = {}'.format(idx, threshold))
        marked_n = len(ora.get_indices_to_mark(distances, threshold))
        
        # create oracle combi and grover algo
        oracle_qc = ora.create_oracle_lincombi(threshold, distances, threshold_oracles)
        grover_qc = gro.grover_circuit(nn, oracle_qc, marked_n)
        
        # apply grover algo (only one shot to get a true collapsed value)
        simulator = Aer.get_backend('qasm_simulator')
        counts = execute(grover_qc, backend=simulator, shots=1000).result().get_counts(grover_qc)
        idx = measure_by_prob(counts, len(distances))
        
    print('final minimum found dist[{}] = {}'.format(idx, distances[idx]))
    return distances[idx]

