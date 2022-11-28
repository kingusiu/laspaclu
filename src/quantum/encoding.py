import numpy as np
from qiskit import QuantumCircuit


def amplitude_encoding_circuit(*vv): 
    '''
        amplitude encoding of real vectors
        vv: tuple of vectors to be encoded
        returns encoding circuit
    '''

    sz = int(np.log2(len(vv[0])))
    
    qc = QuantumCircuit(len(vv)*sz)

    for i, v in enumerate(vv):
        qc.initialize(v, range(i*sz, (i+1)*sz))
    
    return qc
