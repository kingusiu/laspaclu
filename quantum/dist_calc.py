import numpy as np

# import Qiskit
from qiskit import Aer, IBMQ, execute, assemble, transpile
from qiskit import QuantumCircuit


def normalize(v):
    return v / np.linalg.norm(v)

def calc_z(a, b) -> float:
    ''' z = |a|**2 + |b|**2 '''
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)
    return a_mag**2 + b_mag**2


def psi_amp(a, b):
    ''' prepare amplitudes for state psi '''
    
    a_norm = normalize(a)
    b_norm = normalize(b)
    
    return np.hstack([a_norm, b_norm]) * (1/np.sqrt(2))


def phi_amp(a, b):
    ''' prepare amplitudes for state phi '''


    z = calc_z(a, b)
    a_mag =  np.linalg.norm(a)
    b_mag =  np.linalg.norm(b)
    
    return np.hstack([a_mag, -b_mag])/np.sqrt(z)


def psi_circuit(a, b):

    amp = psi_amp(a, b) # 2*n amplitudes 1/sqrt(2) (a0, ..., an, b0, ..., bn)
    sz = int(np.log2(len(amp)))
    
    qc = QuantumCircuit(sz) # 2 qubits if a,b in R^2

    qc.initialize(amp, range(sz))
    
    return qc


def phi_circuit(a, b) -> QuantumCircuit:
    ''' prepare subcircuit for state phi '''

    amp = phi_amp(a, b) # 2 amplitudes 1/sqrt(z) (|a|, |b|)
    sz = 1 # always 2 amplitudes
    
    qc = QuantumCircuit(sz) # 2 qubits if a,b in R^2

    qc.initialize(amp, [0])
    
    return qc


def overlap_circuit(a, b) -> QuantumCircuit:
    ''' full overlap circuit < phi | psi > '''
    
    qc = QuantumCircuit(1+2+1, 1) # ancilla + psi + phi

    psi = psi_circuit(a, b)
    qc.append(psi, [1,2])
    phi = phi_circuit(a, b)
    qc.append(phi, [3])
    
    qc.barrier()
    
    qc.h(0)
    qc.cswap(0, 2, 3) # perform test on v1 ancilla alone
    qc.h(0)
        
    return qc


def run_circuit(qc):
    simulator = Aer.get_backend('qasm_simulator')
    return execute(qc, backend=simulator, shots=1000).result().get_counts(qc)


def calc_overlap(answer, state='0'):
    ''' calculate overlap from experiment measurements '''

    shots = answer[state] if len(answer) == 1 else answer['0']+answer['1']
    return np.abs(counts[state]/1000 - 0.5)*2


def calc_dist(answer, z, state='0'):
    ''' calculate distance proportional to |a-b|**2 '''
    return calc_overlap(answer, state)*2*z