import numpy as np
from symmer.symplectic import QuantumState
from deco import concurrent, synchronized

@concurrent
def _process_GHZ(state, n_samples=100, n_shots = 100_000):
    n_q = state.n_qubits
    fidelity = lambda s: np.square(abs(s.get('0'*n_q, 0) + s.get('1'*n_q, 0)))/2
    bootstrapped = []
    for sample in range(n_samples):
        bs_counts = state.sample_state(n_shots, return_normalized=True).to_dictionary
        bootstrapped.append(
            fidelity(bs_counts)
        )
    return bootstrapped
    

@synchronized
def process_GHZ(data, n_samples, n_shots, use='RAW'):
    data_out = []
    n_qubits = list(data.keys())
    for i, states in data.items():
        counts = states[use]
        state = QuantumState.from_dictionary(counts).normalize_counts
        data_out.append(_process_GHZ(state, n_samples, n_shots))
        
        #data_out[i] = {
        #    'RAW': np.array(_process_GHZ(raw_state, n_samples, n_shots)),
        #    #'MEM': np.array(_process_GHZ(mem_state, n_samples, n_shots))
        #}
    return dict(zip(n_qubits, data_out))