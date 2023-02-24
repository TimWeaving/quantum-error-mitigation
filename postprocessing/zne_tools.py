import numpy as np
from symmer.symplectic import QuantumState
from deco import concurrent, synchronized

@concurrent
def _bootstrap_raw_process(counts, Z_vec, n_resamples, n_shots):
    energies_out = []
    psi_measured = QuantumState.from_dictionary(counts).normalize_counts
    for sample in range(n_resamples):
        psi_resampled = psi_measured.sample_state(n_shots).normalize_counts
        sign_flips = (-1)**np.sum(psi_resampled.state_matrix.astype(bool) & Z_vec, axis=1)
        expval = np.sum(np.square(psi_resampled.state_op.coeff_vec) * sign_flips)
        energies_out.append(expval)
    return energies_out

@synchronized
def bootstrap_raw_process(noise_factor, non_I_matrix, input_data, n_resamples=1000, use='MEM'):
    basis_measurements = input_data[str(noise_factor)]
    
    termwise_expvals = []
    mem_data = basis_measurements.get(use)
    shot_vec = basis_measurements.get('n_shots_by_circuit')
    for Z_vec, counts, shots in zip(non_I_matrix, mem_data, shot_vec):
        termwise_expvals.append(
            _bootstrap_raw_process(
                counts=counts, Z_vec=Z_vec, n_resamples=n_resamples, n_shots=shots
            )
        )
    
    return np.array(termwise_expvals)

def get_energies(input_data, ham, noise_factor=0, n_resamples = 1000, use='MEM', termwise=True):
    
    non_I_block = ham.X_block | ham.Z_block
    expval_estimates = bootstrap_raw_process(
        noise_factor, non_I_block, input_data, n_resamples, use
    )
    mean_expvals = np.mean(expval_estimates, axis=1)
    var_expvals  = np.var(expval_estimates, axis=1)

    if termwise:
        return mean_expvals, var_expvals
    else:
        mean_energy = np.sum(mean_expvals * ham.coeff_vec)
        var_energy = np.sum(var_expvals * np.square(ham.coeff_vec))
        return mean_energy.real, var_energy.real

def get_energies_raw(input_data, ham, noise_factor=0, n_resamples = 1000, use='MEM', termwise=True):
    
    non_I_block = ham.X_block | ham.Z_block
    expval_estimates = bootstrap_raw_process(
        noise_factor, non_I_block, input_data, n_resamples, use
    )
    if termwise:
        return expval_estimates
    else:
        return np.sum(expval_estimates.T * ham.coeff_vec, axis=1).real