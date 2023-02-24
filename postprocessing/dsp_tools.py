import numpy as np
from symmer.symplectic import QuantumState
import statsmodels.api as sm
from deco import concurrent, synchronized

I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])

@concurrent
def _bootstrap_postselection(counts, n_resamples, n_shots):
    energies_out = []
    psi_measured = QuantumState.from_dictionary(counts).normalize_counts
    for sample in range(n_resamples):
        psi_resampled = psi_measured.sample_state(n_shots).normalize_counts
        p0 = psi_resampled.to_dictionary.get('0'*3+'0', 0)**2
        p1 = psi_resampled.to_dictionary.get('0'*3+'1', 0)**2
        if p0 + p1 != 0:
            energies_out.append((p0-p1)/(p0+p1))
        else:
            energies_out.append(0)
    return energies_out

@synchronized
def bootstrap_postselection(noise_factor, input_data, n_resamples=1000, use='MEM'):
    basis_measurements = input_data[str(noise_factor)]
    expvals = {}
    for basis in ['X', 'Z']:
        termwise_expvals = []
        mem_data = basis_measurements.get(basis).get(use)
        shot_vec = basis_measurements.get(basis).get('n_shots_by_circuit')
        for counts, shots in zip(mem_data, shot_vec):
            termwise_expvals.append(_bootstrap_postselection(counts=counts, n_resamples=n_resamples, n_shots=shots))
        expvals[basis] = np.array(termwise_expvals)
    return expvals

@concurrent
def _bootstrap_tomography(density_resamples):
    tp_expvals_termwise_X=[]
    tp_expvals_termwise_Z=[]
    
    for rho in density_resamples:
        eigvals, eigvecs = np.linalg.eig(rho)
        largest_eigvec = eigvecs[:,np.argmax(eigvals)].reshape([-1,1])
        exp_X = (largest_eigvec.T.conjugate() @ X @ largest_eigvec)[0][0]
        exp_Z = (largest_eigvec.T.conjugate() @ Z @ largest_eigvec)[0][0] 
        tp_expvals_termwise_X.append(exp_X)
        tp_expvals_termwise_Z.append(exp_Z)
        
    return tp_expvals_termwise_X, tp_expvals_termwise_Z

@synchronized
def bootstrap_tomography(densities):
    data_dump = []
    for resampled_density_matrices in densities:
        data_dump.append(_bootstrap_tomography(resampled_density_matrices))
    dd = np.array(data_dump)
    tp_expvals = {'X':dd[:,0,:], 'Z':dd[:,1,:]}
    return tp_expvals
    
def ratio_variance(A,B):
    """ Implemented as in https://en.wikipedia.org/wiki/Ratio_distribution#A_transformation_to_Gaussianity
    Given estimators A,B what is the variance of the ratio estimator A/B? Assumed A and B independent.
    """
    E_A2 = np.mean(np.square(A), axis=1)
    E_B2 = np.mean(np.square(1/B), axis=1)
    E2_A = np.square(np.mean(A, axis=1))
    E2_B = np.square(np.mean(1/B, axis=1))

    return E_A2 * E_B2 - E2_A * E2_B

def _get_dsp_estimates(X_dsp, Z_dsp, use_TP=True, threshold=0.1):
    if not use_TP:
        X_dsp[X_dsp<0]=0
        dsp_estimates = np.mean(Z_dsp/(1+X_dsp), axis=1)
        dsp_variances = ratio_variance(Z_dsp, 1+X_dsp)
        return dsp_estimates, dsp_variances
    else:
        densities = (I + np.multiply.outer(X_dsp, X) + np.multiply.outer(Z_dsp, Z))/2
        bs_tp = bootstrap_tomography(densities)
    
        X_dsp_tp = bs_tp['X'].copy()
        X_dsp_tp[X_dsp_tp<0]=0

        Z_dsp_tp = bs_tp['Z'].copy()
        use_tp = abs(np.mean(Z_dsp, axis=1))>threshold
        Z_dsp_tp[~use_tp] = Z_dsp[~use_tp]
        
        dsp_tp_estimates = np.mean(Z_dsp_tp/(1+X_dsp_tp), axis=1)
        dsp_tp_variances = ratio_variance(Z_dsp_tp, 1+X_dsp_tp)
        return dsp_tp_estimates, dsp_tp_variances

def get_dsp_estimates(input_data, noise_factor=0, n_resamples = 1000, use='MEM', use_TP=True, threshold=0.1):

    bs = bootstrap_postselection(noise_factor, input_data, n_resamples, use)
    X_dsp = bs['X'].copy()
    Z_dsp = bs['Z'].copy()

    return _get_dsp_estimates(X_dsp, Z_dsp, use_TP=use_TP, threshold=threshold)

@concurrent
def _extrapolate(noisy_resamples, Lambdas):
    extrapolated_values = []
    for noisy_sample in noisy_resamples:
        ols = sm.OLS(endog=noisy_sample, exog=Lambdas)
        zne_ols_model = ols.fit()
        extrapolated_values.append(zne_ols_model.params[0])
    return extrapolated_values

@synchronized
def extrapolate(termwise_bs_data_for_extrapolation, Lambdas):
    Lambdas = sm.add_constant(Lambdas)
    termwise_bs_data = []
    for term_bs_data in termwise_bs_data_for_extrapolation:
        termwise_bs_data.append(_extrapolate(term_bs_data, Lambdas))
    return np.array(termwise_bs_data)

def get_dsp_zne_estimates(input_data, n_resamples = 1000, use='MEM', use_TP=True, threshold=0.1):
    ZNE_factors = list(map(int, input_data.keys()))
    bs = {n:bootstrap_postselection(str(n), input_data, n_resamples, use) 
            for n in ZNE_factors}
    
    Z_termwise_bs_data_for_extrapolation = np.array([bs[n]['Z'].T for n in ZNE_factors]).T
    X_termwise_bs_data_for_extrapolation = np.array([bs[n]['X'].T for n in ZNE_factors]).T
    
    Z_termwise_bs_data = extrapolate(Z_termwise_bs_data_for_extrapolation, ZNE_factors)
    X_termwise_bs_data = extrapolate(X_termwise_bs_data_for_extrapolation, ZNE_factors)
    
    return _get_dsp_estimates(
        X_termwise_bs_data, 
        Z_termwise_bs_data, 
        use_TP=use_TP, 
        threshold=threshold
    )