import os
from qiskit.providers.ibmq.runtime import RuntimeDecoder
import json
import numpy as np
from copy import deepcopy
import statsmodels.api as sm
from symmer.symplectic import QuantumState, PauliwordOp
from .zne_tools import *
from .dsp_tools import *
# Hamiltonian data - HARDCODED FOR NOW

ham_cs_dict = {'III': (-453.09074243348186+0j), 'IIZ': (0.3938279139008882+0j), 'IZI': (0.6207540014723985+0j), 'IZZ': (0.8467205227384085+0j), 'ZII': (0.6207540014723985+0j), 'ZIZ': (0.8467205227384084+0j), 'ZZI': (0.2583692120830724+0j), 'ZZZ': (0.23804868510762295+0j), 'IIX': (-0.015457549840655714+0j), 'IZX': (0.015457549840655714-0j), 'ZIX': (0.015457549840655714+0j), 'ZZX': (-0.015457549840655714+0j), 'IXI': (0.004504380403727218+0j), 'IXZ': (-0.004504380403727218+0j), 'ZXI': (0.06195857194191802+0j), 'ZXZ': (-0.06195857194191802+0j), 'IXX': (-0.009644441995559343+0j), 'IYY': (-0.009644441995559343+0j), 'ZXX': (0.009644441995559343+0j), 'ZYY': (0.009644441995559343+0j), 'XII': (-0.004504380403727218+0j), 'XIZ': (0.004504380403727218+0j), 'XZI': (-0.06195857194191803+0j), 'XZZ': (0.061958571941918006+0j), 'XIX': (0.009644441995559343+0j), 'XZX': (-0.009644441995559343+0j), 'YIY': (0.009644441995559343-0j), 'YZY': (-0.009644441995559343+0j), 'YYI': (-0.05559874279713038+0j), 'YYZ': (0.05559874279713038+0j), 'XXX': (-0.03521948529328592+0j), 'XYY': (-0.03521948529328592+0j), 'YXY': (-0.03521948529328592+0j), 'YYX': (0.03521948529328592-0j)}
num_op_cs_dict = {'III': (17+0j), 'IIZ': (-1+0j), 'IZI': (-0.5+0j), 'IZZ': (-0.5+0j), 'ZII': (-0.5+0j), 'ZIZ': (-0.5+0j)}
spin_op_cs_dict = {'IZI': (0.25+0j), 'IZZ': (0.25-0j), 'ZII': (-0.25+0j), 'ZIZ': (-0.25+0j)}

n_particles=18
ham_cs = PauliwordOp.from_dictionary(ham_cs_dict)
num_op_cs = PauliwordOp.from_dictionary(num_op_cs_dict)
spin_op_cs = PauliwordOp.from_dictionary(spin_op_cs_dict)

# Wrapper functions for performing the postprocessing
qem_directory =  os.getcwd()
qem_data_dir = os.path.join(qem_directory, 'data/QEM_benchmark')

def get_data(system, suffix=None):
    RD = RuntimeDecoder()

    filename = f'{qem_data_dir}/{system}_QEM.json'

    with open(filename, 'r') as f:
        data_dict = json.load(f)

    data_raw = RD.decode(data_dict['RAW'])
    data_zne = RD.decode(data_dict['ZNE'])
    data_dsp = RD.decode(data_dict['DSP'])
    data_dsp_zne = RD.decode(data_dict['DSP+ZNE'])
    
    return data_raw, data_zne, data_dsp, data_dsp_zne

def update_data(data_in, use='RAW', noise_amp=0):
    updated = []
    for counts, term in zip(data_in['results']['experiment_data'][str(noise_amp)][use], ham_cs[1:]):
        
        if np.all(term.X_block==0):
            psi_counts = QuantumState.from_dictionary(counts)

            new_counts = []

            for bvec in psi_counts:
                bvec_copy = bvec.copy()
                bvec_copy.state_op.coeff_vec[0]=1
                if (
                    bvec_copy.dagger*num_op_cs*bvec_copy == n_particles and
                    bvec_copy.dagger*spin_op_cs*bvec_copy == 0
                ):
                    new_counts.append(bvec)
            new_counts = sum(new_counts)
            new_counts = new_counts * (1/np.sum(new_counts.state_op.coeff_vec))
            new_counts = new_counts.to_dictionary
        else:
            new_counts = counts
        updated.append(new_counts)
    data_in['results']['experiment_data'][str(noise_amp)][use] = updated

def zero_noise_extrapolation(
        data_in, 
        coefficients, 
        use='RAW', 
        n_resamples=1000,
        use_DSP = False,
        use_TP  = False,
        threshold=0.1
    ):
    """
    """
    Lambdas = data_in['results']['ZNE_factors']

    if use_DSP:
        raw_expvals = np.array([
            get_dsp_estimates(
                noise_factor=i,
                input_data=data_in['results']['experiment_data'], 
                n_resamples=n_resamples,
                use=use,
                use_TP = use_TP,
                threshold=threshold
            )
            for i in Lambdas
        ])
    else:
        non_I_block = ham_cs[1:].X_block | ham_cs[1:].Z_block
        raw_expvals = np.array([
            bootstrap_raw_process(
                i, non_I_block, data_in['results']['experiment_data'], use=use, n_resamples=n_resamples
            )
            for i in Lambdas
        ])

    noisy_estimates = np.sum(np.mean(raw_expvals, axis=2) * coefficients, axis=1)
    noisy_variances = np.sum(np.var(raw_expvals, axis=2) * np.square(coefficients), axis=1)
    
    X = sm.add_constant(Lambdas)
    wls = sm.WLS(endog=noisy_estimates, exog=X, weights=1/noisy_variances)
    ols = sm.OLS(endog=noisy_estimates, exog=X)
    
    zne_wls_model = wls.fit()
    zne_ols_model = ols.fit()
    
    zne_wls_curve = np.poly1d(zne_wls_model.params[::-1])
    zne_wls_estimate = zne_wls_model.params[0]
    zne_wls_variance = np.square(zne_wls_model.HC0_se[0])*zne_wls_model.df_resid
    
    zne_ols_curve = np.poly1d(zne_ols_model.params[::-1])
    zne_ols_estimate = zne_ols_model.params[0]
    zne_ols_variance = np.square(zne_ols_model.HC0_se[0])*zne_ols_model.df_resid
    
    output = {
        'WLS':{
            'estimate':zne_wls_estimate.real,'variance':zne_wls_variance.real,
            'rsquared':zne_wls_model.rsquared.real,'curve':zne_wls_curve,
        },
        'OLS':{
            'estimate':zne_ols_estimate.real,'variance':zne_ols_variance.real,
            'rsquared':zne_ols_model.rsquared.real,'curve':zne_ols_curve,
        },
        'lambdas':Lambdas,
        'noisy_estimates':noisy_estimates,
        'noisy_variances':noisy_variances,
        'raw_data': raw_expvals
    }
    
    return output


def process_all(system, n_resamples=100, threshold=0.1, threshold_2=0.1):
    print(f'Processing quantum experiment data from {system}...')
    
    data_raw, data_zne, data_dsp, data_dsp_zne = get_data(system)
    
    #######################################################################
    ################ POST PROCESS #########################################
    #######################################################################
    
    # RAW
    raw_expvals, raw_vars = get_energies(data_raw['results']['experiment_data'], ham_cs[1:], use='RAW', n_resamples=n_resamples)
    raw_nrg, raw_std = np.sum(raw_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(raw_vars * np.square(ham_cs.coeff_vec[1:])))
    print('RAW complete')
    # ZNE
    zne_out = zero_noise_extrapolation(data_in=data_zne, coefficients=ham_cs.coeff_vec[1:], use='RAW', n_resamples=n_resamples)
    zne_nrg, zne_std = zne_out['WLS']['estimate'], np.sqrt(zne_out['WLS']['variance'])
    print('ZNE complete')
    # MEM
    mem_expvals, mem_vars = get_energies(data_raw['results']['experiment_data'], ham_cs[1:], use='MEM', n_resamples=n_resamples)
    mem_nrg, mem_std = np.sum(mem_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(mem_vars * np.square(ham_cs.coeff_vec[1:])))
    print('MEM complete')
    # MEM+ZNE
    mem_zne_out = zero_noise_extrapolation(data_in=data_zne, coefficients=ham_cs.coeff_vec[1:], use='MEM', n_resamples=n_resamples)
    mem_zne_nrg, mem_zne_std = mem_zne_out['WLS']['estimate'], np.sqrt(mem_zne_out['WLS']['variance'])
    print('MEM+ZNE complete')
    # SYM
    sym_corrected_data = deepcopy(data_raw)
    update_data(sym_corrected_data, use='RAW')
    update_data(sym_corrected_data, use='MEM')
    sym_corrected_data_zne = deepcopy(data_zne)
    for i in data_zne['results']['ZNE_factors']:
        update_data(sym_corrected_data_zne, use='RAW', noise_amp=i)
        update_data(sym_corrected_data_zne, use='MEM', noise_amp=i)
        
    sym_expvals, sym_vars = get_energies(sym_corrected_data['results']['experiment_data'], ham_cs[1:], use='RAW', n_resamples=n_resamples)
    sym_nrg, sym_std = np.sum(sym_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(sym_vars * np.square(ham_cs.coeff_vec[1:])))
    print('SYM complete')
    # SYM+ZNE
    sym_zne_out = zero_noise_extrapolation(data_in=sym_corrected_data_zne, coefficients=ham_cs.coeff_vec[1:], use='RAW', n_resamples=n_resamples)
    sym_zne_nrg, sym_zne_std = sym_zne_out['WLS']['estimate'], np.sqrt(sym_zne_out['WLS']['variance'])
    print('SYM+ZNE comlete')
    # MEM+SYM
    mem_sym_expvals, mem_sym_vars = get_energies(sym_corrected_data['results']['experiment_data'], ham_cs[1:], use='MEM', n_resamples=n_resamples)
    mem_sym_nrg, mem_sym_std = np.sum(mem_sym_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(mem_sym_vars * np.square(ham_cs.coeff_vec[1:])))
    print('MEM+SYM complete')
    # MEM+SYM+ZNE
    mem_sym_zne_out = zero_noise_extrapolation(data_in=sym_corrected_data_zne, coefficients=ham_cs.coeff_vec[1:], use='MEM', n_resamples=n_resamples)
    mem_sym_zne_nrg, mem_sym_zne_std = mem_sym_zne_out['WLS']['estimate'], np.sqrt(mem_sym_zne_out['WLS']['variance'])
    print('MEM+SYM+ZNE complete')
    # DSP
    dsp_expvals, dsp_vars = get_dsp_estimates(noise_factor=0,input_data=data_dsp['results']['experiment_data'], n_resamples=n_resamples,use='RAW',use_TP = False, threshold=threshold)
    dsp_nrg, dsp_std = np.sum(dsp_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(dsp_vars * np.square(ham_cs.coeff_vec[1:])))
    print('DSP complete')
    # DSP+TP
    dsp_tp_expvals, dsp_tp_vars = get_dsp_estimates(noise_factor=0,input_data=data_dsp['results']['experiment_data'], n_resamples=n_resamples,use='RAW',use_TP = True, threshold=threshold)
    dsp_tp_nrg, dsp_tp_std = np.sum(dsp_tp_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(dsp_tp_vars * np.square(ham_cs.coeff_vec[1:])))
    print('DSP+TP complete')
    # MEM+DSP
    mem_dsp_expvals, mem_dsp_vars = get_dsp_estimates(noise_factor=0,input_data=data_dsp['results']['experiment_data'], n_resamples=n_resamples,use='MEM',use_TP = False, threshold=threshold)
    mem_dsp_nrg, mem_dsp_std = np.sum(mem_dsp_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(mem_dsp_vars * np.square(ham_cs.coeff_vec[1:])))
    print('MEM+DSP complete')
    # MEM+DSP+TP
    mem_dsp_tp_expvals, mem_dsp_tp_vars = get_dsp_estimates(noise_factor=0,input_data=data_dsp['results']['experiment_data'], n_resamples=n_resamples,use='MEM',use_TP = True, threshold=threshold)
    mem_dsp_tp_nrg, mem_dsp_tp_std = np.sum(mem_dsp_tp_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(mem_dsp_tp_vars * np.square(ham_cs.coeff_vec[1:])))
    print('MEM+DSP+TP complete')
    
    # DSP+ZNE
    dsp_zne_expvals, dsp_zne_vars = get_dsp_zne_estimates(
        data_dsp_zne['results']['experiment_data'],
        n_resamples=n_resamples,use='RAW',use_TP = False, threshold=threshold_2
    )
    dsp_zne_nrg, dsp_zne_std = np.sum(dsp_zne_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(dsp_zne_vars * np.square(ham_cs.coeff_vec[1:])))
    print('DSP+ZNE complete')
    # DSP+ZNE+TP
    dsp_tp_zne_expvals, dsp_tp_zne_vars = get_dsp_zne_estimates(
        data_dsp_zne['results']['experiment_data'],
        n_resamples=n_resamples,use='RAW',use_TP = True, threshold=threshold_2
    )
    dsp_tp_zne_nrg, dsp_tp_zne_std = np.sum(dsp_tp_zne_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(dsp_tp_zne_vars * np.square(ham_cs.coeff_vec[1:])))
    print('DSP+TP+ZNE complete')

    # MEM+DSP+ZNE
    mem_dsp_zne_expvals, mem_dsp_zne_vars = get_dsp_zne_estimates(
        data_dsp_zne['results']['experiment_data'],
        n_resamples=n_resamples,use='MEM',use_TP = False, threshold=threshold_2
    )
    mem_dsp_zne_nrg, mem_dsp_zne_std = np.sum(mem_dsp_zne_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(mem_dsp_zne_vars * np.square(ham_cs.coeff_vec[1:])))
    print('MEM+DSP+ZNE complete')
    # MEM+DSP+ZNE+TP
    mem_dsp_tp_zne_expvals, mem_dsp_tp_zne_vars = get_dsp_zne_estimates(
        data_dsp_zne['results']['experiment_data'],
        n_resamples=n_resamples,use='MEM',use_TP = True, threshold=threshold_2
    )
    mem_dsp_tp_zne_nrg, mem_dsp_tp_zne_std = np.sum(mem_dsp_tp_zne_expvals * ham_cs.coeff_vec[1:]), np.sqrt(np.sum(mem_dsp_tp_zne_vars * np.square(ham_cs.coeff_vec[1:])))
    print('MEM+DSP+TP+ZNE complete')

    # ZNE individual expectation values not calculated
    zne_expvals = zne_vars = None
    sym_zne_expvals = sym_zne_vars = None
    mem_zne_expvals = mem_zne_vars = None
    mem_sym_zne_expvals = mem_sym_zne_vars = None
    dsp_zne_expvals = dsp_zne_vars = None
    dsp_tp_zne_expvals = dsp_tp_zne_vars = None
    mem_dsp_zne_expvals = mem_dsp_zne_vars = None
    mem_dsp_tp_zne_expvals = mem_dsp_tp_zne_vars = None

    final_estimates = {
        'RAW':{'energy':raw_nrg, 'stddev':raw_std, 'expvals':raw_expvals, 'variances':raw_vars}, 
        'ZNE':{'energy':zne_nrg, 'stddev':zne_std, 'expvals':zne_expvals, 'variances':zne_vars},
        'SYM':{'energy':sym_nrg, 'stddev':sym_std, 'expvals':sym_expvals, 'variances':sym_vars},
        'SYM+ZNE':{'energy':sym_zne_nrg, 'stddev':sym_zne_std, 'expvals':sym_zne_expvals, 'variances':sym_zne_vars},
        'MEM':{'energy':mem_nrg, 'stddev':mem_std, 'expvals':mem_expvals, 'variances':mem_vars},
        'MEM+SYM':{'energy':mem_sym_nrg, 'stddev':mem_sym_std, 'expvals':mem_sym_expvals, 'variances':mem_sym_vars},
        'MEM+ZNE':{'energy':mem_zne_nrg, 'stddev':mem_zne_std, 'expvals':mem_zne_expvals, 'variances':mem_zne_vars},
        'MEM+SYM+ZNE':{'energy':mem_sym_zne_nrg, 'stddev':mem_sym_zne_std, 'expvals':mem_sym_zne_expvals, 'variances':mem_sym_zne_vars},
        'DSP':{'energy':dsp_nrg, 'stddev':dsp_std, 'expvals':dsp_expvals, 'variances':dsp_vars},
        'DSP+TP':{'energy':dsp_tp_nrg, 'stddev':dsp_tp_std, 'expvals':dsp_tp_expvals, 'variances':dsp_tp_vars},
        'MEM+DSP':{'energy':mem_dsp_nrg, 'stddev':mem_dsp_std, 'expvals':mem_dsp_expvals, 'variances':mem_dsp_vars},
        'MEM+DSP+TP':{'energy':mem_dsp_tp_nrg, 'stddev':mem_dsp_tp_std, 'expvals':mem_dsp_tp_expvals, 'variances':mem_dsp_tp_vars},
        'DSP+ZNE':{'energy':dsp_zne_nrg, 'stddev':dsp_zne_std, 'expvals':dsp_zne_expvals, 'variances':dsp_zne_vars},
        'DSP+TP+ZNE':{'energy':dsp_tp_zne_nrg, 'stddev':dsp_tp_zne_std, 'expvals':dsp_tp_zne_expvals, 'variances':dsp_tp_zne_vars},
        'MEM+DSP+ZNE':{'energy':mem_dsp_zne_nrg, 'stddev':mem_dsp_zne_std, 'expvals':mem_dsp_zne_expvals, 'variances':mem_dsp_zne_vars},
        'MEM+DSP+TP+ZNE':{'energy':mem_dsp_tp_zne_nrg, 'stddev':mem_dsp_tp_zne_std, 'expvals':mem_dsp_tp_zne_expvals, 'variances':mem_dsp_tp_zne_vars}
    }
    return final_estimates

def get_zne_data_2(data_in, use='RAW', n_resamples=1000):
    non_I_block = ham_cs[1:].X_block | ham_cs[1:].Z_block
    raw_expvals = [
        bootstrap_raw_process(
            i, non_I_block, data_in['results']['experiment_data'], use=use, n_resamples=n_resamples
        )
        for i in data_in['results']['ZNE_factors']
    ]

    termwise_zne = []
    for i in range(ham_cs.n_terms-1):
        bs_raw_zne_termwise=[]
        for ed in np.array(raw_expvals)[:,i,:].T:
            raw_zne_curve = np.poly1d(np.polyfit(x=data_in['results']['ZNE_factors'], y=ed, deg=1))
            bs_raw_zne_termwise.append(raw_zne_curve(0))
        termwise_zne.append(bs_raw_zne_termwise)
    return termwise_zne

def bootstrap_data_for_plotting(system, hamiltonian, n_resamples=100, threshold=0.1, suffix='1mil'):
    print(f'Processing quantum experiment data from {system}...')
    
    data_raw, data_zne, data_dsp, data_dsp_zne = get_data(system, suffix)

    # RAW
    raw_expvals = get_energies_raw(
        data_raw['results']['experiment_data'], 
        hamiltonian, 
        use='RAW', 
        n_resamples=n_resamples, 
        termwise=False
    )
    print('RAW complete')

    # DSP+TP
    dsp_tp_estimate = get_dsp_estimates(noise_factor=0,input_data=data_dsp['results']['experiment_data'], n_resamples=n_resamples,use='RAW',use_TP = True, threshold=threshold)
    dsp_expvals = np.sum(dsp_tp_estimate.T * hamiltonian.coeff_vec, axis=1)
    print('DSP complete')

    #MEM+SS
    sym_corrected_data = deepcopy(data_raw)
    update_data(sym_corrected_data, use='RAW')
    update_data(sym_corrected_data, use='MEM')
    sym_corrected_data_zne = deepcopy(data_zne)
    for i in data_zne['results']['ZNE_factors']:
        update_data(sym_corrected_data_zne, use='RAW', noise_amp=i)
        update_data(sym_corrected_data_zne, use='MEM', noise_amp=i)

    ss_expvals = get_energies_raw(
        sym_corrected_data['results']['experiment_data'], 
        hamiltonian, 
        use='MEM', 
        n_resamples=n_resamples, 
        termwise=False
    )
    print('SS complete')

    # ZNE
    zne_estimates = get_zne_data_2(sym_corrected_data_zne, use='MEM', n_resamples=n_resamples)
    zne_expvals = np.sum(np.array(zne_estimates).T * hamiltonian.coeff_vec, axis=1)
    
    zne_wls = zero_noise_extrapolation(data_in=sym_corrected_data_zne, coefficients=hamiltonian.coeff_vec, use='MEM', n_resamples=n_resamples)
    print('ZNE complete')

    return raw_expvals, ss_expvals, zne_expvals, dsp_expvals, zne_wls


    