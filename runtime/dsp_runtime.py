from multiprocessing.sharedctypes import Value
from posixpath import split
from re import L
from time import perf_counter
import numpy as np
from math import ceil
from typing import List, Union, Tuple, Dict
from collections import Counter
from qiskit.providers.ibmq.runtime import UserMessenger
from qiskit.quantum_info import Pauli, PauliTable
from qiskit.opflow import PauliSumOp, PauliOp, PauliBasisChange
from qiskit import QuantumCircuit, transpile, Aer
import mthree

def split_list(alist, wanted_parts=1):
    """
    """
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
            for i in range(wanted_parts) ]

class CircuitCompiler:
    """
    """
    def __init__(self,
            backend,
            user_messenger,
            observable,
            circuit: QuantumCircuit
        ) -> None:
        """
        """
        self.backend = backend
        self.user_messenger = user_messenger
        self.observable = observable
        self.n_qubits = observable.num_qubits
        self.circuit = circuit

        self.readout_block = {
            'IIZ':self._readout_IIZ,
            'IZI':self._readout_IZI,
            'ZII':self._readout_ZII,
            'IZZ':self._readout_IZZ,
            'ZZI':self._readout_ZZI,
            'ZIZ':self._readout_ZIZ,
            'ZZZ':self._readout_ZZZ
        }
        # identify the qubit cluster with best CNOT error
        self.qubit_cluster_scores = self.score_qubit_clusters()
        self.optimal_cluster = sorted(self.qubit_cluster_scores)[0][1]
        
        measurement_map = {
            'IIZ':{'a':0, 'b':2, 'c':1, 'd':4},
            'IZI':{'a':0, 'b':2, 'c':1, 'd':3},
            'ZII':{'a':1, 'b':3, 'c':2, 'd':4},
            'IZZ':{'a':0, 'b':2, 'c':1, 'd':4},
            'ZZI':{'a':0, 'b':2, 'c':1, 'd':4},
            'ZIZ':{'a':0, 'b':2, 'c':1, 'd':4},
            'ZZZ':{'a':0, 'b':2, 'c':1, 'd':4}
        }
        self.qmap={}
        for term,mapping in measurement_map.items():
            self.qmap[term] = {label:self.optimal_cluster[ind] for label, ind in mapping.items()}
        # initiate the measurement error mitigation package
        self.m3 = mthree.M3Mitigation(self.backend)

    def score_qubit_clusters(self):
        """ Identify cluster of five qubits with best error readings
        """
        # CNOT error map
        coupled_error_map = {}
        for q1,q2 in self.backend.configuration().coupling_map:
            coupled_error_map.setdefault(q1,{})[q2] = self.backend.properties().gate_error('cx', [q1,q2])
        # qubits with three adjacent qubits
        core_qubits = [q for q,adj in coupled_error_map.items() if len(adj)==3]
        # the neighbouring qubits of the core ones above
        core_neighbours = [list(coupled_error_map[q].keys()) for q in core_qubits]
        # find all five qubit clusters of the required shape
        qubit_clusters = []
        for core, neighbours in zip(core_qubits, core_neighbours):
            # core is qubit 1
            for q in neighbours: # qubit 2
                extended_qubit = list(set(coupled_error_map[q].keys()).difference([core]+neighbours)) # qubit 3
                remaining = list(set(neighbours).difference([q])) # these are qubits 0 and 4
                if extended_qubit:
                    qubit_clusters.append({0:remaining[0], 1:core, 2:q, 3:extended_qubit[0], 4:remaining[1]})
        # score each qubit cluster based on SX, CNOT and readout errors
        cluster_scores = [
                np.sqrt(np.sum([
                    self.backend.properties().readout_error(q) * self.backend.properties().gate_error('sx', q) * 
                    np.sum([coupled_error_map[q].get(p, 0) for p in cluster.values()]) 
                    for q in cluster.values()
                ]))
            for cluster in qubit_clusters
        ]
        return list(zip(cluster_scores, qubit_clusters))

    def readout_standard(self, qc, c, t, noise_parameter=0):
        """
        """
        qc.cx(c,t)
    
    def readout_root_product(self, qc, c, t, noise_parameter=1):
        """
        """
        qc.h(t)
        for j in range(noise_parameter):
            qc.cp(np.pi/noise_parameter,c,t)
        qc.h(t)

    def _readout_IIZ(self, qc, amp):
        self.readout_method(qc,1,0,amp)
    
    def _readout_IZI(self, qc, amp):
        self.readout_method(qc,2,0,amp)
    
    def _readout_ZII(self, qc, amp):
        self.readout_method(qc,3,0,amp)
    
    def _readout_IZZ(self, qc, amp):
        qc.cx(2,1)
        self.readout_method(qc,1,0,amp)
        qc.cx(2,1)
    
    def _readout_ZZI(self, qc, amp):
        qc.swap(3,1); qc.cx(2,1)
        self.readout_method(qc,1,0,amp)
        qc.cx(2,1); qc.swap(3,1)
    
    def _readout_ZIZ(self, qc, amp):
        qc.cx(3,1)
        self.readout_method(qc,1,0,amp)
        qc.cx(3,1)
    
    def _readout_ZZZ(self, qc, amp):
        qc.cx(3,1); qc.cx(2,1)
        self.readout_method(qc,1,0,amp)
        qc.cx(2,1); qc.cx(3,1)

    def get_purification_circuits(self, 
            basis: str = 'Z',
            noise_parameter: int = 0
        ):
        """
        """
        if noise_parameter != 0:
            self.readout_method = self.readout_root_product
        else:
            self.readout_method = self.readout_standard

        ancilla = QuantumCircuit(1)
        dsp_circ = self.circuit.tensor(ancilla)

        circuits_out = []

        for pauli, X_term, Z_term in zip(
                self.observable.primitive.paulis,
                self.observable.primitive.paulis.x[:,::-1],
                self.observable.primitive.paulis.z[:,::-1]
            ):
            dsp_circ_term = dsp_circ.copy()

            label = pauli.to_label().replace('X', 'Z').replace('Y', 'Z')
            H_locs = self.n_qubits - np.where(X_term)[0]
            S_locs = self.n_qubits - np.where(X_term & Z_term)[0]

            # basis transformation
            for i in S_locs:
                dsp_circ_term.sdg(i)
            for i in H_locs:
                dsp_circ_term.h(i)

            # readout to ancilla qubit via parity computation (possibly with noise amplification)
            self.readout_block[label](dsp_circ_term, noise_parameter)

            # reverse the circuit
            for i in H_locs:
                dsp_circ_term.h(i)
            for i in S_locs:
                dsp_circ_term.s(i) 
            dsp_circ_term = dsp_circ_term.compose(self.circuit.inverse().tensor(ancilla))

            # basis-change on the ancilla qubit for full state tomography
            if basis=='Z':
                pass
            elif basis=='X':
                dsp_circ_term.h(0)
            elif basis=='Y':
                dsp_circ_term.sdg(0)
                dsp_circ_term.h(0)

            dsp_circ_term.measure_all()

            circuits_out.append(
                transpile(
                    dsp_circ_term,
                    backend=self.backend,
                    optimization_level=1,
                    initial_layout=dict(zip(dsp_circ_term.qregs[0][::-1], self.qmap[label].values()))
                )
            )

        return circuits_out


class CircuitSampling(CircuitCompiler):
    
    def __init__(self, backend, user_messenger, base_circuit, observable):
        super().__init__(
            backend=backend,
            user_messenger=user_messenger,
            observable=observable,
            circuit=base_circuit
        )
        self.max_shots = backend.configuration().max_shots
        self.max_circs = backend.configuration().max_experiments
        self.m3 = mthree.M3Mitigation(self.backend)
        
    def _get_counts(self, circuit, n_shots) -> List[Dict[str, float]]:
        """ Given a list of parametrizations, bind the circuits and submit to the backend

        Divides the circuit jobs into smaller batches to circumvent backend limitations
        
        Returns:
            - result (List[Dict[str:int]]):
                A list of dictionaries in which keys are binary strings and 
                their values the frequency of said measurement outcome 

        """
        assert isinstance(circuit, QuantumCircuit), 'Only works for single QuantumCircuits'
        # if number of shots exceeds backend.max_shots split into smaller shot batches
        if n_shots <= self.max_shots:
            shot_batches = [n_shots]
        else:
            n_full_shot_batches = n_shots // self.max_shots
            remaining_shot_batch = n_shots % self.max_shots
            shot_batches = [self.max_shots]*n_full_shot_batches
            if remaining_shot_batch != 0:
                shot_batches.append(remaining_shot_batch)
        self.user_messenger.publish(f'Extracting {n_shots} circuit shots over {len(shot_batches)} batch(es)')
        
        fragmented_countsets = []
        for ns in shot_batches:
            # submit circuit/shot batch to the QPU
            job = self.backend.run(
                circuits = circuit,
                shots    = ns,
            )
            result = job.result()
            raw_count_fragment = result.get_counts()
            fragmented_countsets.append(raw_count_fragment)
        
        # combine count dictionaries element-wise so that like-values are summed
        raw_counts = dict(sum([Counter(fc) for fc in fragmented_countsets], Counter())) 
    
        return raw_counts
    
    def _mitigate_measurement_error(self, raw_counts):
        """
        """
        self.m3.cals_from_system(self.mapping)
        p_dist_mit = self.m3.apply_correction(
            raw_counts, self.mapping
        ).nearest_probability_distribution()
        return p_dist_mit
    
    def get_counts(self, n_shots, ancilla_basis='Z', noise_parameter=1):
        """
        """
        circuits = self.get_purification_circuits(
            basis=ancilla_basis,
            noise_parameter=noise_parameter
        )
        if not isinstance(n_shots, int):
            assert(len(circuits)==len(n_shots)), 'List of circuits and shots must be of equal length'
            n_shots_list = n_shots
        else:
            n_shots_list = [n_shots]*len(circuits)
        
        raw_counts = [self._get_counts(circ, ns) for circ,ns in zip(circuits, n_shots_list)]
        # qubit mapping for measurement-error mitigation
        self.mapping = mthree.utils.final_measurement_mapping(circuits)
        mem_counts = self._mitigate_measurement_error(raw_counts)
            
        measurements = {
            'RAW': [
                {out:freq/ns for out,freq in counts.items()}
                for ns,counts in zip(n_shots_list, raw_counts)
            ],
            'MEM': mem_counts,
            'circuits': circuits.copy(),
            'n_shots_by_circuit': np.asarray(n_shots_list).tolist()
        }
            
        return measurements


class ShotBalancing(CircuitSampling):
    """
    """
    I = np.array([[1,0],[0,1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    
    n_shots_for_rebalancing = 1_000
    
    def __init__(self, backend, user_messenger, circuit, observable):
        """
        """
        super().__init__(
            backend=backend, 
            user_messenger=user_messenger, 
            base_circuit=circuit,
            observable=observable
        )
        self.pauli_variances = None
        self.shot_proportions = None
        
    def estimate_pauli_variances(self):
        """
        """
        expvals = {}
        for basis in ['X', 'Z']:
            basis_measurements = self.get_counts(
                n_shots=self.n_shots_for_rebalancing//2,
                ancilla_basis=basis,
                noise_parameter=0
            )
            termwise_expvals = []
            mem_data = basis_measurements.get('MEM')
            for counts in mem_data:
                p0 = counts.get('0'*3+'0', 0)
                p1 = counts.get('0'*3+'1', 0)
                termwise_expvals.append((p0-p1)/(p0+p1))
            expvals[basis] = np.array(termwise_expvals)

        variances = []
        for Tr_X_rho,Tr_Z_rho in zip(
                expvals.get('X'),expvals.get('Z')
            ):
            rho = (self.I + Tr_X_rho*self.X + Tr_Z_rho*self.Z)/2 
            eigvals, eigvecs = np.linalg.eig(rho)
            largest_eigvec = eigvecs[:,np.argmax(eigvals)].reshape([-1,1])
            exp_X = abs((largest_eigvec.T.conjugate() @ self.X @ largest_eigvec)[0][0])
            exp_Z = (largest_eigvec.T.conjugate() @ self.Z @ largest_eigvec)[0][0] 
            expval = exp_Z/(1+exp_X)
            variances.append(1-expval**2)

        self.pauli_variances = np.array(variances)
    
    def rebalance(self):
        assert(self.pauli_variances is not None), 'Must first estimate Pauli variances'
        
        shot_proportions = abs(self.observable.coeffs) * np.sqrt(abs(self.pauli_variances))
        shot_proportions /= sum(shot_proportions)
        self.shot_proportions = shot_proportions
        
    def distribute_shots(self, budget):
        dist = np.ceil(budget * self.shot_proportions)
        # in case there are any zeros in the distribution:
        zeros = np.where(dist==0); nonzeros = np.where(dist!=0)
        dist[nonzeros]-=len(zeros[0])
        dist[zeros] = np.count_nonzero(dist)
        return list(map(int, dist))


class DSP_Runtime(ShotBalancing):
    """ Runtime program for performing VQE routines.
    """
    ZNE_factors = [0] # the maximum noise scaling parameter
    
    def __init__(self,
        backend = Aer.get_backend('qasm_simulator'),
        user_messenger = UserMessenger(),
        circuit: QuantumCircuit= None,
        observable: PauliSumOp = None
        ) -> None:
        """
        Args:
            - backend (): 
                The target QPU for circuit jobs
            - user_messenger (): 
                Allows readout during computation
            - circuit (QuantumCircuit)
                Circuit for state preparation
            - observable (PauliSumOp): 
                The observable for expecation value estimation
            - observable_groups (List[PauliSumOp]): 
                Some Pauli operators may be estimated simultaneously. The grouping 
                specifies this - if None, will determine a random grouping.
        """
        super().__init__(
            backend = backend, 
            user_messenger = user_messenger,
            circuit = circuit,
            observable = observable
        )
        
    def run(self, shot_budget=100_000, proportion_used_for_rebalancing=0.001):
        """
        """
        self.n_shots_for_rebalancing = int(shot_budget*proportion_used_for_rebalancing)
        self.estimate_pauli_variances()
        self.rebalance()
        
        shots_per_lambda_factor = (
            shot_budget*(1-proportion_used_for_rebalancing)
        ) // (2*len(self.ZNE_factors)) # factor of 1/2 accounts for the ancilla tomography step
        shot_distribution = self.distribute_shots(shots_per_lambda_factor)

        self.user_messenger.publish(f'Finished shot rebalancing; proceeding onto full ground state preparation.')

        measurement_data = {}        
        for k in self.ZNE_factors:
            self.user_messenger.publish(f'Noise amplification factor {k}...')
            basis_measurements = {}
            for basis in ['X', 'Z']:
                self.user_messenger.publish(f'Performing measurements in {basis} basis.')
                basis_measurements[basis] = self.get_counts(
                    n_shots=shot_distribution,
                    ancilla_basis=basis,
                    noise_parameter=k
                )
            measurement_data[str(k)] = basis_measurements
            
        zne_results = {
            'experiment_data': measurement_data,
            'pauli_variances': list(map(float, self.pauli_variances)),
            'optimal_qubit_cluster': self.optimal_cluster,
            'qubit_routing':self.qmap,
            'qubit_cluster_scores': self.qubit_cluster_scores,
            'ZNE_factors': self.ZNE_factors
        }

        return zne_results


def main(backend, user_messenger, **kwargs):
    """ The main runtime program entry-point.

    All the heavy-lifting is handled by the DSP_Runtime class

    Returns:
       
    """
    data_out = {
        'hardware_spec': backend.configuration().to_dict(),
        'gate_errors':   backend.properties().to_dict()
    }
    observable  = kwargs.get("observable", None)
    circuit     = kwargs.get("circuit", None)
    
    dsp = DSP_Runtime(
        backend=backend,
        user_messenger = user_messenger,
        circuit=circuit,
        observable=observable
    )
    dsp.ZNE_factors = kwargs.get("ZNE_factors", [0])
    
    shot_budget = kwargs.get("shot_budget", 2**10)
    proportion_used_for_rebalancing = kwargs.get(
        "proportion_used_for_rebalancing", 0.001
    )
    
    data_out['observable']  = observable
    data_out['shot_budget'] = shot_budget
    data_out['proportion_used_for_rebalancing'] = proportion_used_for_rebalancing  
    data_out['results'] = dsp.run(
        shot_budget = shot_budget,
        proportion_used_for_rebalancing = proportion_used_for_rebalancing
    )

    return data_out