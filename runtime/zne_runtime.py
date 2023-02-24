import numpy as np
from itertools import combinations
from typing import List, Dict
from collections import Counter
from qiskit.providers.ibmq.runtime import UserMessenger
from qiskit.opflow import PauliSumOp
from qiskit import QuantumCircuit, transpile, Aer
import mthree

def split_list(alist, wanted_parts=1):
    """
    """
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
            for i in range(wanted_parts) ]

class CircuitSampling:
    
    def __init__(self, backend, user_messenger):
        self.backend = backend
        self.user_messenger = user_messenger
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
    
    def get_counts(self, circuits, n_shots):
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
    n_shots_for_rebalancing = 1_000
    
    def __init__(self, backend, user_messenger, circuits, observable):
        """
        """
        self.backend = backend
        self.user_messenger = user_messenger
        self.circuits = circuits
        self.observable = observable
        self.non_I_mask = (
            self.observable.primitive.paulis.x[:,::-1] | 
            self.observable.primitive.paulis.z[:,::-1]
        )
        super().__init__(backend=backend, user_messenger=user_messenger)
        self.pauli_variances = None
        self.shot_proportions = None
        
    def estimate_pauli_variances(self):
        """
        """
        measurement_data = self.get_counts(self.circuits, self.n_shots_for_rebalancing)
        variances = []
        for mask, counts in zip(self.non_I_mask, measurement_data['MEM']):

            expval = sum(
                [
                    prob*(-1)**np.sum(mask & np.array([int(i) for i in list(b_str)]).astype(bool)) 
                    for b_str, prob in counts.items()
                ]
            )
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

class ZNE_Runtime(ShotBalancing):
    """ Runtime program for performing VQE routines.
    """
    n_circs     = 100 # number of experiments allowed per circuit batch
    ZNE_factors = [1,2,3,4] # the maximum noise scaling parameter
    
    def __init__(self,
        backend = Aer.get_backend('qasm_simulator'),
        user_messenger = UserMessenger(),
        circuit: QuantumCircuit= None,
        observable: PauliSumOp = None,
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
        self.backend         = backend
        self.user_messenger  = user_messenger
        self.circuit         = circuit
        self.n_qubits        = observable.num_qubits
        self.observable      = observable
        
        # identify the qubit cluster with best CNOT error
        self.qubit_cluster_scores = self.score_qubit_clusters()
        self.optimal_cluster = sorted(self.qubit_cluster_scores)[0][1]
        
    def score_qubit_clusters(self):
        """ Identify cluster of five qubits with best error readings
        """
        # CNOT error map
        coupled_error_map = {}
        for q1,q2 in self.backend.configuration().coupling_map:
            coupled_error_map.setdefault(q1,{})[q2] = self.backend.properties().gate_error('cx', [q1,q2])
        # qubits with at least two adjacent qubits
        core_qubits = [q for q,adj in coupled_error_map.items() if len(adj)>=2]
        # the neighbouring qubits of the core ones above
        core_neighbours = [(q, list(combinations(coupled_error_map[q].keys(), r=2))) for q in core_qubits]
        # find all five qubit clusters of the required shape
        qubit_clusters = [[{0:core, 1:n[0], 2:n[1]} for n in neighbours] for core, neighbours in core_neighbours]
        qubit_clusters = [a for b in qubit_clusters for a in b] # flatten list
        # score each qubit cluster based on SX, CNOT and readout errors
        cluster_scores = [
                float(np.sqrt(np.sum([
                    self.backend.properties().readout_error(q) * self.backend.properties().gate_error('sx', q) * 
                    np.sum([coupled_error_map[q].get(p, 0) for p in cluster.values()]) 
                    for q in cluster.values()
                ])))
            for cluster in qubit_clusters
        ]
        return list(zip(cluster_scores, qubit_clusters))
    
    def CNOT_root_product(self, c, t, noise_parameter=1):
        """ Decomposition of the CNOT gate into repeated CPhase gates, which are each transpiled into 2 CNOTs
        Therefore, a noise parameter k results in 2k CNOT gates that implement a single CNOT.
        """
        qc_block = QuantumCircuit(self.n_qubits)
        qc_block.h(t)
        for j in range(noise_parameter):
            qc_block.cp(np.pi/noise_parameter,c,t)
        qc_block.h(t)
        return qc_block

    def replace_CNOTs(self, qc, noise_parameter=1):
        """ Strip single CNOT gates from the imput circuit and replace with the CPhase decomposition above
        """
        qc_copy = qc.copy()
        
        where_CX = np.where(np.array([i.operation.name=='cx' for i in qc_copy.data]))[0]
        if where_CX.size == 0:
            return qc_copy
        else:
            first_CX = where_CX[0]
            control, target = qc_copy.data[first_CX].qubits
            control, target = control.index, target.index
            noisy_CNOT = self.CNOT_root_product(control, target, noise_parameter=noise_parameter)
            qc_copy.data.pop(first_CX)
            for insert_operation in noisy_CNOT[::-1]:
                qc_copy.data.insert(first_CX, insert_operation)
            return self.replace_CNOTs(qc_copy, noise_parameter=noise_parameter)
            
    def get_zne_circuits(self, noise_parameter=0):
        """ Get the circuits for zero-noise extrapolation with the intended noise scaling parameter
        """
        circuits_out = []

        for pauli, X_term, Z_term in zip(
                self.observable.primitive.paulis,
                self.observable.primitive.paulis.x[:,::-1],
                self.observable.primitive.paulis.z[:,::-1]
            ):
            circ_term = self.circuit.copy()
            if noise_parameter != 0:
                circ_term = self.replace_CNOTs(
                    qc=circ_term, noise_parameter=noise_parameter
                )

            label = pauli.to_label().replace('X', 'Z').replace('Y', 'Z')
            H_locs = self.n_qubits - np.where(X_term)[0] - 1
            S_locs = self.n_qubits - np.where(X_term & Z_term)[0] - 1
            # basis transformation
            for i in S_locs:
                circ_term.sdg(i)
            for i in H_locs:
                circ_term.h(i)

            circ_term.measure_all()

            circuits_out.append(
                transpile(
                    circ_term,
                    backend=self.backend,
                    optimization_level=1,
                    initial_layout=dict(zip(circ_term.qregs[0], self.optimal_cluster.values()))
                )
            )
        return circuits_out

    def balance_shots(self, n_shots_for_rebalancing):
        """ Initiate the parent ShotBalancing class to estimate Pauli variances
        and redistribute circuit shots according to sqrt(|h_i|) * var(P_i)
        """
        base_circuits = self.get_zne_circuits()

        super().__init__(
            backend = self.backend, 
            user_messenger = self.user_messenger,
            circuits = base_circuits,
            observable=self.observable
        )
        self.n_shots_for_rebalancing = n_shots_for_rebalancing

        self.estimate_pauli_variances()
        self.rebalance()

    def run(self, shot_budget=100_000, proportion_used_for_rebalancing=0.001):
        """
        """
        self.balance_shots(
            n_shots_for_rebalancing=int(shot_budget*proportion_used_for_rebalancing)
        ) # by default 0.1% of the total shots used for rebalancing
        
        shots_per_lambda_factor = (
            shot_budget*(1-proportion_used_for_rebalancing)
        ) // len(self.ZNE_factors)
        shot_distribution = self.distribute_shots(shots_per_lambda_factor)

        self.user_messenger.publish(f'Finished shot rebalancing; proceeding onto full ground state preparation.')

        measurement_data = {}        
        for k in self.ZNE_factors:
            self.user_messenger.publish(f'Noise amplification factor {k}...')
            noisy_circuits = self.get_zne_circuits(noise_parameter=k)
            measurements = self.get_counts(circuits=noisy_circuits, n_shots=shot_distribution)
            measurement_data[str(k)] = measurements
            
        zne_results = {
            'experiment_data': measurement_data,
            'pauli_variances': list(map(float, self.pauli_variances)),
            'optimal_qubit_cluster': self.optimal_cluster,
            'qubit_cluster_scores': self.qubit_cluster_scores,
            'ZNE_factors': self.ZNE_factors
        }

        return zne_results


def main(backend, user_messenger, **kwargs):
    """ The main runtime program entry-point.

    All the heavy-lifting is handled by the ZNE_Runtime class

    Returns:
       
    """
    data_out = {
        'hardware_spec': backend.configuration().to_dict(),
        'gate_errors':   backend.properties().to_dict()
    }
    observable  = kwargs.get("observable", None)
    circuit     = kwargs.get("circuit", None)
    
    zne = ZNE_Runtime(
        backend = backend,
        user_messenger=user_messenger,
        circuit = circuit,
        observable = observable
    )
    zne.n_circs = backend.configuration().max_experiments
    zne.ZNE_factors = kwargs.get("ZNE_factors", [1,2,3,4])
    
    shot_budget = kwargs.get("shot_budget", 2**10)
    proportion_used_for_rebalancing = kwargs.get(
        "proportion_used_for_rebalancing", 0.001
    )
    
    data_out['observable']  = observable
    data_out['shot_budget'] = shot_budget
    #data_out['ZNE_factors'] = zne.ZNE_factors  
    data_out['proportion_used_for_rebalancing'] = proportion_used_for_rebalancing  
    data_out['results'] = zne.run(
        shot_budget = shot_budget,
        proportion_used_for_rebalancing = proportion_used_for_rebalancing
    )

    return data_out