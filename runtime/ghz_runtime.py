from qiskit import QuantumCircuit, transpile
import mthree

class GHZ_Runtime:
    """ Runtime program for GHZ state fidelity to benchmark QPUs.
    """
    n_shots = 2**12 # number of circuit shots per job
    mitigate_errors = True # flag indicating whether error mitigation is to be performed 
    
    def __init__(
        self,
        backend,
        user_messenger,
        max_qubits: int = 10
        ) -> None:

        self.backend = backend
        self.user_messenger = user_messenger
        self.max_qubits = min([backend.configuration().num_qubits, max_qubits])
        self.m3 = mthree.M3Mitigation(backend)

    def GHZ_circuit(self, n_qubits):
        """ Define the quantum circuit preparing the N-qubit GHZ state
        """
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits-1):
            qc.cx(i, i+1)
        qc.measure_all()
        return transpile(qc, backend=self.backend, optimization_level=3)

    def prepare_and_measure(self, n_qubits):
        """
        """
        # define circuit and submit to backend
        qc = self.GHZ_circuit(n_qubits)
        job = self.backend.run(qc, shots=self.n_shots)
        raw_counts = job.result().get_counts()

        # raw probability distribution
        raw_prob_dist = {binstr:count/self.n_shots for binstr,count in raw_counts.items()}
        
        if self.mitigate_errors:
            # error mitigated probability distribution
            self.mapping = mthree.utils.final_measurement_mapping(qc)
            self.m3.cals_from_system(self.mapping)
            quasis = self.m3.apply_correction(raw_counts, list(self.mapping.keys()))
            mit_prob_dist = quasis.nearest_probability_distribution()
        else:
            mit_prob_dist = raw_prob_dist

        return raw_prob_dist, mit_prob_dist, qc

    def run(self):
        """
        """
        GHZ_data_out = {}
        for i in range(1, self.max_qubits+1):
            self.user_messenger.publish(f'Preparing {i}-qubit GHZ state')
            raw_counts, mem_counts, qc = self.prepare_and_measure(i)
            self.user_messenger.publish('Raw state data')
            self.user_messenger.publish(raw_counts)
            self.user_messenger.publish('Error mitigated state data')
            self.user_messenger.publish(mem_counts)
            GHZ_data_out[i] = {
                'RAW':raw_counts,
                'MEM':mem_counts,
                'circuit': qc,
                'mapping': self.mapping.copy()
            }
        
        return GHZ_data_out


def main(backend, user_messenger, **kwargs):
    """ The main runtime program entry-point.

    All the heavy-lifting is handled by the GHZ_Runtime class

    Returns:
       
    """
    ghz = GHZ_Runtime(
        backend=backend,
        user_messenger = user_messenger,
        max_qubits=kwargs.get("max_qubits", 10)
    )
    ghz.n_shots = kwargs.get("n_shots", 2**10)
    ghz.mitigate_errors = kwargs.get("mitigate_errors", True)
    
    return ghz.run()