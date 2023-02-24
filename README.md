# Quantum Error Mitigation

Repository to support the QEM benchmark paper.

The following packages will be required:

- [Symmer](https://github.com/UCL-CCS/symmer)
  - For implementing qubit subspace techniques such as Qubit Tapering and Contextual Subspace.
- [Qiskit](https://github.com/Qiskit)
  - For anything related to the quantum hardware itself, including circuit construction and execution.
- [Deco](https://github.com/alex-sherman/deco)
  - For parallelizing the bootstrapping procedure, but will be replaced with straight multiprocessing shortly.
