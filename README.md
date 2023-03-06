# Quantum Error Mitigation

Code repository supporting the following paper:

[Benchmarking Noisy Intermediate Scale Quantum Error Mitigation Strategies for Ground State Preparation of the HCl Molecule](https://doi.org/10.48550/arXiv.2303.00445)

In this work, we compare quantum error mitigation strageties comprised from various combinations of:
- Measurement-Error Mitigation
- Symmetry Verification
- Zero-Noise Extrapolation
- Dual-State Purification (with or without tomography purification)

The following packages will be required to run the notebooks contained herein:

- [Symmer](https://github.com/UCL-CCS/symmer)
  - For implementing qubit subspace techniques such as Qubit Tapering and Contextual Subspace.
- [Qiskit](https://github.com/Qiskit)
  - For anything related to the quantum hardware itself, including circuit construction and execution.
- [Deco](https://github.com/alex-sherman/deco)
  - For parallelizing the bootstrapping procedure, but will be replaced with straight multiprocessing shortly.

![](https://github.com/TimWeaving/quantum-error-mitigation/blob/main/plots/ibmq_kolkata_hist.png)
