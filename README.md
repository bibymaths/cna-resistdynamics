<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

<div align="center">

<img src="docs/assets/logo.svg" alt="CNA-ResistDynamics" width="140" />

# CNA-ResistDynamics

**Dynamical modelling of chemotherapy resistance from liquidCNA measurements**

[![Tests](https://github.com/bibymaths/cna-resistdynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/bibymaths/cna-resistdynamics/actions/workflows/tests.yml)
[![Docs](https://github.com/bibymaths/cna-resistdynamics/actions/workflows/docs.yml/badge.svg)](https://bibymaths.github.io/cna-resistdynamics/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

📚 **[Documentation](https://bibymaths.github.io/cna-resistdynamics/)** · 🐛 **[Issues](https://github.com/bibymaths/cna-resistdynamics/issues)**

</div>

---

ODE and PDE dynamical models for inferring the temporal evolution of
chemotherapy resistance in ovarian cancer from longitudinal liquid-biopsy
(liquidCNA) and CA125 data.

See the **[documentation](https://bibymaths.github.io/cna-resistdynamics/)** for
installation, quick start, model description, and API reference.

## Quick install

```bash
conda env create -f environment.yml
conda activate cna-resist-dynamics
pip install -e .
tumorfits --help
```

## Citation

> Hockings et al. (2025). *Subclonal copy-number alterations drive resistance
> to platinum-based chemotherapy in ovarian cancer.*
> Cancer Research. DOI: [10.1158/0008-5472.CAN-25-0351](https://doi.org/10.1158/0008-5472.CAN-25-0351)