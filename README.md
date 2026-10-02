# MERMAID

MERMAID is a Reinforcement Learning-based Molecular Optimization Library.

## Installation

Use a separate environment with **Python 3.12.15**. Install that interpreter first;
older uv/micromamba catalogs may not yet provide this patch release. Do not replace
an existing system or research environment. From the repository root:

```sh
uv venv --python /path/to/python3.12 .venv
uv --no-config pip install --python .venv/bin/python -r requirements-lock.txt
uv pip check --python .venv/bin/python
```

The interpreter at `/path/to/python3.12` must report version 3.12.15.
`requirements.txt` lists the nine direct dependencies; `requirements-lock.txt`
records all 99 packages in the tested CPU environment. The original Linux/CUDA
Conda export has been removed because it pins obsolete runtimes and native
libraries. CPU verification used macOS ARM64; CUDA/Linux wheels and execution
have not been validated. PyTorch's CUDA distribution requires a separately
validated environment following its official installation instructions.

Run these commands from the repository root:

```sh
.venv/bin/python Model/preprocess.py
.venv/bin/python Model/train.py
.venv/bin/python Generator/mcts.py
.venv/bin/python -m unittest discover -s tests -v
```

Hydra retains the previous 1.1 working-directory behavior. CLI training retains
local `mlruns` storage in the Hydra run directory; `MLFLOW_TRACKING_URI` overrides
that location. Existing file stores remain readable using
`MLFLOW_ALLOW_FILE_STORE=true`. No tracking server is started by these commands.

`rep_mol.ipynb` can use the same environment with an IPython notebook kernel. It
no longer installs Conda into `/usr/local`. Its fragment-generation calculations
were compared; notebook frontend/kernel installation is outside the runtime lock.

## Compatibility and security

The bundled checkpoint loads through `weights_only=True`. General pickle objects
are rejected; arbitrary legacy pickled model objects are not supported. The
bundled `Data/fpscores.pkl.gz` remains a trusted local SA-score table; do not replace
it with an untrusted pickle. No model architecture, training hyperparameter,
reward formula, or production dropout setting has changed.

See [validation/REPORT.md](validation/REPORT.md) for security rationale, measured
comparisons, and limits, and [validation/REPRODUCE.md](validation/REPRODUCE.md) for
reproduction. Deterministic comparisons passed; stochastic training and generation
can produce different results for the same seed after the library upgrade.
