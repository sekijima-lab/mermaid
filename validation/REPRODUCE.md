# Reproduction

Create independent environments; do not update an existing research environment.
The production CPU environment uses Python 3.12.15, installed separately, and:

```sh
uv venv --python /path/to/python3.12 /path/to/modern-env
uv --no-config pip install --python /path/to/modern-env/bin/python -r requirements-lock.txt
uv pip check --python /path/to/modern-env/bin/python
/path/to/modern-env/bin/python -m unittest discover -s tests -v
```

The tested installed uv was 0.10.2. Its existing user configuration prevented
resolving some newer releases; `--no-config` fixed fresh dependency resolution
without modifying user settings. The 99-pin lock resolved successfully against
official PyPI with this option. The test environment initially used official
PyPI JSON wheel URLs, whose installed versions match the version-only lock.

Archive the original source into a separate directory:

```sh
mkdir /path/to/reference
git archive b2eb1e0ca4501b985f59495f3b478ade5f14827f | tar -x -C /path/to/reference
```

For the macOS Intel reference environment used under Rosetta:

```sh
micromamba create -y -r /path/to/isolated-mamba-root -p /path/to/legacy-env \
  --platform osx-64 -c pytorch -c rdkit -c defaults --no-channel-priority \
  python=3.6.13 pytorch=1.9.1 rdkit=2020.09.1 numpy=1.19.2 pandas=1.1.5 pip=21.2.2
PIP_CACHE_DIR=/path/to/isolated-pip-cache arch -x86_64 /path/to/legacy-env/bin/python -m pip install \
  hydra-core==1.1.1 omegaconf==2.1.1 networkx==2.5.1 tqdm==4.62.3 mlflow==1.20.2
arch -x86_64 /path/to/legacy-env/bin/python -m pip check
```

This is an obsolete validation-only reconstruction. It is not the authors' exact
Linux/CUDA environment and must not be used for production or untrusted inputs.

Use the new repository's Python-3.6-compatible harnesses against each source tree:

```sh
arch -x86_64 /path/to/legacy-env/bin/python tests/benchmark.py /path/to/reference /path/to/results/legacy
/path/to/modern-env/bin/python tests/benchmark.py /path/to/updated/repo /path/to/results/modern
PYTHONHASHSEED=0 arch -x86_64 /path/to/legacy-env/bin/python tests/benchmark_generation.py /path/to/reference /path/to/results/legacy-generation
PYTHONHASHSEED=0 /path/to/modern-env/bin/python tests/benchmark_generation.py /path/to/updated/repo /path/to/results/modern-generation
arch -x86_64 /path/to/legacy-env/bin/python tests/benchmark_smoke.py /path/to/reference /path/to/results/legacy-smoke
/path/to/modern-env/bin/python tests/benchmark_smoke.py /path/to/updated/repo /path/to/results/modern-smoke
```

Compare every array in `prediction.npz` and `training.npz`, the numerical JSON
summaries and the MCTS JSON against `criteria.json`; do not loosen thresholds
after seeing results. Harnesses set CPU threading to one and map Hydra's original
working directory to the selected source tree. They load only the trusted bundled
checkpoint and SA table. The numerical/model comparisons disable dropout;
the smoke harness preserves the original production mode. The latter's stochastic
outputs are expected to differ and are reported without an equality claim.

The benchmark_smoke harness places new checkpoints, logs and MLflow stores inside
the results directory. It does not replace bundled data or checkpoints. To compare
notebook calculations:

```sh
arch -x86_64 /path/to/legacy-env/bin/python tests/benchmark_notebook.py /path/to/reference/rep_mol.ipynb
/path/to/modern-env/bin/python tests/benchmark_notebook.py rep_mol.ipynb
```

This last harness evaluates repository notebook function definitions. Run it only
against the trusted source. Compare canonical SMILES; SVG existence is checked,
not visual equivalence. `osv-snapshot.json` records the dated advisory queries;
repeat those queries before a later release.
