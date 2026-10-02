# Security runtime migration

Validated on 2026-10-03, macOS ARM64, with an isolated Python 3.12.15 environment.
Reference source: `b2eb1e0ca4501b985f59495f3b478ade5f14827f`.

## Rationale and changes

The production definition pinned Python 3.6.13, PyTorch 1.9.1, MLflow 1.20.2,
RDKit 2020.09.1, and vulnerable transitive libraries. Python 3.6 is end of life
([Python version status](https://devguide.python.org/versions/)). PyTorch's
[CVE-2025-32434 advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
describes a weights-only loading vulnerability through 2.5.1, fixed in 2.6.0.
The original application used general pickle loading rather than weights-only
loading; the new checkpoint boundary explicitly uses `weights_only=True` and
rejects general Python objects. Tests use a harmless `object()` fixture.

New direct dependencies: torch 2.14.1, RDKit 2026.3.6, MLflow 3.16.1,
NumPy 2.5.3, pandas 3.0.6, Hydra 1.3.7, OmegaConf 2.3.1, NetworkX 3.7,
tqdm 4.70.1. All 99 installed distributions are pinned in `requirements-lock.txt`.
OSV query-batch found no matching advisory for those exact 99 package versions;
see `osv-snapshot.json`. `legacy-osv-snapshot.json` records matches for five
historical packages. Matches include overlapping databases/advisories and do not
establish exploitation of a particular application route. This is a dated PyPI
advisory check, not an audit of every native library or future vulnerability.

The Linux/CUDA-specific legacy Conda export was removed, with the historical
definition retained in `legacy-environment.md` for reference. A candidate Conda
Python 3.12.15 environment could not be solved by the installed catalog, so it is
not advertised as a working installation path. The tested interpreter was built
from official Python 3.12.15 source in an independent prefix (source archive
SHA256 `c2c4321961fab0fb999d66e0cecf521c2ab3994c7992873ea99e306c1094fd5a`).
No existing environment or global authentication setting was changed.

Compatibility edits remove obsolete `rdkit.six`, use dataclass default factories,
and select Hydra's historical 1.1 behavior. CLI MLflow logging retains local file
stores and respects `MLFLOW_TRACKING_URI`. See the official
[Hydra working-directory migration](https://hydra.cc/docs/upgrades/1.1_to_1.2/changes_to_job_working_dir/)
and [MLflow backend changes](https://mlflow.org/docs/latest/self-hosting/).
The molecular notebook no longer overwrites `/usr/local` with an old Conda/Python
installation; unused obsolete imports and historical cell outputs were removed.

## Quantitative comparison

The reference environment used Python 3.6.13 under Rosetta, the macOS Conda
PyTorch 1.9.1 package (reported runtime `1.9.1.post3`), RDKit 2020.09.1,
NumPy 1.19.2, pandas 1.1.5, Hydra 1.1.1, OmegaConf 2.1.1, NetworkX 2.5.1,
tqdm 4.62.3, and MLflow 1.20.2. Other transitive packages were compatible
reconstructions, not exact historical Linux/CUDA builds. Both old and new
dependency-consistency checks passed.

All numerical thresholds in `criteria.json` were declared before inspecting the
cross-version comparison. `comparison.json` contains the actual differences.

- Bundled checkpoint, first 256 preprocessed fragments, CPU, one thread,
  dropout disabled for comparison: max logits difference **7.6294e-6**
  (limit 1e-4); max probability difference **1.4901e-6** (limit 1e-5).
  All maximum-probability token choices and encoded inputs matched.
- One Adam step on 32 fragments, same checkpoint, dropout disabled:
  loss **0.9845690131187439** in both; max gradient difference **1.0431e-7**;
  max updated-weight difference **8.4676e-6** (both limits 1e-5).
- First 512 sample SMILES plus four explicit ring/invalid/aspirin cases:
  max QED difference **1.1102e-16** (limit 1e-8), penalized logP difference
  **0** (limit 1e-6). Exhaustive/filter preprocessing sets matched for 32
  sample molecules. Source hashes are in `input-manifest.json`.
- Controlled MCTS: bundled checkpoint in evaluation mode, seeds 0/1/2,
  `PYTHONHASHSEED=0`, three steps per input, CCO and the actual bundled starting
  molecule. All **six cases** matched exactly, including **39 generated
  molecules**, rewards and valid/invalid counts. See `generation-comparison.json`.
- Actual original and migrated training functions each completed one epoch on
  40 fragments, stored parameters and step-1 metrics, saved a readable checkpoint,
  and round-tripped the existing `log.txt` artifact through a fresh file backend.
  Short production-mode MCTS runs also generated valid molecules and tree CSVs.
  Old/new results are in `legacy-smoke.json` and `modern-smoke.json`.
- Actual migrated CLI generation used the bundled starting molecule and checkpoint,
  three steps and one iteration: exported **11 molecule rows** with finite rewards.
- The notebook's two specified fragment removals produced identical canonical
  SMILES in old/new RDKit; SVG construction succeeded in both. SVG pixel/layout
  equality and an IPython frontend session were not tested.

Five automated tests passed in 75.818 seconds: isolated configuration defaults,
checkpoint prediction against a numeric-only legacy baseline, general-object
checkpoint rejection, controlled MCTS regression, and actual preprocessing/training
CLI plus MLflow checkpoint/metric/artifact verification. The committed prediction
fixture contains the first 32 rows of the 256-row legacy benchmark.

## Limits and observable differences

**Same-seed stochastic output is not identical.** Production dropout remains
enabled as in the original code. In the short production-mode smoke test the
old run generated four valid molecules and the new run six; the molecule sets
differed. One-epoch training losses were 4.174240827560425 versus
4.173778057098389; validation losses 4.161250114440918 versus
4.161930561065674. Cross-version RNG/dropout/initialization behavior is not
guaranteed by the deterministic numerical comparisons. These are short smoke
checks, not evidence of equal full-training distributions or optimization quality.

The model architecture, vocabulary, reward formulas, hyperparameters and production
dropout setting are unchanged. GPU/CUDA, Linux, complete 100-epoch training, full
1000-step optimization, and paper-result reproduction remain unverified.
Legacy arbitrary pickled model objects are intentionally unsupported. The bundled
SA-score pickle remains trusted local input, not a general safe-pickle interface.
Existing historical MLflow databases were not migrated; the tested path is the
file backend used by this application. No GitHub CI workflow was added.
