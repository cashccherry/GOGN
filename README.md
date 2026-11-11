# GOGN acoustic full-waveform inversion

Research code accompanying **A Gauss–Newton Method with No Additional PDE Solves
Beyond Gradient Evaluation for Large-Scale PDE-Constrained Inverse Problems**,
by Cash Cherry, Samy Wu Fung, Luis Tenorio, and Ebru Bozdağ (2026).

The repository compares nonlinear conjugate gradient (NLCG), L-BFGS,
gradient-only Gauss–Newton (GOGN), and Gauss–Newton with inner CG (GN-CG) on
2D Deepwave acoustic inversions.

**Reproduction status:** this repository contains the source code, packaged
acquisition arrays, experiment scripts, plotting scripts, environment file, and
small regression/smoke tests needed to reproduce the reported experiments. Full
paper inversions are computationally expensive; the included smoke and regression
tests check installation, differentiability, data packaging, plotting entry
points, and optimizer behavior on reduced examples.

## Installation

Create a separate environment;

```bash
conda env create -f environment.yml
conda activate gogn
python -m pip install --no-build-isolation -e .
python -c "import gogn, deepwave; import importlib.metadata as m; print(gogn.__file__, m.version('deepwave'))"
```

The environment specifies Python 3.11, PyTorch 2.6.0, torchvision 0.21.0,
**stock Deepwave 0.0.20**, NumPy 2.3.0, SciPy 1.16.0, Matplotlib 3.10.3,
Pillow 12.3.0, psutil, and test dependencies. It omits machine-specific conda build hashes.
PyTorch/torchvision must be a matching pair. The supplied scripts run on CPU;
GPU execution is not configured or validated here. No external model download is needed.

## Small checks

```bash
python -m pytest -q
MPLBACKEND=Agg python tests/smoke_inversion.py
# Save reconstruction/convergence plots, tensors, histories, and a JSON report:
MPLBACKEND=Agg python tests/smoke_inversion.py --output .
```

With `--output .`, plots are saved to `figures/smoke/reconstructions.png` and
`figures/smoke/convergence.png`. Other output roots place them under
`<root>/figures/smoke/`. Without `--output`, the smoke check saves no files.

The smoke case uses a 20×20 model, two sources, three receivers, 100 time samples,
and two optimization updates per method. It checks finite models and a reduction
in the independently evaluated objective for all four methods. Its geometry,
frequency, time step, and small identity regularizer are deliberately different
from the full paper cases. Regression tests cover finite-difference adjoints,
TV derivatives, source gradients/counters, packaged inputs, and the original
regularizer tests. The GOGN gradient and TV direction are regression-tested; all 24 tests pass.
`python tests/verify_diags.py` additionally checks the symbolic diagonal formula.
To exercise the corrected non-diagonal inverse in an acoustic inversion:

```bash
MPLBACKEND=Agg python tests/smoke_inversion.py --regularizer smoothing --output output/smoke-smoothing
MPLBACKEND=Agg python tests/smoke_inversion.py --regularizer tv --output output/smoke-tv
python tests/diagnose_gogn.py  # algebraic examples of the former issues 2 and 3
```

The smoothing variant uses a small Laplacian-plus-identity operator. Its plots are under
`output/smoke-smoothing/figures/smoke/`; TV plots are under
`output/smoke-tv/figures/smoke/`. TV keeps the existing regularizer-solve settings;
quadratic cases use q=m−m₀ without an extra solve.

## Layout and names

- `gogn/objectives/FWI.py`: acoustic waveform modeling, data, weights, spatial
  operators, plotting, and residual wrappers.
- `gogn/objectives/FWITT.py`: travel-time shifts and p-power misfits used for the
  rebuttal; **TT means travel time**, not an alternative waveform-norm frontend.
- `gogn/objectives/FWI/`: the model image and authoritative acquisition arrays.
- `gogn/optimization/`: existing optimizers, line searches, CG, and histories.
- `gogn/regularization/`: base, quadratic, and TV regularizers and existing tests.
- `tests/`: executable experiment/plotting scripts and small regression tests.
- `results/`, `figures/`, `output/`: generated on demand and ignored by Git.

The import/distribution name is now `gogn`. The optimizer
function is `GOGN`. The historical output suffix `SFGN` is retained
so original result files can still be plotted; figures label it GOGN. Likewise
`CG` means NLCG and `GNCG` means GN-CG. `acousticTT` and `acousticTV` remain intact.

| Coverage argument | Meaning | Sources |
|---|---|---:|
| `good` | Uniform coverage | 8 |
| `bad` | Realistic coverage | 5 |
| `good25` | Uniform coverage with additional sources | 25 |
| `bad25` | Realistic coverage with additional sources | 25 |

`good` and `bad` are retained dataset identifiers. Source/receiver arrays and
`smiley.png` are included in installed package data. `generategoodsources.py` and
`generatebadsources.py` are historical provenance; they are not installation or
reproduction steps and deliberately refuse to overwrite existing directories.

## Run the experiments

After activating `gogn`, use the existing launchers:

```bash
bash runalltests.sh  # quadratic waveform: 4 coverages × 3 noise levels × 4 methods
bash runTVtests.sh   # TV waveform: 4 coverages × 3 noise levels × 4 methods
bash runTTtests.sh   # travel time: 2 coverages × 6 p values × 4 methods
bash plotalltests.sh # replot existing quadratic waveform results only
```

Each experiment launcher now runs the inversion **and** plotting. It stops on
failure. The TT launcher no longer assumes cached inversions exist. All waveform
launchers include σ = 0.01, 0.1, and 1. TT retains σ = 0.01 and
p = 0.5, 1, 1.5, 2, 2.5, 3; only p = 0.5 is shown in the supplied rebuttal.
The TV σ = 0.1 cases are retained from the original launcher even though the
manuscript TV appendix displays 0.01 and 1.

For one coverage and noise level:

```bash
python tests/GOGN-test.py good 1e-1
python tests/plot-GOGN-test.py good 1e-1
python tests/GOGN-TV-test.py bad 1e-2
python tests/plot-GOGN-TV-test.py bad 1e-2
python tests/GOGNTT-test.py good 1e-2 0.5
python tests/plot-GOGNTT-test.py good 1e-2 0.5
```

Scripts find input data relative to their own location. Outputs default to this
repository, irrespective of the invocation directory. Set `GOGN_OUTPUT_DIR` to
an absolute output root to keep runs separate; pass the same value when plotting:

```bash
GOGN_OUTPUT_DIR=/tmp/gogn-run bash runalltests.sh
```

Launchers accept `PYTHON` for an alternative Python executable and default to the
headless Matplotlib backend `Agg`. For direct scripts on a headless machine, set
`MPLBACKEND=Agg`. Repeating a case overwrites its output files. Plot resolution
is controlled by `tests/DPI` (400). Use the same textual noise/p arguments when
running the plot and assembly scripts: plot directories retain argument spelling
(e.g. `1e-1`), while tensor/CSV filenames use Python float formatting (`0.1`).

## Reproduce figures

For each table row, run the named experiment and plot script for **both** coverage
arguments, at the stated noise level. `GOGN-test.py` writes `results/acoustic/`;
`GOGN-TV-test.py` writes `results/acousticTV/`; `GOGNTT-test.py` writes
`results/acousticTT/`. Plotting writes the corresponding `figures/<family>/`.

| Manuscript figures | Experiment/plot stem | Coverages | Noise argument |
|---|---|---|---|
| 1: target and acquisition layouts | `plot-paper-figures.py --figures 1` (no inversion) | all four | — |
| 2–3 | `GOGN-test` / `plot-GOGN-test` | `good`, `bad` | `1e-1` |
| 4–5 | same | `good25`, `bad25` | `1e-1` |
| 6–7 | same | `good`, `bad` | `1e-2` |
| 8–9 | same | `good25`, `bad25` | `1e-2` |
| 10–11 | same | `good`, `bad` | `1` |
| 12–13 | same | `good25`, `bad25` | `1` |
| 14–15 | `GOGN-TV-test` / `plot-GOGN-TV-test` | `good`, `bad` | `1e-2` |
| 16–17 | same | `good`, `bad` | `1` |
| 18–19 | same | `good25`, `bad25` | `1e-2` |
| 20–21 | same | `good25`, `bad25` | `1` |

Even-numbered figures 2–20 assemble `work-err.png`, `work-grad.png`, and
`work-obj.png` in that order, with uniform coverage above realistic coverage.
The following odd-numbered figure assembles the two `reconstructions.png`
panels. The existing per-method reconstruction layout and color limits are
preserved. Other `iter-*`, `time-*`, and `iter-step.png` files are supplementary.

| Rebuttal figures | Scripts | Coverage pair | Noise / p |
|---|---|---|---|
| 1–2 | `GOGNTT-test.py`, `plot-GOGNTT-test.py` | `good`, `bad` | `1e-2` / `0.5` |
| 3–4 | `GOGN-TV-test.py`, `plot-GOGN-TV-test.py` | `good`, `bad` | `1e-2` |
| 5–6 | same | `good`, `bad` | `1` |

After plotting, assemble standalone figure images (the manuscript's LaTeX captions
are not embedded):

```bash
python tests/plot-paper-figures.py                 # figures/paper/figure-01.png … 21
python tests/plot-paper-figures.py --rebuttal      # figures/rebuttal/figure-01.png … 06
python tests/plot-paper-figures.py --figures 2 3   # select already computed cases
```

The assembly script combines existing panels; it does not run inversions or
substitute old results. `tests/make-table.py` summarizes all main waveform CSVs
when they exist. Historical manuscript assets are not evidence of a new run.

## Numerical settings, randomness, and cost

Full runs retain nx = 200 (40,000 model parameters), nt = 200, dt = 1,
frequency 0.1, background speed 3000, eighth-order Deepwave propagation, original
filtering, and supplied geometries. Main/TV optimizer limits are maxiter = 500
and maxsim = 100; TT uses maxiter = 100 and maxsim = 100. The initial history
entry counts toward maxiter, and a final step may exceed maxsim before the next
stopping check. Settings are kept directly in the existing experiment scripts.

The data functions retain `torch.manual_seed(3906)`; geometry generators retain
their original NumPy seeds. Determinism means repeatable input generation, not
bitwise agreement across processors, thread counts, or library builds.

Quadratic waveform and TT runs build/cache a **40000×40000 dense Cholesky
factor**. One float32 matrix occupies 6.4 GB (5.96 GiB); construction and
factorization require several matrices plus workspace. Plan for tens of GB of
RAM and about 12.8 GB of disk for both factors, in addition to wavefields and
outputs. The factor is reused across compatible coverages/noise levels;
filenames under `results/cache/` encode its parameters. Do not use an old
unverified `D2lo.pt`. TV uses an iterative regularizer solve instead.

Full suites contain 144 optimizer runs total and are not quick tests. Wall time
and peak memory depend strongly on CPU/BLAS threading and acquisition count;
no validated full-suite runtime estimate is available. GOGN runs source gradients
in a thread pool. Setting `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` can help avoid
oversubscription; record any such settings when comparing timings.

The CSV columns `Forwards` and `Backwards` count wrapper passes, normalized to
one source batch; plot scripts sum them for the legacy “PDE Solves” axis.
They exclude setup and are **not literal single-source solve counts**.
They should be treated as reproducible work counters for the supplied wrappers,
not as universal hardware-independent timing or cost measurements.

## Citation and licensing

Use `CITATION.cff` and the accompanying manuscript. The `LICENSE` file records
the current reuse terms for this repository.
