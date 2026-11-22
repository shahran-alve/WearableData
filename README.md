# WearableData: participant-level inference for longitudinal wearable data

Code for the paper

> Shahran Rahman Alve, Sabbir Ahmed, and Latifur Khan. **Participant-Level Inference
> for Machine Learning on Large-Scale Longitudinal Wearable Data.** IEEE International
> Conference on Big Data (IEEE BigData), 2026.

Wearable studies record many windows per participant but predict one outcome per
participant. This repository implements a participant-label permutation test that
reruns the complete analysis, including fold construction and model selection, on
every permuted labeling. It also contains null recycling, which estimates the
Type-I error of cheaper shortcut tests from stored permutation statistics, and a
lazy refitting engine that runs the test in parallel on large cohorts.

## Setup

Python 3.11 or 3.12. `requirements-lock.txt` pins the versions the tests were run
with.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest tests -o addopts="" -rs
```

Expected: 115 passed, 1 skipped. The skipped test needs the raw NHANES files. On
macOS, check that `.venv/bin/python --version` reports 3.11 or 3.12 before
installing.

Participant identifiers are salted hashes. Set a salt before building a workspace:

```bash
export WEARABLEDATA_ID_SALT=$(python -c "import secrets;print(secrets.token_hex(32))")
```

## Data

No data are redistributed.

| Dataset | Source | Location |
|---|---|---|
| Depresjon | https://datasets.simula.no/depresjon/ | `data/raw/depresjon/` |
| StudentLife | https://studentlife.cs.dartmouth.edu/dataset.html | `data/raw/studentlife/dataset/` |
| NHANES 2011–2014 | DEMO, DPQ, PAXHD and PAXDAY files for cycles G and H, from https://wwwn.cdc.gov/nchs/nhanes/ | `data/raw/nhanes/` |

`python scripts/inspect_archive.py --dataset all` checks that the Depresjon and
StudentLife files are in the expected layout, and
`python scripts/acquire_data.py --dataset all --register` records their hashes.

## Reproducing the paper

Run the commands from the repository root with the environment active. Permutation
runs write a checkpoint as they go and resume when rerun; `--deadline` and
`--time-budget` split long runs. Seeds are derived from names, so permutation index
`i` always gives the same labeling.

### Simulations (no data needed)

| Paper | Command |
|---|---|
| Table II | `python scripts/simulation_study.py --participants 20 80 --replicates 40 --permutations 99`<br>`python scripts/simulation_study.py --participants 40 --replicates 60 --permutations 99` |
| Table III, Fig. 4 | `python scripts/records_simulation.py --delta 0 --windows 10 30 100 300 --replicates 200`<br>`python scripts/records_simulation.py --delta 0 --windows 1000 --replicates 100`<br>`python scripts/records_simulation.py --delta 0.3 --windows 10 30 100 300 --replicates 100` |
| Table IV | `python scripts/selection_null_simulation.py --K 1 4 8 --replicates 30` |

The outputs behind Tables II and IV are stored in `reports/simulation/`.

### Depresjon and StudentLife

```bash
python scripts/build_workspace.py --source depresjon --out reports/real_depresjon
python scripts/build_workspace.py --source studentlife --target phq9_post --out reports/real_sl_phq9
```

| Paper | Command |
|---|---|
| Table V | `python scripts/participant_level_eval.py --workspace reports/real_depresjon --permutations 201`<br>`python scripts/participant_level_eval.py --workspace reports/real_sl_phq9 --permutations 57` |
| Table VI | `python scripts/records_sweep_depresjon.py --workspace reports/real_depresjon` |
| Section VI, inpatients excluded | `python scripts/participant_level_eval.py --workspace reports/real_depresjon --permutations 112 --tag noinpatient --subset-participants reports/real_depresjon/results/cohorts/inpatients_excluded.json` |
| Section VI, first nine windows | `python scripts/records_sweep_depresjon.py --workspace reports/real_depresjon --windows 9 --permutations 999` |
| Section VI, fold schedules | `python scripts/partition_stability.py --workspace reports/real_depresjon --seeds 23`<br>`python scripts/partition_stability.py --workspace reports/real_sl_phq9 --seeds 6` |
| Section VI, within-fold ranks | `python scripts/scale_robust_pooling.py --workspace reports/real_depresjon --permutations 110 --tag rankpooled` |

The stored results for Table V and Section VI are in
`reports/real_depresjon/results/` and `reports/real_sl_phq9/results/`, and
`reports/final_results.json` collects the Table V values. The files ending in
`_FIXEDFOLDS_SUPERSEDED.csv` are the earlier runs that reused the observed-label
folds (null mean 0.471), which Section VI describes.

### NHANES

```bash
N=reports/real_nhanes
python scripts/build_nhanes.py --download --pytest
python scripts/prepare_nhanes_analyses.py blocks --workspace $N
python scripts/prepare_nhanes_analyses.py subsamples --workspace $N
```

The cohort flow in `$N/validation/cohort_flow.csv` should end with 8,277
participants (744 positive) and 54,532 person-days. The sex-by-age blocks use the
age bands 18–39, 40–59 and 60–80.

The eight nulls of Table VII, with B = 999 each (add `--n-jobs` to match your cores):

```bash
python scripts/fast_selection_permutation.py observed --workspace $N
python scripts/fast_selection_permutation.py run --workspace $N --permutations 999 --fixed-only --typeI
python scripts/fast_selection_permutation.py run --workspace $N --permutations 999 --typeI
python scripts/fast_selection_permutation.py run --workspace $N --permutations 999 --fixed-only --block-col sdmvstra
python scripts/fast_selection_permutation.py run --workspace $N --permutations 999 --block-col sdmvstra
python scripts/fast_selection_permutation.py run --workspace $N --permutations 999 --fixed-only \
    --block-file $N/validation/blocks.csv --block-col strata_psu
python scripts/fast_selection_permutation.py run --workspace $N --permutations 999 --fixed-only \
    --block-file $N/validation/blocks.csv --block-col sex_age
python scripts/perfold_selection_permutation.py observed --workspace $N
python scripts/perfold_selection_permutation.py run --workspace $N --permutations 999 --with-global
python scripts/perfold_selection_permutation.py run --workspace $N --permutations 999 --with-global --block-col sdmvstra
```

| Paper | Source |
|---|---|
| Table VII | the `*_fast.json` and `perfold_*.json` files in `$N/results/` |
| Table VIII | the `--typeI` output of the runs above, and `perfold_selection_permutation.py report` on each per-fold checkpoint (`recycled_typeI`) |
| Table I, Fig. 3 | `python scripts/fast_selection_permutation.py timing --workspace $N`, and `--perm-log` on any run |
| Section VII, BBC-CV | `python scripts/bbc_cv.py run --workspace $N --tag nhanes` |
| Besag–Clifford replay | `python scripts/fast_selection_permutation.py run --workspace $N --permutations 999 --sequential --h 10 --report-only` |

Fig. 1 and Fig. 5 compare global and per-fold selection across cohort sizes. Run
the following for each subsample workspace, `reports/real_nhanes_n100` to
`reports/real_nhanes_n4000`, and for the full cohort:

```bash
W=reports/real_nhanes_n1000
python scripts/perfold_selection_permutation.py observed --workspace $W --out $W/results/perfold_observed.json
python scripts/perfold_selection_permutation.py run --workspace $W --permutations 999 --with-global
python scripts/perfold_selection_permutation.py report --observed $W/results/perfold_observed.json \
    --checkpoint $W/results/perfold_null_checkpoint.csv --out $W/results/perfold_report.json
```

In `perfold_report.json`, `recycled_typeI.global_vs_fixed_null` is the Fig. 1 rate
and `null_shift_vs_fixed` is the Fig. 5 shift. The synthetic 40-participant point
of Fig. 1 and the synthetic column of Table VIII use the same commands on

```bash
python scripts/make_synthetic_workspace.py --out reports/synthetic_n40 --participants 40 \
    --positives 12 --windows 560 --effect 0
```

The eight nulls of Table VII took 419 core-hours.

### Check the engine without data

```bash
W=/tmp/syn
python scripts/make_synthetic_workspace.py --out $W --participants 40 --positives 12 --windows 600 --effect 0
python scripts/fast_selection_permutation.py run --workspace $W --permutations 19 --n-jobs 2
python scripts/fast_selection_permutation.py equality --workspaces $W --out equality.csv
```

`equality.csv` should report zero mismatches between the lazy path and the full
analysis.

## Repository layout

```
configs/          dataset sources, windowing and personalization settings
src/wearabledata/     readers, windowing, features, splits, permutation engines, statistics
scripts/          command-line entry points
tests/            unit and integration tests
reports/          stored results for Tables II, IV, V and Section VI
data/manifests/   file hashes of the raw data
```

## Scope

Research software. Outputs are research estimates, not diagnoses or clinical
recommendations. The permutation test assumes that participant labels are
exchangeable under the null and that the analysis is fixed in advance.

## Citation

See `CITATION.cff`.

## License

MIT
