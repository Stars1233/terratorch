# CLAUDE.md

Guidance for working in this repo. Distilled from an integration-test run on the
IBM BlueVela (Vela) LSF cluster.

## Running integration tests on BlueVela (LSF)

- Remote: `rkie@login3.bluevela.rmf.ibm.com`; work under `/proj/partnership-mat/terratorch`.
- Usable Python: `/gpfs/ess6000-1/optshare/share/miniconda/bin/python3.12`
  (login-node `python3` is 3.9.21 — too old). Build a venv and `pip install -e ".[test]"`.
  An `import terratorch` segfault on the **login node** is an OpenBLAS `RLIMIT_NPROC`
  artifact, not a real failure — compute nodes run fine.
- Submit with `scripts/run_lsf_integrationtest.sh <branch> <workdir>` (per-test bsub
  jobs; `test_models_fit` is the prerequisite that produces checkpoints, dependents
  wait on it). Each job: 1 GPU, 8 CPU, 32GB, queue `normal`. Monitor: `bjobs -J 'tt_rkie_*'`.

### Cluster gotchas
1. **Clone via HTTPS, not SSH.** `git@github.com` fails (no publickey on nodes).
   The shipped `run_lsf_integrationtest.sh` clones via SSH — pre-clone the branch
   dir via HTTPS with an HTTPS `origin` so the script skips its clone.
2. **esub requires `-G grp_partnership_mat` on every bsub.** The shipped script
   omits it → all jobs rejected. Inject `-G grp_partnership_mat` into each `bsub`.
3. **No `/dccstor` mount on this cluster.** Tests hardcode `/dccstor` for datasets,
   testing_models, and tmp. See data override below.

## Test data paths are overridable (PR #1206)

`integrationtests/test_base_set.py` reads `TERRATORCH_DATA_ROOT` (default `/dccstor`,
backward-compatible). `TEST_MODELS_ROOT` and `TMP_ROOT` derive from it. On Vela the
`/dccstor` datasets (~63G: Poland-Buildings, sen1floods11_v1.1, senfloods_multimodal,
fire-scars 6-bands, testing_models) are mirrored at `…/terratorch/datasets`, preserving
`/dccstor/...` sub-paths so a prefix swap resolves them.

To run against the mirror:
```
TERRATORCH_DATA_ROOT=/proj/partnership-mat/terratorch/datasets
TERRATORCH_TMP_ROOT=/proj/partnership-mat/terratorch/tmp
```

Relevant env vars: `TERRATORCH_DATA_ROOT`, `TERRATORCH_TEST_MODELS_ROOT`,
`TERRATORCH_TEST_CHECKPOINTS_ROOT`, `TERRATORCH_TMP_ROOT`
(documented in `scripts/README_run_lsf_integrationtest.md`).

### Test-run gotchas
- Give `test_surya` its **own** `TERRATORCH_TMP_ROOT` (or run it after cleanup):
  it downloads HF data into `TMP_ROOT/experiment`, and `test_cleanup` wipes `TMP_ROOT`
  concurrently → cleanup-vs-surya race. surya does not use the `/dccstor` mirror.
- Retry any **CUDA OOM** test on an **exclusive** GPU — shared-node contention, not
  a data/code issue.

## Releasing (e.g. v1.2.9)

1. Bump version in `pyproject.toml` via a branch + PR to `main`.
2. After merge, create the tag on main: `git tag vX.Y.Z <main-sha> && git push origin vX.Y.Z`.
3. The tag triggers `.github/workflows/publish-pypi.yml`: validates tag is on main →
   builds sdist/wheel → `mkdocs build` → `mike deploy --update-aliases <ver> stable`
   + push gh-pages (docs) → twine upload to PyPI. Docs deploy is part of this workflow.
- Docs live at **http://torchgeo.org/terratorch/** (not ibm.github.io). Verify
  `/<ver>/` and `/stable/`, and that gh-pages `versions.json` lists the new version.
