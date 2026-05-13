# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DIVERGENCE.md` documenting the historical relationship with MARP and QRAcc downstream repos.
- `CHANGELOG.md` (this file).
- `scripts/` folder for non-test utility scripts.
- `notebooks/` folder to consolidate all Jupyter notebooks.
- Git LFS tracking for `*.onnx` model files via `.gitattributes`.
- `pyproject.toml` now declares all runtime and dev dependencies, making `pip install -e ".[dev]"` fully functional.
- Version `0.1.0` declared in `pyproject.toml`.
- `Cgraph.get_matrix_dict()` method (replaces dead `huaimc.py` utility).
- `hwacctools/comp_graph/cnode_factory.py` now contains `get_cnode_from_onnx_node`, cleanly separating node construction from node definitions.
- `hwacctools/accsim/sim.py` now accepts a CLI argument for the model path and core size.

### Changed
- `environment.yml` replaced with a minimal, portable conda spec (no machine-specific prefix, no pinned build hashes).
- CI workflow replaced: pip-based (no conda), Python 3.11, `ruff` linting enabled, caching added, triggers scoped to `main`/`master` branches only.
- `.gitignore` extended with standard Python and notebook patterns.
- README: corrected `env.yml` → `environment.yml`; added CUDA installation note.
- `get_cnode_from_onnx_node` depthwise Conv branch: fixed unreachable `return(catter)` — now correctly returns `convs + [catter]` so the concatenator node is included in the graph.

### Removed
- `model_staging` broken submodule (pointed to a private SSH URL; directory was empty).
- `hwacctools/comp_graph/test_cgraph.py` — broken legacy test file (bare imports, hardcoded Windows paths, non-existent image references).
- `hwacctools/comp_graph/cnodes_hardwarelike.py` — broken file (bare imports, class name mismatch bug).
- `hwacctools/comp_graph/huaimc.py` — 7-line dead file (bare imports, functionality absorbed into `Cgraph.get_matrix_dict()`).

## [0.0-legacy] — snapshot before cleanup

Baseline state of the repository before this cleanup effort. See git tag `v0.0-legacy` for the exact snapshot.
