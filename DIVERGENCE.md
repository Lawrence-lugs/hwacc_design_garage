# Code Divergence Audit

This document tracks the relationship between `hwacc_design_garage` and the
downstream repositories that consume its code.

## Background

`hwacc_design_garage` was originally used as a submodule inside:

- **[MARP](https://github.com/Lawrence-lugs/MARP)** — Multi-Accelerator Research Platform
- **[QrAccelerator](https://github.com/Lawrence-lugs/QrAccelerator)** — Charge-redistribution mixed-signal DNN accelerator (digital part), which uses MARP as a submodule

## Audit Results (performed 2026-05-13 against MARP commit `c6bfcdf`)

QrAccelerator points to the same MARP commit as a submodule, so diffing MARP
is sufficient.  MARP does **not** embed `hwacctools/` as a frozen copy — it has
entirely rewritten its graph-compilation logic into `marp/compile/compile.py`
and related files.  The files that **are** shared (via copy-and-evolve) are:

| Local file | MARP file | Diff summary |
|---|---|---|
| `hwacctools/onnx_tools/onnx_splitter.py` | `marp/onnx_tools/onnx_splitter.py` | **Identical** except MARP adds a module docstring and `from __future__ import annotations`. No logic differences. |
| `hwacctools/onnx_utils.py` | `marp/onnx_tools/onnx_utils.py` | **Identical** except MARP has `from __future__ import annotations` and alphabetically sorted imports. |
| `hwacctools/quantization/quant.py` | `marp/quantization/quant.py` | **Identical** except MARP adds a module docstring and `from __future__ import annotations`. |

### Changes imported from MARP

The minor style improvements (module docstrings, `from __future__ import annotations`,
sorted imports) have been applied to the three files above.

### MARP-only code (not ported back)

MARP's `marp/compile/` is a purpose-built compiler that maps ONNX sub-graphs
to MARP accelerator micro-programs. It has no direct equivalent in
`hwacctools/` and is out of scope for this repo.

## Decision

Going forward, improvements to the shared modules (`onnx_splitter.py`,
`onnx_utils.py`, `quant.py`) should originate here and be manually synced to
MARP, or MARP should depend on this package via `pip install`. The modules are
small enough that either approach works.
