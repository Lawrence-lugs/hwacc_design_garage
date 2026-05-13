# Code Divergence Audit

This document tracks the relationship between `hwacc_design_garage` and the
downstream repositories that once consumed it as a submodule.

## Background

`hwacc_design_garage` was originally used as a submodule inside:

- **[MARP](https://github.com/Lawrence-lugs/MARP)** — Multi-Accelerator Research Platform
- **[QRAcc](https://github.com/Lawrence-lugs/QRAcc)** — Quantized-Re-configurable Accelerator

At some point those projects extracted frozen copies of the relevant modules
rather than continuing to track the submodule. It is currently **unknown**
whether those frozen copies are ahead of, behind, or diverged from the code
in this repository.

## Audit Status

| Module | hwacc_design_garage | MARP frozen copy | QRAcc frozen copy | Canonical source |
|--------|-------------------|-----------------|-------------------|-----------------|
| `comp_graph/cnodes.py` | ? | ? | ? | **TBD** |
| `comp_graph/cgraph.py` | ? | ? | ? | **TBD** |
| `comp_graph/splitter.py` | ? | ? | ? | **TBD** |
| `comp_graph/core.py` | ? | ? | ? | **TBD** |
| `quantization/quant.py` | ? | ? | ? | **TBD** |
| `onnx_tools/onnx_splitter.py` | ? | ? | ? | **TBD** |

## How to Audit

To fill in the table above, run a diff between each module in this repo and
its counterpart in MARP / QRAcc:

```bash
# Example: compare cnodes.py with MARP's frozen copy
diff hwacctools/comp_graph/cnodes.py \
     path/to/MARP/hwacc_design_garage/hwacctools/comp_graph/cnodes.py
```

## Decision

Once the audit is complete, pick one canonical source for each module and
merge improvements back. Going forward, downstream projects should depend on
this package via `pip install` (pinned to a release tag) rather than
embedding a frozen copy.
