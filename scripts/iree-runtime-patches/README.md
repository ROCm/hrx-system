# IREE Runtime Patch Stack

HRX-local IREE patches live here as `*.patch` files with paths relative to the
IREE repository root.

Normal flow:

1. `vendor_iree_runtime.py import-pristine` creates the pristine import commit.
2. `vendor_iree_runtime.py apply-patches` creates one commit per patch.
3. Edit IREE under `third_party/iree-runtime` in follow-up commits.
4. `vendor_iree_runtime.py dump-patches --diffbase <pristine-commit>` refreshes
   this directory from those commits.
