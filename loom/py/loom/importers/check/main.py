# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line implementation for importer checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.importers.check.registry import (
    BackendRegistry,
    make_default_registry,
    print_importers,
    skip_unavailable_backend,
)
from loom.importers.check.results import (
    CheckResult,
    dump_check_results,
    results_to_json,
    summarize_results,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: BackendRegistry | None = None,
) -> int:
    registry = registry or make_default_registry()
    parser = argparse.ArgumentParser(
        prog="loom-import-check",
        description=(
            "Run importer golden tests and optionally update inline Loom IR output."
        ),
    )
    parser.add_argument(
        "--agents_md",
        action="store_true",
        help="print agent-oriented Markdown usage and exit",
    )
    parser.add_argument(
        "--list-importers",
        action="store_true",
        help="list enabled and disabled importer check backends",
    )
    subparsers = parser.add_subparsers(dest="importer")
    for candidate in registry.all():
        help_text = candidate.backend.help
        if not candidate.availability.available:
            help_text = f"{help_text} (disabled: {candidate.availability.message()})"
        subparser = subparsers.add_parser(candidate.backend.name, help=help_text)
        candidate.backend.add_arguments(subparser)
        _add_common_check_arguments(subparser)
    args = parser.parse_args(argv)
    if args.agents_md:
        sys.stdout.write(f"{agent_markdown(registry)}\n")
        return 0
    if args.list_importers:
        sys.stdout.write(f"{print_importers(registry)}\n")
        return 0
    if args.importer is None:
        parser.error("importer is required unless --list-importers is used")
    selected = registry.by_name(args.importer)
    if selected is None:
        parser.error(f"unknown importer: {args.importer}")
        raise AssertionError("argparse.error returned")

    availability = selected.availability
    if availability.available:
        availability = selected.backend.prepare(args)
    if not availability.available:
        results = skip_unavailable_backend(selected, args, availability)
    else:
        results = selected.backend.run(args)

    if args.dump_temp_dir is not None:
        dump_check_results(results, args.dump_temp_dir)

    if args.json:
        sys.stdout.write(f"{results_to_json(results)}\n")
    else:
        for result in results:
            sys.stdout.write(f"{result.status}: {_result_location(result)}\n")
            if result.mismatch:
                sys.stdout.write(f"  {result.mismatch}\n")
                if result.diff:
                    sys.stdout.write(result.diff)
            if result.skipped or not result.passed:
                first_line = _diagnostic_line(result.stderr or result.stdout)
                if first_line:
                    sys.stdout.write(f"  {first_line}\n")
        summary = summarize_results(results)
        sys.stdout.write(
            "summary: "
            f"{summary['passed']} passed, "
            f"{summary['updated']} updated, "
            f"{summary['skipped']} skipped, "
            f"{summary['failed']} failed, "
            f"{summary['crashed']} crashed"
            "\n"
        )
    if not results:
        sys.stderr.write("error: no importer check cases matched\n")
        return 1
    if args.fail_on_skip and any(result.skipped for result in results):
        return 1
    return 0 if all(result.passed for result in results) else 1


def agent_markdown(registry: BackendRegistry) -> str:
    """Returns agent-facing usage documentation for importer checks."""

    importers = print_importers(registry)
    return f"""## loom-import-check

Importer setup and build-system selection are documented in
`loom/py/loom/importers/README.md` and the per-importer READMEs. This command's
agent notes document the shared fixture check runner, inline expected output,
update mode, diagnostics, and oracle sidecar capture.

`loom-import-check` runs importer golden tests for importer-specific fixtures.
Use checked-in Bazel test targets for normal verification and update flows so
the test environment matches CI.

### Verify tests

```shell
iree-bazel-test --config=asan //loom/py/loom/importers/check:check_test
python dev.py importers setup tilelang
iree-bazel-test --config=asan \\
    --importer-env mlir \\
    //loom/py/loom/importers/mlir:mlir_import_test
iree-bazel-test --config=asan \\
    --importer-env tilelang \\
    //loom/py/loom/importers/check/tilelang:tilelang_test
iree-bazel-test --config=asan \\
    --importer-env tilelang \\
    //loom/py/loom/importers/tilelang:tilelang_import_test
```

Importer-specific tests are disabled by default because the frontend packages
are not part of the base Python environment. Bazel enables importer targets
through the importer-level `--//loom/config/import:enable=<name>[,<name>...]`
build setting. Prefer `--importer-env <name>` so the repo-local compatibility
setting and locked site-packages path are selected together. Raw configs such as
`--config=loom-importer-mlir`, `--config=loom-importer-tilelang`, and
`--config=loom-importers` only select build compatibility; they do not fetch
frontend Python packages or expand the base module lock. The importer backends
probe the runtime Python environment and report structured skips when the
selected package is unavailable. CMake uses matching top-level options:

```shell
iree-cmake-configure --importer-env tilelang
iree-cmake-test --importer-env tilelang -R tilelang
```

For a local Bazel checkout that always runs importer tests, put the build
setting in the untracked user bazelrc generated by `iree-bazel-configure`, for
example `build --config=loom-importers`. The selected Python runtime must still
make the relevant frontend package importable.

The TileLang package environment is locked separately in
`requirements-importers-tilelang.lock.txt` and materialized under `.tmp/` by
`python dev.py importers setup tilelang`. This is deliberate: importer
frontends can be large, platform-specific, and faster-moving than the base
developer toolchain, but Bazel and CMake still consume the same checked-in lock
and the same `--importer-env tilelang` selection.

### Update expected output

Run the checker binary from the worktree with the same importer environment
used for verification:

```shell
iree-bazel-build //loom/src/loom/tools/loom-opt
iree-bazel-run --importer-env <name> \\
    //loom/py/loom/importers/check:loom_import_check -- \\
    <name> --update \\
    --loom-opt=bazel-bin/loom/src/loom/tools/loom-opt/loom-opt \\
    <fixture.py> [<fixture.py> ...]
```

Bazel test runfiles are read-only, so `iree-bazel-test` remains the verification
path and source mutation belongs in `iree-bazel-run`. The fixture runner still
fails if an importer crashes, produces a failed case, or emits Loom IR rejected
by the production verifier.

### Fixture shape

Python importer fixtures keep shared imports at the top of the file and split
cases with `# ====`. Each case compares imported Loom IR against the inline
`# ----` block. Expected Loom IR is stored as a raw triple-quoted string literal
so it remains valid Python while still being easy to copy into a `.loom` file.
Run update mode after intentional importer output changes instead of editing
the expected Loom IR by hand.

Importer check file sets belong in BUILD filegroups under the importer they
exercise and are passed to the test runner with `$(locations ...)`. Do not
hardcode testdata paths in Python tests.

Checked-in importer tests also provide `loom-opt` to the check tool so
generated Loom IR is verified before expected-output comparison or `--update`
rewrites. Bazel and CMake tests pass `--loom-opt` explicitly from build
metadata. Direct use resolves `loom-opt` from `--loom-opt`, `LOOM_BIN_DIR`, or
`PATH`; importer-check Python does not inspect Bazel runfiles or repository
paths.

TileLang checks also have an opt-in oracle lane for comparing against
TileLang/TVM generated artifacts. `--oracle=source` asks TileLang for generated
device source, `--oracle=code-object` additionally compiles, unbundles, and
externally disassembles a code object when the ROCm tools are available, and
`--oracle=differential` also compiles the imported Loom IR through
`loom-compile --backend=amdgpu-hal` for side-by-side disassembly-family
comparison. This metadata is sidecar validation evidence: the checked stdout
remains imported Loom IR. Use `--oracle-output-dir` or `--dump-temp-dir` to
retain the generated source, bundled HSACO, unbundled code object, Loom HSACO,
raw disassembly, metadata JSON, and parsed instruction-summary JSON.
Oracle modes detect their optional dependencies at runtime. Missing TileLang
source-codegen hooks, HIP/ROCm tools, or LLVM tools produce skipped cases with
structured JSON metadata, not fake passes, and do not expand the minimal
TileLang import dependency contract.

Expected diagnostics use loom-check-style source annotations:

```python
# ERROR@+1: TYPE/001 {{field_a="lhs"}} "same type"
bad_program()
```

Annotations match the shared Python diagnostic model: severity, source line,
optional error domain/code, optional structured diagnostic params, and quoted
message substrings.

### Direct use

Direct invocations are useful for inspection, but Bazel remains the update path
for checked-in tests:

```shell
iree-bazel-run --importer-env mlir --importer-env tilelang \\
    //loom/py/loom/importers/check:loom_import_check -- --list-importers
iree-bazel-run --importer-env tilelang \\
    //loom/py/loom/importers/check:loom_import_check -- \\
    tilelang path/to/case.py
iree-bazel-run --importer-env tilelang \\
    //loom/py/loom/importers/check:loom_import_check -- \\
    tilelang --update path/to/case.py
```

### Importers

```text
{importers}
```"""


def _add_common_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--filter",
        dest="case_filter",
        help=(
            "only run cases whose path, case index, run directive, or "
            "importer-specific labels contain this substring"
        ),
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="return a failing exit code when optional importer checks are skipped",
    )
    parser.add_argument(
        "--dump-temp-dir",
        type=Path,
        help="write per-case importer check artifacts under this directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable check results",
    )
    parser.add_argument(
        "--loom-opt",
        type=Path,
        help=(
            "path to loom-opt used to verify generated Loom IR before "
            "comparing or updating expected output; when omitted, "
            "LOOM_BIN_DIR/loom-opt or PATH is used if available"
        ),
    )


def _result_location(result: CheckResult) -> str:
    if result.case_index < 0:
        return str(result.path)
    return f"{result.path}:case{result.case_index}"


def _diagnostic_line(text: str) -> str | None:
    fallback: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if fallback is None and "RuntimeWarning:" not in stripped:
            fallback = stripped
        if stripped.startswith(("error:", "fatal:", "Traceback ")):
            return stripped
    return fallback
