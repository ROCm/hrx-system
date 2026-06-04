# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command help for the developer command router."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandHelp:
    description: str
    arguments: str | None = None
    epilog: str | None = None


def root_command_help(command: str) -> CommandHelp:
    if command == "setup":
        return CommandHelp(
            description="Set up shared developer tools.",
            epilog="""Examples:
  python dev.py setup
  python dev.py setup --system
  python dev.py setup --tool-root ../.tools/iree-x

The default setup creates or updates the repo-local .venv. Use --system when
tools are already available on PATH and this checkout should not get a managed
tool environment.""",
        )
    if command == "doctor":
        return CommandHelp(
            description="Check shared developer tools and show their versions.",
            epilog="""Examples:
  python dev.py doctor
  python dev.py doctor --system
  python dev.py doctor --tool-root ../.tools/iree-x""",
        )
    return CommandHelp(description=f"Run {command}.")


def lane_command_help(lane: str, command: str) -> CommandHelp:
    build_system_name = {
        "bazel": "Bazel",
        "cmake": "CMake",
    }[lane]
    if command == "setup":
        return CommandHelp(
            description=f"Set up developer tools for {build_system_name}.",
            epilog=f"""Examples:
  python dev.py {lane} setup
  python dev.py {lane} setup --system
  python dev.py {lane} setup --tool-root ../.tools/iree-x

The default setup creates or updates .venv. Use --system when tools are already
installed and this checkout should not get a local tool environment.""",
        )
    if command == "hook":
        default_profile = {
            "bazel": "paranoid",
            "cmake": "default",
        }[lane]
        return CommandHelp(
            description=f"Install Git hooks for {build_system_name}.",
            epilog=f"""Examples:
  python dev.py {lane} hook
  python dev.py {lane} hook --profile {default_profile}
  python dev.py {lane} hook --verify

This writes ignored lefthook-local.yml with the selected build system/profile
and then runs lefthook install. Re-run this command with a different --profile
to change the default profile used by Git commits. The installed hook uses
commit scope: staged files plus files changed by HEAD, so amended commits
validate the commit being replaced.""",
        )
    if command == "configure":
        if lane == "bazel":
            return CommandHelp(
                description="Configure the Bazel source graph for this checkout.",
                arguments="Portable -D project options or documented native Bazel rc options.",
                epilog="""Examples:
  python dev.py bazel configure
  python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python dev.py bazel configure --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm

This writes .bazelrc.configured. Published portable build options live in
BUILDING.md. Use .bazelrc.local for checkout-specific Bazel overrides.""",
            )
        return CommandHelp(
            description="Configure the CMake package/install-test build tree.",
            arguments="CMake configure options.",
            epilog="""Examples:
  python dev.py cmake configure
  python dev.py cmake configure --fresh
  python dev.py cmake configure -DCMAKE_BUILD_TYPE=Debug
  python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python dev.py --cmake-build-dir build/cmake-asan cmake configure -DIREE_ENABLE_ASAN=ON

The build tree lives outside the checkout at ../builds/<checkout-name>/.
Use --cmake-build-dir to select a different tree for CI, experiments, or
parallel configurations.
Published project build options live in BUILDING.md.""",
        )
    if command == "build":
        if lane == "bazel":
            return CommandHelp(
                description="Build Bazel targets.",
                arguments="Bazel targets or build options. Defaults to //runtime/... and //libhrx/....",
                epilog="""Examples:
  python dev.py bazel build
  python dev.py bazel build //libhrx/...
  python dev.py bazel build //runtime/src/iree/base/...

With no explicit target, this builds //runtime/... and //libhrx/....""",
            )
        return CommandHelp(
            description="Build CMake targets in the configured build tree.",
            arguments="Target names followed by native CMake build options.",
            epilog="""Examples:
  python dev.py cmake build hrx
  python dev.py cmake build libhrx_src_libhrx_hrx
  python dev.py cmake build hrx --parallel 8
  python dev.py cmake build --parallel 8

Positional arguments are target names and become cmake --build ... --target
<name>. Option-looking arguments are forwarded to CMake.""",
        )
    if command == "test":
        if lane == "bazel":
            return CommandHelp(
                description="Run Bazel tests with the presubmit configuration.",
                arguments="Bazel test targets. Defaults to //runtime/... and //libhrx/....",
                epilog="""Examples:
  python dev.py bazel test
  python dev.py bazel test //build_tools/devtools:cli_test
  python dev.py bazel test //libhrx/...""",
            )
        return CommandHelp(
            description="Run CTest in the configured CMake build tree.",
            arguments="CTest options.",
            epilog="""Examples:
  python dev.py cmake test
  python dev.py cmake test -R hrx
  python dev.py cmake test --rerun-failed

CTest runs in ../builds/<checkout-name>/ with --output-on-failure.""",
        )
    if command == "query" and lane == "bazel":
        return CommandHelp(
            description="Run Bazel query on the unevaluated build graph.",
            arguments="Bazel query expression and native bazel query options.",
            epilog="""Examples:
  python dev.py bazel query //runtime/src/iree/base:base
  python dev.py bazel query 'kind(cc_library, //runtime/...)'
  python dev.py bazel query 'rdeps(//runtime/..., //runtime/src/iree/base)'

Use cquery when select() resolution or configured target compatibility matters.""",
        )
    if command == "cquery" and lane == "bazel":
        return CommandHelp(
            description="Run Bazel cquery on the configured build graph.",
            arguments="Bazel cquery expression and native bazel cquery options.",
            epilog="""Examples:
  python dev.py bazel cquery //runtime/src/iree/base:base
  python dev.py bazel cquery 'kind(cc_library, //runtime/...)'
  python dev.py bazel cquery --output=files //runtime/src/iree/base:base

cquery evaluates select(), platform constraints, and configured target state.""",
        )
    if command == "info" and lane == "bazel":
        return CommandHelp(
            description="Run Bazel info.",
            arguments="Bazel info keys and native bazel info options.",
            epilog="""Examples:
  python dev.py bazel info
  python dev.py bazel info execution_root
  python dev.py bazel info bazel-bin""",
        )
    if command == "presubmit":
        return CommandHelp(
            description=f"Run CI-shaped acceptance checks for {build_system_name}.",
            epilog=f"""Examples:
  python dev.py {lane} presubmit
  python dev.py {lane} presubmit --profile paranoid
  python dev.py {lane} presubmit --no-project-tests
  python dev.py {lane} presubmit --verbose

Profiles:
  default   Repository hygiene.
  paranoid  Hygiene plus affected project tests and static-analysis providers.
  ci        Full tracked-tree hygiene plus all configured project tests.

The default profile is ci. This is intentionally the expensive full-tree check
that approximates the project test workflows a developer can run locally. Use
--no-project-tests only for CI jobs that have separate project test workflows.
Use `python dev.py {lane} precommit` for local changes only.""",
        )
    if command == "precommit":
        default_profile = {
            "bazel": "paranoid",
            "cmake": "default",
        }[lane]
        if lane == "bazel":
            lane_scope = (
                "Bazel defaults to the paranoid profile: repository "
                "hygiene, affected project tests, and configured "
                "static-analysis providers."
            )
        else:
            lane_scope = (
                "CMake defaults to the default profile. Select "
                "paranoid or ci to add affected project CMake/CTest checks."
            )
        description = f"Run precommit checks for local {build_system_name} changes."
        return CommandHelp(
            description=description,
            epilog=f"""Examples:
  python dev.py {lane} precommit
  python dev.py {lane} precommit --profile {default_profile}
  python dev.py {lane} precommit --commit
  python dev.py {lane} precommit --base origin/main
  python dev.py {lane} precommit --staged
  python dev.py {lane} precommit README.md CONTRIBUTING.md
  python dev.py {lane} precommit --verbose

With no input option, precommit checks staged, unstaged, and untracked files.
`--commit` checks the Git hook scope: staged files plus files changed by HEAD.
`--base` checks branch changes from the merge base with the given ref through
HEAD, plus local staged, unstaged, and untracked files.
Explicit paths check only those files for narrow manual runs.
The default profile is {default_profile}. Use --profile to select default,
paranoid, or ci for this run.
Commit-scope, staged-file, and explicit-path runs with profiles that run tests,
currently paranoid and ci, apply mechanical fixups before running the same
profile in non-mutating check mode. Broader local-change runs and the profile
named default are check-only.

{lane_scope}""",
        )
    if command == "fix":
        return CommandHelp(
            description="Apply staged mechanical fixups.",
            epilog=f"""Examples:
  python dev.py {lane} fix

Fix runs the staged-file formatter/generated-file repair path. Staged
test-bearing precommit runs also use this repair path before validation.
Presubmit stays non-mutating.""",
        )
    if command == "doctor":
        return CommandHelp(
            description=f"Check tools required by {build_system_name}.",
            epilog=f"""Examples:
  python dev.py {lane} doctor
  python dev.py {lane} doctor --system
  python dev.py {lane} doctor --tool-root ../.tools/iree-x""",
        )
    if command == "run" and lane == "bazel":
        return CommandHelp(
            description="Build a Bazel executable target and run it from the current directory.",
            arguments="Bazel target and build options, followed by -- and program arguments.",
            epilog="""Examples:
  python dev.py bazel run //runtime/src/iree/base:allocator_benchmark
  python dev.py bazel run //runtime/src/tools:iree-run-module -- --help
  python dev.py bazel run --config=asan //runtime/src/iree/base:allocator_test
  python dev.py bazel run -p //runtime/src/tools:iree-run-module

This builds first, resolves the configured executable path with cquery, then
execs the binary from the current directory. The Bazel server lock is not held
while the binary runs.""",
        )
    if command == "try" and lane == "bazel":
        return CommandHelp(
            description="Build and run a temporary C/C++ snippet with Bazel.",
            arguments="Source files, -e inline source, Bazel build options, and optional -- program arguments.",
            epilog="""Examples:
  python dev.py bazel try -e 'int main() { return 0; }'
  python dev.py bazel try -x c++ -e $'#include "iree/base/api.h"\\nint main() { return 0; }'
  python dev.py bazel try --dep //runtime/src/iree/base snippet.c -- --flag
  python dev.py bazel try -k -e 'int main() { return 0; }'
  python dev.py bazel try -c -o build/snippet -e 'int main() { return 0; }'

Temporary packages are written under .iree-bazel-try/. The tool infers common
deps from quoted iree/..., loom/..., and loomc/... includes and accepts
explicit --dep labels.""",
        )
    if command == "fuzz" and lane == "bazel":
        return CommandHelp(
            description="Build and run Bazel libFuzzer targets without holding the Bazel lock.",
            arguments="Fuzz target or ... pattern, Bazel build options, followed by -- and fuzzer arguments.",
            epilog="""Examples:
  python dev.py bazel fuzz //runtime/src/iree/tokenizer:special_tokens_fuzz
  python dev.py bazel fuzz //runtime/src/iree/tokenizer/... -- -max_total_time=60

This builds with --config=fuzzer, resolves executable paths with cquery, then
runs the fuzzer binaries directly. Single-target runs exec the fuzzer process.""",
        )
    if command in ("run", "try", "fuzz"):
        return CommandHelp(
            description=f"{command} support for {build_system_name}.",
            arguments="Command arguments.",
            epilog="This command slot is reserved but not wired yet.",
        )
    return CommandHelp(description=f"Run {build_system_name} {command}.")


def agent_markdown(lanes: tuple[str, ...], *, command: str | None = None) -> str:
    if command is not None:
        if len(lanes) != 1:
            raise ValueError("command-specific agent markdown requires one lane")
        return command_agent_markdown(lanes[0], command)

    sections = [agent_markdown_header(), ""]
    for lane in lanes:
        if lane == "bazel":
            sections.append(bazel_agent_markdown())
        elif lane == "cmake":
            sections.append(cmake_agent_markdown())
        else:
            raise ValueError(f"unknown lane: {lane}")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def command_agent_markdown(lane: str, command: str) -> str:
    if lane == "bazel":
        return with_common_wrapper_flags(bazel_command_agent_markdown(command))
    if lane == "cmake":
        return with_common_wrapper_flags(cmake_command_agent_markdown(command))
    raise ValueError(f"unknown lane: {lane}")


def with_common_wrapper_flags(markdown: str) -> str:
    return (
        markdown.rstrip()
        + "\n\nWrapper flags: `-n/--dry-run`, `-v/--verbose`; tool-environment\n"
        "flags such as `--system`, `--venv`, and `--tool-root DIR` also work\n"
        "after the wrapper name. Use `--` before a native option that conflicts\n"
        "with a wrapper flag."
    )


def agent_markdown_header() -> str:
    return """## Developer Commands

Run from the repository root. Prefer generated wrapper aliases such as
`iree-bazel-build`; checked-in `build_tools/bin/iree-bazel-*` launchers are
available for Bazel root-relative scripts and unconfigured shells. Long flags
accept hyphen or underscore spellings, such as `--dry-run` and `--dry_run`.
Common wrapper flags work after the wrapper name: `-n/--dry-run`,
`-v/--verbose`, `--system`, `--venv`, `--tool-root DIR`, and
`--cmake-build-dir DIR`. Command-specific debugging flags include
`iree-bazel-run -p/--print-path` and `iree-bazel-try -k/--keep`. Use `--`
before a native option that conflicts with a wrapper flag. Published build
configuration options live in `BUILDING.md`."""


def bazel_agent_markdown() -> str:
    return """### Bazel

Use `iree-bazel-*` for Bazel source-tree builds, tests, graph queries, and
review checks. `iree-bazel-build` and `iree-bazel-test` default to
`//runtime/...` and `//libhrx/...` when no targets are given. Use absolute labels
such as `//runtime/src/iree/base/...`, not `:target`.

```bash
iree-bazel-configure
iree-bazel-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
iree-bazel-configure --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm
iree-bazel-build [targets...]
iree-bazel-test [targets...]
iree-bazel-query 'deps(//runtime/src/iree/base)'
iree-bazel-cquery --output=files //runtime/src/iree/base
iree-bazel-info execution_root
iree-bazel-dev precommit [paths...]
iree-bazel-dev presubmit
```

`iree-bazel-query` reads the unevaluated source graph; use
`iree-bazel-cquery` when select() resolution, platform constraints, or output
files matter. `iree-bazel-run` executes built binaries without holding the Bazel
lock. More specialized tools are available: `iree-bazel-try` for scratch C/C++
builds and `iree-bazel-fuzz` for fuzzing sessions.

Outputs: `bazel-bin/`, `bazel-testlogs/`, and `bazel-out/`."""


def cmake_agent_markdown() -> str:
    return """### CMake

Use `iree-cmake-*` for package and install-test workflows. `iree-cmake-configure`
writes `../builds/<checkout-name>/`. `iree-cmake-build TARGET` maps to
`cmake --build ... --target TARGET`.

```bash
iree-cmake-configure
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF
iree-cmake-build hrx
iree-cmake-test -R hrx
iree-cmake-dev precommit
iree-cmake-dev presubmit
```

`iree-cmake-dev precommit` checks local changes. The paranoid profile adds
affected project CMake/CTest checks. `iree-cmake-dev presubmit` is the
full-tree CI-shaped check."""


def bazel_command_agent_markdown(command: str) -> str:
    if command == "configure":
        return """## iree-bazel-configure

Configure Bazel options for this checkout. Run it once after clone and again
when changing project build options. It writes `.bazelrc.configured`; keep
personal Bazel overrides in `.bazelrc.local`.

```bash
iree-bazel-configure
iree-bazel-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
iree-bazel-configure --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm
```

Published portable configuration options live in `BUILDING.md`."""

    if command == "build":
        return """## iree-bazel-build

Build Bazel targets from the repository root. With no targets, this builds
`//runtime/...` and `//libhrx/...`. Use absolute labels such as
`//runtime/src/iree/base/...`, not package-relative labels.

```bash
iree-bazel-build
iree-bazel-build //runtime/src/iree/base/...
iree-bazel-build --config=asan //runtime/src/iree/base:allocator_test
```

Use `iree-bazel-cquery` for configured graph/output questions and
`iree-bazel-run` when the next step is executing a built binary.
Scratch C/C++ probes and fuzzing have separate focused guidance:
`iree-bazel-try` and `iree-bazel-fuzz`."""

    if command == "test":
        return """## iree-bazel-test

Run Bazel tests with the repository presubmit configuration. With no targets,
this tests `//runtime/...` and `//libhrx/...`.

```bash
iree-bazel-test
iree-bazel-test //runtime/src/iree/base/...
iree-bazel-test --config=asan //runtime/src/iree/base:allocator_test
```

Use focused targets while iterating, then run the affected project or default
target set before handing work back."""

    if command == "query":
        return """## iree-bazel-query

Inspect the unevaluated Bazel source graph. This is useful for package
structure, dependency shape, and broad label discovery. Use
`iree-bazel-cquery` when select() resolution, configured compatibility, or
output files matter.

```bash
iree-bazel-query 'kind(cc_library, //runtime/...)'
iree-bazel-query 'deps(//runtime/src/iree/base)'
iree-bazel-query 'rdeps(//runtime/..., //runtime/src/iree/base)'
```

Query expressions and native Bazel query options are forwarded unchanged."""

    if command == "cquery":
        return """## iree-bazel-cquery

Inspect the configured Bazel graph. Use this for porting and build-system
debugging when select() values, platform constraints, target compatibility, or
generated output paths are relevant.

```bash
iree-bazel-cquery 'kind(cc_library, //runtime/...)'
iree-bazel-cquery --output=files //runtime/src/iree/base:base
iree-bazel-cquery --config=asan 'deps(//runtime/src/iree/base:base)'
```

cquery expressions and native Bazel cquery options are forwarded unchanged."""

    if command == "info":
        return """## iree-bazel-info

Inspect Bazel workspace and output paths for this checkout.

```bash
iree-bazel-info
iree-bazel-info execution_root
iree-bazel-info bazel-bin
```

Use this instead of guessing output roots when debugging generated files,
execution roots, or wrapper-resolved binary paths."""

    if command == "run":
        return """## iree-bazel-run

Use `iree-bazel-run` instead of raw `bazel run` when executing repository
binaries. It builds the target, resolves the configured executable with cquery,
then execs the binary from the current directory. The Bazel server lock is not
held while the program runs, and signals/PID-based tools see the final process.

```bash
iree-bazel-run //runtime/src/iree/base:allocator_benchmark
iree-bazel-run //runtime/src/tools:iree-run-module -- --help
iree-bazel-run --config=asan //runtime/src/iree/base:allocator_test
iree-bazel-run -p //runtime/src/tools:iree-run-module
```

Put Bazel build options before the program-argument separator. Put executable
arguments after `--`. `-p/--print-path` prints the resolved binary without running
it."""

    if command == "try":
        return """## iree-bazel-try

Use `iree-bazel-try` for one-shot C/C++ probes without creating a permanent
BUILD target. It writes a temporary package under `.iree-bazel-try/`, builds it
with Bazel, then execs the snippet unless `--compile-only` is used. The scratch
package is removed by default.

```bash
iree-bazel-try -e 'int main() { return 0; }'
iree-bazel-try -x c++ -e $'#include "iree/base/api.h"\\nint main() { return 0; }'
iree-bazel-try -e $'#include "loom/ir/module.h"\\nint main() { return 0; }'
iree-bazel-try --dep //runtime/src/iree/base snippet.c -- --flag
iree-bazel-try -k -e 'int main() { return 0; }'
iree-bazel-try --compile-only --output build/snippet -e 'int main() { return 0; }'
```

Quoted `iree/...`, `loom/...`, and `loomc/...` includes are used to infer common
deps. Add `--dep LABEL` for anything inference misses. Use `--no-infer` to
disable inference and `-k/--keep` to inspect the generated scratch package."""

    if command == "fuzz":
        return """## iree-bazel-fuzz

Use `iree-bazel-fuzz` for libFuzzer sessions without holding the Bazel server
lock. It builds with `--config=fuzzer`, resolves fuzzer binaries with cquery,
and runs the binaries directly. A single target replaces the wrapper process
with the fuzzer; `...` patterns discover labels ending in `_fuzz`, build them
together, and launch each fuzzer binary directly.

```bash
iree-bazel-fuzz //runtime/src/iree/tokenizer:special_tokens_fuzz
iree-bazel-fuzz //runtime/src/iree/tokenizer:special_tokens_fuzz -- -max_total_time=60
iree-bazel-fuzz //runtime/src/iree/tokenizer/... -- -max_total_time=60 -jobs=8
```

Corpus and artifact directories live under
`${IREE_FUZZ_CACHE:-~/.cache/iree-fuzz-cache}/`. Fuzzer arguments go after
`--`."""

    return bazel_agent_markdown()


def cmake_command_agent_markdown(command: str) -> str:
    if command == "configure":
        return """## iree-cmake-configure

Configure the CMake package/install-test build tree for this checkout.

```bash
iree-cmake-configure
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF
```

The default build tree is `../builds/<checkout-name>/`."""

    if command == "build":
        return """## iree-cmake-build

Build targets in the configured CMake build tree.

```bash
iree-cmake-build hrx
iree-cmake-build --parallel 8
iree-cmake-build hrx --parallel 8
```

Positional arguments before option-looking arguments are CMake target names."""

    if command == "test":
        return """## iree-cmake-test

Run CTest in the configured CMake build tree.

```bash
iree-cmake-test
iree-cmake-test -R hrx
iree-cmake-test --rerun-failed
```

CTest options are forwarded unchanged."""

    return cmake_agent_markdown()
