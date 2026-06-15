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


ROOT_COMMAND_EPILOG = """Common build-system commands:
  python dev.py bazel configure
  python dev.py bazel build
  python dev.py bazel test
  python dev.py bazel precommit
  python dev.py bazel presubmit
  python dev.py bazel run
  python dev.py bazel try
  python dev.py bazel fuzz
  python dev.py bazel compile-commands
  python dev.py bazel clang-tidy

  python dev.py cmake configure
  python dev.py cmake build
  python dev.py cmake test
  python dev.py cmake precommit
  python dev.py cmake presubmit
  python dev.py cmake run
  python dev.py cmake try
  python dev.py cmake fuzz
  python dev.py cmake compile-commands
  python dev.py cmake clang-tidy

Most developer scratch/state defaults under .tmp/. Set IREE_DEVTOOLS_TMP to
redirect scratch that does not need to be a Bazel workspace package.

Use `python dev.py bazel --help` or `python dev.py cmake --help` for the full
subcommand list for that build system."""


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
  python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON
  python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=pinned
  python dev.py bazel configure -DLOOM_TARGET_AMDGPU=ON
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
  python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DLIBHRX_BUILD=OFF
  python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=package
  python dev.py --cmake-build-dir build/cmake-asan cmake configure -DIREE_ENABLE_ASAN=ON

The first configure uses build/cmake unless --cmake-build-dir or
IREE_CMAKE_BUILD_DIR selects another tree. The selected tree is recorded for
later iree-cmake-build, iree-cmake-test, iree-cmake-run, iree-cmake-try, and
iree-cmake-fuzz invocations.
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
  python dev.py cmake build hrx::hrx
  python dev.py cmake build hrx --parallel 8
  python dev.py cmake build --parallel 8

Positional arguments are target names and become cmake --build ... --target
<name>. Configured CMake aliases such as iree::base and hrx::hrx are translated
to their concrete generator targets. Option-looking arguments are forwarded to
CMake.""",
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

CTest runs in the selected CMake build tree with --output-on-failure.""",
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
    if command == "compile-commands":
        if lane == "bazel":
            return CommandHelp(
                description="Generate a Bazel-backed compile_commands.json.",
                arguments="Bazel target patterns and build options.",
                epilog="""Examples:
  python dev.py bazel compile-commands
  python dev.py bazel compile-commands //runtime/src/iree/base/...
  python dev.py bazel compile-commands --config=asan //runtime/...
  python dev.py bazel compile-commands -o .cache/compile_commands/runtime.json //runtime/...

With no explicit targets, this generates compile commands for //runtime/...,
//libhrx/..., and //loom/.... The wrapper builds the compile-command aspect
output group, reads the Bazel build event stream to find generated fragments,
and merges them into one local database. The default output is
compile_commands.json at the repository root.""",
            )
        return CommandHelp(
            description="Print or copy the configured CMake compile_commands.json path.",
            arguments="Optional -o/--output path.",
            epilog="""Examples:
  python dev.py cmake compile-commands
  python dev.py cmake compile-commands -o .cache/compile_commands/cmake.json

CMake writes compile_commands.json during configure. This command resolves the
configured build directory, prints that file path by default, and copies it to
the requested output path when -o/--output is supplied.""",
        )
    if command == "clang-tidy" and lane == "bazel":
        return CommandHelp(
            description="Run Bazel-backed clang-tidy checks.",
            epilog="""Examples:
  python dev.py bazel clang-tidy
  python dev.py bazel clang-tidy --base origin/main
  python dev.py bazel clang-tidy --all --profile ci
  python dev.py bazel clang-tidy --fix runtime/src/iree/vm/native_module.c
  python dev.py bazel clang-tidy runtime/src/iree/vm/native_module.c
  python dev.py bazel clang-tidy //runtime/src/iree/vm:all

With no input option, this checks local staged, unstaged, and untracked files.
Repo-relative paths and git scopes use the presubmit clang-tidy provider, which
maps files to owning Bazel packages. Bazel target patterns run the clang-tidy
aspect directly. The ci profile adds Bazel --keep_going so one run reports the
full failure set. --fix is restricted to bounded path and git scopes; it exports
fixes through Bazel, applies them outside Bazel, then re-runs the check.""",
        )
    if command == "clang-tidy" and lane == "cmake":
        return CommandHelp(
            description="Run CMake-backed clang-tidy checks.",
            epilog="""Examples:
  python dev.py cmake clang-tidy
  python dev.py cmake clang-tidy --base origin/main
  python dev.py cmake clang-tidy --all --profile ci
  python dev.py cmake clang-tidy --fix runtime/src/iree/vm/native_module.c
  python dev.py cmake clang-tidy runtime/src/iree/vm/native_module.c
  python dev.py --cmake-build-dir build/cmake-asan cmake clang-tidy runtime/src/iree/base/status.c

With no input option, this checks local staged, unstaged, and untracked files.
Repo-relative paths and git scopes use the configured CMake compile database.
The scan materializes generated compile inputs, then runs through
run-clang-tidy with parallelism controlled by IREE_CLANG_TIDY_JOBS. --fix
applies fixes through run-clang-tidy, then re-runs the check. Run
`python dev.py cmake configure` first. Select a build tree with --cmake-build-dir
or IREE_CMAKE_BUILD_DIR.""",
        )
    if command == "presubmit":
        return CommandHelp(
            description=f"Run CI-shaped acceptance checks for {build_system_name}.",
            epilog=f"""Examples:
  python dev.py {lane} presubmit
  python dev.py {lane} presubmit --profile paranoid
  python dev.py {lane} presubmit --base origin/main
  python dev.py {lane} presubmit --no-project-tests
  python dev.py {lane} presubmit --verbose

Profiles:
  default   Repository hygiene.
  paranoid  Hygiene plus affected project tests and static-analysis providers.
  ci        Full tracked-tree hygiene plus all configured project tests.

The default profile is ci. This is intentionally the expensive full-tree check
that approximates the project test workflows a developer can run locally. Use
--no-project-tests only for CI jobs that have separate project test workflows.
`--base` runs the same presubmit profile on the branch diff and lets static
analysis providers conservatively escalate when a change affects wider state.
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

Temporary packages are written under .iree/bazel-try/ so Bazel can address
them as // workspace labels while .tmp/ remains hidden from Bazel package
scans. The tool infers common deps from quoted iree/...,
loom/..., and loomc/... includes and accepts explicit --dep labels.""",
        )
    if command == "try" and lane == "cmake":
        return CommandHelp(
            description="Build and run a temporary C/C++ snippet with CMake.",
            arguments="Source files, -e inline source, --dep CMake targets, and optional -- program arguments.",
            epilog="""Examples:
  python dev.py cmake try -e 'int main() { return 0; }'
  python dev.py cmake try -e $'#include "iree/base/api.h"\\nint main() { return 0; }'
  python dev.py cmake try --dep iree::base snippet.c -- --flag
  python dev.py cmake try -k -e 'int main() { return 0; }'
  python dev.py cmake try --compile-only --output build/snippet -e 'int main() { return 0; }'

This derives a temporary build tree from the configured CMake build tree and
injects one generated CMake file into the real repository configure. The
temporary root defaults under .tmp/ and follows IREE_DEVTOOLS_TMP when set. Run
iree-cmake-configure first. Quoted iree/... and hrx_... includes infer common
deps; add --dep TARGET for anything inference misses.""",
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
    if command == "fuzz" and lane == "cmake":
        return CommandHelp(
            description="Build and run a CMake libFuzzer target.",
            arguments="CMake fuzz target and build options, followed by -- and fuzzer arguments.",
            epilog="""Examples:
  python dev.py cmake configure -DIREE_ENABLE_FUZZING=ON -DLIBHRX_BUILD=OFF -DIREE_HAL_DRIVER_AMDGPU=OFF
  python dev.py cmake fuzz iree::tokenizer::special_tokens_fuzz
  python dev.py cmake fuzz iree::tokenizer::special_tokens_fuzz -- -max_total_time=60

This builds the named CMake fuzz target, resolves the executable with the CMake
File API, then execs it directly. Fuzz targets are configure-time CMake targets;
run iree-cmake-configure -DIREE_ENABLE_FUZZING=ON first.""",
        )
    if command == "run" and lane == "cmake":
        return CommandHelp(
            description="Run an already-built CMake executable target.",
            arguments="CMake executable target, followed by -- and program arguments.",
            epilog="""Examples:
  python dev.py cmake run iree-run-module -- --help
  python dev.py cmake run iree::tools::iree-run-module -- --help
  python dev.py cmake run -p iree-run-module

This resolves the executable with the CMake File API and then execs it from the
current directory. It does not build; run `iree-cmake-build <target>` first.""",
        )
    if command in ("try", "fuzz"):
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
        if command == "compile-commands":
            return bazel_command_agent_markdown(command)
        return with_common_wrapper_flags(bazel_command_agent_markdown(command))
    if lane == "cmake":
        if command == "compile-commands":
            return cmake_command_agent_markdown(command)
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
`iree-bazel-build` and `iree-cmake-build`; checked-in
`build_tools/bin/iree-*-*` launchers are available for root-relative scripts and
unconfigured shells. Common wrapper flags accept hyphen or underscore spellings,
such as `--dry-run` and `--dry_run`, and work after the wrapper name:
`-n/--dry-run`, `-v/--verbose`, `--system`, `--venv`, `--tool-root DIR`, and
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
iree-bazel-configure -DIREE_HAL_DRIVER_AMDGPU=ON
iree-bazel-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=pinned
iree-bazel-configure --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm --repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned
iree-bazel-build [targets...]
iree-bazel-test [targets...]
iree-bazel-query 'deps(//runtime/src/iree/base)'
iree-bazel-cquery --output=files //runtime/src/iree/base
python dev.py bazel compile-commands
iree-bazel-info execution_root
iree-bazel-dev precommit [paths...]
iree-bazel-dev presubmit
```

`iree-bazel-query` reads the unevaluated source graph; use
`iree-bazel-cquery` when select() resolution, platform constraints, or output
files matter. `iree-bazel-run` executes built binaries without holding the Bazel
lock. More specialized tools are available: `iree-bazel-try` for scratch C/C++
builds and `iree-bazel-fuzz` for fuzzing sessions. Use
`python dev.py bazel compile-commands` for clang tooling.

Outputs: `bazel-bin/`, `bazel-testlogs/`, and `bazel-out/`."""


def cmake_agent_markdown() -> str:
    return """### CMake

Use `iree-cmake-*` for package and install-test workflows. `iree-cmake-configure`
records the selected build tree for later wrappers. `iree-cmake-build TARGET`
maps to `cmake --build ... --target TARGET`.

```bash
iree-cmake-configure
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=package
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF
iree-cmake-build hrx::hrx
iree-cmake-test -R hrx
iree-cmake-run iree::tools::iree-run-module -- --help
iree-cmake-fuzz iree::tokenizer::special_tokens_fuzz -- -max_total_time=60
python dev.py cmake compile-commands
python dev.py cmake clang-tidy
iree-cmake-dev precommit
iree-cmake-dev presubmit
```

`iree-cmake-dev precommit` checks local changes. The paranoid profile adds
affected project CMake/CTest checks. `iree-cmake-dev presubmit` is the
full-tree CI-shaped check. `iree-cmake-run` resolves an already-built
executable and does not build implicitly. `iree-cmake-try` builds temporary
C/C++ snippets against the configured tree. `iree-cmake-fuzz` builds and execs
a configured libFuzzer target. `python dev.py cmake compile-commands` prints
the configured compile database path for clang tooling. `python dev.py cmake
clang-tidy` runs IREE clang-tidy checks against that compile database."""


def bazel_command_agent_markdown(command: str) -> str:
    if command == "configure":
        return """## iree-bazel-configure

Configure Bazel options for this checkout. Run it once after clone and again
when changing project build options. It writes `.bazelrc.configured`; keep
personal Bazel overrides in `.bazelrc.local`.

```bash
iree-bazel-configure
iree-bazel-configure -DIREE_HAL_DRIVER_AMDGPU=ON
iree-bazel-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=pinned
iree-bazel-configure --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm --repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned
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

    if command == "compile-commands":
        return """## python dev.py bazel compile-commands

Generate a `compile_commands.json` from the configured Bazel graph. This is the
entry point for clangd, clang-tidy, and other compile-database consumers that
need Bazel's actual configured C/C++ flags.

```bash
python dev.py bazel compile-commands
python dev.py bazel compile-commands //runtime/src/iree/base/...
python dev.py bazel compile-commands --config=asan //runtime/...
python dev.py bazel compile-commands -o .cache/compile_commands/runtime.json //runtime/...
```

With no explicit targets, this covers `//runtime/...`, `//libhrx/...`, and
`//loom/...`. The wrapper builds the compile-command aspect output group, reads
the Bazel build event stream to find generated fragments, and merges them into
the requested output path. Use `-k/--keep` to inspect the captured build event
stream."""

    if command == "clang-tidy":
        return """## python dev.py bazel clang-tidy

Run Bazel-backed clang-tidy checks from the repository root. Use git scopes or
repo-relative paths while iterating, and explicit Bazel targets when you want to
exercise one package or subtree directly.

```bash
python dev.py bazel clang-tidy
python dev.py bazel clang-tidy --base origin/main
python dev.py bazel clang-tidy --all --profile ci
python dev.py bazel clang-tidy --fix runtime/src/iree/vm/native_module.c
python dev.py bazel clang-tidy runtime/src/iree/vm/native_module.c
python dev.py bazel clang-tidy //runtime/src/iree/vm:all
```

Git scopes and repo-relative paths route through the presubmit provider so the
tool can map files to owning Bazel packages. Explicit Bazel targets run the
clang-tidy aspect directly. The ci profile uses Bazel `--keep_going`.
`--fix` is restricted to bounded path and git scopes; it exports replacements
through Bazel, applies them outside Bazel, then re-runs the normal check."""

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
BUILD target. It writes a temporary package under `.iree/bazel-try/` so the
snippet has a valid Bazel workspace label while `.tmp/` remains hidden from
Bazel package scans. The tool builds the snippet with Bazel, then execs the
snippet unless `--compile-only` is used. The scratch package is removed by
default.

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
`IREE_FUZZ_CACHE` when set, otherwise the platform user cache directory under
`iree-fuzz-cache/`. Fuzzer arguments go after `--`."""

    return bazel_agent_markdown()


def cmake_command_agent_markdown(command: str) -> str:
    if command == "compile-commands":
        return """## python dev.py cmake compile-commands

Print or copy the configured CMake `compile_commands.json`.

```bash
python dev.py cmake configure
python dev.py cmake compile-commands
python dev.py cmake compile-commands -o .cache/compile_commands/cmake.json
```

CMake writes the compile database during configure. This command resolves the
recorded build directory, prints the compile database path by default, and
copies it when `-o/--output` is supplied."""

    if command == "clang-tidy":
        return """## python dev.py cmake clang-tidy

Run CMake-backed clang-tidy checks from the repository root.

```bash
python dev.py cmake configure
python dev.py cmake clang-tidy
python dev.py cmake clang-tidy --base origin/main
python dev.py cmake clang-tidy --fix runtime/src/iree/base/status.c
python dev.py cmake clang-tidy runtime/src/iree/base/status.c
python dev.py --cmake-build-dir build/cmake-asan cmake clang-tidy runtime/src/iree/base/status.c
```

The command builds the IREE clang-tidy plugin with CMake, materializes generated
compile inputs, and runs `run-clang-tidy` against source files using the
configured CMake `compile_commands.json`. The scan uses parallelism controlled
by `IREE_CLANG_TIDY_JOBS`. `--fix` applies fixes through `run-clang-tidy`, then
re-runs the normal check. Select the CMake build tree with `--cmake-build-dir`
or `IREE_CMAKE_BUILD_DIR`."""

    if command == "configure":
        return """## iree-cmake-configure

Configure the CMake package/install-test build tree for this checkout.

```bash
iree-cmake-configure
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=package
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF
```

The first configure uses `build/cmake` unless `--cmake-build-dir` or
`IREE_CMAKE_BUILD_DIR` selects another tree. The selected tree is recorded for
later CMake wrappers."""

    if command == "build":
        return """## iree-cmake-build

Build targets in the configured CMake build tree.

```bash
iree-cmake-build hrx
iree-cmake-build hrx::hrx
iree-cmake-build --parallel 8
iree-cmake-build hrx --parallel 8
```

Positional arguments before option-looking arguments are CMake target names.
Configured aliases such as `iree::base` and `hrx::hrx` are translated to their
concrete generator targets."""

    if command == "test":
        return """## iree-cmake-test

Run CTest in the configured CMake build tree.

```bash
iree-cmake-test
iree-cmake-test -R hrx
iree-cmake-test --rerun-failed
```

CTest options are forwarded unchanged."""

    if command == "run":
        return """## iree-cmake-run

Run an already-built executable target from the configured CMake build tree. It
resolves the target with the CMake File API and execs the binary from the
current directory. It does not build; run `iree-cmake-build <target>` first.

```bash
iree-cmake-build iree-run-module
iree-cmake-run iree-run-module -- --help
iree-cmake-run iree::tools::iree-run-module -- --help
iree-cmake-run -p iree-run-module
```

Program arguments go after `--`. `-p/--print-path` prints the resolved binary
without running it."""

    if command == "try":
        return """## iree-cmake-try

Use `iree-cmake-try` for one-shot C/C++ probes against the configured CMake
tree. It creates a temporary build under `.tmp/iree-cmake-try/` by default,
copies the existing cache configuration, injects one generated CMake file into
the real repository configure, builds the snippet target, then runs it unless
`--compile-only` is used. Set `IREE_DEVTOOLS_TMP` to redirect the scratch root.

```bash
iree-cmake-configure
iree-cmake-try -e 'int main() { return 0; }'
iree-cmake-try -e $'#include "iree/base/api.h"\\nint main() { return 0; }'
iree-cmake-try --dep iree::base snippet.c -- --flag
iree-cmake-try -k -e 'int main() { return 0; }'
iree-cmake-try --compile-only --output build/snippet -e 'int main() { return 0; }'
```

Quoted `iree/...` and `hrx_...` includes infer common deps from configured CMake
aliases. Add `--dep TARGET` for anything inference misses. Use `--no-infer` to
disable inference and `-k/--keep` to inspect the generated scratch build."""

    if command == "fuzz":
        return """## iree-cmake-fuzz

Use `iree-cmake-fuzz` for CMake libFuzzer targets. Configure the build tree
with fuzzing enabled, then name the CMake target or configured alias. The
wrapper builds that target, resolves the executable with the CMake File API,
then execs it directly.

```bash
iree-cmake-configure -DIREE_ENABLE_FUZZING=ON -DLIBHRX_BUILD=OFF -DIREE_HAL_DRIVER_AMDGPU=OFF
iree-cmake-fuzz iree::tokenizer::special_tokens_fuzz
iree-cmake-fuzz iree::tokenizer::special_tokens_fuzz -- -max_total_time=60
```

Corpus and artifact directories live under
`IREE_FUZZ_CACHE` when set, otherwise the platform user cache directory under
`iree-fuzz-cache/cmake/`. Fuzzer arguments go after `--`."""

    return cmake_agent_markdown()
