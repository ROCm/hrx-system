# Contributing

This repository uses Bazel as the source of truth for build graph structure and
uses CMake for package/install-test workflows. `dev.py` is the blessed command
router for local development. It selects a structural build lane, prepares the
tool environment, and delegates real work to Bazel, CMake, Lefthook, Git, and
project-local scripts.

## Build Lanes

Bazel/source graph lane:

```bash
python dev.py bazel setup
python dev.py bazel hook --profile paranoid
python dev.py bazel precommit
```

CMake/package lane:

```bash
python dev.py cmake setup
python dev.py cmake hook --profile default
python dev.py cmake precommit
```

The build lane is part of the command structure instead of a global flag. That
keeps hooks and agent runs unambiguous: a Bazel hook runs Bazel-lane checks, and
a CMake hook runs CMake-lane checks.

## Tool Modes

Standalone checkouts normally use the default repo-local tool environment:

```bash
python dev.py bazel setup --venv
```

Embedding or superproject workflows can use system tools without writing into
the submodule:

```bash
python dev.py bazel setup --system
```

Superprojects can also keep a managed tool environment outside this checkout:

```bash
python dev.py bazel setup --tool-root ../.tools/iree-x
```

The same modes work for the CMake lane. `--dry-run` prints the exact command
plan without executing it.

## Manual Build Commands

Use the lane command names when you want to run Bazel or CMake work directly:

```bash
python dev.py bazel configure
python dev.py bazel build
python dev.py bazel test
```

or:

```bash
python dev.py cmake configure
python dev.py cmake build
python dev.py cmake test
```

With no explicit targets, the Bazel build and test commands cover
`//runtime/...` and `//libhrx/...`. In the CMake lane, positional build
arguments are target names, so `python dev.py cmake build hrx` maps to
`cmake --build ... --target hrx`. The first CMake configure uses `build/cmake`
unless `--cmake-build-dir` or `IREE_CMAKE_BUILD_DIR` selects another tree, and
the selected tree is recorded for later CMake wrapper invocations. Source-build
and embedding configuration options are documented in `BUILDING.md`.

## Before Commit

Use `precommit` for the current local change set:

```bash
python dev.py bazel precommit
```

or:

```bash
python dev.py cmake precommit
```

With no input option, `precommit` checks staged, unstaged, and untracked files.
Use `--base <git-ref>` to check the branch diff from the merge base through
`HEAD` plus local changes. Use `--staged` when you explicitly want staged files
only. Use `--profile default`, `--profile paranoid`, or `--profile ci` to select
the check profile for one manual run. The Bazel lane defaults to `paranoid` for
precommit. The CMake lane defaults to `default`; select `paranoid` or `ci` to
add affected project CMake/CTest checks.

Use the fix command when you want mechanical repairs:

```bash
python dev.py bazel fix
```

or:

```bash
python dev.py cmake fix
```

`fix` applies staged formatting/generated-file repairs and stages only files
owned by those fixers. `presubmit` is non-mutating and runs the full-tree
CI-shaped check.

## Git Hook

Install the hook for the lane you want Git commits to check:

```bash
python dev.py bazel hook --profile paranoid
```

or:

```bash
python dev.py cmake hook --profile default
```

The hook validates commit scope: files staged for commit plus files changed by
`HEAD`, so amended commits include the commit being replaced without checking
the full branch. Test-bearing hook profiles apply mechanical fixups before
running the same profile in non-mutating check mode. Lane-specific hook policy
is stored in ignored `lefthook-local.yml`. Re-run
`python dev.py <lane> hook --profile <profile>` to change the profile used by
Git commits.

## Verification

Core developer-tool tests:

```bash
bazel test --config=presubmit //build_tools/bazel:configure_test //build_tools/devtools:cli_test //build_tools/devtools:command_plan_test //build_tools/devtools:setup_test
```

Checkout smoke tests for the command router and checked-in wrappers:

```bash
python build_tools/devtools/cli_smoke_test.py --from-working-tree --scenario dry-run
python build_tools/devtools/bazel_smoke_test.py --from-working-tree --scenario dry-run
python build_tools/devtools/cmake_smoke_test.py --from-working-tree --scenario dry-run
```

The smoke tests create temporary checkouts, run dry-run setup/configure/wrapper
commands, and verify those dry-runs do not create a venv, hook config,
generated Bazel rc, try workspace, or external tool root.

More detail on profiles and hook internals lives in
`build_tools/lefthook/README.md`.
