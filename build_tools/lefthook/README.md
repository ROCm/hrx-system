# Lefthook Presubmit

The root Lefthook configuration owns Git hook dispatch. `dev.py` selects the
local build lane and installs the hook through Lefthook. The Python presubmit
dispatcher owns file selection, profiles, hygiene checks, and project test
routing. Project-specific policy stays in each project under
`*/build_tools/presubmit.py`.

## Commit Contract

A successful Git commit must not commit stale formatter or generated-file
output. The Git `pre-commit` hook uses commit scope: files staged for commit
plus files changed by `HEAD`, so `git commit --amend` validates the commit being
replaced without scanning the full feature branch. Test-bearing hook profiles
run a mechanical fix pass first, stage files owned by those fixers, and then run
the same profile in non-mutating check mode:

```bash
python dev.py <lane> precommit --profile <profile> --commit
```

This fix-then-check path is used for `paranoid` and `ci` when the input is the
generated hook's commit scope or an explicit `--staged` precommit, because those
profiles run affected tests after any mutation. Broader local-change precommit
runs stay check-only. The `default` profile remains check-only. Local fixups are
also available through explicit commands:

```bash
python dev.py bazel fix
python dev.py cmake fix
```

`fix` applies staged hygiene repairs and stages files owned by those fixers.
`precommit` checks the current local change set, with autofix enabled for
commit-scope, staged, and explicit-path test-bearing runs. `presubmit` is
non-mutating and runs the full-tree CI-shaped profile.

Normal presubmit output is terse: major phases, named checks, pass/fail status,
and failure details. Large captured failures print the beginning and end with an
explicit omitted-line count so repeated test-log noise does not bury the useful
diagnostic. Failed runs end with suggested repair commands. Use `--verbose` when
debugging the dispatcher itself or when exact command lines and full streaming
tool output are useful.

## Profiles

`default` runs repository hygiene: buildifier, Ruff, clang-format,
bazel-to-cmake, generated AMDGPU target metadata, watchwords, merge-conflict
markers, and basic text hygiene.

`paranoid` adds affected project tests for the selected build lane and the
static-analysis lane. Semgrep hard rules are the first configured provider.
Rules that inventory existing drift stay at `WARNING` or `INFO` severity until
their baseline is cleaned up and the rule is promoted.

`ci` is the full-tree, non-mutating profile. It checks all tracked files and
runs all configured project presubmit tests.

The default manual `precommit` and hook-install profiles are:

| Lane | Default Profile | Scope |
|------|-----------------|-------|
| Bazel | `paranoid` | Hygiene, affected project Bazel tests, and configured static-analysis providers. |
| CMake | `default` | Repository hygiene. Select `paranoid` or `ci` to add affected project CMake/CTest checks. |

`presubmit` defaults to `ci` for both lanes:

```bash
python dev.py bazel presubmit
python dev.py cmake presubmit
```

The checked-in GitHub `check` group currently runs the Bazel lane. The CMake
`ci` profile is a local opt-in until the CMake build/test lane is green enough
to gate CI.

Any manual run can select a profile explicitly:

```bash
python dev.py bazel precommit --profile default
python dev.py bazel precommit --profile paranoid
python dev.py bazel precommit --profile ci
python dev.py cmake presubmit --profile default
```

## Lefthook Groups

`pre-commit` is the installed Git hook. With `paranoid` or `ci` it runs fixups
first and then validates the same commit-scope set; with `default` it is
check-only.

`dev.py` installs lane-specific local hook policy into ignored
`lefthook-local.yml`:

```bash
python dev.py bazel hook --profile paranoid
python dev.py cmake hook --profile default
```

Re-run `python dev.py <lane> hook --profile <profile>` to change the default
profile used by Git commits. The generated file records the selected lane and
profile so local hook behavior can be audited without reading Python.

`fix` runs the staged hygiene fixer for manual use through Lefthook:

```bash
lefthook run fix
```

`precommit` runs the paranoid profile over staged, unstaged, and untracked
local changes:

```bash
lefthook run precommit
```

`presubmit` runs the local CI profile, including project tests:

```bash
lefthook run presubmit
```

`check` is kept as the same local CI-profile alias:

```bash
lefthook run check
```

The GitHub Presubmit workflow calls:

```bash
python dev.py bazel presubmit --profile ci --no-project-tests
```

That workflow still runs full tracked-tree hygiene, root devtools tests, and
configured static-analysis providers. It skips runtime/libhrx/loom project
tests because dedicated CI workflows own those project build and test lanes.

`lefthook.yml` requires Lefthook 2.1.9 or newer because the repository uses
custom manual hook groups in addition to Git hook names.

## Tool Installation

Python-packaged local tools are pinned in `requirements-dev.lock.txt`. That
lock contains only local development tools; Bazel build and test dependencies
belong in Bazel module fragments and `MODULE.bazel.lock`. Semgrep is
intentionally outside this lock because it is an optional static-analysis tool,
not part of the Python toolchain used to build or test project code.

Standalone binaries are installed by `build_tools/devtools/install.py` into the
selected tool environment. The Bazel lane installs Bazelisk and buildifier with
pinned URLs and SHA-256 hashes:

```bash
python dev.py bazel setup
```

The CMake lane currently has no standalone tool downloads; it uses system CMake
and CTest plus the shared Python-packaged tools.

Optional static-analysis providers should skip outside CI when their tool is
unavailable and print the missing executable. CI installs those tools explicitly
before running the presubmit profile that requires them. `dev.py doctor`
reports optional tool availability; today Semgrep and clang-tidy are warnings
for local setup rather than part of the locked developer requirements.

## Static Analysis

The root static-analysis lane dispatches providers from
`build_tools/lefthook/presubmit.py`. Provider-specific rule configuration stays
native to each provider. Semgrep is an external developer tool, not a dependency
locked in `requirements-dev.lock.txt`; install the pinned version locally with
your preferred Python environment or package manager:

```bash
python3 -m pip install --user semgrep==1.164.0
```

Semgrep rules live under `build_tools/static_analysis/semgrep/` and can be run
directly:

```bash
semgrep scan --metrics=off --disable-version-check \
  --config build_tools/static_analysis/semgrep/iree.yml runtime/src/iree
```

Presubmit runs Semgrep with `--severity ERROR --error`; only promoted hard
rules gate commits and CI. Lower-severity rules remain available for cleanup
campaigns and rule prototyping without flooding the new CI lane. Semgrep
parallelism is controlled by `IREE_SEMGREP_JOBS`; when unset, the dispatcher
uses roughly 85% of detected logical CPUs capped at 14 jobs. That cap avoids
Semgrep's current high-core-count OCaml-domain failure mode while keeping the
local/CI default comfortably fast for this repository size.

## Project Dispatch

The root dispatcher writes the selected file list once and invokes each affected
project script with `--files-from`. Project scripts decide whether those files
affect their project. Shared build-system paths fan out to every configured
project entry point.

CMake project tests use normal CMake commands: build the configured tree, then
run `ctest` with the project's native selector. Presubmit scripts do not
translate Bazel labels to CMake target names or create synthetic CMake test
groups.

Any path under a `build_tools/` directory is a global project trigger. That rule
is intentionally structural: root `build_tools/`, `runtime/build_tools/`,
`libhrx/build_tools/`, and future project tool directories all fan out to every
configured project entry point instead of maintaining dependency-specific folder
lists.

This keeps root policy focused on repository-wide hygiene while preserving
project ownership of tests, expensive checks, and future project-specific static
analysis.
