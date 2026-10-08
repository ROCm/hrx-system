# Contributing

## How Pull Requests Are Integrated

Contributions are welcome at every stage, from a focused bug report, feature
proposal, or tested design idea through a complete implementation. This
repository is developed by a continuously active team, and changes often cross
shared representations, generators, backends, tests, and performance
constraints. Loom in particular usually has dozens of concurrent workstreams.
A pull request therefore supplies evidence and a concrete implementation to
evaluate. Its submitted diff and commit structure may change before the work
lands.

When maintainer edits are enabled, project maintainers may push corrections,
rework commits, extend tests, or rebase the pull request branch. We may instead
split or combine the work, or carry its findings into a broader implementation
on another branch. The goal is one coherent change for the system as a whole.
Code carried forward retains its authorship. A replacement or superseding
change links the original issue or pull request and credits the discovery,
design, or code it carries forward.

A nontrivial pull request must stand on its own for someone outside the
authoring session. Describe the problem and its impact, include a concrete
reproducer or other evidence, explain the design intent and scope, and report
how the change was verified. Include performance and size results when the
change can affect them. Agent-assisted contributions are welcome and meet the
same standard. A raw generated diff or "an agent made this" does not provide
enough information to triage; the agent should produce the explanation and
evidence as part of the contribution.

[PR #1380](https://github.com/ROCm/hrx-system/pull/1380) is an example of this
model. It isolated a silent wrong-code bug, supplied a compact reproducer,
explained the violated invariant, and proposed a repair. Concurrent
[PR #1377](https://github.com/ROCm/hrx-system/pull/1377) corrected the same
invariant across a broader set of compiler paths. The project verified that
broader fix against #1380's reproducer and closed #1380 as superseded, while
retaining the original report as part of the design history. The contribution
improved Loom even though its submitted patch did not merge unchanged.

## Development Tooling

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
only. `--amend` checks the exact amended candidate by comparing the index with
`HEAD^`; that mode is always non-mutating. Use `--profile default`, `--profile
paranoid`, or `--profile ci` to select the check profile for one manual run. The
Bazel lane defaults to `paranoid` for precommit. The CMake lane defaults to
`default`; select `paranoid` or `ci` to add affected project CMake/CTest checks.

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

The hook treats the Git index as the ordinary commit boundary. Test-bearing
profiles first apply fixups to staged existing files and generated outputs
derived from selected inputs. If the index changes, the attempt stops before
tests and prints every changed staged path. Review `git diff --cached` and retry
the same commit; the repaired candidate then receives read-only hygiene, tests,
and static analysis. Lefthook's `fail_on_changes` check remains an independent
guard against committing hook mutations without review.

A candidate file that also has unstaged changes is an incoherent formatter
input: the index contains one version while whole-file tools see another. The
hook stops with the exact paths. Running `python dev.py <lane> fix` before
selecting hunks, then staging the intended hunks again, keeps the commit narrow
without letting a formatter stage the rest of the file.

Git's pre-commit event does not identify an amend operation. Explicit
`python dev.py <lane> precommit --amend` validates the index against `HEAD^`
without granting mutation authority to files inherited from `HEAD`.
Lane-specific hook policy is stored in ignored `lefthook-local.yml`. Re-run
`python dev.py <lane> hook --profile <profile>` to change the profile used by
Git commits.

## Verification

Core developer-tool tests:

```bash
bazel test --config=presubmit //build_tools/bazel:configure_test //build_tools/devtools:cli_test //build_tools/devtools:command_plan_test //build_tools/devtools:setup_test
```

Smoke test for the command router and checked-in wrappers:

```bash
python build_tools/devtools/cli_smoke_test.py
```

The smoke test runs one direct command, one checked-in wrapper per build-system
lane, and one CI command against the live repository. All commands are dry runs,
and the harness verifies that repository and ignored local configuration state
remain unchanged.

More detail on profiles and hook internals lives in
`build_tools/lefthook/README.md`.
