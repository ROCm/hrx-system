# Lefthook Presubmit

The root Lefthook configuration owns Git hook dispatch. `dev.py` selects the
local build lane and installs the hook through Lefthook. The Python presubmit
dispatcher owns file selection, profiles, hygiene checks, and project test
routing. Project-specific policy stays in each project under
`*/build_tools/presubmit.py`.

## Test Selection

Bazel presubmit selects affected tests across every project and the repository
build tools. Each project supplies its test roots and build configuration to
the shared selector in `build_tools/devtools/bazel_test_selection.py`. A changed
runtime source can therefore select Loom consumers, and a shared generator can
select tests in several projects without a separate path-to-test table.

The selector discovers eligible test rules with `query`, then follows their
configured dependencies with `cquery` using the same options as test execution.
This preserves `select()` choices, tool transitions, and aspect edges. Ordinary
manual tests remain excluded; explicitly registered presubmit suites include
their manual policy tests. The resulting labels go to a target-pattern file,
with compatibility and execution-resource policy applied by Bazel. Query
failures fail presubmit; an empty selection runs no tests.

Changes to package BUILD files seed every rule in that package, then use the
same configured reverse-dependency query to select affected tests. Removed
package boundaries, Starlark rules, module definitions, and root Bazel
configuration conservatively select the full suites because their ownership is
not bounded to one package. CMake validation and source hygiene retain their
project-owned selection and entry points.

## Commit Contract

A successful Git commit must not commit stale formatter or generated-file
output or private work-tracking breadcrumbs. The generated Git commit hook uses
the Git index as both the ordinary validation boundary and the authority for
mechanical mutation. Files present only in `HEAD`, the unstaged worktree, or an
unrelated dirty change set never become fixer inputs merely because a commit is
running. Test-bearing hook profiles start with a bounded mechanical fix pass:

```bash
python dev.py <lane> precommit --profile <profile> --staged
```

This path is used for `paranoid` and `ci` with staged or explicit inputs. When
the pass changes the index, it prints the exact paths and returns nonzero before
tests. Lefthook's `fail_on_changes: always` independently observes the mutation.
The next commit attempt finds the repaired index unchanged and continues through
read-only hygiene, affected tests, and static analysis. This two-pass contract
makes the review boundary visible instead of silently adding formatter output
to a commit. Broader local-change precommit runs and the `default` profile stay
check-only. Local fixups are also available through explicit commands:

```bash
python dev.py bazel fix
python dev.py cmake fix
```

`fix` applies staged hygiene repairs and stages files owned by those fixers.
Loom artifact maintenance updates only families whose owned outputs
were selected; the read-only retry detects source changes that require an
explicit regeneration. `precommit` checks the current local change set, with
autofix enabled for staged and explicit-path test-bearing runs. `presubmit` is
non-mutating and runs the full-tree CI-shaped profile.

Whole-file tools cannot faithfully consume a candidate file that also contains
unstaged hunks. That state fails before mutation and names each conflicting
path. Running `python dev.py <lane> fix` before selecting hunks, then staging the
intended hunks again, retains the narrow commit boundary.

Git's pre-commit event provides no amend discriminator. Amend validation is an
explicit read-only operation:

```bash
python dev.py <lane> precommit --profile <profile> --amend
```

It compares the current index with `HEAD^`, which is the exact tree that an
amend would write relative to the replaced commit's parent. The compatibility
spelling `--commit` now has the same staged-only semantics as `--staged`, keeping
previously installed hook configurations safe.

The `commit-msg` hook validates the final Git commit message text. The subject
line must start with a bracketed subsystem tag such as `[Loom]`, `[HRX]`,
`[HAL]`, `[Runtime]`, `[Infra]`, or `[CI]`, followed by the short description.
Slash-qualified tags such as `[HAL/AMDGPU]`, `[Loom/WASM]`, or `[Runtime/VM]`
are accepted when the narrower ownership surface is useful.

Git autosquash subjects such as `fixup! [Loom] ...`, `squash! [Loom] ...`,
and `amend! [Loom] ...` are accepted. The hook strips the autosquash wrapper
and validates the target subject with the same tag and subject-length policy.

Subjects and prose body lines must stay at or below 72 characters. This is a
formatting policy for readable Git log output, not a request to shorten or
fragment useful commit-message prose. Long fenced code blocks, indented
examples, Markdown tables, URLs, and final Git trailers are preserved as
structured text.

Before validation, the hook canonicalizes ordinary prose paragraphs to 72
columns in the proposed commit message and proceeds without a retry. Inline
code spans remain indivisible. Subject lines, trailers, fenced blocks, indented
examples, tables, URLs, block quotes, and list items are left untouched;
overlong structured lines still require an explicit edit. The same formatter is
available directly:

```bash
python build_tools/lefthook/commit_msg.py --format .git/COMMIT_EDITMSG
```

When a tag is missing, the hook ranks suggested tags from the staged paths and
prints the paths used for the suggestion so the next edit is obvious.

The hook also rejects literal `\n`/`\r` escape sequences that should have been
real paragraph breaks and bead-shaped issue identifiers such as `loom-20r1y`,
`loom-qof73.1`, or `bd-123`. Legitimate tool names such as `loom-check`,
`loom-compile`, and `loom-opt` are not issue identifiers and are accepted.

Normal presubmit output is terse: major phases, named checks, pass/fail status,
and failure details. Large captured failures print the beginning and end with an
explicit omitted-line count so repeated test-log noise does not bury the useful
diagnostic. Failed runs end with suggested repair commands. Use `--verbose` when
debugging the dispatcher itself or when exact command lines and full streaming
tool output are useful.

## Profiles

`default` runs repository hygiene: buildifier, Ruff, clang-format, the CMake
dependency lock, generated AMDGPU target metadata, repository policy,
watchwords, merge-conflict markers, and basic text hygiene.

Every hygiene run checks `MODULE.cmake.lock` against the Bazel dependency graph
with `build_tools/bazel_to_cmake/deps.py --check`, including narrow commits and
fix mode. This check is offline and read-only. Dependency updates regenerate
the lock explicitly with `python build_tools/bazel_to_cmake/deps.py`; resolving
new registry versions may require network access and reviewing a global output.

Buildifier checks formatting and all lint categories in the pinned release,
including unused loads and variables, deprecated Starlark APIs, and declaration
documentation. The fixer applies available repairs and then runs the same lint
check; findings without automatic repairs still fail. Unused Starlark loads
are detectable syntactically, while unused C/C++ `deps` require separate
compilation and link analysis.

clang-format runs C/C++ files in parallel batches. Set `IREE_CLANG_FORMAT_JOBS`
to override the default worker cap when diagnosing local machine behavior.

`paranoid` adds affected project tests for the selected build lane and the
external static-analysis providers.

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

The installed Git commit hook runs staged fixups first with `paranoid` or `ci`.
An unchanged fix pass proceeds directly to read-only validation; a changed pass
stops for review and validation runs on the retry. The `default` profile is
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

That workflow still runs full tracked-tree hygiene, affected repository-tool
tests, and configured static-analysis providers. Developer workflow,
Lefthook, and project-presubmit policy tests are tagged `manual` so product and
platform wildcard builds do not rediscover them. The dispatcher selects their
affected members through the same configured dependency graph as ordinary
tests. Runtime/libamdf/libhrx/loom product tests remain skipped because
dedicated CI workflows own those build and test lanes.

`lefthook.yml` requires Lefthook 2.1.9 or newer because the repository uses
custom manual hook groups in addition to Git hook names.

## Tool Installation

Python-packaged local tools are split by role. Core developer tools are pinned
in `requirements-dev.lock.txt`. Optional Zizmor automation analysis is pinned
in `requirements-analysis.lock.txt` so its binary wheel stays out of the
smaller build/test tool environment. Bazel build and test dependencies belong
in Bazel module fragments and `MODULE.bazel.lock`.

Managed setup installs both Python locks. The analysis lock is installed with
`--only-binary=:all:` so an incompatible Python package wheel fails setup
clearly instead of falling back to a source distribution that may produce a
broken analyzer executable.

Standalone binaries are installed by `build_tools/devtools/install.py` into the
selected tool environment. The Bazel lane installs Bazelisk, buildifier, and
buildozer with pinned URLs and SHA-256 hashes for Linux, macOS, and Windows on
x86-64 and ARM64. Buildifier and buildozer share a buildtools release pin:

```bash
python dev.py bazel setup
```

The CMake lane currently has no standalone tool downloads; it uses system CMake
and CTest plus the shared Python-packaged tools.

Outside CI, optional static-analysis providers skip when their tool is
unavailable and print the missing executable. CI requires the providers selected
by the CI profile. `dev.py doctor` reports optional tool availability. Clang-tidy
uses the Bazel LLVM repository model under `build_tools/clang_tidy`; local
paranoid precommit skips when the LLVM tools are not available. The GitHub
presubmit workflow fetches the ROCm LLVM toolchain and sets
`IREE_CLANG_TIDY_REQUIRED=1` so missing LLVM tools fail loudly instead of
silently skipping.

## Bazel Editing

[Buildozer](https://github.com/bazel-contrib/buildtools/blob/main/buildozer/README.md)
applies explicit edits to authored BUILD files, including repository macros.
It can batch dependency changes, move attributes, update loads, and create or
rename targets. It does not infer C/C++ dependency ownership or evaluate macros
and configurations. Dependency-removal decisions still require include, compile,
and link evidence for the supported configurations.

Its JSON print output is useful for inspecting a target before editing:

```bash
buildozer -output_json 'print name kind deps' \
  //libamdf/src/platform:native_wait | jq .
```

The returned `select()` is expression text, not a resolved dependency list.
Value-based `remove` and `replace` commands affect every matching `select()` arm.
`set_select` replaces the complete attribute; it does not patch one branch.
`-edit-variables` can change a shared variable and therefore affect other rules.
Review the full diff and configuration evidence before applying those edits.

For an established removal from a target such as `//example:library`, preview
the rewritten file with `-stdout`, then apply the same command without it:

```bash
buildozer -stdout 'remove deps //example:unused' //example:library
buildozer 'remove deps //example:unused' //example:library
```

`buildozer -f <commands-file>` batches newline-separated
`command|label|label` records. Exit status `3` means a successful in-place no-op;
`0` means a change or a read-only success. A `-stdout` preview also returns `0`
for a no-op, so its exit status alone does not indicate a proposed change.
Buildozer is an explicit editing tool and is not run automatically by the hooks.

Buildifier also supports structured lint output for selected BUILD/Starlark files:

```bash
buildifier -mode=check -lint=warn -warnings=all -format=json \
  runtime/src/iree/base/BUILD.bazel > /tmp/buildifier.json
jq -c '.files[] | .filename as $file | .warnings[]? |
  {file: $file, category, start, message}' /tmp/buildifier.json
jq -e '.success' /tmp/buildifier.json
```

JSON mode returns exit status `0` even when `.success` is false. Consumers use
that field to decide whether lint passed. Hooks use text output and its failure
exit status. The dispatcher explicitly selects `-warnings=all`; the JSON config
format's `warningsList` accepts category names, not the CLI shorthand `all`.

### Bazel Package Boundaries

Changed-file hygiene maps every selected path to its nearest `BUILD.bazel`,
queries those packages as expanded Bazel XML, and rejects source labels that
contain a descendant directory. `--all` supplies every tracked path and checks
the complete package graph. A directory contributing build inputs is a package
ownership boundary: give it a `BUILD.bazel` and reference its targets instead
of reaching through the parent package.

Some source trees are intentionally one ownership unit even though their
installed or embedded layout contains directories. Those packages declare the
exception locally, next to the rules that consume it:

```starlark
load("//build_tools/bazel:package_boundary.bzl", "iree_package_boundary")

iree_package_boundary(
    allowed_descendant_paths = ["include/example/**"],
    reason = "Installed public headers preserve their SDK-relative layout.",
)
```

An allowed path is an exact descendant file, a directory prefix ending in
`/**`, or `**` when an entire package is genuinely an indivisible source
bundle. The check rejects empty reasons, malformed or overlapping patterns,
and patterns that no longer match a source. Narrow prefixes keep a legitimate
installed-header tree from exempting unrelated implementation or test trees in
the same package.

## Static Analysis

Repository-specific lexical contracts live in
`build_tools/lefthook/repository_policy.py`. The data-driven checker pairs each
forbidden text pattern with the exact repository-relative paths it owns and
reports stable rule identifiers with line and column locations. It runs during
every hygiene pass without an external executable. Editing the checker or its
tests scans all tracked candidate paths; ordinary changes scan only the selected
files.

The policies keep C runtime sources free of goto-based cleanup, prevent
suppression of the designated-initializer check, keep target-generic diagnostics
independent of architecture vocabulary, and enforce documentation ownership
between the AMD hardware reference, libamdf, and CTS. Focused unit cases exercise
every path boundary and accepted/rejected spelling. Semantic C and C++ contracts
remain in the clang-tidy plugin, where parsed source and macro-expansion context
are available.

The root static-analysis lane dispatches external providers from
`build_tools/lefthook/presubmit.py`. Zizmor checks GitHub automation, and
clang-tidy selects C/C++ files under `runtime/src/iree/`, `loom/src/loom/`,
`loom/binding/c/`, `libamdf/`, and `libhrx/`.

In the Bazel lane, clang-tidy maps selected files to their nearest package and
invokes the checked-in clang-tidy aspect. File-based and explicit-target analysis
enable `//libamdf/config:enabled` when the selected scope includes libamdf,
including repository-wide target patterns. Excluding the complete libamdf tree
removes that override unless a later pattern selects it again. Other scopes
preserve the configured value, avoiding analysis-cache invalidation on
runtime-only checks. Transitive dependencies use the configured project
enablement:

```bash
python dev.py bazel precommit --profile paranoid runtime/src/iree/base/status.c
```

Bazel clang-tidy inherits the normal Bazel concurrency policy, including remote
execution capacity configured by the developer. `IREE_CLANG_TIDY_JOBS` selects
an explicit positive job count for clang-tidy in either lane. Without that
override, the CMake lane uses half the local logical CPUs, rounded up and
capped at 96; the Bazel lane does not impose a client-CPU-derived limit.

Header and shared build-infrastructure changes can expand the analysis to all
tracked C/C++ files. Changes under `build_tools/clang_tidy/` also run the plugin
smoke test and action smoke target. The CMake lane analyzes selected sources
present in the configured compilation database and reports the rest as skipped.
For example, `AMDF_BUILD=OFF` skips libamdf while enabled projects still run.
An entirely excluded source selection succeeds as a skip before plugin setup;
selected plugin infrastructure checks still run.
Native Windows hooks run repository policy directly and delegate the external
providers to Linux presubmit CI.
See `build_tools/clang_tidy/README.md` for the direct Bazel commands and LLVM
discovery environment variables.

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
project ownership of cheap invariant checks, tests, expensive checks, and
future project-specific static analysis. `--no-project-tests` suppresses only
the project test phase; project hygiene still runs in CI.
