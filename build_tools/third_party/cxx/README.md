# CXX semantic parser fork

HRX imports the CXX semantic parser from an immutable source archive. Parser
changes live in the maintained fork rather than as patches in this directory:

- Upstream: <https://github.com/robertoraggi/cplusplus>
- Maintained fork: <https://github.com/benvanik/cplusplus>
- Current stack: [`hrx-upstream/66-automatic-storage-identity`](https://github.com/benvanik/cplusplus/commits/hrx-upstream/66-automatic-storage-identity/)

`../deps.MODULE.bazel` owns the archive commit and checksum. The generated root
`MODULE.cmake.lock` carries the same identity for CMake. `cxx.BUILD.bazel` and
`cxx.cmake` adapt the parser library to HRX's build systems; they are not a
second source tree. New parser behavior belongs in the fork, and in-tree patch
files are not part of this dependency model.

## Stack structure

The fork keeps one linear, dependency-ordered commit for each independently
reviewable upstream landing unit. Local cut-point branches use
`hrx-upstream/NN-short-description`, where `NN` is the two-digit landing
position. Each branch points at its commit in the shared linear history; it
does not contain a parallel version of the change.

Only the terminal stack branch and branches with an active upstream pull
request are published. Intermediate cut points remain local until review. If
every cut point is published, GitHub labels each ancestor with every descendant
branch and obscures the history.

Stack commits receive no Git tags. Tags in the CXX repository identify
upstream releases. Rewriting a stack commit changes that commit and every
descendant commit, so the terminal archive identity and HRX pin must change as
part of the same operation.

To create the next landing unit from a clone of the maintained fork:

```sh
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/robertoraggi/cplusplus.git
git fetch origin --prune
git fetch upstream --prune
git switch hrx-upstream/63-portable-wide-integer-limits
git switch -c hrx-upstream/64-short-description
```

Update the terminal branch named in this README whenever the stack grows. A
rebase of the stack uses `git rebase --update-refs` so every local numbered
branch continues to identify its landing unit. Force updates use
`--force-with-lease` and apply only to branches owned by the maintained fork.

## Commit contract

CXX history uses a plain imperative subject such as `Preserve explicit
reference pack identity`. HRX subsystem prefixes such as `[CXX]`, stack
numbers, issue identifiers, and HRX patch filenames do not belong in CXX
subjects. The body explains the language or representation contract, the
mechanism that establishes it, and any non-obvious consequence for later
compiler stages.

Each behavior change carries a focused, production-shaped test in the same
commit. The parent must demonstrate the defect: the new test fails there, or
the recorded expected output differs for the behavior being repaired. Exact
parser, semantic, diagnostic, and code-generation output stays exact when
order, operands, types, or the complete diagnostic set are part of the
contract.

Generated products change with their owning source definitions and must be at
a fixed point in the same commit. A prerequisite with its own contract and
test is an earlier landing unit; test scaffolding or a partial implementation
is not a standalone stack layer.

## CXX qualification

Run commands from the CXX repository root and reuse its single `build/`
directory. `uv sync` supplies lit; FileCheck must also be discoverable.

```sh
uv sync
cmake --preset default-mlir
cmake --build build --parallel "${CXX_BUILD_JOBS:-4}"
ctest --test-dir build --output-on-failure
```

The configure log must report both `Using lit` and `Using filecheck`. CXX
silently omits the lit suites when either tool is absent, so a green CTest run
without those messages is incomplete.

During development, the aggregate lit suites provide useful focused checks:

```sh
ctest --test-dir build --output-on-failure -R '^sema$'
ctest --test-dir build --output-on-failure -R '^codegen$'
```

The full CTest invocation is required before moving the terminal stack or
opening an upstream pull request. It covers the lit suites and the compiled API
and emitter tests.

## Updating the HRX pin

Publish the terminal branch before computing its archive identity. GitHub's
generated archive bytes, including their compression and top-level directory,
define the dependency checksum:

```sh
commit=$(git rev-parse HEAD)
archive=/tmp/cplusplus-${commit}.tar.gz
curl --fail --location --output "${archive}" \
  "https://github.com/benvanik/cplusplus/archive/${commit}.tar.gz"
sha256sum "${archive}"
```

Update the URL, `strip_prefix`, and SHA-256 in
`build_tools/third_party/deps.MODULE.bazel`, then regenerate and verify the
CMake lock from the HRX repository root:

```sh
.venv/bin/python -B build_tools/bazel_to_cmake/deps.py
.venv/bin/python -B build_tools/bazel_to_cmake/deps.py --check
```

Qualify the archive through HRX's importer and normal presubmit path:

```sh
build_tools/bin/iree-bazel-test \
  --config=asan \
  --//loom/config/import:enable=cxx \
  --nocache_test_results \
  --keep_going \
  //loom/src/loom/import/cxx/...

.venv/bin/python -B dev.py bazel precommit \
  --profile paranoid \
  --base origin/main \
  --verbose
```

An upstream landing removes its commit from the private remainder of the stack.
Rebase the remaining local branches onto the new upstream main, run the full
CXX qualification, publish the next review branch, and repin HRX to the new
terminal commit. The archive remains immutable throughout that process even
though its identity advances.
