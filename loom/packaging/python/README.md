# Loom SDK wheel

The `hrx-loom` wheel contains the Loom compiler, runtime, command-line tools, and
`loom` Python authoring and model APIs. Pip installs the commands into the Python
environment's `bin/` directory on Unix or `Scripts/` directory on Windows. An
activated virtual environment makes them available on PATH.

The native commands are `loom-benchmark`, `loom-check`, `loom-compile`,
`loom-format`, `loom-import-cxx`, `loom-link`, `loom-opt`, `loom-run`, `loom-serve`,
and `loom-test`. Python supplies `loom-compile-report`, `loom-export-cxx`,
`loom-fetch`, and `loom-lint`. The wheel includes the matching native shared
libraries beside the executables. C++ import includes its standard and Loom
headers, so it works without a separate compiler or source checkout.
Linux wheel packaging gives each native file an adjacent-library search path;
installed commands load their matching companions without Bazel build paths.

The Python APIs use the standard library. The optional `hf` extra enables Hugging
Face acquisition; local and HTTP model sources work without it. Model weights
are acquired separately with `loom-fetch`.

Build the development wheel through Bazel and install it with Python 3.12 or
newer. For Linux x86-64:

```sh
iree-bazel-build --config=loom-sdk --config=opt //loom/packaging/python:wheel
python -m pip install bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0-py3-none-linux_x86_64.whl
loom-serve --help
loom-benchmark --help
loom-fetch model.json --list --json
loom-fetch model.json --preset=check --output=./model
python -m loom.model.documentation --card model.json --output=model.md
```

To include the optional provider when installing a local wheel:

```sh
python -m pip install 'bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0-py3-none-linux_x86_64.whl[hf]'
```

The wheel filename describes the selected target platform: `linux_x86_64`,
`linux_aarch64`, or `win_amd64`. Windows cross-builds use
`--config=windows-x86_64` with the toolchain setup in
[BUILDING.md](../../../BUILDING.md#windows-from-linux). Linux development wheels
use the build toolchain's libc baseline; these tags make no manylinux portability
claim.

The final release artifact is built through the `release_wheel` target:

```sh
iree-bazel-build --config=loom-sdk --config=opt //loom/packaging/python:release_wheel
```

On Linux x86-64, Bazel audits the ELF ABI before producing
`manylinux_wheel/hrx_loom-VERSION-py3-none-manylinux_2_39_x86_64.whl` beneath this
package's `bazel-bin` directory. An incompatible toolchain fails this action.
The audit changes only wheel tags and their recorded digests; it cannot add
system libraries or alter the installed payload. Windows retains the original
`win_amd64` wheel. Linux aarch64 development builds do not yet have a qualified
release artifact.

The installed-artifact check runs on Linux or native Windows:

```sh
iree-bazel-test --config=loom-sdk --config=opt //loom/src/loom/tooling/target/cpu/serve:wheel_model_card_test
```

It installs the final wheel into an isolated environment, exercises all commands
and C++ import, fetches a fixture over HTTP, reuses it offline, and checks native
inference and graceful shutdown. Two simultaneous servers establish that stopping
one leaves the other serving requests. Windows uses separate consoles and real
console control events. Cross-building the target verifies its inputs; execution
requires the matching operating system.

The [SDK build workflow](../../../.github/workflows/build_loom_sdk.yml) runs
this installed test for all four profiles on Linux and Windows. The producer
exports the exact `release_wheel` target only after that test succeeds:

```sh
python -m build_tools.ci.loom_wheel --profile=release --output=wheel-artifact
```

Run it from the repository root with Bazelisk and the native compiler toolchain
on PATH. The output directory receives one wheel and existing files are never
replaced. Optional Bazel flags follow `--`. PR and manual build jobs have read
permissions and produce test artifacts without publishing a release.

`--config=loom-sdk` selects the SDK's CPU and libamdf runtime drivers, compiler
targets, and media codecs independently of the worktree configuration. Native
AMD execution uses the operating-system device driver; it requires no ROCm SDK.
The `amdgpu2`, `rdna`, `cdna`, and `xdna` drivers use libamdf. The legacy `amdgpu`
driver is excluded from this configuration. Instrumentation and optimization
are selected separately, as shown below.

Windows SDK profiles use the redistributable dynamic C runtime. The debug
profile keeps unoptimized code and assertions without requiring Visual Studio's
development-only debug runtime. Profiles that produce PDB symbols install them
beside their matching executables and libraries, including libamdf's private
Windows bridge, so a debugger can find them in the installed environment.

Installed users need Python, the wheel, and the platform prerequisites in the
[package description](PYPI_README.md); Bazel is a package-author build tool.
`loom-serve --device=task --model=./model/model.json` loads the fetched card's
selected preset directly; choose a device supported by that model. A separate
serving deployment can supply routing aliases, placement, and execution
overrides.

The wheel carries versioned JSON schemas alongside `loom.model`. Applications
can read them without accessing a repository:

```python
from importlib.resources import files

schema = files("loom.model").joinpath("model-card.v1.schema.json").read_text()
```

The checked-in [version.json](version.json) owns the SDK's base package version
and compatible model catalog association. Bazel reads it during analysis; builds
have no environment or timestamp version override. The current development
identity is `0.0.0.dev0` with no associated catalog. This target builds a local
artifact and has no publishing command.

Release preparation uses canonical PEP 440 versions such as `1.2.3`, `1.2.3a1`,
`1.2.3b1`, or `1.2.3rc1`, with a matching `v`-prefixed Git tag. Development
versions can append `.devN` and are buildable but cannot pass the release tag
check. A release also names a compatible published library catalog. For example,
to preview an identity change on a release-preparation branch:

```sh
python loom/packaging/python/release.py \
  --identity loom/packaging/python/version.json prepare \
  --version=1.2.3 --catalog=github:ROCm/hrx-loom-library@v1.2.3 --dry-run
```

Omitting `--dry-run` edits only the named JSON. Review and commit that change
through the normal branch workflow. The helper creates no commit or tag and
performs no network operation. Naming the source JSON explicitly prevents an
invocation through Bazel from editing its runfiles.

The same tool verifies the identity of built artifacts and the complete release
set. It compares the actual filename, wheel metadata, catalog association, and
standalone package description with the checked-in sources:

```sh
python loom/packaging/python/release.py verify /path/to/hrx_loom-1.2.3-py3-none-win_amd64.whl
python loom/packaging/python/release.py check-tag v1.2.3
python loom/packaging/python/release.py bundle /path/to/release-assets --tag=v1.2.3
```

The bundle contains one wheel for each combination of Linux/Windows and the four
native profiles below. An incomplete or mismatched set fails before
`SHA256SUMS` is created. Repeating the command accepts an identical checksum file
and rejects conflicting bytes. These checks establish artifact identity;
installed-wheel tests establish runtime behavior. The normal profile is the
PyPI artifact; the instrumented profiles are explicit GitHub release assets.
These commands prepare and check local files without publishing them.

The native build configuration determines the installed version. Publishers
supply a base version without a `+` suffix; the wheel rule adds the matching
suffix automatically:

| Build configuration | Version | Native behavior |
| --- | --- | --- |
| `--config=opt` | Base version | Optimized compiler and runtime. |
| `--config=opt --//runtime/config/base/tracing:provider=tracy` | Base version plus `+tracy` | Optimized with Tracy instrumentation. |
| `--compilation_mode=dbg` | Base version plus `+debug` | Debug compiler and runtime. |
| `--config=asan` | Base version plus `+asan` | AddressSanitizer with debug information. |

An uninstrumented fastbuild is allowed for a `.dev` version. Release versions
require one of the configurations above. Unsupported combinations fail during
Bazel analysis, so a wheel cannot silently lose its instrumentation identity.
Use the repository's complete `--config=opt` profile for optimized artifacts.

All profiles install the same distribution and commands. Pip replaces the
complete SDK when installing a different profile into the same environment;
separate virtual environments keep profiles side by side. The optional `[hf]`
extra works with every profile and does not choose a native build.

For example, the Tracy development wheel on Linux x86-64 is built and installed
with:

```sh
iree-bazel-build --config=loom-sdk --config=opt \
  --//runtime/config/base/tracing:provider=tracy //loom/packaging/python:wheel
python -m pip install bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0+tracy-py3-none-linux_x86_64.whl
```

Tracy records startup events before a collector attaches. The
[tracing guide](../../../runtime/src/iree/base/tracing/README.md) describes the
matching collector version, connection, and capture lifetime. An instrumented
build is intended for deliberate capture; it is not an idle runtime switch.

Release publishing keeps normal wheels on the package index and instrumented
wheels as explicitly selected release assets. PyPI does not accept local-version
suffixes. A requirement such as `hrx-loom==1.2.3` also accepts `1.2.3+tracy`, so
switching back selects the normal wheel file or URL explicitly. This target
does not publish either kind of artifact.

SDK publishers set `package-version` and the independent `model-catalog`
association in `version.json`, such as
`github:OWNER/REPOSITORY@COMPATIBLE_TAG`. Several SDK versions
may share a tag when the publication is qualified for each consumer. The
association is stored in the wheel's `loom.json` distribution metadata.

With an association, `loom-fetch --list` discovers the current compatible models.
Bare `loom-fetch` opens the terminal picker; `loom-fetch --catalog=./catalog.json`
opens a local publication. Both ask for the selections and installation directory.
Without one, discovery requires an explicit `--catalog`; direct card URLs remain
usable. The development wheel carries no default tag. See the
[catalog workflow](../../py/loom/model/README.md#catalog-discovery) for generation,
selection, and offline snapshot behavior.
