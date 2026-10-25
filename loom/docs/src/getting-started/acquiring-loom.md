# Acquiring Loom

The `hrx-loom` Python wheel installs the Loom compiler, runtime, Python APIs,
and command-line tools, including `loom-fetch`, `loom-serve`, and
`loom-benchmark`. Source models use the included JIT compiler. Running a
published model requires neither a repository checkout nor a build toolchain.

!!! info "First release"

    The first PyPI release has not been published. Until then, install a tested
    development wheel from the [Loom SDK build workflow](https://github.com/ROCm/hrx-system/actions/workflows/build_loom_sdk.yml).
    The PyPI commands below describe installation once a release is available.

## Platform requirements

Use Python 3.12 or newer and one of these platforms:

| Platform | Required system runtime |
| --- | --- |
| Linux x86-64 | glibc 2.39 or newer and `libstdc++` providing `GLIBCXX_3.4.31`; Ubuntu 24.04 with its standard `libstdc++6` satisfies this baseline. |
| Windows 11 x86-64 | [Microsoft Visual C++ v14 x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist). |

AMD GPU and NPU execution uses libamdf with the operating system's device
driver. The account running Loom needs permission to access the device.
The SDK includes its native libraries and C++ importer support; a ROCm SDK
installation is not required. A model must supply programs for the chosen
device: installing the SDK does not make every model portable to every target.

## Install in a virtual environment

=== "Linux"

    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install 'hrx-loom[hf]'
    ```

=== "Windows PowerShell"

    ```powershell
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install 'hrx-loom[hf]'
    ```

The optional `[hf]` extra installs Hugging Face acquisition support. Plain
`hrx-loom` is sufficient for models whose assets come from local directories
or ordinary HTTP URLs. Both install the same native tools.

For a development wheel, download the artifact from a successful job for your
platform and extract its ZIP. Install the `.whl` inside it instead of the
package name. For example, the Linux development release profile uses:

```sh
python -m pip install './hrx_loom-0.0.0.dev0-py3-none-manylinux_2_39_x86_64.whl[hf]'
```

Windows uses the corresponding `win_amd64.whl`. Use the exact filename in the
artifact; its version changes with the build. The `release` profile is the
ordinary optimized SDK. Explicit `tracy`, `debug`, and `asan` artifacts contain
the same package with different native instrumentation. They replace the
installed SDK in that environment; they are not pip extras. Separate virtual
environments let these builds coexist.

## Check the installation

```sh
loom-serve --help
loom-fetch --help
loom-benchmark --help
loom-run --list_drivers
```

Activating the environment puts its `bin/` directory on PATH on Linux and its
`Scripts/` directory on PATH on Windows. The tools load their matching native
libraries from the installation. If a command is not found, activate the same
environment in which pip installed the wheel.

`task` selects CPU execution, `amdgpu` selects native AMD GPU execution, and
`xdna` selects AMD NPU execution. The driver list reports compiled-in drivers;
creating a device still requires supported hardware and operating-system access.

Continue with [fetch and serve a model](../serving/workflows/fetch-and-serve.md)
for a complete speech-recognition request, or [run your first
kernel](first-kernel.md) to author and execute Loom programs.

## Building from source

Compiler contributors can build the tools from the
[HRX System repository](https://github.com/ROCm/hrx-system). The repository's
[Loom quickstart](https://github.com/ROCm/hrx-system/tree/main/loom#build-the-tools)
tracks the source-build prerequisites and Bazel targets. The
[SDK build guide](https://github.com/ROCm/hrx-system/tree/main/loom/packaging/python)
describes building and testing wheels through Bazel.

Direct source builds are a compiler-contributor workflow. The command-line
pages deliberately use installed tools exactly as a kernel or application
author sees them; Bazel authoring repositories consume those tool roles through
the module boundary below.

## Consuming Loom from Bazel

An independent kernel or model repository can depend on the HRX Bzlmod module
and load Loom's public authoring rules through `@hrx//loom/...`. The module
registers source-built toolchains automatically, while a local module override
keeps compiler and kernel edits in one live Bazel graph. [Build libraries and
binaries with Bazel](../workflows/build-with-bazel.md#depend-on-hrx) defines the
module declaration, public loads, target profiles, and deployment products.

Once the tools or Bzlmod dependency are available, [run your first
kernel](first-kernel.md) from one checked source file.
