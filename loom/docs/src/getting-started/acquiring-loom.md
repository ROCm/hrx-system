# Acquiring Loom

The [hrx-loom Python wheel](https://pypi.org/project/hrx-loom/) installs the
Loom compiler, runtime, Python APIs, and command-line tools, including
`loom-fetch`, `loom-serve`, and
`loom-benchmark`. Source models use the included JIT compiler. Running a
published model requires neither a repository checkout nor a build toolchain.

## Platform requirements

Use Python 3.12 or newer and one of these platforms:

| Platform | Required system runtime |
| --- | --- |
| Linux x86-64 | glibc 2.39 or newer and `libstdc++` providing `GLIBCXX_3.4.31`; Ubuntu 24.04 with its standard `libstdc++6` satisfies this baseline. |
| Windows 11 x86-64 | [Microsoft Visual C++ v14 x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist). |

AMD GPU and NPU execution uses libamdf with the operating system's device
driver. The account running Loom needs permission to access the device.
The SDK includes its native libraries and C++ importer support; a ROCm SDK
installation is not required. Vulkan execution dynamically loads the system
Vulkan loader and requires a Vulkan 1.3 device with buffer device address
support. The SDK does not bundle a Vulkan loader or vendor driver. A model must
supply programs for the chosen device: installing the SDK does not make every
model portable to every target.

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

For a development wheel, download `loom-sdk-wheel-linux-Release` or
`loom-sdk-wheel-windows-Release` from a successful Loom CI run and extract its
ZIP. These artifacts contain the qualified wheel for your platform. Install
the `.whl` inside it instead of the package name. For example, the Linux
development release profile uses:

```sh
python -m pip install './hrx_loom-0.2.0.dev0-py3-none-manylinux_2_39_x86_64.whl[hf]'
```

Windows uses the corresponding `win_amd64.whl`. Use the exact filename in the
artifact; its version changes with the build. The `release` profile is the
ordinary optimized SDK. Explicit `tracy`, `debug`, and `asan` artifacts contain
the same package with different native instrumentation. They replace the
installed SDK in that environment; they are not pip extras. Separate virtual
environments let these builds coexist. Instrumented wheels are available from
releases and explicit [SDK workflow runs](https://github.com/ROCm/hrx-system/actions/workflows/build_loom_sdk.yml).

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

`task` selects CPU execution, `amdgpu` selects native AMD GPU execution, `xdna`
selects AMD NPU execution, and `vulkan` selects Vulkan execution. The driver
list reports compiled-in drivers; creating a device still requires supported
hardware and operating-system access.

Continue with [fetch and serve a model](../serving/workflows/fetch-and-serve.md)
for a complete speech-recognition request, or [run your first
kernel](first-kernel.md) to author and execute Loom programs.

### Check device access inside a sandbox

Device discovery and execution access are separate. For example:

```sh
loom-run --list_devices=xdna
```

An NPU can appear in this list through host discovery metadata even when the
process cannot open it. On Linux, a container or agent sandbox needs the host's
selected `/dev/accel/accelN` device node, permission to open it for reading and
writing, and a device-access policy that allows it. Seeing `/sys/class/accel`
does not provide that access. GPU exposure through `/dev/kfd` and `/dev/dri`
does not expose the NPU.

Inspect the selected node from the same environment that will run the server:

```sh
ls -ld /sys/class/accel /dev/accel
ls -l /dev/accel/accel0
id
```

Here `accel0` is an example; use the node for the intended device. If it exists
on the host but not inside the sandbox, the launcher owns exposing it. If it
is visible but opening it returns `EACCES` or `EPERM`, check the account's access
and the sandbox's device policy. An `ENOENT` while opening the accelerator node
means the path is absent; changing model programs or tokenizer files cannot
repair it. The launcher may need to recreate the sandbox after its device
configuration changes.

Starting an empty server checks device creation before downloading weights or
debugging a model. For the first discovered NPU:

```sh
loom-serve --device=xdna://0 --pool_backing=fixed --listen=127.0.0.1:8080
```

The listener's `ready` record establishes server/device startup, not model
execution. A registered model's readiness and a completed inference request
establish the later steps. Device creation can also reject an incompatible
host driver or firmware; the [native AMD requirements](https://github.com/ROCm/hrx-system/tree/main/libamdf#native-drivers)
describe those prerequisites. Ctrl-C stops the empty server. Windows uses its
native device-access mechanism, not Linux device-node mounts.

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
