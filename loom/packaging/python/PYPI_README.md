# Loom SDK

`hrx-loom` contains the Loom compiler, runtime, model tools, and Python APIs.
It compiles programs for CPUs, AMD GPUs, and AMD NPUs, and serves models through
`loom-serve`. C++ import includes its headers and compiler support.

## Install

Use Python 3.12 or newer in a virtual environment:

```sh
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install hrx-loom
loom-serve --help
```

Pip installs the native commands and their matching libraries in the environment's
`bin` directory on Linux or `Scripts` directory on Windows. Activating the
environment puts the commands on PATH. Building models from Loom source uses the
included JIT compiler; it does not require Bazel or a separate toolchain.

The initial platforms are Linux x86-64 with glibc 2.39 or newer and Windows 11
x86-64. Linux also needs a C++ runtime providing `GLIBCXX_3.4.31`; Ubuntu 24.04
with its standard `libstdc++6` package satisfies these requirements. Windows
needs the [Microsoft Visual C++ v14 x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
AMD execution uses libamdf and requires a supported operating-system device
driver and device access. No ROCm SDK installation is required.

## Fetch and serve a model

Hugging Face acquisition uses the optional `hf` extra. Local files and HTTP
sources work with the base package:

```sh
python -m pip install 'hrx-loom[hf]'
loom-fetch --list
loom-fetch
```

The release's compatible model catalog supplies the list and terminal picker.
A development SDK without an associated catalog requires `--catalog` or a model
card path or URL. Cards describe the exact files, variants, and runnable presets.
For example, install the library's Moonshine Streaming Tiny model on an AMD GPU:

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/21637a18079d1f1dd9b8e20d26f700a62e60db4f/model/moonshine-ai/moonshine-streaming-tiny/model.json --preset=streaming --output=./moonshine
loom-serve --model=./moonshine/model.json --device=amdgpu2
```

Choose a device supported by the model: `task` for CPU, `amdgpu2` for AMD GPU,
or `xdna` for AMD NPU. `loom-run --list_drivers` lists the installed drivers.
The fetched card preserves the selected preset and can be served directly.
Model weights are downloaded separately; the SDK does not contain checkpoints.
The [fetch-to-serve walkthrough](https://rocm.github.io/hrx-system/loom/serving/workflows/fetch-and-serve/)
includes the audio request, expected transcript, and offline/cache behavior.

## Tools and Python APIs

The installed commands are `loom-benchmark`, `loom-check`, `loom-compile`,
`loom-compile-report`, `loom-export-cxx`, `loom-fetch`, `loom-format`,
`loom-import-cxx`, `loom-link`, `loom-lint`, `loom-opt`, `loom-run`, `loom-serve`,
and `loom-test`. Each provides `--help`.

The `loom` Python package provides program authoring and model-card APIs using
the Python standard library. Versioned JSON schemas are included with the SDK:

```python
from importlib.resources import files

schema = files("loom.model").joinpath("model-card.v1.schema.json").read_text()
```

The [model format and acquisition guide](https://github.com/ROCm/hrx-system/blob/main/loom/py/loom/model/README.md)
describes origins, variants, presets, locks, offline use, and model publication.
The [Loom library](https://github.com/ROCm/hrx-loom-library) contains model cards,
programs, kernels, and their tests. The
[SDK build guide](https://github.com/ROCm/hrx-system/blob/main/loom/packaging/python/README.md)
describes Bazel builds and explicit Tracy, debug, and AddressSanitizer wheel
profiles.
