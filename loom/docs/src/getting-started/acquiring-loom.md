# Acquiring Loom

The programming guide assumes that the Loom command-line tools are installed
on `PATH`. Guide commands use the public executable names directly; they do not
require a source checkout or expose the repository's build-system labels.

!!! info "Release packages"

    Prebuilt Loom releases are not published yet. Release archives, supported
    host platforms, installation verification, and version-selection guidance
    will be documented here when the first distribution is available.

## Building from source

Compiler contributors can build the tools from the
[HRX System repository](https://github.com/ROCm/hrx-system). The repository's
[Loom quickstart](https://github.com/ROCm/hrx-system/tree/main/loom#build-the-first-slice)
tracks the source-build prerequisites and Bazel targets while release packaging
is being established.

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
