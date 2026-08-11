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

Source builds are a contributor workflow. The rest of this guide deliberately
uses the installed tools exactly as a kernel or application author will see
them.
