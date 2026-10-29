# Maintaining Loom SDK releases

An SDK release publishes `hrx-loom` for Linux x86-64 and Windows x86-64 from one
reviewed source revision. Each platform has Release, Tracy, Debug, and ASAN
wheels. GitHub Releases receives all eight wheels and `SHA256SUMS`; PyPI receives
the two Release wheels. Publication uses the exact artifacts that passed the
installed tests.

This guide covers release ownership and maintenance. The [SDK build guide](README.md)
covers local builds, native profiles, and installed tests. Commands below run
from the repository root with Python 3.12 or newer; GitHub commands use `gh`.

## CI and release artifacts

| Trigger | SDK profiles | Publication |
| --- | --- | --- |
| Ordinary PR / main CI | Release | Workflow artifacts only |
| Manual SDK workflow | Requested profiles | Workflow artifacts only |
| Release tag or manual release workflow on that tag | Release, Tracy, Debug, ASAN | GitHub Releases, then PyPI |

The shared [artifact producer](../../../.github/workflows/build_loom_sdk_artifacts.yml)
runs once per requested profile on Linux. It builds and tests the Linux wheel,
cross-compiles the Windows wheel with clang-cl, and exports the Windows wheel
and small test fixture. Release-size reporting measures these same archives
without another build. No model checkpoints are acquired by this qualification.

In [ordinary CI](../../../.github/workflows/ci_iree.yml), the existing Windows
Clang CL job installs and tests the cross-built Release wheel. The
[manual/release SDK workflow](../../../.github/workflows/build_loom_sdk.yml)
uses one Windows job to qualify every requested profile. Windows qualification
invokes no build tool. Existing PR sanitizer jobs retain their source-test
coverage; they do not produce instrumented SDK wheels.

The publication workflow consumes only `loom-sdk-wheel-*` artifacts. The
`loom-sdk-inputs-windows-*` artifacts contain inputs awaiting native qualification
and are not publication artifacts. Build and qualification artifacts are retained
for 14 days.

## Publishing account setup

The [release workflow](../../../.github/workflows/release_loom_sdk.yml) uses the
repository's `GITHUB_TOKEN` to create or complete the GitHub release. Its PyPI
job runs in the `pypi` GitHub environment and uses OIDC with attestations through
the official PyPA publishing action. It does not require a stored PyPI API token.

The PyPI trusted publisher configuration matches these fields:

| Field | Value |
| --- | --- |
| Project | `hrx-loom` |
| GitHub owner | `ROCm` |
| Repository | `hrx-system` |
| Workflow filename | `release_loom_sdk.yml` |
| Environment | `pypi` |

For the first publication, a
[pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
can create the PyPI project on first use. An existing project uses its
[publisher settings](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
The repository's release owners configure the GitHub environment, its approval
and tag restrictions, and the corresponding PyPI publisher. These account
settings are separate from the checked-in workflow.

## Prepare and qualify a version

[version.json](version.json) owns the base `package-version` and the independent
`model-catalog` association. Bazel reads it during analysis; environment variables
and timestamps do not override the version. Native build configuration adds
`+tracy`, `+debug`, or `+asan` to the instrumented wheels automatically.

Release versions use `X.Y.Z`, `X.Y.ZaN`, `X.Y.ZbN`, or `X.Y.ZrcN`, with the
matching `v`-prefixed Git tag. A `.devN` version is buildable but cannot pass the
release gate. The gate also requires a catalog association, such as
`github:ROCm/hrx-loom-library@v1.2.3`.

The catalog supplies the default `loom-fetch --list` and picker contents.
Several SDK versions may share a catalog reference when its models are qualified
for those consumers. The identity checker validates the reference's syntax; it
does not fetch the catalog or establish model compatibility. Catalog publication
and compatibility qualification precede the SDK release. The
[catalog guide](../../py/loom/model/README.md#catalog-discovery) describes the
publication format and discovery behavior.

On a release-preparation branch, preview the identity change:

```sh
python loom/packaging/python/release.py \
  --identity loom/packaging/python/version.json prepare \
  --version=1.2.3 --catalog=github:ROCm/hrx-loom-library@v1.2.3 --dry-run
```

Repeating without `--dry-run` edits only the named JSON. The release-preparation
PR includes that change and any updates to [PYPI_README.md](PYPI_README.md), the
package description embedded in every wheel. Review and CI establish the source
revision to release; the preparation helper creates neither a commit nor a tag.

Ordinary PR CI covers the Release SDK. To qualify the other profiles on a branch
named `sdk-release-preparation`:

```sh
gh workflow run build_loom_sdk.yml --repo ROCm/hrx-system \
  --ref sdk-release-preparation -f 'profiles=["Tracy","Debug","ASAN"]'
```

This workflow builds and tests artifacts without publishing them. The installed
tests cover command availability, C++ import, HTTP and offline fixture fetching,
native inference, and independent server shutdown. Native Windows additionally
exercises real console control events. The [build guide](README.md#build-and-test)
describes local execution of the same checks.

## Publish the reviewed tag

From the reviewed source revision, the release owner can check the intended tag:

```sh
python loom/packaging/python/release.py check-tag v1.2.3
```

Pushing the matching `v*.*.*` tag starts `Loom / SDK / Release`. An existing tag
can also be selected explicitly:

```sh
gh workflow run release_loom_sdk.yml --repo ROCm/hrx-system --ref v1.2.3
```

This is a publishing workflow. It checks the tag against the committed identity,
builds all four profiles for both platforms, and requires every installed test
to pass. The bundle job verifies the complete set and writes `SHA256SUMS`.
Both publication jobs verify the bundle again and consume its bytes without
recompiling. GitHub publication completes before PyPI publication starts.

GitHub publication writes a compact SDK description naming the PyPI install,
attached native profiles, and checksum document. The body comes from the
committed SDK identity and artifact contract, so the first release cannot expand
into the repository's complete pull-request history.

The normal filenames have no local-version suffix; instrumented filenames use
`+tracy`, `+debug`, or `+asan`. Linux release wheels carry the audited
`manylinux_2_39_x86_64` tag; Windows wheels carry `win_amd64`. Prerelease versions
create GitHub prereleases. All profiles install the same distribution, so users
choose an instrumented wheel explicitly or use separate virtual environments.

A complete release has eight wheels and `SHA256SUMS` on GitHub, and the two
normal wheels on PyPI. From a checkout of the release tag, a downloaded complete
bundle can be checked without network access:

```sh
python -m build_tools.ci.loom_release verify \
  --tag=v1.2.3 --directory=release-assets
```

The checker compares wheel names, metadata, native profile identities, catalog
association, package description, and checksums with that checkout. It does not
rerun native tests or publish anything.

## Recover a failed publication

Retrying failed publication jobs in the same run reuses successful build jobs'
artifacts. Once the failure's cause is resolved, GitHub's **Re-run failed jobs**
action, or the corresponding [CLI command](https://cli.github.com/manual/gh_run_rerun),
keeps successful producers intact:

```sh
gh run rerun RUN_ID --repo ROCm/hrx-system --failed
```

The retry behavior is deliberate:

| Condition | Result |
| --- | --- |
| Matching GitHub assets already exist | Download and compare their SHA-256, then upload only missing assets. |
| Matching normal wheels already exist on PyPI | Compare published hashes and yanked state, then upload only missing wheels. |
| Existing bytes differ, a wheel is unexpected, or a PyPI wheel is yanked | Fail; changed artifacts need a new version. |
| A GitHub asset is incompletely uploaded, or release draft/prerelease metadata conflicts | Fail for release-owner review before further uploads. |
| Authentication, rate limiting, or a service fails | Report the error; retry after the cause is resolved. |

No publication operation overwrites an existing asset. After the 14-day workflow
artifact retention expires, a new build must produce byte-identical artifacts to
resume a partial release. Different bytes require a new version. A new workflow
dispatch starts new producers; it is not the same operation as rerunning failed
publication jobs.

## Maintaining the release machinery

| Concern | Source of truth |
| --- | --- |
| Version, catalog association, and package description | [version.json](version.json), [PYPI_README.md](PYPI_README.md) |
| Wheel contents, native profiles, and Linux ABI audit | [BUILD.bazel](BUILD.bazel), [defs.bzl](defs.bzl), [native.bzl](native.bzl), [manylinux.bzl](manylinux.bzl) |
| Compiler, runner image, and cross-SDK pins | [Artifact workflow](../../../.github/workflows/build_loom_sdk_artifacts.yml), [Windows setup](../../../.github/scripts/setup_windows_cross.sh), [SDK manifest](../../../.github/scripts/windows-sdk-channel.json) |
| Artifact export and installed qualification | [loom_wheel.py](../../../build_tools/ci/loom_wheel.py), [installed oracle](../../src/loom/tooling/target/cpu/serve/wheel_model_card_test.py) |
| Identity, bundle checks, and publication retries | [release.py](release.py), [loom_release.py](../../../build_tools/ci/loom_release.py), [release workflow](../../../.github/workflows/release_loom_sdk.yml) |

A compiler, sysroot, native-library, or packaging change is qualified through
the actual Linux and Windows wheels, including each affected instrumented
profile. The Linux compiler and Windows compiler-rt versions match; SDK/CRT
inputs and their digests are pinned together. Linux ABI compatibility comes from
the `release_wheel` audit. The destination-OS installed tests establish loader
paths and runtime behavior that cross-compilation alone cannot check.
