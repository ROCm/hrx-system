# Maintaining Loom SDK releases

An SDK release publishes `hrx-loom` for Linux x86-64 and Windows x86-64 from one
reviewed source revision. Each platform has Release, Tracy, Debug, and ASAN
wheels. GitHub Releases receives all eight wheels, standalone `loom-serve` and
`amdf-info` archives for each platform, and `SHA256SUMS`. Stable releases also
publish the two Release wheels to PyPI. Publication uses the exact artifacts
that passed the installed tests.

This guide covers release ownership and maintenance. The [SDK build guide](README.md)
covers local builds, native profiles, and installed tests. Commands below run
from the repository root with Python 3.12 or newer; GitHub commands use `gh`.

## CI and release artifacts

| Trigger | SDK profiles | Publication |
| --- | --- | --- |
| Ordinary PR / main CI | Release | Workflow artifacts only |
| Manual SDK workflow | Requested profiles | Workflow artifacts only |
| Prerelease tag or manual release workflow on that tag | Release, Tracy, Debug, ASAN | Public GitHub prerelease |
| Stable tag or manual release workflow on that tag | Release, Tracy, Debug, ASAN | GitHub Releases, then PyPI |

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

The publication workflow consumes `loom-sdk-wheel-*` and
`loom-standalone-archive-*` artifacts. The `loom-sdk-inputs-windows-*` artifacts
contain inputs awaiting native qualification and are not publication artifacts.
Build and qualification artifacts are retained for 14 days.

The standalone `amdf-info` archive contains the exact executable from the Release
wheel. Destination-OS qualification runs help, text inventory, and JSON inventory
with only that executable in its directory, an empty `PATH`, and no SDK loader
overrides before exporting the archive. It adds no build or Python installation.
The executable embeds libamdf; Linux retains the wheel's glibc baseline, and
Windows retains its Microsoft C runtime prerequisite. Device activation remains
separate hardware qualification. CI bootstrap consumers can pin a release archive
and its SHA-256 without configuring the repository's build graph.

## Publishing account setup

The [release workflow](../../../.github/workflows/release_loom_sdk.yml) uses the
repository's `GITHUB_TOKEN` to create or complete a draft GitHub release and
publish it after its asset set is complete. Its PyPI job runs in the `pypi`
GitHub environment and uses OIDC with attestations through the official PyPA
publishing action. It does not require a stored PyPI API token.

Repository owners enable
[immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)
before creating the first release governed by this process. GitHub applies the
setting only to releases created after it is enabled. Drafts remain repairable;
publication then locks the release's tag and assets. The workflow therefore
attaches and verifies every artifact while the release is a draft and performs
publication as its final GitHub mutation.

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

The main branch carries the next planned SDK version as `X.Y.Z.dev0`. Advancing
it after a stable release prevents ordinary CI artifacts from reusing the
published version for different bytes. Its catalog remains an immutable commit
and can advance independently as model changes qualify. A frozen release line
replaces the development version with its first candidate version.

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
package description embedded in every wheel. Its pull request description also
contains the exact version-marked section selected for publication. Candidate
pull requests carry a concise candidate summary and a link to the release
line's evolving unlisted Gist. The stable pull request carries the completed
editorial copied from that Gist, including the notable pull request links that
create GitHub cross-references. The
[release-note authoring contract](release_notes/README.md) defines both forms,
starts from the previous public tag and candidate commit, audits migrations
before feature prose, and records only evidence-backed outcomes and notable
changes. A missing, ambiguous, stale, empty, or placeholder-bearing section
cannot publish.

Review and CI establish the source revision to release. The release tag points
directly at the commit GitHub records as the version-bump pull request's merge
result; the publisher resolves the pull request from that commit rather than
accepting an independently supplied description. The preparation helper creates
neither a commit nor a tag.

Ordinary PR CI covers the Release SDK. To qualify the other profiles on a branch
named `sdk-release-preparation`:

```bash
for profile in Tracy Debug ASAN; do
  gh workflow run build_loom_sdk.yml --repo ROCm/hrx-system \
    --ref sdk-release-preparation -f profile="$profile"
done
```

This workflow builds and tests artifacts without publishing them. The installed
tests cover command availability, C++ import, HTTP and offline fixture fetching,
native inference, and independent server shutdown. Native Windows additionally
exercises real console control events. The [build guide](README.md#build-and-test)
describes local execution of the same checks.

## Run a release-candidate cycle

A release line isolates candidate stabilization from continued work on `main`.
Create `release/vX.Y` from the reviewed main revision selected for the release,
then target the identity-preparation pull request at that branch. The repository
tag and release line can be shared by additional HRX packages in the future;
this workflow publishes the Loom SDK assets. The release branch owns the
candidate and stable identities, while ordinary development continues on
`main`.

Create one unlisted Gist for the `X.Y.Z` release line before opening its first
candidate pull request. Seed it with the stable editorial template and maintain
the full release narrative there throughout candidate qualification. Unlisted
Gists are accessible to anyone with the URL and are not suitable for
confidential information. Gist references do not create pull request timeline
cross-references; that is intentional while the editorial is still changing.

Each candidate pull request uses the
[candidate template](release_notes/CANDIDATE_TEMPLATE.md). Its marked section
contains a short release focus, material changes since the previous candidate
or selected main revision, required migrations or test focus, the shared Gist
URL, and the feedback channel. It does not copy the full editorial or notable
pull request list. The published candidate therefore points testers at one
living draft without making every candidate a duplicate release-note archive.

The first candidate sets `package-version` to `X.Y.Zrc1` and names the
coordinated library tag
`github:ROCm/hrx-loom-library@vX.Y.Zrc1`. Build and qualify the SDK candidate
before that tag exists, then use the library's SDK qualification workflow to
pin the exact candidate wheel and source revision. After its host and hardware
gates pass, merge the library pin and run **Loom Library / Release**. That
workflow creates the same-version library tag, proves that the candidate SDK's
implicit `loom-fetch --list` path resolves the tagged admitted catalog, and
publishes the empty immutable library release.

Only after that catalog release is immutable, tag the merged HRX release-line
commit as `vX.Y.Zrc1`. The SDK workflow requires the matching immutable catalog
release before any SDK release mutation, then publishes a public GitHub
prerelease with direct wheel and standalone-executable installation instructions.
It does not enter the PyPI environment or stage a PyPI upload.

Feedback fixes land on `main` first and are then cherry-picked onto the release
line. Each changed candidate increments the package version and tag (`rc2`,
`rc3`, and so on), which preserves the exact bytes testers used and keeps every
feedback report attributable to one source revision. Unrelated features remain
on `main`; the release line takes only fixes needed to qualify the selected
release. Update the same Gist as evidence and migrations change, and summarize
only the candidate delta in the next version-bump pull request.

When the candidate is accepted, a final identity pull request on the same
release line changes both `package-version` to `X.Y.Z` and `model-catalog` to
the coordinated `github:ROCm/hrx-loom-library@vX.Y.Z` tag. Build the stable SDK
candidate, qualify its exact wheel and source revision through the library pin
workflow, and publish the immutable stable library release before creating the
HRX tag. Copy the completed Gist editorial into the stable pull request's marked
section and review it against the exact stable range. This final pull request is
the durable editorial source and creates GitHub cross-references for every
linked notable pull request. The stable tag then runs the complete qualification
again, publishes a normal GitHub release, and publishes its two Release wheels
to PyPI. Promoting GitHub metadata or reusing candidate artifacts cannot
substitute for the stable build: the stable version has its own library
association, tag, wheel metadata, checksums, and attestations.

After stable publication, replace the Gist draft with a short pointer to the
immutable GitHub release. Candidate links continue to resolve, while the
published release and its source pull request remain the only authoritative
copies of the completed editorial.

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

This is a publishing workflow. It checks the tag against the committed identity
and requires the identity's same-version catalog tag to be an immutable public
library release. It builds all four profiles for both platforms and requires
every installed test to pass. The bundle job verifies the complete set and
writes `SHA256SUMS`.
Publication consumes those bytes without recompiling. The GitHub job creates or
resumes a draft, verifies its exact asset inventory and SHA-256 digests, uploads
only missing assets, and publishes only after the set is complete. GitHub
publication completes before the stable-only PyPI publication starts.

GitHub publication copies the marked section from the exact merged version-bump
pull request, followed by a generated appendix naming the installation path,
attached native profiles, standalone executables, checksum document, and
qualification shape. For a candidate this is the concise summary and evolving
Gist link; for a stable release it is the complete reviewed editorial. The body
records the source pull request, merge commit, version, and editorial digest.
Stable releases install from PyPI; prereleases install their attached Release
wheel directly.

A draft retry resolves the source pull request again and requires the complete
composed body to match. After publication, the immutable release body is the
authoritative snapshot: retries validate its source provenance, editorial
digest, and generated appendix without depending on the still-editable pull
request description.

The normal filenames have no local-version suffix; instrumented filenames use
`+tracy`, `+debug`, or `+asan`. Linux release wheels carry the audited
`manylinux_2_39_x86_64` tag; Windows wheels carry `win_amd64`. Versions containing
`aN`, `bN`, or `rcN` create public GitHub prereleases and never publish to PyPI.
All profiles install the same distribution, so users choose an instrumented
wheel explicitly or use separate virtual environments.

A complete GitHub release has eight wheels, four standalone executable archives,
and `SHA256SUMS`. A stable release additionally has the two normal wheels on PyPI.
From a checkout of the release tag, a downloaded complete bundle can be checked
without network access:

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
| A matching GitHub draft exists | Require its body to match the merged version-bump pull request, compare API-reported SHA-256 digests, upload only missing assets, then publish it. |
| A complete matching GitHub release is already public | Validate its self-contained editorial provenance and exact assets, then accept it without another release mutation. |
| The coordinated library release is absent, mutable, or mismatched | Fail before creating or changing the SDK release. |
| Matching normal wheels already exist on PyPI | Compare published hashes and yanked state, then upload only missing wheels. |
| Existing bytes differ, an asset is unexpected, or a PyPI wheel is yanked | Fail; changed artifacts need a new version. |
| A public GitHub release is incomplete | Fail; an immutable public release cannot be repaired, so the corrected release needs a new version. |
| A GitHub asset is incompletely uploaded, or release tag/channel metadata conflicts | Fail for release-owner review before further uploads. |
| Authentication, rate limiting, or a service fails | Report the error; retry after the cause is resolved. |

No publication operation overwrites an existing asset, and no operation mutates
a completed public release. After the 14-day workflow artifact retention
expires, a new build must produce byte-identical artifacts to resume a partial
draft. Different bytes require a new version. A new workflow dispatch starts new
producers; it is not the same operation as rerunning failed publication jobs.

## Maintaining the release machinery

| Concern | Source of truth |
| --- | --- |
| Version, catalog association, and package description | [version.json](version.json), [PYPI_README.md](PYPI_README.md) |
| Editorial release-note contract and pull request templates | [release_notes/README.md](release_notes/README.md), [release_notes/CANDIDATE_TEMPLATE.md](release_notes/CANDIDATE_TEMPLATE.md), [release_notes/TEMPLATE.md](release_notes/TEMPLATE.md) |
| Wheel contents, native profiles, and Linux ABI audit | [BUILD.bazel](BUILD.bazel), [defs.bzl](defs.bzl), [native.bzl](native.bzl), [manylinux.bzl](manylinux.bzl) |
| Compiler, runner image, and cross-SDK pins | [Artifact workflow](../../../.github/workflows/build_loom_sdk_artifacts.yml), [Windows setup](../../../.github/scripts/setup_windows_cross.sh), [SDK manifest](../../../.github/scripts/windows-sdk-channel.json) |
| Artifact export and installed qualification | [loom_wheel.py](../../../build_tools/ci/loom_wheel.py), [installed oracle](../../src/loom/tooling/target/cpu/serve/wheel_model_card_test.py) |
| Standalone executables | [loom_server.py](../../../build_tools/ci/loom_server.py), [amdf_info_archive.py](../../../build_tools/ci/amdf_info_archive.py) |
| Identity, bundle checks, and publication retries | [release.py](release.py), [loom_release.py](../../../build_tools/ci/loom_release.py), [release workflow](../../../.github/workflows/release_loom_sdk.yml) |

A compiler, sysroot, native-library, or packaging change is qualified through
the actual Linux and Windows wheels, including each affected instrumented
profile. The Linux compiler and Windows compiler-rt versions match; SDK/CRT
inputs and their digests are pinned together. Linux ABI compatibility comes from
the `release_wheel` audit. The destination-OS installed tests establish loader
paths and runtime behavior that cross-compilation alone cannot check.
