# Local NativeLink Quickstart

NativeLink can provide one shared Bazel action cache and execution limit across
multiple local worktrees. This is an opt-in developer configuration: ordinary
builds remain unchanged until `--config=nativelink` is selected. Same-machine
remote execution adds hashing, RPC, and artifact-materialization work, so its
payoff comes from cross-worktree cache hits and globally bounded concurrency,
not from making one cold build faster.

`../../.bazelrc` imports `nativelink.bazelrc`, which only defines the named
configuration. `local.json5` runs a persistent cache, scheduler, and local
worker in one process on two loopback listeners.

## Install NativeLink

The checked-in configuration targets NativeLink 1.6.5 on Linux x86-64. Install
the pinned release from the
[official NativeLink release](https://github.com/TraceMachina/nativelink/releases/tag/v1.6.5):

```bash
NATIVELINK_VERSION=1.6.5
NATIVELINK_ARCHIVE="nativelink-${NATIVELINK_VERSION}-x86_64-unknown-linux-musl.tar.gz"
NATIVELINK_DOWNLOAD_DIRECTORY="$(mktemp -d)"

curl -fL \
  --output "${NATIVELINK_DOWNLOAD_DIRECTORY}/${NATIVELINK_ARCHIVE}" \
  "https://github.com/TraceMachina/nativelink/releases/download/v${NATIVELINK_VERSION}/${NATIVELINK_ARCHIVE}"
printf '%s  %s\n' \
  093d2e0baac5311444762449b703a45b738d570065d2e17c2db1fcf39d8ad051 \
  "${NATIVELINK_DOWNLOAD_DIRECTORY}/${NATIVELINK_ARCHIVE}" | \
  sha256sum --check
tar -xzf "${NATIVELINK_DOWNLOAD_DIRECTORY}/${NATIVELINK_ARCHIVE}" \
  -C "${NATIVELINK_DOWNLOAD_DIRECTORY}"
install -Dm0755 "${NATIVELINK_DOWNLOAD_DIRECTORY}/nativelink" \
  "${HOME}/.local/bin/nativelink"
"${HOME}/.local/bin/nativelink" --version
```

## Start the Local Service

From the repository root, choose a persistent state directory outside the
checkout, create its private stores, and start NativeLink in one terminal:

```bash
export NATIVELINK_STATE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/iree-nativelink"
install -d -m0700 \
  "${NATIVELINK_STATE_ROOT}" \
  "${NATIVELINK_STATE_ROOT}/cas" \
  "${NATIVELINK_STATE_ROOT}/cas/content" \
  "${NATIVELINK_STATE_ROOT}/cas/temp" \
  "${NATIVELINK_STATE_ROOT}/ac" \
  "${NATIVELINK_STATE_ROOT}/ac/content" \
  "${NATIVELINK_STATE_ROOT}/ac/temp" \
  "${NATIVELINK_STATE_ROOT}/worker" \
  "${NATIVELINK_STATE_ROOT}/worker/cas" \
  "${NATIVELINK_STATE_ROOT}/worker/cas/content" \
  "${NATIVELINK_STATE_ROOT}/worker/cas/temp" \
  "${NATIVELINK_STATE_ROOT}/worker/actions"
"${HOME}/.local/bin/nativelink" build_tools/nativelink/local.json5
```

## Opt a Build In

From another terminal, confirm that the client and worker APIs are both
listening:

```bash
ss -ltn '( sport = :50051 or sport = :50061 )'
```

Two `LISTEN` rows mean NativeLink is ready. Opt an ordinary wrapper command
into both the shared cache and local worker:

```bash
build_tools/bin/iree-bazel-build \
  --config=nativelink \
  //runtime/src/iree/base:base
```

The first cache-cold build should report `remote` processes. Run the same
command from another configured worktree to verify `remote cache hit` entries;
different worktrees may still execute actions whose source or configuration
inputs differ. The named config disables any separately configured Bazel disk
cache only while selected, avoiding duplicate local reads and writes without
deleting either cache. Requested outputs are downloaded; intermediate outputs
remain in NativeLink unless the invocation adds
`--remote_download_outputs=all`.

## Capacity and Trust Boundary

The default limits are 48 GB for the persistent CAS, 2 GB for the action cache,
12 GB for the worker's fast CAS, and 16 concurrent actions. Override them
before launch with `NATIVELINK_CAS_MAX_BYTES`, `NATIVELINK_AC_MAX_BYTES`,
`NATIVELINK_WORKER_CACHE_MAX_BYTES`, and
`NATIVELINK_MAX_INFLIGHT_TASKS`.

Both APIs bind to loopback, but actions run directly on the host with the
developer's authority so configured compilers and SDKs remain visible. This is
appropriate only for trusted local checkouts; it is not a network deployment
or an isolation boundary. Stop NativeLink only after builds are idle. Stopping
the process is not a benchmark admission hold, and this quickstart does not
claim restart-safe drain or queue control.
