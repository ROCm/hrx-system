# IREE Remote HAL

The remote HAL lets an IREE process use a HAL device hosted by another process
or machine. A client process creates a normal `iree_hal_device_t`, records and
submits normal HAL work, and the remote client driver translates that work into
protocol messages. A server process owns one or more real local HAL devices,
accepts client sessions, reconstructs resources and queue submissions, and
executes the work on those local devices.

The core design is not "RPC for every C function". It is a distributed HAL
device built from three traffic classes:

```text
client HAL API
  |
  v
remote client driver
  |
  | iree/net session bootstrap and topology exchange
  |
  +-- control channel: resource/device RPCs and lifecycle messages
  +-- queue channel: frontier-ordered HAL queue operations
  +-- bulk channel: large payloads, files, maps, profiling data
  |
  v
remote server session
  |
  v
local HAL device(s)
```

This split is the main mental model for the code. The session layer owns
connection bootstrap, peer topology, endpoint provisioning, shutdown, and proxy
semaphores for remote frontier axes. The remote HAL layer owns the meaning of
control messages, queue commands, resource IDs, executable and command-buffer
serialization, bulk transfer engines, file registration, and profiling relay.

## Entry Points

Client-side public API:

- [`runtime/src/iree/hal/remote/client/api.h`](runtime/src/iree/hal/remote/client/api.h):
  remote client driver/device options, connection lifecycle, and server-side
  file open entry points.
- [`runtime/src/iree/hal/remote/client/registration/driver_module.c`](runtime/src/iree/hal/remote/client/registration/driver_module.c):
  driver registration for URI-style names such as `remote-tcp`, `remote-shm`,
  and `remote-rdma`.
- [`runtime/src/iree/hal/remote/client/device.c`](runtime/src/iree/hal/remote/client/device.c):
  client device creation, session connection, control RPC dispatch, queue/bulk
  endpoint publication, and error propagation.
- [`runtime/src/iree/hal/remote/client/queue.c`](runtime/src/iree/hal/remote/client/queue.c):
  HAL queue operation lowering to queue channel `COMMAND` frames and
  client-side semaphore signal handling.

Server-side public API:

- [`runtime/src/iree/hal/remote/server/api.h`](runtime/src/iree/hal/remote/server/api.h):
  remote server options, start/stop lifecycle, device exposure, topology, and
  file allow-list configuration.
- [`runtime/src/iree/hal/remote/server/server.c`](runtime/src/iree/hal/remote/server/server.c):
  listener setup, accepted session slots, local device retention, bootstrap
  device catalog construction, and server shutdown.
- [`runtime/src/iree/hal/remote/server/session.c`](runtime/src/iree/hal/remote/server/session.c):
  control message handlers, queue command replay on the wrapped device,
  provisional resource resolution, and ordered `ADVANCE` emission.

Protocol and channel contracts:

- [`runtime/src/iree/hal/remote/protocol/common.h`](runtime/src/iree/hal/remote/protocol/common.h):
  shared wire conventions, endpoint counts, resource ID encoding, buffer
  params, bindings, dispatch config, and memory heap descriptions.
- [`runtime/src/iree/hal/remote/protocol/control.h`](runtime/src/iree/hal/remote/protocol/control.h):
  device, buffer, semaphore, file, executable, command-buffer, lifecycle, and
  profiling control messages.
- [`runtime/src/iree/hal/remote/protocol/queue.h`](runtime/src/iree/hal/remote/protocol/queue.h):
  queue `COMMAND` payloads and server `ADVANCE` payloads.
- [`runtime/src/iree/hal/remote/protocol/commands.h`](runtime/src/iree/hal/remote/protocol/commands.h):
  serialized command-buffer command stream format.
- [`runtime/src/iree/hal/remote/protocol/profile.h`](runtime/src/iree/hal/remote/protocol/profile.h):
  bulk payload metadata for server-to-client profile sink callbacks.
- [`runtime/src/iree/net/session.h`](runtime/src/iree/net/session.h):
  net session bootstrap, topology exchange, proxy semaphore creation, control
  DATA, endpoint opening, and GOAWAY/error semantics.
- [`runtime/src/iree/net/channel/queue/queue_channel.h`](runtime/src/iree/net/channel/queue/queue_channel.h):
  ordered queue traffic with frontiers.
- [`runtime/src/iree/net/channel/bulk/bulk_channel.h`](runtime/src/iree/net/channel/bulk/bulk_channel.h):
  transfer-ID keyed bulk START/DATA/COMPLETE/ABORT/CREDIT framing.
- [`runtime/src/iree/net/carrier.h`](runtime/src/iree/net/carrier.h):
  transport-agnostic carrier interface, capabilities, direct operations, remote
  memory handles, and file handle transfer.

Tooling and tests:

- [`runtime/src/iree/tools/iree-serve-device/iree-serve-device-main.c`](runtime/src/iree/tools/iree-serve-device/iree-serve-device-main.c):
  command-line server that exposes a local device over TCP, shared memory, or
  RDMA.
- [`runtime/src/iree/tools/iree-serve-device/transport.c`](runtime/src/iree/tools/iree-serve-device/transport.c):
  server bind URI parsing and transport factory construction.
- [`runtime/src/iree/tools/iree-remote-check/iree-remote-check-main.c`](runtime/src/iree/tools/iree-remote-check/iree-remote-check-main.c):
  remote-only client that selects, uploads, dispatches, and verifies an
  embedded AMDGPU or Vulkan executable without a compiler or VM.
- [`runtime/src/iree/hal/remote/cts/loopback_adapter.cc`](runtime/src/iree/hal/remote/cts/loopback_adapter.cc):
  HAL CTS adapter that wraps ordinary HAL CTS backends behind a loopback remote
  server/client pair.
- [`runtime/src/iree/net/carrier/cts`](runtime/src/iree/net/carrier/cts):
  reusable carrier CTS used by loopback, TCP, shared-memory, and RDMA carriers.
- [`runtime/src/iree/net/carrier/rdma/README.md`](runtime/src/iree/net/carrier/rdma/README.md):
  Linux RDMA and GPU dma-buf registration notes.

## Running It

### Deployment Boundary

Remote HAL is designed for devices served inside a trusted private network or
compute fabric. The protocol currently provides no peer authentication, client
authorization, confidentiality, or integrity protection; admission and
transport protection belong to the surrounding network. Any peer admitted by
that boundary can submit HAL work to the exposed device. Direct exposure to an
untrusted network is outside the deployment contract.

The server defaults to `tcp://0.0.0.0:5000` so clients on the trusted network
can connect without extra server configuration. Bind a specific interface when
the host participates in networks with different trust domains. Remote HAL
deliberately has no in-protocol cryptography: TCP deployments that need peer
authentication or encryption must provide it at the network boundary, such as
with an isolated compute fabric, IPsec, or WireGuard.

### AMDGPU Quick Start

Run the server and client on machines in the same trusted network. The explicit
Bazel driver selection makes the recipe independent of an existing local
configuration:

```bash
# Server: expose AMDGPU device 0 on all IPv4 interfaces (default).
iree-bazel-run \
  --//runtime/config/hal:drivers=amdgpu \
  //tools:iree-serve-device -- \
  --device=amdgpu://0

# Client: upload and dispatch a matching embedded HSACO.
iree-bazel-run \
  --//runtime/config/hal:drivers=remote \
  --//runtime/config/hal:executable_artifacts=amdgpu \
  //tools:iree-remote-check -- \
  --device=remote-tcp://server-host:5000
```

After the listener starts, `iree-serve-device` prints its actual bound address
and a client device flag. A wildcard listener is reported with a
`<server-host>` placeholder because the server cannot choose which hostname or
interface is reachable from a particular client. Replace that placeholder with
the server's reachable hostname or address while preserving the printed port.
An assigned port is included when `--bind=...:0` is used.

`iree-remote-check` is intentionally independent of IREE compiler and VM
artifacts. Its client binary links the remote HAL implementation and executable
target-selection utilities, but no AMDGPU, HIP, or Vulkan HAL driver. The
server is the only process that loads the native driver. The check reads the
immutable executable targets delivered during session bootstrap, selects a
compatible embedded artifact, performs real host-to-device and device-to-host
transfers, submits a queue dispatch, waits on its HAL semaphore, and verifies
the exact result.

CMake builds select the same client payloads with
`IREE_HAL_EXECUTABLE_ARTIFACT_AMDGPU=ON` or
`IREE_HAL_EXECUTABLE_ARTIFACT_VULKAN=ON`. These options embed an executable in
supporting tools without enabling the corresponding native HAL driver; server
driver selection remains independent through `IREE_HAL_DRIVER_AMDGPU` or
`IREE_HAL_DRIVER_VULKAN`.

The AMDGPU artifacts follow
`//runtime/src/iree/hal/drivers/amdgpu:targets`. The default selector set covers
the normal generic gfx9, gfx10, gfx11, and gfx12 families. If the connected
device reports a target not covered by the client binary, the check prints the
server's target table and fails before upload; rebuild it with an exact or
generic selector matching one of those rows.

### Vulkan Quick Start

The same client also embeds a separately assembled Vulkan 1.3 BDA SPIR-V
module. Only the server links the Vulkan HAL driver:

```bash
# Server: expose the default Vulkan device.
iree-bazel-run \
  --//runtime/config/hal:drivers=vulkan \
  //tools:iree-serve-device -- \
  --device=vulkan://

# Client: select and dispatch the embedded SPIR-V module remotely.
iree-bazel-run \
  --//runtime/config/hal:drivers=remote \
  --//runtime/config/hal:executable_artifacts=vulkan \
  //tools:iree-remote-check -- \
  --device=remote-tcp://server-host:5000
```

### Same-Host Development

The wildcard TCP default also accepts same-host clients at `127.0.0.1:5000`.
An explicit loopback bind restricts the listener to same-host development:

```bash
iree-serve-device \
  --device=amdgpu://0 \
  --bind=tcp://127.0.0.1:5000

iree-remote-check \
  --device=remote-tcp://127.0.0.1:5000
```

The shared-memory transport provides a same-host IPC path whose endpoint access
is governed by operating-system permissions.

### Other Transports

Clients use a remote driver name whose suffix selects the transport factory:

```bash
iree-remote-check \
  --device=remote-shm:///dev/shm/iree-gpu
iree-remote-check \
  --device='remote-rdma://192.0.2.10:7471?rdma=true'
```

The client registration module enables TCP by default. Shared-memory and RDMA
client transports are opt-in in both build systems; RDMA is Linux-only:

```bash
# Bazel: the string-list flag replaces the default list.
--//runtime/config/net:transports=tcp,shm,rdma

# CMake
-DIREE_NET_TRANSPORT_SHM=ON -DIREE_NET_TRANSPORT_RDMA=ON
```

`remote-rdma` and `rdma://` select the RDMA carrier. The `rdma=true` client
parameter and `iree-serve-device --rdma` flag require RDMA-capable bulk
transfer during session bootstrap; they are not downgrade preferences.

### Server Files

Server-side files are not implicitly exposed to clients. `iree-serve-device`
builds an explicit logical namespace from repeated allow-list flags:

```bash
iree-serve-device \
  --device=amdgpu://0 \
  --remote_file_allow=weights=/srv/models/weights.bin \
  --remote_file_allow_write=scratch=/srv/iree-scratch
```

The client then opens logical names through
`iree_hal_remote_client_device_open_file`. The open waits for the server to
validate the logical path and return the file extent and granted access; data
movement remains asynchronous and queue ordered. The allow-list restricts the
logical file namespace visible through `FILE_OPEN`; it does not authenticate
clients or restrict their use of the exposed HAL device.

## Protocol Compatibility

The protocol is experimental and does not yet provide backward compatibility
between runtime revisions. Session bootstrap rejects mismatched protocol
versions, and the remote HAL device-catalog parser rejects unsupported catalog
versions instead of interpreting them as an older layout.

Wire fields are defined as little-endian fixed-width values. The current
implementation accesses naturally aligned wire structs in host representation,
so remote HAL builds require little-endian hosts. Cross-endian communication
requires explicit field encoding and decoding at every protocol boundary; the
build fails rather than producing a peer that silently misinterprets traffic.

## Lifecycle

The client-side lifecycle starts with a remote HAL driver and device:

1. The driver is created by transport-specific registration, for example
   `remote-tcp` creating a TCP transport factory.
2. `iree_hal_remote_client_device_create` allocates the HAL proxy device,
   receive pool references, frontier tracker, allocator, queue slab provider,
   and bounded bulk-transfer state.
3. `iree_hal_remote_client_device_connect` starts an asynchronous
   `iree_net_session_connect`.
4. Session bootstrap exchanges topology. Each side learns the peer's frontier
   axes and registers proxy semaphores in its frontier tracker.
5. The HAL client opens the queue endpoint and then the bulk endpoint. The
   device is published as connected only after both endpoints are ready.
6. Normal HAL calls use the standard HAL vtables. Unsupported pre-connect
   operations fail with `FAILED_PRECONDITION`.

The server-side lifecycle mirrors this:

1. A tool or embedding application creates one or more local HAL devices.
2. The caller constructs an `iree_net_session_topology_t` describing the local
   queue axes exposed to clients.
3. `iree_hal_remote_server_create` retains the local devices, copies topology,
   serializes their immutable device specs for bootstrap, and allocates bounded
   session slots.
4. `iree_hal_remote_server_start` creates a listener from the transport factory.
5. Accepted connections become `iree_net_session_accept` sessions. After
   bootstrap, each session opens queue and bulk endpoints.
6. `iree_hal_remote_server_stop` sends GOAWAY, stops the listener, detaches
   channels, drains sessions and bulk transfers, and reports stopped only when
   all session slots are closed.

The net session owns graceful shutdown semantics. The remote HAL session code
owns releasing HAL resources, failing pending queue signals, sending terminal
`ADVANCE` frames where possible, and ensuring in-flight bulk transfers cannot
outlive the channels and resources they reference.

Executable loading uses the target table in the bootstrapped device spec.
Clients select a target locally and send its stable table ordinal with the
artifact, queue affinity, load flags, and specialization constants. The server
resolves the ordinal against its own device spec and calls
`iree_hal_device_load_executable` directly. Capability discovery therefore does
not add a synchronous format-query round trip, and target-family compatibility
remains owned by the local HAL backend.

## Ordering And Semaphores

Queue ordering is represented with async frontiers rather than a single remote
FIFO. Each queue `COMMAND` frame can carry a wait frontier and a signal frontier.
The client assigns monotonically increasing submission epochs on a client queue
axis, records the mapping from HAL semaphore values to `(axis, epoch)`, and
registers frontier waiters that signal local proxy semaphores when the server
advances that epoch.

On the server:

- Queue `COMMAND` frames are decoded by
  `runtime/src/iree/hal/remote/server/session.c`.
- Wait frontiers are translated into local HAL semaphore waits using the
  frontier tracker and per-session epoch-to-semaphore mapping.
- The command is submitted to the wrapped local HAL device.
- Completion is observed through local HAL semaphores.
- The server sends an ordered `ADVANCE` frame carrying the completed signal
  frontier, any provisional resource resolutions, and serialized status data
  when the operation failed.

On the client, an `ADVANCE` frame advances or fails the local frontier tracker.
Client-visible semaphore wait failures therefore surface through the ordinary
HAL semaphore path instead of a side channel.

Some wait semaphores cannot be encoded immediately as a remote frontier. A host
signal or cross-device dependency may not yet have a recorded epoch. The client
queue path gates those submissions locally until the first unsatisfied semaphore
is signaled, then re-enters queue submission with a concrete frontier.

## Resources

Remote resources are session-scoped 64-bit IDs defined in
[`runtime/src/iree/hal/remote/protocol/common.h`](runtime/src/iree/hal/remote/protocol/common.h).
The ID encodes resource type, flags, generation, server proactor index, and
resource-table slot. Client-created resources begin as provisional IDs. The
server resolves them to canonical IDs and piggybacks the mapping on queue
`ADVANCE` frames or control responses.

This provisional path is important for latency: queue-ordered operations can
refer to resources before the client has synchronously waited for a control
round trip. When a queue command reaches the server before a referenced
provisional ID is resolved, the server parks the command on that provisional
entry and resumes it once resolution succeeds or fails.

Server-side resource lifetime is controlled by per-session resource tables in
[`runtime/src/iree/hal/remote/server/resource_table.h`](runtime/src/iree/hal/remote/server/resource_table.h).
Client-side releases are batched and tagged with the greatest submission epoch
that could still mention the resource. The server releases table entries only
after it has observed queue traffic through that epoch, which prevents a
release message from racing ahead of earlier submitted work.

## Control Channel

The control channel carries request/response messages and notifications for
operations that are not naturally queue ordered:

- Device trim; immutable device facts arrive in the bootstrap catalog.
- Semaphore creation and synchronous semaphore query/wait/signal.
- Executable upload and executable metadata queries.
- Command-buffer upload for reusable command buffers.
- File open/register/list/close.
- Buffer allocation, import, map, unmap, and virtual/physical memory control.
- Resource release batches.
- Profiling begin/flush/end.
- Device lost, resource error, and memory pressure notifications.

Messages use a fixed envelope and optional response prefix. Control messages
that create resources increment a control epoch; queue operations that depend
on those resources include the corresponding control frontier so the server
does not execute queue work before the resource exists.

Large control payloads avoid unnecessary copies by using
`iree_net_session_send_control_data`; small stack-backed messages use the copy
helper. Command-buffer upload and executable upload support both inline data
and bulk references, selected by the upload flags in
[`runtime/src/iree/hal/remote/protocol/control.h`](runtime/src/iree/hal/remote/protocol/control.h).

## Queue Channel

The queue channel carries one HAL queue operation per `COMMAND` frame. Current
operation families include:

- Queue-ordered allocation and deallocation.
- Buffer fill/update/copy.
- Server-side file read/write.
- Client-local file read/write bridged through bulk transfers.
- Dispatch.
- Command-buffer execute.
- Queue flush.
- Resource release batches.
- Extension operations.

Immediate dispatches and one-shot command buffers are optimized for the common
compiler-generated path. A one-shot command buffer can stream serialized command
pages during recording and append the final fragment to
`COMMAND_BUFFER_EXECUTE`. Reusable command buffers are uploaded over the control
channel, assigned a resource ID, and later referenced by queue execution.

Queue frames carry frontiers; command-buffer command streams carry the command
payload. The command stream format is self-describing: each command has an
8-byte header with type and padded length, followed by command-specific data.

## Bulk Channel

The bulk channel is for payloads that should not block control or queue traffic
behind large byte streams. It is keyed by 64-bit transfer IDs and uses
START/DATA/COMPLETE/ABORT/CREDIT frames. DATA chunks may arrive out of order;
the embedding transfer engines reconstruct or stage them.

Current remote HAL bulk users:

- Client file reads: client-local file bytes are uploaded to a server buffer.
- Client file writes: server buffer bytes are downloaded into a client-local
  file, and queue completion waits for client acknowledgement.
- Buffer map/unmap: mapped bytes are transferred between client host memory and
  server buffers.
- Profiling relay: server-side profile sink callbacks are sent to the
  client-owned profile sink.

The generic bulk channel only validates framing and manages bounded send
contexts and peer receive credits. The HAL-specific engines under
[`runtime/src/iree/hal/remote/client`](runtime/src/iree/hal/remote/client) and
[`runtime/src/iree/hal/remote/server`](runtime/src/iree/hal/remote/server) own
transfer semantics, staging pools, file views, profile metadata, cancellation,
and completion coupling back to queue/control responses.

## Transports

Remote HAL traffic runs on `iree/net` transports. A transport factory creates
connections and listeners. Each connection exposes enough endpoints for one
session control endpoint plus the remote HAL queue and bulk endpoints; this is
captured by `IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT`.

Available paths in this branch:

- TCP: default client registration and default `iree-serve-device` transport.
  It is the portable path for cross-machine development and review.
- Shared memory: optional registration and `iree-serve-device` support for
  local IPC and low-noise tests.
- Loopback: test-only in-process transport used by the remote HAL CTS adapter.
- RDMA: carrier and CTS infrastructure under
  [`runtime/src/iree/net/carrier/rdma`](runtime/src/iree/net/carrier/rdma).
  The carrier uses libibverbs and librdmacm through dynamic loaders, so
  binaries can run on systems without RDMA installed. Linux RDMA-over-Ethernet
  development and GPU-direct prerequisites live there, including the dma-buf
  registration probe described in
  [`runtime/src/iree/net/carrier/rdma/README.md`](runtime/src/iree/net/carrier/rdma/README.md).

The remote HAL client/server APIs treat `rdma=true` as a requirement, not a
preference. Device or server creation validates that the selected transport
factory advertises `IREE_NET_TRANSPORT_CAPABILITY_RDMA`; session bootstrap then
advertises and requires `IREE_NET_BOOTSTRAP_CAPABILITY_RDMA` so peer mismatch
fails instead of silently falling back to message bulk transfer. Review of RDMA
behavior should start below the HAL layer: transport factory setup, carrier
capabilities, registered regions, direct write/read budget, memory-window
lifetime, and the carrier CTS.

## First Connection Checks

The server readiness block is the source of truth for the listener and client
URI. It is printed only after the listener has started successfully.

- Connection refused: confirm that the trusted-network boundary admits TCP port
  5000 and that the client URI names an interface or hostname reachable from
  the client. An explicitly configured `127.0.0.1` or `[::1]` listener only
  accepts same-host clients.
- Wildcard listener: `0.0.0.0` and `[::]` are bind addresses, not client
  destinations. Replace the printed `<server-host>` placeholder with a
  reachable hostname or interface address while preserving the printed port.
- `transport '...' is not compiled`: TCP is the default. Add optional Bazel
  transports with `--//runtime/config/net:transports=tcp,shm,rdma`, or enable
  `IREE_NET_TRANSPORT_SHM`/`IREE_NET_TRANSPORT_RDMA` in CMake.
- Server starts but bootstrap fails: use client and server binaries from the
  same revision. The experimental protocol deliberately rejects version
  mismatches.
- Executable selection fails after connection: the client has no artifact for
  a target advertised by the served device. For `iree-remote-check`, rebuild
  with `--//runtime/config/hal:executable_artifacts=amdgpu` and an AMDGPU
  selector matching one of the printed target rows, or use
  `--//runtime/config/hal:executable_artifacts=vulkan` with a server advertising
  `spirv / vulkan1.3+bda`.

## Debugging Map

Connection and endpoint setup:

- Client cannot connect: start with
  `iree_hal_remote_client_device_connect` in
  `runtime/src/iree/hal/remote/client/device.c`, then follow
  `iree_net_session_connect` in `runtime/src/iree/net/session.h` and the
  selected transport factory.
- Server accepts but client never becomes ready: inspect session bootstrap,
  topology exchange, and endpoint opening on both sides. The remote HAL device
  is not connected until queue and bulk endpoints have both been published.
- Dynamic port or bind confusion: `iree_hal_remote_server_query_bound_address`
  reaches the listener through the transport factory.

Queue progress:

- Client wait hangs: inspect whether the signal semaphore has an epoch mapping
  in `runtime/src/iree/hal/remote/client/semaphore.*`, whether
  `runtime/src/iree/hal/remote/client/queue.c` registered a frontier waiter,
  whether the server submitted the local command, and whether an `ADVANCE`
  frame was sent.
- Server receives a command before a provisional resource is resolved: inspect
  the provisional map and pending queue command paths in
  `runtime/src/iree/hal/remote/server/session.c`.
- A queue operation fails on the server but the client sees only a semaphore
  failure: inspect serialized status in the `ADVANCE` payload and
  `runtime/src/iree/net/status_wire.*`.

Resources:

- Resource not found on the server: compare provisional and resolved IDs,
  generation bits, and resource table slot lifetime.
- Use-after-release suspicion: inspect resource release batches and the
  observed submission window in `runtime/src/iree/hal/remote/server/session.c`.

Bulk transfers:

- Upload/download stalls: inspect peer DATA chunk credit, send budget,
  transfer table entries, and whether a START frame was acknowledged by local
  transfer state.
- Large payload corrupt or truncated: inspect chunk offset, sequence, total
  size, and the transfer-specific staging/reconstruction path.
- File queue operations fail late: remember that logical open can be
  provisional on the client and final path/access validation happens on the
  server.

Transport and RDMA:

- TCP/SHM failures usually start in the selected transport factory and carrier.
- RDMA availability is runtime capability, not just build configuration. Check
  rdma-core dynamic loading, active port/GID selection, `/dev/infiniband/*`
  permissions, `RLIMIT_MEMLOCK`, SoftRoCE versus hardware provider behavior,
  and dma-buf registration when GPU memory is involved.

## Test And Presubmit Surface

Useful focused checks:

```bash
iree-bazel-test --config=presubmit \
  //runtime/src/iree/hal/remote/... \
  //runtime/src/iree/net/...
```

Fresh execution of the remote HAL and net tests:

```bash
iree-bazel-test --config=presubmit --test_output=errors \
  --nocache_test_results \
  //runtime/src/iree/hal/remote/... \
  //runtime/src/iree/net/...
```

Full Bazel presubmit for this worktree:

```bash
python dev.py bazel presubmit
```

RDMA-focused build and CTS coverage:

```bash
iree-bazel-test --//runtime/config/net:transports=rdma \
  //runtime/src/iree/net/carrier/rdma:all \
  //runtime/src/iree/net/carrier/rdma/cts:all

iree-cmake-configure \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_BENCHMARKS=ON \
  -DIREE_NET_TRANSPORT_RDMA=ON

iree-cmake-build \
  iree_net_carrier_rdma_factory \
  iree_tools_iree-serve-device_transport_test \
  iree_hal_remote_client_registration_registration
```

The remote HAL CTS adapter composes with existing HAL CTS backends by wrapping
each concrete backend behind an in-process loopback remote server/client pair.
Carrier CTS lives in `runtime/src/iree/net/carrier/cts` and is reused by
loopback, TCP, shared memory, and RDMA backends.

RDMA live probes may skip on machines without the required hardware, kernel,
driver, permission, or library support. Once the relevant capabilities are
present, those tests should fail loudly on ordinary provider errors rather than
silently passing.

## Review Pressure

The highest-value review questions are architectural:

- Does every cross-machine ordering edge pass through a frontier, and is the
  producing side responsible for advancing or failing that frontier exactly
  once?
- Does every server-side resource have a single session-local owner, a stable
  ID resolution path, and a release condition tied to observed queue progress?
- Can every large payload move on bulk or a zero-copy control path without
  blocking latency-sensitive control or queue traffic?
- Are failures reported through the same HAL surfaces a local device would use,
  especially semaphore failures for queue-ordered work?
- Does transport-specific complexity stay under `iree/net` carrier/factory
  capability checks instead of leaking into the HAL protocol?
- Can a new developer reproduce the important behaviors with loopback/TCP/SHM
  tests before needing real RDMA hardware?
