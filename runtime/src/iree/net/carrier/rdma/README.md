# IREE RDMA Carrier Notes

The RDMA carrier uses libibverbs and librdmacm through dynamic loaders so IREE
binaries can still run on systems without RDMA installed. Runtime availability
is therefore a capability decision, not a link-time requirement.

## Build Surface

RDMA is an opt-in runtime network transport. Default builds keep
`//third_party:rdma_core_headers` on an empty facade target so broad
`bazel build //...` and `bazel test //...` coverage does not need RDMA headers.
Enabling the RDMA transport switches that facade to the pinned rdma-core header
repository in Bazel and to the CMake adapter under
`build_tools/third_party/rdma-core-headers`.

```bash
# Bazel: RDMA-only transport build.
bazel build --//runtime/config/net:transports=rdma \
  //runtime/src/iree/net/carrier/rdma:factory

# Bazel: TCP, shared memory, and RDMA remote-HAL client registrations.
bazel build --//runtime/config/net:transports=tcp,shm,rdma \
  //runtime/src/iree/hal/remote/client/registration:registration

# CMake: RDMA is Linux-only.
cmake -S . -B /tmp/iree-rdma -GNinja -DIREE_NET_TRANSPORT_RDMA=ON
cmake --build /tmp/iree-rdma --target iree_net_carrier_rdma_factory
```

The carrier still discovers libibverbs and librdmacm at runtime through the
dynamic loaders in this directory. A successful RDMA-enabled compile proves the
headers and code shape, while live execution depends on the local kernel,
provider, permissions, and memory-registration capabilities.

## Remote HAL Path

The remote HAL uses RDMA through ordinary transport selection plus an explicit
bulk-transfer requirement:

```bash
iree-serve-device --device=hip://0 --bind=rdma://192.0.2.10:7471 --rdma

iree-run-module --device='remote-rdma://192.0.2.10:7471?rdma=true' \
  --module=model.vmfb
```

`remote-rdma` selects the RDMA client transport factory and `rdma://` selects
the RDMA listener factory. The `rdma=true` client option and server `--rdma`
flag require both peers to negotiate `IREE_NET_BOOTSTRAP_CAPABILITY_RDMA`.
Mismatched peers fail during device/server setup or session bootstrap instead
of silently using message bulk transfer.

## SEND Data Path

Every native verbs scatter-gather entry must reference RDMA-registered memory.
The carrier prepares each SEND by preserving caller spans that are already
registered and packing only unregistered bytes into one registered staging
buffer. Output spans referencing the staging buffer are interleaved with the
borrowed registered spans in their original wire order.

This matters for the normal framed channel shape: a small sender-owned frame
header followed by a large registered payload. The header is copied into
transport-owned memory, while the payload address and registration remain
unchanged all the way to `ibv_post_send`. The staging-buffer size constrains the
sum of unregistered bytes, not the total message size. Fully unregistered sends
retain the copy path and coalesce adjacent source spans into one verbs entry.

The carrier retains the staging lease through native completion or
cancellation. Callers retain registered source spans until the ordinary send
completion. `IREE_NET_SEND_FLAG_ZERO_COPY` remains strict: a send carrying that
flag fails before admission if any non-empty source span is unregistered.
`send_payload_test` exercises a framed 1 MiB registered payload, interleaved
registered and unregistered spans, the fully copied path, and strict zero-copy
rejection without requiring RDMA hardware.

## Verification Ladder

The low-friction checks prove the opt-in dependency surface and portable carrier
contracts:

```bash
bazel build //third_party:rdma_core_headers

bazel test --//runtime/config/net:transports=rdma \
  //runtime/src/iree/net/carrier/rdma:all \
  //runtime/src/iree/net/carrier/rdma/cts:all
```

The first command resolves to the empty facade under the default transport
configuration. The RDMA-enabled test command builds the real header dependency,
factory, unit tests, and carrier CTS. CTS cases may skip when the machine lacks
RDMA devices or provider support; unexpected provider errors after capability
discovery are failures.

## GPU Dma-Buf Registration

`dmabuf_registration_test` validates the prerequisite for GPU-direct RDMA bulk
transfers with AMDGPU memory:

1. Allocate memory from an AMDGPU HSA global memory pool.
2. Export that allocation with `hsa_amd_portable_export_dmabuf`.
3. Open an RDMA device and protection domain through libibverbs.
4. Register the exported dma-buf fd with `ibv_reg_dmabuf_mr`.

This is intentionally a live capability test. It skips when the machine lacks
HSA, an AMDGPU agent, libibverbs, RDMA devices, dma-buf export support, or the
optional `ibv_reg_dmabuf_mr` symbol. Once all those capabilities are present,
ordinary `ibv_reg_dmabuf_mr` failures are reported as test failures.

Useful command:

```bash
iree-bazel-test \
  //runtime/src/iree/net/carrier/rdma:dmabuf_registration_test \
  --test_tag_filters= \
  --cache_test_results=no \
  --test_output=all
```

The empty `--test_tag_filters=` override is needed when using
`iree-bazel-test`, because the wrapper's default filter excludes `nodocker`
tests and this probe is tagged as a live AMDGPU/RDMA test.

Expected system requirements:

- Linux with dma-buf memory registration support in the kernel and rdma-core
  stack. The libibverbs loader checks for the optional `ibv_reg_dmabuf_mr`
  symbol.
- An RDMA provider whose device supports dma-buf MR registration. SoftRoCE is
  useful for host-memory correctness work, but GPU dma-buf registration is a
  hardware/provider capability.
- ROCR/KFD and the AMDGPU kernel driver must support
  `hsa_amd_portable_export_dmabuf` for the selected GPU allocation.
- The user must have access to the relevant `/dev/kfd`, `/dev/dri/renderD*`,
  and `/dev/infiniband/*` nodes.
- `RLIMIT_MEMLOCK` must be high enough for RDMA memory registration.
