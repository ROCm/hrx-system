# IREE RDMA Carrier Notes

The RDMA carrier uses libibverbs and librdmacm through dynamic loaders so IREE
binaries can still run on systems without RDMA installed. Runtime availability
is therefore a capability decision, not a link-time requirement.

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
