# HRX CTS (Conformance Test Suite)

## Test Coverage

### Tested

| Category | Test file | What's tested |
|----------|-----------|---------------|
| `lifecycle` | lifecycle_test.cpp | GPU/CPU init/shutdown, version query, double-init error |
| `status` | status_test.cpp | Status creation, codes, string conversion, ignore |
| `device` | device_test.cpp | Property queries (name, arch, total_memory, compute_units, warp_size, max_shared_memory, clock_rate, pci_bus_id), device type, synchronize |
| `allocator` | allocator_test.cpp | device_allocator() borrowed ref, allocate_buffer with params, import_buffer from host ptr, retain/release |
| `memory` | memory_test.cpp | Stream-ordered buffer_allocate, map/unmap, read-back, zero-size rejection, buffer_get_size |
| `transfer` | transfer_test.cpp | synchronous_h2d, synchronous_d2h, h2d+d2h roundtrip, out-of-range rejection |
| `semaphore` | semaphore_test.cpp | Create, query, signal, wait, retain/release |
| `stream` | stream_test.cpp | Create, flush, synchronize, query, get_semaphore, timeline position, wait_on |
| `stream_ops` | stream_ops_test.cpp | fill_buffer, copy_buffer, update_buffer with verification |
| `refcount` | refcount_test.cpp | Retain/release on device, semaphore, stream, buffer |
| `virtual_memory` | virtual_memory_test.cpp | Query support (expected false on local-task), reserve failure on unsupported device |

### Not Yet Tested (needs kernels or infrastructure)

| Category | Why |
|----------|-----|
| `queue_dispatch` / `stream_dispatch` | Needs compiled HSACO or VMFB kernel binary |
| `executable_load` / `export_count` / `export_info` | Needs kernel binary to load |
| `executable_lookup_global` | Needs executable with global variables |
| `queue_host_call` | Needs dispatch ordering infrastructure |
| `stream_execution_barrier` | Not yet implemented in libhrx |
| `multithread` | Concurrent stream/semaphore stress tests |
| `multidevice` | Cross-device copy, cross-device semaphore (needs multi-GPU) |
| `fork` | fork() safety for DataLoader scenarios |
| `virtual_memory` (full) | map/unmap/protect needs allocator with VM support (not local-task) |

### Running

```bash
# Build with CTS enabled
cmake -S . -B build/hrx-runtime ... -DLIBHRX_BUILD_CTS=ON -DBUILD_TESTING=ON

# Run all CTS tests
ctest --test-dir build/hrx-runtime --output-on-failure

# Run a specific category
build/hrx-runtime/cts/hrx_cts_allocator
build/hrx-runtime/cts/hrx_cts_transfer
```

### Remote HAL Device

The CTS launcher accepts the same remote HAL URI shape as `hrx-info`. A local
smoke run can serve an AMDGPU device over TCP and run a focused CTS binary
through the remote HAL client:

```bash
# Terminal 1: serve a local device.
iree-bazel-run \
  --//runtime/config/hal:drivers=amdgpu,remote \
  //runtime/src/iree/tools/iree-serve-device:iree-serve-device -- \
  --device=amdgpu://0 \
  --bind=tcp://127.0.0.1:5000

# Terminal 2: run the CTS through the remote HAL client.
iree-bazel-test \
  --//runtime/config/hal:drivers=remote \
  //libhrx/cts:hrx_cts_transfer \
  --test_arg=--hrx-device=remote-tcp://127.0.0.1:5000
```

The launcher maps `remote-*://...` specs to `HRX_GPU_DRIVER` before calling
`hrx_gpu_initialize()`, so the tests exercise the same HRX device lifecycle and
stream/buffer API as a preloaded HIP application.
