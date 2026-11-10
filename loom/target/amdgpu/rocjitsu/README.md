# Loom on simulated AMD GPUs

The concrete `loom-test_<profile>` and `loom-serve_<profile>` targets run the
ordinary Loom tools with a complete multi-GPU rocjitsu topology. The selected
topology replaces native discovery. The library and topology are bound by the
invocation; ordinary workload flags and device selection remain available.
The JIT selects its target from the selected device.

Supply the separately built runtime described in the
[runtime package guide](../../../../build_tools/third_party/rocjitsu/README.md):

```sh
iree-bazel-run --config=asan \
  --repo_env=IREE_ROCJITSU_RUNTIME_PATH=/absolute/path/to/package \
  //loom/target/amdgpu/rocjitsu:loom-test_gfx942 -- \
  --device=amdgpu://0 /absolute/path/to/workload.loom
```

Each catalog row describes one VM containing two or four GPUs. Device indices
select members of that topology; they are not separate simulator instances.
Custom topologies can be supplied to the ordinary tools with
`--amd_rocjitsu_config=<json-file>` and an optional
`--amd_rocjitsu_library=<shared-library>` outside the fixed Bazel invocations.

The `buffer_bulk_test` suite qualifies canonical buffer kernels through the
ordinary HAL and JIT. All execution leaves share `buffer_bulk_module`, including
its imported and linked artifacts. Select the test matrix independently:

```sh
iree-bazel-test --config=asan --config=rocjitsu \
  --//build_tools/bazel:rocjitsu_profiles=gfx942,gfx950,gfx1100,gfx1151,gfx1201,gfx1250 \
  --repo_env=IREE_ROCJITSU_RUNTIME_PATH=/absolute/path/to/package \
  //loom/target/amdgpu/rocjitsu:buffer_bulk_test
```

Add `--//build_tools/bazel:gpu_execution=native,rocjitsu` to include the physical
GPU leaf. Its hardware resource requirement is independent of the simulated
leaves. The bootstrap CLI checks and `workload_flags_test` exercise fixed topology
bindings and ordinary workload selection, including selection of a fourth GPU.

`server_test` uses the same selection flags to run the canonical embedding model
on two devices sharing one pool in each topology. It checks exact HTTP results
and normal SIGTERM shutdown. The ordinary `loom-serve_<profile>` invocations can
serve other model packages with their usual `--devices` and `--model` options.
Serving pools use `"pool_backing":"fixed"` in the device configuration, or
`--pool_backing=fixed` for the single-device CLI form, because the simulator
does not expose libamdf virtual memory reservations and remapping.
