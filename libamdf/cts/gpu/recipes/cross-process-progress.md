# Cross-process compute and readback progress

This recipe qualifies progress and ownership while a second process activates
the same physical GPU. It is a composition witness, not a claim that one PM4
packet or one runtime implementation caused a deployment failure.

The parent owns an already-active device, two case-local queues, coherent
SYSTEM backing, target-selected executable storage, and a guarded 1 MiB
payload. One workitem publishes `started`, advances an exact integer
recurrence, and exits after either a host release or 300,000,000 iterations.
The iteration ceiling lets a provider serialize fresh activation behind
accepted compute without deadlocking the case. Reaching it is recorded; the
child must still activate and the following readback must still complete.

The child is a re-execution of the same CTS binary. It loads the same provider,
opens the exact endpoint ID selected by the parent, and reports three framed
states over an inherited local socket: passive preparation, entry into
`device_create`, and successful activation. It remains alive until the parent
has completed and retired readback. There is no elapsed-time trigger, retry,
helper executable, SDK runtime, or ambient configuration channel.

## Native carriers

The process lifecycle, payload, executable, recurrence, and complete CPU oracle
are common. Each carrier retains its actual transfer mechanism:

| Carrier | Compute completion | Bulk readback |
| --- | --- | --- |
| PM4 | A post-dispatch `CS_PARTIAL_FLUSH` and SYSTEM acquire precede a separate SYSTEM release marker. | A second PM4 queue performs SYSTEM acquisition, `DMA_DATA`, DMA drain, and a SYSTEM release marker. |
| AQL | The dispatch packet has SYSTEM acquire/release scopes and a private USER completion signal. | A second AQL queue dispatches the target-selected byte-copy program with SYSTEM acquire/release scopes. |

Both paths snapshot source, destination, control, and argument backing before
the second queue retires. The oracle checks every payload and guard byte, the
recurrence result for the observed iteration count, unchanged arguments,
independent queue retirement, and child cleanup. Memory-profile and class
properties keep SYSTEM placement visible for integrated, discrete, and
datacenter deployments.

## Required deployment witnesses

Linux RDNA qualification requires both carriers on the exact selected
endpoint. This applies independently to integrated and discrete devices; the
original gfx1201 deployment is its own required witness rather than evidence
inferred from another RDNA target.

```sh
AQL_CASE=CrossProcessProgressAqlTest.ActivationDuringComputePreservesBulkReadback
PM4_CASE=CrossProcessProgressPm4Test.ActivationDuringComputePreservesBulkReadback
iree-bazel-test --config=asan \
  //libamdf/cts/gpu/recipes:recipes_dynamic \
  --test_arg="--amdf_gpu_endpoint_id=${GPU_ENDPOINT_ID}" \
  --test_arg="--amdf_gpu_target=${GPU_TARGET}" \
  --test_arg="--gtest_filter=${AQL_CASE}:${PM4_CASE}" \
  --test_arg="--amdf_require_test=${AQL_CASE}" \
  --test_arg="--amdf_require_test=${PM4_CASE}"
```

Linux CDNA qualification requires the AQL carrier. PM4 compute is not part of
the CDNA deployment contract.

```sh
AQL_CASE=CrossProcessProgressAqlTest.ActivationDuringComputePreservesBulkReadback
iree-bazel-test --config=asan \
  //libamdf/cts/gpu/recipes:recipes_dynamic \
  --test_arg="--amdf_gpu_endpoint_id=${GPU_ENDPOINT_ID}" \
  --test_arg="--amdf_gpu_target=${GPU_TARGET}" \
  --test_arg="--gtest_filter=${AQL_CASE}" \
  --test_arg="--amdf_require_test=${AQL_CASE}"
```

The complete qualification record adds the provider and binary identities,
native lifetime, kernel/driver/firmware deployment, endpoint identity and
topology, raw XML, and post-run device health described by the
[qualification model](../qualification.md#required-witnesses-and-deployment-identity).
