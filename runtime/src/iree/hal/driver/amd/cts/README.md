# AMD heterogeneous conformance

This suite composes CPU, GPU, and NPU execution and memory services. It owns tests
that join engines; `gpu/cts/` and `xdna/cts/` own engine-specific behavior, and
`fabric/cts/` owns shared native memory contracts independently of GPU HAL
execution.

| Target | Participants | Contract |
| --- | --- | --- |
| `cpu_xdna_test` | Task and XDNA HAL | Consumer-first CPU→NPU→CPU arithmetic through one backing, changing-input replay, default and explicit transitions, recorded direct and indirect bindings, deferred allocation, host registration, and public mapping permissions. |
| `execution_test` | Task, XDNA, and RDNA HAL | Consumer-first CPU→NPU→GPU→CPU and CPU→GPU→NPU→CPU arithmetic through shared storage, independent device semaphores, default and table-resolved transitions, reusable indirect recordings, changing-input replay, stable-address backing replacement through each queue implementation, and untouched guard ranges. |
| `virtual_buffer_test` | Task and XDNA HAL, native GPU SDMA | Stable addresses across backing replacement, independently resolved mapping groups, host aliases and device-only reservations, and native process/instance lifetimes. |

The virtual-buffer suite submits GPU work through libamdf. It isolates the
native memory integration contract and does not qualify RDNA HAL queue execution.
The execution suite uses the public driver registry and queue APIs for every
participant. Its NPU/GPU handoffs use asynchronous host admission for foreign
semaphores; it does not claim firmware-only cross-engine waits.

Stable-address cases use conservative queue visibility and alternate physical
backing only after the final CPU reader. They check both the selected backing and
preservation of the previously detached backing before removing all mappings.
A reservation's immutable transition table has no physical backing to qualify;
these cases do not use that table to specialize barriers.

Finite-pool cases allocate through Task, return the only block through XDNA after
the final CPU reader, and reacquire it through RDNA while that reader is still
pending. A GPU fill and CPU readback verify the reused storage; both reservations
are returned explicitly.

Each target declares its required engines and hardware separately. CPU/XDNA
tests need an NPU; execution and virtual-buffer tests need both a GPU and an NPU.
Ordinary runtime capability checks identify unsupported native profiles.

```sh
iree-bazel-test --config=asan \
  --//runtime/config/hal:drivers=task,amd \
  //runtime/src/iree/hal/driver/amd/cts:cpu_xdna_test \
  //runtime/src/iree/hal/driver/amd/cts:execution_test \
  //runtime/src/iree/hal/driver/amd/cts:virtual_buffer_test
```
