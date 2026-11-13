# Two-bank scheduling experiment

`two_bank_schedule.h` supplies a causal submission order for two consecutive prompt chunks. It is experimental and is not connected to `loom-serve`. The existing `serve.py` and `host_adapter.cc` retain their prior behavior.

Each bank has three stages per layer: GPU prefix (including attention and state publication), FFN on the selected device, and GPU residual. After priming bank A, the order is:

```
B.prefix[L], B.ffn[L], A.residual[L],
A.prefix[L+1], A.ffn[L+1], B.residual[L]
```

The next-layer A stages are omitted at the last layer. The emitter submits work without host waits and owns all device dependencies:

- Serialize submissions on each device's timeline.
- Prefix waits for that bank's previous residual. B prefix also waits for A prefix at the same layer, which must publish A's KV/recurrent state.
- FFN waits for its own prefix. Residual waits for its own FFN.
- Keep both banks' scratch, metadata, outputs and capture buffers distinct; immutable weights can be shared.
- Retain resources and drain both final residual completions before recycling banks or admitting another pair. On failure, stop emission, propagate failure to downstream consumers, and drain or cancel already accepted work before destruction.

A single GPU timeline is intentional: B's independent prefix is queued before A's NPU-dependent residual, so a wait on A cannot prevent B's prefix from executing. A global wait on both device timelines after each submit defeats this schedule.

The native example is `.synagent/artifacts/loom-dual-device-runner/parallel-hybrid/probe.cc`. Its GPU stages pack synthetic integer state and consume the resident-external NPU FFN output; consumption flips the Boolean output to make the next layer depend on the prior result. It does not implement attention, model residual arithmetic, DFlash, or a model-serving adapter. The existing 32-worker image is an i8 unit-scale synthetic FFN, not the BF16 model image. The native result is a dependency/coherence test, not model correctness or performance qualification.

Build and run instructions, source hashes, admission failures, and the resource ledger are retained in the artifact directory. `run_probe.py` requires explicit GPU and NPU grant IDs and applies a 45-second timeout with a 3-second kill grace. Acquire both resources before launch and release them immediately after exit.
