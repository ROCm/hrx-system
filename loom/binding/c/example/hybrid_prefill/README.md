# Experimental two-device Qwen prefill adapter

The `compile_npu.c` helper uses the loomc C API to compile the XDNA program;
its output for the 512x1536x5120 bank was byte-identical to `loom-compile`.
This host adapter runs the fetched Qwen3.8-27B Q4_K_M package through the Loom
0.1.0 SDK. Selected prefill projections use both AMDGPU and XDNA HAL queues.
Model loading, attention, recurrent state, sampling, DFlash, and decode stay in
the SDK. This is a research example with a deliberately narrow kernel ABI, not
a new production model-card schema or a generally compatible preload library.

The adapter supports `g_gemm_q4_k_store_m256` at M=1024, K=5120, N=17408 only.
It divides output features between the devices and divides the token dimension
into two independent 512-token banks. Other roots and shapes use the original
recording. This preserves short/tail prefill and decode execution, but performance
and token equivalence must still be measured. In the reference package the
supported subset includes 16 Q4_K FFN gate projections per complete 1024-token
chunk. A fraction is a fraction of those projections' output features, not a
fraction of all model computation.

The qualified whole-array package program uses 32 compute tiles, paired DMA
owners and peers, 128x128 macrotiles and K128 panels. Configuration changes only
`whole_array.steps`, `whole_array.tiles`, and `whole_array.nblocks`. `compile.py`
validates positive aligned dimensions, the 16-pair schedule, and the signed-i8
i32 accumulation bound. Its default artifact is byte-identical to the prior
5.52-TOPS image. That historical integer throughput is not this adapter's
application throughput.

## Numerical path

Q4_K block scales and negated minima are separately rounded to F16 before
the fused dequantization operation is rounded to F16, matching the source GPU
kernel's operand policy. Activations already supplied as F16 and their inverse
row scales come from the original package. `i8` requantizes each activation and
weight row once. The initial single-i8 candidate failed the 8k golden
continuation; this option remains an unqualified diagnostic control. `i8x2` represents each row as primary plus residual
i8 values with independent scales and evaluates all four products. This requires
four whole-array invocations per bank. Token checks, not arithmetic precision
claims, determine acceptance.

Packed layouts (last two axes are contiguous 8x8 fragments):

- A: `[M/128,K/128,16,16,8 rows,8 reduction lanes]`, signed i8.
- B: `[N/128,K/128,16,16,8 reduction lanes,8 output features]`, signed i8.
- C: `[M/128,N/128,16,16,8 rows,8 output features]`, signed i32.

CPU weight conversion is cached per fixed parameter buffer and offset. Packing,
weight uploads, scale epilogues, and assembly currently run through host code and
HAL transfers, and are included in request timings. The prototype is expected to
lose substantial time here. It does not change or duplicate the original GGUF.

## Scheduling and compatibility

Recording captures are replayed as GPU segments around eligible projections.
Accepted host jobs retain the original command buffer, queues, binding buffers,
and semaphores. They return immediately to the SDK: synchronously waiting inside
the SDK submission call deadlocks its later prerequisite publication. Ready jobs
serialize access to shared scratch, while GPU bank 0 and NPU bank 1 run together,
followed by the opposite bank order. Each device uses an explicit timeline;
transfers wait for both producers before reading results. The original completion
is signaled only after assembly. A failed job fails that completion.

The library is coupled to the installed SDK ABI and this package's kernel ABI.
Current checkout headers are incompatible with the tested SDK's memory-pool
structure. Build against the publisher's `d01d1c1` runtime headers. The publisher
pin does not independently attest the wheel's source provenance. The experimental
process retains its auxiliary device group and conversion caches until exit;
use a dedicated server process for each measurement block. It does not support
hot model replacement or general multi-tenant use.

## Validation status

The corrected residual-i8 candidate is **not token equivalent** to pure GPU.
At requested fraction 0.10 it matched the short 64-token continuation, but differed
at tokens 39 and 45 for the 1k and 8k continuations. Its first projection's NPU
prefix relative L2 error was 9.09e-5; the GPU remainder was bit-identical. The
sampled integer-dot oracle and the separate CPU metadata conversion oracle pass.
Those checks do not qualify the model. Performance results are diagnostic only.

## Use

All files, compiler scratch, and `TMPDIR` must be on persistent project storage.
Build the adapter and compile bank images before acquiring devices:

```sh
python loom/binding/c/example/hybrid_prefill/build.py \
  --headers /project/sdk-headers/runtime/src \
  --generated-headers /project/runtime-build/runtime/src \
  --loomc-headers /project/sdk-headers/loom/binding/c/include \
  --sdk /path/to/sdk/bin \
  --output /project/artifacts/sdk_adapter.so
```

Use the publisher-pinned headers described above, not current checkout headers.
Compile bank images before acquiring devices. For the default fraction, N=2560:

```sh
python loom/binding/c/example/hybrid_prefill/compile.py \
  --capi-runner /project/artifacts/compile_npu --m 512 --n 2560 --k 5120 \
  --output /project/artifacts/bank_2560.bin
```

`serve.py` leaves the SDK path unchanged when `--prefill-offload-device` is absent.
To opt in, acquire GPU and NPU grants externally, export `GPU_GRANT` and
`NPU_GRANT`, then launch under a timeout and process-group kill switch:

```sh
python loom/binding/c/example/hybrid_prefill/serve.py \
  --sdk /path/to/sdk/bin --model /path/to/package/model.json \
  --artifacts /project/artifacts --prefill-offload-device xdna \
  --prefill-row-fraction 0.15 --precision i8x2 \
  --memory_bytes=52000000000 --listen=127.0.0.1:18439
```

Release both grants immediately after stopping the server. Never infer a model
speedup from the isolated kernel or from timings excluding host conversion.
The task's incremental report records validation, measurements, and remaining
limitations: `.synagent/reports/2026-11-07-loom-hybrid-runner.md`.
