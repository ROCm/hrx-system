# Final experiment result

The persistent row schedule was rejected for production use. It reduced image
storage substantially but was 4.14 times slower than the faster CPU policy in
the controlled final comparison. The operator selected branch preservation
and ended codec integration in hrx-system.

## Measured execution

The machine was a gfx1151 GPU sharing memory with 16 physical Zen 5 CPU cores.
Both programs completed 128 JPEG-to-resized-RGB operations before one final
completion. The working cohort contained 16 images: four distinct images,
each repeated four times, replayed eight times. Both retained 16 final outputs
and used identical float32 filter weights and tap order. This was batched
execution, not 128 CPU-to-GPU-to-CPU round trips.

| Path | Median batch time | Scope |
| --- | ---: | --- |
| GPU baseline persistent schedule | 2,448.166 ms | Resident compressed input |
| GPU polling variant | 2,139.985 ms | Same input and final output contract |
| CPU libjpeg ISLOW plus resize | 520.481 ms | 16 task workers |
| CPU libjpeg IFAST plus resize | 516.559 ms | 16 task workers |

GPU JIT, allocation, and uploads were outside timing. CPU codec lifecycle and
temporary workspace allocation were inside timing. Output and guards were
verified before and after measurement. IDCT policies have independently
checked output hashes; CPU and GPU pixels are not asserted bit-identical.
Each round had two warmups and nine fixed samples. Raw samples and interleaved
round medians remain in [the conclusion](evidence/resize-conclusion.json).

The polling variant saved 12.6% against the first GPU schedule. A separate
device profile attributes 2,108.448 ms of 2,187.775 ms to
`jpeg_resize_stream`: 96.4% of device time. That worker selects entropy,
horizontal-filter, and vertical-filter roles and processes about 7.25 million
filter jobs. This locates the remaining cost in device work. It does not prove
that queue atomics alone cause the gap; task granularity, repeated work, and
memory traffic would require a new execution-shape experiment.

## Storage and numerical boundaries

Measured live image buffer backing was 94,371,148 bytes, including 12 bytes of
command transient storage, compared with 425,547,140 bytes for the previous
full-resolution path: a 77.8% reduction. The new buffers include 18.6 MB of
compressed input, another 18.6 MB of unstuffed entropy, 39.7 MB of private and
published final RGB, 5.1 MB of component/horizontal rings, and 12.4 MB of
tables, plans, queue state, and certificates.

These counts exclude compiler, driver, executable, and host fixture mirrors.
The CPU's process RSS is not a comparable device-allocation metric. CPU memory
parity was not established. A 12 MP image's component and horizontal rings use
705,024 bytes versus 54,867,456 bytes for its previous full component/RGB
planes, but the complete cohort still owns the other allocations above.

The preserved small cases check exact independent expected bytes, peer-image
isolation, guards, malformed-marker error 258, and malformed-padding error 4.
The large valid output was also compared with the real consumer's native
resize output. Neither this evidence nor a maximum one-sample-value IDCT
difference on earlier images is an error guarantee for every JPEG or model.

## Reusable work

The compiler repairs stand on independent language and backend contracts:
typed subgroup transport, convergent-operation CSE, integer/bitfield
legalization, semantic vector slicing, shared storage snapshots, predicate
lifetimes, unsigned range facts, and volatile completion spaces. Their
regressions exercise those contracts without depending on a JPEG decoder.

The archived u32 scan prototype had a qualified five-size win against rocPRIM
in a common HIP graph harness. Ordered affine scans exposed a remaining
placement-sensitive 2M-element loss, and the complete generic semantic family
was never selected. These are useful examples and compiler witnesses, not a
completed production scan library.
