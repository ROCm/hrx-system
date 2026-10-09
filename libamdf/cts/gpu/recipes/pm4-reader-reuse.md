# PM4 staging and independent readers

The [reader-reuse cases](pm4_reader_reuse_test.cc) qualify a complete finite
staging pipeline through PM4 and SDMA. Reusable SYSTEM source pages and copied
destination slots have different owners: SDMA completion releases a source,
while every independent compute reader must finish before destination reuse.
The destination is either coherent SYSTEM or non-host-mapped LOCAL memory.

The complete command program is recorded before any queue is published. USER
cases publish mapped native rings; KERNEL cases submit retained executable
command buffers. Both use the same packet bodies, immutable kernel arguments,
compiled transform and independent output oracle. This covers pre-recorded
staging with known inputs; GPU-generated transfer commands are a separate
publication contract.

## Owners and dependencies

| Owner | Operation | Completion releases |
| --- | --- | --- |
| PM4 refiller | CP DMA copies a distinct immutable input record into its reusable SYSTEM source slot. | That job's source is ready for SDMA. |
| SDMA uploader | Copies the source into its SYSTEM or LOCAL destination slot. | The source may be overwritten; compute may read the destination. |
| Each PM4 reader | Independently dispatches a multi-workgroup transform into its own retained output. | That reader's use of the destination. |
| Final SDMA readback | Joins the readers and copies complete destination backing, including guards. | The host may snapshot all payloads and control records. |

For job `j`, source credits `S` and destination credits `D`:

- Refill `j` waits for upload `j-S` before source reuse.
- Upload `j` waits for refill `j` and every reader of destination user `j-D`.
- Each reader of `j` waits for upload `j`.
- The held reader of `j` additionally waits for refill `j+S`, when present.

That last dependency proves the copy's independent lifetime. The source
contains different bytes before the held reader consumes the earlier copied
input. The refiller samples that reader's still-pending completion after the
overwrite and before publishing refill completion. The sample is observational;
it never controls progress. The held reader alternates between the two logical
readers so each completion remains necessary to protect the destination.

The chain remains acyclic when `S > D`: refill `j+S` depends on upload `j`,
not its readers. The final `S` jobs drain without the extra dependency. All
waits execute in command processors, with no shader waiting for another shader
to become resident. Each reader queue joins its own dispatch before acquiring
the next job's data; the other reader queue remains independent.

## Visibility and observation

PM4 drains its CP DMA before publishing refill completion. Exact memory-pair
queries select SDMA acquire/release operations and verify PM4's required system
transitions. Native progress-word polling, payload acquisition, shader
completion and command retirement are separate parts of the contract.

Each PM4 reader publishes its immutable code through the initial full system
barrier. Subsequent jobs retain the instruction cache and perform explicit data
acquisition after their dependencies. The compiled image comes from the central
physical-target catalog; no target-specific shader is embedded in this case.

The host submits each complete stream once and observes the final in-stream
completion. It snapshots all payloads and control before waiting for native
retirement, so a driver's retirement path cannot supply missing payload
visibility. Every native stream retires before any oracle failure can return
from the case. Checks cover all reader outputs, final source and destination
contents, guards, completion/observation records, immutable inputs, arguments,
code and complete command backing. Queue-first teardown retains each backing
through its last native user.

## Matrix and required execution

The matrix crosses USER/KERNEL publication, SYSTEM/LOCAL destination, 4/64-KiB
payloads, and `(source, destination, reader)` counts `(1,1,1)`, `(1,2,2)`,
`(2,1,2)` and `(2,4,2)`. Each case has 33 jobs. They share the existing
`recipes_dynamic_bin`; PROCESS and INSTANCE are invocations of that binary.
The finite command owners request 64 KiB of storage on either native transport.

Both engines must advertise the selected publication mode. LOCAL placement
requires its own heap. Windows additionally requires an HWS SDMA family;
PM4 compute support alone cannot establish that transfer service. Missing
capabilities produce skips in ordinary discovery, while an exact required
case makes an unmet deployment contract fail qualification.

For example, a Windows SYSTEM witness selects the KERNEL case:

```sh
iree-bazel-test --config=asan \
  //libamdf/cts/gpu/recipes:recipes_dynamic \
  --test_filter='Publication/Pm4ReaderReuseTest.SourceCopyAndEveryReaderOwnSeparateReuseBoundaries/KernelSystem64KiBS1D2R2' \
  --test_arg=--amdf_require_test=Publication/Pm4ReaderReuseTest.SourceCopyAndEveryReaderOwnSeparateReuseBoundaries/KernelSystem64KiBS1D2R2
```

Linux mapped-queue qualification uses the corresponding `User` case, with
`Local` when qualifying device-local placement. Cross-compilation or a skipped
case does not establish native execution. Successful output establishes
ownership and progress, not transfer throughput or physical engine overlap.
