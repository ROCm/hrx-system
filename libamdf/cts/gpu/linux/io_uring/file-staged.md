# File ingress through SDMA to independent compute readers

The [staged-file cases](file_staged_test.cc) read a GPU-selected block from a
private file into a reusable registered SYSTEM page, copy it through SDMA, and
consume the copy in independent multi-workgroup dispatches. The destination is
either coherent SYSTEM memory or non-host-mapped LOCAL memory. This exercises
the ownership needed when an upcoming model-table lookup is known before the
current computation finishes; it does not implement a cache or model scheduler.

The source and destination have separate reuse boundaries:

| Storage | Becomes readable after | May be overwritten after |
| --- | --- | --- |
| Registered source page | Complete file read, including any positive short completions. | SDMA has finished reading it. |
| Copied destination slot | SDMA completion and the compute dispatch's SYSTEM acquire. | Every independent reader dispatch completes. |
| SDMA command bytes | Release publication of the completed packet sequence. | The native command read frontier passes them. |
| Reader output | That reader's native completion. | Final host verification in this finite witness. |

## Queue ownership and progress

Three finite AQL batches and one SDMA ring share one cached device. The file
queue owns the native io_uring submission/completion indices. Each of its
one-workitem dispatches computes a file-block selection, publishes READ_FIXED
requests, consumes CQEs and records the actual read result. The upload queue
owns SDMA packet generation and its persistent byte frontier. The reader queue
launches one or two separately completed dispatches per copied input. Each
reader transforms every input word into its own retained output.

For job `j`, `S` source pages and `D` destination slots:

- File read `j` waits for upload `j-S` before reusing its source page.
- Upload `j` waits for file read `j` and all readers of destination user `j-D`.
- Every reader of `j` waits for upload `j`.
- The final reader of `j` additionally waits for file read `j+S`, when present.

The last edge makes source reuse observable: the original source page contains
different data before the final reader consumes the retained copy. Read `j+S`
depends on upload `j`, so it can finish without waiting for that reader. The
schedule remains acyclic with more source credits than destination credits.
The held reader alternates logical ordinals so both completion signals protect
the retained destination in two-reader cases.
The final `S` jobs drain without this extra edge. The file shader records that
the previous source user's final native reader signal is still pending; this
observation never controls progress.

Dependency waits are native AQL barrier packets and occupy no shader workgroup.
The file shader waits only on Linux I/O; the upload shader waits only on the
independent SDMA engine. Neither waits for another shader to become resident.
Each dispatch owns its SYSTEM acquire/release scopes. Dependency-only barriers
use NONE scopes, and header barriers explicitly serialize the two publishers.

The host publishes all immutable packet batches and arguments once. During
execution it services only SQPOLL idle wakes. It does not create per-job SQEs,
consume CQEs, copy payloads or dispatch successors. Linux file I/O still uses
CPU/kernel service; this is not a claim of CPU-free storage access.

## Visibility, errors and the oracle

CPU buffer pointers and GPU virtual addresses are separate arguments. The
[native resource owner](file_io_resources.h) retains ordinary registered WB
pages, restricted native rings and an unlinked private regular file. Exact
registered-memory pair queries establish SDMA cache work before publication;
an owned SYSTEM allocation's visibility policy is not substituted. SDMA
completion admits the reader dispatch, whose SYSTEM acquire covers the copied
payload. Cold initialization and final LOCAL readback use the same queried
memory-edge contracts.

Every native request is consumed before the file owner advances its persistent
ticket. Positive short completions advance both file and buffer offsets. EOF
before the complete payload records `-ENODATA`; an absent fixed-file entry
records `-EBADF`. The first error stops new file requests. All already-published
AQL dispatches still drain: uploads and readers of failed jobs leave payloads
and outputs untouched. Earlier successful jobs retain their own success result
and finish even if a later read fails. No partial input becomes compute input.

The oracle checks every reader's full output, every final source/destination
word and guard page, all result fields, native signals, unchanged requests and
arguments, SQ/CQ progress and overflow counters, and the complete file.
Successful jobs replace each reused source with a different file block.
Native request counts permit positive short completions and must agree with
the final SQ/CQ frontiers. Copy completion and command retirement are checked
independently before returning SDMA publication to the host.

## Matrix and invocation

One grouped `file_staged_dynamic_bin` contains all cases. Its PROCESS and
INSTANCE invocations exercise the same shared-library path. Kernels are built
from [file_staged.loom](../../kernels/file_staged.loom) and the shared
[SDMA copy program](../../kernels/sdma_copy.loom) for the complete configured
GPU target matrix. AQL admission permits both RDNA and CDNA; the exact device
must separately advertise registered-memory SDMA visibility. PM4 is not a
requirement of this corpus.

The success matrix crosses buffered/direct I/O, SYSTEM/LOCAL destinations,
4/64-KiB payloads and `(source, destination, reader)` counts `(1,1,1)`,
`(1,2,2)`, `(2,1,2)` and `(2,4,2)`. Each batch has 33 jobs. Buffered error
cases fail job seven after earlier successes, with both source-heavy and
destination-heavy credit arrangements. Positive partial EOF and invalid-file
errors have separate cases for each placement.

```sh
iree-bazel-test --config=asan //libamdf/cts/gpu/linux/io_uring:file_staged
```

Host registration, SQPOLL permission, LOCAL placement and the direct-I/O
filesystem contract are independent capabilities. Their absence produces
explicit skips before GPU submission. A qualification invocation can require
an exact case with `--amdf_require_test`, for example
`FileToSdma/FileStagedOwnershipTest.CopyReleasesSourceAndEveryReaderReleasesDestination/DirectLocal64KiBS1D2R2`.
A generic GPU resource tag does not establish disk-backed scratch. Direct cases
require `O_DIRECT` and `STATX_DIOALIGN`; tmpfs is not substituted for disk-backed
evidence. Correctness receipts establish byte ownership and progress, not
throughput or a preferred scheduling policy.
