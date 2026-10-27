# Loom XDNA arithmetic fixtures

[mul_i32.loom](mul_i32.loom) is a High `pipeline.def<kernel>` program whose
strands own placement, channel storage, and asynchronous transport. Its
signature-only `multiply_record` declaration leaves the record computation to
a separately linked module. [mul_i32_worker.loom](mul_i32_worker.loom)
supplies the ordinary authored-Loom implementation. When C++ import is enabled,
[mul_i32_worker.cxx](mul_i32_worker.cxx) supplies an interchangeable helper:

```c++
namespace loomt = loom::type;
using input_record = loomt::view<loomt::shape<16>, const int>;
using output_record = loomt::view<loomt::shape<16>, int>;

[[loom::symbol("multiply_record"), loom::force_inline]] void multiply_record(input_record lhs, input_record rhs, output_record output) {
  for (unsigned i = 0; i < 16; ++i) {
    const auto index = loom::index::cast(i);
    loom::view::store(loom::view::load(lhs, index) * loom::view::load(rhs, index), output, index);
  }
}
```

Each helper compiles into an independent relocatable Loom module. The ordinary
Loom dependency closure links one provider into the High pipeline before AIE2P
realization. The target pipeline assigns the targetless helper to its compute
strand and inlines it; neither provider authors a physical worker ABI. There is
no C++-specific linker or source-combining path. Both helpers multiply two
records of sixteen `i32` values and retain the low 32 bits of each product.
Their native images and generated C sources stay in the build tree, so source
and compiler changes regenerate the fixtures, including changes to the native
file format. The CTS executes the C++-provided image through the same
allocations, bindings, native submission, and complete CPU oracle as the
authored-Loom image.

[recurrence_i32.loom](recurrence_i32.loom) uses the same bindings and completion
protocol. For each word it starts with `lhs[i]` and applies
`value = value * lhs[i] + rhs[i]` for `rhs[15]` iterations, modulo 2^32. Its
data-dependent loop supplies bounded useful work for firmware marker timing.
Every output is checked independently on the CPU. The worker contains no timer
reads, delay instructions or extra profiling DMA; controller markers surround
its ordinary execution.

The build uses the shared `loom_kernel_binary` and `iree_c_embed_data` rules.
Loom is a build-tool dependency of these corpora; the test executables load
libamdf dynamically and use a constrained [CTS image reader](../util/executable.h).
Production libamdf has no compiler or image-loader dependency. No external AIE
compiler, SDK, transaction generator or linker is needed.

## Program and completion

The pipeline has one compute strand, an ingress/egress strand, two input
records, and one output record. Each record contains sixteen words. The image
declares three complete 64-byte bindings in `lhs`, `rhs`, `output` order. The
caller supplies DMA addresses, external guards, directional cache transitions,
and native completion waits. NPU4 uses `amd.xdna.strix.17f0_10` for
Strix/Krackan; NPU5 uses `amd.xdna.strix_halo.17f0_11`. The reader checks the
selected image's execution profile against the native endpoint before
allocating a placement.

Each generation submits the establishing invocation. It loads the worker and
configures the local rings before issuing finite shim transfers. Both complete
input records are required to produce the single output record, so the output
DMA join ends this program's external payload accesses. Checked native command
completion also ends the controller's use of command storage. Host and GPU
cache actions remain separate requirements.

The worker and circular compute DMA remain resident in tile-local storage.
Output completion does not mean that every core and DMA channel has stopped.
The next establishing invocation disables and resets its owned cores and
compute DMA before rewriting local state. The tests use that complete
invocation every time, including after time sharing; they do not assume that
resident state survives between native submissions. Normal context destruction
releases the placement after queue retirement and command-storage release.

## Resident services

[resident_exchange.loom](resident_exchange.loom) supplies the finite resident
service for the [GPU/XDNA exchange recipes](../../interop/gpu/xdna/recipes/README.md).
One immutable 64-byte configuration specifies generation count, payload extent
and one or two request/response credits. The service consumes GPU-produced
requests through direct scalar streams and returns transformed payloads through
shim DMA. A separate startup decision permits RUN or prestart ABORT. A final
GPU acknowledgement ends custom traffic before the ordinary 64-byte terminal
record reports completion.

The source has three array roots sharing the same service: `resident_service_array`
places one worker in column zero; `resident_channels_array` places independent workers
in columns zero and one. The latter exposes four complete 64-byte bindings in
configuration-zero, terminal-zero, configuration-one, terminal-one order.
Each worker retains its own configuration, local rings and shim resources.
The caller supplies a matching context width and composes the disjoint custom
routes and descriptors around the bound establishing invocation. The ordinary
terminal waits precede the caller's final custom DMA idle observations.

`resident_terminal_relay_array` keeps one service in column zero and routes its
complete terminal record through a second worker in column one. The relay
copies all sixteen words to its ordinary output. Its column owns the terminal
S2MM0 task and completion token, while configuration input and custom traffic
remain in column zero. The array retains two 64-byte host bindings and needs
a two-column context. Normal completion, zero work and prestart ABORT use the
same terminal route; no host action relays the record between workers.

`resident_split_response_array` uses that same relay with a service whose
response payload and ready word use different shim S2MM channels. The caller
assigns the two channel roles and queues both descriptors independently.
Payload completion releases one unit to shim lock zero; the ready descriptor
acquires and consumes one unit. Both channels operate in order. The same
worker image supports either channel assignment and either task-arming order.
Configuration words 3 through 6 contain the first task's control header and
slot-relative descriptor offset, followed by the second task's header and
offset. These values agree with the caller's cold descriptor/routing setup;
the image contains no external allocation address. Packet ID 9 carries the
payload and ID 11 carries ready. Each output drops its packet header.

The split worker keeps the chained service's arithmetic, credit reuse,
startup decision, full terminal record and final GPU acknowledgement. Its
terminal relay leaves both output channels on the service column available
for the response. Native retirement also joins both output DMA channels.
The [split-response matrix](../../interop/gpu/xdna/recipes/README.md#split-response-publication)
checks the complete GPU-observed payload for every role/arming combination.

[resident_npu_initiated.loom](resident_npu_initiated.loom) starts from an
NPU-produced payload and transforms each actual GPU return into the next NPU
payload. Its one worker has the same two 64-byte configuration/terminal
bindings and the same custom descriptor ownership as the one-credit service.
Configuration adds an immutable seed. A closing full payload exposes every
word of the final GPU return before the final acknowledgement and terminal
drain. Zero returns and prestart ABORT issue no payload transfers. The
[recipe](../../interop/gpu/xdna/recipes/README.md#npu-initiated-dataflow) specifies
the complete recurrence, credit reuse and independent output checks.

[resident_file_channel.loom](resident_file_channel.loom) uses that same
NPU-initiated recurrence with a CPU peer performing actual file operations.
It additionally accepts a negative errno in the peer generation word and
returns a terminal error after draining its custom DMA traffic. The
[Linux file-channel recipe](../linux/file_channel/README.md) owns cache-line
publication, POSIX I/O, complete output checks and CPU/NPU comparisons.

## Building and inspecting images

Enable `AMDF_BUILD`, `LOOM_BUILD` and `LOOM_TARGET_XDNA` in the repository
configuration. Building either arithmetic corpus automatically compiles both
profiles and embeds their outputs:

```sh
iree-bazel-build //libamdf/cts/xdna/recipes:execution_dynamic_bin
iree-bazel-build //libamdf/cts/interop/gpu/xdna/recipes:execution_dynamic_bin
```

Build the standalone image targets to inspect or export fresh compiler output:

```sh
iree-bazel-build \
  //libamdf/cts/xdna/programs:mul_i32_npu4 \
  //libamdf/cts/xdna/programs:mul_i32_npu5
```

With C++ import enabled, the corresponding linked C++ worker images are built
with:

```sh
iree-bazel-build \
  //libamdf/cts/xdna/programs:mul_i32_cxx_npu4 \
  //libamdf/cts/xdna/programs:mul_i32_cxx_npu5
```

Their outputs are `bazel-bin/libamdf/cts/xdna/programs/mul_i32_npu4.xdna` and
`mul_i32_npu5.xdna`; the C++ variants use the corresponding `mul_i32_cxx_*`
names. CMake generates the same named images in its corresponding package build
directory. Cross builds run the compilation and embedding tools on the build
host; the generated data is then compiled into the destination platform's test
executable. For CMake cross builds, `IREE_HOST_BIN_DIR` supplies the host
`loom-link`, `loom-compile` and `iree-c-embed-data` executables.
