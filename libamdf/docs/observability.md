# Timing, observation, and telemetry

libamdf exposes native observation sources, their storage contracts, and their
clock metadata. The caller owns workload identity, queues, result storage,
collection, and the relationship between device and host clocks. Profiling
does not introduce a library-owned collector or change ordinary submission.

The shared [observation types](../include/amdf/observation.h) describe a clock
and the progress of a native read. Source-specific operations live in the
[GPU](../include/amdf/gpu.h) and [XDNA](../include/amdf/xdna.h) extensions.
Extension negotiation establishes that an API table is present. The selected
queue family, live device, context, or native service establishes which
operations are available.

The [telemetry extension](../include/amdf/telemetry.h) supplies measured clock
rates, power, temperature, and activity with physical units and source identity.
Its two operations enumerate complete metric descriptions and read selected
fields from one source. The caller schedules reads and owns the resulting
history; libamdf creates no sampler, thread, GPU allocation, or sensor session.

The device-independent `AMDF_EXTENSION_OBSERVATION` table supplies the profiling
host clock. `query_host_clock_info` returns its frequency and counter width;
`query_host_clock` reads its ticks. Neither requires an instance or device.
CPU intervals and the host anchors for every GPU/NPU source can therefore use
one portable timebase without calling native clock APIs. The caller queries
units once and subtracts timestamps before converting a duration:

```c
const void* extension = NULL;
amdf_status_t status = api->query_extension(
    AMDF_EXTENSION_OBSERVATION, AMDF_OBSERVATION_EXTENSION_VERSION_1,
    AMDF_OBSERVATION_EXTENSION_VERSION_LATEST, &extension);
// Handle status before consuming extension or any subsequent output.
const amdf_observation_api_t* observation = extension;
amdf_observation_clock_info_t host_clock = {0};
status = observation->query_host_clock_info(&host_clock);
uint64_t before = 0, after = 0;
status = observation->query_host_clock(&before);
// Perform CPU work, or publish and observe the intended device work.
status = observation->query_host_clock(&after);
double duration_ns = (double)(after - before) * 1e9 / host_clock.frequency_hz;
```

The host clock is independent of wall-clock adjustments. Its epoch is shared
by callers on that host within one boot, including separate provider instances.
It does not establish a common epoch between network nodes. Suspend, device
reset, and observation-source loss require refreshing device/host relations.

## Selecting a source

| Observation | Selection and output | Readiness and reuse |
| --- | --- | --- |
| CPU submission and completion observation | Observation extension host-clock reads around the caller's actual publication and completion operations. | Measures host-observed latency, including scheduling and observation delay. It does not supply device execution timestamps. |
| GPU PM4 or AQL-carried command-processor clock | Caller-written native timestamp commands and destination storage. `device_query_queue_clock_info` supplies nominal units for the selected queue family. | The command's pipeline stage and explicit dependencies determine what it measures. Result-write completion and the memory edge precede collection. |
| GPU SDMA clock | Caller-written `TIMESTAMP_GET_GLOBAL` commands around the intended transfer work. The same queue-clock query supplies nominal units. | The caller orders payload release and completion after the ending sample, then acquires the result before reuse. |
| AQL dispatch interval | An AQL family advertises `DISPATCH_TIMESTAMPS`; queue creation selects `AMDF_GPU_USER_QUEUE_FLAG_DISPATCH_TIMESTAMPS`. Firmware writes the dispatch's native completion signal. | Each observed dispatch owns an exclusive signal through completion and timestamp consumption. Capture is a queue-lifetime option, while the caller chooses which dispatches receive observation signals. |
| XDNA transaction marker | `context_query_result_buffer_info` supplies destination geometry. The caller attaches backing and supplies `RECORD_TIMER` instructions and the finite record protocol. | Every result writer retires before native collection and host cache acquire. Valid extent follows the caller's record protocol. |
| XDNA automatic firmware events | `device_query_firmware_trace_info` reports recorder state, schema, clock, and read-storage requirements. `device_read_firmware_trace` returns device-wide events with native context and opcode fields, without application-submission attribution. | Native transport can lose data. Read flags and native byte positions describe progress; neither an event nor an empty read retires application work. |

The [GPU timing contract](gpu.md#queue-timing) describes explicit timestamps
and AQL signal ownership. [XDNA result buffers](xdna.md#finite-firmware-results)
and [automatic firmware timing](xdna.md#automatic-firmware-timing) describe
the two distinct NPU sources.

PM4/AQL queue availability follows the native provider. AQL is a Linux service;
Windows GPU queues retain their PM4/SDMA mechanisms. XDNA automatic events use
the native MCDM service on Windows and the SDK DPT interface when supplied by
the Linux driver. An absent trace service returns `UNSUPPORTED`. A permission
failure remains a native error and is not evidence that the hardware lacks
the source. Finite transaction markers have their own admission independently
of that recorder.

Explicit timestamp destinations, finite marker IDs, and exclusive AQL
completion signals associate observations with the caller's work. Automatic
XDNA records instead expose firmware identities. The trace interface supplies
no mapping from those identities to a public XDNA context or submission.
Enabling the recorder establishes configuration ownership, not exclusive
workload execution. A device-wide diagnostic trace is therefore available
independently of an attributed application timeline; an arbitrary concurrent
dispatch cannot be identified from an opcode or its position in that trace.

## Device telemetry

`AMDF_EXTENSION_TELEMETRY` operates on an already-live device. Enumeration and
reads are explicit cold native queries that may block and wake a suspended
device. Discovery and ordinary submission perform no sensor work. A device
with no supported source enumerates zero metrics; permission, removal, and
native query errors remain failures. Obtaining the extension table itself is
inert and does not imply that a particular device has a sensor service.

Each descriptor carries a portable measurement kind, physical domain, instance,
exact rational unit conversion, temporal aggregation, and native provenance.
Selection uses those typed facts: a GPU compute clock differs from a memory
clock, and an edge temperature differs from a hotspot. Diagnostic names identify
native fields without making the application carry an architecture-specific
name or register table. IDs are opaque within the live device and source;
saved captures copy the descriptor strings and device identity.

A caller can select a measured GPU clock with no platform code:

```c
// device is already live. The caller owns catalog with nonzero capacity.
const void* extension = NULL;
amdf_status_t status = api->query_extension(
    AMDF_EXTENSION_TELEMETRY, 1, AMDF_TELEMETRY_EXTENSION_VERSION_LATEST,
    &extension);
if (!amdf_status_is_ok(status)) return status;
const amdf_telemetry_api_t* telemetry = extension;
uint32_t count = 0;
status = telemetry->device_enumerate_metrics(
    device, catalog_capacity, catalog, &count);
if (!amdf_status_is_ok(status)) return status;
for (uint32_t i = 0; i < count; ++i) {
  const amdf_telemetry_metric_info_t* metric = &catalog[i];
  if (metric->kind != AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE ||
      metric->domain != AMDF_TELEMETRY_DOMAIN_GPU) continue;
  amdf_telemetry_value_t value = {0};
  amdf_telemetry_sample_t sample = {0};
  status = telemetry->device_read(device, metric->source_id, 1,
                                 &metric->metric_id, &value, &sample);
  if (!amdf_status_is_ok(status)) return status;
  // Consume value.status before interpreting its selected numeric member.
  // Retain metric and sample alongside the value in caller-owned history.
}
```

Zero-capacity enumeration returns the required count without writing a catalog.
Insufficient nonzero capacity returns `BUFFER_TOO_SMALL`, a valid prefix, and
the total count so the caller can resize its storage. Other errors preserve
both outputs. Reads preserve all values and the sample descriptor on a
call-level failure. A successful read gives every selected value its own
status: `NOT_FOUND` means the native field is unavailable, while a successful
zero remains a real reading. Duplicate selections share the same observation.

Fields from one native table share a source ID and can be read together through
one query. Linux hwmon attributes are separate sources because their reads are
sequential native operations. Windows PMLog supplies a native table through the
existing WKMI dependency, without starting its shared-memory sampler. That
query has no physical-adapter selector; linked adapters therefore contribute
no PMLog source rather than assigning an ambiguous observation to one GPU.
Only sensors with established measurement semantics and units are enumerated.
Configured clock targets, power caps, and maximum rates are not measurements.

Linux identifies the native SMU producer during the explicit telemetry query.
The same hwmon name can describe different quantities: SMU14 APU clock fields
are filtered targets, the Renoir memory-clock route reads its fabric clock,
and APU temperature routes identify GPU and SoC regions rather than edge and
hotspot sensors. The descriptors preserve those distinctions. Clock routes
that synthesize zero on older firmware require the native firmware capability
before enumeration. An unavailable producer identity leaves those sources
absent; malformed identity metadata remains an error.

Physical values are `native_value * scale_numerator / scale_denominator` in
the descriptor's unit. Integer storage preserves native precision until the
caller chooses a display representation. Package power can include CPU power
on an APU; it does not become GPU-only consumption because the GPU exposes it.
An APU's firmware power budget can also include a participating discrete GPU;
`POWER_BUDGET` preserves that scope separately from package power.
Related domains and multiple sources can overlap and are not additive. Native
sensor names such as "average" or "current" do not establish a filter window;
`UNSPECIFIED` aggregation preserves that absence of information.

The host timestamps bracket the explicit native read in the observation
extension's clock. They measure query timing, not when the sensor acquired or
filtered its value. Optional driver and source timestamps retain their separate
clock domains; absent timestamps stay absent. Repeated reads can return the
same cached observation. Actual filter constants or averaging windows belong
to individual values when provided. This information supports a device-level
timeline without claiming per-dispatch power attribution or sensor freshness.

Measured operating clocks describe device behavior. Queue timestamp frequency
comes from the queue-clock APIs below, independently of these clock sensors.

## Clock units and correlation

`amdf_observation_clock_info_t` carries `frequency_hz` and meaningful
`counter_bit_count`. Zero means the corresponding fact is unspecified. The
source and its capture lifetime identify the clock; the record is not a
globally comparable clock identifier. Equal frequencies, a shared device,
and a common storage width do not establish a shared epoch.

GPU extension version 2 exposes the native nominal queue-clock rate. It is
separate from shader-cycle and performance-event units. XDNA format-2 automatic
events represent nanoseconds. XDNA extension version 2 exposes the live
partition-shim operating rate for `RECORD_TIMER`; the cached result-buffer
geometry deliberately leaves that rate unspecified. These queries allocate
no collector and retain no calibration history.

`gpu->device_query_clock_sample(device, family_ordinal, &sample)` supplies a
complete native device/host tuple through one portable operation. The sample
contains device ticks and their nominal rate, the native paired host timestamp,
and common host-clock reads bracketing the native operation. Linux retains all
KFD counters, including the auxiliary system-clock timestamp and frequency;
Windows retains the selected engine's KMT GPU/QPC pair. The primary host values
always use the observation extension's clock. Clients need no native descriptor,
ioctl, KMT call, or platform clock selection.

An application's existing host clock can use a different epoch or rate. Its
own portable reads can bracket the libamdf sampling call to relate the returned
observations to that clock. Scaling ticks to nanoseconds changes units; it does
not make timestamps from distinct host clocks interchangeable.

This operation samples the native driver's counter. A queue or shader can use
a different counter with the same frequency. A native tuple therefore does not
establish the relation of every observation on that GPU. Actual source anchors
supply that relation without a device-name or OS branch in the consumer. The
query allocates nothing, retains no calibration state, and does not submit or
wait for workload commands. It belongs at explicit sampling points, outside
ordinary queue publication.

A same-epoch, non-wrapping delta at a constant rate converts to nanoseconds as
`delta_ticks * 1e9 / frequency_hz`. Subtract the source origin before converting
to floating point or widening the arithmetic. A rate describes units; it does
not establish oscillator accuracy or continuity through gating, preemption,
reconfiguration, or reset. For the XDNA array timer, equal rate snapshots before
and after work do not prove that the intervening rate stayed constant.

A caller can bound a source-to-host offset with an independently attributed
interval. If `host_before` precedes its submission and `host_after` follows
its completion, and the source endpoints have been converted to nanoseconds,
a constant-offset model gives:

```text
host_before - source_begin <= offset <= host_after - source_end
```

The interval width includes scheduling and host-observation delay. Its midpoint
is an estimate with half that width as sampling uncertainty, not an exact
calibration. Intersecting bounds from several observations can narrow a stable
segment; an empty intersection exposes inconsistent attribution, rate, or
epoch. A long capture needs a relation that accounts for drift and invalidates
old segments after loss or reset. Each GPU/NPU source gets its own relation to
the common observation host clock before events from different sources are
compared. A host bracket remains a valid enclosing interval when a finer device
clock relation is unavailable; its uncertainty stays visible in the output.

## Multi-device timelines

A CPU-to-GPU-to-NPU-to-GPU pipeline uses one host origin and a separate clock
relation for each observed device source. The producer records host ticks before
publishing an independently identified command; its completion observer records
host ticks after acquiring that command's completion. The collector pairs the
command's device interval with those host observations. The identity is supplied
by the application, not inferred from adjacent records in a shared firmware log.

The same procedure applies to PM4 and SDMA timestamp slots, AQL completion-signal
timestamps, and attributed XDNA firmware events. For a finite `RECORD_TIMER`
interval, the marker epoch and operating-rate condition additionally belong to
that command. A kernel's shader-cycle counter requires its own source contract.
Recording a GPU queue timestamp on either side of a GPU/NPU dependency can also
measure the entire offload interval in one clock; the intervening NPU interval
describes the work inside that envelope.

Collection follows the existing completion protocol and may run in batches.
Calibration does not require a host wait between each pipeline stage. Anchor
cadence, retained history, and interpolation policy belong to the collector;
native services, clock units, and platform clock reads belong to libamdf.
Durations come from one continuous source. Cross-device edge latency comes from
the mapped endpoints with both uncertainties retained. If those uncertainty
ranges overlap, the observations do not resolve that edge's latency precisely.

## Sustained collection and storage

Caller-controlled GPU observations can use a fixed ring of slots. The producer
obtains a free slot, writes its observation, establishes the required visibility,
and publishes that generation. The consumer acquires the matching generation,
copies the record to its sink, and returns the slot. Work can continue on other
slots throughout collection. Queue-ring consumption permits command-storage
reuse; it does not replace the observation's result-publication boundary.

AQL dispatch timing uses the dispatch completion signal as that publication
boundary. Its timestamp fields remain borrowed until the caller has consumed
them. Each dispatch carries its own signal; a shared counter or a later barrier
cannot preserve individual intervals. A resident dispatch supplies only its
outer interval. Internal regions require an observation source that runs while
that dispatch remains live.

[Memory reservations](memory.md#stable-reservations-and-replaceable-backing) can project one
backing range repeatedly into a contiguous address range when the selected
memory profile supports it. Such aliases simplify wraparound access; they do
not create publication, credits, or overwrite protection. Native record headers
and payload stride still determine the source's actual ring geometry.

Automatic XDNA trace reads instead copy from a lossy native recorder into
caller-provided host storage. Each reader owns a position and a capture
generation. A successful `amdf_observation_read_result_t` identifies the valid
output extent and next native position. `GAP`, `RESTART`, and `DISCARDED` are
processed before the batch's records; interval pairing cannot cross those
discontinuities. Decoded output bytes and native cursor units can differ.
Storage is borrowed only during the read and may change even on failure;
failure leaves the result descriptor unchanged.

Finite XDNA result buffers have a different lifecycle. The caller budgets every
record, publishes its destination, attaches it, and runs the planned commands.
After all result writers retire, it collects and acquires the host view before
consuming records. Detachment ends the native association; memory remains
caller-owned. Neither an append position nor a live-read cursor is supplied by
this mode, so it cannot be consumed as the automatic trace stream.

## Native controls and teardown

XDNA firmware recording is device-wide. Its configuration owner arranges
cross-process authority and serializes reconfiguration against reads. The
library does not acquire that authority, and device destruction does not
disable the recorder. An external privileged recorder can own configuration
while a workload remains unprivileged, subject to the reader's independent
native permission requirements.

Normal teardown follows the source's last writer and last reader. GPU slots
are released after their producers retire and their records are consumed.
Finite XDNA result backing stays live until its writers retire and its native
association is removed. A failed native control can have made progress; its
error is not rollback or permission to free borrowed backing. Automatic trace
storage is borrowed only for each read, while final event draining follows the
application's completion and expected-event protocol. An empty batch alone
does not establish that all events have arrived.
