// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_PROFILE_H_
#define IREE_HAL_DRIVER_AMD_GPU_PROFILE_H_

#include "iree/hal/api.h"
#include "iree/hal/driver/amd/gpu/image/image.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/perf_counters.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/thread_trace.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device-owned session with bounded, independently synchronized queue streams.
// Begin/end require idle queues and external serialization with submission.
// Appending never allocates or calls a sink. Begin, flush, end, and cold queue
// registration own sink calls; new queues emit metadata before accepting work.
typedef struct iree_hal_amd_gpu_profile_t iree_hal_amd_gpu_profile_t;

// Host-only reflection and original artifact bytes, independent of native
// executable allocations.
// Captured events may retain this resource until a later explicit flush without
// retaining the executable or any of its device memory.
typedef struct iree_hal_amd_gpu_profile_executable_t
    iree_hal_amd_gpu_profile_executable_t;

// Captures the admitted image and its native placements without retaining the
// executable. The native executable loader supplies a contiguous image source;
// metadata shares that host storage and copies the load addresses. All
// addresses refer to normalized image offset zero on this device.
iree_status_t iree_hal_amd_gpu_profile_executable_create(
    iree_hal_executable_t* executable, const iree_hal_amd_gpu_image_t* image,
    uint32_t load_count, const uint64_t* load_addresses,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_executable_t** out_metadata);

// Appends a newly placed queue variant to the retained host history. Flush
// serializes the resulting load records independently of native image lifetime.
// The executable serializes additions; flush may run concurrently with them.
// NULL metadata requires no work; failure leaves the prior history unchanged.
iree_status_t iree_hal_amd_gpu_profile_executable_add_load(
    iree_hal_amd_gpu_profile_executable_t* metadata, uint64_t virtual_base,
    uint64_t byte_length, uint64_t address);

void iree_hal_amd_gpu_profile_executable_retain(
    iree_hal_amd_gpu_profile_executable_t* metadata);
void iree_hal_amd_gpu_profile_executable_release(
    iree_hal_amd_gpu_profile_executable_t* metadata);
uint64_t iree_hal_amd_gpu_profile_executable_id(
    const iree_hal_amd_gpu_profile_executable_t* metadata);

// Recording-time attribution. The executable metadata is borrowed from the
// recording owner or direct capture. Completed profile records retain it.
typedef struct iree_hal_amd_gpu_profile_dispatch_t {
  // Host-only executable identity and function reflection.
  iree_hal_amd_gpu_profile_executable_t* executable;
  // Static attribution; invocation identity and timestamps are filled later.
  iree_hal_profile_dispatch_event_t event;
} iree_hal_amd_gpu_profile_dispatch_t;

// Native dispatch interval owned through host retirement. PM4 samples bracket
// dispatch issue and completion joins; AQL samples are written by firmware in
// native completion signals. Raw ticks use the queue's declared frequency
// without an inferred host epoch and do not describe wave occupancy.
typedef struct iree_hal_amd_gpu_profile_sample_t {
  // Native dispatch-start clock tick.
  uint64_t begin;
  // Native dispatch-end clock tick.
  uint64_t end;
} iree_hal_amd_gpu_profile_sample_t;

// A PM4 dispatch of a counter-capable owner extends its sample with two raw
// counter samples, taken before the start tick and after the end tick. Their
// layout belongs to the counter selection the owner was prepared with.

// An independently retiring dispatch range. Its native owner keeps samples
// alive until completion copies them into the bounded host stream. Chunked
// publication stores this view in each retirement record, not the shared
// invocation: later publication can overlap earlier retirement.
typedef struct iree_hal_amd_gpu_profile_completion_t {
  // First timestamp pair, or NULL for metadata-only capture.
  const iree_hal_amd_gpu_profile_sample_t* samples;
  // Byte distance between timestamp pairs; zero for metadata-only capture.
  // Bytes beyond the pair hold the dispatch's two raw counter samples.
  uint32_t sample_stride;
  // First dispatch in the immutable command metadata, or zero for direct work.
  uint32_t first;
  // Completed dispatches in this range, including metadata-only dispatches.
  uint32_t count;
  // Native sample-storage prefix returned after host collection; zero when
  // this range has no timestamp storage.
  uint64_t samples_end;
} iree_hal_amd_gpu_profile_completion_t;

// Host-only command metadata retained independently of resident IB storage.
typedef struct iree_hal_amd_gpu_profile_commands_t
    iree_hal_amd_gpu_profile_commands_t;
iree_status_t iree_hal_amd_gpu_profile_commands_create(
    const iree_hal_profile_command_buffer_record_t* record,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_commands_t** out_commands);
void iree_hal_amd_gpu_profile_commands_retain(
    iree_hal_amd_gpu_profile_commands_t* commands);
void iree_hal_amd_gpu_profile_commands_release(
    iree_hal_amd_gpu_profile_commands_t* commands);
// Appends while recording and retains dispatch.executable. After sealing, all
// consumers only read the array; no invocation revisits native command bytes.
iree_status_t iree_hal_amd_gpu_profile_commands_append(
    iree_hal_amd_gpu_profile_commands_t* commands,
    iree_hal_amd_gpu_profile_dispatch_t dispatch);
uint32_t iree_hal_amd_gpu_profile_commands_count(
    const iree_hal_amd_gpu_profile_commands_t* commands);

// Optional inline capture metadata, present only during a profiling session.
// The operation's publication owner alone updates this record. A native
// operation may publish several chunks but contributes one public queue event.
typedef struct iree_hal_amd_gpu_profile_operation_t {
  // Public identity and admission facts, copied to the stream at readiness.
  iree_hal_profile_queue_event_t event;
  // Borrowed host metadata for recorded dispatches, or NULL for direct work.
  iree_hal_amd_gpu_profile_commands_t* commands;
  // Direct dispatch attribution; executable metadata is borrowed by capture.
  iree_hal_amd_gpu_profile_dispatch_t dispatch;
  // Whole-operation completion facts. Chunked execution derives independent
  // subranges and sample views in each native retirement record.
  iree_hal_amd_gpu_profile_completion_t completion;
  // The stream has received this operation, including a possible ring drop.
  bool recorded;
  // This operation's publication carries the session's claimed thread-trace
  // bracket; completion hands the written extent to the next flush.
  bool traced;
  // A claim for this operation was refused because the bracket was held. Set
  // once, so publication retries count one untraced operation, not several.
  bool trace_refused;
  // This operation's publication starts, samples and stops the session's
  // hardware counters; completion turns its raw samples into records.
  bool counted;
  // A counter claim for this operation was refused. Set once, like
  // trace_refused.
  bool counter_refused;
  // Dispatches of this operation the session's sample bound still admits.
  uint32_t counter_grant;
} iree_hal_amd_gpu_profile_operation_t;

// Device-owned native resources of the session's single thread-trace bracket.
// The device prepares them before queues join the session and releases them
// after the session ends. Buffers are GPU-writable and host-readable; the
// hardware writes trace bytes directly and the stop sequence fills the info
// words. Only one bracket exists per device because the trace registers are
// device-wide.
typedef struct iree_hal_amd_gpu_profile_trace_storage_t {
  // Validated bracket naming every buffer and the info words.
  iree_hal_amd_gpu_pm4_thread_trace_config_t config;
  // Host view of each traced engine's buffer, indexed by engine.
  const uint8_t* buffers[IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MAX_ENGINE_COUNT];
  // Host view of the info words, INFO_WORD_COUNT per engine.
  uint32_t* info_words;
  // Upper bound on traced operations in the session.
  uint32_t trace_limit;
  // A stable device power state is held for the session.
  bool stable_power_state;
} iree_hal_amd_gpu_profile_trace_storage_t;

// Installs the bracket of a session that admitted executable traces. Called
// once, before any queue borrows the session. The storage is borrowed until
// the session ends. Without it the session traces nothing.
void iree_hal_amd_gpu_profile_set_trace_storage(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_trace_storage_t* storage);

// True when the session admitted the executable-trace family.
bool iree_hal_amd_gpu_profile_traces_executables(
    const iree_hal_amd_gpu_profile_t* profile);

// Claims the device's bracket for a captured operation about to be published.
// Returns NULL when the session does not trace, the operation is not captured
// or matches no trace predicate, the trace bound is reached, or an earlier
// trace is still in flight or awaiting flush. A claim is exclusive across
// queues. The publisher either publishes both sequences around the operation
// and sets operation->traced, or returns the claim with trace_cancel. A refusal
// by a held bracket is recorded on the operation, which the publisher keeps
// across its retries. Performs no allocation, sink call or native call.
const iree_hal_amd_gpu_pm4_thread_trace_config_t*
iree_hal_amd_gpu_profile_trace_claim(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation);

// Returns an unpublished claim; the bracket and its budget are available again.
void iree_hal_amd_gpu_profile_trace_cancel(iree_hal_amd_gpu_profile_t* profile);

// Device-owned resources of a session that samples hardware counters. The
// device prepares them before queues join the session and releases them after
// the session ends.
typedef struct iree_hal_amd_gpu_profile_counter_storage_t {
  // Validated selection shared by every sequence and raw sample.
  iree_hal_amd_gpu_pm4_perf_counter_config_t config;
  // Layout identity of |config|; recordings prepared with another one are not
  // sampled.
  uint64_t signature;
  // GPU address and host view of the STATE_WORD_COUNT words every start
  // sequence copies. Host-initialized to the unwritten pattern once.
  uint64_t state_address;
  const uint32_t* state_words;
  // Upper bound on sampled dispatches in the session.
  uint64_t sample_limit;
  // A stable device power state is held for the session.
  bool stable_power_state;
} iree_hal_amd_gpu_profile_counter_storage_t;

// Installs the counter selection of a session that admitted counter samples
// and allocates each registered queue's sample records. Called once, before
// any queue borrows the session. The storage is borrowed until the session
// ends. Without it the session samples nothing.
iree_status_t iree_hal_amd_gpu_profile_set_counter_storage(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_counter_storage_t* storage);

// True when the session admitted the counter-sample family.
bool iree_hal_amd_gpu_profile_counts_dispatches(
    const iree_hal_amd_gpu_profile_t* profile);

// Bytes a directly published dispatch reserves after its timestamp pair: two
// raw counter samples, or zero when the session samples no counters.
uint32_t iree_hal_amd_gpu_profile_counter_sample_byte_length(
    const iree_hal_amd_gpu_profile_t* profile);

// Admits a captured operation about to be published into a sampled range.
// Returns NULL when the session does not sample, the operation is not captured
// or matches no predicate, the sample bound is reached, or another queue holds
// the range. A recording passes the layout signature it was prepared with and
// is refused unless that is the session's; a direct dispatch passes NULL.
//
// Counting is per queue but the registers are device-wide, so the range is
// exclusive across queues: operations of the owning queue share it, because
// one ring serializes them, and every other queue's operations run uncounted
// until the owner's last sampled operation completes. That is what makes a
// sample hold exactly its own dispatch among this device's queues.
//
// The publisher either publishes the start, sample and stop sequences around
// the operation and sets operation->counted, or returns the claim with
// counter_cancel. Performs no allocation, sink call or native call.
const iree_hal_amd_gpu_profile_counter_storage_t*
iree_hal_amd_gpu_profile_counter_claim(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation,
    const uint64_t* recorded_signature);

// Returns an unpublished claim: the range and the bound are available again.
void iree_hal_amd_gpu_profile_counter_cancel(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation);

// Reports a matching operation the publisher has to run unsampled for a
// reason of its own, such as sample sequences its ring cannot hold. Counted
// once per operation, like a refused claim.
void iree_hal_amd_gpu_profile_counter_skip(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation);

// Begins the requested capture and emits device/queue metadata. Unsupported
// families fail before starting a sink session. An empty request returns NULL.
// The filter pattern is copied and the sink remains retained until end.
// Executable traces and counter samples are admitted here and need
// set_trace_storage and set_counter_storage to produce data.
iree_status_t iree_hal_amd_gpu_profile_begin(
    const iree_hal_device_profiling_options_t* options,
    iree_hal_profile_session_id_t session_id, uint32_t queue_count,
    uint64_t timestamp_frequency_hz, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_t** out_profile);

// Prepares one device-assigned coordinate before the queue accepts work.
// Existing streams and their completed records remain stable through queue
// replacement. The caller serializes registration per coordinate; registration
// may overlap flush and other producers. Metadata is emitted by explicit flush
// before any events from the new stream. No sink callback runs during
// registration. NULL is a no-op; failure leaves the coordinate unregistered.
iree_status_t iree_hal_amd_gpu_profile_register_queue(
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal);

// These session queries consume immutable captured options and do no work.
bool iree_hal_amd_gpu_profile_captures_dispatches(
    const iree_hal_amd_gpu_profile_t* profile);
bool iree_hal_amd_gpu_profile_timestamps_dispatches(
    const iree_hal_amd_gpu_profile_t* profile);

// Selects an entire recording using predicates known without visiting its
// commands. Individual command/function predicates are evaluated at retirement
// using retained metadata; the native sample layout stays unchanged. False
// suppresses capture and its metadata requirement for the whole invocation.
bool iree_hal_amd_gpu_profile_captures_commands(
    const iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    uint64_t command_buffer_id);

// Selects dispatch artifacts using all active predicates. Direct calls use
// command_buffer_id=0 and cannot match a command-index predicate. Names are
// borrowed for this call; no allocation or executable lookup is performed.
bool iree_hal_amd_gpu_profile_captures_dispatch(
    const iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    uint64_t command_buffer_id, uint32_t command_index,
    iree_string_view_t function_name);

// Called before returning native sample-storage credit or captured resource
// references. Copies completed samples and immutable attribution to bounded
// host rings, retaining only host metadata. No sink/native call or allocation.
void iree_hal_amd_gpu_profile_operation_complete(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_operation_t* operation,
    const iree_hal_amd_gpu_profile_completion_t* completion);

// Writes stable stream snapshots outside their mutexes. Failed writes retain
// the snapshot for retry. The caller serializes flush/end with each other.
iree_status_t iree_hal_amd_gpu_profile_flush(
    iree_hal_amd_gpu_profile_t* profile);

// Flushes, ends, and frees the session even when sink callbacks fail.
iree_status_t iree_hal_amd_gpu_profile_end(iree_hal_amd_gpu_profile_t* profile);

// Initializes identity at acceptance. The caller fills type-specific fields
// before transferring the capture to its execution owner.
void iree_hal_amd_gpu_profile_operation_initialize(
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    iree_hal_profile_queue_event_type_t type, iree_host_size_t wait_count,
    iree_host_size_t signal_count, iree_host_size_t operation_count,
    iree_hal_amd_gpu_profile_operation_t* operation);

// Records actual software deferral before first native publication/readiness.
static inline void iree_hal_amd_gpu_profile_operation_defer(
    iree_hal_amd_gpu_profile_operation_t* operation) {
  if (operation && !operation->recorded) {
    operation->event.flags |=
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
    operation->event.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
  }
}

// Publishes an operation once. Readiness is host-observable admission to the
// executor, not GPU execution time. Failed admission uses ready=false. Native
// wait packets are counted independently of a prior host deferral.
void iree_hal_amd_gpu_profile_operation_record(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation, bool ready,
    uint32_t native_wait_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_PROFILE_H_
