// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_EVENT_H_
#define LIBHRX_SRC_BINDING_COMMON_EVENT_H_

#include "common/completion.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_event_timestamp_slot_t
    iree_hal_streaming_event_timestamp_slot_t;
typedef struct iree_hal_streaming_graph_t iree_hal_streaming_graph_t;
typedef struct iree_hal_streaming_graph_node_t iree_hal_streaming_graph_node_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;

// Timeline advanced by accepted operations in one binding scheduling domain.
// The semaphore is owned by the containing object and |pending_value| is the
// largest value an accepted queue operation will signal. Callers provide the
// synchronization protecting |pending_value|.
typedef struct iree_hal_streaming_operation_timeline_t {
  // Timeline semaphore signaled by operations in the scheduling domain.
  iree_hal_semaphore_t* semaphore;
  // Largest value an accepted operation will signal.
  uint64_t pending_value;
} iree_hal_streaming_operation_timeline_t;

// Facts converting a pair of device ticks captured on one device into a
// duration. Populated or zeroed as a unit: a zero |frequency_hz| means the
// device advertises no domain whose ticks this layer can convert, and is the
// one state in which a timing-enabled record captures no tick.
typedef struct iree_hal_streaming_timestamp_domain_t {
  // Ticks per second of the domain, or 0 when the device advertises none.
  uint64_t frequency_hz;
  // Number of low bits defined in a tick, in [1, 64]; the counter wraps at
  // this width. Zero exactly when |frequency_hz| is zero.
  uint32_t valid_bits;
} iree_hal_streaming_timestamp_domain_t;

typedef enum iree_hal_streaming_event_flag_bits_e {
  IREE_HAL_STREAMING_EVENT_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_EVENT_FLAG_BLOCKING_SYNC = 1ull << 0,
  IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING = 1ull << 1,
  IREE_HAL_STREAMING_EVENT_FLAG_INTERPROCESS = 1ull << 2,
} iree_hal_streaming_event_flags_t;

// The timeline point a submitted event record names, together with the stream
// timeline point that reaching it implies and the device tick slot the record
// captures into. Published and read as one value so no reader can pair one
// record's semaphore, value or tick with another record's.
//
// A point is either owning or under construction. An owning point holds one
// reference to everything it names and is what every holder outside a record
// path has: iree_hal_streaming_event_acquire_recorded_point produces one,
// iree_hal_streaming_event_commit_recorded_point consumes one, and
// iree_hal_streaming_event_release_recorded_point drops what one names. A
// point under construction names the timeline a record is about to signal and
// owns nothing, which is how a record path builds its point before
// iree_hal_streaming_event_enqueue_record completes it into an owning one.
typedef struct iree_hal_streaming_recorded_point_t {
  // Timeline semaphore the record's submission signals, or NULL when no record
  // has been submitted. Retained by whoever holds the point.
  iree_hal_semaphore_t* semaphore;
  // Value |semaphore| reaches once the recorded work completes, or 0 when
  // |semaphore| is NULL.
  uint64_t value;
  // Stream whose timeline this point is ordered after, or 0 when the point
  // follows no stream timeline point. Identifies the timeline a cross-stream
  // wait on this point can claim ordering against.
  unsigned long long ordered_after_stream_id;
  // Value on |ordered_after_stream_id|'s timeline this point is ordered after,
  // or 0 when there is none. A lower bound, not the point itself: a record
  // inside a graph launch is ordered after the tail the launch waited on,
  // which is earlier than anything the launch signals.
  uint64_t ordered_after_stream_value;
  // True when the record executes outside its context's shared primary queue.
  // A dependent stream must keep its wait off that shared queue so producers
  // submitted later can continue to run.
  bool requires_independent_queue;
  // Slot the device writes this record's tick into at the point |value| names,
  // or NULL when the record captured no tick because timing is disabled on the
  // event or the device advertises no domain. Retained by whoever holds the
  // point; the tick is defined once |semaphore| reaches |value|.
  iree_hal_streaming_event_timestamp_slot_t* timestamp_slot;
} iree_hal_streaming_recorded_point_t;

// Event for synchronization.
typedef struct iree_hal_streaming_event_t {
  // References held by public handles and in-flight operations.
  iree_atomic_ref_count_t ref_count;

  // Event properties.
  iree_hal_streaming_event_flags_t flags;

  // Guards |recorded_point| and |capture_graph|, which move together: a
  // submitted record installs a point and ends any capture association in one
  // transition, so no reader can see the new point while the event still reads
  // as captured. The point carries the record's timeline point and the slot its
  // tick lands in as one value, so no reader can pair one record's point with
  // another record's slot. It does not reach the capture dependency frontier
  // below, whose fields each say what orders them.
  // Acquired after the recording stream's mutex and after the graph
  // executable's mutex; no path takes either while holding this one.
  // Waits and reference releases happen outside it: readers copy and retain
  // what they need under it and drop it once unlocked.
  iree_slim_mutex_t mutex;
  // Point the last submitted record names, or a zeroed point when no record
  // has been submitted. The event owns no timeline: a record names a point on
  // the timeline of whichever submission carries it, and the retained
  // reference in |recorded_point.semaphore| is what keeps a submitted record
  // queryable after the stream or graph executable that carried it is gone.
  iree_hal_streaming_recorded_point_t recorded_point;

  // Context that created the event, retained.
  iree_hal_streaming_context_t* context;

  // Platform-specific IPC handle, if the event is IPC enabled.
  void* ipc_handle;

  // Graph and exact session a capture-time record last associated this event
  // with, retained, or NULL/zero when the last record was submitted. The graph,
  // session ID, dependency pointer/count/capacity, and dependency contents are
  // one value guarded by |mutex|.
  iree_hal_streaming_graph_t* capture_graph;
  // Capture session identifier associated with |capture_graph|.
  unsigned long long capture_id;
  // Frontier of graph nodes the captured event record follows.
  iree_hal_streaming_graph_node_t** capture_dependencies;
  // Number of entries in |capture_dependencies|.
  iree_host_size_t capture_dependency_count;
  // Allocated capacity of |capture_dependencies|.
  iree_host_size_t capture_dependency_capacity;

  // Allocator used for the event and its dependency storage.
  iree_allocator_t host_allocator;
} iree_hal_streaming_event_t;

// Outcome of measuring the interval between two event records. Carried out of
// band from the status because a failed timeline propagates its own status
// verbatim, and that status can carry any code, including whichever one a
// measurement outcome would otherwise have used.
typedef enum iree_hal_streaming_event_timing_e {
  // Both records were reached and the interval between them was measured.
  IREE_HAL_STREAMING_EVENT_TIMING_MEASURED = 0,
  // At least one of the events carries no record to measure, because timing is
  // disabled on it or because no record of it has been submitted.
  IREE_HAL_STREAMING_EVENT_TIMING_UNTIMED,
  // Both events carry a record but at least one has not been reached.
  IREE_HAL_STREAMING_EVENT_TIMING_INCOMPLETE,
  // At least one event's last record went into a stream capture, which records
  // a dependency frontier and no queue point, so it names no time.
  IREE_HAL_STREAMING_EVENT_TIMING_CAPTURED,
  // The device the records were made on advertises no timestamp domain, so no
  // clock the two records share can measure the interval between them.
  IREE_HAL_STREAMING_EVENT_TIMING_UNSUPPORTED,
} iree_hal_streaming_event_timing_t;

// Reads the clock of the queue family used by a context's event records.
// All context queues, including independently acquired record and stream-wait
// queues, belong to this family. Other families do not affect comparability.
// Returns a zeroed domain if the family has no timestamp clock or covers more
// than one physical device. A NULL family publishes no facts.
// Synchronization: none (reads immutable device facts).
iree_hal_streaming_timestamp_domain_t iree_hal_streaming_query_timestamp_domain(
    const iree_hal_queue_family_spec_t* family);

// Creates an event owned by |context|.
// Synchronization: none (creates new event).
iree_status_t iree_hal_streaming_event_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_event_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_event_t** out_event);

// Retains or releases an event reference.
// Synchronization: none (reference counting).
void iree_hal_streaming_event_retain(iree_hal_streaming_event_t* event);
void iree_hal_streaming_event_release(iree_hal_streaming_event_t* event);

// Clears captured metadata when a binding retires the public event handle.
// Graph nodes can retain the event for replay, so the handle must first drop
// its reference back to the capture graph to avoid an ownership cycle.
// Releases the captured graph outside the event mutex.
void iree_hal_streaming_event_clear_capture(iree_hal_streaming_event_t* event);

// Queries whether the event's last submitted record has completed without
// waiting. An event with no submitted record is complete. On failure
// |out_completion_state| is unchanged.
iree_status_t iree_hal_streaming_event_query(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_completion_state_t* out_completion_state);

// Takes a reference to the point |event| was last recorded at, or a zeroed
// point when no record has been submitted. Callers release the point with
// iree_hal_streaming_event_release_recorded_point.
// Synchronization: event (event mutex held while copying the point).
void iree_hal_streaming_event_acquire_recorded_point(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_recorded_point_t* out_point);

// Releases the references |point| holds and zeroes it. Releasing a tick slot
// can return it to its pool, so callers holding the event mutex drop the point
// after unlocking.
// Synchronization: pool (the slot's pool mutex is held while the last
// reference returns it). A caller holding a stream or graph executable mutex
// nests the pool mutex under it.
void iree_hal_streaming_event_release_recorded_point(
    iree_hal_streaming_recorded_point_t* point);

// Adopts |point| as the point |event| is recorded at, consuming the references
// it holds and dropping the references the previous point held. Called only
// once the submission that signals |point| has been accepted, with a point
// iree_hal_streaming_event_enqueue_record completed.
//
// A submitted record ends the event's association with any graph a capture-time
// record left on it, in the same transition, so no reader can see the new point
// while the event still reads as captured. Returns that graph reference for
// release outside the caller's stream or graph executable mutex, keeping
// destruction of graph-owned resources outside submission locks.
// Synchronization: event (event mutex held while replacing the point).
IREE_MUST_USE_RESULT iree_hal_streaming_graph_t*
iree_hal_streaming_event_commit_recorded_point(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_recorded_point_t point);

// Returns whether a capture-time record last associated |event| with a graph.
// An event names none once its last record has been submitted, and none before
// any record has been made.
//
// Answers from the association alone without acquiring ownership or running
// graph destruction when the caller only needs to query capture state.
// Synchronization: event (event mutex held while reading).
bool iree_hal_streaming_event_has_capture_graph(
    iree_hal_streaming_event_t* event);

// Returns a retained reference to the graph and exact session a capture-time
// record last associated |event| with, or NULL/zero when the event names no
// capture. Both outputs are acquired atomically under the event mutex.
// Release the returned reference outside the caller's stream mutex so graph
// destruction runs outside submission locks.
// Synchronization: event (event mutex held while retaining).
IREE_MUST_USE_RESULT iree_hal_streaming_graph_t*
iree_hal_streaming_event_acquire_capture_graph(
    iree_hal_streaming_event_t* event, unsigned long long* out_capture_id);

// Records |event| after the current tails of every stream in |streams|. Each
// stream must belong to the context that created |event| and none may be
// capturing. The caller keeps the borrowed stream references live for the
// duration of the call. The fan-in record is submitted directly on the
// context's lazily acquired event queue and retains no single recording stream.
//
// All records advance the context's event-record timeline. When
// |additional_timeline| is non-NULL the same submission also waits on and
// advances it, and the caller must serialize access to it for the duration of
// the call. Accepted submissions update both timelines before returning OK.
// The selected queue is flushed after acceptance; a flush error preserves the
// accepted timeline values and recorded point.
iree_status_t iree_hal_streaming_event_record_after_streams(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_stream_t* const* streams, iree_host_size_t stream_count,
    iree_hal_streaming_operation_timeline_t* additional_timeline);

// Enqueues |event|'s record on |queue| at the point reached once
// |wait_semaphores| is satisfied, signaling |signal_semaphores| there.
//
// |context| must be the context that created |event|. The record's tick slot
// comes from that context's pool and outlives the record on the point the event
// holds, and nothing the point names keeps that pool alive: only the reference
// the event holds on its own context does. This is the streaming layer's own
// enforcement of the rule, covering callers that have not already decided it.
//
// |point| arrives describing the timeline point that record signals and owning
// nothing. On success it additionally names the slot the device writes this
// record's tick into and holds one reference to everything it names, which the
// caller hands to iree_hal_streaming_event_commit_recorded_point; that call
// consumes them. On failure |point| is left exactly as it arrived, owing
// nothing.
//
// A timing-enabled event on a device advertising a timestamp domain always
// captures a tick: a slot that cannot be obtained fails the record rather than
// leaving it silently untimed. Every other record enqueues a plain barrier.
//
// All submitted record paths enqueue through here, so none can forget the
// timestamp substitution, seat a cross-context record, leak a slot on a
// rejected enqueue, or produce a point owning only part of what it names.
//
// Synchronization: pool (the context's timestamp pool mutex is held while a
// tick slot is acquired, and covers the device allocation a pool growth
// performs). Stream and graph callers hold their submission mutexes across the
// call; a context-wide record has no single stream mutex to hold.
IREE_MUST_USE_RESULT iree_status_t iree_hal_streaming_event_enqueue_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_context_t* context,
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_streaming_recorded_point_t* point);

// Records |event| at the point |stream| has reached. On a stream that is not
// capturing that point is a queue point: |stream| is flushed so the record
// lands behind everything already recorded on it, and the record is enqueued
// there. |stream| must then belong to |event|'s context, or the record is
// refused with IREE_STATUS_INCOMPATIBLE.
//
// A default captured record names the stream's dependency frontier and no
// queue point, so nothing is flushed or enqueued and it is accepted from any
// context. A binding may be stricter: hipEventRecord holds a capturing stream
// to the context rule too, refusing the pair before it reaches here.
// When |capture_external_record| is true, capture also adds an event-record
// node that publishes a submitted event point on each graph launch. Such a
// node requires the event's context to match the graph's. Outside capture,
// both modes record the ordinary stream point.
// Synchronization: stream flush (flushes a stream that is not capturing).
iree_status_t iree_hal_streaming_event_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream,
    bool capture_external_record);

// Waits until the event's last submitted record is signaled.
iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event);

// Converts the interval between two ticks captured in |domain| to milliseconds.
// Only the low |domain.valid_bits| of a tick are defined and the counter wraps
// there, so the difference is reduced modulo that width; a width of 64 makes
// the reduction the identity.
//
// Reading that reduced difference as a signed offset from the counter's top bit
// is this layer's choice and not something the device facts state. It is what
// makes a pair captured in order a positive duration and a reversed pair a
// negative one, and what it costs is that an interval longer than half the
// counter range reports negative: out of reach at 64 bits and 100 MHz, but 21
// seconds on a 32-bit counter at the same rate.
//
// |domain| must be populated; the only caller reaches this through a record
// that captured a tick, which a zeroed domain makes impossible.
// Synchronization: none (pure arithmetic).
float iree_hal_streaming_timestamp_domain_elapsed_ms(
    iree_hal_streaming_timestamp_domain_t domain, uint64_t start_tick,
    uint64_t stop_tick);

// Measures the interval between the records |start| and |stop| name and stores
// it in milliseconds in |*ms|. Writes |*ms| only when |*out_timing| is
// MEASURED; every other outcome leaves it untouched.
//
// |*out_timing| says why no interval was produced and is meaningful only when
// this returns ok. A non-ok status comes from querying a timeline or reading a
// captured tick back and belongs to whatever failed the device, not to the
// events.
//
// Synchronization: both events (each event's mutex held while its record is
// copied; no waiting).
iree_status_t iree_hal_streaming_event_elapsed_time(
    float* ms, iree_hal_streaming_event_t* start,
    iree_hal_streaming_event_t* stop,
    iree_hal_streaming_event_timing_t* out_timing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_EVENT_H_
