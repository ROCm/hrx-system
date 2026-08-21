// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_QUERY_TEST_UTIL_H_
#define IREE_EXPERIMENTAL_STREAMING_QUERY_TEST_UTIL_H_

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "common/internal.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"

namespace iree::hal::stream {

// Timeline value an implementation reports for a timeline that has failed with
// |code|. iree_hal_semaphore_query decodes it back into a status and reports
// IREE_HAL_SEMAPHORE_FAILURE_VALUE as the value, which is above every value
// these fixtures record, so a query that dropped the status would read the
// timeline as reached and answer complete. A bare code owns no allocation, so
// the encoded status needs no consumer to free it.
//
// IREE_STATUS_INTERNAL is rejected: it is what the decoder synthesizes for a
// failure value carrying no encoded status, so a suite asserting on it would
// pass whether or not the code survived the round trip.
inline uint64_t FailedTimelineValue(iree_status_code_t code) {
  EXPECT_NE(code, IREE_STATUS_INTERNAL)
      << "iree_hal_semaphore_failure_as_status synthesizes "
         "IREE_STATUS_INTERNAL for a failure value carrying no encoded status, "
         "so asserting on it cannot show the code made the round trip";
  return iree_hal_status_as_semaphore_failure(iree_status_from_code(code));
}

// HAL semaphore reporting a caller-chosen timeline value from every query.
//
// Standard layout with |resource| first, so the address of a spy is a valid
// iree_hal_semaphore_t. Its storage is automatic, so destruction through the
// HAL is never correct: the destroy entry fails the test, and the destructor
// requires the reference count back at the one initialization left it at. That
// is what makes a query's retain and release observable.
struct TimelineSemaphoreSpy {
  explicit TimelineSemaphoreSpy(uint64_t initial_timeline_value);
  ~TimelineSemaphoreSpy();
  TimelineSemaphoreSpy(const TimelineSemaphoreSpy&) = delete;
  TimelineSemaphoreSpy& operator=(const TimelineSemaphoreSpy&) = delete;

  iree_hal_semaphore_t* get();

  // {ref_count, vtable} header that iree_hal_resource_t and
  // iree_async_semaphore_t share, which is what lets the HAL dispatch to the
  // vtable below.
  iree_hal_resource_t resource;
  // Value every query reports.
  uint64_t timeline_value;
};

// Reports the caller-chosen value. Reached through iree_hal_semaphore_query.
inline uint64_t TimelineSemaphoreSpyQuery(iree_async_semaphore_t* semaphore) {
  return reinterpret_cast<TimelineSemaphoreSpy*>(semaphore)->timeline_value;
}

// Reached only when a release drops the last reference, which for a spy with
// automatic storage means one release too many.
inline void TimelineSemaphoreSpyDestroy(iree_async_semaphore_t* semaphore) {
  (void)semaphore;
  ADD_FAILURE() << "test semaphore released more times than it was retained";
}

// Only destroy and query are ever dispatched: iree_hal_semaphore_query
// dispatches query, and a release that drops the last reference dispatches
// destroy. The remaining entries stay null; no query path reaches them.
inline constexpr iree_hal_semaphore_vtable_t kTimelineSemaphoreSpyVtable = {
    /*.async=*/{
        /*.destroy=*/TimelineSemaphoreSpyDestroy,
        /*.query=*/TimelineSemaphoreSpyQuery,
        /*.signal=*/nullptr,
        /*.on_fail=*/nullptr,
    },
    /*.wait=*/nullptr,
    /*.import_timepoint=*/nullptr,
    /*.export_timepoint=*/nullptr,
};

inline TimelineSemaphoreSpy::TimelineSemaphoreSpy(
    uint64_t initial_timeline_value)
    : timeline_value(initial_timeline_value) {
  iree_hal_resource_initialize(&kTimelineSemaphoreSpyVtable, &resource);
}

inline TimelineSemaphoreSpy::~TimelineSemaphoreSpy() {
  EXPECT_EQ(iree_atomic_ref_count_load(&resource.ref_count), 1)
      << "test semaphore left with an unreleased reference";
}

inline iree_hal_semaphore_t* TimelineSemaphoreSpy::get() {
  return reinterpret_cast<iree_hal_semaphore_t*>(this);
}

// An iree_hal_streaming_stream_t carrying only the timeline state a query
// reads. No context, no device and no command buffer, so the flush a query
// performs first takes the stream mutex and submits nothing.
class QueryableStream {
 public:
  // A stream whose timeline reports |timeline_value| and whose last accepted
  // submission signals |pending_value|.
  QueryableStream(uint64_t timeline_value, uint64_t pending_value);
  ~QueryableStream();
  QueryableStream(const QueryableStream&) = delete;
  QueryableStream& operator=(const QueryableStream&) = delete;

  // Moves the timeline the stream queries to |timeline_value|.
  void set_timeline_value(uint64_t timeline_value);

  iree_hal_streaming_stream_t* get();

 private:
  // Declared first so it is destroyed last, after |stream_| is torn down.
  TimelineSemaphoreSpy semaphore_;
  // The constructor sets the reference count, the mutex, the timeline
  // semaphore and the pending value; the member initializer zeroes the rest.
  // That zeroing is load-bearing: a query also reads command_buffer, through
  // the flush it performs first, and reads and updates completed_value. The
  // reference count is initialized the way iree_hal_streaming_stream_create
  // initializes it, so the fixture holds the one reference a create caller
  // owns; a context here borrows its streams and takes no second reference, so
  // the count moves only when a query moves it.
  iree_hal_streaming_stream_t stream_ = {};
};

inline QueryableStream::QueryableStream(uint64_t timeline_value,
                                        uint64_t pending_value)
    : semaphore_(timeline_value) {
  iree_atomic_ref_count_init(&stream_.ref_count);
  iree_slim_mutex_initialize(&stream_.mutex);
  stream_.timeline_semaphore = semaphore_.get();
  stream_.pending_value = pending_value;
}

// The stream borrows the spy: production has a stream create and own its
// timeline, but the fixture holds this one as a member declared ahead of
// |stream_|, so it already outlives the stream and a retain/release pair would
// only cancel itself. The count the spy checks at teardown therefore moves only
// when a query moves it.
inline QueryableStream::~QueryableStream() {
  // A context query retains every stream in its snapshot and releases it after
  // the walk. No tested path moves the event or context fixture's own count, so
  // this is the only one of the three that has anything to check.
  EXPECT_EQ(iree_atomic_ref_count_load(&stream_.ref_count), 1)
      << "test stream left with an unreleased reference";
  iree_slim_mutex_deinitialize(&stream_.mutex);
}

inline void QueryableStream::set_timeline_value(uint64_t timeline_value) {
  semaphore_.timeline_value = timeline_value;
}

inline iree_hal_streaming_stream_t* QueryableStream::get() { return &stream_; }

// An iree_hal_streaming_event_t carrying only the state a query reads.
class QueryableEvent {
 public:
  // An event with no submitted record.
  QueryableEvent();
  // An event whose record names |record_value| on a timeline reporting
  // |timeline_value|, published through the commit the record paths use.
  QueryableEvent(uint64_t timeline_value, uint64_t record_value);
  ~QueryableEvent();
  QueryableEvent(const QueryableEvent&) = delete;
  QueryableEvent& operator=(const QueryableEvent&) = delete;

  // Moves the timeline the recorded point names to |timeline_value|.
  void set_timeline_value(uint64_t timeline_value);

  iree_hal_streaming_event_t* get();

 private:
  // Declared first so it is destroyed last, after |event_| has released the
  // reference its recorded point holds. An event with no submitted record never
  // names it, which keeps one fixture shape for both cases.
  TimelineSemaphoreSpy semaphore_;
  // The constructor sets the reference count, the mutex and, for a recorded
  // event, the point; the member initializer zeroes the rest. The reference
  // count is initialized the way iree_hal_streaming_event_create initializes
  // it; no query path moves it.
  iree_hal_streaming_event_t event_ = {};
};

inline QueryableEvent::QueryableEvent() : semaphore_(0) {
  iree_atomic_ref_count_init(&event_.ref_count);
  iree_slim_mutex_initialize(&event_.mutex);
}

inline QueryableEvent::QueryableEvent(uint64_t timeline_value,
                                      uint64_t record_value)
    : semaphore_(timeline_value) {
  iree_atomic_ref_count_init(&event_.ref_count);
  iree_slim_mutex_initialize(&event_.mutex);
  const iree_hal_streaming_recorded_point_t point = {
      /*.semaphore=*/semaphore_.get(),
      /*.value=*/record_value,
      /*.ordered_after_stream_id=*/0,
      /*.ordered_after_stream_value=*/0,
      /*.record_time_ns=*/0,
  };
  iree_hal_streaming_event_commit_recorded_point(&event_, point);
}

// Mirrors iree_hal_streaming_event_destroy: the event owns the reference its
// recorded point holds, and the mutex is deinitialized after it.
inline QueryableEvent::~QueryableEvent() {
  iree_hal_semaphore_release(event_.recorded_point.semaphore);
  iree_slim_mutex_deinitialize(&event_.mutex);
}

inline void QueryableEvent::set_timeline_value(uint64_t timeline_value) {
  semaphore_.timeline_value = timeline_value;
}

inline iree_hal_streaming_event_t* QueryableEvent::get() { return &event_; }

// An iree_hal_streaming_context_t carrying only the stream list a query reads.
// The streams are borrowed: production's list retains every stream it names and
// releases it on unregister, but the QueryableStream fixtures here stay owned
// by the caller, which must keep them alive for as long as the context.
class QueryableContext {
 public:
  // A context whose stream list holds |streams| in the order given.
  explicit QueryableContext(std::initializer_list<QueryableStream*> streams);
  ~QueryableContext();
  QueryableContext(const QueryableContext&) = delete;
  QueryableContext& operator=(const QueryableContext&) = delete;

  iree_hal_streaming_context_t* get();

 private:
  // Backing storage for |context_.streams|. Nothing may register a stream into
  // this context: the array is owned by this vector, while
  // iree_hal_streaming_context_register_stream would grow it through
  // |context_.host_allocator|.
  std::vector<iree_hal_streaming_stream_t*> streams_;
  // The constructor sets the reference count, the stream-list mutex, the stream
  // array and the host allocator; the member initializer zeroes the rest. The
  // allocator is load-bearing: a query copies the stream list into an
  // allocation taken from it and frees that allocation through it. The
  // reference count is initialized the way iree_hal_streaming_context_create
  // initializes it; no query path moves it.
  iree_hal_streaming_context_t context_ = {};
};

inline QueryableContext::QueryableContext(
    std::initializer_list<QueryableStream*> streams) {
  streams_.reserve(streams.size());
  for (QueryableStream* stream : streams) {
    streams_.push_back(stream->get());
  }
  iree_atomic_ref_count_init(&context_.ref_count);
  iree_slim_mutex_initialize(&context_.stream_list_mutex);
  context_.streams = streams_.data();
  context_.stream_count = (iree_host_size_t)streams_.size();
  // No query path reads the capacity, but a capacity below the count would
  // describe an array this is not.
  context_.stream_capacity = (iree_host_size_t)streams_.size();
  context_.host_allocator = iree_allocator_system();
}

inline QueryableContext::~QueryableContext() {
  iree_slim_mutex_deinitialize(&context_.stream_list_mutex);
}

inline iree_hal_streaming_context_t* QueryableContext::get() {
  return &context_;
}

}  // namespace iree::hal::stream

#endif  // IREE_EXPERIMENTAL_STREAMING_QUERY_TEST_UTIL_H_
