// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/ipc_event.h"

#include <cstring>
#include <utility>

#include "common/internal.h"
#include "common/stream.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  Cleanup cleanup_;
};
template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

struct FakeIpcEventStats {
  iree_hal_semaphore_t* proxy_semaphore = nullptr;
  int record_begin_count = 0;
  int record_commit_count = 0;
  int record_abort_count = 0;
  int active_record_state_count = 0;
  int query_count = 0;
  int synchronize_count = 0;
  int wait_reserve_count = 0;
  int wait_arm_count = 0;
  int wait_commit_count = 0;
  int wait_abort_count = 0;
  int active_wait_state_count = 0;
  bool carrier_active = true;
  bool signal_on_commit = true;
  bool fail_wait_commit = false;
};

struct FakeIpcEvent {
  iree_hal_streaming_ipc_event_t base;
  FakeIpcEventStats* stats;
  iree_allocator_t host_allocator;
};

struct FakeIpcRecord {
  FakeIpcEventStats* stats;
  iree_hal_semaphore_t* semaphore;
  iree_allocator_t host_allocator;
};

struct FakeIpcWait {
  FakeIpcEventStats* stats;
  iree_hal_semaphore_t* semaphore;
  iree_allocator_t host_allocator;
  bool armed;
};

FakeIpcEvent* CastFakeIpcEvent(iree_hal_streaming_ipc_event_t* base) {
  return reinterpret_cast<FakeIpcEvent*>(base);
}

void FakeIpcEventDestroy(iree_hal_streaming_ipc_event_t* base) {
  FakeIpcEvent* event = CastFakeIpcEvent(base);
  iree_allocator_free(event->host_allocator, event);
}

iree_status_t FakeIpcEventBeginRecord(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_semaphore_t* recorded_semaphore, uint64_t recorded_value,
    iree_hal_streaming_ipc_event_record_state_t* out_record_state) {
  FakeIpcEvent* event = CastFakeIpcEvent(base);
  ++event->stats->record_begin_count;
  *out_record_state = nullptr;
  if (!recorded_semaphore || !recorded_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "record point is empty");
  }

  FakeIpcRecord* record = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(event->host_allocator, sizeof(*record),
                            reinterpret_cast<void**>(&record)));
  record->stats = event->stats;
  record->semaphore = recorded_semaphore;
  iree_hal_semaphore_retain(recorded_semaphore);
  record->host_allocator = event->host_allocator;
  ++event->stats->active_record_state_count;
  *out_record_state = record;
  return iree_ok_status();
}

void ReleaseFakeIpcRecord(FakeIpcRecord* record, bool committed) {
  if (committed) {
    ++record->stats->record_commit_count;
  } else {
    ++record->stats->record_abort_count;
  }
  --record->stats->active_record_state_count;
  iree_hal_semaphore_release(record->semaphore);
  iree_allocator_free(record->host_allocator, record);
}

void FakeIpcEventCommitRecord(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_record_state_t record_state) {
  (void)base;
  ReleaseFakeIpcRecord(reinterpret_cast<FakeIpcRecord*>(record_state),
                       /*committed=*/true);
}

void FakeIpcEventAbortRecord(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_record_state_t record_state) {
  (void)base;
  ReleaseFakeIpcRecord(reinterpret_cast<FakeIpcRecord*>(record_state),
                       /*committed=*/false);
}

iree_status_t FakeIpcEventQuery(iree_hal_streaming_ipc_event_t* base,
                                int* out_status) {
  FakeIpcEvent* event = CastFakeIpcEvent(base);
  ++event->stats->query_count;
  *out_status = 0;
  return iree_ok_status();
}

iree_status_t FakeIpcEventSynchronize(iree_hal_streaming_ipc_event_t* base) {
  FakeIpcEvent* event = CastFakeIpcEvent(base);
  ++event->stats->synchronize_count;
  return iree_ok_status();
}

iree_status_t FakeIpcEventReserveWait(
    iree_hal_streaming_ipc_event_t* base, iree_hal_device_t* destination_device,
    iree_hal_semaphore_t** out_semaphore, uint64_t* out_value,
    iree_hal_streaming_ipc_event_wait_state_t* out_wait_state) {
  FakeIpcEvent* event = CastFakeIpcEvent(base);
  ++event->stats->wait_reserve_count;
  *out_semaphore = nullptr;
  *out_value = 0;
  *out_wait_state = nullptr;

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      destination_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  FakeIpcWait* wait = nullptr;
  iree_status_t status = iree_allocator_malloc(
      event->host_allocator, sizeof(*wait), reinterpret_cast<void**>(&wait));
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_release(semaphore);
    return status;
  }
  wait->stats = event->stats;
  wait->semaphore = semaphore;
  iree_hal_semaphore_retain(semaphore);
  wait->host_allocator = event->host_allocator;
  wait->armed = false;
  ++event->stats->active_wait_state_count;

  IREE_ASSERT(event->stats->proxy_semaphore == nullptr);
  event->stats->proxy_semaphore = semaphore;
  iree_hal_semaphore_retain(semaphore);
  *out_semaphore = semaphore;
  *out_value = 1;
  *out_wait_state = wait;
  return iree_ok_status();
}

bool FakeIpcEventArmWait(iree_hal_streaming_ipc_event_t* base,
                         iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  FakeIpcEvent* event = CastFakeIpcEvent(base);
  FakeIpcWait* wait = reinterpret_cast<FakeIpcWait*>(wait_state);
  ++event->stats->wait_arm_count;
  wait->armed = event->stats->carrier_active;
  return wait->armed;
}

iree_status_t FakeIpcEventCommitWait(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  (void)base;
  FakeIpcWait* wait = reinterpret_cast<FakeIpcWait*>(wait_state);
  ++wait->stats->wait_commit_count;
  iree_status_t status = iree_ok_status();
  if (wait->armed && wait->stats->fail_wait_commit) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected IPC wait activation failure");
    iree_hal_semaphore_fail(wait->semaphore, iree_status_clone(status));
  } else if (wait->armed && wait->stats->signal_on_commit) {
    status = iree_hal_semaphore_signal(wait->semaphore, 1, nullptr);
  }
  --wait->stats->active_wait_state_count;
  iree_hal_semaphore_release(wait->semaphore);
  iree_allocator_free(wait->host_allocator, wait);
  return status;
}

void FakeIpcEventAbortWait(
    iree_hal_streaming_ipc_event_t* base,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  (void)base;
  FakeIpcWait* wait = reinterpret_cast<FakeIpcWait*>(wait_state);
  ++wait->stats->wait_abort_count;
  --wait->stats->active_wait_state_count;
  iree_hal_semaphore_release(wait->semaphore);
  iree_allocator_free(wait->host_allocator, wait);
}

iree_status_t FakeIpcEventExportToken(
    iree_hal_streaming_ipc_event_t* base,
    uint8_t out_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE]) {
  (void)base;
  memset(out_token, 0, IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE);
  return iree_ok_status();
}

const iree_hal_streaming_ipc_event_ops_t kFakeIpcEventOps = {
    /*.destroy=*/FakeIpcEventDestroy,
    /*.begin_record=*/FakeIpcEventBeginRecord,
    /*.commit_record=*/FakeIpcEventCommitRecord,
    /*.abort_record=*/FakeIpcEventAbortRecord,
    /*.query=*/FakeIpcEventQuery,
    /*.synchronize=*/FakeIpcEventSynchronize,
    /*.reserve_wait=*/FakeIpcEventReserveWait,
    /*.arm_wait=*/FakeIpcEventArmWait,
    /*.commit_wait=*/FakeIpcEventCommitWait,
    /*.abort_wait=*/FakeIpcEventAbortWait,
    /*.export_token=*/FakeIpcEventExportToken,
};

iree_status_t AttachFakeIpcEvent(iree_hal_streaming_event_t* event,
                                 FakeIpcEventStats* stats) {
  iree_allocator_t host_allocator = iree_allocator_system();
  FakeIpcEvent* ipc_event = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*ipc_event),
                            reinterpret_cast<void**>(&ipc_event)));
  ipc_event->base.ops = &kFakeIpcEventOps;
  ipc_event->stats = stats;
  ipc_event->host_allocator = host_allocator;
  iree_status_t status =
      iree_hal_streaming_event_attach_ipc_event(event, &ipc_event->base);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, ipc_event);
  }
  return status;
}

iree_status_t ReleaseFakeIpcProxy(FakeIpcEventStats* stats, bool resolve) {
  if (!stats->proxy_semaphore) return iree_ok_status();
  iree_status_t status = iree_ok_status();
  if (resolve) {
    uint64_t value = 0;
    status = iree_hal_semaphore_query(stats->proxy_semaphore, &value);
    if (iree_status_is_ok(status) && value < 1) {
      status = iree_hal_semaphore_signal(stats->proxy_semaphore, 1, nullptr);
    }
  }
  iree_hal_semaphore_release(stats->proxy_semaphore);
  stats->proxy_semaphore = nullptr;
  return status;
}

struct RejectBarrierQueue {
  iree_hal_queue_t base;
  iree_hal_queue_t* delegate;
  int reject_barrier_on_call;
  int barrier_count;
  // Number of flush operations observed by this wrapper.
  int flush_count;
};

RejectBarrierQueue* CastRejectBarrierQueue(iree_hal_queue_t* base) {
  return reinterpret_cast<RejectBarrierQueue*>(base);
}

void RejectBarrierQueueDestroy(iree_hal_queue_t* base) {
  RejectBarrierQueue* queue = CastRejectBarrierQueue(base);
  iree_hal_queue_release(queue->delegate);
  queue->delegate = nullptr;
}

iree_status_t RejectBarrierQueueBarrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_queue_barrier_flags_t flags) {
  RejectBarrierQueue* queue = CastRejectBarrierQueue(base);
  ++queue->barrier_count;
  if (queue->barrier_count == queue->reject_barrier_on_call) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected queue barrier rejection");
  }
  return iree_hal_queue_barrier(queue->delegate, wait_semaphore_list,
                                signal_semaphore_list, flags);
}

iree_status_t RejectBarrierQueueExecute(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  RejectBarrierQueue* queue = CastRejectBarrierQueue(base);
  return iree_hal_queue_execute(queue->delegate, wait_semaphore_list,
                                signal_semaphore_list, command_buffer,
                                binding_table, flags);
}

iree_status_t RejectBarrierQueueHostCall(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  RejectBarrierQueue* queue = CastRejectBarrierQueue(base);
  return iree_hal_queue_host_call(queue->delegate, wait_semaphore_list,
                                  signal_semaphore_list, call, args, flags);
}

iree_status_t RejectBarrierQueueFlush(iree_hal_queue_t* base) {
  RejectBarrierQueue* queue = CastRejectBarrierQueue(base);
  ++queue->flush_count;
  return iree_hal_queue_flush(queue->delegate);
}

const iree_hal_queue_vtable_t kRejectBarrierQueueVtable = {
    /*.destroy=*/RejectBarrierQueueDestroy,
    /*.barrier=*/RejectBarrierQueueBarrier,
    /*.execute=*/RejectBarrierQueueExecute,
    /*.host_call=*/RejectBarrierQueueHostCall,
    /*.query_dispatch_concurrency=*/nullptr,
    /*.dispatch=*/nullptr,
    /*.atomic_wait=*/nullptr,
    /*.atomic_store=*/nullptr,
    /*.atomic_rmw=*/nullptr,
    /*.timestamp=*/nullptr,
    /*.flush=*/RejectBarrierQueueFlush,
};

void InitializeRejectBarrierQueue(iree_hal_queue_t* delegate,
                                  int reject_barrier_on_call,
                                  RejectBarrierQueue* out_queue) {
  out_queue->delegate = delegate;
  out_queue->reject_barrier_on_call = reject_barrier_on_call;
  out_queue->barrier_count = 0;
  out_queue->flush_count = 0;
  iree_hal_queue_retain(delegate);
  iree_hal_queue_params_t queue_params;
  iree_hal_queue_params_initialize(&queue_params);
  queue_params.priority = iree_hal_queue_priority(delegate);
  queue_params.features = iree_hal_queue_features(delegate);
  queue_params.execution_resources =
      iree_hal_queue_execution_resources(delegate);
  iree_hal_queue_initialize(iree_hal_queue_family(delegate), &queue_params,
                            &kRejectBarrierQueueVtable, &out_queue->base);
}

class IpcEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device_)));
    InitializeDeviceEntry(&device_entry_);
    IREE_ASSERT_OK(CreateContext(&device_entry_, &context_));
  }

  void TearDown() override {
    iree_hal_streaming_context_release(context_);
    DeinitializeDeviceEntry(&device_entry_);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  void InitializeDeviceEntry(iree_hal_streaming_device_t* device_entry) {
    memset(device_entry, 0, sizeof(*device_entry));
    device_entry->hrx_device = hrx_device_;
    device_entry->hal_device = hrx_device_hal(hrx_device_);
    iree_slim_mutex_initialize(&device_entry->primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry->graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry->block_pool);
  }

  void DeinitializeDeviceEntry(iree_hal_streaming_device_t* device_entry) {
    iree_arena_block_pool_deinitialize(&device_entry->block_pool);
    iree_slim_mutex_deinitialize(&device_entry->graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry->primary_context_mutex);
  }

  iree_status_t CreateContext(iree_hal_streaming_device_t* device_entry,
                              iree_hal_streaming_context_t** out_context) {
    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    return iree_hal_streaming_context_create(
        device_entry, context_flags, iree_allocator_system(), out_context);
  }

  iree_status_t CreateNonBlockingStream(
      iree_hal_streaming_context_t* context,
      iree_hal_streaming_stream_t** out_stream) {
    IREE_ASSERT_ARGUMENT(context);
    IREE_ASSERT_ARGUMENT(out_stream);
    *out_stream = nullptr;
    const iree_hal_queue_family_t* queue_family =
        iree_hal_device_queue_family(context->device, /*family_ordinal=*/0);
    if (IREE_UNLIKELY(!queue_family)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "task device has no queue family");
    }
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    iree_hal_queue_t* queue = nullptr;
    iree_status_t status =
        iree_hal_queue_acquire(queue_family, &queue_params, &queue);
    iree_hal_streaming_stream_t* stream = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_stream_create(
          context, queue, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING,
          /*priority=*/0, iree_allocator_system(), &stream);
    }
    iree_hal_queue_release(queue);
    if (iree_status_is_ok(status)) *out_stream = stream;
    return status;
  }

  hrx_device_t hrx_device_ = nullptr;
  iree_hal_streaming_device_t device_entry_ = {};
  iree_hal_streaming_context_t* context_ = nullptr;
};

TEST_F(IpcEventTest, QueryAndSynchronizeIgnoreLocalProducerFailure) {
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] { iree_hal_streaming_event_release(event); });
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  iree_hal_semaphore_t* failed_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &failed_semaphore));
  iree_hal_semaphore_fail(
      failed_semaphore,
      iree_make_status(IREE_STATUS_ABORTED, "injected local producer failure"));
  iree_hal_streaming_recorded_point_t failed_point = {
      /*.semaphore=*/failed_semaphore,
      /*.value=*/1,
  };
  iree_hal_streaming_event_displaced_capture_t displaced_capture =
      iree_hal_streaming_event_commit_recorded_point(event, failed_point);
  EXPECT_EQ(nullptr, displaced_capture.graph);
  EXPECT_EQ(nullptr, displaced_capture.recording_stream);
  iree_hal_streaming_event_release_displaced_capture(&displaced_capture);

  int query_status = 1;
  IREE_EXPECT_OK(iree_hal_streaming_event_query(event, &query_status));
  EXPECT_EQ(0, query_status);
  IREE_EXPECT_OK(iree_hal_streaming_event_synchronize(event));
  EXPECT_EQ(1, stats.query_count);
  EXPECT_EQ(1, stats.synchronize_count);
}

TEST_F(IpcEventTest, RejectedRecordAbortsPreparedState) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] {
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  iree_hal_queue_t* delegate = stream->queue;
  RejectBarrierQueue rejecting_queue;
  InitializeRejectBarrierQueue(delegate, /*reject_barrier_on_call=*/1,
                               &rejecting_queue);
  stream->queue = &rejecting_queue.base;
  ScopeExit restore_queue([&] {
    stream->queue = delegate;
    iree_hal_queue_release(&rejecting_queue.base);
  });

  iree_status_t status = iree_hal_streaming_event_record(event, stream);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(1, rejecting_queue.barrier_count);
  EXPECT_EQ(1, stats.record_begin_count);
  EXPECT_EQ(0, stats.record_commit_count);
  EXPECT_EQ(1, stats.record_abort_count);
  EXPECT_EQ(0, stats.active_record_state_count);
  EXPECT_EQ(0u, stream->pending_value);

  iree_hal_streaming_recorded_point_t point;
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_EQ(nullptr, point.semaphore);
  iree_hal_streaming_event_release_recorded_point(&point);
}

TEST_F(IpcEventTest, SameDeviceDifferentContextOrdinaryRecordRejects) {
  iree_hal_streaming_context_t* recording_context = nullptr;
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  ScopeExit cleanup([&] {
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
    iree_hal_streaming_context_release(recording_context);
  });
  IREE_ASSERT_OK(CreateContext(&device_entry_, &recording_context));
  IREE_ASSERT_OK(CreateNonBlockingStream(recording_context, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));

  EXPECT_FALSE(
      iree_hal_streaming_event_can_record_in_context(event, recording_context));
  iree_status_t status = iree_hal_streaming_event_record(event, stream);
  EXPECT_EQ(IREE_STATUS_INCOMPATIBLE, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(0u, stream->pending_value);

  iree_hal_streaming_recorded_point_t point;
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_EQ(nullptr, point.semaphore);
  iree_hal_streaming_event_release_recorded_point(&point);
}

TEST_F(IpcEventTest, SameDeviceDifferentContextRecordCommits) {
  iree_hal_streaming_context_t* recording_context = nullptr;
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] {
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
    iree_hal_streaming_context_release(recording_context);
  });
  IREE_ASSERT_OK(CreateContext(&device_entry_, &recording_context));
  IREE_ASSERT_OK(CreateNonBlockingStream(recording_context, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  EXPECT_TRUE(
      iree_hal_streaming_event_can_record_in_context(event, recording_context));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event, stream));
  EXPECT_EQ(1, stats.record_begin_count);
  EXPECT_EQ(1, stats.record_commit_count);
  EXPECT_EQ(0, stats.record_abort_count);
  EXPECT_EQ(0, stats.active_record_state_count);
  ASSERT_EQ(nullptr, event->capture_graph);
  ASSERT_EQ(nullptr, event->recording_stream);
  ASSERT_EQ(0u, event->capture_id);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));

  iree_hal_streaming_recorded_point_t point;
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_NE(nullptr, point.semaphore);
  iree_hal_streaming_event_release_recorded_point(&point);

  // Mirror public stream and context destruction while the event remains
  // live. A submitted record must carry its lifetime only through the point;
  // retaining this stream would leave its raw context pointer dangling.
  iree_hal_streaming_context_unregister_stream(recording_context, stream);
  iree_hal_streaming_stream_release(stream);
  stream = nullptr;
  iree_hal_streaming_context_release(recording_context);
  recording_context = nullptr;
  int query_status = 1;
  IREE_EXPECT_OK(iree_hal_streaming_event_query(event, &query_status));
  EXPECT_EQ(0, query_status);
  iree_hal_streaming_event_release(event);
  event = nullptr;
}

TEST_F(IpcEventTest, CapturedOrdinaryRecordRetainsRecordingStream) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  bool capture_active = false;
  ScopeExit cleanup([&] {
    if (capture_active) {
      IREE_EXPECT_OK(iree_hal_streaming_end_capture(stream, &captured_graph));
    }
    iree_hal_streaming_graph_release(captured_graph);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  capture_active = true;

  IREE_ASSERT_OK(iree_hal_streaming_event_record(event, stream));
  EXPECT_EQ(stream, event->recording_stream);
  EXPECT_EQ(stream->capture_graph, event->capture_graph);

  IREE_ASSERT_OK(iree_hal_streaming_end_capture(stream, &captured_graph));
  capture_active = false;
}

TEST_F(IpcEventTest, DifferentDeviceEntryRecordRejectsAnUnrecordedEvent) {
  iree_hal_streaming_device_t other_device_entry;
  InitializeDeviceEntry(&other_device_entry);
  iree_hal_streaming_context_t* recording_context = nullptr;
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] {
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
    iree_hal_streaming_context_release(recording_context);
    DeinitializeDeviceEntry(&other_device_entry);
  });
  IREE_ASSERT_OK(CreateContext(&other_device_entry, &recording_context));
  IREE_ASSERT_OK(CreateNonBlockingStream(recording_context, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  EXPECT_FALSE(
      iree_hal_streaming_event_can_record_in_context(event, recording_context));
  iree_status_t status = iree_hal_streaming_event_record(event, stream);
  EXPECT_EQ(IREE_STATUS_INCOMPATIBLE, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(0, stats.record_begin_count);
  EXPECT_EQ(0, stats.record_commit_count);
  EXPECT_EQ(0, stats.record_abort_count);
  EXPECT_EQ(0u, stream->pending_value);

  iree_hal_streaming_recorded_point_t point;
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_EQ(nullptr, point.semaphore);
  iree_hal_streaming_event_release_recorded_point(&point);
}

TEST_F(IpcEventTest, ContextWideRecordRejectsBeforeAdapterPreparation) {
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] { iree_hal_streaming_event_release(event); });
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  iree_status_t status =
      iree_hal_streaming_context_record_event(context_, event);
  EXPECT_EQ(IREE_STATUS_UNIMPLEMENTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(0, stats.record_begin_count);
  EXPECT_EQ(0, stats.record_commit_count);
  EXPECT_EQ(0, stats.record_abort_count);

  iree_hal_streaming_recorded_point_t point;
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_EQ(nullptr, point.semaphore);
  iree_hal_streaming_event_release_recorded_point(&point);
}

TEST_F(IpcEventTest, CapturingStreamRejectsIpcRecordBeforeCaptureMutation) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  bool capture_active = false;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] {
    if (capture_active) {
      IREE_EXPECT_OK(iree_hal_streaming_end_capture(stream, &captured_graph));
    }
    iree_hal_streaming_graph_release(captured_graph);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  capture_active = true;
  iree_hal_streaming_graph_t* const original_graph = stream->capture_graph;

  iree_status_t status = iree_hal_streaming_event_record(event, stream);
  EXPECT_EQ(IREE_STATUS_UNIMPLEMENTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(0, stats.record_begin_count);
  EXPECT_EQ(original_graph, stream->capture_graph);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE, stream->capture_status);
  EXPECT_EQ(0u, stream->capture_dependency_count);
  EXPECT_EQ(nullptr, event->capture_graph);

  IREE_ASSERT_OK(iree_hal_streaming_end_capture(stream, &captured_graph));
  capture_active = false;
}

TEST_F(IpcEventTest, CapturingStreamRejectsIpcWaitBeforeCaptureMutation) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  bool capture_active = false;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] {
    if (capture_active) {
      IREE_EXPECT_OK(iree_hal_streaming_end_capture(stream, &captured_graph));
    }
    iree_hal_streaming_graph_release(captured_graph);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  capture_active = true;
  iree_hal_streaming_graph_t* const original_graph = stream->capture_graph;

  for (bool capture_external_wait : {false, true}) {
    iree_status_t status = iree_hal_streaming_stream_wait_event(
        stream, event, capture_external_wait);
    EXPECT_EQ(IREE_STATUS_UNIMPLEMENTED, iree_status_code(status));
    iree_status_free(status);
  }
  EXPECT_EQ(0, stats.wait_reserve_count);
  EXPECT_EQ(0, stats.wait_arm_count);
  EXPECT_EQ(0, stats.wait_commit_count);
  EXPECT_EQ(0, stats.wait_abort_count);
  EXPECT_EQ(original_graph, stream->capture_graph);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE, stream->capture_status);
  EXPECT_EQ(0u, stream->capture_dependency_count);

  IREE_ASSERT_OK(iree_hal_streaming_end_capture(stream, &captured_graph));
  capture_active = false;
}

TEST_F(IpcEventTest, AcceptedDirectWaitCommitsPreparedState) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  stats.signal_on_commit = false;
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseFakeIpcProxy(&stats, /*resolve=*/true));
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_event(
      stream, event, /*capture_external_wait=*/false));
  EXPECT_EQ(1, stats.wait_reserve_count);
  EXPECT_EQ(1, stats.wait_arm_count);
  EXPECT_EQ(1, stats.wait_commit_count);
  EXPECT_EQ(0, stats.wait_abort_count);
  EXPECT_EQ(0, stats.active_wait_state_count);

  int stream_status = 0;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(stream, &stream_status));
  EXPECT_EQ(1, stream_status);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(stats.proxy_semaphore, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));
}

TEST_F(IpcEventTest,
       AcceptedDirectWaitCommitFailureFailsProxyAndFlushesBarrier) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  stats.fail_wait_commit = true;
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseFakeIpcProxy(&stats, /*resolve=*/false));
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  iree_hal_queue_t* delegate = stream->queue;
  RejectBarrierQueue tracking_queue;
  InitializeRejectBarrierQueue(delegate, /*reject_barrier_on_call=*/0,
                               &tracking_queue);
  stream->queue = &tracking_queue.base;
  ScopeExit restore_queue([&] {
    stream->queue = delegate;
    iree_hal_queue_release(&tracking_queue.base);
  });

  const uint64_t initial_pending_value = stream->pending_value;
  iree_status_t status = iree_hal_streaming_stream_wait_event(
      stream, event, /*capture_external_wait=*/false);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(1, tracking_queue.barrier_count);
  EXPECT_EQ(1, stats.wait_reserve_count);
  EXPECT_EQ(1, stats.wait_arm_count);
  EXPECT_EQ(1, stats.wait_commit_count);
  EXPECT_EQ(0, stats.wait_abort_count);
  EXPECT_EQ(0, stats.active_wait_state_count);
  EXPECT_GT(stream->pending_value, initial_pending_value);
  EXPECT_EQ(1, tracking_queue.flush_count);

  ASSERT_NE(nullptr, stats.proxy_semaphore);
  uint64_t proxy_value = 0;
  status = iree_hal_semaphore_query(stats.proxy_semaphore, &proxy_value);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_free(status);
}

TEST_F(IpcEventTest, RejectedDirectWaitAbortsPreparedState) {
  iree_hal_streaming_stream_t* stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  FakeIpcEventStats stats;
  ScopeExit cleanup([&] {
    IREE_EXPECT_OK(ReleaseFakeIpcProxy(&stats, /*resolve=*/true));
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(stream);
  });
  IREE_ASSERT_OK(CreateNonBlockingStream(context_, &stream));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(AttachFakeIpcEvent(event, &stats));

  iree_hal_queue_t* delegate = stream->queue;
  RejectBarrierQueue rejecting_queue;
  InitializeRejectBarrierQueue(delegate, /*reject_barrier_on_call=*/1,
                               &rejecting_queue);
  stream->queue = &rejecting_queue.base;
  ScopeExit restore_queue([&] {
    stream->queue = delegate;
    iree_hal_queue_release(&rejecting_queue.base);
  });

  iree_status_t status = iree_hal_streaming_stream_wait_event(
      stream, event, /*capture_external_wait=*/false);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(1, stats.wait_reserve_count);
  EXPECT_EQ(1, stats.wait_arm_count);
  EXPECT_EQ(0, stats.wait_commit_count);
  EXPECT_EQ(1, stats.wait_abort_count);
  EXPECT_EQ(0, stats.active_wait_state_count);
  EXPECT_EQ(0u, stream->pending_value);
}

}  // namespace
