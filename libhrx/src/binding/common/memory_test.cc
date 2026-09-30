// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/memory.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "common/graph.h"
#include "common/graph_memory.h"
#include "common/internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_LINUX)
static std::atomic<bool> g_fail_next_memory_barrier = false;
static std::atomic<bool> g_emulate_graph_virtual_memory = false;
static std::atomic<bool> g_pause_graph_pointer_prepare = false;
static std::atomic<bool> g_graph_pointer_prepare_entered = false;
static std::atomic<bool> g_allow_graph_pointer_prepare_return = false;
static std::atomic<bool> g_graph_prepare_held_stream_reservation = false;
static std::atomic<iree_hal_streaming_stream_t*>
    g_graph_pointer_prepare_stream = nullptr;
static std::atomic<bool> g_pause_after_graph_pointer_restore = false;
static std::atomic<bool> g_graph_pointer_restore_entered = false;
static std::atomic<bool> g_allow_graph_pointer_restore_return = false;
static std::atomic<iree_hal_streaming_allocation_preparation_t*>
    g_allocation_preparation_await_idle_target = nullptr;
static std::atomic<bool> g_allocation_preparation_await_idle_entered = false;

extern "C" void __real_iree_hal_streaming_allocation_preparation_await_idle(
    iree_hal_streaming_allocation_preparation_t* preparation);

extern "C" void __wrap_iree_hal_streaming_allocation_preparation_await_idle(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  iree_hal_streaming_allocation_preparation_t* expected = preparation;
  if (g_allocation_preparation_await_idle_target.compare_exchange_strong(
          expected, nullptr, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    g_allocation_preparation_await_idle_entered.store(
        true, std::memory_order_release);
  }
  __real_iree_hal_streaming_allocation_preparation_await_idle(preparation);
}

static void ArmAllocationPreparationAwaitIdleObservation(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  g_allocation_preparation_await_idle_entered.store(false,
                                                    std::memory_order_release);
  g_allocation_preparation_await_idle_target.store(preparation,
                                                   std::memory_order_release);
}

static bool WaitForAllocationPreparationAwaitIdleEntry(
    const std::atomic<bool>& operation_returned) {
  const iree_time_t deadline_ns = iree_relative_timeout_to_deadline_ns(
      iree_make_duration_ms(/*timeout_ms=*/10000));
  while (!g_allocation_preparation_await_idle_entered.load(
             std::memory_order_acquire) &&
         !operation_returned.load(std::memory_order_acquire) &&
         iree_time_now() < deadline_ns) {
    std::this_thread::yield();
  }
  return g_allocation_preparation_await_idle_entered.load(
      std::memory_order_acquire);
}

extern "C" iree_status_t __real_iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

extern "C" iree_status_t __wrap_iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (g_fail_next_memory_barrier.exchange(false, std::memory_order_acq_rel)) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "injected post-copy barrier failure");
  }
  return __real_iree_hal_command_buffer_execution_barrier(
      command_buffer, source_stage_mask, target_stage_mask, flags,
      memory_barrier_count, memory_barriers, buffer_barrier_count,
      buffer_barriers);
}

extern "C" bool __real_iree_hal_allocator_supports_virtual_memory(
    iree_hal_allocator_t* allocator);

extern "C" bool __wrap_iree_hal_allocator_supports_virtual_memory(
    iree_hal_allocator_t* allocator) {
  if (g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return true;
  }
  return __real_iree_hal_allocator_supports_virtual_memory(allocator);
}

extern "C" iree_status_t
__real_iree_hal_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_device_size_t* out_minimum_page_size,
    iree_device_size_t* out_recommended_page_size);

extern "C" iree_status_t
__wrap_iree_hal_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_device_size_t* out_minimum_page_size,
    iree_device_size_t* out_recommended_page_size) {
  if (g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    *out_minimum_page_size = 4096;
    *out_recommended_page_size = 4096;
    return iree_ok_status();
  }
  return __real_iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, out_minimum_page_size, out_recommended_page_size);
}

extern "C" iree_status_t __real_iree_hal_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t size, iree_hal_buffer_t** out_virtual_buffer);

extern "C" iree_status_t __wrap_iree_hal_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t size, iree_hal_buffer_t** out_virtual_buffer) {
  if (!g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return __real_iree_hal_allocator_virtual_memory_reserve(
        allocator, queue_family_affinity, size, out_virtual_buffer);
  }
  iree_hal_buffer_params_t params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER |
          IREE_HAL_BUFFER_USAGE_DISPATCH,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      /*.queue_family_affinity=*/queue_family_affinity,
  };
  return iree_hal_allocator_allocate_buffer(allocator, params, size,
                                            out_virtual_buffer);
}

extern "C" iree_status_t __real_iree_hal_allocator_virtual_memory_release(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer);

extern "C" iree_status_t __wrap_iree_hal_allocator_virtual_memory_release(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer) {
  if (!g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return __real_iree_hal_allocator_virtual_memory_release(allocator,
                                                            virtual_buffer);
  }
  iree_hal_buffer_release(virtual_buffer);
  return iree_ok_status();
}

extern "C" iree_status_t __real_iree_hal_allocator_physical_memory_allocate(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory);

extern "C" iree_status_t __wrap_iree_hal_allocator_physical_memory_allocate(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  if (!g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return __real_iree_hal_allocator_physical_memory_allocate(
        allocator, params, size, host_allocator, out_physical_memory);
  }
  return iree_allocator_malloc(iree_allocator_system(), /*byte_length=*/1,
                               (void**)out_physical_memory);
}

extern "C" iree_status_t __real_iree_hal_allocator_physical_memory_free(
    iree_hal_allocator_t* allocator,
    iree_hal_physical_memory_t* physical_memory);

extern "C" iree_status_t __wrap_iree_hal_allocator_physical_memory_free(
    iree_hal_allocator_t* allocator,
    iree_hal_physical_memory_t* physical_memory) {
  if (!g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return __real_iree_hal_allocator_physical_memory_free(allocator,
                                                          physical_memory);
  }
  iree_allocator_free(iree_allocator_system(), physical_memory);
  return iree_ok_status();
}

extern "C" iree_status_t __real_iree_hal_allocator_virtual_memory_map(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size);

extern "C" iree_status_t __wrap_iree_hal_allocator_virtual_memory_map(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  if (g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return iree_ok_status();
  }
  return __real_iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer, virtual_offset, physical_memory,
      physical_offset, size);
}

extern "C" iree_status_t __real_iree_hal_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size);

extern "C" iree_status_t __wrap_iree_hal_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  if (g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return iree_ok_status();
  }
  return __real_iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer, virtual_offset, size);
}

extern "C" iree_status_t __real_iree_hal_allocator_virtual_memory_protect(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection);

extern "C" iree_status_t __wrap_iree_hal_allocator_virtual_memory_protect(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection) {
  if (g_emulate_graph_virtual_memory.load(std::memory_order_acquire)) {
    return iree_ok_status();
  }
  return __real_iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer, virtual_offset, size, queue_family_affinity,
      access_scope, protection);
}

extern "C" iree_status_t
__real_iree_hal_streaming_graph_memory_allocation_prepare_pointer_for_launch(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

extern "C" iree_status_t
__wrap_iree_hal_streaming_graph_memory_allocation_prepare_pointer_for_launch(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  iree_status_t status =
      __real_iree_hal_streaming_graph_memory_allocation_prepare_pointer_for_launch(
          allocation);
  if (iree_status_is_ok(status) && g_pause_graph_pointer_prepare.exchange(
                                       false, std::memory_order_acq_rel)) {
    iree_hal_streaming_stream_t* stream =
        g_graph_pointer_prepare_stream.load(std::memory_order_acquire);
    const bool acquired = iree_slim_mutex_try_lock(&stream->mutex);
    if (acquired) {
      iree_slim_mutex_unlock(&stream->mutex);
    }
    g_graph_prepare_held_stream_reservation.store(!acquired,
                                                  std::memory_order_release);
    g_graph_pointer_prepare_entered.store(true, std::memory_order_release);
    while (
        !g_allow_graph_pointer_prepare_return.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
  return status;
}

extern "C" void
__real_iree_hal_streaming_graph_memory_allocation_restore_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

extern "C" void
__wrap_iree_hal_streaming_graph_memory_allocation_restore_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  __real_iree_hal_streaming_graph_memory_allocation_restore_pointer_reference(
      allocation);
  if (g_pause_after_graph_pointer_restore.exchange(false,
                                                   std::memory_order_acq_rel)) {
    g_graph_pointer_restore_entered.store(true, std::memory_order_release);
    while (
        !g_allow_graph_pointer_restore_return.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
}

#endif  // IREE_PLATFORM_LINUX

namespace {

void NoopHostCallback(void* user_data) { (void)user_data; }

iree_status_t LaunchGraphExecAndCheckResult(
    iree_hal_streaming_graph_exec_t* executable,
    iree_hal_streaming_stream_t* stream) {
  iree_hal_streaming_graph_exec_launch_result_t result =
      IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR;
  iree_status_t status =
      iree_hal_streaming_graph_exec_launch(executable, stream, &result);
  EXPECT_EQ(iree_status_is_ok(status)
                ? IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS
                : IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR,
            result);
  return status;
}

#if defined(IREE_PLATFORM_LINUX)
class ScopedGraphVirtualMemoryEmulation {
 public:
  ScopedGraphVirtualMemoryEmulation()
      : previous_(g_emulate_graph_virtual_memory.exchange(
            true, std::memory_order_acq_rel)) {}

  ~ScopedGraphVirtualMemoryEmulation() {
    g_emulate_graph_virtual_memory.store(previous_, std::memory_order_release);
  }

 private:
  bool previous_;
};
#endif  // IREE_PLATFORM_LINUX

struct InjectedFlushQueue {
  iree_hal_queue_t base;
  iree_hal_queue_t* target = nullptr;
  iree_hal_semaphore_t* execute_gate = nullptr;
  uint64_t execute_gate_value = 0;
  std::atomic<bool> fail_flush = false;
  std::atomic<int> injected_execute_count = 0;
  std::atomic<int> injected_flush_count = 0;
};

InjectedFlushQueue* CastInjectedFlushQueue(iree_hal_queue_t* base_queue) {
  return reinterpret_cast<InjectedFlushQueue*>(base_queue);
}

void DestroyInjectedFlushQueue(iree_hal_queue_t* base_queue) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  iree_hal_queue_release(queue->target);
  queue->target = nullptr;
}

iree_status_t InjectedFlushQueueBarrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_queue_barrier(CastInjectedFlushQueue(base_queue)->target,
                                wait_semaphore_list, signal_semaphore_list,
                                flags);
}

iree_status_t InjectedFlushQueueExecute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  if (!queue->execute_gate) {
    return iree_hal_queue_execute(queue->target, wait_semaphore_list,
                                  signal_semaphore_list, command_buffer,
                                  binding_table, flags);
  }
  if (wait_semaphore_list.count > 1) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected queue supports at most one input wait");
  }
  std::array<iree_hal_semaphore_t*, 2> wait_semaphores = {};
  std::array<uint64_t, 2> wait_values = {};
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    wait_semaphores[i] = wait_semaphore_list.semaphores[i];
    wait_values[i] = wait_semaphore_list.payload_values[i];
  }
  wait_semaphores[wait_semaphore_list.count] = queue->execute_gate;
  wait_values[wait_semaphore_list.count] = queue->execute_gate_value;
  const iree_hal_semaphore_list_t gated_waits = {
      /*.count=*/wait_semaphore_list.count + 1,
      /*.semaphores=*/wait_semaphores.data(),
      /*.payload_values=*/wait_values.data(),
  };
  iree_status_t status =
      iree_hal_queue_execute(queue->target, gated_waits, signal_semaphore_list,
                             command_buffer, binding_table, flags);
  if (iree_status_is_ok(status)) {
    queue->injected_execute_count.fetch_add(1, std::memory_order_acq_rel);
  }
  return status;
}

iree_status_t InjectedFlushQueueFlush(iree_hal_queue_t* base_queue) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  iree_status_t status = iree_hal_queue_flush(queue->target);
  if (iree_status_is_ok(status) &&
      queue->fail_flush.load(std::memory_order_acquire)) {
    queue->injected_flush_count.fetch_add(1, std::memory_order_acq_rel);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "injected post-accept queue flush failure");
  }
  return status;
}

const iree_hal_queue_vtable_t kInjectedFlushQueueVtable = {
    /*.destroy=*/DestroyInjectedFlushQueue,
    /*.barrier=*/InjectedFlushQueueBarrier,
    /*.execute=*/InjectedFlushQueueExecute,
    /*.host_call=*/nullptr,
    /*.query_dispatch_concurrency=*/nullptr,
    /*.dispatch=*/nullptr,
    /*.atomic_wait=*/nullptr,
    /*.atomic_store=*/nullptr,
    /*.atomic_rmw=*/nullptr,
    /*.timestamp=*/nullptr,
    /*.flush=*/InjectedFlushQueueFlush,
    /*.alloca=*/nullptr,
    /*.dealloca=*/nullptr,
    /*.transfer=*/nullptr,
    /*.read=*/nullptr,
    /*.write=*/nullptr,
};

void InitializeInjectedFlushQueue(iree_hal_queue_t* target,
                                  InjectedFlushQueue* out_queue) {
  out_queue->target = target;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target);
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kInjectedFlushQueueVtable, &out_queue->base);
}

// Forwards every operation except one selected host call. Graph allocation,
// free, and user host nodes are separate queue host-call blocks, so selecting
// an index deterministically exposes each accepted-prefix boundary.
struct RejectNthHostCallQueue {
  iree_hal_queue_t base;
  iree_hal_queue_t* target = nullptr;
  std::atomic<int> host_call_count = 0;
  std::atomic<int> reject_host_call_index = -1;
};

RejectNthHostCallQueue* CastRejectNthHostCallQueue(
    iree_hal_queue_t* base_queue) {
  return reinterpret_cast<RejectNthHostCallQueue*>(base_queue);
}

void DestroyRejectNthHostCallQueue(iree_hal_queue_t* base_queue) {
  auto* queue = CastRejectNthHostCallQueue(base_queue);
  iree_hal_queue_release(queue->target);
  queue->target = nullptr;
}

iree_status_t RejectNthHostCallQueueBarrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_queue_barrier(CastRejectNthHostCallQueue(base_queue)->target,
                                wait_semaphore_list, signal_semaphore_list,
                                flags);
}

iree_status_t RejectNthHostCallQueueExecute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  return iree_hal_queue_execute(CastRejectNthHostCallQueue(base_queue)->target,
                                wait_semaphore_list, signal_semaphore_list,
                                command_buffer, binding_table, flags);
}

iree_status_t RejectNthHostCallQueueHostCall(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  auto* queue = CastRejectNthHostCallQueue(base_queue);
  const int call_index =
      queue->host_call_count.fetch_add(1, std::memory_order_acq_rel);
  if (call_index ==
      queue->reject_host_call_index.load(std::memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected graph host-call rejection");
  }
  return iree_hal_queue_host_call(queue->target, wait_semaphore_list,
                                  signal_semaphore_list, call, args, flags);
}

iree_status_t RejectNthHostCallQueueFlush(iree_hal_queue_t* base_queue) {
  return iree_hal_queue_flush(CastRejectNthHostCallQueue(base_queue)->target);
}

const iree_hal_queue_vtable_t kRejectNthHostCallQueueVtable = {
    /*.destroy=*/DestroyRejectNthHostCallQueue,
    /*.barrier=*/RejectNthHostCallQueueBarrier,
    /*.execute=*/RejectNthHostCallQueueExecute,
    /*.host_call=*/RejectNthHostCallQueueHostCall,
    /*.query_dispatch_concurrency=*/nullptr,
    /*.dispatch=*/nullptr,
    /*.atomic_wait=*/nullptr,
    /*.atomic_store=*/nullptr,
    /*.atomic_rmw=*/nullptr,
    /*.timestamp=*/nullptr,
    /*.flush=*/RejectNthHostCallQueueFlush,
    /*.alloca=*/nullptr,
    /*.dealloca=*/nullptr,
    /*.transfer=*/nullptr,
    /*.read=*/nullptr,
    /*.write=*/nullptr,
};

void InitializeRejectNthHostCallQueue(iree_hal_queue_t* target,
                                      RejectNthHostCallQueue* out_queue) {
  out_queue->target = target;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target);
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kRejectNthHostCallQueueVtable, &out_queue->base);
}

void ResetRejectNthHostCallQueue(RejectNthHostCallQueue* queue,
                                 int reject_host_call_index) {
  queue->host_call_count.store(0, std::memory_order_release);
  queue->reject_host_call_index.store(reject_host_call_index,
                                      std::memory_order_release);
}

iree_hal_queue_t* ReplaceStreamQueue(iree_hal_streaming_stream_t* stream,
                                     iree_hal_queue_t* replacement) {
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_queue_t* previous = stream->queue;
  stream->queue = replacement;
  iree_slim_mutex_unlock(&stream->mutex);
  return previous;
}

struct ControlledCancellationQueue {
  iree_hal_queue_t base;
  iree_hal_queue_t* target = nullptr;
  iree_hal_resource_t* retained_resource = nullptr;
  iree_hal_semaphore_t* wait_semaphore = nullptr;
  uint64_t wait_value = 0;
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  std::atomic<bool> accepted = false;
};

ControlledCancellationQueue* CastControlledCancellationQueue(
    iree_hal_queue_t* base_queue) {
  return reinterpret_cast<ControlledCancellationQueue*>(base_queue);
}

void DestroyControlledCancellationQueue(iree_hal_queue_t* base_queue) {
  auto* queue = CastControlledCancellationQueue(base_queue);
  IREE_ASSERT(!queue->retained_resource);
  IREE_ASSERT(!queue->wait_semaphore);
  IREE_ASSERT(!queue->signal_semaphore);
  iree_hal_queue_release(queue->target);
  queue->target = nullptr;
}

iree_status_t ControlledCancellationQueueBarrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_queue_barrier(
      CastControlledCancellationQueue(base_queue)->target, wait_semaphore_list,
      signal_semaphore_list, flags);
}

iree_status_t ControlledCancellationQueueExecute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  return iree_hal_queue_execute(
      CastControlledCancellationQueue(base_queue)->target, wait_semaphore_list,
      signal_semaphore_list, command_buffer, binding_table, flags);
}

iree_status_t ControlledCancellationQueueHostCall(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  (void)args;
  (void)flags;
  auto* queue = CastControlledCancellationQueue(base_queue);
  if (!call.resource || wait_semaphore_list.count > 1 ||
      signal_semaphore_list.count != 1 || queue->retained_resource ||
      queue->wait_semaphore || queue->signal_semaphore) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "controlled queue requires one resource-backed "
                            "host call, at most one wait, and one signal");
  }
  queue->retained_resource = call.resource;
  iree_hal_resource_retain(queue->retained_resource);
  if (wait_semaphore_list.count == 1) {
    queue->wait_semaphore = wait_semaphore_list.semaphores[0];
    queue->wait_value = wait_semaphore_list.payload_values[0];
    iree_hal_semaphore_retain(queue->wait_semaphore);
  }
  queue->signal_semaphore = signal_semaphore_list.semaphores[0];
  iree_hal_semaphore_retain(queue->signal_semaphore);
  queue->accepted.store(true, std::memory_order_release);
  return iree_ok_status();
}

iree_status_t ControlledCancellationQueueFlush(iree_hal_queue_t* base_queue) {
  return iree_hal_queue_flush(
      CastControlledCancellationQueue(base_queue)->target);
}

const iree_hal_queue_vtable_t kControlledCancellationQueueVtable = {
    /*.destroy=*/DestroyControlledCancellationQueue,
    /*.barrier=*/ControlledCancellationQueueBarrier,
    /*.execute=*/ControlledCancellationQueueExecute,
    /*.host_call=*/ControlledCancellationQueueHostCall,
    /*.query_dispatch_concurrency=*/nullptr,
    /*.dispatch=*/nullptr,
    /*.atomic_wait=*/nullptr,
    /*.atomic_store=*/nullptr,
    /*.atomic_rmw=*/nullptr,
    /*.timestamp=*/nullptr,
    /*.flush=*/ControlledCancellationQueueFlush,
    /*.alloca=*/nullptr,
    /*.dealloca=*/nullptr,
    /*.transfer=*/nullptr,
    /*.read=*/nullptr,
    /*.write=*/nullptr,
};

void InitializeControlledCancellationQueue(
    iree_hal_queue_t* target, ControlledCancellationQueue* out_queue) {
  out_queue->target = target;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target);
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kControlledCancellationQueueVtable,
                            &out_queue->base);
}

void ReleaseControlledCancellationResource(ControlledCancellationQueue* queue) {
  IREE_ASSERT(queue->retained_resource);
  iree_hal_resource_release(queue->retained_resource);
  queue->retained_resource = nullptr;
  iree_hal_semaphore_release(queue->wait_semaphore);
  queue->wait_semaphore = nullptr;
  queue->wait_value = 0;
  iree_hal_semaphore_release(queue->signal_semaphore);
  queue->signal_semaphore = nullptr;
}

iree_host_size_t PendingTerminalResourceCount(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->terminal_resource_mutex);
  const iree_host_size_t count = device->pending_terminal_resource_count;
  iree_slim_mutex_unlock(&device->terminal_resource_mutex);
  return count;
}

struct FailingSnapshotAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  std::atomic<bool> fail_allocations = false;
  std::atomic<int> allocation_attempt_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailingSnapshotAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      allocator->allocation_attempt_count.fetch_add(1,
                                                    std::memory_order_acq_rel);
      if (allocator->fail_allocations.load(std::memory_order_acquire)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected stream snapshot allocation failure");
      }
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &FailingSnapshotAllocator::Control};
  }
};

class CpuStreamingMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));

    device_entry_.hrx_device = hrx_device;
    device_entry_.hal_device = hrx_device_hal(hrx_device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_trim_mutex);
    iree_slim_mutex_initialize(&device_entry_.terminal_resource_mutex);
    iree_notification_initialize(&device_entry_.terminal_resource_notification);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);

    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, iree_allocator_system(), &context_));
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
        /*priority=*/0, iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    iree_hal_streaming_memory_release_wrapped_buffer(buffer_);
    buffer_ = nullptr;
    device_pointer_ = 0;
    iree_hal_streaming_stream_release(stream_);
    iree_hal_streaming_context_release(context_);
    iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
    EXPECT_EQ(0, device_entry_.pending_terminal_resource_count);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_notification_deinitialize(
        &device_entry_.terminal_resource_notification);
    iree_slim_mutex_deinitialize(&device_entry_.terminal_resource_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_trim_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  void AddGraphMemoryAllocationNode(
      iree_hal_streaming_graph_t* graph,
      iree_hal_streaming_graph_memory_allocation_t** out_allocation,
      iree_hal_streaming_deviceptr_t* out_pointer,
      iree_hal_streaming_graph_node_t** out_node = nullptr) {
    *out_allocation = nullptr;
    *out_pointer = 0;
    IREE_ASSERT_OK(iree_hal_streaming_graph_memory_allocation_create(
        context_, /*size=*/4096, out_allocation));
    iree_hal_streaming_graph_node_t* allocation_node = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
        graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
        &allocation_node));
    allocation_node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC;
    allocation_node->attrs.mem_alloc.params = nullptr;
    allocation_node->attrs.mem_alloc.params_size = 0;
    allocation_node->attrs.mem_alloc.dptr =
        iree_hal_streaming_graph_memory_allocation_device_pointer(
            *out_allocation);
    allocation_node->attrs.mem_alloc.bytesize = 4096;
    allocation_node->attrs.mem_alloc.allocation = *out_allocation;
    iree_hal_streaming_graph_memory_allocation_retain(*out_allocation);
    allocation_node->attrs.mem_alloc.has_in_graph_free_node = false;
    graph->has_graph_memory_nodes = true;
    *out_pointer =
        (iree_hal_streaming_deviceptr_t)allocation_node->attrs.mem_alloc.dptr;
    if (out_node) {
      *out_node = allocation_node;
    }
  }

  void AddGraphMemoryFreeNode(
      iree_hal_streaming_graph_t* graph,
      iree_hal_streaming_graph_memory_allocation_t* allocation,
      iree_hal_streaming_graph_node_t* dependency,
      iree_hal_streaming_graph_node_t** out_node) {
    *out_node = nullptr;
    iree_hal_streaming_graph_memory_allocation_retain(allocation);
    if (!iree_hal_streaming_graph_memory_allocation_try_claim_free_node_reference(
            allocation)) {
      iree_hal_streaming_graph_memory_allocation_release(allocation);
      FAIL() << "graph allocation pointer could not be claimed by free node";
      return;
    }
    iree_hal_streaming_graph_node_t* free_node = nullptr;
    iree_status_t status = iree_hal_streaming_graph_add_empty_node(
        graph, &dependency, /*dependency_count=*/1, &free_node);
    if (!iree_status_is_ok(status)) {
      iree_hal_streaming_graph_memory_allocation_release_free_node_reference(
          allocation);
      iree_hal_streaming_graph_memory_allocation_release(allocation);
      IREE_ASSERT_OK(status);
      return;
    }
    free_node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE;
    free_node->attrs.mem_free.dptr =
        iree_hal_streaming_graph_memory_allocation_device_pointer(allocation);
    free_node->attrs.mem_free.allocation = allocation;
    iree_hal_streaming_graph_refresh_memory_allocation_free_node_state(
        graph, allocation);
    graph->has_graph_memory_nodes = true;
    *out_node = free_node;
  }

  void CreateGraphMemoryAllocationGraph(
      iree_hal_streaming_graph_t** out_graph,
      iree_hal_streaming_graph_memory_allocation_t** out_allocation,
      iree_hal_streaming_deviceptr_t* out_pointer) {
    *out_graph = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        out_graph));
    AddGraphMemoryAllocationNode(*out_graph, out_allocation, out_pointer);
  }

  iree_hal_streaming_device_t device_entry_ = {};
  iree_hal_streaming_context_t* context_ = nullptr;
  iree_hal_streaming_stream_t* stream_ = nullptr;
  iree_hal_streaming_buffer_t* buffer_ = nullptr;
  iree_hal_streaming_deviceptr_t device_pointer_ = 0;
};

TEST_F(CpuStreamingMemoryTest,
       GraphMemoryChildTopologyIsRejectedAndRevalidated) {
#if !defined(IREE_PLATFORM_LINUX)
  GTEST_SKIP() << "virtual-memory dependency injection requires ELF wrapping";
#else
  ScopedGraphVirtualMemoryEmulation emulate_virtual_memory;

  // A graph that already owns a memory allocation cannot be inserted as a
  // child: the legacy HIP child API has no way to transfer graph-memory
  // ownership to the parent executable.
  iree_hal_streaming_graph_t* memory_child = nullptr;
  iree_hal_streaming_graph_memory_allocation_t* direct_allocation = nullptr;
  iree_hal_streaming_deviceptr_t direct_pointer = 0;
  CreateGraphMemoryAllocationGraph(&memory_child, &direct_allocation,
                                   &direct_pointer);
  iree_hal_streaming_graph_t* direct_parent = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &direct_parent));
  iree_hal_streaming_graph_node_t* rejected_child_node = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_streaming_graph_add_child_graph_node(
          direct_parent, /*dependencies=*/nullptr, /*dependency_count=*/0,
          memory_child, &rejected_child_node));
  EXPECT_EQ(nullptr, rejected_child_node);
  iree_hal_streaming_graph_release(direct_parent);
  iree_hal_streaming_graph_release(memory_child);

  // Insertion-time validation is not sufficient: a leaf can gain a memory
  // node after an otherwise valid nested hierarchy has been assembled.
  iree_hal_streaming_graph_t* leaf = nullptr;
  iree_hal_streaming_graph_t* middle = nullptr;
  iree_hal_streaming_graph_t* root = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &leaf));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &middle));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &root));
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      middle, /*dependencies=*/nullptr, /*dependency_count=*/0, leaf,
      /*out_node=*/nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      root, /*dependencies=*/nullptr, /*dependency_count=*/0, middle,
      /*out_node=*/nullptr));
  iree_hal_streaming_graph_memory_allocation_t* nested_allocation = nullptr;
  iree_hal_streaming_deviceptr_t nested_pointer = 0;
  AddGraphMemoryAllocationNode(leaf, &nested_allocation, &nested_pointer);

  iree_hal_streaming_graph_exec_t* rejected_exec = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_streaming_graph_instantiate(
          root, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE,
          &rejected_exec));
  EXPECT_EQ(nullptr, rejected_exec);
  iree_hal_streaming_graph_release(root);
  iree_hal_streaming_graph_release(middle);
  iree_hal_streaming_graph_release(leaf);

  // Whole-executable update revalidates the source so the same post-insertion
  // mutation is rejected before recursive child compilation.
  iree_hal_streaming_graph_t* update_child = nullptr;
  iree_hal_streaming_graph_t* update_parent = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &update_child));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &update_parent));
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      update_parent, /*dependencies=*/nullptr, /*dependency_count=*/0,
      update_child, /*out_node=*/nullptr));
  iree_hal_streaming_graph_exec_t* update_exec = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      update_parent, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE,
      &update_exec));
  iree_hal_streaming_graph_memory_allocation_t* update_allocation = nullptr;
  iree_hal_streaming_deviceptr_t update_pointer = 0;
  AddGraphMemoryAllocationNode(update_child, &update_allocation,
                               &update_pointer);

  iree_hal_streaming_graph_node_t* error_node = nullptr;
  iree_hal_streaming_graph_exec_update_result_t update_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_streaming_graph_exec_update(update_exec, update_parent,
                                           &error_node, &update_result));
  EXPECT_EQ(nullptr, error_node);
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NOT_SUPPORTED, update_result);

  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(update_exec));
  iree_hal_streaming_graph_release(update_parent);
  iree_hal_streaming_graph_release(update_child);
#endif  // IREE_PLATFORM_LINUX
}

TEST_F(CpuStreamingMemoryTest,
       RejectedGraphMemoryPrefixLeavesRetryableMappingResidue) {
#if !defined(IREE_PLATFORM_LINUX)
  GTEST_SKIP() << "virtual-memory dependency injection requires ELF wrapping";
#else
  ScopedGraphVirtualMemoryEmulation emulate_virtual_memory;
  auto graph_memory_allocation_count = [&] {
    iree_slim_mutex_lock(&device_entry_.graph_memory_mutex);
    const uint32_t count = device_entry_.graph_memory_allocation_count;
    iree_slim_mutex_unlock(&device_entry_.graph_memory_mutex);
    return count;
  };
  const uint32_t initial_allocation_count = graph_memory_allocation_count();
  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_memory_allocation_t* allocation = nullptr;
  iree_hal_streaming_deviceptr_t pointer = 0;
  iree_hal_streaming_graph_node_t* allocation_node = nullptr;
  AddGraphMemoryAllocationNode(graph, &allocation, &pointer, &allocation_node);
  iree_hal_streaming_graph_node_t* free_node = nullptr;
  AddGraphMemoryFreeNode(graph, allocation, allocation_node, &free_node);
  ASSERT_NE(nullptr, free_node);
  iree_hal_streaming_graph_node_t* terminal_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, &free_node, /*dependency_count=*/1, &NoopHostCallback,
      /*user_data=*/nullptr, &terminal_node));

  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  RejectNthHostCallQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeRejectNthHostCallQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);

  auto launch_with_rejection = [&](int reject_host_call_index) {
    ResetRejectNthHostCallQueue(&fault_queue, reject_host_call_index);
    iree_status_t status = LaunchGraphExecAndCheckResult(exec, stream_);
    EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
    const bool was_rejected =
        iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED;
    iree_status_ignore(status);
    return was_rejected;
  };
  auto pointer_is_published = [&] {
    iree_hal_streaming_buffer_ref_t ref = {};
    iree_status_t status =
        iree_hal_streaming_memory_lookup(context_, pointer, &ref);
    const iree_status_code_t code = iree_status_code(status);
    EXPECT_TRUE(code == IREE_STATUS_OK || code == IREE_STATUS_NOT_FOUND);
    iree_status_ignore(status);
    return code == IREE_STATUS_OK;
  };
  auto launch_and_synchronize = [&] {
    ResetRejectNthHostCallQueue(&fault_queue, /*reject_host_call_index=*/-1);
    iree_status_t launch_status = LaunchGraphExecAndCheckResult(exec, stream_);
    EXPECT_EQ(IREE_STATUS_OK, iree_status_code(launch_status));
    const bool launch_succeeded = iree_status_is_ok(launch_status);
    iree_status_ignore(launch_status);
    if (launch_succeeded) {
      iree_status_t synchronize_status =
          iree_hal_streaming_stream_synchronize(stream_);
      EXPECT_EQ(IREE_STATUS_OK, iree_status_code(synchronize_status));
      const bool synchronize_succeeded = iree_status_is_ok(synchronize_status);
      iree_status_ignore(synchronize_status);
      return synchronize_succeeded;
    }
    return false;
  };

  // Allocation creation publishes the returned pointer. Accept its first map
  // block, reject the dependent free block, and retain that fully usable
  // mapping and publication as retry residue.
  ASSERT_TRUE(launch_with_rejection(/*reject_host_call_index=*/1));
  EXPECT_EQ(2, fault_queue.host_call_count.load(std::memory_order_acquire));
  EXPECT_TRUE(iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_TRUE(pointer_is_published());
  ASSERT_TRUE(launch_and_synchronize());
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_FALSE(pointer_is_published());

  // After that successful launch, a repeat launch must use the same residue
  // path when its map is accepted and its free is rejected. Preparation
  // republishes the pointer, and failure leaves both mapping and publication
  // available to the following retry.
  ASSERT_TRUE(launch_with_rejection(/*reject_host_call_index=*/1));
  EXPECT_EQ(2, fault_queue.host_call_count.load(std::memory_order_acquire));
  EXPECT_TRUE(iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_TRUE(pointer_is_published());
  ASSERT_TRUE(launch_and_synchronize());
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_FALSE(pointer_is_published());

  // If both map and free were accepted before a third block rejected, the
  // accepted free consumes both the pointer and any per-attempt map evidence.
  // The next launch follows the ordinary publish/map/free path.
  ASSERT_TRUE(launch_with_rejection(/*reject_host_call_index=*/2));
  EXPECT_EQ(3, fault_queue.host_call_count.load(std::memory_order_acquire));
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_FALSE(pointer_is_published());
  ASSERT_TRUE(launch_and_synchronize());
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_FALSE(pointer_is_published());

  // A successful free ends the previous pointer generation. Reject the first
  // map of the next generation: its persistent preparation publication is
  // unexecuted, so graph teardown must retire the table entry, virtual
  // reservation, and context reference.
  ASSERT_TRUE(launch_with_rejection(/*reject_host_call_index=*/0));
  EXPECT_EQ(1, fault_queue.host_call_count.load(std::memory_order_acquire));
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_TRUE(pointer_is_published());

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(exec));
  iree_hal_streaming_graph_release(graph);
  // Graph-exec destruction is nonblocking. Keep VMM emulation installed until
  // its queued retirement has released the executable's allocation records.
  iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
  EXPECT_EQ(0, PendingTerminalResourceCount(&device_entry_));
  EXPECT_FALSE(pointer_is_published());
  EXPECT_EQ(initial_allocation_count, graph_memory_allocation_count());
#endif  // IREE_PLATFORM_LINUX
}

TEST_F(CpuStreamingMemoryTest,
       RejectedNoFreeGraphMemoryPrefixBypassesCompletedLaunchPreflight) {
#if !defined(IREE_PLATFORM_LINUX)
  GTEST_SKIP() << "virtual-memory dependency injection requires ELF wrapping";
#else
  ScopedGraphVirtualMemoryEmulation emulate_virtual_memory;
  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_memory_allocation_t* allocation = nullptr;
  iree_hal_streaming_deviceptr_t pointer = 0;
  iree_hal_streaming_graph_node_t* allocation_node = nullptr;
  AddGraphMemoryAllocationNode(graph, &allocation, &pointer, &allocation_node);
  iree_hal_streaming_graph_node_t* terminal_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, &allocation_node, /*dependency_count=*/1, &NoopHostCallback,
      /*user_data=*/nullptr, &terminal_node));

  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  RejectNthHostCallQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeRejectNthHostCallQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  iree_hal_queue_release(original_queue);

  auto pointer_is_published = [&] {
    iree_hal_streaming_buffer_ref_t ref = {};
    iree_status_t status =
        iree_hal_streaming_memory_lookup(context_, pointer, &ref);
    const iree_status_code_t code = iree_status_code(status);
    EXPECT_TRUE(code == IREE_STATUS_OK || code == IREE_STATUS_NOT_FOUND);
    iree_status_ignore(status);
    return code == IREE_STATUS_OK;
  };
  auto launch_and_synchronize = [&] {
    ResetRejectNthHostCallQueue(&fault_queue, /*reject_host_call_index=*/-1);
    iree_status_t status = LaunchGraphExecAndCheckResult(exec, stream_);
    EXPECT_EQ(IREE_STATUS_OK, iree_status_code(status));
    const bool launched = iree_status_is_ok(status);
    iree_status_ignore(status);
    if (!launched) {
      return false;
    }
    status = iree_hal_streaming_stream_synchronize(stream_);
    EXPECT_EQ(IREE_STATUS_OK, iree_status_code(status));
    const bool synchronized = iree_status_is_ok(status);
    iree_status_ignore(status);
    return synchronized;
  };
  auto free_async_and_synchronize = [&] {
    ResetRejectNthHostCallQueue(&fault_queue, /*reject_host_call_index=*/-1);
    iree_status_t status =
        iree_hal_streaming_memory_free_device_async(context_, pointer, stream_);
    EXPECT_EQ(IREE_STATUS_OK, iree_status_code(status));
    const bool enqueued = iree_status_is_ok(status);
    iree_status_ignore(status);
    if (!enqueued) {
      return false;
    }
    status = iree_hal_streaming_stream_synchronize(stream_);
    EXPECT_EQ(IREE_STATUS_OK, iree_status_code(status));
    const bool synchronized = iree_status_is_ok(status);
    iree_status_ignore(status);
    return synchronized;
  };

  // Establish a completed launch, then externally retire its live unmatched
  // allocation so a repeat launch is otherwise valid.
  ASSERT_TRUE(launch_and_synchronize());
  EXPECT_TRUE(iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_TRUE(pointer_is_published());
  ASSERT_TRUE(free_async_and_synchronize());
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_FALSE(pointer_is_published());

  // The repeat launch republishes and maps, but its later host node rejects.
  // Its mapping is residue from an incomplete attempt, not a live unmatched
  // allocation from the earlier successful launch.
  ResetRejectNthHostCallQueue(&fault_queue, /*reject_host_call_index=*/1);
  iree_status_t status = LaunchGraphExecAndCheckResult(exec, stream_);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  const bool was_rejected =
      iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED;
  iree_status_ignore(status);
  ASSERT_TRUE(was_rejected);
  EXPECT_EQ(2, fault_queue.host_call_count.load(std::memory_order_acquire));
  EXPECT_TRUE(iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_TRUE(pointer_is_published());

  // launch_count still reflects the first successful launch. Retry residue
  // must bypass the ordinary live-unfreed rejection and be consumed by map.
  ASSERT_TRUE(launch_and_synchronize());
  EXPECT_TRUE(iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_TRUE(pointer_is_published());

  // The graph has no free node, so the test owns final retirement of the live
  // allocation produced by its successful retry.
  ASSERT_TRUE(free_async_and_synchronize());
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  EXPECT_FALSE(pointer_is_published());

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(exec));
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
  EXPECT_EQ(0, PendingTerminalResourceCount(&device_entry_));
#endif  // IREE_PLATFORM_LINUX
}

TEST_F(CpuStreamingMemoryTest,
       PitchedCopyDrainsAcceptedRowsBeforeReturningRecordingError) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);

  const std::array<uint8_t, kAllocationSize> initial = {
      1, 2, 3, 4, 21, 22, 23, 24, 41, 42, 43, 44, 61, 62, 63, 64};
  std::memcpy(buffer_->host_ptr, initial.data(), initial.size());

  // Row zero copies [0, 4) to the disjoint range [4, 8). Row one then tries
  // to copy [8, 12) onto itself, forcing command-buffer validation to fail
  // only after the first row has been accepted.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_memcpy_device_to_device_2d(
          context_, device_pointer_ + 4, /*dst_pitch=*/4, device_pointer_,
          /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_));

  const auto* contents = static_cast<const uint8_t*>(buffer_->host_ptr);
  EXPECT_EQ(0, std::memcmp(contents + 4, initial.data(), 4));
  EXPECT_EQ(0, std::memcmp(contents + 8, initial.data() + 8, 4));
}

TEST_F(CpuStreamingMemoryTest,
       PostAcceptFlushFailureDrainsPrefixBeforeReturningOriginalError) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);

  const std::array<uint8_t, kAllocationSize> initial = {
      1, 2, 3, 4, 21, 22, 23, 24, 41, 42, 43, 44, 61, 62, 63, 64};
  std::memcpy(buffer_->host_ptr, initial.data(), initial.size());

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  fault_queue.fail_flush.store(true, std::memory_order_release);

  std::atomic<bool> copy_returned = false;
  std::atomic<iree_status_code_t> copy_status_code = IREE_STATUS_UNKNOWN;
  std::thread copy_thread([&] {
    iree_status_t status = iree_hal_streaming_memcpy_device_to_device_2d(
        context_, device_pointer_ + 4, /*dst_pitch=*/4, device_pointer_,
        /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_);
    copy_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    copy_returned.store(true, std::memory_order_release);
  });

  bool observed_flush_failure = false;
  for (int i = 0; i < 1000000; ++i) {
    if (fault_queue.injected_flush_count.load(std::memory_order_acquire) > 0) {
      observed_flush_failure = true;
      break;
    }
    std::this_thread::yield();
  }
  if (!observed_flush_failure) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                             /*frontier=*/nullptr));
    copy_thread.join();
    iree_hal_queue_retain(original_queue);
    iree_hal_queue_t* installed_queue =
        ReplaceStreamQueue(stream_, original_queue);
    iree_hal_queue_release(installed_queue);
    iree_hal_semaphore_release(gate);
    FAIL() << "accepted command buffer did not reach injected queue_flush";
    return;
  }
  EXPECT_FALSE(copy_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  copy_thread.join();
  EXPECT_TRUE(copy_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT,
            copy_status_code.load(std::memory_order_acquire));
  const auto* contents = static_cast<const uint8_t*>(buffer_->host_ptr);
  EXPECT_EQ(0, std::memcmp(contents + 4, initial.data(), 4));
  EXPECT_EQ(0, std::memcmp(contents + 8, initial.data() + 8, 4));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  iree_hal_semaphore_release(gate);
}

TEST_F(CpuStreamingMemoryTest,
       SnapshotAllocationFailureDrainsEverySelectedStreamTail) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);

  iree_hal_streaming_stream_t* second_stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &second_stream));
  iree_hal_streaming_buffer_t* staging[2] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(staging); ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host_staging(
        context_, kAllocationSize, &staging[i]));
    ASSERT_NE(nullptr, staging[i]->host_ptr);
  }
  const std::array<uint8_t, kAllocationSize> expected = {
      3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48};
  std::memcpy(buffer_->host_ptr, expected.data(), expected.size());
  std::memset(staging[0]->host_ptr, 0, expected.size());
  std::memset(staging[1]->host_ptr, 0, expected.size());

  // Leave one recorded D2H-shaped copy behind each independently gated
  // registered stream. The allocation-free fallback visits streams in stable
  // ID order, so releasing only the first gate proves it cannot return without
  // submitting and draining the second selected tail.
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_values[2] = {1, 2};
  iree_hal_streaming_stream_t* streams[2] = {stream_, second_stream};
  uint64_t second_initial_pending = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(streams); ++i) {
    const iree_hal_semaphore_list_t gate_wait = {
        /*.count=*/1,
        /*.semaphores=*/&gate,
        /*.payload_values=*/&gate_values[i],
    };
    IREE_ASSERT_OK(
        iree_hal_streaming_stream_wait_semaphores(streams[i], gate_wait));
    IREE_ASSERT_OK(iree_hal_streaming_memory_memcpy(
        context_, iree_hal_streaming_buffer_device_pointer(staging[i]),
        device_pointer_, kAllocationSize, streams[i]));
  }
  iree_slim_mutex_lock(&second_stream->mutex);
  second_initial_pending = second_stream->pending_value;
  iree_slim_mutex_unlock(&second_stream->mutex);

  FailingSnapshotAllocator allocator;
  const iree_allocator_t original_allocator = context_->host_allocator;
  context_->host_allocator = allocator.AsAllocator();
  allocator.fail_allocations.store(true, std::memory_order_release);

  std::atomic<bool> synchronize_returned = false;
  std::atomic<iree_status_code_t> synchronize_status_code = IREE_STATUS_UNKNOWN;
  std::thread synchronize_thread([&] {
    iree_status_t status = iree_hal_streaming_context_synchronize(context_);
    synchronize_status_code.store(iree_status_code(status),
                                  std::memory_order_release);
    iree_status_ignore(status);
    synchronize_returned.store(true, std::memory_order_release);
  });

  bool observed_snapshot_failure = false;
  for (int i = 0; i < 1000000; ++i) {
    if (allocator.allocation_attempt_count.load(std::memory_order_acquire) >
        0) {
      observed_snapshot_failure = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_snapshot_failure);
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_values[0],
                                           /*frontier=*/nullptr));
  bool observed_second_submission = false;
  for (int i = 0; i < 1000000; ++i) {
    iree_slim_mutex_lock(&second_stream->mutex);
    observed_second_submission =
        second_stream->pending_value > second_initial_pending;
    iree_slim_mutex_unlock(&second_stream->mutex);
    if (observed_second_submission) {
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_second_submission);
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_values[1],
                                           /*frontier=*/nullptr));
  synchronize_thread.join();
  allocator.fail_allocations.store(false, std::memory_order_release);
  context_->host_allocator = original_allocator;

  EXPECT_TRUE(synchronize_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED,
            synchronize_status_code.load(std::memory_order_acquire));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(staging); ++i) {
    EXPECT_EQ(
        0, std::memcmp(staging[i]->host_ptr, expected.data(), expected.size()));
  }

  iree_hal_semaphore_release(gate);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(staging); ++i) {
    iree_hal_streaming_memory_release_wrapped_buffer(staging[i]);
  }
  iree_hal_streaming_context_unregister_stream(context_, second_stream);
  iree_hal_streaming_stream_release(second_stream);
}

TEST_F(CpuStreamingMemoryTest,
       PostAcceptFlushFailureKeepsD2HStagingUntilTailCompletes) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);
  const std::array<uint8_t, kAllocationSize> expected = {
      7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112};
  std::memcpy(buffer_->host_ptr, expected.data(), expected.size());
  // The task allocator is host-visible; clear only the wrapper classification
  // so this test exercises the production staging/callback route.
  buffer_->memory_type =
      (iree_hal_memory_type_t)(buffer_->memory_type &
                               ~IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
  std::array<uint8_t, kAllocationSize> destination = {};

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  fault_queue.fail_flush.store(true, std::memory_order_release);

  std::atomic<bool> copy_returned = false;
  std::atomic<iree_status_code_t> copy_status_code = IREE_STATUS_UNKNOWN;
  std::thread copy_thread([&] {
    iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
        context_, destination.data(), device_pointer_, kAllocationSize,
        stream_);
    copy_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    copy_returned.store(true, std::memory_order_release);
  });

  bool observed_flush_failure = false;
  for (int i = 0; i < 1000000; ++i) {
    if (fault_queue.injected_flush_count.load(std::memory_order_acquire) > 0) {
      observed_flush_failure = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_flush_failure);
  for (int i = 0; i < 100000 && !copy_returned.load(std::memory_order_acquire);
       ++i) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(copy_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  copy_thread.join();
  EXPECT_TRUE(copy_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_INTERNAL,
            copy_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(0,
            std::memcmp(destination.data(), expected.data(), expected.size()));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  iree_hal_semaphore_release(gate);
}

#if defined(IREE_PLATFORM_LINUX)
TEST_F(CpuStreamingMemoryTest,
       ScalarD2HBarrierFailureDrainsRecordedCopyBeforeFreeingStaging) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);
  buffer_->memory_type =
      (iree_hal_memory_type_t)(buffer_->memory_type &
                               ~IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
  std::array<uint8_t, kAllocationSize> destination = {};

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  iree_hal_queue_release(original_queue);

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  g_fail_next_memory_barrier.store(true, std::memory_order_release);

  std::atomic<bool> copy_returned = false;
  std::atomic<iree_status_code_t> copy_status_code = IREE_STATUS_UNKNOWN;
  std::thread copy_thread([&] {
    iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
        context_, destination.data(), device_pointer_, kAllocationSize,
        stream_);
    copy_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    copy_returned.store(true, std::memory_order_release);
  });

  bool observed_execute = false;
  for (int i = 0; i < 1000000; ++i) {
    if (fault_queue.injected_execute_count.load(std::memory_order_acquire) >
        0) {
      observed_execute = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_execute);
  EXPECT_FALSE(copy_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  copy_thread.join();
  g_fail_next_memory_barrier.store(false, std::memory_order_release);
  EXPECT_TRUE(copy_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_ABORTED,
            copy_status_code.load(std::memory_order_acquire));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  iree_hal_semaphore_release(gate);
}
#endif  // IREE_PLATFORM_LINUX

TEST_F(CpuStreamingMemoryTest,
       AsyncFreeFlushFailureRestoresAllocationWhileWorkIsPending) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);
  std::memset(buffer_->host_ptr, 0, kAllocationSize);
  uint32_t pattern = 0xA5A5A5A5u;
  IREE_ASSERT_OK(iree_hal_streaming_memory_memset(context_, device_pointer_,
                                                  kAllocationSize, &pattern,
                                                  sizeof(pattern), stream_));

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  iree_hal_queue_release(original_queue);
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  fault_queue.fail_flush.store(true, std::memory_order_release);

  // The fill is accepted, but the flush error prevents enqueueing the free.
  // Rollback restores the caller's allocation using its reserved table slot;
  // it neither allocates nor waits for accepted work to complete.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_streaming_memory_free_device_async(
                            context_, device_pointer_, stream_));
  EXPECT_EQ(1, fault_queue.injected_execute_count.load());
  EXPECT_EQ(1, fault_queue.injected_flush_count.load());

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);

  uint64_t completed_value = 0;
  IREE_EXPECT_OK(
      iree_hal_semaphore_query(stream_->timeline_semaphore, &completed_value));
  EXPECT_LT(completed_value, stream_->pending_value);

  // A normal preparation lease proves both lookup and admission were restored.
  iree_hal_streaming_retained_buffer_ref_t restored_ref = {};
  IREE_EXPECT_OK(iree_hal_streaming_memory_lookup_range_retain(
      context_, device_pointer_, kAllocationSize, &restored_ref));
  EXPECT_EQ(buffer_, restored_ref.owner_wrapper);
  iree_hal_streaming_retained_buffer_ref_deinitialize(&restored_ref);

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
  iree_hal_semaphore_release(gate);

  std::array<uint8_t, kAllocationSize> expected;
  expected.fill(0xA5);
  EXPECT_EQ(0,
            std::memcmp(buffer_->host_ptr, expected.data(), expected.size()));

  // The failed call did not consume ownership: the caller can free it normally.
  IREE_ASSERT_OK(iree_hal_streaming_memory_free_device_async(
      context_, device_pointer_, stream_));
  buffer_ = nullptr;
  device_pointer_ = 0;
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
}

TEST(GraphMemoryReferenceTest, FreeNodeClaimRequiresLiveUnclaimedPointer) {
  // A live, otherwise unowned pointer can acquire its first free-node claim.
  EXPECT_TRUE(iree_hal_streaming_graph_memory_reference_can_claim_free_node(
      /*has_pointer_reference=*/true,
      /*is_pointer_reference_claimed=*/false,
      /*has_free_node_reference=*/false));
  // A synchronous or async external free already owns the pointer reference.
  EXPECT_FALSE(iree_hal_streaming_graph_memory_reference_can_claim_free_node(
      /*has_pointer_reference=*/true,
      /*is_pointer_reference_claimed=*/true,
      /*has_free_node_reference=*/false));
  // A completed free has retired the pointer reference.
  EXPECT_FALSE(iree_hal_streaming_graph_memory_reference_can_claim_free_node(
      /*has_pointer_reference=*/false,
      /*is_pointer_reference_claimed=*/false,
      /*has_free_node_reference=*/false));
  // An existing graph free node already owns the single free-node slot.
  EXPECT_FALSE(iree_hal_streaming_graph_memory_reference_can_claim_free_node(
      /*has_pointer_reference=*/true,
      /*is_pointer_reference_claimed=*/false,
      /*has_free_node_reference=*/true));
}

TEST_F(CpuStreamingMemoryTest,
       ExternalFreeArbitratesBeforeUnexecutedGraphDestruction) {
#if !defined(IREE_PLATFORM_LINUX)
  GTEST_SKIP() << "virtual-memory dependency injection requires ELF wrapping";
#else
  ScopedGraphVirtualMemoryEmulation emulate_virtual_memory;
  for (bool use_async_free : {false, true}) {
    SCOPED_TRACE(testing::Message() << "use_async_free=" << use_async_free);

    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));
    iree_hal_streaming_graph_memory_allocation_t* allocation = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_memory_allocation_create(
        context_, /*size=*/4096, &allocation));
    iree_hal_streaming_graph_node_t* allocation_node = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
        graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
        &allocation_node));
    allocation_node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC;
    allocation_node->attrs.mem_alloc.params = nullptr;
    allocation_node->attrs.mem_alloc.params_size = 0;
    allocation_node->attrs.mem_alloc.dptr =
        iree_hal_streaming_graph_memory_allocation_device_pointer(allocation);
    allocation_node->attrs.mem_alloc.bytesize = 4096;
    allocation_node->attrs.mem_alloc.allocation = allocation;
    iree_hal_streaming_graph_memory_allocation_retain(allocation);
    allocation_node->attrs.mem_alloc.has_in_graph_free_node = false;
    graph->has_graph_memory_nodes = true;

    const iree_hal_streaming_deviceptr_t pointer =
        (iree_hal_streaming_deviceptr_t)allocation_node->attrs.mem_alloc.dptr;
    iree_hal_streaming_retained_buffer_ref_t retained_ref = {};
    IREE_ASSERT_OK(iree_hal_streaming_memory_lookup_range_retain(
        context_, pointer, /*size=*/1, &retained_ref));

    std::atomic<bool> free_returned = false;
    std::atomic<iree_status_code_t> free_status_code = IREE_STATUS_UNKNOWN;
    std::thread free_thread([&] {
      iree_status_t status =
          use_async_free
              ? iree_hal_streaming_memory_free_device_async(context_, pointer,
                                                            stream_)
              : iree_hal_streaming_memory_free_device(context_, pointer);
      free_status_code.store(iree_status_code(status),
                             std::memory_order_release);
      iree_status_ignore(status);
      free_returned.store(true, std::memory_order_release);
    });

    // The retained preparation lease holds an accepted external free between
    // table removal and its post-removal work. A synchronous free of an
    // unexecuted graph allocation must instead be rejected before removal.
    bool observed_table_removal = false;
    iree_status_code_t lookup_error = IREE_STATUS_OK;
    while (!free_returned.load(std::memory_order_acquire)) {
      iree_hal_streaming_buffer_ref_t raw_ref = {};
      iree_status_t lookup_status =
          iree_hal_streaming_memory_lookup(context_, pointer, &raw_ref);
      const iree_status_code_t lookup_code = iree_status_code(lookup_status);
      iree_status_ignore(lookup_status);
      if (lookup_code == IREE_STATUS_NOT_FOUND) {
        observed_table_removal = true;
        break;
      }
      if (lookup_code != IREE_STATUS_OK) {
        lookup_error = lookup_code;
        break;
      }
      std::this_thread::yield();
    }
    EXPECT_EQ(IREE_STATUS_OK, lookup_error);

    if (use_async_free && observed_table_removal) {
      // The async free owns the pointer claim before publication disappears,
      // so graph destruction cannot attempt a second close or table removal.
      iree_hal_streaming_graph_release(graph);
      graph = nullptr;
    }

    iree_hal_streaming_retained_buffer_ref_deinitialize(&retained_ref);
    free_thread.join();

    if (use_async_free) {
      EXPECT_TRUE(observed_table_removal);
      EXPECT_EQ(IREE_STATUS_OK,
                free_status_code.load(std::memory_order_acquire));
      IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    } else {
      EXPECT_FALSE(observed_table_removal);
      EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT,
                free_status_code.load(std::memory_order_acquire));
    }
    iree_hal_streaming_graph_release(graph);
  }
#endif  // IREE_PLATFORM_LINUX
}

TEST_F(CpuStreamingMemoryTest,
       HostFlagQueryReadsPublishedMetadataWhileAllocationIsClosing) {
  const iree_hal_streaming_host_register_flags_t allocation_flags =
      static_cast<iree_hal_streaming_host_register_flags_t>(
          IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE |
          IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, /*size=*/4096, allocation_flags, &buffer_));
  ASSERT_NE(nullptr, buffer_->host_ptr);
  iree_hal_streaming_allocation_preparation_begin_close(&buffer_->preparation);

  iree_hal_streaming_host_register_flags_t queried_flags =
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  IREE_EXPECT_OK(iree_hal_streaming_memory_host_flags(
      context_, buffer_->host_ptr, &queried_flags));
  EXPECT_EQ(allocation_flags, queried_flags);

  iree_hal_streaming_allocation_preparation_reopen(&buffer_->preparation);
}

TEST_F(CpuStreamingMemoryTest,
       SameStreamGraphLaunchReservesPointerPreparationThroughEnqueue) {
#if !defined(IREE_PLATFORM_LINUX)
  GTEST_SKIP() << "virtual-memory dependency injection requires ELF wrapping";
#else
  ScopedGraphVirtualMemoryEmulation emulate_virtual_memory;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_memory_allocation_t* allocation = nullptr;
  iree_hal_streaming_deviceptr_t pointer = 0;
  CreateGraphMemoryAllocationGraph(&graph, &allocation, &pointer);
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  g_graph_pointer_prepare_stream.store(stream_, std::memory_order_release);
  g_graph_prepare_held_stream_reservation.store(false,
                                                std::memory_order_release);
  g_graph_pointer_prepare_entered.store(false, std::memory_order_release);
  g_allow_graph_pointer_prepare_return.store(false, std::memory_order_release);
  g_pause_graph_pointer_prepare.store(true, std::memory_order_release);

  std::atomic<bool> launch_returned = false;
  std::atomic<iree_status_code_t> launch_status_code = IREE_STATUS_UNKNOWN;
  std::thread launch_thread([&] {
    iree_status_t status = LaunchGraphExecAndCheckResult(exec, stream_);
    launch_status_code.store(iree_status_code(status),
                             std::memory_order_release);
    iree_status_ignore(status);
    launch_returned.store(true, std::memory_order_release);
  });

  bool prepare_entered = false;
  for (int i = 0; i < 1000000; ++i) {
    if (g_graph_pointer_prepare_entered.load(std::memory_order_acquire)) {
      prepare_entered = true;
      break;
    }
    if (launch_returned.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  if (!prepare_entered) {
    g_allow_graph_pointer_prepare_return.store(true, std::memory_order_release);
    launch_thread.join();
    IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(exec));
    iree_hal_streaming_graph_release(graph);
    g_graph_pointer_prepare_stream.store(nullptr, std::memory_order_release);
    FAIL() << "graph launch did not reach pointer preparation";
    return;
  }

  // The same stream lock must already be held while the pointer state is
  // inspected. This synchronous probe fails deterministically if launch
  // locking is moved back after pointer preparation.
  EXPECT_TRUE(
      g_graph_prepare_held_stream_reservation.load(std::memory_order_acquire));

  std::atomic<bool> free_returned = false;
  std::atomic<iree_status_code_t> free_status_code = IREE_STATUS_UNKNOWN;
  std::thread free_thread([&] {
    iree_status_t status =
        iree_hal_streaming_memory_free_device_async(context_, pointer, stream_);
    free_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    free_returned.store(true, std::memory_order_release);
  });

  bool observed_table_removal = false;
  for (int i = 0; i < 1000000; ++i) {
    iree_hal_streaming_buffer_ref_t raw_ref = {};
    iree_status_t status =
        iree_hal_streaming_memory_lookup(context_, pointer, &raw_ref);
    if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
      observed_table_removal = true;
      iree_status_ignore(status);
      break;
    }
    iree_status_ignore(status);
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_table_removal);

  g_allow_graph_pointer_prepare_return.store(true, std::memory_order_release);
  launch_thread.join();
  free_thread.join();
  EXPECT_EQ(IREE_STATUS_OK, launch_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK, free_status_code.load(std::memory_order_acquire));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));

  // Correct same-stream ordering executes map before free, leaving neither a
  // mapping nor an unpublished pointer reference. If free overtakes enqueue,
  // the later map recreates both and this assertion exposes the leak.
  EXPECT_FALSE(
      iree_hal_streaming_graph_memory_allocation_is_mapped(allocation));
  iree_hal_streaming_buffer_ref_t raw_ref = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_hal_streaming_memory_lookup(context_, pointer, &raw_ref));

  // Keep mutation runs self-cleaning if the old ordering recreated the
  // pointer reference after free completed.
  if (iree_hal_streaming_graph_memory_allocation_claim_async_free_reference(
          allocation)) {
    IREE_EXPECT_OK(
        iree_hal_streaming_graph_memory_allocation_unmap_if_mapped(allocation));
    iree_hal_streaming_graph_memory_allocation_complete_pointer_reference(
        allocation);
    iree_hal_streaming_graph_memory_allocation_release(allocation);
  }
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(exec));
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
  EXPECT_EQ(0, device_entry_.pending_terminal_resource_count);
  g_graph_pointer_prepare_stream.store(nullptr, std::memory_order_release);
#endif  // IREE_PLATFORM_LINUX
}

TEST_F(CpuStreamingMemoryTest,
       GraphAsyncFreeRollbackRestoresVisibilityBeforeReleasingClaim) {
#if !defined(IREE_PLATFORM_LINUX)
  GTEST_SKIP() << "rollback phase injection requires ELF wrapping";
#else
  ScopedGraphVirtualMemoryEmulation emulate_virtual_memory;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_memory_allocation_t* allocation = nullptr;
  iree_hal_streaming_deviceptr_t pointer = 0;
  CreateGraphMemoryAllocationGraph(&graph, &allocation, &pointer);

  iree_hal_streaming_retained_buffer_ref_t retained_ref = {};
  IREE_ASSERT_OK(iree_hal_streaming_memory_lookup_range_retain(
      context_, pointer, /*size=*/1, &retained_ref));

  // Leave a real command buffer pending so queue_host_call must submit and
  // flush it before attempting the callback. The injected post-accept flush
  // failure then exercises the production graph-free rollback path without
  // invoking the wrapper queue's intentionally absent host-call method.
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, /*size=*/16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  const uint32_t pattern = 0xA5A5A5A5u;
  IREE_ASSERT_OK(iree_hal_streaming_memory_memset(context_, device_pointer_,
                                                  /*count=*/16, &pattern,
                                                  sizeof(pattern), stream_));

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  iree_hal_queue_release(original_queue);
  fault_queue.fail_flush.store(true, std::memory_order_release);

  g_graph_pointer_restore_entered.store(false, std::memory_order_release);
  g_allow_graph_pointer_restore_return.store(false, std::memory_order_release);
  g_pause_after_graph_pointer_restore.store(true, std::memory_order_release);

  std::atomic<iree_status_code_t> free_status_code = IREE_STATUS_UNKNOWN;
  std::thread free_thread([&] {
    iree_status_t status =
        iree_hal_streaming_memory_free_device_async(context_, pointer, stream_);
    free_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
  });

  bool observed_table_removal = false;
  for (int i = 0; i < 1000000; ++i) {
    iree_hal_streaming_buffer_ref_t raw_ref = {};
    iree_status_t status =
        iree_hal_streaming_memory_lookup(context_, pointer, &raw_ref);
    if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
      observed_table_removal = true;
      iree_status_ignore(status);
      break;
    }
    iree_status_ignore(status);
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_table_removal);
  iree_hal_streaming_retained_buffer_ref_deinitialize(&retained_ref);

  bool restore_entered = false;
  for (int i = 0; i < 1000000; ++i) {
    if (g_graph_pointer_restore_entered.load(std::memory_order_acquire)) {
      restore_entered = true;
      break;
    }
    std::this_thread::yield();
  }
  if (restore_entered) {
    // The wrapper pauses immediately after unclaim. Correct rollback has
    // already reinserted and reopened the exact reserved table entry, so real
    // graph destruction can claim and unpublish it without a second close.
    iree_hal_streaming_graph_release(graph);
    graph = nullptr;
  }
  g_allow_graph_pointer_restore_return.store(true, std::memory_order_release);
  free_thread.join();

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));

  EXPECT_TRUE(restore_entered);
  EXPECT_EQ(IREE_STATUS_INTERNAL,
            free_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(1,
            fault_queue.injected_flush_count.load(std::memory_order_acquire));
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
  EXPECT_EQ(0, device_entry_.pending_terminal_resource_count);
#endif  // IREE_PLATFORM_LINUX
}

TEST_F(CpuStreamingMemoryTest,
       GraphAsyncFreeCancellationRemainsInTerminalResourceDrain) {
#if !defined(IREE_PLATFORM_LINUX)
  GTEST_SKIP() << "virtual-memory dependency injection requires ELF wrapping";
#else
  ScopedGraphVirtualMemoryEmulation emulate_virtual_memory;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_memory_allocation_t* allocation = nullptr;
  iree_hal_streaming_deviceptr_t pointer = 0;
  CreateGraphMemoryAllocationGraph(&graph, &allocation, &pointer);

  ControlledCancellationQueue cancellation_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeControlledCancellationQueue(original_queue, &cancellation_queue);
  EXPECT_EQ(original_queue,
            ReplaceStreamQueue(stream_, &cancellation_queue.base));
  iree_hal_queue_release(original_queue);

  IREE_ASSERT_OK(
      iree_hal_streaming_memory_free_device_async(context_, pointer, stream_));
  ASSERT_TRUE(cancellation_queue.accepted.load(std::memory_order_acquire));
  ASSERT_TRUE(cancellation_queue.retained_resource);
  EXPECT_EQ(1, PendingTerminalResourceCount(&device_entry_));

  // The pending free owns the returned-pointer claim, so destroying its
  // unexecuted allocation graph leaves the pointer reference for either
  // callback completion or cancellation rollback.
  iree_hal_streaming_graph_release(graph);
  graph = nullptr;

  // Publish terminal stream failure first, matching backends that release a
  // notification-ring resource only after waiters can observe cancellation.
  iree_hal_semaphore_fail(
      cancellation_queue.signal_semaphore,
      iree_make_status(IREE_STATUS_CANCELLED,
                       "injected queued host-call cancellation"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(1, PendingTerminalResourceCount(&device_entry_));

  std::atomic<bool> drain_started = false;
  std::atomic<bool> drain_returned = false;
  std::thread drain_thread([&] {
    drain_started.store(true, std::memory_order_release);
    iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
    drain_returned.store(true, std::memory_order_release);
  });
  while (!drain_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(drain_returned.load(std::memory_order_acquire));

  // Releasing the queue-owned resource performs cancellation rollback,
  // including table restoration and final context release, before it drops
  // the generalized teardown token and wakes the drain.
  ReleaseControlledCancellationResource(&cancellation_queue);
  drain_thread.join();
  EXPECT_TRUE(drain_returned.load(std::memory_order_acquire));
  EXPECT_EQ(0, PendingTerminalResourceCount(&device_entry_));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);

  // Cancellation restored an unexecuted pointer after its source graph was
  // destroyed. Retry the stream-ordered free on a fresh stream to retire it.
  iree_hal_streaming_stream_t* retry_stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &retry_stream));
  IREE_ASSERT_OK(iree_hal_streaming_memory_free_device_async(context_, pointer,
                                                             retry_stream));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(retry_stream));
  iree_hal_streaming_stream_release(retry_stream);
  iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
  EXPECT_EQ(0, PendingTerminalResourceCount(&device_entry_));
#endif  // IREE_PLATFORM_LINUX
}

TEST_F(CpuStreamingMemoryTest,
       AsyncFreeCancellationRemainsInTerminalResourceDrain) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);

  ControlledCancellationQueue cancellation_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeControlledCancellationQueue(original_queue, &cancellation_queue);
  EXPECT_EQ(original_queue,
            ReplaceStreamQueue(stream_, &cancellation_queue.base));
  iree_hal_queue_release(original_queue);

  IREE_ASSERT_OK(iree_hal_streaming_memory_free_device_async(
      context_, device_pointer_, stream_));
  buffer_ = nullptr;
  device_pointer_ = 0;
  ASSERT_TRUE(cancellation_queue.accepted.load(std::memory_order_acquire));
  ASSERT_TRUE(cancellation_queue.retained_resource);
  EXPECT_EQ(1, PendingTerminalResourceCount(&device_entry_));

  // A backend may fail the stream signal before releasing its retained host
  // call state. Teardown must still see the pending terminal resource.
  iree_hal_semaphore_fail(
      cancellation_queue.signal_semaphore,
      iree_make_status(IREE_STATUS_CANCELLED,
                       "injected queued async-free cancellation"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(1, PendingTerminalResourceCount(&device_entry_));

  std::atomic<bool> drain_started = false;
  std::atomic<bool> drain_returned = false;
  std::thread drain_thread([&] {
    drain_started.store(true, std::memory_order_release);
    iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
    drain_returned.store(true, std::memory_order_release);
  });
  while (!drain_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(drain_returned.load(std::memory_order_acquire));

  ReleaseControlledCancellationResource(&cancellation_queue);
  drain_thread.join();
  EXPECT_TRUE(drain_returned.load(std::memory_order_acquire));
  EXPECT_EQ(0, PendingTerminalResourceCount(&device_entry_));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
}

TEST_F(CpuStreamingMemoryTest,
       HostD2HStagingCancellationRemainsInTerminalResourceDrain) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);
  std::memset(buffer_->host_ptr, 0x5A, kAllocationSize);
  // The CPU allocator is host-visible. Clear only the wrapper classification
  // so the production D2H staging/callback path is used.
  buffer_->memory_type =
      (iree_hal_memory_type_t)(buffer_->memory_type &
                               ~IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
  std::array<uint8_t, kAllocationSize> destination = {};

  ControlledCancellationQueue cancellation_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeControlledCancellationQueue(original_queue, &cancellation_queue);
  EXPECT_EQ(original_queue,
            ReplaceStreamQueue(stream_, &cancellation_queue.base));
  iree_hal_queue_release(original_queue);

  IREE_ASSERT_OK(iree_hal_streaming_memcpy_device_to_host(
      context_, destination.data(), device_pointer_, kAllocationSize, stream_));
  ASSERT_TRUE(cancellation_queue.accepted.load(std::memory_order_acquire));
  ASSERT_TRUE(cancellation_queue.retained_resource);
  ASSERT_TRUE(cancellation_queue.wait_semaphore);
  EXPECT_EQ(1, PendingTerminalResourceCount(&device_entry_));

  // Let the accepted device-to-staging copy finish before emulating
  // cancellation of the later host scatter operation.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      cancellation_queue.wait_semaphore, cancellation_queue.wait_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_fail(
      cancellation_queue.signal_semaphore,
      iree_make_status(IREE_STATUS_CANCELLED,
                       "injected staged D2H host-call cancellation"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(1, PendingTerminalResourceCount(&device_entry_));

  std::atomic<bool> drain_started = false;
  std::atomic<bool> drain_returned = false;
  std::thread drain_thread([&] {
    drain_started.store(true, std::memory_order_release);
    iree_hal_streaming_device_terminal_resource_await_idle(&device_entry_);
    drain_returned.store(true, std::memory_order_release);
  });
  while (!drain_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(drain_returned.load(std::memory_order_acquire));

  ReleaseControlledCancellationResource(&cancellation_queue);
  drain_thread.join();
  EXPECT_TRUE(drain_returned.load(std::memory_order_acquire));
  EXPECT_EQ(0, PendingTerminalResourceCount(&device_entry_));
  const std::array<uint8_t, kAllocationSize> empty_destination = {};
  EXPECT_EQ(empty_destination, destination);

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
}

TEST_F(CpuStreamingMemoryTest,
       RetainedAcrossContextLookupKeepsMetadataLiveAcrossTableRemoval) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);

  iree_hal_streaming_retained_buffer_ref_t retained_ref = {};
  IREE_ASSERT_OK(iree_hal_streaming_memory_lookup_range_retain_across_contexts(
      context_, device_pointer_, kAllocationSize, &retained_ref));
  ASSERT_EQ(buffer_, retained_ref.owner_wrapper);

#if defined(IREE_PLATFORM_LINUX)
  iree_hal_streaming_allocation_preparation_t* observed_preparation =
      &buffer_->preparation;
  ArmAllocationPreparationAwaitIdleObservation(observed_preparation);
#endif  // IREE_PLATFORM_LINUX
  std::atomic<bool> free_returned = false;
  std::atomic<iree_status_code_t> free_status_code = IREE_STATUS_UNKNOWN;
  std::thread free_thread([&] {
    iree_status_t status =
        iree_hal_streaming_memory_free_device(context_, device_pointer_);
    free_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    free_returned.store(true, std::memory_order_release);
  });

#if defined(IREE_PLATFORM_LINUX)
  // Wait until this allocation's free enters the production lease drain. A
  // return or deadline before entry is a deterministic contract failure.
  if (!WaitForAllocationPreparationAwaitIdleEntry(free_returned)) {
    // Once the sole lifetime barrier is absent, neither the raw preparation
    // pointer nor the retained reference can be touched without racing free.
    std::abort();
  }
#else
  // The free first removes the pointer-table entry, then must wait for the
  // retained lookup's preparation lease before it can release the wrapper.
  bool observed_table_removal = false;
  for (int i = 0; i < 1000000; ++i) {
    iree_hal_streaming_buffer_ref_t raw_ref = {};
    iree_status_t status =
        iree_hal_streaming_memory_lookup(context_, device_pointer_, &raw_ref);
    if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
      observed_table_removal = true;
      iree_status_ignore(status);
      break;
    }
    iree_status_ignore(status);
    std::this_thread::yield();
  }
  if (!observed_table_removal) {
    iree_hal_streaming_retained_buffer_ref_deinitialize(&retained_ref);
    free_thread.join();
    FAIL() << "competing free never removed the retained allocation";
    return;
  }
#endif  // IREE_PLATFORM_LINUX

  EXPECT_FALSE(free_returned.load(std::memory_order_acquire));
  iree_hal_streaming_buffer_ref_t raw_ref = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_hal_streaming_memory_lookup(context_, device_pointer_, &raw_ref));
  // This is the same owner metadata graph-allocation lookup reads before
  // taking its independent allocation reference. It remains safe after table
  // removal because the retained reference still owns the wrapper and lease.
  EXPECT_EQ(buffer_, retained_ref.owner_wrapper);
  EXPECT_EQ(nullptr, retained_ref.owner_wrapper->graph_memory_allocation);
  EXPECT_NE(nullptr, retained_ref.buffer);

  iree_hal_streaming_retained_buffer_ref_deinitialize(&retained_ref);
  free_thread.join();
  EXPECT_TRUE(free_returned.load(std::memory_order_acquire));
  // This isolated fixture has no global registry, so synchronization fails
  // after the lease drains and the free atomically republishes the allocation.
  EXPECT_EQ(IREE_STATUS_FAILED_PRECONDITION,
            free_status_code.load(std::memory_order_acquire));

  iree_hal_streaming_retained_buffer_ref_t restored_ref = {};
  IREE_EXPECT_OK(iree_hal_streaming_memory_lookup_range_retain(
      context_, device_pointer_, kAllocationSize, &restored_ref));
  EXPECT_EQ(buffer_, restored_ref.owner_wrapper);
  iree_hal_streaming_retained_buffer_ref_deinitialize(&restored_ref);
}

TEST_F(CpuStreamingMemoryTest,
       WrappedBufferFinalReleaseDrainsRetainedLookupLease) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);

  iree_hal_streaming_retained_buffer_ref_t retained_ref = {};
  IREE_ASSERT_OK(iree_hal_streaming_memory_lookup_range_retain_across_contexts(
      context_, device_pointer_, kAllocationSize, &retained_ref));
  iree_hal_streaming_buffer_t* wrapper = buffer_;
  buffer_ = nullptr;
  device_pointer_ = 0;

  std::atomic<bool> release_returned = false;
  std::thread release_thread([&] {
    iree_hal_streaming_memory_release_wrapped_buffer(wrapper);
    release_returned.store(true, std::memory_order_release);
  });

  bool observed_table_removal = false;
  for (int i = 0; i < 1000000; ++i) {
    iree_hal_streaming_buffer_ref_t raw_ref = {};
    iree_status_t status = iree_hal_streaming_memory_lookup(
        context_, iree_hal_streaming_buffer_device_pointer(wrapper), &raw_ref);
    if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
      observed_table_removal = true;
      iree_status_ignore(status);
      break;
    }
    iree_status_ignore(status);
    std::this_thread::yield();
  }
  if (!observed_table_removal) {
    iree_hal_streaming_retained_buffer_ref_deinitialize(&retained_ref);
    release_thread.join();
    FAIL() << "final release never removed the retained allocation";
    return;
  }

  EXPECT_FALSE(release_returned.load(std::memory_order_acquire));
  EXPECT_EQ(wrapper, retained_ref.owner_wrapper);
  iree_hal_streaming_retained_buffer_ref_deinitialize(&retained_ref);
  release_thread.join();
  EXPECT_TRUE(release_returned.load(std::memory_order_acquire));
}
}  // namespace
