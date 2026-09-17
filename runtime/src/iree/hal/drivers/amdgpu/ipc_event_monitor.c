// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/ipc_event_monitor.h"

#include <limits.h>
#include <stdint.h>

#include "iree/base/allocator.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/base/time.h"

typedef enum iree_hal_amdgpu_ipc_event_monitor_operation_state_e {
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_INITIALIZED = 0,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_ADMITTED,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_PUBLISHED,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_SCHEDULED,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_POLLING,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_CANCELLED,
} iree_hal_amdgpu_ipc_event_monitor_operation_state_t;

typedef enum iree_hal_amdgpu_ipc_event_monitor_schedule_kind_e {
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_NONE = 0,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_IMMEDIATE,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DORMANT,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_CALLBACK_WAIT,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_100_US,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_200_US,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_400_US,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_800_US,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_1000_US,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_COUNT,
} iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t;

typedef enum iree_hal_amdgpu_ipc_event_monitor_state_e {
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_STOPPED = 0,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_RUNNING,
  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_CLOSING,
} iree_hal_amdgpu_ipc_event_monitor_state_t;

typedef struct iree_hal_amdgpu_ipc_event_monitor_schedule_list_t {
  // First operation in FIFO order.
  iree_hal_amdgpu_ipc_event_monitor_operation_t* head;
  // Last operation in FIFO order.
  iree_hal_amdgpu_ipc_event_monitor_operation_t* tail;
} iree_hal_amdgpu_ipc_event_monitor_schedule_list_t;

// One lazy process-wide worker used only by IPC event record and wait bridges.
// Synchronization objects have process lifetime so shutdown can join and a
// later admission can restart without observing a prior worker generation.
typedef struct iree_hal_amdgpu_ipc_event_monitor_t {
  // Serializes worker creation, admission closure, and restart.
  iree_mutex_t mutex;
  // Protects the intrusive scheduler lists and active-node state transitions.
  iree_mutex_t schedule_mutex;
  // Wakes the worker after publication, callback readiness, or close.
  iree_notification_t wake_notification;
  // Wakes local preparation blocked on shutdown rather than worker activity.
  iree_notification_t shutdown_notification;
  // Wakes concurrent shutdown callers after the worker has stopped.
  iree_notification_t state_changed;
  // Atomic MPSC stack of committed operations awaiting worker ownership.
  iree_atomic_intptr_t pending_head;
  // Intrusive lists indexed by schedule kind and protected by |schedule_mutex|.
  iree_hal_amdgpu_ipc_event_monitor_schedule_list_t
      schedule_lists[IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_COUNT];
  // Number of admitted operations not yet completed or cancelled.
  iree_atomic_int64_t outstanding_operation_count;
  // Saturating scheduler-node inspection count guarded by |schedule_mutex|.
  int64_t schedule_node_visit_count;
  // Number of cancellation notifications posted while shutdown is active.
  iree_atomic_int64_t cancellation_wake_count_for_testing;
  // True when active operations must take their shutdown-resolution path.
  iree_atomic_int32_t shutdown_requested;
  // Lifecycle state protected by |mutex|.
  iree_hal_amdgpu_ipc_event_monitor_state_t state;
  // Joinable worker owned while RUNNING or CLOSING.
  iree_thread_t* thread;
  // Injects worker creation failure for deterministic rollback testing.
  bool fail_thread_create_for_testing;
} iree_hal_amdgpu_ipc_event_monitor_t;

static iree_once_flag iree_hal_amdgpu_ipc_event_monitor_once =
    IREE_ONCE_FLAG_INIT;
static iree_hal_amdgpu_ipc_event_monitor_t iree_hal_amdgpu_ipc_event_monitor;

// Bound consecutive immediate polls before giving an expired finite operation
// one service opportunity. Both classes retain progress under sustained load.
#define IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_IMMEDIATE_SERVICE_QUOTA 4

static int iree_hal_amdgpu_ipc_event_monitor_thread_main(void* user_data);

static void iree_hal_amdgpu_ipc_event_monitor_initialize(void) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      &iree_hal_amdgpu_ipc_event_monitor;
  iree_mutex_initialize(&monitor->mutex);
  iree_mutex_initialize(&monitor->schedule_mutex);
  iree_notification_initialize(&monitor->wake_notification);
  iree_notification_initialize(&monitor->shutdown_notification);
  iree_notification_initialize(&monitor->state_changed);
  iree_atomic_store(&monitor->pending_head, 0, iree_memory_order_relaxed);
  for (iree_host_size_t i = 0;
       i < IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_COUNT; ++i) {
    monitor->schedule_lists[i].head = NULL;
    monitor->schedule_lists[i].tail = NULL;
  }
  iree_atomic_store(&monitor->outstanding_operation_count, 0,
                    iree_memory_order_relaxed);
  monitor->schedule_node_visit_count = 0;
  iree_atomic_store(&monitor->cancellation_wake_count_for_testing, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&monitor->shutdown_requested, 0, iree_memory_order_relaxed);
  monitor->state = IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_STOPPED;
  monitor->thread = NULL;
  monitor->fail_thread_create_for_testing = false;
}

static iree_hal_amdgpu_ipc_event_monitor_t*
iree_hal_amdgpu_ipc_event_monitor_get(void) {
  iree_call_once(&iree_hal_amdgpu_ipc_event_monitor_once,
                 iree_hal_amdgpu_ipc_event_monitor_initialize);
  return &iree_hal_amdgpu_ipc_event_monitor;
}

static iree_hal_amdgpu_ipc_event_monitor_schedule_list_t*
iree_hal_amdgpu_ipc_event_monitor_schedule_list(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor,
    iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t schedule_kind) {
  IREE_ASSERT_GT(schedule_kind,
                 IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_NONE);
  IREE_ASSERT_LT(schedule_kind,
                 IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_COUNT);
  return &monitor->schedule_lists[schedule_kind];
}

static void iree_hal_amdgpu_ipc_event_monitor_note_node_visit(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor) {
  IREE_ASSERT(monitor->schedule_node_visit_count >= 0);
  if (monitor->schedule_node_visit_count < INT64_MAX) {
    ++monitor->schedule_node_visit_count;
  }
}

// Appends |operation| to |schedule_kind|. Must hold |schedule_mutex|.
static void iree_hal_amdgpu_ipc_event_monitor_schedule_append_locked(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor,
    iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t schedule_kind,
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation) {
  IREE_ASSERT_EQ(operation->schedule_kind,
                 IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_NONE);
  IREE_ASSERT(!operation->schedule_next);
  IREE_ASSERT(!operation->schedule_previous);
  iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* list =
      iree_hal_amdgpu_ipc_event_monitor_schedule_list(monitor, schedule_kind);
  operation->schedule_previous = list->tail;
  operation->schedule_next = NULL;
  if (list->tail) {
    list->tail->schedule_next = operation;
  } else {
    list->head = operation;
  }
  list->tail = operation;
  operation->schedule_kind = (uint8_t)schedule_kind;
}

// Unlinks |operation| from its scheduler list. Must hold |schedule_mutex|.
static void iree_hal_amdgpu_ipc_event_monitor_schedule_unlink_locked(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor,
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation) {
  const iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t schedule_kind =
      (iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t)
          operation->schedule_kind;
  iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* list =
      iree_hal_amdgpu_ipc_event_monitor_schedule_list(monitor, schedule_kind);
  if (operation->schedule_previous) {
    operation->schedule_previous->schedule_next = operation->schedule_next;
  } else {
    IREE_ASSERT(list->head == operation);
    list->head = operation->schedule_next;
  }
  if (operation->schedule_next) {
    operation->schedule_next->schedule_previous = operation->schedule_previous;
  } else {
    IREE_ASSERT(list->tail == operation);
    list->tail = operation->schedule_previous;
  }
  operation->schedule_next = NULL;
  operation->schedule_previous = NULL;
  operation->schedule_kind =
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_NONE;
}

void iree_hal_amdgpu_ipc_event_monitor_set_fail_thread_create_for_testing(
    bool fail_thread_create) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_mutex_lock(&monitor->mutex);
  IREE_ASSERT(monitor->state ==
              IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_STOPPED);
  monitor->fail_thread_create_for_testing = fail_thread_create;
  iree_mutex_unlock(&monitor->mutex);
}

int64_t iree_hal_amdgpu_ipc_event_monitor_schedule_node_visit_count_for_testing(
    void) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_mutex_lock(&monitor->schedule_mutex);
  const int64_t visit_count = monitor->schedule_node_visit_count;
  iree_mutex_unlock(&monitor->schedule_mutex);
  return visit_count;
}

void iree_hal_amdgpu_ipc_event_monitor_reset_schedule_node_visit_count_for_testing(
    void) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_mutex_lock(&monitor->schedule_mutex);
  monitor->schedule_node_visit_count = 0;
  iree_mutex_unlock(&monitor->schedule_mutex);
}

int64_t iree_hal_amdgpu_ipc_event_monitor_cancellation_wake_count_for_testing(
    void) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  return iree_atomic_load(&monitor->cancellation_wake_count_for_testing,
                          iree_memory_order_acquire);
}

void iree_hal_amdgpu_ipc_event_monitor_reset_cancellation_wake_count_for_testing(
    void) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_atomic_store(&monitor->cancellation_wake_count_for_testing, 0,
                    iree_memory_order_release);
}

void iree_hal_amdgpu_ipc_event_monitor_make_finite_operations_due_for_testing(
    void) {
  static const iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t
      kDeadlineKinds[] = {
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_100_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_200_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_400_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_800_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_1000_US,
      };
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_mutex_lock(&monitor->schedule_mutex);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kDeadlineKinds); ++i) {
    iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* list =
        iree_hal_amdgpu_ipc_event_monitor_schedule_list(monitor,
                                                        kDeadlineKinds[i]);
    for (iree_hal_amdgpu_ipc_event_monitor_operation_t* operation = list->head;
         operation; operation = operation->schedule_next) {
      operation->next_poll_deadline_ns = IREE_TIME_INFINITE_PAST;
    }
  }
  iree_mutex_unlock(&monitor->schedule_mutex);
  iree_notification_post(&monitor->wake_notification, IREE_ALL_WAITERS);
}

iree_host_size_t
iree_hal_amdgpu_ipc_event_monitor_make_dormant_operations_due_for_testing(
    void) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_host_size_t operation_count = 0;
  iree_mutex_lock(&monitor->schedule_mutex);
  iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* dormant_list =
      iree_hal_amdgpu_ipc_event_monitor_schedule_list(
          monitor, IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DORMANT);
  while (dormant_list->head) {
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation =
        dormant_list->head;
    iree_hal_amdgpu_ipc_event_monitor_schedule_unlink_locked(monitor,
                                                             operation);
    operation->next_poll_deadline_ns = IREE_TIME_INFINITE_PAST;
    iree_hal_amdgpu_ipc_event_monitor_schedule_append_locked(
        monitor, IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_100_US,
        operation);
    ++operation_count;
  }
  iree_mutex_unlock(&monitor->schedule_mutex);
  iree_notification_post(&monitor->wake_notification, IREE_ALL_WAITERS);
  return operation_count;
}

static bool iree_hal_amdgpu_ipc_event_monitor_is_shutting_down(
    void* user_data) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      (iree_hal_amdgpu_ipc_event_monitor_t*)user_data;
  return iree_atomic_load(&monitor->shutdown_requested,
                          iree_memory_order_acquire) != 0;
}

bool iree_hal_amdgpu_ipc_event_monitor_await_shutdown(iree_timeout_t timeout) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  return iree_notification_await(
      &monitor->shutdown_notification,
      iree_hal_amdgpu_ipc_event_monitor_is_shutting_down, monitor, timeout);
}

void iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
    iree_hal_amdgpu_ipc_event_monitor_poll_fn_t poll,
    iree_duration_t maximum_poll_delay_ns,
    iree_hal_amdgpu_ipc_event_monitor_operation_t* out_operation) {
  IREE_ASSERT_ARGUMENT(poll);
  IREE_ASSERT_ARGUMENT(out_operation);
  IREE_ASSERT(maximum_poll_delay_ns == IREE_DURATION_INFINITE ||
              maximum_poll_delay_ns ==
                  IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_EXTERNAL_MAX_POLL_DELAY_NS);
  out_operation->pending_next = NULL;
  out_operation->schedule_next = NULL;
  out_operation->schedule_previous = NULL;
  iree_atomic_store(
      &out_operation->state,
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_INITIALIZED,
      iree_memory_order_relaxed);
  iree_atomic_store(&out_operation->poll_requested, 0,
                    iree_memory_order_relaxed);
  out_operation->maximum_poll_delay_ns = maximum_poll_delay_ns;
  out_operation->current_poll_delay_ns =
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_INITIAL_POLL_DELAY_NS;
  out_operation->next_poll_deadline_ns = IREE_TIME_INFINITE_FUTURE;
  out_operation->schedule_kind =
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_NONE;
  out_operation->has_been_polled = false;
  out_operation->poll = poll;
}

iree_status_t iree_hal_amdgpu_ipc_event_monitor_operation_admit(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation) {
  IREE_ASSERT_ARGUMENT(operation);
  IREE_ASSERT_EQ(iree_atomic_load(&operation->state, iree_memory_order_relaxed),
                 IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_INITIALIZED);

  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_mutex_lock(&monitor->mutex);

  iree_status_t status = iree_ok_status();
  if (monitor->state == IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_CLOSING) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "IPC event monitor is shutting down");
  } else {
    const int64_t outstanding_operation_count = iree_atomic_load(
        &monitor->outstanding_operation_count, iree_memory_order_relaxed);
    IREE_ASSERT_GE(outstanding_operation_count, 0);
    if (IREE_UNLIKELY(outstanding_operation_count == INT64_MAX)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "too many outstanding IPC event operations");
    } else if (monitor->state ==
               IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_STOPPED) {
      iree_atomic_store(&monitor->shutdown_requested, 0,
                        iree_memory_order_relaxed);
      const iree_thread_create_params_t thread_params = {
          .name = IREE_SV("amdgpu-ipc-monitor"),
          .priority_class = IREE_THREAD_PRIORITY_CLASS_LOW,
      };
      if (monitor->fail_thread_create_for_testing) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "injected IPC event monitor worker creation failure");
      } else {
        status = iree_thread_create(
            iree_hal_amdgpu_ipc_event_monitor_thread_main, monitor,
            thread_params, iree_allocator_system(), &monitor->thread);
      }
      if (iree_status_is_ok(status)) {
        monitor->state = IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_RUNNING;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const int64_t old_count = iree_atomic_fetch_add(
        &monitor->outstanding_operation_count, 1, iree_memory_order_relaxed);
    IREE_ASSERT_GE(old_count, 0);
    iree_atomic_store(
        &operation->state,
        IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_ADMITTED,
        iree_memory_order_release);
  }

  iree_mutex_unlock(&monitor->mutex);
  return status;
}

void iree_hal_amdgpu_ipc_event_monitor_operation_publish(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation) {
  IREE_ASSERT_ARGUMENT(operation);
  const int32_t old_state = iree_atomic_exchange(
      &operation->state,
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_PUBLISHED,
      iree_memory_order_acq_rel);
  IREE_ASSERT_EQ(old_state,
                 IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_ADMITTED);

  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  intptr_t pending_head =
      iree_atomic_load(&monitor->pending_head, iree_memory_order_relaxed);
  do {
    operation->pending_next =
        (iree_hal_amdgpu_ipc_event_monitor_operation_t*)pending_head;
  } while (!iree_atomic_compare_exchange_weak(
      &monitor->pending_head, &pending_head, (intptr_t)operation,
      iree_memory_order_release, iree_memory_order_relaxed));
  // The successful MPSC publication above is the caller's final operation
  // access. Only process-lifetime monitor state is touched from here onward.
  iree_notification_post(&monitor->wake_notification, IREE_ALL_WAITERS);
}

void iree_hal_amdgpu_ipc_event_monitor_operation_request_poll(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation) {
  IREE_ASSERT_ARGUMENT(operation);
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_mutex_lock(&monitor->schedule_mutex);
  const int32_t state =
      iree_atomic_load(&operation->state, iree_memory_order_acquire);
  if (state == IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_SCHEDULED) {
    iree_hal_amdgpu_ipc_event_monitor_note_node_visit(monitor);
    if (operation->schedule_kind !=
        IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_IMMEDIATE) {
      iree_hal_amdgpu_ipc_event_monitor_schedule_unlink_locked(monitor,
                                                               operation);
      iree_hal_amdgpu_ipc_event_monitor_schedule_append_locked(
          monitor, IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_IMMEDIATE,
          operation);
    }
  } else {
    IREE_ASSERT(
        state == IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_ADMITTED ||
        state == IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_PUBLISHED ||
        state == IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_POLLING);
  }
  // This release store is the callback's final operation access and ownership
  // handoff. A transaction waiting on it may free unpublished storage, and the
  // worker may poll and free published storage after acquiring it.
  iree_atomic_store(&operation->poll_requested, 1, iree_memory_order_release);
  iree_mutex_unlock(&monitor->schedule_mutex);
  iree_notification_post(&monitor->wake_notification, IREE_ALL_WAITERS);
}

static bool iree_hal_amdgpu_ipc_event_monitor_poll_was_requested(
    void* user_data) {
  iree_hal_amdgpu_ipc_event_monitor_operation_t* operation =
      (iree_hal_amdgpu_ipc_event_monitor_operation_t*)user_data;
  return iree_atomic_load(&operation->poll_requested,
                          iree_memory_order_acquire) != 0;
}

bool iree_hal_amdgpu_ipc_event_monitor_operation_await_poll_request(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(operation);
  return iree_notification_await(
      &iree_hal_amdgpu_ipc_event_monitor.wake_notification,
      iree_hal_amdgpu_ipc_event_monitor_poll_was_requested, operation, timeout);
}

static void iree_hal_amdgpu_ipc_event_monitor_retire_operation(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor) {
  const int64_t old_count = iree_atomic_fetch_add(
      &monitor->outstanding_operation_count, -1, iree_memory_order_acq_rel);
  IREE_ASSERT_GT(old_count, 0);
}

void iree_hal_amdgpu_ipc_event_monitor_operation_cancel(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation) {
  IREE_ASSERT_ARGUMENT(operation);
  const int32_t old_state = iree_atomic_exchange(
      &operation->state,
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_CANCELLED,
      iree_memory_order_acq_rel);
  IREE_ASSERT_EQ(old_state,
                 IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_ADMITTED);
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_hal_amdgpu_ipc_event_monitor_retire_operation(monitor);
  // Only shutdown waits for unpublished admissions to retire. The acquire
  // pairs with admission closure so a cancellation racing close cannot miss
  // the worker wake that lets it observe the final outstanding count.
  if (iree_atomic_load(&monitor->shutdown_requested,
                       iree_memory_order_acquire) != 0) {
    iree_atomic_fetch_add(&monitor->cancellation_wake_count_for_testing, 1,
                          iree_memory_order_relaxed);
    iree_notification_post(&monitor->wake_notification, IREE_ALL_WAITERS);
  }
}

static iree_duration_t iree_hal_amdgpu_ipc_event_monitor_advance_delay(
    iree_duration_t current_delay_ns) {
  switch (current_delay_ns) {
    case 100000:
      return 200000;
    case 200000:
      return 400000;
    case 400000:
      return 800000;
    case 800000:
    case 1000000:
      return 1000000;
    default:
      IREE_ASSERT(false, "unexpected IPC event monitor poll delay");
      return 1000000;
  }
}

static iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t
iree_hal_amdgpu_ipc_event_monitor_schedule_kind_for_delay(
    iree_duration_t delay_ns) {
  switch (delay_ns) {
    case 100000:
      return IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_100_US;
    case 200000:
      return IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_200_US;
    case 400000:
      return IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_400_US;
    case 800000:
      return IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_800_US;
    case 1000000:
      return IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_1000_US;
    default:
      IREE_ASSERT(false, "unexpected IPC event monitor poll delay");
      return IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_1000_US;
  }
}

static iree_time_t iree_hal_amdgpu_ipc_event_monitor_deadline_after(
    iree_time_t now_ns, iree_duration_t delay_ns) {
  IREE_ASSERT_GT(delay_ns, 0);
  if (now_ns > IREE_TIME_INFINITE_FUTURE - delay_ns) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  return now_ns + delay_ns;
}

// Transfers the current MPSC batch into the immediate FIFO. Reversing the
// stack restores producer publication order without touching dormant nodes.
static void iree_hal_amdgpu_ipc_event_monitor_activate_published(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor) {
  iree_hal_amdgpu_ipc_event_monitor_operation_t* published_head =
      (iree_hal_amdgpu_ipc_event_monitor_operation_t*)iree_atomic_exchange(
          &monitor->pending_head, 0, iree_memory_order_acquire);
  iree_hal_amdgpu_ipc_event_monitor_operation_t* ordered_head = NULL;
  while (published_head) {
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation = published_head;
    published_head = operation->pending_next;
    operation->pending_next = ordered_head;
    ordered_head = operation;
  }
  if (!ordered_head) return;

  iree_mutex_lock(&monitor->schedule_mutex);
  while (ordered_head) {
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation = ordered_head;
    ordered_head = operation->pending_next;
    operation->pending_next = NULL;
    iree_hal_amdgpu_ipc_event_monitor_note_node_visit(monitor);
    const int32_t old_state = iree_atomic_exchange(
        &operation->state,
        IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_SCHEDULED,
        iree_memory_order_acq_rel);
    IREE_ASSERT_EQ(old_state,
                   IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_PUBLISHED);
    iree_hal_amdgpu_ipc_event_monitor_schedule_append_locked(
        monitor, IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_IMMEDIATE,
        operation);
  }
  iree_mutex_unlock(&monitor->schedule_mutex);
}

// Moves nodes that require a shutdown poll to the immediate FIFO exactly once.
// Callback-wait nodes remain dormant until their excluded callback publishes
// readiness; retrying cancellation cannot make progress.
static void iree_hal_amdgpu_ipc_event_monitor_sweep_for_shutdown(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor) {
  static const iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t kSweepKinds[] =
      {
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DORMANT,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_100_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_200_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_400_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_800_US,
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_1000_US,
      };
  iree_mutex_lock(&monitor->schedule_mutex);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kSweepKinds); ++i) {
    iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* list =
        iree_hal_amdgpu_ipc_event_monitor_schedule_list(monitor,
                                                        kSweepKinds[i]);
    while (list->head) {
      iree_hal_amdgpu_ipc_event_monitor_operation_t* operation = list->head;
      iree_hal_amdgpu_ipc_event_monitor_note_node_visit(monitor);
      iree_hal_amdgpu_ipc_event_monitor_schedule_unlink_locked(monitor,
                                                               operation);
      iree_hal_amdgpu_ipc_event_monitor_schedule_append_locked(
          monitor, IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_IMMEDIATE,
          operation);
    }
  }
  iree_mutex_unlock(&monitor->schedule_mutex);
}

// Selects one immediate or expired operation and transfers it to POLLING.
// Returns NULL with the earliest finite deadline in |out_next_deadline_ns| when
// no operation is ready. All list work is constant time per fixed bucket.
static iree_hal_amdgpu_ipc_event_monitor_operation_t*
iree_hal_amdgpu_ipc_event_monitor_take_ready(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor, iree_time_t now_ns,
    iree_host_size_t* inout_immediate_service_count, bool* out_is_first_poll,
    bool* out_was_requested, iree_time_t* out_next_deadline_ns) {
  *out_is_first_poll = false;
  *out_was_requested = false;
  *out_next_deadline_ns = IREE_TIME_INFINITE_FUTURE;

  iree_mutex_lock(&monitor->schedule_mutex);
  iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* immediate_list =
      iree_hal_amdgpu_ipc_event_monitor_schedule_list(
          monitor, IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_IMMEDIATE);
  iree_hal_amdgpu_ipc_event_monitor_operation_t* immediate_operation =
      immediate_list->head;
  iree_hal_amdgpu_ipc_event_monitor_operation_t* operation = NULL;
  if (immediate_operation &&
      *inout_immediate_service_count <
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_IMMEDIATE_SERVICE_QUOTA) {
    operation = immediate_operation;
    ++*inout_immediate_service_count;
    iree_hal_amdgpu_ipc_event_monitor_note_node_visit(monitor);
  } else {
    static const iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t
        kDeadlineKinds[] = {
            IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_100_US,
            IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_200_US,
            IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_400_US,
            IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_800_US,
            IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DELAY_1000_US,
        };
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kDeadlineKinds); ++i) {
      iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* list =
          iree_hal_amdgpu_ipc_event_monitor_schedule_list(monitor,
                                                          kDeadlineKinds[i]);
      iree_hal_amdgpu_ipc_event_monitor_operation_t* candidate = list->head;
      if (!candidate) continue;
      iree_hal_amdgpu_ipc_event_monitor_note_node_visit(monitor);
      if (candidate->next_poll_deadline_ns < *out_next_deadline_ns) {
        *out_next_deadline_ns = candidate->next_poll_deadline_ns;
        operation = candidate;
      }
    }
    if (*out_next_deadline_ns > now_ns) operation = NULL;

    if (!operation && immediate_operation) {
      // No finite operation is due. Start a fresh immediate quota so a
      // deadline that expires under continuous traffic is checked again after
      // a bounded number of polls.
      operation = immediate_operation;
      *inout_immediate_service_count = 1;
      iree_hal_amdgpu_ipc_event_monitor_note_node_visit(monitor);
    } else {
      // A due operation consumed this service opportunity, or the immediate
      // queue is empty. The next immediate burst begins with a fresh quota.
      *inout_immediate_service_count = 0;
    }
  }

  if (operation) {
    iree_hal_amdgpu_ipc_event_monitor_schedule_unlink_locked(monitor,
                                                             operation);
    const int32_t old_state = iree_atomic_exchange(
        &operation->state,
        IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_POLLING,
        iree_memory_order_acq_rel);
    IREE_ASSERT_EQ(old_state,
                   IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_SCHEDULED);
    *out_is_first_poll = !operation->has_been_polled;
    operation->has_been_polled = true;
    *out_was_requested = iree_atomic_exchange(&operation->poll_requested, 0,
                                              iree_memory_order_acq_rel) != 0;
    *out_next_deadline_ns = IREE_TIME_INFINITE_FUTURE;
  }
  iree_mutex_unlock(&monitor->schedule_mutex);
  return operation;
}

// Requeues an operation whose unlocked poll remained pending. A callback that
// races POLLING sets |poll_requested| under the same mutex and therefore wins
// over a dormant or finite-delay destination.
static void iree_hal_amdgpu_ipc_event_monitor_reschedule_pending(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor,
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool is_first_poll, bool was_requested) {
  const iree_time_t now_ns = iree_time_now();
  iree_mutex_lock(&monitor->schedule_mutex);
  IREE_ASSERT_EQ(iree_atomic_load(&operation->state, iree_memory_order_relaxed),
                 IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_POLLING);

  iree_hal_amdgpu_ipc_event_monitor_schedule_kind_t schedule_kind;
  if (iree_atomic_load(&operation->poll_requested, iree_memory_order_acquire) !=
      0) {
    schedule_kind = IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_IMMEDIATE;
  } else if (is_shutting_down) {
    schedule_kind =
        IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_CALLBACK_WAIT;
  } else if (operation->maximum_poll_delay_ns == IREE_DURATION_INFINITE) {
    operation->next_poll_deadline_ns = IREE_TIME_INFINITE_FUTURE;
    schedule_kind = IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_DORMANT;
  } else {
    if (is_first_poll || was_requested) {
      operation->current_poll_delay_ns =
          IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_INITIAL_POLL_DELAY_NS;
    } else {
      operation->current_poll_delay_ns =
          iree_hal_amdgpu_ipc_event_monitor_advance_delay(
              operation->current_poll_delay_ns);
    }
    operation->next_poll_deadline_ns =
        iree_hal_amdgpu_ipc_event_monitor_deadline_after(
            now_ns, operation->current_poll_delay_ns);
    schedule_kind = iree_hal_amdgpu_ipc_event_monitor_schedule_kind_for_delay(
        operation->current_poll_delay_ns);
    iree_hal_amdgpu_ipc_event_monitor_schedule_list_t* list =
        iree_hal_amdgpu_ipc_event_monitor_schedule_list(monitor, schedule_kind);
    IREE_ASSERT(!list->tail || list->tail->next_poll_deadline_ns <=
                                   operation->next_poll_deadline_ns);
  }

  iree_hal_amdgpu_ipc_event_monitor_schedule_append_locked(
      monitor, schedule_kind, operation);
  iree_atomic_store(&operation->state,
                    IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_OPERATION_STATE_SCHEDULED,
                    iree_memory_order_release);
  iree_mutex_unlock(&monitor->schedule_mutex);
}

static bool iree_hal_amdgpu_ipc_event_monitor_schedule_is_empty(
    iree_hal_amdgpu_ipc_event_monitor_t* monitor) {
  bool is_empty = true;
  iree_mutex_lock(&monitor->schedule_mutex);
  for (iree_host_size_t i = 1;
       i < IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_SCHEDULE_KIND_COUNT; ++i) {
    if (monitor->schedule_lists[i].head || monitor->schedule_lists[i].tail) {
      is_empty = false;
      break;
    }
  }
  iree_mutex_unlock(&monitor->schedule_mutex);
  return is_empty;
}

static int iree_hal_amdgpu_ipc_event_monitor_thread_main(void* user_data) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      (iree_hal_amdgpu_ipc_event_monitor_t*)user_data;
  bool shutdown_swept = false;
  iree_host_size_t immediate_service_count = 0;

  for (;;) {
    const iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&monitor->wake_notification);
    iree_hal_amdgpu_ipc_event_monitor_activate_published(monitor);

    const bool is_shutting_down =
        iree_atomic_load(&monitor->shutdown_requested,
                         iree_memory_order_acquire) != 0;
    if (is_shutting_down && !shutdown_swept) {
      iree_hal_amdgpu_ipc_event_monitor_sweep_for_shutdown(monitor);
      shutdown_swept = true;
    }

    bool is_first_poll = false;
    bool was_requested = false;
    iree_time_t next_poll_deadline_ns = IREE_TIME_INFINITE_FUTURE;
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation =
        iree_hal_amdgpu_ipc_event_monitor_take_ready(
            monitor, iree_time_now(), &immediate_service_count, &is_first_poll,
            &was_requested, &next_poll_deadline_ns);
    if (operation) {
      iree_notification_cancel_wait(&monitor->wake_notification);
      iree_hal_amdgpu_ipc_event_monitor_poll_fn_t poll = operation->poll;
      if (poll(operation, is_shutting_down, was_requested)) {
        // |poll| may have freed |operation|. Retire only monitor-owned state.
        iree_hal_amdgpu_ipc_event_monitor_retire_operation(monitor);
      } else {
        iree_hal_amdgpu_ipc_event_monitor_reschedule_pending(
            monitor, operation, is_shutting_down, is_first_poll, was_requested);
      }
      continue;
    }

    if (is_shutting_down &&
        iree_atomic_load(&monitor->outstanding_operation_count,
                         iree_memory_order_acquire) == 0) {
      const bool schedule_is_empty =
          iree_hal_amdgpu_ipc_event_monitor_schedule_is_empty(monitor);
      IREE_ASSERT(schedule_is_empty);
      (void)schedule_is_empty;
      IREE_ASSERT_EQ(
          iree_atomic_load(&monitor->pending_head, iree_memory_order_relaxed),
          0);
      iree_notification_cancel_wait(&monitor->wake_notification);
      break;
    }

    iree_notification_commit_wait(&monitor->wake_notification, wait_token,
                                  IREE_DURATION_ZERO, next_poll_deadline_ns);
  }

  return 0;
}

void iree_hal_amdgpu_ipc_event_monitor_shutdown(void) {
  iree_hal_amdgpu_ipc_event_monitor_t* monitor =
      iree_hal_amdgpu_ipc_event_monitor_get();
  iree_mutex_lock(&monitor->mutex);
  if (monitor->state == IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_STOPPED) {
    iree_mutex_unlock(&monitor->mutex);
    return;
  }
  if (monitor->state == IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_CLOSING) {
    const iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&monitor->state_changed);
    iree_mutex_unlock(&monitor->mutex);
    iree_notification_commit_wait(&monitor->state_changed, wait_token,
                                  IREE_DURATION_ZERO,
                                  IREE_TIME_INFINITE_FUTURE);
    return;
  }

  monitor->state = IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_CLOSING;
  iree_atomic_store(&monitor->shutdown_requested, 1, iree_memory_order_release);
  iree_thread_t* thread = monitor->thread;
  iree_mutex_unlock(&monitor->mutex);

  // Keep preparation cancellation independent from ordinary worker traffic.
  iree_notification_post(&monitor->shutdown_notification, IREE_ALL_WAITERS);
  iree_notification_post(&monitor->wake_notification, IREE_ALL_WAITERS);
  iree_thread_release(thread);

  iree_mutex_lock(&monitor->mutex);
  IREE_ASSERT(monitor->state ==
              IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_CLOSING);
  IREE_ASSERT_EQ(iree_atomic_load(&monitor->outstanding_operation_count,
                                  iree_memory_order_relaxed),
                 0);
  IREE_ASSERT_EQ(
      iree_atomic_load(&monitor->pending_head, iree_memory_order_relaxed), 0);
  const bool schedule_is_empty =
      iree_hal_amdgpu_ipc_event_monitor_schedule_is_empty(monitor);
  IREE_ASSERT(schedule_is_empty);
  (void)schedule_is_empty;
  monitor->thread = NULL;
  iree_atomic_store(&monitor->shutdown_requested, 0, iree_memory_order_relaxed);
  monitor->state = IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_STATE_STOPPED;
  // Publish completion while admission remains closed so this notification
  // cannot be mistaken for completion of a later worker generation.
  iree_notification_post(&monitor->state_changed, IREE_ALL_WAITERS);
  iree_mutex_unlock(&monitor->mutex);
}
