// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Enable GNU extensions for POLLRDHUP (peer half-close detection).
// Must be defined before any includes.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "iree/async/platform/io_uring/proactor.h"

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/futex.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/io_uring/buffer_ring.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/relay.h"
#include "iree/async/platform/io_uring/socket.h"
#include "iree/async/platform/io_uring/socket_completion.h"
#include "iree/async/semaphore.h"
#include "iree/async/types.h"
#include "iree/async/util/continuation.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/operation_completion.h"
#include "iree/async/util/semaphore_wait.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/memory.h"

// Maximum number of cascading CQE snapshots processed in one poll turn.
// Additional in-flight operations remain kernel-owned and are resumed by the
// caller's next poll turn.
#define IREE_ASYNC_IO_URING_CQE_DRAIN_PASS_BUDGET 32

// TSAN cannot observe the happens-before relationship provided by io_uring's
// shared memory rings (SQ/CQ). When a submitter thread fills an SQE and the
// poll thread later processes the corresponding CQE, the kernel's ring protocol
// provides ordering, but TSAN sees no userspace synchronization between the
// submitter's writes and the completer's reads.
//
// We bridge this gap with a C11 atomic on the operation struct
// (iree_async_operation_t::tsan_bridge). The submitter stores with release
// ordering after filling the operation; the completer loads with acquire
// ordering before reading operation fields. TSAN intercepts C11 atomics
// through compiler instrumentation (its core tracking mechanism).
//
// After the TSAN release on submit, the submitter must NOT read any operation
// fields — the operation is logically owned by the kernel and then the poll
// thread. Phase 3 (software op execution) uses a bitmap from Phase 1 analysis
// to skip kernel ops without re-reading their types.
#if defined(IREE_SANITIZER_THREAD)
#define IREE_IO_URING_TSAN_COMPLETE(operation) \
  iree_atomic_load(&(operation)->tsan_bridge, iree_memory_order_acquire)
#else
#define IREE_IO_URING_TSAN_COMPLETE(operation) ((void)0)
#endif  // IREE_SANITIZER_THREAD

// Wakes the proactor after another task queues a ring registration request.
static void iree_async_proactor_io_uring_wake_registration_owner(
    void* user_data) {
  iree_async_proactor_io_uring_t* proactor =
      (iree_async_proactor_io_uring_t*)user_data;
  iree_async_proactor_wake(&proactor->base);
}

iree_status_t iree_async_proactor_create_io_uring(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  // Message pool capacity from options, with default fallback.
  iree_host_size_t message_pool_capacity = options.message_pool_capacity;
  if (message_pool_capacity == 0) {
    message_pool_capacity = IREE_ASYNC_MESSAGE_POOL_DEFAULT_CAPACITY;
  }

  // Calculate allocation layout with trailing message pool entries.
  iree_host_size_t total_size = 0;
  iree_host_size_t message_entries_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(sizeof(iree_async_proactor_io_uring_t), &total_size,
                         IREE_STRUCT_FIELD(message_pool_capacity,
                                           iree_async_message_pool_entry_t,
                                           &message_entries_offset)));

  // Allocate the proactor structure with trailing data.
  iree_async_proactor_io_uring_t* proactor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&proactor));
  memset(proactor, 0, total_size);

  // Initialize base fields.
  iree_async_proactor_initialize(&iree_async_proactor_io_uring_vtable,
                                 options.debug_name, allocator,
                                 &proactor->base);

  // Initialize sequence emulator for SEQUENCE operation support.
  // Uses vtable-dispatched submit_one which routes individual step operations
  // through this backend's normal submit path.
  iree_async_sequence_emulator_initialize(&proactor->sequence_emulator,
                                          &proactor->base,
                                          iree_async_proactor_submit_one);

  // Initialize message pool with trailing data entries.
  iree_async_message_pool_entry_t* message_entries =
      (iree_async_message_pool_entry_t*)((uint8_t*)proactor +
                                         message_entries_offset);
  iree_async_message_pool_initialize(message_pool_capacity, message_entries,
                                     &proactor->message_pool);

  // Initialize our fields.
  proactor->wake_eventfd = -1;
  proactor->wake_poll_armed = false;
  iree_atomic_store(&proactor->polling.owner_tid, 0, iree_memory_order_relaxed);
  iree_atomic_store(&proactor->polling.dispatch_tid, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&proactor->next_buffer_group_id, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&proactor->legacy_buffer_table_state,
                    IREE_ASYNC_IO_URING_LEGACY_BUFFER_TABLE_STATE_FREE,
                    iree_memory_order_relaxed);
  proactor->capabilities = IREE_ASYNC_PROACTOR_CAPABILITY_NONE;
  iree_atomic_slist_initialize(&proactor->pending_software_operations);
  iree_atomic_slist_initialize(&proactor->pending_notifications);
  iree_atomic_slist_initialize(&proactor->pending_semaphore_waits);
  iree_async_semaphore_wait_context_initialize(
      &proactor->semaphore_wait_context);
  proactor->event_sources = NULL;
  proactor->relays = NULL;

  // Initialize signal handling state (lazy-initialized on first subscribe).
  proactor->signal.initialized = false;
  iree_async_linux_signal_state_initialize(&proactor->signal.linux_state);
  iree_async_signal_dispatch_state_initialize(&proactor->signal.dispatch_state);
  for (int i = 0; i < IREE_ASYNC_SIGNAL_COUNT; ++i) {
    proactor->signal.subscriptions[i] = NULL;
    proactor->signal.subscriber_count[i] = 0;
  }
  proactor->signal.event_source = NULL;

  // Translate proactor threading mode to ring threading mode.
  iree_io_uring_ring_options_t ring_options =
      iree_io_uring_ring_options_default();
  ring_options.threading_mode =
      (options.threading_mode == IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD)
          ? IREE_IO_URING_RING_THREADING_CROSS_THREAD
          : IREE_IO_URING_RING_THREADING_SAME_THREAD;
  if (options.max_concurrent_operations > 0) {
    ring_options.sq_entries = (uint32_t)options.max_concurrent_operations;
  }

  // Initialize the io_uring ring.
  iree_status_t status =
      iree_io_uring_ring_initialize(ring_options, &proactor->ring);

  // Detect capabilities from kernel feature flags and opcode probing.
  // This verifies we have a new enough kernel and determines what features
  // are available.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_io_uring_detect_capabilities(
        &proactor->ring, proactor->ring.features,
        &proactor->kernel_capabilities);
  }

  // Sparse fixed-buffer tables are an internal kernel mechanism, not a public
  // capability applications can disable.
  bool supports_sparse_buffer_table = iree_any_bit_set(
      proactor->kernel_capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT);

  // Probe registration itself: an opcode from the same kernel release is not
  // evidence that provided-buffer rings are usable under the current policy.
  // Keep the probe page alive through destruction if its retirement fails.
  void* buffer_ring_probe_page = NULL;
  if (iree_status_is_ok(status) && supports_sparse_buffer_table) {
    iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
    status = iree_allocator_malloc_aligned(allocator, page_size, page_size, 0,
                                           &buffer_ring_probe_page);
    if (iree_status_is_ok(status)) {
      memset(buffer_ring_probe_page, 0, page_size);
      status = iree_async_proactor_io_uring_probe_buffer_ring(
          &proactor->ring, buffer_ring_probe_page,
          &proactor->supports_buffer_ring);
    }
  }

  // Apply the allowed_capabilities mask from options.
  if (iree_status_is_ok(status)) {
    proactor->capabilities =
        proactor->kernel_capabilities & options.allowed_capabilities;
    if (!proactor->supports_buffer_ring) {
      proactor->capabilities &= ~IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT;
    }
  }

  // Create sparse buffer table on 5.19+ kernels for dynamic buffer
  // registration. This pre-allocates an empty table in the kernel so
  // individual slots can be populated later via IORING_REGISTER_BUFFERS_UPDATE.
  // MULTISHOT capability implies 5.19+ (the probe checks SOCKET opcode 45).
  if (iree_status_is_ok(status) && supports_sparse_buffer_table) {
    uint16_t table_capacity = IREE_IO_URING_SPARSE_TABLE_DEFAULT_CAPACITY;
    status = iree_io_uring_sparse_table_allocate(table_capacity, allocator,
                                                 &proactor->buffer_table);
    if (iree_status_is_ok(status)) {
      iree_io_uring_rsrc_register_t reg = {
          .nr = table_capacity,
          .flags = IREE_IORING_RSRC_REGISTER_SPARSE,
          .resv2 = 0,
          .data = 0,
          .tags = 0,
      };
      int register_result = iree_io_uring_ring_register(
          &proactor->ring, IREE_IORING_REGISTER_BUFFERS2, &reg, sizeof(reg));
      if (register_result < 0) {
        int error_number = -register_result;
        iree_io_uring_sparse_table_free(proactor->buffer_table, allocator);
        proactor->buffer_table = NULL;
        status = iree_make_status(
            iree_status_code_from_errno(error_number),
            "IORING_REGISTER_BUFFERS2 (sparse, capacity=%u) failed (%d)",
            (unsigned)table_capacity, error_number);
      }
    }
  }

  // Create the wake eventfd.
  if (iree_status_is_ok(status)) {
    proactor->wake_eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (proactor->wake_eventfd < 0) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "eventfd creation failed (%d)", errno);
    } else {
      iree_io_uring_registration_wake_callback_t wake_callback = {
          .fn = iree_async_proactor_io_uring_wake_registration_owner,
          .user_data = proactor,
      };
      iree_io_uring_ring_set_registration_wake_callback(&proactor->ring,
                                                        wake_callback);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_proactor = &proactor->base;
  } else {
    iree_async_proactor_io_uring_destroy(&proactor->base);
  }
  iree_allocator_free_aligned(allocator, buffer_ring_probe_page);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

static iree_async_proactor_capabilities_t
iree_async_proactor_io_uring_query_capabilities(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return proactor->capabilities;
}

//===----------------------------------------------------------------------===//
// Message delivery
//===----------------------------------------------------------------------===//

// Drains pending messages from the message pool and invokes the callback.
// Called from poll() after processing CQEs.
static void iree_async_proactor_io_uring_drain_pending_messages(
    iree_async_proactor_io_uring_t* proactor) {
  iree_async_message_pool_entry_t* entry =
      iree_async_message_pool_flush(&proactor->message_pool);
  while (entry) {
    iree_async_message_pool_entry_t* next =
        iree_async_message_pool_entry_next(entry);
    if (proactor->message_callback.fn) {
      proactor->message_callback.fn(&proactor->base, entry->message_data,
                                    proactor->message_callback.user_data);
    }
    iree_async_message_pool_release(&proactor->message_pool, entry);
    entry = next;
  }
}

//===----------------------------------------------------------------------===//
// LINKED chain dispatch helpers
//===----------------------------------------------------------------------===//

// Submits a continuation chain from an operation's linked_next pointer.
// Builds a temporary array from the linked list and submits as a batch.
// On submit failure, queues the failed head and cancelled tail through the
// poll-owned software completion path.
void iree_async_proactor_io_uring_submit_continuation_chain(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head) {
  if (!chain_head) {
    return;
  }

  // Count operations in the chain (must happen before submit, which rebuilds
  // linked_next from LINKED flags and destroys the incoming chain).
  iree_host_size_t chain_count = 0;
  for (iree_async_operation_t* op = chain_head; op; op = op->linked_next) {
    ++chain_count;
  }

  // Build array from linked_next chain. Use stack for the common case (chains
  // are typically 1-5 operations), heap-allocate for longer chains.
  iree_async_operation_t* stack_array[16];
  iree_async_operation_t** chain_array = stack_array;
  if (chain_count > IREE_ARRAYSIZE(stack_array)) {
    iree_status_t alloc_status = iree_allocator_malloc(
        proactor->base.allocator, chain_count * sizeof(iree_async_operation_t*),
        (void**)&chain_array);
    if (!iree_status_is_ok(alloc_status)) {
      // The failed head receives the allocation error and the unsubmitted tail
      // is cancelled. Callback-bearing operations use the poll-owned
      // completion path; suppressed tails are consumed before their storage
      // may be released.
      iree_async_operation_t* remaining_chain = chain_head->linked_next;
      chain_head->linked_next = NULL;
      if (chain_head->completion_fn) {
        iree_async_proactor_io_uring_push_software_operation(
            proactor, chain_head, alloc_status);
      } else {
        // A deliberately suppressed tail has no callback to order and may be
        // released as soon as its predecessor callback begins.
        iree_async_operation_complete(chain_head, alloc_status,
                                      IREE_ASYNC_COMPLETION_FLAG_NONE);
      }
      if (remaining_chain) {
        iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
            proactor, remaining_chain);
      }
      return;
    }
  }

  iree_host_size_t index = 0;
  for (iree_async_operation_t* op = chain_head; op; op = op->linked_next) {
    chain_array[index++] = op;
  }

  iree_async_operation_list_t chain_list = {chain_array, chain_count};
  iree_status_t submit_status =
      iree_async_proactor_io_uring_submit(&proactor->base, chain_list);
  if (!iree_status_is_ok(submit_status)) {
    // Submission may have rebuilt or consumed the intrusive links. Restore the
    // caller-owned chain before queueing the failed head and cancelled tail.
    for (iree_host_size_t i = 0; i < chain_count; ++i) {
      chain_array[i]->linked_next =
          i + 1 < chain_count ? chain_array[i + 1] : NULL;
    }
    iree_async_operation_t* failed_head = chain_array[0];
    iree_async_operation_t* remaining_chain = failed_head->linked_next;
    failed_head->linked_next = NULL;
    if (failed_head->completion_fn) {
      iree_async_proactor_io_uring_push_software_operation(
          proactor, failed_head, submit_status);
    } else {
      iree_async_operation_complete(failed_head, submit_status,
                                    IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    if (remaining_chain) {
      iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
          proactor, remaining_chain);
    }
  }

  if (chain_array != stack_array) {
    iree_allocator_free(proactor->base.allocator, chain_array);
  }
}

//===----------------------------------------------------------------------===//
// Poll-owned software operation dispatch
//===----------------------------------------------------------------------===//

// Pushes software work to the poll-owned MPSC queue. The status is stored in
// base.pending_status after any continuation chain has been consumed. SEQUENCE
// operations use an OK status to request startup instead of completion.
void iree_async_proactor_io_uring_push_software_operation(
    iree_async_proactor_io_uring_t* proactor, iree_async_operation_t* operation,
    iree_status_t status) {
  operation->pending_status = status;
  iree_atomic_slist_push(&proactor->pending_software_operations,
                         (iree_atomic_slist_entry_t*)operation);
}

// Drains pending software work and invokes or starts it.
// Called from poll() BEFORE CQE processing to preserve callback ordering for
// chains where a software operation precedes a kernel operation (e.g.,
// SIGNAL(LINKED) → RECV: SIGNAL callback must fire before RECV CQE callback).
//
// Returns the number of terminal completions drained (for inclusion in poll's
// completion count). Successfully started sequences do not count as completed.
static iree_host_size_t
iree_async_proactor_io_uring_drain_pending_software_operations(
    iree_async_proactor_io_uring_t* proactor) {
  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&proactor->pending_software_operations,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return 0;
  }

  iree_host_size_t drained_count = 0;
  iree_atomic_slist_entry_t* entry = head;
  while (entry != NULL) {
    iree_async_operation_t* operation = (iree_async_operation_t*)entry;
    iree_atomic_slist_entry_t* next = entry->next;
    operation->next = NULL;

    // Transfer ownership of the pending status out of the queue entry.
    iree_status_t status = operation->pending_status;
    operation->pending_status = iree_ok_status();

    if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
      iree_async_sequence_operation_t* sequence =
          (iree_async_sequence_operation_t*)operation;

      if (iree_status_is_ok(status)) {
        if (sequence->step_count == 0) {
          if (iree_any_bit_set(
                  iree_async_operation_load_internal_flags(operation),
                  IREE_ASYNC_SEQUENCE_INTERNAL_CANCEL_REQUESTED)) {
            status = iree_status_from_code(IREE_STATUS_CANCELLED);
          }
        } else if (!sequence->step_fn) {
          status =
              iree_async_sequence_submit_as_linked(&proactor->base, sequence);
          if (iree_status_is_ok(status)) {
            entry = next;
            continue;
          }
        } else {
          status = iree_async_sequence_emulation_begin(
              &proactor->sequence_emulator, sequence);
          if (iree_status_is_ok(status)) {
            entry = next;
            continue;
          }
        }
      }
      iree_async_sequence_prepare_for_completion(sequence);
    }

    drained_count += iree_async_operation_complete(
        operation, status, IREE_ASYNC_COMPLETION_FLAG_NONE);

    entry = next;
  }

  return drained_count;
}

//===----------------------------------------------------------------------===//
// Semaphore wait drain
//===----------------------------------------------------------------------===//

// Drains pending semaphore wait completions and invokes callbacks.
// Called from poll() after processing CQEs. Returns the number of completions
// drained (for inclusion in poll's completion count).
static iree_host_size_t
iree_async_proactor_io_uring_drain_pending_semaphore_waits(
    iree_async_proactor_io_uring_t* proactor) {
  // Flush all pending completions with approximate FIFO ordering.
  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&proactor->pending_semaphore_waits,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return 0;  // No completions pending.
  }

  // Process each completed semaphore wait.
  iree_host_size_t drained_count = 0;
  iree_atomic_slist_entry_t* entry = head;
  while (entry != NULL) {
    iree_async_semaphore_wait_tracker_t* tracker =
        (iree_async_semaphore_wait_tracker_t*)entry;
    iree_atomic_slist_entry_t* next = entry->next;

    if (!iree_async_semaphore_wait_tracker_try_prepare_completion(tracker)) {
      entry = next;
      continue;
    }

    iree_async_semaphore_wait_completion_t completion;
    iree_async_semaphore_wait_tracker_finalize(tracker, &completion);
    iree_async_semaphore_wait_operation_t* wait_op = completion.operation;
    iree_async_operation_t* continuation = completion.continuation_head;
    iree_status_t status = completion.status;
    iree_status_code_t status_code = iree_status_code(status);

    // Consume continuation fields before the trigger callback. Dispatch only
    // queues work, so successor callbacks remain ordered after the trigger.
    if (continuation) {
      if (status_code == IREE_STATUS_OK) {
        iree_async_proactor_io_uring_dispatch_continuation_chain(proactor,
                                                                 continuation);
      } else {
        iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
            proactor, continuation);
      }
    }

    drained_count +=
        iree_async_operation_complete((iree_async_operation_t*)wait_op, status,
                                      IREE_ASYNC_COMPLETION_FLAG_NONE);

    entry = next;
  }

  return drained_count;
}

//===----------------------------------------------------------------------===//
// Wake mechanism
//===----------------------------------------------------------------------===//

// Arms a POLL_ADD on the wake eventfd. This allows wake() from another thread
// to interrupt a blocking poll().
static iree_status_t iree_async_proactor_io_uring_arm_wake(
    iree_async_proactor_io_uring_t* proactor) {
  if (proactor->wake_poll_armed) {
    return iree_ok_status();
  }
  if (proactor->wake_eventfd < 0) {
    return iree_ok_status();
  }

  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full when arming wake");
  }

  sqe->opcode = IREE_IORING_OP_POLL_ADD;
  sqe->fd = proactor->wake_eventfd;
  sqe->poll32_events = POLLIN;
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_WAKE, 0);
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  proactor->wake_poll_armed = true;
  return iree_ok_status();
}

// Handles completion of the wake POLL_ADD and drains the eventfd. poll() arms
// a fresh wait before it can block again; any intervening wake remains recorded
// in the eventfd counter until then.
static void iree_async_proactor_io_uring_handle_wake_completion(
    iree_async_proactor_io_uring_t* proactor) {
  proactor->wake_poll_armed = false;

  // Drain the eventfd (read returns the count of wake() calls).
  uint64_t value;
  ssize_t result = 0;
  do {
    result = read(proactor->wake_eventfd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  IREE_ASSERT(result == sizeof(value),
              "failed to drain io_uring wake eventfd: %zd (errno=%d)", result,
              errno);

  // Registration is a cold control path. Only wake completions pay this
  // pending-bit check; ordinary submission and io_uring_enter remain
  // unchanged.
  iree_io_uring_ring_drain_registration_requests(&proactor->ring);
}

//===----------------------------------------------------------------------===//
// Fence import completion
//===----------------------------------------------------------------------===//

// Handles completion of a fence import POLL_ADD.
// On success (POLLIN set), signals the semaphore. On error, fails it.
// Always closes the fd, releases the semaphore, and frees the tracker.
static void iree_async_proactor_io_uring_handle_fence_import_completion(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  (void)proactor;

  // Extract the tracker pointer from user_data payload.
  iree_async_io_uring_fence_import_tracker_t* tracker =
      (iree_async_io_uring_fence_import_tracker_t*)(uintptr_t)
          iree_io_uring_internal_payload(cqe->user_data);

  if (cqe->res < 0) {
    // Kernel error (EBADF, ECANCELED, etc.). Fail the semaphore.
    iree_async_semaphore_fail(
        tracker->semaphore,
        iree_make_status(iree_status_code_from_errno(-cqe->res),
                         "fence import POLL_ADD failed (%d)", -cqe->res));
  } else if (!iree_any_bit_set((uint32_t)cqe->res, POLLIN)) {
    // No POLLIN: pure POLLHUP or POLLERR (fd error condition).
    iree_async_semaphore_fail(
        tracker->semaphore,
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "fence fd signaled error (poll events=0x%04x)",
                         cqe->res));
  } else {
    // POLLIN set: fd became readable, fence signaled.
    iree_status_t signal_status = iree_async_semaphore_signal(
        tracker->semaphore, tracker->signal_value, /*frontier=*/NULL);
    if (!iree_status_is_ok(signal_status)) {
      // Signal failed (e.g., non-monotonic value). Fail the semaphore.
      iree_async_semaphore_fail(tracker->semaphore, signal_status);
    }
  }

  // Cleanup: close fd, release semaphore, free tracker.
  close(tracker->fence_fd);
  iree_async_semaphore_release(tracker->semaphore);
  iree_allocator_free(tracker->allocator, tracker);
}

//===----------------------------------------------------------------------===//
// Fence export timepoint callback
//===----------------------------------------------------------------------===//

// Timepoint callback that signals the exported eventfd when the semaphore
// reaches the target value. On success, writes to the eventfd to make it
// readable. On failure (semaphore failed or cancelled), leaves the fd
// unreadable. Always releases the semaphore and frees the tracker.
//
// write() to a non-blocking eventfd is ~100-200ns and never blocks, which is
// within the "must be fast" callback contract.
static void iree_async_io_uring_fence_export_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_async_io_uring_fence_export_tracker_t* tracker =
      (iree_async_io_uring_fence_export_tracker_t*)user_data;

  if (iree_status_is_ok(status)) {
    // Semaphore reached the target value. Write to eventfd to make it readable.
    uint64_t value = 1;
    ssize_t result = write(tracker->eventfd, &value, sizeof(value));
    // EAGAIN means counter is already nonzero (redundant signal, benign).
    // Any other failure (EBADF) indicates a lifecycle bug — the eventfd was
    // closed while a timepoint callback was still pending.
    IREE_ASSERT(result >= 0 || errno == EAGAIN);
  } else {
    // Semaphore failed or was cancelled. Leave fd unreadable so consumers
    // observe a timeout or check the semaphore for failure status.
    iree_status_ignore(status);
  }

  // Cleanup: release semaphore, free tracker. The eventfd is caller-owned and
  // must not be closed here.
  iree_async_semaphore_release(tracker->semaphore);
  iree_allocator_free(tracker->allocator, tracker);
}

//===----------------------------------------------------------------------===//
// CQE Processing Helpers
//===----------------------------------------------------------------------===//

// Handles MESSAGE_RECEIVE internal CQE.
// Incoming message from another proactor via MSG_RING.
static inline void iree_async_proactor_io_uring_handle_message_receive_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  // The 64-bit message_data was split across two CQE fields:
  //   - Lower 32 bits: sqe->len → cqe->res
  //   - Upper 32 bits: encoded in sqe->off → cqe->user_data payload
  uint64_t upper_32 = iree_io_uring_internal_payload(cqe->user_data);
  uint64_t message_data = (upper_32 << 32) | (uint32_t)cqe->res;
  if (proactor->message_callback.fn) {
    proactor->message_callback.fn(&proactor->base, message_data,
                                  proactor->message_callback.user_data);
  }
}

// Callback for signalfd dispatch - invoked for each signal read.
static void iree_async_proactor_io_uring_signal_dispatch_callback(
    void* user_data, iree_async_signal_t signal) {
  iree_async_proactor_io_uring_t* proactor =
      (iree_async_proactor_io_uring_t*)user_data;

  // Dispatch to all subscribers for this signal.
  iree_async_signal_subscription_t* head =
      proactor->signal.subscriptions[signal];
  iree_async_signal_subscription_t* to_free =
      iree_async_signal_subscription_dispatch(
          head, &proactor->signal.dispatch_state, signal);

  // Free any subscriptions that were unsubscribed during dispatch.
  while (to_free) {
    iree_async_signal_subscription_t* subscription = to_free;
    iree_async_signal_t signal = subscription->signal;
    to_free = to_free->pending_next;
    iree_async_signal_subscription_unlink(
        &proactor->signal.subscriptions[signal], subscription);
    iree_allocator_free(proactor->base.allocator, subscription);

    // Decrement count and remove signal from mask if last subscriber.
    if (--proactor->signal.subscriber_count[signal] == 0) {
      iree_async_linux_signal_remove_signal(&proactor->signal.linux_state,
                                            signal);
    }
  }
}

// Handles SIGNAL internal CQE.
// Multishot poll completion for the signalfd.
static void iree_async_proactor_io_uring_handle_signal_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  // Only process if signal handling is still active.
  if (!proactor->signal.initialized) {
    return;
  }

  // Read and dispatch all pending signals from signalfd.
  if (cqe->res >= 0 && (cqe->res & POLLIN)) {
    IREE_CHECK_OK(iree_async_linux_signal_read_signalfd(
        &proactor->signal.linux_state,
        (iree_async_signal_dispatch_callback_t){
            .fn = iree_async_proactor_io_uring_signal_dispatch_callback,
            .user_data = proactor,
        }));
  }

  // Multishot poll remains armed until cancelled. CQE_F_MORE indicates more
  // CQEs will follow. If MORE is clear, the poll was cancelled (proactor
  // teardown). No cleanup needed here - destroy() handles it.
}

// Converts a CQE result to an iree_status_t.
// Handles special cases like timer expiration and futex value mismatch.
static iree_status_t iree_async_proactor_io_uring_cqe_to_status(
    const iree_io_uring_cqe_t* cqe, iree_async_operation_t* operation) {
  if (cqe->res >= 0) {
    return iree_ok_status();
  }

  // Timer: -ETIME is the expected success case (timeout expired).
  if (cqe->res == -ETIME &&
      operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER) {
    return iree_ok_status();
  }

  // FUTEX_WAIT: -EAGAIN means value mismatch (already changed), not error.
  if (cqe->res == -EAGAIN &&
      operation->type == IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT) {
    return iree_ok_status();
  }

  return iree_make_status(iree_status_code_from_errno(-cqe->res),
                          "io_uring operation type %d failed (%d)",
                          (int)operation->type, -cqe->res);
}

// Handles SOCKET_ACCEPT completion: imports the accepted fd as a socket.
static iree_status_t iree_async_proactor_io_uring_complete_socket_accept(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_async_socket_accept_operation_t* accept) {
  int accepted_fd = cqe->res;

  // Propagate ZERO_COPY flag from listener.
  // SO_ZEROCOPY is NOT inherited on accept(), so we must set it
  // explicitly on each accepted socket if the listener has it enabled.
  iree_async_socket_flags_t inherited_flags = IREE_ASYNC_SOCKET_FLAG_NONE;
  bool listener_wants_zc = iree_any_bit_set(accept->listen_socket->flags,
                                            IREE_ASYNC_SOCKET_FLAG_ZERO_COPY);
#if defined(SO_ZEROCOPY)
  if (listener_wants_zc) {
    int optval = 1;
    if (setsockopt(accepted_fd, SOL_SOCKET, SO_ZEROCOPY, &optval,
                   sizeof(optval)) == 0) {
      inherited_flags |= IREE_ASYNC_SOCKET_FLAG_ZERO_COPY;
    }
  }
#else
  (void)listener_wants_zc;
#endif  // SO_ZEROCOPY

  iree_async_primitive_t accepted_primitive =
      iree_async_primitive_from_fd(accepted_fd);
  iree_status_t status = iree_async_io_uring_socket_import(
      proactor, accepted_primitive, accept->listen_socket->type,
      inherited_flags, &accept->accepted_socket);
  if (iree_status_is_ok(status)) {
    iree_atomic_store(&accept->accepted_socket->bind_state,
                      IREE_ASYNC_SOCKET_BIND_STATE_BOUND,
                      iree_memory_order_release);
    accept->accepted_socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
  } else {
    // Import failed (e.g., allocation failure). Close the accepted fd.
    close(accepted_fd);
  }
  return status;
}

// Handles SOCKET_RECV_POOL completion: builds the buffer lease from CQE.
static inline void iree_async_proactor_io_uring_complete_socket_recv_pool(
    const iree_io_uring_cqe_t* cqe,
    iree_async_socket_recv_pool_operation_t* recv_pool) {
  if (cqe->res >= 0 && (cqe->flags & IREE_IORING_CQE_F_BUFFER)) {
    uint16_t buffer_index =
        (uint16_t)(cqe->flags >> IREE_IORING_CQE_BUFFER_SHIFT);

    iree_async_region_t* region =
        iree_async_buffer_pool_region(recv_pool->pool);
    IREE_ASSERT(buffer_index < region->buffer_count);
    recv_pool->lease.span = iree_async_span_make(
        region, (iree_host_size_t)buffer_index * region->buffer_size,
        region->buffer_size);
    recv_pool->lease.release = region->recycle;
    recv_pool->lease.buffer_index = (iree_async_buffer_index_t)buffer_index;
  }
  // Without buffer selection the lease was reserved before submission.
  recv_pool->bytes_received = (iree_host_size_t)cqe->res;
}

// Handles SOCKET_RECVFROM completion: extracts sender address.
static inline void iree_async_proactor_io_uring_complete_socket_recvfrom(
    const iree_io_uring_cqe_t* cqe,
    iree_async_socket_recvfrom_operation_t* recvfrom) {
  recvfrom->bytes_received = (iree_host_size_t)cqe->res;
  struct msghdr* msg = (struct msghdr*)recvfrom->platform.posix.msg_header;
  recvfrom->sender.length = (iree_host_size_t)msg->msg_namelen;
}

// Handles FILE_OPEN completion: imports the opened fd as a file handle.
static iree_status_t iree_async_proactor_io_uring_complete_file_open(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_async_file_open_operation_t* open_op) {
  int fd = cqe->res;
  iree_async_primitive_t primitive = iree_async_primitive_from_fd(fd);
  iree_status_t status =
      iree_async_file_import(&proactor->base, primitive, &open_op->opened_file);
  if (!iree_status_is_ok(status)) {
    // Import failed (e.g., allocation failure). Close the opened fd.
    close(fd);
  }
  return status;
}

// Populates operation result fields from a successful CQE.
// Returns updated status (may fail for SOCKET_ACCEPT or FILE_OPEN if import
// fails).
static iree_status_t iree_async_proactor_io_uring_populate_result(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT: {
      iree_async_socket_connect_operation_t* connect =
          (iree_async_socket_connect_operation_t*)operation;
      iree_atomic_store(&connect->socket->bind_state,
                        IREE_ASYNC_SOCKET_BIND_STATE_BOUND,
                        iree_memory_order_release);
      connect->socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return iree_async_proactor_io_uring_complete_socket_accept(
          proactor, cqe, (iree_async_socket_accept_operation_t*)operation);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv =
          (iree_async_socket_recv_operation_t*)operation;
      recv->bytes_received = (iree_host_size_t)cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      iree_async_proactor_io_uring_complete_socket_recv_pool(
          cqe, (iree_async_socket_recv_pool_operation_t*)operation);
      break;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      iree_async_proactor_io_uring_complete_socket_recvfrom(
          cqe, (iree_async_socket_recvfrom_operation_t*)operation);
      break;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE: {
      iree_async_socket_close_operation_t* close_op =
          (iree_async_socket_close_operation_t*)operation;
      close_op->socket->state = IREE_ASYNC_SOCKET_STATE_CLOSED;
      close_op->socket->primitive.value.fd = -1;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE: {
      iree_async_futex_wake_operation_t* futex_wake =
          (iree_async_futex_wake_operation_t*)operation;
      futex_wake->woken_count = cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
      return iree_async_proactor_io_uring_complete_file_open(
          proactor, cqe, (iree_async_file_open_operation_t*)operation);
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      iree_async_file_read_operation_t* read_op =
          (iree_async_file_read_operation_t*)operation;
      read_op->bytes_read = (iree_host_size_t)cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_file_write_operation_t* write_op =
          (iree_async_file_write_operation_t*)operation;
      write_op->bytes_written = (iree_host_size_t)cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE: {
      iree_async_file_close_operation_t* close_op =
          (iree_async_file_close_operation_t*)operation;
      close_op->file->primitive.value.fd = -1;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL: {
      iree_async_handle_poll_operation_t* handle_poll =
          (iree_async_handle_poll_operation_t*)operation;
      // cqe->res for POLL_ADD contains the Linux poll event mask.
      // Translate to portable iree_async_poll_events_t.
      int revents = cqe->res;
      iree_async_poll_events_t events = IREE_ASYNC_POLL_EVENT_NONE;
      if (revents & POLLIN) {
        events |= IREE_ASYNC_POLL_EVENT_IN;
      }
      if (revents & POLLOUT) {
        events |= IREE_ASYNC_POLL_EVENT_OUT;
      }
      if (revents & POLLERR) {
        events |= IREE_ASYNC_POLL_EVENT_ERR;
      }
      if (revents & POLLHUP) {
        events |= IREE_ASYNC_POLL_EVENT_HUP;
      }
      handle_poll->result_events = events;
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

// Computes completion flags from CQE and operation type.
static inline iree_async_completion_flags_t
iree_async_proactor_io_uring_completion_flags(
    const iree_io_uring_cqe_t* cqe, iree_async_operation_t* operation) {
  iree_async_completion_flags_t flags = IREE_ASYNC_COMPLETION_FLAG_NONE;

  if (iree_any_bit_set(cqe->flags, IREE_IORING_CQE_F_MORE)) {
    flags |= IREE_ASYNC_COMPLETION_FLAG_MORE;
  }

  return flags;
}

// Handles internal CQEs (wake poll, cancel, fence import, message, event).
// Owned cancellation receipts count as user completions and report native
// errors through the poll result instead of the target callback.
static inline iree_host_size_t
iree_async_proactor_io_uring_process_internal_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_status_t* inout_poll_status) {
  switch (iree_io_uring_internal_tag(cqe->user_data)) {
    case IREE_IO_URING_TAG_WAKE:
      iree_async_proactor_io_uring_handle_wake_completion(proactor);
      break;
    case IREE_IO_URING_TAG_CANCEL: {
      iree_host_size_t completed_count = 0;
      *inout_poll_status =
          iree_status_join(*inout_poll_status,
                           iree_async_proactor_io_uring_complete_cancel_request(
                               cqe, &completed_count));
      return completed_count;
    }
    case IREE_IO_URING_TAG_FENCE_IMPORT:
      iree_async_proactor_io_uring_handle_fence_import_completion(proactor,
                                                                  cqe);
      break;
    case IREE_IO_URING_TAG_MESSAGE_RECEIVE:
      iree_async_proactor_io_uring_handle_message_receive_cqe(proactor, cqe);
      break;
    case IREE_IO_URING_TAG_MESSAGE_SOURCE:
      // Source-side completion of a message send where the user requested
      // SKIP_SOURCE_COMPLETION (fire-and-forget semantics).
      //
      // We get CQEs here when:
      // - FEAT_CQE_SKIP is not available (kernel always generates source CQE)
      // - The operation failed (error CQEs bypass CQE_SKIP)
      //
      // Fire-and-forget means the user explicitly chose to not be notified
      // of either success or failure, so we discard the result.
      break;
    case IREE_IO_URING_TAG_EVENT_SOURCE:
      iree_async_io_uring_event_source_dispatch(proactor, cqe);
      break;
    case IREE_IO_URING_TAG_EVENT_SOURCE_CANCEL:
      *inout_poll_status = iree_status_join(
          *inout_poll_status,
          iree_async_io_uring_event_source_complete_cancel(proactor, cqe));
      break;
    case IREE_IO_URING_TAG_RELAY: {
      // Extract the relay pointer from the payload.
      iree_async_relay_t* relay =
          (iree_async_relay_t*)(uintptr_t)iree_io_uring_internal_payload(
              cqe->user_data);
      iree_async_io_uring_handle_relay_cqe(proactor, relay, cqe->res,
                                           cqe->flags);
      break;
    }
    case IREE_IO_URING_TAG_RELAY_CANCEL: {
      iree_async_relay_t* relay =
          (iree_async_relay_t*)(uintptr_t)iree_io_uring_internal_payload(
              cqe->user_data);
      *inout_poll_status = iree_status_join(
          *inout_poll_status,
          iree_async_io_uring_relay_complete_cancel(proactor, relay, cqe->res));
      break;
    }
    case IREE_IO_URING_TAG_SIGNAL:
      iree_async_proactor_io_uring_handle_signal_cqe(proactor, cqe);
      break;
    default:
      IREE_ASSERT(false,
                  "unrecognized internal tag %d in CQE (user_data=0x%" PRIx64
                  ", res=%d, flags=0x%x)",
                  (int)iree_io_uring_internal_tag(cqe->user_data),
                  cqe->user_data, cqe->res, cqe->flags);
      break;
  }
  return 0;
}

// Returns the socket for a socket I/O operation that should propagate sticky
// failure on error. Returns NULL for non-socket operations and for types where
// errors don't indicate a broken socket (accept errors reflect transient
// resource issues, not a broken listener; close errors are moot).
static iree_async_socket_t*
iree_async_proactor_io_uring_socket_from_io_operation(
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return ((iree_async_socket_connect_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      return ((iree_async_socket_recv_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      return ((iree_async_socket_send_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return ((iree_async_socket_recv_pool_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      return ((iree_async_socket_recvfrom_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return ((iree_async_socket_sendto_operation_t*)operation)->socket;
    default:
      return NULL;
  }
}

//===----------------------------------------------------------------------===//
// Poll
//===----------------------------------------------------------------------===//

// Processes a single CQE, invoking the operation's callback.
// Returns the number of user completions (typically 1, but can be more if
// continuation callbacks were invoked directly; 0 if suppressed, e.g., first
// CQE of a zero-copy send waiting for NOTIF).
iree_host_size_t iree_async_proactor_io_uring_process_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_status_t* inout_poll_status) {
  // Linked POLL_ADD head CQE for EVENT_WAIT.
  // Always ignored — the linked READ CQE handles resource release and user
  // callback dispatch for both success (READ returns data) and failure
  // (READ gets -ECANCELED when POLL_ADD fails).
  if (iree_io_uring_is_internal_cqe(cqe) &&
      iree_io_uring_internal_tag(cqe->user_data) ==
          IREE_IO_URING_TAG_LINKED_POLL) {
    return 0;
  }

  // Handle internal operations (wake poll, cancel, fence import, message).
  if (iree_io_uring_is_internal_cqe(cqe)) {
    return iree_async_proactor_io_uring_process_internal_cqe(proactor, cqe,
                                                             inout_poll_status);
  }

  // User operation: extract the operation pointer.
  iree_async_operation_t* operation =
      (iree_async_operation_t*)(uintptr_t)cqe->user_data;

  // Acquire pairs with the release store to tsan_bridge in submit_one.
  // Makes the submitter's writes to operation fields visible to this thread.
  IREE_IO_URING_TSAN_COMPLETE(operation);

  iree_status_t status = iree_ok_status();
  iree_async_completion_flags_t flags = IREE_ASYNC_COMPLETION_FLAG_NONE;
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND) {
    iree_async_io_uring_socket_send_completion_t completion =
        iree_async_io_uring_socket_process_send_cqe(
            cqe, (iree_async_socket_send_operation_t*)operation);
    if (!completion.dispatch) {
      return 0;
    }
    status = completion.status;
    flags = completion.flags;
  } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO) {
    iree_async_io_uring_socket_send_completion_t completion =
        iree_async_io_uring_socket_process_sendto_cqe(
            cqe, (iree_async_socket_sendto_operation_t*)operation);
    if (!completion.dispatch) {
      return 0;
    }
    status = completion.status;
    flags = completion.flags;
  } else {
    status = iree_async_proactor_io_uring_cqe_to_status(cqe, operation);
    if (iree_status_is_ok(status)) {
      status = iree_async_proactor_io_uring_populate_result(proactor, cqe,
                                                            operation);
    } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL) {
      iree_async_socket_recv_pool_operation_t* receive =
          (iree_async_socket_recv_pool_operation_t*)operation;
      if (iree_async_buffer_pool_region(receive->pool)
              ->handles.iouring.buffer_group_id < 0) {
        // The kernel has retired this one-shot receive, including cancellation.
        // Only now may another operation acquire its destination buffer.
        iree_async_buffer_lease_release(&receive->lease);
      }
    }
    flags = iree_async_proactor_io_uring_completion_flags(cqe, operation);
  }

  // Propagate error to socket's sticky failure status.
  if (!iree_status_is_ok(status)) {
    iree_async_socket_t* socket =
        iree_async_proactor_io_uring_socket_from_io_operation(operation);
    if (socket) {
      iree_async_socket_set_failure(socket, iree_status_code(status));
    }
  }

  const bool is_final =
      !iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE);

  // Detach the LINKED continuation before the final callback can release the
  // trigger. Intermediate multishot completions leave the continuation owned
  // by the in-flight operation.
  iree_async_operation_t* continuation =
      is_final ? iree_async_continuation_take(operation) : NULL;

  iree_status_code_t status_code = iree_status_code(status);

  // Consume successor fields before the trigger callback. Software completions
  // and failures are queued to the MPSC path, while kernel operations produce
  // later CQEs, so user callbacks remain trigger-first.
  if (continuation) {
    if (status_code == IREE_STATUS_OK) {
      iree_async_proactor_io_uring_dispatch_continuation_chain(proactor,
                                                               continuation);
    } else {
      iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
          proactor, continuation);
    }
  }

  bool is_notification_monitor =
      operation->type == IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL &&
      iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                       IREE_ASYNC_IO_URING_NOTIFICATION_OPERATION_MONITOR);
  iree_host_size_t completed_count =
      iree_async_operation_complete(operation, status, flags);

  return is_notification_monitor ? 0 : completed_count;
}

// Submits poll-owned event source and relay operations in SQ-sized batches.
// Registration only creates logical handles; this is the boundary that makes
// them kernel-visible while preserving SINGLE_ISSUER ownership.
iree_status_t iree_async_proactor_io_uring_submit_pending_event_monitors(
    iree_async_proactor_io_uring_t* proactor) {
  bool has_pending = false;
  do {
    bool has_pending_event_sources =
        iree_async_io_uring_event_source_submit_pending(proactor);
    bool has_pending_relays =
        iree_async_io_uring_retry_pending_relays(proactor);
    has_pending = has_pending_event_sources || has_pending_relays;

    // This also flushes SQEs queued by ordinary operations before the first
    // poll. Submitting each full batch advances the SQ head so registration is
    // not limited by the ring's instantaneous capacity.
    IREE_RETURN_IF_ERROR(
        iree_io_uring_ring_submit(&proactor->ring,
                                  /*min_complete=*/0,
                                  /*flags=*/IREE_IORING_ENTER_GETEVENTS));
  } while (has_pending);
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_poll(
    iree_async_proactor_t* base_proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  if (out_completed_count) {
    *out_completed_count = 0;
  }

  // Enable the ring on first poll. When created with R_DISABLED (which happens
  // automatically when SINGLE_ISSUER is in the setup flags), this call binds
  // the current thread as the exclusive submitter for all io_uring_enter calls.
  IREE_RETURN_IF_ERROR(iree_io_uring_ring_enable(&proactor->ring));

  int32_t owner_tid = (int32_t)syscall(__NR_gettid);
  iree_atomic_store(&proactor->polling.owner_tid, owner_tid,
                    iree_memory_order_relaxed);

  iree_convert_timeout_to_absolute(&timeout);
  bool is_immediate = iree_timeout_is_immediate(timeout);

  // Accepted notification consumers share one poll-owned native monitor.
  // Any staged SQEs are flushed with the registrations below.
  iree_async_io_uring_notification_drain_pending(proactor);

  // Submit registrations and any SQEs queued before the first poll. This must
  // happen before arming the wake source: a full pre-poll SQ must not prevent
  // the poll owner from establishing its own wake path.
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_io_uring_submit_pending_event_monitors(proactor));

  // Arm the wake poll before potentially blocking.
  IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_arm_wake(proactor));

  // Flush the wake SQE and deferred completions. With DEFER_TASKRUN, the kernel
  // defers CQE generation for async notifications until we explicitly request
  // processing via GETEVENTS. This flush must happen even when CQEs are already
  // available because those completions may have triggered deferred task work.
  IREE_RETURN_IF_ERROR(
      iree_io_uring_ring_submit(&proactor->ring,
                                /*min_complete=*/0,
                                /*flags=*/IREE_IORING_ENTER_GETEVENTS));

  // Run poll-owner work before waiting for native completions.
  iree_host_size_t progress_count = 0;
  iree_status_t progress_status =
      iree_async_proactor_run_progress(base_proactor, &progress_count);
  if (!iree_status_is_ok(progress_status)) {
    if (out_completed_count) {
      *out_completed_count = progress_count;
    }
    return progress_status;
  }
  if (progress_count > 0 || base_proactor->progress_list) {
    is_immediate = true;
  }

  // If no CQEs are available after flushing and timeout allows blocking,
  // wait for completions.
  if (!iree_io_uring_ring_cq_ready(&proactor->ring) && !is_immediate &&
      !base_proactor->cancellations.list.head) {
    iree_duration_t timeout_ns = iree_timeout_as_duration_ns(timeout);
    iree_status_t wait_status =
        iree_io_uring_ring_wait_cqe(&proactor->ring, /*min_complete=*/1,
                                    /*flush_pending=*/true, timeout_ns);
    if (iree_status_is_deadline_exceeded(wait_status)) {
      iree_status_free(wait_status);
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    IREE_RETURN_IF_ERROR(wait_status);
  }

  // Mark the poll thread as active. Submit paths check this to decide whether
  // to flush SQEs directly (poll thread) or defer to wake (cross-thread).
  // Set before the first MPSC drain because drain callbacks may submit ops.
  iree_atomic_store(&proactor->polling.dispatch_tid, owner_tid,
                    iree_memory_order_relaxed);

  // First MPSC drain: software work from submit threads. This starts admitted
  // sequences and delivers terminal completions whose side effects are already
  // done. Draining these first preserves callback ordering for chains like
  // SIGNAL(LINKED) → RECV: the SIGNAL callback fires before the RECV CQE
  // callback. A second drain occurs after CQE processing and semaphore wait
  // dispatch to catch software work pushed during those phases.
  iree_host_size_t completed =
      progress_count +
      iree_async_proactor_io_uring_drain_pending_software_operations(proactor);
  iree_status_t status = iree_ok_status();

  // Process available CQEs using a CQ tail snapshot. The snapshot bounds the
  // loop to CQEs that existed before processing started. CQEs generated by
  // SQE flushes during CQE callback processing (Phase 4 of submit) are past
  // the snapshot and handled by the drain loop below.
  //
  // SQEs submitted during CQE callback processing (via submit's Phase 4)
  // are flushed immediately for latency — inline sends complete during
  // io_uring_enter. The CQ tail snapshot prevents processing CQEs from
  // those flushes in the same iteration (which would create infinite
  // re-submission loops for operations that complete inline). Those CQEs
  // are handled by the drain loop below.
  {
    uint32_t cq_tail_snapshot =
        iree_atomic_load((iree_atomic_int32_t*)proactor->ring.cq_tail,
                         iree_memory_order_acquire);
    while (*proactor->ring.cq_head != cq_tail_snapshot) {
      iree_io_uring_cqe_t* cqe =
          &proactor->ring
               .cqes[*proactor->ring.cq_head & proactor->ring.cq_mask];
      completed +=
          iree_async_proactor_io_uring_process_cqe(proactor, cqe, &status);
      iree_io_uring_ring_cq_advance(&proactor->ring, 1);
    }
  }

  // Second MPSC drain: software work pushed during the main CQE loop. CQE
  // processing dispatches continuation chains that push software work to the
  // MPSC. Terminal completions must fire BEFORE the drain loop processes CQEs
  // from Phase 4 flushes, because those CQEs correspond to operations later in
  // the chain. Example: [RECV → SIGNAL → SEND] — SIGNAL's callback must
  // fire before SEND's CQE is processed.
  iree_async_io_uring_notification_drain_pending(proactor);
  completed +=
      iree_async_proactor_io_uring_drain_pending_software_operations(proactor);

  // Drain loop: process CQEs generated by Phase 4 flushes and run DEFER_TASKRUN
  // task_work. Each pass calls ring_submit with GETEVENTS to flush any
  // remaining SQEs and run deferred task_work (e.g., multishot recv triggered
  // by a send on the same ring). The resulting CQEs are processed with a fresh
  // snapshot to prevent cascading loops.
  //
  // GETEVENTS is required because DEFER_TASKRUN defers async completions as
  // task_work. Without GETEVENTS, io_uring_enter only processes inline
  // completions and a poll loop could sleep despite runnable task work.
  //
  // The pass budget is a fairness boundary, not a completion join. Every
  // processed CQE either terminally returns its operation to the caller or
  // leaves it retained as in flight. If callbacks keep generating new work
  // after the budget is exhausted, the current poll returns OK and the
  // caller's next poll turn observes the ready CQE or blocks on the submitted
  // operation. No operation lifetime depends on the number of drain passes.
  for (int drain_pass = 0;
       drain_pass < IREE_ASYNC_IO_URING_CQE_DRAIN_PASS_BUDGET &&
       iree_status_is_ok(status);
       ++drain_pass) {
    iree_async_io_uring_notification_drain_pending(proactor);
    completed += iree_async_proactor_io_uring_drain_pending_software_operations(
        proactor);
    status = iree_async_proactor_io_uring_submit_cancel_requests(proactor);
    if (iree_status_is_ok(status)) {
      status = iree_io_uring_ring_submit(&proactor->ring, /*min_complete=*/0,
                                         IREE_IORING_ENTER_GETEVENTS);
    }
    if (iree_status_is_ok(status)) {
      uint32_t drain_tail_snapshot =
          iree_atomic_load((iree_atomic_int32_t*)proactor->ring.cq_tail,
                           iree_memory_order_acquire);
      if (*proactor->ring.cq_head == drain_tail_snapshot) {
        break;
      }
      while (*proactor->ring.cq_head != drain_tail_snapshot) {
        iree_io_uring_cqe_t* cqe =
            &proactor->ring
                 .cqes[*proactor->ring.cq_head & proactor->ring.cq_mask];
        completed +=
            iree_async_proactor_io_uring_process_cqe(proactor, cqe, &status);
        iree_io_uring_ring_cq_advance(&proactor->ring, 1);
      }
    }
  }

  // The final drains can enqueue software work for the next poll. Preserve
  // their normal wake path, including when kernel submission failed.
  // Acquire submissions from producers that suppressed their wake while this
  // dispatch interval was active. Later producers observe idle and wake us.
  iree_atomic_exchange(&proactor->polling.dispatch_tid, 0,
                       iree_memory_order_acq_rel);

  if (iree_status_is_ok(status)) {
    // Acquire notification intents from submitters that suppressed their wake
    // before dispatch_tid became idle. A monitor staged here must be submitted
    // before returning; future submitters use the normal wake path.
    if (iree_async_io_uring_notification_drain_pending(proactor)) {
      status = iree_io_uring_ring_submit(&proactor->ring, /*min_complete=*/0,
                                         IREE_IORING_ENTER_GETEVENTS);
    }

    // Drain pending messages from the fallback MPSC queue.
    // This handles messages that arrived via eventfd wake rather than MSG_RING.
    iree_async_proactor_io_uring_drain_pending_messages(proactor);

    // Drain pending semaphore wait completions.
    // These are pushed by timepoint callbacks when semaphores reach target
    // values. Count them as completions since they invoke user callbacks.
    completed +=
        iree_async_proactor_io_uring_drain_pending_semaphore_waits(proactor);

    // Third MPSC drain: software work pushed during the CQE drain loop and
    // semaphore wait dispatch. Each CQE drain pass may dispatch continuations
    // that push software work. Semaphore wait dispatch similarly pushes
    // continuation completions.
    completed += iree_async_proactor_io_uring_drain_pending_software_operations(
        proactor);
  }

  if (out_completed_count) {
    *out_completed_count = completed;
  }

  // Cooperative work can force a nonblocking native turn without expiring
  // the caller's timeout.
  if (iree_status_is_ok(status) && completed == 0 &&
      iree_timeout_as_duration_ns(timeout) == 0) {
    status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Poll owner lifecycle
//===----------------------------------------------------------------------===//

static void iree_async_proactor_io_uring_end_polling(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_io_uring_ring_end_polling(&proactor->ring);
}

//===----------------------------------------------------------------------===//
// Wake
//===----------------------------------------------------------------------===//

static void iree_async_proactor_io_uring_wake(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  if (proactor->wake_eventfd < 0) {
    return;
  }

  // Write to eventfd to wake a blocked poll. This is thread-safe and
  // signal-safe. EAGAIN means the counter is saturated and therefore already
  // readable; all other failures violate the live-proactor invariant.
  uint64_t value = 1;
  ssize_t result = 0;
  do {
    result = write(proactor->wake_eventfd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  if (result < 0 && errno == EAGAIN) {
    return;
  }
  IREE_ASSERT(result == sizeof(value),
              "failed to signal io_uring wake eventfd: %zd (errno=%d)", result,
              errno);
}

//===----------------------------------------------------------------------===//
// Socket
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_create_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_socket_create(proactor, type, options, out_socket);
}

static iree_status_t iree_async_proactor_io_uring_import_socket(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_socket_import(proactor, primitive, type, flags,
                                           out_socket);
}

static void iree_async_proactor_io_uring_destroy_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_t* socket) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_io_uring_socket_destroy(proactor, socket);
}

//===----------------------------------------------------------------------===//
// File vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_import_file(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Validate the file descriptor.
  int fd = primitive.value.fd;
  if (fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid file descriptor %d", fd);
  }

  // Allocate the file structure.
  iree_async_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*file), (void**)&file));
  memset(file, 0, sizeof(*file));

  // Initialize the file.
  iree_atomic_ref_count_init(&file->ref_count);
  file->proactor = base_proactor;
  file->primitive = primitive;
  file->fixed_file_index = -1;  // Not using io_uring fixed files yet.

  IREE_TRACE({
    iree_snprintf(file->debug_path, sizeof(file->debug_path), "fd:%d", fd);
  });

  *out_file = file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_io_uring_destroy_file(
    iree_async_proactor_t* base_proactor, iree_async_file_t* file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(file);

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Close the file descriptor if still open. The fd may already be -1 if
  // a FILE_CLOSE operation was submitted before the final release.
  int fd = file->primitive.value.fd;
  if (fd >= 0) {
    close(fd);
  }

  iree_allocator_free(allocator, file);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_proactor_io_uring_create_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t** out_event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  iree_async_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*event), (void**)&event));
  iree_status_t status = iree_async_event_native_initialize(&event->native);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&event->ref_count);
    event->proactor = base_proactor;
    event->fixed_file_index = -1;
    *out_event = event;
  } else {
    iree_allocator_free(allocator, event);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_async_proactor_io_uring_destroy_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!event) {
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  iree_async_event_native_deinitialize(&event->native);
  iree_allocator_free(allocator, event);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Notification vtable implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_create_notification(
    iree_async_proactor_t* base_proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_notification_create(proactor, flags,
                                                 out_notification);
}

static iree_status_t iree_async_proactor_io_uring_create_notification_shared(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_native_t* native,
    iree_async_notification_t** out_notification) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_notification_create_shared(proactor, native,
                                                        out_notification);
}

static void iree_async_proactor_io_uring_destroy_notification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_io_uring_notification_destroy(proactor, notification);
}

static iree_status_t iree_async_proactor_io_uring_import_fence(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  // Validate: must be a POSIX fd with a valid descriptor.
  // On validation error the caller retains fd ownership.
  if (fence.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "import_fence requires an FD primitive (got type %d)", (int)fence.type);
  }
  if (fence.value.fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import_fence fd must be >= 0 (got %d)",
                            fence.value.fd);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Get an SQE for POLL_ADD on the fence fd.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full, cannot submit fence import POLL_ADD");
  }

  // Allocate a tracker to associate the fd with the semaphore.
  iree_async_io_uring_fence_import_tracker_t* tracker = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      proactor->base.allocator, sizeof(*tracker), (void**)&tracker);
  if (!iree_status_is_ok(alloc_status)) {
    iree_io_uring_ring_sq_rollback(&proactor->ring, 1);
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    IREE_TRACE_ZONE_END(z0);
    return alloc_status;
  }

  // Retain the semaphore so it survives until the CQE fires.
  iree_async_semaphore_retain(semaphore);
  tracker->semaphore = semaphore;
  tracker->signal_value = signal_value;
  tracker->fence_fd = fence.value.fd;
  tracker->allocator = proactor->base.allocator;

  // Fill the POLL_ADD SQE.
  sqe->opcode = IREE_IORING_OP_POLL_ADD;
  sqe->fd = fence.value.fd;
  sqe->poll32_events = POLLIN;
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_FENCE_IMPORT,
                                                 (uintptr_t)tracker);
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  // Wake the poll thread to flush the SQE. Do NOT call ring_submit()
  // directly — only the poll thread may call io_uring_enter (SINGLE_ISSUER).
  // The SQE is already committed to sq_local_tail and will be included in
  // the next ring_submit() call during poll().
  iree_async_proactor_wake(&proactor->base);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_export_fence(
    iree_async_proactor_t* base_proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  IREE_ASSERT_ARGUMENT(out_fence);
  *out_fence = iree_async_primitive_none();

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a non-blocking eventfd. The caller will own this fd and can poll it,
  // pass it to Vulkan (VK_KHR_external_semaphore_fd), HIP, DRM/KMS, or use it
  // in io_uring LINK chains.
  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "eventfd creation failed for export_fence");
  }

  // Allocate a tracker to bridge the semaphore timepoint to the eventfd.
  iree_async_io_uring_fence_export_tracker_t* tracker = NULL;
  iree_status_t status = iree_allocator_malloc(
      proactor->base.allocator, sizeof(*tracker), (void**)&tracker);
  if (!iree_status_is_ok(status)) {
    close(efd);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Retain the semaphore so it survives until the timepoint callback fires.
  // The timepoint API does not hold a reference.
  iree_async_semaphore_retain(semaphore);
  tracker->semaphore = semaphore;
  tracker->eventfd = efd;
  tracker->allocator = proactor->base.allocator;

  // Set callback on the embedded timepoint before registering.
  tracker->timepoint.callback = iree_async_io_uring_fence_export_callback;
  tracker->timepoint.user_data = tracker;

  // Register the timepoint. If the semaphore has already reached wait_value,
  // the callback fires synchronously (writing to eventfd before we return).
  // If the semaphore is already failed, the callback fires with the failure
  // status (eventfd stays unreadable). acquire_timepoint always returns OK.
  status = iree_async_semaphore_acquire_timepoint(semaphore, wait_value,
                                                  &tracker->timepoint);

  if (iree_status_is_ok(status)) {
    // Return the eventfd as the exported fence. Caller owns the fd.
    *out_fence = iree_async_primitive_from_fd(efd);
  } else {
    // Should not happen (acquire_timepoint only fails on NULL semaphore), but
    // handle defensively.
    close(efd);
    iree_async_semaphore_release(semaphore);
    iree_allocator_free(proactor->base.allocator, tracker);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Relay vtable wrappers
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_register_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_register_relay(proactor, source, sink, flags,
                                            error_callback, out_relay);
}

static void iree_async_proactor_io_uring_unregister_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_t* relay,
    iree_async_relay_unregistered_callback_t callback) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_io_uring_unregister_relay(proactor, relay, callback);
}

//===----------------------------------------------------------------------===//
// Signal handling
//===----------------------------------------------------------------------===//

// Submits a multishot POLL_ADD for the signalfd. Called when the first signal
// subscription is added and we have a signalfd to monitor.
static iree_status_t iree_async_proactor_io_uring_submit_signal_poll(
    iree_async_proactor_io_uring_t* proactor, int signal_fd) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get an SQE for multishot POLL_ADD on the signalfd.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  iree_status_t status =
      sqe ? iree_ok_status()
          : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "SQ full, cannot submit signal poll");

  // Fill POLL_ADD SQE for multishot monitoring.
  if (iree_status_is_ok(status)) {
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IREE_IORING_OP_POLL_ADD;
    sqe->fd = signal_fd;
    sqe->poll32_events = POLLIN;
    sqe->len = IREE_IORING_POLL_ADD_MULTI;
    sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_SIGNAL, 0);
  }
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  if (iree_status_is_ok(status)) {
    proactor->signal.initialized = true;
    // Wake the poll thread to submit the signal POLL_ADD.
    iree_async_proactor_wake(&proactor->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_async_proactor_io_uring_subscribe_signal(
    iree_async_proactor_t* base_proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback,
    iree_async_signal_subscription_t** out_subscription) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_subscription);
  *out_subscription = NULL;

  if (signal <= IREE_ASYNC_SIGNAL_NONE || signal >= IREE_ASYNC_SIGNAL_COUNT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid signal type %d", (int)signal);
  }

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Claim global signal ownership (first proactor wins).
  iree_status_t status = iree_async_signal_claim_ownership(base_proactor);

  // If this is the first subscriber for this specific signal, add it to the
  // signalfd mask. This blocks only the signals that actually have subscribers.
  int signal_fd = -1;
  if (iree_status_is_ok(status) &&
      proactor->signal.subscriber_count[signal] == 0) {
    status = iree_async_linux_signal_add_signal(&proactor->signal.linux_state,
                                                signal, &signal_fd);
    if (!iree_status_is_ok(status)) {
      // Failed to add signal - release ownership if we were the first.
      iree_async_signal_release_ownership(base_proactor);
    }
  }

  // If this is the first signal subscription overall, submit POLL_ADD.
  if (iree_status_is_ok(status) && !proactor->signal.initialized &&
      signal_fd >= 0) {
    status =
        iree_async_proactor_io_uring_submit_signal_poll(proactor, signal_fd);
    if (!iree_status_is_ok(status)) {
      // Failed to submit poll - remove the signal we just added.
      iree_async_linux_signal_remove_signal(&proactor->signal.linux_state,
                                            signal);
      iree_async_signal_release_ownership(base_proactor);
    }
  }

  // Allocate and initialize the subscription.
  iree_async_signal_subscription_t* subscription = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, sizeof(*subscription),
                                   (void**)&subscription);
  }
  if (iree_status_is_ok(status)) {
    iree_async_signal_subscription_initialize(subscription, base_proactor,
                                              signal, callback);
    iree_async_signal_subscription_link(&proactor->signal.subscriptions[signal],
                                        subscription);
    ++proactor->signal.subscriber_count[signal];
    *out_subscription = subscription;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_async_proactor_io_uring_unsubscribe_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_signal_subscription_t* subscription) {
  if (!subscription) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_signal_t signal = subscription->signal;

  // If we're currently dispatching signals, defer the unsubscribe to avoid
  // corrupting the iteration. The subscription will be freed after dispatch.
  if (proactor->signal.dispatch_state.dispatching) {
    iree_async_signal_subscription_defer_unsubscribe(
        &proactor->signal.dispatch_state, subscription);
    // Note: count decrement and remove_signal happen in dispatch callback
    // when the deferred subscription is actually processed.
  } else {
    // Immediate unsubscribe: unlink, decrement count, and free.
    iree_async_signal_subscription_unlink(
        &proactor->signal.subscriptions[signal], subscription);
    iree_allocator_free(proactor->base.allocator, subscription);

    // If this was the last subscriber for this signal, remove it from the
    // signalfd mask so default behavior is restored.
    if (--proactor->signal.subscriber_count[signal] == 0) {
      iree_async_linux_signal_remove_signal(&proactor->signal.linux_state,
                                            signal);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_async_proactor_io_uring_set_message_callback(
    iree_async_proactor_t* base_proactor,
    iree_async_proactor_message_callback_t callback) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  proactor->message_callback = callback;
}

static iree_status_t iree_async_proactor_io_uring_send_message(
    iree_async_proactor_t* base_target, uint64_t message_data) {
  iree_async_proactor_io_uring_t* target =
      iree_async_proactor_io_uring_cast(base_target);
  iree_status_t status =
      iree_async_message_pool_send(&target->message_pool, message_data);
  if (iree_status_is_ok(status)) {
    base_target->vtable->wake(base_target);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

const iree_async_proactor_vtable_t iree_async_proactor_io_uring_vtable = {
    .destroy = iree_async_proactor_io_uring_destroy,
    .query_capabilities = iree_async_proactor_io_uring_query_capabilities,
    .submit = iree_async_proactor_io_uring_submit,
    .poll = iree_async_proactor_io_uring_poll,
    .end_polling = iree_async_proactor_io_uring_end_polling,
    .wake = iree_async_proactor_io_uring_wake,
    .cancel = iree_async_proactor_io_uring_cancel,
    .create_socket = iree_async_proactor_io_uring_create_socket,
    .import_socket = iree_async_proactor_io_uring_import_socket,
    .destroy_socket = iree_async_proactor_io_uring_destroy_socket,
    .import_file = iree_async_proactor_io_uring_import_file,
    .destroy_file = iree_async_proactor_io_uring_destroy_file,
    .create_event = iree_async_proactor_io_uring_create_event,
    .destroy_event = iree_async_proactor_io_uring_destroy_event,
    .register_event_source = iree_async_io_uring_event_source_register,
    .unregister_event_source = iree_async_io_uring_event_source_unregister,
    .create_notification = iree_async_proactor_io_uring_create_notification,
    .create_notification_shared =
        iree_async_proactor_io_uring_create_notification_shared,
    .destroy_notification = iree_async_proactor_io_uring_destroy_notification,
    .notification_signal = iree_async_io_uring_notification_signal,
    .notification_wait = iree_async_io_uring_notification_wait,
    .register_relay = iree_async_proactor_io_uring_register_relay,
    .unregister_relay = iree_async_proactor_io_uring_unregister_relay,
    .register_buffer = iree_async_proactor_io_uring_register_buffer,
    .register_dmabuf = iree_async_proactor_io_uring_register_dmabuf,
    .unregister_buffer = iree_async_proactor_io_uring_unregister_buffer,
    .register_slab = iree_async_proactor_io_uring_register_slab,
    .import_fence = iree_async_proactor_io_uring_import_fence,
    .export_fence = iree_async_proactor_io_uring_export_fence,
    .set_message_callback = iree_async_proactor_io_uring_set_message_callback,
    .send_message = iree_async_proactor_io_uring_send_message,
    .subscribe_signal = iree_async_proactor_io_uring_subscribe_signal,
    .unsubscribe_signal = iree_async_proactor_io_uring_unsubscribe_signal,
};
