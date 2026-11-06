// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/membership.h"

#include "iree/async/frontier_tracker.h"
#include "iree/hal/driver/amd/status.h"

typedef enum iree_hal_amd_gpu_queue_slot_state_e {
  IREE_HAL_AMD_GPU_QUEUE_SLOT_VACANT = 0,
  IREE_HAL_AMD_GPU_QUEUE_SLOT_CONSTRUCTING,
  IREE_HAL_AMD_GPU_QUEUE_SLOT_LIVE,
  IREE_HAL_AMD_GPU_QUEUE_SLOT_DRAINING,
  IREE_HAL_AMD_GPU_QUEUE_SLOT_REMOVING,
  IREE_HAL_AMD_GPU_QUEUE_SLOT_PRESERVED,
} iree_hal_amd_gpu_queue_slot_state_t;

struct iree_hal_amd_gpu_queue_slot_t {
  // Device-owned membership outliving native signal readers.
  iree_hal_amd_gpu_queue_membership_t* membership;
  // Older stable coordinate; the link never changes after installation.
  iree_hal_amd_gpu_queue_slot_t* next;
  // Borrowed queue; construction publishes this before worker installation.
  // LIVE slots admit trims; DRAINING and PRESERVED admit terminal inspection
  // only. Removal joins every borrower before native shutdown.
  iree_hal_queue_t* queue;
  // Native causal coordinate; source indices additionally skip device services.
  uint32_t ordinal;
  // Native ownership phase. Mutation requires both membership mutexes;
  // lifecycle traversal reads it under slots_mutex alone.
  iree_hal_amd_gpu_queue_slot_state_t state;
  // Completed epoch continued by the next queue using this coordinate.
  uint64_t epoch;
  // Lifecycle operations borrowing queue/cache state under slots_mutex.
  // Ordinary submission and retirement never touch this count.
  uint32_t borrowers;
  // Native cell ownership survives the producer and its callback records.
  iree_hal_amd_gpu_signal_cache_t signals;
};

// Construction and removal exclude all new borrowers. Draining excludes new
// trims while terminal callbacks may still inspect the queue's accepted work.
static bool iree_hal_amd_gpu_queue_slot_is_unborrowed(void* user_data) {
  iree_hal_amd_gpu_queue_slot_t* slot = user_data;
  iree_slim_mutex_lock(&slot->membership->slots_mutex);
  const bool idle = slot->borrowers == 0;
  iree_slim_mutex_unlock(&slot->membership->slots_mutex);
  return idle;
}

static void iree_hal_amd_gpu_queue_slot_return(
    iree_hal_amd_gpu_queue_slot_t* slot) {
  iree_hal_amd_gpu_queue_membership_t* membership = slot->membership;
  iree_slim_mutex_lock(&membership->slots_mutex);
  const bool idle = --slot->borrowers == 0;
  iree_slim_mutex_unlock(&membership->slots_mutex);
  if (idle) {
    iree_notification_post(&membership->notification, IREE_ALL_WAITERS);
  }
}

struct iree_hal_amd_gpu_queue_sources_t {
  // Extent of each table, including one temporary release observer.
  uint32_t capacity;
  // Table currently borrowed by the workers; the other is cold scratch.
  uint32_t active;
  // Publication tables with stable source indices and vacant NULL callbacks.
  iree_hal_amd_gpu_publication_source_t* publication[2];
  // Dense retirement tables; callback order puts feedback after native work.
  iree_hal_amd_gpu_completion_source_t* completion[2];
};

struct iree_hal_amd_gpu_queue_join_t {
  // Borrowed device membership, live through this caller's final release.
  iree_hal_amd_gpu_queue_membership_t* membership;
  // Borrowed queue whose accepted work must finish before removal.
  iree_hal_queue_t* queue;
  // Next waiting release, protected by the completion drain mutex.
  iree_hal_amd_gpu_queue_join_t* next;
  // List link pointing here, or NULL when not registered.
  iree_hal_amd_gpu_queue_join_t** previous;
  // Release-publishes status after the observer establishes idleness/failure.
  iree_atomic_int32_t completed;
  // Native observation result, distinct from terminal cleanup ownership.
  amdf_status_t status;
};

void iree_hal_amd_gpu_queue_membership_initialize(
    iree_hal_amd_gpu_queue_membership_t* membership) {
  membership->frontier_count = membership->provisioned_count;
  iree_slim_mutex_initialize(&membership->mutex);
  iree_slim_mutex_initialize(&membership->slots_mutex);
  iree_notification_initialize(&membership->notification);
}

static void iree_hal_amd_gpu_queue_membership_clear_frontier(
    iree_hal_amd_gpu_queue_membership_t* membership) {
  if (!membership->frontier.tracker) {
    return;
  }
  for (uint32_t i = 0; i < membership->frontier.count; ++i) {
    iree_async_frontier_tracker_retire_axis(
        membership->frontier.tracker,
        membership->frontier.base_axis | ((uint64_t)i << 24),
        iree_status_from_code(IREE_STATUS_CANCELLED));
  }
  iree_async_frontier_tracker_release(membership->frontier.tracker);
  memset(&membership->frontier, 0, sizeof(membership->frontier));
}

bool iree_hal_amd_gpu_queue_membership_is_empty(
    iree_hal_amd_gpu_queue_membership_t* membership) {
  for (iree_hal_amd_gpu_queue_slot_t* slot = membership->slots; slot;
       slot = slot->next) {
    if (slot->state != IREE_HAL_AMD_GPU_QUEUE_SLOT_VACANT) {
      return false;
    }
  }
  return true;
}

bool iree_hal_amd_gpu_queue_membership_deinitialize(
    iree_hal_amd_gpu_queue_membership_t* membership) {
  if (!iree_hal_amd_gpu_queue_membership_is_empty(membership)) {
    return false;
  }
  while (membership->slots) {
    iree_hal_amd_gpu_queue_slot_t* slot = membership->slots;
    membership->slots = slot->next;
    iree_hal_amd_gpu_signal_cache_deinitialize(membership->context,
                                               &slot->signals);
    iree_allocator_free_aligned(membership->host_allocator, slot);
  }
  iree_hal_amd_gpu_queue_membership_clear_frontier(membership);
  iree_allocator_free(membership->host_allocator, membership->sources);
  iree_notification_deinitialize(&membership->notification);
  iree_slim_mutex_deinitialize(&membership->slots_mutex);
  iree_slim_mutex_deinitialize(&membership->mutex);
  return true;
}

static bool iree_hal_amd_gpu_queue_membership_poll_join(
    void* user_data, amdf_status_t status,
    iree_hal_amd_gpu_completion_scope_t* scope,
    iree_hal_amd_gpu_completion_wait_t* out_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy) {
  iree_hal_amd_gpu_queue_membership_t* membership = user_data;
  bool completed = false;
  for (iree_hal_amd_gpu_queue_join_t* join = membership->joins; join;
       join = join->next) {
    if (!iree_atomic_load(&join->completed, iree_memory_order_relaxed) &&
        (!amdf_status_is_ok(status) ||
         membership->lifecycle->is_idle(join->queue))) {
      join->status = status;
      iree_atomic_store(&join->completed, 1, iree_memory_order_release);
      completed = true;
    }
  }
  if (completed) {
    iree_notification_post(&membership->notification, IREE_ALL_WAITERS);
  }
  return false;
}

// Rebuild only the spare callback table, preserving retained sources' order.
// The completion handoff joins removed sources and coherent observation while
// relocating active callbacks that remain members. Neither worker keeps the
// old table after its handoff returns.
static amdf_status_t iree_hal_amd_gpu_queue_membership_rebind(
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_hal_queue_t* excluded_queue) {
  iree_hal_amd_gpu_queue_sources_t* sources = membership->sources;
  const uint32_t next = sources->active ^ 1;
  iree_hal_amd_gpu_publication_source_t* publication =
      sources->publication[next];
  iree_hal_amd_gpu_completion_source_t* completion = sources->completion[next];
  memset(publication, 0, sources->capacity * sizeof(*publication));
  memset(completion, 0, sources->capacity * sizeof(*completion));
  uint32_t completion_count = 0;
  for (uint32_t i = 0; i < membership->provisioned_count; ++i) {
    publication[i] =
        membership->lifecycle->publication_source(membership->provisioned[i]);
    completion[completion_count++] =
        membership->lifecycle->completion_source(membership->provisioned[i]);
  }
  memcpy(publication + membership->provisioned_count,
         membership->publication_tail, sizeof(membership->publication_tail));
  for (iree_hal_amd_gpu_queue_slot_t* slot = membership->slots; slot;
       slot = slot->next) {
    if (!slot->queue || slot->queue == excluded_queue ||
        slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_REMOVING ||
        slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_PRESERVED) {
      continue;
    }
    publication[slot->ordinal + 3] =
        membership->lifecycle->publication_source(slot->queue);
    completion[completion_count++] =
        membership->lifecycle->completion_source(slot->queue);
  }
  memcpy(completion + completion_count, membership->completion_tail,
         membership->completion_tail_count * sizeof(*completion));
  completion_count += membership->completion_tail_count;
  if (membership->joins) {
    completion[completion_count++] = (iree_hal_amd_gpu_completion_source_t){
        .fn = iree_hal_amd_gpu_queue_membership_poll_join,
        .user_data = membership};
  }
  iree_hal_amd_gpu_publication_set_sources(
      membership->publication,
      membership->provisioned_count + membership->slot_count + 3, publication);
  iree_hal_amd_gpu_completion_set_sources(membership->completion,
                                          completion_count, completion);
  sources->active = next;
  // Every replacement interrupts observation, including removal of an idle
  // queue. The worker may have consumed the last native event while replacement
  // excluded a retained source's ready copy. Rearm that work after the swap.
  // A terminal observer has stopped using the native event. Table handoffs
  // still join their host readers, without reentering the failed device.
  return amdf_status_is_ok(
             iree_hal_amd_gpu_completion_query_status(membership->completion))
             ? iree_hal_amd_gpu_context_signal(membership->context)
             : AMDF_STATUS_OK;
}

// Acquisitions secure both source tables, including the release observer's
// entry. Removing a queue cannot fail because host scratch is unavailable.
static iree_status_t iree_hal_amd_gpu_queue_membership_reserve_sources(
    iree_hal_amd_gpu_queue_membership_t* membership, uint32_t required) {
  iree_hal_amd_gpu_queue_sources_t* previous = membership->sources;
  if (previous && previous->capacity >= required) {
    return iree_ok_status();
  }
  uint32_t capacity = previous ? previous->capacity * 2 : 8;
  while (capacity < required) {
    capacity *= 2;
  }
  const iree_host_size_t total_size =
      sizeof(iree_hal_amd_gpu_queue_sources_t) +
      2 * capacity *
          (sizeof(iree_hal_amd_gpu_publication_source_t) +
           sizeof(iree_hal_amd_gpu_completion_source_t));
  iree_hal_amd_gpu_queue_sources_t* sources = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(membership->host_allocator,
                                             total_size, (void**)&sources));
  sources->capacity = capacity;
  sources->publication[0] = (void*)(sources + 1);
  sources->publication[1] = sources->publication[0] + capacity;
  sources->completion[0] = (void*)(sources->publication[1] + capacity);
  sources->completion[1] = sources->completion[0] + capacity;
  membership->sources = sources;
  const amdf_status_t status =
      iree_hal_amd_gpu_queue_membership_rebind(membership, NULL);
  iree_allocator_free(membership->host_allocator, previous);
  return IREE_HAL_AMD_STATUS_FROM_AMDF(status, "GPU source installation wake");
}

static bool iree_hal_amd_gpu_queue_membership_joined(void* user_data) {
  iree_hal_amd_gpu_queue_join_t* join = user_data;
  return iree_atomic_load(&join->completed, iree_memory_order_acquire) ||
         !amdf_status_is_ok(iree_hal_amd_gpu_completion_query_status(
             join->membership->completion));
}

static bool iree_hal_amd_gpu_queue_membership_terminal_joined(void* user_data) {
  iree_hal_amd_gpu_queue_membership_t* membership = user_data;
  return iree_atomic_load(&membership->terminal.completed,
                          iree_memory_order_acquire);
}

static bool iree_hal_amd_gpu_queue_membership_release(void* user_data,
                                                      iree_hal_queue_t* queue) {
  iree_hal_amd_gpu_queue_slot_t* slot = user_data;
  iree_hal_amd_gpu_queue_membership_t* membership = slot->membership;
  iree_slim_mutex_lock(&membership->mutex);
  membership->lifecycle->close(queue);
  iree_slim_mutex_lock(&membership->slots_mutex);
  slot->state = IREE_HAL_AMD_GPU_QUEUE_SLOT_DRAINING;
  iree_slim_mutex_unlock(&membership->slots_mutex);
  iree_slim_mutex_unlock(&membership->mutex);
  // An already-admitted trim may still enqueue cleanup or begin an async
  // scratch release. Join it before proving idleness, with this queue's worker
  // sources still installed to finish that work. Terminal inspection remains
  // admitted so failure can stop and retire accepted work during this join.
  iree_notification_await(&membership->notification,
                          iree_hal_amd_gpu_queue_slot_is_unborrowed, slot,
                          iree_infinite_timeout());
  iree_slim_mutex_lock(&membership->mutex);
  iree_hal_amd_gpu_queue_join_t join = {.membership = membership,
                                        .queue = queue};
  // An idle queue already has its accepted-work proof. Its source removal
  // below joins the final observer access without asking that observer to
  // acknowledge idleness again. This also permits an independent callback to
  // release an unused queue while terminal traversal joins other owners.
  const bool idle = membership->lifecycle->is_idle(queue);
  amdf_status_t status =
      idle ? AMDF_STATUS_OK
           : iree_hal_amd_gpu_completion_query_status(membership->completion);
  const bool joining = amdf_status_is_ok(status) && !idle;
  if (joining) {
    iree_slim_mutex_lock(&membership->completion->drain_mutex);
    join.next = membership->joins;
    join.previous = &membership->joins;
    if (join.next) {
      join.next->previous = &join.next;
    }
    *join.previous = &join;
    iree_slim_mutex_unlock(&membership->completion->drain_mutex);
    status = iree_hal_amd_gpu_queue_membership_rebind(membership, NULL);
  }
  iree_slim_mutex_unlock(&membership->mutex);
  if (amdf_status_is_ok(status) && joining) {
    iree_notification_await(&membership->notification,
                            iree_hal_amd_gpu_queue_membership_joined, &join,
                            iree_infinite_timeout());
    status = iree_hal_amd_gpu_completion_query_status(membership->completion);
    if (amdf_status_is_ok(status)) {
      status = join.status;
    }
  }
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_context_fail_execution(
        membership->context, status, "GPU acquired queue release observation");
    // The first failed observation is not a lifetime proof. The terminal
    // owner uses slots_mutex; this caller holds no membership exclusion.
    if (membership->failure) {
      if (!iree_notification_await(
              &membership->notification,
              iree_hal_amd_gpu_queue_membership_terminal_joined, membership,
              iree_make_deadline(membership->failure->deadline))) {
        iree_hal_amd_gpu_failure_poll(membership->failure, iree_time_now());
        // A diagnostic deadline cannot detach accepted CPU callbacks. The
        // terminal owner still joins them even when native access is unproven.
        iree_notification_await(
            &membership->notification,
            iree_hal_amd_gpu_queue_membership_terminal_joined, membership,
            iree_infinite_timeout());
      }
      status = membership->terminal.status;
    }
  }
  iree_slim_mutex_lock(&membership->mutex);
  if (joining) {
    // Join the observer's final list/status access before returning this stack
    // record. Other releases keep their own records and can continue waiting.
    iree_slim_mutex_lock(&membership->completion->drain_mutex);
    *join.previous = join.next;
    if (join.next) {
      join.next->previous = join.previous;
    }
    iree_slim_mutex_unlock(&membership->completion->drain_mutex);
  }
  const amdf_status_t removal_status = iree_hal_amd_gpu_queue_membership_rebind(
      membership, amdf_status_is_ok(status) ? queue : NULL);
  if (amdf_status_is_ok(status)) {
    status = removal_status;
  }
  iree_slim_mutex_lock(&membership->slots_mutex);
  slot->state = IREE_HAL_AMD_GPU_QUEUE_SLOT_REMOVING;
  iree_slim_mutex_unlock(&membership->slots_mutex);
  iree_slim_mutex_unlock(&membership->mutex);
  // Only this slot's lifecycle borrowers can still access its native state.
  // Other acquired queues remain independent of the native stop/unmap calls.
  iree_notification_await(&membership->notification,
                          iree_hal_amd_gpu_queue_slot_is_unborrowed, slot,
                          iree_infinite_timeout());
  if (amdf_status_is_ok(status)) {
    status = membership->lifecycle->shutdown(queue);
  }
  if (amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_tsan_registry_remove_queue(membership->tsan,
                                                slot->ordinal);
  }
  iree_slim_mutex_lock(&membership->mutex);
  iree_slim_mutex_lock(&membership->slots_mutex);
  if (amdf_status_is_ok(status)) {
    slot->epoch = membership->lifecycle->completed_epoch(queue);
    slot->queue = NULL;
    slot->state = IREE_HAL_AMD_GPU_QUEUE_SLOT_VACANT;
  } else {
    slot->state = IREE_HAL_AMD_GPU_QUEUE_SLOT_PRESERVED;
  }
  iree_slim_mutex_unlock(&membership->slots_mutex);
  iree_slim_mutex_unlock(&membership->mutex);
  if (!amdf_status_is_ok(status)) {
    iree_atomic_store(&membership->context->cleanup_blocked, 1,
                      iree_memory_order_relaxed);
    if (!amdf_status_is_ok(removal_status)) {
      iree_hal_amd_gpu_context_fail_execution(
          membership->context, removal_status, "GPU source removal wake");
    } else {
      iree_hal_amd_gpu_context_report(
          membership->context, status,
          "GPU acquired queue release could not prove native quiescence");
    }
  }
  iree_hal_amd_gpu_context_wake_failure(membership->context);
  return amdf_status_is_ok(status);
}

static iree_status_t iree_hal_amd_gpu_queue_membership_reserve_frontiers(
    iree_hal_amd_gpu_queue_membership_t* membership, uint32_t required) {
  if (required <= membership->frontier_count) {
    return iree_ok_status();
  }
  uint32_t count = membership->frontier_count;
  while (count < required) {
    count = iree_min(count * 2, 256u);
  }
  iree_status_t status =
      iree_hal_amd_gpu_program_pool_reserve_queues(membership->programs, count);
  for (uint32_t i = 0;
       i < membership->provisioned_count && iree_status_is_ok(status); ++i) {
    status = membership->lifecycle->reserve_frontiers(
        membership->provisioned[i], count, &membership->notification);
  }
  for (iree_hal_amd_gpu_queue_slot_t* slot = membership->slots;
       slot && iree_status_is_ok(status); slot = slot->next) {
    if (slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_LIVE) {
      status = membership->lifecycle->reserve_frontiers(
          slot->queue, count, &membership->notification);
    }
  }
  if (iree_status_is_ok(status)) {
    membership->frontier_count = count;
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_queue_membership_acquire(
    iree_hal_amd_gpu_queue_membership_t* membership,
    const iree_hal_queue_params_t* params, iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  iree_slim_mutex_lock(&membership->mutex);
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      iree_hal_amd_gpu_completion_query_status(membership->completion),
      "GPU queue acquisition after observer failure");
  iree_hal_amd_gpu_queue_slot_t* slot = membership->slots;
  while (slot && slot->state != IREE_HAL_AMD_GPU_QUEUE_SLOT_VACANT) {
    slot = slot->next;
  }
  const uint32_t ordinal =
      slot ? slot->ordinal
           : membership->provisioned_count + membership->slot_count;
  if (iree_status_is_ok(status) && ordinal >= 256) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "GPU device has 256 live queue coordinates");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_queue_membership_reserve_sources(membership,
                                                               ordinal + 5);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_queue_membership_reserve_frontiers(membership,
                                                                 ordinal + 1);
  }
  if (iree_status_is_ok(status) && !slot) {
    status = iree_allocator_malloc_aligned(
        membership->host_allocator, sizeof(*slot),
        iree_alignof(iree_hal_amd_gpu_queue_slot_t), 0, (void**)&slot);
    if (iree_status_is_ok(status)) {
      slot->membership = membership;
      slot->ordinal = ordinal;
      iree_hal_amd_gpu_signal_cache_initialize(membership->publication,
                                               ordinal + 3, &slot->signals);
      iree_slim_mutex_lock(&membership->slots_mutex);
      slot->next = membership->slots;
      membership->slots = slot;
      ++membership->slot_count;
      iree_slim_mutex_unlock(&membership->slots_mutex);
    }
  }
  if (iree_status_is_ok(status) && membership->frontier.tracker &&
      ordinal >= membership->frontier.count) {
    status = iree_async_frontier_tracker_register_axis(
        membership->frontier.tracker,
        membership->frontier.base_axis | ((uint64_t)ordinal << 24), NULL);
    if (iree_status_is_ok(status)) {
      ++membership->frontier.count;
    }
  }
  const uint32_t frontier_count = membership->frontier_count;
  iree_hal_queue_t* queue = NULL;
  bool constructing = false;
  if (iree_status_is_ok(status)) {
    // Reserve the coordinate before releasing membership exclusion. Terminal
    // traversal cannot finish while this native owner is under construction.
    iree_slim_mutex_lock(&membership->slots_mutex);
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        iree_hal_amd_gpu_completion_query_status(membership->completion),
        "GPU queue acquisition after native failure");
    if (iree_status_is_ok(status)) {
      slot->state = IREE_HAL_AMD_GPU_QUEUE_SLOT_CONSTRUCTING;
      constructing = true;
    }
    iree_slim_mutex_unlock(&membership->slots_mutex);
  }
  iree_slim_mutex_unlock(&membership->mutex);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&membership->notification,
                            iree_hal_amd_gpu_queue_slot_is_unborrowed, slot,
                            iree_infinite_timeout());
    status = iree_hal_amd_gpu_signal_cache_reserve_frontiers(
        &slot->signals, frontier_count, membership->host_allocator);
    if (iree_status_is_ok(status)) {
      status = membership->create(membership->user_data, ordinal, ordinal + 3,
                                  frontier_count, slot->epoch, &slot->signals,
                                  params, &queue);
    }
  }
  iree_slim_mutex_lock(&membership->mutex);
  if (iree_status_is_ok(status)) {
    membership->lifecycle->assign_frontier(
        queue, membership->frontier.tracker,
        membership->frontier.base_axis | ((uint64_t)ordinal << 24));
    iree_slim_mutex_lock(&membership->slots_mutex);
    slot->queue = queue;
    iree_slim_mutex_unlock(&membership->slots_mutex);
    membership->lifecycle->set_release(
        queue, iree_hal_amd_gpu_queue_membership_release, slot);
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        iree_hal_amd_gpu_queue_membership_rebind(membership, NULL),
        "GPU acquired queue source wake");
    // A concurrent acquisition may have widened the coordinate space while
    // native construction ran. The unexposed queue catches up through its
    // exact publisher before any operation can use its new coordinate.
    if (iree_status_is_ok(status) &&
        frontier_count < membership->frontier_count) {
      status = membership->lifecycle->reserve_frontiers(
          queue, membership->frontier_count, &membership->notification);
    }
    if (iree_status_is_ok(status) && membership->profile) {
      status = membership->lifecycle->set_profile(queue, membership->profile);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_amd_gpu_tsan_registry_add_queue(membership->tsan, ordinal);
    }
    if (iree_status_is_ok(status)) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          iree_hal_amd_gpu_completion_query_status(membership->completion),
          "GPU acquired queue after native failure");
    }
  }
  if (constructing) {
    iree_slim_mutex_lock(&membership->slots_mutex);
    slot->state = queue ? IREE_HAL_AMD_GPU_QUEUE_SLOT_LIVE
                        : IREE_HAL_AMD_GPU_QUEUE_SLOT_VACANT;
    iree_slim_mutex_unlock(&membership->slots_mutex);
  }
  iree_slim_mutex_unlock(&membership->mutex);
  iree_hal_amd_gpu_context_wake_failure(membership->context);
  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else {
    iree_hal_queue_release(queue);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_queue_membership_configure_profile(
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_hal_amd_gpu_profile_t* profile) {
  iree_status_t status = iree_ok_status();
  // Disabling visits every live queue even after an error: session destruction
  // must never leave a queue borrowing freed host capture storage.
  for (uint32_t i = 0; i < membership->provisioned_count &&
                       (!profile || iree_status_is_ok(status));
       ++i) {
    status = iree_status_join(status, membership->lifecycle->set_profile(
                                          membership->provisioned[i], profile));
  }
  for (iree_hal_amd_gpu_queue_slot_t* slot = membership->slots;
       slot && (!profile || iree_status_is_ok(status)); slot = slot->next) {
    if (slot->queue) {
      status = iree_status_join(
          status, membership->lifecycle->set_profile(slot->queue, profile));
    }
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_queue_membership_set_profile(
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_hal_amd_gpu_profile_t* profile) {
  iree_slim_mutex_lock(&membership->mutex);
  iree_status_t status =
      iree_hal_amd_gpu_queue_membership_configure_profile(membership, profile);
  if (!iree_status_is_ok(status) && profile) {
    status = iree_status_join(
        status,
        iree_hal_amd_gpu_queue_membership_configure_profile(membership, NULL));
  }
  membership->profile = iree_status_is_ok(status) ? profile : NULL;
  iree_slim_mutex_unlock(&membership->mutex);
  return status;
}

iree_status_t iree_hal_amd_gpu_queue_membership_assign_frontier(
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t base_axis) {
  const uint32_t count = membership->provisioned_count + membership->slot_count;
  uint32_t registered_count = 0;
  iree_status_t status = iree_ok_status();
  while (tracker && registered_count < count && iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_register_axis(
        tracker, base_axis | ((uint64_t)registered_count << 24), NULL);
    if (iree_status_is_ok(status)) {
      ++registered_count;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_retain(tracker);
    iree_hal_amd_gpu_queue_membership_clear_frontier(membership);
    membership->frontier.tracker = tracker;
    membership->frontier.base_axis = base_axis;
    membership->frontier.count = registered_count;
    for (uint32_t i = 0; i < membership->provisioned_count; ++i) {
      membership->lifecycle->assign_frontier(
          membership->provisioned[i], tracker, base_axis | ((uint64_t)i << 24));
    }
    for (iree_hal_amd_gpu_queue_slot_t* slot = membership->slots; slot;
         slot = slot->next) {
      if (slot->queue) {
        membership->lifecycle->assign_frontier(
            slot->queue, tracker, base_axis | ((uint64_t)slot->ordinal << 24));
      }
    }
  } else {
    for (uint32_t i = 0; i < registered_count; ++i) {
      iree_async_frontier_tracker_retire_axis(
          tracker, base_axis | ((uint64_t)i << 24),
          iree_status_from_code(IREE_STATUS_CANCELLED));
    }
  }
  return status;
}

void iree_hal_amd_gpu_queue_membership_notify_failure(
    iree_hal_amd_gpu_queue_membership_t* membership) {
  iree_slim_mutex_lock(&membership->slots_mutex);
  iree_hal_amd_gpu_queue_slot_t* head = membership->slots;
  iree_slim_mutex_unlock(&membership->slots_mutex);
  for (iree_hal_amd_gpu_queue_slot_t* slot = head; slot; slot = slot->next) {
    iree_slim_mutex_lock(&membership->slots_mutex);
    iree_hal_queue_t* queue =
        slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_LIVE ? slot->queue : NULL;
    if (queue) {
      ++slot->borrowers;
    }
    iree_slim_mutex_unlock(&membership->slots_mutex);
    if (queue) {
      membership->lifecycle->notify_failure(queue);
      iree_hal_amd_gpu_queue_slot_return(slot);
    }
  }
}

static bool iree_hal_amd_gpu_queue_membership_step_failure(
    iree_hal_amd_gpu_queue_membership_t* membership, iree_hal_queue_t* queue,
    iree_hal_amd_gpu_failure_phase_t phase, iree_status_code_t code,
    uint32_t ordinal, uint32_t completed, uint32_t total,
    amdf_status_t* out_status) {
  iree_hal_amd_gpu_failure_set_phase(membership->failure, phase, ordinal,
                                     completed, total);
  switch (phase) {
    case IREE_HAL_AMD_GPU_FAILURE_CLOSING:
      membership->lifecycle->close(queue);
      membership->lifecycle->notify_failure(queue);
      return true;
    case IREE_HAL_AMD_GPU_FAILURE_INACTIVATING:
      return membership->lifecycle->try_inactivate(queue, out_status);
    case IREE_HAL_AMD_GPU_FAILURE_RETIRING:
      return membership->lifecycle->retire_failed(queue, code);
    case IREE_HAL_AMD_GPU_FAILURE_JOINING:
      return membership->lifecycle->is_host_idle(queue);
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
}

bool iree_hal_amd_gpu_queue_membership_try_failure(
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_hal_amd_gpu_failure_phase_t phase, iree_status_code_t code,
    amdf_status_t* out_status) {
  if (!iree_slim_mutex_try_lock(&membership->slots_mutex)) {
    return false;
  }
  // Once failure is published, no new construction can reserve a coordinate.
  // The stable list includes every constructor already admitted; its slot
  // remains pending until native initialization or rollback returns.
  iree_hal_amd_gpu_queue_slot_t* head = membership->slots;
  const uint32_t total = membership->provisioned_count + membership->slot_count;
  iree_slim_mutex_unlock(&membership->slots_mutex);
  amdf_status_t status = AMDF_STATUS_OK;
  bool ready = true;
  uint32_t completed = 0;
  for (uint32_t i = 0;
       i < membership->provisioned_count && amdf_status_is_ok(status); ++i) {
    const bool queue_ready = iree_hal_amd_gpu_queue_membership_step_failure(
        membership, membership->provisioned[i], phase, code, i, completed,
        total, &status);
    completed += queue_ready;
    ready &= queue_ready;
  }
  for (iree_hal_amd_gpu_queue_slot_t* slot = head;
       slot && amdf_status_is_ok(status); slot = slot->next) {
    iree_slim_mutex_lock(&membership->slots_mutex);
    const bool pending =
        slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_CONSTRUCTING ||
        slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_REMOVING;
    iree_hal_queue_t* queue = pending ? NULL : slot->queue;
    if (queue) {
      ++slot->borrowers;
    }
    iree_slim_mutex_unlock(&membership->slots_mutex);
    // A slow native call on one coordinate cannot prevent closing or stopping
    // the others. It still prevents this phase's overall ownership proof.
    ready &= !pending;
    if (queue) {
      const bool queue_ready = iree_hal_amd_gpu_queue_membership_step_failure(
          membership, queue, phase, code, slot->ordinal, completed, total,
          &status);
      completed += queue_ready;
      ready &= queue_ready;
      iree_hal_amd_gpu_queue_slot_return(slot);
    } else if (!pending) {
      ++completed;
    }
  }
  if (ready || !amdf_status_is_ok(status)) {
    *out_status = status;
    return true;
  }
  return false;
}

void iree_hal_amd_gpu_queue_membership_finish_failure(
    iree_hal_amd_gpu_queue_membership_t* membership, amdf_status_t status) {
  membership->terminal.status = status;
  iree_atomic_store(&membership->terminal.completed, 1,
                    iree_memory_order_release);
  iree_notification_post(&membership->notification, IREE_ALL_WAITERS);
}

amdf_status_t iree_hal_amd_gpu_queue_membership_trim(
    iree_hal_amd_gpu_queue_membership_t* membership) {
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0;
       i < membership->provisioned_count && amdf_status_is_ok(status); ++i) {
    if (membership->lifecycle->trim_scratch) {
      status = membership->lifecycle->trim_scratch(membership->provisioned[i]);
    }
    membership->lifecycle->trim(membership->provisioned[i]);
  }
  iree_slim_mutex_lock(&membership->slots_mutex);
  iree_hal_amd_gpu_queue_slot_t* head = membership->slots;
  iree_slim_mutex_unlock(&membership->slots_mutex);
  for (iree_hal_amd_gpu_queue_slot_t* slot = head; slot; slot = slot->next) {
    iree_slim_mutex_lock(&membership->slots_mutex);
    iree_hal_queue_t* queue =
        slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_LIVE ? slot->queue : NULL;
    const bool vacant = slot->state == IREE_HAL_AMD_GPU_QUEUE_SLOT_VACANT &&
                        slot->borrowers == 0;
    if (queue || vacant) {
      ++slot->borrowers;
    }
    iree_slim_mutex_unlock(&membership->slots_mutex);
    if (queue) {
      if (membership->lifecycle->trim_scratch && amdf_status_is_ok(status)) {
        status = membership->lifecycle->trim_scratch(queue);
      }
      membership->lifecycle->trim(queue);
    } else if (vacant) {
      iree_hal_amd_gpu_signal_cache_trim(membership->context, 0,
                                         &slot->signals);
    }
    if (queue || vacant) {
      iree_hal_amd_gpu_queue_slot_return(slot);
    }
  }
  return status;
}
