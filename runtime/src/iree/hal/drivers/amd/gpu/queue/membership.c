// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/membership.h"

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/amd/status.h"

struct iree_hal_amd_gpu_queue_slot_t {
  // Device-owned membership outliving native signal readers.
  iree_hal_amd_gpu_queue_membership_t* membership;
  // Older stable coordinate; the link never changes after installation.
  iree_hal_amd_gpu_queue_slot_t* next;
  // Borrowed public queue, or NULL after a successful synchronous release.
  iree_hal_queue_t* queue;
  // Native causal coordinate; source indices additionally skip device services.
  uint32_t ordinal;
  // Completed epoch continued by the next queue using this coordinate.
  uint64_t epoch;
  // Native cell ownership survives the producer and its callback records.
  iree_hal_amd_gpu_signal_cache_t signals;
};

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
    if (slot->queue) {
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
    iree_hal_amd_gpu_completion_wait_t* out_wait) {
  iree_hal_amd_gpu_queue_membership_t* membership = user_data;
  if (!iree_atomic_load(&membership->join.completed,
                        iree_memory_order_relaxed) &&
      (!amdf_status_is_ok(status) ||
       membership->lifecycle->is_idle(membership->join.queue))) {
    membership->join.status = status;
    iree_atomic_store(&membership->join.completed, 1,
                      iree_memory_order_release);
    iree_notification_post(&membership->notification, 1);
  }
  return false;
}

// Rebuild only the spare callback table. Both handoffs join all reads of the
// old table, including callbacks and borrowed coherent wait addresses.
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
    if (!slot->queue || slot->queue == excluded_queue) {
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
  if (membership->join.queue && membership->join.queue != excluded_queue) {
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
  return iree_hal_amd_gpu_context_signal(membership->context);
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
  iree_hal_amd_gpu_queue_membership_t* membership = user_data;
  return iree_atomic_load(&membership->join.completed,
                          iree_memory_order_acquire);
}

static bool iree_hal_amd_gpu_queue_membership_release(void* user_data,
                                                      iree_hal_queue_t* queue) {
  iree_hal_amd_gpu_queue_slot_t* slot = user_data;
  iree_hal_amd_gpu_queue_membership_t* membership = slot->membership;
  iree_slim_mutex_lock(&membership->mutex);
  membership->lifecycle->close(queue);
  membership->join.queue = queue;
  membership->join.status = AMDF_STATUS_OK;
  iree_atomic_store(&membership->join.completed, 0, iree_memory_order_relaxed);
  amdf_status_t status =
      iree_hal_amd_gpu_queue_membership_rebind(membership, NULL);
  if (amdf_status_is_ok(status)) {
    status = iree_hal_amd_gpu_completion_query_status(membership->completion);
  }
  if (amdf_status_is_ok(status)) {
    iree_notification_await(&membership->notification,
                            iree_hal_amd_gpu_queue_membership_joined,
                            membership, iree_infinite_timeout());
    status = membership->join.status;
  }
  // The final source replacement joins the release observer too. Its last
  // notification cannot race reuse of the join record by another release.
  const amdf_status_t removal_status =
      iree_hal_amd_gpu_queue_membership_rebind(membership, queue);
  membership->join.queue = NULL;
  if (amdf_status_is_ok(status)) {
    status = removal_status;
  }
  if (amdf_status_is_ok(status)) {
    status = membership->lifecycle->shutdown(queue);
  }
  if (amdf_status_is_ok(status)) {
    slot->epoch = membership->lifecycle->completed_epoch(queue);
    iree_slim_mutex_lock(&membership->slots_mutex);
    slot->queue = NULL;
    iree_slim_mutex_unlock(&membership->slots_mutex);
  } else {
    membership->context->cleanup_blocked = true;
    iree_hal_amd_gpu_context_report(
        membership->context, status,
        "GPU acquired queue release could not prove native quiescence");
  }
  iree_slim_mutex_unlock(&membership->mutex);
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
    if (slot->queue) {
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
  while (slot && slot->queue) {
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
    status =
        iree_allocator_malloc_aligned(membership->host_allocator, sizeof(*slot),
                                      iree_alignof(*slot), 0, (void**)&slot);
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
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_signal_cache_reserve_frontiers(
        &slot->signals, membership->frontier_count, membership->host_allocator);
  }
  iree_hal_queue_t* queue = NULL;
  if (iree_status_is_ok(status)) {
    status = membership->create(membership->user_data, ordinal, ordinal + 3,
                                membership->frontier_count, slot->epoch,
                                &slot->signals, params, &queue);
  }
  if (iree_status_is_ok(status)) {
    membership->lifecycle->assign_frontier(
        queue, membership->frontier.tracker,
        membership->frontier.base_axis | ((uint64_t)ordinal << 24));
    iree_slim_mutex_lock(&membership->slots_mutex);
    slot->queue = queue;
    iree_slim_mutex_unlock(&membership->slots_mutex);
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        iree_hal_amd_gpu_queue_membership_rebind(membership, NULL),
        "GPU acquired queue source wake");
    membership->lifecycle->set_release(
        queue, iree_hal_amd_gpu_queue_membership_release, slot);
  }
  iree_slim_mutex_unlock(&membership->mutex);
  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else {
    iree_hal_queue_release(queue);
  }
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
  for (iree_hal_amd_gpu_queue_slot_t* slot = membership->slots; slot;
       slot = slot->next) {
    if (slot->queue) {
      membership->lifecycle->notify_failure(slot->queue);
    }
  }
  iree_slim_mutex_unlock(&membership->slots_mutex);
}

amdf_status_t iree_hal_amd_gpu_queue_membership_trim(
    iree_hal_amd_gpu_queue_membership_t* membership) {
  iree_slim_mutex_lock(&membership->mutex);
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0;
       i < membership->provisioned_count && amdf_status_is_ok(status); ++i) {
    if (membership->lifecycle->trim_scratch) {
      status = membership->lifecycle->trim_scratch(membership->provisioned[i]);
    }
    membership->lifecycle->trim(membership->provisioned[i]);
  }
  for (iree_hal_amd_gpu_queue_slot_t* slot = membership->slots; slot;
       slot = slot->next) {
    if (slot->queue) {
      if (membership->lifecycle->trim_scratch && amdf_status_is_ok(status)) {
        status = membership->lifecycle->trim_scratch(slot->queue);
      }
      membership->lifecycle->trim(slot->queue);
    } else {
      iree_hal_amd_gpu_signal_cache_trim(membership->context, 0,
                                         &slot->signals);
    }
  }
  iree_slim_mutex_unlock(&membership->mutex);
  return status;
}
