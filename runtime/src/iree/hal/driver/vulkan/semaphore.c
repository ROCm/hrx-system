// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/semaphore.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_semaphore_t {
  // Embedded async semaphore at offset 0 for HAL/async toll-free bridging.
  iree_async_semaphore_t async;

  // Device-level Vulkan dispatch table copied from the creating device.
  iree_hal_vulkan_device_syms_t syms;

  // Back-pointer to the logical device that created this semaphore.
  iree_hal_vulkan_logical_device_t* device;

  // Vulkan logical device that owns the semaphore handle.
  VkDevice logical_device;

  // Native Vulkan timeline semaphore handle.
  VkSemaphore handle;

  // Once exposed, native producers may advance the timeline outside the HAL.
  iree_atomic_int32_t handle_exposed;

  // Creation flags controlling synchronization behavior.
  iree_hal_semaphore_flags_t flags;

  // Latest accepted producer, protected by async.mutex. Independent of the
  // optional accumulated frontier's capacity.
  iree_hal_vulkan_semaphore_submission_t submitted;

  // Lazily allocated availability timeline, owned here and protected by
  // async.mutex. It never owns the parent semaphore or any queue submission.
  iree_async_semaphore_t* submission_timeline;
} iree_hal_vulkan_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_vulkan_semaphore_vtable;

static iree_hal_vulkan_semaphore_t* iree_hal_vulkan_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_semaphore_vtable);
  return (iree_hal_vulkan_semaphore_t*)base_value;
}

static iree_status_code_t iree_hal_vulkan_semaphore_failure_code(
    VkResult result) {
  switch (result) {
    default:
      return IREE_STATUS_INTERNAL;
    case VK_ERROR_DEVICE_LOST:
      return IREE_STATUS_DATA_LOSS;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
  }
}

iree_status_t iree_hal_vulkan_semaphore_create(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (initial_value > IREE_HAL_SEMAPHORE_MAX_VALUE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan semaphore initial value %" PRIu64
                            " exceeds the maximum HAL semaphore value %" PRIu64,
                            initial_value,
                            (uint64_t)IREE_HAL_SEMAPHORE_MAX_VALUE);
  }

  VkSemaphoreTypeCreateInfo timeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = initial_value,
  };
  VkSemaphoreCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timeline_create_info,
  };

  VkSemaphore handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vkCreateSemaphore(IREE_VULKAN_DEVICE(syms), logical_device,
                                 &create_info, /*pAllocator=*/NULL, &handle));

  iree_hal_vulkan_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), 0, &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    memset(semaphore, 0, total_size);
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_vulkan_semaphore_vtable,
        proactor, initial_value, frontier_offset, 0, host_allocator,
        &semaphore->async);
    semaphore->syms = *syms;
    semaphore->device = device;
    semaphore->logical_device = logical_device;
    semaphore->handle = handle;
    semaphore->flags = flags;
    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  } else {
    iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(syms), logical_device, handle,
                            /*pAllocator=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->async.allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(&semaphore->syms),
                          semaphore->logical_device, semaphore->handle,
                          /*pAllocator=*/NULL);
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_async_semaphore_release(semaphore->submission_timeline);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_vulkan_semaphore_vtable);
}

bool iree_hal_vulkan_semaphore_is_local(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_vulkan_logical_device_t* device) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_vulkan_semaphore_vtable) &&
         ((const iree_hal_vulkan_semaphore_t*)semaphore)->device == device;
}

iree_hal_semaphore_flags_t iree_hal_vulkan_semaphore_flags(
    iree_hal_semaphore_t* semaphore) {
  return ((const iree_hal_vulkan_semaphore_t*)semaphore)->flags;
}

iree_status_t iree_hal_vulkan_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore, VkSemaphore* out_handle) {
  IREE_ASSERT_ARGUMENT(base_semaphore);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  iree_atomic_store(&semaphore->handle_exposed, 1, iree_memory_order_release);
  *out_handle = semaphore->handle;
  return iree_ok_status();
}

bool iree_hal_vulkan_semaphore_query_submission(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_vulkan_semaphore_submission_t* out_submission) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  *out_submission = semaphore->submitted;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  return out_submission->semaphore != VK_NULL_HANDLE &&
         out_submission->value == value;
}

iree_status_t iree_hal_vulkan_semaphore_submission_timeline(
    iree_hal_semaphore_t* base_semaphore,
    iree_async_semaphore_t** out_timeline) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_status_t status = iree_status_clone((iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire));
  if (iree_status_is_ok(status) && !semaphore->submission_timeline) {
    uint64_t value = (uint64_t)iree_atomic_load(
        &semaphore->async.timeline_value, iree_memory_order_acquire);
    if (semaphore->submitted.semaphore) {
      value = iree_max(value, semaphore->submitted.value);
    }
    status = iree_async_semaphore_create(
        semaphore->async.proactor, value, /*frontier_capacity=*/0,
        semaphore->async.allocator, &semaphore->submission_timeline);
  }
  *out_timeline = semaphore->submission_timeline;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  return status;
}

void iree_hal_vulkan_semaphore_notify_submission(
    iree_hal_semaphore_t* base_semaphore, uint64_t value) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_async_semaphore_t* timeline = semaphore->submission_timeline;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  if (timeline) {
    iree_status_t status =
        iree_async_semaphore_publish_untainted(timeline, value, NULL);
    if (!iree_status_is_ok(status)) {
      iree_async_semaphore_fail(&semaphore->async, status);
    }
  }
}

void iree_hal_vulkan_semaphore_publish_signal(
    iree_hal_semaphore_t* base_semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier,
    VkSemaphore producer_semaphore, uint64_t producer_epoch,
    uint64_t producer_value) {
  IREE_ASSERT_ARGUMENT(producer_frontier);
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_async_frontier_merge(semaphore->async.frontier,
                            semaphore->async.frontier_capacity,
                            producer_frontier);
  if (producer_value > semaphore->submitted.value) {
    semaphore->submitted = (iree_hal_vulkan_semaphore_submission_t){
        .semaphore = producer_semaphore,
        .axis = producer_axis,
        .epoch = producer_epoch,
        .value = producer_value,
    };
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);
}

static uint64_t iree_hal_vulkan_semaphore_observe_native_value(
    iree_async_semaphore_t* base_semaphore, uint64_t value) {
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &base_semaphore->timeline_value, iree_memory_order_acquire);
  if (value <= current_value) {
    return current_value;
  }

  // A native query may observe a host signal before that signal publishes its
  // frontier and host value. Serialize publication with the signaling thread.
  iree_slim_mutex_lock(&base_semaphore->mutex);
  current_value = (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                             iree_memory_order_acquire);
  const bool advanced = value > current_value;
  if (advanced) {
    iree_atomic_store(&base_semaphore->timeline_value, (int64_t)value,
                      iree_memory_order_release);
    current_value = value;
  }
  iree_slim_mutex_unlock(&base_semaphore->mutex);

  if (advanced) {
    iree_async_semaphore_dispatch_timepoints(base_semaphore, value);
  }
  return current_value;
}

static uint64_t iree_hal_vulkan_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));

  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    return iree_hal_status_as_semaphore_failure(failure_status);
  }

  // Until a native caller obtains the handle, HAL publication owns every
  // timeline update. Native queue dependencies use their private queue epochs.
  if (!iree_atomic_load(&semaphore->handle_exposed,
                        iree_memory_order_acquire)) {
    return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                      iree_memory_order_acquire);
  }

  uint64_t value = 0;
  VkResult result = iree_vkGetSemaphoreCounterValue_raw(
      &semaphore->syms, semaphore->logical_device, semaphore->handle, &value);
  if (IREE_UNLIKELY(result != VK_SUCCESS)) {
    return iree_hal_status_as_semaphore_failure(
        iree_status_from_code(iree_hal_vulkan_semaphore_failure_code(result)));
  }

  if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    failure_status = (iree_status_t)iree_atomic_load(
        &base_semaphore->failure_status, iree_memory_order_acquire);
    if (!iree_status_is_ok(failure_status)) {
      return iree_hal_status_as_semaphore_failure(failure_status);
    }
    return IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  }

  return iree_hal_vulkan_semaphore_observe_native_value(base_semaphore, value);
}

static iree_status_t iree_hal_vulkan_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  if (new_value > IREE_HAL_SEMAPHORE_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan semaphore signal value %" PRIu64
                            " exceeds the maximum HAL semaphore value %" PRIu64,
                            new_value, (uint64_t)IREE_HAL_SEMAPHORE_MAX_VALUE);
  }

  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  VkSemaphoreSignalInfo signal_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = semaphore->handle,
      .value = new_value,
  };

  // Native signals must be ordered with the terminal failure signal. Publish
  // the host value only after native success: newly acquired timepoints observe
  // it directly, without waiting for dispatch_timepoints.
  iree_slim_mutex_lock(&base_semaphore->mutex);
  iree_status_t status = iree_status_clone((iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire));
  VkResult result = VK_SUCCESS;
  if (iree_status_is_ok(status)) {
    uint64_t current_value = (uint64_t)iree_atomic_load(
        &base_semaphore->timeline_value, iree_memory_order_acquire);
    if (new_value <= current_value) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "semaphore already signaled past %" PRIu64,
                                new_value);
    } else {
      result = iree_vkSignalSemaphore_raw(
          &semaphore->syms, semaphore->logical_device, &signal_info);
      // Failure publication is independent of the mutex. A failure arriving
      // during the native call takes precedence over this signal's completion.
      status = iree_status_clone((iree_status_t)iree_atomic_load(
          &base_semaphore->failure_status, iree_memory_order_acquire));
      if (result == VK_SUCCESS && iree_status_is_ok(status)) {
        if (frontier != NULL && frontier->entry_count > 0) {
          iree_async_frontier_merge(base_semaphore->frontier,
                                    base_semaphore->frontier_capacity,
                                    frontier);
        }
        iree_atomic_store(&base_semaphore->timeline_value, (int64_t)new_value,
                          iree_memory_order_release);
      }
    }
  }
  iree_slim_mutex_unlock(&base_semaphore->mutex);

  // Failure propagation and callbacks may reenter this semaphore.
  if (result != VK_SUCCESS && iree_status_is_ok(status)) {
    status = iree_status_from_vk_result(__FILE__, __LINE__, result,
                                        "vkSignalSemaphore");
    iree_async_semaphore_fail(base_semaphore, iree_status_clone(status));
  }
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
    iree_hal_vulkan_semaphore_notify_submission(
        iree_hal_semaphore_cast(base_semaphore), new_value);
  }
  return status;
}

static void iree_hal_vulkan_semaphore_on_fail(
    iree_async_semaphore_t* base_semaphore, iree_status_code_t status_code) {
  (void)status_code;
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));

  VkSemaphoreSignalInfo signal_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = semaphore->handle,
      .value = IREE_HAL_SEMAPHORE_FAILURE_VALUE,
  };
  iree_slim_mutex_lock(&base_semaphore->mutex);
  VkResult result = iree_vkSignalSemaphore_raw(
      &semaphore->syms, semaphore->logical_device, &signal_info);
  iree_async_semaphore_t* timeline = semaphore->submission_timeline;
  iree_slim_mutex_unlock(&base_semaphore->mutex);
  if (timeline) {
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &base_semaphore->failure_status, iree_memory_order_acquire);
    iree_async_semaphore_fail(timeline, iree_status_clone(failure));
  }
  IREE_ASSERT(result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST);
  (void)result;
}

static uint64_t iree_hal_vulkan_timeout_to_nanoseconds(iree_timeout_t timeout) {
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    return UINT64_MAX;
  }
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    return 0;
  }
  iree_time_t now_ns = iree_time_now();
  return deadline_ns < now_ns ? 0 : (uint64_t)(deadline_ns - now_ns);
}

static iree_status_t iree_hal_vulkan_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  (void)flags;
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Failure callbacks run before on_fail wakes native waiters. A callback may
  // wait on this semaphore, so the recorded failure must bypass the native
  // wait.
  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_status_clone(failure_status);
  }

  VkSemaphore vk_semaphore = semaphore->handle;
  VkSemaphoreWaitInfo wait_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &vk_semaphore,
      .pValues = &value,
  };
  VkResult result = iree_vkWaitSemaphores_raw(
      &semaphore->syms, semaphore->logical_device, &wait_info,
      iree_hal_vulkan_timeout_to_nanoseconds(timeout));

  iree_status_t status = iree_ok_status();
  failure_status = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    status = iree_status_clone(failure_status);
  } else if (result == VK_SUCCESS) {
    uint64_t current_value = 0;
    result = iree_vkGetSemaphoreCounterValue_raw(
        &semaphore->syms, semaphore->logical_device, semaphore->handle,
        &current_value);
    if (result == VK_SUCCESS &&
        current_value < IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      iree_hal_vulkan_semaphore_observe_native_value(
          (iree_async_semaphore_t*)base_semaphore, current_value);
    } else if (result != VK_SUCCESS) {
      status = iree_status_from_vk_result(__FILE__, __LINE__, result,
                                          "vkGetSemaphoreCounterValue");
    }
  } else {
    status = iree_status_from_vk_result(__FILE__, __LINE__, result,
                                        "vkWaitSemaphores");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_semaphore_vtable_t iree_hal_vulkan_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_vulkan_semaphore_destroy,
            .query = iree_hal_vulkan_semaphore_query,
            .signal = iree_hal_vulkan_semaphore_signal,
            .on_fail = iree_hal_vulkan_semaphore_on_fail,
        },
    .wait = iree_hal_vulkan_semaphore_wait,
};
