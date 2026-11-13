// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/failure.h"

#include <inttypes.h>

void iree_hal_amd_gpu_failure_initialize(
    const amdf_endpoint_info_t* endpoint, iree_hal_device_t* device,
    iree_hal_device_event_sink_t event_sink, uint32_t timeout_ms,
    iree_hal_amd_gpu_failure_t* failure) {
  *failure = (iree_hal_amd_gpu_failure_t){.endpoint = endpoint,
                                          .event_sink = event_sink,
                                          .device = device,
                                          .timeout_ms = timeout_ms};
  iree_notification_initialize(&failure->changed);
}

void iree_hal_amd_gpu_failure_deinitialize(
    iree_hal_amd_gpu_failure_t* failure) {
  if (!failure->endpoint) {
    return;
  }
  iree_notification_deinitialize(&failure->changed);
  failure->endpoint = NULL;
}

bool iree_hal_amd_gpu_failure_begin(iree_hal_amd_gpu_failure_t* failure,
                                    amdf_status_t status, const char* operation,
                                    iree_time_t now) {
  const bool first =
      !iree_atomic_load(&failure->status, iree_memory_order_relaxed);
  if (first) {
    failure->operation = operation;
    failure->started = now;
    const iree_duration_t budget =
        (iree_duration_t)failure->timeout_ms * 1000000;
    failure->deadline = now < IREE_TIME_INFINITE_FUTURE - budget
                            ? now + budget
                            : IREE_TIME_INFINITE_FUTURE;
    iree_hal_amd_gpu_failure_set_phase(
        failure, IREE_HAL_AMD_GPU_FAILURE_CLOSING, UINT32_MAX, 0, 0);
    iree_atomic_store(&failure->status, status, iree_memory_order_release);
  }
  return first;
}

void iree_hal_amd_gpu_failure_set_phase(iree_hal_amd_gpu_failure_t* failure,
                                        iree_hal_amd_gpu_failure_phase_t phase,
                                        uint32_t queue_ordinal,
                                        uint32_t completed_count,
                                        uint32_t total_count) {
  iree_atomic_store(&failure->completed_count, completed_count,
                    iree_memory_order_relaxed);
  iree_atomic_store(&failure->total_count, total_count,
                    iree_memory_order_relaxed);
  iree_atomic_store(&failure->position,
                    ((uint64_t)queue_ordinal << 32) | (uint32_t)phase,
                    iree_memory_order_release);
}

iree_time_t iree_hal_amd_gpu_failure_poll(iree_hal_amd_gpu_failure_t* failure,
                                          iree_time_t now) {
  const amdf_status_t status = (amdf_status_t)iree_atomic_load(
      &failure->status, iree_memory_order_acquire);
  const uint64_t position =
      (uint64_t)iree_atomic_load(&failure->position, iree_memory_order_acquire);
  if (!status || (uint32_t)position == IREE_HAL_AMD_GPU_FAILURE_FINISHED ||
      iree_atomic_load(&failure->reported, iree_memory_order_acquire)) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  if (now < failure->deadline) {
    return failure->deadline;
  }
  int32_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(&failure->reported, &expected, 1,
                                           iree_memory_order_acq_rel,
                                           iree_memory_order_relaxed)) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  static const char* const phases[] = {"closing", "inactivating", "retiring",
                                       "joining callbacks", "finished"};
  const amdf_endpoint_info_t* endpoint = failure->endpoint;
  const uint32_t ordinal = (uint32_t)(position >> 32);
  const char* owner = ordinal < 256               ? "compute"
                      : ordinal == UINT32_MAX - 1 ? "transfer"
                      : (uint32_t)position == IREE_HAL_AMD_GPU_FAILURE_CLOSING
                          ? "device"
                          : "code";
  char message[768];
  iree_snprintf(message, sizeof(message),
                "GPU terminal cleanup exceeded %u ms (%" PRId64
                " ms elapsed) for %s (%016" PRIx64 "%016" PRIx64
                "): phase=%s queue=%u progress=%u/%u owner=%s original=%s "
                "status=0x%016" PRIx64
                "; unproven native/callback ownership remains retained",
                failure->timeout_ms,
                (int64_t)((now - failure->started) / 1000000), endpoint->name,
                endpoint->id.words[0], endpoint->id.words[1],
                phases[(uint32_t)position], ordinal,
                (uint32_t)iree_atomic_load(&failure->completed_count,
                                           iree_memory_order_relaxed),
                (uint32_t)iree_atomic_load(&failure->total_count,
                                           iree_memory_order_relaxed),
                owner, failure->operation, status);
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = IREE_STATUS_DEADLINE_EXCEEDED,
      .backend_result_code = status,
      .message = iree_make_cstring_view(message),
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.device = failure->device;
  event.source.driver_id = IREE_SV("amd");
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(failure->event_sink, &event);
  return IREE_TIME_INFINITE_FUTURE;
}

void iree_hal_amd_gpu_failure_finish(iree_hal_amd_gpu_failure_t* failure) {
  iree_atomic_store(&failure->position, IREE_HAL_AMD_GPU_FAILURE_FINISHED,
                    iree_memory_order_release);
  iree_notification_post(&failure->changed, IREE_ALL_WAITERS);
}

bool iree_hal_amd_gpu_failure_wake(iree_hal_amd_gpu_failure_t* failure) {
  if (!iree_atomic_load(&failure->status, iree_memory_order_acquire)) {
    return false;
  }
  iree_notification_post(&failure->changed, IREE_ALL_WAITERS);
  return true;
}
