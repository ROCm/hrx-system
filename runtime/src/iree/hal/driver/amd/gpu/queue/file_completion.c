// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/file_completion.h"

#include "iree/hal/driver/amd/status.h"

enum {
  IREE_HAL_AMD_GPU_FILE_COMPLETION_ACTIVE = 1 << 0,
  IREE_HAL_AMD_GPU_FILE_COMPLETION_RETIRED = 1 << 1,
};

static void iree_hal_amd_gpu_file_completion_deliver(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_amd_gpu_file_completion_t* completion = user_data;
  void (*notify)(void*, iree_status_t, bool) = completion->notify;
  void* notify_data = completion->user_data;
  const amdf_status_t native_status = (amdf_status_t)iree_atomic_load(
      &completion->status, iree_memory_order_acquire);
  status = iree_status_join(
      IREE_HAL_AMD_STATUS_FROM_AMDF(native_status, "SDMA file copy"), status);
  if (iree_atomic_load(&completion->state, iree_memory_order_acquire) &
      IREE_HAL_AMD_GPU_FILE_COMPLETION_RETIRED) {
    iree_atomic_fetch_and(&completion->state,
                          ~IREE_HAL_AMD_GPU_FILE_COMPLETION_ACTIVE,
                          iree_memory_order_release);
    notify(notify_data, status, true);
    return;
  }
  notify(notify_data, iree_status_clone(status), false);
  // A final native notification either coalesced with this callback or claims
  // a new NOP after this exchange. No completion access follows the handoff.
  const int32_t previous = iree_atomic_fetch_and(
      &completion->state, ~IREE_HAL_AMD_GPU_FILE_COMPLETION_ACTIVE,
      iree_memory_order_acq_rel);
  if (previous & IREE_HAL_AMD_GPU_FILE_COMPLETION_RETIRED) {
    notify(notify_data, status, true);
  } else {
    iree_status_free(status);
  }
}

void iree_hal_amd_gpu_file_completion_initialize(
    iree_async_proactor_t* proactor, void (*notify)(void*, iree_status_t, bool),
    void* user_data, iree_hal_amd_gpu_file_completion_t* completion) {
  completion->proactor = proactor;
  completion->notify = notify;
  completion->user_data = user_data;
  iree_atomic_store(&completion->status, 0, iree_memory_order_relaxed);
  iree_atomic_store(&completion->state, 0, iree_memory_order_relaxed);
}

void iree_hal_amd_gpu_file_completion_notify(
    iree_hal_amd_gpu_file_completion_t* completion, amdf_status_t status,
    bool retired) {
  int64_t expected = 0;
  iree_atomic_compare_exchange_strong(&completion->status, &expected, status,
                                      iree_memory_order_release,
                                      iree_memory_order_relaxed);
  const int32_t previous = iree_atomic_fetch_or(
      &completion->state,
      IREE_HAL_AMD_GPU_FILE_COMPLETION_ACTIVE |
          (retired ? IREE_HAL_AMD_GPU_FILE_COMPLETION_RETIRED : 0),
      iree_memory_order_acq_rel);
  if (previous & IREE_HAL_AMD_GPU_FILE_COMPLETION_ACTIVE) {
    return;
  }
  iree_async_proactor_t* proactor = completion->proactor;
  iree_async_operation_initialize(
      &completion->publication.base, IREE_ASYNC_OPERATION_TYPE_NOP, 0,
      iree_hal_amd_gpu_file_completion_deliver, completion);
  // Standalone NOP uses caller-owned storage; live proactor admission cannot
  // fail for native capacity or allocation. Final callback may reclaim storage.
  IREE_CHECK_OK(
      iree_async_proactor_submit_one(proactor, &completion->publication.base));
}
