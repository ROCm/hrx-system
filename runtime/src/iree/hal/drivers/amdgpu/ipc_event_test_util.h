// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_TEST_UTIL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void (*iree_hal_amdgpu_ipc_event_record_callback_hook_fn_t)(
    void* user_data);
typedef void (*iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_fn_t)(
    void* user_data);
typedef void (
    *iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook_fn_t)(
    void* user_data);
typedef void (*iree_hal_amdgpu_ipc_event_import_candidate_hook_fn_t)(
    void* user_data);
typedef void (*iree_hal_amdgpu_ipc_event_ref_decrement_hook_fn_t)(
    void* user_data, int32_t decrement_kind, int32_t old_ref_count);

enum {
  IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_CANDIDATE_DISCARD = 0,
  IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_ATTACH_FAILURE = 1,
  IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_REGISTRY_ANCHOR = 2,
  IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_COUNT = 3,
};

// Installs a deterministic record-timepoint callback gate. Tests must remove
// the hook only after every callback that could observe it has returned.
void iree_hal_amdgpu_ipc_event_set_record_callback_hook_for_testing(
    iree_hal_amdgpu_ipc_event_record_callback_hook_fn_t hook, void* user_data);

// Installs a hook reached after a record timepoint callback has taken ownership
// and cancellation can no longer exclude it. Tests remove the hook only after
// the corresponding abort or shutdown has returned.
void iree_hal_amdgpu_ipc_event_set_record_cancel_miss_hook_for_testing(
    iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_fn_t hook,
    void* user_data);

// Installs a hook before an anchored public release acquires the registry
// mutex. Tests use this to suspend a releaser while its caller reference still
// keeps the carrier alive through a concurrent import.
void iree_hal_amdgpu_ipc_event_set_before_anchored_release_lock_hook_for_testing(
    iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook_fn_t hook,
    void* user_data);

// Installs a hook after an import candidate is allocated but before it retries
// registry lookup. Tests use this to force the candidate-discard race.
void iree_hal_amdgpu_ipc_event_set_import_candidate_hook_for_testing(
    iree_hal_amdgpu_ipc_event_import_candidate_hook_fn_t hook, void* user_data);

// Observes internal carrier decrements that immediately precede direct
// destruction. Used to verify those side effects remain live with NDEBUG.
void iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(
    iree_hal_amdgpu_ipc_event_ref_decrement_hook_fn_t hook, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_TEST_UTIL_H_
