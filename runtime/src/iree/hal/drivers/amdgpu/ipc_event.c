// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/ipc_event.h"

#include <string.h>

#include "iree/async/semaphore.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amdgpu/ipc_event_monitor.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"

static_assert(sizeof(iree_hal_amdgpu_ipc_event_token_t) ==
                  sizeof(hsa_amd_ipc_signal_t),
              "IPC event token must match the ROCr signal token size");

static iree_status_t iree_hal_amdgpu_ipc_event_create_for_logical_device(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_ipc_event_t** out_event);

static iree_status_t iree_hal_amdgpu_ipc_event_import_for_logical_device(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_ipc_event_token_t token,
    iree_hal_amdgpu_ipc_event_t** out_event);

static iree_status_t iree_hal_amdgpu_ipc_event_wait_reserve_for_logical_device(
    iree_hal_amdgpu_logical_device_t* destination_device,
    iree_hal_amdgpu_ipc_event_wait_point_t* out_wait_point,
    iree_hal_amdgpu_ipc_event_wait_t** out_wait);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_ipc_event_t
//===----------------------------------------------------------------------===//

typedef enum iree_hal_amdgpu_ipc_event_registry_state_e {
  IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_UNMANAGED = 0,
  IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_ATTACHING,
  IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_READY,
  IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_DETACHING,
} iree_hal_amdgpu_ipc_event_registry_state_t;

struct iree_hal_amdgpu_ipc_event_t {
  // Intrusive reference count owning this carrier.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for the carrier and its monitor operation state.
  iree_allocator_t host_allocator;
  // Logical device retained to keep |libhsa| and its HSA runtime live.
  iree_hal_device_t* device;
  // HSA dispatch table borrowed from the retained logical device.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Locally created or attached process-local handle to the IPC signal.
  hsa_signal_t signal;
  // Next entry in the process-wide carrier registry.
  struct iree_hal_amdgpu_ipc_event_t* registry_next;
  // Stable process-independent token represented by this carrier.
  iree_hal_amdgpu_ipc_event_token_t token;
  // State protected by the carrier registry mutex.
  iree_hal_amdgpu_ipc_event_registry_state_t registry_state;
  // Immutable after publication; true when the registry owns one anchor
  // reference in addition to caller and operation references.
  bool has_registry_anchor;
  // Serializes process-local record and pending-wait preparation through
  // commit or abort. Cross-process event reuse requires external ordering.
  iree_slim_mutex_t generation_mutex;
  // True after at least one local record operation has been committed.
  bool has_local_record;
  // True when shutdown abandoned a pending local record generation.
  iree_atomic_int32_t local_record_abandoned;
  // Number of committed local rearms after the first record. Waits use this to
  // recognize a prior completion hidden by a fast process-local rerecord.
  iree_atomic_int64_t local_rearm_count;
};

// Indexes the one live process-local carrier for each token.
//
// Locally created owners are registered directly. Duplicate imports, including
// a source process reopening a token re-exported by a foreign process, retain
// that owner instead of attaching a second signal with independent generation
// state. An imported token with no live local owner attaches exactly once.
//
// Each linked carrier's initial reference is a registry anchor. Attach and
// detach execute outside |mutex| while a transition state prevents concurrent
// imports from starting a second ROCr operation for the token. This also keeps
// a final release from holding the registry mutex across HSA. The anchor is
// dropped only after DETACHING has completed and the entry is unlinked, so the
// registry creates no device-retention cycle.
// Deliberately process-lifetime: it spans the per-ordinal AMDGPU systems.
typedef struct iree_hal_amdgpu_ipc_event_registry_t {
  // Guards carrier lookup and registry state transitions.
  iree_mutex_t mutex;
  // Wakes imports waiting for an attach or detach transition to complete.
  iree_notification_t changed;
  // Intrusive list of carriers keyed by their process-wide token.
  iree_hal_amdgpu_ipc_event_t* events;
} iree_hal_amdgpu_ipc_event_registry_t;

static iree_once_flag iree_hal_amdgpu_ipc_event_registry_once =
    IREE_ONCE_FLAG_INIT;
static iree_hal_amdgpu_ipc_event_registry_t iree_hal_amdgpu_ipc_event_registry;

static void iree_hal_amdgpu_ipc_event_registry_initialize(void) {
  iree_mutex_initialize(&iree_hal_amdgpu_ipc_event_registry.mutex);
  iree_notification_initialize(&iree_hal_amdgpu_ipc_event_registry.changed);
}

static iree_hal_amdgpu_ipc_event_registry_t*
iree_hal_amdgpu_ipc_event_registry_get(void) {
  iree_call_once(&iree_hal_amdgpu_ipc_event_registry_once,
                 iree_hal_amdgpu_ipc_event_registry_initialize);
  return &iree_hal_amdgpu_ipc_event_registry;
}

struct iree_hal_amdgpu_ipc_event_record_t {
  // Common monitor operation; must be the first field.
  iree_hal_amdgpu_ipc_event_monitor_operation_t monitor_operation;
  // Cancellable producer-semaphore completion registration.
  iree_async_semaphore_timepoint_t timepoint;
  // Host allocator used to free this operation.
  iree_allocator_t host_allocator;
  // Carrier retained until the monitor completes or cancels the operation.
  iree_hal_amdgpu_ipc_event_t* event;
  // Ordinary HAL semaphore retained until its accepted timepoint completes.
  iree_hal_semaphore_t* recorded_semaphore;
  // Value on |recorded_semaphore| accepted by the producer queue.
  uint64_t recorded_value;
};

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
};

// Test-only callback gate. Tests install and remove it while no record callback
// can begin, so these borrowed values require no production synchronization.
static iree_hal_amdgpu_ipc_event_record_callback_hook_fn_t
    iree_hal_amdgpu_ipc_event_record_callback_hook;
static void* iree_hal_amdgpu_ipc_event_record_callback_hook_user_data;
static iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_fn_t
    iree_hal_amdgpu_ipc_event_record_cancel_miss_hook;
static void* iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_user_data;
static iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook_fn_t
    iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook;
static void*
    iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook_user_data;
static iree_hal_amdgpu_ipc_event_import_candidate_hook_fn_t
    iree_hal_amdgpu_ipc_event_import_candidate_hook;
static void* iree_hal_amdgpu_ipc_event_import_candidate_hook_user_data;
static iree_hal_amdgpu_ipc_event_ref_decrement_hook_fn_t
    iree_hal_amdgpu_ipc_event_ref_decrement_hook;
static void* iree_hal_amdgpu_ipc_event_ref_decrement_hook_user_data;

void iree_hal_amdgpu_ipc_event_set_record_callback_hook_for_testing(
    iree_hal_amdgpu_ipc_event_record_callback_hook_fn_t hook, void* user_data) {
  iree_hal_amdgpu_ipc_event_record_callback_hook_user_data = user_data;
  iree_hal_amdgpu_ipc_event_record_callback_hook = hook;
}

void iree_hal_amdgpu_ipc_event_set_record_cancel_miss_hook_for_testing(
    iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_fn_t hook,
    void* user_data) {
  iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_user_data = user_data;
  iree_hal_amdgpu_ipc_event_record_cancel_miss_hook = hook;
}

void iree_hal_amdgpu_ipc_event_set_before_anchored_release_lock_hook_for_testing(
    iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook_fn_t hook,
    void* user_data) {
  iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook_user_data =
      user_data;
  iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook = hook;
}

void iree_hal_amdgpu_ipc_event_set_import_candidate_hook_for_testing(
    iree_hal_amdgpu_ipc_event_import_candidate_hook_fn_t hook,
    void* user_data) {
  iree_hal_amdgpu_ipc_event_import_candidate_hook_user_data = user_data;
  iree_hal_amdgpu_ipc_event_import_candidate_hook = hook;
}

void iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(
    iree_hal_amdgpu_ipc_event_ref_decrement_hook_fn_t hook, void* user_data) {
  iree_hal_amdgpu_ipc_event_ref_decrement_hook_user_data = user_data;
  iree_hal_amdgpu_ipc_event_ref_decrement_hook = hook;
}

static void iree_hal_amdgpu_ipc_event_notify_ref_decrement_for_testing(
    int32_t decrement_kind, int32_t old_ref_count) {
  if (iree_hal_amdgpu_ipc_event_ref_decrement_hook) {
    iree_hal_amdgpu_ipc_event_ref_decrement_hook(
        iree_hal_amdgpu_ipc_event_ref_decrement_hook_user_data, decrement_kind,
        old_ref_count);
  }
}

static iree_status_t iree_hal_amdgpu_ipc_event_create_empty(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_ipc_event_t** out_event) {
  *out_event = NULL;
  const iree_allocator_t host_allocator = logical_device->host_allocator;
  iree_hal_amdgpu_ipc_event_t* event = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event));

  iree_atomic_ref_count_init(&event->ref_count);
  event->host_allocator = host_allocator;
  event->device = (iree_hal_device_t*)logical_device;
  iree_hal_device_retain(event->device);
  event->libhsa = &logical_device->system->libhsa;
  event->signal = (hsa_signal_t){0};
  event->registry_next = NULL;
  memset(&event->token, 0, sizeof(event->token));
  event->registry_state = IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_UNMANAGED;
  event->has_registry_anchor = false;
  iree_slim_mutex_initialize(&event->generation_mutex);
  event->has_local_record = false;
  iree_atomic_store(&event->local_record_abandoned, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&event->local_rearm_count, 0, iree_memory_order_relaxed);
  *out_event = event;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_ipc_event_adopt_signal(
    iree_hal_amdgpu_logical_device_t* logical_device, hsa_signal_t signal,
    iree_hal_amdgpu_ipc_event_t** out_event) {
  iree_status_t status =
      iree_hal_amdgpu_ipc_event_create_empty(logical_device, out_event);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_signal_destroy_raw(&logical_device->system->libhsa, signal));
    return status;
  }
  (*out_event)->signal = signal;
  return iree_ok_status();
}

static void iree_hal_amdgpu_ipc_event_destroy(
    iree_hal_amdgpu_ipc_event_t* event, bool destroy_signal) {
  iree_hal_device_t* device = event->device;
  const iree_allocator_t host_allocator = event->host_allocator;
  iree_slim_mutex_deinitialize(&event->generation_mutex);
  if (destroy_signal) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_signal_destroy_raw(event->libhsa, event->signal));
  }
  iree_allocator_free(host_allocator, event);
  iree_hal_device_release(device);
}

// Adds a newly created owner as the canonical carrier for its token. The
// owner's allocation begins with the caller reference; registration adds the
// anchor that keeps the entry valid through concurrent lookup.
static iree_status_t iree_hal_amdgpu_ipc_event_register_owner(
    iree_hal_amdgpu_ipc_event_t* event) {
  iree_hal_amdgpu_ipc_event_registry_t* registry =
      iree_hal_amdgpu_ipc_event_registry_get();
  iree_mutex_lock(&registry->mutex);
  iree_hal_amdgpu_ipc_event_t* existing = registry->events;
  while (existing &&
         memcmp(&existing->token, &event->token, sizeof(event->token)) != 0) {
    existing = existing->registry_next;
  }
  if (IREE_UNLIKELY(existing)) {
    iree_mutex_unlock(&registry->mutex);
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "IPC event token is already registered");
  }

  event->has_registry_anchor = true;
  event->registry_state = IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_READY;
  event->registry_next = registry->events;
  registry->events = event;
  iree_atomic_ref_count_inc(&event->ref_count);
  iree_mutex_unlock(&registry->mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_ipc_event_create_for_logical_device(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_ipc_event_t** out_event) {
  const iree_hal_amdgpu_libhsa_t* libhsa = &logical_device->system->libhsa;
  hsa_signal_t signal = {0};
  IREE_RETURN_IF_ERROR(iree_hsa_amd_signal_create(
      IREE_LIBHSA(libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, HSA_AMD_SIGNAL_IPC, &signal));
  iree_hal_amdgpu_ipc_event_t* event = NULL;
  iree_status_t status =
      iree_hal_amdgpu_ipc_event_adopt_signal(logical_device, signal, &event);
  if (!iree_status_is_ok(status)) return status;

  hsa_amd_ipc_signal_t hsa_token;
  status =
      iree_hsa_amd_ipc_signal_create(IREE_LIBHSA(libhsa), signal, &hsa_token);
  if (iree_status_is_ok(status)) {
    memcpy(event->token.data, &hsa_token, sizeof(hsa_token));
    status = iree_hal_amdgpu_ipc_event_register_owner(event);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_ipc_event_release(event);
    return status;
  }
  *out_event = event;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_ipc_event_create(
    iree_hal_device_t* device, iree_hal_amdgpu_ipc_event_t** out_event) {
  if (IREE_UNLIKELY(!out_event)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_event must be non-NULL");
  }
  *out_event = NULL;
  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_logical_device_cast_checked(device, &logical_device));
  return iree_hal_amdgpu_ipc_event_create_for_logical_device(logical_device,
                                                             out_event);
}

static iree_status_t iree_hal_amdgpu_ipc_event_import_for_logical_device(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_ipc_event_token_t token,
    iree_hal_amdgpu_ipc_event_t** out_event) {
  *out_event = NULL;
  iree_hal_amdgpu_ipc_event_registry_t* registry =
      iree_hal_amdgpu_ipc_event_registry_get();
  iree_hal_amdgpu_ipc_event_t* candidate = NULL;

  for (;;) {
    iree_mutex_lock(&registry->mutex);
    iree_hal_amdgpu_ipc_event_t* existing = registry->events;
    while (existing && memcmp(&existing->token, &token, sizeof(token)) != 0) {
      existing = existing->registry_next;
    }
    if (existing && existing->registry_state ==
                        IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_READY) {
      iree_atomic_ref_count_inc(&existing->ref_count);
      *out_event = existing;
      iree_mutex_unlock(&registry->mutex);
      if (candidate) {
        const int32_t old_candidate_ref_count =
            iree_atomic_ref_count_dec(&candidate->ref_count);
        iree_hal_amdgpu_ipc_event_notify_ref_decrement_for_testing(
            IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_CANDIDATE_DISCARD,
            old_candidate_ref_count);
        IREE_ASSERT_EQ(old_candidate_ref_count, 1);
        iree_hal_amdgpu_ipc_event_destroy(candidate, /*destroy_signal=*/false);
      }
      return iree_ok_status();
    }
    if (existing) {
      // Keep the token entry linked while its HSA operation runs. Preparing the
      // wait under the mutex prevents a missed transition notification.
      iree_wait_token_t wait_token =
          iree_notification_prepare_wait(&registry->changed);
      iree_mutex_unlock(&registry->mutex);
      iree_notification_commit_wait(&registry->changed, wait_token,
                                    IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
      continue;
    }
    if (!candidate) {
      iree_mutex_unlock(&registry->mutex);
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_ipc_event_create_empty(logical_device, &candidate));
      candidate->token = token;
      if (iree_hal_amdgpu_ipc_event_import_candidate_hook) {
        iree_hal_amdgpu_ipc_event_import_candidate_hook(
            iree_hal_amdgpu_ipc_event_import_candidate_hook_user_data);
      }
      continue;
    }

    candidate->registry_state =
        IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_ATTACHING;
    candidate->has_registry_anchor = true;
    candidate->registry_next = registry->events;
    registry->events = candidate;
    iree_mutex_unlock(&registry->mutex);

    hsa_amd_ipc_signal_t hsa_token;
    memcpy(&hsa_token, token.data, sizeof(hsa_token));
    hsa_signal_t signal = {0};
    iree_status_t status = iree_hsa_amd_ipc_signal_attach(
        IREE_LIBHSA(candidate->libhsa), &hsa_token, &signal);

    iree_mutex_lock(&registry->mutex);
    IREE_ASSERT(candidate->registry_state ==
                IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_ATTACHING);
    if (iree_status_is_ok(status)) {
      candidate->signal = signal;
      candidate->registry_state =
          IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_READY;
      iree_atomic_ref_count_inc(&candidate->ref_count);
      *out_event = candidate;
    } else {
      iree_hal_amdgpu_ipc_event_t** current = &registry->events;
      while (*current && *current != candidate) {
        current = &(*current)->registry_next;
      }
      IREE_ASSERT(*current == candidate);
      if (*current == candidate) *current = candidate->registry_next;
      candidate->registry_next = NULL;
      candidate->registry_state =
          IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_UNMANAGED;
      candidate->has_registry_anchor = false;
    }
    iree_mutex_unlock(&registry->mutex);
    iree_notification_post(&registry->changed, IREE_ALL_WAITERS);

    if (!iree_status_is_ok(status)) {
      const int32_t old_candidate_ref_count =
          iree_atomic_ref_count_dec(&candidate->ref_count);
      iree_hal_amdgpu_ipc_event_notify_ref_decrement_for_testing(
          IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_ATTACH_FAILURE,
          old_candidate_ref_count);
      IREE_ASSERT_EQ(old_candidate_ref_count, 1);
      iree_hal_amdgpu_ipc_event_destroy(candidate, /*destroy_signal=*/false);
    }
    return status;
  }
}

iree_status_t iree_hal_amdgpu_ipc_event_import(
    iree_hal_device_t* device, iree_hal_amdgpu_ipc_event_token_t token,
    iree_hal_amdgpu_ipc_event_t** out_event) {
  if (IREE_UNLIKELY(!out_event)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_event must be non-NULL");
  }
  *out_event = NULL;
  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_logical_device_cast_checked(device, &logical_device));
  return iree_hal_amdgpu_ipc_event_import_for_logical_device(logical_device,
                                                             token, out_event);
}

void iree_hal_amdgpu_ipc_event_retain(iree_hal_amdgpu_ipc_event_t* event) {
  if (event) {
    iree_atomic_ref_count_inc(&event->ref_count);
  }
}

void iree_hal_amdgpu_ipc_event_release(iree_hal_amdgpu_ipc_event_t* event) {
  if (!event) return;
  if (!event->has_registry_anchor) {
    const int32_t old_ref_count = iree_atomic_ref_count_dec(&event->ref_count);
    if (old_ref_count == 1) {
      iree_hal_amdgpu_ipc_event_destroy(event, /*destroy_signal=*/true);
    }
    return;
  }

  iree_hal_amdgpu_ipc_event_registry_t* registry =
      iree_hal_amdgpu_ipc_event_registry_get();
  if (iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook) {
    iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook(
        iree_hal_amdgpu_ipc_event_before_anchored_release_lock_hook_user_data);
  }

  // Import retains and anchored release decrements linearize under the same
  // mutex while this caller's reference still keeps |event| alive. A 2 -> 1
  // transition therefore owns detachment before another import can retain the
  // registry anchor as though it were a caller reference.
  iree_mutex_lock(&registry->mutex);
  const int32_t old_ref_count = iree_atomic_ref_count_dec(&event->ref_count);
  IREE_ASSERT_GT(old_ref_count, 1);
  if (old_ref_count != 2) {
    iree_mutex_unlock(&registry->mutex);
    return;
  }
  IREE_ASSERT(event->registry_state ==
              IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_READY);
  event->registry_state = IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_DETACHING;
  iree_mutex_unlock(&registry->mutex);

  // Do not hold the registry mutex across HSA. An HSA entry point must not
  // block the registry state transitions used by another HSA caller.
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_signal_destroy_raw(event->libhsa, event->signal));

  iree_mutex_lock(&registry->mutex);
  IREE_ASSERT(event->registry_state ==
              IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_DETACHING);
  iree_hal_amdgpu_ipc_event_t** current = &registry->events;
  while (*current && *current != event) current = &(*current)->registry_next;
  IREE_ASSERT(*current == event,
              "detaching IPC event must retain its registry anchor");
  if (*current == event) *current = event->registry_next;
  event->registry_next = NULL;
  event->registry_state = IREE_HAL_AMDGPU_IPC_EVENT_REGISTRY_STATE_UNMANAGED;
  iree_mutex_unlock(&registry->mutex);
  iree_notification_post(&registry->changed, IREE_ALL_WAITERS);

  const int32_t old_anchor_ref_count =
      iree_atomic_ref_count_dec(&event->ref_count);
  iree_hal_amdgpu_ipc_event_notify_ref_decrement_for_testing(
      IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_REGISTRY_ANCHOR,
      old_anchor_ref_count);
  IREE_ASSERT_EQ(old_anchor_ref_count, 1);
  iree_hal_amdgpu_ipc_event_destroy(event, /*destroy_signal=*/false);
}

static bool iree_hal_amdgpu_ipc_event_local_record_was_abandoned(
    iree_hal_amdgpu_ipc_event_t* event) {
  return iree_atomic_load(&event->local_record_abandoned,
                          iree_memory_order_acquire) != 0;
}

static iree_status_t iree_hal_amdgpu_ipc_event_abandoned_status(void) {
  return iree_make_status(
      IREE_STATUS_ABORTED,
      "IPC event local record generation was abandoned during shutdown");
}

iree_status_t iree_hal_amdgpu_ipc_event_export(
    iree_hal_amdgpu_ipc_event_t* event,
    iree_hal_amdgpu_ipc_event_token_t* out_token) {
  if (IREE_UNLIKELY(!event || !out_token)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event and out_token must be non-NULL");
  }
  if (IREE_UNLIKELY(
          iree_hal_amdgpu_ipc_event_local_record_was_abandoned(event))) {
    return iree_hal_amdgpu_ipc_event_abandoned_status();
  }
  *out_token = event->token;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_ipc_event_query(
    iree_hal_amdgpu_ipc_event_t* event, bool* out_reached) {
  if (IREE_UNLIKELY(!event || !out_reached)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event and out_reached must be non-NULL");
  }
  *out_reached = false;

  // The binary transport deliberately carries completion but no producer
  // status. This acquire load never waits behind process-local record or wait
  // serialization, preserving the nonblocking query contract.
  const hsa_signal_value_t signal_value =
      iree_hsa_signal_load_scacquire(IREE_LIBHSA(event->libhsa), event->signal);
  *out_reached = signal_value < 1;
  if (!*out_reached &&
      IREE_UNLIKELY(
          iree_hal_amdgpu_ipc_event_local_record_was_abandoned(event))) {
    return iree_hal_amdgpu_ipc_event_abandoned_status();
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_ipc_event_wait(
    iree_hal_amdgpu_ipc_event_t* event) {
  if (IREE_UNLIKELY(!event)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event must be non-NULL");
  }

  // Bind a pending signal to an asynchronous proxy, or return immediately when
  // arm observes a completed generation.
  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {0};
  iree_hal_amdgpu_ipc_event_wait_t* wait = NULL;
  iree_status_t status =
      iree_hal_amdgpu_ipc_event_wait_reserve_for_logical_device(
          (iree_hal_amdgpu_logical_device_t*)event->device, &wait_point, &wait);
  if (!iree_status_is_ok(status)) return status;
  const bool is_armed = iree_hal_amdgpu_ipc_event_wait_arm(event, wait);
  status = iree_hal_amdgpu_ipc_event_wait_commit(wait);
  if (iree_status_is_ok(status) && is_armed) {
    status = iree_hal_semaphore_wait(wait_point.semaphore, wait_point.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(wait_point.semaphore);
  return status;
}

static void iree_hal_amdgpu_ipc_event_record_destroy(
    iree_hal_amdgpu_ipc_event_record_t* record, bool cancel_admission) {
  const iree_allocator_t host_allocator = record->host_allocator;
  iree_hal_semaphore_release(record->recorded_semaphore);
  iree_hal_amdgpu_ipc_event_release(record->event);
  if (cancel_admission) {
    // Retire admission only after every backend-lifetime reference is gone so
    // a concurrent shutdown cannot return ahead of this abort.
    iree_hal_amdgpu_ipc_event_monitor_operation_cancel(
        &record->monitor_operation);
  }
  iree_allocator_free(host_allocator, record);
}

static void iree_hal_amdgpu_ipc_event_record_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amdgpu_ipc_event_record_t* record =
      (iree_hal_amdgpu_ipc_event_record_t*)user_data;
  (void)timepoint;
  iree_status_free(status);

  if (iree_hal_amdgpu_ipc_event_record_callback_hook) {
    iree_hal_amdgpu_ipc_event_record_callback_hook(
        iree_hal_amdgpu_ipc_event_record_callback_hook_user_data);
  }

  // Final callback access. The monitor may release the record immediately
  // after acquiring this request, so nothing may touch |record| afterward.
  iree_hal_amdgpu_ipc_event_monitor_operation_request_poll(
      &record->monitor_operation);
}

static bool iree_hal_amdgpu_ipc_event_record_poll(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested) {
  iree_hal_amdgpu_ipc_event_record_t* record =
      (iree_hal_amdgpu_ipc_event_record_t*)operation;
  if (was_requested) {
    // Producer completion and failure share binary transport completion: the
    // external signal has no cross-process error channel.
    iree_hsa_signal_store_screlease(IREE_LIBHSA(record->event->libhsa),
                                    record->event->signal, 0);
    iree_hal_amdgpu_ipc_event_record_destroy(record, false);
    return true;
  }
  if (!is_shutting_down) return false;

  if (iree_async_semaphore_cancel_timepoint(
          (iree_async_semaphore_t*)record->recorded_semaphore,
          &record->timepoint)) {
    // Cancellation excludes the callback but can race the semaphore's release
    // publication before timepoint dispatch. Reconcile the acquire-visible
    // terminal state so shutdown cannot discard a completion or failure that
    // already happened before this query.
    uint64_t current_value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(record->recorded_semaphore, &current_value);
    const bool is_terminal = !iree_status_is_ok(query_status) ||
                             current_value >= record->recorded_value;
    iree_status_free(query_status);
    if (is_terminal) {
      // Producer completion and failure share binary transport completion: the
      // external signal has no cross-process error channel.
      iree_hsa_signal_store_screlease(IREE_LIBHSA(record->event->libhsa),
                                      record->event->signal, 0);
    } else {
      // Accepted producer work is still pending, so completion cannot be
      // manufactured safely. Persist terminal local state before releasing the
      // operation so a later monitor generation cannot wait on its cancelled
      // timepoint forever.
      iree_atomic_store(&record->event->local_record_abandoned, 1,
                        iree_memory_order_release);
    }
    // If the acquire query is still pending, accepted producer work may still
    // be executing. Publishing 0 would expose its output before completion.
    iree_hal_amdgpu_ipc_event_record_destroy(record, false);
    return true;
  }

  // The callback was detached for dispatch and owns the timepoint storage. It
  // performs no teardown and will request another poll as its final access.
  if (iree_hal_amdgpu_ipc_event_record_cancel_miss_hook) {
    iree_hal_amdgpu_ipc_event_record_cancel_miss_hook(
        iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_user_data);
  }
  return false;
}

// Cancels the timepoint owned by an unpublished record. A false cancellation
// result means its callback was already detached; that callback's final atomic
// request returns ownership without performing teardown.
static void iree_hal_amdgpu_ipc_event_record_cancel_timepoint(
    iree_hal_amdgpu_ipc_event_record_t* record) {
  if (!iree_async_semaphore_cancel_timepoint(
          (iree_async_semaphore_t*)record->recorded_semaphore,
          &record->timepoint)) {
    if (iree_hal_amdgpu_ipc_event_record_cancel_miss_hook) {
      iree_hal_amdgpu_ipc_event_record_cancel_miss_hook(
          iree_hal_amdgpu_ipc_event_record_cancel_miss_hook_user_data);
    }
    const bool was_requested =
        iree_hal_amdgpu_ipc_event_monitor_operation_await_poll_request(
            &record->monitor_operation, iree_infinite_timeout());
    IREE_ASSERT(was_requested);
    (void)was_requested;
  }
}

iree_status_t iree_hal_amdgpu_ipc_event_record_prepare(
    iree_hal_amdgpu_ipc_event_t* event,
    iree_hal_semaphore_t* recorded_semaphore, uint64_t recorded_value,
    iree_hal_amdgpu_ipc_event_record_t** out_record) {
  if (IREE_UNLIKELY(!out_record)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_record must be non-NULL");
  }
  *out_record = NULL;
  if (IREE_UNLIKELY(!event || !recorded_semaphore)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event and recorded_semaphore must be non-NULL");
  }

  // Allocate and retain everything commit needs before taking the record lock
  // and before the caller offers work to a queue.
  iree_hal_amdgpu_ipc_event_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(event->host_allocator,
                                             sizeof(*record), (void**)&record));
  iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
      iree_hal_amdgpu_ipc_event_record_poll, IREE_DURATION_INFINITE,
      &record->monitor_operation);
  iree_status_t status = iree_hal_amdgpu_ipc_event_monitor_operation_admit(
      &record->monitor_operation);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(event->host_allocator, record);
    return status;
  }
  record->host_allocator = event->host_allocator;
  record->event = event;
  iree_hal_amdgpu_ipc_event_retain(event);
  record->recorded_semaphore = recorded_semaphore;
  iree_hal_semaphore_retain(recorded_semaphore);
  record->recorded_value = recorded_value;
  record->timepoint.callback =
      iree_hal_amdgpu_ipc_event_record_timepoint_callback;
  record->timepoint.user_data = record;
  status = iree_async_semaphore_acquire_timepoint(
      (iree_async_semaphore_t*)recorded_semaphore, recorded_value,
      &record->timepoint);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_ipc_event_record_destroy(record, true);
    return status;
  }

  // Hold this lock across the caller's queue enqueue. No other local record can
  // observe or rearm the carrier until this one commits or aborts.
  iree_slim_mutex_lock(&event->generation_mutex);
  if (IREE_UNLIKELY(
          iree_hal_amdgpu_ipc_event_local_record_was_abandoned(event))) {
    iree_slim_mutex_unlock(&event->generation_mutex);
    iree_hal_amdgpu_ipc_event_record_cancel_timepoint(record);
    iree_hal_amdgpu_ipc_event_record_destroy(record, true);
    return iree_hal_amdgpu_ipc_event_abandoned_status();
  }
  if (event->has_local_record) {
    hsa_signal_value_t prior_value = 1;
    bool cancelled = false;
    bool abandoned = false;
    while (prior_value >= 1 && !cancelled && !abandoned) {
      prior_value = iree_hsa_signal_load_scacquire(IREE_LIBHSA(event->libhsa),
                                                   event->signal);
      if (prior_value < 1) break;
      abandoned = iree_hal_amdgpu_ipc_event_local_record_was_abandoned(event);
      if (abandoned) break;
      if (iree_hal_amdgpu_ipc_event_monitor_await_shutdown(
              iree_make_timeout_ms(1))) {
        // Prefer an observable completion that raced shutdown. Only abandon
        // this admission when the prior generation is still genuinely pending.
        prior_value = iree_hsa_signal_load_scacquire(IREE_LIBHSA(event->libhsa),
                                                     event->signal);
        if (prior_value >= 1) {
          abandoned =
              iree_hal_amdgpu_ipc_event_local_record_was_abandoned(event);
          cancelled = !abandoned;
        }
      }
    }
    if (cancelled || abandoned) {
      iree_slim_mutex_unlock(&event->generation_mutex);
      iree_hal_amdgpu_ipc_event_record_cancel_timepoint(record);
      iree_hal_amdgpu_ipc_event_record_destroy(record, true);
      if (abandoned) {
        return iree_hal_amdgpu_ipc_event_abandoned_status();
      }
      return iree_make_status(
          IREE_STATUS_CANCELLED,
          "IPC event monitor shut down while waiting to rerecord");
    }
  }

  *out_record = record;
  return iree_ok_status();
}

void iree_hal_amdgpu_ipc_event_record_commit(
    iree_hal_amdgpu_ipc_event_record_t* record) {
  if (IREE_UNLIKELY(!record)) return;

  iree_hal_amdgpu_ipc_event_t* event = record->event;
  if (event->has_local_record) {
    // The prior completion may be hidden by the following rearm before a local
    // wait's monitor poll. Release-publish that completed generation first.
    iree_atomic_fetch_add(&event->local_rearm_count, 1,
                          iree_memory_order_release);
  }
  event->has_local_record = true;

  iree_hsa_signal_store_screlease(IREE_LIBHSA(event->libhsa), event->signal, 1);
  iree_slim_mutex_unlock(&event->generation_mutex);
  // Final transaction access: publication transfers ownership to the worker.
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(
      &record->monitor_operation);
}

void iree_hal_amdgpu_ipc_event_record_abort(
    iree_hal_amdgpu_ipc_event_record_t* record) {
  if (IREE_UNLIKELY(!record)) return;
  iree_hal_amdgpu_ipc_event_t* event = record->event;
  iree_slim_mutex_unlock(&event->generation_mutex);
  iree_hal_amdgpu_ipc_event_record_cancel_timepoint(record);
  iree_hal_amdgpu_ipc_event_record_destroy(record, true);
}

//===----------------------------------------------------------------------===//
// External wait bridge
//===----------------------------------------------------------------------===//

struct iree_hal_amdgpu_ipc_event_wait_t {
  // Common monitor operation; must be the first field.
  iree_hal_amdgpu_ipc_event_monitor_operation_t monitor_operation;
  // Host allocator used to free this state.
  iree_allocator_t host_allocator;
  // Carrier retained from arm through monitor completion. NULL when arm found
  // no pending generation requiring a proxy.
  iree_hal_amdgpu_ipc_event_t* event;
  // Destination device retained through monitor completion.
  iree_hal_device_t* destination_device;
  // Destination semaphore retained through monitor completion.
  iree_hal_semaphore_t* semaphore;
  // Local rearm count captured while selecting the pending generation.
  int64_t observed_rearm_count;
};

static void iree_hal_amdgpu_ipc_event_wait_destroy(
    iree_hal_amdgpu_ipc_event_wait_t* wait, bool cancel_admission) {
  const iree_allocator_t host_allocator = wait->host_allocator;
  iree_hal_amdgpu_ipc_event_release(wait->event);
  iree_hal_device_release(wait->destination_device);
  iree_hal_semaphore_release(wait->semaphore);
  if (cancel_admission) {
    // Retire admission only after every backend-lifetime reference is gone so
    // a concurrent shutdown cannot return ahead of this abort.
    iree_hal_amdgpu_ipc_event_monitor_operation_cancel(
        &wait->monitor_operation);
  }
  iree_allocator_free(host_allocator, wait);
}

static bool iree_hal_amdgpu_ipc_event_wait_poll(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested) {
  iree_hal_amdgpu_ipc_event_wait_t* wait =
      (iree_hal_amdgpu_ipc_event_wait_t*)operation;
  (void)was_requested;
  // Read the transport first. A new local rearm release-stores its generation
  // count before release-storing signal 1, so either acquire observes the
  // prior 0 directly or the counter proves that completion was hidden.
  const hsa_signal_value_t observed_value = iree_hsa_signal_load_scacquire(
      IREE_LIBHSA(wait->event->libhsa), wait->event->signal);
  const int64_t current_rearm_count = iree_atomic_load(
      &wait->event->local_rearm_count, iree_memory_order_acquire);
  const bool is_pending =
      observed_value >= 1 && current_rearm_count == wait->observed_rearm_count;
  const bool was_abandoned =
      is_pending &&
      iree_hal_amdgpu_ipc_event_local_record_was_abandoned(wait->event);
  if (is_pending && !was_abandoned && !is_shutting_down) return false;

  iree_status_t resolution_status = iree_ok_status();
  if (was_abandoned) {
    resolution_status = iree_hal_amdgpu_ipc_event_abandoned_status();
  } else if (is_pending) {
    resolution_status =
        iree_make_status(IREE_STATUS_CANCELLED, "IPC event monitor shut down");
  }

  // Resolve the accepted proxy before any release capable of synchronously
  // entering destination-device or carrier teardown.
  if (iree_status_is_ok(resolution_status)) {
    resolution_status =
        iree_hal_semaphore_signal(wait->semaphore, 1, /*frontier=*/NULL);
  }
  if (!iree_status_is_ok(resolution_status)) {
    iree_hal_semaphore_fail(wait->semaphore, resolution_status);
  }
  iree_hal_amdgpu_ipc_event_wait_destroy(wait, false);
  return true;
}

static iree_status_t iree_hal_amdgpu_ipc_event_wait_reserve_for_logical_device(
    iree_hal_amdgpu_logical_device_t* destination_device,
    iree_hal_amdgpu_ipc_event_wait_point_t* out_wait_point,
    iree_hal_amdgpu_ipc_event_wait_t** out_wait) {
  out_wait_point->semaphore = NULL;
  out_wait_point->value = 0;
  *out_wait = NULL;

  iree_hal_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_hal_amdgpu_semaphore_create(
      destination_device, destination_device->proactor,
      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, destination_device->host_allocator,
      &semaphore);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_hal_amdgpu_ipc_event_wait_t* wait = NULL;
  status = iree_allocator_malloc(destination_device->host_allocator,
                                 sizeof(*wait), (void**)&wait);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_release(semaphore);
    return status;
  }
  iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
      iree_hal_amdgpu_ipc_event_wait_poll,
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_EXTERNAL_MAX_POLL_DELAY_NS,
      &wait->monitor_operation);
  status = iree_hal_amdgpu_ipc_event_monitor_operation_admit(
      &wait->monitor_operation);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(destination_device->host_allocator, wait);
    iree_hal_semaphore_release(semaphore);
    return status;
  }
  wait->host_allocator = destination_device->host_allocator;
  wait->event = NULL;
  wait->destination_device = (iree_hal_device_t*)destination_device;
  iree_hal_device_retain(wait->destination_device);
  wait->semaphore = semaphore;
  iree_hal_semaphore_retain(semaphore);
  wait->observed_rearm_count = 0;

  out_wait_point->semaphore = semaphore;
  out_wait_point->value = 1;
  *out_wait = wait;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_ipc_event_wait_reserve(
    iree_hal_device_t* destination_device,
    iree_hal_amdgpu_ipc_event_wait_point_t* out_wait_point,
    iree_hal_amdgpu_ipc_event_wait_t** out_wait) {
  if (IREE_UNLIKELY(!out_wait_point || !out_wait)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_wait_point and out_wait must be non-NULL");
  }
  out_wait_point->semaphore = NULL;
  out_wait_point->value = 0;
  *out_wait = NULL;
  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_cast_checked(
      destination_device, &logical_device));
  return iree_hal_amdgpu_ipc_event_wait_reserve_for_logical_device(
      logical_device, out_wait_point, out_wait);
}

bool iree_hal_amdgpu_ipc_event_wait_arm(
    iree_hal_amdgpu_ipc_event_t* event,
    iree_hal_amdgpu_ipc_event_wait_t* wait) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(wait);
  IREE_ASSERT(!wait->event);

  // Taking the acquire load under the process-local generation lock makes a
  // later local rerecord order after this decision; no proxy is needed for the
  // already-complete generation.
  iree_slim_mutex_lock(&event->generation_mutex);
  const hsa_signal_value_t signal_value =
      iree_hsa_signal_load_scacquire(IREE_LIBHSA(event->libhsa), event->signal);
  if (IREE_LIKELY(signal_value < 1)) {
    iree_slim_mutex_unlock(&event->generation_mutex);
    return false;
  }

  // Hold a pending process-local generation stable until activation or
  // abandonment so a local rerecord cannot rearm between selection and
  // registration.
  wait->event = event;
  iree_hal_amdgpu_ipc_event_retain(event);
  wait->observed_rearm_count =
      iree_atomic_load(&event->local_rearm_count, iree_memory_order_relaxed);
  return true;
}

iree_status_t iree_hal_amdgpu_ipc_event_wait_commit(
    iree_hal_amdgpu_ipc_event_wait_t* wait) {
  if (IREE_UNLIKELY(!wait)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wait must be non-NULL");
  }
  if (!wait->event) {
    // The source adapter had no carrier or its generation was already complete
    // when this reserved wait was armed. Its proxy was omitted from the
    // accepted queue operation.
    iree_hal_amdgpu_ipc_event_wait_destroy(wait, true);
    return iree_ok_status();
  }

  iree_hal_amdgpu_ipc_event_t* event = wait->event;
  iree_slim_mutex_unlock(&event->generation_mutex);
  // Final transaction access: publication transfers ownership to the worker.
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&wait->monitor_operation);
  return iree_ok_status();
}

void iree_hal_amdgpu_ipc_event_wait_abort(
    iree_hal_amdgpu_ipc_event_wait_t* wait) {
  if (wait) {
    if (wait->event) {
      iree_slim_mutex_unlock(&wait->event->generation_mutex);
    }
    iree_hal_amdgpu_ipc_event_wait_destroy(wait, true);
  }
}
