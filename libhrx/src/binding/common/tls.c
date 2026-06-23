// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/tls.h"

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/internal/atomics.h"

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !defined(IREE_PLATFORM_WINDOWS)
#include <pthread.h>
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !IREE_PLATFORM_WINDOWS

typedef enum iree_hal_streaming_tls_slot_state_e {
  IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY = 0,
  IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING = 1,
  IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED = 2,
} iree_hal_streaming_tls_slot_state_t;

typedef struct iree_hal_streaming_tls_slot_t {
  // Allocation state for this process-global slot.
  iree_atomic_int32_t state;
  // Destructor registered for values stored in this slot.
  iree_hal_streaming_tls_destructor_t destructor;
#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !defined(IREE_PLATFORM_WINDOWS)
  // Native pthread key backing this slot.
  pthread_key_t pthread_key;
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !IREE_PLATFORM_WINDOWS
} iree_hal_streaming_tls_slot_t;

static iree_hal_streaming_tls_slot_t
    iree_hal_streaming_tls_slots[IREE_HAL_STREAMING_TLS_KEY_CAPACITY] = {{0}};

static bool iree_hal_streaming_tls_slot_is_allocated(
    iree_hal_streaming_tls_key_t key) {
  return key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY &&
         iree_atomic_load(&iree_hal_streaming_tls_slots[key].state,
                          iree_memory_order_acquire) ==
             IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED;
}

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

static void*
    iree_hal_streaming_tls_values[IREE_HAL_STREAMING_TLS_KEY_CAPACITY] = {0};

IREE_API_EXPORT iree_status_t iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  for (iree_hal_streaming_tls_key_t key = 0;
       key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++key) {
    int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY;
    if (!iree_atomic_compare_exchange_strong(
            &iree_hal_streaming_tls_slots[key].state, &expected_state,
            IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    iree_hal_streaming_tls_slots[key].destructor = destructor;
    iree_hal_streaming_tls_values[key] = NULL;
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    *out_key = key;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "binding TLS key capacity exhausted");
}

IREE_API_EXPORT void iree_hal_streaming_tls_key_delete(
    iree_hal_streaming_tls_key_t key) {
  if (!iree_hal_streaming_tls_slot_is_allocated(key)) return;
  iree_hal_streaming_tls_values[key] = NULL;
  iree_hal_streaming_tls_slots[key].destructor = NULL;
  iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                    IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                    iree_memory_order_release);
}

IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key) {
  return iree_hal_streaming_tls_slot_is_allocated(key)
             ? iree_hal_streaming_tls_values[key]
             : NULL;
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value) {
  if (IREE_UNLIKELY(!iree_hal_streaming_tls_slot_is_allocated(key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid binding TLS key");
  }
  iree_hal_streaming_tls_values[key] = value;
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_WINDOWS)

static __declspec(thread) void*
    iree_hal_streaming_tls_values[IREE_HAL_STREAMING_TLS_KEY_CAPACITY] = {0};

static void iree_hal_streaming_tls_cleanup_current_thread(void) {
  for (int iteration = 0; iteration < 4; ++iteration) {
    bool invoked_destructor = false;
    for (iree_hal_streaming_tls_key_t key = 0;
         key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++key) {
      void* value = iree_hal_streaming_tls_values[key];
      if (!value) continue;
      if (!iree_hal_streaming_tls_slot_is_allocated(key)) {
        iree_hal_streaming_tls_values[key] = NULL;
        continue;
      }
      iree_hal_streaming_tls_destructor_t destructor =
          iree_hal_streaming_tls_slots[key].destructor;
      if (!destructor) continue;
      iree_hal_streaming_tls_values[key] = NULL;
      destructor(value);
      invoked_destructor = true;
    }
    if (!invoked_destructor) break;
  }
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
  (void)instance;
  (void)reserved;
  switch (reason) {
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
      iree_hal_streaming_tls_cleanup_current_thread();
      break;
    default:
      break;
  }
  return TRUE;
}

IREE_API_EXPORT iree_status_t iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  for (iree_hal_streaming_tls_key_t key = 0;
       key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++key) {
    int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY;
    if (!iree_atomic_compare_exchange_strong(
            &iree_hal_streaming_tls_slots[key].state, &expected_state,
            IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    iree_hal_streaming_tls_slots[key].destructor = destructor;
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    *out_key = key;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "binding TLS key capacity exhausted");
}

IREE_API_EXPORT void iree_hal_streaming_tls_key_delete(
    iree_hal_streaming_tls_key_t key) {
  if (!iree_hal_streaming_tls_slot_is_allocated(key)) return;
  int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED;
  if (!iree_atomic_compare_exchange_strong(
          &iree_hal_streaming_tls_slots[key].state, &expected_state,
          IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return;
  }
  iree_hal_streaming_tls_values[key] = NULL;
  iree_hal_streaming_tls_slots[key].destructor = NULL;
  iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                    IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                    iree_memory_order_release);
}

IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key) {
  return iree_hal_streaming_tls_slot_is_allocated(key)
             ? iree_hal_streaming_tls_values[key]
             : NULL;
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value) {
  if (IREE_UNLIKELY(!iree_hal_streaming_tls_slot_is_allocated(key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid binding TLS key");
  }
  iree_hal_streaming_tls_values[key] = value;
  return iree_ok_status();
}

#else

IREE_API_EXPORT iree_status_t iree_hal_streaming_tls_key_create(
    iree_hal_streaming_tls_key_t* out_key,
    iree_hal_streaming_tls_destructor_t destructor) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  for (iree_hal_streaming_tls_key_t key = 0;
       key < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++key) {
    int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY;
    if (!iree_atomic_compare_exchange_strong(
            &iree_hal_streaming_tls_slots[key].state, &expected_state,
            IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    int result =
        pthread_key_create(&iree_hal_streaming_tls_slots[key].pthread_key,
                           (void (*)(void*))destructor);
    if (result != 0) {
      iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                        IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                        iree_memory_order_release);
      return iree_make_status(iree_status_code_from_errno(result),
                              "pthread_key_create failed: %d", result);
    }
    iree_hal_streaming_tls_slots[key].destructor = destructor;
    iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                      IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED,
                      iree_memory_order_release);
    *out_key = key;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "binding TLS key capacity exhausted");
}

IREE_API_EXPORT void iree_hal_streaming_tls_key_delete(
    iree_hal_streaming_tls_key_t key) {
  if (!iree_hal_streaming_tls_slot_is_allocated(key)) return;
  int32_t expected_state = IREE_HAL_STREAMING_TLS_SLOT_STATE_ALLOCATED;
  if (!iree_atomic_compare_exchange_strong(
          &iree_hal_streaming_tls_slots[key].state, &expected_state,
          IREE_HAL_STREAMING_TLS_SLOT_STATE_INITIALIZING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return;
  }
  pthread_key_delete(iree_hal_streaming_tls_slots[key].pthread_key);
  iree_hal_streaming_tls_slots[key].destructor = NULL;
  iree_atomic_store(&iree_hal_streaming_tls_slots[key].state,
                    IREE_HAL_STREAMING_TLS_SLOT_STATE_EMPTY,
                    iree_memory_order_release);
}

IREE_API_EXPORT void* iree_hal_streaming_tls_get(
    iree_hal_streaming_tls_key_t key) {
  return iree_hal_streaming_tls_slot_is_allocated(key)
             ? pthread_getspecific(
                   iree_hal_streaming_tls_slots[key].pthread_key)
             : NULL;
}

IREE_API_EXPORT iree_status_t
iree_hal_streaming_tls_set(iree_hal_streaming_tls_key_t key, void* value) {
  if (IREE_UNLIKELY(!iree_hal_streaming_tls_slot_is_allocated(key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid binding TLS key");
  }
  int result =
      pthread_setspecific(iree_hal_streaming_tls_slots[key].pthread_key, value);
  if (IREE_UNLIKELY(result != 0)) {
    return iree_make_status(iree_status_code_from_errno(result),
                            "pthread_setspecific failed: %d", result);
  }
  return iree_ok_status();
}

#endif  // IREE_SYNCHRONIZATION_DISABLE_UNSAFE
