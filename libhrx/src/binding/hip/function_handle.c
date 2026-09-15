// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/function_handle.h"

#include <string.h>

#include "binding/hip/handle_registry.h"
#include "iree/base/threading/call_once.h"

typedef struct iree_hip_function_handle_record_t {
  // References held by the registry and active lookups.
  iree_atomic_ref_count_t ref_count;
  // Module retained for the lifetime of this record.
  iree_hal_streaming_module_t* module;
  // Symbol owned by |module|.
  iree_hal_streaming_symbol_t* symbol;
  // Monotonic opaque token exposed through the public ABI.
  uintptr_t handle;
  // Allocator used for this record.
  iree_allocator_t host_allocator;
  // Next record in the module-retirement index.
  struct iree_hip_function_handle_record_t* next;
} iree_hip_function_handle_record_t;

static iree_once_flag iree_hip_function_handle_once = IREE_ONCE_FLAG_INIT;
static iree_hip_handle_registry_t iree_hip_function_handle_registry;
static iree_slim_mutex_t iree_hip_function_handle_mutex;
static iree_hip_function_handle_record_t* iree_hip_function_handle_head;
static uint64_t iree_hip_function_handle_next_id;

static void iree_hip_function_handle_initialize(void) {
  iree_hip_handle_registry_initialize(&iree_hip_function_handle_registry);
  iree_slim_mutex_initialize(&iree_hip_function_handle_mutex);
  iree_hip_function_handle_head = NULL;
  iree_hip_function_handle_next_id = 1;
}

static void iree_hip_function_handle_ensure_initialized(void) {
  iree_call_once(&iree_hip_function_handle_once,
                 iree_hip_function_handle_initialize);
}

static void iree_hip_function_handle_record_retain(uintptr_t value) {
  iree_hip_function_handle_record_t* record =
      (iree_hip_function_handle_record_t*)value;
  iree_atomic_ref_count_inc(&record->ref_count);
}

static void iree_hip_function_handle_record_release(
    iree_hip_function_handle_record_t* record) {
  if (!record) return;
  if (iree_atomic_ref_count_dec(&record->ref_count) != 1) return;
  iree_hal_streaming_module_release(record->module);
  iree_allocator_free(record->host_allocator, record);
}

static iree_status_t iree_hip_function_handle_allocate_token(
    uintptr_t* out_handle) {
#if UINTPTR_MAX > UINT32_MAX
  const uint64_t max_id = UINT64_C(0x0000FFFFFFFFFFFF);
#else
  const uint64_t max_id = UINT32_C(0x7FFFFFFF);
#endif  // UINTPTR_MAX > UINT32_MAX
  if (IREE_UNLIKELY(iree_hip_function_handle_next_id > max_id)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HIP function handle space is exhausted");
  }
  const uintptr_t id = (uintptr_t)iree_hip_function_handle_next_id++;
#if UINTPTR_MAX > UINT32_MAX
  *out_handle = id | IREE_HAL_STREAMING_SYMBOL_TAG_VALUE;
#else
  *out_handle = id | UINT32_C(0x80000000);
#endif  // UINTPTR_MAX > UINT32_MAX
  return iree_ok_status();
}

iree_status_t iree_hip_function_handle_create(
    iree_hal_streaming_module_t* module, iree_hal_streaming_symbol_t* symbol,
    void** out_handle) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;

  iree_hip_function_handle_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(module->host_allocator,
                                             sizeof(*record), (void**)&record));
  memset(record, 0, sizeof(*record));
  iree_atomic_ref_count_init(&record->ref_count);
  record->module = module;
  iree_hal_streaming_module_retain(module);
  record->symbol = symbol;
  record->host_allocator = module->host_allocator;

  iree_hip_function_handle_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_function_handle_mutex);
  iree_status_t status =
      iree_hip_function_handle_allocate_token(&record->handle);
  if (iree_status_is_ok(status)) {
    status = iree_hip_handle_registry_insert_value(
        &iree_hip_function_handle_registry, record->handle, (uintptr_t)record);
  }
  if (iree_status_is_ok(status)) {
    record->next = iree_hip_function_handle_head;
    iree_hip_function_handle_head = record;
    *out_handle = (void*)record->handle;
  }
  iree_slim_mutex_unlock(&iree_hip_function_handle_mutex);

  if (!iree_status_is_ok(status)) {
    iree_hip_function_handle_record_release(record);
  }
  return status;
}

bool iree_hip_function_handle_lookup(const void* handle,
                                     iree_hal_streaming_symbol_t** out_symbol,
                                     iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_symbol);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_symbol = NULL;
  *out_module = NULL;
  if (!handle) return false;

  iree_hip_function_handle_ensure_initialized();
  uintptr_t record_value = 0;
  if (!iree_hip_handle_registry_lookup_retain_value(
          &iree_hip_function_handle_registry, (uintptr_t)handle,
          iree_hip_function_handle_record_retain, &record_value)) {
    return false;
  }

  iree_hip_function_handle_record_t* record =
      (iree_hip_function_handle_record_t*)record_value;
  iree_hal_streaming_module_retain(record->module);
  *out_symbol = record->symbol;
  *out_module = record->module;
  iree_hip_function_handle_record_release(record);
  return true;
}

void iree_hip_function_handle_retire_module(
    iree_hal_streaming_module_t* module) {
  if (!module) return;
  iree_hip_function_handle_ensure_initialized();

  iree_hip_function_handle_record_t* retired_head = NULL;
  iree_slim_mutex_lock(&iree_hip_function_handle_mutex);
  iree_hip_function_handle_record_t** link = &iree_hip_function_handle_head;
  while (*link) {
    iree_hip_function_handle_record_t* record = *link;
    if (record->module != module) {
      link = &record->next;
      continue;
    }
    uintptr_t record_value = 0;
    const bool removed = iree_hip_handle_registry_remove_value(
        &iree_hip_function_handle_registry, record->handle, &record_value);
    IREE_ASSERT(removed && record_value == (uintptr_t)record);
    *link = record->next;
    record->next = retired_head;
    retired_head = record;
  }
  iree_slim_mutex_unlock(&iree_hip_function_handle_mutex);

  while (retired_head) {
    iree_hip_function_handle_record_t* record = retired_head;
    retired_head = record->next;
    iree_hip_function_handle_record_release(record);
  }
}
