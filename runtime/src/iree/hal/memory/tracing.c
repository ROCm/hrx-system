// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tracing.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/debugging.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
// Immutable once published; Tracy may request these strings after every HAL
// owner has gone away. Entries are never removed, so readers need no
// reclamation.
typedef struct iree_hal_memory_trace_name_t {
  // Previously published names.
  struct iree_hal_memory_trace_name_t* next;
  // Length excluding the trailing NUL.
  iree_host_size_t length;
  // Process-owned name storage.
  char value[];
} iree_hal_memory_trace_name_t;

static iree_atomic_intptr_t iree_hal_memory_trace_names =
    IREE_ATOMIC_VAR_INIT(0);

static iree_status_t iree_hal_memory_trace_intern(iree_string_view_t name,
                                                  const char** out_name) {
  if (name.size >
      IREE_HOST_SIZE_MAX - sizeof(iree_hal_memory_trace_name_t) - 1) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "trace memory identifier is too large");
  }
  iree_hal_memory_trace_name_t* candidate = NULL;
  intptr_t head =
      iree_atomic_load(&iree_hal_memory_trace_names, iree_memory_order_acquire);
  for (;;) {
    for (iree_hal_memory_trace_name_t* entry =
             (iree_hal_memory_trace_name_t*)head;
         entry; entry = entry->next) {
      if (entry->length == name.size &&
          memcmp(entry->value, name.data, name.size) == 0) {
        iree_allocator_free(iree_allocator_system(), candidate);
        *out_name = entry->value;
        return iree_ok_status();
      }
    }
    if (!candidate) {
      IREE_LEAK_CHECK_DISABLE_PUSH();
      iree_status_t status = iree_allocator_malloc(
          iree_allocator_system(), sizeof(*candidate) + name.size + 1,
          (void**)&candidate);
      IREE_LEAK_CHECK_DISABLE_POP();
      IREE_RETURN_IF_ERROR(status);
      candidate->length = name.size;
      memcpy(candidate->value, name.data, name.size);
      candidate->value[name.size] = '\0';
    }
    candidate->next = (iree_hal_memory_trace_name_t*)head;
    if (iree_atomic_compare_exchange_strong(
            &iree_hal_memory_trace_names, &head, (intptr_t)candidate,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      *out_name = candidate->value;
      return iree_ok_status();
    }
  }
}
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING

IREE_API_EXPORT iree_status_t iree_hal_memory_trace_initialize(
    iree_string_view_t trace_name, const char* default_memory_id,
    iree_hal_memory_trace_t* out_trace) {
  out_trace->memory_id = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  return iree_hal_memory_trace_intern(
      iree_string_view_is_empty(trace_name)
          ? iree_make_cstring_view(default_memory_id)
          : trace_name,
      &out_trace->memory_id);
#else
  (void)trace_name;
  (void)default_memory_id;
  return iree_ok_status();
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
}

IREE_API_EXPORT iree_status_t iree_hal_memory_trace_initialize_domain(
    const char* domain, uint32_t ordinal, iree_hal_memory_trace_t* out_trace) {
  out_trace->memory_id = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (ordinal == UINT32_MAX) {
    return iree_hal_memory_trace_initialize(iree_string_view_empty(), domain,
                                            out_trace);
  }
  char name[128];
  int length = iree_snprintf(name, sizeof(name), "%s-%u", domain, ordinal);
  if (length < 0 || (iree_host_size_t)length >= sizeof(name)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "trace memory domain name is too large");
  }
  return iree_hal_memory_trace_initialize(
      iree_make_string_view(name, (iree_host_size_t)length), domain, out_trace);
#else
  (void)domain;
  (void)ordinal;
  return iree_ok_status();
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
}

IREE_API_EXPORT iree_status_t iree_hal_memory_trace_initialize_pool(
    iree_string_view_t trace_name, const char* default_memory_id,
    iree_hal_memory_trace_t* out_trace) {
  if (IREE_HAL_MEMORY_TRACE_ENABLE_POOL_STREAMS) {
    return iree_hal_memory_trace_initialize(trace_name, default_memory_id,
                                            out_trace);
  }
  out_trace->memory_id = NULL;
  return iree_ok_status();
}
