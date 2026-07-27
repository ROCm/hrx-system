// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <string.h>

#include "hrx_internal.h"

_Static_assert(sizeof(hrx_host_allocator_t) == sizeof(iree_allocator_t),
               "host allocator layout mismatch");
_Static_assert(_Alignof(hrx_host_allocator_t) == _Alignof(iree_allocator_t),
               "host allocator alignment mismatch");

// Type-pun allocator values. Both are two-word structs {self, ctl} with
// identical layout, but HRX keeps the control pointer opaque in the public API.
static inline hrx_host_allocator_t iree_to_hrx_allocator(iree_allocator_t a) {
  hrx_host_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

static inline iree_allocator_t hrx_to_iree_allocator(hrx_host_allocator_t a) {
  iree_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

hrx_host_allocator_t hrx_host_allocator_system(void) {
  return iree_to_hrx_allocator(iree_allocator_system());
}

//===----------------------------------------------------------------------===//
// Standard allocation
//===----------------------------------------------------------------------===//

hrx_status_t hrx_host_allocator_malloc(hrx_host_allocator_t allocator,
                                       size_t byte_length, void** out_ptr) {
  return hrx_status_from_iree(
      iree_allocator_malloc(hrx_to_iree_allocator(allocator),
                            (iree_host_size_t)byte_length, out_ptr));
}

hrx_status_t hrx_host_allocator_malloc_uninitialized(
    hrx_host_allocator_t allocator, size_t byte_length, void** out_ptr) {
  return hrx_status_from_iree(iree_allocator_malloc_uninitialized(
      hrx_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      out_ptr));
}

hrx_status_t hrx_host_allocator_realloc(hrx_host_allocator_t allocator,
                                        size_t byte_length, void** inout_ptr) {
  return hrx_status_from_iree(
      iree_allocator_realloc(hrx_to_iree_allocator(allocator),
                             (iree_host_size_t)byte_length, inout_ptr));
}

hrx_status_t hrx_host_allocator_clone(hrx_host_allocator_t allocator,
                                      const void* src, size_t byte_length,
                                      void** out_ptr) {
  iree_const_byte_span_t span = {
      .data = (const uint8_t*)src,
      .data_length = (iree_host_size_t)byte_length,
  };
  return hrx_status_from_iree(
      iree_allocator_clone(hrx_to_iree_allocator(allocator), span, out_ptr));
}

void hrx_host_allocator_free(hrx_host_allocator_t allocator, void* ptr) {
  iree_allocator_free(hrx_to_iree_allocator(allocator), ptr);
}

//===----------------------------------------------------------------------===//
// Aligned allocation
//===----------------------------------------------------------------------===//

hrx_status_t hrx_host_allocator_malloc_aligned(hrx_host_allocator_t allocator,
                                               size_t byte_length,
                                               size_t min_alignment,
                                               size_t offset, void** out_ptr) {
  return hrx_status_from_iree(iree_allocator_malloc_aligned(
      hrx_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      (iree_host_size_t)min_alignment, (iree_host_size_t)offset, out_ptr));
}

hrx_status_t hrx_host_allocator_realloc_aligned(hrx_host_allocator_t allocator,
                                                size_t byte_length,
                                                size_t min_alignment,
                                                size_t offset,
                                                void** inout_ptr) {
  return hrx_status_from_iree(iree_allocator_realloc_aligned(
      hrx_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      (iree_host_size_t)min_alignment, (iree_host_size_t)offset, inout_ptr));
}

void hrx_host_allocator_free_aligned(hrx_host_allocator_t allocator,
                                     void* ptr) {
  iree_allocator_free_aligned(hrx_to_iree_allocator(allocator), ptr);
}
