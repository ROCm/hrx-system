// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <string.h>

_Static_assert(sizeof(pyre_host_allocator_t) == sizeof(iree_allocator_t),
               "host allocator layout mismatch");
_Static_assert(_Alignof(pyre_host_allocator_t) == _Alignof(iree_allocator_t),
               "host allocator alignment mismatch");

// Initialized before main() via constructor. iree_allocator_system() is
// safe to call at load time (it's an inline returning two constants).
pyre_host_allocator_t pyre_host_allocator_system_value;

__attribute__((constructor))
static void pyre_host_allocator_init(void) {
  iree_allocator_t sys = iree_allocator_system();
  pyre_host_allocator_system_value.self = sys.self;
  pyre_host_allocator_system_value.ctl = (void*)sys.ctl;
}

// Type-pun pyre_host_allocator_t to iree_allocator_t.
static inline iree_allocator_t pyre_to_iree_allocator(
    pyre_host_allocator_t a) {
  iree_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

//===----------------------------------------------------------------------===//
// Standard allocation
//===----------------------------------------------------------------------===//

pyre_status_t pyre_host_allocator_malloc(pyre_host_allocator_t allocator,
                                         size_t byte_length, void** out_ptr) {
  return pyre_status_from_iree(iree_allocator_malloc(
      pyre_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      out_ptr));
}

pyre_status_t pyre_host_allocator_malloc_uninitialized(
    pyre_host_allocator_t allocator, size_t byte_length, void** out_ptr) {
  return pyre_status_from_iree(iree_allocator_malloc_uninitialized(
      pyre_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      out_ptr));
}

pyre_status_t pyre_host_allocator_realloc(pyre_host_allocator_t allocator,
                                          size_t byte_length,
                                          void** inout_ptr) {
  return pyre_status_from_iree(iree_allocator_realloc(
      pyre_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      inout_ptr));
}

pyre_status_t pyre_host_allocator_clone(pyre_host_allocator_t allocator,
                                        const void* src, size_t byte_length,
                                        void** out_ptr) {
  iree_const_byte_span_t span = {
      .data = (const uint8_t*)src,
      .data_length = (iree_host_size_t)byte_length,
  };
  return pyre_status_from_iree(
      iree_allocator_clone(pyre_to_iree_allocator(allocator), span, out_ptr));
}

void pyre_host_allocator_free(pyre_host_allocator_t allocator, void* ptr) {
  iree_allocator_free(pyre_to_iree_allocator(allocator), ptr);
}

//===----------------------------------------------------------------------===//
// Aligned allocation
//===----------------------------------------------------------------------===//

pyre_status_t pyre_host_allocator_malloc_aligned(
    pyre_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** out_ptr) {
  return pyre_status_from_iree(iree_allocator_malloc_aligned(
      pyre_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      (iree_host_size_t)min_alignment, (iree_host_size_t)offset, out_ptr));
}

pyre_status_t pyre_host_allocator_realloc_aligned(
    pyre_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** inout_ptr) {
  return pyre_status_from_iree(iree_allocator_realloc_aligned(
      pyre_to_iree_allocator(allocator), (iree_host_size_t)byte_length,
      (iree_host_size_t)min_alignment, (iree_host_size_t)offset, inout_ptr));
}

void pyre_host_allocator_free_aligned(pyre_host_allocator_t allocator,
                                      void* ptr) {
  iree_allocator_free_aligned(pyre_to_iree_allocator(allocator), ptr);
}
