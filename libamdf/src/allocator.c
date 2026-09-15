// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200112L
#endif

#include "libamdf/src/allocator.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

static void* AMDF_CALL amdf_system_allocator_allocate(
    void* user_data, uint64_t byte_length, uint64_t minimum_alignment) {
  (void)user_data;
#if defined(_WIN32)
  return _aligned_malloc((size_t)byte_length, (size_t)minimum_alignment);
#else
  void* pointer = NULL;
  return posix_memalign(&pointer, (size_t)minimum_alignment,
                        (size_t)byte_length) == 0
             ? pointer
             : NULL;
#endif
}

static void* AMDF_CALL amdf_system_allocator_resize(
    void* user_data, void* allocation, uint64_t old_byte_length,
    uint64_t new_byte_length, uint64_t minimum_alignment) {
  (void)user_data;
#if defined(_WIN32)
  (void)old_byte_length;
  return _aligned_realloc(allocation, (size_t)new_byte_length,
                          (size_t)minimum_alignment);
#else
  void* new_allocation =
      amdf_system_allocator_allocate(NULL, new_byte_length, minimum_alignment);
  if (new_allocation == NULL) return NULL;
  memcpy(new_allocation, allocation,
         (size_t)(old_byte_length < new_byte_length ? old_byte_length
                                                    : new_byte_length));
  free(allocation);
  return new_allocation;
#endif
}

static void AMDF_CALL amdf_system_allocator_free(void* user_data,
                                                 void* allocation) {
  (void)user_data;
#if defined(_WIN32)
  _aligned_free(allocation);
#else
  free(allocation);
#endif
}

amdf_allocator_t amdf_allocator_system(void) {
  return (amdf_allocator_t){
      .allocate = amdf_system_allocator_allocate,
      .resize = amdf_system_allocator_resize,
      .free = amdf_system_allocator_free,
  };
}

static bool amdf_allocator_is_zero(amdf_allocator_t allocator) {
  return allocator.user_data == NULL && allocator.allocate == NULL &&
         allocator.resize == NULL && allocator.free == NULL;
}

static bool amdf_allocator_normalize_alignment(size_t minimum_alignment,
                                               size_t* out_alignment) {
  if (minimum_alignment == 0 ||
      (minimum_alignment & (minimum_alignment - 1)) != 0) {
    return false;
  }
  const size_t natural_alignment = amdf_max_align_t;
  *out_alignment = minimum_alignment < natural_alignment ? natural_alignment
                                                         : minimum_alignment;
  return true;
}

amdf_status_t amdf_allocator_resolve(
    const amdf_allocator_t* requested_allocator,
    amdf_allocator_t* out_allocator) {
  if (requested_allocator == NULL || out_allocator == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_allocator_t allocator = *requested_allocator;
  if (amdf_allocator_is_zero(allocator)) {
    allocator = amdf_allocator_system();
  } else if (allocator.allocate == NULL || allocator.free == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  *out_allocator = allocator;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_malloc(amdf_allocator_t allocator, size_t byte_length,
                          size_t minimum_alignment, void** out_pointer) {
  if (out_pointer == NULL || allocator.allocate == NULL || byte_length == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  size_t alignment = 0;
  if (!amdf_allocator_normalize_alignment(minimum_alignment, &alignment)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  void* pointer =
      allocator.allocate(allocator.user_data, byte_length, alignment);
  if (pointer == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  if (((uintptr_t)pointer & (alignment - 1)) != 0) {
    allocator.free(allocator.user_data, pointer);
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  *out_pointer = pointer;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_calloc(amdf_allocator_t allocator, size_t byte_length,
                          size_t minimum_alignment, void** out_pointer) {
  void* pointer = NULL;
  const amdf_status_t status =
      amdf_malloc(allocator, byte_length, minimum_alignment, &pointer);
  if (!amdf_status_is_ok(status)) return status;
  memset(pointer, 0, byte_length);
  *out_pointer = pointer;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_calloc_array(amdf_allocator_t allocator, size_t count,
                                size_t element_size, size_t minimum_alignment,
                                void** out_pointer) {
  if (count == 0 || element_size == 0 || count > SIZE_MAX / element_size) {
    return amdf_make_api_status(count == 0 || element_size == 0
                                    ? AMDF_STATUS_CODE_INVALID_ARGUMENT
                                    : AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_calloc(allocator, count * element_size, minimum_alignment,
                     out_pointer);
}

amdf_status_t amdf_calloc_with_trailing(amdf_allocator_t allocator,
                                        size_t header_byte_length,
                                        size_t trailing_byte_length,
                                        size_t minimum_alignment,
                                        void** out_pointer) {
  if (header_byte_length == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (trailing_byte_length > SIZE_MAX - header_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_calloc(allocator, header_byte_length + trailing_byte_length,
                     minimum_alignment, out_pointer);
}

amdf_status_t amdf_realloc(amdf_allocator_t allocator, size_t old_byte_length,
                           size_t new_byte_length, size_t minimum_alignment,
                           void** inout_pointer) {
  if (inout_pointer == NULL || allocator.allocate == NULL ||
      allocator.free == NULL || new_byte_length == 0 ||
      ((*inout_pointer == NULL) != (old_byte_length == 0))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  size_t alignment = 0;
  if (!amdf_allocator_normalize_alignment(minimum_alignment, &alignment)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (old_byte_length == new_byte_length) return AMDF_STATUS_OK;

  void* new_pointer = NULL;
  if (*inout_pointer == NULL) {
    return amdf_malloc(allocator, new_byte_length, alignment, inout_pointer);
  } else if (allocator.resize != NULL) {
    new_pointer = allocator.resize(allocator.user_data, *inout_pointer,
                                   old_byte_length, new_byte_length, alignment);
    if (new_pointer == NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
  } else {
    amdf_status_t status =
        amdf_malloc(allocator, new_byte_length, alignment, &new_pointer);
    if (!amdf_status_is_ok(status)) return status;
    memcpy(
        new_pointer, *inout_pointer,
        old_byte_length < new_byte_length ? old_byte_length : new_byte_length);
    allocator.free(allocator.user_data, *inout_pointer);
  }
  *inout_pointer = new_pointer;
  return AMDF_STATUS_OK;
}

void amdf_free(amdf_allocator_t allocator, void* pointer) {
  if (pointer != NULL) allocator.free(allocator.user_data, pointer);
}
