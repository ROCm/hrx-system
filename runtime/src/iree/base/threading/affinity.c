// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "iree/base/threading/affinity.h"

#include <string.h>

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
#include <errno.h>
#include <sched.h>
#endif

void iree_thread_affinity_set_any(iree_thread_affinity_t* out_thread_affinity) {
  memset(out_thread_affinity, 0x00, sizeof(*out_thread_affinity));
}

void iree_thread_affinity_set_group_any(
    uint32_t group, iree_thread_affinity_t* out_thread_affinity) {
  memset(out_thread_affinity, 0x00, sizeof(*out_thread_affinity));
  out_thread_affinity->group_any = 1;
  out_thread_affinity->group = group;
}

iree_status_t iree_thread_affinity_query_current(
    iree_allocator_t allocator, iree_bitmap_t* out_processor_mask) {
  *out_processor_mask = (iree_bitmap_t){0};
#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
  cpu_set_t inline_set;
  cpu_set_t* native_set = &inline_set;
  iree_host_size_t native_size = sizeof(inline_set);
  iree_status_t status = iree_ok_status();
  bool complete = false;
  while (!complete && iree_status_is_ok(status)) {
    CPU_ZERO_S(native_size, native_set);
    if (sched_getaffinity(0, native_size, native_set) == 0) {
      complete = true;
    } else if (errno != EINVAL) {
      int error = errno;
      status = iree_make_status(iree_status_code_from_errno(error),
                                "cannot query current processor affinity: %s "
                                "(errno %d)",
                                strerror(error), error);
    } else {
      // EINVAL means the supplied mask cannot represent the kernel CPU IDs.
      // Grow storage instead of dropping the application's affinity
      // restriction.
      if (native_size > SIZE_MAX / 16) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "processor affinity exceeds addressable "
                                  "bitmap storage");
      } else {
        if (native_set != &inline_set) {
          iree_allocator_free(allocator, native_set);
        }
        native_set = NULL;
        native_size *= 2;
        status =
            iree_allocator_malloc(allocator, native_size, (void**)&native_set);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t bit_count = native_size * 8;
    uint64_t* words = NULL;
    status = iree_allocator_malloc_array(allocator,
                                         iree_bitmap_calculate_words(bit_count),
                                         sizeof(*words), (void**)&words);
    if (iree_status_is_ok(status)) {
      iree_bitmap_t processor_mask = {.bit_count = bit_count, .words = words};
      // cpu_set_t is opaque and may use a different word size than our bitmap.
      for (iree_host_size_t processor = 0; processor < bit_count; ++processor) {
        if (CPU_ISSET_S(processor, native_size, native_set)) {
          iree_bitmap_set(processor_mask, processor);
        }
      }
      *out_processor_mask = processor_mask;
    }
  }
  if (native_set != &inline_set) {
    iree_allocator_free(allocator, native_set);
  }
  return status;
#else
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "effective processor affinity capture is not "
                          "supported on this platform");
#endif
}
