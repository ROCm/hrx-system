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

#include "iree/base/internal/math.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
#include <errno.h>
#include <sched.h>
#endif

iree_host_size_t iree_thread_affinity_set_find_first_set(
    const iree_thread_affinity_set_t* set, iree_host_size_t processor_offset) {
  if (processor_offset >= IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT) {
    return IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT;
  }
  iree_host_size_t word_index = processor_offset / 64;
  uint64_t word =
      set->words[word_index] & (UINT64_MAX << (processor_offset % 64));
  for (; word_index < IREE_ARRAYSIZE(set->words); ++word_index) {
    if (word) {
      return word_index * 64 + iree_math_count_trailing_zeros_u64(word);
    }
    if (word_index + 1 < IREE_ARRAYSIZE(set->words)) {
      word = set->words[word_index + 1];
    }
  }
  return IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT;
}

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
    iree_thread_affinity_set_t* out_processor_set) {
  memset(out_processor_set, 0, sizeof(*out_processor_set));
#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
  const iree_host_size_t native_size =
      CPU_ALLOC_SIZE(IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  cpu_set_t* native_set = iree_alloca(native_size);
  CPU_ZERO_S(native_size, native_set);
  if (sched_getaffinity(0, native_size, native_set) != 0) {
    const int error = errno;
    if (error == EINVAL) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "kernel processor affinity exceeds "
          "IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT (%u CPU IDs)",
          (unsigned)IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "cannot query current processor affinity: %s "
                            "(errno %d)",
                            strerror(error), error);
  }
  iree_bitmap_t processor_mask =
      iree_thread_affinity_set_bitmap(out_processor_set);
  // cpu_set_t is opaque and may use a different word size than our bitmap.
  for (iree_host_size_t processor = 0; processor < processor_mask.bit_count;
       ++processor) {
    if (CPU_ISSET_S(processor, native_size, native_set)) {
      iree_bitmap_set(processor_mask, processor);
    }
  }
  return iree_ok_status();
#else
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "effective processor affinity capture is not "
                          "supported on this platform");
#endif
}
