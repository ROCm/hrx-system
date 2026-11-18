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

#include "iree/base/internal/sysfs.h"
#elif defined(IREE_PLATFORM_WINDOWS)
// clang-format off
#include <windows.h>
// clang-format on
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

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

typedef struct iree_thread_cpu_set_builder_t {
  // Native mask populated by the sysfs CPU-list parser.
  cpu_set_t* set;
  // Native mask capacity in bytes.
  iree_host_size_t size;
  // Optional application budget bounding the requested node's processors.
  const iree_thread_affinity_set_t* processor_set;
  // Whether the node contains CPUs outside the native mask representation.
  bool overflow;
} iree_thread_cpu_set_builder_t;

static bool iree_thread_cpu_set_add_range(uint32_t start_cpu, uint32_t end_cpu,
                                          void* user_data) {
  iree_thread_cpu_set_builder_t* builder = user_data;
  if (builder->processor_set) {
    end_cpu = iree_min(end_cpu, IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  } else if (end_cpu > builder->size * 8) {
    builder->overflow = true;
    return false;
  }
  for (uint32_t cpu = start_cpu; cpu < end_cpu; ++cpu) {
    if (!builder->processor_set ||
        iree_thread_affinity_set_test(builder->processor_set, cpu)) {
      CPU_SET_S(cpu, builder->size, builder->set);
    }
  }
  return true;
}

// Uses the shared sysfs CPU-list parser so NUMA masks have the same numbering
// as topology discovery. An absent or unrepresentable node is an error.
static iree_status_t iree_thread_make_cpu_set_from_node_id(
    uint32_t node_id, const iree_thread_affinity_set_t* processor_set,
    iree_host_size_t native_size, cpu_set_t* out_set) {
  char path[128];
  iree_snprintf(path, sizeof(path), "%s/node/node%u/cpulist",
                iree_sysfs_get_root_path(), node_id);
  char buffer[4096];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length));
  iree_thread_cpu_set_builder_t builder = {
      .set = out_set,
      .size = native_size,
      .processor_set = processor_set,
  };
  IREE_RETURN_IF_ERROR(
      iree_sysfs_parse_cpu_list(iree_make_string_view(buffer, length),
                                iree_thread_cpu_set_add_range, &builder));
  if (builder.overflow) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node %u CPU mask exceeds %u CPUs", node_id,
                            (unsigned)(native_size * 8));
  }
  return iree_ok_status();
}

iree_status_t iree_thread_affinity_resolve_linux(
    iree_thread_affinity_t affinity,
    const iree_thread_affinity_set_t* processor_set,
    iree_host_size_t native_size, void* out_native_set) {
  cpu_set_t* native_set = (cpu_set_t*)out_native_set;
  CPU_ZERO_S(native_size, native_set);
  if (affinity.group_any) {
    return iree_thread_make_cpu_set_from_node_id(affinity.group, processor_set,
                                                 native_size, native_set);
  }
  if (affinity.id_assigned) {
    if (!processor_set && affinity.id + affinity.smt >= native_size * 8) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "CPU %u exceeds native affinity mask",
                              affinity.id);
    }
    for (uint32_t cpu = affinity.id; cpu <= affinity.id + affinity.smt; ++cpu) {
      if (!processor_set || iree_thread_affinity_set_test(processor_set, cpu)) {
        CPU_SET_S(cpu, native_size, native_set);
      }
    }
  } else if (processor_set) {
    for (iree_host_size_t cpu =
             iree_thread_affinity_set_find_first_set(processor_set, 0);
         cpu < IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT;
         cpu =
             iree_thread_affinity_set_find_first_set(processor_set, cpu + 1)) {
      CPU_SET_S(cpu, native_size, native_set);
    }
  }
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_ANDROID || IREE_PLATFORM_LINUX

iree_status_t iree_thread_affinity_query_node(
    uint32_t node_id, const iree_thread_affinity_set_t* processor_set,
    iree_thread_affinity_set_t* out_processor_set) {
  memset(out_processor_set, 0, sizeof(*out_processor_set));
#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
  const iree_host_size_t native_size =
      CPU_ALLOC_SIZE(IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  cpu_set_t* native_set = iree_alloca(native_size);
  CPU_ZERO_S(native_size, native_set);
  IREE_RETURN_IF_ERROR(iree_thread_make_cpu_set_from_node_id(
      node_id, processor_set, native_size, native_set));
  iree_bitmap_t bitmap = iree_thread_affinity_set_bitmap(out_processor_set);
  for (iree_host_size_t cpu = 0; cpu < bitmap.bit_count; ++cpu) {
    if (CPU_ISSET_S(cpu, native_size, native_set)) {
      iree_bitmap_set(bitmap, cpu);
    }
  }
  return iree_ok_status();
#elif defined(IREE_PLATFORM_WINDOWS)
  if (node_id > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node %u exceeds Windows node IDs", node_id);
  }
  GROUP_AFFINITY native = {0};
  if (!GetNumaNodeProcessorMaskEx((USHORT)node_id, &native)) {
    const DWORD error = GetLastError();
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "cannot resolve NUMA node %u CPUs: %lu", node_id,
                            error);
  }
  if (native.Group >= IREE_ARRAYSIZE(out_processor_set->words)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node %u CPU mask exceeds %u CPUs", node_id,
                            IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  }
  out_processor_set->words[native.Group] = native.Mask;
  if (processor_set) {
    out_processor_set->words[native.Group] &=
        processor_set->words[native.Group];
  }
  return iree_ok_status();
#else
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "NUMA processor queries are unsupported on this "
                          "platform");
#endif
}
