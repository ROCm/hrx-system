// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ALLOCATOR_H_
#define IREE_HAL_ALLOCATOR_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/atomic.h"
#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Describes one native memory heap available to a device.
//
// Heap discovery is informational: explicit pool construction selects a native
// route and owns subsequent buffer allocation or import. Heaps are enumerated
// in the backend's preferred order for common execution, with the most
// preferred first. A local CPU or unified-memory device may expose one heap,
// while a discrete or remote device may expose several distinct placements.
typedef struct iree_hal_allocator_memory_heap_t {
  // Bits that describe the residency and behavior of the memory type.
  iree_hal_memory_type_t type;

  // Indicates what kind of usage is allowed of buffers allocated from this
  // heap. For example, exclusive device local memory may not allow mapping
  // while cached host local memory may not allow usage in dispatches.
  iree_hal_buffer_usage_t allowed_usage;

  // Atomic operations supported by naturally aligned locations allocated from
  // this heap. Concrete placement may narrow these prospective capabilities,
  // and queue-family capabilities are required independently.
  iree_hal_atomic_operation_capabilities_t atomic_operations;

  // Maximum size, in bytes, of any individual native allocation of this type.
  // Smaller allocations may still fail due to fragmentation or exhaustion.
  iree_device_size_t max_allocation_size;

  // Minimum alignment, in bytes, of native allocations of this type.
  iree_device_size_t min_alignment;
} iree_hal_allocator_memory_heap_t;

// Memory protection flags for controlling access to virtual address ranges.
// Maps to the equivalent host or device virtual memory protection API.
enum iree_hal_memory_protection_bits_t {
  // No access. Removes existing access permissions.
  // Platform: PROT_NONE (POSIX), PAGE_NOACCESS (Windows), or equivalent.
  IREE_HAL_MEMORY_PROTECTION_NONE = 0ull,

  // Read-only access.
  // Platform: PROT_READ (POSIX), PAGE_READONLY (Windows), or equivalent.
  IREE_HAL_MEMORY_PROTECTION_READ = 1ull << 0,

  // Write-only access (uncommon, typically combined with READ).
  // This is not directly supported on all platforms and may require READ.
  IREE_HAL_MEMORY_PROTECTION_WRITE = 1ull << 1,

  // Read-write access (most common).
  // Platform: PROT_READ|PROT_WRITE (POSIX), PAGE_READWRITE (Windows), or
  // equivalent.
  IREE_HAL_MEMORY_PROTECTION_READ_WRITE =
      IREE_HAL_MEMORY_PROTECTION_READ | IREE_HAL_MEMORY_PROTECTION_WRITE,
};
typedef uint64_t iree_hal_memory_protection_t;

// Memory usage advice flags for optimizing access patterns.
// Maps to madvise (POSIX) or an implementation-defined behavior.
enum iree_hal_memory_advice_bits_t {
  // No specific advice. Use default behavior.
  // Platform: MADV_NORMAL (POSIX).
  IREE_HAL_MEMORY_ADVICE_NORMAL = 0ull,

  // Will access this memory soon. Prefetch into cache/VRAM.
  // Platform: MADV_WILLNEED (POSIX); no direct equivalent on Windows.
  IREE_HAL_MEMORY_ADVICE_WILL_NEED = 1ull << 0,

  // Don't need this memory anymore. Can discard and zero on next access.
  // Platform: MADV_DONTNEED (POSIX - zeros immediately),
  // MADV_FREE (POSIX - lazy); no direct equivalent on Windows.
  // NOTE: On CPU, this may immediately free physical memory.
  IREE_HAL_MEMORY_ADVICE_DONT_NEED = 1ull << 1,
};
typedef uint64_t iree_hal_memory_advice_t;

//===----------------------------------------------------------------------===//
// Statistics/reporting
//===----------------------------------------------------------------------===//

// Aggregate allocation statistics.
typedef struct iree_hal_allocator_statistics_t {
#if IREE_STATISTICS_ENABLE
  iree_device_size_t host_bytes_peak;
  iree_device_size_t host_bytes_allocated;
  iree_device_size_t host_bytes_freed;
  iree_device_size_t device_bytes_peak;
  iree_device_size_t device_bytes_allocated;
  iree_device_size_t device_bytes_freed;
  // TODO(benvanik): mapping information (discarded, mapping ranges,
  //                 flushed/invalidated, etc).
#else
  int reserved;
#endif  // IREE_STATISTICS_ENABLE
} iree_hal_allocator_statistics_t;

// Formats allocator statistics as a pretty-printed multi-line string.
IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_format(
    const iree_hal_allocator_statistics_t* statistics,
    iree_string_builder_t* builder);

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_allocator_t iree_hal_allocator_t;

// Retains the given |allocator| for the caller.
IREE_API_EXPORT void iree_hal_allocator_retain(iree_hal_allocator_t* allocator);

// Releases the given |allocator| from the caller.
IREE_API_EXPORT void iree_hal_allocator_release(
    iree_hal_allocator_t* allocator);

// Returns the host allocator used for allocating host objects.
IREE_API_EXPORT iree_allocator_t iree_hal_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT allocator);

// Queries aggregate native backing statistics from the allocator since
// creation. Explicit pool allocations are reported by
// iree_hal_pool_query_stats() instead. Thread-safe; statistics are captured at
// the time the call is made.
//
// NOTE: statistics may be compiled out in some configurations and this call
// will become a memset(0).
IREE_API_EXPORT void iree_hal_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics);

// Prints the current allocation statistics of |allocator| to |file|.
// No-op if statistics are not enabled (IREE_STATISTICS_ENABLE).
IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_fprint(
    FILE* file, iree_hal_allocator_t* IREE_RESTRICT allocator);

// Queries the native memory heaps visible to the allocator.
// The resulting heaps are sorted in preferred performance order for common
// execution, with the most preferred first. Pool factories use this metadata
// as one input when qualifying concrete memory routes.
//
// If |heaps| is NULL then the call will query the heap count. This allows for
// preallocation of storage:
//   iree_host_size_t count = 0;
//   iree_hal_allocator_query_memory_heaps(allocator, 0, NULL, &count);
//   ... heaps = iree_alloca(sizeof(heap) * count);
//   iree_hal_allocator_query_memory_heaps(allocator, count, heaps, &count);
//
// Returns the total count in |out_count| and if |capacity| is large enough
// will fill |heaps| with the heap information.
// Returns IREE_STATUS_OUT_OF_RANGE if |capacity| is too small.
IREE_API_EXPORT iree_status_t iree_hal_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT allocator, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count);

//===----------------------------------------------------------------------===//
// iree_hal_heap_allocator_t
//===----------------------------------------------------------------------===//

// Creates allocator metadata describing ordinary coherent host memory.
// Buffer storage is acquired through an explicit heap pool.
IREE_API_EXPORT iree_status_t iree_hal_allocator_create_heap(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_allocator_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_allocator_t* IREE_RESTRICT allocator);

  iree_allocator_t(IREE_API_PTR* host_allocator)(
      const iree_hal_allocator_t* IREE_RESTRICT allocator);

  void(IREE_API_PTR* query_statistics)(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics);

  iree_status_t(IREE_API_PTR* query_memory_heaps)(
      iree_hal_allocator_t* IREE_RESTRICT allocator, iree_host_size_t capacity,
      iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
      iree_host_size_t* IREE_RESTRICT out_count);

} iree_hal_allocator_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_allocator_vtable_t);

IREE_API_EXPORT void iree_hal_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT allocator);

#if IREE_STATISTICS_ENABLE

// Records a native backing allocation to |statistics|.
static inline void iree_hal_allocator_statistics_record_alloc(
    iree_hal_allocator_statistics_t* statistics,
    iree_hal_memory_type_t memory_type, iree_device_size_t allocation_size) {
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    statistics->host_bytes_allocated += allocation_size;
    statistics->host_bytes_peak =
        iree_max(statistics->host_bytes_peak, statistics->host_bytes_allocated -
                                                  statistics->host_bytes_freed);
  } else {
    statistics->device_bytes_allocated += allocation_size;
    statistics->device_bytes_peak = iree_max(
        statistics->device_bytes_peak,
        statistics->device_bytes_allocated - statistics->device_bytes_freed);
  }
}

// Records a native backing deallocation to |statistics|.
static inline void iree_hal_allocator_statistics_record_free(
    iree_hal_allocator_statistics_t* statistics,
    iree_hal_memory_type_t memory_type, iree_device_size_t allocation_size) {
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    statistics->host_bytes_freed += allocation_size;
  } else {
    statistics->device_bytes_freed += allocation_size;
  }
}

#else
#define iree_hal_allocator_statistics_record_alloc(statistics, ...)
#define iree_hal_allocator_statistics_record_free(statistics, ...)
#endif  // IREE_STATISTICS_ENABLE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ALLOCATOR_H_
