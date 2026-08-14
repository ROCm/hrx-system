// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_descriptor.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/barrier.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/pm4_command_atomic.h"
#include "iree/hal/drivers/amdgpu/pm4_command_builder.h"
#include "iree/hal/drivers/amdgpu/pm4_command_record.h"
#include "iree/hal/drivers/amdgpu/pm4_command_transfer.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/hal/drivers/amdgpu/util/signal_pool.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// PM4 command-buffer storage
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_pm4_command_buffer_vtable;

typedef enum iree_hal_amdgpu_pm4_command_buffer_recording_state_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_INITIAL = 0,
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING = 1,
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FINALIZED = 2,
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FAILED = 3,
} iree_hal_amdgpu_pm4_command_buffer_recording_state_t;

typedef enum iree_hal_amdgpu_pm4_dispatch_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER = 1u << 0,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER = 1u << 1,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER = 1u << 2,
} iree_hal_amdgpu_pm4_dispatch_record_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_dispatch_record_flags_t;

typedef enum iree_hal_amdgpu_pm4_binding_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC = 1u << 0,
} iree_hal_amdgpu_pm4_binding_record_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_binding_record_flags_t;

typedef struct iree_hal_amdgpu_pm4_dispatch_record_t {
  // Common command-record header.
  iree_hal_amdgpu_pm4_command_record_header_t header;
  // Host descriptor with executable-load PM4 metadata for this dispatch.
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor;
  // Session-local executable identifier, or 0 when unavailable.
  uint64_t executable_id;
  // Dispatch workgroup counts.
  uint32_t workgroup_count[3];
  // DISPATCH_DIRECT thread dimensions.
  uint32_t dispatch_thread_count[3];
  // HAL command ordinal within this command buffer.
  uint32_t command_index;
  // Executable export ordinal dispatched by this command.
  uint32_t export_ordinal;
  // Byte offset of this dispatch's kernarg template in resident template
  // memory.
  uint32_t template_offset;
  // Byte length of this dispatch's kernarg template.
  uint32_t template_length;
  // Byte length of inline constant data following the record.
  uint32_t constant_length;
  // Number of inline native binding records following constants.
  uint32_t binding_record_count;
  // iree_hal_amdgpu_pm4_dispatch_record_flag_bits_t mask.
  iree_hal_amdgpu_pm4_dispatch_record_flags_t flags;
  // Acquire fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_acquire_scope;
  // Release fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_release_scope;
} iree_hal_amdgpu_pm4_dispatch_record_t;

typedef struct iree_hal_amdgpu_pm4_binding_record_t {
  // Static binding device pointer, or dynamic binding byte offset.
  uint64_t value;
  // Dynamic binding table slot when the dynamic flag is set.
  uint32_t binding_slot;
  // Byte offset in resident kernarg template storage.
  uint32_t target_offset;
  // iree_hal_amdgpu_pm4_binding_record_flag_bits_t mask.
  iree_hal_amdgpu_pm4_binding_record_flags_t flags;
  // Reserved padding; must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_pm4_binding_record_t;

enum {
  IREE_HAL_AMDGPU_PM4_RETAINED_RESOURCE_INLINE_CAPACITY = 64,
};

typedef struct iree_hal_amdgpu_pm4_retained_resource_table_t {
  // Open-addressed table of retained resources seen during recording.
  iree_hal_resource_t** resources;
  // Number of occupied resource slots.
  iree_host_size_t count;
  // Power-of-two capacity of |resources|.
  iree_host_size_t capacity;
  // Inline slots used before recording sees enough unique resources to spill.
  iree_hal_resource_t*
      inline_resources[IREE_HAL_AMDGPU_PM4_RETAINED_RESOURCE_INLINE_CAPACITY];
} iree_hal_amdgpu_pm4_retained_resource_table_t;

typedef struct iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t {
  // Next allocation in the resident pool free list.
  struct iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* next;
  // Device-visible executable allocation base pointer.
  IREE_AMDGPU_DEVICE_PTR uint8_t* base;
  // Allocated byte capacity of |base|.
  iree_host_size_t capacity;
} iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t;

struct iree_hal_amdgpu_pm4_command_buffer_resident_pool_t {
  // HSA API table used to allocate and free resident storage.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // CPU agent associated with host-side materialization buffers.
  hsa_agent_t host_agent;
  // GPU agent allowed to fetch resident PM4 storage.
  hsa_agent_t device_agent;
  // HSA memory pool used for executable PM4 storage.
  hsa_amd_memory_pool_t memory_pool;
  // HSA host memory pool used for async-copy staging sources.
  hsa_amd_memory_pool_t host_staging_memory_pool;
  // Host allocator used for allocation metadata and this pool.
  iree_allocator_t host_allocator;
  // Recommended HSA allocation granule used to round backing allocations.
  iree_host_size_t allocation_granule;
  // Recommended HSA allocation granule used to round host staging allocations.
  iree_host_size_t host_staging_allocation_granule;
  // Mutex guarding the free list and outstanding allocation count.
  iree_slim_mutex_t mutex;
  // HSA signals used for synchronous waits on async publication copies.
  iree_hal_amdgpu_host_signal_pool_t copy_signal_pool;
  // Cached allocations available for command-buffer publication.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* free_list
      IREE_GUARDED_BY(mutex);
  // Cached host staging allocations available for async-copy publication.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* staging_free_list
      IREE_GUARDED_BY(mutex);
  // Number of allocations currently borrowed by finalized command buffers.
  iree_host_size_t outstanding_count IREE_GUARDED_BY(mutex);
  // Number of staging allocations currently borrowed during finalization.
  iree_host_size_t outstanding_staging_count IREE_GUARDED_BY(mutex);
};

static void iree_hal_amdgpu_pm4_command_buffer_resident_allocation_free(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation) {
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_memory_pool_free_raw(
      resident_pool->libhsa, allocation->base));
  iree_allocator_free(resident_pool->host_allocator, allocation);
}

iree_status_t iree_hal_amdgpu_pm4_command_buffer_resident_pool_create(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t resident_memory_pool,
    hsa_amd_memory_pool_t host_staging_memory_pool,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  if (IREE_UNLIKELY(!libhsa)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resident pool HSA API table is required");
  }
  if (IREE_UNLIKELY(!resident_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resident pool memory pool is required");
  }
  if (IREE_UNLIKELY(!host_staging_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer host staging memory pool is required");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*resident_pool), (void**)&resident_pool);
  if (iree_status_is_ok(status)) {
    memset(resident_pool, 0, sizeof(*resident_pool));
    resident_pool->libhsa = libhsa;
    resident_pool->host_agent = host_agent;
    resident_pool->device_agent = device_agent;
    resident_pool->memory_pool = resident_memory_pool;
    resident_pool->host_staging_memory_pool = host_staging_memory_pool;
    resident_pool->host_allocator = host_allocator;
    iree_slim_mutex_initialize(&resident_pool->mutex);

    size_t allocation_granule = 0;
    status = iree_hsa_amd_memory_pool_get_info(
        IREE_LIBHSA(libhsa), resident_memory_pool,
        HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
        &allocation_granule);
    if (iree_status_is_ok(status)) {
      if (IREE_UNLIKELY(allocation_granule == 0 ||
                        !iree_host_size_is_power_of_two(
                            (iree_host_size_t)allocation_granule))) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "PM4 command-buffer resident pool allocation granule %zu is "
            "invalid",
            allocation_granule);
      } else {
        resident_pool->allocation_granule =
            (iree_host_size_t)allocation_granule;
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_host_signal_pool_initialize(
          libhsa, /*initial_capacity=*/0,
          IREE_HAL_AMDGPU_HOST_SIGNAL_POOL_BATCH_SIZE_DEFAULT, host_allocator,
          &resident_pool->copy_signal_pool);
    }
    if (iree_status_is_ok(status)) {
      size_t host_staging_allocation_granule = 0;
      status = iree_hsa_amd_memory_pool_get_info(
          IREE_LIBHSA(libhsa), host_staging_memory_pool,
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
          &host_staging_allocation_granule);
      if (iree_status_is_ok(status)) {
        if (IREE_UNLIKELY(
                host_staging_allocation_granule == 0 ||
                !iree_host_size_is_power_of_two(
                    (iree_host_size_t)host_staging_allocation_granule))) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "PM4 command-buffer host staging allocation granule %zu is "
              "invalid",
              host_staging_allocation_granule);
        } else {
          resident_pool->host_staging_allocation_granule =
              (iree_host_size_t)host_staging_allocation_granule;
        }
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_pool = resident_pool;
  } else if (resident_pool) {
    if (resident_pool->copy_signal_pool.libhsa) {
      iree_hal_amdgpu_host_signal_pool_deinitialize(
          &resident_pool->copy_signal_pool);
    }
    iree_slim_mutex_deinitialize(&resident_pool->mutex);
    iree_allocator_free(host_allocator, resident_pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_pm4_command_buffer_resident_pool_trim(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool) {
  if (!resident_pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&resident_pool->mutex);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation =
      resident_pool->free_list;
  resident_pool->free_list = NULL;
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* staging_allocation =
      resident_pool->staging_free_list;
  resident_pool->staging_free_list = NULL;
  iree_slim_mutex_unlock(&resident_pool->mutex);

  while (allocation) {
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* next_allocation =
        allocation->next;
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_free(resident_pool,
                                                                allocation);
    allocation = next_allocation;
  }
  while (staging_allocation) {
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* next_allocation =
        staging_allocation->next;
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_free(
        resident_pool, staging_allocation);
    staging_allocation = next_allocation;
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_pm4_command_buffer_resident_pool_destroy(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool) {
  if (!resident_pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(resident_pool->outstanding_count == 0,
              "PM4 command-buffer resident pool destroyed with %" PRIhsz
              " outstanding allocations",
              resident_pool->outstanding_count);
  IREE_ASSERT(resident_pool->outstanding_staging_count == 0,
              "PM4 command-buffer resident pool destroyed with %" PRIhsz
              " outstanding staging allocations",
              resident_pool->outstanding_staging_count);
  iree_hal_amdgpu_pm4_command_buffer_resident_pool_trim(resident_pool);
  iree_hal_amdgpu_host_signal_pool_deinitialize(
      &resident_pool->copy_signal_pool);
  iree_slim_mutex_deinitialize(&resident_pool->mutex);
  iree_allocator_t host_allocator = resident_pool->host_allocator;
  iree_allocator_free(host_allocator, resident_pool);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_round_pooled_capacity(
    iree_host_size_t required_byte_length, iree_host_size_t allocation_granule,
    iree_host_size_t* out_capacity) {
  *out_capacity = 0;
  if (IREE_UNLIKELY(required_byte_length == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resident allocation cannot be empty");
  }
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
          required_byte_length, allocation_granule, out_capacity))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident allocation size overflows");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_host_size_t required_byte_length, bool collect_timings,
    iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** out_allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  IREE_ASSERT_ARGUMENT(out_allocation);
  *out_allocation = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, required_byte_length);

  iree_host_size_t capacity = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_pm4_command_buffer_round_pooled_capacity(
          required_byte_length, resident_pool->allocation_granule, &capacity));

  iree_slim_mutex_lock(&resident_pool->mutex);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** inout_link =
      &resident_pool->free_list;
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation =
      resident_pool->free_list;
  while (allocation) {
    if (allocation->capacity >= required_byte_length) {
      *inout_link = allocation->next;
      allocation->next = NULL;
      ++resident_pool->outstanding_count;
      break;
    }
    inout_link = &allocation->next;
    allocation = allocation->next;
  }
  iree_slim_mutex_unlock(&resident_pool->mutex);

  if (allocation) {
    *out_allocation = allocation;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status = iree_allocator_malloc(
      resident_pool->host_allocator, sizeof(*allocation), (void**)&allocation);
  if (iree_status_is_ok(status)) {
    memset(allocation, 0, sizeof(*allocation));
    allocation->capacity = capacity;

    const bool should_collect_stats = collect_timings && publish_stats;
    iree_time_t time_start = should_collect_stats ? iree_time_now() : 0;
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(resident_pool->libhsa), resident_pool->memory_pool,
        allocation->capacity, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG,
        (void**)&allocation->base);
    if (should_collect_stats) {
      publish_stats->resident_allocate_ns += iree_time_now() - time_start;
    }
    if (iree_status_is_ok(status)) {
      time_start = should_collect_stats ? iree_time_now() : 0;
      status = iree_hsa_amd_agents_allow_access(
          IREE_LIBHSA(resident_pool->libhsa),
          /*num_agents=*/1, &resident_pool->device_agent,
          /*flags=*/NULL, allocation->base);
      if (should_collect_stats) {
        publish_stats->resident_allow_access_ns += iree_time_now() - time_start;
      }
    }
    if (iree_status_is_ok(status) && publish_stats) {
      publish_stats->resident_allocation_count += 1;
      publish_stats->resident_allow_access_agent_count += 1;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&resident_pool->mutex);
    ++resident_pool->outstanding_count;
    iree_slim_mutex_unlock(&resident_pool->mutex);
    *out_allocation = allocation;
  } else if (allocation) {
    if (allocation->base) {
      status = iree_status_join(
          status, iree_hsa_amd_memory_pool_free(
                      IREE_LIBHSA(resident_pool->libhsa), allocation->base));
    }
    iree_allocator_free(resident_pool->host_allocator, allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_pm4_command_buffer_resident_pool_release(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  if (!allocation) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation->capacity);

  iree_slim_mutex_lock(&resident_pool->mutex);
  IREE_ASSERT(resident_pool->outstanding_count > 0,
              "PM4 command-buffer resident allocation released without an "
              "outstanding borrow");
  allocation->next = resident_pool->free_list;
  resident_pool->free_list = allocation;
  --resident_pool->outstanding_count;
  iree_slim_mutex_unlock(&resident_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire_staging(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_host_size_t required_byte_length, bool collect_timings,
    iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** out_allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  IREE_ASSERT_ARGUMENT(out_allocation);
  *out_allocation = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, required_byte_length);

  iree_host_size_t capacity = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_pm4_command_buffer_round_pooled_capacity(
              required_byte_length,
              resident_pool->host_staging_allocation_granule, &capacity));

  iree_slim_mutex_lock(&resident_pool->mutex);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** inout_link =
      &resident_pool->staging_free_list;
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation =
      resident_pool->staging_free_list;
  while (allocation) {
    if (allocation->capacity >= required_byte_length) {
      *inout_link = allocation->next;
      allocation->next = NULL;
      ++resident_pool->outstanding_staging_count;
      break;
    }
    inout_link = &allocation->next;
    allocation = allocation->next;
  }
  iree_slim_mutex_unlock(&resident_pool->mutex);

  if (allocation) {
    *out_allocation = allocation;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status = iree_allocator_malloc(
      resident_pool->host_allocator, sizeof(*allocation), (void**)&allocation);
  if (iree_status_is_ok(status)) {
    memset(allocation, 0, sizeof(*allocation));
    allocation->capacity = capacity;

    const bool should_collect_stats = collect_timings && publish_stats;
    iree_time_t time_start = should_collect_stats ? iree_time_now() : 0;
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(resident_pool->libhsa),
        resident_pool->host_staging_memory_pool, allocation->capacity,
        HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&allocation->base);
    if (should_collect_stats) {
      publish_stats->host_staging_allocate_ns += iree_time_now() - time_start;
    }
    if (iree_status_is_ok(status)) {
      time_start = should_collect_stats ? iree_time_now() : 0;
      status = iree_hsa_amd_agents_allow_access(
          IREE_LIBHSA(resident_pool->libhsa),
          /*num_agents=*/1, &resident_pool->device_agent,
          /*flags=*/NULL, allocation->base);
      if (should_collect_stats) {
        publish_stats->host_staging_allow_access_ns +=
            iree_time_now() - time_start;
      }
    }
    if (iree_status_is_ok(status) && publish_stats) {
      publish_stats->host_staging_allocation_count += 1;
      publish_stats->host_staging_allow_access_agent_count += 1;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&resident_pool->mutex);
    ++resident_pool->outstanding_staging_count;
    iree_slim_mutex_unlock(&resident_pool->mutex);
    *out_allocation = allocation;
  } else if (allocation) {
    if (allocation->base) {
      status = iree_status_join(
          status, iree_hsa_amd_memory_pool_free(
                      IREE_LIBHSA(resident_pool->libhsa), allocation->base));
    }
    iree_allocator_free(resident_pool->host_allocator, allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_pm4_command_buffer_resident_pool_release_staging(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  if (!allocation) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation->capacity);

  iree_slim_mutex_lock(&resident_pool->mutex);
  IREE_ASSERT(resident_pool->outstanding_staging_count > 0,
              "PM4 command-buffer host staging allocation released without an "
              "outstanding borrow");
  allocation->next = resident_pool->staging_free_list;
  resident_pool->staging_free_list = allocation;
  --resident_pool->outstanding_staging_count;
  iree_slim_mutex_unlock(&resident_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

typedef enum iree_hal_amdgpu_pm4_materialization_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_NONE = 0u,
  // Emits the dispatch-attributed profiling program for the active queue.
  IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_PROFILE = 1u << 0,
  // Reuses templates already populated by the first normal program variant.
  IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_REUSE_TEMPLATES = 1u << 1,
} iree_hal_amdgpu_pm4_materialization_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_materialization_flags_t;

// Mutable state scoped to one finalize-time replay of compact command records.
typedef struct iree_hal_amdgpu_pm4_materialization_state_t {
  // Resident PM4 program being populated by this replay.
  iree_hal_amdgpu_pm4_program_t* program;
  // Shared normal fixup and template plan.
  iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup;
  // Profile plan being populated, or NULL for normal execution.
  iree_hal_amdgpu_pm4_command_buffer_profile_plan_t* profile;
  // Borrowed views over the resident regions populated by this replay.
  struct {
    // Destination PM4 program builder.
    iree_hal_amdgpu_pm4_dword_builder_t program;
    // Shared resident kernarg-template builder.
    iree_hal_amdgpu_pm4_byte_builder_t template;
    // Destination dynamic binding fixup builder.
    iree_hal_amdgpu_pm4_fixup_entry_builder_t fixup;
  } builders;
  // Byte offset of the destination program from the resident allocation base.
  iree_host_size_t program_offset;
  // PM4 packet-family capabilities for barrier emission.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // Most recent shader launch state emitted into this program.
  iree_hal_amdgpu_pm4_dispatch_launch_state_t previous_launch_state;
  // Whether |previous_launch_state| contains a valid value.
  bool has_previous_launch_state;
  // Whether the fixup-to-IB visibility barrier has been emitted.
  bool has_emitted_fixup_barrier;
  // iree_hal_amdgpu_pm4_materialization_flag_bits_t mask.
  iree_hal_amdgpu_pm4_materialization_flags_t flags;
} iree_hal_amdgpu_pm4_materialization_state_t;
typedef struct iree_hal_amdgpu_pm4_command_buffer_t {
  // Base HAL command-buffer resource.
  iree_hal_command_buffer_t base;
  // Host allocator used for command-buffer-owned host storage.
  iree_allocator_t host_allocator;
  // Borrowed block pool used for retained resource sets.
  iree_arena_block_pool_t* resource_set_block_pool;
  // Resource set retaining static buffers and executables unless unretained.
  iree_hal_resource_set_t* resource_set;
  // Exact first-use table for resources retained into |resource_set|.
  iree_hal_amdgpu_pm4_retained_resource_table_t retained_resources;
  // Last executable retained into |resource_set| during this recording.
  iree_hal_executable_t* last_retained_executable;
  // Pool that owns resident PM4 storage allocations.
  iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool;
  // Stable opaque hostcall device address written into implicit templates.
  void* hostcall_buffer;
  // Borrowed resident storage allocation returned to |resident_pool|.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* resident_allocation;
  // Borrowed host staging allocation returned to |resident_pool|.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t*
      host_staging_allocation;
  // Mutex guarding nonblocking publication signal ownership.
  iree_slim_mutex_t publication_mutex;
  // Pending async-copy publication signal, or null when resident bytes no
  // longer need queue-side publication ordering.
  hsa_signal_t publication_signal IREE_GUARDED_BY(publication_mutex);
  // Number of queued AQL barriers that may still reference
  // |publication_signal|.
  uint32_t publication_reference_count IREE_GUARDED_BY(publication_mutex);
  // HSA API table used to copy materialized resident PM4 storage.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // PM4 packet-family capabilities validated for this physical device.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // Borrowed fallback atomic launch metadata for this physical device.
  const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context;
  // Borrowed builtin transfer planning metadata for this physical device.
  const iree_hal_amdgpu_device_buffer_transfer_context_t*
      buffer_transfer_context;
  // Borrowed builtin transfer PM4 launch metadata for this physical device.
  const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
      buffer_transfer_pm4_context;
  // PM4 timestamp packet strategy selected for this physical device.
  iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy;
  // iree_hal_amdgpu_pm4_command_buffer_flag_bits_t mask.
  iree_hal_amdgpu_pm4_command_buffer_flags_t flags;
  // Physical device ordinal this command buffer was recorded for.
  uint32_t device_ordinal;
  // Recording lifecycle state.
  iree_hal_amdgpu_pm4_command_buffer_recording_state_t recording_state;
  // Compact command record planning and storage state.
  iree_hal_amdgpu_pm4_command_recording_state_t recording;
  // Per-binding atomic memory cells required at queue submission, or NULL
  // when no dynamic atomic targets were recorded.
  iree_hal_amdgpu_atomic_memory_cell_flags_t* atomic_binding_requirements;
  // Profile metadata retained for iree-profile command-buffer records.
  struct {
    // Borrowed metadata registry owned by the logical device.
    iree_hal_amdgpu_profile_metadata_registry_t* metadata;
    // Session-local command-buffer identifier, or 0 when metadata is not
    // retained.
    uint64_t id;
    // Host copy of profile-visible dispatch operation records.
    iree_hal_profile_command_operation_record_t* operations;
    // Number of entries in |operations|.
    uint32_t operation_count;
  } profile;
  // Byte offset of resident kernarg-template storage within the allocation.
  iree_host_size_t resident_template_offset;
  // Finalize-time publication stats.
  iree_hal_amdgpu_pm4_command_buffer_publish_stats_t publish_stats;
  // Shared execution and queue-indexed profile plans produced by end().
  iree_hal_amdgpu_pm4_command_program_set_t program_set;
  // Total byte length of the resident allocation backing |program_set|.
  iree_host_size_t resident_byte_length;
} iree_hal_amdgpu_pm4_command_buffer_t;

static iree_hal_amdgpu_pm4_command_buffer_t*
iree_hal_amdgpu_pm4_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_pm4_command_buffer_vtable);
  return (iree_hal_amdgpu_pm4_command_buffer_t*)base_value;
}

static bool iree_hal_amdgpu_pm4_command_buffer_retains_resources(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return !iree_all_bits_set(command_buffer->base.mode,
                            IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED);
}

static bool iree_hal_amdgpu_pm4_command_buffer_retains_profile_metadata(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_all_bits_set(
      command_buffer->base.mode,
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA);
}

static void iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  if (!command_buffer->host_staging_allocation) return;
  iree_hal_amdgpu_pm4_command_buffer_resident_pool_release_staging(
      command_buffer->resident_pool, command_buffer->host_staging_allocation);
  command_buffer->host_staging_allocation = NULL;
}

static bool iree_hal_amdgpu_pm4_command_buffer_has_pending_publication(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  const bool has_pending_publication =
      !iree_hsa_signal_is_null(command_buffer->publication_signal);
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
  return has_pending_publication;
}

static void
iree_hal_amdgpu_pm4_command_buffer_release_publication_resources_locked(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  IREE_ASSERT(!iree_hsa_signal_is_null(command_buffer->publication_signal),
              "release requires a pending publication signal");
  iree_hal_amdgpu_host_signal_pool_release(
      &command_buffer->resident_pool->copy_signal_pool,
      command_buffer->publication_signal);
  command_buffer->publication_signal = iree_hsa_signal_null();
  iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(command_buffer);
}

static void iree_hal_amdgpu_pm4_command_buffer_wait_for_publication(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  IREE_ASSERT(command_buffer->publication_reference_count == 0,
              "PM4 command-buffer publication has %u live AQL references at "
              "destruction",
              command_buffer->publication_reference_count);
  if (!iree_hsa_signal_is_null(command_buffer->publication_signal)) {
    const hsa_signal_value_t completion_value = iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(command_buffer->libhsa), command_buffer->publication_signal,
        HSA_SIGNAL_CONDITION_LT, /*compare_value=*/1, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    IREE_ASSERT(completion_value == 0,
                "PM4 command-buffer async resident copy failed with signal "
                "value %" PRId64,
                (int64_t)completion_value);
    iree_hal_amdgpu_pm4_command_buffer_release_publication_resources_locked(
        command_buffer);
  } else {
    iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(command_buffer);
  }
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
}

static bool iree_hal_amdgpu_pm4_command_buffer_validates(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  return !iree_any_bit_set(command_buffer->base.mode,
                           IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED);
#else
  (void)command_buffer;
  return false;
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
}

static bool iree_hal_amdgpu_pm4_command_buffer_collects_finalize_timings(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_COLLECT_FINALIZE_TIMINGS);
}

static bool iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_COPY |
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_ASYNC_COPY);
}

static bool iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_ASYNC_COPY);
}

static bool iree_hal_amdgpu_pm4_command_buffer_uses_nonblocking_publication(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_NONBLOCKING_PUBLICATION);
}

static bool
iree_hal_amdgpu_pm4_command_buffer_materializes_profile_dispatch_timestamps(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_PROFILE_DISPATCH_TIMESTAMPS);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_ensure_resource_set(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  if (!iree_hal_amdgpu_pm4_command_buffer_retains_resources(command_buffer) ||
      command_buffer->resource_set) {
    return iree_ok_status();
  }
  return iree_hal_resource_set_allocate(command_buffer->resource_set_block_pool,
                                        &command_buffer->resource_set);
}

static void iree_hal_amdgpu_pm4_retained_resource_table_initialize(
    iree_hal_amdgpu_pm4_retained_resource_table_t* table) {
  memset(table->inline_resources, 0, sizeof(table->inline_resources));
  table->resources = table->inline_resources;
  table->count = 0;
  table->capacity = IREE_ARRAYSIZE(table->inline_resources);
}

static void iree_hal_amdgpu_pm4_retained_resource_table_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_retained_resource_table_t* table) {
  if (table->resources != table->inline_resources) {
    iree_allocator_free(host_allocator, table->resources);
  }
  table->resources = NULL;
  table->count = 0;
  table->capacity = 0;
  memset(table->inline_resources, 0, sizeof(table->inline_resources));
}

static iree_host_size_t iree_hal_amdgpu_pm4_retained_resource_table_hash(
    iree_hal_resource_t* resource) {
  uint64_t value = (uint64_t)((uintptr_t)resource >> 4);
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return (iree_host_size_t)value;
}

static iree_host_size_t iree_hal_amdgpu_pm4_retained_resource_table_find_slot(
    const iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_hal_resource_t* resource, bool* out_found) {
  const iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot =
      iree_hal_amdgpu_pm4_retained_resource_table_hash(resource) & mask;
  while (true) {
    iree_hal_resource_t* existing_resource = table->resources[slot];
    if (existing_resource == resource) {
      *out_found = true;
      return slot;
    }
    if (!existing_resource) {
      *out_found = false;
      return slot;
    }
    slot = (slot + 1) & mask;
  }
}

static iree_status_t iree_hal_amdgpu_pm4_retained_resource_table_reserve(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_host_size_t required_count) {
  if (required_count <= table->capacity - table->capacity / 4) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity = table->capacity * 2;
  while (required_count > new_capacity - new_capacity / 4) {
    if (IREE_UNLIKELY(new_capacity > IREE_HOST_SIZE_MAX / 2)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "PM4 command-buffer retained resource table exceeds host size");
    }
    new_capacity *= 2;
  }

  iree_hal_resource_t** new_resources = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, new_capacity,
                                                   sizeof(*new_resources),
                                                   (void**)&new_resources));
  iree_hal_amdgpu_pm4_retained_resource_table_t new_table = {
      .resources = new_resources,
      .capacity = new_capacity,
  };
  for (iree_host_size_t i = 0; i < table->capacity; ++i) {
    iree_hal_resource_t* resource = table->resources[i];
    if (!resource) continue;
    bool found = false;
    const iree_host_size_t slot =
        iree_hal_amdgpu_pm4_retained_resource_table_find_slot(&new_table,
                                                              resource, &found);
    IREE_ASSERT(!found);
    new_resources[slot] = resource;
    ++new_table.count;
  }
  IREE_ASSERT(new_table.count == table->count);
  if (table->resources != table->inline_resources) {
    iree_allocator_free(host_allocator, table->resources);
  }
  table->resources = new_resources;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_pm4_retained_resource_table_contains(
    const iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_hal_resource_t* resource) {
  bool found = false;
  iree_hal_amdgpu_pm4_retained_resource_table_find_slot(table, resource,
                                                        &found);
  return found;
}

static void iree_hal_amdgpu_pm4_retained_resource_table_insert_prepared(
    iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_hal_resource_t* resource) {
  bool found = false;
  const iree_host_size_t slot =
      iree_hal_amdgpu_pm4_retained_resource_table_find_slot(table, resource,
                                                            &found);
  IREE_ASSERT(!found);
  table->resources[slot] = resource;
  ++table->count;
}

static void iree_hal_amdgpu_pm4_recording_builders_initialize(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_hal_amdgpu_pm4_retained_resource_table_initialize(
      &command_buffer->retained_resources);
  iree_hal_amdgpu_pm4_byte_builder_initialize(
      command_buffer->host_allocator,
      &command_buffer->recording.record_builder);
}

static void iree_hal_amdgpu_pm4_recording_builders_deinitialize(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_hal_amdgpu_pm4_retained_resource_table_deinitialize(
      command_buffer->host_allocator, &command_buffer->retained_resources);
  iree_hal_amdgpu_pm4_byte_builder_deinitialize(
      &command_buffer->recording.record_builder);
  if (!iree_hal_amdgpu_pm4_command_buffer_has_pending_publication(
          command_buffer)) {
    iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(command_buffer);
  }
  command_buffer->recording.record_ib_dword_count = 0;
  command_buffer->recording.record_template_byte_length = 0;
  command_buffer->recording.record_fixup_entry_count = 0;
  command_buffer->resident_template_offset = 0;
  command_buffer->recording.profile.record_program_dword_count = 0;
  command_buffer->recording.profile.record_fixup_entry_count = 0;
}

//===----------------------------------------------------------------------===//
// PM4 packet emission
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_pm4_barrier_state_accumulate(
    iree_hal_amdgpu_pm4_command_barrier_state_t* barrier_state,
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  barrier_state->pending = true;
  barrier_state->acquire_scope = iree_hal_amdgpu_pm4_max_fence_scope(
      barrier_state->acquire_scope, acquire_scope);
  barrier_state->release_scope = iree_hal_amdgpu_pm4_max_fence_scope(
      barrier_state->release_scope, release_scope);
}

static void iree_hal_amdgpu_pm4_barrier_state_reset(
    iree_hal_amdgpu_pm4_command_barrier_state_t* barrier_state) {
  *barrier_state = (iree_hal_amdgpu_pm4_command_barrier_state_t){0};
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_nop(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t dword_count) {
  if (IREE_UNLIKELY(dword_count < 2)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 NOP packet requires at least 2 dwords");
  }
  uint32_t* nop_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, dword_count, &nop_dwords));
  nop_dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_NOP, dword_count);
  memset(nop_dwords + 1, 0, (dword_count - 1) * sizeof(nop_dwords[0]));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_align_timestamp_target(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    iree_host_size_t program_offset) {
  if (IREE_UNLIKELY(program_offset % sizeof(uint64_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 profile program offset is not aligned for timestamp fixup");
  }
  const iree_host_size_t program_dword_offset =
      program_offset / sizeof(uint32_t);
  const bool target_is_aligned =
      ((program_dword_offset + builder->dword_count + 4u) & 1u) == 0;
  return target_is_aligned ? iree_ok_status()
                           : iree_hal_amdgpu_pm4_dword_builder_emit_nop(
                                 builder, /*dword_count=*/3);
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_timestamp(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy, void* target,
    uint32_t* out_target_dword_offset) {
  IREE_ASSERT_ARGUMENT(out_target_dword_offset);
  *out_target_dword_offset = 0;
  const uint32_t control = iree_hal_amdgpu_pm4_copy_timestamp_control(strategy);
  if (IREE_UNLIKELY(control == 0 || !iree_host_ptr_has_alignment(target, 8))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 profile timestamp strategy cannot write aligned timestamp ranges");
  }

  const uint32_t packet_dword_offset = builder->dword_count;
  uint32_t* dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT, &dwords));
  const uintptr_t target_address = (uintptr_t)target;
  dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
      IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT);
  dwords[1] = control;
  dwords[2] = 0;
  dwords[3] = 0;
  dwords[4] = iree_hal_amdgpu_pm4_addr_lo_8(target_address);
  dwords[5] = iree_hal_amdgpu_pm4_addr_hi(target_address);
  *out_target_dword_offset = packet_dword_offset + 4u;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Resident storage publication
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_pm4_command_buffer_reset_resident_plans(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  memset(&command_buffer->program_set.program, 0,
         sizeof(command_buffer->program_set.program));
  memset(&command_buffer->program_set.fixup, 0,
         sizeof(command_buffer->program_set.fixup));
  if (command_buffer->program_set.profile_plans) {
    memset(command_buffer->program_set.profile_plans, 0,
           command_buffer->program_set.physical_queue_count *
               sizeof(command_buffer->program_set.profile_plans[0]));
  }
  command_buffer->resident_byte_length = 0;
}

static void iree_hal_amdgpu_pm4_command_buffer_bind_resident_plans(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_pm4_command_program_layout_t* layout,
    uint8_t* resident_base, uint32_t profile_binding_count) {
  uint8_t* const template_base = layout->template_byte_length != 0
                                     ? resident_base + layout->template_offset
                                     : NULL;
  command_buffer->program_set.program = (iree_hal_amdgpu_pm4_program_t){
      .libhsa = command_buffer->libhsa,
      .memory_pool = command_buffer->resident_pool->memory_pool,
      .dwords = (uint32_t*)(resident_base + layout->program_offset),
      .dword_count = (uint32_t)(layout->program_byte_length / sizeof(uint32_t)),
      .byte_length = layout->program_byte_length,
  };
  command_buffer->program_set
      .fixup = (iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t){
      .entries =
          layout->fixup_byte_length != 0
              ? (const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*)(resident_base +
                                                                          layout
                                                                              ->fixup_offset)
              : NULL,
      .entry_count = command_buffer->recording.record_fixup_entry_count,
      .target_base = resident_base,
      .target_byte_length = layout->total_byte_length,
      .template_base = template_base,
      .template_byte_length = layout->template_byte_length,
  };

  for (uint32_t profile_plan_ordinal = 0;
       profile_plan_ordinal < command_buffer->program_set.profile_plan_count;
       ++profile_plan_ordinal) {
    const iree_host_size_t profile_program_offset =
        iree_hal_amdgpu_pm4_command_program_layout_profile_program_offset(
            layout, profile_plan_ordinal);
    const iree_host_size_t profile_fixup_offset =
        iree_hal_amdgpu_pm4_command_program_layout_profile_fixup_offset(
            layout, profile_plan_ordinal);
    const iree_host_size_t dummy_ticks_offset =
        iree_hal_amdgpu_pm4_command_program_layout_dummy_ticks_offset(
            layout, profile_plan_ordinal);
    command_buffer->program_set.profile_plans
        [profile_plan_ordinal] = (iree_hal_amdgpu_pm4_command_buffer_profile_plan_t){
        .program =
            {
                .libhsa = command_buffer->libhsa,
                .memory_pool = command_buffer->resident_pool->memory_pool,
                .dwords = (uint32_t*)(resident_base + profile_program_offset),
                .dword_count = (uint32_t)(layout->profile_program_byte_length /
                                          sizeof(uint32_t)),
                .byte_length = layout->profile_program_byte_length,
            },
        .entries =
            layout->profile_fixup_byte_length != 0
                ? (const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*)(resident_base +
                                                                            profile_fixup_offset)
                : NULL,
        .entry_count =
            command_buffer->recording.profile.record_fixup_entry_count,
        .timestamp_binding_base = command_buffer->base.binding_count,
        .binding_count = profile_binding_count,
        .operation_count = command_buffer->recording.record_command_count,
        .target_base = resident_base,
        .dummy_ticks = (iree_hal_amdgpu_timestamp_range_t*)(resident_base +
                                                            dummy_ticks_offset),
    };
  }
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_prepare_resident_storage(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    uint32_t program_dword_count,
    iree_hal_amdgpu_pm4_command_program_layout_t* out_layout,
    uint8_t** out_materialization_base) {
  memset(out_layout, 0, sizeof(*out_layout));
  *out_materialization_base = NULL;
  if (IREE_UNLIKELY(command_buffer->recording.record_template_byte_length ==
                        0 &&
                    command_buffer->recording.record_fixup_entry_count != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command-buffer fixup entries require target storage");
  }

  uint32_t profile_binding_count = 0;
  if (command_buffer->program_set.profile_plan_count != 0) {
    if (IREE_UNLIKELY(command_buffer->recording.record_command_count >
                      (UINT32_MAX - command_buffer->base.binding_count) / 2u)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile timestamp binding count exceeds uint32_t storage");
    }
    profile_binding_count = command_buffer->base.binding_count +
                            2u * command_buffer->recording.record_command_count;
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_program_layout_calculate(
      &command_buffer->program_set, program_dword_count,
      command_buffer->recording.profile.record_program_dword_count,
      command_buffer->recording.record_template_byte_length,
      command_buffer->recording.record_fixup_entry_count,
      command_buffer->recording.profile.record_fixup_entry_count, out_layout));
  command_buffer->publish_stats.program_bytes = out_layout->program_byte_length;
  command_buffer->publish_stats.template_bytes =
      out_layout->template_byte_length;
  command_buffer->publish_stats.fixup_entry_bytes =
      out_layout->fixup_byte_length;
  command_buffer->publish_stats.resident_bytes = out_layout->total_byte_length;
  command_buffer->resident_template_offset = out_layout->template_offset;

  const bool collect_timings =
      iree_hal_amdgpu_pm4_command_buffer_collects_finalize_timings(
          command_buffer);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t*
      resident_allocation = NULL;
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire(
          command_buffer->resident_pool, out_layout->total_byte_length,
          collect_timings, &command_buffer->publish_stats,
          &resident_allocation);
  uint8_t* materialization_base = NULL;
  if (iree_status_is_ok(status)) {
    command_buffer->resident_allocation = resident_allocation;
    command_buffer->resident_byte_length = out_layout->total_byte_length;
    iree_hal_amdgpu_pm4_command_buffer_bind_resident_plans(
        command_buffer, out_layout, resident_allocation->base,
        profile_binding_count);

    if (iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(
            command_buffer)) {
      command_buffer->publish_stats.host_staging_bytes =
          out_layout->total_byte_length;
      iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t*
          staging_allocation = NULL;
      status = iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire_staging(
          command_buffer->resident_pool, out_layout->total_byte_length,
          collect_timings, &command_buffer->publish_stats, &staging_allocation);
      if (iree_status_is_ok(status)) {
        command_buffer->host_staging_allocation = staging_allocation;
        materialization_base = staging_allocation->base;
      }
    } else if (iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(
                   command_buffer)) {
      command_buffer->publish_stats.host_staging_bytes =
          out_layout->total_byte_length;
      const iree_time_t time_start = collect_timings ? iree_time_now() : 0;
      status = iree_allocator_malloc(command_buffer->host_allocator,
                                     out_layout->total_byte_length,
                                     (void**)&materialization_base);
      if (collect_timings) {
        command_buffer->publish_stats.host_staging_allocate_ns +=
            iree_time_now() - time_start;
      }
    } else {
      materialization_base = resident_allocation->base;
    }
  }
  if (iree_status_is_ok(status)) {
    memset(materialization_base, 0, out_layout->total_byte_length);
    *out_materialization_base = materialization_base;
  } else if (resident_allocation) {
    if (materialization_base &&
        iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(
            command_buffer) &&
        !iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(
            command_buffer)) {
      iree_allocator_free(command_buffer->host_allocator, materialization_base);
    }
    iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(command_buffer);
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_release(
        command_buffer->resident_pool, resident_allocation);
    command_buffer->resident_allocation = NULL;
    iree_hal_amdgpu_pm4_command_buffer_reset_resident_plans(command_buffer);
  }
  return status;
}

static void iree_hal_amdgpu_pm4_command_buffer_initialize_materialization_state(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_pm4_command_program_layout_t* layout,
    uint8_t* materialization_base, uint32_t profile_plan_ordinal,
    iree_hal_amdgpu_pm4_materialization_flags_t flags,
    iree_hal_amdgpu_pm4_materialization_state_t* out_state) {
  const bool is_profile =
      iree_any_bit_set(flags, IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_PROFILE);
  const iree_host_size_t program_offset =
      is_profile
          ? iree_hal_amdgpu_pm4_command_program_layout_profile_program_offset(
                layout, profile_plan_ordinal)
          : layout->program_offset;
  const iree_host_size_t fixup_offset =
      is_profile
          ? iree_hal_amdgpu_pm4_command_program_layout_profile_fixup_offset(
                layout, profile_plan_ordinal)
          : layout->fixup_offset;
  const uint32_t program_dword_count =
      (uint32_t)((is_profile ? layout->profile_program_byte_length
                             : layout->program_byte_length) /
                 sizeof(uint32_t));
  const uint32_t fixup_entry_count =
      is_profile ? command_buffer->recording.profile.record_fixup_entry_count
                 : command_buffer->recording.record_fixup_entry_count;

  *out_state = (iree_hal_amdgpu_pm4_materialization_state_t){
      .program = is_profile ? &command_buffer->program_set
                                   .profile_plans[profile_plan_ordinal]
                                   .program
                            : &command_buffer->program_set.program,
      .fixup = &command_buffer->program_set.fixup,
      .profile =
          is_profile
              ? &command_buffer->program_set.profile_plans[profile_plan_ordinal]
              : NULL,
      .program_offset = program_offset,
      .vendor_packet_capabilities = command_buffer->vendor_packet_capabilities,
      .flags = flags,
  };
  iree_hal_amdgpu_pm4_dword_builder_initialize(command_buffer->host_allocator,
                                               &out_state->builders.program);
  iree_hal_amdgpu_pm4_dword_builder_borrow_storage(
      &out_state->builders.program,
      (uint32_t*)(materialization_base + program_offset), program_dword_count);
  iree_hal_amdgpu_pm4_byte_builder_initialize(command_buffer->host_allocator,
                                              &out_state->builders.template);
  iree_hal_amdgpu_pm4_byte_builder_borrow_storage(
      &out_state->builders.template,
      layout->template_byte_length != 0
          ? materialization_base + layout->template_offset
          : NULL,
      layout->template_byte_length);
  if (iree_any_bit_set(
          flags, IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_REUSE_TEMPLATES)) {
    out_state->builders.template.length = layout->template_byte_length;
  }
  iree_hal_amdgpu_pm4_fixup_entry_builder_initialize(
      command_buffer->host_allocator, &out_state->builders.fixup);
  iree_hal_amdgpu_pm4_fixup_entry_builder_borrow_storage(
      &out_state->builders.fixup,
      fixup_entry_count != 0
          ? (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*)(materialization_base +
                                                                fixup_offset)
          : NULL,
      fixup_entry_count);
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_copy_materialized_image_sync(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer, void* target,
    const void* source, iree_host_size_t byte_length) {
  if (byte_length == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hsa_memory_copy(IREE_LIBHSA(command_buffer->libhsa),
                                            target, source, byte_length));
  command_buffer->publish_stats.resident_copy_bytes += byte_length;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_launch_materialized_image_async(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer, void* target,
    const void* source, iree_host_size_t byte_length,
    hsa_signal_t* out_completion_signal) {
  IREE_ASSERT_ARGUMENT(out_completion_signal);
  *out_completion_signal = iree_hsa_signal_null();
  if (byte_length == 0) return iree_ok_status();

  hsa_signal_t completion_signal = {0};
  iree_status_t status = iree_hal_amdgpu_host_signal_pool_acquire(
      &command_buffer->resident_pool->copy_signal_pool, /*initial_value=*/1,
      &completion_signal);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_async_copy(
        IREE_LIBHSA(command_buffer->libhsa), target,
        command_buffer->resident_pool->device_agent, source,
        command_buffer->resident_pool->host_agent, byte_length,
        /*num_dep_signals=*/0, /*dep_signals=*/NULL, completion_signal);
  }
  if (iree_status_is_ok(status)) {
    *out_completion_signal = completion_signal;
  } else if (completion_signal.handle) {
    iree_hal_amdgpu_host_signal_pool_release(
        &command_buffer->resident_pool->copy_signal_pool, completion_signal);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_copy_materialized_image_async(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer, void* target,
    const void* source, iree_host_size_t byte_length) {
  hsa_signal_t completion_signal = {0};
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_launch_materialized_image_async(
          command_buffer, target, source, byte_length, &completion_signal);
  if (iree_status_is_ok(status) && completion_signal.handle) {
    const hsa_signal_value_t completion_value = iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(command_buffer->libhsa), completion_signal,
        HSA_SIGNAL_CONDITION_LT, /*compare_value=*/1, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    if (IREE_UNLIKELY(completion_value != 0)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "PM4 command-buffer async resident copy failed "
                                "with signal value %" PRId64,
                                (int64_t)completion_value);
    }
  }
  if (completion_signal.handle) {
    iree_hal_amdgpu_host_signal_pool_release(
        &command_buffer->resident_pool->copy_signal_pool, completion_signal);
  }
  if (iree_status_is_ok(status)) {
    command_buffer->publish_stats.resident_copy_bytes += byte_length;
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_copy_materialized_storage(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const uint8_t* materialization_base) {
  IREE_TRACE_ZONE_BEGIN(z0);
  void* const resident_base = command_buffer->resident_allocation->base;
  const iree_host_size_t byte_length = command_buffer->resident_byte_length;
  iree_status_t status = iree_ok_status();
  if (iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(command_buffer)) {
    if (iree_hal_amdgpu_pm4_command_buffer_uses_nonblocking_publication(
            command_buffer)) {
      hsa_signal_t completion_signal = {0};
      status =
          iree_hal_amdgpu_pm4_command_buffer_launch_materialized_image_async(
              command_buffer, resident_base, materialization_base, byte_length,
              &completion_signal);
      if (iree_status_is_ok(status) && completion_signal.handle) {
        iree_slim_mutex_lock(&command_buffer->publication_mutex);
        IREE_ASSERT(iree_hsa_signal_is_null(command_buffer->publication_signal),
                    "PM4 command buffer already has a pending publication");
        IREE_ASSERT(command_buffer->publication_reference_count == 0,
                    "PM4 command buffer already has pending publication "
                    "references");
        command_buffer->publication_signal = completion_signal;
        iree_slim_mutex_unlock(&command_buffer->publication_mutex);
        command_buffer->publish_stats.resident_copy_bytes += byte_length;
      }
    } else {
      status = iree_hal_amdgpu_pm4_command_buffer_copy_materialized_image_async(
          command_buffer, resident_base, materialization_base, byte_length);
    }
  } else {
    status = iree_hal_amdgpu_pm4_command_buffer_copy_materialized_image_sync(
        command_buffer, resident_base, materialization_base, byte_length);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static uint64_t iree_hal_amdgpu_pm4_command_buffer_host_record_bytes(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return command_buffer->recording.record_builder.length;
}

static iree_status_t iree_hal_amdgpu_pm4_align_host_size(
    iree_host_size_t value, iree_host_size_t alignment,
    iree_host_size_t* out_aligned_value) {
  *out_aligned_value = 0;
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 alignment must be a power-of-two");
  }
  if (IREE_UNLIKELY(value > UINTPTR_MAX - (alignment - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 alignment overflows host size");
  }
  *out_aligned_value = iree_host_align(value, alignment);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dispatch recording
//===----------------------------------------------------------------------===//

static bool iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(
    const iree_hal_dispatch_config_t config) {
  return config.workgroup_size[0] || config.workgroup_size[1] ||
         config.workgroup_size[2];
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_check_dispatch_flags(
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_arguments(flags) ||
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 command-buffer indirect dispatch is not "
                            "implemented");
  }
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported PM4 dispatch flags: 0x%" PRIx64,
                            flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_validate_dispatch_shape(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    const iree_hal_dispatch_config_t config) {
  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(config.workgroup_count[i] >
                      descriptor->maximum_workgroup_count[i])) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch grid dimension %" PRIhsz
          " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
          i, config.workgroup_count[i],
          descriptor->kernel_args.workgroup_size[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_resolve_dispatch_thread_count(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    const iree_hal_dispatch_config_t config,
    uint32_t out_dispatch_thread_count[3]) {
  for (iree_host_size_t i = 0; i < 3; ++i) {
    const uint64_t thread_count = (uint64_t)config.workgroup_count[i] *
                                  descriptor->kernel_args.workgroup_size[i];
    if (IREE_UNLIKELY(thread_count > UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 DISPATCH_DIRECT dimension %" PRIhsz
          " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
          i, config.workgroup_count[i],
          descriptor->kernel_args.workgroup_size[i]);
    }
    out_dispatch_thread_count[i] = (uint32_t)thread_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_resolve_buffer_ref(
    const iree_hal_buffer_ref_t* buffer_ref, uint64_t* out_device_pointer) {
  *out_device_pointer = 0;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref->buffer);
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer buffer reference must be backed by an AMDGPU "
        "allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer_ref->buffer), buffer_ref->offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer buffer reference device pointer offset overflows");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer buffer reference device pointer offset exceeds "
        "host pointer size");
  }
  *out_device_pointer =
      (uint64_t)((uintptr_t)device_ptr + (uintptr_t)device_offset);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_write_template(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* fixup_builder,
    iree_host_size_t resident_template_offset,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_kernarg_layout_t* layout, uint32_t template_offset,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_amdgpu_pm4_binding_record_t* binding_records,
    uint32_t binding_record_count, void* hostcall_buffer,
    uint8_t* template_bytes) {
  uint64_t* binding_ptrs = binding_record_count
                               ? (uint64_t*)iree_alloca(binding_record_count *
                                                        sizeof(binding_ptrs[0]))
                               : NULL;
  for (uint32_t i = 0; i < binding_record_count; ++i) {
    const iree_hal_amdgpu_pm4_binding_record_t* binding_record =
        &binding_records[i];
    if (IREE_UNLIKELY(binding_record->target_offset < template_offset)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 command-buffer binding target offset precedes template");
    }
    if (iree_any_bit_set(binding_record->flags,
                         IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC)) {
      if (IREE_UNLIKELY(resident_template_offset >
                        UINT32_MAX - binding_record->target_offset)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "PM4 command-buffer dynamic template fixup offset overflows");
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_fixup_entry_builder_append(
          fixup_builder,
          (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
              .target_offset = (uint32_t)resident_template_offset +
                               binding_record->target_offset,
              .binding_slot = binding_record->binding_slot,
              .binding_offset = binding_record->value,
          }));
      binding_ptrs[i] = 0;
      continue;
    }
    binding_ptrs[i] = binding_record->value;
  }

  iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
      layout, binding_ptrs, constants, template_bytes);
  if (iree_any_bit_set(layout->flags,
                       IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS)) {
    iree_amdgpu_kernel_implicit_args_t* implicit_args =
        (iree_amdgpu_kernel_implicit_args_t*)(template_bytes +
                                              layout
                                                  ->implicit_args_byte_offset);
    iree_hal_amdgpu_device_dispatch_initialize_implicit_args(
        kernel_args, config.workgroup_count,
        config.dynamic_workgroup_local_memory, hostcall_buffer, implicit_args);
  }
  return iree_ok_status();
}

static const uint8_t* iree_hal_amdgpu_pm4_dispatch_record_constants(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  return (const uint8_t*)record + sizeof(*record);
}

static const iree_hal_amdgpu_pm4_binding_record_t*
iree_hal_amdgpu_pm4_dispatch_record_bindings(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  const uintptr_t constants_end =
      (uintptr_t)iree_hal_amdgpu_pm4_dispatch_record_constants(record) +
      record->constant_length;
  return (const iree_hal_amdgpu_pm4_binding_record_t*)iree_host_align(
      constants_end, iree_alignof(iree_hal_amdgpu_pm4_binding_record_t));
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_append_preloaded_binding_fixups(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* fixup_builder,
    iree_host_size_t program_offset,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state,
    const iree_hal_amdgpu_pm4_binding_record_t* binding_records,
    uint32_t binding_record_count, uint32_t template_offset,
    uint32_t user_data_program_dword_offset) {
  if (launch_state->kernarg_preload_dword_count == 0) return iree_ok_status();
  for (uint32_t i = 0; i < binding_record_count; ++i) {
    const iree_hal_amdgpu_pm4_binding_record_t* binding_record =
        &binding_records[i];
    if (!iree_any_bit_set(binding_record->flags,
                          IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC)) {
      continue;
    }
    if (IREE_UNLIKELY(binding_record->target_offset < template_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 command-buffer dynamic binding target offset precedes template");
    }
    const uint32_t binding_kernarg_offset =
        binding_record->target_offset - template_offset;
    bool is_preloaded = false;
    uint32_t preload_dword_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
            launch_state, binding_kernarg_offset, sizeof(uint64_t),
            &is_preloaded, &preload_dword_offset));
    if (!is_preloaded) continue;

    const uint32_t user_data_payload_dword_offset =
        launch_state->kernarg_preload_user_data_offset + preload_dword_offset;
    if (IREE_UNLIKELY(user_data_program_dword_offset >
                      UINT32_MAX - 2u - user_data_payload_dword_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 command-buffer dynamic user-data fixup dword offset overflows");
    }
    const uint32_t target_dword_offset =
        user_data_program_dword_offset + 2u + user_data_payload_dword_offset;
    if (IREE_UNLIKELY(target_dword_offset > UINT32_MAX / sizeof(uint32_t))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 command-buffer dynamic user-data fixup byte offset overflows");
    }
    iree_host_size_t target_offset = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_add(program_offset,
                                        target_dword_offset * sizeof(uint32_t),
                                        &target_offset) ||
            target_offset > UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 command-buffer dynamic user-data fixup byte offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_fixup_entry_builder_append(
        fixup_builder, (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
                           .target_offset = (uint32_t)target_offset,
                           .binding_slot = binding_record->binding_slot,
                           .binding_offset = binding_record->value,
                       }));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_resource_t* resource) {
  if (!resource) return iree_ok_status();
  iree_hal_amdgpu_pm4_retained_resource_table_t* retained_resources =
      &command_buffer->retained_resources;
  if (iree_hal_amdgpu_pm4_retained_resource_table_contains(retained_resources,
                                                           resource)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_retained_resource_table_reserve(
      command_buffer->host_allocator, retained_resources,
      retained_resources->count + 1));
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set,
                                   /*count=*/1, &resource));
  iree_hal_amdgpu_pm4_retained_resource_table_insert_prepared(
      retained_resources, resource);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_retain_dispatch(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, iree_hal_buffer_ref_list_t bindings) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_ensure_resource_set(command_buffer));
  if (!command_buffer->resource_set) return iree_ok_status();

  if (command_buffer->last_retained_executable != executable) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
            command_buffer, (iree_hal_resource_t*)executable));
    command_buffer->last_retained_executable = executable;
  }
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
            command_buffer, (iree_hal_resource_t*)bindings.values[i].buffer));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_append_dispatch_record(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    uint64_t executable_id, uint32_t command_index, uint32_t export_ordinal,
    const iree_hal_dispatch_config_t config,
    const uint32_t dispatch_thread_count[3], iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_pm4_dispatch_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope) {
  const iree_hal_amdgpu_kernarg_layout_t* layout = descriptor->kernarg_layout;
  if (IREE_UNLIKELY(constants.data_length > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dispatch constants exceed uint32_t storage");
  }
  if (IREE_UNLIKELY(bindings.count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dispatch bindings exceed uint32_t storage");
  }
  if (IREE_UNLIKELY(bindings.count != layout->binding_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 command-buffer dispatch binding count %" PRIhsz
                            " does not match executable binding count %u",
                            bindings.count, (uint32_t)layout->binding_count);
  }
  if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer dispatch bindings must be non-null when count is "
        "non-zero");
  }
  const iree_host_size_t expected_constant_bytes = layout->constant_byte_length;
  if (IREE_UNLIKELY(constants.data_length != expected_constant_bytes)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer dispatch constant byte length mismatch; expected "
        "%" PRIhsz " but got %" PRIhsz,
        expected_constant_bytes, constants.data_length);
  }

  iree_host_size_t template_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_align_host_size(
      command_buffer->recording.record_template_byte_length,
      layout->kernarg_alignment, &template_offset));
  if (IREE_UNLIKELY(template_offset > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer kernarg template offset exceeds uint32_t storage");
  }
  iree_host_size_t new_record_template_byte_length = 0;
  if (!iree_host_size_checked_add(template_offset, layout->kernarg_byte_length,
                                  &new_record_template_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer kernarg template storage overflows");
  }
  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &descriptor->pm4_launch_state;
  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  uint32_t dynamic_template_fixup_count = 0;
  uint32_t dynamic_preload_fixup_count = 0;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    if (!binding->buffer) {
      if (IREE_UNLIKELY(binding->buffer_slot == UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "PM4 command-buffer dispatch binding slot %u exceeds binding "
            "count storage",
            binding->buffer_slot);
      }
      command_buffer->base.binding_count = iree_max(
          command_buffer->base.binding_count, binding->buffer_slot + 1);
      if (IREE_UNLIKELY(dynamic_template_fixup_count == UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "PM4 command-buffer dynamic fixup entry count exceeds uint32_t "
            "storage");
      }
      ++dynamic_template_fixup_count;
      const uint32_t binding_target_offset =
          (uint32_t)binding_slots[i].target_qword_index * sizeof(uint64_t);
      bool is_preloaded = false;
      uint32_t preload_dword_offset = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
              launch_state, binding_target_offset, sizeof(uint64_t),
              &is_preloaded, &preload_dword_offset));
      if (is_preloaded) {
        if (IREE_UNLIKELY(dynamic_preload_fixup_count == UINT32_MAX)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "PM4 command-buffer dynamic fixup entry count exceeds uint32_t "
              "storage");
        }
        ++dynamic_preload_fixup_count;
      }
    }
  }
  if (IREE_UNLIKELY(dynamic_template_fixup_count >
                    UINT32_MAX - dynamic_preload_fixup_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dynamic fixup entry count exceeds uint32_t "
        "storage");
  }
  const uint32_t dynamic_fixup_count =
      dynamic_template_fixup_count + dynamic_preload_fixup_count;
  if (IREE_UNLIKELY(dynamic_fixup_count >
                    UINT32_MAX -
                        command_buffer->recording.record_fixup_entry_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dynamic fixup entry count exceeds uint32_t "
        "storage");
  }
  const uint32_t new_record_fixup_entry_count =
      command_buffer->recording.record_fixup_entry_count + dynamic_fixup_count;

  uint32_t shared_dispatch_dwords =
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  if (iree_any_bit_set(
          flags, IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    shared_dispatch_dwords += iree_hal_amdgpu_pm4_barrier_dword_count(
        command_buffer->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION, barrier_acquire_scope,
        barrier_release_scope);
  }
  const bool needs_dispatch_setup =
      !command_buffer->recording.has_previous_launch_state ||
      memcmp(&command_buffer->recording.previous_launch_state, launch_state,
             sizeof(*launch_state)) != 0;
  if (needs_dispatch_setup) {
    shared_dispatch_dwords += descriptor->pm4_setup_dword_count;
  }
  if (launch_state->user_data_dword_count != 0) {
    shared_dispatch_dwords += 2u + launch_state->user_data_dword_count;
  }
  uint32_t dispatch_dwords = shared_dispatch_dwords;
  if (iree_any_bit_set(
          flags, IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER)) {
    dispatch_dwords += iree_hal_amdgpu_pm4_barrier_dword_count(
        command_buffer->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE);
  }
  if (IREE_UNLIKELY(dispatch_dwords >
                    IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                        command_buffer->recording.record_ib_dword_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires more than the PM4-IB maximum %u dwords",
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }
  const uint32_t new_record_ib_dword_count =
      command_buffer->recording.record_ib_dword_count + dispatch_dwords;

  uint32_t new_profile_program_dword_count =
      command_buffer->recording.profile.record_program_dword_count;
  uint32_t new_profile_fixup_entry_count =
      command_buffer->recording.profile.record_fixup_entry_count;
  if (iree_hal_amdgpu_pm4_command_buffer_materializes_profile_dispatch_timestamps(
          command_buffer)) {
    const uint32_t profile_timestamp_dwords =
        2u * IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT;
    const uint32_t profile_timestamp_alignment_dwords = 6u;
    const uint32_t profile_dispatch_flush_dwords =
        iree_hal_amdgpu_pm4_barrier_dword_count(
            command_buffer->vendor_packet_capabilities,
            IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
            IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE);
    const uint32_t profile_fixup_barrier_dwords =
        iree_any_bit_set(
            flags,
            IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER)
            ? iree_hal_amdgpu_pm4_barrier_dword_count(
                  command_buffer->vendor_packet_capabilities,
                  IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB,
                  IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE)
            : 0u;
    if (IREE_UNLIKELY(profile_dispatch_flush_dwords == 0)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "PM4 dispatch profiling requires execution-flush barrier support");
    } else if (
        IREE_UNLIKELY(
            profile_fixup_barrier_dwords == 0 &&
            iree_any_bit_set(
                flags,
                IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER))) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "PM4 dispatch profiling requires fixup-to-IB barrier support");
    }
    const uint32_t profile_dispatch_dwords =
        shared_dispatch_dwords + profile_timestamp_dwords +
        profile_timestamp_alignment_dwords + profile_dispatch_flush_dwords +
        profile_fixup_barrier_dwords;
    if (IREE_UNLIKELY(
            command_buffer->recording.profile.record_program_dword_count >
            IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT - profile_dispatch_dwords)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile command buffer requires more than the PM4-IB maximum %u "
          "dwords",
          IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
    }
    new_profile_program_dword_count += profile_dispatch_dwords;
    if (IREE_UNLIKELY(
            dynamic_fixup_count > UINT32_MAX - 2u ||
            command_buffer->recording.profile.record_fixup_entry_count >
                UINT32_MAX - dynamic_fixup_count - 2u)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile command-buffer fixup entry count exceeds uint32_t "
          "storage");
    }
    new_profile_fixup_entry_count += dynamic_fixup_count + 2u;
  }

  const iree_host_size_t binding_record_count =
      (iree_host_size_t)layout->binding_count;
  iree_host_size_t binding_record_bytes = 0;
  if (!iree_host_size_checked_mul(binding_record_count,
                                  sizeof(iree_hal_amdgpu_pm4_binding_record_t),
                                  &binding_record_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 binding records overflow");
  }

  iree_host_size_t constants_end = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_amdgpu_pm4_dispatch_record_t),
                                  constants.data_length, &constants_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch record constants overflow");
  }
  const iree_host_size_t bindings_offset = iree_host_align(
      constants_end, iree_alignof(iree_hal_amdgpu_pm4_binding_record_t));
  iree_host_size_t record_length = 0;
  if (!iree_host_size_checked_add(bindings_offset, binding_record_bytes,
                                  &record_length) ||
      record_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch record length overflows");
  }

  uint8_t* record_bytes = NULL;
  const iree_host_size_t record_builder_base_length =
      command_buffer->recording.record_builder.length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_record(
      &command_buffer->recording.record_builder, record_length, &record_bytes));
  memset(record_bytes, 0, record_length);
  iree_hal_amdgpu_pm4_dispatch_record_t* record =
      (iree_hal_amdgpu_pm4_dispatch_record_t*)record_bytes;
  record->header.length = (uint32_t)record_length;
  record->header.opcode = IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH;
  record->descriptor = descriptor;
  record->executable_id = executable_id;
  record->workgroup_count[0] = config.workgroup_count[0];
  record->workgroup_count[1] = config.workgroup_count[1];
  record->workgroup_count[2] = config.workgroup_count[2];
  record->dispatch_thread_count[0] = dispatch_thread_count[0];
  record->dispatch_thread_count[1] = dispatch_thread_count[1];
  record->dispatch_thread_count[2] = dispatch_thread_count[2];
  record->command_index = command_index;
  record->export_ordinal = export_ordinal;
  record->template_offset = (uint32_t)template_offset;
  record->template_length = (uint32_t)layout->kernarg_byte_length;
  record->constant_length = (uint32_t)constants.data_length;
  record->binding_record_count = (uint32_t)binding_record_count;
  record->flags = flags;
  record->barrier_acquire_scope = barrier_acquire_scope;
  record->barrier_release_scope = barrier_release_scope;
  if (constants.data_length > 0) {
    memcpy(record_bytes + sizeof(*record), constants.data,
           constants.data_length);
  }
  if (binding_record_bytes > 0) {
    iree_hal_amdgpu_pm4_binding_record_t* binding_records =
        (iree_hal_amdgpu_pm4_binding_record_t*)(record_bytes + bindings_offset);
    memset(binding_records, 0, binding_record_bytes);
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      const iree_hal_buffer_ref_t* binding = &bindings.values[i];
      iree_hal_amdgpu_pm4_binding_record_t* binding_record =
          &binding_records[i];
      const uint32_t binding_target_offset =
          (uint32_t)binding_slots[i].target_qword_index * sizeof(uint64_t);
      if (IREE_UNLIKELY((uint32_t)template_offset >
                        UINT32_MAX - binding_target_offset)) {
        command_buffer->recording.record_builder.length =
            record_builder_base_length;
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "PM4 command-buffer binding target offset overflows");
      }
      binding_record->target_offset =
          (uint32_t)template_offset + binding_target_offset;
      if (!binding->buffer) {
        binding_record->value = binding->offset;
        binding_record->binding_slot = binding->buffer_slot;
        binding_record->flags = IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC;
        continue;
      }
      iree_status_t status =
          iree_hal_amdgpu_pm4_command_buffer_resolve_buffer_ref(
              binding, &binding_record->value);
      if (!iree_status_is_ok(status)) {
        command_buffer->recording.record_builder.length =
            record_builder_base_length;
        return status;
      }
    }
  }
  command_buffer->recording.record_template_byte_length =
      new_record_template_byte_length;
  command_buffer->recording.record_fixup_entry_count =
      new_record_fixup_entry_count;
  command_buffer->recording.record_ib_dword_count = new_record_ib_dword_count;
  command_buffer->recording.profile.record_program_dword_count =
      new_profile_program_dword_count;
  command_buffer->recording.profile.record_fixup_entry_count =
      new_profile_fixup_entry_count;
  if (needs_dispatch_setup) {
    command_buffer->recording.previous_launch_state = *launch_state;
    command_buffer->recording.has_previous_launch_state = true;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_materialize_dispatch(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_amdgpu_pm4_materialization_state_t* state,
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
      record->descriptor;
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &descriptor->kernel_args;
  const iree_hal_amdgpu_kernarg_layout_t* layout = descriptor->kernarg_layout;
  iree_const_byte_span_t constants = iree_make_const_byte_span(
      iree_hal_amdgpu_pm4_dispatch_record_constants(record),
      record->constant_length);
  const iree_hal_amdgpu_pm4_binding_record_t* binding_records =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  const iree_hal_dispatch_config_t config = {
      .workgroup_count = {record->workgroup_count[0],
                          record->workgroup_count[1],
                          record->workgroup_count[2]},
  };

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    const uint32_t dword_count_before = state->builders.program.dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        &state->builders.program, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope));
    command_buffer->publish_stats.execution_barrier_dwords +=
        state->builders.program.dword_count - dword_count_before;
  }

  uint8_t* template_bytes = NULL;
  if (iree_any_bit_set(
          state->flags,
          IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_REUSE_TEMPLATES)) {
    if (IREE_UNLIKELY(
            record->template_offset > state->builders.template.length ||
            layout->kernarg_byte_length >
                state->builders.template.length - record->template_offset)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 dispatch template lies outside populated resident storage");
    }
    template_bytes = state->builders.template.bytes + record->template_offset;
  } else {
    uint32_t template_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_aligned(
        &state->builders.template, layout->kernarg_alignment,
        layout->kernarg_byte_length, &template_offset, &template_bytes));
    if (IREE_UNLIKELY(template_offset != record->template_offset)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 template materialization offset changed");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_write_template(
        &state->builders.fixup, command_buffer->resident_template_offset,
        kernel_args, layout, record->template_offset, config, constants,
        binding_records, record->binding_record_count,
        command_buffer->hostcall_buffer, template_bytes));
  }

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER)) {
    if (IREE_UNLIKELY(state->has_emitted_fixup_barrier)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 program contains multiple fixup barriers");
    }
    const uint32_t dword_count_before = state->builders.program.dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        &state->builders.program, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE));
    command_buffer->publish_stats.fixup_barrier_dwords +=
        state->builders.program.dword_count - dword_count_before;
    state->has_emitted_fixup_barrier = true;
  }

  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &descriptor->pm4_launch_state;
  if (!state->has_previous_launch_state ||
      memcmp(&state->previous_launch_state, launch_state,
             sizeof(*launch_state)) != 0) {
    const uint32_t dword_count_before = state->builders.program.dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
        &state->builders.program, descriptor->pm4_setup_dwords,
        descriptor->pm4_setup_dword_count));
    command_buffer->publish_stats.dispatch_setup_dwords +=
        state->builders.program.dword_count - dword_count_before;
    state->previous_launch_state = *launch_state;
    state->has_previous_launch_state = true;
  }

  if (IREE_UNLIKELY(!state->fixup->template_base)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command-buffer dispatch requires resident kernarg storage");
  }
  if (IREE_UNLIKELY(launch_state->user_data_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command-buffer dispatch has no kernarg user-data dwords");
  }
  const uintptr_t kernarg_address =
      (uintptr_t)state->fixup->template_base + record->template_offset;
  uint32_t dword_count_before = state->builders.program.dword_count;
  const uint32_t user_data_program_dword_offset =
      state->builders.program.dword_count;
  const uint8_t* kernarg_preload_data =
      launch_state->kernarg_preload_dword_count != 0
          ? template_bytes +
                launch_state->kernarg_preload_dword_offset * sizeof(uint32_t)
          : NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_user_data(
      &state->builders.program, launch_state, kernarg_address,
      kernarg_preload_data));
  command_buffer->publish_stats.dispatch_user_data_dwords +=
      state->builders.program.dword_count - dword_count_before;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_append_preloaded_binding_fixups(
          &state->builders.fixup, state->program_offset, launch_state,
          binding_records, record->binding_record_count,
          record->template_offset, user_data_program_dword_offset));
  dword_count_before = state->builders.program.dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
      &state->builders.program, record->dispatch_thread_count,
      launch_state->dispatch_initiator));
  command_buffer->publish_stats.dispatch_direct_dwords +=
      state->builders.program.dword_count - dword_count_before;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_materialize_builtin_record(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_amdgpu_pm4_materialization_state_t* state,
    const iree_hal_amdgpu_pm4_command_record_header_t* record) {
  const bool is_profile = iree_any_bit_set(
      state->flags, IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_PROFILE);
  iree_hal_amdgpu_pm4_command_record_flags_t record_flags = 0;
  switch ((iree_hal_amdgpu_pm4_command_record_opcode_t)record->opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
      record_flags =
          ((const iree_hal_amdgpu_pm4_atomic_record_t*)record)->flags;
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY:
      record_flags =
          ((const iree_hal_amdgpu_pm4_transfer_record_t*)record)->flags;
      break;
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown builtin PM4 command record opcode %u",
                              record->opcode);
  }
  const iree_hal_amdgpu_pm4_command_record_flags_t fixup_barrier_flag =
      is_profile ? IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_PROFILE_FIXUP_BARRIER
                 : IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_FIXUP_BARRIER;
  if (iree_any_bit_set(record_flags, fixup_barrier_flag) &&
      IREE_UNLIKELY(state->has_emitted_fixup_barrier)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 program contains multiple fixup barriers");
  }
  iree_hal_amdgpu_pm4_command_materialization_state_t builtin_state = {
      .dword_builder = &state->builders.program,
      .template_builder = &state->builders.template,
      .fixup_builder = &state->builders.fixup,
      .template_base = state->fixup->template_base,
      .resident_template_offset = command_buffer->resident_template_offset,
      .program_offset = state->program_offset,
      .vendor_packet_capabilities = state->vendor_packet_capabilities,
      .previous_launch_state = state->previous_launch_state,
      .has_previous_launch_state = state->has_previous_launch_state,
      .flags = is_profile
                   ? IREE_HAL_AMDGPU_PM4_COMMAND_MATERIALIZATION_FLAG_PROFILE
                   : IREE_HAL_AMDGPU_PM4_COMMAND_MATERIALIZATION_FLAG_NONE,
  };
  iree_hal_amdgpu_pm4_command_materialization_stats_t stats = {0};
  switch ((iree_hal_amdgpu_pm4_command_record_opcode_t)record->opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW: {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_record_materialize(
          (const iree_hal_amdgpu_pm4_atomic_record_t*)record,
          command_buffer->atomic_context, &builtin_state,
          is_profile ? NULL : &stats));
      break;
    }
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY: {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_record_materialize(
          (const iree_hal_amdgpu_pm4_transfer_record_t*)record,
          command_buffer->buffer_transfer_pm4_context, &builtin_state,
          is_profile ? NULL : &stats));
      break;
    }
    default:
      break;
  }
  state->previous_launch_state = builtin_state.previous_launch_state;
  state->has_previous_launch_state = builtin_state.has_previous_launch_state;
  state->has_emitted_fixup_barrier |=
      iree_any_bit_set(record_flags, fixup_barrier_flag);
  command_buffer->publish_stats.execution_barrier_dwords +=
      stats.execution_barrier_dwords;
  command_buffer->publish_stats.fixup_barrier_dwords +=
      stats.fixup_barrier_dwords;
  command_buffer->publish_stats.dispatch_setup_dwords +=
      stats.dispatch_setup_dwords;
  command_buffer->publish_stats.dispatch_user_data_dwords +=
      stats.dispatch_user_data_dwords;
  command_buffer->publish_stats.dispatch_direct_dwords +=
      stats.dispatch_direct_dwords;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_append_profile_template_fixups(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_amdgpu_pm4_materialization_state_t* state,
    const iree_hal_amdgpu_pm4_binding_record_t* binding_records,
    uint32_t binding_record_count) {
  for (uint32_t i = 0; i < binding_record_count; ++i) {
    const iree_hal_amdgpu_pm4_binding_record_t* binding_record =
        &binding_records[i];
    if (!iree_any_bit_set(binding_record->flags,
                          IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC)) {
      continue;
    }
    if (IREE_UNLIKELY(command_buffer->resident_template_offset >
                      UINT32_MAX - binding_record->target_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile command-buffer dynamic template fixup offset "
          "overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_fixup_entry_builder_append(
        &state->builders.fixup,
        (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
            .target_offset =
                (uint32_t)command_buffer->resident_template_offset +
                binding_record->target_offset,
            .binding_slot = binding_record->binding_slot,
            .binding_offset = binding_record->value,
        }));
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_append_profile_timestamp_fixup(
    iree_hal_amdgpu_pm4_materialization_state_t* state,
    uint32_t target_dword_offset, uint32_t binding_slot) {
  iree_host_size_t target_byte_offset = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(
              state->program_offset,
              (iree_host_size_t)target_dword_offset * sizeof(uint32_t),
              &target_byte_offset) ||
          target_byte_offset > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 profile timestamp fixup target offset exceeds uint32_t storage");
  }
  return iree_hal_amdgpu_pm4_fixup_entry_builder_append(
      &state->builders.fixup,
      (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
          .target_offset = (uint32_t)target_byte_offset,
          .binding_slot = binding_slot,
          .binding_offset = 0,
      });
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_materialize_profile_dispatch(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_amdgpu_pm4_materialization_state_t* state,
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
      record->descriptor;
  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &descriptor->pm4_launch_state;
  const iree_hal_amdgpu_pm4_binding_record_t* binding_records =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        &state->builders.program, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope));
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_append_profile_template_fixups(
          command_buffer, state, binding_records,
          record->binding_record_count));

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER)) {
    if (IREE_UNLIKELY(state->has_emitted_fixup_barrier)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 profile program contains multiple fixup barriers");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        &state->builders.program, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE));
    state->has_emitted_fixup_barrier = true;
  }

  if (!state->has_previous_launch_state ||
      memcmp(&state->previous_launch_state, launch_state,
             sizeof(*launch_state)) != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
        &state->builders.program, descriptor->pm4_setup_dwords,
        descriptor->pm4_setup_dword_count));
    state->previous_launch_state = *launch_state;
    state->has_previous_launch_state = true;
  }

  if (IREE_UNLIKELY(!state->fixup->template_base)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 profile command-buffer dispatch requires resident kernarg "
        "storage");
  }
  if (IREE_UNLIKELY(launch_state->user_data_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 profile command-buffer dispatch has no kernarg user-data dwords");
  }
  const uintptr_t kernarg_address =
      (uintptr_t)state->fixup->template_base + record->template_offset;
  const uint32_t user_data_program_dword_offset =
      state->builders.program.dword_count;
  const uint8_t* template_bytes =
      state->builders.template.bytes + record->template_offset;
  const uint8_t* kernarg_preload_data =
      launch_state->kernarg_preload_dword_count != 0
          ? template_bytes +
                launch_state->kernarg_preload_dword_offset * sizeof(uint32_t)
          : NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_user_data(
      &state->builders.program, launch_state, kernarg_address,
      kernarg_preload_data));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_append_preloaded_binding_fixups(
          &state->builders.fixup, state->program_offset, launch_state,
          binding_records, record->binding_record_count,
          record->template_offset, user_data_program_dword_offset));

  if (IREE_UNLIKELY(record->command_index >= state->profile->operation_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 profile command index exceeds retained dispatch count");
  }
  const uint32_t timestamp_binding_slot =
      state->profile->timestamp_binding_base + record->command_index * 2u;
  uint32_t timestamp_target_dword_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_align_timestamp_target(
      &state->builders.program, state->program_offset));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_timestamp(
      &state->builders.program, command_buffer->pm4_timestamp_strategy,
      &state->profile->dummy_ticks->start_tick,
      &timestamp_target_dword_offset));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_append_profile_timestamp_fixup(
          state, timestamp_target_dword_offset, timestamp_binding_slot));

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
      &state->builders.program, record->dispatch_thread_count,
      launch_state->dispatch_initiator));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
      &state->builders.program, state->vendor_packet_capabilities,
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE));

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_align_timestamp_target(
      &state->builders.program, state->program_offset));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_timestamp(
      &state->builders.program, command_buffer->pm4_timestamp_strategy,
      &state->profile->dummy_ticks->end_tick, &timestamp_target_dword_offset));
  return iree_hal_amdgpu_pm4_command_buffer_append_profile_timestamp_fixup(
      state, timestamp_target_dword_offset, timestamp_binding_slot + 1u);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_materialize_records(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_amdgpu_pm4_materialization_state_t* state) {
  const uint8_t* cursor = command_buffer->recording.record_builder.bytes;
  const uint8_t* const end =
      cursor + command_buffer->recording.record_builder.length;
  while (cursor < end) {
    if (IREE_UNLIKELY((iree_host_size_t)(end - cursor) <
                      sizeof(iree_hal_amdgpu_pm4_command_record_header_t))) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 command record is truncated");
    }
    const iree_hal_amdgpu_pm4_command_record_header_t* header =
        (const iree_hal_amdgpu_pm4_command_record_header_t*)cursor;
    if (IREE_UNLIKELY(header->length == 0 ||
                      (iree_host_size_t)(end - cursor) < header->length)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 command record length is invalid");
    }
    switch ((iree_hal_amdgpu_pm4_command_record_opcode_t)header->opcode) {
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH: {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_pm4_command_buffer_materialize_dispatch(
                command_buffer, state,
                (const iree_hal_amdgpu_pm4_dispatch_record_t*)cursor));
        break;
      }
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY: {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_pm4_command_buffer_materialize_builtin_record(
                command_buffer, state, header));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "unknown PM4 command record opcode %u",
                                header->opcode);
    }
    cursor += header->length;
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_materialize_profile_records(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_amdgpu_pm4_materialization_state_t* state) {
  if (command_buffer->recording.profile.record_program_dword_count == 0) {
    return iree_ok_status();
  }

  const uint8_t* cursor = command_buffer->recording.record_builder.bytes;
  const uint8_t* const end =
      cursor + command_buffer->recording.record_builder.length;
  while (cursor < end) {
    if (IREE_UNLIKELY((iree_host_size_t)(end - cursor) <
                      sizeof(iree_hal_amdgpu_pm4_command_record_header_t))) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 profile command record is truncated");
    }
    const iree_hal_amdgpu_pm4_command_record_header_t* header =
        (const iree_hal_amdgpu_pm4_command_record_header_t*)cursor;
    if (IREE_UNLIKELY(header->length == 0 ||
                      (iree_host_size_t)(end - cursor) < header->length)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 profile command record length is invalid");
    }
    switch ((iree_hal_amdgpu_pm4_command_record_opcode_t)header->opcode) {
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH: {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_pm4_command_buffer_materialize_profile_dispatch(
                command_buffer, state,
                (const iree_hal_amdgpu_pm4_dispatch_record_t*)cursor));
        break;
      }
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY: {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_pm4_command_buffer_materialize_builtin_record(
                command_buffer, state, header));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "unknown PM4 profile command record opcode %u",
                                header->opcode);
    }
    cursor += header->length;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
      &state->builders.program, state->vendor_packet_capabilities,
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
      command_buffer->recording.barrier_state.acquire_scope,
      command_buffer->recording.barrier_state.release_scope));
  if (IREE_UNLIKELY(state->builders.program.dword_count >
                    state->profile->program.dword_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 profile IB materialization exceeded reserved capacity");
  }
  state->profile->program.dword_count = state->builders.program.dword_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_materialize_program_set(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_pm4_command_program_layout_t* layout,
    uint8_t* materialization_base) {
  iree_hal_amdgpu_pm4_materialization_state_t state;
  iree_hal_amdgpu_pm4_command_buffer_initialize_materialization_state(
      command_buffer, layout, materialization_base,
      /*profile_plan_ordinal=*/0, IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_NONE,
      &state);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_materialize_records(
      command_buffer, &state));
  if (IREE_UNLIKELY(state.builders.template.length !=
                    command_buffer->recording.record_template_byte_length)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 template materialization produced %" PRIhsz
        " bytes, expected %" PRIhsz,
        state.builders.template.length,
        command_buffer->recording.record_template_byte_length);
  }
  if (IREE_UNLIKELY(state.builders.fixup.count !=
                    command_buffer->recording.record_fixup_entry_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 fixup materialization produced %u entries, expected %u",
        state.builders.fixup.count,
        command_buffer->recording.record_fixup_entry_count);
  }
  if (IREE_UNLIKELY(state.builders.program.dword_count !=
                    command_buffer->recording.record_ib_dword_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 IB materialization produced %u dwords, expected %u",
        state.builders.program.dword_count,
        command_buffer->recording.record_ib_dword_count);
  }
  const uint32_t dword_count_before = state.builders.program.dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
      &state.builders.program, state.vendor_packet_capabilities,
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
      command_buffer->recording.barrier_state.acquire_scope,
      command_buffer->recording.barrier_state.release_scope));
  command_buffer->publish_stats.terminal_barrier_dwords +=
      state.builders.program.dword_count - dword_count_before;
  if (IREE_UNLIKELY(state.builders.program.dword_count !=
                    state.program->dword_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL, "PM4 resident IB contains %u dwords, expected %u",
        state.builders.program.dword_count, state.program->dword_count);
  }

  for (uint32_t profile_plan_ordinal = 0;
       profile_plan_ordinal < command_buffer->program_set.profile_plan_count;
       ++profile_plan_ordinal) {
    iree_hal_amdgpu_pm4_command_buffer_initialize_materialization_state(
        command_buffer, layout, materialization_base, profile_plan_ordinal,
        IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_PROFILE |
            IREE_HAL_AMDGPU_PM4_MATERIALIZATION_FLAG_REUSE_TEMPLATES,
        &state);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_materialize_profile_records(
            command_buffer, &state));
    if (IREE_UNLIKELY(
            state.builders.fixup.count !=
            command_buffer->recording.profile.record_fixup_entry_count)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 profile fixup materialization produced %u entries, expected %u",
          state.builders.fixup.count,
          command_buffer->recording.profile.record_fixup_entry_count);
    }
  }
  return iree_ok_status();
}

static iree_hal_profile_command_operation_flags_t
iree_hal_amdgpu_pm4_command_buffer_profile_dispatch_binding_flags(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  iree_hal_profile_command_operation_flags_t flags =
      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_NONE;
  const iree_hal_amdgpu_pm4_binding_record_t* binding_records =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  for (uint32_t i = 0; i < record->binding_record_count; ++i) {
    flags |= iree_any_bit_set(binding_records[i].flags,
                              IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC)
                 ? IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS
                 : IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
  }
  return flags;
}

static void iree_hal_amdgpu_pm4_command_buffer_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_dispatch_record_t* dispatch_record,
    iree_hal_profile_command_operation_record_t* out_record) {
  iree_hal_profile_command_operation_record_t record =
      iree_hal_profile_command_operation_record_default();
  record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  record.command_buffer_id = command_buffer_id;
  record.command_index = dispatch_record->command_index;
  if (iree_any_bit_set(
          dispatch_record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
  }
  const iree_hal_profile_command_operation_flags_t binding_flags =
      iree_hal_amdgpu_pm4_command_buffer_profile_dispatch_binding_flags(
          dispatch_record);
  record.flags |= binding_flags;
  if (!iree_any_bit_set(
          binding_flags,
          IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS)) {
    record.flags |=
        IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_PREPUBLISHED_ARGUMENTS;
  }
  record.executable_id = dispatch_record->executable_id;
  record.function_ordinal = dispatch_record->export_ordinal;
  record.binding_count = dispatch_record->binding_record_count;
  record.workgroup_count[0] = dispatch_record->workgroup_count[0];
  record.workgroup_count[1] = dispatch_record->workgroup_count[1];
  record.workgroup_count[2] = dispatch_record->workgroup_count[2];
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &dispatch_record->descriptor->kernel_args;
  record.workgroup_size[0] = kernel_args->workgroup_size[0];
  record.workgroup_size[1] = kernel_args->workgroup_size[1];
  record.workgroup_size[2] = kernel_args->workgroup_size[2];
  *out_record = record;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_register_profile_operations(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  if (command_buffer->recording.record_command_count == 0)
    return iree_ok_status();

  iree_host_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &byte_length,
      IREE_STRUCT_FIELD(command_buffer->recording.record_command_count,
                        iree_hal_profile_command_operation_record_t, NULL)));

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, command_buffer->recording.record_command_count);

  iree_status_t status =
      iree_allocator_malloc(command_buffer->host_allocator, byte_length,
                            (void**)&command_buffer->profile.operations);

  iree_host_size_t record_count = 0;
  if (iree_status_is_ok(status)) {
    iree_hal_profile_command_operation_record_t* records =
        command_buffer->profile.operations;
    const uint8_t* cursor = command_buffer->recording.record_builder.bytes;
    const uint8_t* const end =
        cursor + command_buffer->recording.record_builder.length;
    while (cursor < end &&
           record_count < command_buffer->recording.record_command_count) {
      if (IREE_UNLIKELY((iree_host_size_t)(end - cursor) <
                        sizeof(iree_hal_amdgpu_pm4_command_record_header_t))) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "PM4 command record header is truncated");
        break;
      }
      const iree_hal_amdgpu_pm4_command_record_header_t* header =
          (const iree_hal_amdgpu_pm4_command_record_header_t*)cursor;
      if (IREE_UNLIKELY(header->length == 0 ||
                        (iree_host_size_t)(end - cursor) < header->length)) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "PM4 command record length is invalid");
        break;
      }
      switch ((iree_hal_amdgpu_pm4_command_record_opcode_t)header->opcode) {
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH:
          iree_hal_amdgpu_pm4_command_buffer_initialize_profile_operation(
              command_buffer->profile.id,
              (const iree_hal_amdgpu_pm4_dispatch_record_t*)cursor,
              &records[record_count++]);
          break;
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
          iree_hal_amdgpu_pm4_atomic_record_initialize_profile_operation(
              command_buffer->profile.id,
              (const iree_hal_amdgpu_pm4_atomic_record_t*)cursor,
              &records[record_count++]);
          break;
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY:
          iree_hal_amdgpu_pm4_transfer_record_initialize_profile_operation(
              command_buffer->profile.id,
              (const iree_hal_amdgpu_pm4_transfer_record_t*)cursor,
              &records[record_count++]);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "unknown PM4 command record opcode %u",
                                    header->opcode);
          break;
      }
      cursor += header->length;
    }
  }
  if (iree_status_is_ok(status) &&
      record_count != command_buffer->recording.record_command_count) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 profile command-operation count mismatch: "
                              "expected %u but got %" PRIhsz,
                              command_buffer->recording.record_command_count,
                              record_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_metadata_register_command_operations(
        command_buffer->profile.metadata, record_count,
        command_buffer->profile.operations);
  }
  if (iree_status_is_ok(status)) {
    command_buffer->profile.operation_count = (uint32_t)record_count;
  } else {
    iree_allocator_free(command_buffer->host_allocator,
                        command_buffer->profile.operations);
    command_buffer->profile.operations = NULL;
    command_buffer->profile.operation_count = 0;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_record_dispatch(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_check_dispatch_flags(flags));
  if (IREE_UNLIKELY(
          iree_hal_amdgpu_executable_requires_queue_scope(executable))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue-scoped executable dispatch from PM4 command buffers is "
        "not implemented");
  }

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
          executable, export_ordinal, command_buffer->device_ordinal,
          &descriptor));
  if (IREE_UNLIKELY(descriptor->kernel_args.workgroup_cluster_size[0] != 0 ||
                    descriptor->kernel_args.workgroup_cluster_size[1] != 0 ||
                    descriptor->kernel_args.workgroup_cluster_size[2] != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "clustered AMDGPU dispatch requires an AQL command buffer");
  }
  if (IREE_UNLIKELY(!descriptor->pm4_launch_state_valid)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dispatch requires executable-load PM4 metadata");
  }
  if (IREE_UNLIKELY(iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(
          config))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dispatch workgroup-size override is not "
        "implemented");
  }

  const bool validates =
      iree_hal_amdgpu_pm4_command_buffer_validates(command_buffer);
  if (validates) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_validate_dispatch_shape(descriptor,
                                                                   config));
    const iree_host_size_t expected_constant_length =
        descriptor->kernarg_layout->constant_byte_length;
    if (IREE_UNLIKELY(constants.data_length != expected_constant_length)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch constant byte length mismatch; expected %" PRIhsz
          " but got %" PRIhsz,
          expected_constant_length, constants.data_length);
    }
    if (IREE_UNLIKELY(bindings.count !=
                      descriptor->kernarg_layout->binding_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)descriptor->kernarg_layout->binding_count, bindings.count);
    }
    if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch bindings must be non-null when count is non-zero");
    }
  }

  if (IREE_UNLIKELY(config.dynamic_workgroup_local_memory != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dynamic LDS is not implemented");
  }

  uint32_t workgroup_count[3] = {
      config.workgroup_count[0],
      config.workgroup_count[1],
      config.workgroup_count[2],
  };
  if (workgroup_count[0] == 0 || workgroup_count[1] == 0 ||
      workgroup_count[2] == 0) {
    return iree_ok_status();
  }
  uint32_t dispatch_thread_count[3] = {0, 0, 0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_resolve_dispatch_thread_count(
          descriptor, config, dispatch_thread_count));

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_retain_dispatch(
      command_buffer, executable, bindings));
  if (IREE_UNLIKELY(descriptor->pm4_group_segment_fixed_size !=
                    descriptor->kernel_args.group_segment_size)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dynamic LDS is not implemented; descriptor LDS "
        "size %" PRIu32 " differs from dispatch group segment size %" PRIu32,
        descriptor->pm4_group_segment_fixed_size,
        descriptor->kernel_args.group_segment_size);
  }

  iree_hal_amdgpu_pm4_dispatch_record_flags_t record_flags =
      IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_NONE;
  iree_hsa_fence_scope_t barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t barrier_release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  const bool consumes_barrier_state =
      command_buffer->recording.barrier_state.pending;
  if (consumes_barrier_state) {
    record_flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER;
    barrier_acquire_scope =
        command_buffer->recording.barrier_state.acquire_scope;
    barrier_release_scope =
        command_buffer->recording.barrier_state.release_scope;
  }
  bool has_dynamic_binding = false;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) {
      has_dynamic_binding = true;
      break;
    }
  }
  const bool adds_fixup_barrier =
      has_dynamic_binding &&
      !command_buffer->recording.has_planned_fixup_barrier;
  if (adds_fixup_barrier) {
    record_flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER;
  }
  const bool adds_profile_fixup_barrier =
      iree_hal_amdgpu_pm4_command_buffer_materializes_profile_dispatch_timestamps(
          command_buffer) &&
      !command_buffer->recording.profile.has_planned_fixup_barrier;
  if (adds_profile_fixup_barrier) {
    record_flags |=
        IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER;
  }
  if (IREE_UNLIKELY(command_buffer->recording.record_command_count ==
                    UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer command index exceeds uint32_t storage");
  }
  const uint32_t command_index = command_buffer->recording.record_command_count;
  const uint32_t function_index =
      iree_hal_executable_function_index(export_ordinal);
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_append_dispatch_record(
          command_buffer, descriptor, iree_hal_amdgpu_executable_id(executable),
          command_index, function_index, config, dispatch_thread_count,
          constants, bindings, record_flags, barrier_acquire_scope,
          barrier_release_scope);
  if (iree_status_is_ok(status)) {
    if (consumes_barrier_state) {
      iree_hal_amdgpu_pm4_barrier_state_reset(
          &command_buffer->recording.barrier_state);
    }
    command_buffer->recording.has_planned_fixup_barrier |= adds_fixup_barrier;
    command_buffer->recording.profile.has_planned_fixup_barrier |=
        adds_profile_fixup_barrier;
    ++command_buffer->recording.record_command_count;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_pm4_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer);

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_verify_create(
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_hal_amdgpu_pm4_command_buffer_flags_t flags,
    iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_arena_block_pool_t* resource_set_block_pool) {
  if (iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command buffers require reusable command-buffer mode");
  }
  const iree_hal_command_buffer_mode_t unsupported_modes =
      IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA;
  if (iree_any_bit_set(mode, unsupported_modes)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 command-buffer mode bits 0x%08" PRIx32
                            " are not implemented",
                            mode & unsupported_modes);
  }
  const iree_hal_command_category_t supported_categories =
      IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_ATOMIC |
      IREE_HAL_COMMAND_CATEGORY_TRANSFER;
  if (command_categories == 0 ||
      iree_any_bit_set(command_categories, ~supported_categories)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 command-buffer categories 0x%08" PRIx32
                            " are not implemented",
                            command_categories & ~supported_categories);
  }
  if (!iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          vendor_packet_capabilities)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command buffers require PM4-IB, EVENT_WRITE, SET_SH_REG, "
        "ACQUIRE_MEM with a supported packet layout, and DISPATCH_DIRECT "
        "capabilities");
  }
  if (IREE_UNLIKELY(!resident_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 command-buffer resident pool is required");
  }
  if (IREE_UNLIKELY(
          iree_all_bits_set(
              mode, IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA) &&
          !profile_metadata)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 command-buffer profile metadata is required");
  }
  if (iree_any_bit_set(
          flags,
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_PROFILE_DISPATCH_TIMESTAMPS)) {
    if (!iree_all_bits_set(
            mode, IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "PM4 dispatch profiling requires retained profile metadata");
    }
    if (!iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
            pm4_timestamp_strategy)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "PM4 dispatch profiling requires queue-local timestamp packet "
          "support");
    }
  }
  if (IREE_UNLIKELY(!resource_set_block_pool)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resource set block pool is required");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_host_size_t device_ordinal, iree_host_size_t physical_queue_count,
    iree_hal_amdgpu_pm4_command_buffer_flags_t flags,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context,
    const iree_hal_amdgpu_device_buffer_transfer_context_t*
        buffer_transfer_context,
    const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
        buffer_transfer_pm4_context,
    iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    void* hostcall_buffer,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_arena_block_pool_t* resource_set_block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_verify_create(
      mode, command_categories, vendor_packet_capabilities, flags,
      pm4_timestamp_strategy, resident_pool, profile_metadata,
      resource_set_block_pool));
  if (IREE_UNLIKELY(!atomic_context)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 atomic launch metadata is required");
  }
  if (IREE_UNLIKELY(!buffer_transfer_context || !buffer_transfer_pm4_context)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 transfer launch metadata is required");
  }
  if (IREE_UNLIKELY(device_ordinal > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 command-buffer device ordinal %" PRIhsz
                            " exceeds uint32_t storage",
                            device_ordinal);
  }
  if (IREE_UNLIKELY(physical_queue_count == 0 ||
                    physical_queue_count > IREE_HAL_MAX_QUEUES)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 physical queue count %" PRIhsz " must be in [1, %" PRIhsz "]",
        physical_queue_count, (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t validation_state_offset = 0;
  iree_host_size_t profile_plans_offset = 0;
  const iree_host_size_t profile_plan_capacity =
      iree_any_bit_set(
          flags,
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_PROFILE_DISPATCH_TIMESTAMPS)
          ? physical_queue_count
          : 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_pm4_command_buffer_t), &total_size,
      IREE_STRUCT_FIELD(
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
          uint8_t, &validation_state_offset),
      IREE_STRUCT_FIELD(profile_plan_capacity,
                        iree_hal_amdgpu_pm4_command_buffer_profile_plan_t,
                        &profile_plans_offset)));

  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer));
  iree_slim_mutex_initialize(&command_buffer->publication_mutex);
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + validation_state_offset,
      &iree_hal_amdgpu_pm4_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->resource_set_block_pool = resource_set_block_pool;
  command_buffer->resident_pool = resident_pool;
  command_buffer->hostcall_buffer = hostcall_buffer;
  command_buffer->profile.metadata = profile_metadata;
  command_buffer->libhsa = resident_pool->libhsa;
  command_buffer->vendor_packet_capabilities = vendor_packet_capabilities;
  command_buffer->atomic_context = atomic_context;
  command_buffer->buffer_transfer_context = buffer_transfer_context;
  command_buffer->buffer_transfer_pm4_context = buffer_transfer_pm4_context;
  command_buffer->pm4_timestamp_strategy = pm4_timestamp_strategy;
  command_buffer->flags = flags;
  command_buffer->device_ordinal = (uint32_t)device_ordinal;

  iree_hal_amdgpu_pm4_command_program_set_flags_t program_set_flags =
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_NONE;
  if (binding_capacity != 0 ||
      iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    program_set_flags |=
        IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_SERIAL_PROFILE;
  }
  iree_status_t status = iree_hal_amdgpu_pm4_command_program_set_initialize(
      queue_affinity, device_ordinal, physical_queue_count, program_set_flags,
      profile_plan_capacity != 0
          ? (iree_hal_amdgpu_pm4_command_buffer_profile_plan_t*)((uint8_t*)
                                                                     command_buffer +
                                                                 profile_plans_offset)
          : NULL,
      &command_buffer->program_set);
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_pm4_command_buffer_retains_profile_metadata(
          command_buffer)) {
    status = iree_hal_amdgpu_profile_metadata_register_command_buffer(
        profile_metadata, mode, command_categories, queue_affinity,
        device_ordinal, &command_buffer->profile.id);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_amdgpu_pm4_command_buffer_destroy(&command_buffer->base);
  }
  return status;
}

static void iree_hal_amdgpu_pm4_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;

  iree_hal_amdgpu_pm4_command_buffer_wait_for_publication(command_buffer);
  if (command_buffer->resident_allocation) {
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_release(
        command_buffer->resident_pool, command_buffer->resident_allocation);
    command_buffer->resident_allocation = NULL;
    iree_hal_amdgpu_pm4_command_buffer_reset_resident_plans(command_buffer);
  }
  if (command_buffer->recording_state ==
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING) {
    iree_hal_amdgpu_pm4_recording_builders_deinitialize(command_buffer);
  }
  iree_allocator_free(host_allocator, command_buffer->profile.operations);
  iree_allocator_free(host_allocator,
                      command_buffer->atomic_binding_requirements);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_slim_mutex_deinitialize(&command_buffer->publication_mutex);
  iree_allocator_free(host_allocator, command_buffer);
}

bool iree_hal_amdgpu_pm4_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_amdgpu_pm4_command_buffer_vtable);
}

iree_host_size_t iree_hal_amdgpu_pm4_command_buffer_device_ordinal(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return command_buffer->device_ordinal;
}

const iree_hal_amdgpu_pm4_program_t* iree_hal_amdgpu_pm4_command_buffer_program(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return &command_buffer->program_set.program;
}

const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t*
iree_hal_amdgpu_pm4_command_buffer_fixup_plan(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return &command_buffer->program_set.fixup;
}

const iree_hal_amdgpu_pm4_command_buffer_profile_plan_t*
iree_hal_amdgpu_pm4_command_buffer_profile_plan(
    iree_hal_command_buffer_t* base_command_buffer,
    uint32_t physical_queue_ordinal) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return iree_hal_amdgpu_pm4_command_program_set_select_profile(
      &command_buffer->program_set, physical_queue_ordinal);
}

uint64_t iree_hal_amdgpu_pm4_command_buffer_profile_id(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return command_buffer->profile.id;
}

const iree_hal_profile_command_operation_record_t*
iree_hal_amdgpu_pm4_command_buffer_profile_operations(
    iree_hal_command_buffer_t* base_command_buffer, uint32_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  *out_count = command_buffer->profile.operation_count;
  return command_buffer->profile.operations;
}

uint32_t iree_hal_amdgpu_pm4_command_buffer_operation_count(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return command_buffer->recording.record_command_count;
}

const iree_hal_amdgpu_atomic_memory_cell_flags_t*
iree_hal_amdgpu_pm4_command_buffer_atomic_binding_requirements(
    iree_hal_command_buffer_t* base_command_buffer, uint32_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  *out_count = command_buffer->atomic_binding_requirements
                   ? command_buffer->base.binding_count
                   : 0;
  return command_buffer->atomic_binding_requirements;
}

const iree_hal_amdgpu_pm4_command_buffer_publish_stats_t*
iree_hal_amdgpu_pm4_command_buffer_publish_stats(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return &command_buffer->publish_stats;
}

hsa_signal_t iree_hal_amdgpu_pm4_command_buffer_acquire_publication_reference(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  hsa_signal_t publication_signal = iree_hsa_signal_null();
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  if (!iree_hsa_signal_is_null(command_buffer->publication_signal)) {
    publication_signal = command_buffer->publication_signal;
    IREE_ASSERT(command_buffer->publication_reference_count != UINT32_MAX,
                "PM4 command-buffer publication reference count overflowed");
    ++command_buffer->publication_reference_count;
  }
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
  return publication_signal;
}

void iree_hal_amdgpu_pm4_command_buffer_cancel_publication_reference(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  IREE_ASSERT(command_buffer->publication_reference_count > 0,
              "PM4 command-buffer publication reference cancelled without a "
              "matching acquire");
  --command_buffer->publication_reference_count;
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
}

void iree_hal_amdgpu_pm4_command_buffer_retire_publication_reference(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_status_code_t status_code) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  IREE_ASSERT(command_buffer->publication_reference_count > 0,
              "PM4 command-buffer publication reference retired without a "
              "matching acquire");
  --command_buffer->publication_reference_count;
  if (command_buffer->publication_reference_count == 0 &&
      status_code == IREE_STATUS_OK &&
      !iree_hsa_signal_is_null(command_buffer->publication_signal)) {
    iree_hal_amdgpu_pm4_command_buffer_release_publication_resources_locked(
        command_buffer);
  }
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
}

//===----------------------------------------------------------------------===//
// Recording operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  switch (command_buffer->recording_state) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_INITIAL:
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "PM4 command buffer is already recording");
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FINALIZED:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "PM4 command buffer has already been recorded");
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FAILED:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "PM4 command buffer recording failed and cannot be reused");
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "invalid PM4 command-buffer recording state %d",
                              (int)command_buffer->recording_state);
  }
  iree_hal_amdgpu_pm4_recording_builders_initialize(command_buffer);
  memset(&command_buffer->publish_stats, 0,
         sizeof(command_buffer->publish_stats));
  command_buffer->recording.barrier_state =
      (iree_hal_amdgpu_pm4_command_barrier_state_t){0};
  command_buffer->recording.previous_launch_state =
      (iree_hal_amdgpu_pm4_dispatch_launch_state_t){0};
  command_buffer->last_retained_executable = NULL;
  command_buffer->recording.has_previous_launch_state = false;
  command_buffer->recording.has_planned_fixup_barrier = false;
  command_buffer->recording.profile.has_planned_fixup_barrier = false;
  command_buffer->recording.record_command_count = 0;
  command_buffer->recording_state =
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(
          command_buffer->recording_state !=
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "PM4 command buffer is not recording");
  }

  const bool collect_timings =
      iree_hal_amdgpu_pm4_command_buffer_collects_finalize_timings(
          command_buffer);
  const iree_time_t finalize_start = collect_timings ? iree_time_now() : 0;
  const uint32_t terminal_barrier_dword_count =
      iree_hal_amdgpu_pm4_barrier_dword_count(
          command_buffer->vendor_packet_capabilities,
          IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
          command_buffer->recording.barrier_state.acquire_scope,
          command_buffer->recording.barrier_state.release_scope);
  iree_hal_amdgpu_pm4_command_program_layout_t program_layout = {0};
  uint8_t* materialization_base = NULL;
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pm4_command_program_set_flags_t program_set_flags =
      command_buffer->program_set.flags &
      ~IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE;
  if (command_buffer->recording.profile.record_program_dword_count != 0) {
    program_set_flags |= IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE;
  }
  status = iree_hal_amdgpu_pm4_command_program_set_initialize(
      command_buffer->base.queue_affinity, command_buffer->device_ordinal,
      command_buffer->program_set.physical_queue_count, program_set_flags,
      command_buffer->program_set.profile_plans, &command_buffer->program_set);
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(terminal_barrier_dword_count == 0)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer terminal barrier cannot be emitted with "
        "capabilities 0x%08" PRIx32,
        command_buffer->vendor_packet_capabilities);
  } else if (iree_status_is_ok(status) &&
             IREE_UNLIKELY(
                 terminal_barrier_dword_count >
                 IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                     command_buffer->recording.record_ib_dword_count)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires more than the PM4-IB maximum %u dwords",
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }
  if (iree_status_is_ok(status) &&
      command_buffer->recording.profile.record_program_dword_count > 0) {
    if (IREE_UNLIKELY(
            terminal_barrier_dword_count >
            IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                command_buffer->recording.profile.record_program_dword_count)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile command buffer requires more than the PM4-IB maximum %u "
          "dwords",
          IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
    } else {
      command_buffer->recording.profile.record_program_dword_count +=
          terminal_barrier_dword_count;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pm4_command_buffer_prepare_resident_storage(
        command_buffer,
        command_buffer->recording.record_ib_dword_count +
            terminal_barrier_dword_count,
        &program_layout, &materialization_base);
  }
  if (iree_status_is_ok(status)) {
    iree_time_t time_start = collect_timings ? iree_time_now() : 0;
    status = iree_hal_amdgpu_pm4_command_buffer_materialize_program_set(
        command_buffer, &program_layout, materialization_base);
    if (collect_timings) {
      command_buffer->publish_stats.materialize_ns +=
          iree_time_now() - time_start;
    }
  }
  if (iree_status_is_ok(status)) {
    command_buffer->publish_stats.host_record_bytes =
        iree_hal_amdgpu_pm4_command_buffer_host_record_bytes(command_buffer);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(command_buffer)) {
    iree_time_t time_start = collect_timings ? iree_time_now() : 0;
    status = iree_hal_amdgpu_pm4_command_buffer_copy_materialized_storage(
        command_buffer, materialization_base);
    if (collect_timings) {
      command_buffer->publish_stats.resident_copy_ns +=
          iree_time_now() - time_start;
    }
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_pm4_command_buffer_retains_profile_metadata(
          command_buffer)) {
    status = iree_hal_amdgpu_pm4_command_buffer_register_profile_operations(
        command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(command_buffer->resource_set);
  }

  iree_hal_amdgpu_pm4_recording_builders_deinitialize(command_buffer);
  if (materialization_base &&
      iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(command_buffer) &&
      !iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(
          command_buffer)) {
    iree_allocator_free(command_buffer->host_allocator, materialization_base);
  }
  if (!iree_status_is_ok(status) && command_buffer->resident_allocation) {
    iree_hal_amdgpu_pm4_command_buffer_wait_for_publication(command_buffer);
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_release(
        command_buffer->resident_pool, command_buffer->resident_allocation);
    command_buffer->resident_allocation = NULL;
    iree_hal_amdgpu_pm4_command_buffer_reset_resident_plans(command_buffer);
  }
  if (collect_timings) {
    command_buffer->publish_stats.total_finalize_ns =
        iree_time_now() - finalize_start;
  }
  if (iree_status_is_ok(status)) {
    command_buffer->recording_state =
        IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FINALIZED;
  } else {
    command_buffer->recording_state =
        IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FAILED;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  const iree_hal_execution_barrier_flags_t supported_flags =
      IREE_HAL_EXECUTION_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
      IREE_HAL_EXECUTION_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
  if (IREE_UNLIKELY(flags & ~supported_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported AMDGPU PM4 execution barrier flags: 0x%016" PRIx64,
        flags & ~supported_flags);
  }
  const iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          source_stage_mask, target_stage_mask, flags, memory_barrier_count,
          memory_barriers, buffer_barrier_count, buffer_barriers);
  iree_hal_amdgpu_pm4_barrier_state_accumulate(
      &command_buffer->recording.barrier_state, scopes.acquire, scopes.release);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_ensure_atomic_binding_requirements(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  if (command_buffer->atomic_binding_requirements) return iree_ok_status();
  if (IREE_UNLIKELY(command_buffer->base.binding_capacity == 0)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 dynamic atomic target requires a non-zero binding capacity");
  }
  iree_host_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &byte_length,
      IREE_STRUCT_FIELD(command_buffer->base.binding_capacity,
                        iree_hal_amdgpu_atomic_memory_cell_flags_t, NULL)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      command_buffer->host_allocator, byte_length,
      (void**)&command_buffer->atomic_binding_requirements));
  memset(command_buffer->atomic_binding_requirements, 0, byte_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_prepare_atomic_target(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_width_t width,
    iree_hal_atomic_flags_t atomic_flags,
    iree_hal_amdgpu_pm4_buffer_ref_record_t* out_target,
    iree_hal_amdgpu_atomic_memory_cell_flags_t* out_required_cell) {
  memset(out_target, 0, sizeof(*out_target));
  out_target->binding_slot = UINT32_MAX;
  *out_required_cell =
      iree_hal_amdgpu_atomic_memory_required_cell(width, atomic_flags);
  if (IREE_UNLIKELY(*out_required_cell ==
                    IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 atomic target has malformed width %u", width);
  }

  out_target->profile_offset = target_ref.offset;
  if (!target_ref.buffer) {
    if (IREE_UNLIKELY(target_ref.buffer_slot >=
                      command_buffer->base.binding_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 atomic binding slot %u exceeds binding capacity %u",
          target_ref.buffer_slot, command_buffer->base.binding_capacity);
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_ensure_atomic_binding_requirements(
            command_buffer));
    out_target->value = target_ref.offset;
    out_target->binding_slot = target_ref.buffer_slot;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_resolve_buffer_ref(
      &target_ref, &out_target->value));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(target_ref.buffer);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_atomic_memory_validate_target(
      iree_hal_amdgpu_buffer_atomic_memory_cells(allocated_buffer),
      (const void*)(uintptr_t)out_target->value, width, atomic_flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_ensure_resource_set(command_buffer));
  if (command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
            command_buffer, (iree_hal_resource_t*)target_ref.buffer));
  }
  return iree_ok_status();
}

static iree_hal_amdgpu_pm4_command_barrier_state_t
iree_hal_amdgpu_pm4_command_buffer_atomic_target_barrier_state(
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_atomic_flags_t atomic_flags,
    iree_hal_atomic_flags_t ordering_flag) {
  iree_hal_amdgpu_pm4_command_barrier_state_t barrier_state = {0};
  if (target_stage_mask != 0) {
    iree_hal_amdgpu_pm4_barrier_state_accumulate(
        &barrier_state,
        iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
            target_stage_mask, atomic_flags, ordering_flag),
        IREE_HSA_FENCE_SCOPE_NONE);
  }
  return barrier_state;
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_prepare_atomic_record(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_width_t width,
    iree_hal_atomic_flags_t atomic_flags,
    iree_hal_atomic_flags_t source_ordering_flag,
    iree_hal_atomic_flags_t target_ordering_flag,
    iree_hal_amdgpu_pm4_buffer_ref_record_t* out_target,
    iree_hal_amdgpu_atomic_memory_cell_flags_t* out_required_cell,
    iree_hal_amdgpu_pm4_atomic_record_flags_t* out_record_flags,
    iree_hsa_fence_scope_t* out_barrier_acquire_scope,
    iree_hsa_fence_scope_t* out_barrier_release_scope,
    iree_hal_amdgpu_pm4_command_barrier_state_t* out_target_barrier_state) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_atomic_target(
      command_buffer, target_ref, width, atomic_flags, out_target,
      out_required_cell));

  iree_hal_amdgpu_pm4_command_barrier_state_t source_barrier_state =
      command_buffer->recording.barrier_state;
  if (source_stage_mask != 0) {
    iree_hal_amdgpu_pm4_barrier_state_accumulate(
        &source_barrier_state, IREE_HSA_FENCE_SCOPE_NONE,
        iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
            source_stage_mask, atomic_flags, source_ordering_flag));
  }

  iree_hal_amdgpu_pm4_atomic_record_flags_t record_flags =
      IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_NONE;
  if (source_barrier_state.pending) {
    record_flags |= IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_EXECUTION_BARRIER;
  }
  const bool has_dynamic_target = out_target->binding_slot != UINT32_MAX;
  if (has_dynamic_target) {
    record_flags |= IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET;
    if (!command_buffer->recording.has_planned_fixup_barrier) {
      record_flags |= IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_FIXUP_BARRIER;
    }
    if (iree_hal_amdgpu_pm4_command_buffer_materializes_profile_dispatch_timestamps(
            command_buffer) &&
        !command_buffer->recording.profile.has_planned_fixup_barrier) {
      record_flags |=
          IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_PROFILE_FIXUP_BARRIER;
    }
  }

  *out_record_flags = record_flags;
  *out_barrier_acquire_scope = source_barrier_state.acquire_scope;
  *out_barrier_release_scope = source_barrier_state.release_scope;
  *out_target_barrier_state =
      iree_hal_amdgpu_pm4_command_buffer_atomic_target_barrier_state(
          target_stage_mask, atomic_flags, target_ordering_flag);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_append_atomic_record(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_amdgpu_pm4_atomic_record_t* record,
    iree_hal_amdgpu_atomic_memory_cell_flags_t required_cell,
    iree_hal_amdgpu_pm4_command_barrier_state_t target_barrier_state) {
  iree_hal_amdgpu_pm4_command_record_measurement_t measurement;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_record_measure(
      record, command_buffer->atomic_context,
      command_buffer->vendor_packet_capabilities,
      command_buffer->recording.record_template_byte_length,
      command_buffer->recording.has_previous_launch_state,
      &command_buffer->recording.previous_launch_state, &measurement));
  const bool materializes_profile =
      iree_hal_amdgpu_pm4_command_buffer_materializes_profile_dispatch_timestamps(
          command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_recording_state_validate_measurement(
          &command_buffer->recording, materializes_profile, &measurement));

  uint8_t* record_bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_record(
      &command_buffer->recording.record_builder, sizeof(*record),
      &record_bytes));
  memcpy(record_bytes, record, sizeof(*record));

  iree_hal_amdgpu_pm4_command_recording_state_commit_measurement(
      &command_buffer->recording, materializes_profile, &measurement);
  const iree_hal_amdgpu_device_kernel_pm4_launch_t* launch =
      iree_hal_amdgpu_pm4_atomic_record_launch(record,
                                               command_buffer->atomic_context);
  command_buffer->recording.previous_launch_state = launch->launch_state;
  command_buffer->recording.has_previous_launch_state = true;
  command_buffer->recording.has_planned_fixup_barrier |= iree_any_bit_set(
      record->flags, IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_FIXUP_BARRIER);
  command_buffer->recording.profile.has_planned_fixup_barrier |=
      iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_PROFILE_FIXUP_BARRIER);
  command_buffer->recording.barrier_state = target_barrier_state;
  if (iree_any_bit_set(record->flags,
                       IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET)) {
    command_buffer->base.binding_count = iree_max(
        command_buffer->base.binding_count, record->target.binding_slot + 1u);
    command_buffer->atomic_binding_requirements[record->target.binding_slot] |=
        required_cell;
  }
  ++command_buffer->recording.record_command_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(command_buffer->recording.record_command_count ==
                    UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer command index exceeds uint32_t storage");
  }
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
  iree_hal_amdgpu_atomic_memory_cell_flags_t required_cell = 0;
  iree_hal_amdgpu_pm4_atomic_record_flags_t record_flags = 0;
  iree_hsa_fence_scope_t barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t barrier_release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hal_amdgpu_pm4_command_barrier_state_t target_barrier_state;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_atomic_record(
      command_buffer, source_stage_mask, target_stage_mask, target_ref,
      params.width, params.flags, IREE_HAL_ATOMIC_FLAG_NONE,
      IREE_HAL_ATOMIC_FLAG_ACQUIRE, &target, &required_cell, &record_flags,
      &barrier_acquire_scope, &barrier_release_scope, &target_barrier_state));
  iree_hal_amdgpu_pm4_atomic_record_t record;
  iree_hal_amdgpu_pm4_atomic_record_initialize_wait(
      target, params, command_buffer->recording.record_command_count,
      record_flags, barrier_acquire_scope, barrier_release_scope, &record);
  return iree_hal_amdgpu_pm4_command_buffer_append_atomic_record(
      command_buffer, &record, required_cell, target_barrier_state);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_atomic_store(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(command_buffer->recording.record_command_count ==
                    UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer command index exceeds uint32_t storage");
  }
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
  iree_hal_amdgpu_atomic_memory_cell_flags_t required_cell = 0;
  iree_hal_amdgpu_pm4_atomic_record_flags_t record_flags = 0;
  iree_hsa_fence_scope_t barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t barrier_release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hal_amdgpu_pm4_command_barrier_state_t target_barrier_state;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_atomic_record(
      command_buffer, source_stage_mask, target_stage_mask, target_ref,
      params.width, params.flags, IREE_HAL_ATOMIC_FLAG_RELEASE,
      IREE_HAL_ATOMIC_FLAG_NONE, &target, &required_cell, &record_flags,
      &barrier_acquire_scope, &barrier_release_scope, &target_barrier_state));
  iree_hal_amdgpu_pm4_atomic_record_t record;
  iree_hal_amdgpu_pm4_atomic_record_initialize_store(
      target, params, command_buffer->recording.record_command_count,
      record_flags, barrier_acquire_scope, barrier_release_scope, &record);
  return iree_hal_amdgpu_pm4_command_buffer_append_atomic_record(
      command_buffer, &record, required_cell, target_barrier_state);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_rmw_params_t params) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(command_buffer->recording.record_command_count ==
                    UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer command index exceeds uint32_t storage");
  }
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
  iree_hal_amdgpu_atomic_memory_cell_flags_t required_cell = 0;
  iree_hal_amdgpu_pm4_atomic_record_flags_t record_flags = 0;
  iree_hsa_fence_scope_t barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t barrier_release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hal_amdgpu_pm4_command_barrier_state_t target_barrier_state;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_atomic_record(
      command_buffer, source_stage_mask, target_stage_mask, target_ref,
      params.width, params.flags, IREE_HAL_ATOMIC_FLAG_RELEASE,
      IREE_HAL_ATOMIC_FLAG_ACQUIRE, &target, &required_cell, &record_flags,
      &barrier_acquire_scope, &barrier_release_scope, &target_barrier_state));
  iree_hal_amdgpu_pm4_atomic_record_t record;
  iree_hal_amdgpu_pm4_atomic_record_initialize_rmw(
      target, params, command_buffer->recording.record_command_count,
      record_flags, barrier_acquire_scope, barrier_release_scope, &record);
  return iree_hal_amdgpu_pm4_command_buffer_append_atomic_record(
      command_buffer, &record, required_cell, target_barrier_state);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer events are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer events are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer events are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_prepare_transfer_ref(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref, uint64_t dynamic_alignment,
    iree_hal_amdgpu_pm4_buffer_ref_record_t* out_record, uint64_t* out_length,
    uint64_t* out_alignment) {
  memset(out_record, 0, sizeof(*out_record));
  out_record->binding_slot = UINT32_MAX;
  *out_length = 0;
  *out_alignment = 0;

  if (!buffer_ref.buffer) {
    if (IREE_UNLIKELY(buffer_ref.buffer_slot >=
                      command_buffer->base.binding_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 transfer binding slot %u exceeds binding capacity %u",
          buffer_ref.buffer_slot, command_buffer->base.binding_capacity);
    }
    out_record->value = buffer_ref.offset;
    out_record->profile_offset = buffer_ref.offset;
    out_record->binding_slot = buffer_ref.buffer_slot;
    *out_length = buffer_ref.length;
    *out_alignment = dynamic_alignment;
    return iree_ok_status();
  }

  iree_device_size_t resolved_offset = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(buffer_ref.buffer),
      buffer_ref.offset, buffer_ref.length, &resolved_offset,
      &resolved_length));
  const iree_hal_buffer_ref_t resolved_ref = iree_hal_make_buffer_ref(
      buffer_ref.buffer, resolved_offset, resolved_length);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_resolve_buffer_ref(
      &resolved_ref, &out_record->value));
  out_record->profile_offset = resolved_offset;
  *out_length = resolved_length;
  *out_alignment = iree_hal_amdgpu_device_buffer_transfer_pointer_alignment(
      (const void*)(uintptr_t)out_record->value);

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_ensure_resource_set(command_buffer));
  if (command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
            command_buffer, (iree_hal_resource_t*)buffer_ref.buffer));
  }
  return iree_ok_status();
}

static iree_hal_amdgpu_pm4_transfer_recorder_t
iree_hal_amdgpu_pm4_command_buffer_transfer_recorder(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return (iree_hal_amdgpu_pm4_transfer_recorder_t){
      .recording_state = &command_buffer->recording,
      .binding_count = &command_buffer->base.binding_count,
      .transfer_context = command_buffer->buffer_transfer_context,
      .transfer_pm4_context = command_buffer->buffer_transfer_pm4_context,
      .vendor_packet_capabilities = command_buffer->vendor_packet_capabilities,
      .materializes_profile =
          iree_hal_amdgpu_pm4_command_buffer_materializes_profile_dispatch_timestamps(
              command_buffer),
  };
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  if (IREE_UNLIKELY(!pattern)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill pattern must be non-null");
  }
  if (IREE_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                    pattern_length != 4)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill patterns must be 1, 2, or 4 bytes (got %" PRIhsz ")",
        pattern_length);
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_FILL_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported fill flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY((target_ref.offset % pattern_length) != 0 ||
                    (target_ref.length % pattern_length) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill target offset and length must be aligned to the pattern length");
  }

  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
  uint64_t length = 0;
  uint64_t target_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_transfer_ref(
      command_buffer, target_ref, pattern_length, &target, &length,
      &target_alignment));
  iree_hal_amdgpu_pm4_transfer_recorder_t recorder =
      iree_hal_amdgpu_pm4_command_buffer_transfer_recorder(command_buffer);
  return iree_hal_amdgpu_pm4_transfer_recorder_fill(
      &recorder, target, target_alignment, length, pattern, pattern_length);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  if (IREE_UNLIKELY(!source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "update source buffer must be non-null");
  }
  if (IREE_UNLIKELY(target_ref.length >
                    IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer update length %" PRIdsz
                            " exceeds maximum update size %" PRIdsz,
                            target_ref.length,
                            IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE);
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_UPDATE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported update flags: 0x%" PRIx64, flags);
  }
  const uintptr_t source_address = (uintptr_t)source_buffer;
  if (IREE_UNLIKELY(source_offset > UINTPTR_MAX - source_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "update source offset overflows host pointer");
  }

  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
  uint64_t length = 0;
  uint64_t target_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_transfer_ref(
      command_buffer, target_ref, /*dynamic_alignment=*/1, &target, &length,
      &target_alignment));
  if (IREE_UNLIKELY(length > IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolved command-buffer update length %" PRIu64
                            " exceeds maximum update size %" PRIdsz,
                            length, IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE);
  }
  iree_hal_amdgpu_pm4_transfer_recorder_t recorder =
      iree_hal_amdgpu_pm4_command_buffer_transfer_recorder(command_buffer);
  return iree_hal_amdgpu_pm4_transfer_recorder_update(
      &recorder, target, target_alignment,
      iree_make_const_byte_span(
          (const uint8_t*)(source_address + source_offset),
          (iree_host_size_t)length));
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  if (IREE_UNLIKELY(flags != IREE_HAL_COPY_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported copy flags: 0x%" PRIx64, flags);
  }

  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_hal_amdgpu_pm4_buffer_ref_record_t source;
  uint64_t source_length = 0;
  uint64_t source_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_transfer_ref(
      command_buffer, source_ref, /*dynamic_alignment=*/1, &source,
      &source_length, &source_alignment));
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
  uint64_t target_length = 0;
  uint64_t target_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_prepare_transfer_ref(
      command_buffer, target_ref, /*dynamic_alignment=*/1, &target,
      &target_length, &target_alignment));
  if (IREE_UNLIKELY(source_length != target_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy spans between source and target must match "
                            "(source_length=%" PRIu64 ", target_length=%" PRIu64
                            ")",
                            source_length, target_length);
  }
  iree_hal_amdgpu_pm4_transfer_recorder_t recorder =
      iree_hal_amdgpu_pm4_command_buffer_transfer_recorder(command_buffer);
  return iree_hal_amdgpu_pm4_transfer_recorder_copy(
      &recorder, source, source_alignment, target, target_alignment,
      source_length);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer collectives are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(
          command_buffer->recording_state !=
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "PM4 command buffer is not recording");
  }
  return iree_hal_amdgpu_pm4_command_buffer_record_dispatch(
      command_buffer, executable, export_ordinal, config, constants, bindings,
      flags);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_pm4_command_buffer_vtable = {
        .destroy = iree_hal_amdgpu_pm4_command_buffer_destroy,
        .begin = iree_hal_amdgpu_pm4_command_buffer_begin,
        .end = iree_hal_amdgpu_pm4_command_buffer_end,
        .begin_debug_group =
            iree_hal_amdgpu_pm4_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amdgpu_pm4_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_amdgpu_pm4_command_buffer_execution_barrier,
        .atomic_wait = iree_hal_amdgpu_pm4_command_buffer_atomic_wait,
        .atomic_store = iree_hal_amdgpu_pm4_command_buffer_atomic_store,
        .atomic_rmw = iree_hal_amdgpu_pm4_command_buffer_atomic_rmw,
        .signal_event = iree_hal_amdgpu_pm4_command_buffer_signal_event,
        .reset_event = iree_hal_amdgpu_pm4_command_buffer_reset_event,
        .wait_events = iree_hal_amdgpu_pm4_command_buffer_wait_events,
        .advise_buffer = iree_hal_amdgpu_pm4_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amdgpu_pm4_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdgpu_pm4_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdgpu_pm4_command_buffer_copy_buffer,
        .collective = iree_hal_amdgpu_pm4_command_buffer_collective,
        .dispatch = iree_hal_amdgpu_pm4_command_buffer_dispatch,
};
