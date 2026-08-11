// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// _GNU_SOURCE must be defined before any system header so that <dlfcn.h>
// exposes the GNU extensions dladdr()/Dl_info and the RTLD_NOLOAD flag, which
// hipGetProcAddress() uses to resolve symbols against this shared object
// itself rather than the process-global scope.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "binding/hip/api.h"

#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "binding/hip/binding_internal.h"
#include "binding/hip/blocking_printf_provider.h"
#include "binding/hip/handle_registry.h"
#include "binding/hip/launch_params.h"
#include "common/graph.h"
#include "common/internal.h"
#include "common/stream.h"
#include "common/tls.h"
#include "hrx_runtime.h"
#include "iree/base/threading/call_once.h"
#include "iree/hal/drivers/amdgpu/api.h"

//===----------------------------------------------------------------------===//
// Debug logging
//===----------------------------------------------------------------------===//

#ifndef IREE_HIP_VERBOSE_DEBUG
#define IREE_HIP_VERBOSE_DEBUG 0
#endif

#if IREE_HIP_VERBOSE_DEBUG
#define HIP_DEBUG_LOG(fmt, ...)          \
  do {                                   \
    fprintf(stderr, fmt, ##__VA_ARGS__); \
    fflush(stderr);                      \
  } while (0)
#else
#define HIP_DEBUG_LOG(fmt, ...) ((void)0)
#endif

//===----------------------------------------------------------------------===//
// HRX binding identification
//===----------------------------------------------------------------------===//

typedef struct hrx_hip_info_t {
  int version_major;
  int version_minor;
  int version_patch;
  const char* build_type;
  const char* iree_hal_driver;
  const char* backend_lib;
} hrx_hip_info_t;

static void iree_hip_sanitize_device_name(char* name) {
  if (!name) return;
  char* node_suffix = strstr(name, " (Node ");
  if (node_suffix) {
    *node_suffix = '\0';
  }
}

HIPAPI int hrx_hip_binding_active(void) { return 1; }

HIPAPI void hrx_hip_binding_info(hrx_hip_info_t* out_info) {
  if (!out_info) return;
  out_info->version_major = HRX_VERSION_MAJOR;
  out_info->version_minor = HRX_VERSION_MINOR;
  out_info->version_patch = HRX_VERSION_PATCH;
#ifdef NDEBUG
  out_info->build_type = "Release";
#else
  out_info->build_type = "Debug";
#endif
  const char* driver = getenv("IREE_HAL_DRIVER");
  out_info->iree_hal_driver = driver ? driver : "(default)";
  const char* backend = getenv("HIP_PASSTHROUGH_BACKEND_LIB");
  out_info->backend_lib = backend ? backend : "(none)";
}

typedef enum hrx_hip_stream_batch_mem_op_type_e {
  HRX_HIP_STREAM_MEM_OP_WAIT_VALUE_32 = 0x1,
  HRX_HIP_STREAM_MEM_OP_WRITE_VALUE_32 = 0x2,
  HRX_HIP_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 0x3,
  HRX_HIP_STREAM_MEM_OP_WAIT_VALUE_64 = 0x4,
  HRX_HIP_STREAM_MEM_OP_WRITE_VALUE_64 = 0x5,
  HRX_HIP_STREAM_MEM_OP_BARRIER = 0x6,
} hrx_hip_stream_batch_mem_op_type_t;

typedef union hrx_hip_stream_batch_mem_op_params_t {
  hrx_hip_stream_batch_mem_op_type_t operation;
  struct {
    hrx_hip_stream_batch_mem_op_type_t operation;
    hipDeviceptr_t address;
    union {
      uint32_t value;
      uint64_t value64;
    };
    unsigned int flags;
    hipDeviceptr_t alias;
  } waitValue;
  struct {
    hrx_hip_stream_batch_mem_op_type_t operation;
    hipDeviceptr_t address;
    union {
      uint32_t value;
      uint64_t value64;
    };
    unsigned int flags;
    hipDeviceptr_t alias;
  } writeValue;
  struct {
    hrx_hip_stream_batch_mem_op_type_t operation;
    unsigned int flags;
  } flushRemoteWrites;
  struct {
    hrx_hip_stream_batch_mem_op_type_t operation;
    unsigned int flags;
  } memoryBarrier;
  uint64_t pad[6];
} hrx_hip_stream_batch_mem_op_params_t;

typedef struct hrx_hip_batch_mem_op_node_params_t {
  hipCtx_t ctx;
  unsigned int count;
  hrx_hip_stream_batch_mem_op_params_t* paramArray;
  unsigned int flags;
} hrx_hip_batch_mem_op_node_params_t;

static bool iree_hip_graph_handle_is_live(hipGraph_t graph);
static hipError_t iree_status_to_hip_result(iree_status_t status);

#define IREE_HIP_ARRAY_MAGIC 0x6872786869706179ull
// HRX HIP arrays are backed by normal device allocations. The practical limit
// is checked allocation size and available device memory; HIP device attributes
// expose these dimensions as int-valued fields.
#define IREE_HIP_ARRAY_MAX_DIMENSION INT_MAX
#define IREE_HIP_ARRAY_MAX_1D IREE_HIP_ARRAY_MAX_DIMENSION
#define IREE_HIP_ARRAY_MAX_2D_WIDTH IREE_HIP_ARRAY_MAX_DIMENSION
#define IREE_HIP_ARRAY_MAX_2D_HEIGHT IREE_HIP_ARRAY_MAX_DIMENSION
#define IREE_HIP_ARRAY_MAX_3D_WIDTH IREE_HIP_ARRAY_MAX_DIMENSION
#define IREE_HIP_ARRAY_MAX_3D_HEIGHT IREE_HIP_ARRAY_MAX_DIMENSION
#define IREE_HIP_ARRAY_MAX_3D_DEPTH IREE_HIP_ARRAY_MAX_DIMENSION
struct hipArray_st {
  // References held by the registry and active API callers.
  iree_atomic_ref_count_t ref_count;
  // Next live array handle in the process registry.
  struct hipArray_st* next_live_array;
  // Magic value used to reject invalid or freed handles.
  uint64_t magic;
  // Channel descriptor supplied at allocation time.
  hipChannelFormatDesc desc;
  // Public logical extent reported by descriptor APIs.
  hipExtent public_extent;
  // Normalized backing extent used for storage and copies.
  hipExtent extent;
  // Driver API format equivalent for desc.
  hipArray_Format format;
  // Driver API channel count equivalent for desc.
  unsigned int num_channels;
  // Allocation flags supplied at allocation time.
  unsigned int flags;
  // Context that owns the backing device allocation.
  iree_hal_streaming_context_t* context;
  // Backing device buffer wrapper registered in the context table.
  iree_hal_streaming_buffer_t* buffer;
  // Backing device pointer exported from the buffer wrapper.
  iree_hal_streaming_deviceptr_t device_ptr;
  // Bytes per array element.
  size_t element_size;
  // Logical row width in bytes.
  size_t width_bytes;
  // Pitched row stride in bytes.
  size_t pitch;
  // Rows per depth slice.
  size_t rows_per_slice;
  // Bytes per depth slice.
  size_t slice_pitch;
  // Total allocation size in bytes.
  size_t allocation_size;
};

static iree_once_flag iree_hip_array_registry_mutex_once = IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hip_array_registry_mutex;
static struct hipArray_st* iree_hip_array_registry_head = NULL;

static void iree_hip_array_registry_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_array_registry_mutex);
}

static void iree_hip_array_registry_lock(void) {
  iree_call_once(&iree_hip_array_registry_mutex_once,
                 iree_hip_array_registry_mutex_initialize);
  iree_slim_mutex_lock(&iree_hip_array_registry_mutex);
}

static hipError_t iree_hip_array_retain(hipArray_const_t array,
                                        struct hipArray_st** out);

static void iree_hip_array_release(struct hipArray_st* array);

static hipError_t iree_hip_array_byte_range_to_elements(
    hipArray_const_t array, size_t byte_offset, size_t byte_width,
    size_t* out_element_offset, size_t* out_element_width);

static hipError_t iree_hip_array_byte_range_to_elements_for_array(
    const struct hipArray_st* array, size_t byte_offset, size_t byte_width,
    size_t* out_element_offset, size_t* out_element_width);

static hipError_t iree_hip_memcpy2d_array_to_3d_params(
    const hip_Memcpy2D* copy, hipMemcpy3DParms* out_params);

static hipError_t iree_hip_array_validate_extent_limits(hipExtent extent);

static void iree_hip_array_registry_insert(struct hipArray_st* array) {
  iree_hip_array_registry_lock();
  array->next_live_array = iree_hip_array_registry_head;
  iree_hip_array_registry_head = array;
  iree_slim_mutex_unlock(&iree_hip_array_registry_mutex);
}

static bool iree_hip_array_registry_remove(hipArray_t array,
                                           struct hipArray_st** out_array) {
  if (out_array) *out_array = NULL;
  if (!array) return false;
  bool removed = false;
  iree_hip_array_registry_lock();
  struct hipArray_st** current = &iree_hip_array_registry_head;
  while (*current) {
    if (*current == array) {
      if (out_array) *out_array = *current;
      *current = array->next_live_array;
      array->next_live_array = NULL;
      removed = true;
      break;
    }
    current = &(*current)->next_live_array;
  }
  iree_slim_mutex_unlock(&iree_hip_array_registry_mutex);
  return removed;
}

static bool iree_hip_array_registry_lookup(hipArray_const_t array,
                                           struct hipArray_st** out) {
  if (out) *out = NULL;
  if (!array) return false;
  bool found = false;
  iree_hip_array_registry_lock();
  for (struct hipArray_st* current = iree_hip_array_registry_head; current;
       current = current->next_live_array) {
    if (current == array && current->magic == IREE_HIP_ARRAY_MAGIC) {
      iree_atomic_ref_count_inc(&current->ref_count);
      if (out) *out = current;
      found = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_array_registry_mutex);
  return found;
}

//===----------------------------------------------------------------------===//
// Memory pool handles
//===----------------------------------------------------------------------===//

struct hipMemPool_st {
  // References held by the live-handle registry and active API callers.
  iree_atomic_ref_count_t ref_count;
  // Next live pool handle in the process registry.
  struct hipMemPool_st* next_live_pool;
  // HRX pool owned by a user-created handle, or NULL for a default handle.
  hrx_mem_pool_t hrx_pool;
  // Device ordinal associated with the pool's allocation location.
  int device_ordinal;
  // Allocation type accepted by this pool handle.
  hipMemAllocationType allocation_type;
  // True when this handle resolves the device's default pool dynamically.
  bool is_default;
};

typedef struct iree_hip_managed_mem_pool_selection_t {
  // Next managed-pool selection in the process registry.
  struct iree_hip_managed_mem_pool_selection_t* next;
  // Device ordinal associated with the selected managed pool.
  int device_ordinal;
  // Retained pool handle selected for managed allocations on the device.
  struct hipMemPool_st* pool;
} iree_hip_managed_mem_pool_selection_t;

static iree_once_flag iree_hip_mem_pool_registry_mutex_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hip_mem_pool_registry_mutex;
static struct hipMemPool_st* iree_hip_mem_pool_registry_head = NULL;
static iree_hip_managed_mem_pool_selection_t*
    iree_hip_managed_mem_pool_selection_head = NULL;

static void iree_hip_mem_pool_registry_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_mem_pool_registry_mutex);
}

static void iree_hip_mem_pool_registry_lock(void) {
  iree_call_once(&iree_hip_mem_pool_registry_mutex_once,
                 iree_hip_mem_pool_registry_mutex_initialize);
  iree_slim_mutex_lock(&iree_hip_mem_pool_registry_mutex);
}

static void iree_hip_mem_pool_release(struct hipMemPool_st* pool) {
  if (!pool) return;
  if (iree_atomic_ref_count_dec(&pool->ref_count) != 1) return;
  hrx_mem_pool_release(pool->hrx_pool);
  iree_allocator_free(iree_allocator_system(), pool);
}

static hipError_t iree_hip_mem_pool_registry_acquire(
    hipMemPool_t pool, struct hipMemPool_st** out_pool) {
  if (out_pool) *out_pool = NULL;
  if (!pool) return hipErrorInvalidValue;

  iree_hip_mem_pool_registry_lock();
  for (struct hipMemPool_st* current = iree_hip_mem_pool_registry_head; current;
       current = current->next_live_pool) {
    if (current == pool) {
      iree_atomic_ref_count_inc(&current->ref_count);
      if (out_pool) *out_pool = current;
      iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
      return hipSuccess;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
  return hipErrorInvalidValue;
}

static hipError_t iree_hip_mem_pool_create_handle(
    hrx_mem_pool_t hrx_pool, int device_ordinal,
    hipMemAllocationType allocation_type, bool is_default,
    hipMemPool_t* out_pool) {
  *out_pool = NULL;
  struct hipMemPool_st* pool = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*pool), (void**)&pool);
  if (!iree_status_is_ok(status)) return iree_status_to_hip_result(status);

  memset(pool, 0, sizeof(*pool));
  iree_atomic_ref_count_init(&pool->ref_count);
  pool->hrx_pool = hrx_pool;
  pool->device_ordinal = device_ordinal;
  pool->allocation_type = allocation_type;
  pool->is_default = is_default;

  iree_hip_mem_pool_registry_lock();
  pool->next_live_pool = iree_hip_mem_pool_registry_head;
  iree_hip_mem_pool_registry_head = pool;
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
  *out_pool = pool;
  return hipSuccess;
}

static hipError_t iree_hip_mem_pool_get_default(
    int device_ordinal, hipMemAllocationType allocation_type,
    hipMemPool_t* out_pool) {
  *out_pool = NULL;
  iree_hip_mem_pool_registry_lock();
  for (struct hipMemPool_st* current = iree_hip_mem_pool_registry_head; current;
       current = current->next_live_pool) {
    if (current->is_default && current->device_ordinal == device_ordinal &&
        current->allocation_type == allocation_type) {
      *out_pool = current;
      iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
      return hipSuccess;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);

  struct hipMemPool_st* pool = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*pool), (void**)&pool);
  if (!iree_status_is_ok(status)) return iree_status_to_hip_result(status);
  memset(pool, 0, sizeof(*pool));
  iree_atomic_ref_count_init(&pool->ref_count);
  pool->device_ordinal = device_ordinal;
  pool->allocation_type = allocation_type;
  pool->is_default = true;

  iree_hip_mem_pool_registry_lock();
  for (struct hipMemPool_st* current = iree_hip_mem_pool_registry_head; current;
       current = current->next_live_pool) {
    if (current->is_default && current->device_ordinal == device_ordinal &&
        current->allocation_type == allocation_type) {
      *out_pool = current;
      iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
      iree_allocator_free(iree_allocator_system(), pool);
      return hipSuccess;
    }
  }
  pool->next_live_pool = iree_hip_mem_pool_registry_head;
  iree_hip_mem_pool_registry_head = pool;
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
  *out_pool = pool;
  return hipSuccess;
}

static hipError_t iree_hip_mem_pool_retain_backend(
    const struct hipMemPool_st* pool, hrx_mem_pool_t* out_hrx_pool) {
  *out_hrx_pool = NULL;
  if (!pool->is_default) {
    hrx_mem_pool_retain(pool->hrx_pool);
    *out_hrx_pool = pool->hrx_pool;
    return hipSuccess;
  }

  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(pool->device_ordinal);
  if (!device) return hipErrorInvalidDevice;
  iree_status_t status =
      iree_hal_streaming_device_ensure_default_mem_pool(device);
  if (!iree_status_is_ok(status)) return iree_status_to_hip_result(status);
  *out_hrx_pool = iree_hal_streaming_device_retain_default_mem_pool(device);
  return *out_hrx_pool ? hipSuccess : hipErrorInvalidValue;
}

static bool iree_hip_mem_pool_registry_remove(hipMemPool_t pool,
                                              struct hipMemPool_st** out_pool) {
  if (out_pool) *out_pool = NULL;
  if (!pool) return false;

  iree_hip_managed_mem_pool_selection_t* removed_selection = NULL;
  iree_hip_mem_pool_registry_lock();
  struct hipMemPool_st** current = &iree_hip_mem_pool_registry_head;
  while (*current) {
    if (*current == pool && !pool->is_default) {
      if (out_pool) *out_pool = *current;
      *current = pool->next_live_pool;
      pool->next_live_pool = NULL;

      iree_hip_managed_mem_pool_selection_t** selection =
          &iree_hip_managed_mem_pool_selection_head;
      while (*selection) {
        if ((*selection)->pool == pool) {
          removed_selection = *selection;
          *selection = removed_selection->next;
          break;
        }
        selection = &(*selection)->next;
      }
      iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
      if (removed_selection) {
        iree_hip_mem_pool_release(removed_selection->pool);
        iree_allocator_free(iree_allocator_system(), removed_selection);
      }
      return true;
    }
    current = &(*current)->next_live_pool;
  }
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
  return false;
}

static hipError_t iree_hip_mem_pool_get_current(
    iree_hal_streaming_device_t* device, int device_ordinal,
    hipMemPool_t* out_pool) {
  *out_pool = NULL;
  hrx_mem_pool_t hrx_pool = iree_hal_streaming_device_retain_mem_pool(device);
  if (!hrx_pool) return hipErrorInvalidValue;

  iree_hip_mem_pool_registry_lock();
  for (struct hipMemPool_st* current = iree_hip_mem_pool_registry_head; current;
       current = current->next_live_pool) {
    if (!current->is_default &&
        current->allocation_type == hipMemAllocationTypePinned &&
        current->device_ordinal == device_ordinal &&
        current->hrx_pool == hrx_pool) {
      *out_pool = current;
      break;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);
  hrx_mem_pool_release(hrx_pool);

  return *out_pool ? hipSuccess
                   : iree_hip_mem_pool_get_default(
                         device_ordinal, hipMemAllocationTypePinned, out_pool);
}

static hipError_t iree_hip_mem_pool_get_managed_selection(
    int device_ordinal, hipMemPool_t* out_pool) {
  *out_pool = NULL;
  iree_hip_mem_pool_registry_lock();
  for (iree_hip_managed_mem_pool_selection_t* selection =
           iree_hip_managed_mem_pool_selection_head;
       selection; selection = selection->next) {
    if (selection->device_ordinal == device_ordinal) {
      *out_pool = selection->pool;
      break;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);

  return *out_pool ? hipSuccess
                   : iree_hip_mem_pool_get_default(
                         device_ordinal, hipMemAllocationTypeManaged, out_pool);
}

static hipError_t iree_hip_mem_pool_set_managed_selection(
    int device_ordinal, struct hipMemPool_st* pool) {
  iree_hip_managed_mem_pool_selection_t* new_selection = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*new_selection), (void**)&new_selection);
  if (!iree_status_is_ok(status)) return iree_status_to_hip_result(status);
  memset(new_selection, 0, sizeof(*new_selection));

  iree_hip_managed_mem_pool_selection_t* previous_selection = NULL;
  iree_hip_mem_pool_registry_lock();
  iree_hip_managed_mem_pool_selection_t** selection =
      &iree_hip_managed_mem_pool_selection_head;
  while (*selection && (*selection)->device_ordinal != device_ordinal) {
    selection = &(*selection)->next;
  }
  if (*selection) {
    previous_selection = *selection;
    *selection = previous_selection->next;
  }
  new_selection->device_ordinal = device_ordinal;
  new_selection->pool = pool;
  iree_atomic_ref_count_inc(&pool->ref_count);
  new_selection->next = iree_hip_managed_mem_pool_selection_head;
  iree_hip_managed_mem_pool_selection_head = new_selection;
  iree_slim_mutex_unlock(&iree_hip_mem_pool_registry_mutex);

  if (previous_selection) {
    iree_hip_mem_pool_release(previous_selection->pool);
    iree_allocator_free(iree_allocator_system(), previous_selection);
  }
  return hipSuccess;
}

static bool iree_hip_mem_pool_allocation_type_is_supported(
    hipMemAllocationType allocation_type) {
  return allocation_type == hipMemAllocationTypePinned ||
         allocation_type == hipMemAllocationTypeManaged;
}

// Pool allocation and stream-ordered release are implemented for every
// registered streaming device. This capability is independent of GPU
// architecture.
static bool iree_hip_memory_pools_supported(void) { return true; }

//===----------------------------------------------------------------------===//
// Flag translation functions
//===----------------------------------------------------------------------===//

static iree_hal_streaming_stream_flags_t iree_hip_stream_flags_to_internal(
    unsigned int hip_flags) {
  iree_hal_streaming_stream_flags_t flags = IREE_HAL_STREAMING_STREAM_FLAG_NONE;
  if (hip_flags & hipStreamNonBlocking) {
    flags |= IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING;
  }
  return flags;
}

static bool iree_hip_stream_create_flags_are_valid(unsigned int flags) {
  return (flags & ~hipStreamNonBlocking) == 0;
}

static bool iree_hip_synchronization_policy_is_valid(
    hipSynchronizationPolicy policy) {
  switch (policy) {
    case hipSyncPolicyAuto:
    case hipSyncPolicySpin:
    case hipSyncPolicyYield:
    case hipSyncPolicyBlockingSync:
      return true;
  }
  return false;
}

static int iree_hip_clamp_stream_priority(int priority) {
  int least_priority = 0;
  int greatest_priority = 0;
  hipError_t result =
      hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  if (result != hipSuccess) return priority;
  if (priority < greatest_priority) return greatest_priority;
  if (priority > least_priority) return least_priority;
  return priority;
}

static iree_hal_streaming_event_flags_t iree_hip_event_flags_to_internal(
    unsigned int hip_flags) {
  iree_hal_streaming_event_flags_t flags = IREE_HAL_STREAMING_EVENT_FLAG_NONE;
  if (hip_flags & hipEventBlockingSync) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_BLOCKING_SYNC;
  }
  if (hip_flags & hipEventDisableTiming) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING;
  }
  if (hip_flags & hipEventInterprocess) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_INTERPROCESS;
  }
  return flags;
}

static bool iree_hip_event_create_flags_are_valid(unsigned int flags) {
  const unsigned int supported_flags =
      hipEventBlockingSync | hipEventDisableTiming | hipEventInterprocess |
      hipEventDisableSystemFence | hipEventReleaseToDevice |
      hipEventReleaseToSystem;
  if ((flags & ~supported_flags) != 0) return false;
  if ((flags & hipEventInterprocess) && !(flags & hipEventDisableTiming)) {
    return false;
  }
  if ((flags & hipEventReleaseToDevice) && (flags & hipEventReleaseToSystem)) {
    return false;
  }
  return true;
}

static iree_hal_streaming_memory_flags_t iree_hip_memory_flags_to_internal(
    unsigned int hip_flags) {
  iree_hal_streaming_memory_flags_t flags = IREE_HAL_STREAMING_MEMORY_FLAG_NONE;
  if (hip_flags & hipHostRegisterPortable) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_PORTABLE;
  }
  if (hip_flags & hipHostRegisterMapped) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_PINNED;
  }
  // hipHostRegisterIoMemory could map to write-combined.
  if (hip_flags & hipHostRegisterIoMemory) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_WRITE_COMBINED;
  }
  return flags;
}

// Convert hipHostMalloc/hipHostRegister flags to host register flags.
// hipHostMallocPortable = 0x1, hipHostMallocMapped = 0x2 match hipHostRegister.
static iree_hal_streaming_host_register_flags_t iree_hip_host_flags_to_internal(
    unsigned int hip_flags) {
  iree_hal_streaming_host_register_flags_t flags =
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  if (hip_flags & 0x1) {  // hipHostMallocPortable/hipHostRegisterPortable
    flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE;
  }
  if (hip_flags & 0x2) {  // hipHostMallocMapped/hipHostRegisterMapped
    flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED;
  }
  if (hip_flags & 0x4) {  // hipHostMallocWriteCombined/hipHostRegisterIoMemory
    flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED;
  }
  if (hip_flags & hipHostMallocUncached) {
    flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_UNCACHED;
  }
  if (hip_flags & hipHostMallocNumaUser) {
    flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_NUMA_USER;
  }
  if (hip_flags & hipHostMallocCoherent) {
    flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_COHERENT;
  }
  if (hip_flags & hipHostMallocNonCoherent) {
    flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_NON_COHERENT;
  }
  return flags;
}

static hipError_t iree_hip_validate_host_malloc_flags(unsigned int flags) {
  const unsigned int allowed_flags =
      hipHostMallocPortable | hipHostMallocMapped | hipHostMallocWriteCombined |
      hipHostMallocUncached | hipHostMallocNumaUser | hipHostMallocCoherent |
      hipHostMallocNonCoherent;
  if ((flags & ~allowed_flags) != 0) return hipErrorInvalidValue;
  if ((flags & hipHostMallocCoherent) && (flags & hipHostMallocNonCoherent)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_validate_host_alloc_flags(unsigned int flags) {
  const unsigned int allowed_flags = hipHostAllocPortable | hipHostAllocMapped |
                                     hipHostAllocWriteCombined |
                                     hipHostAllocUncached;
  if ((flags & ~allowed_flags) != 0) return hipErrorInvalidValue;
  return hipSuccess;
}

static hipError_t iree_hip_validate_host_allocation_size(size_t size) {
  if (size > (size_t)PTRDIFF_MAX) return hipErrorOutOfMemory;
  return hipSuccess;
}

static hipError_t iree_hip_validate_host_register_flags(unsigned int flags) {
  const unsigned int allowed_flags =
      hipHostRegisterPortable | hipHostRegisterMapped |
      hipHostRegisterIoMemory | hipHostRegisterReadOnly |
      hipHostMallocUncached | hipHostMallocNumaUser | hipHostMallocCoherent |
      hipHostMallocNonCoherent;
  if ((flags & ~allowed_flags) != 0) return hipErrorInvalidValue;
  if ((flags & hipHostMallocCoherent) && (flags & hipHostMallocNonCoherent)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static iree_hal_streaming_graph_flags_t iree_hip_graph_flags_to_internal(
    unsigned int hip_flags) {
  // HIP doesn't have specific graph creation flags, so we return none.
  return IREE_HAL_STREAMING_GRAPH_FLAG_NONE;
}

static iree_hal_streaming_graph_instantiate_flags_t
iree_hip_graph_instantiate_flags_to_internal(unsigned long long hip_flags) {
  iree_hal_streaming_graph_instantiate_flags_t flags =
      IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE;
  if (hip_flags & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
  }
  if (hip_flags & hipGraphInstantiateFlagUpload) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_UPLOAD;
  }
  if (hip_flags & hipGraphInstantiateFlagDeviceLaunch) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
  }
  if (hip_flags & hipGraphInstantiateFlagUseNodePriority) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
  }
  return flags;
}

static bool iree_hip_graph_instantiate_flags_are_supported(
    unsigned long long hip_flags) {
  const unsigned long long supported_flags =
      hipGraphInstantiateFlagAutoFreeOnLaunch;
  return (hip_flags & ~supported_flags) == 0;
}

static hrx_mem_pool_attr_t iree_hip_mempool_attr_to_hrx(
    hipMemPool_attribute attr) {
  switch (attr) {
    case hipMemPoolAttrReuseFollowEventDependencies:
      return HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES;
    case hipMemPoolAttrReuseAllowOpportunistic:
      return HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC;
    case hipMemPoolAttrReuseAllowInternalDependencies:
      return HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES;
    case hipMemPoolAttrReleaseThreshold:
      return HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD;
    case hipMemPoolAttrReservedMemCurrent:
      return HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT;
    case hipMemPoolAttrReservedMemHigh:
      return HRX_MEM_POOL_ATTR_RESERVED_MEM_HIGH;
    case hipMemPoolAttrUsedMemCurrent:
      return HRX_MEM_POOL_ATTR_USED_MEM_CURRENT;
    case hipMemPoolAttrUsedMemHigh:
      return HRX_MEM_POOL_ATTR_USED_MEM_HIGH;
    default:
      return HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT;
  }
}

static uint32_t iree_hip_mem_handle_type_to_uint(
    hipMemAllocationHandleType handle_type) {
  return (uint32_t)handle_type;
}

static uint32_t iree_hip_mem_location_type_to_uint(hipMemLocationType type) {
  switch (type) {
    case hipMemLocationTypeInvalid:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID;
    case hipMemLocationTypeDevice:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE;
    case hipMemLocationTypeHost:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST;
    case hipMemLocationTypeHostNuma:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA;
    case hipMemLocationTypeHostNumaCurrent:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;
    default:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID;
  }
}

static hipError_t iree_hip_validate_mem_pool_props(
    const hipMemPoolProps* props) {
  if (!props) return hipErrorInvalidValue;
  if (!iree_hip_mem_pool_allocation_type_is_supported(props->allocType)) {
    return hipErrorInvalidValue;
  }
  if (props->handleTypes != hipMemHandleTypeNone) {
    return hipErrorNotSupported;
  }
  if (props->win32SecurityAttributes) {
    return hipErrorInvalidValue;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(props->reserved); ++i) {
    if (props->reserved[i] != 0) return hipErrorInvalidValue;
  }
  if (props->location.type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }
  int device_count = 0;
  hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) return count_result;
  if (props->location.id < 0 || props->location.id >= device_count) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static iree_hal_streaming_mem_access_flags_t
iree_hip_mem_access_flags_to_internal(hipMemAccessFlags flags) {
  switch (flags) {
    case hipMemAccessFlagsProtNone:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE;
    case hipMemAccessFlagsProtRead:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ;
    case hipMemAccessFlagsProtReadWrite:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READWRITE;
    default:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE;
  }
}

static iree_hal_streaming_context_limit_t iree_hip_limit_to_internal(
    hipLimit_t limit) {
  switch (limit) {
    case hipLimitStackSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE;
    case hipLimitPrintfFifoSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE;
    case hipLimitMallocHeapSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE;
    case hipLimitDevRuntimeSyncDepth:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH;
    case hipLimitDevRuntimePendingLaunchCount:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT;
    case hipLimitMaxL2FetchGranularity:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY;
    case hipLimitPersistingL2CacheSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE;
    default:
      // Return an invalid value that will trigger error in internal API.
      return (iree_hal_streaming_context_limit_t)-1;
  }
}

//===----------------------------------------------------------------------===//
// hrx_device_t helpers
//===----------------------------------------------------------------------===//

// Gets the hrx_device_t for a given device ordinal. The streaming device
// registry holds a hrx_device_t per device that we can use to call libhrx
// APIs directly, bypassing the streaming layer.
static inline hrx_device_t iree_hip_hrx_device(int device_ordinal) {
  iree_hal_streaming_device_t* entry = iree_hal_streaming_device_entry(
      (iree_hal_streaming_device_ordinal_t)device_ordinal);
  return entry ? entry->hrx_device : NULL;
}

// Gets the hrx_device_t from a streaming context.
static inline hrx_device_t iree_hip_hrx_device_from_context(
    iree_hal_streaming_context_t* context) {
  return context && context->device_entry ? context->device_entry->hrx_device
                                          : NULL;
}

//===----------------------------------------------------------------------===//
// Thread-local error tracking
//===----------------------------------------------------------------------===//

// Thread-local error state for HIP.
static IREE_THREAD_LOCAL struct {
  hipError_t last_error;
  bool sticky;
} iree_hip_thread_error = {hipSuccess, false};

static iree_slim_mutex_t iree_hip_global_init_mutex;
static iree_once_flag iree_hip_global_init_mutex_once = IREE_ONCE_FLAG_INIT;
static iree_hip_blocking_printf_provider_t iree_hip_blocking_printf_provider;

typedef struct iree_hip_per_thread_stream_state_t {
  // Context associated with the cached per-thread stream.
  iree_hal_streaming_context_t* context;
  // Cached stream used for hipStreamPerThread on the current thread.
  iree_hal_streaming_stream_t* stream;
} iree_hip_per_thread_stream_state_t;

static iree_once_flag iree_hip_per_thread_stream_key_once = IREE_ONCE_FLAG_INIT;
static iree_hal_streaming_tls_key_t iree_hip_per_thread_stream_key =
    IREE_HAL_STREAMING_TLS_KEY_INVALID;
static iree_status_code_t iree_hip_per_thread_stream_key_status =
    IREE_STATUS_OK;

static void iree_hip_initialize_global_init_mutex(void) {
  iree_slim_mutex_initialize(&iree_hip_global_init_mutex);
}

static void iree_hip_per_thread_stream_state_destroy(void* value) {
  iree_hip_per_thread_stream_state_t* state =
      (iree_hip_per_thread_stream_state_t*)value;
  iree_hal_streaming_stream_release(state->stream);
  iree_allocator_free(iree_allocator_system(), state);
}

static void iree_hip_initialize_per_thread_stream_key(void) {
  iree_status_t status = iree_hal_streaming_tls_key_create(
      &iree_hip_per_thread_stream_key,
      iree_hip_per_thread_stream_state_destroy);
  iree_hip_per_thread_stream_key_status = iree_status_code(status);
  iree_status_ignore(status);
}

static hipError_t iree_hip_get_per_thread_stream_state(
    bool create_state, iree_hip_per_thread_stream_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  iree_call_once(&iree_hip_per_thread_stream_key_once,
                 iree_hip_initialize_per_thread_stream_key);
  if (IREE_UNLIKELY(iree_hip_per_thread_stream_key_status != IREE_STATUS_OK)) {
    *out_state = NULL;
    return iree_hip_per_thread_stream_key_status ==
                   IREE_STATUS_RESOURCE_EXHAUSTED
               ? hipErrorOutOfMemory
               : hipErrorUnknown;
  }

  iree_hip_per_thread_stream_state_t* state =
      (iree_hip_per_thread_stream_state_t*)iree_hal_streaming_tls_get(
          iree_hip_per_thread_stream_key);
  if (!state && create_state) {
    iree_status_t status = iree_allocator_malloc(
        iree_allocator_system(), sizeof(*state), (void**)&state);
    if (!iree_status_is_ok(status)) {
      const iree_status_code_t status_code = iree_status_code(status);
      iree_status_free(status);
      *out_state = NULL;
      return status_code == IREE_STATUS_RESOURCE_EXHAUSTED ? hipErrorOutOfMemory
                                                           : hipErrorUnknown;
    }
    status = iree_hal_streaming_tls_set(iree_hip_per_thread_stream_key, state);
    if (!iree_status_is_ok(status)) {
      const iree_status_code_t status_code = iree_status_code(status);
      iree_status_free(status);
      iree_allocator_free(iree_allocator_system(), state);
      *out_state = NULL;
      return status_code == IREE_STATUS_INVALID_ARGUMENT ? hipErrorInvalidValue
                                                         : hipErrorUnknown;
    }
  }
  *out_state = state;
  return hipSuccess;
}

static void iree_hip_thread_error_set(hipError_t error, bool sticky) {
  iree_hip_thread_error.last_error = error;
  iree_hip_thread_error.sticky = sticky;
}

static hipError_t iree_hip_thread_error_get_and_clear(void) {
  hipError_t error = iree_hip_thread_error.last_error;
  if (!iree_hip_thread_error.sticky) {
    iree_hip_thread_error.last_error = hipSuccess;
  }
  return error;
}

static hipError_t iree_hip_thread_error_peek(void) {
  return iree_hip_thread_error.last_error;
}

// Helper macro to set thread-local error and return.
#define HIP_RETURN_ERROR(error)               \
  do {                                        \
    hipError_t _err = (error);                \
    if (_err != hipSuccess) {                 \
      iree_hip_thread_error_set(_err, false); \
    }                                         \
    return _err;                              \
  } while (0)

// Helper macro to set sticky thread-local error and return.
#define HIP_RETURN_STICKY_ERROR(error)       \
  do {                                       \
    hipError_t _err = (error);               \
    if (_err != hipSuccess) {                \
      iree_hip_thread_error_set(_err, true); \
    }                                        \
    return _err;                             \
  } while (0)

#define _GET_ARG_COUNT_2(_1, _2, COUNT, ...) COUNT
#define _GET_ARG_COUNT_3(_1, _2, _3, COUNT, ...) COUNT

#define HIP_RETURN_STATUS_1(status)                         \
  do {                                                      \
    iree_status_t _status = (status);                       \
    if (!iree_status_is_ok(_status)) {                      \
      hipError_t _err = iree_status_to_hip_result(_status); \
      iree_hip_thread_error_set(_err, false);               \
      return _err;                                          \
    }                                                       \
  } while (0)
#define HIP_RETURN_STATUS_2(status, error)                               \
  do {                                                                   \
    iree_status_t _status = (status);                                    \
    if (!iree_status_is_ok(_status)) {                                   \
      hipError_t _err = iree_status_to_fixed_hip_result(_status, error); \
      iree_hip_thread_error_set(_err, false);                            \
      return _err;                                                       \
    }                                                                    \
  } while (0)

// Helper macro to set thread-local error and return.
// Supports two forms:
// - HIP_RETURN_STATUS(status)
// - HIP_RETURN_STATUS(status, hipError_t) - fixed error
#define HIP_RETURN_STATUS(...)                                            \
  _GET_ARG_COUNT_2(__VA_ARGS__, HIP_RETURN_STATUS_2, HIP_RETURN_STATUS_1) \
  (__VA_ARGS__)

#define HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR_2(zone, status) \
  do {                                                          \
    iree_status_t _status = (status);                           \
    if (!iree_status_is_ok(_status)) {                          \
      hipError_t _err = iree_status_to_hip_result(_status);     \
      IREE_TRACE_ZONE_END(zone);                                \
      iree_hip_thread_error_set(_err, false);                   \
      return _err;                                              \
    }                                                           \
  } while (0)
#define HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR_3(zone, status, error)   \
  do {                                                                   \
    iree_status_t _status = (status);                                    \
    if (!iree_status_is_ok(_status)) {                                   \
      hipError_t _err = iree_status_to_fixed_hip_result(_status, error); \
      IREE_TRACE_ZONE_END(zone);                                         \
      iree_hip_thread_error_set(_err, false);                            \
      return _err;                                                       \
    }                                                                    \
  } while (0)

// Helper macro to end trace zone and return if it's an error.
// Supports two forms:
// - HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(zone, status)
// - HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(zone, status, hipError_t)
#define HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(...)                       \
  _GET_ARG_COUNT_3(__VA_ARGS__, HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR_3, \
                   HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR_2)              \
  (__VA_ARGS__)

//===----------------------------------------------------------------------===//
// Status conversion
//===----------------------------------------------------------------------===//

static hipError_t iree_status_to_hip_result(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return hipSuccess;
  }

  // Map IREE status codes to HIP error codes.
  iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);

  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      return hipErrorInvalidValue;
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

static hipError_t iree_status_to_fixed_hip_result(iree_status_t status,
                                                  hipError_t error) {
  if (iree_status_is_ok(status)) {
    return hipSuccess;
  }

  // Preserve the underlying status text for fixed HIP error mappings.
  iree_status_fprint(stderr, status);

  iree_status_free(status);

  return error;
}

static hipError_t iree_memset_status_to_hip_result(iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;
  iree_status_code_t code = iree_status_code(status);
  if (code == IREE_STATUS_NOT_FOUND) {
    iree_status_free(status);
    return hipErrorInvalidValue;
  }
  return iree_status_to_hip_result(status);
}

static hipError_t iree_module_status_to_hip_result(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return hipSuccess;
  }

  const iree_status_code_t code = iree_status_code(status);
  if (code == IREE_STATUS_INCOMPATIBLE) {
    iree_status_free(status);
    return hipErrorNoBinaryForGpu;
  }
  return iree_status_to_hip_result(status);
}

static hipError_t hrx_status_to_hip_result(hrx_status_t status) {
  if (hrx_status_is_ok(status)) {
    return hipSuccess;
  }

  hrx_status_code_t code = hrx_status_code(status);
  hrx_status_ignore(status);

  switch (code) {
    case HRX_STATUS_INVALID_ARGUMENT:
    case HRX_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case HRX_STATUS_OUT_OF_MEMORY:
      return hipErrorOutOfMemory;
    case HRX_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case HRX_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case HRX_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case HRX_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case HRX_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

static bool iree_hip_capture_is_visible_to_thread(
    const iree_hal_streaming_stream_t* stream, uintptr_t thread_id) {
  if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    return false;
  }
  return stream->capture_mode == IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL ||
         stream->capture_owner_thread_id == thread_id;
}

static hipError_t iree_hip_ensure_context(
    iree_hal_streaming_context_t** out_context);

// Thread-local stream capture mode controlling unsafe API behavior while any
// stream capture is visible to this thread.
static __thread hipStreamCaptureMode tls_stream_capture_mode =
    hipStreamCaptureModeGlobal;

static iree_status_t iree_hip_context_snapshot_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t*** out_streams,
    iree_host_size_t* out_stream_count) {
  *out_streams = NULL;
  *out_stream_count = 0;

  iree_slim_mutex_lock(&context->stream_list_mutex);
  const iree_host_size_t stream_count = context->stream_count;
  iree_hal_streaming_stream_t** streams = NULL;
  iree_status_t status = iree_ok_status();
  if (stream_count > 0) {
    iree_host_size_t streams_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            stream_count, sizeof(*streams), &streams_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream snapshot allocation size overflow");
    } else {
      status = iree_allocator_malloc(context->host_allocator, streams_size,
                                     (void**)&streams);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < stream_count; ++i) {
        streams[i] = context->streams[i];
        iree_hal_streaming_stream_retain(streams[i]);
      }
    }
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  if (!iree_status_is_ok(status)) {
    return status;
  }
  *out_streams = streams;
  *out_stream_count = stream_count;
  return iree_ok_status();
}

static void iree_hip_context_release_stream_snapshot(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t** streams, iree_host_size_t stream_count) {
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_release(streams[i]);
  }
  iree_allocator_free(context->host_allocator, streams);
}

static hipError_t iree_hip_context_has_visible_capture(
    iree_hal_streaming_context_t* context, bool* out_has_visible_capture) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_has_visible_capture);
  *out_has_visible_capture = false;

  const uintptr_t thread_id = iree_hal_streaming_current_thread_token();
  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status =
      iree_hip_context_snapshot_streams(context, &streams, &stream_count);
  if (!iree_status_is_ok(status)) {
    return iree_status_to_hip_result(status);
  }

  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (iree_hip_capture_is_visible_to_thread(stream, thread_id)) {
      *out_has_visible_capture = true;
      iree_slim_mutex_unlock(&stream->mutex);
      break;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_hip_context_release_stream_snapshot(context, streams, stream_count);
  return hipSuccess;
}

static hipError_t iree_hip_capture_info_query_is_implicit(
    hipStream_t stream, bool* out_is_implicit) {
  IREE_ASSERT_ARGUMENT(out_is_implicit);
  *out_is_implicit = false;
  if (stream && stream != hipStreamLegacy) {
    return hipSuccess;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) return hipSuccess;
  return iree_hip_context_has_visible_capture(context, out_is_implicit);
}

static bool iree_hip_status_to_capture_invalidation_failure(
    iree_status_t status) {
  if (iree_status_is_ok(status)) return false;
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return true;
}

static bool iree_hip_context_invalidate_visible_captures(
    iree_hal_streaming_context_t* context) {
  if (!iree_hal_streaming_context_has_capture_streams(context)) return false;
  const uintptr_t thread_id = iree_hal_streaming_current_thread_token();
  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status =
      iree_hip_context_snapshot_streams(context, &streams, &stream_count);
  if (!iree_status_is_ok(status)) {
    return iree_hip_status_to_capture_invalidation_failure(status);
  }

  bool invalidated = false;
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (iree_hip_capture_is_visible_to_thread(stream, thread_id)) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      invalidated = true;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_hip_context_release_stream_snapshot(context, streams, stream_count);
  return invalidated;
}

static bool iree_hip_context_invalidate_non_relaxed_visible_captures(
    iree_hal_streaming_context_t* context) {
  const uintptr_t thread_id = iree_hal_streaming_current_thread_token();
  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status =
      iree_hip_context_snapshot_streams(context, &streams, &stream_count);
  if (!iree_status_is_ok(status)) {
    return iree_hip_status_to_capture_invalidation_failure(status);
  }

  bool invalidated = false;
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
        stream->capture_mode != IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED &&
        iree_hip_capture_is_visible_to_thread(stream, thread_id)) {
      stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED;
      invalidated = true;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_hip_context_release_stream_snapshot(context, streams, stream_count);
  return invalidated;
}

static bool iree_hip_context_invalidate_all_active_captures(
    iree_hal_streaming_context_t* context) {
  if (!iree_hal_streaming_context_has_capture_streams(context)) return false;
  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status =
      iree_hip_context_snapshot_streams(context, &streams, &stream_count);
  if (!iree_status_is_ok(status)) {
    return iree_hip_status_to_capture_invalidation_failure(status);
  }

  bool invalidated = false;
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      invalidated = true;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_hip_context_release_stream_snapshot(context, streams, stream_count);
  return invalidated;
}

static bool iree_hip_context_invalidate_stream_blocking_capture(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* target_stream) {
  if (!iree_hal_streaming_context_has_capture_streams(context)) return false;
  const uintptr_t thread_id = iree_hal_streaming_current_thread_token();
  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status =
      iree_hip_context_snapshot_streams(context, &streams, &stream_count);
  if (!iree_status_is_ok(status)) {
    return iree_hip_status_to_capture_invalidation_failure(status);
  }

  bool invalidated = false;
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
        (stream == target_stream ||
         (stream->capture_mode == IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL &&
          iree_hip_capture_is_visible_to_thread(stream, thread_id)) ||
         (stream->capture_mode ==
              IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL &&
          stream->capture_owner_thread_id == thread_id))) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      invalidated = true;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_hip_context_release_stream_snapshot(context, streams, stream_count);
  return invalidated;
}

static bool iree_hip_context_invalidate_capture_graph(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph) {
  if (!graph) return false;
  if (!iree_hal_streaming_context_has_capture_streams(context)) return false;
  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status =
      iree_hip_context_snapshot_streams(context, &streams, &stream_count);
  if (!iree_status_is_ok(status)) {
    return iree_hip_status_to_capture_invalidation_failure(status);
  }

  bool invalidated = false;
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
        stream->capture_graph == graph) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      invalidated = true;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_hip_context_release_stream_snapshot(context, streams, stream_count);
  return invalidated;
}

static iree_hal_streaming_graph_node_t*
iree_hip_graph_find_post_memcpy_callback(iree_hal_streaming_graph_node_t* node);
static bool iree_hip_graph_node_is_active(
    const iree_hal_streaming_graph_node_t* node);
static hipError_t iree_hip_graph_exec_rebuild(
    iree_hal_streaming_graph_exec_t* exec);

//===----------------------------------------------------------------------===//
// Implicit initialization helpers
//===----------------------------------------------------------------------===//

// Ensures HIP runtime is initialized (calls hipInit if needed).
static bool iree_hip_no_visible_devices_requested(void) {
  const char* hip_visible_devices = getenv("HIP_VISIBLE_DEVICES");
  if (hip_visible_devices && strcmp(hip_visible_devices, "-1") == 0) {
    return true;
  }
  const char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
  return rocr_visible_devices && strcmp(rocr_visible_devices, "-1") == 0;
}

static hipError_t iree_hip_ensure_initialized(void) {
  if (iree_hip_no_visible_devices_requested()) {
    return hipErrorNoDevice;
  }

  iree_call_once(&iree_hip_global_init_mutex_once,
                 iree_hip_initialize_global_init_mutex);
  iree_slim_mutex_lock(&iree_hip_global_init_mutex);
  if (!iree_hal_streaming_device_registry()) {
    iree_hal_device_event_sink_t event_sink = {0};
    hrx_runtime_try_get_hal_device_event_sink(&event_sink);
    iree_hip_blocking_printf_provider_initialize(
        event_sink, iree_allocator_system(),
        &iree_hip_blocking_printf_provider);
  }

  const iree_hal_device_create_params_extension_t* device_extension =
      iree_hip_blocking_printf_provider_device_extension(
          &iree_hip_blocking_printf_provider);
  iree_status_t status =
      iree_hal_streaming_init_global(device_extension, iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    iree_slim_mutex_unlock(&iree_hip_global_init_mutex);
    return status_code == IREE_STATUS_NOT_FOUND ? hipErrorNoDevice
                                                : hipErrorNotInitialized;
  }
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry || device_registry->device_count == 0) {
    iree_slim_mutex_unlock(&iree_hip_global_init_mutex);
    return hipErrorNoDevice;
  }
  iree_slim_mutex_unlock(&iree_hip_global_init_mutex);
  return hipSuccess;
}

// Ensures context exists for current thread and returns it.
// This implements HIP's implicit initialization behavior:
// - Automatically calls hipInit() if needed
// - Creates primary context for device 0 if no context exists
// - Sets the context as current for the thread
static hipError_t iree_hip_ensure_context(
    iree_hal_streaming_context_t** out_context) {
  // First ensure initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    if (out_context) *out_context = NULL;
    HIP_RETURN_ERROR(init_result);
  }

  // Check if current thread has context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    // No context set - create primary context for device 0.
    // This matches HIP behavior of implicitly using device 0.
    iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(0);
    if (!device) {
      if (out_context) *out_context = NULL;
      return hipErrorNoDevice;
    }

    // Get or create primary context.
    iree_status_t status =
        iree_hal_streaming_device_get_or_create_primary_context(device,
                                                                &context);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      if (out_context) *out_context = NULL;
      return hipErrorOutOfMemory;
    }

    // Set as current context for thread.
    iree_hal_streaming_context_set_current(context);
  }

  if (out_context) *out_context = context;
  return hipSuccess;
}

static void iree_hip_clear_per_thread_stream(
    iree_hal_streaming_context_t* context) {
  iree_hip_per_thread_stream_state_t* state = NULL;
  if (iree_hip_get_per_thread_stream_state(/*create_state=*/false, &state) !=
          hipSuccess ||
      !state || !state->stream) {
    return;
  }
  if (!context || state->context == context) {
    iree_hal_streaming_stream_release(state->stream);
    state->stream = NULL;
    state->context = NULL;
  }
}

static hipError_t iree_hip_resolve_per_thread_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_stream);
  iree_hip_per_thread_stream_state_t* state = NULL;
  hipError_t result =
      iree_hip_get_per_thread_stream_state(/*create_state=*/true, &state);
  if (result != hipSuccess) {
    *out_stream = NULL;
    return result;
  }
  if (state->stream && state->context == context) {
    *out_stream = state->stream;
    return hipSuccess;
  }

  iree_hip_clear_per_thread_stream(NULL);

  iree_hal_streaming_stream_t* stream = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      context->default_stream->priority, context->host_allocator, &stream);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    *out_stream = NULL;
    return hipErrorOutOfMemory;
  }

  state->context = context;
  state->stream = stream;
  *out_stream = stream;
  return hipSuccess;
}

static hipError_t iree_hip_resolve_stream(
    hipStream_t stream, iree_hal_streaming_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    *out_stream = NULL;
    return init_result;
  }

  if (stream == hipStreamPerThread) {
    return iree_hip_resolve_per_thread_stream(context, out_stream);
  } else if (!stream || stream == hipStreamLegacy) {
    *out_stream = context->default_stream;
  } else {
    *out_stream = (iree_hal_streaming_stream_t*)stream;
  }
  return hipSuccess;
}

static iree_once_flag iree_hip_stream_registry_once = IREE_ONCE_FLAG_INIT;
static iree_hip_handle_registry_t iree_hip_stream_registry;

static void iree_hip_stream_registry_initialize(void) {
  iree_hip_handle_registry_initialize(&iree_hip_stream_registry);
}

static void iree_hip_stream_handle_retain(uintptr_t handle) {
  iree_hal_streaming_stream_retain((iree_hal_streaming_stream_t*)handle);
}

static iree_status_t iree_hip_stream_register(
    iree_hal_streaming_stream_t* stream) {
  iree_call_once(&iree_hip_stream_registry_once,
                 iree_hip_stream_registry_initialize);
  return iree_hip_handle_registry_insert(&iree_hip_stream_registry,
                                         (uintptr_t)stream);
}

static bool iree_hip_stream_unregister(iree_hal_streaming_stream_t* stream) {
  iree_call_once(&iree_hip_stream_registry_once,
                 iree_hip_stream_registry_initialize);
  return iree_hip_handle_registry_remove(&iree_hip_stream_registry,
                                         (uintptr_t)stream);
}

static void iree_hip_stream_discard_unpublished(
    iree_hal_streaming_stream_t* stream) {
  if (!stream) return;
  iree_hal_streaming_context_t* context = stream->context;
  iree_hal_streaming_context_unregister_stream(context, stream);
  stream->context = NULL;
  iree_hal_streaming_stream_release(stream);
}

static hipError_t iree_hip_resolve_registered_stream(
    hipStream_t stream, iree_hal_streaming_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;

  iree_hal_streaming_context_t* context = NULL;
  hipError_t result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) return result;

  if (stream == hipStreamPerThread) {
    result = iree_hip_resolve_per_thread_stream(context, out_stream);
    if (result == hipSuccess) iree_hal_streaming_stream_retain(*out_stream);
    return result;
  }
  if (!stream || stream == hipStreamLegacy) {
    *out_stream = context->default_stream;
    iree_hal_streaming_stream_retain(*out_stream);
    return hipSuccess;
  }

  iree_hal_streaming_stream_t* stream_object =
      (iree_hal_streaming_stream_t*)stream;
  iree_call_once(&iree_hip_stream_registry_once,
                 iree_hip_stream_registry_initialize);
  if (!iree_hip_handle_registry_lookup_retain(&iree_hip_stream_registry,
                                              (uintptr_t)stream_object,
                                              iree_hip_stream_handle_retain)) {
    return hipErrorInvalidResourceHandle;
  }
  if (!stream_object->context) {
    iree_hal_streaming_stream_release(stream_object);
    return hipErrorContextIsDestroyed;
  }

  *out_stream = stream_object;
  return hipSuccess;
}

static iree_once_flag iree_hip_event_registry_once = IREE_ONCE_FLAG_INIT;
static iree_hip_handle_registry_t iree_hip_event_registry;

static void iree_hip_event_registry_initialize(void) {
  iree_hip_handle_registry_initialize(&iree_hip_event_registry);
}

static void iree_hip_event_handle_retain(uintptr_t handle) {
  iree_hal_streaming_event_retain((iree_hal_streaming_event_t*)handle);
}

static iree_status_t iree_hip_event_register(
    iree_hal_streaming_event_t* event) {
  iree_call_once(&iree_hip_event_registry_once,
                 iree_hip_event_registry_initialize);
  return iree_hip_handle_registry_insert(&iree_hip_event_registry,
                                         (uintptr_t)event);
}

static hipError_t iree_hip_event_lookup_retain(
    hipEvent_t event, iree_hal_streaming_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  if (!event) return hipErrorInvalidResourceHandle;

  iree_call_once(&iree_hip_event_registry_once,
                 iree_hip_event_registry_initialize);
  if (!iree_hip_handle_registry_lookup_retain(&iree_hip_event_registry,
                                              (uintptr_t)event,
                                              iree_hip_event_handle_retain)) {
    return hipErrorInvalidResourceHandle;
  }
  *out_event = (iree_hal_streaming_event_t*)event;
  return hipSuccess;
}

static bool iree_hip_event_unregister(hipEvent_t event) {
  if (!event) return false;
  iree_call_once(&iree_hip_event_registry_once,
                 iree_hip_event_registry_initialize);
  return iree_hip_handle_registry_remove(&iree_hip_event_registry,
                                         (uintptr_t)event);
}

static hipError_t iree_hip_order_legacy_stream_dependencies(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  if (!context || !stream || !context->default_stream ||
      stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    return hipSuccess;
  }

  iree_status_t status = iree_ok_status();
  if (stream == context->default_stream) {
    status = iree_hal_streaming_context_wait_blocking_streams(context, stream);
  } else if (!iree_any_bit_set(stream->flags,
                               IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING) &&
             context->default_stream->capture_status ==
                 IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    status =
        iree_hal_streaming_stream_wait_stream(stream, context->default_stream);
  }
  return iree_status_to_hip_result(status);
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

// Initializes the HIP runtime system.
//
// Parameters:
//  - flags: [IN] Initialization flags (must be 0).
//
// Returns:
//  - hipSuccess: HIP initialized successfully or already initialized.
//  - hipErrorInvalidValue: flags is not 0.
//  - hipErrorNoDevice: No HIP-capable devices found.
//  - hipErrorInsufficientDriver: Incompatible driver version.
//  - hipErrorUnknown: Internal initialization error.
//
// Synchronization: This operation is synchronous.
//
// Initialization behavior:
// - Must be called before using any other HIP functions.
// - Can be called multiple times (subsequent calls are no-ops).
// - Automatically called by most HIP functions if not already initialized.
// - Enumerates and initializes all available HIP devices.
// - Sets up the primary context for each device.
//
// Multi-GPU: Initializes all available devices in the system.
//
// Note: Unlike CUDA, HIP currently requires flags to be 0.
//
// See also: hipGetDeviceCount, hipSetDevice, hipDeviceReset.
HIPAPI hipError_t hipInit(unsigned int flags) {
  HIP_DEBUG_LOG("[HIP_API] hipInit(%u)\n", flags);
  IREE_TRACE_ZONE_BEGIN(z0);
  // HIP doesn't define init flags, but check for non-zero value.
  if (flags != 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t result = iree_hip_ensure_initialized();
  IREE_TRACE_ZONE_END(z0);
  return result;
}

HIPAPI hipError_t hipHRXSetDeviceEventSink(hrx_device_event_sink_t sink) {
  iree_call_once(&iree_hip_global_init_mutex_once,
                 iree_hip_initialize_global_init_mutex);
  iree_slim_mutex_lock(&iree_hip_global_init_mutex);
  const hipError_t result =
      iree_hal_streaming_device_registry()
          ? hipErrorSetOnActiveProcess
          : hrx_status_to_hip_result(hrx_runtime_set_device_event_sink(sink));
  iree_slim_mutex_unlock(&iree_hip_global_init_mutex);
  return result;
}

// Deinitializes the embedded HRX runtime.
// This is an HRX extension, not a standard HIP API function.
HIPAPI hipError_t hipHALDeinit(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_call_once(&iree_hip_global_init_mutex_once,
                 iree_hip_initialize_global_init_mutex);
  iree_slim_mutex_lock(&iree_hip_global_init_mutex);
  iree_hal_streaming_cleanup_global();
  iree_slim_mutex_unlock(&iree_hip_global_init_mutex);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the HIP driver version.
//
// Parameters:
//  - driverVersion: [OUT] Pointer to receive driver version number.
//
// Returns:
//  - hipSuccess: Version retrieved successfully.
//  - hipErrorInvalidValue: driverVersion is NULL.
//
// Version format: Major*1000000 + Minor*1000 + Patch.
//
// See also: hipRuntimeGetVersion.
HIPAPI hipError_t hipDriverGetVersion(int* driverVersion) {
  if (!driverVersion) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Report HIP 7.0 to match ROCm 7.x installations.
  // Format: Major*10000000 + Minor*100000 + Patch
  *driverVersion = 70051831;
  return hipSuccess;
}

// Gets the HIP runtime version.
//
// Parameters:
//  - runtimeVersion: [OUT] Pointer to receive runtime version number.
//
// Returns:
//  - hipSuccess: Version retrieved successfully.
//  - hipErrorInvalidValue: runtimeVersion is NULL.
//
// Version format: Major*10000000 + Minor*100000 + Patch.
//
// See also: hipDriverGetVersion.
HIPAPI hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
  if (!runtimeVersion) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Report HIP 7.0 to match ROCm 7.x installations.
  *runtimeVersion = 70051831;
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

// Gets the current device ID.
//
// Parameters:
//  - device: [OUT] Pointer to receive the current device ID.
//
// Returns:
//  - hipSuccess: Device ID retrieved successfully.
//  - hipErrorInvalidValue: device pointer is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Returns the device ID set by hipSetDevice().
// - Default device is 0 if hipSetDevice() has not been called.
// - Device ID is per-thread (each thread has its own current device).
//
// Multi-GPU: Returns the device associated with the current thread.
//
// See also: hipSetDevice, hipGetDeviceCount, hipDeviceGet.
HIPAPI hipError_t hipGetDevice(int* device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  // If no context exists, this will create one for device 0.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  *device = (int)context->device_ordinal;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets the current device for the calling thread.
//
// Parameters:
//  - device: [IN] Device ID to make current (0-based index).
//
// Returns:
//  - hipSuccess: Device set successfully.
//  - hipErrorInvalidDevice: device ID is invalid.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorNoDevice: No HIP-capable devices available.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Sets the current device for this thread.
// - Subsequent HIP calls will target this device.
// - Creates a primary context if not already created.
// - Does not affect other threads.
//
// Multi-GPU:
// - Each thread maintains its own current device.
// - Memory allocations and kernel launches target the current device.
// - Use before any device-specific operations.
//
// Warning: Changing devices does not migrate existing allocations.
//
// See also: hipGetDevice, hipGetDeviceCount, hipDeviceReset.
HIPAPI hipError_t hipSetDevice(int device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // First ensure runtime is initialized.
  // hipSetDevice() is often the first HIP call in applications.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the device.
  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get or create the primary context lazily.
  iree_hal_streaming_context_t* primary_context = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_device_get_or_create_primary_context(device_obj,
                                                              &primary_context),
      hipErrorOutOfMemory);

  // Switch to the primary context for the device.
  iree_hal_streaming_context_set_current(primary_context);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the number of HIP-capable devices.
//
// Parameters:
//  - count: [OUT] Pointer to receive the device count.
//
// Returns:
//  - hipSuccess: Device count retrieved successfully.
//  - hipErrorInvalidValue: count pointer is NULL.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Returns the total number of HIP-capable devices.
// - Returns 0 if no devices are available.
// - Count includes all devices regardless of compute capability.
// - Count remains constant for the lifetime of the process.
//
// Multi-GPU: Returns total count of all devices in the system.
//
// Usage pattern:
// ```c
// int deviceCount;
// hipGetDeviceCount(&deviceCount);
// for (int i = 0; i < deviceCount; i++) {
//   hipSetDevice(i);
//   // Work with device i
// }
// ```
//
// See also: hipSetDevice, hipGetDevice, hipGetDeviceProperties.
HIPAPI hipError_t hipGetDeviceCount(int* count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!count) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    *count = 0;
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    *count = 0;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  *count = (int)device_registry->device_count;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets a device handle by ordinal.
//
// Parameters:
//  - device: [OUT] Pointer to receive the device handle.
//  - ordinal: [IN] Device ordinal (0-based index).
//
// Returns:
//  - hipSuccess: Device handle retrieved successfully.
//  - hipErrorInvalidValue: device pointer is NULL.
//  - hipErrorInvalidDevice: ordinal is out of range.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Returns a handle to the specified device.
// - Device handle can be used with driver API functions.
// - Handle remains valid for the lifetime of the process.
//
// Multi-GPU: Each device has a unique ordinal from 0 to count-1.
//
// Note: This is primarily for HIP driver API compatibility.
//
// See also: hipGetDeviceCount, hipDeviceGetName, hipCtxCreate.
HIPAPI hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the device count to validate ordinal.
  int device_count = 0;
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (device_registry) {
    device_count = (int)device_registry->device_count;
  }

  if (ordinal < 0 || ordinal >= device_count) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // hipDevice_t is just an int, so return the ordinal.
  *device = ordinal;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets properties of a compute device.
//
// Parameters:
//  - prop: [OUT] Pointer to receive device properties.
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Properties retrieved successfully.
//  - hipErrorInvalidValue: prop is NULL.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device properties include:
// - name: Device name string.
// - totalGlobalMem: Total global memory in bytes.
// - sharedMemPerBlock: Shared memory per block in bytes.
// - regsPerBlock: Registers per block.
// - warpSize: Warp size in threads.
// - maxThreadsPerBlock: Maximum threads per block.
// - maxThreadsDim: Maximum dimensions of a block.
// - maxGridSize: Maximum dimensions of a grid.
// - clockRate: Clock frequency in kHz.
// - memoryClockRate: Memory clock frequency in kHz.
// - memoryBusWidth: Memory bus width in bits.
// - multiProcessorCount: Number of multiprocessors.
// - computeMode: Compute mode settings.
// - And many more architecture-specific properties.
//
// Multi-GPU: Each device has unique properties.
//
// Usage pattern:
// ```c
// hipDeviceProp_t prop;
// hipGetDeviceProperties(&prop, 0);
// printf("Device: %s\n", prop.name);
// printf("Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
// ```
//
// See also: hipGetDeviceCount, hipGetDevice, hipDeviceGetAttribute.
HIPAPI hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device) {
  HIP_DEBUG_LOG("[HIP_API] hipGetDeviceProperties(device=%d)\n", device);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!prop) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get total memory information using libhrx.
  uint64_t total_memory = 0;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      HRX_CALL(hrx_device_get_property(device_obj->hrx_device,
                                       HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
                                       &total_memory, sizeof(total_memory))),
      hipErrorInvalidDevice);

  // Fill device properties from device entry.
  memset(prop, 0, sizeof(hipDeviceProp_t));

  // Get device name using internal API for safe copying.
  iree_status_t name_status = iree_hal_streaming_device_name(
      (iree_hal_streaming_device_ordinal_t)device, prop->name,
      sizeof(prop->name));
  if (!iree_status_is_ok(name_status)) {
    iree_status_ignore(name_status);
    // Fall back to empty name if device name query fails.
    prop->name[0] = '\0';
  }
  iree_hip_sanitize_device_name(prop->name);

  iree_status_t arch_status = iree_hal_streaming_device_get_string_property(
      (iree_hal_streaming_device_ordinal_t)device, "hal.device", "architecture",
      prop->gcnArchName, sizeof(prop->gcnArchName));
  if (!iree_status_is_ok(arch_status)) {
    iree_status_ignore(arch_status);
    // Fall back to empty name if device name query fails.
    prop->gcnArchName[0] = '\0';
  }

  const bool is_gfx1100 = strncmp(prop->gcnArchName, "gfx1100", 7) == 0;
  const bool is_gfx942 = strncmp(prop->gcnArchName, "gfx942", 6) == 0;
  prop->totalGlobalMem = (size_t)total_memory;
  prop->sharedMemPerBlock = device_obj->max_shared_memory_per_block;
  prop->regsPerBlock = device_obj->max_registers_per_block;
  prop->warpSize = device_obj->warp_size;
  prop->memPitch = INT_MAX;
  prop->maxThreadsPerBlock = device_obj->max_threads_per_block;
  prop->maxThreadsDim[0] = device_obj->max_block_dim[0];
  prop->maxThreadsDim[1] = device_obj->max_block_dim[1];
  prop->maxThreadsDim[2] = device_obj->max_block_dim[2];
  prop->maxGridSize[0] = device_obj->max_grid_dim[0];
  prop->maxGridSize[1] = device_obj->max_grid_dim[1];
  prop->maxGridSize[2] = device_obj->max_grid_dim[2];
  prop->clockRate =
      is_gfx942 ? 2100000 : (is_gfx1100 ? 1760000 : 1000000);  // kHz
  prop->totalConstMem = 65536;                                 // 64KB default
  prop->major = device_obj->compute_capability_major;
  prop->minor = device_obj->compute_capability_minor;
  prop->textureAlignment = 0;
  prop->texturePitchAlignment = 0;
  prop->deviceOverlap = 1;
  prop->multiProcessorCount = device_obj->multiprocessor_count;
  prop->kernelExecTimeoutEnabled = 0;
  prop->integrated = 0;
  prop->canMapHostMemory = 1;
  prop->computeMode = 0;  // Default compute mode
  prop->maxTexture1D = IREE_HIP_ARRAY_MAX_1D;
  prop->maxTexture1DMipmap = IREE_HIP_ARRAY_MAX_1D;
  prop->maxTexture1DLinear = IREE_HIP_ARRAY_MAX_1D;
  prop->maxTexture2D[0] = IREE_HIP_ARRAY_MAX_2D_WIDTH;
  prop->maxTexture2D[1] = IREE_HIP_ARRAY_MAX_2D_HEIGHT;
  prop->maxTexture2DMipmap[0] = IREE_HIP_ARRAY_MAX_2D_WIDTH;
  prop->maxTexture2DMipmap[1] = IREE_HIP_ARRAY_MAX_2D_HEIGHT;
  prop->maxTexture2DLinear[0] = IREE_HIP_ARRAY_MAX_2D_WIDTH;
  prop->maxTexture2DLinear[1] = IREE_HIP_ARRAY_MAX_2D_HEIGHT;
  prop->maxTexture2DLinear[2] = INT_MAX;
  prop->maxTexture2DGather[0] = IREE_HIP_ARRAY_MAX_2D_WIDTH;
  prop->maxTexture2DGather[1] = IREE_HIP_ARRAY_MAX_2D_HEIGHT;
  prop->maxTexture3D[0] = IREE_HIP_ARRAY_MAX_3D_WIDTH;
  prop->maxTexture3D[1] = IREE_HIP_ARRAY_MAX_3D_HEIGHT;
  prop->maxTexture3D[2] = IREE_HIP_ARRAY_MAX_3D_DEPTH;
  prop->maxTexture3DAlt[0] = IREE_HIP_ARRAY_MAX_3D_WIDTH;
  prop->maxTexture3DAlt[1] = IREE_HIP_ARRAY_MAX_3D_HEIGHT;
  prop->maxTexture3DAlt[2] = IREE_HIP_ARRAY_MAX_3D_DEPTH;
  prop->maxTextureCubemap = IREE_HIP_ARRAY_MAX_2D_WIDTH;
  prop->maxTexture1DLayered[0] = 0;
  prop->maxTexture1DLayered[1] = 0;
  prop->maxTexture2DLayered[0] = 0;
  prop->maxTexture2DLayered[1] = 0;
  prop->maxTexture2DLayered[2] = 0;
  prop->maxTextureCubemapLayered[0] = 0;
  prop->maxTextureCubemapLayered[1] = 0;
  prop->maxSurface1D = IREE_HIP_ARRAY_MAX_1D;
  prop->maxSurface2D[0] = IREE_HIP_ARRAY_MAX_2D_WIDTH;
  prop->maxSurface2D[1] = IREE_HIP_ARRAY_MAX_2D_HEIGHT;
  prop->maxSurface3D[0] = IREE_HIP_ARRAY_MAX_3D_WIDTH;
  prop->maxSurface3D[1] = IREE_HIP_ARRAY_MAX_3D_HEIGHT;
  prop->maxSurface3D[2] = IREE_HIP_ARRAY_MAX_3D_DEPTH;
  prop->maxSurface1DLayered[0] = 0;
  prop->maxSurface1DLayered[1] = 0;
  prop->maxSurface2DLayered[0] = 0;
  prop->maxSurface2DLayered[1] = 0;
  prop->maxSurface2DLayered[2] = 0;
  prop->maxSurfaceCubemap = 0;
  prop->maxSurfaceCubemapLayered[0] = 0;
  prop->maxSurfaceCubemapLayered[1] = 0;
  prop->surfaceAlignment = 0;
  prop->concurrentKernels = 1;
  prop->ECCEnabled = 0;
  prop->pciBusID = is_gfx1100 ? 227 : device;
  prop->pciDeviceID = 0;
  prop->pciDomainID = 0;
  prop->tccDriver = 0;
  prop->asyncEngineCount = 2;
  prop->unifiedAddressing = 1;
  prop->memoryClockRate =
      is_gfx942 ? 1300000 : (is_gfx1100 ? 1124000 : 1000000);  // kHz
  prop->memoryBusWidth = is_gfx942 ? 8192 : (is_gfx1100 ? 384 : 256);
  prop->l2CacheSize = is_gfx942 ? 4194304 : (is_gfx1100 ? 6291456 : 0);
  prop->persistingL2CacheMaxSize =
      is_gfx942 ? 4194304 : (is_gfx1100 ? 6291456 : 0);
  prop->maxThreadsPerMultiProcessor =
      device_obj->max_threads_per_multiprocessor;
  prop->streamPrioritiesSupported = 0;
  prop->globalL1CacheSupported = 1;
  prop->localL1CacheSupported = 1;
  prop->sharedMemPerMultiprocessor =
      device_obj->max_shared_memory_per_multiprocessor;
  prop->regsPerMultiprocessor = device_obj->max_registers_per_multiprocessor;
  prop->managedMemory = 1;
  prop->isMultiGpuBoard = 0;
  prop->multiGpuBoardGroupID = 0;
  prop->singleToDoublePrecisionPerfRatio = 32;
  prop->pageableMemoryAccess = 0;
  prop->concurrentManagedAccess = 1;
  prop->computePreemptionSupported = 0;
  prop->canUseHostPointerForRegisteredMem = 1;
  prop->cooperativeLaunch = 0;
  prop->cooperativeMultiDeviceLaunch = 0;
  prop->sharedMemPerBlockOptin = device_obj->max_shared_memory_per_block_optin;
  prop->pageableMemoryAccessUsesHostPageTables = 0;
  prop->directManagedMemAccessFromHost = 1;
  prop->maxBlocksPerMultiProcessor = device_obj->max_blocks_per_multiprocessor;
  prop->accessPolicyMaxWindowSize = 0;
  prop->reservedSharedMemPerBlock = 0;
  // HIP-on-AMDGPU supports native host/device atomics on the fine-grained
  // shared memory required during device initialization.
  prop->hostNativeAtomicSupported = 1;
  prop->memoryPoolsSupported = iree_hip_memory_pools_supported() ? 1 : 0;
  prop->hostRegisterSupported = 1;
  prop->hostRegisterReadOnlySupported = 1;
  prop->maxSharedMemoryPerMultiProcessor =
      device_obj->max_shared_memory_per_multiprocessor;
  prop->clockInstructionRate = is_gfx1100 ? 1000000 : 0;
  prop->isLargeBar = is_gfx1100 ? 1 : 0;
  if (is_gfx1100) {
    prop->arch.hasGlobalInt32Atomics = 1;
    prop->arch.hasGlobalFloatAtomicExch = 1;
    prop->arch.hasSharedInt32Atomics = 1;
    prop->arch.hasSharedFloatAtomicExch = 1;
    prop->arch.hasFloatAtomicAdd = 1;
    prop->arch.hasGlobalInt64Atomics = 1;
    prop->arch.hasSharedInt64Atomics = 1;
    prop->arch.hasDoubles = 1;
    prop->arch.hasWarpVote = 1;
    prop->arch.hasWarpBallot = 1;
    prop->arch.hasWarpShuffle = 1;
    prop->arch.hasThreadFenceSystem = 1;
    prop->arch.has3dGrid = 1;
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Alias for ROCm 6.0.0+ compatibility - exports the versioned symbol.
// Modern HIP runtimes expect this versioned symbol for ABI stability.
HIPAPI hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t* prop,
                                              int device) {
  return hipGetDeviceProperties(prop, device);
}

// Gets a specific attribute of a compute device.
//
// Parameters:
//  - value: [OUT] Pointer to receive the attribute value.
//  - attr: [IN] Attribute to query (hipDeviceAttributeMaxThreadsPerBlock,
//               hipDeviceAttributeMaxBlockDimX, hipDeviceAttributeMaxGridDimX,
//               hipDeviceAttributeComputeCapabilityMajor, etc.).
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Attribute retrieved successfully.
//  - hipErrorInvalidValue: value is NULL or attr is invalid.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Common attributes:
// - hipDeviceAttributeMaxThreadsPerBlock: Max threads per block.
// - hipDeviceAttributeMaxBlockDimX/Y/Z: Max block dimensions.
// - hipDeviceAttributeMaxGridDimX/Y/Z: Max grid dimensions.
// - hipDeviceAttributeMaxSharedMemoryPerBlock: Shared memory per block.
// - hipDeviceAttributeWarpSize: Number of threads in a warp.
// - hipDeviceAttributeComputeCapabilityMajor/Minor: Compute capability.
// - hipDeviceAttributeMultiprocessorCount: Number of SMs/CUs.
// - hipDeviceAttributeClockRate: Core clock frequency.
// - hipDeviceAttributeMemoryClockRate: Memory clock frequency.
// - hipDeviceAttributeGlobalMemoryBusWidth: Memory bus width.
// - hipDeviceAttributeL2CacheSize: L2 cache size.
// - hipDeviceAttributeComputeMode: Current compute mode.
//
// Multi-GPU: Each device has unique attributes.
//
// Note: More efficient than hipGetDeviceProperties() when only specific
// attributes are needed.
//
// See also: hipGetDeviceProperties, hipDeviceGetName, hipGetDevice.
HIPAPI hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr,
                                        int device) {
  HIP_DEBUG_LOG("[HIP_API] hipDeviceGetAttribute(attr=%d, device=%d)\n",
                (int)attr, device);
  if (!value) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Map attributes to device properties.
  const bool is_gfx1100 = strncmp(device_obj->gcn_arch_name, "gfx1100", 7) == 0;
  const bool is_gfx942 = strncmp(device_obj->gcn_arch_name, "gfx942", 6) == 0;
  switch (attr) {
    case hipDeviceAttributeMaxThreadsPerBlock:
      *value = device_obj->max_threads_per_block;
      break;
    case hipDeviceAttributeMaxBlockDimX:
      *value = device_obj->max_block_dim[0];
      break;
    case hipDeviceAttributeMaxBlockDimY:
      *value = device_obj->max_block_dim[1];
      break;
    case hipDeviceAttributeMaxBlockDimZ:
      *value = device_obj->max_block_dim[2];
      break;
    case hipDeviceAttributeMaxGridDimX:
      *value = device_obj->max_grid_dim[0];
      break;
    case hipDeviceAttributeMaxGridDimY:
      *value = device_obj->max_grid_dim[1];
      break;
    case hipDeviceAttributeMaxGridDimZ:
      *value = device_obj->max_grid_dim[2];
      break;
    case hipDeviceAttributeWarpSize:
      *value = device_obj->warp_size;
      break;
    case hipDeviceAttributeCanMapHostMemory:
      *value = 1;
      break;
    case hipDeviceAttributeCanUseHostPointerForRegisteredMem:
      *value = 1;
      break;
    case hipDeviceAttributeHostNativeAtomicSupported:
      // HIP-on-AMDGPU requires fine-grained host/device atomic shared memory.
      *value = 1;
      break;
    case hipDeviceAttributeMultiprocessorCount:
      *value = device_obj->multiprocessor_count;
      break;
    case hipDeviceAttributeComputeCapabilityMajor:
      *value = device_obj->compute_capability_major;
      break;
    case hipDeviceAttributeComputeCapabilityMinor:
      *value = device_obj->compute_capability_minor;
      break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:
      *value = device_obj->max_shared_memory_per_block;
      break;
    case hipDeviceAttributeMaxRegistersPerBlock:
      *value = device_obj->max_registers_per_block;
      break;
    case hipDeviceAttributeClockRate:
      *value = is_gfx942 ? 2100000 : (is_gfx1100 ? 1760000 : 1000000);  // kHz
      break;
    case hipDeviceAttributeMemoryClockRate:
      *value = is_gfx942 ? 1300000 : (is_gfx1100 ? 1124000 : 1000000);  // kHz
      break;
    case hipDeviceAttributeMemoryBusWidth:
      *value = is_gfx942 ? 8192 : (is_gfx1100 ? 384 : 256);
      break;
    case hipDeviceAttributeL2CacheSize:
      *value = is_gfx942 ? 4194304 : (is_gfx1100 ? 6291456 : 0);
      break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:
      *value = device_obj->max_threads_per_multiprocessor;
      break;
    case hipDeviceAttributeSharedMemPerBlockOptin:
      *value = device_obj->max_shared_memory_per_block_optin;
      break;
    case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
      *value = device_obj->max_shared_memory_per_multiprocessor;
      break;
    case hipDeviceAttributeMaxPitch:
      *value = INT_MAX;
      break;
    case hipDeviceAttributeSharedMemPerMultiprocessor:
      *value = device_obj->max_shared_memory_per_multiprocessor;
      break;
    case hipDeviceAttributeTotalGlobalMem:
      *value = device_obj->total_memory > 2147483647ull
                   ? 2147483647
                   : (int)device_obj->total_memory;
      break;
    case hipDeviceAttributeManagedMemory:
      *value = 1;
      break;
    case hipDeviceAttributeHostRegisterSupported:
      *value = 1;
      break;
    case hipDeviceAttributeMemoryPoolsSupported:
      *value = iree_hip_memory_pools_supported() ? 1 : 0;
      break;
    case hipDeviceAttributeConcurrentManagedAccess:
      *value = 1;
      break;
    case hipDeviceAttributeDirectManagedMemAccessFromHost:
      *value = 1;
      break;
    case hipDeviceAttributePhysicalMultiProcessorCount:
      // Physical multi-processor (compute unit) count.
      // Same as hipDeviceAttributeMultiprocessorCount for most devices.
      *value = device_obj->multiprocessor_count;
      break;
    case hipDeviceAttributeFineGrainSupport:
      *value = is_gfx1100 ? 0 : 1;
      break;
    case hipDeviceAttributeWallClockRate:
      *value = 100000;  // Wall clock rate in kHz
      break;
    case hipDeviceAttributeIsLargeBar:
      *value = 1;  // Assume large BAR support
      break;
    case hipDeviceAttributeCanUseStreamWaitValue:
      *value = 1;
      break;
    case hipDeviceAttributeImageSupport:
      *value = 1;
      break;
    case hipDeviceAttributeMaxBlocksPerMultiProcessor:
      *value = device_obj->max_blocks_per_multiprocessor;
      break;
    case hipDeviceAttributePersistingL2CacheMaxSize:
      *value = is_gfx942 ? 4194304 : (is_gfx1100 ? 6291456 : 0);
      break;
    case hipDeviceAttributeNumberOfXccs:
      *value = is_gfx942 ? 8 : (is_gfx1100 ? 1 : 0);
      break;
    case hipDeviceAttributeMaxTexture1DWidth:
    case hipDeviceAttributeMaxTexture1DLinear:
    case hipDeviceAttributeMaxTexture1DMipmap:
    case hipDeviceAttributeMaxSurface1D:
      *value = IREE_HIP_ARRAY_MAX_1D;
      break;
    case hipDeviceAttributeMaxTexture2DWidth:
    case hipDeviceAttributeMaxTexture2DLinear:
    case hipDeviceAttributeMaxTexture2DMipmap:
    case hipDeviceAttributeMaxTexture2DGather:
    case hipDeviceAttributeMaxTextureCubemap:
    case hipDeviceAttributeMaxSurface2D:
    case hipDeviceAttributeMaxSurfaceCubemap:
      *value = IREE_HIP_ARRAY_MAX_2D_WIDTH;
      break;
    case hipDeviceAttributeMaxTexture2DHeight:
      *value = IREE_HIP_ARRAY_MAX_2D_HEIGHT;
      break;
    case hipDeviceAttributeMaxTexture3DWidth:
    case hipDeviceAttributeMaxTexture3DAlt:
    case hipDeviceAttributeMaxSurface3D:
      *value = IREE_HIP_ARRAY_MAX_3D_WIDTH;
      break;
    case hipDeviceAttributeMaxTexture3DHeight:
      *value = IREE_HIP_ARRAY_MAX_3D_HEIGHT;
      break;
    case hipDeviceAttributeMaxTexture3DDepth:
      *value = IREE_HIP_ARRAY_MAX_3D_DEPTH;
      break;
    default:
      // Return sensible defaults for other attributes.
      *value = 0;
      break;
  }

  return hipSuccess;
}

// Gets the name of a compute device.
//
// Parameters:
//  - name: [OUT] Buffer to receive the device name.
//  - len: [IN] Maximum length of name buffer including null terminator.
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Name retrieved successfully.
//  - hipErrorInvalidValue: name is NULL or len <= 0.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Name behavior:
// - Returns human-readable device name.
// - String is null-terminated.
// - Truncated if longer than len-1 characters.
// - Typically includes manufacturer and model.
//
// Multi-GPU: Each device has a unique name.
//
// Usage pattern:
// ```c
// char deviceName[256];
// hipDeviceGetName(deviceName, sizeof(deviceName), 0);
// printf("Device 0: %s\n", deviceName);
// ```
//
// See also: hipGetDeviceProperties, hipDeviceGetAttribute, hipGetDevice.
HIPAPI hipError_t hipDeviceGetName(char* name, int len, int device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!name || len <= 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_device_name(
      (iree_hal_streaming_device_ordinal_t)device, name, (size_t)len);
  if (iree_status_is_ok(status)) {
    iree_hip_sanitize_device_name(name);
  }
  hipError_t result = iree_status_to_hip_result(status);

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Gets the UUID of a compute device.
//
// Parameters:
//  - uuid: [OUT] Pointer to receive the device UUID.
//  - dev: [IN] Device handle to query.
//
// Returns:
//  - hipSuccess: UUID retrieved successfully.
//  - hipErrorInvalidValue: uuid is NULL.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorNotSupported: UUID not supported (current implementation).
//
// Synchronization: This operation is synchronous.
//
// UUID behavior:
// - Returns a unique identifier for the physical device.
// - UUID persists across reboots and driver reloads.
// - Can be used to identify specific GPUs in multi-GPU systems.
// - Format is 16-byte binary value.
//
// Multi-GPU: Each physical device has a unique UUID.
//
// Note: Currently not implemented in StreamHAL.
//
// See also: hipDeviceGet, hipGetDeviceProperties.
HIPAPI hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t dev) {
  // UUID support is not currently implemented.
  (void)uuid;
  (void)dev;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Gets the total memory of a compute device.
//
// Parameters:
//  - bytes: [OUT] Pointer to receive total memory in bytes.
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Total memory retrieved successfully.
//  - hipErrorInvalidValue: bytes is NULL.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory information:
// - Returns total global memory available on the device.
// - Does not include shared memory or constant memory.
// - Value is fixed for the device (doesn't change with allocations).
//
// Multi-GPU: Each device has its own memory capacity.
//
// Usage pattern:
// ```c
// size_t totalMem;
// hipDeviceTotalMem(&totalMem, 0);
// printf("Device 0 has %zu GB of memory\n", totalMem / (1024*1024*1024));
// ```
//
// See also: hipMemGetInfo, hipGetDeviceProperties.
HIPAPI hipError_t hipDeviceTotalMem(size_t* bytes, int device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!bytes) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *bytes = 0;

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  uint64_t total_memory = 0;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      HRX_CALL(hrx_device_get_property(iree_hip_hrx_device(device),
                                       HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
                                       &total_memory, sizeof(total_memory))),
      hipErrorInvalidDevice);

  *bytes = (size_t)total_memory;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Queries if one device can directly access another device's memory.
//
// Parameters:
//  - canAccessPeer: [OUT] Set to 1 if access is possible, 0 otherwise.
//  - device: [IN] Device that would be accessing memory.
//  - peerDevice: [IN] Device whose memory would be accessed.
//
// Returns:
//  - hipSuccess: Query completed successfully.
//  - hipErrorInvalidValue: canAccessPeer is NULL.
//  - hipErrorInvalidDevice: Invalid device or peerDevice.
//
// Synchronization: This operation is synchronous.
//
// Peer access behavior:
// - Returns 1 if device can directly access peerDevice's memory.
// - Returns 0 if direct access is not possible.
// - Does not enable peer access, only queries capability.
// - Symmetric access is not guaranteed (A→B doesn't imply B→A).
//
// Multi-GPU:
// - Typically supported between GPUs on the same PCIe root complex.
// - May be supported across PCIe switches with P2P capability.
// - Not supported between discrete and integrated GPUs.
//
// See also: hipDeviceEnablePeerAccess, hipDeviceDisablePeerAccess,
//           hipMemcpyPeer.
HIPAPI hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device,
                                         int peerDevice) {
  if (!canAccessPeer) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *canAccessPeer = 0;

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry || device < 0 ||
      device >= (int)device_registry->device_count || peerDevice < 0 ||
      peerDevice >= (int)device_registry->device_count) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }
  if (device == peerDevice) {
    // A device can access its own allocations, but that is not peer access.
    return hipSuccess;
  }

  // Direct peer access is not advertised until backend topology reports it.
  // Copy APIs may still stage transfers through host memory.
  return hipSuccess;
}

// Gets peer-to-peer attributes between two devices.
//
// Parameters:
//  - value: [OUT] Pointer to receive the attribute value.
//  - attrib: [IN] P2P attribute to query (hipDevP2PAttrPerformanceRank,
//                 hipDevP2PAttrAccessSupported,
//                 hipDevP2PAttrNativeAtomicSupported,
//                 hipDevP2PAttrCudaArrayAccessSupported).
//  - srcDevice: [IN] Source device in P2P pair.
//  - dstDevice: [IN] Destination device in P2P pair.
//
// Returns:
//  - hipSuccess: Attribute retrieved successfully.
//  - hipErrorInvalidValue: value is NULL or invalid attribute.
//  - hipErrorInvalidDevice: Invalid device ordinals.
//
// Synchronization: This operation is synchronous.
//
// P2P attributes:
// - hipDevP2PAttrPerformanceRank: Relative performance (higher is better).
// - hipDevP2PAttrAccessSupported: 1 if P2P access is supported.
// - hipDevP2PAttrNativeAtomicSupported: 1 if atomic operations supported.
// - hipDevP2PAttrCudaArrayAccessSupported: 1 if array access supported.
//
// Multi-GPU:
// - Queries capabilities of direct GPU-to-GPU communication.
// - Asymmetric: srcDevice→dstDevice may differ from dstDevice→srcDevice.
// - Performance varies based on PCIe topology.
//
// See also: hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess.
HIPAPI hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attrib,
                                           int srcDevice, int dstDevice) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up P2P link.
  iree_hal_streaming_p2p_link_t* link =
      iree_hal_streaming_device_lookup_p2p_link(srcDevice, dstDevice);
  if (!link) {
    *value = 0;
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Map HIP P2P attribute enum to the appropriate link field.
  switch (attrib) {
    case hipDevP2PAttrAccessSupported:
      *value = link->access_supported ? 1 : 0;
      break;
    case hipDevP2PAttrNativeAtomicSupported:
      *value = link->native_atomic_supported ? 1 : 0;
      break;
    case hipDevP2PAttrHipArrayAccessSupported:
      *value = link->cuda_array_access_supported ? 1 : 0;
      break;
    case hipDevP2PAttrPerformanceRank:
      *value = link->performance_rank;
      break;
    default:
      // Unsupported attribute.
      *value = 0;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Enables direct memory access from current device to peer device.
//
// Parameters:
//  - peerDevice: [IN] Peer device to enable access to.
//  - flags: [IN] Reserved for future use (must be 0).
//
// Returns:
//  - hipSuccess: Peer access enabled successfully.
//  - hipErrorInvalidDevice: Invalid peer device.
//  - hipErrorInvalidValue: Invalid flags.
//  - hipErrorPeerAccessAlreadyEnabled: Access already enabled.
//  - hipErrorPeerAccessUnsupported: Devices cannot access each other.
//
// Synchronization: This operation is synchronous.
//
// Peer access behavior:
// - Enables current device to access peerDevice's memory.
// - Access is unidirectional (must enable separately for bidirectional).
// - Remains enabled until explicitly disabled or context destroyed.
// - Allows direct memory copies without staging through host.
//
// Multi-GPU:
// - Improves performance for device-to-device transfers.
// - Enables single-copy transfers instead of staged copies.
// - May increase memory bandwidth utilization.
//
// Warning: Check hipDeviceCanAccessPeer() before enabling.
//
// See also: hipDeviceDisablePeerAccess, hipDeviceCanAccessPeer,
//           hipMemcpyPeer.
HIPAPI hipError_t hipDeviceEnablePeerAccess(int peerDevice,
                                            unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get the current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the peer device's default context.
  iree_hal_streaming_device_t* peer_device =
      iree_hal_streaming_device_entry(peerDevice);
  if (!peer_device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get or create the peer device's primary context.
  iree_hal_streaming_context_t* peer_primary_context = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_device_get_or_create_primary_context(
          peer_device, &peer_primary_context),
      hipErrorOutOfMemory);

  // Enable peer access between the current context and the peer device's
  // context.
  // TODO(benvanik): check for hipErrorPeerAccessAlreadyEnabled.
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_context_enable_peer_access(context,
                                                    peer_primary_context),
      hipErrorPeerAccessUnsupported);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Disables direct memory access from current device to peer device.
//
// Parameters:
//  - peerDevice: [IN] Peer device to disable access to.
//
// Returns:
//  - hipSuccess: Peer access disabled successfully.
//  - hipErrorInvalidDevice: Invalid peer device.
//  - hipErrorPeerAccessNotEnabled: Peer access was not enabled.
//
// Synchronization: This operation is synchronous. Waits for all peer
// operations to complete.
//
// Peer access behavior:
// - Disables current device's access to peerDevice's memory.
// - Only affects current device's access (unidirectional).
// - Subsequent peer operations will fail or fall back to staged copies.
// - All ongoing peer operations complete before disabling.
//
// Multi-GPU:
// - Returns to staged copy behavior through host memory.
// - May reduce memory bandwidth utilization.
//
// Warning: Ensure no kernels are actively using peer memory.
//
// See also: hipDeviceEnablePeerAccess, hipDeviceCanAccessPeer.
HIPAPI hipError_t hipDeviceDisablePeerAccess(int peerDevice) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get the current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the peer device's default context.
  iree_hal_streaming_device_t* peer_device =
      iree_hal_streaming_device_entry(peerDevice);
  if (!peer_device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get or create the peer device's primary context.
  iree_hal_streaming_context_t* peer_primary_context = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_device_get_or_create_primary_context(
          peer_device, &peer_primary_context),
      hipErrorOutOfMemory);

  // Disable peer access between the current context and the peer device's
  // context.
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_context_disable_peer_access(context,
                                                     peer_primary_context),
      hipErrorPeerAccessNotEnabled);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the PCI bus ID string for a device.
// Returns a placeholder string since we don't have real PCI info.
HIPAPI hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  if (!pciBusId || len <= 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Validate device ordinal.
  int device_count = 0;
  hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) {
    return count_result;
  }
  if (device < 0 || device >= device_count) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Return a placeholder PCI bus ID.
  // Format: domain:bus:device.function (e.g., "0000:00:00.0")
  int written = snprintf(pciBusId, len, "0000:00:0%d.0", device);
  if (written < 0 || written >= len) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  return hipSuccess;
}

// Gets the device ordinal for a PCI bus ID string.
// We return device 0 for any valid-looking bus ID.
HIPAPI hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  if (!device || !pciBusId) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // For simplicity, just return device 0.
  // A proper implementation would parse the bus ID and match it.
  *device = 0;
  return hipSuccess;
}

// Gets the range of stream priorities supported by the device.
// Lower values have higher priority (with 0 being the default).
HIPAPI hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority,
                                                  int* greatestPriority) {
  // Return a simple priority range (0 = default, -1 = high priority).
  // On most AMD GPUs, stream priorities don't have significant effect.
  if (leastPriority) {
    *leastPriority = 0;  // Lowest priority (default)
  }
  if (greatestPriority) {
    *greatestPriority = -1;  // Highest priority
  }
  return hipSuccess;
}

static hipError_t iree_hip_graph_memory_device(
    int device, iree_hal_streaming_device_t** out_device) {
  if (out_device) *out_device = NULL;
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) return init_result;
  iree_hal_streaming_device_t* device_entry = iree_hal_streaming_device_entry(
      (iree_hal_streaming_device_ordinal_t)device);
  if (!device_entry) return hipErrorInvalidDevice;
  if (out_device) *out_device = device_entry;
  return hipSuccess;
}

HIPAPI hipError_t hipDeviceGetGraphMemAttribute(int device, int attr,
                                                void* value) {
  iree_hal_streaming_device_t* device_entry = NULL;
  hipError_t device_result =
      iree_hip_graph_memory_device(device, &device_entry);
  if (device_result != hipSuccess) {
    HIP_RETURN_ERROR(device_result);
  }
  if (!value) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  switch ((hipGraphMemAttributeType)attr) {
    case hipGraphMemAttrUsedMemCurrent:
      *(uint64_t*)value =
          iree_hal_streaming_graph_memory_used_current(device_entry);
      break;
    case hipGraphMemAttrUsedMemHigh:
      *(uint64_t*)value =
          iree_hal_streaming_graph_memory_used_high(device_entry);
      break;
    case hipGraphMemAttrReservedMemCurrent:
      *(uint64_t*)value =
          iree_hal_streaming_graph_memory_reserved_current(device_entry);
      break;
    case hipGraphMemAttrReservedMemHigh:
      *(uint64_t*)value =
          iree_hal_streaming_graph_memory_reserved_high(device_entry);
      break;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipDeviceSetGraphMemAttribute(int device, int attr,
                                                void* value) {
  iree_hal_streaming_device_t* device_entry = NULL;
  hipError_t device_result =
      iree_hip_graph_memory_device(device, &device_entry);
  if (device_result != hipSuccess) {
    HIP_RETURN_ERROR(device_result);
  }
  if (!value) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  switch ((hipGraphMemAttributeType)attr) {
    case hipGraphMemAttrUsedMemHigh:
      if (*(const uint64_t*)value != 0) {
        HIP_RETURN_ERROR(hipErrorInvalidValue);
      }
      iree_hal_streaming_graph_memory_reset_used_high(device_entry);
      break;
    case hipGraphMemAttrReservedMemHigh:
      if (*(const uint64_t*)value != 0) {
        HIP_RETURN_ERROR(hipErrorInvalidValue);
      }
      iree_hal_streaming_graph_memory_reset_reserved_high(device_entry);
      break;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipDeviceGraphMemTrim(int device) {
  iree_hal_streaming_device_t* device_entry = NULL;
  hipError_t device_result =
      iree_hip_graph_memory_device(device, &device_entry);
  if (device_result != hipSuccess) {
    HIP_RETURN_ERROR(device_result);
  }
  iree_hal_streaming_graph_memory_trim(device_entry);
  return hipSuccess;
}

// Waits for all operations on the current device to complete.
//
// Parameters: None.
//
// Returns:
//  - hipSuccess: All operations completed successfully.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: A kernel launch failed.
//  - hipErrorIllegalAddress: Invalid memory access occurred.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation blocks the host thread until all
// previously enqueued operations on the current device have completed.
//
// Device behavior:
// - Synchronizes all streams on the current device.
// - Waits for all kernels, memory copies, and events.
// - More heavyweight than hipStreamSynchronize().
// - Does not synchronize with other devices.
//
// Multi-GPU: Only synchronizes the current device set by hipSetDevice().
//
// Performance note: Use stream-specific synchronization when possible for
// better performance.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
HIPAPI hipError_t hipDeviceSynchronize(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG("[HIP_API] hipDeviceSynchronize() called\n");
  // Ensure initialization and get context.
  static int sync_count = 0;
  sync_count++;
  (void)sync_count;
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_all_active_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_status_t status = iree_hal_streaming_context_synchronize(context);
  hipError_t result = iree_status_to_hip_result(status);
  HIP_DEBUG_LOG(
      "[HIP_API] hipDeviceSynchronize() returned %d (sync_count=%d)\n", result,
      sync_count);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Resets the current device and destroys all allocations.
//
// Parameters: None.
//
// Returns:
//  - hipSuccess: Device reset successfully.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Waits for all operations
// to complete before resetting.
//
// Reset behavior:
// - Destroys all allocations on the current device.
// - Destroys all streams, events, and modules.
// - Resets all device state to initial values.
// - Primary context is destroyed and recreated.
// - Subsequent API calls reinitialize the device.
//
// Multi-GPU: Only resets the current device set by hipSetDevice().
//
// Warning: This is a heavyweight operation that affects all contexts on the
// device. All pointers and handles become invalid. Use with caution in
// multi-threaded applications.
//
// Usage pattern:
// ```c
// hipSetDevice(0);
// // ... work with device ...
// hipDeviceReset();  // Clean slate for device 0
// ```
//
// See also: hipSetDevice, hipDeviceSynchronize, hipCtxDestroy.
HIPAPI hipError_t hipDeviceReset(void) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get current context to determine which device to reset.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Reset the primary context for the current device.
  hipDevice_t current_device = context->device_ordinal;
  iree_hip_clear_per_thread_stream(context);
  hipError_t result = hipDevicePrimaryCtxReset(current_device);

  IREE_TRACE_ZONE_END(z0);
  iree_hip_thread_error_set(result, false);
  return result;
}

//===----------------------------------------------------------------------===//
// Device flags
//===----------------------------------------------------------------------===//

// Sets flags for the current device.
//
// Parameters:
//  - flags: [IN] Flags to set for the device. See hipDeviceSchedule* flags.
//
// Returns:
//  - hipSuccess: Flags set successfully.
//  - hipErrorSetOnActiveProcess: Device already has active contexts.
//
// Notes:
// - Must be called before any other HIP call that activates the device.
// - Common flags:
//   - hipDeviceScheduleAuto: Let driver choose scheduling.
//   - hipDeviceScheduleSpin: Busy-wait for synchronization.
//   - hipDeviceScheduleYield: Yield CPU during synchronization.
//   - hipDeviceScheduleBlockingSync: Block thread during synchronization.
//   - hipDeviceLmemResizeToMax: Allocate max local memory for all kernels.
//   - hipDeviceMapHost: Allow mapping of host memory.
//
// See also: hipGetDeviceFlags, hipSetDevice.
HIPAPI hipError_t hipSetDeviceFlags(unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure initialization.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get current context to check if device is already active.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (context != NULL) {
    // Device already has an active context - can't change flags.
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorSetOnActiveProcess);
  }

  // For now, we accept the flags but don't enforce them.
  // The streaming backend uses its own scheduling model.
  (void)flags;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the flags for the current device.
//
// Parameters:
//  - flags: [OUT] Pointer to receive the current device flags.
//
// Returns:
//  - hipSuccess: Flags retrieved successfully.
//  - hipErrorInvalidValue: flags is NULL.
//
// Notes:
// - Returns the flags set by hipSetDeviceFlags() or defaults.
//
// See also: hipSetDeviceFlags, hipGetDevice.
HIPAPI hipError_t hipGetDeviceFlags(unsigned int* flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flags) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Return default flags (auto scheduling).
  // The streaming backend doesn't currently track user-set flags.
  *flags = 0;  // hipDeviceScheduleAuto

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Cache configuration (no-ops on AMD devices)
//===----------------------------------------------------------------------===//

// Sets the preferred cache configuration for the current device.
// Note: These hints are ignored on AMD devices per the HIP documentation.
HIPAPI hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  (void)cacheConfig;
  return hipSuccess;  // No-op on AMD
}

// Gets the current cache configuration for the current device.
HIPAPI hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig) {
  if (!cacheConfig) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *cacheConfig = hipFuncCachePreferNone;
  return hipSuccess;
}

// Sets the shared memory configuration for the current device.
HIPAPI hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
  (void)config;
  return hipSuccess;  // No-op on AMD
}

// Gets the shared memory configuration for the current device.
HIPAPI hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* config) {
  if (!config) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *config = hipSharedMemBankSizeDefault;
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Primary context
//===----------------------------------------------------------------------===//

// Retains the primary context for a device.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the primary context handle.
//  - dev: [IN] Device handle.
//
// Returns:
//  - hipSuccess: Primary context retained successfully.
//  - hipErrorInvalidValue: pctx is NULL.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorOutOfMemory: Insufficient memory to create context.
//
// Synchronization: This operation is synchronous.
//
// Primary context behavior:
// - Creates primary context if it doesn't exist.
// - Increments reference count if it already exists.
// - Primary context is shared by all threads.
// - More lightweight than hipCtxCreate contexts.
// - Automatically created when needed by runtime API.
//
// Multi-GPU: Each device has its own primary context.
//
// Warning: Must balance with hipDevicePrimaryCtxRelease().
//
// See also: hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxSetFlags,
//           hipCtxCreate.
HIPAPI hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP runtime is initialized (but don't create context yet).
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Retain the primary context, creating it if necessary.
  iree_hal_streaming_context_t* primary_context = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_device_retain_primary_context(device,
                                                       &primary_context),
      hipErrorOutOfMemory);

  *pctx = (hipCtx_t)primary_context;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Releases the primary context for a device.
//
// Parameters:
//  - dev: [IN] Device handle.
//
// Returns:
//  - hipSuccess: Primary context released successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//
// Synchronization: This operation is synchronous.
//
// Primary context behavior:
// - Decrements reference count.
// - Destroys context when reference count reaches zero.
// - All resources in the context are released.
// - Subsequent API calls may recreate the context.
//
// Multi-GPU: Only affects the specified device's primary context.
//
// Warning: Must balance with hipDevicePrimaryCtxRetain().
//
// See also: hipDevicePrimaryCtxRetain, hipDevicePrimaryCtxReset.
HIPAPI hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Release the primary context (destroys when ref count reaches 0).
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_device_release_primary_context(device),
      hipErrorInvalidContext);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets flags for the primary context.
//
// Parameters:
//  - dev: [IN] Device handle.
//  - flags: [IN] Context flags (hipCtxSchedAuto, hipCtxSchedSpin,
//                hipCtxSchedYield, hipCtxSchedBlockingSync, etc.).
//
// Returns:
//  - hipSuccess: Flags set successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorInvalidValue: Invalid flags.
//  - hipErrorContextAlreadyInUse: Primary context already active.
//
// Synchronization: This operation is synchronous.
//
// Flag behavior:
// - Must be called before primary context is created.
// - Cannot change flags after context is active.
// - Affects scheduling behavior and resource allocation.
//
// Scheduling flags:
// - hipCtxSchedAuto: Automatic scheduling.
// - hipCtxSchedSpin: Spin-wait (low latency, high CPU).
// - hipCtxSchedYield: Yield CPU (higher latency, low CPU).
// - hipCtxSchedBlockingSync: Block on synchronization.
//
// Multi-GPU: Each device's primary context has independent flags.
//
// See also: hipDevicePrimaryCtxGetState, hipDevicePrimaryCtxRetain.
HIPAPI hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev,
                                              unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Convert HIP flags to strongly-typed internal flags.
  iree_hal_streaming_context_flags_t internal_flags = {0};

  // Extract scheduling mode from lower bits.
  unsigned int sched_flags = flags & 0x07;
  switch (sched_flags) {
    case hipDeviceScheduleAuto:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
      break;
    case hipDeviceScheduleSpin:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN;
      break;
    case hipDeviceScheduleYield:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD;
      break;
    case hipDeviceScheduleBlockingSync:
      internal_flags.scheduling_mode =
          IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC;
      break;
    default:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
      break;
  }

  // Extract other flags.
  internal_flags.map_host_memory = (flags & hipDeviceMapHost) != 0;
  internal_flags.resize_local_mem_to_max =
      (flags & hipDeviceLmemResizeToMax) != 0;

  // Set the primary context flags.
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_device_set_primary_context_flags(dev, &internal_flags),
      hipErrorInvalidValue);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the state of the primary context.
//
// Parameters:
//  - dev: [IN] Device handle.
//  - flags: [OUT] Pointer to receive context flags (can be NULL).
//  - active: [OUT] Pointer to receive active state (can be NULL).
//
// Returns:
//  - hipSuccess: State retrieved successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//
// Synchronization: This operation is synchronous.
//
// State information:
// - flags: Current context creation flags.
// - active: 1 if context is active, 0 if inactive.
// - Context is active while the device owns a live primary context.
//
// Multi-GPU: Queries state of specified device's primary context.
//
// Usage pattern:
// ```c
// unsigned int flags;
// int active;
// hipDevicePrimaryCtxGetState(device, &flags, &active);
// if (active) {
//   printf("Primary context is active\n");
// }
// ```
//
// See also: hipDevicePrimaryCtxSetFlags, hipDevicePrimaryCtxRetain.
HIPAPI hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev,
                                              unsigned int* flags,
                                              int* active) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get the primary context state.
  iree_hal_streaming_context_flags_t internal_flags = {0};
  bool is_active = false;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_device_primary_context_state(dev, &internal_flags,
                                                      &is_active),
      hipErrorInvalidDevice);

  if (flags) {
    // Convert internal flags back to HIP flags.
    unsigned int hip_flags = 0;

    // Set scheduling mode.
    switch (internal_flags.scheduling_mode) {
      case IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO:
        hip_flags |= hipDeviceScheduleAuto;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN:
        hip_flags |= hipDeviceScheduleSpin;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD:
        hip_flags |= hipDeviceScheduleYield;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC:
        hip_flags |= hipDeviceScheduleBlockingSync;
        break;
    }

    // Set other flags.
    if (internal_flags.map_host_memory) {
      hip_flags |= hipDeviceMapHost;
    }
    if (internal_flags.resize_local_mem_to_max) {
      hip_flags |= hipDeviceLmemResizeToMax;
    }

    *flags = hip_flags;
  }

  if (active) {
    *active = is_active ? 1 : 0;
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Resets the primary context for a device.
//
// Parameters:
//  - dev: [IN] Device handle.
//
// Returns:
//  - hipSuccess: Primary context reset successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//
// Synchronization: This operation is synchronous. Waits for all operations
// to complete.
//
// Reset behavior:
// - Destroys the primary context regardless of reference count.
// - All resources in the context are released.
// - All allocations are freed.
// - All streams and events are destroyed.
// - Context will be recreated on next use.
//
// Multi-GPU: Only affects the specified device's primary context.
//
// Warning: This is a heavyweight operation that affects all users of the
// primary context. Use with caution in multi-threaded applications.
//
// See also: hipDeviceReset, hipDevicePrimaryCtxRelease, hipCtxDestroy.
HIPAPI hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Validate device.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (dev < 0 || !device_registry ||
      dev >= (int)device_registry->device_count) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Reset the primary context by:
  // 1. Waiting for all operations to complete (if it exists)
  // 2. Releasing the current context
  // 3. The context will be recreated lazily on next access

  if (device->primary_context) {
    // Wait for all operations on the context to complete.
    iree_status_t status = iree_hal_streaming_context_wait_idle(
        device->primary_context, iree_infinite_timeout());
    if (!iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(
          iree_status_to_fixed_hip_result(status, hipErrorUnknown));
    }

    // Clear current context if it was the primary context (before release).
    iree_hal_streaming_context_t* current_context =
        iree_hal_streaming_context_current();
    if (current_context == device->primary_context) {
      iree_hal_streaming_context_set_current(NULL);
    }

    // All allocations are released with the context — reset free memory.
    iree_atomic_store(&device->free_memory, device->total_memory,
                      iree_memory_order_relaxed);

    // Lock to ensure thread safety during reset.
    iree_slim_mutex_lock(&device->primary_context_mutex);

    // Release the old context.
    iree_hal_streaming_context_release(device->primary_context);
    device->primary_context = NULL;

    // Reset reference count to 0.
    device->primary_context_ref_count = 0;

    // Also clear memory pools.
    hrx_mem_pool_release(device->current_mem_pool);
    device->current_mem_pool = NULL;
    hrx_mem_pool_release(device->default_mem_pool);
    device->default_mem_pool = NULL;

    iree_slim_mutex_unlock(&device->primary_context_mutex);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Helper function to convert HIP context flags to internal flags.
static iree_hal_streaming_context_flags_t
iree_hal_streaming_hip_context_flags_to_internal(unsigned int hip_flags) {
  iree_hal_streaming_context_flags_t flags = {0};

  // Convert scheduling flags.
  int sched_flags = hip_flags & hipDeviceScheduleMask;
  switch (sched_flags) {
    case hipDeviceScheduleSpin:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN;
      break;
    case hipDeviceScheduleYield:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD;
      break;
    case hipDeviceScheduleBlockingSync:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC;
      break;
    default:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
      break;
  }

  // Convert other flags.
  if (hip_flags & hipDeviceMapHost) {
    flags.map_host_memory = true;
  }
  if (hip_flags & hipDeviceLmemResizeToMax) {
    flags.resize_local_mem_to_max = true;
  }

  return flags;
}

// Creates a new HIP context for a device.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the created context handle.
//  - flags: [IN] Context creation flags (hipCtxSchedAuto, hipCtxSchedSpin,
//                hipCtxSchedYield, hipCtxSchedBlockingSync, hipCtxMapHost,
//                hipCtxLmemResizeToMax).
//  - dev: [IN] Device handle for which to create the context.
//
// Returns:
//  - hipSuccess: Context created successfully.
//  - hipErrorInvalidValue: pctx is NULL or invalid flags.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorOutOfMemory: Insufficient memory to create context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during context creation.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Creates a new context and makes it current for this thread.
// - Each context maintains its own address space and resources.
// - Multiple contexts can exist per device.
// - Context must be destroyed with hipCtxDestroy().
//
// Scheduling flags:
// - hipCtxSchedAuto: Automatic scheduling (default).
// - hipCtxSchedSpin: Spin-wait for synchronization (low latency).
// - hipCtxSchedYield: Yield CPU for synchronization (low CPU usage).
// - hipCtxSchedBlockingSync: Block on synchronization.
//
// Multi-GPU: Each device can have multiple contexts.
//
// Warning: Creating multiple contexts per device may impact performance.
//
// See also: hipCtxDestroy, hipCtxPushCurrent, hipCtxPopCurrent,
//           hipCtxSetCurrent.
HIPAPI hipError_t hipCtxCreate(hipCtx_t* pctx, unsigned int flags,
                               hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Create a new context for the device.
  iree_hal_streaming_context_t* context = NULL;
  // Get the host allocator from the device registry.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorNotInitialized);
  }

  iree_status_t status = iree_hal_streaming_context_create(
      device, iree_hal_streaming_hip_context_flags_to_internal(flags),
      device_registry->host_allocator, &context);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_device_ensure_default_mem_pool(device);
  }

  if (iree_status_is_ok(status)) {
    *pctx = (hipCtx_t)context;
    // Make it current.
    iree_hal_streaming_context_set_current(context);
  } else {
    iree_hal_streaming_context_release(context);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Destroys a HIP context.
//
// Parameters:
//  - ctx: [IN] Context handle to destroy.
//
// Returns:
//  - hipSuccess: Context destroyed successfully.
//  - hipErrorInvalidValue: ctx is NULL.
//  - hipErrorInvalidContext: Invalid context handle.
//  - hipErrorContextIsDestroyed: Context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for all operations
// in the context to complete before destroying.
//
// Context behavior:
// - All resources associated with the context are released.
// - If context is current, it is popped from the stack.
// - All memory allocations in the context are freed.
// - All streams and events in the context are destroyed.
// - Using a destroyed context results in undefined behavior.
//
// Multi-GPU: Only affects the specified context on its device.
//
// Warning: Ensure all operations using this context have completed.
// Destroying a context with active operations may cause errors.
//
// See also: hipCtxCreate, hipCtxPushCurrent, hipCtxPopCurrent.
HIPAPI hipError_t hipCtxDestroy(hipCtx_t ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Check if this is the current context.
  // If so, clear it from TLS to avoid a dangling reference.
  if (ctx == (hipCtx_t)iree_hal_streaming_context_current()) {
    // This will release the TLS reference.
    iree_hal_streaming_context_set_current(NULL);
  }

  // Release the context.
  iree_hal_streaming_context_release((iree_hal_streaming_context_t*)ctx);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Pushes a context onto the current thread's context stack.
//
// Parameters:
//  - ctx: [IN] Context to push (becomes current context).
//
// Returns:
//  - hipSuccess: Context pushed successfully.
//  - hipErrorInvalidValue: ctx is NULL.
//  - hipErrorInvalidContext: Invalid context handle.
//  - hipErrorContextIsDestroyed: Context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context stack behavior:
// - Pushes context onto the thread's context stack.
// - The pushed context becomes the current context.
// - Previous current context remains on the stack.
// - Must be balanced with hipCtxPopCurrent().
// - Stack depth is implementation-defined.
//
// Multi-GPU: Can push contexts from different devices onto the same stack.
//
// Usage pattern:
// ```c
// hipCtxPushCurrent(ctx1);  // ctx1 is now current
// // Operations use ctx1
// hipCtxPopCurrent(&old);    // Restore previous context
// ```
//
// Warning: Unbalanced push/pop operations can cause resource leaks.
//
// See also: hipCtxPopCurrent, hipCtxCreate, hipCtxSetCurrent,
//           hipCtxGetCurrent.
HIPAPI hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_status_t status =
      iree_hal_streaming_context_push((iree_hal_streaming_context_t*)ctx);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Pops a context from the current thread's context stack.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the popped context (can be NULL).
//
// Returns:
//  - hipSuccess: Context popped successfully.
//  - hipErrorInvalidContext: Context stack is empty.
//  - hipErrorContextIsDestroyed: Current context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context stack behavior:
// - Removes the current context from the stack.
// - Previous context on the stack becomes current.
// - If pctx is not NULL, receives the popped context.
// - If stack becomes empty, no context is current.
// - Must balance with hipCtxPushCurrent() calls.
//
// Multi-GPU: Restores the previous context which may be on a different
// device.
//
// Usage pattern:
// ```c
// hipCtx_t old;
// hipCtxPushCurrent(newCtx);
// // Work with newCtx
// hipCtxPopCurrent(&old);  // old == newCtx
// ```
//
// Warning: Popping from an empty stack is an error.
//
// See also: hipCtxPushCurrent, hipCtxGetCurrent, hipCtxSetCurrent.
HIPAPI hipError_t hipCtxPopCurrent(hipCtx_t* pctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = NULL;
  iree_status_t status = iree_hal_streaming_context_pop(&context);
  if (iree_status_is_ok(status) && pctx) {
    *pctx = (hipCtx_t)context;
  }
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets the current context for the calling thread.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the current context handle.
//
// Returns:
//  - hipSuccess: Current context retrieved successfully.
//  - hipErrorInvalidValue: pctx is NULL.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Returns the context at the top of the thread's context stack.
// - Returns NULL if no context is current.
// - Does not modify the context stack.
// - Returned context remains current.
//
// Multi-GPU: Returns the current context regardless of which device it
// belongs to.
//
// Usage pattern:
// ```c
// hipCtx_t current;
// hipCtxGetCurrent(&current);
// if (current) {
//   // Context is active
// }
// ```
//
// See also: hipCtxSetCurrent, hipCtxPushCurrent, hipGetDevice.
HIPAPI hipError_t hipCtxGetCurrent(hipCtx_t* pctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Do NOT implicitly initialize here - just return what's current.
  // hipCtxGetCurrent should return NULL if no context is set.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  *pctx = (hipCtx_t)context;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets the current context for the calling thread.
//
// Parameters:
//  - ctx: [IN] Context to make current (NULL to clear current context).
//
// Returns:
//  - hipSuccess: Context set successfully.
//  - hipErrorInvalidContext: Invalid context handle.
//  - hipErrorContextIsDestroyed: Context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Replaces the current context without modifying the stack.
// - If ctx is NULL, clears the current context.
// - Does not push or pop the context stack.
// - Previous current context is not retained.
//
// Multi-GPU: Can set a context from any device as current.
//
// Warning: This can leave the context stack in an inconsistent state if
// used with push/pop operations. Prefer push/pop for nested context
// management.
//
// See also: hipCtxGetCurrent, hipCtxPushCurrent, hipCtxPopCurrent.
HIPAPI hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_set_current((iree_hal_streaming_context_t*)ctx);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the device associated with the current context.
//
// Parameters:
//  - device: [OUT] Pointer to receive the device handle.
//
// Returns:
//  - hipSuccess: Device retrieved successfully.
//  - hipErrorInvalidValue: device is NULL.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorContextIsDestroyed: Current context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Returns the device that owns the current context.
// - Works with both primary and created contexts.
// - Each context is associated with exactly one device.
//
// Multi-GPU: Returns the device for the current context, which may be
// different from the current device set by hipSetDevice().
//
// See also: hipCtxGetCurrent, hipGetDevice, hipCtxCreate.
HIPAPI hipError_t hipCtxGetDevice(hipDevice_t* device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  *device = (hipDevice_t)context->device_ordinal;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Synchronizes all operations in the current context.
//
// Parameters: None.
//
// Returns:
//  - hipSuccess: All operations completed successfully.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorLaunchFailure: A kernel launch failed.
//  - hipErrorIllegalAddress: Invalid memory access occurred.
//
// Synchronization: This operation blocks until all previously enqueued
// operations in the current context have completed.
//
// Context behavior:
// - Waits for all streams in the current context.
// - Waits for all kernels, copies, and events.
// - More comprehensive than hipStreamSynchronize().
// - Does not affect other contexts.
//
// Multi-GPU: Only synchronizes the current context, not other contexts
// on the same or different devices.
//
// Performance note: Use stream-specific synchronization when possible.
//
// See also: hipDeviceSynchronize, hipStreamSynchronize, hipEventSynchronize.
HIPAPI hipError_t hipCtxSynchronize(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_context_synchronize(context);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Enables peer access from current context to peer context.
//
// Parameters:
//  - peerContext: [IN] Peer context to enable access to.
//  - flags: [IN] Reserved for future use (must be 0).
//
// Returns:
//  - hipSuccess: Peer access enabled successfully.
//  - hipErrorInvalidValue: peerContext is NULL or invalid flags.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorPeerAccessAlreadyEnabled: Access already enabled.
//  - hipErrorPeerAccessUnsupported: Devices cannot access each other.
//  - hipErrorInvalidDevice: Contexts on same device.
//
// Synchronization: This operation is synchronous.
//
// Peer access behavior:
// - Enables current context to access memory in peer context.
// - Access is unidirectional (must enable separately for bidirectional).
// - Contexts must be on different devices.
// - Devices must support P2P access.
// - Remains enabled until explicitly disabled.
//
// Multi-GPU:
// - Enables direct GPU-to-GPU memory access.
// - Avoids staging through host memory.
// - Improves performance for multi-GPU applications.
//
// See also: hipCtxDisablePeerAccess, hipDeviceCanAccessPeer,
//           hipDeviceEnablePeerAccess.
HIPAPI hipError_t hipCtxEnablePeerAccess(hipCtx_t peerContext,
                                         unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!peerContext) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_context_enable_peer_access(
      context, (iree_hal_streaming_context_t*)peerContext);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Disables peer access from current context to peer context.
//
// Parameters:
//  - peerContext: [IN] Peer context to disable access to.
//
// Returns:
//  - hipSuccess: Peer access disabled successfully.
//  - hipErrorInvalidValue: peerContext is NULL.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorPeerAccessNotEnabled: Peer access was not enabled.
//
// Synchronization: This operation is synchronous. Waits for all peer
// operations to complete.
//
// Peer access behavior:
// - Disables current context's access to peer context memory.
// - Only affects current context's access (unidirectional).
// - Subsequent peer operations will fail.
// - All ongoing peer operations complete before disabling.
//
// Multi-GPU:
// - Returns to staged copy behavior through host memory.
// - May reduce performance for multi-GPU operations.
//
// Warning: Ensure no kernels are actively using peer memory.
//
// See also: hipCtxEnablePeerAccess, hipDeviceDisablePeerAccess.
HIPAPI hipError_t hipCtxDisablePeerAccess(hipCtx_t peerContext) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!peerContext) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_context_disable_peer_access(
      context, (iree_hal_streaming_context_t*)peerContext);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets resource limits for the current device.
//
// Parameters:
//  - pValue: [OUT] Pointer to receive the limit value.
//  - limit: [IN] Limit to query (hipLimitStackSize, hipLimitPrintfFifoSize,
//                hipLimitMallocHeapSize, hipLimitDevRuntimeSyncDepth,
//                hipLimitDevRuntimePendingLaunchCount,
//                hipLimitMaxL2FetchGranularity, hipLimitPersistingL2CacheSize).
//
// Returns:
//  - hipSuccess: Limit retrieved successfully.
//  - hipErrorInvalidValue: pValue is NULL or invalid limit.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorUnsupportedLimit: Limit not supported on this device.
//
// Synchronization: This operation is synchronous.
//
// Common limits:
// - hipLimitStackSize: Stack size per thread in bytes.
// - hipLimitPrintfFifoSize: Size of printf FIFO in bytes.
// - hipLimitMallocHeapSize: Size of device malloc heap in bytes.
// - hipLimitDevRuntimeSyncDepth: Maximum nesting depth of sync operations.
// - hipLimitDevRuntimePendingLaunchCount: Maximum pending launches.
// - hipLimitMaxL2FetchGranularity: L2 cache fetch granularity.
// - hipLimitPersistingL2CacheSize: Persisting L2 cache size.
//
// Multi-GPU: Limits are per-device and context-specific.
//
// See also: hipDeviceSetLimit, hipDeviceGetAttribute.
HIPAPI hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pValue) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the limit value using internal API.
  iree_status_t status = iree_hal_streaming_context_limit(
      context, iree_hip_limit_to_internal(limit), pValue);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets resource limits for the current device.
//
// Parameters:
//  - limit: [IN] Limit to set (hipLimitStackSize, hipLimitPrintfFifoSize,
//                hipLimitMallocHeapSize, etc.).
//  - value: [IN] New limit value in bytes.
//
// Returns:
//  - hipSuccess: Limit set successfully.
//  - hipErrorInvalidValue: Invalid limit or value.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorUnsupportedLimit: Limit not supported or read-only.
//  - hipErrorMemoryAllocation: Insufficient memory for new limit.
//
// Synchronization: This operation is synchronous.
//
// Limit behavior:
// - Changes take effect for subsequent kernel launches.
// - Does not affect currently running kernels.
// - Some limits may require context reset to take effect.
// - Setting to 0 may restore default value.
//
// Multi-GPU: Limits are per-device and context-specific.
//
// Warning: Increasing limits may reduce available memory for allocations.
// Setting limits too low may cause kernel launch failures.
//
// See also: hipDeviceGetLimit, hipFuncSetAttribute.
HIPAPI hipError_t hipDeviceSetLimit(hipLimit_t limit, size_t value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Set the limit value using internal API.
  iree_status_t status = iree_hal_streaming_context_set_limit(
      context, iree_hip_limit_to_internal(limit), value);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

// Gets the amount of free and total device memory.
//
// Parameters:
//  - free: [OUT] Pointer to receive free memory in bytes (can be NULL).
//  - total: [OUT] Pointer to receive total memory in bytes (can be NULL).
//
// Returns:
//  - hipSuccess: Memory info retrieved successfully.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Current device is invalid.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory info behavior:
// - Free memory is currently available for allocation.
// - Total memory is the total device memory capacity.
// - Values may change between calls due to other allocations.
// - Does not include host memory.
//
// Multi-GPU: Returns memory info for the current device.
//
// Usage pattern:
// ```c
// size_t free_mem, total_mem;
// hipMemGetInfo(&free_mem, &total_mem);
// printf("GPU memory: %zu/%zu bytes free\n", free_mem, total_mem);
// ```
//
// Note: Free memory may be fragmented; large allocations may fail even if
// total free memory exceeds the requested size.
//
// See also: hipDeviceTotalMem, hipMalloc, hipSetDevice.
HIPAPI hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_device_size_t free_memory = 0;
  iree_device_size_t total_memory = 0;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_device_memory_info(context->device_ordinal,
                                            &free_memory, &total_memory),
      hipErrorInvalidDevice);

  if (free) *free = (size_t)free_memory;
  if (total) *total = (size_t)total_memory;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

static hipError_t iree_hip_current_mem_pool(
    iree_hal_streaming_context_t* context, hrx_mem_pool_t* out_pool) {
  if (!context || !context->device_entry) return hipErrorInvalidDevice;
  hrx_mem_pool_t pool =
      iree_hal_streaming_device_retain_mem_pool(context->device_entry);
  if (!pool) return hipErrorInvalidDevice;
  *out_pool = pool;
  return hipSuccess;
}

static hipError_t iree_hip_context_total_memory_from_spec(
    iree_hal_streaming_context_t* context, bool* out_known,
    iree_device_size_t* out_total) {
  if (!context || !out_known || !out_total) return hipErrorInvalidDevice;
  *out_known = false;
  *out_total = 0;

  iree_hal_device_t* hal_device =
      hrx_device_hal(iree_hip_hrx_device_from_context(context));
  if (!hal_device) return hipErrorInvalidDevice;

  iree_hal_device_observation_t observation;
  iree_hal_device_observation_initialize(
      IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY, &observation);
  iree_status_t status =
      iree_hal_device_observation_populate_memory_total_from_spec(
          iree_hal_device_spec(hal_device), &observation);
  if (!iree_status_is_ok(status)) {
    return iree_status_to_hip_result(status);
  }
  if (iree_all_bits_set(observation.memory.flags,
                        IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES)) {
    *out_known = true;
    *out_total = observation.memory.total_bytes;
  }
  return hipSuccess;
}

static hipError_t iree_hip_malloc_from_pool(
    iree_hal_streaming_context_t* context, hrx_mem_pool_t pool, size_t size,
    iree_hal_streaming_stream_t* stream, void** out_ptr) {
  if (!pool) return hipErrorInvalidValue;

  // Reject absurdly large sizes that can't possibly succeed.
  {
    bool total_memory_known = false;
    iree_device_size_t total_memory = 0;
    hipError_t query_result = iree_hip_context_total_memory_from_spec(
        context, &total_memory_known, &total_memory);
    if (query_result != hipSuccess) return query_result;
    if (size > (size_t)IREE_DEVICE_SIZE_MAX ||
        (total_memory_known && (iree_device_size_t)size > total_memory)) {
      return hipErrorOutOfMemory;
    }
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  const size_t allocation_size = iree_max(size, (size_t)8);
  iree_status_t status =
      stream ? iree_hal_streaming_memory_allocate_device_from_pool_async(
                   context, pool, allocation_size, /*flags=*/0, stream, &buffer)
             : iree_hal_streaming_memory_allocate_device_from_pool(
                   context, pool, allocation_size, /*flags=*/0, &buffer);
  if (!iree_status_is_ok(status)) return iree_status_to_hip_result(status);
  buffer->logical_size = (iree_device_size_t)size;
  *out_ptr = (void*)buffer->device_ptr;
  return hipSuccess;
}

enum {
  IREE_HIP_MANAGED_PAGE_SIZE = 4096,
  IREE_HIP_MANAGED_CPU_ACCESS_BIT = 63,
};

static hipError_t iree_hip_validate_managed_location(int device,
                                                     bool allow_cpu) {
  if (device == hipCpuDeviceId) {
    return allow_cpu ? hipSuccess : hipErrorInvalidDevice;
  }
  if (device < 0 || device >= IREE_HIP_MANAGED_CPU_ACCESS_BIT) {
    return hipErrorInvalidDevice;
  }
  return iree_hal_streaming_device_entry(device) ? hipSuccess
                                                 : hipErrorInvalidDevice;
}

static uint64_t iree_hip_managed_location_mask(int device) {
  return device == hipCpuDeviceId ? (1ull << IREE_HIP_MANAGED_CPU_ACCESS_BIT)
                                  : (1ull << (uint32_t)device);
}

static bool iree_hip_managed_memory_page_range(
    const iree_hal_streaming_buffer_ref_t* buffer_ref, iree_device_size_t count,
    iree_host_size_t* out_first_page, iree_host_size_t* out_page_count) {
  IREE_ASSERT_ARGUMENT(buffer_ref);
  IREE_ASSERT_ARGUMENT(out_first_page);
  IREE_ASSERT_ARGUMENT(out_page_count);
  *out_first_page = 0;
  *out_page_count = 0;
  if (!buffer_ref->buffer || count == 0 ||
      !buffer_ref->buffer->managed_page_count) {
    return false;
  }
  if (buffer_ref->offset > IREE_DEVICE_SIZE_MAX - count) return false;

  const iree_device_size_t first_offset = buffer_ref->offset;
  const iree_device_size_t end_offset = first_offset + count;
  iree_device_size_t rounded_end = 0;
  if (end_offset > IREE_DEVICE_SIZE_MAX - (IREE_HIP_MANAGED_PAGE_SIZE - 1)) {
    return false;
  }
  rounded_end = end_offset + (IREE_HIP_MANAGED_PAGE_SIZE - 1);

  const iree_host_size_t first_page =
      (iree_host_size_t)(first_offset / IREE_HIP_MANAGED_PAGE_SIZE);
  const iree_host_size_t end_page =
      (iree_host_size_t)(rounded_end / IREE_HIP_MANAGED_PAGE_SIZE);
  if (end_page < first_page ||
      end_page > buffer_ref->buffer->managed_page_count) {
    return false;
  }
  *out_first_page = first_page;
  *out_page_count = end_page - first_page;
  return *out_page_count != 0;
}

static hipError_t iree_hip_lookup_streaming_range_with_owner(
    iree_hal_streaming_context_t* context, const void* dev_ptr, size_t count,
    iree_hal_streaming_context_t** out_owner_context,
    iree_hal_streaming_buffer_ref_t* out_buffer_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dev_ptr);
  IREE_ASSERT_ARGUMENT(out_owner_context);
  IREE_ASSERT_ARGUMENT(out_buffer_ref);
  *out_owner_context = NULL;
  memset(out_buffer_ref, 0, sizeof(*out_buffer_ref));

  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)(uintptr_t)dev_ptr, count,
      out_buffer_ref);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_context_retain(context);
    *out_owner_context = context;
    return hipSuccess;
  }
  if (!iree_status_is_ok(status) &&
      iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
    iree_status_ignore(status);
    status = iree_hal_streaming_memory_lookup_range_across_contexts(
        (iree_hal_streaming_deviceptr_t)(uintptr_t)dev_ptr, count,
        out_owner_context, out_buffer_ref);
  }
  if (!iree_status_is_ok(status)) {
    return iree_status_to_fixed_hip_result(status, hipErrorInvalidValue);
  }
  return hipSuccess;
}

static hipError_t iree_hip_lookup_managed_range(
    iree_hal_streaming_context_t* context, const void* dev_ptr, size_t count,
    iree_hal_streaming_context_t** out_owner_context,
    iree_hal_streaming_buffer_ref_t* out_buffer_ref) {
  hipError_t result = iree_hip_lookup_streaming_range_with_owner(
      context, dev_ptr, count, out_owner_context, out_buffer_ref);
  if (result != hipSuccess) return result;
  if (out_buffer_ref->buffer->is_managed) return hipSuccess;
  iree_hal_streaming_context_release(*out_owner_context);
  *out_owner_context = NULL;
  return hipErrorInvalidValue;
}

static hipError_t iree_hip_lookup_advisable_range(
    iree_hal_streaming_context_t* context, const void* dev_ptr, size_t count,
    iree_hal_streaming_context_t** out_owner_context,
    iree_hal_streaming_buffer_ref_t* out_buffer_ref) {
  hipError_t result = iree_hip_lookup_streaming_range_with_owner(
      context, dev_ptr, count, out_owner_context, out_buffer_ref);
  if (result != hipSuccess) return result;
  if ((out_buffer_ref->buffer->is_managed ||
       out_buffer_ref->buffer->imported_host_allocation) &&
      out_buffer_ref->buffer->managed_page_count) {
    return hipSuccess;
  }
  iree_hal_streaming_context_release(*out_owner_context);
  *out_owner_context = NULL;
  return hipErrorInvalidValue;
}

static uint64_t iree_hip_managed_common_accessed_by_mask(
    const uint64_t* masks, iree_host_size_t first_page,
    iree_host_size_t page_count);

static hipError_t iree_hip_managed_location_to_device(hipMemLocation location,
                                                      int* out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = hipInvalidDeviceId;
  switch (location.type) {
    case hipMemLocationTypeDevice: {
      if (location.id < 0) return hipErrorInvalidValue;
      hipError_t result =
          iree_hip_validate_managed_location(location.id, /*allow_cpu=*/false);
      if (result != hipSuccess) return result;
      *out_device = location.id;
      return hipSuccess;
    }
    case hipMemLocationTypeHost:
    case hipMemLocationTypeHostNumaCurrent:
      *out_device = hipCpuDeviceId;
      return hipSuccess;
    case hipMemLocationTypeHostNuma:
      if (location.id < 0) return hipErrorInvalidValue;
      *out_device = hipCpuDeviceId;
      return hipSuccess;
    case hipMemLocationTypeInvalid:
    default:
      return hipErrorInvalidValue;
  }
}

static hipError_t iree_hip_managed_apply_advice(
    iree_hal_streaming_context_t* context, const void* dev_ptr, size_t count,
    hipMemAdvise_t advice, int device) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dev_ptr);

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_ref_t buffer_ref;
  hipError_t result = iree_hip_lookup_advisable_range(
      context, dev_ptr, count, &owner_context, &buffer_ref);
  if (result != hipSuccess) return result;

  iree_host_size_t first_page = 0;
  iree_host_size_t page_count = 0;
  if (!iree_hip_managed_memory_page_range(
          &buffer_ref, (iree_device_size_t)count, &first_page, &page_count)) {
    iree_hal_streaming_context_release(owner_context);
    return hipErrorInvalidValue;
  }

  switch (advice) {
    case hipMemAdviseSetReadMostly:
    case hipMemAdviseUnsetReadMostly: {
      const bool read_mostly = advice == hipMemAdviseSetReadMostly;
      for (iree_host_size_t i = 0; i < page_count; ++i) {
        buffer_ref.buffer->managed_read_mostly_pages[first_page + i] =
            read_mostly;
      }
      buffer_ref.buffer->read_mostly_hint = read_mostly;
      iree_hal_streaming_context_release(owner_context);
      return hipSuccess;
    }
    case hipMemAdviseSetPreferredLocation:
    case hipMemAdviseUnsetPreferredLocation: {
      result = iree_hip_validate_managed_location(device, /*allow_cpu=*/true);
      if (result != hipSuccess) {
        iree_hal_streaming_context_release(owner_context);
        return result;
      }
      const int32_t location = advice == hipMemAdviseSetPreferredLocation
                                   ? device
                                   : hipInvalidDeviceId;
      for (iree_host_size_t i = 0; i < page_count; ++i) {
        buffer_ref.buffer->managed_preferred_locations[first_page + i] =
            location;
      }
      buffer_ref.buffer->preferred_location = location;
      iree_hal_streaming_context_release(owner_context);
      return hipSuccess;
    }
    case hipMemAdviseSetAccessedBy:
    case hipMemAdviseUnsetAccessedBy: {
      result = iree_hip_validate_managed_location(device, /*allow_cpu=*/true);
      if (result != hipSuccess) {
        iree_hal_streaming_context_release(owner_context);
        return result;
      }
      const uint64_t mask = iree_hip_managed_location_mask(device);
      for (iree_host_size_t i = 0; i < page_count; ++i) {
        uint64_t* page_mask =
            &buffer_ref.buffer
                 ->managed_accessed_by_device_masks[first_page + i];
        if (advice == hipMemAdviseSetAccessedBy) {
          *page_mask |= mask;
        } else {
          *page_mask &= ~mask;
        }
      }
      buffer_ref.buffer->accessed_by_device_mask =
          iree_hip_managed_common_accessed_by_mask(
              buffer_ref.buffer->managed_accessed_by_device_masks, 0,
              buffer_ref.buffer->managed_page_count);
      iree_hal_streaming_context_release(owner_context);
      return hipSuccess;
    }
    case hipMemAdviseSetCoarseGrain:
    case hipMemAdviseUnsetCoarseGrain: {
      const int32_t coherency_mode = advice == hipMemAdviseSetCoarseGrain
                                         ? hipMemRangeCoherencyModeCoarseGrain
                                         : hipMemRangeCoherencyModeFineGrain;
      for (iree_host_size_t i = 0; i < page_count; ++i) {
        buffer_ref.buffer->managed_coherency_modes[first_page + i] =
            coherency_mode;
      }
      buffer_ref.buffer->coherency_mode = coherency_mode;
      iree_hal_streaming_context_release(owner_context);
      return hipSuccess;
    }
    default:
      iree_hal_streaming_context_release(owner_context);
      return hipErrorInvalidValue;
  }
}

static hipError_t iree_hip_managed_record_prefetch(
    iree_hal_streaming_context_t* context, const void* dev_ptr, size_t count,
    int device) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dev_ptr);

  hipError_t result =
      iree_hip_validate_managed_location(device, /*allow_cpu=*/true);
  if (result != hipSuccess) return result;

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_ref_t buffer_ref;
  result = iree_hip_lookup_managed_range(context, dev_ptr, count,
                                         &owner_context, &buffer_ref);
  if (result != hipSuccess) return result;

  iree_host_size_t first_page = 0;
  iree_host_size_t page_count = 0;
  if (!iree_hip_managed_memory_page_range(
          &buffer_ref, (iree_device_size_t)count, &first_page, &page_count)) {
    iree_hal_streaming_context_release(owner_context);
    return hipErrorInvalidValue;
  }
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    buffer_ref.buffer->managed_last_prefetch_locations[first_page + i] = device;
  }
  buffer_ref.buffer->last_prefetch_location = device;
  iree_hal_streaming_context_release(owner_context);
  return hipSuccess;
}

static int32_t iree_hip_managed_uniform_location(const int32_t* locations,
                                                 iree_host_size_t first_page,
                                                 iree_host_size_t page_count) {
  int32_t location = locations[first_page];
  for (iree_host_size_t i = 1; i < page_count; ++i) {
    if (locations[first_page + i] != location) return hipInvalidDeviceId;
  }
  return location;
}

static int32_t iree_hip_managed_uniform_coherency_mode(
    const int32_t* modes, iree_host_size_t first_page,
    iree_host_size_t page_count) {
  int32_t mode = modes[first_page];
  for (iree_host_size_t i = 1; i < page_count; ++i) {
    if (modes[first_page + i] != mode) {
      return hipMemRangeCoherencyModeIndeterminate;
    }
  }
  return mode;
}

static bool iree_hip_managed_all_pages_read_mostly(
    const bool* pages, iree_host_size_t first_page,
    iree_host_size_t page_count) {
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    if (!pages[first_page + i]) return false;
  }
  return true;
}

static uint64_t iree_hip_managed_common_accessed_by_mask(
    const uint64_t* masks, iree_host_size_t first_page,
    iree_host_size_t page_count) {
  uint64_t mask = masks[first_page];
  for (iree_host_size_t i = 1; i < page_count; ++i) {
    mask &= masks[first_page + i];
  }
  return mask;
}

static void iree_hip_managed_fill_accessed_by(uint64_t mask, int* devices,
                                              iree_host_size_t device_count) {
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    devices[i] = hipInvalidDeviceId;
  }
  iree_host_size_t position = 0;
  for (int device = 0;
       device < IREE_HIP_MANAGED_CPU_ACCESS_BIT && position < device_count;
       ++device) {
    if (mask & (1ull << (uint32_t)device)) {
      devices[position++] = device;
    }
  }
  if ((mask & (1ull << IREE_HIP_MANAGED_CPU_ACCESS_BIT)) &&
      position < device_count) {
    devices[position++] = hipCpuDeviceId;
  }
}

// Allocates memory on the device.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive the allocated device memory pointer.
//  - size: [IN] Size in bytes to allocate.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient device memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal allocation error.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Allocated memory is uninitialized.
// - Memory persists until freed with hipFree().
// - Memory is accessible from all streams on the device.
// - Allocation is aligned to meet all alignment requirements.
// - Zero-size allocations are allowed and return NULL.
//
// Multi-GPU: Memory is allocated on the current device.
//
// Warning: Always check return value before using the allocated pointer.
// Dereferencing a failed allocation results in undefined behavior.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// See also: hipFree, hipMallocPitch, hipMallocHost, hipMallocManaged,
//           hipMallocAsync.
HIPAPI hipError_t hipMalloc(void** ptr, size_t size) {
  HIP_DEBUG_LOG("[HIP_API] hipMalloc(%zu)\n", size);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *ptr = NULL;

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (tls_stream_capture_mode != hipStreamCaptureModeRelaxed &&
      iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  if (size == 0) {
    *ptr = NULL;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  const size_t allocation_size = iree_max(size, (size_t)8);
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, allocation_size, /*flags=*/0, &buffer);
  hipError_t result = iree_status_to_hip_result(status);
  if (result == hipSuccess) {
    buffer->logical_size = (iree_device_size_t)size;
    *ptr = (void*)buffer->device_ptr;
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates device memory with specific memory type flags.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive the allocated memory pointer.
//  - sizeBytes: [IN] Requested memory size in bytes.
//  - flags: [IN] Memory allocation flags (hipDeviceMallocDefault, etc.)
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL or invalid flags.
//  - hipErrorOutOfMemory: Insufficient device memory.
//
// Note: Default, uncached, and fixed-size signal-memory allocation modes are
// implemented. Fine-grained allocations require a distinct HAL memory property
// and fail loudly instead of falling back to default memory.
//
// See also: hipMalloc, hipFree.
HIPAPI hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes,
                                        unsigned int flags) {
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }
  if (!ptr) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *ptr = NULL;
  switch (flags) {
    case hipDeviceMallocDefault:
      return hipMalloc(ptr, sizeBytes);
    case hipMallocSignalMemory: {
      if (sizeBytes != sizeof(uint64_t)) {
        HIP_RETURN_ERROR(hipErrorInvalidValue);
      }

      iree_hal_streaming_context_t* context = NULL;
      hipError_t context_result = iree_hip_ensure_context(&context);
      if (context_result != hipSuccess) {
        HIP_RETURN_ERROR(context_result);
      }
      if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
        HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
      }

      iree_hal_streaming_buffer_t* buffer = NULL;
      iree_status_t status = iree_hal_streaming_memory_allocate_host(
          context, sizeBytes,
          IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_SIGNAL_MEMORY, &buffer);
      if (iree_status_is_ok(status)) {
        *ptr = buffer->host_ptr;
      }
      return iree_status_to_hip_result(status);
    }
    case hipDeviceMallocUncached: {
      if (sizeBytes == 0) {
        return hipSuccess;
      }
      iree_hal_streaming_context_t* context = NULL;
      hipError_t context_result = iree_hip_ensure_context(&context);
      if (context_result != hipSuccess) {
        HIP_RETURN_ERROR(context_result);
      }
      if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
        HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
      }
      iree_hal_streaming_buffer_t* buffer = NULL;
      iree_status_t status = iree_hal_streaming_memory_allocate_device(
          context, sizeBytes, IREE_HAL_STREAMING_MEMORY_FLAG_UNCACHED, &buffer);
      if (iree_status_is_ok(status)) {
        *ptr = (void*)iree_hal_streaming_buffer_device_pointer(buffer);
      }
      return iree_status_to_hip_result(status);
    }
    case hipDeviceMallocFinegrained: {
      // Zero-byte allocations never consume a mode-specific resource, while
      // unrepresentable requests must not be misreported as missing hardware
      // support.
      if (sizeBytes == 0) {
        return hipSuccess;
      }
      hipError_t size_result =
          iree_hip_validate_host_allocation_size(sizeBytes);
      if (size_result != hipSuccess) {
        HIP_RETURN_ERROR(size_result);
      }
      HIP_RETURN_ERROR(hipErrorNotSupported);
    }
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
}

// Allocates pitched linear memory on the device.
//
// Parameters:
//  - devPtr: [OUT] Pointer to receive the allocated device memory pointer.
//  - pitch: [OUT] Pointer to receive the pitch in bytes.
//  - width: [IN] Requested width of allocation in bytes.
//  - height: [IN] Requested height of allocation in rows.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: devPtr or pitch is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorOutOfMemory: Insufficient device memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Allocates at least width bytes per row.
// - Actual pitch may be larger than width for alignment.
// - Use returned pitch for row-to-row calculations.
// - Memory layout: row[i] starts at devPtr + i * pitch.
// - Allocated memory is uninitialized.
//
// Multi-GPU: Memory is allocated on the current device.
//
// Performance note: Pitched memory can improve coalescing for 2D data
// access patterns.
//
// Warning: Always use the returned pitch value, not width, when accessing
// rows.
//
// See also: hipMalloc, hipMemcpy2D, hipFree.
HIPAPI hipError_t hipMallocPitch(void** devPtr, size_t* pitch, size_t width,
                                 size_t height) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!devPtr || !pitch) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *devPtr = NULL;
  *pitch = 0;
  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Get current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Allocate pitched memory.
  // HIP doesn't have an ElementSizeBytes parameter like CUDA, so we pass 0.
  size_t calculated_pitch = 0;
  iree_hal_streaming_buffer_t* buffer = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_memory_allocate_device_pitched(
          context, width, height, 0, &calculated_pitch, &buffer),
      hipErrorOutOfMemory);

  // Return device pointer and pitch.
  *devPtr = (void*)iree_hal_streaming_buffer_device_pointer(buffer);
  *pitch = calculated_pitch;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Frees memory allocated with hipMalloc.
//
// Parameters:
//  - ptr: [IN] Device pointer to free (can be NULL).
//
// Returns:
//  - hipSuccess: Memory freed successfully or ptr was NULL.
//  - hipErrorInvalidDevicePointer: ptr is not a valid allocation.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorDeinitialized: HIP runtime has been deinitialized.
//
// Synchronization: This operation is synchronous. Waits for all operations
// using the memory to complete before freeing.
//
// Memory behavior:
// - Freeing NULL is a no-op and returns hipSuccess.
// - After freeing, the pointer becomes invalid.
// - Using freed memory results in undefined behavior.
// - Double-free results in undefined behavior.
//
// Multi-GPU: Memory must be freed from the same context that allocated it.
//
// Warning: Ensure all kernels using this memory have completed before
// freeing. Use hipDeviceSynchronize() if unsure.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// See also: hipMalloc, hipFreeHost, hipFreeAsync.
HIPAPI hipError_t hipFree(void* ptr) {
  HIP_DEBUG_LOG("[HIP_API] hipFree(%p)\n", ptr);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  // hipFree is synchronous - the internal function handles synchronization.
  iree_status_t status = iree_hal_streaming_memory_free_device(
      context, (iree_hal_streaming_deviceptr_t)ptr);

  // A signal allocation is host-visible storage but, unlike ordinary host
  // allocations, this API owns it. Keep that exception explicit so hipFree
  // does not accidentally accept arbitrary pinned host pointers.
  hipError_t result;
  if (iree_status_is_ok(status)) {
    result = hipSuccess;
  } else if (iree_status_code(status) == IREE_STATUS_NOT_FOUND ||
             iree_status_code(status) == IREE_STATUS_INVALID_ARGUMENT) {
    iree_status_free(status);
    iree_hal_streaming_host_register_flags_t host_flags =
        IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
    status = iree_hal_streaming_memory_host_flags(context, ptr, &host_flags);
    if (!iree_status_is_ok(status)) {
      if (iree_status_code(status) == IREE_STATUS_NOT_FOUND ||
          iree_status_code(status) == IREE_STATUS_INVALID_ARGUMENT) {
        iree_status_free(status);
        result = hipErrorInvalidValue;
      } else {
        result = iree_status_to_hip_result(status);
      }
    } else if (iree_any_bit_set(
                   host_flags,
                   IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_SIGNAL_MEMORY)) {
      status = iree_hal_streaming_memory_free_host(context, ptr);
      if (iree_status_is_ok(status)) {
        result = hipSuccess;
      } else if (iree_status_code(status) == IREE_STATUS_NOT_FOUND ||
                 iree_status_code(status) == IREE_STATUS_INVALID_ARGUMENT) {
        iree_status_free(status);
        result = hipErrorInvalidValue;
      } else {
        result = iree_status_to_hip_result(status);
      }
    } else {
      result = hipErrorInvalidValue;
    }
  } else {
    result = iree_status_to_hip_result(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

HIPAPI hipError_t hipFreeArray(hipArray_t array) {
  if (!array) HIP_RETURN_ERROR(hipErrorInvalidValue);
  struct hipArray_st* removed_array = NULL;
  if (!iree_hip_array_registry_remove(array, &removed_array)) {
    HIP_RETURN_ERROR(hipErrorContextIsDestroyed);
  }
  iree_hip_array_release(removed_array);
  return hipSuccess;
}

// Allocates page-locked host memory accessible from device.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive the allocated host memory pointer.
//  - size: [IN] Size in bytes to allocate.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient host memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Allocates pinned (page-locked) host memory.
// - Memory is accessible from both host and device.
// - Enables higher bandwidth for host-device transfers.
// - Memory is uninitialized.
// - Must be freed with hipFreeHost().
//
// Multi-GPU: Memory is accessible from all devices in the system.
//
// Performance note: Pinned memory improves transfer performance but reduces
// available memory for other processes. Use judiciously.
//
// Warning: Excessive pinned memory allocation can degrade system
// performance.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// See also: hipMalloc, hipFreeHost, hipFreeAsync.
HIPAPI hipError_t hipMallocHost(void** ptr, size_t size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *ptr = NULL;
  hipError_t size_result = iree_hip_validate_host_allocation_size(size);
  if (size_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(size_result);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_host(
      context, size, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT, &buffer);

  if (iree_status_is_ok(status)) {
    *ptr = buffer->host_ptr;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Frees page-locked host memory allocated with hipMallocHost.
//
// Parameters:
//  - ptr: [IN] Host pointer to free (can be NULL).
//
// Returns:
//  - hipSuccess: Memory freed successfully or ptr was NULL.
//  - hipErrorInvalidValue: ptr is not a valid pinned allocation.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Freeing NULL is a no-op and returns hipSuccess.
// - After freeing, the pointer becomes invalid.
// - Using freed memory results in undefined behavior.
//
// Multi-GPU: Can be called from any device context.
//
// Warning: Only use for memory allocated with hipMallocHost or
// hipHostMalloc.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// See also: hipMalloc, hipFreeHost, hipFreeAsync.
HIPAPI hipError_t hipFreeHost(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  // hipFreeHost is synchronous - the internal function handles synchronization.
  iree_status_t status = iree_hal_streaming_memory_free_host(context, ptr);

  // Convert status to HIP error. Invalid pointers (including already-freed
  // pointers) should return hipErrorInvalidValue, not hipErrorNotFound.
  hipError_t result;
  if (iree_status_is_ok(status)) {
    result = hipSuccess;
  } else if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
    // Pointer not found in allocation table - invalid or already freed.
    iree_status_free(status);
    result = hipErrorInvalidValue;
  } else {
    result = iree_status_to_hip_result(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates host memory with specified properties.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive the allocated host memory pointer.
//  - size: [IN] Size in bytes to allocate.
//  - flags: [IN] Allocation flags (hipHostMallocDefault, hipHostMallocPortable,
//                hipHostMallocMapped, hipHostMallocWriteCombined, etc).
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL or invalid flags.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient host memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - hipHostMallocDefault: Page-locked, accessible from all devices.
// - hipHostMallocPortable: Accessible from all devices in all contexts.
// - hipHostMallocMapped: Maps into device address space.
// - hipHostMallocWriteCombined: Optimized for device reads.
// - hipHostMallocCoherent: Coherent between host and device.
// - hipHostMallocNonCoherent: May require explicit synchronization.
// - Memory is uninitialized.
// - Must be freed with hipHostFree().
//
// Multi-GPU: Flags control multi-device accessibility.
//
// Performance note: Write-combined memory is fast for device reads but slow
// for host reads.
//
// See also: hipHostFree, hipMallocHost, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *ptr = NULL;

  // Zero-size allocations return nullptr.
  if (size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  hipError_t size_result = iree_hip_validate_host_allocation_size(size);
  if (size_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(size_result);
  }
  hipError_t flags_result = iree_hip_validate_host_malloc_flags(flags);
  if (flags_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(flags_result);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_host(
      context, size, iree_hip_host_flags_to_internal(flags), &buffer);

  if (iree_status_is_ok(status)) {
    *ptr = buffer->host_ptr;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates page-locked host memory through the legacy hipHostAlloc API.
//
// Parameters:
//  - ptr: [OUT] Pointer to allocated host memory.
//  - size: [IN] Size of allocation in bytes.
//  - flags: [IN] Allocation flags (hipHostAllocDefault, hipHostAllocPortable,
//                hipHostAllocMapped, hipHostAllocWriteCombined, etc).
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL.
//  - hipErrorOutOfMemory: Insufficient memory available.
//
// Note: hipHostAlloc uses the legacy hipHostAlloc flag surface. HIP-only
// hipHostMalloc flags such as coherent, non-coherent, and NUMA-user are
// rejected.
//
// See also: hipHostMalloc, hipHostFree.
HIPAPI hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *ptr = NULL;

  hipError_t flags_result = iree_hip_validate_host_alloc_flags(flags);
  if (flags_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(flags_result);
  }
  hipError_t size_result = iree_hip_validate_host_allocation_size(size);
  if (size_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(size_result);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipHostMalloc(ptr, size, flags);
}

// Frees host memory allocated with hipHostMalloc.
//
// Parameters:
//  - ptr: [IN] Host pointer to free (can be NULL).
//
// Returns:
//  - hipSuccess: Memory freed successfully or ptr was NULL.
//  - hipErrorInvalidValue: ptr is not a valid allocation.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Freeing NULL is a no-op and returns hipSuccess.
// - Unmaps any device mappings if hipHostMallocMapped was used.
// - After freeing, both host and device pointers become invalid.
//
// Multi-GPU: Can be called from any device context.
//
// See also: hipHostMalloc, hipFreeHost, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostFree(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  // hipHostFree is synchronous - the internal function handles synchronization.
  iree_status_t status = iree_hal_streaming_memory_free_host(context, ptr);

  // Convert status to HIP error. Invalid pointers (including already-freed
  // pointers) should return hipErrorInvalidValue, not hipErrorNotFound.
  hipError_t result;
  if (iree_status_is_ok(status)) {
    result = hipSuccess;
  } else if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
    // Pointer not found in allocation table - invalid or already freed.
    iree_status_free(status);
    result = hipErrorInvalidValue;
  } else {
    result = iree_status_to_hip_result(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates host-visible memory that is also addressable from device code.
HIPAPI hipError_t hipMallocManaged(void** dev_ptr, size_t size,
                                   unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dev_ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *dev_ptr = NULL;

  if (size == 0 || (flags != hipMemAttachGlobal && flags != hipMemAttachHost &&
                    flags != hipMemAttachSingle)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (size > SIZE_MAX - (IREE_HIP_MANAGED_PAGE_SIZE - 1)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (tls_stream_capture_mode != hipStreamCaptureModeRelaxed &&
      iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_managed(
      context, (iree_host_size_t)size, flags, &buffer);
  if (!iree_status_is_ok(status)) {
    hipError_t result = iree_status_to_hip_result(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  if (!buffer->host_ptr ||
      (iree_hal_streaming_deviceptr_t)(uintptr_t)buffer->host_ptr !=
          buffer->device_ptr) {
    iree_status_ignore(
        iree_hal_streaming_memory_free_device(context, buffer->device_ptr));
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorNotSupported);
  }

  *dev_ptr = buffer->host_ptr;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Registers existing host memory for use by the device.
//
// Parameters:
//  - ptr: [IN] Host memory pointer to register.
//  - size: [IN] Size in bytes of memory to register.
//  - flags: [IN] Registration flags (hipHostRegisterDefault,
//                hipHostRegisterPortable, hipHostRegisterMapped,
//                hipHostRegisterIoMemory).
//
// Returns:
//  - hipSuccess: Memory registered successfully.
//  - hipErrorInvalidValue: ptr is NULL, size is 0, or invalid flags.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorMemoryAllocation: Registration failed.
//  - hipErrorNotSupported: Flags not supported.
//
// Synchronization: This operation is synchronous.
//
// Registration behavior:
// - Pins the memory pages to prevent swapping.
// - Enables high-bandwidth transfers to/from device.
// - Memory must be page-aligned for best performance.
// - Must be unregistered with hipHostUnregister().
//
// Flags:
// - hipHostRegisterDefault: Basic pinning.
// - hipHostRegisterPortable: Accessible from all devices.
// - hipHostRegisterMapped: Maps into device address space.
// - hipHostRegisterIoMemory: Memory is IO memory.
//
// Multi-GPU: Portable flag enables access from all devices.
//
// Warning: Registering too much memory can degrade system performance.
//
// See also: hipHostUnregister, hipHostMalloc, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr || size == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t flags_result = iree_hip_validate_host_register_flags(flags);
  if (flags_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(flags_result);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_hal_streaming_host_register_flags_t internal_flags =
      iree_hip_host_flags_to_internal(flags);
  if (flags & hipHostRegisterReadOnly) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY;
  }

  // Register the host memory using the internal function.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_register_host(
      context, ptr, size, internal_flags, &buffer);

  hipError_t result = hipSuccess;
  if (iree_status_is_ok(status)) {
    result = hipSuccess;
  } else if (iree_status_code(status) == IREE_STATUS_INVALID_ARGUMENT ||
             iree_status_code(status) == IREE_STATUS_OUT_OF_RANGE ||
             iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED ||
             iree_status_code(status) == IREE_STATUS_UNKNOWN ||
             iree_status_code(status) == IREE_STATUS_INTERNAL) {
    iree_status_free(status);
    result = hipErrorInvalidValue;
  } else {
    result = iree_status_to_hip_result(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Unregisters host memory previously registered with hipHostRegister.
//
// Parameters:
//  - ptr: [IN] Host memory pointer to unregister.
//
// Returns:
//  - hipSuccess: Memory unregistered successfully.
//  - hipErrorInvalidValue: ptr is NULL or not registered.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorHostMemoryNotRegistered: Memory was not registered.
//
// Synchronization: This operation is synchronous.
//
// Unregistration behavior:
// - Unpins the memory pages.
// - Memory can be swapped again.
// - Removes device mappings if hipHostRegisterMapped was used.
// - Must match the pointer from hipHostRegister().
//
// Multi-GPU: Unregisters from all devices if portable.
//
// Warning: Ensure no device operations are using this memory.
//
// See also: hipHostRegister, hipHostFree.
HIPAPI hipError_t hipHostUnregister(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_non_relaxed_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  // Unregister the host memory using the internal function.
  iree_status_t status =
      iree_hal_streaming_memory_unregister_host(context, ptr);

  hipError_t result = hipSuccess;
  if (iree_status_is_ok(status)) {
    result = hipSuccess;
  } else {
    iree_status_free(status);
    result = hipErrorHostMemoryNotRegistered;
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets the address range of a device allocation.
//
// Parameters:
//  - pbase: [OUT] Pointer to receive the base address of allocation.
//  - psize: [OUT] Pointer to receive the size of allocation.
//  - dptr: [IN] Device pointer to query.
//
// Returns:
//  - hipSuccess: Address range retrieved successfully.
//  - hipErrorInvalidValue: pbase or psize is NULL.
//  - hipErrorInvalidDevicePointer: dptr is not a valid allocation.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Address range behavior:
// - Returns the original allocation containing dptr.
// - dptr can be anywhere within the allocation.
// - Useful for finding allocation boundaries.
//
// Multi-GPU: Queries allocation on current device.
//
// See also: hipMemPtrGetInfo, hipMalloc.
HIPAPI hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                        hipDeviceptr_t dptr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pbase || !psize) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_deviceptr_t base = 0;
  size_t size = 0;
  iree_status_t status = iree_hal_streaming_memory_address_range(
      context, (iree_hal_streaming_deviceptr_t)dptr, &base, &size);

  if (iree_status_is_ok(status)) {
    *pbase = (hipDeviceptr_t)base;
    *psize = size;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets device pointer for mapped host memory.
//
// Parameters:
//  - pdptr: [OUT] Pointer to receive the device pointer.
//  - p: [IN] Host pointer to query.
//  - flags: [IN] Reserved (must be 0).
//
// Returns:
//  - hipSuccess: Device pointer retrieved successfully.
//  - hipErrorInvalidValue: pdptr or p is NULL, or invalid flags.
//  - hipErrorMemoryAllocation: Host memory not mapped to device.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Mapping behavior:
// - Host memory must be allocated with hipHostMalloc(hipHostMallocMapped).
// - Or registered with hipHostRegister(hipHostRegisterMapped).
// - Returns device-accessible pointer to same memory.
// - Single memory location, two addresses.
//
// Multi-GPU: Device pointer valid on current device.
//
// Warning: Device pointer invalid if host memory unmapped.
//
// See also: hipHostMalloc, hipHostRegister, hipHostGetFlags.
HIPAPI hipError_t hipHostGetDevicePointer(hipDeviceptr_t* pdptr, void* p,
                                          unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pdptr || !p) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Flags must be 0.
  if (flags != 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_hal_streaming_context_t* owner_context = NULL;
  hipError_t lookup_result = iree_hip_lookup_streaming_range_with_owner(
      context, p, 1, &owner_context, &buffer_ref);
  if (lookup_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  *pdptr = (hipDeviceptr_t)(buffer_ref.buffer->device_ptr + buffer_ref.offset);
  iree_hal_streaming_context_release(owner_context);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets flags used to allocate pinned host memory.
//
// Parameters:
//  - flagsPtr: [OUT] Pointer to receive the allocation flags.
//  - hostPtr: [IN] Host pointer to query.
//
// Returns:
//  - hipSuccess: Flags retrieved successfully.
//  - hipErrorInvalidValue: flagsPtr or hostPtr is NULL.
//  - hipErrorInvalidHostPointer: hostPtr not allocated with HIP.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Flag information:
// - Returns flags from hipHostMalloc() or hipHostRegister().
// - hipHostMallocDefault, hipHostMallocPortable, hipHostMallocMapped,
//   hipHostMallocWriteCombined, hipHostMallocCoherent, etc.
// - Returns 0 for non-HIP allocations.
//
// Multi-GPU: Flags indicate portability across devices.
//
// See also: hipHostMalloc, hipHostRegister, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flagsPtr || !hostPtr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the internal flags.
  iree_hal_streaming_host_register_flags_t internal_flags = {0};
  iree_status_t status =
      iree_hal_streaming_memory_host_flags(context, hostPtr, &internal_flags);

  if (iree_status_is_ok(status)) {
    // Convert internal flags back to HIP flags.
    // hipHostMalloc and hipHostRegister use the same flag values:
    // hipHostMallocPortable/hipHostRegisterPortable = 0x1
    // hipHostMallocMapped/hipHostRegisterMapped = 0x2
    // hipHostMallocWriteCombined/hipHostRegisterIoMemory = 0x4
    unsigned int hip_flags = 0;
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE) {
      hip_flags |= 0x1;  // hipHostMallocPortable
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED) {
      hip_flags |= 0x2;  // hipHostMallocMapped
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED) {
      hip_flags |= 0x4;  // hipHostMallocWriteCombined
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_UNCACHED) {
      hip_flags |= hipHostMallocUncached;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_NUMA_USER) {
      hip_flags |= hipHostMallocNumaUser;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_COHERENT) {
      hip_flags |= hipHostMallocCoherent;
    }
    if (internal_flags &
        IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_NON_COHERENT) {
      hip_flags |= hipHostMallocNonCoherent;
    }
    *flagsPtr = hip_flags;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets information about a memory pointer.
//
// Parameters:
//  - ptr: [IN] Pointer to query (host or device).
//  - size: [OUT] Pointer to receive allocation size.
//
// Returns:
//  - hipSuccess: Information retrieved successfully.
//  - hipErrorInvalidValue: ptr or size is NULL.
//  - hipErrorInvalidDevicePointer: ptr is not a valid allocation.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Pointer information:
// - Works with device allocations from hipMalloc().
// - Works with host allocations from hipHostMalloc().
// - Returns size of original allocation.
// - ptr can be anywhere within allocation.
//
// Multi-GPU: Queries current device's allocations.
//
// Note: Extended version hipDrvMemGetInfo provides more details.
//
// See also: hipMemGetAddressRange, hipPointerGetAttributes.
HIPAPI hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!ptr || !size) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up the buffer from the pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)ptr, &buffer_ref);

  if (iree_status_is_ok(status)) {
    *size = buffer_ref.buffer->logical_size;
  }

  hipError_t result = hipSuccess;
  if (iree_status_is_ok(status)) {
    result = hipSuccess;
  } else if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
    iree_status_free(status);
    result = hipErrorInvalidValue;
  } else {
    result = iree_status_to_hip_result(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

static hipError_t iree_hip_resolve_memcpy_kind(
    iree_hal_streaming_context_t* context, const void* dst, const void* src,
    hipMemcpyKind kind, hipMemcpyKind* out_kind) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_kind);

  if (kind == hipMemcpyDeviceToDeviceNoCU) {
    *out_kind = hipMemcpyDeviceToDevice;
    return hipSuccess;
  }
  if (kind != hipMemcpyDefault) {
    if (kind == hipMemcpyHostToHost || kind == hipMemcpyHostToDevice ||
        kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice) {
      *out_kind = kind;
      return hipSuccess;
    }
    return hipErrorInvalidValue;
  }

  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t dst_status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)dst, &dst_ref);
  bool dst_is_device = iree_status_is_ok(dst_status);
  if (!dst_is_device && iree_status_code(dst_status) == IREE_STATUS_NOT_FOUND) {
    iree_status_ignore(dst_status);
    iree_hal_streaming_context_t* owner_context = NULL;
    dst_status = iree_hal_streaming_memory_lookup_range_across_contexts(
        (iree_hal_streaming_deviceptr_t)dst, 1, &owner_context, &dst_ref);
    if (iree_status_is_ok(dst_status)) {
      dst_is_device = dst_ref.buffer->is_managed;
      iree_hal_streaming_context_release(owner_context);
    }
  }
  iree_status_ignore(dst_status);

  iree_hal_streaming_buffer_ref_t src_ref;
  iree_status_t src_status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)src, &src_ref);
  bool src_is_device = iree_status_is_ok(src_status);
  if (!src_is_device && iree_status_code(src_status) == IREE_STATUS_NOT_FOUND) {
    iree_status_ignore(src_status);
    iree_hal_streaming_context_t* owner_context = NULL;
    src_status = iree_hal_streaming_memory_lookup_range_across_contexts(
        (iree_hal_streaming_deviceptr_t)src, 1, &owner_context, &src_ref);
    if (iree_status_is_ok(src_status)) {
      src_is_device = src_ref.buffer->is_managed;
      iree_hal_streaming_context_release(owner_context);
    }
  }
  iree_status_ignore(src_status);

  if (dst_is_device && src_is_device) {
    *out_kind = hipMemcpyDeviceToDevice;
  } else if (dst_is_device && !src_is_device) {
    *out_kind = hipMemcpyHostToDevice;
  } else if (!dst_is_device && src_is_device) {
    *out_kind = hipMemcpyDeviceToHost;
  } else {
    *out_kind = hipMemcpyHostToHost;
  }
  return hipSuccess;
}

static hipError_t iree_hip_validate_hip_visible_memcpy_range(
    iree_hal_streaming_buffer_ref_t range_ref, size_t count) {
  if (!range_ref.buffer || count > (size_t)IREE_DEVICE_SIZE_MAX) {
    return hipErrorInvalidValue;
  }
  const iree_device_size_t device_count = (iree_device_size_t)count;
  const iree_device_size_t logical_size = range_ref.buffer->logical_size;
  if (range_ref.offset > logical_size ||
      device_count > logical_size - range_ref.offset) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_validate_known_memcpy_range(
    iree_hal_streaming_context_t* context, const void* ptr, size_t count,
    bool use_hip_visible_pool_size) {
  if (count == 0) return hipSuccess;

  iree_hal_streaming_buffer_ref_t range_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)ptr, count, &range_ref);
  if (iree_status_is_ok(status)) {
    if (use_hip_visible_pool_size) {
      return iree_hip_validate_hip_visible_memcpy_range(range_ref, count);
    }
    return hipSuccess;
  }
  const iree_status_code_t code = iree_status_code(status);
  if (code != IREE_STATUS_NOT_FOUND) {
    return iree_status_to_hip_result(status);
  }
  iree_status_ignore(status);

  iree_hal_streaming_context_t* owner_context = NULL;
  status = iree_hal_streaming_memory_lookup_range_across_contexts(
      (iree_hal_streaming_deviceptr_t)ptr, count, &owner_context, &range_ref);
  if (iree_status_is_ok(status)) {
    hipError_t result = hipSuccess;
    if (use_hip_visible_pool_size) {
      result = iree_hip_validate_hip_visible_memcpy_range(range_ref, count);
    }
    iree_hal_streaming_context_release(owner_context);
    return result;
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return iree_status_to_hip_result(status);
  }
  iree_status_ignore(status);

  iree_hal_streaming_buffer_ref_t buffer_ref;
  status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)ptr, &buffer_ref);
  if (iree_status_is_ok(status)) {
    return hipErrorInvalidValue;
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return iree_status_to_hip_result(status);
  }
  iree_status_ignore(status);

  status = iree_hal_streaming_memory_lookup_range_across_contexts(
      (iree_hal_streaming_deviceptr_t)ptr, 1, &owner_context, &buffer_ref);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    return hipErrorInvalidValue;
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return iree_status_to_hip_result(status);
  }
  iree_status_ignore(status);
  return hipSuccess;
}

static hipError_t iree_hip_validate_memcpy_ranges(
    iree_hal_streaming_context_t* context, const void* dst, const void* src,
    size_t count, hipMemcpyKind kind, bool use_hip_visible_pool_size) {
  hipError_t result = hipSuccess;
  switch (kind) {
    case hipMemcpyHostToDevice:
      return iree_hip_validate_known_memcpy_range(context, dst, count,
                                                  use_hip_visible_pool_size);
    case hipMemcpyDeviceToHost:
      return iree_hip_validate_known_memcpy_range(context, src, count,
                                                  use_hip_visible_pool_size);
    case hipMemcpyDeviceToDevice:
      result = iree_hip_validate_known_memcpy_range(context, dst, count,
                                                    use_hip_visible_pool_size);
      if (result == hipSuccess) {
        result = iree_hip_validate_known_memcpy_range(
            context, src, count, use_hip_visible_pool_size);
      }
      return result;
    case hipMemcpyHostToHost:
      return hipSuccess;
    default:
      return hipErrorInvalidValue;
  }
}

// Returns true when |dst| is not backed by a host allocation tracked in the
// current context. HIP completes D2H copies to such pageable memory before the
// API call returns, while tracked host memory remains stream ordered.
static bool iree_hip_memcpy_destination_is_pageable(
    iree_hal_streaming_context_t* context, const void* dst, size_t size) {
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)(uintptr_t)dst, size, &dst_ref);
  const bool is_pageable = !iree_status_is_ok(status);
  iree_status_ignore(status);
  return is_pageable;
}

static hipError_t iree_hip_memcpy_peer_staged(
    iree_hal_streaming_context_t* dst_context, void* dst,
    iree_hal_streaming_context_t* src_context, const void* src,
    size_t size_bytes, iree_hal_streaming_stream_t* stream);

static hipError_t iree_hip_try_cross_context_h2d(
    iree_hal_streaming_context_t* context, void* dst, const void* src,
    size_t size, bool* out_handled) {
  *out_handled = false;
  iree_hal_streaming_buffer_ref_t current_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)(uintptr_t)dst, size,
      &current_ref);
  if (iree_status_is_ok(status)) return hipSuccess;
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return iree_status_to_hip_result(status);
  }
  iree_status_ignore(status);

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_ref_t dst_ref;
  status = iree_hal_streaming_memory_lookup_range_across_contexts(
      (iree_hal_streaming_deviceptr_t)(uintptr_t)dst, size, &owner_context,
      &dst_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipSuccess;
  }
  *out_handled = true;
  if (dst_ref.buffer->host_ptr &&
      iree_any_bit_set((iree_hal_memory_type_t)dst_ref.buffer->memory_type,
                       IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    iree_status_t sync_status = iree_hal_streaming_context_synchronize_all();
    if (iree_status_is_ok(sync_status)) {
      memcpy((uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset, src, size);
    }
    iree_hal_streaming_context_release(owner_context);
    return iree_status_to_hip_result(sync_status);
  }
  if (dst_ref.buffer->is_managed) {
    if (!dst_ref.buffer->host_ptr) {
      iree_hal_streaming_context_release(owner_context);
      return hipErrorInvalidValue;
    }
    iree_status_t sync_status = iree_hal_streaming_context_synchronize_all();
    if (iree_status_is_ok(sync_status)) {
      memcpy((uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset, src, size);
    }
    iree_hal_streaming_context_release(owner_context);
    return iree_status_to_hip_result(sync_status);
  }

  const uint8_t* src_ptr = (const uint8_t*)src;
  iree_device_size_t remaining = size;
  iree_device_size_t offset = 0;
  iree_status_t transfer_status =
      iree_hal_streaming_context_synchronize(owner_context);
  while (remaining > 0 && iree_status_is_ok(transfer_status)) {
    const iree_device_size_t chunk_size = 4 * 1024 * 1024;
    const iree_device_size_t this_chunk =
        remaining < chunk_size ? remaining : chunk_size;
    transfer_status = iree_hal_device_transfer_h2d(
        owner_context->device, src_ptr + offset, dst_ref.buffer->buffer,
        dst_ref.offset + offset, this_chunk,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    offset += this_chunk;
    remaining -= this_chunk;
  }
  iree_hal_streaming_context_release(owner_context);
  return iree_status_to_hip_result(transfer_status);
}

static hipError_t iree_hip_try_cross_context_d2h(
    iree_hal_streaming_context_t* context, void* dst, const void* src,
    size_t size, bool* out_handled) {
  *out_handled = false;
  iree_hal_streaming_buffer_ref_t current_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)(uintptr_t)src, size,
      &current_ref);
  if (iree_status_is_ok(status)) return hipSuccess;
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return iree_status_to_hip_result(status);
  }
  iree_status_ignore(status);

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_ref_t src_ref;
  status = iree_hal_streaming_memory_lookup_range_across_contexts(
      (iree_hal_streaming_deviceptr_t)(uintptr_t)src, size, &owner_context,
      &src_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipSuccess;
  }
  *out_handled = true;
  if (src_ref.buffer->host_ptr &&
      iree_any_bit_set((iree_hal_memory_type_t)src_ref.buffer->memory_type,
                       IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    iree_status_t sync_status = iree_hal_streaming_context_synchronize_all();
    if (iree_status_is_ok(sync_status)) {
      memcpy(dst, (const uint8_t*)src_ref.buffer->host_ptr + src_ref.offset,
             size);
    }
    iree_hal_streaming_context_release(owner_context);
    return iree_status_to_hip_result(sync_status);
  }
  if (src_ref.buffer->is_managed) {
    if (!src_ref.buffer->host_ptr) {
      iree_hal_streaming_context_release(owner_context);
      return hipErrorInvalidValue;
    }
    iree_status_t sync_status = iree_hal_streaming_context_synchronize_all();
    if (iree_status_is_ok(sync_status)) {
      memcpy(dst, (const uint8_t*)src_ref.buffer->host_ptr + src_ref.offset,
             size);
    }
    iree_hal_streaming_context_release(owner_context);
    return iree_status_to_hip_result(sync_status);
  }

  uint8_t* dst_ptr = (uint8_t*)dst;
  iree_device_size_t remaining = size;
  iree_device_size_t offset = 0;
  iree_status_t transfer_status =
      iree_hal_streaming_context_synchronize(owner_context);
  while (remaining > 0 && iree_status_is_ok(transfer_status)) {
    const iree_device_size_t chunk_size = 4 * 1024 * 1024;
    const iree_device_size_t this_chunk =
        remaining < chunk_size ? remaining : chunk_size;
    transfer_status = iree_hal_device_transfer_d2h(
        owner_context->device, src_ref.buffer->buffer, src_ref.offset + offset,
        dst_ptr + offset, this_chunk, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
    offset += this_chunk;
    remaining -= this_chunk;
  }
  iree_hal_streaming_context_release(owner_context);
  return iree_status_to_hip_result(transfer_status);
}

static hipError_t iree_hip_lookup_memcpy_range_with_owner(
    iree_hal_streaming_context_t* context, const void* ptr, size_t size,
    iree_hal_streaming_context_t** out_owner_context,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  *out_owner_context = NULL;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)(uintptr_t)ptr, size, out_ref);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_context_retain(context);
    *out_owner_context = context;
    return hipSuccess;
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return iree_status_to_hip_result(status);
  }
  iree_status_ignore(status);

  status = iree_hal_streaming_memory_lookup_range_across_contexts(
      (iree_hal_streaming_deviceptr_t)(uintptr_t)ptr, size, out_owner_context,
      out_ref);
  if (!iree_status_is_ok(status)) {
    hipError_t result = iree_status_to_hip_result(status);
    *out_owner_context = NULL;
    return result;
  }
  return hipSuccess;
}

static hipError_t iree_hip_try_managed_d2d(
    iree_hal_streaming_context_t* context, void* dst, const void* src,
    size_t size, bool* out_handled) {
  *out_handled = false;
  iree_hal_streaming_context_t* dst_context = NULL;
  iree_hal_streaming_context_t* src_context = NULL;
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_hal_streaming_buffer_ref_t src_ref;
  hipError_t result = iree_hip_lookup_memcpy_range_with_owner(
      context, dst, size, &dst_context, &dst_ref);
  if (result != hipSuccess) return hipSuccess;
  result = iree_hip_lookup_memcpy_range_with_owner(context, src, size,
                                                   &src_context, &src_ref);
  if (result != hipSuccess) {
    iree_hal_streaming_context_release(dst_context);
    return hipSuccess;
  }

  const bool dst_is_managed = dst_ref.buffer->is_managed;
  const bool src_is_managed = src_ref.buffer->is_managed;
  if (!dst_is_managed && !src_is_managed) {
    iree_hal_streaming_context_release(src_context);
    iree_hal_streaming_context_release(dst_context);
    return hipSuccess;
  }
  *out_handled = true;

  iree_status_t status = iree_ok_status();
  if (dst_context == src_context) {
    status = iree_hal_streaming_context_synchronize(dst_context);
  } else {
    status = iree_hal_streaming_context_synchronize(dst_context);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_context_synchronize(src_context);
    }
  }

  if (iree_status_is_ok(status) && dst_is_managed && src_is_managed) {
    if (!dst_ref.buffer->host_ptr || !src_ref.buffer->host_ptr) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "managed D2D copy requires host backing");
    } else {
      memmove((uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset,
              (const uint8_t*)src_ref.buffer->host_ptr + src_ref.offset, size);
    }
  } else if (iree_status_is_ok(status) && dst_is_managed) {
    if (!dst_ref.buffer->host_ptr || !src_ref.buffer->buffer) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "managed D2D destination copy is not supported");
    } else {
      status = iree_hal_device_transfer_d2h(
          src_context->device, src_ref.buffer->buffer, src_ref.offset,
          (uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset, size,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    }
  } else if (iree_status_is_ok(status) && src_is_managed) {
    if (!src_ref.buffer->host_ptr || !dst_ref.buffer->buffer) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "managed D2D source copy is not supported");
    } else {
      status = iree_hal_device_transfer_h2d(
          dst_context->device,
          (const uint8_t*)src_ref.buffer->host_ptr + src_ref.offset,
          dst_ref.buffer->buffer, dst_ref.offset, size,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    }
  }

  iree_hal_streaming_context_release(src_context);
  iree_hal_streaming_context_release(dst_context);
  return iree_status_to_hip_result(status);
}

// Copies data between host and device.
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - src: [IN] Source pointer (host or device).
//  - sizeBytes: [IN] Number of bytes to copy.
//  - kind: [IN] Type of copy (hipMemcpyHostToHost, hipMemcpyHostToDevice,
//               hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice).
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers, invalid size, or invalid kind.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevicePointer: Device pointer not valid.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until copy
// completes.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// Copy behavior:
// - hipMemcpyHostToHost: CPU memcpy.
// - hipMemcpyHostToDevice: Transfer from host to device.
// - hipMemcpyDeviceToHost: Transfer from device to host.
// - hipMemcpyDeviceToDevice: Copy within or between devices.
// - hipMemcpyDefault: Runtime determines direction from pointer types.
//
// Multi-GPU:
// - Device-to-device copies within same GPU are supported.
// - Cross-device copies require peer access or staging through host.
//
// Performance note: For asynchronous transfers, use hipMemcpyAsync().
//
// See also: hipMemcpyAsync, hipMemcpy2D, hipMemcpyHtoD, hipMemcpyDtoH.
HIPAPI hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                            hipMemcpyKind kind) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (sizeBytes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate pointers.
  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  hipError_t kind_result =
      iree_hip_resolve_memcpy_kind(context, dst, src, kind, &kind);
  if (kind_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(kind_result);
  }
  hipError_t range_result =
      iree_hip_validate_memcpy_ranges(context, dst, src, sizeBytes, kind, true);
  if (range_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(range_result);
  }

  bool handled = false;
  hipError_t special_result = hipSuccess;
  if (kind == hipMemcpyHostToDevice) {
    special_result =
        iree_hip_try_cross_context_h2d(context, dst, src, sizeBytes, &handled);
  } else if (kind == hipMemcpyDeviceToHost) {
    special_result =
        iree_hip_try_cross_context_d2h(context, dst, src, sizeBytes, &handled);
  } else if (kind == hipMemcpyDeviceToDevice) {
    special_result =
        iree_hip_try_managed_d2d(context, dst, src, sizeBytes, &handled);
  }
  if (handled || special_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(special_result);
  }

  iree_status_t status = iree_ok_status();
  switch (kind) {
    case hipMemcpyHostToDevice:
      iree_hal_streaming_context_synchronize(context);
      status = iree_hal_streaming_memcpy_host_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst, src, sizeBytes, NULL);
      break;
    case hipMemcpyDeviceToHost:
      iree_hal_streaming_context_synchronize(context);
      status = iree_hal_streaming_memcpy_device_to_host(
          context, dst, (iree_hal_streaming_deviceptr_t)src, sizeBytes, NULL);
      break;
    case hipMemcpyDeviceToDevice:
      iree_hal_streaming_context_synchronize(context);
      status = iree_hal_streaming_memcpy_device_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst,
          (iree_hal_streaming_deviceptr_t)src, sizeBytes, NULL);
      break;
    case hipMemcpyHostToHost:
      status = iree_hal_streaming_context_synchronize(context);
      if (iree_status_is_ok(status)) {
        memcpy(dst, src, sizeBytes);
      }
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      break;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data between host and device asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - src: [IN] Source pointer (host or device).
//  - sizeBytes: [IN] Number of bytes to copy.
//  - kind: [IN] Type of copy (hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
//               hipMemcpyDeviceToDevice, hipMemcpyDefault).
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers, invalid size, or invalid kind.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevicePointer: Device pointer not valid.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous. Returns immediately after
// enqueueing the copy.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Copy executes after all previously enqueued operations in the stream.
// - Subsequent operations in the stream wait for copy to complete.
// - Graph capture: Supported. Creates memcpy node when capturing.
//
// Memory requirements:
// - Host memory must be pinned for async H2D/D2H transfers.
// - Use hipHostMalloc() or hipHostRegister() to pin memory.
// - Non-pinned memory falls back to synchronous copy.
//
// Multi-GPU:
// - Device-to-device copies within same GPU are supported.
// - Cross-device copies require peer access.
//
// Warning: Host memory must remain valid until copy completes.
//
// See also: hipMemcpy, hipStreamSynchronize, hipHostMalloc,
//           hipMemcpyHtoDAsync.
HIPAPI hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                                 hipMemcpyKind kind, hipStream_t stream) {
  HIP_DEBUG_LOG(
      "[HIP_API] hipMemcpyAsync(dst=%p, src=%p, size=%zu, kind=%d, "
      "stream=%p)\n",
      dst, src, sizeBytes, (int)kind, (void*)stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (sizeBytes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate pointers.
  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Explicit streams carry their owning context and can be used from worker
  // threads that have no current context set.
  iree_hal_streaming_context_t* context = NULL;
  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result = hipSuccess;
  if (stream && stream != hipStreamLegacy && stream != hipStreamPerThread) {
    stream_obj = (iree_hal_streaming_stream_t*)stream;
    context = stream_obj->context;
    if (!context) {
      init_result = hipErrorContextIsDestroyed;
    }
  } else {
    init_result = iree_hip_ensure_context(&context);
    if (init_result == hipSuccess) {
      init_result = iree_hip_resolve_stream(stream, &stream_obj);
    }
  }
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t kind_result =
      iree_hip_resolve_memcpy_kind(context, dst, src, kind, &kind);
  if (kind_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(kind_result);
  }
  hipError_t range_result =
      iree_hip_validate_memcpy_ranges(context, dst, src, sizeBytes, kind, true);
  if (range_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(range_result);
  }

  const bool pageable_d2h =
      kind == hipMemcpyDeviceToHost &&
      stream_obj->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
      iree_hip_memcpy_destination_is_pageable(context, dst, sizeBytes);

  if (!stream || stream == hipStreamLegacy) {
    iree_status_t order_status =
        iree_hal_streaming_context_synchronize_legacy_default(context);
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(z0, order_status);
  }

  if (stream_obj->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    bool handled = false;
    hipError_t special_result = hipSuccess;
    if (kind == hipMemcpyHostToDevice) {
      special_result = iree_hip_try_cross_context_h2d(context, dst, src,
                                                      sizeBytes, &handled);
    } else if (kind == hipMemcpyDeviceToHost) {
      special_result = iree_hip_try_cross_context_d2h(context, dst, src,
                                                      sizeBytes, &handled);
    } else if (kind == hipMemcpyDeviceToDevice) {
      special_result =
          iree_hip_try_managed_d2d(context, dst, src, sizeBytes, &handled);
    }
    if (handled || special_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(special_result);
    }
  }

  hipError_t dependency_result =
      iree_hip_order_legacy_stream_dependencies(context, stream_obj);
  if (dependency_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }

  iree_status_t status = iree_ok_status();
  switch (kind) {
    case hipMemcpyHostToDevice:
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_memcpy_host_to_device(
            context, (iree_hal_streaming_deviceptr_t)dst, src, sizeBytes,
            stream_obj);
      }
      break;
    case hipMemcpyDeviceToHost:
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_memcpy_device_to_host(
            context, dst, (iree_hal_streaming_deviceptr_t)src, sizeBytes,
            stream_obj);
      }
      break;
    case hipMemcpyDeviceToDevice:
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_memcpy_device_to_device(
            context, (iree_hal_streaming_deviceptr_t)dst,
            (iree_hal_streaming_deviceptr_t)src, sizeBytes, stream_obj);
      }
      break;
    case hipMemcpyHostToHost:
      // Host-to-host copies are synchronous.
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_stream_synchronize(stream_obj);
      }
      if (iree_status_is_ok(status)) {
        memcpy(dst, src, sizeBytes);
      }
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      break;
  }
  if (iree_status_is_ok(status) && pageable_d2h) {
    status = iree_hal_streaming_stream_synchronize(stream_obj);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data between host and device with stream (deprecated).
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - src: [IN] Source pointer (host or device).
//  - sizeBytes: [IN] Number of bytes to copy.
//  - kind: [IN] Type of copy (hipMemcpyHostToDevice, etc.).
//  - stream: [IN] Stream for asynchronous execution.
//
// Returns: Same as hipMemcpyAsync.
//
// Synchronization: This operation is asynchronous.
//
// Note: This function is deprecated. Use hipMemcpyAsync instead.
//
// See also: hipMemcpyAsync, hipMemcpy.
HIPAPI hipError_t hipMemcpyWithStream(void* dst, const void* src,
                                      size_t sizeBytes, hipMemcpyKind kind,
                                      hipStream_t stream) {
  hipError_t result = hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  if (result == hipSuccess) {
    result = hipStreamSynchronize(stream);
  }
  return result;
}

static bool iree_hip_calculate_2d_copy_span(size_t pitch, size_t width,
                                            size_t height, size_t* out_span) {
  *out_span = 0;
  if (width == 0 || height == 0) return true;
  return iree_host_size_checked_mul(height - 1, pitch, out_span) &&
         iree_host_size_checked_add(*out_span, width, out_span);
}

static hipError_t iree_hip_validate_2d_copy_shape(size_t dst_pitch,
                                                  size_t src_pitch,
                                                  size_t width, size_t height,
                                                  hipError_t pitch_error) {
  if (width == 0 || height == 0) return hipSuccess;
  if (width > dst_pitch || width > src_pitch) {
    return pitch_error;
  }
  size_t dst_span = 0;
  size_t src_span = 0;
  if (!iree_hip_calculate_2d_copy_span(dst_pitch, width, height, &dst_span) ||
      !iree_hip_calculate_2d_copy_span(src_pitch, width, height, &src_span)) {
    return hipErrorInvalidValue;
  }
  int max_pitch = 0;
  if (hipDeviceGetAttribute(&max_pitch, hipDeviceAttributeMaxPitch, 0) ==
          hipSuccess &&
      max_pitch > 0 &&
      (dst_pitch > (size_t)max_pitch || src_pitch > (size_t)max_pitch)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_capture_memcpy3d_node(
    iree_hal_streaming_stream_t* stream, const hipMemcpy3DParms* params) {
  hipGraphNode_t node = NULL;
  hipError_t result =
      hipGraphAddMemcpyNode(&node, (hipGraph_t)stream->capture_graph,
                            (const hipGraphNode_t*)stream->capture_dependencies,
                            stream->capture_dependency_count, params);
  if (result != hipSuccess) {
    return result;
  }

  iree_hal_streaming_graph_node_t* terminal_node =
      (iree_hal_streaming_graph_node_t*)node;
  iree_hal_streaming_graph_node_t* post_callback =
      iree_hip_graph_find_post_memcpy_callback(terminal_node);
  if (post_callback) {
    terminal_node = post_callback;
  }

  iree_status_t status =
      iree_hal_streaming_capture_set_last_node(stream, terminal_node);
  if (!iree_status_is_ok(status)) {
    result = iree_status_to_hip_result(status);
  }
  return result;
}

static hipError_t iree_hip_memcpy2d_to_3d_params(const hip_Memcpy2D* copy,
                                                 hipError_t pitch_error,
                                                 hipMemcpy3DParms* out_params) {
  if (!copy || !out_params) return hipErrorInvalidValue;
  memset(out_params, 0, sizeof(*out_params));

  if (copy->srcMemoryType == hipMemoryTypeArray ||
      copy->dstMemoryType == hipMemoryTypeArray) {
    return iree_hip_memcpy2d_array_to_3d_params(copy, out_params);
  }

  hipError_t shape_result = iree_hip_validate_2d_copy_shape(
      copy->dstPitch, copy->srcPitch, copy->WidthInBytes, copy->Height,
      pitch_error);
  if (shape_result != hipSuccess) return shape_result;

  iree_host_size_t src_width = 0;
  iree_host_size_t dst_width = 0;
  iree_host_size_t src_height = 0;
  iree_host_size_t dst_height = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(copy->srcXInBytes, copy->WidthInBytes,
                                      &src_width) ||
          !iree_host_size_checked_add(copy->dstXInBytes, copy->WidthInBytes,
                                      &dst_width) ||
          !iree_host_size_checked_add(copy->srcY, copy->Height, &src_height) ||
          !iree_host_size_checked_add(copy->dstY, copy->Height, &dst_height))) {
    return hipErrorInvalidValue;
  }
  if (copy->WidthInBytes != 0 &&
      (src_width > copy->srcPitch || dst_width > copy->dstPitch)) {
    return hipErrorInvalidValue;
  }

  const void* src = NULL;
  switch (copy->srcMemoryType) {
    case hipMemoryTypeHost:
      src = copy->srcHost;
      break;
    case hipMemoryTypeDevice:
    case hipMemoryTypeUnified:
      src = copy->srcDevice;
      break;
    default:
      return hipErrorInvalidValue;
  }

  void* dst = NULL;
  switch (copy->dstMemoryType) {
    case hipMemoryTypeHost:
      dst = copy->dstHost;
      break;
    case hipMemoryTypeDevice:
    case hipMemoryTypeUnified:
      dst = copy->dstDevice;
      break;
    default:
      return hipErrorInvalidValue;
  }
  if (!dst || !src) return hipErrorInvalidValue;

  if (copy->srcMemoryType == hipMemoryTypeUnified ||
      copy->dstMemoryType == hipMemoryTypeUnified) {
    out_params->kind = hipMemcpyDefault;
  } else if (copy->srcMemoryType == hipMemoryTypeHost &&
             copy->dstMemoryType == hipMemoryTypeHost) {
    out_params->kind = hipMemcpyHostToHost;
  } else if (copy->srcMemoryType == hipMemoryTypeHost &&
             copy->dstMemoryType == hipMemoryTypeDevice) {
    out_params->kind = hipMemcpyHostToDevice;
  } else if (copy->srcMemoryType == hipMemoryTypeDevice &&
             copy->dstMemoryType == hipMemoryTypeHost) {
    out_params->kind = hipMemcpyDeviceToHost;
  } else if (copy->srcMemoryType == hipMemoryTypeDevice &&
             copy->dstMemoryType == hipMemoryTypeDevice) {
    out_params->kind = hipMemcpyDeviceToDevice;
  } else {
    return hipErrorInvalidValue;
  }

  out_params->srcPtr.ptr = (void*)src;
  out_params->srcPtr.pitch = copy->srcPitch;
  out_params->srcPtr.xsize = src_width;
  out_params->srcPtr.ysize = src_height;
  out_params->srcPos.x = copy->srcXInBytes;
  out_params->srcPos.y = copy->srcY;
  out_params->dstPtr.ptr = dst;
  out_params->dstPtr.pitch = copy->dstPitch;
  out_params->dstPtr.xsize = dst_width;
  out_params->dstPtr.ysize = dst_height;
  out_params->dstPos.x = copy->dstXInBytes;
  out_params->dstPos.y = copy->dstY;
  out_params->extent.width = copy->WidthInBytes;
  out_params->extent.height = copy->Height;
  out_params->extent.depth = 1;
  return hipSuccess;
}

// Copies 2D pitched data between host and device asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - dpitch: [IN] Pitch (stride in bytes) of destination memory.
//  - src: [IN] Source pointer (host or device).
//  - spitch: [IN] Pitch (stride in bytes) of source memory.
//  - width: [IN] Width of the 2D copy region in bytes.
//  - height: [IN] Height of the 2D copy region (number of rows).
//  - kind: [IN] Type of copy (hipMemcpyHostToDevice, etc.).
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy was successfully queued.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidPitchValue: Invalid pitch values.
//  - hipErrorInvalidMemcpyDirection: Invalid copy direction.
//
// Synchronization: This operation is asynchronous on the specified stream.
//
// Threading: Thread-safe.
//
// Notes:
// - The copy region is defined as a rectangle of `height` rows, each `width`
//   bytes wide.
// - For each row, `width` bytes are copied; remaining bytes up to the pitch
//   are not accessed.
// - Host memory should be pinned for best async performance.
//
// See also: hipMemcpy2D, hipMemcpyAsync, hipMallocPitch.
HIPAPI hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                                   size_t spitch, size_t width, size_t height,
                                   hipMemcpyKind kind, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate pointers.
  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Validate pitch values - width cannot exceed pitch.
  if (width > dpitch || width > spitch) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidPitchValue);
  }

  // Zero-size copy is a no-op.
  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  init_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t shape_result = iree_hip_validate_2d_copy_shape(
      dpitch, spitch, width, height, hipErrorInvalidPitchValue);
  if (shape_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(shape_result);
  }

  const hipMemcpyKind requested_kind = kind;
  hipError_t kind_result =
      iree_hip_resolve_memcpy_kind(context, dst, src, kind, &kind);
  if (kind_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(kind_result);
  }

  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    hipMemcpy3DParms params;
    memset(&params, 0, sizeof(params));
    params.srcPtr.ptr = (void*)src;
    params.srcPtr.pitch = spitch;
    params.srcPtr.xsize = width;
    params.srcPtr.ysize = height;
    params.dstPtr.ptr = dst;
    params.dstPtr.pitch = dpitch;
    params.dstPtr.xsize = width;
    params.dstPtr.ysize = height;
    params.extent.width = width;
    params.extent.height = height;
    params.extent.depth = 1;
    params.kind = requested_kind;

    hipError_t result = iree_hip_capture_memcpy3d_node(stream_obj, &params);
    IREE_TRACE_ZONE_END(z0);
    if (result != hipSuccess) {
      HIP_RETURN_ERROR(result);
    }
    return hipSuccess;
  }

  size_t dst_span = 0;
  if (!iree_hip_calculate_2d_copy_span(dpitch, width, height, &dst_span)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const bool pageable_d2h =
      kind == hipMemcpyDeviceToHost &&
      iree_hip_memcpy_destination_is_pageable(context, dst, dst_span);

  if (kind == hipMemcpyDeviceToHost) {
    iree_status_t packed_status = iree_hal_streaming_memcpy_device_to_host_2d(
        context, dst, dpitch, (iree_hal_streaming_deviceptr_t)src, spitch,
        width, height, stream_obj);
    if (!iree_status_is_unavailable(packed_status)) {
      if (iree_status_is_ok(packed_status) && pageable_d2h) {
        packed_status = iree_hal_streaming_stream_synchronize(stream_obj);
      }
      hipError_t result = iree_status_to_hip_result(packed_status);
      IREE_TRACE_ZONE_END(z0);
      return result;
    }
    iree_status_ignore(packed_status);
  }

  // Copy row by row.
  iree_status_t status = iree_ok_status();
  if (kind == hipMemcpyHostToHost) {
    status = iree_hal_streaming_stream_synchronize(stream_obj);
  }
  const uint8_t* src_ptr = (const uint8_t*)src;
  uint8_t* dst_ptr = (uint8_t*)dst;

  for (size_t row = 0; row < height && iree_status_is_ok(status); ++row) {
    switch (kind) {
      case hipMemcpyHostToDevice:
        status = iree_hal_streaming_memcpy_host_to_device(
            context, (iree_hal_streaming_deviceptr_t)dst_ptr, src_ptr, width,
            stream_obj);
        break;
      case hipMemcpyDeviceToHost:
        status = iree_hal_streaming_memcpy_device_to_host(
            context, dst_ptr, (iree_hal_streaming_deviceptr_t)src_ptr, width,
            stream_obj);
        break;
      case hipMemcpyDeviceToDevice:
        status = iree_hal_streaming_memcpy_device_to_device(
            context, (iree_hal_streaming_deviceptr_t)dst_ptr,
            (iree_hal_streaming_deviceptr_t)src_ptr, width, stream_obj);
        break;
      case hipMemcpyHostToHost:
        memcpy(dst_ptr, src_ptr, width);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid memory copy direction");
        break;
    }
    src_ptr += spitch;
    dst_ptr += dpitch;
  }
  if (iree_status_is_ok(status) && pageable_d2h) {
    status = iree_hal_streaming_stream_synchronize(stream_obj);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies 2D pitched data between host and device synchronously.
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - dpitch: [IN] Pitch (stride in bytes) of destination memory.
//  - src: [IN] Source pointer (host or device).
//  - spitch: [IN] Pitch (stride in bytes) of source memory.
//  - width: [IN] Width of the 2D copy region in bytes.
//  - height: [IN] Height of the 2D copy region (number of rows).
//  - kind: [IN] Type of copy (hipMemcpyHostToDevice, etc.).
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidPitchValue: Invalid pitch values.
//  - hipErrorInvalidMemcpyDirection: Invalid copy direction.
//
// Synchronization: This operation is synchronous - it blocks until complete.
//
// Threading: Thread-safe.
//
// Notes:
// - The copy region is defined as a rectangle of `height` rows, each `width`
//   bytes wide.
// - For each row, `width` bytes are copied; remaining bytes up to the pitch
//   are not accessed.
//
// See also: hipMemcpy2DAsync, hipMemcpy, hipMallocPitch.
HIPAPI hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src,
                              size_t spitch, size_t width, size_t height,
                              hipMemcpyKind kind) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t shape_result = iree_hip_validate_2d_copy_shape(
      dpitch, spitch, width, height, hipErrorInvalidPitchValue);
  if (shape_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(shape_result);
  }
  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  // Call async version with NULL stream (default stream).
  hipError_t result =
      hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, NULL);

  if (result == hipSuccess) {
    // Synchronize to make it blocking.
    result = hipDeviceSynchronize();
  }

  IREE_TRACE_ZONE_END(z0);
  iree_hip_thread_error_set(result, false);
  return result;
}

static hipError_t iree_hip_resolve_memcpy3d_array_params(
    const hipMemcpy3DParms* params, hipMemcpy3DParms* out_params);

static hipError_t iree_hip_graph_memcpy3d_span_bytes(
    size_t pitch, size_t ysize, size_t width, size_t height, size_t depth,
    iree_host_size_t* out_span);

static hipError_t iree_hip_memcpy3d_staged_rows(
    iree_hal_streaming_context_t* dst_context, void* dst,
    iree_hal_streaming_context_t* src_context, const void* src, size_t width,
    size_t height, size_t depth, size_t dst_pitch, size_t src_pitch,
    size_t dst_slice_pitch, size_t src_slice_pitch,
    iree_hal_streaming_stream_t* stream) {
  if (width == 0 || height == 0 || depth == 0) return hipSuccess;
  if (!dst || !src) return hipErrorInvalidValue;
  if (!dst_context && !src_context) return hipErrorInvalidValue;

  iree_status_t status = iree_ok_status();
  uint8_t* dst_base = (uint8_t*)dst;
  const uint8_t* src_base = (const uint8_t*)src;
  if (dst_context && src_context) {
    uint8_t* staging = (uint8_t*)malloc(width);
    if (!staging) return hipErrorOutOfMemory;
    for (size_t z = 0; z < depth && iree_status_is_ok(status); ++z) {
      uint8_t* dst_slice = dst_base + z * dst_slice_pitch;
      const uint8_t* src_slice = src_base + z * src_slice_pitch;
      for (size_t y = 0; y < height && iree_status_is_ok(status); ++y) {
        status = iree_hal_streaming_memcpy_device_to_host(
            src_context, staging,
            (iree_hal_streaming_deviceptr_t)(src_slice + y * src_pitch), width,
            NULL);
        if (iree_status_is_ok(status)) {
          status = iree_hal_streaming_memcpy_host_to_device(
              dst_context,
              (iree_hal_streaming_deviceptr_t)(dst_slice + y * dst_pitch),
              staging, width, NULL);
        }
      }
    }
    free(staging);
    return iree_status_to_hip_result(status);
  }

  if (src_context) {
    // A stream can only record commands for buffers owned by its HAL device.
    // Cross-device copies may be requested while another device is current,
    // making the implicit stream foreign to the source allocation. Preserve
    // ordering with that stream, then use blocking transfers on the source
    // device instead of recording an invalid cross-device command buffer.
    if (stream->context != src_context) {
      status = iree_hal_streaming_stream_synchronize(stream);
      for (size_t z = 0; z < depth && iree_status_is_ok(status); ++z) {
        uint8_t* dst_slice = dst_base + z * dst_slice_pitch;
        const uint8_t* src_slice = src_base + z * src_slice_pitch;
        for (size_t y = 0; y < height && iree_status_is_ok(status); ++y) {
          status = iree_hal_streaming_memcpy_device_to_host(
              src_context, dst_slice + y * dst_pitch,
              (iree_hal_streaming_deviceptr_t)(src_slice + y * src_pitch),
              width, NULL);
        }
      }
      return iree_status_to_hip_result(status);
    }
    for (size_t z = 0; z < depth && iree_status_is_ok(status); ++z) {
      status = iree_hal_streaming_memcpy_device_to_host_2d(
          src_context, dst_base + z * dst_slice_pitch, dst_pitch,
          (iree_hal_streaming_deviceptr_t)(src_base + z * src_slice_pitch),
          src_pitch, width, height, stream);
    }
    return iree_status_to_hip_result(status);
  }

  if (stream->context != dst_context) {
    status = iree_hal_streaming_stream_synchronize(stream);
    stream = NULL;
  }
  for (size_t z = 0; z < depth && iree_status_is_ok(status); ++z) {
    uint8_t* dst_slice = dst_base + z * dst_slice_pitch;
    const uint8_t* src_slice = src_base + z * src_slice_pitch;
    for (size_t y = 0; y < height && iree_status_is_ok(status); ++y) {
      status = iree_hal_streaming_memcpy_host_to_device(
          dst_context,
          (iree_hal_streaming_deviceptr_t)(dst_slice + y * dst_pitch),
          src_slice + y * src_pitch, width, stream);
    }
  }
  return iree_status_to_hip_result(status);
}

static bool iree_hip_memcpy3d_is_device_to_device(
    const hipMemcpy3DParms* params) {
  switch (params->kind) {
    case hipMemcpyDeviceToDevice:
    case hipMemcpyDeviceToDeviceNoCU:
      return true;
    case hipMemcpyDefault:
      break;
    default:
      return false;
  }

  iree_hal_streaming_context_t* dst_context = NULL;
  iree_hal_streaming_buffer_ref_t dst_ref = {0};
  iree_status_t dst_status =
      iree_hal_streaming_memory_lookup_range_across_contexts(
          (iree_hal_streaming_deviceptr_t)params->dstPtr.ptr, 1, &dst_context,
          &dst_ref);
  iree_hal_streaming_context_t* src_context = NULL;
  iree_hal_streaming_buffer_ref_t src_ref = {0};
  iree_status_t src_status =
      iree_hal_streaming_memory_lookup_range_across_contexts(
          (iree_hal_streaming_deviceptr_t)params->srcPtr.ptr, 1, &src_context,
          &src_ref);
  const bool result =
      iree_status_is_ok(dst_status) && iree_status_is_ok(src_status) &&
      (dst_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      (src_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  if (iree_status_is_ok(dst_status)) {
    iree_hal_streaming_context_release(dst_context);
  } else {
    iree_status_ignore(dst_status);
  }
  if (iree_status_is_ok(src_status)) {
    iree_hal_streaming_context_release(src_context);
  } else {
    iree_status_ignore(src_status);
  }
  return result;
}

HIPAPI hipError_t hipMemcpy3D(const hipMemcpy3DParms* p) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!p) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemcpy3DParms resolved_params;
  hipError_t resolve_result =
      iree_hip_resolve_memcpy3d_array_params(p, &resolved_params);
  if (resolve_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(resolve_result);
  }
  p = &resolved_params;
  if (p->extent.width == 0 || p->extent.height == 0 || p->extent.depth == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  if (!p->srcPtr.ptr || !p->dstPtr.ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  const bool device_to_device = iree_hip_memcpy3d_is_device_to_device(p);
  hipError_t result = hipMemcpy3DAsync(p, NULL);
  if (result == hipSuccess && !device_to_device) {
    result = hipDeviceSynchronize();
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

HIPAPI hipError_t hipMemcpy3DAsync(const hipMemcpy3DParms* p,
                                   hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!p) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipMemcpy3DParms* original_params = p;
  hipMemcpy3DParms resolved_params;
  hipError_t resolve_result =
      iree_hip_resolve_memcpy3d_array_params(p, &resolved_params);
  if (resolve_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(resolve_result);
  }
  p = &resolved_params;
  if (p->extent.width == 0 || p->extent.height == 0 || p->extent.depth == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  if (!p->srcPtr.ptr || !p->dstPtr.ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (p->extent.width > p->srcPtr.pitch || p->extent.width > p->dstPtr.pitch ||
      p->srcPos.x > p->srcPtr.pitch || p->dstPos.x > p->dstPtr.pitch ||
      p->extent.width > p->srcPtr.pitch - p->srcPos.x ||
      p->extent.width > p->dstPtr.pitch - p->dstPos.x) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidPitchValue);
  }
  int max_pitch = 0;
  if (hipDeviceGetAttribute(&max_pitch, hipDeviceAttributeMaxPitch, 0) ==
          hipSuccess &&
      max_pitch > 0 &&
      (p->srcPtr.pitch >= (size_t)max_pitch ||
       p->dstPtr.pitch >= (size_t)max_pitch)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if ((p->srcPtr.ysize != 0 &&
       (p->srcPos.y >= p->srcPtr.ysize ||
        p->extent.height > p->srcPtr.ysize - p->srcPos.y)) ||
      (p->dstPtr.ysize != 0 &&
       (p->dstPos.y >= p->dstPtr.ysize ||
        p->extent.height > p->dstPtr.ysize - p->dstPos.y))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_host_size_t src_rows_per_slice = p->srcPtr.ysize;
  iree_host_size_t dst_rows_per_slice = p->dstPtr.ysize;
  iree_host_size_t src_slice_pitch = 0;
  iree_host_size_t dst_slice_pitch = 0;
  iree_host_size_t src_base_offset = 0;
  iree_host_size_t dst_base_offset = 0;
  iree_host_size_t src_row_offset = 0;
  iree_host_size_t dst_row_offset = 0;
  iree_host_size_t src_z_offset = 0;
  iree_host_size_t dst_z_offset = 0;
  if (IREE_UNLIKELY(
          (src_rows_per_slice == 0 &&
           !iree_host_size_checked_add(p->srcPos.y, p->extent.height,
                                       &src_rows_per_slice)) ||
          (dst_rows_per_slice == 0 &&
           !iree_host_size_checked_add(p->dstPos.y, p->extent.height,
                                       &dst_rows_per_slice)) ||
          !iree_host_size_checked_mul(p->srcPtr.pitch, src_rows_per_slice,
                                      &src_slice_pitch) ||
          !iree_host_size_checked_mul(p->dstPtr.pitch, dst_rows_per_slice,
                                      &dst_slice_pitch) ||
          !iree_host_size_checked_mul(p->srcPos.y, p->srcPtr.pitch,
                                      &src_row_offset) ||
          !iree_host_size_checked_mul(p->dstPos.y, p->dstPtr.pitch,
                                      &dst_row_offset) ||
          !iree_host_size_checked_mul(p->srcPos.z, src_slice_pitch,
                                      &src_z_offset) ||
          !iree_host_size_checked_mul(p->dstPos.z, dst_slice_pitch,
                                      &dst_z_offset) ||
          !iree_host_size_checked_add(src_z_offset, src_row_offset,
                                      &src_base_offset) ||
          !iree_host_size_checked_add(dst_z_offset, dst_row_offset,
                                      &dst_base_offset) ||
          !iree_host_size_checked_add(src_base_offset, p->srcPos.x,
                                      &src_base_offset) ||
          !iree_host_size_checked_add(dst_base_offset, p->dstPos.x,
                                      &dst_base_offset))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  const uint8_t* src_base = (const uint8_t*)p->srcPtr.ptr + src_base_offset;
  uint8_t* dst_base = (uint8_t*)p->dstPtr.ptr + dst_base_offset;

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t stream_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (stream_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(stream_result);
  }
  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    hipError_t result =
        iree_hip_capture_memcpy3d_node(stream_obj, original_params);
    IREE_TRACE_ZONE_END(z0);
    if (result != hipSuccess) {
      HIP_RETURN_ERROR(result);
    }
    return hipSuccess;
  }

  iree_host_size_t dst_span = 0;
  hipError_t span_result = iree_hip_graph_memcpy3d_span_bytes(
      p->dstPtr.pitch, dst_rows_per_slice, p->extent.width, p->extent.height,
      p->extent.depth, &dst_span);
  if (span_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(span_result);
  }
  iree_host_size_t src_span = 0;
  span_result = iree_hip_graph_memcpy3d_span_bytes(
      p->srcPtr.pitch, src_rows_per_slice, p->extent.width, p->extent.height,
      p->extent.depth, &src_span);
  if (span_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(span_result);
  }

  iree_hal_streaming_context_t* dst_context = NULL;
  iree_hal_streaming_buffer_ref_t dst_ref = {0};
  iree_status_t dst_status =
      iree_hal_streaming_memory_lookup_range_across_contexts(
          (iree_hal_streaming_deviceptr_t)dst_base, dst_span, &dst_context,
          &dst_ref);
  iree_hal_streaming_context_t* src_context = NULL;
  iree_hal_streaming_buffer_ref_t src_ref = {0};
  iree_status_t src_status =
      iree_hal_streaming_memory_lookup_range_across_contexts(
          (iree_hal_streaming_deviceptr_t)src_base, src_span, &src_context,
          &src_ref);
  const bool have_dst = iree_status_is_ok(dst_status);
  const bool have_src = iree_status_is_ok(src_status);
  const bool dst_is_device = have_dst && (dst_ref.buffer->memory_type &
                                          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  const bool src_is_device = have_src && (src_ref.buffer->memory_type &
                                          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  hipMemcpyKind effective_kind = p->kind;
  if (effective_kind == hipMemcpyDefault) {
    if (dst_is_device && src_is_device) {
      effective_kind = hipMemcpyDeviceToDevice;
    } else if (dst_is_device) {
      effective_kind = hipMemcpyHostToDevice;
    } else if (src_is_device) {
      effective_kind = hipMemcpyDeviceToHost;
    } else {
      effective_kind = hipMemcpyHostToHost;
    }
  }
  hipError_t result = hipSuccess;
  bool return_after_lookup = false;
  if ((effective_kind == hipMemcpyHostToDevice && !dst_is_device) ||
      (effective_kind == hipMemcpyDeviceToHost && !src_is_device) ||
      ((effective_kind == hipMemcpyDeviceToDevice ||
        effective_kind == hipMemcpyDeviceToDeviceNoCU) &&
       (!dst_is_device || !src_is_device))) {
    result = hipErrorInvalidValue;
    return_after_lookup = true;
  }

  const bool src_is_contiguous = p->srcPos.x == 0 && p->srcPos.y == 0 &&
                                 p->extent.width == p->srcPtr.pitch &&
                                 p->extent.height == src_rows_per_slice;
  const bool dst_is_contiguous = p->dstPos.x == 0 && p->dstPos.y == 0 &&
                                 p->extent.width == p->dstPtr.pitch &&
                                 p->extent.height == dst_rows_per_slice;
  const bool is_cross_context_device_copy =
      (effective_kind == hipMemcpyDeviceToDevice ||
       effective_kind == hipMemcpyDeviceToDeviceNoCU) &&
      dst_is_device && src_is_device && dst_context != src_context;
  if (!return_after_lookup && !is_cross_context_device_copy &&
      src_is_contiguous && dst_is_contiguous) {
    iree_host_size_t byte_count = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            src_slice_pitch, p->extent.depth, &byte_count))) {
      result = hipErrorInvalidValue;
    } else {
      result = hipMemcpyAsync(dst_base, src_base, byte_count, effective_kind,
                              stream);
    }
    return_after_lookup = true;
  }

  if (!return_after_lookup) {
    hipError_t staged_result = hipSuccess;
    bool use_staged_copy = false;
    switch (effective_kind) {
      case hipMemcpyDeviceToHost:
        use_staged_copy = src_is_device;
        if (use_staged_copy) {
          staged_result = iree_hip_memcpy3d_staged_rows(
              NULL, dst_base, src_context, src_base, p->extent.width,
              p->extent.height, p->extent.depth, p->dstPtr.pitch,
              p->srcPtr.pitch, dst_slice_pitch, src_slice_pitch, stream_obj);
        }
        break;
      case hipMemcpyHostToDevice:
        use_staged_copy = dst_is_device;
        if (use_staged_copy) {
          staged_result = iree_hip_memcpy3d_staged_rows(
              dst_context, dst_base, NULL, src_base, p->extent.width,
              p->extent.height, p->extent.depth, p->dstPtr.pitch,
              p->srcPtr.pitch, dst_slice_pitch, src_slice_pitch, stream_obj);
        }
        break;
      case hipMemcpyDeviceToDevice:
      case hipMemcpyDeviceToDeviceNoCU:
        use_staged_copy = dst_is_device && src_is_device && dst_context &&
                          src_context && dst_context != src_context;
        if (use_staged_copy) {
          staged_result = iree_hip_memcpy3d_staged_rows(
              dst_context, dst_base, src_context, src_base, p->extent.width,
              p->extent.height, p->extent.depth, p->dstPtr.pitch,
              p->srcPtr.pitch, dst_slice_pitch, src_slice_pitch, stream_obj);
        }
        break;
      default:
        break;
    }
    if (use_staged_copy) {
      result = staged_result;
      // Strided copies queue one transfer per row. Complete the pageable D2H
      // operation only after its final row is queued on the source stream.
      if (result == hipSuccess && effective_kind == hipMemcpyDeviceToHost &&
          !have_dst && stream_obj->context == src_context) {
        result = iree_status_to_hip_result(
            iree_hal_streaming_stream_synchronize(stream_obj));
      }
      return_after_lookup = true;
    }
  }

  if (have_dst) {
    iree_hal_streaming_context_release(dst_context);
  }
  iree_status_ignore(dst_status);
  if (have_src) {
    iree_hal_streaming_context_release(src_context);
  }
  iree_status_ignore(src_status);
  if (return_after_lookup) {
    IREE_TRACE_ZONE_END(z0);
    if (result != hipSuccess) {
      HIP_RETURN_ERROR(result);
    }
    return hipSuccess;
  }

  for (size_t z = 0; z < p->extent.depth; ++z) {
    iree_host_size_t src_offset = 0;
    iree_host_size_t dst_offset = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(z, src_slice_pitch, &src_offset) ||
            !iree_host_size_checked_mul(z, dst_slice_pitch, &dst_offset))) {
      result = hipErrorInvalidValue;
      break;
    }
    result = hipMemcpy2DAsync(dst_base + dst_offset, p->dstPtr.pitch,
                              src_base + src_offset, p->srcPtr.pitch,
                              p->extent.width, p->extent.height, effective_kind,
                              stream);
    if (result != hipSuccess) break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

HIPAPI hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hipMemcpy3DParms params;
  hipError_t result =
      iree_hip_memcpy2d_to_3d_params(pCopy, hipErrorInvalidValue, &params);
  if (result == hipSuccess) {
    result = hipMemcpy3D(&params);
  }
  IREE_TRACE_ZONE_END(z0);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy,
                                        hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hipMemcpy3DParms params;
  hipError_t result =
      iree_hip_memcpy2d_to_3d_params(pCopy, hipErrorInvalidValue, &params);
  if (result == hipSuccess) {
    result = hipMemcpy3DAsync(&params, stream);
  }
  IREE_TRACE_ZONE_END(z0);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                 hipChannelFormatKind f) {
  hipChannelFormatDesc desc = {
      .x = x,
      .y = y,
      .z = z,
      .w = w,
      .f = f,
  };
  return desc;
}

static bool iree_hip_array_channel_bits_supported(hipChannelFormatKind kind,
                                                  int bits) {
  switch (kind) {
    case hipChannelFormatKindSigned:
    case hipChannelFormatKindUnsigned:
      return bits == 8 || bits == 16 || bits == 32;
    case hipChannelFormatKindFloat:
      return bits == 16 || bits == 32;
    default:
      return false;
  }
}

static hipError_t iree_hip_array_desc_to_format(
    const hipChannelFormatDesc* desc, hipArray_Format* out_format,
    unsigned int* out_num_channels, size_t* out_element_size) {
  if (out_format) *out_format = 0;
  if (out_num_channels) *out_num_channels = 0;
  if (out_element_size) *out_element_size = 0;
  if (!desc || desc->x <= 0 || desc->y < 0 || desc->z < 0 || desc->w < 0) {
    return hipErrorInvalidValue;
  }
  if (!iree_hip_array_channel_bits_supported(desc->f, desc->x)) {
    return hipErrorInvalidValue;
  }

  const int bits[4] = {desc->x, desc->y, desc->z, desc->w};
  unsigned int channel_count = 0;
  bool saw_zero_channel = false;
  for (size_t i = 0; i < IREE_ARRAYSIZE(bits); ++i) {
    if (bits[i] == 0) {
      saw_zero_channel = true;
      continue;
    }
    if (saw_zero_channel || bits[i] != desc->x) {
      return hipErrorInvalidValue;
    }
    ++channel_count;
  }
  if (channel_count != 1 && channel_count != 2 && channel_count != 4) {
    return hipErrorInvalidValue;
  }

  hipArray_Format format = 0;
  switch (desc->f) {
    case hipChannelFormatKindUnsigned:
      format = desc->x == 8    ? HIP_AD_FORMAT_UNSIGNED_INT8
               : desc->x == 16 ? HIP_AD_FORMAT_UNSIGNED_INT16
                               : HIP_AD_FORMAT_UNSIGNED_INT32;
      break;
    case hipChannelFormatKindSigned:
      format = desc->x == 8    ? HIP_AD_FORMAT_SIGNED_INT8
               : desc->x == 16 ? HIP_AD_FORMAT_SIGNED_INT16
                               : HIP_AD_FORMAT_SIGNED_INT32;
      break;
    case hipChannelFormatKindFloat:
      format = desc->x == 16 ? HIP_AD_FORMAT_HALF : HIP_AD_FORMAT_FLOAT;
      break;
    default:
      return hipErrorInvalidValue;
  }

  if (out_format) *out_format = format;
  if (out_num_channels) *out_num_channels = channel_count;
  if (out_element_size) {
    *out_element_size = ((size_t)desc->x * channel_count) / 8;
  }
  return hipSuccess;
}

static hipError_t iree_hip_array_format_to_desc(hipArray_Format format,
                                                unsigned int num_channels,
                                                hipChannelFormatDesc* out_desc,
                                                size_t* out_element_size) {
  if (out_desc) memset(out_desc, 0, sizeof(*out_desc));
  if (out_element_size) *out_element_size = 0;
  if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
    return hipErrorInvalidValue;
  }

  int bits = 0;
  hipChannelFormatKind kind = hipChannelFormatKindNone;
  switch (format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      bits = 8;
      kind = hipChannelFormatKindUnsigned;
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      bits = 16;
      kind = hipChannelFormatKindUnsigned;
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      bits = 32;
      kind = hipChannelFormatKindUnsigned;
      break;
    case HIP_AD_FORMAT_SIGNED_INT8:
      bits = 8;
      kind = hipChannelFormatKindSigned;
      break;
    case HIP_AD_FORMAT_SIGNED_INT16:
      bits = 16;
      kind = hipChannelFormatKindSigned;
      break;
    case HIP_AD_FORMAT_SIGNED_INT32:
      bits = 32;
      kind = hipChannelFormatKindSigned;
      break;
    case HIP_AD_FORMAT_HALF:
      bits = 16;
      kind = hipChannelFormatKindFloat;
      break;
    case HIP_AD_FORMAT_FLOAT:
      bits = 32;
      kind = hipChannelFormatKindFloat;
      break;
    default:
      return hipErrorInvalidValue;
  }

  hipChannelFormatDesc desc = {
      .x = num_channels >= 1 ? bits : 0,
      .y = num_channels >= 2 ? bits : 0,
      .z = num_channels >= 4 ? bits : 0,
      .w = num_channels >= 4 ? bits : 0,
      .f = kind,
  };
  if (out_desc) *out_desc = desc;
  if (out_element_size) *out_element_size = ((size_t)bits * num_channels) / 8;
  return hipSuccess;
}

static void iree_hip_array_destroy(struct hipArray_st* array) {
  iree_hal_streaming_context_t* context = array->context;
  const iree_hal_streaming_deviceptr_t device_ptr = array->device_ptr;
  array->magic = 0;
  array->context = NULL;
  array->buffer = NULL;
  array->device_ptr = 0;
  if (context && device_ptr) {
    iree_status_ignore(
        iree_hal_streaming_memory_free_device(context, device_ptr));
  }
  iree_hal_streaming_context_release(context);
  free(array);
}

static void iree_hip_array_release(struct hipArray_st* array) {
  if (array && iree_atomic_ref_count_dec(&array->ref_count) == 1) {
    iree_hip_array_destroy(array);
  }
}

static hipError_t iree_hip_array_retain(hipArray_const_t array,
                                        struct hipArray_st** out) {
  if (out) *out = NULL;
  struct hipArray_st* array_info = NULL;
  if (!iree_hip_array_registry_lookup(array, &array_info) ||
      !array_info->context || !array_info->buffer || !array_info->device_ptr) {
    iree_hip_array_release(array_info);
    return hipErrorInvalidValue;
  }
  if (out) *out = array_info;
  return hipSuccess;
}

static hipError_t iree_hip_array_byte_range_to_elements_for_array(
    const struct hipArray_st* array_info, size_t byte_offset, size_t byte_width,
    size_t* out_element_offset, size_t* out_element_width) {
  if (out_element_offset) *out_element_offset = 0;
  if (out_element_width) *out_element_width = 0;
  if (!array_info) return hipErrorInvalidValue;
  if (array_info->element_size == 0 ||
      byte_offset % array_info->element_size != 0 ||
      byte_width % array_info->element_size != 0) {
    return hipErrorInvalidValue;
  }
  const size_t element_offset = byte_offset / array_info->element_size;
  const size_t element_width = byte_width / array_info->element_size;
  if (element_offset > array_info->extent.width ||
      element_width > array_info->extent.width - element_offset) {
    return hipErrorInvalidValue;
  }
  if (out_element_offset) *out_element_offset = element_offset;
  if (out_element_width) *out_element_width = element_width;
  return hipSuccess;
}

static hipError_t iree_hip_array_byte_range_to_elements(
    hipArray_const_t array, size_t byte_offset, size_t byte_width,
    size_t* out_element_offset, size_t* out_element_width) {
  struct hipArray_st* array_info = NULL;
  hipError_t result = iree_hip_array_retain(array, &array_info);
  if (result != hipSuccess) return result;
  result = iree_hip_array_byte_range_to_elements_for_array(
      array_info, byte_offset, byte_width, out_element_offset,
      out_element_width);
  iree_hip_array_release(array_info);
  return result;
}

static hipError_t iree_hip_array_row_device_pointer(
    const struct hipArray_st* array_info, size_t row_offset, size_t byte_offset,
    iree_hal_streaming_deviceptr_t* out_device_ptr) {
  *out_device_ptr = 0;
  if (!array_info || row_offset >= array_info->rows_per_slice ||
      byte_offset >= array_info->width_bytes) {
    return hipErrorInvalidValue;
  }
  iree_host_size_t row_start = 0;
  iree_host_size_t range_start = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(row_offset, array_info->pitch,
                                      &row_start) ||
          !iree_host_size_checked_add(row_start, byte_offset, &range_start) ||
          range_start > array_info->allocation_size)) {
    return hipErrorInvalidValue;
  }
  *out_device_ptr = array_info->device_ptr + range_start;
  return hipSuccess;
}

static hipError_t iree_hip_array_enqueue_packed_row_copy(
    bool array_is_destination, iree_hal_streaming_deviceptr_t array_ptr,
    void* external_ptr, size_t byte_count, hipMemcpyKind kind) {
  if (array_is_destination) {
    return hipMemcpyAsync((void*)array_ptr, external_ptr, byte_count, kind,
                          NULL);
  }
  return hipMemcpyAsync(external_ptr, (const void*)array_ptr, byte_count, kind,
                        NULL);
}

static hipError_t iree_hip_array_enqueue_packed_rows_copy(
    bool array_is_destination, iree_hal_streaming_deviceptr_t array_ptr,
    size_t array_pitch, void* external_ptr, size_t row_width, size_t row_count,
    hipMemcpyKind kind) {
  if (array_is_destination) {
    return hipMemcpy2DAsync((void*)array_ptr, array_pitch, external_ptr,
                            row_width, row_width, row_count, kind, NULL);
  }
  return hipMemcpy2DAsync(external_ptr, row_width, (const void*)array_ptr,
                          array_pitch, row_width, row_count, kind, NULL);
}

// Copies the selected 2D array slice as a packed byte stream. The allocation
// pitch is an implementation detail, so a range spanning rows must skip its
// padding rather than expose that padding to the caller. The synchronous APIs
// enqueue a prefix, complete-row range, and suffix on the default stream, then
// synchronize once after all successfully queued work has completed.
static hipError_t iree_hip_array_copy_packed_rows(
    hipArray_const_t array, size_t byte_offset, size_t row_offset,
    void* external_ptr, size_t byte_count, hipMemcpyKind kind,
    bool array_is_destination) {
  struct hipArray_st* array_info = NULL;
  hipError_t result = iree_hip_array_retain(array, &array_info);
  if (result != hipSuccess) return result;

  iree_host_size_t whole_row_byte_count = 0;
  iree_host_size_t remaining_byte_count = 0;
  if (byte_offset >= array_info->width_bytes ||
      row_offset >= array_info->rows_per_slice ||
      !iree_host_size_checked_mul(array_info->rows_per_slice - row_offset - 1,
                                  array_info->width_bytes,
                                  &whole_row_byte_count) ||
      !iree_host_size_checked_add(array_info->width_bytes - byte_offset,
                                  whole_row_byte_count,
                                  &remaining_byte_count) ||
      byte_count > remaining_byte_count) {
    iree_hip_array_release(array_info);
    return hipErrorInvalidValue;
  }

  iree_hal_streaming_context_t* context = NULL;
  result = iree_hip_ensure_context(&context);
  if (result == hipSuccess &&
      iree_hip_context_invalidate_visible_captures(context)) {
    result = hipErrorStreamCaptureImplicit;
  }

  bool may_have_enqueued_work = false;
  size_t external_offset = 0;
  size_t current_row = row_offset;
  size_t remaining = byte_count;
  const size_t first_row_byte_count =
      byte_offset == 0 ? 0
                       : (remaining < array_info->width_bytes - byte_offset
                              ? remaining
                              : array_info->width_bytes - byte_offset);
  if (result == hipSuccess && first_row_byte_count != 0) {
    iree_hal_streaming_deviceptr_t array_ptr = 0;
    result = iree_hip_array_row_device_pointer(array_info, current_row,
                                               byte_offset, &array_ptr);
    if (result == hipSuccess) {
      may_have_enqueued_work = true;
      result = iree_hip_array_enqueue_packed_row_copy(
          array_is_destination, array_ptr, (uint8_t*)external_ptr,
          first_row_byte_count, kind);
    }
    external_offset = first_row_byte_count;
    remaining -= first_row_byte_count;
    ++current_row;
  }

  const size_t full_row_count = remaining / array_info->width_bytes;
  if (result == hipSuccess && full_row_count != 0) {
    iree_hal_streaming_deviceptr_t array_ptr = 0;
    result = iree_hip_array_row_device_pointer(array_info, current_row, 0,
                                               &array_ptr);
    if (result == hipSuccess) {
      may_have_enqueued_work = true;
      result = iree_hip_array_enqueue_packed_rows_copy(
          array_is_destination, array_ptr, array_info->pitch,
          (uint8_t*)external_ptr + external_offset, array_info->width_bytes,
          full_row_count, kind);
    }
    const size_t full_row_byte_count = full_row_count * array_info->width_bytes;
    external_offset += full_row_byte_count;
    remaining -= full_row_byte_count;
    current_row += full_row_count;
  }

  if (result == hipSuccess && remaining != 0) {
    iree_hal_streaming_deviceptr_t array_ptr = 0;
    result = iree_hip_array_row_device_pointer(array_info, current_row, 0,
                                               &array_ptr);
    if (result == hipSuccess) {
      may_have_enqueued_work = true;
      result = iree_hip_array_enqueue_packed_row_copy(
          array_is_destination, array_ptr,
          (uint8_t*)external_ptr + external_offset, remaining, kind);
    }
  }

  if (may_have_enqueued_work) {
    hipError_t synchronize_result = hipDeviceSynchronize();
    if (result == hipSuccess) result = synchronize_result;
  }
  iree_hip_array_release(array_info);
  return result;
}

static hipError_t iree_hip_array_legacy_row_range(
    hipArray_const_t array, size_t byte_offset, size_t byte_count,
    const struct hipArray_st** out_array,
    iree_hal_streaming_deviceptr_t* out_device_ptr) {
  if (out_array) *out_array = NULL;
  if (out_device_ptr) *out_device_ptr = 0;
  if (!array) return hipErrorInvalidValue;
  struct hipArray_st* array_info = NULL;
  hipError_t result = iree_hip_array_retain(array, &array_info);
  if (result != hipSuccess) return result;
  if (byte_offset > array_info->width_bytes ||
      byte_count > array_info->width_bytes - byte_offset) {
    iree_hip_array_release(array_info);
    return hipErrorInvalidValue;
  }
  if (out_array) *out_array = array_info;
  if (out_device_ptr) *out_device_ptr = array_info->device_ptr + byte_offset;
  if (!out_array) iree_hip_array_release(array_info);
  return hipSuccess;
}

static hipError_t iree_hip_memcpy2d_array_to_3d_params(
    const hip_Memcpy2D* copy, hipMemcpy3DParms* out_params) {
  if (!copy || !out_params) return hipErrorInvalidValue;
  memset(out_params, 0, sizeof(*out_params));

  const bool src_is_array = copy->srcMemoryType == hipMemoryTypeArray;
  const bool dst_is_array = copy->dstMemoryType == hipMemoryTypeArray;
  if (!src_is_array && !dst_is_array) return hipErrorInvalidValue;

  size_t array_width_elements = 0;
  bool array_width_set = false;
  if (src_is_array) {
    struct hipArray_st* src_array = NULL;
    hipError_t result =
        iree_hip_array_retain((hipArray_const_t)copy->srcArray, &src_array);
    if (result != hipSuccess) return result;
    size_t src_element_offset = 0;
    size_t src_element_width = 0;
    result = iree_hip_array_byte_range_to_elements_for_array(
        src_array, copy->srcXInBytes, copy->WidthInBytes, &src_element_offset,
        &src_element_width);
    if (result != hipSuccess) {
      iree_hip_array_release(src_array);
      return result;
    }
    if (copy->Height != 0 &&
        (copy->srcY >= src_array->extent.height ||
         copy->Height > src_array->extent.height - copy->srcY)) {
      iree_hip_array_release(src_array);
      return hipErrorInvalidValue;
    }
    out_params->srcArray = copy->srcArray;
    out_params->srcPos.x = src_element_offset;
    out_params->srcPos.y = copy->srcY;
    array_width_elements = src_element_width;
    array_width_set = true;
    iree_hip_array_release(src_array);
  } else {
    const void* src = NULL;
    switch (copy->srcMemoryType) {
      case hipMemoryTypeHost:
        src = copy->srcHost;
        break;
      case hipMemoryTypeDevice:
      case hipMemoryTypeUnified:
        src = copy->srcDevice;
        break;
      default:
        return hipErrorInvalidValue;
    }
    if ((copy->WidthInBytes != 0 || copy->Height != 0) && !src) {
      return hipErrorInvalidValue;
    }
    iree_host_size_t src_width = 0;
    iree_host_size_t src_height = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
                          copy->srcXInBytes, copy->WidthInBytes, &src_width) ||
                      !iree_host_size_checked_add(copy->srcY, copy->Height,
                                                  &src_height))) {
      return hipErrorInvalidValue;
    }
    if (copy->WidthInBytes != 0 && src_width > copy->srcPitch) {
      return hipErrorInvalidValue;
    }
    out_params->srcPtr.ptr = (void*)src;
    out_params->srcPtr.pitch = copy->srcPitch;
    out_params->srcPtr.xsize = src_width;
    out_params->srcPtr.ysize = src_height;
    out_params->srcPos.x = copy->srcXInBytes;
    out_params->srcPos.y = copy->srcY;
  }

  if (dst_is_array) {
    struct hipArray_st* dst_array = NULL;
    hipError_t result =
        iree_hip_array_retain((hipArray_const_t)copy->dstArray, &dst_array);
    if (result != hipSuccess) return result;
    size_t dst_element_offset = 0;
    size_t dst_element_width = 0;
    result = iree_hip_array_byte_range_to_elements_for_array(
        dst_array, copy->dstXInBytes, copy->WidthInBytes, &dst_element_offset,
        &dst_element_width);
    if (result != hipSuccess) {
      iree_hip_array_release(dst_array);
      return result;
    }
    if (copy->Height != 0 &&
        (copy->dstY >= dst_array->extent.height ||
         copy->Height > dst_array->extent.height - copy->dstY)) {
      iree_hip_array_release(dst_array);
      return hipErrorInvalidValue;
    }
    if (array_width_set && array_width_elements != dst_element_width) {
      iree_hip_array_release(dst_array);
      return hipErrorInvalidValue;
    }
    out_params->dstArray = copy->dstArray;
    out_params->dstPos.x = dst_element_offset;
    out_params->dstPos.y = copy->dstY;
    array_width_elements = dst_element_width;
    array_width_set = true;
    iree_hip_array_release(dst_array);
  } else {
    void* dst = NULL;
    switch (copy->dstMemoryType) {
      case hipMemoryTypeHost:
        dst = copy->dstHost;
        break;
      case hipMemoryTypeDevice:
      case hipMemoryTypeUnified:
        dst = copy->dstDevice;
        break;
      default:
        return hipErrorInvalidValue;
    }
    if ((copy->WidthInBytes != 0 || copy->Height != 0) && !dst) {
      return hipErrorInvalidValue;
    }
    iree_host_size_t dst_width = 0;
    iree_host_size_t dst_height = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
                          copy->dstXInBytes, copy->WidthInBytes, &dst_width) ||
                      !iree_host_size_checked_add(copy->dstY, copy->Height,
                                                  &dst_height))) {
      return hipErrorInvalidValue;
    }
    if (copy->WidthInBytes != 0 && dst_width > copy->dstPitch) {
      return hipErrorInvalidValue;
    }
    out_params->dstPtr.ptr = dst;
    out_params->dstPtr.pitch = copy->dstPitch;
    out_params->dstPtr.xsize = dst_width;
    out_params->dstPtr.ysize = dst_height;
    out_params->dstPos.x = copy->dstXInBytes;
    out_params->dstPos.y = copy->dstY;
  }

  out_params->extent.width =
      array_width_set ? array_width_elements : copy->WidthInBytes;
  out_params->extent.height = copy->Height;
  out_params->extent.depth = 1;

  if (copy->srcMemoryType == hipMemoryTypeUnified ||
      copy->dstMemoryType == hipMemoryTypeUnified) {
    out_params->kind = hipMemcpyDefault;
  } else if (copy->srcMemoryType == hipMemoryTypeHost &&
             copy->dstMemoryType == hipMemoryTypeHost) {
    out_params->kind = hipMemcpyHostToHost;
  } else if (copy->srcMemoryType == hipMemoryTypeHost) {
    out_params->kind = hipMemcpyHostToDevice;
  } else if (copy->dstMemoryType == hipMemoryTypeHost) {
    out_params->kind = hipMemcpyDeviceToHost;
  } else {
    out_params->kind = hipMemcpyDeviceToDevice;
  }
  return hipSuccess;
}

static hipError_t iree_hip_array_create(hipArray_t* array,
                                        const hipChannelFormatDesc* desc,
                                        hipExtent extent, unsigned int flags) {
  if (!array || !desc || extent.width == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *array = NULL;

  size_t element_size = 0;
  hipArray_Format format = 0;
  unsigned int num_channels = 0;
  hipError_t element_result = iree_hip_array_desc_to_format(
      desc, &format, &num_channels, &element_size);
  if (element_result != hipSuccess) {
    HIP_RETURN_ERROR(element_result);
  }
  if (flags != hipArrayDefault) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t extent_result = iree_hip_array_validate_extent_limits(extent);
  if (extent_result != hipSuccess) HIP_RETURN_ERROR(extent_result);

  const hipExtent public_extent = extent;
  if (extent.height == 0) extent.height = 1;
  if (extent.depth == 0) extent.depth = 1;

  iree_host_size_t width_bytes = 0;
  iree_host_size_t slice_pitch = 0;
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(extent.width, element_size,
                                                &width_bytes) ||
                    !iree_host_size_checked_mul(width_bytes, extent.height,
                                                &slice_pitch) ||
                    !iree_host_size_checked_mul(slice_pitch, extent.depth,
                                                &allocation_size))) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, allocation_size, 0, &buffer);
  if (!iree_status_is_ok(status)) {
    return iree_status_to_hip_result(status);
  }

  struct hipArray_st* new_array =
      (struct hipArray_st*)calloc(1, sizeof(*new_array));
  if (!new_array) {
    iree_status_ignore(
        iree_hal_streaming_memory_free_device(context, buffer->device_ptr));
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  new_array->magic = IREE_HIP_ARRAY_MAGIC;
  iree_atomic_ref_count_init(&new_array->ref_count);
  new_array->desc = *desc;
  new_array->public_extent = public_extent;
  new_array->extent = extent;
  new_array->format = format;
  new_array->num_channels = num_channels;
  new_array->flags = flags;
  new_array->context = context;
  iree_hal_streaming_context_retain(context);
  new_array->buffer = buffer;
  new_array->device_ptr = buffer->device_ptr;
  new_array->element_size = element_size;
  new_array->width_bytes = width_bytes;
  new_array->pitch = width_bytes;
  new_array->rows_per_slice = extent.height;
  new_array->slice_pitch = slice_pitch;
  new_array->allocation_size = allocation_size;
  iree_hip_array_registry_insert(new_array);
  *array = new_array;
  return hipSuccess;
}

static hipError_t iree_hip_resolve_memcpy3d_array_params(
    const hipMemcpy3DParms* params, hipMemcpy3DParms* out_params) {
  if (!params || !out_params) return hipErrorInvalidValue;
  *out_params = *params;
  struct hipArray_st* src_array = NULL;
  struct hipArray_st* dst_array = NULL;
  hipError_t result = hipSuccess;
  if (params->srcArray) {
    if (params->srcPtr.ptr) {
      result = hipErrorInvalidValue;
    } else {
      result =
          iree_hip_array_retain((hipArray_const_t)params->srcArray, &src_array);
    }
  }
  if (result == hipSuccess && params->dstArray) {
    if (params->dstPtr.ptr) {
      result = hipErrorInvalidValue;
    } else {
      result =
          iree_hip_array_retain((hipArray_const_t)params->dstArray, &dst_array);
    }
  }
  if (result == hipSuccess && src_array && dst_array) {
    if (src_array->element_size != dst_array->element_size) {
      result = hipErrorInvalidValue;
    }
  }
  if (result == hipSuccess && (src_array || dst_array)) {
    const size_t element_size =
        dst_array ? dst_array->element_size : src_array->element_size;
    iree_host_size_t width_bytes = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            params->extent.width, element_size, &width_bytes))) {
      result = hipErrorInvalidValue;
    } else {
      out_params->extent.width = width_bytes;
    }
  }
  if (result == hipSuccess && src_array) {
    iree_host_size_t src_x_in_bytes = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            params->srcPos.x, src_array->element_size, &src_x_in_bytes))) {
      result = hipErrorInvalidValue;
    } else {
      out_params->srcArray = NULL;
      out_params->srcPos.x = src_x_in_bytes;
      out_params->srcPtr.ptr = (void*)src_array->device_ptr;
      out_params->srcPtr.pitch = src_array->pitch;
      out_params->srcPtr.xsize = src_array->width_bytes;
      out_params->srcPtr.ysize = src_array->rows_per_slice;
    }
  }
  if (result == hipSuccess && dst_array) {
    iree_host_size_t dst_x_in_bytes = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            params->dstPos.x, dst_array->element_size, &dst_x_in_bytes))) {
      result = hipErrorInvalidValue;
    } else {
      out_params->dstArray = NULL;
      out_params->dstPos.x = dst_x_in_bytes;
      out_params->dstPtr.ptr = (void*)dst_array->device_ptr;
      out_params->dstPtr.pitch = dst_array->pitch;
      out_params->dstPtr.xsize = dst_array->width_bytes;
      out_params->dstPtr.ysize = dst_array->rows_per_slice;
    }
  }
  iree_hip_array_release(dst_array);
  iree_hip_array_release(src_array);
  return result;
}

HIPAPI hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset,
                                          size_t hOffset, const void* src,
                                          size_t spitch, size_t width,
                                          size_t height, hipMemcpyKind kind,
                                          hipStream_t stream) {
  if (!dst) HIP_RETURN_ERROR(hipErrorInvalidHandle);
  if (!src || spitch < width) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (width == 0 || height == 0) return hipSuccess;
  size_t element_offset = 0;
  size_t element_width = 0;
  hipError_t range_result = iree_hip_array_byte_range_to_elements(
      (hipArray_const_t)dst, wOffset, width, &element_offset, &element_width);
  if (range_result != hipSuccess) {
    HIP_RETURN_ERROR(range_result);
  }
  hipMemcpy3DParms params = {0};
  params.srcPtr.ptr = (void*)src;
  params.srcPtr.pitch = spitch;
  params.srcPtr.xsize = width;
  params.srcPtr.ysize = height;
  params.dstArray = dst;
  params.dstPos.x = element_offset;
  params.dstPos.y = hOffset;
  params.extent.width = element_width;
  params.extent.height = height;
  params.extent.depth = 1;
  params.kind = kind;
  return hipMemcpy3DAsync(&params, stream);
}

HIPAPI hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset,
                                     size_t hOffset, const void* src,
                                     size_t spitch, size_t width, size_t height,
                                     hipMemcpyKind kind) {
  if (!dst) HIP_RETURN_ERROR(hipErrorInvalidHandle);
  if (!src || spitch < width) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (width == 0 || height == 0) return hipSuccess;
  size_t element_offset = 0;
  size_t element_width = 0;
  hipError_t range_result = iree_hip_array_byte_range_to_elements(
      (hipArray_const_t)dst, wOffset, width, &element_offset, &element_width);
  if (range_result != hipSuccess) {
    HIP_RETURN_ERROR(range_result);
  }
  hipMemcpy3DParms params = {0};
  params.srcPtr.ptr = (void*)src;
  params.srcPtr.pitch = spitch;
  params.srcPtr.xsize = width;
  params.srcPtr.ysize = height;
  params.dstArray = dst;
  params.dstPos.x = element_offset;
  params.dstPos.y = hOffset;
  params.extent.width = element_width;
  params.extent.height = height;
  params.extent.depth = 1;
  params.kind = kind;
  return hipMemcpy3D(&params);
}

HIPAPI hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch,
                                            hipArray_const_t src,
                                            size_t wOffset, size_t hOffset,
                                            size_t width, size_t height,
                                            hipMemcpyKind kind,
                                            hipStream_t stream) {
  if (!src) HIP_RETURN_ERROR(hipErrorInvalidHandle);
  if (!dst || dpitch < width) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (width == 0 || height == 0) return hipSuccess;
  size_t element_offset = 0;
  size_t element_width = 0;
  hipError_t range_result = iree_hip_array_byte_range_to_elements(
      src, wOffset, width, &element_offset, &element_width);
  if (range_result != hipSuccess) {
    HIP_RETURN_ERROR(range_result);
  }
  hipMemcpy3DParms params = {0};
  params.srcArray = (hipArray_t)src;
  params.srcPos.x = element_offset;
  params.srcPos.y = hOffset;
  params.dstPtr.ptr = dst;
  params.dstPtr.pitch = dpitch;
  params.dstPtr.xsize = width;
  params.dstPtr.ysize = height;
  params.extent.width = element_width;
  params.extent.height = height;
  params.extent.depth = 1;
  params.kind = kind;
  return hipMemcpy3DAsync(&params, stream);
}

HIPAPI hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch,
                                       hipArray_const_t src, size_t wOffset,
                                       size_t hOffset, size_t width,
                                       size_t height, hipMemcpyKind kind) {
  if (!src) HIP_RETURN_ERROR(hipErrorInvalidHandle);
  if (!dst || dpitch < width) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (width == 0 || height == 0) return hipSuccess;
  size_t element_offset = 0;
  size_t element_width = 0;
  hipError_t range_result = iree_hip_array_byte_range_to_elements(
      src, wOffset, width, &element_offset, &element_width);
  if (range_result != hipSuccess) {
    HIP_RETURN_ERROR(range_result);
  }
  hipMemcpy3DParms params = {0};
  params.srcArray = (hipArray_t)src;
  params.srcPos.x = element_offset;
  params.srcPos.y = hOffset;
  params.dstPtr.ptr = dst;
  params.dstPtr.pitch = dpitch;
  params.dstPtr.xsize = width;
  params.dstPtr.ysize = height;
  params.extent.width = element_width;
  params.extent.height = height;
  params.extent.depth = 1;
  params.kind = kind;
  return hipMemcpy3D(&params);
}

HIPAPI hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst,
                                          size_t hOffsetDst,
                                          hipArray_const_t src,
                                          size_t wOffsetSrc, size_t hOffsetSrc,
                                          size_t width, size_t height,
                                          hipMemcpyKind kind) {
  if (!dst || !src) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (width == 0 || height == 0) return hipSuccess;
  size_t src_element_offset = 0;
  size_t src_element_width = 0;
  hipError_t range_result = iree_hip_array_byte_range_to_elements(
      src, wOffsetSrc, width, &src_element_offset, &src_element_width);
  if (range_result != hipSuccess) {
    HIP_RETURN_ERROR(range_result);
  }
  size_t dst_element_offset = 0;
  size_t dst_element_width = 0;
  range_result = iree_hip_array_byte_range_to_elements(
      (hipArray_const_t)dst, wOffsetDst, width, &dst_element_offset,
      &dst_element_width);
  if (range_result != hipSuccess || src_element_width != dst_element_width) {
    HIP_RETURN_ERROR(range_result == hipSuccess ? hipErrorInvalidValue
                                                : range_result);
  }
  hipMemcpy3DParms params = {0};
  params.srcArray = (hipArray_t)src;
  params.srcPos.x = src_element_offset;
  params.srcPos.y = hOffsetSrc;
  params.dstArray = dst;
  params.dstPos.x = dst_element_offset;
  params.dstPos.y = hOffsetDst;
  params.extent.width = src_element_width;
  params.extent.height = height;
  params.extent.depth = 1;
  params.kind = kind;
  return hipMemcpy3D(&params);
}

HIPAPI hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset,
                                   size_t hOffset, const void* src,
                                   size_t count, hipMemcpyKind kind) {
  if (!src) HIP_RETURN_ERROR(hipErrorInvalidValue);
  hipError_t result = iree_hip_array_copy_packed_rows(
      (hipArray_const_t)dst, wOffset, hOffset, (void*)src, count, kind,
      /*array_is_destination=*/true);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipSuccess;
}

HIPAPI hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray,
                                     size_t wOffset, size_t hOffset,
                                     size_t count, hipMemcpyKind kind) {
  if (!dst) HIP_RETURN_ERROR(hipErrorInvalidValue);
  hipError_t result = iree_hip_array_copy_packed_rows(
      srcArray, wOffset, hOffset, dst, count, kind,
      /*array_is_destination=*/false);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipSuccess;
}

HIPAPI hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset,
                                     const void* srcHost, size_t ByteCount,
                                     hipStream_t stream) {
  if (!srcHost) HIP_RETURN_ERROR(hipErrorInvalidValue);
  if (ByteCount == 0) return hipSuccess;
  iree_hal_streaming_deviceptr_t dst_ptr = 0;
  hipError_t result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)dstArray, dstOffset, ByteCount, NULL, &dst_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipMemcpyAsync((void*)dst_ptr, srcHost, ByteCount,
                        hipMemcpyHostToDevice, stream);
}

HIPAPI hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset,
                                const void* srcHost, size_t count) {
  if (!srcHost) HIP_RETURN_ERROR(hipErrorInvalidValue);
  if (count == 0) return hipSuccess;
  iree_hal_streaming_deviceptr_t dst_ptr = 0;
  hipError_t result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)dstArray, dstOffset, count, NULL, &dst_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipMemcpy((void*)dst_ptr, srcHost, count, hipMemcpyHostToDevice);
}

HIPAPI hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray,
                                     size_t srcOffset, size_t ByteCount,
                                     hipStream_t stream) {
  if (!dstHost) HIP_RETURN_ERROR(hipErrorInvalidValue);
  if (ByteCount == 0) return hipSuccess;
  iree_hal_streaming_deviceptr_t src_ptr = 0;
  hipError_t result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)srcArray, srcOffset, ByteCount, NULL, &src_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipMemcpyAsync(dstHost, (const void*)src_ptr, ByteCount,
                        hipMemcpyDeviceToHost, stream);
}

HIPAPI hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray,
                                size_t srcOffset, size_t count) {
  if (!dst) HIP_RETURN_ERROR(hipErrorInvalidValue);
  if (count == 0) return hipSuccess;
  iree_hal_streaming_deviceptr_t src_ptr = 0;
  hipError_t result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)srcArray, srcOffset, count, NULL, &src_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipMemcpy(dst, (const void*)src_ptr, count, hipMemcpyDeviceToHost);
}

HIPAPI hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset,
                                hipDeviceptr_t srcDevice, size_t ByteCount) {
  if (!srcDevice) HIP_RETURN_ERROR(hipErrorInvalidValue);
  if (ByteCount == 0) return hipSuccess;
  iree_hal_streaming_deviceptr_t dst_ptr = 0;
  hipError_t result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)dstArray, dstOffset, ByteCount, NULL, &dst_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipMemcpy((void*)dst_ptr, srcDevice, ByteCount,
                   hipMemcpyDeviceToDevice);
}

HIPAPI hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray,
                                size_t srcOffset, size_t ByteCount) {
  if (!dstDevice) HIP_RETURN_ERROR(hipErrorInvalidValue);
  if (ByteCount == 0) return hipSuccess;
  iree_hal_streaming_deviceptr_t src_ptr = 0;
  hipError_t result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)srcArray, srcOffset, ByteCount, NULL, &src_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipMemcpy(dstDevice, (const void*)src_ptr, ByteCount,
                   hipMemcpyDeviceToDevice);
}

static hipError_t iree_hip_array_validate_extent_limits(hipExtent extent) {
  if (extent.width == 0) return hipErrorInvalidValue;
  if (extent.height == 0 && extent.depth != 0) return hipErrorInvalidValue;
  if (extent.height == 0) {
    return extent.width <= IREE_HIP_ARRAY_MAX_1D ? hipSuccess
                                                 : hipErrorInvalidValue;
  }
  if (extent.depth == 0) {
    return extent.width <= IREE_HIP_ARRAY_MAX_2D_WIDTH &&
                   extent.height <= IREE_HIP_ARRAY_MAX_2D_HEIGHT
               ? hipSuccess
               : hipErrorInvalidValue;
  }
  return extent.width <= IREE_HIP_ARRAY_MAX_3D_WIDTH &&
                 extent.height <= IREE_HIP_ARRAY_MAX_3D_HEIGHT &&
                 extent.depth <= IREE_HIP_ARRAY_MAX_3D_DEPTH
             ? hipSuccess
             : hipErrorInvalidValue;
}

HIPAPI hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset,
                                hipArray_t srcArray, size_t srcOffset,
                                size_t ByteCount) {
  if (ByteCount == 0) return hipSuccess;
  iree_hal_streaming_deviceptr_t dst_ptr = 0;
  hipError_t result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)dstArray, dstOffset, ByteCount, NULL, &dst_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  iree_hal_streaming_deviceptr_t src_ptr = 0;
  result = iree_hip_array_legacy_row_range(
      (hipArray_const_t)srcArray, srcOffset, ByteCount, NULL, &src_ptr);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  return hipMemcpy((void*)dst_ptr, (const void*)src_ptr, ByteCount,
                   hipMemcpyDeviceToDevice);
}

// Sets 2D device memory to a value (asynchronous).
//
// Parameters:
//  - dst: [OUT] Device pointer to 2D memory to set.
//  - pitch: [IN] Pitch in bytes of the 2D memory.
//  - value: [IN] Value to set (interpreted as unsigned char).
//  - width: [IN] Width in bytes to set.
//  - height: [IN] Number of rows to set.
//  - stream: [IN] Stream for asynchronous execution.
//
// Returns:
//  - hipSuccess: Operation enqueued successfully.
//  - hipErrorInvalidValue: dst is NULL or dimensions invalid.
HIPAPI hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value,
                                   size_t width, size_t height,
                                   hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!dst) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  if (width > pitch) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_host_size_t byte_span = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(height - 1, pitch, &byte_span) ||
          !iree_host_size_checked_add(byte_span, width, &byte_span))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  init_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t dependency_result =
      iree_hip_order_legacy_stream_dependencies(context, stream_obj);
  if (dependency_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }

  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t range_status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)dst, byte_span, &dst_ref);
  if (!iree_status_is_ok(range_status)) {
    hipError_t result = iree_memset_status_to_hip_result(range_status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    hipMemsetParams params = {
        .dst = dst,
        .value = value,
        .pitch = pitch,
        .elementSize = 1,
        .width = width,
        .height = height,
    };
    hipGraphNode_t node = NULL;
    hipError_t result = hipGraphAddMemsetNode(
        &node, (hipGraph_t)stream_obj->capture_graph,
        (const hipGraphNode_t*)stream_obj->capture_dependencies,
        stream_obj->capture_dependency_count, &params);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_capture_set_last_node(
            stream_obj, (iree_hal_streaming_graph_node_t*)node),
        hipErrorInvalidValue);
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Perform row-by-row memset.
  uint8_t* dst_ptr = (uint8_t*)dst;
  for (size_t row = 0; row < height; ++row) {
    iree_status_t status = iree_hal_streaming_memory_memset(
        context, (iree_hal_streaming_deviceptr_t)(dst_ptr + row * pitch), width,
        &value, 1, stream_obj);
    if (!iree_status_is_ok(status)) {
      hipError_t result = iree_memset_status_to_hip_result(status);
      IREE_TRACE_ZONE_END(z0);
      return result;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets 2D device memory to a value (synchronous).
//
// Parameters:
//  - dst: [OUT] Device pointer to 2D memory to set.
//  - pitch: [IN] Pitch in bytes of the 2D memory.
//  - value: [IN] Value to set (interpreted as unsigned char).
//  - width: [IN] Width in bytes to set.
//  - height: [IN] Number of rows to set.
//
// Returns:
//  - hipSuccess: Operation completed successfully.
//  - hipErrorInvalidValue: dst is NULL or dimensions invalid.
HIPAPI hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width,
                              size_t height) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dst) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  if (width > pitch) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  hipError_t result = hipMemset2DAsync(dst, pitch, value, width, height, NULL);

  if (result == hipSuccess) {
    result = hipDeviceSynchronize();
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

static hipError_t iree_hip_validate_memset3d_shape(
    hipPitchedPtr pitchedDevPtr, hipExtent extent,
    iree_host_size_t* out_slice_pitch) {
  *out_slice_pitch = 0;
  if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
    return hipSuccess;
  }
  if (!pitchedDevPtr.ptr || pitchedDevPtr.pitch == 0 ||
      extent.width > pitchedDevPtr.pitch) {
    return hipErrorInvalidValue;
  }
  const iree_host_size_t rows_per_slice =
      pitchedDevPtr.ysize ? pitchedDevPtr.ysize : extent.height;
  if (extent.height > rows_per_slice) {
    return hipErrorInvalidValue;
  }
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          pitchedDevPtr.pitch, rows_per_slice, out_slice_pitch))) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_memset3d_byte_span(hipPitchedPtr pitchedDevPtr,
                                              hipExtent extent,
                                              iree_host_size_t slice_pitch,
                                              iree_host_size_t* out_byte_span) {
  *out_byte_span = 0;
  if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
    return hipSuccess;
  }
  iree_host_size_t last_slice_offset = 0;
  iree_host_size_t last_row_offset = 0;
  iree_host_size_t last_element_offset = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(extent.depth - 1, slice_pitch,
                                      &last_slice_offset) ||
          !iree_host_size_checked_mul(extent.height - 1, pitchedDevPtr.pitch,
                                      &last_row_offset) ||
          !iree_host_size_checked_add(last_slice_offset, last_row_offset,
                                      &last_element_offset) ||
          !iree_host_size_checked_add(last_element_offset, extent.width,
                                      out_byte_span))) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

HIPAPI hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value,
                                   hipExtent extent, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t slice_pitch = 0;
  hipError_t result =
      iree_hip_validate_memset3d_shape(pitchedDevPtr, extent, &slice_pitch);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  iree_host_size_t byte_span = 0;
  result = iree_hip_memset3d_byte_span(pitchedDevPtr, extent, slice_pitch,
                                       &byte_span);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_context_t* context = NULL;
  result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t range_status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)pitchedDevPtr.ptr, byte_span,
      &dst_ref);
  if (!iree_status_is_ok(range_status)) {
    result = iree_memset_status_to_hip_result(range_status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  const iree_host_size_t rows_per_slice =
      pitchedDevPtr.ysize ? pitchedDevPtr.ysize : extent.height;
  if (extent.width == pitchedDevPtr.pitch && extent.height == rows_per_slice) {
    iree_host_size_t byte_count = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(slice_pitch, extent.depth,
                                                  &byte_count))) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    hipError_t linear_result =
        hipMemsetAsync(pitchedDevPtr.ptr, value, byte_count, stream);
    IREE_TRACE_ZONE_END(z0);
    return linear_result;
  }

  uint8_t* base = (uint8_t*)pitchedDevPtr.ptr;
  for (size_t z = 0; z < extent.depth; ++z) {
    iree_host_size_t slice_offset = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(z, slice_pitch, &slice_offset))) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    result = hipMemset2DAsync(base + slice_offset, pitchedDevPtr.pitch, value,
                              extent.width, extent.height, stream);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

HIPAPI hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value,
                              hipExtent extent) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t slice_pitch = 0;
  hipError_t result =
      iree_hip_validate_memset3d_shape(pitchedDevPtr, extent, &slice_pitch);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  iree_hal_streaming_context_t* context = NULL;
  result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  result = hipMemset3DAsync(pitchedDevPtr, value, extent, NULL);
  if (result == hipSuccess) {
    result = hipDeviceSynchronize();
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates 3D device memory.
//
// Parameters:
//  - pitchedDevPtr: [OUT] Pointer to receive the pitched pointer structure.
//  - extent: [IN] Extent of the 3D allocation (width, height, depth).
//
// Returns:
//  - hipSuccess: Allocation successful.
//  - hipErrorInvalidValue: pitchedDevPtr is NULL.
//  - hipErrorOutOfMemory: Allocation failed.
HIPAPI hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pitchedDevPtr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Initialize output.
  pitchedDevPtr->ptr = NULL;
  pitchedDevPtr->pitch = 0;
  pitchedDevPtr->xsize = 0;
  pitchedDevPtr->ysize = 0;

  // Zero extent is technically valid but produces NULL allocation.
  if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  const size_t alignment = IREE_HAL_STREAMING_PITCHED_ALLOCATION_ALIGNMENT;
  size_t pitch = 0;
  size_t slice_size = 0;
  size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(extent.width, alignment - 1, &pitch) ||
          !iree_host_size_checked_mul(pitch / alignment, alignment, &pitch) ||
          !iree_host_size_checked_mul(pitch, extent.height, &slice_size) ||
          !iree_host_size_checked_mul(slice_size, extent.depth, &total_size))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  // Allocate the memory.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, total_size, 0, &buffer);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_status_to_hip_result(status);
  }

  // Fill in the pitched pointer structure.
  pitchedDevPtr->ptr = (void*)iree_hal_streaming_buffer_device_pointer(buffer);
  pitchedDevPtr->pitch = pitch;
  pitchedDevPtr->xsize = extent.width;
  pitchedDevPtr->ysize = extent.height;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

HIPAPI hipError_t hipMalloc3DArray(hipArray_t* array,
                                   const hipChannelFormatDesc* desc,
                                   hipExtent extent, unsigned int flags) {
  return iree_hip_array_create(array, desc, extent, flags);
}

HIPAPI hipError_t hipMallocArray(hipArray_t* array,
                                 const hipChannelFormatDesc* desc, size_t width,
                                 size_t height, unsigned int flags) {
  return iree_hip_array_create(array, desc, (hipExtent){width, height, 0},
                               flags);
}

// Copies data from host memory to device memory (synchronous).
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source host pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorInvalidContext: No active context.
//  - hipErrorInvalidDevice: Device pointer not valid for current device.
//
// Synchronization: SYNCHRONOUS - blocks until copy completes.
// Stream behavior: Executes on default stream (stream 0).
// Multi-GPU: Current device context determines which device performs copy.
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
// Limitations: None in current implementation.
//
// Warning: Passing device pointer as src or host pointer as dst causes
//          undefined behavior.
//
// See also: hipMemcpy, hipMemcpyHtoDAsync, hipMemcpyDtoH.
HIPAPI hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src,
                                size_t sizeBytes) {
  // Synchronous host-to-device copy.
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
}

// Copies data from device memory to host memory (synchronous).
//
// Parameters:
//  - dst: [OUT] Destination host pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorInvalidContext: No active context.
//  - hipErrorInvalidDevice: Device pointer not valid for current device.
//
// Synchronization: SYNCHRONOUS - blocks until copy completes.
// Stream behavior: Executes on default stream (stream 0).
// Multi-GPU: Current device context determines which device performs copy.
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
// Limitations: None in current implementation.
//
// Warning: Passing host pointer as src or device pointer as dst causes
//          undefined behavior.
//
// See also: hipMemcpy, hipMemcpyDtoHAsync, hipMemcpyHtoD.
HIPAPI hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src,
                                size_t sizeBytes) {
  // Synchronous device-to-host copy.
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
}

// Copies data from device memory to device memory (synchronous).
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorInvalidContext: No active context.
//  - hipErrorInvalidDevice: Device pointers not valid for current device.
//
// Synchronization: SYNCHRONOUS - blocks until copy completes.
// Stream behavior: Executes on default stream (stream 0).
// Multi-GPU: Both pointers must be accessible from current device. For
//            cross-device copies, use hipMemcpyPeer.
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
// Limitations: Intra-device copy only; no automatic peer access.
//
// Note: Source and destination regions may overlap; behavior is equivalent
//       to memmove.
//
// See also: hipMemcpy, hipMemcpyDtoDAsync, hipMemcpyPeer.
HIPAPI hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                                size_t sizeBytes) {
  // Synchronous device-to-device copy.
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
}

// Copies data from host to device memory asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source host pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorInvalidDevicePointer: Invalid device pointer.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Host memory should be pinned for true async behavior.
// - Graph capture: Supported. Creates memcpy node when capturing.
//
// Multi-GPU: Copies to current device.
//
// See also: hipMemcpyHtoD, hipMemcpyAsync, hipMemcpyDtoHAsync.
HIPAPI hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src,
                                     size_t sizeBytes, hipStream_t stream) {
  // Asynchronous host-to-device copy.
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream);
}

// Copies data from device to host memory asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination host pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorInvalidDevicePointer: Invalid device pointer.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Host memory should be pinned for true async behavior.
// - Graph capture: Supported. Creates memcpy node when capturing.
//
// Multi-GPU: Copies from current device.
//
// Warning: Host memory must remain valid until copy completes.
//
// See also: hipMemcpyDtoH, hipMemcpyAsync, hipMemcpyHtoDAsync.
HIPAPI hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src,
                                     size_t sizeBytes, hipStream_t stream) {
  // Asynchronous device-to-host copy.
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream);
}

// Copies data from device to device memory asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorInvalidDevicePointer: Invalid device pointers.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Executes after all previously enqueued operations.
// - Graph capture: Supported. Creates memcpy node when capturing.
//
// Multi-GPU: Both pointers must be accessible from current device.
// For cross-device copies, use hipMemcpyPeerAsync.
//
// Note: Source and destination may overlap (behaves like memmove).
//
// See also: hipMemcpyDtoD, hipMemcpyAsync, hipMemcpyPeerAsync.
HIPAPI hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
                                     size_t sizeBytes, hipStream_t stream) {
  // Asynchronous device-to-device copy.
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream);
}

//===----------------------------------------------------------------------===//
// Peer-to-Peer Memory Operations
//===----------------------------------------------------------------------===//

static hipError_t iree_hip_primary_context_for_device_id(
    int device_id, iree_hal_streaming_context_t** out_context) {
  if (out_context) *out_context = NULL;
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) return init_result;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry || device_id < 0 ||
      device_id >= (int)device_registry->device_count) {
    return hipErrorInvalidDevice;
  }

  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(device_id);
  if (!device) return hipErrorInvalidDevice;

  iree_hal_streaming_context_t* context = NULL;
  iree_status_t status =
      iree_hal_streaming_device_get_or_create_primary_context(device, &context);
  if (!iree_status_is_ok(status)) return iree_status_to_hip_result(status);

  if (out_context) *out_context = context;
  return hipSuccess;
}

static hipError_t iree_hip_validate_peer_copy_range(
    iree_hal_streaming_context_t* context, const void* ptr, size_t size) {
  if (size == 0) return hipSuccess;
  iree_hal_streaming_buffer_ref_t ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)ptr, size, &ref);
  if (iree_status_is_ok(status)) {
    return (ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
               ? hipSuccess
               : hipErrorInvalidValue;
  }
  iree_status_ignore(status);
  return hipErrorInvalidValue;
}

static hipError_t iree_hip_memcpy_peer_staged(
    iree_hal_streaming_context_t* dst_context, void* dst,
    iree_hal_streaming_context_t* src_context, const void* src,
    size_t size_bytes, iree_hal_streaming_stream_t* stream) {
  if (size_bytes == 0) return hipSuccess;
  if (!dst_context || !src_context || !dst || !src) return hipErrorInvalidValue;

  hipError_t validate_result =
      iree_hip_validate_peer_copy_range(dst_context, dst, size_bytes);
  if (validate_result != hipSuccess) return validate_result;
  validate_result =
      iree_hip_validate_peer_copy_range(src_context, src, size_bytes);
  if (validate_result != hipSuccess) return validate_result;

  uint8_t* staging = (uint8_t*)malloc(size_bytes);
  if (!staging) return hipErrorOutOfMemory;

  iree_status_t status = iree_ok_status();
  iree_hal_streaming_stream_t* src_stream = NULL;
  if (stream && stream->context == src_context) {
    src_stream = stream;
  } else if (stream) {
    status = iree_hal_streaming_stream_synchronize(stream);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_memcpy_device_to_host(
        src_context, staging, (iree_hal_streaming_deviceptr_t)src, size_bytes,
        src_stream);
  }
  if (iree_status_is_ok(status) && src_stream) {
    // This fallback consumes and releases |staging| before returning. A D2H
    // queued on the source stream must complete before the destination
    // transfer can read that storage.
    status = iree_hal_streaming_stream_synchronize(src_stream);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_memcpy_host_to_device(
        dst_context, (iree_hal_streaming_deviceptr_t)dst, staging, size_bytes,
        NULL);
  }

  free(staging);
  return iree_status_to_hip_result(status);
}

// Copies memory between two peer accessible devices asynchronously.
//
// Returns:
//  - hipErrorNotSupported: Peer-to-peer memory operations are not supported.
//
HIPAPI hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId,
                                     const void* src, int srcDeviceId,
                                     size_t sizeBytes, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* dst_context = NULL;
  hipError_t result =
      iree_hip_primary_context_for_device_id(dstDeviceId, &dst_context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_context_t* src_context = NULL;
  result = iree_hip_primary_context_for_device_id(srcDeviceId, &src_context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (sizeBytes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  // Peer copies name both participating devices explicitly, so the stream is
  // not required to belong to the caller's current context. Resolve it through
  // the process-wide live handle registry and retain it for the operation.
  result = iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (!stream_obj || !stream_obj->context) {
    iree_hal_streaming_stream_release(stream_obj);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorContextIsDestroyed);
  }

  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    hipMemcpy3DParms params;
    memset(&params, 0, sizeof(params));
    params.dstPtr.ptr = dst;
    params.dstPtr.pitch = sizeBytes;
    params.dstPtr.xsize = sizeBytes;
    params.dstPtr.ysize = 1;
    params.srcPtr.ptr = (void*)src;
    params.srcPtr.pitch = sizeBytes;
    params.srcPtr.xsize = sizeBytes;
    params.srcPtr.ysize = 1;
    params.extent.width = sizeBytes;
    params.extent.height = 1;
    params.extent.depth = 1;
    params.kind = hipMemcpyDeviceToDevice;
    result = iree_hip_capture_memcpy3d_node(stream_obj, &params);
    iree_hal_streaming_stream_release(stream_obj);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  result = iree_hip_memcpy_peer_staged(dst_context, dst, src_context, src,
                                       sizeBytes, stream_obj);
  iree_hal_streaming_stream_release(stream_obj);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Copies memory between two peer accessible devices synchronously.
//
// Returns:
//  - hipErrorNotSupported: Peer-to-peer memory operations are not supported.
HIPAPI hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                                int srcDeviceId, size_t sizeBytes) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* dst_context = NULL;
  hipError_t result =
      iree_hip_primary_context_for_device_id(dstDeviceId, &dst_context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_context_t* src_context = NULL;
  result = iree_hip_primary_context_for_device_id(srcDeviceId, &src_context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (sizeBytes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (iree_hip_context_invalidate_visible_captures(src_context) ||
      (dst_context != src_context &&
       iree_hip_context_invalidate_visible_captures(dst_context))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  result = iree_hip_memcpy_peer_staged(dst_context, dst, src_context, src,
                                       sizeBytes, NULL);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

HIPAPI hipError_t hipArrayCreate(hipArray_t* pHandle,
                                 const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  if (!pHandle || !pAllocateArray) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipChannelFormatDesc desc;
  hipError_t result = iree_hip_array_format_to_desc(
      pAllocateArray->Format, pAllocateArray->NumChannels, &desc, NULL);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return iree_hip_array_create(
      pHandle, &desc,
      (hipExtent){pAllocateArray->Width, pAllocateArray->Height, 0},
      hipArrayDefault);
}

HIPAPI hipError_t hipArray3DCreate(
    hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  if (!array || !pAllocateArray) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipChannelFormatDesc desc;
  hipError_t result = iree_hip_array_format_to_desc(
      pAllocateArray->Format, pAllocateArray->NumChannels, &desc, NULL);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return iree_hip_array_create(
      array, &desc,
      (hipExtent){pAllocateArray->Width, pAllocateArray->Height,
                  pAllocateArray->Depth},
      pAllocateArray->Flags);
}

HIPAPI hipError_t hipArrayDestroy(hipArray_t array) {
  return hipFreeArray(array);
}

HIPAPI hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                        hipArray_t array) {
  if (!array) HIP_RETURN_ERROR(hipErrorInvalidHandle);
  if (!pArrayDescriptor) HIP_RETURN_ERROR(hipErrorInvalidValue);
  struct hipArray_st* array_info = NULL;
  hipError_t result =
      iree_hip_array_retain((hipArray_const_t)array, &array_info);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  pArrayDescriptor->Width = array_info->public_extent.width;
  pArrayDescriptor->Height = array_info->public_extent.height;
  pArrayDescriptor->Format = array_info->format;
  pArrayDescriptor->NumChannels = array_info->num_channels;
  iree_hip_array_release(array_info);
  return hipSuccess;
}

HIPAPI hipError_t hipArray3DGetDescriptor(
    HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  if (!array) HIP_RETURN_ERROR(hipErrorInvalidHandle);
  if (!pArrayDescriptor) HIP_RETURN_ERROR(hipErrorInvalidValue);
  struct hipArray_st* array_info = NULL;
  hipError_t result =
      iree_hip_array_retain((hipArray_const_t)array, &array_info);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  pArrayDescriptor->Width = array_info->public_extent.width;
  pArrayDescriptor->Height = array_info->public_extent.height;
  pArrayDescriptor->Depth = array_info->public_extent.depth;
  pArrayDescriptor->Format = array_info->format;
  pArrayDescriptor->NumChannels = array_info->num_channels;
  pArrayDescriptor->Flags = array_info->flags;
  iree_hip_array_release(array_info);
  return hipSuccess;
}

HIPAPI hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent,
                                  unsigned int* flags, hipArray_t array) {
  if (!array) HIP_RETURN_ERROR(hipErrorInvalidHandle);
  if (!desc && !extent && !flags) return hipSuccess;
  struct hipArray_st* array_info = NULL;
  hipError_t result =
      iree_hip_array_retain((hipArray_const_t)array, &array_info);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  if (desc) *desc = array_info->desc;
  if (extent) *extent = array_info->public_extent;
  if (flags) *flags = array_info->flags;
  iree_hip_array_release(array_info);
  return hipSuccess;
}

HIPAPI hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc,
                                    hipArray_const_t array) {
  if (!desc) HIP_RETURN_ERROR(hipErrorInvalidValue);
  struct hipArray_st* array_info = NULL;
  hipError_t result = iree_hip_array_retain(array, &array_info);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  *desc = array_info->desc;
  iree_hip_array_release(array_info);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// IPC Memory Operations (not supported)
//===----------------------------------------------------------------------===//

// Gets an IPC memory handle for a device allocation.
// Not supported - returns hipErrorNotSupported.
HIPAPI hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
  (void)handle;
  (void)devPtr;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Opens an IPC memory handle exported from another process.
// Not supported - returns hipErrorNotSupported.
HIPAPI hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle,
                                      unsigned int flags) {
  (void)devPtr;
  (void)handle;
  (void)flags;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Closes an IPC memory handle.
// Not supported - returns hipErrorNotSupported.
HIPAPI hipError_t hipIpcCloseMemHandle(void* devPtr) {
  (void)devPtr;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Gets an IPC event handle for an event.
// Not supported - returns hipErrorNotSupported.
HIPAPI hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle,
                                       hipEvent_t event) {
  (void)handle;
  (void)event;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Opens an IPC event handle.
// Not supported - returns hipErrorNotSupported.
HIPAPI hipError_t hipIpcOpenEventHandle(hipEvent_t* event,
                                        hipIpcEventHandle_t handle) {
  (void)event;
  (void)handle;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

//===----------------------------------------------------------------------===//
// Symbol Memory Operations
//===----------------------------------------------------------------------===//

// Gets the device address of a symbol (device variable).
//
// Parameters:
//  - devPtr: [OUT] Pointer to receive the device address.
//  - symbol: [IN] Pointer to the host shadow variable.
//
// Returns:
//  - hipSuccess: Address retrieved successfully.
//  - hipErrorInvalidValue: devPtr or symbol is NULL.
//  - hipErrorInvalidSymbol: Symbol not found in registry.
//
// Notes:
// - The symbol must have been registered via __hipRegisterVar.
// - The returned address can be used with hipMemcpy functions.
//
// See also: hipGetSymbolSize, hipMemcpyToSymbol, hipMemcpyFromSymbol.
HIPAPI hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!devPtr || !symbol) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up the symbol in the context's symbol map.
  iree_hal_streaming_symbol_t* sym = NULL;
  iree_status_t status = iree_hal_streaming_context_symbol_map_lookup(
      &context->symbol_map, (void*)symbol, &sym);

  if (!iree_status_is_ok(status) || !sym) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidSymbol);
  }

  // Check that this is a variable (GLOBAL or DATA type).
  if (sym->type != IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL &&
      sym->type != IREE_HAL_STREAMING_SYMBOL_TYPE_DATA) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidSymbol);
  }

  *devPtr = (void*)sym->device_address;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the size of a symbol (device variable).
//
// Parameters:
//  - size: [OUT] Pointer to receive the size in bytes.
//  - symbol: [IN] Pointer to the host shadow variable.
//
// Returns:
//  - hipSuccess: Size retrieved successfully.
//  - hipErrorInvalidValue: size or symbol is NULL.
//  - hipErrorInvalidSymbol: Symbol not found in registry.
//
// See also: hipGetSymbolAddress, hipMemcpyToSymbol, hipMemcpyFromSymbol.
HIPAPI hipError_t hipGetSymbolSize(size_t* size, const void* symbol) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!size || !symbol) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up the symbol in the context's symbol map.
  iree_hal_streaming_symbol_t* sym = NULL;
  iree_status_t status = iree_hal_streaming_context_symbol_map_lookup(
      &context->symbol_map, (void*)symbol, &sym);

  if (!iree_status_is_ok(status) || !sym) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidSymbol);
  }

  if (sym->type != IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL &&
      sym->type != IREE_HAL_STREAMING_SYMBOL_TYPE_DATA) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidSymbol);
  }

  *size = (size_t)sym->size_bytes;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Copies data to a symbol (device variable) asynchronously.
//
// Parameters:
//  - symbol: [IN] Pointer to the host shadow variable.
//  - src: [IN] Source host pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - offset: [IN] Offset in bytes from start of symbol.
//  - kind: [IN] Type of copy (must be hipMemcpyHostToDevice or default).
//  - stream: [IN] Stream for asynchronous execution.
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidSymbol: Symbol not found.
//  - hipErrorInvalidMemcpyDirection: Invalid copy direction.
//
// See also: hipMemcpyToSymbol, hipMemcpyFromSymbolAsync.
HIPAPI hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src,
                                         size_t sizeBytes, size_t offset,
                                         hipMemcpyKind kind,
                                         hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!symbol || !src) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the device address of the symbol.
  void* devPtr = NULL;
  hipError_t result = hipGetSymbolAddress(&devPtr, symbol);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  // Validate the copy direction.
  if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDefault) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidMemcpyDirection);
  }

  // Perform the copy with offset.
  result = hipMemcpyAsync((uint8_t*)devPtr + offset, src, sizeBytes,
                          hipMemcpyHostToDevice, stream);

  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data to a symbol (device variable) synchronously.
//
// Parameters:
//  - symbol: [IN] Pointer to the host shadow variable.
//  - src: [IN] Source host pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - offset: [IN] Offset in bytes from start of symbol.
//  - kind: [IN] Type of copy (default: hipMemcpyHostToDevice).
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidSymbol: Symbol not found.
//  - hipErrorInvalidMemcpyDirection: Invalid copy direction.
//
// Synchronization: This operation is synchronous.
//
// See also: hipMemcpyToSymbolAsync, hipMemcpyFromSymbol.
HIPAPI hipError_t hipMemcpyToSymbol(const void* symbol, const void* src,
                                    size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind) {
  IREE_TRACE_ZONE_BEGIN(z0);

  hipError_t result =
      hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind, NULL);

  if (result == hipSuccess) {
    result = hipDeviceSynchronize();
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data from a symbol (device variable) asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination host pointer.
//  - symbol: [IN] Pointer to the host shadow variable.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - offset: [IN] Offset in bytes from start of symbol.
//  - kind: [IN] Type of copy (must be hipMemcpyDeviceToHost or default).
//  - stream: [IN] Stream for asynchronous execution.
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidSymbol: Symbol not found.
//  - hipErrorInvalidMemcpyDirection: Invalid copy direction.
//
// See also: hipMemcpyFromSymbol, hipMemcpyToSymbolAsync.
HIPAPI hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol,
                                           size_t sizeBytes, size_t offset,
                                           hipMemcpyKind kind,
                                           hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!dst || !symbol) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the device address of the symbol.
  void* devPtr = NULL;
  hipError_t result = hipGetSymbolAddress(&devPtr, symbol);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  // Validate the copy direction.
  if (kind != hipMemcpyDeviceToHost && kind != hipMemcpyDefault) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidMemcpyDirection);
  }

  // Perform the copy with offset.
  result = hipMemcpyAsync(dst, (const uint8_t*)devPtr + offset, sizeBytes,
                          hipMemcpyDeviceToHost, stream);

  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data from a symbol (device variable) synchronously.
//
// Parameters:
//  - dst: [OUT] Destination host pointer.
//  - symbol: [IN] Pointer to the host shadow variable.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - offset: [IN] Offset in bytes from start of symbol.
//  - kind: [IN] Type of copy (default: hipMemcpyDeviceToHost).
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidSymbol: Symbol not found.
//  - hipErrorInvalidMemcpyDirection: Invalid copy direction.
//
// Synchronization: This operation is synchronous.
//
// See also: hipMemcpyFromSymbolAsync, hipMemcpyToSymbol.
HIPAPI hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol,
                                      size_t sizeBytes, size_t offset,
                                      hipMemcpyKind kind) {
  IREE_TRACE_ZONE_BEGIN(z0);

  hipError_t result =
      hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind, NULL);

  if (result == hipSuccess) {
    result = hipDeviceSynchronize();
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

static hipError_t iree_hip_enqueue_memory_fill(
    iree_hal_streaming_context_t* context, hipDeviceptr_t destination,
    iree_device_size_t byte_count, const void* pattern,
    iree_host_size_t pattern_length, hipStream_t stream) {
  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result = iree_hip_resolve_stream(stream, &stream_object);
  if (result != hipSuccess) return result;

  result = iree_hip_order_legacy_stream_dependencies(context, stream_object);
  if (result != hipSuccess) return result;

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)destination, byte_count, pattern,
      pattern_length, stream_object);
  return iree_memset_status_to_hip_result(status);
}

// Sets device memory to a value.
//
// Parameters:
//  - dst: [IN/OUT] Device pointer to memory to set.
//  - value: [IN] Value to set (interpreted as unsigned char).
//  - sizeBytes: [IN] Number of bytes to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dst is NULL or sizeBytes overflows.
//  - hipErrorInvalidDevicePointer: dst is not a valid device pointer.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// Memory behavior:
// - Sets each byte to (unsigned char)(value & 0xFF).
// - Works with any device memory regardless of allocation type.
// - Can set partial allocations.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// Performance note: For large memory regions, consider using hipMemsetAsync
// to overlap with other operations.
//
// See also: hipMemsetAsync, hipMemsetD8, hipMemsetD16, hipMemsetD32.
HIPAPI hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG("[HIP_API] hipMemset(dst=%p, value=%d, size=%zu) ENTRY\n", dst,
                value, sizeBytes);
  if (sizeBytes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dst pointer.
  if (!dst) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  hipError_t dependency_result = iree_hip_order_legacy_stream_dependencies(
      context, context->default_stream);
  if (dependency_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dst, sizeBytes, &value, 1,
      context->default_stream);

  if (iree_status_is_ok(status)) {
    // hipMemset is synchronous - wait for completion on the default stream.
    HIP_DEBUG_LOG("[HIP_API] hipMemset about to sync...\n");
    status = iree_hal_streaming_stream_synchronize(context->default_stream);
    HIP_DEBUG_LOG("[HIP_API] hipMemset sync done\n");
  }

  hipError_t result = iree_memset_status_to_hip_result(status);
  HIP_DEBUG_LOG("[HIP_API] hipMemset EXIT result=%d\n", result);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to a value asynchronously.
//
// Parameters:
//  - dst: [IN/OUT] Device pointer to memory to set.
//  - value: [IN] Value to set (interpreted as unsigned char).
//  - sizeBytes: [IN] Number of bytes to set.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Operation enqueued successfully.
//  - hipErrorInvalidValue: dst is NULL or sizeBytes overflows.
//  - hipErrorInvalidDevicePointer: dst is not a valid device pointer.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Operation is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Executes after all previously enqueued operations in the stream.
// - Subsequent operations in the stream wait for this to complete.
// - Graph capture: Supported. Creates memset node when capturing.
//
// Memory behavior:
// - Sets each byte to (unsigned char)(value & 0xFF).
// - Memory contents are undefined if read before operation completes.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// See also: hipMemset, hipMemsetD8Async, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes,
                                 hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG(
      "[HIP_API] hipMemsetAsync(dst=%p, value=%d, size=%zu, stream=%p) ENTRY\n",
      dst, value, sizeBytes, (void*)stream);
  if (sizeBytes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dst pointer.
  if (!dst) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t result = iree_hip_enqueue_memory_fill(
      context, (hipDeviceptr_t)dst, sizeBytes, &value, 1, stream);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to an 8-bit value.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - uc: [IN] 8-bit value to set.
//  - N: [IN] Number of 8-bit values to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// Memory behavior:
// - Sets N consecutive bytes to the value uc.
// - Total bytes modified: N.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// See also: hipMemset, hipMemsetD8Async, hipMemsetD16, hipMemsetD32.
HIPAPI hipError_t hipMemsetD8(hipDeviceptr_t dstDevice, unsigned char uc,
                              size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (N == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dstDevice pointer.
  if (!dstDevice) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  hipError_t dependency_result = iree_hip_order_legacy_stream_dependencies(
      context, context->default_stream);
  if (dependency_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N, &uc, 1,
      context->default_stream);

  if (iree_status_is_ok(status)) {
    // hipMemsetD8 is synchronous - wait for completion on the default stream.
    status = iree_hal_streaming_stream_synchronize(context->default_stream);
  }

  hipError_t result = iree_memset_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to a 16-bit value.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - us: [IN] 16-bit value to set.
//  - N: [IN] Number of 16-bit values to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// Memory behavior:
// - Sets N consecutive 16-bit values to us.
// - Total bytes modified: N * 2.
// - Memory must be 2-byte aligned.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// Warning: dstDevice must be aligned to 2-byte boundary.
//
// See also: hipMemset, hipMemsetD16Async, hipMemsetD8, hipMemsetD32.
HIPAPI hipError_t hipMemsetD16(hipDeviceptr_t dstDevice, unsigned short us,
                               size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (N == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dstDevice pointer.
  if (!dstDevice) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  iree_host_size_t byte_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(N, 2, &byte_count))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t dependency_result = iree_hip_order_legacy_stream_dependencies(
      context, context->default_stream);
  if (dependency_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, byte_count, &us, 2,
      context->default_stream);

  if (iree_status_is_ok(status)) {
    // hipMemsetD16 is synchronous - wait for completion on the default stream.
    status = iree_hal_streaming_stream_synchronize(context->default_stream);
  }

  hipError_t result = iree_memset_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to a 32-bit value.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - i: [IN] 32-bit value to set.
//  - N: [IN] Number of 32-bit values to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// Memory behavior:
// - Sets N consecutive 32-bit values to i.
// - Total bytes modified: N * 4.
// - Memory must be 4-byte aligned.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// Warning: dstDevice must be aligned to 4-byte boundary.
//
// See also: hipMemset, hipMemsetD32Async, hipMemsetD8, hipMemsetD16.
HIPAPI hipError_t hipMemsetD32(hipDeviceptr_t dstDevice, int i, size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (N == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dstDevice pointer.
  if (!dstDevice) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_visible_captures(context)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  iree_host_size_t byte_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(N, 4, &byte_count))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t dependency_result = iree_hip_order_legacy_stream_dependencies(
      context, context->default_stream);
  if (dependency_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, byte_count, &i, 4,
      context->default_stream);

  if (iree_status_is_ok(status)) {
    // hipMemsetD32 is synchronous - wait for completion on the default stream.
    status = iree_hal_streaming_stream_synchronize(context->default_stream);
  }

  hipError_t result = iree_memset_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to an 8-bit value asynchronously.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - uc: [IN] 8-bit value to set.
//  - N: [IN] Number of 8-bit values to set.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Operation enqueued successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Operation is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Memory contents are undefined if read before operation completes.
// - Graph capture: Supported. Creates memset node when capturing.
//
// Memory behavior:
// - Sets N consecutive bytes to the value uc.
// - Total bytes modified: N.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// See also: hipMemsetD8, hipMemsetAsync, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetD8Async(hipDeviceptr_t dstDevice, unsigned char uc,
                                   size_t N, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (N == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dstDevice pointer.
  if (!dstDevice) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t result =
      iree_hip_enqueue_memory_fill(context, dstDevice, N, &uc, 1, stream);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Asynchronously sets memory to a 16-bit value.
//
// Parameters:
//  - dstDevice: [IN] Device memory pointer to set.
//  - us: [IN] 16-bit value to set in each element.
//  - N: [IN] Number of 16-bit elements to set.
//  - stream: [IN] Stream for asynchronous operation.
//
// Returns:
//  - hipSuccess: Memory set operation queued successfully.
//  - hipErrorInvalidValue: Invalid pointer or count.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorUnknown: Internal error.
//
// Synchronization: This operation is asynchronous and completes
// when the stream reaches this operation.
//
// Memory behavior:
// - Sets N consecutive 16-bit values starting at dstDevice.
// - The destination address must be 2-byte aligned.
// - Total bytes written: N * sizeof(uint16_t).
// - Pattern is replicated as 16-bit values, not bytes.
//
// Multi-GPU: Memory must be accessible from the current device.
//
// Performance notes:
// - May be optimized for aligned addresses and specific patterns.
// - Large fills may be broken into multiple operations internally.
// - Consider coalescing multiple small fills for better performance.
// - Graph capture: Supported. Creates memset node when capturing.
//
// Warning: Ensure proper alignment for 16-bit access. Unaligned
// access may cause performance degradation or errors.
//
// See also: hipMemsetD16, hipMemsetD8Async, hipMemsetD32Async,
//           hipMemsetAsync, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetD16Async(hipDeviceptr_t dstDevice, unsigned short us,
                                    size_t N, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (N == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dstDevice pointer.
  if (!dstDevice) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_host_size_t byte_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(N, 2, &byte_count))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t result = iree_hip_enqueue_memory_fill(context, dstDevice,
                                                   byte_count, &us, 2, stream);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Asynchronously sets memory to a 32-bit value.
//
// Parameters:
//  - dstDevice: [IN] Device memory pointer to set.
//  - i: [IN] 32-bit value to set in each element.
//  - N: [IN] Number of 32-bit elements to set.
//  - stream: [IN] Stream for asynchronous operation.
//
// Returns:
//  - hipSuccess: Memory set operation queued successfully.
//  - hipErrorInvalidValue: Invalid pointer or count.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorUnknown: Internal error.
//
// Synchronization: This operation is asynchronous and completes
// when the stream reaches this operation.
//
// Memory behavior:
// - Sets N consecutive 32-bit values starting at dstDevice.
// - The destination address must be 4-byte aligned.
// - Total bytes written: N * sizeof(int32_t).
// - Pattern is replicated as 32-bit values, not bytes.
//
// Multi-GPU: Memory must be accessible from the current device.
//
// Performance notes:
// - May be optimized for aligned addresses and specific patterns.
// - Large fills may be broken into multiple operations internally.
// - Consider coalescing multiple small fills for better performance.
// - Often most efficient for clearing large buffers.
// - Graph capture: Supported. Creates memset node when capturing.
//
// Warning: Ensure proper alignment for 32-bit access. Unaligned
// access may cause performance degradation or errors.
//
// See also: hipMemsetD32, hipMemsetD8Async, hipMemsetD16Async,
//           hipMemsetAsync, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetD32Async(hipDeviceptr_t dstDevice, int i, size_t N,
                                    hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (N == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Validate dstDevice pointer.
  if (!dstDevice) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_host_size_t byte_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(N, 4, &byte_count))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t result = iree_hip_enqueue_memory_fill(context, dstDevice,
                                                   byte_count, &i, 4, stream);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

// Creates a new asynchronous stream for independent command execution.
//
// Parameters:
//  - stream: [OUT] Pointer to receive the created stream handle.
//
// Returns:
//  - hipSuccess: Stream created successfully.
//  - hipErrorInvalidValue: stream pointer is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create stream.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during stream creation.
//
// Synchronization: This operation is synchronous.
//
// Stream behavior:
// - The new stream is independent of the default stream.
// - Operations on different streams may execute concurrently.
// - Operations within a stream execute in order.
// - Use hipStreamSynchronize() to wait for all operations to complete.
// - Use hipStreamQuery() to check completion status without blocking.
// - Stream must be destroyed with hipStreamDestroy() when no longer needed.
//
// Multi-GPU: The stream is associated with the current device context.
//
// Note: NULL stream (0) represents the default stream which synchronizes
// with all other streams on the device.
//
// See also: hipStreamCreateWithFlags, hipStreamCreateWithPriority,
//           hipStreamDestroy, hipStreamSynchronize.
HIPAPI hipError_t hipStreamCreate(hipStream_t* stream) {
  HIP_DEBUG_LOG("[HIP_API] hipStreamCreate(stream=%p)\n", (void*)stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, context->host_allocator,
      &stream_obj);

  if (iree_status_is_ok(status)) {
    status = iree_hip_stream_register(stream_obj);
  }
  if (iree_status_is_ok(status)) {
    *stream = (hipStream_t)stream_obj;
  } else {
    iree_hip_stream_discard_unpublished(stream_obj);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Creates a new asynchronous stream with specified flags.
//
// Parameters:
//  - stream: [OUT] Pointer to receive the created stream handle.
//  - flags: [IN] Stream creation flags (hipStreamDefault or
//                hipStreamNonBlocking).
//
// Returns:
//  - hipSuccess: Stream created successfully.
//  - hipErrorInvalidValue: stream pointer is NULL or invalid flags.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create stream.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during stream creation.
//
// Synchronization: This operation is synchronous.
//
// Stream flags:
// - hipStreamDefault (0): Default stream behavior.
// - hipStreamNonBlocking: Stream does not synchronize with NULL stream.
//
// Stream behavior:
// - The new stream is independent of other streams.
// - Operations on different streams may execute concurrently.
// - Operations within a stream execute in order.
// - Non-blocking streams do not implicitly synchronize with the NULL
//   stream, allowing better overlap of host and device execution.
// - Stream must be destroyed with hipStreamDestroy() when no longer needed.
//
// Multi-GPU: Stream is created on the current device and can only
// execute operations on that device.
//
// Warning: Using non-blocking streams requires careful synchronization
// to avoid race conditions with host code.
//
// See also: hipStreamCreate, hipStreamCreateWithPriority,
//           hipStreamDestroy, hipStreamSynchronize.
HIPAPI hipError_t hipStreamCreateWithFlags(hipStream_t* stream,
                                           unsigned int flags) {
  HIP_DEBUG_LOG("[HIP_API] hipStreamCreateWithFlags(stream=%p, flags=%u)\n",
                (void*)stream, flags);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_stream_create_flags_are_valid(flags)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, iree_hip_stream_flags_to_internal(flags), 0,
      context->host_allocator, &stream_obj);

  if (iree_status_is_ok(status)) {
    status = iree_hip_stream_register(stream_obj);
  }
  if (iree_status_is_ok(status)) {
    *stream = (hipStream_t)stream_obj;
  } else {
    iree_hip_stream_discard_unpublished(stream_obj);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Creates a new asynchronous stream with specified flags and priority.
//
// Parameters:
//  - stream: [OUT] Pointer to receive the created stream handle.
//  - flags: [IN] Stream creation flags (hipStreamDefault or
//                hipStreamNonBlocking).
//  - priority: [IN] Stream priority (higher values = higher priority).
//
// Returns:
//  - hipSuccess: Stream created successfully.
//  - hipErrorInvalidValue: stream pointer is NULL, invalid flags, or
//                          priority outside valid range.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create stream.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during stream creation.
//
// Synchronization: This operation is synchronous.
//
// Priority behavior:
// - Higher numerical values indicate higher priority.
// - Priority range can be queried with hipDeviceGetStreamPriorityRange().
// - Work in higher priority streams may preempt lower priority streams.
// - Priorities are hints; actual scheduling depends on hardware support.
// - If priorities are not supported, all streams execute with equal
//   priority.
//
// Stream behavior:
// - The new stream is independent of other streams.
// - Operations on different streams may execute concurrently.
// - Operations within a stream execute in order.
// - Non-blocking streams do not implicitly synchronize with NULL stream.
// - Stream must be destroyed with hipStreamDestroy() when no longer needed.
//
// Multi-GPU: Stream is created on the current device and can only
// execute operations on that device.
//
// Warning: Priority scheduling is a performance hint and may not be
// supported on all devices. Check device capabilities before relying
// on priority behavior.
//
// See also: hipStreamCreate, hipStreamCreateWithFlags,
//           hipDeviceGetStreamPriorityRange, hipStreamGetPriority.
HIPAPI hipError_t hipStreamCreateWithPriority(hipStream_t* stream,
                                              unsigned int flags,
                                              int priority) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_stream_create_flags_are_valid(flags)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, iree_hip_stream_flags_to_internal(flags),
      iree_hip_clamp_stream_priority(priority), context->host_allocator,
      &stream_obj);

  if (iree_status_is_ok(status)) {
    status = iree_hip_stream_register(stream_obj);
  }
  if (iree_status_is_ok(status)) {
    *stream = (hipStream_t)stream_obj;
  } else {
    iree_hip_stream_discard_unpublished(stream_obj);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Destroys a stream previously created with hipStreamCreate.
//
// Parameters:
//  - stream: [IN] Stream handle to destroy.
//
// Returns:
//  - hipSuccess: Stream destroyed successfully.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorContextIsDestroyed: Associated context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for all pending
// operations in the stream to complete before destroying.
//
// Stream behavior:
// - All operations in the stream must complete before destruction.
// - After destruction, the stream handle becomes invalid.
// - Using a destroyed stream results in undefined behavior.
//
// Multi-GPU: Stream must be destroyed from the same context that created it.
//
// Warning: Ensure all operations using this stream have completed. The
// function implicitly synchronizes the stream before destruction.
//
// See also: hipStreamCreate, hipStreamCreateWithFlags,
//           hipStreamCreateWithPriority.
HIPAPI hipError_t hipStreamDestroy(hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidResourceHandle);
  }

  iree_hal_streaming_stream_t* streaming_stream = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &streaming_stream);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_status_t status =
      iree_hal_streaming_stream_synchronize(streaming_stream);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_stream_release(streaming_stream);
    result = iree_status_to_hip_result(status);
    IREE_TRACE_ZONE_END(z0);
    return result;
  }
  if (!iree_hip_stream_unregister(streaming_stream)) {
    iree_hal_streaming_stream_release(streaming_stream);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidResourceHandle);
  }

  // Removing the handle prevents new API calls from retaining the stream.
  // Keep the context association intact for operations that already retained
  // the stream, including context-wide synchronization snapshots. Context
  // teardown detaches any streams that outlive their owning context.
  iree_hal_streaming_context_t* context = streaming_stream->context;
  iree_hal_streaming_context_unregister_stream(context, streaming_stream);
  // Release the registry lookup reference and the public handle reference.
  iree_hal_streaming_stream_release(streaming_stream);
  iree_hal_streaming_stream_release(streaming_stream);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Queries the priority of a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//  - priority: [OUT] Pointer to receive the stream priority.
//
// Returns:
//  - hipSuccess: Priority queried successfully.
//  - hipErrorInvalidValue: priority pointer is NULL.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//
// Synchronization: This operation is synchronous and immediate.
//
// Priority behavior:
// - Returns the priority assigned when the stream was created.
//  - Default streams have priority 0.
// - Higher numerical values indicate higher priority.
// - If priorities are not supported, returns 0.
//
// Multi-GPU: Queries the stream associated with the current context.
//
// See also: hipStreamCreateWithPriority, hipStreamGetFlags,
//           hipDeviceGetStreamPriorityRange.
HIPAPI hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  if (!priority) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  *priority = stream_obj->priority;
  iree_hal_streaming_stream_release(stream_obj);
  return hipSuccess;
}

// Queries the flags of a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//  - flags: [OUT] Pointer to receive the stream flags.
//
// Returns:
//  - hipSuccess: Flags queried successfully.
//  - hipErrorInvalidValue: flags pointer is NULL.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//
// Synchronization: This operation is synchronous and immediate.
//
// Flag values:
// - hipStreamDefault (0): Default stream behavior.
// - hipStreamNonBlocking: Stream does not synchronize with NULL stream.
//
// Stream behavior:
// - Returns the flags specified when the stream was created.
// - Default streams return hipStreamDefault.
// - Flags determine synchronization behavior with NULL stream.
//
// Multi-GPU: Queries the stream associated with the current context.
//
// See also: hipStreamCreateWithFlags, hipStreamGetPriority,
//           hipStreamGetDevice.
HIPAPI hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  if (!flags) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!stream) {
    hipError_t init_result = iree_hip_ensure_initialized();
    if (init_result != hipSuccess) {
      HIP_RETURN_ERROR(init_result);
    }
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  *flags = stream_obj->flags;
  iree_hal_streaming_stream_release(stream_obj);
  return hipSuccess;
}

// Queries the device associated with a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//  - device: [OUT] Pointer to receive the device ordinal.
//
// Returns:
//  - hipSuccess: Device queried successfully.
//  - hipErrorInvalidValue: device pointer is NULL.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//
// Synchronization: This operation is synchronous and immediate.
//
// Device association:
// - Returns the device ordinal where the stream was created.
// - Streams can only execute operations on their associated device.
// - Device association is fixed for the lifetime of the stream.
//
// Multi-GPU:
// - Each stream is bound to a specific device.
// - Operations queued to a stream execute on its associated device.
// - Use hipSetDevice() before creating streams to control placement.
//
// See also: hipStreamCreate, hipGetDevice, hipSetDevice,
//           hipStreamGetFlags.
HIPAPI hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  *device = (hipDevice_t)stream_obj->context->device_ordinal;
  iree_hal_streaming_stream_release(stream_obj);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

HIPAPI hipError_t hipStreamGetAttribute(hipStream_t stream,
                                        hipStreamAttrID attribute,
                                        hipStreamAttrValue* value_out) {
  if (attribute != hipStreamAttributeSynchronizationPolicy || !value_out) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &stream_object);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);

  iree_slim_mutex_lock(&stream_object->mutex);
  value_out->syncPolicy =
      (hipSynchronizationPolicy)stream_object->synchronization_policy;
  iree_slim_mutex_unlock(&stream_object->mutex);
  iree_hal_streaming_stream_release(stream_object);
  return hipSuccess;
}

HIPAPI hipError_t hipStreamSetAttribute(hipStream_t stream,
                                        hipStreamAttrID attribute,
                                        const hipStreamAttrValue* value) {
  if (attribute != hipStreamAttributeSynchronizationPolicy || !value ||
      !iree_hip_synchronization_policy_is_valid(value->syncPolicy)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &stream_object);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);

  iree_slim_mutex_lock(&stream_object->mutex);
  stream_object->synchronization_policy =
      (iree_hal_streaming_synchronization_policy_t)value->syncPolicy;
  iree_slim_mutex_unlock(&stream_object->mutex);
  iree_hal_streaming_stream_release(stream_object);
  return hipSuccess;
}

HIPAPI hipError_t hipStreamCopyAttributes(hipStream_t destination,
                                          hipStream_t source) {
  iree_hal_streaming_stream_t* source_stream = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(source, &source_stream);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);

  iree_hal_streaming_stream_t* destination_stream = NULL;
  result = iree_hip_resolve_registered_stream(destination, &destination_stream);
  if (result != hipSuccess) {
    iree_hal_streaming_stream_release(source_stream);
    HIP_RETURN_ERROR(result);
  }

  iree_slim_mutex_lock(&source_stream->mutex);
  const iree_hal_streaming_synchronization_policy_t synchronization_policy =
      source_stream->synchronization_policy;
  iree_slim_mutex_unlock(&source_stream->mutex);

  iree_slim_mutex_lock(&destination_stream->mutex);
  destination_stream->synchronization_policy = synchronization_policy;
  iree_slim_mutex_unlock(&destination_stream->mutex);
  iree_hal_streaming_stream_release(destination_stream);
  iree_hal_streaming_stream_release(source_stream);
  return hipSuccess;
}

HIPAPI hipError_t hipStreamGetId(hipStream_t stream,
                                 unsigned long long* stream_id) {
  if (!stream_id) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  *stream_id = stream_obj->stream_id;
  iree_hal_streaming_stream_release(stream_obj);
  return hipSuccess;
}

// Returns the device ID associated with a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//
// Returns:
//  - Device ordinal (>= 0) on success.
//  - -1 on error (sets last error).
//
// Synchronization: This operation is synchronous and immediate.
//
// Threading: Thread-safe.
//
// Multi-GPU Notes:
// - Each stream is bound to a specific device at creation time.
// - The NULL stream is bound to the current context's device.
//
// See also: hipStreamGetDevice, hipGetDevice, hipSetDevice.
HIPAPI int hipGetStreamDeviceId(hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    iree_hip_thread_error_set(init_result, false);
    return -1;
  }

  int device_id = (int)stream_obj->context->device_ordinal;
  iree_hal_streaming_stream_release(stream_obj);

  IREE_TRACE_ZONE_END(z0);
  return device_id;
}

// Waits for all operations in a stream to complete.
//
// Parameters:
//  - stream: [IN] Stream to synchronize (NULL = default stream).
//
// Returns:
//  - hipSuccess: All operations completed successfully.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: A kernel launch in the stream failed.
//  - hipErrorIllegalAddress: Invalid memory access in stream operations.
//  - hipErrorUnknown: Internal error during synchronization.
//
// Synchronization: This operation blocks the host thread until all
// previously enqueued operations in the stream have completed.
//
// Stream behavior:
// - If stream is NULL, synchronizes the default stream.
// - Blocks until all operations enqueued before this call complete.
// - Operations enqueued after this call are not affected.
// - Does not synchronize with other streams.
//
// Multi-GPU: Synchronizes operations on the device associated with the
// stream.
//
// Performance note: Consider using hipStreamQuery for non-blocking status
// checks or hipEventSynchronize for finer-grained synchronization.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// See also: hipStreamQuery, hipDeviceSynchronize, hipEventSynchronize.
HIPAPI hipError_t hipStreamSynchronize(hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG("[HIP_API] hipStreamSynchronize(stream=%p) called\n",
                (void*)stream);

  iree_hal_streaming_stream_t* streaming_stream = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &streaming_stream);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (!stream || stream == hipStreamLegacy) {
    iree_hal_streaming_context_t* context = streaming_stream->context;
    if (iree_hip_context_invalidate_visible_captures(context)) {
      iree_hal_streaming_stream_release(streaming_stream);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
    }

    iree_status_t status =
        iree_hal_streaming_context_synchronize_legacy_default(context);
    iree_hal_streaming_stream_release(streaming_stream);
    hipError_t result = iree_status_to_hip_result(status);
    IREE_TRACE_ZONE_END(z0);
    return result;
  }

  if (iree_hip_context_invalidate_stream_blocking_capture(
          streaming_stream->context, streaming_stream)) {
    iree_hal_streaming_stream_release(streaming_stream);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_status_t status =
      iree_hal_streaming_stream_synchronize(streaming_stream);
  iree_hal_streaming_stream_release(streaming_stream);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Queries the completion status of operations in a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//
// Returns:
//  - hipSuccess: All operations in the stream have completed.
//  - hipErrorNotReady: Operations are still in progress.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: A kernel launch in the stream failed.
//
// Synchronization: This operation is non-blocking. Returns immediately with
// the current status.
//
// Stream behavior:
// - If stream is NULL, queries the default stream.
// - Checks if all operations enqueued before this call have completed.
// - Does not wait for operations to complete.
// - Does not affect operations in other streams.
//
// Multi-GPU: Queries operations on the device associated with the stream.
//
// Usage pattern:
// ```c
// while (hipStreamQuery(stream) == hipErrorNotReady) {
//   // Do other work while waiting
// }
// ```
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
//
// See also: hipStreamSynchronize, hipEventQuery, hipDeviceSynchronize.
HIPAPI hipError_t hipStreamQuery(hipStream_t stream) {
  iree_hal_streaming_stream_t* streaming_stream = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &streaming_stream);
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  if (iree_hip_context_invalidate_stream_blocking_capture(
          streaming_stream->context, streaming_stream)) {
    iree_hal_streaming_stream_release(streaming_stream);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  int is_complete = 0;
  iree_status_t status = iree_ok_status();
  if (!stream || stream == hipStreamLegacy) {
    status = iree_hal_streaming_context_query(streaming_stream->context,
                                              &is_complete);
  } else {
    status = iree_hal_streaming_stream_query(streaming_stream, &is_complete);
  }
  iree_hal_streaming_stream_release(streaming_stream);
  // is_complete == 0 means complete, is_complete == 1 means not complete.
  hipError_t result = iree_status_is_ok(status)
                          ? (is_complete == 0 ? hipSuccess : hipErrorNotReady)
                          : iree_status_to_hip_result(status);
  return result;
}

// Makes a stream wait for an event to complete.
//
// Parameters:
//  - stream: [IN] Stream that will wait (NULL = default stream).
//  - event: [IN] Event to wait for.
//  - flags: [IN] Reserved for future use (must be 0).
//
// Returns:
//  - hipSuccess: Wait dependency added successfully.
//  - hipErrorInvalidResourceHandle: Invalid stream or event handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidValue: Invalid flags or NULL event.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous. Adds a dependency to the
// stream but does not block the host.
//
// Stream behavior:
// - If stream is NULL, uses the default stream.
// - The event can be from the same or different device.
// - Graph capture: Supported. Creates wait node when capturing.
// - Stream will wait for the event before executing subsequent operations.
// - Does not affect operations already enqueued in the stream.
// - The event can be from the same or different stream.
// - The event can be from the same or different device.
//
// Multi-GPU: Enables cross-device synchronization when event is from a
// different device.
//
// Usage pattern:
// ```c
// hipEventRecord(event, stream1);
// hipStreamWaitEvent(stream2, event, 0);  // stream2 waits for stream1
// ```
//
// See also: hipEventRecord, hipEventSynchronize, hipStreamSynchronize.
HIPAPI hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                                     unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (flags != 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_event_t* event_object = NULL;
  hipError_t event_result = iree_hip_event_lookup_retain(event, &event_object);
  if (event_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(event_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    iree_hal_streaming_event_release(event_object);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status =
      iree_hal_streaming_stream_wait_event(stream_obj, event_object);
  iree_hal_streaming_stream_release(stream_obj);
  iree_hal_streaming_event_release(event_object);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Stream memory operations
//===----------------------------------------------------------------------===//

enum {
  IREE_HIP_STREAM_WAIT_VALUE_GTE = 0x0,
  IREE_HIP_STREAM_WAIT_VALUE_EQ = 0x1,
  IREE_HIP_STREAM_WAIT_VALUE_AND = 0x2,
  IREE_HIP_STREAM_WAIT_VALUE_NOR = 0x3,
  IREE_HIP_STREAM_WRITE_VALUE_DEFAULT = 0x0,
  IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT = 0x1000,
  IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT = 0x1001,
};

typedef struct iree_hip_stream_value_target_t {
  // Context retaining the allocation registry while the operation is built.
  iree_hal_streaming_context_t* owner_context;
  // Allocation and byte offset resolved from the caller's device pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
} iree_hip_stream_value_target_t;

typedef enum iree_hip_stream_value_operation_kind_e {
  IREE_HIP_STREAM_VALUE_OPERATION_WAIT = 0,
  IREE_HIP_STREAM_VALUE_OPERATION_WRITE = 1,
} iree_hip_stream_value_operation_kind_t;

typedef struct iree_hip_stream_value_operation_t {
  // Operation executed at this position in the batch.
  iree_hip_stream_value_operation_kind_t kind;
  // Resolved allocation retained until the complete batch is enqueued.
  iree_hip_stream_value_target_t target;
  // Comparison or replacement value supplied by the caller.
  uint64_t value;
  // Mask applied by wait operations.
  uint64_t mask;
  // Condition or update behavior flags for the operation.
  unsigned int flags;
  // Width of the memory value in bytes.
  iree_host_size_t byte_length;
} iree_hip_stream_value_operation_t;

typedef struct iree_hip_stream_value_write_t {
  // Resource retained until the stream-ordered host write completes.
  iree_hal_resource_t resource;
  // Context retaining the allocation registry and its imported host memory.
  iree_hal_streaming_context_t* context;
  // HAL allocation retained while the write is pending.
  iree_hal_buffer_t* buffer;
  // Host address updated by the stream-ordered callback.
  void* host_pointer;
  // Value supplied by the caller, narrowed to |byte_length|.
  uint64_t value;
  // Width of the value in bytes.
  iree_host_size_t byte_length;
  // Operation selecting assignment, atomic addition, or atomic subtraction.
  unsigned int flags;
} iree_hip_stream_value_write_t;

static void iree_hip_stream_value_write_destroy(
    iree_hal_resource_t* base_resource) {
  iree_hip_stream_value_write_t* write =
      (iree_hip_stream_value_write_t*)base_resource;
  iree_hal_buffer_release(write->buffer);
  iree_hal_streaming_context_release(write->context);
  iree_allocator_free(iree_allocator_system(), write);
}

static const iree_hal_resource_vtable_t iree_hip_stream_value_write_vtable = {
    .destroy = iree_hip_stream_value_write_destroy,
};

static bool iree_hip_stream_write_value_flags_are_valid(unsigned int flags) {
  return flags == IREE_HIP_STREAM_WRITE_VALUE_DEFAULT ||
         flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT ||
         flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT;
}

static bool iree_hip_stream_wait_value_flags_are_valid(unsigned int flags) {
  return flags == IREE_HIP_STREAM_WAIT_VALUE_GTE ||
         flags == IREE_HIP_STREAM_WAIT_VALUE_EQ ||
         flags == IREE_HIP_STREAM_WAIT_VALUE_AND ||
         flags == IREE_HIP_STREAM_WAIT_VALUE_NOR;
}

static void iree_hip_stream_value_target_deinitialize(
    iree_hip_stream_value_target_t* target) {
  iree_hal_streaming_context_release(target->owner_context);
  memset(target, 0, sizeof(*target));
}

static hipError_t iree_hip_stream_value_target_initialize(
    iree_hal_streaming_stream_t* stream, const void* ptr,
    iree_host_size_t byte_length, iree_hip_stream_value_target_t* out_target) {
  memset(out_target, 0, sizeof(*out_target));
  if (!ptr || (uintptr_t)ptr % byte_length != 0) {
    return hipErrorInvalidValue;
  }
  hipError_t result = iree_hip_lookup_streaming_range_with_owner(
      stream->context, ptr, byte_length, &out_target->owner_context,
      &out_target->buffer_ref);
  if (result != hipSuccess || !out_target->buffer_ref.buffer) {
    iree_hip_stream_value_target_deinitialize(out_target);
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static iree_status_t iree_hip_stream_value_host_write(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* call_context) {
  (void)args;
  (void)call_context;
  iree_hip_stream_value_write_t* write =
      (iree_hip_stream_value_write_t*)user_data;
  if (write->byte_length == sizeof(uint32_t)) {
    uint32_t* target = (uint32_t*)write->host_pointer;
    const uint32_t value = (uint32_t)write->value;
    if (write->flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT) {
      __atomic_fetch_add(target, value, __ATOMIC_ACQ_REL);
    } else if (write->flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT) {
      __atomic_fetch_sub(target, value, __ATOMIC_ACQ_REL);
    } else {
      __atomic_store_n(target, value, __ATOMIC_RELEASE);
    }
  } else {
    uint64_t* target = (uint64_t*)write->host_pointer;
    if (write->flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT) {
      __atomic_fetch_add(target, write->value, __ATOMIC_ACQ_REL);
    } else if (write->flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT) {
      __atomic_fetch_sub(target, write->value, __ATOMIC_ACQ_REL);
    } else {
      __atomic_store_n(target, write->value, __ATOMIC_RELEASE);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hip_enqueue_stream_value_write_resolved(
    iree_hal_streaming_stream_t* stream,
    const iree_hip_stream_value_target_t* target, uint64_t value,
    unsigned int flags, iree_host_size_t byte_length) {
  if (!target->buffer_ref.buffer->host_ptr) {
    if (flags != IREE_HIP_STREAM_WRITE_VALUE_DEFAULT) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "atomic stream value writes require host-visible memory");
    }
    return iree_hal_streaming_memcpy_host_to_device(
        target->owner_context,
        target->buffer_ref.buffer->device_ptr + target->buffer_ref.offset,
        &value, byte_length, stream);
  }

  iree_hip_stream_value_write_t* write = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             sizeof(*write), (void**)&write));
  iree_hal_resource_initialize(&iree_hip_stream_value_write_vtable,
                               &write->resource);
  write->context = target->owner_context;
  iree_hal_streaming_context_retain(write->context);
  write->buffer = target->buffer_ref.buffer->buffer;
  write->host_pointer =
      (uint8_t*)target->buffer_ref.buffer->host_ptr + target->buffer_ref.offset;
  write->value = value;
  write->byte_length = byte_length;
  write->flags = flags;
  iree_hal_buffer_retain(write->buffer);

  const uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call = iree_hal_make_host_call_with_resource(
      iree_hip_stream_value_host_write, write, &write->resource);
  iree_status_t status = iree_hal_streaming_queue_host_call(
      stream, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);
  iree_hal_resource_release(&write->resource);
  return status;
}

static hipError_t iree_hip_enqueue_stream_value_write(
    hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
    iree_host_size_t byte_length) {
  if (!ptr || !iree_hip_stream_write_value_flags_are_valid(flags)) {
    return hipErrorInvalidValue;
  }
  if ((uintptr_t)ptr % byte_length != 0) return hipErrorInvalidValue;

  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &stream_object);
  if (result != hipSuccess) return hipErrorContextIsDestroyed;

  iree_hip_stream_value_target_t target;
  result = iree_hip_stream_value_target_initialize(stream_object, ptr,
                                                   byte_length, &target);
  if (result != hipSuccess) {
    iree_hal_streaming_stream_release(stream_object);
    return result;
  }

  iree_status_t status = iree_hip_enqueue_stream_value_write_resolved(
      stream_object, &target, value, flags, byte_length);
  iree_hip_stream_value_target_deinitialize(&target);
  iree_hal_streaming_stream_release(stream_object);
  return iree_status_to_hip_result(status);
}

static iree_hal_amdgpu_wait_value_condition_t
iree_hip_stream_wait_value_condition(unsigned int flags) {
  switch (flags) {
    case IREE_HIP_STREAM_WAIT_VALUE_EQ:
      return IREE_HAL_AMDGPU_WAIT_VALUE_CONDITION_EQUAL;
    case IREE_HIP_STREAM_WAIT_VALUE_AND:
      return IREE_HAL_AMDGPU_WAIT_VALUE_CONDITION_BITWISE_AND;
    case IREE_HIP_STREAM_WAIT_VALUE_NOR:
      return IREE_HAL_AMDGPU_WAIT_VALUE_CONDITION_BITWISE_NOR;
    case IREE_HIP_STREAM_WAIT_VALUE_GTE:
    default:
      return IREE_HAL_AMDGPU_WAIT_VALUE_CONDITION_GREATER_THAN_OR_EQUAL;
  }
}

static iree_status_t iree_hip_enqueue_stream_value_wait_resolved(
    iree_hal_streaming_stream_t* stream,
    const iree_hip_stream_value_target_t* target, uint64_t value,
    unsigned int flags, uint64_t mask, iree_host_size_t byte_length) {
  return iree_hal_streaming_queue_wait_value(
      stream, target->buffer_ref.buffer->buffer, target->buffer_ref.offset,
      value, mask, byte_length, iree_hip_stream_wait_value_condition(flags),
      IREE_HAL_AMDGPU_WAIT_VALUE_FLAG_NONE,
      iree_hal_amdgpu_device_queue_wait_value);
}

static hipError_t iree_hip_enqueue_stream_value_wait(
    hipStream_t stream, const void* ptr, uint64_t value, unsigned int flags,
    uint64_t mask, iree_host_size_t byte_length) {
  if (!ptr || !iree_hip_stream_wait_value_flags_are_valid(flags)) {
    return hipErrorInvalidValue;
  }
  if ((uintptr_t)ptr % byte_length != 0) return hipErrorInvalidValue;

  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &stream_object);
  if (result != hipSuccess) return hipErrorContextIsDestroyed;

  iree_hip_stream_value_target_t target;
  result = iree_hip_stream_value_target_initialize(stream_object, ptr,
                                                   byte_length, &target);
  if (result != hipSuccess) {
    iree_hal_streaming_stream_release(stream_object);
    return result;
  }

  iree_status_t status = iree_hip_enqueue_stream_value_wait_resolved(
      stream_object, &target, value, flags, mask, byte_length);
  iree_hip_stream_value_target_deinitialize(&target);
  iree_hal_streaming_stream_release(stream_object);
  return iree_status_to_hip_result(status);
}

// Writes a 32-bit value to memory as part of stream execution.
HIPAPI hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr,
                                        uint32_t value, unsigned int flags) {
  return iree_hip_enqueue_stream_value_write(stream, ptr, value, flags,
                                             sizeof(value));
}

// Writes a 64-bit value to device memory as part of stream execution.
HIPAPI hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr,
                                        uint64_t value, unsigned int flags) {
  return iree_hip_enqueue_stream_value_write(stream, ptr, value, flags,
                                             sizeof(value));
}

// Waits until a 32-bit value meets a condition as part of stream execution.
HIPAPI hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr,
                                       uint32_t value, unsigned int flags,
                                       uint32_t mask) {
  return iree_hip_enqueue_stream_value_wait(stream, ptr, value, flags, mask,
                                            sizeof(value));
}

// Waits until a 64-bit value meets a condition as part of stream execution.
HIPAPI hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr,
                                       uint64_t value, unsigned int flags,
                                       uint64_t mask) {
  return iree_hip_enqueue_stream_value_wait(stream, ptr, value, flags, mask,
                                            sizeof(value));
}

static hipError_t iree_hip_stream_value_operation_initialize(
    iree_hal_streaming_stream_t* stream,
    const hipStreamBatchMemOpParams* params,
    iree_hip_stream_value_operation_t* out_operation) {
  memset(out_operation, 0, sizeof(*out_operation));

  const void* address = NULL;
  switch (params->operation) {
    case hipStreamMemOpWaitValue32:
      out_operation->kind = IREE_HIP_STREAM_VALUE_OPERATION_WAIT;
      out_operation->value = params->waitValue.value;
      out_operation->mask = UINT32_MAX;
      out_operation->flags = params->waitValue.flags;
      out_operation->byte_length = sizeof(uint32_t);
      address = params->waitValue.address;
      if (!iree_hip_stream_wait_value_flags_are_valid(out_operation->flags)) {
        return hipErrorInvalidValue;
      }
      break;
    case hipStreamMemOpWriteValue32:
      out_operation->kind = IREE_HIP_STREAM_VALUE_OPERATION_WRITE;
      out_operation->value = params->writeValue.value;
      out_operation->mask = UINT32_MAX;
      out_operation->flags = params->writeValue.flags;
      out_operation->byte_length = sizeof(uint32_t);
      address = params->writeValue.address;
      if (!iree_hip_stream_write_value_flags_are_valid(out_operation->flags)) {
        return hipErrorInvalidValue;
      }
      break;
    case hipStreamMemOpWaitValue64:
      out_operation->kind = IREE_HIP_STREAM_VALUE_OPERATION_WAIT;
      out_operation->value = params->waitValue.value64;
      out_operation->mask = UINT64_MAX;
      out_operation->flags = params->waitValue.flags;
      out_operation->byte_length = sizeof(uint64_t);
      address = params->waitValue.address;
      if (!iree_hip_stream_wait_value_flags_are_valid(out_operation->flags)) {
        return hipErrorInvalidValue;
      }
      break;
    case hipStreamMemOpWriteValue64:
      out_operation->kind = IREE_HIP_STREAM_VALUE_OPERATION_WRITE;
      out_operation->value = params->writeValue.value64;
      out_operation->mask = UINT64_MAX;
      out_operation->flags = params->writeValue.flags;
      out_operation->byte_length = sizeof(uint64_t);
      address = params->writeValue.address;
      if (!iree_hip_stream_write_value_flags_are_valid(out_operation->flags)) {
        return hipErrorInvalidValue;
      }
      break;
    case hipStreamMemOpFlushRemoteWrites:
    case hipStreamMemOpBarrier:
      return hipErrorNotSupported;
    default:
      return hipErrorInvalidValue;
  }

  hipError_t result = iree_hip_stream_value_target_initialize(
      stream, address, out_operation->byte_length, &out_operation->target);
  if (result != hipSuccess) return result;
  if (out_operation->kind == IREE_HIP_STREAM_VALUE_OPERATION_WRITE &&
      out_operation->flags != IREE_HIP_STREAM_WRITE_VALUE_DEFAULT &&
      !out_operation->target.buffer_ref.buffer->host_ptr) {
    iree_hip_stream_value_target_deinitialize(&out_operation->target);
    return hipErrorNotSupported;
  }
  return hipSuccess;
}

HIPAPI hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                                      hipStreamBatchMemOpParams* param_array,
                                      unsigned int flags) {
  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &stream_object);
  if (result != hipSuccess) HIP_RETURN_ERROR(hipErrorContextIsDestroyed);

  if (!param_array || count == 0 || count > 256 || flags != 0) {
    iree_hal_streaming_stream_release(stream_object);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_host_size_t operations_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          count, sizeof(iree_hip_stream_value_operation_t),
          &operations_size))) {
    iree_hal_streaming_stream_release(stream_object);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hip_stream_value_operation_t* operations = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), operations_size, (void**)&operations);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_stream_release(stream_object);
    HIP_RETURN_ERROR(iree_status_to_hip_result(status));
  }

  iree_host_size_t initialized_count = 0;
  for (; initialized_count < count; ++initialized_count) {
    result = iree_hip_stream_value_operation_initialize(
        stream_object, &param_array[initialized_count],
        &operations[initialized_count]);
    if (result != hipSuccess) break;
  }

  for (iree_host_size_t i = 0; i < initialized_count && result == hipSuccess;
       ++i) {
    const iree_hip_stream_value_operation_t* operation = &operations[i];
    if (operation->kind == IREE_HIP_STREAM_VALUE_OPERATION_WAIT) {
      status = iree_hip_enqueue_stream_value_wait_resolved(
          stream_object, &operation->target, operation->value, operation->flags,
          operation->mask, operation->byte_length);
    } else {
      status = iree_hip_enqueue_stream_value_write_resolved(
          stream_object, &operation->target, operation->value, operation->flags,
          operation->byte_length);
    }
    if (!iree_status_is_ok(status)) {
      result = iree_status_to_hip_result(status);
    }
  }

  for (iree_host_size_t i = 0; i < initialized_count; ++i) {
    iree_hip_stream_value_target_deinitialize(&operations[i].target);
  }
  iree_allocator_free(iree_allocator_system(), operations);
  iree_hal_streaming_stream_release(stream_object);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// Extended stream creation (CU mask)
//===----------------------------------------------------------------------===//

static void iree_hip_fill_default_cu_mask(
    const iree_hal_streaming_context_t* context, uint32_t mask_count,
    uint32_t* mask) {
  memset(mask, 0, mask_count * sizeof(*mask));
  const iree_host_size_t compute_unit_count =
      context && context->device_entry
          ? context->device_entry->multiprocessor_count
          : 0;
  for (iree_host_size_t compute_unit = 0; compute_unit < compute_unit_count;
       ++compute_unit) {
    const iree_host_size_t word = compute_unit / 32;
    if (word >= mask_count) break;
    mask[word] |= 1u << (compute_unit % 32);
  }
}

static bool iree_hip_cu_mask_has_enabled_compute_unit(
    const iree_hal_streaming_context_t* context, uint32_t mask_count,
    const uint32_t* mask) {
  const iree_host_size_t compute_unit_count =
      context && context->device_entry
          ? context->device_entry->multiprocessor_count
          : 0;
  for (iree_host_size_t compute_unit = 0; compute_unit < compute_unit_count;
       ++compute_unit) {
    const iree_host_size_t word = compute_unit / 32;
    if (word >= mask_count) return false;
    if (mask[word] & (1u << (compute_unit % 32))) return true;
  }
  return false;
}

HIPAPI hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream,
                                               uint32_t mask_count,
                                               const uint32_t* mask) {
  iree_hal_streaming_context_t* context = NULL;
  hipError_t result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);

  if (!stream || mask_count == 0 || !mask) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *stream = NULL;

  iree_host_size_t byte_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(mask_count, sizeof(*mask),
                                                &byte_count))) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  uint32_t* copied_mask = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, byte_count, (void**)&copied_mask);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }
  if (iree_hip_cu_mask_has_enabled_compute_unit(context, mask_count, mask)) {
    memcpy(copied_mask, mask, byte_count);
  } else {
    // An all-clear mask selects the device's default compute-unit set.
    iree_hip_fill_default_cu_mask(context, mask_count, copied_mask);
  }

  result = hipStreamCreate(stream);
  if (result != hipSuccess) {
    iree_allocator_free(context->host_allocator, copied_mask);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_stream_t* stream_object =
      (iree_hal_streaming_stream_t*)*stream;
  iree_slim_mutex_lock(&stream_object->mutex);
  iree_allocator_free(stream_object->host_allocator, stream_object->cu_mask);
  stream_object->cu_mask = copied_mask;
  stream_object->cu_mask_count = mask_count;
  iree_slim_mutex_unlock(&stream_object->mutex);
  return hipSuccess;
}

HIPAPI hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t mask_count,
                                        uint32_t* mask) {
  if (!mask || mask_count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &stream_object);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);

  iree_slim_mutex_lock(&stream_object->mutex);
  if (stream_object->cu_mask) {
    memset(mask, 0, mask_count * sizeof(*mask));
    const iree_host_size_t copy_count =
        iree_min(stream_object->cu_mask_count, mask_count);
    memcpy(mask, stream_object->cu_mask, copy_count * sizeof(*mask));
    iree_slim_mutex_unlock(&stream_object->mutex);
    iree_hal_streaming_stream_release(stream_object);
    return hipSuccess;
  }
  iree_slim_mutex_unlock(&stream_object->mutex);

  iree_hip_fill_default_cu_mask(stream_object->context, mask_count, mask);
  iree_hal_streaming_stream_release(stream_object);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Stream capture mode management
//===----------------------------------------------------------------------===//

// Exchanges the thread's stream capture mode.
// Sets the new mode and returns the previous mode via the mode pointer.
HIPAPI hipError_t
hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  if (!mode) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  switch (*mode) {
    case hipStreamCaptureModeGlobal:
    case hipStreamCaptureModeThreadLocal:
    case hipStreamCaptureModeRelaxed:
      break;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipStreamCaptureMode old_mode = tls_stream_capture_mode;
  tls_stream_capture_mode = *mode;
  *mode = old_mode;
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

// Creates an event object for timing and synchronization.
//
// Parameters:
//  - event: [OUT] Pointer to receive the created event handle.
//
// Returns:
//  - hipSuccess: Event created successfully.
//  - hipErrorInvalidValue: event pointer is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create event.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Event behavior:
// - Created with default flags (no special behavior).
// - Can be recorded in any stream.
// - Can be used for timing measurements.
// - Can be used for stream synchronization.
// - Must be destroyed with hipEventDestroy().
//
// Multi-GPU: Event can be used across devices for synchronization.
//
// Usage pattern:
// ```c
// hipEvent_t start, stop;
// hipEventCreate(&start);
// hipEventCreate(&stop);
// hipEventRecord(start, stream);
// // ... operations ...
// hipEventRecord(stop, stream);
// hipEventSynchronize(stop);
// float ms;
// hipEventElapsedTime(&ms, start, stop);
// ```
//
// See also: hipEventCreateWithFlags, hipEventDestroy, hipEventRecord,
//           hipEventSynchronize.
HIPAPI hipError_t hipEventCreate(hipEvent_t* event) {
  HIP_DEBUG_LOG("[HIP_API] hipEventCreate(event=%p)\n", (void*)event);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!event) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *event = NULL;

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_event_t* event_obj = NULL;
  iree_status_t status = iree_hal_streaming_event_create(
      context, IREE_HAL_STREAMING_EVENT_FLAG_NONE, context->host_allocator,
      &event_obj);

  if (iree_status_is_ok(status)) {
    status = iree_hip_event_register(event_obj);
  }
  if (iree_status_is_ok(status)) {
    *event = (hipEvent_t)event_obj;
  } else {
    iree_hal_streaming_event_release(event_obj);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Creates an event object with specified flags.
//
// Parameters:
//  - event: [OUT] Pointer to receive the created event handle.
//  - flags: [IN] Event creation flags (hipEventDefault, hipEventBlockingSync,
//                hipEventDisableTiming, hipEventInterprocess).
//
// Returns:
//  - hipSuccess: Event created successfully.
//  - hipErrorInvalidValue: event pointer is NULL or invalid flags.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create event.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Event flags:
// - hipEventDefault (0): Standard event with timing support.
// - hipEventBlockingSync: CPU thread blocks on hipEventSynchronize.
// - hipEventDisableTiming: Faster but cannot be used with
//   hipEventElapsedTime.
// - hipEventInterprocess: Event can be shared across processes.
//
// Multi-GPU: Event can be used across devices for synchronization.
//
// Performance note: hipEventDisableTiming creates lighter-weight events
// when timing is not needed.
//
// See also: hipEventCreate, hipEventDestroy, hipEventRecord.
HIPAPI hipError_t hipEventCreateWithFlags(hipEvent_t* event,
                                          unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!event) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *event = NULL;
  if (!iree_hip_event_create_flags_are_valid(flags)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_event_t* event_obj = NULL;
  iree_status_t status = iree_hal_streaming_event_create(
      context, iree_hip_event_flags_to_internal(flags), context->host_allocator,
      &event_obj);

  if (iree_status_is_ok(status)) {
    status = iree_hip_event_register(event_obj);
  }
  if (iree_status_is_ok(status)) {
    *event = (hipEvent_t)event_obj;
  } else {
    iree_hal_streaming_event_release(event_obj);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Destroys an event object.
//
// Parameters:
//  - event: [IN] Event handle to destroy.
//
// Returns:
//  - hipSuccess: Event destroyed successfully.
//  - hipErrorInvalidResourceHandle: Invalid event handle.
//  - hipErrorContextIsDestroyed: Associated context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for the event to
// complete if it has been recorded but not yet reached.
//
// Event behavior:
// - All uses of the event must complete before destruction.
// - After destruction, the event handle becomes invalid.
// - A destroyed event is rejected as an invalid resource handle.
//
// Multi-GPU: Event must be destroyed from a context that can access it.
//
// Warning: Ensure the event has completed or been synchronized before
// destroying.
//
// See also: hipEventCreate, hipEventCreateWithFlags, hipEventSynchronize.
HIPAPI hipError_t hipEventDestroy(hipEvent_t event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!iree_hip_event_unregister(event)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidResourceHandle);
  }
  iree_hal_streaming_event_release((iree_hal_streaming_event_t*)event);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Records an event in a stream for timing and synchronization.
//
// Parameters:
//  - event: [IN] Event handle to record.
//  - stream: [IN] Stream to record the event in (NULL = default stream).
//
// Returns:
//  - hipSuccess: Event recorded successfully.
//  - hipErrorInvalidValue: event is NULL or invalid.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorLaunchFailure: Previous kernel launch failed.
//  - hipErrorUnknown: Internal error during recording.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - The event captures the current position in the stream's command queue.
// - All previously enqueued operations in the stream must complete before
//   the event is signaled.
// - If stream is NULL, uses the default stream.
// - Graph capture: Supported. Creates event node when capturing.
// - The event can be waited on by other streams using hipStreamWaitEvent().
// - The event can be queried with hipEventQuery() or synchronized with
//   hipEventSynchronize().
//
// Multi-GPU: Events can be used for synchronization across devices.
//
// Warning: Recording an event multiple times overwrites the previous
// recording. Wait for the event to complete before re-recording.
//
// Note: Use hipEventElapsedTime() to measure time between two events
// recorded in the same stream.
HIPAPI hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_event_t* event_object = NULL;
  hipError_t event_result = iree_hip_event_lookup_retain(event, &event_object);
  if (event_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(event_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    iree_hal_streaming_event_release(event_object);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (event_object->context != stream_obj->context) {
    iree_hal_streaming_stream_release(stream_obj);
    iree_hal_streaming_event_release(event_object);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  iree_status_t status =
      iree_hal_streaming_event_record(event_object, stream_obj);
  iree_hal_streaming_stream_release(stream_obj);
  iree_hal_streaming_event_release(event_object);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Waits for an event to complete.
//
// Parameters:
//  - event: [IN] Event to wait for.
//
// Returns:
//  - hipSuccess: Event has completed.
//  - hipErrorInvalidResourceHandle: Invalid event handle.
//  - hipErrorLaunchFailure: A kernel launch associated with event failed.
//  - hipErrorIllegalAddress: Invalid memory access in associated operations.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation blocks the host thread until the event
// completes.
//
// Event behavior:
// - Blocks until all operations before the event recording have completed.
// - If event has not been recorded, returns immediately with hipSuccess.
// - If hipEventBlockingSync flag was used, may yield CPU to other threads.
//
// Multi-GPU: Can synchronize events from other devices.
//
// Performance note: For polling without blocking, use hipEventQuery.
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// See also: hipEventQuery, hipEventRecord, hipStreamSynchronize.
HIPAPI hipError_t hipEventSynchronize(hipEvent_t event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG("[HIP_API] hipEventSynchronize(event=%p) called\n",
                (void*)event);

  iree_hal_streaming_event_t* streaming_event = NULL;
  hipError_t event_result =
      iree_hip_event_lookup_retain(event, &streaming_event);
  if (event_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(event_result);
  }

  // Check if any stream is capturing - synchronous operations not allowed
  // during capture. Note: We need to check global capture status since events
  // can be shared across streams.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    iree_hal_streaming_event_release(streaming_event);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (streaming_event->capture_graph) {
    iree_hip_context_invalidate_capture_graph(context,
                                              streaming_event->capture_graph);
    iree_hal_streaming_event_release(streaming_event);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorCapturedEvent);
  }

  iree_status_t status = iree_hal_streaming_event_synchronize(streaming_event);
  iree_hal_streaming_event_release(streaming_event);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Queries the completion status of an event.
//
// Parameters:
//  - event: [IN] Event to query.
//
// Returns:
//  - hipSuccess: Event has completed.
//  - hipErrorNotReady: Event has not completed.
//  - hipErrorInvalidResourceHandle: Invalid event handle.
//  - hipErrorLaunchFailure: A kernel launch associated with event failed.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is non-blocking. Returns immediately with
// the current status.
//
// Event behavior:
// - Checks if all operations before the event recording have completed.
// - If event has not been recorded, returns hipSuccess.
// - Does not wait for the event to complete.
//
// Multi-GPU: Can query events from other devices.
//
// Usage pattern:
// ```c
// while (hipEventQuery(event) == hipErrorNotReady) {
//   // Do other work while waiting
// }
// ```
//
// Graph capture: Not supported. Returns hipErrorStreamCaptureUnsupported.
//
// See also: hipEventSynchronize, hipEventRecord, hipStreamQuery.
HIPAPI hipError_t hipEventQuery(hipEvent_t event) {
  iree_hal_streaming_event_t* streaming_event = NULL;
  hipError_t event_result =
      iree_hip_event_lookup_retain(event, &streaming_event);
  if (event_result != hipSuccess) HIP_RETURN_ERROR(event_result);

  // Check if any stream is capturing - synchronous operations not allowed
  // during capture. Note: We need to check global capture status since events
  // can be shared across streams.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    iree_hal_streaming_event_release(streaming_event);
    HIP_RETURN_ERROR(init_result);
  }

  if (streaming_event->capture_graph) {
    iree_hip_context_invalidate_capture_graph(context,
                                              streaming_event->capture_graph);
    iree_hal_streaming_event_release(streaming_event);
    HIP_RETURN_ERROR(hipErrorCapturedEvent);
  }

  int is_complete = 0;
  iree_status_t status =
      iree_hal_streaming_event_query(streaming_event, &is_complete);
  iree_hal_streaming_event_release(streaming_event);
  // is_complete == 0 means complete, is_complete == 1 means not complete.
  hipError_t result = iree_status_is_ok(status)
                          ? (is_complete == 0 ? hipSuccess : hipErrorNotReady)
                          : iree_status_to_hip_result(status);
  return result;
}

// Computes elapsed time between two events.
//
// Parameters:
//  - ms: [OUT] Pointer to receive elapsed time in milliseconds.
//  - start: [IN] Start event (must have been recorded earlier).
//  - stop: [IN] Stop event (must have been recorded later).
//
// Returns:
//  - hipSuccess: Time computed successfully.
//  - hipErrorInvalidValue: ms is NULL or events are NULL.
//  - hipErrorInvalidResourceHandle: Invalid event handles.
//  - hipErrorNotReady: One or both events have not completed.
//  - hipErrorInvalidHandle: Events created with hipEventDisableTiming.
//
// Synchronization: This operation may block if events have not completed.
//
// Timing behavior:
// - Both events must be recorded in the same stream.
// - Stop event must be recorded after start event.
// - Events must not have hipEventDisableTiming flag.
// - Returns time in milliseconds with ~0.5 microsecond resolution.
// - Time measurement includes all operations between events.
//
// Multi-GPU: Both events must be from the same device.
//
// Usage pattern:
// ```c
// hipEventRecord(start, stream);
// // ... operations to time ...
// hipEventRecord(stop, stream);
// hipEventSynchronize(stop);
// float milliseconds;
// hipEventElapsedTime(&milliseconds, start, stop);
// ```
//
// Warning: Events must be recorded in the same stream for accurate timing.
//
// See also: hipEventCreate, hipEventRecord, hipEventSynchronize.
HIPAPI hipError_t hipEventElapsedTime(float* ms, hipEvent_t start,
                                      hipEvent_t stop) {
  if (!ms) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_event_t* start_event = NULL;
  hipError_t result = iree_hip_event_lookup_retain(start, &start_event);
  if (result != hipSuccess) HIP_RETURN_ERROR(hipErrorInvalidHandle);

  iree_hal_streaming_event_t* stop_event = NULL;
  result = iree_hip_event_lookup_retain(stop, &stop_event);
  if (result != hipSuccess) {
    iree_hal_streaming_event_release(start_event);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  if (start_event->context != stop_event->context) {
    iree_hal_streaming_event_release(stop_event);
    iree_hal_streaming_event_release(start_event);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  // Check if either event has timing disabled.
  if ((start_event->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING) ||
      (stop_event->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING)) {
    iree_hal_streaming_event_release(stop_event);
    iree_hal_streaming_event_release(start_event);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  if (start_event->record_time_ns == 0 || stop_event->record_time_ns == 0) {
    iree_hal_streaming_event_release(stop_event);
    iree_hal_streaming_event_release(start_event);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  iree_status_t status =
      iree_hal_streaming_event_elapsed_time(ms, start_event, stop_event);
  iree_hal_streaming_event_release(stop_event);
  iree_hal_streaming_event_release(start_event);
  result = iree_status_to_hip_result(status);
  return result;
}

//===----------------------------------------------------------------------===//
// Module and kernel execution
//===----------------------------------------------------------------------===//

// Loads a compute module from a file.
//
// Parameters:
//  - module: [OUT] Pointer to receive the loaded module handle.
//  - fname: [IN] Path to the module file (.hsaco, .co, etc.).
//
// Returns:
//  - hipSuccess: Module loaded successfully.
//  - hipErrorInvalidValue: module or fname is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorFileNotFound: Module file does not exist.
//  - hipErrorInvalidImage: File is not a valid module format.
//  - hipErrorNoBinaryForGpu: Module has no binary for the current device.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorOutOfMemory: Insufficient memory to load module.
//
// Synchronization: This operation is synchronous.
//
// Module behavior:
// - Loads compiled GPU code from file.
// - Module remains loaded until hipModuleUnload() is called.
// - Module can contain multiple kernels and global variables.
// - Module is associated with the current context.
//
// Supported formats:
// - .hsaco: AMD GPU code object (GCN/RDNA ISA).
// - .co: NVIDIA GPU code object (PTX/SASS).
// - Architecture-specific binary formats.
//
// Multi-GPU: Module is loaded for the current device's architecture.
//
// See also: hipModuleLoadData, hipModuleUnload, hipModuleGetFunction.
HIPAPI hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!module || !fname) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Load module from file.
  iree_hal_streaming_module_t* stream_module = NULL;
  iree_hal_executable_load_flags_t load_flags =
      IREE_HAL_EXECUTABLE_LOAD_FLAG_ALLOW_OPTIMIZATION;
  iree_status_t status = iree_hal_streaming_module_create_from_file(
      context, load_flags, iree_make_cstring_view(fname),
      context->host_allocator, &stream_module);

  if (iree_status_is_ok(status)) {
    *module = (hipModule_t)stream_module;
  }

  hipError_t result = iree_module_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Loads a compute module from memory.
//
// Parameters:
//  - module: [OUT] Pointer to receive the loaded module handle.
//  - image: [IN] Pointer to module data in memory.
//
// Returns:
//  - hipSuccess: Module loaded successfully.
//  - hipErrorInvalidValue: module or image is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidImage: Data is not a valid module format.
//  - hipErrorNoBinaryForGpu: Module has no binary for the current device.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorOutOfMemory: Insufficient memory to load module.
//
// Synchronization: This operation is synchronous.
//
// Module behavior:
// - Loads compiled GPU code from memory buffer.
// - Buffer must contain valid module data.
// - Module remains loaded until hipModuleUnload() is called.
// - Image data can be freed after loading.
//
// Multi-GPU: Module is loaded for the current device's architecture.
//
// See also: hipModuleLoad, hipModuleLoadDataEx, hipModuleUnload,
//           hipModuleGetFunction.
HIPAPI hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  // Call the extended version with no options.
  return hipModuleLoadDataEx(module, image, 0, NULL, NULL);
}

// Loads a compute module from memory with extended options.
//
// Parameters:
//  - module: [OUT] Pointer to receive the loaded module handle.
//  - image: [IN] Pointer to module data in memory.
//  - numOptions: [IN] Number of JIT options provided.
//  - options: [IN] Array of option types (hipJitOption enum values).
//  - optionValues: [IN] Array of pointers to option values.
//
// Returns:
//  - hipSuccess: Module loaded successfully.
//  - hipErrorInvalidValue: module or image is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidImage: Data is not a valid module format.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorOutOfMemory: Insufficient memory to load module.
//  - hipErrorInvalidValue: Unknown JIT option.
//
// Synchronization: This operation is synchronous.
//
// Module behavior:
// - Loads compiled GPU code from memory buffer.
// - Module remains loaded until hipModuleUnload() is called.
// - Module can contain multiple kernels and global variables.
// - Module is associated with the current context.
//
// JIT options:
// - hipJitOptionMaxRegisters: Max registers per thread.
// - hipJitOptionThreadsPerBlock: Min threads per block.
// - hipJitOptionWallTime: Compilation time limit (ms).
// - hipJitOptionInfoLogBuffer: Buffer for info messages.
// - hipJitOptionErrorLogBuffer: Buffer for error messages.
// - hipJitOptionOptimizationLevel: Optimization level (0-4).
// - hipJitOptionTargetFromContext: Use context's target.
// - hipJitOptionTarget: Specify compute capability.
// - hipJitOptionFallbackStrategy: Fallback behavior.
//
// Option behavior:
// - Recognized options are accepted as hints.
// - Log buffers are filled with null-terminated strings.
// - Option values are type-specific (int*, char**, etc.).
//
// Multi-GPU: Module is loaded for the current device's architecture.
//
// Image lifetime: The image buffer must remain valid until this call returns
// and may be released immediately afterward.
//
// See also: hipModuleLoad, hipModuleLoadData, hipModuleUnload,
//           hipModuleGetFunction.
HIPAPI hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                      unsigned int numOptions,
                                      hipJitOption* options,
                                      void** optionValues) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG(
      "[HIP_API] hipModuleLoadDataEx(module=%p, image=%p, numOptions=%u) "
      "ENTRY\n",
      (void*)module, image, numOptions);
  if (!module || !image) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Process JIT options if provided.
  // Note: Most JIT options are informational or optimization hints that may
  // not apply to our HAL backend. We parse them for compatibility but may not
  // use all of them.
  if (numOptions > 0 && (!options || !optionValues)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  for (unsigned int i = 0; i < numOptions; ++i) {
    switch (options[i]) {
      case hipJitOptionMaxRegisters:
        // Maximum number of registers per thread.
        // This could influence kernel compilation but may be backend-specific.
        break;
      case hipJitOptionThreadsPerBlock:
        // Minimum number of threads per block.
        break;
      case hipJitOptionWallTime:
        // Wall time for compilation in milliseconds.
        break;
      case hipJitOptionInfoLogBuffer:
        // Buffer for informational log.
        break;
      case hipJitOptionInfoLogBufferSizeBytes:
        // Size of info log buffer.
        break;
      case hipJitOptionErrorLogBuffer:
        // Buffer for error log.
        break;
      case hipJitOptionErrorLogBufferSizeBytes:
        // Size of error log buffer.
        break;
      case hipJitOptionOptimizationLevel:
        // Optimization level (0-4).
        break;
      case hipJitOptionTargetFromContext:
        // Use target from current context.
        break;
      case hipJitOptionTarget:
        // Explicit compute capability target.
        break;
      case hipJitOptionFallbackStrategy:
        // Fallback strategy for compilation.
        break;
      case hipJitOptionGenerateDebugInfo:
        // Generate debug information.
        break;
      case hipJitOptionLogVerbose:
        // Enable verbose logging.
        break;
      case hipJitOptionGenerateLineInfo:
        // Generate line number information.
        break;
      case hipJitOptionCacheMode:
        // Cache mode for compiled kernels.
        break;
      case hipJitOptionSm3xOpt:
        // SM 3.x specific optimizations.
        break;
      case hipJitOptionFastCompile:
        // Fast compilation mode.
        break;
      default:
        IREE_TRACE_ZONE_END(z0);
        HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }

  iree_hal_executable_load_flags_t load_flags =
      IREE_HAL_EXECUTABLE_LOAD_FLAG_ALLOW_OPTIMIZATION;

  iree_hal_streaming_module_t* stream_module = NULL;
  iree_status_t status = iree_hal_streaming_module_create_from_memory(
      context, load_flags, iree_make_const_byte_span(image, 0),
      context->host_allocator, &stream_module);

  if (iree_status_is_ok(status)) {
    *module = (hipModule_t)stream_module;
  } else {
    char buf[512] = {0};
    iree_host_size_t buf_len = 0;
    iree_status_format(status, sizeof(buf), buf, &buf_len);
    fprintf(stderr,
            "[HRX] hipModuleLoadDataEx FAILED image=%p numOptions=%u: %.*s\n",
            image, numOptions, (int)buf_len, buf);
    fflush(stderr);
  }

  hipError_t result = iree_module_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Unloads a compute module.
//
// Parameters:
//  - module: [IN] Module handle to unload.
//
// Returns:
//  - hipSuccess: Module unloaded successfully.
//  - hipErrorInvalidResourceHandle: Invalid module handle.
//  - hipErrorContextIsDestroyed: Associated context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for all operations
// using the module to complete.
//
// Module behavior:
// - Releases all resources associated with the module.
// - All kernel functions from the module become invalid.
// - All global variables from the module become inaccessible.
// - Module handle becomes invalid after unloading.
//
// Multi-GPU: Only affects the module in the current context.
//
// Warning: Ensure all kernels from this module have completed execution
// before unloading.
//
// See also: hipModuleLoad, hipModuleLoadData.
HIPAPI hipError_t hipModuleUnload(hipModule_t module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!module) {
    IREE_TRACE_ZONE_END(z0);
    return hipErrorInvalidResourceHandle;
  }

  iree_hal_streaming_module_t* streaming_module =
      (iree_hal_streaming_module_t*)module;
  iree_status_t status =
      iree_hal_streaming_context_synchronize(streaming_module->context);
  if (!iree_status_is_ok(status)) {
    hipError_t result = iree_status_to_hip_result(status);
    IREE_TRACE_ZONE_END(z0);
    return result;
  }

  iree_hal_streaming_module_release(streaming_module);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets a kernel function handle from a module.
//
// Parameters:
//  - function: [OUT] Pointer to receive the function handle.
//  - module: [IN] Module containing the function.
//  - kname: [IN] Name of the kernel function to retrieve.
//
// Returns:
//  - hipSuccess: Function retrieved successfully.
//  - hipErrorInvalidValue: function or kname is NULL.
//  - hipErrorInvalidHandle: Invalid module handle.
//  - hipErrorNotFound: Function with given name not found in module.
//
// Synchronization: This operation is synchronous.
//
// Function behavior:
// - Retrieves a handle to a __global__ kernel function.
// - Function name must match exactly (including C++ mangling).
// - Function handle can be used with hipModuleLaunchKernel().
// - Function remains valid until module is unloaded.
//
// Name lookup:
// - For C kernels: Use the exact function name.
// - For C++ kernels: Use the mangled name.
// - For templated kernels: Use the instantiated mangled name.
//
// Multi-GPU: Function is specific to the module's device.
//
// Usage pattern:
// ```c
// hipFunction_t kernel;
// hipModuleGetFunction(&kernel, module, "vector_add");
// hipModuleLaunchKernel(kernel, ...);
// ```
//
// See also: hipModuleLoad, hipModuleLaunchKernel, hipModuleGetGlobal.
HIPAPI hipError_t hipModuleGetFunction(hipFunction_t* function,
                                       hipModule_t module, const char* kname) {
  HIP_DEBUG_LOG("[HIP_API] hipModuleGetFunction(module=%p, kname='%s')\n",
                (void*)module, kname ? kname : "(null)");
  if (!function || !kname) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_module_t* stream_module =
      (iree_hal_streaming_module_t*)module;
  if (!stream_module) {
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  iree_hal_streaming_symbol_t* stream_symbol = NULL;
  iree_status_t status =
      iree_hal_streaming_module_function(stream_module, kname, &stream_symbol);
  if (iree_status_is_ok(status)) {
    *function = (hipFunction_t)iree_hal_streaming_symbol_tag(stream_symbol);
    HIP_DEBUG_LOG(
        "[HIP_API] hipModuleGetFunction: found symbol %p -> tagged %p\n",
        (void*)stream_symbol, (void*)*function);
  } else {
    if (getenv("HRX_DEBUG_MODULE_LOOKUP")) {
      fprintf(stderr,
              "[HRX] hipModuleGetFunction failed for '%s'; module has %" PRIhsz
              " symbols\n",
              kname, stream_module->symbol_count);
      for (iree_host_size_t i = 0; i < stream_module->symbol_count; ++i) {
        const iree_hal_streaming_symbol_t* symbol = &stream_module->symbols[i];
        fprintf(stderr, "[HRX]   symbol[%" PRIhsz "] type=%d name='%.*s'\n", i,
                (int)symbol->type, (int)symbol->name.size,
                symbol->name.data ? symbol->name.data : "");
      }
    }
    // NOT_FOUND is common and expected: rocBLAS scans multiple modules and
    // relies on this call to probe for kernels. Leave normal error signalling
    // via the return value and avoid spamming stderr.
    HIP_DEBUG_LOG("[HIP_API] hipModuleGetFunction: FAILED to find '%s'\n",
                  kname);
  }

  hipError_t result = iree_status_to_hip_result(status);
  HIP_RETURN_ERROR(result);
}

// Gets a global variable pointer from a module.
//
// Parameters:
//  - dptr: [OUT] Pointer to receive the device pointer to the global.
//  - bytes: [OUT] Pointer to receive the size of the global (can be NULL).
//  - hmod: [IN] Module containing the global variable.
//  - name: [IN] Name of the global variable.
//
// Returns:
//  - hipSuccess: Global variable found and pointer retrieved.
//  - hipErrorInvalidValue: dptr, hmod, or name is NULL.
//  - hipErrorInvalidHandle: Invalid module handle.
//  - hipErrorNotFound: Global variable with given name not found.
//
// Synchronization: This operation is synchronous.
//
// Global variable behavior:
// - Retrieves device pointer to a __device__ or __constant__ variable.
// - Variable name must match exactly (including C++ mangling).
// - Pointer remains valid until module is unloaded.
// - Can read/write the variable using hipMemcpy functions.
//
// Name lookup:
// - For C globals: Use the exact variable name.
// - For C++ globals: Use the mangled name.
// - Namespace and class scope affect mangling.
//
// Multi-GPU: Pointer is valid only on the module's device.
//
// Usage pattern:
// ```c
// hipDeviceptr_t d_global;
// size_t global_size;
// hipModuleGetGlobal(&d_global, &global_size, module, "globalVar");
// hipMemcpyHtoD(d_global, &host_value, sizeof(int));
// ```
//
// See also: hipModuleLoad, hipModuleGetFunction, hipGetSymbolAddress.
HIPAPI hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                                     hipModule_t hmod, const char* name) {
  if (!dptr || !hmod || !name) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_module_t* module = (iree_hal_streaming_module_t*)hmod;
  iree_hal_streaming_deviceptr_t device_ptr = 0;
  iree_device_size_t size = 0;
  iree_status_t status =
      iree_hal_streaming_module_global(module, name, &device_ptr, &size);
  if (iree_status_is_ok(status)) {
    *dptr = (hipDeviceptr_t)device_ptr;
    if (bytes) *bytes = (size_t)size;
  }

  hipError_t result = iree_status_to_hip_result(status);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// Function management
//===----------------------------------------------------------------------====//

// Queries a single attribute of a kernel function.
//
// Parameters:
//  - pi: [OUT] Pointer to receive the attribute value.
//  - attrib: [IN] Attribute to query (hipFuncAttribute_t enum).
//  - hfunc: [IN] Function handle to query.
//
// Returns:
//  - hipSuccess: Attribute queried successfully.
//  - hipErrorInvalidValue: pi is NULL or hfunc is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidDeviceFunction: Not a valid kernel function.
//
// Synchronization: This operation is synchronous and immediate.
//
// Available attributes:
// - hipFuncAttributeMaxThreadsPerBlock: Max threads per block.
// - hipFuncAttributeSharedSizeBytes: Static shared memory usage.
// - hipFuncAttributeConstSizeBytes: Constant memory usage.
// - hipFuncAttributeLocalSizeBytes: Local memory per thread.
// - hipFuncAttributeNumRegs: Register usage per thread.
// - hipFuncAttributePtxVersion: PTX version (CUDA compatibility).
// - hipFuncAttributeBinaryVersion: Binary version.
// - hipFuncAttributeCacheModeCA: Cache configuration.
// - hipFuncAttributeMaxDynamicSharedSizeBytes: Max dynamic shared.
// - hipFuncAttributePreferredSharedMemoryCarveout: Shared mem percent.
//
// Attribute values:
// - Values are kernel-specific and architecture-dependent.
// - Can be used for occupancy calculations.
// - Some attributes may return 0 if not applicable.
//
// Multi-GPU: Attributes are specific to the device that compiled
// the kernel.
//
// See also: hipFuncGetAttributes, hipFuncSetAttribute,
//           hipFuncSetCacheConfig.
HIPAPI hipError_t hipFuncGetAttribute(int* pi, hipFuncAttribute_t attrib,
                                      hipFunction_t hfunc) {
  HIP_DEBUG_LOG("[HIP_API] hipFuncGetAttribute(attrib=%d, hfunc=%p)\n",
                (int)attrib, (void*)hfunc);
  if (!pi || !hfunc) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Untag the function pointer if it was tagged by hipModuleGetFunction.
  iree_hal_streaming_symbol_t* symbol = iree_hal_streaming_symbol_untag(hfunc);

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }
  iree_hal_streaming_function_attributes_t* attributes =
      &symbol->function_attributes;

  // Return attribute value based on what we have cached.
  switch (attrib) {
    case hipFuncAttributeMaxThreadsPerBlock:
      *pi = attributes->maximum_threads_per_block;
      break;
    case hipFuncAttributeSharedSizeBytes:
      *pi = attributes->fixed_shared_memory_size;
      break;
    case hipFuncAttributeConstSizeBytes:
      // We don't track constant memory usage.
      *pi = 0;
      break;
    case hipFuncAttributeLocalSizeBytes:
      *pi = attributes->fixed_local_memory_size;
      break;
    case hipFuncAttributeNumRegs:
      *pi = attributes->register_count;
      break;
    case hipFuncAttributePtxVersion:
      // Return a default PTX version equivalent for HIP.
      *pi = 0;  // Not applicable to HIP/ROCm.
      break;
    case hipFuncAttributeBinaryVersion:
      // Return a default binary version.
      *pi = 0;  // Not tracked.
      break;
    case hipFuncAttributeCacheModeCA:
      // Cache mode is not tracked.
      *pi = 0;
      break;
    case hipFuncAttributeMaxDynamicSharedSizeBytes:
      *pi = iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
          attributes);
      break;
    case hipFuncAttributePreferredSharedMemoryCarveout:
      // Carveout percentage not tracked.
      *pi = 0;
      break;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  HIP_RETURN_ERROR(hipSuccess);
}

// Queries all attributes of a kernel function.
//
// Parameters:
//  - attr: [OUT] Pointer to structure to receive all attributes.
//  - hfunc: [IN] Function handle to query.
//
// Returns:
//  - hipSuccess: Attributes queried successfully.
//  - hipErrorInvalidValue: attr or hfunc is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidDeviceFunction: Not a valid kernel function.
//
// Synchronization: This operation is synchronous and immediate.
//
// Attribute structure fields:
// - sharedSizeBytes: Static shared memory in bytes.
// - constSizeBytes: Constant memory in bytes.
// - localSizeBytes: Local memory per thread in bytes.
// - maxThreadsPerBlock: Maximum threads per block.
// - numRegs: Registers per thread.
// - ptxVersion: PTX version (CUDA compatibility).
// - binaryVersion: Binary version.
// - cacheModeCA: Cache configuration.
// - maxDynamicSharedSizeBytes: Max dynamic shared memory.
// - preferredShmemCarveout: Preferred shared memory percentage.
//
// Usage:
// - More efficient than multiple hipFuncGetAttribute calls.
// - Use for occupancy calculations and launch configuration.
// - Values are kernel and architecture specific.
//
// Multi-GPU: Attributes are specific to the device that compiled
// the kernel.
//
// See also: hipFuncGetAttribute, hipFuncSetAttribute,
//           hipOccupancyMaxActiveBlocksPerMultiprocessor.
HIPAPI hipError_t hipFuncGetAttributes(hipFuncAttributes* attr,
                                       hipFunction_t hfunc) {
  if (!attr || !hfunc) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Untag the function pointer if it was tagged by hipModuleGetFunction.
  iree_hal_streaming_symbol_t* symbol = iree_hal_streaming_symbol_untag(hfunc);

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }
  iree_hal_streaming_function_attributes_t* attributes =
      &symbol->function_attributes;

  // Fill in the attributes structure.
  memset(attr, 0, sizeof(hipFuncAttributes));
  attr->maxThreadsPerBlock = attributes->maximum_threads_per_block;
  attr->sharedSizeBytes = attributes->fixed_shared_memory_size;
  attr->constSizeBytes = 0;  // Not tracked.
  attr->localSizeBytes = attributes->fixed_local_memory_size;
  attr->numRegs = attributes->register_count;
  attr->ptxVersion = 0;     // Not applicable to HIP/ROCm.
  attr->binaryVersion = 0;  // Not tracked.
  attr->cacheModeCA = 0;    // Not tracked.
  attr->maxDynamicSharedSizeBytes =
      iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
          attributes);
  attr->preferredShmemCarveout = 0;  // Not tracked.

  HIP_RETURN_ERROR(hipSuccess);
}

// Sets a specific attribute of a kernel function.
//
// Parameters:
//  - hfunc: [IN] Function handle to modify.
//  - attrib: [IN] Attribute to set (hipFuncAttribute_t enum).
//  - value: [IN] New value for the attribute.
//
// Returns:
//  - hipSuccess: Attribute set successfully.
//  - hipErrorInvalidValue: Invalid attribute or value.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidDeviceFunction: Not a valid kernel function.
//  - hipErrorNotSupported: Attribute cannot be modified.
//
// Synchronization: This operation is synchronous.
//
// Settable attributes:
// - hipFuncAttributeMaxDynamicSharedSizeBytes: Set max dynamic shared
//   memory for kernels that use more than 48KB.
// - hipFuncAttributePreferredSharedMemoryCarveout: Set L1/shared split
//   (percentage 0-100).
//
// Attribute effects:
// - Changes apply to all subsequent launches of this function.
// - Does not affect currently executing kernels.
// - Settings persist until module is unloaded.
// - May affect occupancy and performance.
//
// Restrictions:
// - Most attributes are read-only and cannot be set.
// - Value must be within hardware-supported range.
// - Some attributes require specific GPU architectures.
//
// Multi-GPU: Settings apply only to the function on the current device.
//
// Warning: Changing attributes may reduce occupancy or cause launch
// failures if values exceed hardware limits.
//
// See also: hipFuncGetAttribute, hipFuncSetCacheConfig,
//           hipFuncSetSharedMemConfig.
HIPAPI hipError_t hipFuncSetAttribute(hipFunction_t hfunc,
                                      hipFuncAttribute_t attrib, int value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Untag the function pointer if it was tagged by hipModuleGetFunction.
  iree_hal_streaming_symbol_t* symbol = iree_hal_streaming_symbol_untag(hfunc);

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }
  iree_hal_streaming_function_attributes_t* attributes =
      &symbol->function_attributes;

  // Only certain attributes can be set.
  hipError_t result = hipSuccess;
  switch (attrib) {
    case hipFuncAttributeMaxDynamicSharedSizeBytes:
      if (value < 0 ||
          !iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
              attributes, (uint32_t)value)) {
        result = hipErrorInvalidValue;
      }
      break;
    case hipFuncAttributePreferredSharedMemoryCarveout:
      // This controls the L1/shared memory split.
      // Values are percentages (0, 25, 50, 75, 100).
      // We don't actually configure this in the stream HAL yet.
      if (value != 0 && value != 25 && value != 50 && value != 75 &&
          value != 100) {
        result = hipErrorInvalidValue;
      }
      break;
    default:
      // Most attributes are read-only.
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets the preferred cache configuration for a kernel function.
//
// Parameters:
//  - hfunc: [IN] Function handle to configure.
//  - config: [IN] Cache configuration preference.
//
// Returns:
//  - hipSuccess: Cache configuration set successfully.
//  - hipErrorInvalidValue: Invalid function handle or config.
//  - hipErrorInvalidHandle: Not a valid kernel function.
//  - hipErrorNotSupported: Configuration not supported on device.
//
// Synchronization: This operation is synchronous.
//
// Cache configurations:
// - hipFuncCachePreferNone: No preference (default).
// - hipFuncCachePreferShared: Prefer larger shared memory.
// - hipFuncCachePreferL1: Prefer larger L1 cache.
// - hipFuncCachePreferEqual: Equal L1 and shared memory.
//
// Cache behavior:
// - Controls L1 cache vs shared memory allocation.
// - Total L1 + shared memory is fixed per SM.
// - Configuration is a hint; hardware may override.
// - Affects all subsequent launches of this function.
//
// Performance considerations:
// - PreferShared: Good for kernels with heavy shared memory use.
// - PreferL1: Good for kernels with scattered memory access.
// - PreferEqual: Balanced for mixed workloads.
//
// Multi-GPU: Configuration applies per device and context.
//
// Warning: Configuration may not be honored if it violates hardware
// constraints or kernel requirements.
//
// See also: hipFuncSetAttribute, hipFuncSetSharedMemConfig,
//           hipDeviceSetCacheConfig.
HIPAPI hipError_t hipFuncSetCacheConfig(hipFunction_t hfunc,
                                        hipFuncCache_t config) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Validate cache configuration.
  hipError_t result = hipSuccess;
  switch (config) {
    case hipFuncCachePreferNone:
    case hipFuncCachePreferShared:
    case hipFuncCachePreferL1:
    case hipFuncCachePreferEqual:
      // These are all valid configurations.
      // We don't actually configure cache in the stream HAL yet,
      // but we accept the values.
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets the shared memory bank configuration for a kernel function.
//
// Parameters:
//  - hfunc: [IN] Function handle to configure.
//  - config: [IN] Shared memory bank size configuration.
//
// Returns:
//  - hipSuccess: Configuration set successfully.
//  - hipErrorInvalidValue: Invalid function handle or config.
//  - hipErrorInvalidHandle: Not a valid kernel function.
//  - hipErrorNotSupported: Configuration not supported on device.
//
// Synchronization: This operation is synchronous.
//
// Bank configurations:
// - hipSharedMemBankSizeDefault: Default bank size.
// - hipSharedMemBankSizeFourByte: 4-byte banks (32 banks).
// - hipSharedMemBankSizeEightByte: 8-byte banks (16 banks).
//
// Bank conflict behavior:
// - 4-byte banks: Better for 32-bit data access patterns.
// - 8-byte banks: Better for 64-bit data access patterns.
// - Bank conflicts occur when multiple threads access same bank.
// - Conflicts serialize memory access, reducing performance.
//
// Performance optimization:
// - Choose based on dominant data type in shared memory.
// - 4-byte for float/int, 8-byte for double/long.
// - Proper padding can avoid bank conflicts.
// - Use stride access patterns to minimize conflicts.
//
// Hardware notes:
// - Not all devices support all configurations.
// - Newer GPUs may have different bank architectures.
// - Configuration is a performance hint.
//
// Multi-GPU: Configuration applies per device and context.
//
// See also: hipFuncSetCacheConfig, hipFuncSetAttribute,
//           hipDeviceSetSharedMemConfig.
HIPAPI hipError_t hipFuncSetSharedMemConfig(hipFunction_t hfunc,
                                            hipSharedMemConfig config) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Validate shared memory configuration.
  hipError_t result = hipSuccess;
  switch (config) {
    case hipSharedMemBankSizeDefault:
    case hipSharedMemBankSizeFourByte:
    case hipSharedMemBankSizeEightByte:
      // These are all valid configurations.
      // We don't actually configure shared memory banks in the stream HAL yet,
      // but we accept the values.
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// Kernel name functions
//===----------------------------------------------------------------------===//

// Returns the name of a kernel function.
// Returns a placeholder since we don't track kernel names.
HIPAPI const char* hipKernelNameRef(const hipFunction_t f) {
  if (!f) return NULL;
  return "<unknown kernel>";
}

// Returns the name of a kernel function by host pointer.
// Returns a placeholder since we don't track kernel names.
HIPAPI const char* hipKernelNameRefByPtr(const void* hostFunction,
                                         hipStream_t stream) {
  if (!hostFunction) return NULL;
  (void)stream;
  return "<unknown kernel>";
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

// Launches a kernel with specified configuration.
//
// Parameters:
//  - function_address: [IN] Host function pointer (registered kernel stub).
//  - numBlocks: [IN] Grid dimensions in blocks.
//  - dimBlocks: [IN] Block dimensions in threads.
//  - args: [IN] Array of kernel arguments.
//  - sharedMemBytes: [IN] Dynamic shared memory size.
//  - stream: [IN] Stream for kernel execution.
//
// Returns:
//  - hipSuccess: Kernel launched successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidDeviceFunction: Unregistered kernel function.
//  - hipErrorLaunchFailure: Kernel launch failed.
//
// Behavior:
// - This function is typically called from compiler-generated kernel stubs
//   after they pop the configuration from __hipPopCallConfiguration.
// - It does NOT read the thread-local configuration directly.
// - The function_address should be registered via __hipRegisterFunction.
//
// Stream behavior:
// - Kernel execution is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Kernel executes after all previously enqueued operations in the stream.
// - Graph capture: Supported. Creates kernel node when capturing.
//
// Thread safety: Thread-safe.
#ifndef IREE_HIP_SYNC_AFTER_EVERY_LAUNCH
#define IREE_HIP_SYNC_AFTER_EVERY_LAUNCH 0
#endif
HIPAPI hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks,
                                  dim3 dimBlocks, void** args,
                                  size_t sharedMemBytes, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG(
      "[HIP_API] hipLaunchKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), "
      "shared=%zu, stream=%p)\n",
      function_address, numBlocks.x, numBlocks.y, numBlocks.z, dimBlocks.x,
      dimBlocks.y, dimBlocks.z, sharedMemBytes, (void*)stream);

  if (!function_address) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDeviceFunction);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  init_result = iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Resolve the host function pointer to a symbol.
  // Check if this is already a tagged symbol from hipModuleGetFunction (driver
  // API) or if we need to do the host lookup.
  iree_hal_streaming_symbol_t* symbol = NULL;
  if (iree_hal_streaming_symbol_has_tag(function_address)) {
    // Fast path: a symbol pointer - just untag and use directly.
    symbol = iree_hal_streaming_symbol_untag(function_address);
    HIP_DEBUG_LOG(
        "[DEBUG_TAG] Using tagged symbol %p -> copy=%u bind=%u name=%.*s\n",
        function_address, symbol->parameters.copy_count,
        symbol->parameters.binding_count,
        (int)(symbol->name.size > 80 ? 80 : symbol->name.size),
        symbol->name.data ? symbol->name.data : "(null)");
  } else {
    // Slow path: must look up in symbol map.
    // This may demand-load the entire parent module of the function.
    iree_status_t lookup_status = iree_hal_streaming_context_symbol_map_lookup(
        &context->symbol_map, (void*)function_address, &symbol);
    if (!iree_status_is_ok(lookup_status)) {
      // Symbol not found in registry - invalid function.
      iree_status_ignore(lookup_status);
      HIP_DEBUG_LOG("[DEBUG_LOOKUP] Lookup failed for %p\n", function_address);
      symbol = NULL;
    } else if (symbol == (iree_hal_streaming_symbol_t*)function_address) {
      // Symbol was not found in registry - the lookup returns the host pointer
      // as a fallback which is not a valid symbol.
      HIP_DEBUG_LOG("[DEBUG_LOOKUP] Identity returned for %p\n",
                    function_address);
      symbol = NULL;
    } else {
      // Found valid symbol
      if (symbol->name.data && strstr(symbol->name.data, "indexSelect")) {
        HIP_DEBUG_LOG(
            "[DEBUG_LOOKUP] Found indexSelect: copy=%u bind=%u const=%u\n",
            symbol->parameters.copy_count, symbol->parameters.binding_count,
            (unsigned)symbol->parameters.constant_bytes);
      }
    }
  }

  if (!symbol) {
    iree_hal_streaming_stream_release(stream_obj);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDeviceFunction);
  }

  hipError_t launch_config_result = iree_hip_validate_launch_configuration(
      stream_obj->context ? stream_obj->context->device_entry : NULL, symbol,
      numBlocks.x, numBlocks.y, numBlocks.z, dimBlocks.x, dimBlocks.y,
      dimBlocks.z, sharedMemBytes);
  if (launch_config_result != hipSuccess) {
    iree_hal_streaming_stream_release(stream_obj);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(launch_config_result);
  }

  const iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {numBlocks.x, numBlocks.y, numBlocks.z},
      .block_dim = {dimBlocks.x, dimBlocks.y, dimBlocks.z},
      .shared_memory_bytes = (uint32_t)sharedMemBytes,
      .buffer = args,  // args is already the kernelParams array
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };
  hipError_t dependency_result =
      iree_hip_order_legacy_stream_dependencies(context, stream_obj);
  if (dependency_result != hipSuccess) {
    iree_hal_streaming_stream_release(stream_obj);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }
  iree_status_t status =
      iree_hal_streaming_launch_kernel(symbol, &params, stream_obj);

  hipError_t result = iree_status_to_hip_result(status);

#if IREE_HIP_SYNC_AFTER_EVERY_LAUNCH
  if (result == hipSuccess) {
    iree_hal_streaming_context_t* ctx = iree_hal_streaming_context_current();
    if (ctx) {
      iree_status_t sync_status = iree_hal_streaming_context_synchronize(ctx);
      if (!iree_status_is_ok(sync_status)) {
        HIP_DEBUG_LOG(
            "[HIP_API] Warning: device sync after hipLaunchKernel failed\n");
        iree_status_ignore(sync_status);
      }
    }
  }
#endif

  iree_hal_streaming_stream_release(stream_obj);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Launches a kernel with specified configuration.
//
// Parameters:
//  - function_address: [IN] Host function pointer (registered kernel stub).
//  - numBlocks: [IN] Grid dimensions in blocks.
//  - dimBlocks: [IN] Block dimensions in threads.
//  - args: [IN] Array of kernel arguments.
//  - sharedMemBytes: [IN] Dynamic shared memory size.
//  - stream: [IN] Stream for kernel execution.
//  - startEvent: [IN] Event to record start time.
//  - stopEvent: [IN] Event to record stop time.
//  - flags: AdditionalFlags
//       - hipExtAnyOrderLaunch: Allow kernels to launch in any order.
//                               currently does nothing.
// Returns:
//  - hipSuccess: Kernel launched successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidDeviceFunction: Unregistered kernel function.
//  - hipErrorLaunchFailure: Kernel launch failed.
//
// Behavior:
// - This function is typically called from compiler-generated kernel stubs
//   after they pop the configuration from __hipPopCallConfiguration.
// - It does NOT read the thread-local configuration directly.
// - The function_address should be registered via __hipRegisterFunction.
//
// Stream behavior:
// - Kernel execution is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Kernel executes after all previously enqueued operations in the stream.
// - Graph capture: Supported. Creates kernel node when capturing.
//
// Thread safety: Thread-safe.
HIPAPI hipError_t hipExtLaunchKernel(const void* function_address,
                                     dim3 numBlocks, dim3 dimBlocks,
                                     void** args, size_t sharedMemBytes,
                                     hipStream_t stream, hipEvent_t startEvent,
                                     hipEvent_t stopEvent, int flags) {
  if (!function_address) HIP_RETURN_ERROR(hipErrorInvalidDeviceFunction);
  if (flags & ~hipExtAnyOrderLaunch) HIP_RETURN_ERROR(hipErrorInvalidValue);

  // Validate the opaque handles before passing them to the normal launch path.
  // A stale handle is only an address and must never be dereferenced to decide
  // whether it is still live.
  iree_hal_streaming_stream_t* stream_object = NULL;
  hipError_t result =
      iree_hip_resolve_registered_stream(stream, &stream_object);
  if (result != hipSuccess) HIP_RETURN_ERROR(hipErrorInvalidValue);

  iree_hal_streaming_event_t* start_event = NULL;
  if (startEvent) {
    result = iree_hip_event_lookup_retain(startEvent, &start_event);
    if (result != hipSuccess ||
        start_event->context != stream_object->context) {
      if (start_event) iree_hal_streaming_event_release(start_event);
      iree_hal_streaming_stream_release(stream_object);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }

  iree_hal_streaming_event_t* stop_event = NULL;
  if (stopEvent) {
    result = iree_hip_event_lookup_retain(stopEvent, &stop_event);
    if (result != hipSuccess || stop_event->context != stream_object->context) {
      if (stop_event) iree_hal_streaming_event_release(stop_event);
      if (start_event) iree_hal_streaming_event_release(start_event);
      iree_hal_streaming_stream_release(stream_object);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }

  if (start_event) {
    result = iree_status_to_hip_result(
        iree_hal_streaming_event_record(start_event, stream_object));
  }
  if (result == hipSuccess) {
    result = hipLaunchKernel(function_address, numBlocks, dimBlocks, args,
                             sharedMemBytes, stream);
  }
  if (result == hipSuccess && stop_event) {
    result = iree_status_to_hip_result(
        iree_hal_streaming_event_record(stop_event, stream_object));
  }

  if (stop_event) iree_hal_streaming_event_release(stop_event);
  if (start_event) iree_hal_streaming_event_release(start_event);
  iree_hal_streaming_stream_release(stream_object);
  return result;
}

// Launches a kernel function with specified dimensions and parameters.
//
// Parameters:
//  - f: [IN] Kernel function handle obtained from hipModuleGetFunction().
//  - gridDimX: [IN] Grid X dimension in blocks.
//  - gridDimY: [IN] Grid Y dimension in blocks.
//  - gridDimZ: [IN] Grid Z dimension in blocks.
//  - blockDimX: [IN] Block X dimension in threads.
//  - blockDimY: [IN] Block Y dimension in threads.
//  - blockDimZ: [IN] Block Z dimension in threads.
//  - sharedMemBytes: [IN] Dynamic shared memory size per block in bytes.
//  - stream: [IN] Stream for kernel execution (NULL = default stream).
//  - kernelParams: [IN] Array of kernel parameters, NULL-terminated.
//  - extra: [IN] Extra options (currently unused, should be NULL).
//
// Returns:
//  - hipSuccess: Kernel launched successfully.
//  - hipErrorInvalidValue: Invalid function handle or dimensions.
//  - hipErrorInvalidConfiguration: Invalid launch configuration.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorSharedObjectInitFailed: Shared memory allocation failed.
//  - hipErrorLaunchOutOfResources: Insufficient resources for launch.
//  - hipErrorLaunchTimeOut: Previous kernel execution timed out.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during launch.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Kernel execution is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Kernel executes after all previously enqueued operations in the stream.
// - Use hipStreamSynchronize() to wait for kernel completion.
// - Use hipEventRecord() after launch to mark completion point.
// - Graph capture: Supported. Creates kernel node when capturing.
//
// Launch configuration:
// - Total threads = gridDim * blockDim.
// - Grid dimensions must be > 0 and within device limits.
// - Block dimensions must be > 0 and within device limits.
// - Total threads per block must not exceed device maximum.
// - Shared memory size must not exceed device maximum.
//
// Kernel parameters:
// - kernelParams is an array of void* pointers to actual arguments.
// - Array must be NULL-terminated.
// - Each pointer points to the argument value (not a pointer to pointer).
// - Arguments are passed by value to the kernel.
//
// Multi-GPU: Kernel executes on the device associated with the current
// context.
//
// WARNING: Ensure all kernel arguments remain valid until kernel completes.
// Do not modify or free argument memory while kernel is executing.
//
// Note: Check device properties with hipDeviceGetAttribute() to determine
// maximum grid/block dimensions and shared memory limits.
// Counter for tracking hipModuleLaunchKernel calls (for debugging NaN issue)
static int g_module_launch_count = 0;

HIPAPI hipError_t hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra) {
  IREE_TRACE_ZONE_BEGIN(z0);
  int launch_num = ++g_module_launch_count;
  HIP_DEBUG_LOG(
      "[HIP_API] hipModuleLaunchKernel #%d: f=%p grid=(%u,%u,%u) "
      "block=(%u,%u,%u) shared=%u stream=%p extra=%p kernelParams=%p\n",
      launch_num, (void*)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
      blockDimZ, sharedMemBytes, (void*)stream, (void*)extra,
      (void*)kernelParams);

#if IREE_HIP_VERBOSE_DEBUG
  // Debug: dump the extra buffer contents to understand the kernel ABI
  void* debug_buf = NULL;
  size_t debug_sz = 0;
  if (extra) {
    HIP_DEBUG_LOG("[HIP_API]   extra format:\n");
    for (int i = 0; extra[i] != HIP_LAUNCH_PARAM_END; i += 2) {
      if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
        debug_buf = extra[i + 1];
        HIP_DEBUG_LOG("[HIP_API]     BUFFER_POINTER: %p\n", debug_buf);
      } else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
        debug_sz = *(size_t*)extra[i + 1];
        HIP_DEBUG_LOG("[HIP_API]     BUFFER_SIZE: %zu\n", debug_sz);
      }
    }
    // Dump first 64 bytes of buffer as hex and pointers
    if (debug_buf && debug_sz >= 64) {
      HIP_DEBUG_LOG("[HIP_API]   buffer contents (first 64 bytes as ptrs):\n");
      uint64_t* ptrs = (uint64_t*)debug_buf;
      for (int i = 0; i < 8; ++i) {
        HIP_DEBUG_LOG("[HIP_API]     [%d]: 0x%016lx\n", i, ptrs[i]);
      }
    }
  }
#endif
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  init_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Validate the function handle before untagging it. An untagged or
  // non-function handle is rejected here (as the other driver-style entry
  // points do) so the extra-buffer handling below can safely dereference
  // symbol->parameters.
  if (!iree_hal_streaming_symbol_has_tag(f)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }
  iree_hal_streaming_symbol_t* symbol = iree_hal_streaming_symbol_untag(f);
  if (!symbol || symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  if (kernelParams && extra) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t launch_config_result = iree_hip_validate_launch_configuration(
      stream_obj->context ? stream_obj->context->device_entry : NULL, symbol,
      gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes);
  if (launch_config_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(launch_config_result);
  }

  // Extract params pointer and size from HIP's parameter format.
  void* params_ptr = NULL;
  size_t params_size = 0;
  iree_hal_streaming_dispatch_flags_t dispatch_flags =
      IREE_HAL_STREAMING_DISPATCH_FLAG_NONE;
  if (extra) {
    hipError_t parse_result =
        iree_hip_parse_launch_extra(extra, &params_ptr, &params_size);
    if (parse_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(parse_result);
    }
    // The extra buffer is already laid out in the kernel's native kernarg ABI.
    // Preserve it byte-for-byte: HIP device pointers may appear in formal
    // pointer parameters or inside opaque data copied by the caller.
    dispatch_flags |= IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED;
  } else if (kernelParams) {
    // kernelParams is an array of pointers to the actual parameters.
    params_ptr = kernelParams;
    dispatch_flags |= IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY;
  }

  const iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {gridDimX, gridDimY, gridDimZ},
      .block_dim = {blockDimX, blockDimY, blockDimZ},
      .shared_memory_bytes = sharedMemBytes,
      .buffer = params_ptr,
      .buffer_size = params_size,
      .flags = dispatch_flags,
  };
  hipError_t dependency_result =
      iree_hip_order_legacy_stream_dependencies(context, stream_obj);
  if (dependency_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }
  iree_status_t status =
      iree_hal_streaming_launch_kernel(symbol, &params, stream_obj);

  if (!iree_status_is_ok(status)) {
    char buf[512] = {0};
    iree_host_size_t buf_len = 0;
    iree_status_format(status, sizeof(buf), buf, &buf_len);
    fprintf(stderr,
            "[HRX] hipModuleLaunchKernel FAILED #%d symbol=%p buffer=%p "
            "buffer_size=%zu grid=(%u,%u,%u) block=(%u,%u,%u): %.*s\n",
            launch_num, (void*)symbol, params_ptr, params_size, gridDimX,
            gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, (int)buf_len,
            buf);
    fflush(stderr);
  }

  hipError_t result = iree_status_to_hip_result(status);
  HIP_DEBUG_LOG("[HIP_API] hipModuleLaunchKernel: returned result=%d\n",
                result);

  // AGGRESSIVE SYNC: Wait for device to complete after every kernel
  // to rule out race conditions between kernel dispatches.
#ifndef IREE_HIP_SYNC_AFTER_EVERY_LAUNCH
#define IREE_HIP_SYNC_AFTER_EVERY_LAUNCH 0
#endif
#if IREE_HIP_SYNC_AFTER_EVERY_LAUNCH
  if (result == hipSuccess) {
    iree_hal_streaming_context_t* ctx = iree_hal_streaming_context_current();
    if (ctx) {
      iree_status_t sync_status = iree_hal_streaming_context_synchronize(ctx);
      if (!iree_status_is_ok(sync_status)) {
        HIP_DEBUG_LOG("[HIP_API] Warning: device sync after launch failed\n");
        iree_status_ignore(sync_status);
      }
    }
  }
#endif

  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Launches a kernel function with specified dimensions and parameters.
//
// Parameters:
//  - f: [IN] Kernel function handle obtained from hipModuleGetFunction().
//  - gridDimX: [IN] Grid X dimension in blocks.
//  - gridDimY: [IN] Grid Y dimension in blocks.
//  - gridDimZ: [IN] Grid Z dimension in blocks.
//  - blockDimX: [IN] Block X dimension in threads.
//  - blockDimY: [IN] Block Y dimension in threads.
//  - blockDimZ: [IN] Block Z dimension in threads.
//  - sharedMemBytes: [IN] Dynamic shared memory size per block in bytes.
//  - stream: [IN] Stream for kernel execution (NULL = default stream).
//  - kernelParams: [IN] Array of kernel parameters, NULL-terminated.
//  - extra: [IN] Extra options (currently unused, should be NULL).
//  - startEvent: [IN] Event to record start time.
//  - stopEvent: [IN] Event to record stop time.
//  - flags: AdditionalFlags
//       - hipExtAnyOrderLaunch: Allow kernels to launch in any order.
//                               currently does nothing.
//
// Returns:
//  - hipSuccess: Kernel launched successfully.
//  - hipErrorInvalidValue: Invalid function handle or dimensions.
//  - hipErrorInvalidConfiguration: Invalid launch configuration.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorSharedObjectInitFailed: Shared memory allocation failed.
//  - hipErrorLaunchOutOfResources: Insufficient resources for launch.
//  - hipErrorLaunchTimeOut: Previous kernel execution timed out.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during launch.
//
// Synchronization: This operation is asynchronous.
//
// Extended kernel launch with OpenCL-style global/local work sizes.
//
// IMPORTANT: Unlike hipModuleLaunchKernel which takes grid dimensions
// (number of workgroups), this function takes GLOBAL work sizes
// (total number of threads). The relationship is:
//   gridDim = globalWorkSize / localWorkSize
//
// Parameters:
//  - f: [IN] Kernel function handle from hipModuleGetFunction().
//  - globalWorkSizeX: [IN] Total threads in X (NOT number of workgroups).
//  - globalWorkSizeY: [IN] Total threads in Y.
//  - globalWorkSizeZ: [IN] Total threads in Z.
//  - localWorkSizeX: [IN] Threads per workgroup in X (same as blockDimX).
//  - localWorkSizeY: [IN] Threads per workgroup in Y.
//  - localWorkSizeZ: [IN] Threads per workgroup in Z.
//  - sharedMemBytes: [IN] Dynamic shared memory size per block in bytes.
//  - stream: [IN] Stream for kernel execution (NULL = default stream).
//  - kernelParams: [IN] Array of kernel parameters, NULL-terminated.
//  - extra: [IN] Extra launch parameters (HIP_LAUNCH_PARAM_BUFFER).
//  - startEvent: [IN] Event to record at kernel start (can be NULL).
//  - stopEvent: [IN] Event to record at kernel end (can be NULL).
//  - flags: [IN] Launch flags (e.g., hipExtAnyOrderLaunch).
//
// Note: This converts globalWorkSize to gridDim before calling
// hipModuleLaunchKernel: gridDim = ceil(globalWorkSize / localWorkSize).
HIPAPI hipError_t hipExtModuleLaunchKernel(
    hipFunction_t f, unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ, unsigned int localWorkSizeX,
    unsigned int localWorkSizeY, unsigned int localWorkSizeZ,
    unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams,
    void** extra, hipEvent_t startEvent, hipEvent_t stopEvent, int flags) {
  // Convert OpenCL-style global/local work sizes to CUDA-style grid/block
  // dimensions. hipModuleLaunchKernel expects gridDim (number of workgroups),
  // but hipExtModuleLaunchKernel receives globalWorkSize (total threads).
  unsigned int gridDimX =
      localWorkSizeX ? ((globalWorkSizeX + localWorkSizeX - 1) / localWorkSizeX)
                     : 1;
  unsigned int gridDimY =
      localWorkSizeY ? ((globalWorkSizeY + localWorkSizeY - 1) / localWorkSizeY)
                     : 1;
  unsigned int gridDimZ =
      localWorkSizeZ ? ((globalWorkSizeZ + localWorkSizeZ - 1) / localWorkSizeZ)
                     : 1;
  return hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, localWorkSizeX,
                               localWorkSizeY, localWorkSizeZ, sharedMemBytes,
                               stream, kernelParams, extra);
}

// Launches a cooperative kernel with grid-wide synchronization support.
//
// Parameters:
//  - f: [IN] Kernel function handle from hipModuleGetFunction().
//  - gridDimX: [IN] Grid X dimension in blocks.
//  - gridDimY: [IN] Grid Y dimension in blocks.
//  - gridDimZ: [IN] Grid Z dimension in blocks.
//  - blockDimX: [IN] Block X dimension in threads.
//  - blockDimY: [IN] Block Y dimension in threads.
//  - blockDimZ: [IN] Block Z dimension in threads.
//  - sharedMemBytes: [IN] Dynamic shared memory size per block in bytes.
//  - stream: [IN] Stream for kernel execution (NULL = default stream).
//  - kernelParams: [IN] Array of kernel parameters, NULL-terminated.
//
// Returns:
//  - hipSuccess: Cooperative kernel launched successfully.
//  - hipErrorInvalidValue: Invalid function handle or dimensions.
//  - hipErrorInvalidConfiguration: Invalid launch configuration.
//  - hipErrorCooperativeLaunchTooLarge: Grid exceeds max cooperative size.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorLaunchOutOfResources: Insufficient resources for launch.
//  - hipErrorNotSupported: Device doesn't support cooperative launch.
//
// Synchronization: This operation is asynchronous.
//
// Cooperative kernel requirements:
// - Kernel must use cooperative grid synchronization primitives.
// - All blocks in the grid must be resident simultaneously.
// - Grid size limited by device occupancy and resources.
// - Use hipOccupancyMaxActiveBlocksPerMultiprocessor() to determine
//   maximum grid size.
//
// Cooperative features:
// - Enables grid-wide synchronization via cooperative_groups::this_grid().
// - All blocks can synchronize at barriers.
// - Useful for iterative algorithms requiring global synchronization.
// - Higher launch overhead than regular kernels.
//
// Device requirements:
// - Device must support cooperative launch (check device attributes).
// - Compute capability 6.0+ for NVIDIA, RDNA+ for AMD.
// - Limited by SM/CU count and available resources.
//
// Performance considerations:
// - May reduce occupancy to ensure all blocks are resident.
// - Launch overhead higher than regular kernels.
// - Use only when grid-wide sync is necessary.
//
// Multi-GPU: Cooperative kernels cannot span multiple devices.
// Use hipLaunchCooperativeKernelMultiDevice() for multi-GPU.
//
// Warning: Grid size must not exceed the maximum determined by
// occupancy calculations, or launch will fail.
//
// See also: hipModuleLaunchKernel,
//           hipOccupancyMaxActiveBlocksPerMultiprocessor,
//           hipLaunchCooperativeKernelMultiDevice.
HIPAPI hipError_t hipModuleLaunchCooperativeKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!f) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Resolve NULL stream to default stream.
  if (!stream) {
    stream = (hipStream_t)context->default_stream;
  }

  // Get the device from the stream's context.
  iree_hal_streaming_stream_t* hal_stream =
      (iree_hal_streaming_stream_t*)stream;
  iree_hal_streaming_device_t* device = hal_stream->context->device_entry;

  // Get symbol.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)f;

  // Calculate maximum blocks for cooperative launch.
  // This will return 0 if the device doesn't support cooperative launch.
  int block_size = blockDimX * blockDimY * blockDimZ;
  uint32_t max_blocks = 0;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_calculate_max_cooperative_blocks(
          device, symbol, block_size, sharedMemBytes, &max_blocks),
      hipErrorInvalidValue);

  // Verify grid size doesn't exceed max active blocks.
  // If max_blocks is 0 (device doesn't support cooperative launch) or
  // grid is too large, return error.
  int total_blocks = gridDimX * gridDimY * gridDimZ;
  if (max_blocks == 0 || total_blocks > max_blocks) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorCooperativeLaunchTooLarge);
  }

  // Set up dispatch params with cooperative flag.
  // Cooperative launch always uses kernelParams format.
  const iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {gridDimX, gridDimY, gridDimZ},
      .block_dim = {blockDimX, blockDimY, blockDimZ},
      .shared_memory_bytes = sharedMemBytes,
      .buffer = kernelParams,  // Array of pointers to parameters.
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE,
  };

  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_launch_kernel((iree_hal_streaming_symbol_t*)f, &params,
                                       (iree_hal_streaming_stream_t*)stream),
      hipErrorInvalidConfiguration);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Enqueues a host function callback in a stream.
//
// Parameters:
//  - stream: [IN] Stream to enqueue the callback (NULL = default stream).
//  - fn: [IN] Host function to call when stream reaches this point.
//  - userData: [IN] User data pointer passed to the callback.
//
// Returns:
//  - hipSuccess: Host function enqueued successfully.
//  - hipErrorInvalidValue: fn is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//
// Synchronization: The host function is called asynchronously when
// the stream reaches this operation.
//
// Callback signature:
// ```c
// void hostFunction(void* userData);
// ```
//
// Callback behavior:
// - Called on a runtime thread when stream reaches this point.
// - Blocks stream execution until callback completes.
// - Can make HIP API calls except stream synchronization.
// - Should complete quickly to avoid blocking the stream.
// - Runs after all prior operations in stream complete.
// - Subsequent operations wait for callback to finish.
//
// Restrictions in callback:
// - Cannot call hipStreamSynchronize() on any stream.
// - Cannot call hipDeviceSynchronize().
// - Cannot wait on events from the same stream.
// - Can launch new work to different streams.
//
// Use cases:
// - CPU-GPU task pipelining.
// - Signaling completion to host code.
// - Triggering dependent CPU work.
// - Resource management between kernels.
//
// Performance considerations:
// - Callback runs on runtime thread, not application thread.
// - Long-running callbacks block stream progress.
// - Consider using events for simple synchronization.
//
// Multi-GPU: Callback executes in context of the stream's device.
//
// Warning: Avoid heavy computation in callbacks. Use callbacks for
// quick signaling or launching work on other streams.
//
// Graph capture: Supported. Creates host function node when capturing.
//
// See also: hipStreamAddCallback, hipEventRecord,
//           hipStreamWaitEvent.
HIPAPI hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn,
                                    void* userData) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG("[HIP_API] hipLaunchHostFunc(stream=%p, fn=%p, userData=%p)\n",
                (void*)stream, (void*)fn, userData);
  if (!fn) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status =
      iree_hal_streaming_launch_host_function(stream_obj, fn, userData);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Occupancy functions
//===----------------------------------------------------------------------===//

static bool iree_hip_occupancy_flags_are_valid(unsigned int flags) {
  return (flags & ~(unsigned int)hipOccupancyDisableCachingOverride) == 0;
}

// Calculates maximum active blocks per multiprocessor for a kernel.
//
// Parameters:
//  - numBlocks: [OUT] Pointer to receive max active blocks per SM/CU.
//  - f: [IN] Kernel function handle to analyze.
//  - blockSize: [IN] Block size in threads for the calculation.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: numBlocks or f is NULL, or blockSize <= 0.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Occupancy calculation:
// - Determines theoretical maximum blocks that can be resident.
// - Considers register usage, shared memory, and block size.
// - Returns blocks per single SM/CU, not total device capacity.
// - Actual occupancy may be lower due to launch configuration.
//
// Limiting factors:
// - Register usage per thread.
// - Shared memory (static + dynamic) per block.
// - Maximum threads per multiprocessor.
// - Maximum blocks per multiprocessor.
// - Warp/wavefront scheduling limits.
//
// Usage pattern:
// ```c
// int maxBlocks;
// hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
//     &maxBlocks, kernel, 256, sharedMemSize);
// int numSMs = deviceProps.multiProcessorCount;
// int totalBlocks = maxBlocks * numSMs;
// ```
//
// Performance optimization:
// - Use to find optimal block size for maximum occupancy.
// - Balance between occupancy and resource usage.
// - Higher occupancy doesn't always mean better performance.
//
// Multi-GPU: Calculation is specific to the current device's
// architecture and capabilities.
//
// See also: hipModuleOccupancyMaxPotentialBlockSize,
//           hipFuncGetAttributes, hipDeviceGetAttribute.
HIPAPI hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!numBlocks || !f || blockSize <= 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context and device.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get device properties.
  iree_hal_streaming_device_t* device = context->device_entry;
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  if (!iree_hal_streaming_symbol_has_tag(f)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }
  iree_hal_streaming_symbol_t* symbol = iree_hal_streaming_symbol_untag(f);

  // Verify it's a function.
  if (!symbol || symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  // Use shared occupancy calculation.
  uint32_t max_blocks = 0;
  iree_status_t status =
      iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
          device, symbol, blockSize, dynSharedMemPerBlk, &max_blocks);

  if (iree_status_is_ok(status)) {
    *numBlocks = max_blocks;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Calculates maximum active blocks per multiprocessor with flags.
//
// Parameters:
//  - numBlocks: [OUT] Pointer to receive max active blocks per SM/CU.
//  - f: [IN] Kernel function handle to analyze.
//  - blockSize: [IN] Block size in threads for the calculation.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//  - flags: [IN] Flags to control occupancy calculation behavior.
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: numBlocks or f is NULL, or blockSize <= 0.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Flag options:
// - hipOccupancyDefault: Default behavior.
// - hipOccupancyDisableCachingOverride: Don't override cache config.
// - Additional flags may be defined for specific architectures.
//
// Occupancy calculation:
// - Same as hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.
// - Flags may modify how cache configuration affects calculation.
// - Useful for kernels with specific cache requirements.
//
// Cache considerations:
// - Default may assume optimal cache configuration.
// - Flags can preserve kernel's specified cache config.
// - Important for kernels tuned for specific L1/shared split.
//
// Multi-GPU: Calculation is specific to the current device's
// architecture and capabilities.
//
// See also: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
//           hipFuncSetCacheConfig, hipOccupancyMaxPotentialBlockSize.
HIPAPI hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk,
    unsigned int flags) {
  if (!iree_hip_occupancy_flags_are_valid(flags)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // HRX does not model cache-override effects in occupancy calculation.
  return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, f, blockSize, dynSharedMemPerBlk);
}

// Calculates optimal block and grid size for maximum occupancy.
//
// Parameters:
//  - gridSize: [OUT] Pointer to receive optimal grid size (in blocks).
//  - blockSize: [OUT] Pointer to receive optimal block size (in threads).
//  - f: [IN] Kernel function handle to analyze.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//  - blockSizeLimit: [IN] Maximum block size to consider (0 = no limit).
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: Output pointers or f is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Optimization strategy:
// - Tests multiple block sizes to find best occupancy.
// - Returns block size that maximizes multiprocessor occupancy.
// - Grid size calculated to fully utilize the device.
// - Balances threads per block with active blocks.
//
// Block size selection:
// - Tests powers of 2 and warp/wavefront multiples.
// - Respects kernel's max threads per block limit.
// - Considers register and shared memory constraints.
// - blockSizeLimit caps the maximum tested size.
//
// Grid size calculation:
// - Returns minimum grid to achieve maximum occupancy.
// - Grid size = (maxActiveBlocks * numSMs).
// - May be larger than needed for actual problem size.
// - Application should adjust based on actual work.
//
// Usage pattern:
// ```c
// int minGridSize, blockSize;
// hipModuleOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
//                                         kernel, 0, 0);
// int actualGridSize = (problemSize + blockSize - 1) / blockSize;
// hipModuleLaunchKernel(kernel, actualGridSize, 1, 1,
//                      blockSize, 1, 1, ...);
// ```
//
// Performance notes:
// - Optimal occupancy doesn't guarantee best performance.
// - Consider memory access patterns and arithmetic intensity.
// - May need to tune based on actual kernel behavior.
//
// Multi-GPU: Calculation is specific to the current device.
//
// See also: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
//           hipModuleOccupancyMaxPotentialBlockSizeWithFlags.
HIPAPI hipError_t hipModuleOccupancyMaxPotentialBlockSize(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk,
    int blockSizeLimit) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!gridSize || !blockSize || !f) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context and device.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get device properties.
  iree_hal_streaming_device_t* device = context->device_entry;
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  if (!iree_hal_streaming_symbol_has_tag(f)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }
  iree_hal_streaming_symbol_t* symbol = iree_hal_streaming_symbol_untag(f);

  // Verify it's a function.
  if (!symbol || symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  uint32_t block_size = 0;
  uint32_t min_grid_size = 0;

  // Use shared occupancy calculation.
  // HIP doesn't yet have a C API for dynamic shared memory callbacks.
  // Pass NULL for the callback to use fixed dynamic shared memory size.
  iree_status_t status = iree_hal_streaming_calculate_optimal_block_size(
      device, symbol, (uint32_t)dynSharedMemPerBlk, NULL,
      (uint32_t)blockSizeLimit, &block_size, &min_grid_size);
  *gridSize = min_grid_size;
  *blockSize = block_size;

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Calculates optimal block and grid size with flags.
//
// Parameters:
//  - gridSize: [OUT] Pointer to receive optimal grid size (in blocks).
//  - blockSize: [OUT] Pointer to receive optimal block size (in threads).
//  - f: [IN] Kernel function handle to analyze.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//  - blockSizeLimit: [IN] Maximum block size to consider (0 = no limit).
//  - flags: [IN] Flags to control occupancy calculation behavior.
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: Output pointers or f is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Flag options:
// - hipOccupancyDefault: Default optimization behavior.
// - hipOccupancyDisableCachingOverride: Preserve cache configuration.
// - Additional flags may affect optimization strategy.
//
// Extended behavior:
// - Same as hipModuleOccupancyMaxPotentialBlockSize.
// - Flags may affect how cache configuration is considered.
// - Useful for kernels with specific performance requirements.
//
// Multi-GPU: Calculation is specific to the current device.
//
// See also: hipModuleOccupancyMaxPotentialBlockSize,
//           hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.
HIPAPI hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk,
    int blockSizeLimit, unsigned int flags) {
  if (!iree_hip_occupancy_flags_are_valid(flags)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  return hipModuleOccupancyMaxPotentialBlockSize(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
}

//===----------------------------------------------------------------------===//
// Runtime occupancy functions (for host function pointers)
//===----------------------------------------------------------------------===//

// Calculates maximum active blocks per SM for a host function pointer.
HIPAPI hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk) {
  if (!numBlocks || !f) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Conservative default for AMD GPUs.
  *numBlocks = 1;
  return hipSuccess;
}

// Calculates maximum active blocks per SM with flags.
HIPAPI hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk,
    unsigned int flags) {
  (void)flags;
  return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize,
                                                      dynSharedMemPerBlk);
}

// Calculates optimal block and grid size for a host function pointer.
HIPAPI hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize,
                                                    int* blockSize,
                                                    const void* f,
                                                    size_t dynSharedMemPerBlk,
                                                    int blockSizeLimit) {
  if (!gridSize || !blockSize || !f) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Return conservative defaults that work for most kernels.
  // Block size of 256 is commonly optimal for AMD GPUs.
  *blockSize =
      (blockSizeLimit > 0 && blockSizeLimit < 256) ? blockSizeLimit : 256;
  // Grid size of 1 is a minimum that will work.
  *gridSize = 1;
  return hipSuccess;
}

// Calculates optimal block and grid size with flags.
HIPAPI hipError_t hipOccupancyMaxPotentialBlockSizeWithFlags(
    int* gridSize, int* blockSize, const void* f, size_t dynSharedMemPerBlk,
    int blockSizeLimit, unsigned int flags) {
  (void)flags;
  return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f,
                                           dynSharedMemPerBlk, blockSizeLimit);
}

//===----------------------------------------------------------------------===//
// Unified memory management
//===----------------------------------------------------------------------===//

// Advises about the usage pattern of managed memory.
//
// Parameters:
//  - dev_ptr: [IN] Pointer to managed memory to advise about.
//  - count: [IN] Size in bytes of the memory range.
//  - advice: [IN] Advice to apply to the memory range.
//  - device: [IN] Device ID for device-specific advice.
//
// Returns:
//  - hipSuccess: Advice applied successfully (or ignored).
//  - hipErrorInvalidValue: Invalid pointer, size, or advice.
//  - hipErrorInvalidDevice: Invalid device ID.
//  - hipErrorNotSupported: Advice not supported on this system.
//
// Synchronization: This operation is synchronous.
//
// Advice options:
// - hipMemAdviseSetReadMostly: Data mostly read, rarely written.
// - hipMemAdviseUnsetReadMostly: Clear read-mostly setting.
// - hipMemAdviseSetPreferredLocation: Set preferred device location.
// - hipMemAdviseUnsetPreferredLocation: Clear preferred location.
// - hipMemAdviseSetAccessedBy: Memory will be accessed by device.
// - hipMemAdviseUnsetAccessedBy: Clear accessed-by setting.
//
// Memory migration hints:
// - ReadMostly: Enables read duplication across devices.
// - PreferredLocation: Migrates pages to specified device.
// - AccessedBy: Establishes direct access mapping.
//
// Performance optimization:
// - Reduces page fault overhead for managed memory.
// - Improves memory access patterns across devices.
// - Hints are advisory; system may ignore them.
//
// Managed memory behavior:
// - Only applies to memory allocated with hipMallocManaged.
// - Advice persists until explicitly changed or memory freed.
// - Can significantly impact multi-GPU performance.
//
// Multi-GPU: Advice can specify different devices for different
// memory regions to optimize NUMA behavior.
//
// Warning: Incorrect advice may degrade performance. Profile to
// verify improvements.
//
// See also: hipMallocManaged, hipMemPrefetchAsync,
//           hipMemRangeGetAttribute.
HIPAPI hipError_t hipMemAdvise(const void* dev_ptr, size_t count,
                               hipMemAdvise_t advice, int device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dev_ptr || count == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t result =
      iree_hip_managed_apply_advice(context, dev_ptr, count, advice, device);

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

HIPAPI hipError_t hipMemAdvise_v2(const void* dev_ptr, size_t count,
                                  hipMemoryAdvise advice,
                                  hipMemLocation location) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dev_ptr || count == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  int device = hipInvalidDeviceId;
  hipError_t result = iree_hip_managed_location_to_device(location, &device);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_context_t* context = NULL;
  result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  result = iree_hip_managed_apply_advice(context, dev_ptr, count,
                                         (hipMemAdvise_t)advice, device);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Asynchronously prefetches managed memory to a device.
//
// Parameters:
//  - dev_ptr: [IN] Pointer to managed memory to prefetch.
//  - count: [IN] Size in bytes to prefetch.
//  - device: [IN] Destination device ID (or hipCpuDeviceId for host).
//  - stream: [IN] Stream for asynchronous prefetch.
//
// Returns:
//  - hipSuccess: Prefetch initiated successfully (or ignored).
//  - hipErrorInvalidValue: Invalid pointer, size, or stream.
//  - hipErrorInvalidDevice: Invalid device ID.
//  - hipErrorNotSupported: Prefetch not supported on this system.
//
// Synchronization: This operation is asynchronous.
//
// Prefetch behavior:
// - Migrates pages to specified device before access.
// - Reduces page fault latency during kernel execution.
// - Operation is enqueued in the specified stream.
// - Overlaps with other stream operations.
//
// Performance benefits:
// - Eliminates page faults during kernel execution.
// - Enables overlap of data migration with computation.
// - Critical for managed memory performance.
//
// Memory requirements:
// - Only applies to hipMallocManaged allocations.
// - Pages must be resident in system memory.
// - Prefetch may be ignored if pages already on device.
//
// Stream ordering:
// - Prefetch completes before subsequent operations in stream.
// - Use events or stream synchronization to ensure completion.
// - Can prefetch to different devices in different streams.
//
// Multi-GPU patterns:
// - Prefetch to device before kernel launch.
// - Prefetch to hipCpuDeviceId for host access.
// - Pipeline prefetches across multiple devices.
//
// Warning: Prefetching large amounts may cause memory pressure.
// Monitor available device memory.
//
// See also: hipMallocManaged, hipMemAdvise, hipStreamSynchronize,
//           hipMemRangeGetAttribute.
HIPAPI hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count,
                                      int device, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dev_ptr || count == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t result =
      iree_hip_validate_managed_location(device, /*allow_cpu=*/true);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_context_t* context = NULL;
  result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  result = iree_hip_resolve_stream(stream, &stream_obj);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  result = iree_hip_managed_record_prefetch(context, dev_ptr, count, device);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

HIPAPI hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count,
                                         hipMemLocation location,
                                         unsigned int flags,
                                         hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dev_ptr || count == 0 || flags != 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  int device = hipInvalidDeviceId;
  hipError_t result = iree_hip_managed_location_to_device(location, &device);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_context_t* context = NULL;
  result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  result = iree_hip_resolve_stream(stream, &stream_obj);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  result = iree_hip_managed_record_prefetch(context, dev_ptr, count, device);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

static hipError_t iree_hip_prefetch_batch_location_for_operation(
    size_t operation, const hipMemLocation* prefetch_locs,
    const size_t* prefetch_loc_idxs, size_t num_prefetch_locs,
    hipMemLocation* out_location) {
  IREE_ASSERT_ARGUMENT(prefetch_locs);
  IREE_ASSERT_ARGUMENT(prefetch_loc_idxs);
  IREE_ASSERT_ARGUMENT(out_location);

  size_t location_index = 0;
  for (size_t i = 1; i < num_prefetch_locs; ++i) {
    if (prefetch_loc_idxs[i] > operation) break;
    location_index = i;
  }
  *out_location = prefetch_locs[location_index];
  return hipSuccess;
}

HIPAPI hipError_t hipMemPrefetchBatchAsync(
    void** dev_ptrs, size_t* sizes, size_t count, hipMemLocation* prefetch_locs,
    size_t* prefetch_loc_idxs, size_t num_prefetch_locs,
    unsigned long long flags, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dev_ptrs || !sizes || !prefetch_locs || !prefetch_loc_idxs ||
      count == 0 || num_prefetch_locs == 0 || num_prefetch_locs > count ||
      flags != 0 || !stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (prefetch_loc_idxs[0] != 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  for (size_t i = 1; i < num_prefetch_locs; ++i) {
    if (prefetch_loc_idxs[i] <= prefetch_loc_idxs[i - 1] ||
        prefetch_loc_idxs[i] >= count) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }

  int ignored_device = hipInvalidDeviceId;
  for (size_t i = 0; i < num_prefetch_locs; ++i) {
    hipError_t result =
        iree_hip_managed_location_to_device(prefetch_locs[i], &ignored_device);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
  }

  iree_hal_streaming_context_t* context = NULL;
  hipError_t result = iree_hip_ensure_context(&context);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  result = iree_hip_resolve_stream(stream, &stream_obj);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  for (size_t i = 0; i < count; ++i) {
    if (!dev_ptrs[i] || sizes[i] == 0) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }

    hipMemLocation location;
    result = iree_hip_prefetch_batch_location_for_operation(
        i, prefetch_locs, prefetch_loc_idxs, num_prefetch_locs, &location);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
    int device = hipInvalidDeviceId;
    result = iree_hip_managed_location_to_device(location, &device);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
    result = iree_hip_managed_record_prefetch(context, dev_ptrs[i], sizes[i],
                                              device);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

typedef enum iree_hip_pointer_metadata_kind_e {
  IREE_HIP_POINTER_METADATA_INVALID = 0,
  IREE_HIP_POINTER_METADATA_BUFFER,
  IREE_HIP_POINTER_METADATA_ARRAY,
  IREE_HIP_POINTER_METADATA_STATIC_MANAGED,
  IREE_HIP_POINTER_METADATA_UNREGISTERED_HOST,
} iree_hip_pointer_metadata_kind_t;

typedef struct iree_hip_pointer_metadata_t {
  // Kind of pointer metadata represented by this value.
  iree_hip_pointer_metadata_kind_t kind;
  // HIP memory type reported for the pointer.
  hipMemoryType memory_type;
  // Owner context retained while |buffer_ref| is used.
  iree_hal_streaming_context_t* owner_context;
  // Streaming buffer reference when |kind| is BUFFER.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  // Array metadata when |kind| is ARRAY.
  struct hipArray_st* array;
  // Registered symbol size when |kind| is STATIC_MANAGED.
  size_t symbol_size;
} iree_hip_pointer_metadata_t;

static hipMemoryType iree_hip_memory_type_for_buffer(
    const iree_hal_streaming_buffer_t* buffer) {
  if (buffer->is_managed) return hipMemoryTypeManaged;
  return buffer->host_ptr ? hipMemoryTypeHost : hipMemoryTypeDevice;
}

static bool iree_hip_pointer_is_cpu_mapped(const void* ptr) {
#ifdef __linux__
  if (!ptr) return false;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) return false;
  const uintptr_t address = (uintptr_t)ptr;
  const uintptr_t page_address = address - (address % (uintptr_t)page_size);
  unsigned char residency = 0;
  errno = 0;
  return mincore((void*)page_address, 1, &residency) == 0;
#else
  (void)ptr;
  return false;
#endif
}

static bool iree_hip_pointer_is_registered_managed_variable(const void* ptr,
                                                            size_t* out_size) {
  if (out_size) *out_size = 0;
  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) return false;
  iree_hal_streaming_symbol_type_t symbol_type =
      IREE_HAL_STREAMING_SYMBOL_TYPE_UNDEFINED;
  size_t symbol_size = 0;
  if (!iree_hal_streaming_global_symbol_registry_query_variable(
          registry, (void*)ptr, &symbol_type, &symbol_size)) {
    return false;
  }
  if (symbol_type != IREE_HAL_STREAMING_SYMBOL_TYPE_DATA) return false;
  if (out_size) *out_size = symbol_size;
  return true;
}

static hipError_t iree_hip_query_pointer_metadata(
    iree_hal_streaming_context_t* context, const void* ptr,
    iree_hip_pointer_metadata_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_metadata);
  memset(out_metadata, 0, sizeof(*out_metadata));
  out_metadata->kind = IREE_HIP_POINTER_METADATA_INVALID;

  hipError_t lookup_result = iree_hip_lookup_streaming_range_with_owner(
      context, ptr, 1, &out_metadata->owner_context, &out_metadata->buffer_ref);
  if (lookup_result == hipSuccess) {
    out_metadata->kind = IREE_HIP_POINTER_METADATA_BUFFER;
    out_metadata->memory_type =
        iree_hip_memory_type_for_buffer(out_metadata->buffer_ref.buffer);
    return hipSuccess;
  }

  if (iree_hip_array_registry_lookup((hipArray_const_t)ptr,
                                     &out_metadata->array)) {
    out_metadata->kind = IREE_HIP_POINTER_METADATA_ARRAY;
    out_metadata->memory_type = hipMemoryTypeArray;
    return hipSuccess;
  }

  if (iree_hip_pointer_is_registered_managed_variable(
          ptr, &out_metadata->symbol_size)) {
    out_metadata->kind = IREE_HIP_POINTER_METADATA_STATIC_MANAGED;
    out_metadata->memory_type = hipMemoryTypeManaged;
    return hipSuccess;
  }

  if (iree_hip_pointer_is_cpu_mapped(ptr)) {
    out_metadata->kind = IREE_HIP_POINTER_METADATA_UNREGISTERED_HOST;
    out_metadata->memory_type = hipMemoryTypeUnregistered;
    return hipSuccess;
  }

  return hipErrorInvalidValue;
}

static void iree_hip_pointer_metadata_release(
    iree_hip_pointer_metadata_t* metadata) {
  if (!metadata) return;
  iree_hal_streaming_context_release(metadata->owner_context);
  iree_hip_array_release(metadata->array);
  memset(metadata, 0, sizeof(*metadata));
}

// Queries a single attribute of a pointer.
//
// Parameters:
//  - data: [OUT] Pointer to receive the attribute value.
//  - attribute: [IN] Attribute to query (hipPointer_attribute_t enum).
//  - ptr: [IN] Pointer to query.
//
// Returns:
//  - hipSuccess: Attribute queried successfully.
//  - hipErrorInvalidValue: data is NULL or invalid attribute.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidMemoryHandle: Pointer not recognized.
//
// Synchronization: This operation is synchronous and immediate.
//
// Available attributes:
// - HIP_POINTER_ATTRIBUTE_CONTEXT: Context that owns the memory.
// - HIP_POINTER_ATTRIBUTE_MEMORY_TYPE: Type of memory.
// - HIP_POINTER_ATTRIBUTE_DEVICE_POINTER: Device pointer value.
// - HIP_POINTER_ATTRIBUTE_HOST_POINTER: Host pointer value.
// - HIP_POINTER_ATTRIBUTE_P2P_TOKENS: P2P tokens for IPC.
// - HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: Synchronous mem ops flag.
// - HIP_POINTER_ATTRIBUTE_BUFFER_ID: Unique buffer identifier.
// - HIP_POINTER_ATTRIBUTE_IS_MANAGED: Is managed memory.
// - HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL: Device ordinal.
// - HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: Allowed export types.
// - HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: RDMA capable.
// - HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS: Access permissions.
// - HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: Memory pool handle.
//
// Memory types:
// - Device memory: Allocated with hipMalloc.
// - Host memory: Allocated with hipHostMalloc or registered.
// - Managed memory: Allocated with hipMallocManaged.
// - Unregistered: Regular system memory.
//
// Usage patterns:
// - Determine memory type before operations.
// - Check if pointer is valid device memory.
// - Get device association for multi-GPU.
//
// Multi-GPU: Returns device-specific information for the pointer.
//
// See also: hipPointerGetAttributes, hipPointerSetAttribute,
//           hipMemGetInfo.
HIPAPI hipError_t hipPointerGetAttribute(void* data,
                                         hipPointer_attribute_t attribute,
                                         hipDeviceptr_t ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  if (attribute == HIP_POINTER_ATTRIBUTE_MEMORY_TYPE ||
      attribute == HIP_POINTER_ATTRIBUTE_RANGE_SIZE) {
    iree_hip_pointer_metadata_t metadata;
    hipError_t metadata_result =
        iree_hip_query_pointer_metadata(context, ptr, &metadata);
    if (metadata_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(metadata_result);
    }
    if (attribute == HIP_POINTER_ATTRIBUTE_MEMORY_TYPE) {
      *(hipMemoryType*)data = metadata.memory_type;
    } else {
      switch (metadata.kind) {
        case IREE_HIP_POINTER_METADATA_BUFFER:
          *(size_t*)data = metadata.buffer_ref.buffer->logical_size;
          break;
        case IREE_HIP_POINTER_METADATA_ARRAY:
          *(size_t*)data = metadata.array ? metadata.array->allocation_size : 0;
          break;
        case IREE_HIP_POINTER_METADATA_STATIC_MANAGED:
          *(size_t*)data = metadata.symbol_size;
          break;
        case IREE_HIP_POINTER_METADATA_UNREGISTERED_HOST:
        case IREE_HIP_POINTER_METADATA_INVALID:
        default:
          iree_hip_pointer_metadata_release(&metadata);
          IREE_TRACE_ZONE_END(z0);
          HIP_RETURN_ERROR(hipErrorInvalidValue);
      }
    }
    iree_hip_pointer_metadata_release(&metadata);
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_ref_t buffer_ref;
  hipError_t lookup_result = iree_hip_lookup_streaming_range_with_owner(
      context, (const void*)ptr, 1, &owner_context, &buffer_ref);
  if (lookup_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(lookup_result);
  }

  hipError_t result = hipSuccess;

  switch (attribute) {
    case HIP_POINTER_ATTRIBUTE_CONTEXT: {
      // Return the context handle.
      *(hipCtx_t*)data = (hipCtx_t)buffer_ref.buffer->context;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_DEVICE_POINTER: {
      // Return the device pointer.
      *(hipDeviceptr_t*)data =
          (hipDeviceptr_t)((iree_device_size_t)buffer_ref.buffer->device_ptr +
                           buffer_ref.offset);
      break;
    }
    case HIP_POINTER_ATTRIBUTE_HOST_POINTER: {
      if (!buffer_ref.buffer->host_ptr) {
        result = hipErrorInvalidValue;
        break;
      }
      *(void**)data = (void*)((iree_host_size_t)buffer_ref.buffer->host_ptr +
                              buffer_ref.offset);
      break;
    }
    case HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL: {
      // Return the device ordinal.
      *(int*)data = (int)buffer_ref.buffer->context->device_ordinal;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_IS_MANAGED: {
      *(unsigned int*)data = buffer_ref.buffer->is_managed ? 1 : 0;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR: {
      // Return the base address of the allocation.
      if (buffer_ref.buffer->is_managed && buffer_ref.buffer->host_ptr) {
        *(hipDeviceptr_t*)data = (hipDeviceptr_t)buffer_ref.buffer->host_ptr;
      } else if (buffer_ref.buffer->host_ptr) {
        *(hipDeviceptr_t*)data = (hipDeviceptr_t)buffer_ref.buffer->host_ptr;
      } else {
        *(hipDeviceptr_t*)data = (hipDeviceptr_t)buffer_ref.buffer->device_ptr;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_RANGE_SIZE: {
      // Return the size of the allocation.
      *(size_t*)data = buffer_ref.buffer->logical_size;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_MAPPED: {
      // This attribute describes whether the allocation has a runtime-visible
      // device mapping, not whether it also has a host alias.
      *(unsigned int*)data = 1;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: {
      // Synchronous memory operations flag.
      *(unsigned int*)data = 1;  // Default to synchronous.
      break;
    }
    case HIP_POINTER_ATTRIBUTE_BUFFER_ID: {
      // Return a unique buffer ID (use pointer as ID).
      *(unsigned long long*)data = (unsigned long long)buffer_ref.buffer;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_P2P_TOKENS:
    case HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS:
    case HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
      // These attributes are not supported yet.
      result = hipErrorNotSupported;
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  iree_hal_streaming_context_release(owner_context);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets an attribute of a pointer.
//
// Parameters:
//  - value: [IN] Pointer to the new attribute value.
//  - attribute: [IN] Attribute to set (hipPointer_attribute_t enum).
//  - ptr: [IN] Pointer to modify.
//
// Returns:
//  - hipSuccess: Attribute set successfully.
//  - hipErrorInvalidValue: value is NULL or invalid attribute.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidMemoryHandle: Pointer not recognized.
//  - hipErrorNotSupported: Attribute cannot be modified.
//
// Synchronization: This operation is synchronous.
//
// Settable attributes:
// - HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: Enable synchronous operations.
// - HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS: Set access permissions.
// - Limited subset of attributes are modifiable.
//
// Sync memory operations:
// - When enabled, memory operations complete synchronously.
// - Affects hipMemcpy behavior with this pointer.
// - Default is typically asynchronous for device memory.
//
// Access flags:
// - Control read/write permissions.
// - May affect memory protection and caching.
// - Platform and device specific.
//
// Restrictions:
// - Most attributes are read-only.
// - Changes may not take effect immediately.
// - Some attributes require specific hardware support.
//
// Usage patterns:
// - Configure memory behavior for specific use cases.
// - Optimize memory access patterns.
// - Control synchronization behavior.
//
// Multi-GPU: Attributes are set per pointer, affecting all
// devices that access the memory.
//
// Warning: Changing attributes may affect performance or
// correctness of concurrent operations.
//
// See also: hipPointerGetAttribute, hipPointerGetAttributes,
//           hipMemcpyAsync.
HIPAPI hipError_t hipPointerSetAttribute(const void* value,
                                         hipPointer_attribute_t attribute,
                                         hipDeviceptr_t ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)(uintptr_t)ptr, &buffer_ref);

  // If lookup fails, the pointer might not be a valid allocation.
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(
        iree_status_to_fixed_hip_result(status, hipErrorInvalidDevicePointer));
  }

  hipError_t result = hipSuccess;
  switch (attribute) {
    case HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: {
      // Set synchronous memory operations flag.
      // Note: We accept the value but don't store it since all our operations
      // are currently synchronous by default.
      unsigned int sync_value = *(const unsigned int*)value;
      (void)sync_value;  // Suppress unused variable warning.
      break;
    }
    case HIP_POINTER_ATTRIBUTE_CONTEXT:
    case HIP_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case HIP_POINTER_ATTRIBUTE_DEVICE_POINTER:
    case HIP_POINTER_ATTRIBUTE_HOST_POINTER:
    case HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case HIP_POINTER_ATTRIBUTE_IS_MANAGED:
    case HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    case HIP_POINTER_ATTRIBUTE_RANGE_SIZE:
    case HIP_POINTER_ATTRIBUTE_MAPPED:
    case HIP_POINTER_ATTRIBUTE_BUFFER_ID:
    case HIP_POINTER_ATTRIBUTE_P2P_TOKENS:
    case HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS:
    case HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
      // These attributes are read-only and cannot be set.
      result = hipErrorNotSupported;
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Queries multiple attributes of a pointer in one call.
//
// Parameters:
//  - numAttributes: [IN] Number of attributes to query.
//  - attributes: [IN] Array of attributes to query.
//  - data: [IN/OUT] Array of pointers to receive attribute values.
//  - ptr: [IN] Pointer to query.
//
// Returns:
//  - hipSuccess: All attributes queried successfully.
//  - hipErrorInvalidValue: Invalid parameters or numAttributes is 0.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidMemoryHandle: Pointer not recognized.
//
// Synchronization: This operation is synchronous and immediate.
//
// Batch query behavior:
// - More efficient than multiple hipPointerGetAttribute calls.
// - Each data[i] receives value for attributes[i].
// - All attributes queried even if some fail.
// - Returns first error encountered, but continues.
//
// Data array setup:
// - Each data[i] must point to appropriate type for attribute.
// - Size depends on attribute (int*, void**, size_t*, etc.).
// - Caller must allocate storage before call.
//
// Example usage:
// ```c
// hipPointer_attribute_t attrs[] = {
//     HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
//     HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
// };
// hipMemoryType memType;
// int device;
// void* data[] = {&memType, &device};
// hipPointerGetAttributes(2, attrs, data, ptr);
// ```
//
// Error handling:
// - If any attribute fails, that data element is undefined.
// - Check return value before using results.
// - Some attributes may not be supported for all pointers.
//
// Multi-GPU: Returns device-specific information for each attribute.
//
// See also: hipPointerGetAttribute, hipPointerSetAttribute,
//           hipMemGetInfo.
HIPAPI hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes,
                                             hipPointer_attribute_t* attributes,
                                             void** data, const void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!attributes || !data || numAttributes == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Query each attribute individually using hipPointerGetAttribute.
  hipError_t result = hipSuccess;
  for (unsigned int i = 0; i < numAttributes; i++) {
    hipError_t attr_result =
        hipPointerGetAttribute(data[i], attributes[i], (hipDeviceptr_t)ptr);
    if (attr_result != hipSuccess) {
      // Return the first error encountered.
      if (result == hipSuccess) {
        result = attr_result;
      }
      // Continue to try other attributes even if one fails.
    }
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Queries pointer attributes (runtime API).
//
// Parameters:
//  - attributes: [OUT] Structure to receive pointer attributes.
//  - ptr: [IN] Pointer to query.
//
// Returns:
//  - hipSuccess: Attributes queried successfully.
//  - hipErrorInvalidValue: Null attributes pointer.
//  - hipErrorInvalidDevice: Pointer not found in any device context.
//
// The attributes structure includes:
//  - type: Memory type (host, device, managed, unified).
//  - device: Device ordinal where memory resides.
//  - devicePointer: Device pointer (same as ptr for device memory).
//  - hostPointer: Host pointer (if available).
//  - isManaged: Whether memory is managed (unified memory).
//  - allocationFlags: Flags used during allocation.
//
// See also: hipPointerGetAttribute, hipMalloc, hipHostMalloc.
HIPAPI hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes,
                                          const void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!attributes) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Initialize with defaults.
  memset(attributes, 0, sizeof(*attributes));

  iree_hip_pointer_metadata_t metadata;
  hipError_t metadata_result =
      iree_hip_query_pointer_metadata(context, ptr, &metadata);
  if (metadata_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(metadata_result);
  }

  attributes->type = metadata.memory_type;
  switch (metadata.kind) {
    case IREE_HIP_POINTER_METADATA_BUFFER: {
      iree_hal_streaming_buffer_ref_t buffer_ref = metadata.buffer_ref;
      if (buffer_ref.buffer->host_ptr) {
        attributes->hostPointer =
            (void*)((iree_host_size_t)buffer_ref.buffer->host_ptr +
                    buffer_ref.offset);
      }
      attributes->devicePointer =
          (void*)((iree_device_size_t)buffer_ref.buffer->device_ptr +
                  buffer_ref.offset);
      attributes->device = (int)buffer_ref.buffer->context->device_ordinal;
      attributes->isManaged = buffer_ref.buffer->is_managed ? 1 : 0;
      attributes->allocationFlags =
          (unsigned int)buffer_ref.buffer->host_register_flags;
      break;
    }
    case IREE_HIP_POINTER_METADATA_ARRAY:
      attributes->device = metadata.array && metadata.array->context
                               ? (int)metadata.array->context->device_ordinal
                               : (int)context->device_ordinal;
      attributes->devicePointer =
          metadata.array ? (void*)metadata.array->device_ptr : NULL;
      attributes->allocationFlags = metadata.array ? metadata.array->flags : 0;
      break;
    case IREE_HIP_POINTER_METADATA_STATIC_MANAGED:
      attributes->device = (int)context->device_ordinal;
      attributes->devicePointer = (void*)ptr;
      attributes->hostPointer = (void*)ptr;
      attributes->isManaged = 1;
      break;
    case IREE_HIP_POINTER_METADATA_UNREGISTERED_HOST:
      attributes->device = (int)context->device_ordinal;
      break;
    case IREE_HIP_POINTER_METADATA_INVALID:
    default:
      iree_hip_pointer_metadata_release(&metadata);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hip_pointer_metadata_release(&metadata);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipSuccess);
}

// Queries an attribute of a memory range.
//
// Parameters:
//  - data: [OUT] Buffer to receive attribute values.
//  - data_size: [IN] Size of the data buffer in bytes.
//  - attribute: [IN] Attribute to query (hipMemRangeAttribute enum).
//  - dev_ptr: [IN] Start of memory range to query.
//  - count: [IN] Size of memory range in bytes.
//
// Returns:
//  - hipSuccess: Attribute queried successfully.
//  - hipErrorInvalidValue: Invalid parameters or buffer too small.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotSupported: Attribute not supported.
//
// Synchronization: This operation is synchronous and immediate.
//
// Available attributes:
// - hipMemRangeAttributeReadMostly: Pages are read-mostly.
// - hipMemRangeAttributePreferredLocation: Preferred location.
// - hipMemRangeAttributeAccessedBy: Devices with access.
// - hipMemRangeAttributeLastPrefetchLocation: Last prefetch location.
//
// Attribute data formats:
// - ReadMostly: int (0 or 1).
// - PreferredLocation: int (device ID or hipCpuDeviceId).
// - AccessedBy: Array of int (device IDs).
// - LastPrefetchLocation: int (device ID).
//
// Memory range behavior:
// - Range can span multiple pages.
// - Attributes may vary across pages.
// - Returns aggregate or first value depending on attribute.
//
// Managed memory specific:
// - Most attributes only apply to hipMallocManaged memory.
// - Regular allocations return default values.
// - Use to verify memory migration and access patterns.
//
// Performance analysis:
// - Check where pages are currently resident.
// - Verify prefetch and migration effectiveness.
// - Understand access patterns across devices.
//
// Multi-GPU: Attributes reflect multi-device state and
// can show which devices have access.
//
// See also: hipMemRangeGetAttributes, hipMemAdvise,
//           hipMemPrefetchAsync.
HIPAPI hipError_t hipMemRangeGetAttribute(void* data, size_t data_size,
                                          hipMemRangeAttribute attribute,
                                          const void* dev_ptr, size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data || data_size == 0 || !dev_ptr || count == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_ref_t buffer_ref;
  hipError_t lookup_result = iree_hip_lookup_advisable_range(
      context, dev_ptr, count, &owner_context, &buffer_ref);
  if (lookup_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(lookup_result);
  }

  iree_host_size_t first_page = 0;
  iree_host_size_t page_count = 0;
  if (!iree_hip_managed_memory_page_range(
          &buffer_ref, (iree_device_size_t)count, &first_page, &page_count)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Map the attribute and return the appropriate value.
  hipError_t result = hipSuccess;
  switch (attribute) {
    case hipMemRangeAttributeReadMostly:
      // Return 1 if all pages have read-duplication enabled.
      if (data_size != sizeof(int)) {
        result = hipErrorInvalidValue;
      } else {
        *(int*)data = iree_hip_managed_all_pages_read_mostly(
                          buffer_ref.buffer->managed_read_mostly_pages,
                          first_page, page_count)
                          ? 1
                          : 0;
      }
      break;

    case hipMemRangeAttributePreferredLocation:
      // Return the preferred device ID or hipCpuDeviceId (-1).
      if (data_size != sizeof(int)) {
        result = hipErrorInvalidValue;
      } else {
        *(int*)data = iree_hip_managed_uniform_location(
            buffer_ref.buffer->managed_preferred_locations, first_page,
            page_count);
      }
      break;

    case hipMemRangeAttributeAccessedBy:
      if (data_size == 0 || data_size % sizeof(int) != 0) {
        result = hipErrorInvalidValue;
      } else {
        const iree_host_size_t device_count =
            (iree_host_size_t)(data_size / sizeof(int));
        uint64_t mask = iree_hip_managed_common_accessed_by_mask(
            buffer_ref.buffer->managed_accessed_by_device_masks, first_page,
            page_count);
        iree_hip_managed_fill_accessed_by(mask, (int*)data, device_count);
      }
      break;

    case hipMemRangeAttributeLastPrefetchLocation:
      // Return the last prefetch location.
      if (data_size != sizeof(int)) {
        result = hipErrorInvalidValue;
      } else {
        *(int*)data = iree_hip_managed_uniform_location(
            buffer_ref.buffer->managed_last_prefetch_locations, first_page,
            page_count);
      }
      break;

    case hipMemRangeAttributeCoherencyMode:
      if (data_size != sizeof(int)) {
        result = hipErrorInvalidValue;
      } else {
        *(int*)data = iree_hip_managed_uniform_coherency_mode(
            buffer_ref.buffer->managed_coherency_modes, first_page, page_count);
      }
      break;

    default:
      result = hipErrorInvalidValue;
      break;
  }

  iree_hal_streaming_context_release(owner_context);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Queries multiple attributes of a memory range in one call.
//
// Parameters:
//  - data: [OUT] Array of buffers to receive attribute values.
//  - data_sizes: [IN] Array of buffer sizes in bytes.
//  - attributes: [IN] Array of attributes to query.
//  - num_attributes: [IN] Number of attributes to query.
//  - dev_ptr: [IN] Start of memory range to query.
//  - count: [IN] Size of memory range in bytes.
//
// Returns:
//  - hipSuccess: All attributes queried successfully.
//  - hipErrorInvalidValue: Invalid parameters or num_attributes is 0.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotSupported: One or more attributes not supported.
//
// Synchronization: This operation is synchronous and immediate.
//
// Batch query behavior:
// - More efficient than multiple hipMemRangeGetAttribute calls.
// - Each data[i] receives value for attributes[i].
// - data_sizes[i] must be large enough for attribute data.
// - All attributes queried even if some fail.
//
// Example usage:
// ```c
// hipMemRangeAttribute attrs[] = {
//     hipMemRangeAttributeReadMostly,
//     hipMemRangeAttributePreferredLocation
// };
// int readMostly, preferredLoc;
// void* data[] = {&readMostly, &preferredLoc};
// size_t sizes[] = {sizeof(int), sizeof(int)};
// hipMemRangeGetAttributes(data, sizes, attrs, 2, ptr, size);
// ```
//
// Error handling:
// - If any attribute fails, that data element is undefined.
// - Returns first error encountered but continues.
// - Check return value before using results.
//
// Multi-GPU: Attributes reflect state across all devices
// that have access to the memory range.
//
// See also: hipMemRangeGetAttribute, hipMemAdvise,
//           hipPointerGetAttributes.
HIPAPI hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                           hipMemRangeAttribute* attributes,
                                           size_t num_attributes,
                                           const void* dev_ptr, size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data || !data_sizes || !attributes || num_attributes == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Query each attribute individually using hipMemRangeGetAttribute.
  hipError_t result = hipSuccess;
  for (size_t i = 0; i < num_attributes; i++) {
    hipError_t attr_result = hipMemRangeGetAttribute(
        data[i], data_sizes[i], attributes[i], dev_ptr, count);
    if (attr_result != hipSuccess) {
      // Return the first error encountered.
      if (result == hipSuccess) {
        result = attr_result;
      }
      // Continue to try other attributes even if one fails.
    }
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// HIP graphs
//===----------------------------------------------------------------------===//

// Creates an empty task graph.
//
// Parameters:
//  - pGraph: [OUT] Pointer to receive the created graph handle.
//  - flags: [IN] Graph creation flags (must be 0).
//
// Returns:
//  - hipSuccess: Graph created successfully.
//  - hipErrorInvalidValue: pGraph is NULL or flags is non-zero.
//  - hipErrorMemoryAllocation: Insufficient memory.
//  - hipErrorNotSupported: Graphs not supported on this device.
//
// Synchronization: This operation is synchronous.
//
// Graph concepts:
// - Directed acyclic graph (DAG) of GPU operations.
// - Nodes represent kernels, memcpy, memset, or host callbacks.
// - Edges represent dependencies between operations.
// - Captured once, launched multiple times.
//
// Graph benefits:
// - Reduced launch overhead for repeated workloads.
// - Optimized scheduling and resource allocation.
// - Better GPU utilization through known dependencies.
// - Enables whole-graph optimizations.
//
// Graph workflow:
// 1. Create empty graph with hipGraphCreate.
// 2. Add nodes with hipGraphAdd* functions.
// 3. Define dependencies between nodes.
// 4. Instantiate to create executable with hipGraphInstantiate.
// 5. Launch executable multiple times with hipGraphLaunch.
// 6. Destroy graph and executable when done.
//
//===----------------------------------------------------------------------===//
// User Objects
//===----------------------------------------------------------------------===//

struct hipUserObject {
  // References held by the live-handle registry, graphs, and active API calls.
  // This count protects the control block but is not its public reference
  // count.
  iree_atomic_int32_t handle_reference_count;
  // Next live user-object handle in the process registry.
  struct hipUserObject* next_live_object;
  // Protects reference count changes from user and graph-owner threads.
  iree_slim_mutex_t mutex;
  // Current retained reference count.
  uint64_t ref_count;
  // Number of graph records that still reference this control block.
  uint64_t graph_association_count;
  // Whether the user payload destructor has been invoked.
  bool destructor_called;
  // User pointer passed to the destructor callback.
  void* ptr;
  // Destructor callback invoked when the retained count reaches zero.
  hipHostFn_t destroy;
  // Allocator used for this handle.
  iree_allocator_t host_allocator;
};

static iree_once_flag iree_hip_user_object_registry_mutex_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hip_user_object_registry_mutex;
static hipUserObject_t iree_hip_user_object_registry_head = NULL;

static void iree_hip_user_object_registry_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_user_object_registry_mutex);
}

static void iree_hip_user_object_registry_lock(void) {
  iree_call_once(&iree_hip_user_object_registry_mutex_once,
                 iree_hip_user_object_registry_mutex_initialize);
  iree_slim_mutex_lock(&iree_hip_user_object_registry_mutex);
}

static void iree_hip_user_object_handle_release(hipUserObject_t user_object) {
  if (user_object && iree_atomic_fetch_sub(&user_object->handle_reference_count,
                                           1, iree_memory_order_acq_rel) == 1) {
    iree_slim_mutex_deinitialize(&user_object->mutex);
    iree_allocator_free(user_object->host_allocator, user_object);
  }
}

static void iree_hip_user_object_registry_insert(hipUserObject_t user_object) {
  iree_hip_user_object_registry_lock();
  user_object->next_live_object = iree_hip_user_object_registry_head;
  iree_hip_user_object_registry_head = user_object;
  iree_slim_mutex_unlock(&iree_hip_user_object_registry_mutex);
}

static bool iree_hip_user_object_registry_lookup_retain(
    hipUserObject_t object, hipUserObject_t* out_object) {
  if (out_object) *out_object = NULL;
  if (!object) return false;

  bool found = false;
  iree_hip_user_object_registry_lock();
  for (hipUserObject_t current = iree_hip_user_object_registry_head; current;
       current = current->next_live_object) {
    if (current == object) {
      iree_atomic_fetch_add(&current->handle_reference_count, 1,
                            iree_memory_order_relaxed);
      if (out_object) *out_object = current;
      found = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_user_object_registry_mutex);
  return found;
}

static bool iree_hip_user_object_registry_remove(hipUserObject_t user_object) {
  bool removed = false;
  iree_hip_user_object_registry_lock();
  hipUserObject_t* current = &iree_hip_user_object_registry_head;
  while (*current) {
    if (*current == user_object) {
      *current = user_object->next_live_object;
      user_object->next_live_object = NULL;
      removed = true;
      break;
    }
    current = &(*current)->next_live_object;
  }
  iree_slim_mutex_unlock(&iree_hip_user_object_registry_mutex);
  if (removed) {
    iree_hip_user_object_handle_release(user_object);
  }
  return removed;
}

static hipError_t iree_hip_user_object_retain_refs_checked(
    hipUserObject_t user_object, uint64_t count) {
  if (!user_object || count == 0) return hipErrorInvalidValue;
  hipError_t result = hipSuccess;
  iree_slim_mutex_lock(&user_object->mutex);
  if (user_object->destructor_called) {
    // A completed user object cannot be resurrected.
  } else if (UINT64_MAX - user_object->ref_count < count) {
    result = hipErrorInvalidValue;
  } else {
    user_object->ref_count += count;
  }
  iree_slim_mutex_unlock(&user_object->mutex);
  return result;
}

static hipError_t iree_hip_user_object_check_ref_count(
    hipUserObject_t user_object, uint64_t count) {
  if (!user_object || count == 0) return hipErrorInvalidValue;
  hipError_t result = hipSuccess;
  iree_slim_mutex_lock(&user_object->mutex);
  if (!user_object->destructor_called && count > user_object->ref_count) {
    result = hipErrorInvalidValue;
  }
  iree_slim_mutex_unlock(&user_object->mutex);
  return result;
}

static iree_status_t iree_hip_user_object_retain_graph_refs(void* object,
                                                            uint64_t count) {
  hipUserObject_t user_object = (hipUserObject_t)object;
  if (!user_object || count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&user_object->mutex);
  if (user_object->graph_association_count == UINT64_MAX ||
      (!user_object->destructor_called &&
       UINT64_MAX - user_object->ref_count < count)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  } else {
    ++user_object->graph_association_count;
    iree_atomic_fetch_add(&user_object->handle_reference_count, 1,
                          iree_memory_order_relaxed);
    if (!user_object->destructor_called) {
      user_object->ref_count += count;
    }
  }
  iree_slim_mutex_unlock(&user_object->mutex);
  return status;
}

static hipError_t iree_hip_user_object_add_graph_association(
    hipUserObject_t user_object) {
  if (!user_object) return hipErrorInvalidValue;
  hipError_t result = hipSuccess;
  iree_slim_mutex_lock(&user_object->mutex);
  if (user_object->graph_association_count == UINT64_MAX) {
    result = hipErrorInvalidValue;
  } else {
    ++user_object->graph_association_count;
    iree_atomic_fetch_add(&user_object->handle_reference_count, 1,
                          iree_memory_order_relaxed);
  }
  iree_slim_mutex_unlock(&user_object->mutex);
  return result;
}

static hipError_t iree_hip_user_object_release_refs_internal(
    hipUserObject_t user_object, uint64_t count,
    bool release_graph_association) {
  if (!user_object || count == 0) return hipErrorInvalidValue;

  bool should_destroy = false;
  void* ptr = NULL;
  hipHostFn_t destroy = NULL;
  hipError_t result = hipSuccess;
  iree_slim_mutex_lock(&user_object->mutex);
  if (release_graph_association && user_object->graph_association_count == 0) {
    result = hipErrorInvalidValue;
  } else if (!user_object->destructor_called &&
             count > user_object->ref_count) {
    result = hipErrorInvalidValue;
  } else {
    if (!user_object->destructor_called) {
      user_object->ref_count -= count;
      if (user_object->ref_count == 0) {
        user_object->destructor_called = true;
        should_destroy = true;
        ptr = user_object->ptr;
        destroy = user_object->destroy;
      }
    }
    if (release_graph_association) {
      --user_object->graph_association_count;
    }
  }
  iree_slim_mutex_unlock(&user_object->mutex);

  if (should_destroy) {
    // Remove the public handle before invoking user code so reentrant API calls
    // observe a completed object without dereferencing its control block.
    (void)iree_hip_user_object_registry_remove(user_object);
    destroy(ptr);
  }
  return result;
}

static hipError_t iree_hip_user_object_release_refs_checked(
    hipUserObject_t user_object, uint64_t count) {
  return iree_hip_user_object_release_refs_internal(
      user_object, count, /*release_graph_association=*/false);
}

static void iree_hip_user_object_release_graph_refs(void* object,
                                                    uint64_t count) {
  hipUserObject_t user_object = (hipUserObject_t)object;
  hipError_t result = iree_hip_user_object_release_refs_internal(
      user_object, count, /*release_graph_association=*/true);
  if (result == hipSuccess) {
    iree_hip_user_object_handle_release(user_object);
  }
}

static void iree_hip_user_object_release_refs(void* object, uint64_t count) {
  (void)iree_hip_user_object_release_refs_checked((hipUserObject_t)object,
                                                  count);
}

static hipError_t iree_hip_user_object_remove_graph_association(
    hipUserObject_t user_object, uint64_t count) {
  hipError_t result = iree_hip_user_object_release_refs_internal(
      user_object, count, /*release_graph_association=*/true);
  if (result == hipSuccess) {
    iree_hip_user_object_handle_release(user_object);
  }
  return result;
}

HIPAPI hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr,
                                      hipHostFn_t destroy,
                                      unsigned int initialRefcount,
                                      unsigned int flags) {
  if (!object_out || !destroy || initialRefcount == 0 ||
      flags != hipUserObjectNoDestructorSync) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipUserObject_t object = NULL;
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*object), (void**)&object);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }
  iree_slim_mutex_initialize(&object->mutex);
  iree_atomic_store(&object->handle_reference_count, 1,
                    iree_memory_order_relaxed);
  object->next_live_object = NULL;
  object->ref_count = initialRefcount;
  object->graph_association_count = 0;
  object->destructor_called = false;
  object->ptr = ptr;
  object->destroy = destroy;
  object->host_allocator = host_allocator;
  iree_hip_user_object_registry_insert(object);
  *object_out = object;
  return hipSuccess;
}

HIPAPI hipError_t hipUserObjectRelease(hipUserObject_t object,
                                       unsigned int count) {
  if (!object || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipUserObject_t retained_object = NULL;
  if (!iree_hip_user_object_registry_lookup_retain(object, &retained_object)) {
    return hipSuccess;
  }
  hipError_t result =
      iree_hip_user_object_release_refs_checked(retained_object, count);
  iree_hip_user_object_handle_release(retained_object);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipUserObjectRetain(hipUserObject_t object,
                                      unsigned int count) {
  if (!object || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipUserObject_t retained_object = NULL;
  if (!iree_hip_user_object_registry_lookup_retain(object, &retained_object)) {
    return hipSuccess;
  }
  hipError_t result =
      iree_hip_user_object_retain_refs_checked(retained_object, count);
  iree_hip_user_object_handle_release(retained_object);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipGraphRetainUserObject(hipGraph_t graph,
                                           hipUserObject_t object,
                                           unsigned int count,
                                           unsigned int flags) {
  if (!graph || !object || count == 0 ||
      (flags != 0 && flags != hipGraphUserObjectMove)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const bool move_refs = flags == hipGraphUserObjectMove;
  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  for (iree_hal_streaming_graph_user_object_ref_t* ref =
           stream_graph->user_object_refs;
       ref; ref = ref->next) {
    if (ref->object == object) {
      if (UINT64_MAX - ref->count < count) {
        HIP_RETURN_ERROR(hipErrorInvalidValue);
      }
      const uint64_t new_graph_ref_count = ref->count + count;
      hipError_t retain_result = hipSuccess;
      if (move_refs) {
        retain_result =
            iree_hip_user_object_check_ref_count(object, new_graph_ref_count);
      } else {
        retain_result = iree_hip_user_object_retain_refs_checked(object, count);
      }
      if (retain_result != hipSuccess) HIP_RETURN_ERROR(retain_result);
      ref->count += count;
      return hipSuccess;
    }
  }

  hipUserObject_t retained_object = NULL;
  if (!iree_hip_user_object_registry_lookup_retain(object, &retained_object)) {
    return hipSuccess;
  }
  object = retained_object;
  hipError_t retain_result =
      move_refs ? iree_hip_user_object_check_ref_count(object, count)
                : iree_hip_user_object_retain_refs_checked(object, count);
  if (retain_result != hipSuccess) {
    iree_hip_user_object_handle_release(retained_object);
    HIP_RETURN_ERROR(retain_result);
  }
  iree_hal_streaming_graph_user_object_ref_t* ref = NULL;
  iree_status_t status =
      iree_arena_allocate(&stream_graph->arena, sizeof(*ref), (void**)&ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    if (!move_refs) {
      iree_hip_user_object_release_refs(object, count);
    }
    iree_hip_user_object_handle_release(retained_object);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }
  hipError_t association_result =
      iree_hip_user_object_add_graph_association(object);
  if (association_result != hipSuccess) {
    if (!move_refs) {
      (void)iree_hip_user_object_release_refs_checked(object, count);
    }
    iree_hip_user_object_handle_release(retained_object);
    HIP_RETURN_ERROR(association_result);
  }
  ref->object = object;
  ref->count = count;
  ref->retain = iree_hip_user_object_retain_graph_refs;
  ref->release = iree_hip_user_object_release_graph_refs;
  ref->next = stream_graph->user_object_refs;
  stream_graph->user_object_refs = ref;
  iree_hip_user_object_handle_release(retained_object);
  return hipSuccess;
}

HIPAPI hipError_t hipGraphReleaseUserObject(hipGraph_t graph,
                                            hipUserObject_t object,
                                            unsigned int count) {
  if (!graph || !object || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  for (iree_hal_streaming_graph_user_object_ref_t** previous_next =
           &stream_graph->user_object_refs;
       *previous_next;) {
    iree_hal_streaming_graph_user_object_ref_t* ref = *previous_next;
    if (ref->object == object) {
      if (count > ref->count) {
        HIP_RETURN_ERROR(hipErrorInvalidValue);
      }
      const bool removes_association = count == ref->count;
      hipError_t release_result =
          removes_association
              ? iree_hip_user_object_remove_graph_association(object, count)
              : iree_hip_user_object_release_refs_checked(object, count);
      if (release_result != hipSuccess) {
        HIP_RETURN_ERROR(release_result);
      }
      ref->count -= count;
      if (ref->count == 0) {
        *previous_next = ref->next;
      }
      return hipSuccess;
    }
    previous_next = &ref->next;
  }
  HIP_RETURN_ERROR(hipErrorInvalidValue);
}

//===----------------------------------------------------------------------===//
// Graphs
//===----------------------------------------------------------------------===//

typedef struct iree_hip_live_graph_entry_t {
  // Public HIP graph handle that is currently owned by the caller.
  hipGraph_t graph;
  // Next entry in the process-global live graph registry.
  struct iree_hip_live_graph_entry_t* next;
} iree_hip_live_graph_entry_t;

typedef struct iree_hip_live_graph_exec_entry_t {
  // Public HIP executable graph handle that is currently owned by the caller.
  hipGraphExec_t graph_exec;
  // Next entry in the process-global live executable graph registry.
  struct iree_hip_live_graph_exec_entry_t* next;
} iree_hip_live_graph_exec_entry_t;

static iree_once_flag iree_hip_live_graph_mutex_once = IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hip_live_graph_mutex;
static iree_hip_live_graph_entry_t* iree_hip_live_graph_head = NULL;

static iree_once_flag iree_hip_live_graph_exec_mutex_once = IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hip_live_graph_exec_mutex;
static iree_hip_live_graph_exec_entry_t* iree_hip_live_graph_exec_head = NULL;

static void iree_hip_live_graph_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_live_graph_mutex);
}

static void iree_hip_live_graph_lock(void) {
  iree_call_once(&iree_hip_live_graph_mutex_once,
                 iree_hip_live_graph_mutex_initialize);
  iree_slim_mutex_lock(&iree_hip_live_graph_mutex);
}

static void iree_hip_live_graph_exec_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_live_graph_exec_mutex);
}

static void iree_hip_live_graph_exec_lock(void) {
  iree_call_once(&iree_hip_live_graph_exec_mutex_once,
                 iree_hip_live_graph_exec_mutex_initialize);
  iree_slim_mutex_lock(&iree_hip_live_graph_exec_mutex);
}

static bool iree_hip_live_graph_register(hipGraph_t graph) {
  if (!graph) return false;
  iree_hip_live_graph_entry_t* entry =
      (iree_hip_live_graph_entry_t*)malloc(sizeof(*entry));
  if (!entry) return false;
  entry->graph = graph;

  iree_hip_live_graph_lock();
  for (iree_hip_live_graph_entry_t* current = iree_hip_live_graph_head; current;
       current = current->next) {
    if (current->graph == graph) {
      iree_slim_mutex_unlock(&iree_hip_live_graph_mutex);
      free(entry);
      return true;
    }
  }
  entry->next = iree_hip_live_graph_head;
  iree_hip_live_graph_head = entry;
  iree_slim_mutex_unlock(&iree_hip_live_graph_mutex);
  return true;
}

static bool iree_hip_live_graph_unregister(hipGraph_t graph) {
  iree_hip_live_graph_lock();
  iree_hip_live_graph_entry_t** current = &iree_hip_live_graph_head;
  while (*current) {
    if ((*current)->graph == graph) {
      iree_hip_live_graph_entry_t* entry = *current;
      *current = entry->next;
      iree_slim_mutex_unlock(&iree_hip_live_graph_mutex);
      free(entry);
      return true;
    }
    current = &(*current)->next;
  }
  iree_slim_mutex_unlock(&iree_hip_live_graph_mutex);
  return false;
}

static bool iree_hip_graph_handle_is_live(hipGraph_t graph) {
  if (!graph) return false;
  iree_hip_live_graph_lock();
  for (iree_hip_live_graph_entry_t* current = iree_hip_live_graph_head; current;
       current = current->next) {
    if (current->graph == graph) {
      iree_slim_mutex_unlock(&iree_hip_live_graph_mutex);
      return true;
    }
  }
  iree_slim_mutex_unlock(&iree_hip_live_graph_mutex);
  return false;
}

static bool iree_hip_live_graph_exec_register(hipGraphExec_t graph_exec) {
  if (!graph_exec) return false;
  iree_hip_live_graph_exec_entry_t* entry =
      (iree_hip_live_graph_exec_entry_t*)malloc(sizeof(*entry));
  if (!entry) return false;
  entry->graph_exec = graph_exec;

  iree_hip_live_graph_exec_lock();
  for (iree_hip_live_graph_exec_entry_t* current =
           iree_hip_live_graph_exec_head;
       current; current = current->next) {
    if (current->graph_exec == graph_exec) {
      iree_slim_mutex_unlock(&iree_hip_live_graph_exec_mutex);
      free(entry);
      return true;
    }
  }
  iree_hal_streaming_graph_exec_retain(
      (iree_hal_streaming_graph_exec_t*)graph_exec);
  entry->next = iree_hip_live_graph_exec_head;
  iree_hip_live_graph_exec_head = entry;
  iree_slim_mutex_unlock(&iree_hip_live_graph_exec_mutex);
  return true;
}

static bool iree_hip_live_graph_exec_unregister(hipGraphExec_t graph_exec) {
  iree_hip_live_graph_exec_lock();
  iree_hip_live_graph_exec_entry_t** current = &iree_hip_live_graph_exec_head;
  while (*current) {
    if ((*current)->graph_exec == graph_exec) {
      iree_hip_live_graph_exec_entry_t* entry = *current;
      *current = entry->next;
      iree_slim_mutex_unlock(&iree_hip_live_graph_exec_mutex);
      iree_hal_streaming_graph_exec_release(
          (iree_hal_streaming_graph_exec_t*)entry->graph_exec);
      free(entry);
      return true;
    }
    current = &(*current)->next;
  }
  iree_slim_mutex_unlock(&iree_hip_live_graph_exec_mutex);
  return false;
}

// Alternative creation:
// - Stream capture: Record operations to build graph.
// - Graph cloning: Copy existing graph structure.
//
// Limitations:
// - No cycles allowed (must be DAG).
// - Some operations cannot be captured.
// - Device-specific node limits.
//
// Multi-GPU: Graphs can contain operations for multiple devices
// but require careful dependency management.
//
// See also: hipGraphDestroy, hipGraphAddKernelNode,
//           hipGraphInstantiate, hipStreamBeginCapture.
HIPAPI hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraph || flags != 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraph = NULL;

  // Get current context.
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = iree_hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Create graph.
  iree_hal_streaming_graph_t* graph = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_create(context,
                                      iree_hip_graph_flags_to_internal(flags),
                                      context->host_allocator, &graph),
      hipErrorOutOfMemory);

  if (!iree_hip_live_graph_register((hipGraph_t)graph)) {
    iree_hal_streaming_graph_release(graph);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }
  *pGraph = (hipGraph_t)graph;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Destroys a task graph.
//
// Parameters:
//  - graph: [IN] Graph handle to destroy.
//
// Returns:
//  - hipSuccess: Graph destroyed successfully.
//  - hipErrorInvalidValue: graph is NULL or invalid.
//
// Synchronization: This operation is synchronous.
//
// Destruction behavior:
// - Releases all resources associated with the graph.
// - All nodes in the graph are destroyed.
// - Graph handle becomes invalid after destruction.
// - Does not affect instantiated executables.
//
// Resource management:
// - Graph can be destroyed after instantiation.
// - Executable graphs remain valid after source destruction.
// - Must destroy both graph and executables to free all resources.
//
// Warning: Using a destroyed graph results in undefined behavior.
//
// See also: hipGraphCreate, hipGraphExecDestroy,
//           hipGraphInstantiate.
HIPAPI hipError_t hipGraphDestroy(hipGraph_t graph) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_live_graph_unregister(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_graph_release(stream_graph);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Instantiates a graph to create an executable graph.
//
// Parameters:
//  - pGraphExec: [OUT] Pointer to receive executable graph handle.
//  - graph: [IN] Source graph to instantiate.
//  - pErrorNode: [OUT] Optional pointer to receive error node (can be NULL).
//  - pLogBuffer: [OUT] Optional buffer for error messages (can be NULL).
//  - bufferSize: [IN] Size of pLogBuffer in bytes.
//
// Returns:
//  - hipSuccess: Graph instantiated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//  - hipErrorGraphCyclesDetected: Graph contains cycles.
//  - hipErrorInvalidDeviceFunction: Invalid kernel in graph.
//
// Synchronization: This operation is synchronous.
//
// Instantiation process:
// - Validates graph structure (DAG, no cycles).
// - Allocates resources for execution.
// - Optimizes node scheduling.
// - Creates device-specific command buffers.
// - Returns executable that can be launched.
//
// Error reporting:
// - pErrorNode receives first problematic node if provided.
// - pLogBuffer receives detailed error message if provided.
// - Both can be NULL if error details not needed.
//
// Performance optimization:
// - Instantiation is expensive; do once, launch many times.
// - Enables whole-graph optimizations.
// - May merge adjacent compatible operations.
// - Optimizes memory allocation and reuse.
//
// Resource lifetime:
// - Executable is independent of source graph.
// - Source graph can be destroyed after instantiation.
// - Must destroy executable with hipGraphExecDestroy.
//
// Multi-GPU: Instantiation binds operations to specific devices
// based on current context and node specifications.
//
// See also: hipGraphInstantiateWithFlags, hipGraphLaunch,
//           hipGraphExecDestroy, hipGraphExecUpdate.
HIPAPI hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec,
                                      hipGraph_t graph,
                                      hipGraphNode_t* pErrorNode,
                                      char* pLogBuffer, size_t bufferSize) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraphExec || !graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Always NUL terminate the log buffer and clear error node in case we fail
  // early.
  *pGraphExec = NULL;
  if (pErrorNode) *pErrorNode = NULL;
  if (pLogBuffer && bufferSize > 0) {
    pLogBuffer[0] = 0;
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_graph_exec_t* exec = NULL;
  iree_status_t status = iree_hal_streaming_graph_instantiate(
      stream_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec);
  if (!iree_status_is_ok(status)) {
    hipError_t result = iree_status_to_hip_result(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  if (!iree_hip_live_graph_exec_register((hipGraphExec_t)exec)) {
    iree_hal_streaming_graph_exec_release(exec);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }
  *pGraphExec = (hipGraphExec_t)exec;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Instantiates a graph with flags to create an executable graph.
//
// Parameters:
//  - pGraphExec: [OUT] Pointer to receive executable graph handle.
//  - graph: [IN] Source graph to instantiate.
//  - flags: [IN] Instantiation flags for optimization hints.
//
// Returns:
//  - hipSuccess: Graph instantiated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//  - hipErrorGraphCyclesDetected: Graph contains cycles.
//
// Synchronization: This operation is synchronous.
//
// Flag options:
// - hipGraphInstantiateFlagAutoFreeOnLaunch: Auto-free after launch.
// - hipGraphInstantiateFlagUpload: Upload to device immediately.
// - hipGraphInstantiateFlagDeviceLaunch: Enable device-side launch.
// - hipGraphInstantiateFlagUseNodePriority: Honor node priorities.
//
// Extended behavior:
// - Same as hipGraphInstantiate but with optimization hints.
// - Flags control resource management and scheduling.
// - May affect performance and memory usage.
//
// Performance flags:
// - AutoFree reduces memory pressure for one-shot graphs.
// - Upload flag can reduce first launch latency.
// - DeviceLaunch enables GPU-driven execution.
//
// Multi-GPU: Flags may affect multi-device scheduling
// and resource allocation strategies.
//
// See also: hipGraphInstantiate, hipGraphLaunch,
//           hipGraphExecDestroy.
HIPAPI hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec,
                                               hipGraph_t graph,
                                               unsigned long long flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraphExec || !graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraphExec = NULL;
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_instantiate_flags_are_supported(flags)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_graph_exec_t* exec = NULL;
  iree_status_t status = iree_hal_streaming_graph_instantiate(
      stream_graph, iree_hip_graph_instantiate_flags_to_internal(flags), &exec);
  if (!iree_status_is_ok(status)) {
    hipError_t result = iree_status_to_hip_result(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  if (!iree_hip_live_graph_exec_register((hipGraphExec_t)exec)) {
    iree_hal_streaming_graph_exec_release(exec);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }
  *pGraphExec = (hipGraphExec_t)exec;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

static hipError_t iree_hip_resolve_graph_exec(
    hipGraphExec_t graphExec, iree_hal_streaming_graph_exec_t** out_exec) {
  if (out_exec) *out_exec = NULL;
  if (!graphExec) {
    return hipErrorInvalidValue;
  }

  iree_hip_live_graph_exec_lock();
  for (iree_hip_live_graph_exec_entry_t* current =
           iree_hip_live_graph_exec_head;
       current; current = current->next) {
    if (current->graph_exec != graphExec) continue;
    iree_hal_streaming_graph_exec_t* exec =
        (iree_hal_streaming_graph_exec_t*)graphExec;
    const bool is_live = iree_hal_streaming_graph_exec_try_retain_live(exec);
    iree_slim_mutex_unlock(&iree_hip_live_graph_exec_mutex);
    if (!is_live) return hipErrorInvalidValue;
    if (out_exec) *out_exec = exec;
    return hipSuccess;
  }
  iree_slim_mutex_unlock(&iree_hip_live_graph_exec_mutex);
  return hipErrorInvalidValue;
}

// Destroys an executable graph.
//
// Parameters:
//  - graphExec: [IN] Executable graph handle to destroy.
//
// Returns:
//  - hipSuccess: Executable destroyed successfully.
//  - hipErrorInvalidValue: graphExec is NULL or invalid.
//
// Synchronization: This operation is synchronous.
//
// Destruction behavior:
// - Waits for any pending executions to complete.
// - Releases all resources associated with executable.
// - Handle becomes invalid after destruction.
// - Does not affect source graph used for instantiation.
//
// Resource management:
// - Must destroy to free device resources.
// - Independent of source graph lifetime.
// - Multiple executables can exist from same graph.
//
// Warning: Using a destroyed executable results in undefined behavior.
// Ensure no launches are pending before destruction.
//
// See also: hipGraphInstantiate, hipGraphDestroy, hipGraphLaunch.
HIPAPI hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_status_t status = iree_hal_streaming_graph_exec_destroy_handle(exec);
  if (iree_status_is_ok(status)) {
    (void)iree_hip_live_graph_exec_unregister(graphExec);
  }
  iree_hal_streaming_graph_exec_release(exec);
  if (!iree_status_is_ok(status)) {
    result = iree_status_to_fixed_hip_result(status, hipErrorInvalidValue);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Launches an executable graph in a stream.
//
// Parameters:
//  - graphExec: [IN] Executable graph to launch.
//  - stream: [IN] Stream for asynchronous execution.
//
// Returns:
//  - hipSuccess: Graph launched successfully.
//  - hipErrorInvalidValue: Invalid executable or stream.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: Launch failed on device.
//
// Synchronization: This operation is asynchronous.
//
// Launch behavior:
// - Enqueues entire graph as single operation.
// - Graph executes after prior stream operations.
// - Subsequent operations wait for graph completion.
// - Internal node dependencies are preserved.
//
// Performance benefits:
// - Single launch for complex workload.
// - Reduced CPU-GPU communication overhead.
// - Optimized scheduling and resource usage.
// - Better than individual kernel launches.
//
// Execution model:
// - Nodes execute based on dependencies.
// - Parallel nodes may run concurrently.
// - Graph completion when all nodes finish.
// - Use hipStreamSynchronize() to wait.
//
// Reusability:
// - Same executable can be launched multiple times.
// - Each launch is independent execution.
// - Resources are reused across launches.
// - Efficient for repeated workloads.
//
// Multi-GPU:
// - Graph may contain operations for multiple devices.
// - Cross-device dependencies handled automatically.
// - Stream determines primary execution context.
//
// Warning: Ensure input/output buffers are valid for each launch.
// Graph captures buffer addresses, not contents.
//
// See also: hipGraphInstantiate, hipGraphExecUpdate,
//           hipStreamSynchronize, hipGraphExecDestroy.
HIPAPI hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t init_result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  init_result = iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    iree_hal_streaming_graph_exec_release(exec);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t dependency_result = iree_hip_order_legacy_stream_dependencies(
      stream_obj->context, stream_obj);
  if (dependency_result != hipSuccess) {
    iree_hal_streaming_stream_release(stream_obj);
    iree_hal_streaming_graph_exec_release(exec);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(dependency_result);
  }

  iree_status_t launch_status =
      iree_hal_streaming_graph_exec_launch(exec, stream_obj);
  iree_hal_streaming_stream_release(stream_obj);
  iree_hal_streaming_graph_exec_release(exec);
  if (!iree_status_is_ok(launch_status)) {
    hipError_t result =
        iree_status_to_fixed_hip_result(launch_status, hipErrorInvalidValue);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Updates an executable graph with a modified source graph.
//
// Parameters:
//  - hGraphExec: [IN/OUT] Executable graph to update.
//  - hGraph: [IN] Modified source graph with updates.
//  - hErrorNode_out: [OUT] Pointer for the first incompatible node.
//  - updateResult_out: [OUT] Detailed update compatibility result.
//
// Returns:
//  - hipSuccess: Update successful.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorGraphExecUpdateFailure: Update not possible.
//
// Synchronization: This operation is synchronous.
//
// Update behavior:
// - Attempts to update executable without full re-instantiation.
// - Preserves optimizations when possible.
// - Faster than destroy + re-instantiate.
// - May fail if topology changed significantly.
//
// Supported updates:
// - Kernel parameters and launch configuration.
// - Memory copy parameters (source/dest/size).
// - Host callback function pointers.
// - Node enable/disable state.
//
// Unsupported updates:
// - Adding or removing nodes.
// - Changing node types.
// - Modifying graph topology/dependencies.
// - Changing device assignments.
//
// Update strategy:
// - Try update first for performance.
// - Fall back to re-instantiation if update fails.
// - Check hErrorNode_out for failure location.
//
// Performance optimization:
// - Update is much faster than re-instantiation.
// - Preserves device-side optimizations.
// - Ideal for parameter-only changes.
//
// Multi-GPU: Updates must preserve device assignments.
// Cannot move operations between devices.
//
// Warning: Updated executable must not be executing during update.
// Ensure stream synchronization before updating.
//
// See also: hipGraphInstantiate, hipGraphLaunch,
//           hipGraphExecDestroy.
static hipGraphExecUpdateResult iree_hip_graph_exec_update_result(
    iree_hal_streaming_graph_exec_update_result_t result) {
  switch (result) {
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS:
      return hipGraphExecUpdateSuccess;
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED:
      return hipGraphExecUpdateErrorTopologyChanged;
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NODE_TYPE_CHANGED:
      return hipGraphExecUpdateErrorNodeTypeChanged;
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_FUNCTION_CHANGED:
      return hipGraphExecUpdateErrorFunctionChanged;
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED:
      return hipGraphExecUpdateErrorParametersChanged;
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NOT_SUPPORTED:
      return hipGraphExecUpdateErrorNotSupported;
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_UNSUPPORTED_FUNCTION_CHANGE:
      return hipGraphExecUpdateErrorUnsupportedFunctionChange;
    case IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR:
    default:
      return hipGraphExecUpdateError;
  }
}

HIPAPI hipError_t
hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                   hipGraphNode_t* hErrorNode_out,
                   hipGraphExecUpdateResult* updateResult_out) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hGraphExec || !hGraph || !hErrorNode_out || !updateResult_out) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t init_result = iree_hip_resolve_graph_exec(hGraphExec, &exec);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;

  iree_hal_streaming_graph_node_t* error_node = NULL;
  iree_hal_streaming_graph_exec_update_result_t update_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  iree_status_t status = iree_hal_streaming_graph_exec_update(
      exec, graph, &error_node, &update_result);
  iree_hal_streaming_graph_exec_release(exec);
  *hErrorNode_out = (hipGraphNode_t)error_node;
  *updateResult_out = iree_hip_graph_exec_update_result(update_result);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorGraphExecUpdateFailure);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

static hipError_t iree_hip_resolve_function_symbol(
    iree_hal_streaming_context_t* context, const void* function_address,
    iree_hal_streaming_symbol_t** out_symbol) {
  *out_symbol = NULL;
  if (!context || !function_address) return hipErrorInvalidValue;

  if (iree_hal_streaming_symbol_has_tag(function_address)) {
    *out_symbol = iree_hal_streaming_symbol_untag(function_address);
  } else {
    iree_status_t lookup_status = iree_hal_streaming_context_symbol_map_lookup(
        &context->symbol_map, (void*)function_address, out_symbol);
    if (!iree_status_is_ok(lookup_status)) {
      iree_status_ignore(lookup_status);
      return hipErrorInvalidDeviceFunction;
    }
    if (*out_symbol == (iree_hal_streaming_symbol_t*)function_address) {
      *out_symbol = NULL;
      return hipErrorInvalidDeviceFunction;
    }
  }

  if (!*out_symbol ||
      (*out_symbol)->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    *out_symbol = NULL;
    return hipErrorInvalidDeviceFunction;
  }
  return hipSuccess;
}

static bool iree_hip_symbol_accepts_empty_kernel_params(
    const iree_hal_streaming_symbol_t* symbol) {
  return symbol && symbol->parameters.binding_count == 0 &&
         symbol->parameters.copy_count == 0 &&
         symbol->parameters.constant_bytes == 0 &&
         symbol->parameters.direct_arg_bytes == 0;
}

// Adds a kernel execution node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pNodeParams: [IN] Kernel parameters (hipKernelNodeParams structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: This operation modifies the graph structure.
//
// Node parameters (hipKernelNodeParams):
// - func: Kernel function to execute.
// - gridDim: Grid dimensions (blocks).
// - blockDim: Block dimensions (threads).
// - sharedMemBytes: Dynamic shared memory size.
// - kernelParams: Array of kernel arguments.
// - extra: Extra launch parameters.
//
// Dependency behavior:
// - Node waits for all dependencies to complete.
// - Creates edges from dependency nodes to this node.
// - NULL dependencies means no predecessors.
// - Forms directed acyclic graph (DAG).
//
// Graph construction:
// - Nodes can be added in any order.
// - Dependencies must already exist in graph.
// - Multiple nodes can have same dependencies.
// - Node becomes dependency for subsequent nodes.
//
// Kernel execution:
// - Parameters captured at node creation.
// - Kernel launches when dependencies satisfied.
// - Uses captured parameters for each graph launch.
// - Parameters can be updated via node update APIs.
//
// Multi-GPU:
// - Kernel executes on device associated with function.
// - Cross-device dependencies handled automatically.
//
// Argument values are captured when the node is created or updated. Device
// memory referenced by pointer-valued arguments must remain valid until graph
// executions using the node complete.
//
// See also: hipGraphAddMemcpyNode, hipGraphAddMemsetNode,
//           hipGraphNodeGetType, hipGraphKernelNodeSetParams.
HIPAPI hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pNodeParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pNodeParams ||
      (numDependencies > 0 && !pDependencies)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipKernelNodeParams* params = (const hipKernelNodeParams*)pNodeParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // Resolve the kernel symbol first so the extra-buffer handling below can key
  // off its reflected parameter metadata.
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t symbol_result = iree_hip_resolve_function_symbol(
      stream_graph->context, params->func, &symbol);
  if (symbol_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(symbol_result);
  }
  if (!params->kernelParams && !params->extra &&
      !iree_hip_symbol_accepts_empty_kernel_params(symbol)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (params->kernelParams && params->extra) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t launch_config_result = iree_hip_validate_launch_configuration(
      stream_graph->context ? stream_graph->context->device_entry : NULL,
      symbol, params->gridDim.x, params->gridDim.y, params->gridDim.z,
      params->blockDim.x, params->blockDim.y, params->blockDim.z,
      params->sharedMemBytes);
  if (launch_config_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(launch_config_result);
  }

  // Create dispatch params from kernel node params.
  // Extract params pointer from HIP's parameter format.
  void* params_ptr = NULL;
  size_t params_size = 0;
  iree_hal_streaming_dispatch_flags_t dispatch_flags =
      IREE_HAL_STREAMING_DISPATCH_FLAG_NONE;
  if (params->extra) {
    hipError_t parse_result =
        iree_hip_parse_launch_extra(params->extra, &params_ptr, &params_size);
    if (parse_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(parse_result);
    }
    // The extra buffer is already laid out in the kernel's native kernarg ABI.
    // Preserve it byte-for-byte; graph capture stores a private copy.
    dispatch_flags |= IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED;
  } else if (params->kernelParams) {
    // kernelParams is an array of pointers to the actual parameters.
    params_ptr = params->kernelParams;
    dispatch_flags |= IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY;
  }

  iree_hal_streaming_dispatch_params_t dispatch_params = {
      .grid_dim = {params->gridDim.x, params->gridDim.y, params->gridDim.z},
      .block_dim = {params->blockDim.x, params->blockDim.y, params->blockDim.z},
      .shared_memory_bytes = params->sharedMemBytes,
      .buffer = params_ptr,
      .buffer_size = params_size,
      .flags = dispatch_flags,
  };

  // Add kernel node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_kernel_node(
          stream_graph, deps, numDependencies, symbol, &dispatch_params, &node),
      hipErrorInvalidValue);
  node->attrs.kernel.hip_function = params->func;
  node->attrs.kernel.hip_kernel_params = params->kernelParams;
  node->attrs.kernel.hip_extra = params->extra;

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

static hipError_t iree_hip_graph_validate_memcpy3d_params(
    iree_hal_streaming_context_t* context, const hipMemcpy3DParms* params,
    hipMemcpyKind* out_kind, void** out_dst, const void** out_src,
    iree_host_size_t* out_size, iree_host_size_t* out_dst_span,
    iree_host_size_t* out_src_span);

static hipMemcpyKind iree_hip_graph_metadata_memcpy_kind(hipMemcpyKind kind) {
  return kind == hipMemcpyDeviceToDeviceNoCU ? hipMemcpyDeviceToDevice : kind;
}

static void iree_hip_graph_store_memcpy_query_metadata(
    iree_hal_streaming_graph_node_t* node, const hipMemcpy3DParms* params) {
  if (!node || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY ||
      !params) {
    return;
  }
  node->attrs.memcpy.hip_dst = params->dstPtr.ptr;
  node->attrs.memcpy.hip_src = params->srcPtr.ptr;
  node->attrs.memcpy.hip_dst_array = params->dstArray;
  node->attrs.memcpy.hip_src_array = params->srcArray;
  node->attrs.memcpy.hip_dst_position_x = params->dstPos.x;
  node->attrs.memcpy.hip_dst_position_y = params->dstPos.y;
  node->attrs.memcpy.hip_dst_position_z = params->dstPos.z;
  node->attrs.memcpy.hip_src_position_x = params->srcPos.x;
  node->attrs.memcpy.hip_src_position_y = params->srcPos.y;
  node->attrs.memcpy.hip_src_position_z = params->srcPos.z;
  node->attrs.memcpy.hip_dst_pitch = params->dstPtr.pitch;
  node->attrs.memcpy.hip_src_pitch = params->srcPtr.pitch;
  node->attrs.memcpy.hip_dst_xsize = params->dstPtr.xsize;
  node->attrs.memcpy.hip_src_xsize = params->srcPtr.xsize;
  node->attrs.memcpy.hip_dst_ysize = params->dstPtr.ysize;
  node->attrs.memcpy.hip_src_ysize = params->srcPtr.ysize;
  node->attrs.memcpy.hip_extent_width = params->extent.width;
  node->attrs.memcpy.hip_extent_height = params->extent.height;
  node->attrs.memcpy.hip_extent_depth = params->extent.depth;
  node->attrs.memcpy.hip_kind =
      (int)iree_hip_graph_metadata_memcpy_kind(params->kind);
}

static void iree_hip_graph_store_1d_memcpy_metadata(
    iree_hal_streaming_graph_node_t* node, void* dst, const void* src,
    size_t count, hipMemcpyKind kind) {
  if (!node || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) return;
  node->attrs.memcpy.execution_dst_pitch = count;
  node->attrs.memcpy.execution_src_pitch = count;
  node->attrs.memcpy.execution_dst_ysize = 1;
  node->attrs.memcpy.execution_src_ysize = 1;
  node->attrs.memcpy.execution_extent_width = count;
  node->attrs.memcpy.execution_extent_height = 1;
  node->attrs.memcpy.execution_extent_depth = 1;
  node->attrs.memcpy.hip_dst = dst;
  node->attrs.memcpy.hip_src = src;
  node->attrs.memcpy.hip_dst_array = NULL;
  node->attrs.memcpy.hip_src_array = NULL;
  node->attrs.memcpy.hip_dst_position_x = 0;
  node->attrs.memcpy.hip_dst_position_y = 0;
  node->attrs.memcpy.hip_dst_position_z = 0;
  node->attrs.memcpy.hip_src_position_x = 0;
  node->attrs.memcpy.hip_src_position_y = 0;
  node->attrs.memcpy.hip_src_position_z = 0;
  node->attrs.memcpy.hip_dst_pitch = count;
  node->attrs.memcpy.hip_src_pitch = count;
  node->attrs.memcpy.hip_dst_xsize = count;
  node->attrs.memcpy.hip_src_xsize = count;
  node->attrs.memcpy.hip_dst_ysize = 1;
  node->attrs.memcpy.hip_src_ysize = 1;
  node->attrs.memcpy.hip_extent_width = count;
  node->attrs.memcpy.hip_extent_height = 1;
  node->attrs.memcpy.hip_extent_depth = 1;
  node->attrs.memcpy.hip_kind = (int)iree_hip_graph_metadata_memcpy_kind(kind);
}

static void iree_hip_graph_store_memcpy3d_metadata(
    iree_hal_streaming_graph_node_t* node, const hipMemcpy3DParms* params) {
  if (!node || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) return;
  node->attrs.memcpy.execution_dst_pitch = params->dstPtr.pitch;
  node->attrs.memcpy.execution_src_pitch = params->srcPtr.pitch;
  node->attrs.memcpy.execution_dst_ysize = params->dstPtr.ysize;
  node->attrs.memcpy.execution_src_ysize = params->srcPtr.ysize;
  node->attrs.memcpy.execution_extent_width = params->extent.width;
  node->attrs.memcpy.execution_extent_height = params->extent.height;
  node->attrs.memcpy.execution_extent_depth = params->extent.depth;
  iree_hip_graph_store_memcpy_query_metadata(node, params);
}

static void iree_hip_graph_store_memcpy_execution_geometry(
    iree_hal_streaming_graph_node_t* node, iree_device_size_t dst_pitch,
    iree_device_size_t src_pitch, iree_device_size_t dst_ysize,
    iree_device_size_t src_ysize, iree_device_size_t width,
    iree_device_size_t height, iree_device_size_t depth) {
  if (!node || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) return;
  node->attrs.memcpy.execution_dst_pitch = dst_pitch;
  node->attrs.memcpy.execution_src_pitch = src_pitch;
  node->attrs.memcpy.execution_dst_ysize = dst_ysize;
  node->attrs.memcpy.execution_src_ysize = src_ysize;
  node->attrs.memcpy.execution_extent_width = width;
  node->attrs.memcpy.execution_extent_height = height;
  node->attrs.memcpy.execution_extent_depth = depth;
}

typedef struct iree_hip_graph_memcpy_callback_data_t {
  // Destination base pointer used by the host callback.
  void* dst;
  // Source base pointer used by the host callback.
  const void* src;
  // Byte count used by 1D copies.
  size_t count;
  // Row width in bytes used by 2D/3D copies.
  size_t width;
  // Row count used by 2D/3D copies.
  size_t height;
  // Slice count used by 3D copies.
  size_t depth;
  // Destination row pitch in bytes.
  size_t dst_pitch;
  // Source row pitch in bytes.
  size_t src_pitch;
  // Destination slice pitch in bytes.
  size_t dst_slice_pitch;
  // Source slice pitch in bytes.
  size_t src_slice_pitch;
  // Destination context for staged peer copies, or NULL for host memory copies.
  iree_hal_streaming_context_t* dst_context;
  // Source context for staged peer copies, or NULL for host memory copies.
  iree_hal_streaming_context_t* src_context;
  // HIP-visible memcpy parameters returned by graph query APIs.
  hipMemcpy3DParms hip_params;
} iree_hip_graph_memcpy_callback_data_t;

static void iree_hip_streaming_context_retain_refs_unchecked(void* object,
                                                             uint64_t count) {
  iree_hal_streaming_context_t* context = (iree_hal_streaming_context_t*)object;
  for (uint64_t i = 0; i < count; ++i) {
    iree_hal_streaming_context_retain(context);
  }
}

static iree_status_t iree_hip_streaming_context_retain_refs(void* object,
                                                            uint64_t count) {
  iree_hip_streaming_context_retain_refs_unchecked(object, count);
  return iree_ok_status();
}

static void iree_hip_streaming_context_release_refs(void* object,
                                                    uint64_t count) {
  iree_hal_streaming_context_t* context = (iree_hal_streaming_context_t*)object;
  for (uint64_t i = 0; i < count; ++i) {
    iree_hal_streaming_context_release(context);
  }
}

static hipError_t iree_hip_graph_retain_remote_context(
    iree_hal_streaming_graph_t* stream_graph,
    iree_hal_streaming_context_t* context) {
  if (!stream_graph || !context || context == stream_graph->context) {
    return hipSuccess;
  }

  for (iree_hal_streaming_graph_user_object_ref_t* ref =
           stream_graph->user_object_refs;
       ref; ref = ref->next) {
    if (ref->object == context &&
        ref->retain == iree_hip_streaming_context_retain_refs &&
        ref->release == iree_hip_streaming_context_release_refs) {
      if (ref->count == UINT64_MAX) return hipErrorInvalidValue;
      iree_hal_streaming_context_retain(context);
      ++ref->count;
      return hipSuccess;
    }
  }

  iree_hal_streaming_graph_user_object_ref_t* ref = NULL;
  iree_status_t status =
      iree_arena_allocate(&stream_graph->arena, sizeof(*ref), (void**)&ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  ref->object = context;
  ref->count = 1;
  ref->retain = iree_hip_streaming_context_retain_refs;
  ref->release = iree_hip_streaming_context_release_refs;
  ref->next = stream_graph->user_object_refs;
  stream_graph->user_object_refs = ref;
  iree_hal_streaming_context_retain(context);
  return hipSuccess;
}

static void iree_hip_graph_release_remote_context(
    iree_hal_streaming_graph_t* stream_graph,
    iree_hal_streaming_context_t* context) {
  if (!stream_graph || !context || context == stream_graph->context) return;

  for (iree_hal_streaming_graph_user_object_ref_t** previous_next =
           &stream_graph->user_object_refs;
       *previous_next;) {
    iree_hal_streaming_graph_user_object_ref_t* ref = *previous_next;
    if (ref->object == context &&
        ref->retain == iree_hip_streaming_context_retain_refs &&
        ref->release == iree_hip_streaming_context_release_refs) {
      if (ref->count > 0) {
        --ref->count;
        iree_hal_streaming_context_release(context);
        if (ref->count == 0) {
          *previous_next = ref->next;
        }
      }
      return;
    }
    previous_next = &ref->next;
  }
}

static void iree_hip_graph_release_memcpy_callback_context_refs(
    iree_hal_streaming_graph_t* stream_graph,
    const iree_hip_graph_memcpy_callback_data_t* callback_data) {
  if (!callback_data) return;
  iree_hip_graph_release_remote_context(stream_graph,
                                        callback_data->dst_context);
  iree_hip_graph_release_remote_context(stream_graph,
                                        callback_data->src_context);
}

static hipError_t iree_hip_graph_memcpy_rows_per_slice(size_t position_y,
                                                       size_t height,
                                                       size_t ysize,
                                                       iree_host_size_t* out);

static hipError_t iree_hip_graph_configure_memcpy_callback_data(
    iree_hip_graph_memcpy_callback_data_t* callback_data,
    const hipMemcpy3DParms* params, void* dst, const void* src,
    iree_host_size_t size, size_t dst_pitch, size_t src_pitch,
    size_t dst_rows_per_slice, size_t src_rows_per_slice);

static void iree_hip_graph_memcpy_callback(void* user_data);

static hipError_t iree_hip_graph_add_staged_peer_memcpy_node(
    hipGraphNode_t* pGraphNode, iree_hal_streaming_graph_t* stream_graph,
    iree_hal_streaming_graph_node_t** deps, size_t numDependencies,
    iree_hal_streaming_context_t* dst_context, void* dst,
    iree_hal_streaming_context_t* src_context, const void* src,
    const hipMemcpy3DParms* params, const hipMemcpy3DParms* execution_params,
    iree_host_size_t size, iree_host_size_t dst_span,
    iree_host_size_t src_span) {
  if (!dst_context || !src_context || !stream_graph) {
    return hipErrorInvalidValue;
  }

  iree_host_size_t src_rows_per_slice = 0;
  hipError_t result = iree_hip_graph_memcpy_rows_per_slice(
      execution_params->srcPos.y, execution_params->extent.height,
      execution_params->srcPtr.ysize, &src_rows_per_slice);
  if (result != hipSuccess) return result;
  iree_host_size_t dst_rows_per_slice = 0;
  result = iree_hip_graph_memcpy_rows_per_slice(
      execution_params->dstPos.y, execution_params->extent.height,
      execution_params->dstPtr.ysize, &dst_rows_per_slice);
  if (result != hipSuccess) return result;

  if (dst_context == stream_graph->context &&
      src_context == stream_graph->context) {
    return hipErrorInvalidValue;
  }

  iree_hal_streaming_buffer_t* staging = NULL;
  iree_status_t status = iree_hal_streaming_graph_allocate_host_staging(
      stream_graph, size, &staging);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }

  if (src_context == stream_graph->context) {
    iree_hal_streaming_buffer_ref_t staging_ref;
    status = iree_hal_streaming_memory_lookup_range(
        stream_graph->context, staging->device_ptr, size, &staging_ref);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorInvalidValue;
    }
    iree_hal_streaming_buffer_ref_t src_ref;
    status = iree_hal_streaming_memory_lookup_range(
        stream_graph->context, (iree_hal_streaming_deviceptr_t)src, src_span,
        &src_ref);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorInvalidValue;
    }

    iree_hal_streaming_graph_node_t* copy_node = NULL;
    status = iree_hal_streaming_graph_add_copy_buffer_node(
        stream_graph, deps, numDependencies, staging_ref, src_ref, size,
        &copy_node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorInvalidValue;
    }
    iree_hip_graph_store_memcpy3d_metadata(copy_node, params);
    iree_hip_graph_store_memcpy_execution_geometry(
        copy_node, execution_params->extent.width,
        execution_params->srcPtr.pitch, execution_params->extent.height,
        src_rows_per_slice, execution_params->extent.width,
        execution_params->extent.height, execution_params->extent.depth);

    iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
    status = iree_arena_allocate(&stream_graph->arena, sizeof(*callback_data),
                                 (void**)&callback_data);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorOutOfMemory;
    }
    result = iree_hip_graph_configure_memcpy_callback_data(
        callback_data, execution_params, dst, staging->host_ptr, size,
        execution_params->dstPtr.pitch, execution_params->extent.width,
        dst_rows_per_slice, execution_params->extent.height);
    if (result != hipSuccess) return result;
    callback_data->dst_context = dst_context;

    result = iree_hip_graph_retain_remote_context(stream_graph, dst_context);
    if (result != hipSuccess) return result;

    iree_hal_streaming_graph_node_t* copy_deps[] = {copy_node};
    iree_hal_streaming_graph_node_t* callback_node = NULL;
    status = iree_hal_streaming_graph_add_host_call_node(
        stream_graph, copy_deps, IREE_ARRAYSIZE(copy_deps),
        iree_hip_graph_memcpy_callback, callback_data, &callback_node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_hip_graph_release_remote_context(stream_graph, dst_context);
      return hipErrorInvalidValue;
    }
    callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    callback_node->attrs.host.user_data_size = sizeof(*callback_data);

    *pGraphNode = (hipGraphNode_t)copy_node;
    return hipSuccess;
  }

  if (dst_context == stream_graph->context) {
    iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
    status = iree_arena_allocate(&stream_graph->arena, sizeof(*callback_data),
                                 (void**)&callback_data);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorOutOfMemory;
    }
    result = iree_hip_graph_configure_memcpy_callback_data(
        callback_data, execution_params, staging->host_ptr, src, size,
        execution_params->extent.width, execution_params->srcPtr.pitch,
        execution_params->extent.height, src_rows_per_slice);
    if (result != hipSuccess) return result;
    callback_data->src_context = src_context;

    result = iree_hip_graph_retain_remote_context(stream_graph, src_context);
    if (result != hipSuccess) return result;

    iree_hal_streaming_graph_node_t* callback_node = NULL;
    status = iree_hal_streaming_graph_add_host_call_node(
        stream_graph, deps, numDependencies, iree_hip_graph_memcpy_callback,
        callback_data, &callback_node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_hip_graph_release_remote_context(stream_graph, src_context);
      return hipErrorInvalidValue;
    }
    callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    callback_node->attrs.host.user_data_size = sizeof(*callback_data);

    iree_hal_streaming_graph_node_t* copy_node = NULL;
    status = iree_hal_streaming_graph_add_copy_ptr_node_with_extra_dependency(
        stream_graph, deps, numDependencies, callback_node,
        (iree_hal_streaming_deviceptr_t)dst, staging->device_ptr, size,
        &copy_node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_hip_graph_release_remote_context(stream_graph, src_context);
      return hipErrorInvalidValue;
    }
    iree_hip_graph_store_memcpy3d_metadata(copy_node, params);
    iree_hip_graph_store_memcpy_execution_geometry(
        copy_node, execution_params->dstPtr.pitch,
        execution_params->extent.width, dst_rows_per_slice,
        execution_params->extent.height, execution_params->extent.width,
        execution_params->extent.height, execution_params->extent.depth);

    *pGraphNode = (hipGraphNode_t)copy_node;
    return hipSuccess;
  }

  iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
  status = iree_arena_allocate(&stream_graph->arena, sizeof(*callback_data),
                               (void**)&callback_data);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  result = iree_hip_graph_configure_memcpy_callback_data(
      callback_data, execution_params, dst, src, size,
      execution_params->dstPtr.pitch, execution_params->srcPtr.pitch,
      dst_rows_per_slice, src_rows_per_slice);
  if (result != hipSuccess) return result;
  callback_data->dst_context = dst_context;
  callback_data->src_context = src_context;
  callback_data->hip_params = *params;
  callback_data->hip_params.kind =
      iree_hip_graph_metadata_memcpy_kind(params->kind);

  result = iree_hip_graph_retain_remote_context(stream_graph, dst_context);
  if (result != hipSuccess) return result;
  result = iree_hip_graph_retain_remote_context(stream_graph, src_context);
  if (result != hipSuccess) {
    iree_hip_graph_release_remote_context(stream_graph, dst_context);
    return result;
  }

  iree_hal_streaming_graph_node_t* node = NULL;
  status = iree_hal_streaming_graph_add_host_call_node(
      stream_graph, deps, numDependencies, iree_hip_graph_memcpy_callback,
      callback_data, &node);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_hip_graph_release_remote_context(stream_graph, src_context);
    iree_hip_graph_release_remote_context(stream_graph, dst_context);
    return hipErrorInvalidValue;
  }
  node->attrs.host.user_data_size = sizeof(*callback_data);
  *pGraphNode = (hipGraphNode_t)node;
  return hipSuccess;
}

// Host callback function for memcpy operations.
static void iree_hip_graph_memcpy_callback(void* user_data) {
  iree_hip_graph_memcpy_callback_data_t* data =
      (iree_hip_graph_memcpy_callback_data_t*)user_data;
  const size_t width = data->width ? data->width : data->count;
  const size_t height = data->height ? data->height : 1;
  const size_t depth = data->depth ? data->depth : 1;
  const size_t dst_pitch = data->dst_pitch ? data->dst_pitch : width;
  const size_t src_pitch = data->src_pitch ? data->src_pitch : width;
  const size_t dst_slice_pitch =
      data->dst_slice_pitch ? data->dst_slice_pitch : dst_pitch * height;
  const size_t src_slice_pitch =
      data->src_slice_pitch ? data->src_slice_pitch : src_pitch * height;
  uint8_t* dst_base = (uint8_t*)data->dst;
  const uint8_t* src_base = (const uint8_t*)data->src;
  if (data->src_context && data->dst_context) {
    if (width == 0) return;
    uint8_t* staging = (uint8_t*)malloc(width);
    if (!staging) return;
    for (size_t z = 0; z < depth; ++z) {
      uint8_t* dst_slice = dst_base + z * dst_slice_pitch;
      const uint8_t* src_slice = src_base + z * src_slice_pitch;
      for (size_t y = 0; y < height; ++y) {
        const uint8_t* src_row = src_slice + y * src_pitch;
        uint8_t* dst_row = dst_slice + y * dst_pitch;
        iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
            data->src_context, staging, (iree_hal_streaming_deviceptr_t)src_row,
            width, NULL);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          free(staging);
          return;
        }
        status = iree_hal_streaming_memcpy_host_to_device(
            data->dst_context, (iree_hal_streaming_deviceptr_t)dst_row, staging,
            width, NULL);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          free(staging);
          return;
        }
      }
    }
    free(staging);
    return;
  }
  if (data->src_context) {
    for (size_t z = 0; z < depth; ++z) {
      const uint8_t* src_slice = src_base + z * src_slice_pitch;
      uint8_t* dst_slice = dst_base + z * dst_slice_pitch;
      for (size_t y = 0; y < height; ++y) {
        iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
            data->src_context, dst_slice + y * dst_pitch,
            (iree_hal_streaming_deviceptr_t)(src_slice + y * src_pitch), width,
            NULL);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          return;
        }
      }
    }
    return;
  }
  if (data->dst_context) {
    for (size_t z = 0; z < depth; ++z) {
      uint8_t* dst_slice = dst_base + z * dst_slice_pitch;
      const uint8_t* src_slice = src_base + z * src_slice_pitch;
      for (size_t y = 0; y < height; ++y) {
        iree_status_t status = iree_hal_streaming_memcpy_host_to_device(
            data->dst_context,
            (iree_hal_streaming_deviceptr_t)(dst_slice + y * dst_pitch),
            src_slice + y * src_pitch, width, NULL);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          return;
        }
      }
    }
    return;
  }
  for (size_t z = 0; z < depth; ++z) {
    uint8_t* dst_slice = dst_base + z * dst_slice_pitch;
    const uint8_t* src_slice = src_base + z * src_slice_pitch;
    for (size_t y = 0; y < height; ++y) {
      memcpy(dst_slice + y * dst_pitch, src_slice + y * src_pitch, width);
    }
  }
}

static bool iree_hip_graph_node_is_memcpy_host_call(
    const iree_hal_streaming_graph_node_t* node) {
  return node && node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL &&
         node->attrs.host.fn == iree_hip_graph_memcpy_callback &&
         node->attrs.host.user_data;
}

static bool iree_hip_graph_node_is_visible_memcpy_host_call(
    const iree_hal_streaming_graph_node_t* node) {
  return iree_hip_graph_node_is_memcpy_host_call(node) &&
         (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) == 0 &&
         node->attrs.host.user_data_size >=
             sizeof(iree_hip_graph_memcpy_callback_data_t);
}

static void iree_hip_graph_store_host_memcpy3d_metadata(
    iree_hal_streaming_graph_node_t* node, const hipMemcpy3DParms* params) {
  if (!iree_hip_graph_node_is_memcpy_host_call(node) || !params ||
      node->attrs.host.user_data_size <
          sizeof(iree_hip_graph_memcpy_callback_data_t)) {
    return;
  }
  iree_hip_graph_memcpy_callback_data_t* callback_data =
      (iree_hip_graph_memcpy_callback_data_t*)node->attrs.host.user_data;
  callback_data->hip_params = *params;
  callback_data->hip_params.kind =
      iree_hip_graph_metadata_memcpy_kind(params->kind);
}

// Adds a memory copy node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pCopyParams: [IN] Copy parameters (hipMemcpy3DParms structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: This operation modifies the graph structure.
//
// Copy parameters (hipMemcpy3DParms):
// - srcArray/srcPtr: Source memory (array or pointer).
// - srcPos: Source position for 3D copies.
// - dstArray/dstPtr: Destination memory.
// - dstPos: Destination position for 3D copies.
// - extent: Size of copy region (width, height, depth).
// - kind: Memory copy kind (D2D, H2D, D2H, etc.).
//
// Copy behavior:
// - Captures copy parameters at node creation.
// - Executes copy when dependencies satisfied.
// - Supports 1D, 2D, and 3D memory copies.
// - Handles all memory types (host/device/managed).
//
// Dependency management:
// - Waits for all dependency nodes to complete.
// - Copy occurs after dependencies in graph execution.
// - Can be dependency for subsequent nodes.
//
// Performance optimization:
// - Enables copy-compute overlap in graph.
// - May merge adjacent copies when possible.
// - Optimizes for memory bandwidth utilization.
//
// Multi-GPU:
// - Supports peer-to-peer copies between devices.
// - Cross-device copies handled transparently.
// - Routing optimized based on topology.
//
// Warning: Source and destination addresses captured at creation.
// Ensure memory remains valid for all graph launches.
//
// See also: hipGraphAddKernelNode, hipGraphAddMemsetNode,
//           hipMemcpy3DAsync, hipGraphMemcpyNodeSetParams.
HIPAPI hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pCopyParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pCopyParams ||
      (numDependencies > 0 && !pDependencies)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipMemcpy3DParms* params = (const hipMemcpy3DParms*)pCopyParams;
  hipMemcpy3DParms execution_params;
  hipError_t resolve_result =
      iree_hip_resolve_memcpy3d_array_params(params, &execution_params);
  if (resolve_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(resolve_result);
  }
  hipMemcpyKind kind = hipMemcpyDefault;
  void* dst = NULL;
  const void* src = NULL;
  iree_host_size_t size = 0;
  iree_host_size_t dst_span = 0;
  iree_host_size_t src_span = 0;
  hipError_t validate_result = iree_hip_graph_validate_memcpy3d_params(
      stream_graph->context, params, &kind, &dst, &src, &size, &dst_span,
      &src_span);
  if (validate_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(validate_result);
  }

  if (execution_params.extent.height == 1 &&
      execution_params.extent.depth == 1) {
    hipError_t result =
        hipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies,
                                numDependencies, dst, src, size, kind);
    if (result == hipSuccess) {
      iree_hal_streaming_graph_node_t* node =
          (iree_hal_streaming_graph_node_t*)*pGraphNode;
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
        iree_hip_graph_store_memcpy3d_metadata(node, params);
        iree_hip_graph_store_memcpy_execution_geometry(
            node, execution_params.dstPtr.pitch, execution_params.srcPtr.pitch,
            execution_params.dstPtr.ysize, execution_params.srcPtr.ysize,
            execution_params.extent.width, execution_params.extent.height,
            execution_params.extent.depth);
      } else {
        iree_hip_graph_store_host_memcpy3d_metadata(node, params);
      }
    }
    IREE_TRACE_ZONE_END(z0);
    return result;
  }

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  if (kind == hipMemcpyHostToHost) {
    iree_host_size_t src_rows_per_slice = execution_params.srcPtr.ysize;
    iree_host_size_t dst_rows_per_slice = execution_params.dstPtr.ysize;
    iree_host_size_t src_slice_pitch = 0;
    iree_host_size_t dst_slice_pitch = 0;
    if (IREE_UNLIKELY(
            (src_rows_per_slice == 0 &&
             !iree_host_size_checked_add(execution_params.srcPos.y,
                                         execution_params.extent.height,
                                         &src_rows_per_slice)) ||
            (dst_rows_per_slice == 0 &&
             !iree_host_size_checked_add(execution_params.dstPos.y,
                                         execution_params.extent.height,
                                         &dst_rows_per_slice)) ||
            !iree_host_size_checked_mul(execution_params.srcPtr.pitch,
                                        src_rows_per_slice, &src_slice_pitch) ||
            !iree_host_size_checked_mul(execution_params.dstPtr.pitch,
                                        dst_rows_per_slice,
                                        &dst_slice_pitch))) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }

    iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
    iree_status_t alloc_status = iree_arena_allocate(
        &stream_graph->arena, sizeof(*callback_data), (void**)&callback_data);
    if (!iree_status_is_ok(alloc_status)) {
      iree_status_ignore(alloc_status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    memset(callback_data, 0, sizeof(*callback_data));
    callback_data->dst = dst;
    callback_data->src = src;
    callback_data->count = size;
    callback_data->width = execution_params.extent.width;
    callback_data->height = execution_params.extent.height;
    callback_data->depth = execution_params.extent.depth;
    callback_data->dst_pitch = execution_params.dstPtr.pitch;
    callback_data->src_pitch = execution_params.srcPtr.pitch;
    callback_data->dst_slice_pitch = dst_slice_pitch;
    callback_data->src_slice_pitch = src_slice_pitch;

    iree_hal_streaming_graph_node_t* node = NULL;
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_graph_add_host_call_node(
            stream_graph, deps, numDependencies, iree_hip_graph_memcpy_callback,
            callback_data, &node),
        hipErrorInvalidValue);
    node->attrs.host.user_data_size = sizeof(*callback_data);
    iree_hip_graph_store_host_memcpy3d_metadata(node, params);

    *pGraphNode = (hipGraphNode_t)node;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  if (kind == hipMemcpyHostToDevice) {
    iree_hal_streaming_buffer_t* staging = NULL;
    iree_status_t status = iree_hal_streaming_graph_allocate_host_staging(
        stream_graph, size, &staging);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }

    iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
    iree_status_t alloc_status = iree_arena_allocate(
        &stream_graph->arena, sizeof(*callback_data), (void**)&callback_data);
    if (!iree_status_is_ok(alloc_status)) {
      iree_status_ignore(alloc_status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    iree_host_size_t src_rows_per_slice = 0;
    hipError_t callback_result = iree_hip_graph_memcpy_rows_per_slice(
        execution_params.srcPos.y, execution_params.extent.height,
        execution_params.srcPtr.ysize, &src_rows_per_slice);
    if (callback_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(callback_result);
    }
    callback_result = iree_hip_graph_configure_memcpy_callback_data(
        callback_data, &execution_params, staging->host_ptr, src, size,
        execution_params.extent.width, execution_params.srcPtr.pitch,
        execution_params.extent.height, src_rows_per_slice);
    if (callback_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(callback_result);
    }
    callback_data->hip_params = *params;
    callback_data->hip_params.kind =
        iree_hip_graph_metadata_memcpy_kind(params->kind);

    iree_hal_streaming_graph_node_t* callback_node = NULL;
    status = iree_hal_streaming_graph_add_host_call_node(
        stream_graph, deps, numDependencies, iree_hip_graph_memcpy_callback,
        callback_data, &callback_node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    callback_node->attrs.host.user_data_size = sizeof(*callback_data);

    iree_hal_streaming_graph_node_t* node = NULL;
    status = iree_hal_streaming_graph_add_copy_ptr_node_with_extra_dependency(
        stream_graph, deps, numDependencies, callback_node,
        (iree_hal_streaming_deviceptr_t)dst, staging->device_ptr, size, &node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    iree_hip_graph_store_memcpy3d_metadata(node, params);
    iree_host_size_t dst_rows_per_slice = 0;
    callback_result = iree_hip_graph_memcpy_rows_per_slice(
        execution_params.dstPos.y, execution_params.extent.height,
        execution_params.dstPtr.ysize, &dst_rows_per_slice);
    if (callback_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(callback_result);
    }
    iree_hip_graph_store_memcpy_execution_geometry(
        node, execution_params.dstPtr.pitch, execution_params.extent.width,
        dst_rows_per_slice, execution_params.extent.height,
        execution_params.extent.width, execution_params.extent.height,
        execution_params.extent.depth);

    *pGraphNode = (hipGraphNode_t)node;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  if (kind == hipMemcpyDeviceToDevice) {
    iree_hal_streaming_context_t* dst_context = NULL;
    iree_hal_streaming_buffer_ref_t dst_ref = {0};
    iree_status_t dst_lookup_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)dst, dst_span, &dst_context,
            &dst_ref);
    iree_hal_streaming_context_t* src_context = NULL;
    iree_hal_streaming_buffer_ref_t src_ref = {0};
    iree_status_t src_lookup_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)src, src_span, &src_context,
            &src_ref);
    const bool have_dst = iree_status_is_ok(dst_lookup_status);
    const bool have_src = iree_status_is_ok(src_lookup_status);
    hipError_t peer_result = hipSuccess;
    bool used_staged_peer = false;
    if (have_dst && have_src &&
        (dst_context != stream_graph->context ||
         src_context != stream_graph->context)) {
      peer_result = iree_hip_graph_add_staged_peer_memcpy_node(
          pGraphNode, stream_graph, deps, numDependencies, dst_context, dst,
          src_context, src, params, &execution_params, size, dst_span,
          src_span);
      used_staged_peer = true;
    }
    if (have_dst) {
      iree_hal_streaming_context_release(dst_context);
    }
    iree_status_ignore(dst_lookup_status);
    if (have_src) {
      iree_hal_streaming_context_release(src_context);
    }
    iree_status_ignore(src_lookup_status);
    if (used_staged_peer) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(peer_result);
    }
  }

  iree_hal_streaming_buffer_ref_t src_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      stream_graph->context, (iree_hal_streaming_deviceptr_t)src, src_span,
      &src_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_buffer_ref_t dst_ref;
  status = iree_hal_streaming_memory_lookup_range(
      stream_graph->context, (iree_hal_streaming_deviceptr_t)dst, dst_span,
      &dst_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    if (kind != hipMemcpyDeviceToHost) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }

    const iree_device_size_t compact_src_pitch = execution_params.extent.width;
    const iree_device_size_t compact_src_ysize = execution_params.extent.height;
    iree_host_size_t dst_rows_per_slice = execution_params.dstPtr.ysize;
    iree_host_size_t dst_slice_pitch = 0;
    iree_host_size_t src_slice_pitch = 0;
    if (IREE_UNLIKELY(
            (dst_rows_per_slice == 0 &&
             !iree_host_size_checked_add(execution_params.dstPos.y,
                                         execution_params.extent.height,
                                         &dst_rows_per_slice)) ||
            !iree_host_size_checked_mul(execution_params.dstPtr.pitch,
                                        dst_rows_per_slice, &dst_slice_pitch) ||
            !iree_host_size_checked_mul(compact_src_pitch, compact_src_ysize,
                                        &src_slice_pitch))) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }

    iree_hal_streaming_buffer_t* staging = NULL;
    status = iree_hal_streaming_graph_allocate_host_staging(stream_graph, size,
                                                            &staging);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    status = iree_hal_streaming_memory_lookup_range(
        stream_graph->context, staging->device_ptr, size, &dst_ref);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }

    iree_hal_streaming_graph_node_t* copy_node = NULL;
    status = iree_hal_streaming_graph_add_copy_buffer_node(
        stream_graph, deps, numDependencies, dst_ref, src_ref, size,
        &copy_node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    iree_hip_graph_store_memcpy3d_metadata(copy_node, params);
    iree_hip_graph_store_memcpy_execution_geometry(
        copy_node, execution_params.extent.width, execution_params.srcPtr.pitch,
        execution_params.extent.height, execution_params.srcPtr.ysize,
        execution_params.extent.width, execution_params.extent.height,
        execution_params.extent.depth);

    iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
    iree_status_t alloc_status = iree_arena_allocate(
        &stream_graph->arena, sizeof(*callback_data), (void**)&callback_data);
    if (!iree_status_is_ok(alloc_status)) {
      iree_status_ignore(alloc_status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    memset(callback_data, 0, sizeof(*callback_data));
    callback_data->dst = dst;
    callback_data->src = staging->host_ptr;
    callback_data->count = size;
    callback_data->width = execution_params.extent.width;
    callback_data->height = execution_params.extent.height;
    callback_data->depth = execution_params.extent.depth;
    callback_data->dst_pitch = execution_params.dstPtr.pitch;
    callback_data->src_pitch = compact_src_pitch;
    callback_data->dst_slice_pitch = dst_slice_pitch;
    callback_data->src_slice_pitch = src_slice_pitch;

    iree_hal_streaming_graph_node_t* copy_deps[] = {copy_node};
    iree_hal_streaming_graph_node_t* callback_node = NULL;
    status = iree_hal_streaming_graph_add_host_call_node(
        stream_graph, copy_deps, IREE_ARRAYSIZE(copy_deps),
        iree_hip_graph_memcpy_callback, callback_data, &callback_node);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    callback_node->attrs.host.user_data_size = sizeof(*callback_data);

    *pGraphNode = (hipGraphNode_t)copy_node;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  iree_hal_streaming_graph_node_t* node = NULL;
  status = iree_hal_streaming_graph_add_copy_buffer_node(
      stream_graph, deps, numDependencies, dst_ref, src_ref, size, &node);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hip_graph_store_memcpy3d_metadata(node, params);
  iree_hip_graph_store_memcpy_execution_geometry(
      node, execution_params.dstPtr.pitch, execution_params.srcPtr.pitch,
      execution_params.dstPtr.ysize, execution_params.srcPtr.ysize,
      execution_params.extent.width, execution_params.extent.height,
      execution_params.extent.depth);

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds a 1D memory copy node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - dst: [IN] Destination memory address.
//  - src: [IN] Source memory address.
//  - count: [IN] Number of bytes to copy.
//  - kind: [IN] Type of transfer (host/device).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//
// See also: hipGraphAddMemcpyNode.
HIPAPI hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode,
                                          hipGraph_t graph,
                                          const hipGraphNode_t* pDependencies,
                                          size_t numDependencies, void* dst,
                                          const void* src, size_t count,
                                          hipMemcpyKind kind) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !dst || !src ||
      (numDependencies > 0 && !pDependencies)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipMemcpyKind requested_kind = kind;
  hipError_t kind_result = iree_hip_resolve_memcpy_kind(stream_graph->context,
                                                        dst, src, kind, &kind);
  if (kind_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(kind_result);
  }
  hipError_t range_result = iree_hip_validate_memcpy_ranges(
      stream_graph->context, dst, src, count, kind, true);
  if (range_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(range_result);
  }

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  iree_hal_streaming_graph_node_t* node = NULL;

  // For device-to-device copies, try to use the optimized memcpy node.
  // For copies involving host memory, use graph-owned host staging buffers so
  // callbacks never submit blocking transfers from a queue completion thread.
  if (kind == hipMemcpyDeviceToDevice) {
    iree_hal_streaming_context_t* dst_context = NULL;
    iree_hal_streaming_buffer_ref_t dst_ref = {0};
    iree_status_t dst_lookup_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)dst, count, &dst_context, &dst_ref);
    iree_hal_streaming_context_t* src_context = NULL;
    iree_hal_streaming_buffer_ref_t src_ref = {0};
    iree_status_t src_lookup_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)src, count, &src_context, &src_ref);
    const bool have_dst = iree_status_is_ok(dst_lookup_status);
    const bool have_src = iree_status_is_ok(src_lookup_status);
    hipError_t peer_result = hipSuccess;
    bool used_staged_peer = false;
    if (have_dst && have_src &&
        (dst_context != stream_graph->context ||
         src_context != stream_graph->context)) {
      hipMemcpy3DParms params = {0};
      params.dstPtr.ptr = dst;
      params.dstPtr.pitch = count;
      params.dstPtr.xsize = count;
      params.dstPtr.ysize = 1;
      params.srcPtr.ptr = (void*)src;
      params.srcPtr.pitch = count;
      params.srcPtr.xsize = count;
      params.srcPtr.ysize = 1;
      params.extent.width = count;
      params.extent.height = 1;
      params.extent.depth = 1;
      params.kind = requested_kind;
      peer_result = iree_hip_graph_add_staged_peer_memcpy_node(
          pGraphNode, stream_graph, deps, numDependencies, dst_context, dst,
          src_context, src, &params, &params, count, count, count);
      used_staged_peer = true;
    }
    if (have_dst) {
      iree_hal_streaming_context_release(dst_context);
    }
    iree_status_ignore(dst_lookup_status);
    if (have_src) {
      iree_hal_streaming_context_release(src_context);
    }
    iree_status_ignore(src_lookup_status);
    if (used_staged_peer) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(peer_result);
    }

    iree_status_t status = iree_hal_streaming_graph_add_copy_ptr_node(
        stream_graph, deps, numDependencies,
        (iree_hal_streaming_deviceptr_t)dst,
        (iree_hal_streaming_deviceptr_t)src, count, &node);
    if (iree_status_is_ok(status)) {
      iree_hip_graph_store_1d_memcpy_metadata(node, dst, src, count,
                                              requested_kind);
      *pGraphNode = (hipGraphNode_t)node;
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  } else if (kind == hipMemcpyHostToDevice) {
    iree_hal_streaming_buffer_t* staging = NULL;
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_graph_allocate_host_staging(stream_graph, count,
                                                       &staging),
        hipErrorOutOfMemory);

    iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
    iree_status_t alloc_status = iree_arena_allocate(
        &stream_graph->arena, sizeof(iree_hip_graph_memcpy_callback_data_t),
        (void**)&callback_data);
    if (!iree_status_is_ok(alloc_status)) {
      iree_status_ignore(alloc_status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    memset(callback_data, 0, sizeof(*callback_data));
    callback_data->dst = staging->host_ptr;
    callback_data->src = src;
    callback_data->count = count;

    iree_hal_streaming_graph_node_t* callback_node = NULL;
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_graph_add_host_call_node(
            stream_graph, deps, numDependencies, iree_hip_graph_memcpy_callback,
            callback_data, &callback_node),
        hipErrorInvalidValue);
    callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    callback_node->attrs.host.user_data_size = sizeof(*callback_data);

    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_graph_add_copy_ptr_node_with_extra_dependency(
            stream_graph, deps, numDependencies, callback_node,
            (iree_hal_streaming_deviceptr_t)dst, staging->device_ptr, count,
            &node),
        hipErrorInvalidValue);
    iree_hip_graph_store_1d_memcpy_metadata(node, dst, src, count,
                                            requested_kind);
    *pGraphNode = (hipGraphNode_t)node;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  } else if (kind == hipMemcpyDeviceToHost) {
    iree_hal_streaming_buffer_t* staging = NULL;
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_graph_allocate_host_staging(stream_graph, count,
                                                       &staging),
        hipErrorOutOfMemory);
    iree_hal_streaming_graph_node_t* copy_node = NULL;
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_graph_add_copy_ptr_node(
            stream_graph, deps, numDependencies, staging->device_ptr,
            (iree_hal_streaming_deviceptr_t)src, count, &copy_node),
        hipErrorInvalidValue);
    iree_hip_graph_store_1d_memcpy_metadata(copy_node, dst, src, count,
                                            requested_kind);

    iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
    iree_status_t alloc_status = iree_arena_allocate(
        &stream_graph->arena, sizeof(iree_hip_graph_memcpy_callback_data_t),
        (void**)&callback_data);
    if (!iree_status_is_ok(alloc_status)) {
      iree_status_ignore(alloc_status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }

    memset(callback_data, 0, sizeof(*callback_data));
    callback_data->dst = dst;
    callback_data->src = staging->host_ptr;
    callback_data->count = count;

    iree_hal_streaming_graph_node_t* copy_deps[] = {copy_node};
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_graph_add_host_call_node(
            stream_graph, copy_deps, IREE_ARRAYSIZE(copy_deps),
            iree_hip_graph_memcpy_callback, callback_data, &node),
        hipErrorInvalidValue);
    node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    node->attrs.host.user_data_size = sizeof(*callback_data);

    *pGraphNode = (hipGraphNode_t)copy_node;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  if (kind != hipMemcpyHostToHost) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Host-to-host copies do not touch the device queue.
  iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
  iree_status_t alloc_status = iree_arena_allocate(
      &stream_graph->arena, sizeof(iree_hip_graph_memcpy_callback_data_t),
      (void**)&callback_data);
  if (!iree_status_is_ok(alloc_status)) {
    iree_status_ignore(alloc_status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  memset(callback_data, 0, sizeof(*callback_data));
  callback_data->dst = dst;
  callback_data->src = src;
  callback_data->count = count;

  // Create a host callback node that performs the memcpy.
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_host_call_node(
          stream_graph, deps, numDependencies, iree_hip_graph_memcpy_callback,
          callback_data, &node),
      hipErrorInvalidValue);
  node->attrs.host.user_data_size = sizeof(*callback_data);
  hipMemcpy3DParms params = {0};
  params.dstPtr.ptr = dst;
  params.dstPtr.pitch = count;
  params.dstPtr.xsize = count;
  params.dstPtr.ysize = 1;
  params.srcPtr.ptr = (void*)src;
  params.srcPtr.pitch = count;
  params.srcPtr.xsize = count;
  params.srcPtr.ysize = 1;
  params.extent.width = count;
  params.extent.height = 1;
  params.extent.depth = 1;
  params.kind = requested_kind;
  iree_hip_graph_store_host_memcpy3d_metadata(node, &params);

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds a memory set node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pMemsetParams: [IN] Memset parameters (hipMemsetParams structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: This operation modifies the graph structure.
//
// Memset parameters (hipMemsetParams):
// - dst: Destination device pointer.
// - value: Value to set (supports 1, 2, or 4 byte values).
// - elementSize: Size of each element (1, 2, or 4 bytes).
// - width: Width of 2D memset area in elements.
// - height: Height of 2D memset area.
// - pitch: Pitch of destination memory.
//
// Memset behavior:
// - Captures parameters at node creation.
// - Executes memset when dependencies satisfied.
// - Supports 1D and 2D memory regions.
// - Fills memory with specified pattern.
//
// Pattern replication:
// - 1-byte: Pattern replicated across region.
// - 2-byte: 16-bit pattern (requires alignment).
// - 4-byte: 32-bit pattern (requires alignment).
//
// Performance optimization:
// - Optimized for memory bandwidth.
// - May use specialized hardware for fills.
// - Coalesced memory access patterns.
//
// Multi-GPU:
// - Memset executes on device owning the memory.
// - Cross-device dependencies handled automatically.
//
// Warning: Destination address captured at creation.
// Ensure memory remains valid for all graph launches.
//
// See also: hipGraphAddMemcpyNode, hipGraphAddKernelNode,
//           hipMemsetAsync, hipGraphMemsetNodeSetParams.
HIPAPI hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pMemsetParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pMemsetParams ||
      (numDependencies > 0 && !pDependencies)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipMemsetParams* params = (const hipMemsetParams*)pMemsetParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  iree_hal_streaming_graph_node_t* node = NULL;
  if (!params->dst || params->width == 0 || params->height == 0 ||
      (params->elementSize != 1 && params->elementSize != 2 &&
       params->elementSize != 4)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_host_size_t row_bytes = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          params->width, params->elementSize, &row_bytes))) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const size_t pitch = params->pitch ? params->pitch : row_bytes;
  if (params->height > 1 && pitch < row_bytes) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_host_size_t leading_rows = params->height - 1;
  iree_host_size_t leading_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(leading_rows, pitch, &leading_size) ||
          !iree_host_size_checked_add(leading_size, row_bytes, &total_size) ||
          total_size % params->elementSize != 0)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const iree_host_size_t memset_count = total_size / params->elementSize;

  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_fill_ptr_node(
          stream_graph, deps, numDependencies,
          (iree_hal_streaming_deviceptr_t)params->dst, params->value,
          params->elementSize, memset_count, &node),
      hipErrorInvalidValue);
  node->attrs.memset.hip_dst = params->dst;
  node->attrs.memset.hip_width = params->width;
  node->attrs.memset.hip_height = params->height;
  node->attrs.memset.hip_pitch = params->pitch;

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds a host callback node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pNodeParams: [IN] Host node parameters (hipHostNodeParams structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: Host callback blocks graph execution.
//
// Host node parameters (hipHostNodeParams):
// - fn: Host function to call.
// - userData: User data passed to callback.
//
// Callback signature:
// ```c
// void hostFunction(void* userData);
// ```
//
// Callback behavior:
// - Executes on host when dependencies complete.
// - Blocks graph execution until callback returns.
// - Runs on runtime thread, not application thread.
// - Can make HIP API calls except synchronization.
//
// Restrictions in callback:
// - Cannot call hipStreamSynchronize().
// - Cannot call hipDeviceSynchronize().
// - Cannot wait on events from same graph.
// - Should complete quickly to avoid stalls.
//
// Use cases:
// - CPU computation between GPU operations.
// - Logging or debugging within graphs.
// - Dynamic parameter updates.
// - Resource management.
//
// Performance considerations:
// - Host nodes serialize graph execution.
// - Long callbacks hurt GPU utilization.
// - Consider async alternatives when possible.
//
// Multi-GPU:
// - Callback executes once regardless of devices.
// - Can access resources from multiple devices.
//
// Warning: Avoid heavy computation in callbacks.
// Graph execution stalls until callback completes.
//
// See also: hipGraphAddKernelNode, hipLaunchHostFunc,
//           hipGraphHostNodeSetParams.
HIPAPI hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode,
                                      hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies,
                                      size_t numDependencies,
                                      const void* pNodeParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pNodeParams ||
      (numDependencies > 0 && !pDependencies)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipHostNodeParams* params = (const hipHostNodeParams*)pNodeParams;
  if (!params->fn) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // Add host node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_host_call_node(
          stream_graph, deps, numDependencies, (void (*)(void*))params->fn,
          params->userData, &node),
      hipErrorInvalidValue);

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds an empty node for synchronization to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: Empty nodes are pure synchronization points.
//
// Empty node behavior:
// - No operation performed.
// - Only waits for dependencies.
// - Acts as synchronization barrier.
// - Zero runtime overhead.
//
// Use cases:
// - Join multiple parallel paths.
// - Create synchronization points.
// - Simplify complex dependencies.
// - Graph structure organization.
//
// Dependency patterns:
// - Fan-in: Multiple nodes converge to empty node.
// - Fan-out: Empty node fans out to multiple nodes.
// - Barrier: Forces serialization point.
//
// Graph optimization:
// - May be optimized away if redundant.
// - Helps graph analysis and scheduling.
// - No device resources required.
//
// Multi-GPU:
// - Synchronizes across device boundaries.
// - Useful for cross-device coordination.
//
// Example:
// ```c
// // Create barrier after parallel operations
// hipGraphNode_t nodes[] = {kernel1, kernel2, kernel3};
// hipGraphNode_t barrier;
// hipGraphAddEmptyNode(&barrier, graph, nodes, 3);
// ```
//
// See also: hipGraphAddKernelNode, hipGraphNodeGetType,
//           hipGraphNodeGetDependencies.
HIPAPI hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode,
                                       hipGraph_t graph,
                                       const hipGraphNode_t* pDependencies,
                                       size_t numDependencies) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (numDependencies > 0 && !pDependencies) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // Empty nodes are just synchronization points.
  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_empty_node(stream_graph, deps,
                                              numDependencies, &node),
      hipErrorInvalidValue);

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets all nodes from a graph.
//
// Parameters:
//  - graph: [IN] Graph to query nodes from.
//  - pNodes: [OUT] Array to store node pointers (optional).
//  - numNodes: [IN/OUT] Size of pNodes array on input, actual count on output.
//
// Returns:
//  - hipSuccess: Nodes retrieved successfully.
//  - hipErrorInvalidValue: Invalid graph or numNodes is NULL.
//
// Query behavior:
// - If pNodes is NULL, returns only the count in *numNodes.
// - If pNodes is non-NULL, fills array up to *numNodes elements.
// - Always updates *numNodes with actual node count.
//
// Usage pattern:
// ```c
// // Query count first.
// size_t nodeCount = 0;
// hipGraphGetNodes(graph, NULL, &nodeCount);
//
// // Allocate and get nodes.
// hipGraphNode_t* nodes = malloc(nodeCount * sizeof(hipGraphNode_t));
// hipGraphGetNodes(graph, nodes, &nodeCount);
// ```
//
// Node ordering:
// - No guaranteed order (implementation defined).
// - May differ between calls.
// - Includes all node types.
//
// Memory management:
// - Caller allocates pNodes array.
// - Node pointers remain valid until graph destruction.
//
// Thread safety:
// - Safe to call concurrently for same graph.
// - Graph must not be modified during call.
//
// See also: hipGraphGetRootNodes, hipGraphNodeGetType,
//           hipGraphNodeGetDependencies.
static bool iree_hip_graph_node_is_hidden(
    const iree_hal_streaming_graph_node_t* node) {
  return node && (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) != 0;
}

static bool iree_hip_graph_edge_is_hidden(
    const iree_hal_streaming_graph_node_t* from,
    const iree_hal_streaming_graph_node_t* to) {
  return iree_hip_graph_node_is_hidden(from) ||
         iree_hip_graph_node_is_hidden(to);
}

HIPAPI hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* pNodes,
                                   size_t* numNodes) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!graph || !numNodes) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;

  size_t total_count = 0;
  for (iree_hal_streaming_node_block_t* block = stream_graph->node_blocks;
       block; block = block->next) {
    for (size_t i = 0; i < block->count; ++i) {
      if (!iree_hip_graph_node_is_hidden(block->nodes[i])) ++total_count;
    }
  }

  // Get nodes from the graph (up to the requested count).
  if (pNodes != NULL) {
    const size_t requested_count = *numNodes;
    const size_t copied_count =
        total_count < requested_count ? total_count : requested_count;
    size_t copied_index = 0;
    for (iree_hal_streaming_node_block_t* block = stream_graph->node_blocks;
         block && copied_index < copied_count; block = block->next) {
      for (size_t i = 0; i < block->count && copied_index < copied_count; ++i) {
        if (iree_hip_graph_node_is_hidden(block->nodes[i])) continue;
        pNodes[copied_index++] = (hipGraphNode_t)block->nodes[i];
      }
    }
    for (size_t i = copied_count; i < requested_count; ++i) {
      pNodes[i] = NULL;
    }
    *numNodes = copied_count;
  } else {
    *numNodes = total_count;
  }
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds an event record node to a graph.
HIPAPI hipError_t
hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                           const hipGraphNode_t* pDependencies,
                           size_t numDependencies, hipEvent_t event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraphNode || !graph || !event) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (numDependencies > 0 && !pDependencies) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_event_node(
          (iree_hal_streaming_graph_t*)graph,
          (iree_hal_streaming_graph_node_t**)pDependencies, numDependencies,
          IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD,
          (iree_hal_streaming_event_t*)event, &node),
      hipErrorInvalidValue);
  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds an event wait node to a graph.
HIPAPI hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode,
                                           hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies,
                                           hipEvent_t event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraphNode || !graph || !event) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (numDependencies > 0 && !pDependencies) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_event_node(
          (iree_hal_streaming_graph_t*)graph,
          (iree_hal_streaming_graph_node_t**)pDependencies, numDependencies,
          IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT,
          (iree_hal_streaming_event_t*)event, &node),
      hipErrorInvalidValue);
  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds dependencies between nodes in a graph.
HIPAPI hipError_t hipGraphAddDependencies(hipGraph_t graph,
                                          const hipGraphNode_t* from,
                                          const hipGraphNode_t* to,
                                          size_t numDependencies) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (numDependencies == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  if (!from || !to) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;

  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_dependencies(
          stream_graph, (iree_hal_streaming_graph_node_t**)from,
          (iree_hal_streaming_graph_node_t**)to, numDependencies),
      hipErrorInvalidValue);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Removes dependencies between nodes in a graph.
static bool iree_hip_graph_dependency_pair_exists(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t* from_node,
    iree_hal_streaming_graph_node_t* to_node) {
  for (uint32_t i = 0; i < to_node->dependency_count; ++i) {
    if (to_node->dependencies[i] == from_node) return true;
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (edge->from == from_node && edge->to == to_node) return true;
  }
  return false;
}

static void iree_hip_graph_remove_dependency_pair(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t* from_node,
    iree_hal_streaming_graph_node_t* to_node) {
  for (uint32_t i = 0; i < to_node->dependency_count; ++i) {
    if (to_node->dependencies[i] != from_node) continue;
    for (uint32_t j = i + 1; j < to_node->dependency_count; ++j) {
      to_node->dependencies[j - 1] = to_node->dependencies[j];
    }
    --to_node->dependency_count;
    return;
  }

  iree_hal_streaming_graph_edge_t** next_edge = &graph->additional_edges;
  while (*next_edge) {
    iree_hal_streaming_graph_edge_t* edge = *next_edge;
    if (edge->from == from_node && edge->to == to_node) {
      *next_edge = edge->next;
      --graph->additional_edge_count;
      return;
    }
    next_edge = &edge->next;
  }
}

HIPAPI hipError_t hipGraphRemoveDependencies(hipGraph_t graph,
                                             const hipGraphNode_t* from,
                                             const hipGraphNode_t* to,
                                             size_t numDependencies) {
  if (!graph) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (numDependencies == 0) {
    return hipSuccess;
  }
  if (!from || !to) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  for (size_t i = 0; i < numDependencies; ++i) {
    iree_hal_streaming_graph_node_t* from_node =
        (iree_hal_streaming_graph_node_t*)from[i];
    iree_hal_streaming_graph_node_t* to_node =
        (iree_hal_streaming_graph_node_t*)to[i];
    if (!from_node || !to_node || from_node->graph != stream_graph ||
        to_node->graph != stream_graph ||
        !iree_hip_graph_dependency_pair_exists(stream_graph, from_node,
                                               to_node)) {
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }

  for (size_t i = 0; i < numDependencies; ++i) {
    iree_hip_graph_remove_dependency_pair(
        stream_graph, (iree_hal_streaming_graph_node_t*)from[i],
        (iree_hal_streaming_graph_node_t*)to[i]);
  }

  return hipSuccess;
}

// Gets edges in a graph.
HIPAPI hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from,
                                   hipGraphNode_t* to, size_t* numEdges) {
  if (!graph || !numEdges || (!from && to) || (from && !to)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const size_t capacity = from && to ? *numEdges : 0;
  size_t edge_count = 0;
  size_t copied_count = 0;

  for (iree_hal_streaming_node_block_t* block = stream_graph->node_blocks;
       block; block = block->next) {
    for (size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (iree_hip_graph_node_is_hidden(node)) continue;
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        if (iree_hip_graph_edge_is_hidden(node->dependencies[j], node)) {
          continue;
        }
        if (copied_count < capacity) {
          from[copied_count] = (hipGraphNode_t)node->dependencies[j];
          to[copied_count] = (hipGraphNode_t)node;
          ++copied_count;
        }
        ++edge_count;
      }
    }
  }

  for (iree_hal_streaming_graph_edge_t* edge = stream_graph->additional_edges;
       edge; edge = edge->next) {
    if (iree_hip_graph_edge_is_hidden(edge->from, edge->to)) continue;
    if (copied_count < capacity) {
      from[copied_count] = (hipGraphNode_t)edge->from;
      to[copied_count] = (hipGraphNode_t)edge->to;
      ++copied_count;
    }
    ++edge_count;
  }

  if (!from) {
    *numEdges = edge_count;
  } else if (capacity > edge_count) {
    for (size_t i = edge_count; i < capacity; ++i) {
      from[i] = NULL;
      to[i] = NULL;
    }
    *numEdges = edge_count;
  }
  return hipSuccess;
}

// Gets root nodes (nodes with no dependencies) in a graph.
HIPAPI hipError_t hipGraphGetRootNodes(hipGraph_t graph,
                                       hipGraphNode_t* pRootNodes,
                                       size_t* pNumRootNodes) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!graph || !pNumRootNodes) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;

  // Iterate through all nodes to find those with no incoming dependencies.
  // A node is a root if:
  // 1. It has no embedded dependencies (node->dependency_count == 0)
  // 2. It has no incoming additional edges

  // Build a set of nodes that have incoming edges from additional_edges.
  // For simplicity, we use a linear search approach for small graphs.
  const size_t capacity = pRootNodes ? *pNumRootNodes : 0;
  size_t root_count = 0;
  size_t copied_count = 0;

  iree_hal_streaming_node_block_t* block = stream_graph->node_blocks;
  while (block) {
    for (size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (iree_hip_graph_node_is_hidden(node)) continue;

      // Check if node has any embedded dependencies.
      bool has_embedded_dependency = false;
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        if (!iree_hip_graph_edge_is_hidden(node->dependencies[j], node)) {
          has_embedded_dependency = true;
          break;
        }
      }
      if (has_embedded_dependency) {
        continue;
      }

      // Check if node has any incoming additional edges.
      bool has_incoming_edge = false;
      iree_hal_streaming_graph_edge_t* edge = stream_graph->additional_edges;
      while (edge) {
        if (edge->to == node &&
            !iree_hip_graph_edge_is_hidden(edge->from, edge->to)) {
          has_incoming_edge = true;
          break;
        }
        edge = edge->next;
      }

      if (!has_incoming_edge) {
        // This is a root node.
        if (copied_count < capacity) {
          pRootNodes[copied_count] = (hipGraphNode_t)node;
          ++copied_count;
        }
        ++root_count;
      }
    }
    block = block->next;
  }

  // If pRootNodes is NULL, just return the count.
  if (!pRootNodes) {
    *pNumRootNodes = root_count;
  } else if (capacity > root_count) {
    for (size_t i = root_count; i < capacity; ++i) {
      pRootNodes[i] = NULL;
    }
    *pNumRootNodes = root_count;
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets dependencies of a node.
HIPAPI hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,
                                              hipGraphNode_t* pDependencies,
                                              size_t* pNumDependencies) {
  if (!node || !pNumDependencies) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  iree_hal_streaming_graph_t* graph = stream_node->graph;
  if (!graph || iree_hip_graph_node_is_hidden(stream_node)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  const size_t capacity = pDependencies ? *pNumDependencies : 0;
  size_t dependency_count = 0;
  size_t copied_count = 0;
  for (uint32_t i = 0; i < stream_node->dependency_count; ++i) {
    if (iree_hip_graph_edge_is_hidden(stream_node->dependencies[i],
                                      stream_node)) {
      continue;
    }
    if (copied_count < capacity) {
      pDependencies[copied_count] =
          (hipGraphNode_t)stream_node->dependencies[i];
      ++copied_count;
    }
    ++dependency_count;
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (edge->to != stream_node) continue;
    if (iree_hip_graph_edge_is_hidden(edge->from, edge->to)) continue;
    if (copied_count < capacity) {
      pDependencies[copied_count] = (hipGraphNode_t)edge->from;
      ++copied_count;
    }
    ++dependency_count;
  }

  if (!pDependencies) {
    *pNumDependencies = dependency_count;
  } else if (capacity > dependency_count) {
    for (size_t i = dependency_count; i < capacity; ++i) {
      pDependencies[i] = NULL;
    }
    *pNumDependencies = dependency_count;
  }
  return hipSuccess;
}

// Gets dependent nodes of a node.
HIPAPI hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,
                                                hipGraphNode_t* pDependentNodes,
                                                size_t* pNumDependentNodes) {
  if (!node || !pNumDependentNodes) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  iree_hal_streaming_graph_t* graph = stream_node->graph;
  if (!graph || iree_hip_graph_node_is_hidden(stream_node)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  const size_t capacity = pDependentNodes ? *pNumDependentNodes : 0;
  size_t dependent_count = 0;
  size_t copied_count = 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* candidate = block->nodes[i];
      if (iree_hip_graph_node_is_hidden(candidate)) continue;
      for (uint32_t j = 0; j < candidate->dependency_count; ++j) {
        if (candidate->dependencies[j] != stream_node) continue;
        if (iree_hip_graph_edge_is_hidden(stream_node, candidate)) continue;
        if (copied_count < capacity) {
          pDependentNodes[copied_count] = (hipGraphNode_t)candidate;
          ++copied_count;
        }
        ++dependent_count;
        break;
      }
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (edge->from != stream_node) continue;
    if (iree_hip_graph_edge_is_hidden(edge->from, edge->to)) continue;
    if (copied_count < capacity) {
      pDependentNodes[copied_count] = (hipGraphNode_t)edge->to;
      ++copied_count;
    }
    ++dependent_count;
  }

  if (!pDependentNodes) {
    *pNumDependentNodes = dependent_count;
  } else if (capacity > dependent_count) {
    for (size_t i = dependent_count; i < capacity; ++i) {
      pDependentNodes[i] = NULL;
    }
    *pNumDependentNodes = dependent_count;
  }
  return hipSuccess;
}

static hipGraphNodeType hip_graph_node_type_from_streaming_type(
    iree_hal_streaming_graph_node_type_t type) {
  switch (type) {
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL:
      return hipGraphNodeTypeKernel;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY:
      return hipGraphNodeTypeMemcpy;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET:
      return hipGraphNodeTypeMemset;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL:
      return hipGraphNodeTypeHost;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
      return hipGraphNodeTypeGraph;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT:
      return hipGraphNodeTypeWaitEvent;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD:
      return hipGraphNodeTypeEventRecord;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC:
      return hipGraphNodeTypeMemAlloc;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE:
      return hipGraphNodeTypeMemFree;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP:
      return hipGraphNodeTypeBatchMemOp;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY:
    default:
      return hipGraphNodeTypeEmpty;
  }
}

// Gets the type of a node.
HIPAPI hipError_t hipGraphNodeGetType(hipGraphNode_t node,
                                      hipGraphNodeType* pType) {
  if (!node || !pType) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!stream_node->graph || iree_hip_graph_node_is_hidden(stream_node)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pType = hip_graph_node_type_from_streaming_type(stream_node->type);
  return hipSuccess;
}

// Destroys a graph node.
HIPAPI hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  HIP_RETURN_STATUS(iree_hal_streaming_graph_destroy_node(
                        (iree_hal_streaming_graph_node_t*)node),
                    hipErrorInvalidValue);
  return hipSuccess;
}

// Clones a graph.
HIPAPI hipError_t hipGraphClone(hipGraph_t* pGraphClone,
                                hipGraph_t originalGraph) {
  if (!pGraphClone || !originalGraph) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraphClone = NULL;
  if (!iree_hip_graph_handle_is_live(originalGraph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_t* clone_graph = NULL;
  HIP_RETURN_STATUS(
      iree_hal_streaming_graph_clone((iree_hal_streaming_graph_t*)originalGraph,
                                     &clone_graph),
      hipErrorInvalidValue);
  if (!iree_hip_live_graph_register((hipGraph_t)clone_graph)) {
    iree_hal_streaming_graph_release(clone_graph);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }
  *pGraphClone = (hipGraph_t)clone_graph;
  return hipSuccess;
}

// Finds a node in a cloned graph.
HIPAPI hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode,
                                          hipGraphNode_t originalNode,
                                          hipGraph_t clonedGraph) {
  if (!pNode || !originalNode || !clonedGraph) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(clonedGraph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* source_node =
      (iree_hal_streaming_graph_node_t*)originalNode;
  if (!source_node->graph) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* clone_graph =
      (iree_hal_streaming_graph_t*)clonedGraph;
  if (clone_graph->clone_source_graph_debug_id !=
      source_node->graph->debug_id) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  for (iree_hal_streaming_node_block_t* block = clone_graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* clone_node = block->nodes[i];
      if (clone_node->clone_source_node_index ==
          source_node->clone_source_node_index) {
        *pNode = (hipGraphNode_t)clone_node;
        return hipSuccess;
      }
    }
  }

  HIP_RETURN_ERROR(hipErrorInvalidValue);
}

static const char* iree_hip_graph_debug_node_label(
    iree_hal_streaming_graph_node_type_t type) {
  switch (type) {
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL:
      return "KERNEL";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY:
      return "MEMCPY";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET:
      return "MEMSET";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL:
      return "HOST";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
      return "GRAPH";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT:
      return "EVENT_WAIT";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD:
      return "EVENT_RECORD";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC:
      return "MEM_ALLOC";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE:
      return "MEM_FREE";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP:
      return "BATCH_MEM_OP";
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY:
    default:
      return "EMPTY";
  }
}

static const char* iree_hip_graph_debug_memcpy_direction(
    const iree_hal_streaming_graph_node_t* node) {
  const bool src_array = node->attrs.memcpy.hip_driver.valid &&
                         node->attrs.memcpy.hip_driver.src_array;
  const bool dst_array = node->attrs.memcpy.hip_driver.valid &&
                         node->attrs.memcpy.hip_driver.dst_array;
  switch ((hipMemcpyKind)node->attrs.memcpy.hip_kind) {
    case hipMemcpyHostToHost:
      return "HtoH";
    case hipMemcpyHostToDevice:
      return dst_array ? "HtoA" : "HtoD";
    case hipMemcpyDeviceToHost:
      return src_array ? "AtoH" : "DtoH";
    case hipMemcpyDeviceToDevice:
      if (src_array && dst_array) return "AtoA";
      if (src_array) return "AtoD";
      if (dst_array) return "DtoA";
      return "DtoD";
    case hipMemcpyDefault:
    default:
      return "Default";
  }
}

static int iree_hip_graph_debug_write_dot_graph(
    FILE* file, const iree_hal_streaming_graph_t* graph) {
  if (fprintf(file, "  subgraph cluster_%llu {\n",
              (unsigned long long)graph->debug_id) < 0 ||
      fprintf(file, "    label=\"graph_%llu\";\n",
              (unsigned long long)graph->debug_id) < 0) {
    return -1;
  }

  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      const iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (iree_hip_graph_node_is_hidden(node)) continue;
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
        if (fprintf(file,
                    "    \"graph_%llu_node_%llu\"[label=\"MEMCPY %s\"];\n",
                    (unsigned long long)node->graph->debug_id,
                    (unsigned long long)node->debug_id,
                    iree_hip_graph_debug_memcpy_direction(node)) < 0) {
          return -1;
        }
      } else {
        if (fprintf(file, "    \"graph_%llu_node_%llu\"[label=\"%s\"];\n",
                    (unsigned long long)node->graph->debug_id,
                    (unsigned long long)node->debug_id,
                    iree_hip_graph_debug_node_label(node->type)) < 0) {
          return -1;
        }
      }
    }
  }

  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      const iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (iree_hip_graph_node_is_hidden(node)) continue;
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        const iree_hal_streaming_graph_node_t* dependency =
            node->dependencies[j];
        if (iree_hip_graph_edge_is_hidden(dependency, node)) continue;
        if (fprintf(file,
                    "    \"graph_%llu_node_%llu\" -> "
                    "\"graph_%llu_node_%llu\";\n",
                    (unsigned long long)dependency->graph->debug_id,
                    (unsigned long long)dependency->debug_id,
                    (unsigned long long)node->graph->debug_id,
                    (unsigned long long)node->debug_id) < 0) {
          return -1;
        }
      }
    }
  }

  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (iree_hip_graph_edge_is_hidden(edge->from, edge->to)) continue;
    if (fprintf(file,
                "    \"graph_%llu_node_%llu\" -> "
                "\"graph_%llu_node_%llu\";\n",
                (unsigned long long)edge->from->graph->debug_id,
                (unsigned long long)edge->from->debug_id,
                (unsigned long long)edge->to->graph->debug_id,
                (unsigned long long)edge->to->debug_id) < 0) {
      return -1;
    }
  }

  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      const iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (iree_hip_graph_node_is_hidden(node) ||
          node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
        continue;
      }
      if (iree_hip_graph_debug_write_dot_graph(
              file, node->attrs.child_graph.graph) != 0) {
        return -1;
      }
    }
  }

  return fprintf(file, "  }\n") < 0 ? -1 : 0;
}

// Prints a graph in DOT format for debugging.
HIPAPI hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path,
                                        unsigned int flags) {
  (void)flags;
  if (!graph || !path || !iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (path[0] == '\0') {
    HIP_RETURN_ERROR(hipErrorOperatingSystem);
  }

  FILE* file = fopen(path, "w");
  if (!file) {
    HIP_RETURN_ERROR(hipErrorOperatingSystem);
  }

  const iree_hal_streaming_graph_t* stream_graph =
      (const iree_hal_streaming_graph_t*)graph;
  int result = fprintf(file, "digraph hip_graph {\n");
  if (result >= 0) {
    result = iree_hip_graph_debug_write_dot_graph(file, stream_graph);
  }
  if (result >= 0 && fprintf(file, "}\n") < 0) {
    result = -1;
  }
  if (fclose(file) != 0) {
    result = -1;
  }
  if (result < 0) {
    HIP_RETURN_ERROR(hipErrorOperatingSystem);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode,
                                            hipGraph_t graph,
                                            const hipGraphNode_t* pDependencies,
                                            size_t numDependencies,
                                            hipGraph_t childGraph) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraphNode || !graph || !childGraph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraphNode = NULL;
  if (numDependencies > 0 && !pDependencies) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph) ||
      !iree_hip_graph_handle_is_live(childGraph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_add_child_graph_node(
          (iree_hal_streaming_graph_t*)graph,
          (iree_hal_streaming_graph_node_t**)pDependencies, numDependencies,
          (iree_hal_streaming_graph_t*)childGraph, &node),
      hipErrorInvalidValue);
  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

static hipError_t iree_hip_resolve_symbol_copy_range(const void* symbol,
                                                     size_t count,
                                                     size_t offset,
                                                     void** out_symbol_ptr) {
  *out_symbol_ptr = NULL;
  if (!symbol) return hipErrorInvalidSymbol;
  if (count == 0 || offset > SIZE_MAX - count) return hipErrorInvalidValue;

  size_t symbol_size = 0;
  hipError_t result = hipGetSymbolSize(&symbol_size, symbol);
  if (result != hipSuccess) return result;
  if (count > symbol_size || offset > symbol_size - count) {
    return hipErrorInvalidValue;
  }

  void* symbol_ptr = NULL;
  result = hipGetSymbolAddress(&symbol_ptr, symbol);
  if (result != hipSuccess) return result;
  *out_symbol_ptr = (uint8_t*)symbol_ptr + offset;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphAddMemcpyNodeFromSymbol(
    hipGraphNode_t* pGraphNode, hipGraph_t graph,
    const hipGraphNode_t* pDependencies, size_t numDependencies, void* dst,
    const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
  if (!pGraphNode || !graph || !dst || count == 0 ||
      (numDependencies > 0 && !pDependencies)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  void* symbol_ptr = NULL;
  hipError_t result =
      iree_hip_resolve_symbol_copy_range(symbol, count, offset, &symbol_ptr);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }

  hipMemcpyKind copy_kind =
      kind == hipMemcpyDefault ? hipMemcpyDeviceToHost : kind;
  return hipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies,
                                 numDependencies, dst, symbol_ptr, count,
                                 copy_kind);
}

HIPAPI hipError_t hipGraphAddMemcpyNodeToSymbol(
    hipGraphNode_t* pGraphNode, hipGraph_t graph,
    const hipGraphNode_t* pDependencies, size_t numDependencies,
    const void* symbol, const void* src, size_t count, size_t offset,
    hipMemcpyKind kind) {
  if (!pGraphNode || !graph || !src || count == 0 ||
      (numDependencies > 0 && !pDependencies)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  void* symbol_ptr = NULL;
  hipError_t result =
      iree_hip_resolve_symbol_copy_range(symbol, count, offset, &symbol_ptr);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }

  hipMemcpyKind copy_kind =
      kind == hipMemcpyDefault ? hipMemcpyHostToDevice : kind;
  return hipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies,
                                 numDependencies, symbol_ptr, src, count,
                                 copy_kind);
}

HIPAPI hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode,
                                          hipGraph_t graph,
                                          const hipGraphNode_t* pDependencies,
                                          size_t numDependencies,
                                          void* allocParams) {
  if (!pGraphNode || !graph || !allocParams ||
      (numDependencies > 0 && !pDependencies)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraphNode = NULL;
  hipMemAllocNodeParams* params = (hipMemAllocNodeParams*)allocParams;
  params->dptr = NULL;

  int device_count = 0;
  hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) {
    HIP_RETURN_ERROR(count_result);
  }
  if (params->bytesize == 0 ||
      params->poolProps.allocType != hipMemAllocationTypePinned ||
      params->poolProps.location.type != hipMemLocationTypeDevice ||
      params->poolProps.location.id < 0 ||
      params->poolProps.location.id >= device_count ||
      (params->accessDescCount > 0 && !params->accessDescs) ||
      params->accessDescCount > (size_t)device_count) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  for (size_t i = 0; i < params->accessDescCount; ++i) {
    const hipMemAccessDesc* desc = &params->accessDescs[i];
    if (desc->location.type != hipMemLocationTypeDevice ||
        desc->location.id < 0 || desc->location.id >= device_count ||
        (desc->flags != hipMemAccessFlagsProtNone &&
         desc->flags != hipMemAccessFlagsProtRead &&
         desc->flags != hipMemAccessFlagsProtReadWrite)) {
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS(
      iree_hal_streaming_graph_add_empty_node(
          stream_graph, (iree_hal_streaming_graph_node_t**)pDependencies,
          numDependencies, &node),
      hipErrorInvalidValue);

  iree_host_size_t access_desc_size = 0;
  iree_host_size_t metadata_size = sizeof(*params);
  if (params->accessDescCount > 0) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(params->accessDescCount,
                                                  sizeof(*params->accessDescs),
                                                  &access_desc_size) ||
                      !iree_host_size_checked_add(
                          metadata_size, access_desc_size, &metadata_size))) {
      iree_status_ignore(iree_hal_streaming_graph_destroy_node(node));
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
  }
  uint8_t* metadata_storage = NULL;
  iree_status_t metadata_status = iree_arena_allocate(
      &stream_graph->arena, metadata_size, (void**)&metadata_storage);
  if (!iree_status_is_ok(metadata_status)) {
    iree_status_ignore(iree_hal_streaming_graph_destroy_node(node));
    HIP_RETURN_STATUS(metadata_status, hipErrorOutOfMemory);
  }

  hrx_mem_pool_t pool = NULL;
  hipError_t pool_result =
      iree_hip_current_mem_pool(stream_graph->context, &pool);
  if (pool_result != hipSuccess) {
    iree_status_ignore(iree_hal_streaming_graph_destroy_node(node));
    HIP_RETURN_ERROR(pool_result);
  }
  // This graph-memory implementation allocates backing storage when the graph
  // template is built. Graph exec launch serialization prevents concurrent
  // launches from sharing the same backing allocation.
  hipError_t alloc_result =
      iree_hip_malloc_from_pool(stream_graph->context, pool, params->bytesize,
                                /*stream=*/NULL, &params->dptr);
  hrx_mem_pool_release(pool);
  if (alloc_result != hipSuccess) {
    iree_status_ignore(iree_hal_streaming_graph_destroy_node(node));
    HIP_RETURN_ERROR(alloc_result);
  }

  hipMemAllocNodeParams* stored_params =
      (hipMemAllocNodeParams*)metadata_storage;
  *stored_params = *params;
  if (access_desc_size > 0) {
    hipMemAccessDesc* stored_descs =
        (hipMemAccessDesc*)(metadata_storage + sizeof(*params));
    memcpy(stored_descs, params->accessDescs, access_desc_size);
    stored_params->accessDescs = stored_descs;
  }

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC;
  node->attrs.mem_alloc.params = stored_params;
  node->attrs.mem_alloc.params_size = metadata_size;
  node->attrs.mem_alloc.dptr = params->dptr;
  node->attrs.mem_alloc.bytesize = params->bytesize;
  node->attrs.mem_alloc.owns_device_allocation = true;
  stream_graph->has_graph_memory_nodes = true;
  *pGraphNode = (hipGraphNode_t)node;
  return hipSuccess;
}

static bool iree_hip_graph_has_mem_alloc_node_for_pointer(
    iree_hal_streaming_graph_t* graph, void* dptr) {
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC &&
          node->attrs.mem_alloc.dptr == dptr) {
        return true;
      }
    }
  }
  return false;
}

HIPAPI hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode,
                                         hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, void* dptr) {
  if (!pGraphNode || !graph || !dptr ||
      (numDependencies > 0 && !pDependencies)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraphNode = NULL;

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_buffer_ref_t ref;
  iree_status_t lookup_status = iree_hal_streaming_memory_lookup(
      stream_graph->context, (iree_hal_streaming_deviceptr_t)dptr, &ref);
  if (!iree_status_is_ok(lookup_status)) {
    iree_status_free(lookup_status);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_has_mem_alloc_node_for_pointer(stream_graph, dptr)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_node_t* node = NULL;
  HIP_RETURN_STATUS(
      iree_hal_streaming_graph_add_empty_node(
          stream_graph, (iree_hal_streaming_graph_node_t**)pDependencies,
          numDependencies, &node),
      hipErrorInvalidValue);
  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE;
  node->attrs.mem_free.dptr = dptr;
  stream_graph->has_graph_memory_nodes = true;
  *pGraphNode = (hipGraphNode_t)node;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphAddNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies,
                                  size_t numDependencies,
                                  const void* nodeParams) {
  if (!pGraphNode || !graph || !nodeParams ||
      (numDependencies > 0 && !pDependencies)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipGraphNodeParams* params = (const hipGraphNodeParams*)nodeParams;
  switch (params->type) {
    case hipGraphNodeTypeKernel:
      return hipGraphAddKernelNode(pGraphNode, graph, pDependencies,
                                   numDependencies, &params->kernel);
    case hipGraphNodeTypeMemcpy:
      return hipGraphAddMemcpyNode(pGraphNode, graph, pDependencies,
                                   numDependencies, &params->memcpy.copyParams);
    case hipGraphNodeTypeMemset:
      return hipGraphAddMemsetNode(pGraphNode, graph, pDependencies,
                                   numDependencies, &params->memset);
    case hipGraphNodeTypeHost:
      return hipGraphAddHostNode(pGraphNode, graph, pDependencies,
                                 numDependencies, &params->host);
    case hipGraphNodeTypeGraph:
      return hipGraphAddChildGraphNode(pGraphNode, graph, pDependencies,
                                       numDependencies, params->graph.graph);
    case hipGraphNodeTypeWaitEvent:
      return hipGraphAddEventWaitNode(pGraphNode, graph, pDependencies,
                                      numDependencies, params->eventWait.event);
    case hipGraphNodeTypeEventRecord:
      return hipGraphAddEventRecordNode(pGraphNode, graph, pDependencies,
                                        numDependencies,
                                        params->eventRecord.event);
    case hipGraphNodeTypeMemAlloc:
      return hipGraphAddMemAllocNode(pGraphNode, graph, pDependencies,
                                     numDependencies, (void*)&params->alloc);
    case hipGraphNodeTypeMemFree:
      return hipGraphAddMemFreeNode(pGraphNode, graph, pDependencies,
                                    numDependencies, params->free.dptr);
    default:
      *pGraphNode = NULL;
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
}

static hipError_t iree_hip_graph_validate_batch_mem_op_node_params(
    const hrx_hip_batch_mem_op_node_params_t* params,
    iree_host_size_t* out_param_array_size) {
  if (out_param_array_size) *out_param_array_size = 0;
  if (!params || !params->ctx || params->count == 0 || !params->paramArray ||
      params->flags != 0) {
    return hipErrorInvalidValue;
  }
  iree_host_size_t param_array_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          params->count, sizeof(*params->paramArray), &param_array_size))) {
    return hipErrorInvalidValue;
  }
  if (out_param_array_size) *out_param_array_size = param_array_size;
  return hipSuccess;
}

static hipError_t iree_hip_graph_copy_batch_mem_op_node_params(
    const iree_hal_streaming_graph_batch_mem_op_node_attrs_t* attrs,
    hrx_hip_batch_mem_op_node_params_t* out_params) {
  if (!attrs || !out_params || attrs->params_size != sizeof(*out_params) ||
      !attrs->params) {
    return hipErrorInvalidValue;
  }
  memcpy(out_params, attrs->params, sizeof(*out_params));
  out_params->paramArray =
      (hrx_hip_stream_batch_mem_op_params_t*)attrs->param_array;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t* pGraphNode,
                                            hipGraph_t graph,
                                            const hipGraphNode_t* pDependencies,
                                            size_t numDependencies,
                                            const void* nodeParams) {
  if (!pGraphNode || !graph || !nodeParams ||
      (numDependencies > 0 && !pDependencies)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraphNode = NULL;
  if (!iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

HIPAPI hipError_t hipGraphBatchMemOpNodeGetParams(hipGraphNode_t node,
                                                  void* nodeParams) {
  if (!node || !nodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t result = iree_hip_graph_copy_batch_mem_op_node_params(
      &stream_node->attrs.batch_mem_op,
      (hrx_hip_batch_mem_op_node_params_t*)nodeParams);
  HIP_RETURN_ERROR(result);
}

HIPAPI hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t node,
                                                  const void* nodeParams) {
  if (!node || !nodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hrx_hip_batch_mem_op_node_params_t* params =
      (const hrx_hip_batch_mem_op_node_params_t*)nodeParams;
  iree_host_size_t param_array_size = 0;
  hipError_t validate_result = iree_hip_graph_validate_batch_mem_op_node_params(
      params, &param_array_size);
  if (validate_result != hipSuccess) {
    HIP_RETURN_ERROR(validate_result);
  }
  HIP_RETURN_STATUS(iree_hal_streaming_graph_set_batch_mem_op_node_params(
                        stream_node, params, sizeof(*params),
                        params->paramArray, param_array_size),
                    hipErrorInvalidValue);
  return hipSuccess;
}

HIPAPI hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node,
                                                 hipGraph_t* pGraph) {
  if (!node || !pGraph) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!stream_node->graph ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH ||
      !stream_node->attrs.child_graph.graph) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraph = (hipGraph_t)stream_node->attrs.child_graph.graph;
  return hipSuccess;
}

static bool iree_hip_graph_node_is_active(
    const iree_hal_streaming_graph_node_t* node) {
  return node && node->graph && !iree_hip_graph_node_is_hidden(node);
}

static hipError_t iree_hip_graph_validate_memset_params(
    iree_hal_streaming_context_t* context, const hipMemsetParams* params,
    iree_hal_streaming_buffer_ref_t* out_dst_ref,
    iree_device_size_t* out_element_count) {
  if (out_dst_ref) memset(out_dst_ref, 0, sizeof(*out_dst_ref));
  if (out_element_count) *out_element_count = 0;
  if (!context || !params || !params->dst || params->width == 0 ||
      params->height == 0 ||
      (params->elementSize != 1 && params->elementSize != 2 &&
       params->elementSize != 4)) {
    return hipErrorInvalidValue;
  }

  iree_host_size_t row_bytes = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          params->width, params->elementSize, &row_bytes))) {
    return hipErrorInvalidValue;
  }
  const size_t pitch = params->pitch ? params->pitch : row_bytes;
  if (params->height > 1 && pitch < row_bytes) {
    return hipErrorInvalidValue;
  }

  const iree_host_size_t leading_rows = params->height - 1;
  iree_host_size_t leading_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(leading_rows, pitch, &leading_size) ||
          !iree_host_size_checked_add(leading_size, row_bytes, &total_size) ||
          total_size % params->elementSize != 0)) {
    return hipErrorInvalidValue;
  }
  const iree_host_size_t element_count = total_size / params->elementSize;

  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)params->dst, total_size,
      &dst_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorInvalidValue;
  }
  if (out_dst_ref) *out_dst_ref = dst_ref;
  if (out_element_count) *out_element_count = element_count;
  return hipSuccess;
}

static hipError_t iree_hip_graph_set_memset_node_params(
    iree_hal_streaming_graph_node_t* node, const hipMemsetParams* params) {
  if (!iree_hip_graph_node_is_active(node) ||
      node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET) {
    return hipErrorInvalidValue;
  }
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_device_size_t element_count = 0;
  hipError_t result = iree_hip_graph_validate_memset_params(
      node->graph->context, params, &dst_ref, &element_count);
  if (result != hipSuccess) return result;

  node->attrs.memset.dst_ref = dst_ref;
  node->attrs.memset.pattern = params->value;
  node->attrs.memset.pattern_size = (uint8_t)params->elementSize;
  node->attrs.memset.count = element_count;
  node->attrs.memset.hip_dst = params->dst;
  node->attrs.memset.hip_width = params->width;
  node->attrs.memset.hip_height = params->height;
  node->attrs.memset.hip_pitch = params->pitch;
  return hipSuccess;
}

static hipError_t iree_hip_graph_validate_memcpy_kind(hipMemcpyKind kind) {
  switch (kind) {
    case hipMemcpyHostToHost:
    case hipMemcpyHostToDevice:
    case hipMemcpyDeviceToHost:
    case hipMemcpyDeviceToDevice:
    case hipMemcpyDefault:
    case hipMemcpyDeviceToDeviceNoCU:
      return hipSuccess;
    default:
      return hipErrorInvalidMemcpyDirection;
  }
}

static hipError_t iree_hip_graph_memcpy3d_span_bytes(
    size_t pitch, size_t ysize, size_t width, size_t height, size_t depth,
    iree_host_size_t* out_span) {
  *out_span = 0;
  if (width == 0 || height == 0 || depth == 0) return hipSuccess;
  const size_t rows_per_slice = ysize ? ysize : height;
  if (height > rows_per_slice) return hipErrorInvalidValue;

  iree_host_size_t slice_pitch = 0;
  iree_host_size_t leading_slices = 0;
  iree_host_size_t leading_rows = 0;
  iree_host_size_t leading_bytes = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(pitch, rows_per_slice, &slice_pitch) ||
          !iree_host_size_checked_mul(depth - 1, slice_pitch,
                                      &leading_slices) ||
          !iree_host_size_checked_mul(height - 1, pitch, &leading_rows) ||
          !iree_host_size_checked_add(leading_slices, leading_rows,
                                      &leading_bytes) ||
          !iree_host_size_checked_add(leading_bytes, width, out_span))) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_graph_validate_memcpy3d_params(
    iree_hal_streaming_context_t* context, const hipMemcpy3DParms* params,
    hipMemcpyKind* out_kind, void** out_dst, const void** out_src,
    iree_host_size_t* out_size, iree_host_size_t* out_dst_span,
    iree_host_size_t* out_src_span) {
  if (out_kind) *out_kind = hipMemcpyDefault;
  if (out_dst) *out_dst = NULL;
  if (out_src) *out_src = NULL;
  if (out_size) *out_size = 0;
  if (out_dst_span) *out_dst_span = 0;
  if (out_src_span) *out_src_span = 0;
  if (!context || !params) {
    return hipErrorInvalidValue;
  }
  hipMemcpy3DParms resolved_params;
  hipError_t array_result =
      iree_hip_resolve_memcpy3d_array_params(params, &resolved_params);
  if (array_result != hipSuccess) return array_result;
  params = &resolved_params;
  if (!context || !params->srcPtr.ptr || !params->dstPtr.ptr ||
      params->extent.width == 0 || params->extent.height == 0 ||
      params->extent.depth == 0) {
    return hipErrorInvalidValue;
  }
  if (iree_hip_graph_validate_memcpy_kind(params->kind) != hipSuccess) {
    return hipErrorInvalidMemcpyDirection;
  }
  if (params->dstPtr.pitch < params->extent.width ||
      params->srcPtr.pitch < params->extent.width) {
    return hipErrorInvalidPitchValue;
  }
  int max_pitch = 0;
  if (hipDeviceGetAttribute(&max_pitch, hipDeviceAttributeMaxPitch, 0) ==
          hipSuccess &&
      max_pitch > 0 &&
      (params->dstPtr.pitch >= (size_t)max_pitch ||
       params->srcPtr.pitch >= (size_t)max_pitch)) {
    return hipErrorInvalidValue;
  }
  if (params->dstPos.x > params->dstPtr.pitch ||
      params->srcPos.x > params->srcPtr.pitch ||
      params->extent.width > params->dstPtr.pitch - params->dstPos.x ||
      params->extent.width > params->srcPtr.pitch - params->srcPos.x) {
    return hipErrorInvalidValue;
  }
  if ((params->dstPtr.ysize != 0 &&
       (params->dstPos.y >= params->dstPtr.ysize ||
        params->extent.height > params->dstPtr.ysize - params->dstPos.y)) ||
      (params->srcPtr.ysize != 0 &&
       (params->srcPos.y >= params->srcPtr.ysize ||
        params->extent.height > params->srcPtr.ysize - params->srcPos.y))) {
    return hipErrorInvalidValue;
  }
  if (params->dstPos.z != 0 || params->srcPos.z != 0) {
    return hipErrorInvalidValue;
  }
  iree_host_size_t size_xy = 0;
  iree_host_size_t size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(params->extent.width,
                                      params->extent.height, &size_xy) ||
          !iree_host_size_checked_mul(size_xy, params->extent.depth, &size))) {
    return hipErrorInvalidValue;
  }
  iree_host_size_t src_rows_per_slice = params->srcPtr.ysize;
  iree_host_size_t dst_rows_per_slice = params->dstPtr.ysize;
  if (src_rows_per_slice == 0 &&
      IREE_UNLIKELY(!iree_host_size_checked_add(
          params->srcPos.y, params->extent.height, &src_rows_per_slice))) {
    return hipErrorInvalidValue;
  }
  if (dst_rows_per_slice == 0 &&
      IREE_UNLIKELY(!iree_host_size_checked_add(
          params->dstPos.y, params->extent.height, &dst_rows_per_slice))) {
    return hipErrorInvalidValue;
  }
  iree_host_size_t dst_span = 0;
  hipError_t span_result = iree_hip_graph_memcpy3d_span_bytes(
      params->dstPtr.pitch, dst_rows_per_slice, params->extent.width,
      params->extent.height, params->extent.depth, &dst_span);
  if (span_result != hipSuccess) return span_result;
  iree_host_size_t src_span = 0;
  span_result = iree_hip_graph_memcpy3d_span_bytes(
      params->srcPtr.pitch, src_rows_per_slice, params->extent.width,
      params->extent.height, params->extent.depth, &src_span);
  if (span_result != hipSuccess) return span_result;

  iree_host_size_t src_slice_pitch = 0;
  iree_host_size_t dst_slice_pitch = 0;
  iree_host_size_t src_row_offset = 0;
  iree_host_size_t dst_row_offset = 0;
  iree_host_size_t src_z_offset = 0;
  iree_host_size_t dst_z_offset = 0;
  iree_host_size_t src_base_offset = 0;
  iree_host_size_t dst_base_offset = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(params->srcPtr.pitch, src_rows_per_slice,
                                      &src_slice_pitch) ||
          !iree_host_size_checked_mul(params->dstPtr.pitch, dst_rows_per_slice,
                                      &dst_slice_pitch) ||
          !iree_host_size_checked_mul(params->srcPos.y, params->srcPtr.pitch,
                                      &src_row_offset) ||
          !iree_host_size_checked_mul(params->dstPos.y, params->dstPtr.pitch,
                                      &dst_row_offset) ||
          !iree_host_size_checked_mul(params->srcPos.z, src_slice_pitch,
                                      &src_z_offset) ||
          !iree_host_size_checked_mul(params->dstPos.z, dst_slice_pitch,
                                      &dst_z_offset) ||
          !iree_host_size_checked_add(src_z_offset, src_row_offset,
                                      &src_base_offset) ||
          !iree_host_size_checked_add(dst_z_offset, dst_row_offset,
                                      &dst_base_offset) ||
          !iree_host_size_checked_add(src_base_offset, params->srcPos.x,
                                      &src_base_offset) ||
          !iree_host_size_checked_add(dst_base_offset, params->dstPos.x,
                                      &dst_base_offset))) {
    return hipErrorInvalidValue;
  }

  void* dst = (uint8_t*)params->dstPtr.ptr + dst_base_offset;
  const void* src = (const uint8_t*)params->srcPtr.ptr + src_base_offset;
  hipMemcpyKind kind = params->kind;
  hipError_t result = hipSuccess;
  if (kind == hipMemcpyDefault) {
    iree_hal_streaming_context_t* dst_context = NULL;
    iree_hal_streaming_buffer_ref_t dst_ref = {0};
    iree_status_t dst_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)dst, dst_span, &dst_context,
            &dst_ref);
    iree_hal_streaming_context_t* src_context = NULL;
    iree_hal_streaming_buffer_ref_t src_ref = {0};
    iree_status_t src_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)src, src_span, &src_context,
            &src_ref);
    const bool dst_is_device =
        iree_status_is_ok(dst_status) &&
        (dst_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
    const bool src_is_device =
        iree_status_is_ok(src_status) &&
        (src_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
    if (iree_status_is_ok(dst_status)) {
      iree_hal_streaming_context_release(dst_context);
    } else {
      iree_status_ignore(dst_status);
    }
    if (iree_status_is_ok(src_status)) {
      iree_hal_streaming_context_release(src_context);
    } else {
      iree_status_ignore(src_status);
    }
    if (dst_is_device && src_is_device) {
      kind = hipMemcpyDeviceToDevice;
    } else {
      result = iree_hip_resolve_memcpy_kind(context, dst, src, kind, &kind);
    }
  } else {
    result = iree_hip_resolve_memcpy_kind(context, dst, src, kind, &kind);
  }
  if (result != hipSuccess) {
    return params->kind == hipMemcpyDefault ? result
                                            : hipErrorInvalidMemcpyDirection;
  }
  switch (kind) {
    case hipMemcpyHostToDevice:
      result =
          iree_hip_validate_known_memcpy_range(context, dst, dst_span, true);
      break;
    case hipMemcpyDeviceToHost:
      result =
          iree_hip_validate_known_memcpy_range(context, src, src_span, true);
      break;
    case hipMemcpyDeviceToDevice:
      result =
          iree_hip_validate_known_memcpy_range(context, dst, dst_span, true);
      if (result == hipSuccess) {
        result =
            iree_hip_validate_known_memcpy_range(context, src, src_span, true);
      }
      if (result != hipSuccess) {
        iree_hal_streaming_context_t* dst_context = NULL;
        iree_hal_streaming_buffer_ref_t dst_ref = {0};
        iree_status_t dst_status =
            iree_hal_streaming_memory_lookup_range_across_contexts(
                (iree_hal_streaming_deviceptr_t)dst, dst_span, &dst_context,
                &dst_ref);
        iree_hal_streaming_context_t* src_context = NULL;
        iree_hal_streaming_buffer_ref_t src_ref = {0};
        iree_status_t src_status =
            iree_hal_streaming_memory_lookup_range_across_contexts(
                (iree_hal_streaming_deviceptr_t)src, src_span, &src_context,
                &src_ref);
        const bool valid =
            iree_status_is_ok(dst_status) && iree_status_is_ok(src_status) &&
            (dst_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
            (src_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
        if (iree_status_is_ok(dst_status)) {
          iree_hal_streaming_context_release(dst_context);
        } else {
          iree_status_ignore(dst_status);
        }
        if (iree_status_is_ok(src_status)) {
          iree_hal_streaming_context_release(src_context);
        } else {
          iree_status_ignore(src_status);
        }
        result = valid ? hipSuccess : hipErrorInvalidValue;
      }
      break;
    case hipMemcpyHostToHost:
      result = hipSuccess;
      break;
    default:
      result = hipErrorInvalidMemcpyDirection;
      break;
  }
  if (result != hipSuccess) return result;

  if (out_kind) *out_kind = kind;
  if (out_dst) *out_dst = dst;
  if (out_src) *out_src = src;
  if (out_size) *out_size = size;
  if (out_dst_span) *out_dst_span = dst_span;
  if (out_src_span) *out_src_span = src_span;
  return hipSuccess;
}

typedef struct iree_hip_graph_memcpy_node_update_t {
  iree_hal_streaming_graph_memcpy_node_attrs_t old_memcpy_attrs;
  iree_hal_streaming_graph_node_t* pre_callback_node;
  iree_hip_graph_memcpy_callback_data_t old_pre_callback_data;
  iree_hal_streaming_graph_node_t* post_callback_node;
  iree_hip_graph_memcpy_callback_data_t old_post_callback_data;
} iree_hip_graph_memcpy_node_update_t;

static bool iree_hip_ranges_overlap(uintptr_t lhs_start, uintptr_t rhs_start,
                                    size_t count) {
  if (count == 0) return false;
  if (lhs_start > UINTPTR_MAX - count || rhs_start > UINTPTR_MAX - count) {
    return true;
  }
  const uintptr_t lhs_end = lhs_start + count;
  const uintptr_t rhs_end = rhs_start + count;
  return lhs_start < rhs_end && rhs_start < lhs_end;
}

static hipError_t iree_hip_graph_validate_1d_memcpy_update(
    hipMemcpyKind kind, void* dst, const void* src, iree_host_size_t size) {
  if (kind == hipMemcpyDeviceToDevice &&
      iree_hip_ranges_overlap((uintptr_t)dst, (uintptr_t)src, size)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_graph_memcpy_slice_pitch(size_t pitch,
                                                    size_t rows_per_slice,
                                                    iree_host_size_t* out) {
  if (!iree_host_size_checked_mul(pitch, rows_per_slice, out)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_graph_memcpy_rows_per_slice(size_t position_y,
                                                       size_t height,
                                                       size_t ysize,
                                                       iree_host_size_t* out) {
  *out = ysize;
  if (*out == 0 && !iree_host_size_checked_add(position_y, height, out)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_graph_configure_memcpy_callback_data(
    iree_hip_graph_memcpy_callback_data_t* callback_data,
    const hipMemcpy3DParms* params, void* dst, const void* src,
    iree_host_size_t size, size_t dst_pitch, size_t src_pitch,
    size_t dst_rows_per_slice, size_t src_rows_per_slice) {
  iree_host_size_t dst_slice_pitch = 0;
  iree_host_size_t src_slice_pitch = 0;
  hipError_t result = iree_hip_graph_memcpy_slice_pitch(
      dst_pitch, dst_rows_per_slice, &dst_slice_pitch);
  if (result != hipSuccess) return result;
  result = iree_hip_graph_memcpy_slice_pitch(src_pitch, src_rows_per_slice,
                                             &src_slice_pitch);
  if (result != hipSuccess) return result;

  memset(callback_data, 0, sizeof(*callback_data));
  callback_data->dst = dst;
  callback_data->src = src;
  callback_data->count = size;
  callback_data->width = params->extent.width;
  callback_data->height = params->extent.height;
  callback_data->depth = params->extent.depth;
  callback_data->dst_pitch = dst_pitch;
  callback_data->src_pitch = src_pitch;
  callback_data->dst_slice_pitch = dst_slice_pitch;
  callback_data->src_slice_pitch = src_slice_pitch;
  callback_data->hip_params = *params;
  callback_data->hip_params.kind =
      iree_hip_graph_metadata_memcpy_kind(params->kind);
  return hipSuccess;
}

static void iree_hip_graph_disable_memcpy_callback(
    iree_hal_streaming_graph_node_t* callback_node) {
  if (!callback_node || callback_node->attrs.host.user_data_size <
                            sizeof(iree_hip_graph_memcpy_callback_data_t)) {
    return;
  }
  iree_hip_graph_memcpy_callback_data_t* callback_data =
      (iree_hip_graph_memcpy_callback_data_t*)
          callback_node->attrs.host.user_data;
  memset(callback_data, 0, sizeof(*callback_data));
}

static bool iree_hip_graph_node_is_hidden_memcpy_callback(
    const iree_hal_streaming_graph_node_t* node) {
  return node && node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL &&
         (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) != 0 &&
         node->attrs.host.fn == iree_hip_graph_memcpy_callback &&
         node->attrs.host.user_data;
}

static iree_hal_streaming_graph_node_t* iree_hip_graph_find_pre_memcpy_callback(
    iree_hal_streaming_graph_node_t* node) {
  for (uint32_t i = 0; i < node->dependency_count; ++i) {
    iree_hal_streaming_graph_node_t* dependency = node->dependencies[i];
    if (iree_hip_graph_node_is_hidden_memcpy_callback(dependency)) {
      return dependency;
    }
  }
  return NULL;
}

static iree_hal_streaming_graph_node_t*
iree_hip_graph_find_post_memcpy_callback(
    iree_hal_streaming_graph_node_t* node) {
  iree_hal_streaming_graph_t* graph = node->graph;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* candidate = block->nodes[i];
      if (!iree_hip_graph_node_is_hidden_memcpy_callback(candidate)) continue;
      for (uint32_t j = 0; j < candidate->dependency_count; ++j) {
        if (candidate->dependencies[j] == node) return candidate;
      }
    }
  }
  return NULL;
}

static hipError_t iree_hip_graph_add_hidden_memcpy_callback(
    iree_hal_streaming_graph_node_t* node, bool after_memcpy,
    iree_hal_streaming_graph_node_t** out_callback_node) {
  *out_callback_node = NULL;
  iree_hip_graph_memcpy_callback_data_t* callback_data = NULL;
  iree_status_t status = iree_arena_allocate(
      &node->graph->arena, sizeof(*callback_data), (void**)&callback_data);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  memset(callback_data, 0, sizeof(*callback_data));

  iree_hal_streaming_graph_node_t* callback_node = NULL;
  if (after_memcpy) {
    iree_hal_streaming_graph_node_t* dependencies[] = {node};
    status = iree_hal_streaming_graph_add_host_call_node(
        node->graph, dependencies, IREE_ARRAYSIZE(dependencies),
        iree_hip_graph_memcpy_callback, callback_data, &callback_node);
  } else {
    status = iree_hal_streaming_graph_add_host_call_node(
        node->graph, NULL, 0, iree_hip_graph_memcpy_callback, callback_data,
        &callback_node);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorInvalidValue;
  }
  callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
  callback_node->attrs.host.user_data_size = sizeof(*callback_data);
  if (!after_memcpy) {
    status = iree_hal_streaming_graph_add_dependencies(
        node->graph, &callback_node, &node, 1);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorInvalidValue;
    }
  }
  *out_callback_node = callback_node;
  return hipSuccess;
}

static hipError_t iree_hip_graph_update_memcpy_metadata(
    iree_hal_streaming_graph_node_t* node, const hipMemcpy3DParms* params,
    hipMemcpyKind kind, void* dst, const void* src, iree_host_size_t size,
    iree_host_size_t dst_span, iree_host_size_t src_span) {
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      node->graph->context, (iree_hal_streaming_deviceptr_t)dst, dst_span,
      &dst_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorInvalidValue;
  }
  iree_hal_streaming_buffer_ref_t src_ref;
  status = iree_hal_streaming_memory_lookup_range(
      node->graph->context, (iree_hal_streaming_deviceptr_t)src, src_span,
      &src_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorInvalidValue;
  }

  node->attrs.memcpy.dst_ref = dst_ref;
  node->attrs.memcpy.src_ref = src_ref;
  node->attrs.memcpy.size = size;
  node->attrs.memcpy.execution_dst_pitch = params->dstPtr.pitch;
  node->attrs.memcpy.execution_src_pitch = params->srcPtr.pitch;
  node->attrs.memcpy.execution_dst_ysize = params->dstPtr.ysize;
  node->attrs.memcpy.execution_src_ysize = params->srcPtr.ysize;
  node->attrs.memcpy.execution_extent_width = params->extent.width;
  node->attrs.memcpy.execution_extent_height = params->extent.height;
  node->attrs.memcpy.execution_extent_depth = params->extent.depth;
  iree_hip_graph_store_memcpy_query_metadata(node, params);
  node->attrs.memcpy.hip_driver.valid = false;
  return hipSuccess;
}

static void iree_hip_graph_rollback_memcpy_node_update(
    iree_hal_streaming_graph_node_t* node,
    const iree_hip_graph_memcpy_node_update_t* update) {
  node->attrs.memcpy = update->old_memcpy_attrs;
  if (update->pre_callback_node) {
    iree_hip_graph_memcpy_callback_data_t* callback_data =
        (iree_hip_graph_memcpy_callback_data_t*)
            update->pre_callback_node->attrs.host.user_data;
    iree_hip_graph_release_memcpy_callback_context_refs(node->graph,
                                                        callback_data);
    *callback_data = update->old_pre_callback_data;
  }
  if (update->post_callback_node) {
    iree_hip_graph_memcpy_callback_data_t* callback_data =
        (iree_hip_graph_memcpy_callback_data_t*)
            update->post_callback_node->attrs.host.user_data;
    iree_hip_graph_release_memcpy_callback_context_refs(node->graph,
                                                        callback_data);
    *callback_data = update->old_post_callback_data;
  }
}

static void iree_hip_graph_commit_memcpy_node_update(
    iree_hal_streaming_graph_node_t* node,
    const iree_hip_graph_memcpy_node_update_t* update) {
  if (update->pre_callback_node) {
    iree_hip_graph_release_memcpy_callback_context_refs(
        node->graph, &update->old_pre_callback_data);
  }
  if (update->post_callback_node) {
    iree_hip_graph_release_memcpy_callback_context_refs(
        node->graph, &update->old_post_callback_data);
  }
}

static hipError_t iree_hip_graph_set_host_memcpy_callback_params(
    iree_hal_streaming_graph_node_t* node, const hipMemcpy3DParms* params,
    hipMemcpyKind kind, void* dst, const void* src, iree_host_size_t size,
    iree_hip_graph_memcpy_callback_data_t* out_old_callback_data);

static hipError_t iree_hip_graph_set_same_context_device_memcpy_node_params(
    iree_hal_streaming_graph_node_t* node,
    const hipMemcpy3DParms* execution_params, void* dst, const void* src,
    iree_host_size_t size, iree_host_size_t dst_span, iree_host_size_t src_span,
    iree_hal_streaming_graph_node_t* pre_callback,
    iree_hal_streaming_graph_node_t* post_callback,
    iree_hip_graph_memcpy_node_update_t* update) {
  if (pre_callback) {
    update->pre_callback_node = pre_callback;
    update->old_pre_callback_data = *(iree_hip_graph_memcpy_callback_data_t*)
                                         pre_callback->attrs.host.user_data;
    iree_hip_graph_disable_memcpy_callback(pre_callback);
  }
  if (post_callback) {
    update->post_callback_node = post_callback;
    update->old_post_callback_data = *(iree_hip_graph_memcpy_callback_data_t*)
                                          post_callback->attrs.host.user_data;
    iree_hip_graph_disable_memcpy_callback(post_callback);
  }
  hipError_t result = iree_hip_graph_update_memcpy_metadata(
      node, execution_params, hipMemcpyDeviceToDevice, dst, src, size, dst_span,
      src_span);
  if (result != hipSuccess) {
    iree_hip_graph_rollback_memcpy_node_update(node, update);
  }
  return result;
}

static hipError_t iree_hip_graph_set_remote_destination_memcpy_node_params(
    iree_hal_streaming_graph_node_t* node,
    const hipMemcpy3DParms* execution_params, void* dst, const void* src,
    iree_host_size_t size, iree_host_size_t src_span,
    iree_hal_streaming_context_t* dst_context,
    iree_hal_streaming_graph_node_t* pre_callback,
    iree_hal_streaming_graph_node_t* post_callback, bool allow_hidden_callbacks,
    iree_host_size_t dst_rows_per_slice, iree_host_size_t src_rows_per_slice,
    iree_hip_graph_memcpy_node_update_t* update) {
  if (!post_callback) {
    if (!allow_hidden_callbacks) return hipErrorInvalidValue;
    hipError_t result =
        iree_hip_graph_add_hidden_memcpy_callback(node, true, &post_callback);
    if (result != hipSuccess) return result;
  }
  update->post_callback_node = post_callback;
  iree_hip_graph_memcpy_callback_data_t* callback_data =
      (iree_hip_graph_memcpy_callback_data_t*)
          post_callback->attrs.host.user_data;
  update->old_post_callback_data = *callback_data;
  if (pre_callback) {
    update->pre_callback_node = pre_callback;
    update->old_pre_callback_data = *(iree_hip_graph_memcpy_callback_data_t*)
                                         pre_callback->attrs.host.user_data;
    iree_hip_graph_disable_memcpy_callback(pre_callback);
  }

  iree_hal_streaming_buffer_t* staging = NULL;
  iree_status_t status = iree_hal_streaming_graph_allocate_host_staging(
      node->graph, size, &staging);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  hipError_t result = iree_hip_graph_configure_memcpy_callback_data(
      callback_data, execution_params, dst, staging->host_ptr, size,
      execution_params->dstPtr.pitch, execution_params->extent.width,
      dst_rows_per_slice, execution_params->extent.height);
  if (result != hipSuccess) return result;
  callback_data->dst_context = dst_context;
  result = iree_hip_graph_retain_remote_context(node->graph, dst_context);
  if (result != hipSuccess) {
    callback_data->dst_context = NULL;
    iree_hip_graph_rollback_memcpy_node_update(node, update);
    return result;
  }
  result = iree_hip_graph_update_memcpy_metadata(
      node, execution_params, hipMemcpyDeviceToDevice,
      (void*)staging->device_ptr, src, size, size, src_span);
  if (result != hipSuccess) {
    iree_hip_graph_rollback_memcpy_node_update(node, update);
    return result;
  }
  iree_hip_graph_store_memcpy_execution_geometry(
      node, execution_params->extent.width, execution_params->srcPtr.pitch,
      execution_params->extent.height, src_rows_per_slice,
      execution_params->extent.width, execution_params->extent.height,
      execution_params->extent.depth);
  return hipSuccess;
}

static hipError_t iree_hip_graph_set_remote_source_memcpy_node_params(
    iree_hal_streaming_graph_node_t* node,
    const hipMemcpy3DParms* execution_params, void* dst, const void* src,
    iree_host_size_t size, iree_host_size_t dst_span,
    iree_hal_streaming_context_t* src_context,
    iree_hal_streaming_graph_node_t* pre_callback,
    iree_hal_streaming_graph_node_t* post_callback, bool allow_hidden_callbacks,
    iree_host_size_t dst_rows_per_slice, iree_host_size_t src_rows_per_slice,
    iree_hip_graph_memcpy_node_update_t* update) {
  if (!pre_callback) {
    if (!allow_hidden_callbacks) return hipErrorInvalidValue;
    hipError_t result =
        iree_hip_graph_add_hidden_memcpy_callback(node, false, &pre_callback);
    if (result != hipSuccess) return result;
  }
  update->pre_callback_node = pre_callback;
  iree_hip_graph_memcpy_callback_data_t* callback_data =
      (iree_hip_graph_memcpy_callback_data_t*)
          pre_callback->attrs.host.user_data;
  update->old_pre_callback_data = *callback_data;
  if (post_callback) {
    update->post_callback_node = post_callback;
    update->old_post_callback_data = *(iree_hip_graph_memcpy_callback_data_t*)
                                          post_callback->attrs.host.user_data;
    iree_hip_graph_disable_memcpy_callback(post_callback);
  }

  iree_hal_streaming_buffer_t* staging = NULL;
  iree_status_t status = iree_hal_streaming_graph_allocate_host_staging(
      node->graph, size, &staging);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  hipError_t result = iree_hip_graph_configure_memcpy_callback_data(
      callback_data, execution_params, staging->host_ptr, src, size,
      execution_params->extent.width, execution_params->srcPtr.pitch,
      execution_params->extent.height, src_rows_per_slice);
  if (result != hipSuccess) return result;
  callback_data->src_context = src_context;
  result = iree_hip_graph_retain_remote_context(node->graph, src_context);
  if (result != hipSuccess) {
    callback_data->src_context = NULL;
    iree_hip_graph_rollback_memcpy_node_update(node, update);
    return result;
  }
  result = iree_hip_graph_update_memcpy_metadata(
      node, execution_params, hipMemcpyDeviceToDevice, dst,
      (const void*)staging->device_ptr, size, dst_span, size);
  if (result != hipSuccess) {
    iree_hip_graph_rollback_memcpy_node_update(node, update);
    return result;
  }
  iree_hip_graph_store_memcpy_execution_geometry(
      node, execution_params->dstPtr.pitch, execution_params->extent.width,
      dst_rows_per_slice, execution_params->extent.height,
      execution_params->extent.width, execution_params->extent.height,
      execution_params->extent.depth);
  return hipSuccess;
}

static hipError_t iree_hip_graph_set_memcpy_node_params(
    iree_hal_streaming_graph_node_t* node, const hipMemcpy3DParms* params,
    iree_hip_graph_memcpy_node_update_t* out_update) {
  if (out_update) memset(out_update, 0, sizeof(*out_update));
  if (!iree_hip_graph_node_is_active(node) ||
      node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
    return hipErrorInvalidValue;
  }
  hipMemcpyKind kind = hipMemcpyDefault;
  void* dst = NULL;
  const void* src = NULL;
  iree_host_size_t size = 0;
  iree_host_size_t dst_span = 0;
  iree_host_size_t src_span = 0;
  hipError_t result = iree_hip_graph_validate_memcpy3d_params(
      node->graph->context, params, &kind, &dst, &src, &size, &dst_span,
      &src_span);
  if (result != hipSuccess) return result;
  hipMemcpy3DParms execution_params;
  result = iree_hip_resolve_memcpy3d_array_params(params, &execution_params);
  if (result != hipSuccess) return result;
  result = iree_hip_graph_validate_1d_memcpy_update(kind, dst, src, size);
  if (result != hipSuccess) return result;

  iree_hip_graph_memcpy_node_update_t update = {
      .old_memcpy_attrs = node->attrs.memcpy,
  };
  iree_hal_streaming_graph_node_t* pre_callback =
      iree_hip_graph_find_pre_memcpy_callback(node);
  iree_hal_streaming_graph_node_t* post_callback =
      iree_hip_graph_find_post_memcpy_callback(node);

  if (kind == hipMemcpyDeviceToDevice) {
    iree_hal_streaming_context_t* dst_context = NULL;
    iree_hal_streaming_buffer_ref_t dst_ref = {0};
    iree_status_t dst_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)dst, dst_span, &dst_context,
            &dst_ref);
    iree_hal_streaming_context_t* src_context = NULL;
    iree_hal_streaming_buffer_ref_t src_ref = {0};
    iree_status_t src_status =
        iree_hal_streaming_memory_lookup_range_across_contexts(
            (iree_hal_streaming_deviceptr_t)src, src_span, &src_context,
            &src_ref);
    const bool have_dst = iree_status_is_ok(dst_status);
    const bool have_src = iree_status_is_ok(src_status);
    const bool valid =
        have_dst && have_src &&
        (dst_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
        (src_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
    iree_host_size_t dst_rows_per_slice = 0;
    iree_host_size_t src_rows_per_slice = 0;
    if (!valid) {
      result = hipErrorInvalidValue;
    } else {
      result = iree_hip_graph_memcpy_rows_per_slice(
          execution_params.srcPos.y, execution_params.extent.height,
          execution_params.srcPtr.ysize, &src_rows_per_slice);
    }
    if (result == hipSuccess) {
      result = iree_hip_graph_memcpy_rows_per_slice(
          execution_params.dstPos.y, execution_params.extent.height,
          execution_params.dstPtr.ysize, &dst_rows_per_slice);
    }

    if (result == hipSuccess) {
      const bool allow_hidden_callbacks = out_update == NULL;
      if (dst_context == node->graph->context &&
          src_context == node->graph->context) {
        result = iree_hip_graph_set_same_context_device_memcpy_node_params(
            node, &execution_params, dst, src, size, dst_span, src_span,
            pre_callback, post_callback, &update);
      } else if (src_context == node->graph->context) {
        result = iree_hip_graph_set_remote_destination_memcpy_node_params(
            node, &execution_params, dst, src, size, src_span, dst_context,
            pre_callback, post_callback, allow_hidden_callbacks,
            dst_rows_per_slice, src_rows_per_slice, &update);
      } else if (dst_context == node->graph->context) {
        result = iree_hip_graph_set_remote_source_memcpy_node_params(
            node, &execution_params, dst, src, size, dst_span, src_context,
            pre_callback, post_callback, allow_hidden_callbacks,
            dst_rows_per_slice, src_rows_per_slice, &update);
      } else {
        result = hipErrorInvalidValue;
      }
    }

    if (have_dst) {
      iree_hal_streaming_context_release(dst_context);
    }
    iree_status_ignore(dst_status);
    if (have_src) {
      iree_hal_streaming_context_release(src_context);
    }
    iree_status_ignore(src_status);
    if (result != hipSuccess) return result;
  } else if (kind == hipMemcpyHostToDevice) {
    if (!pre_callback) {
      if (out_update) return hipErrorInvalidValue;
      result =
          iree_hip_graph_add_hidden_memcpy_callback(node, false, &pre_callback);
      if (result != hipSuccess) return result;
    }
    update.pre_callback_node = pre_callback;
    iree_hip_graph_memcpy_callback_data_t* callback_data =
        (iree_hip_graph_memcpy_callback_data_t*)
            pre_callback->attrs.host.user_data;
    update.old_pre_callback_data = *callback_data;
    if (post_callback) {
      update.post_callback_node = post_callback;
      update.old_post_callback_data = *(iree_hip_graph_memcpy_callback_data_t*)
                                           post_callback->attrs.host.user_data;
      iree_hip_graph_disable_memcpy_callback(post_callback);
    }

    iree_hal_streaming_buffer_t* staging = NULL;
    iree_status_t status = iree_hal_streaming_graph_allocate_host_staging(
        node->graph, size, &staging);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorOutOfMemory;
    }
    iree_host_size_t src_rows_per_slice = 0;
    result = iree_hip_graph_memcpy_rows_per_slice(
        execution_params.srcPos.y, execution_params.extent.height,
        execution_params.srcPtr.ysize, &src_rows_per_slice);
    if (result != hipSuccess) return result;
    result = iree_hip_graph_configure_memcpy_callback_data(
        callback_data, &execution_params, staging->host_ptr, src, size,
        execution_params.extent.width, execution_params.srcPtr.pitch,
        execution_params.extent.height, src_rows_per_slice);
    if (result != hipSuccess) return result;
    callback_data->hip_params = *params;
    callback_data->hip_params.kind =
        iree_hip_graph_metadata_memcpy_kind(params->kind);

    result = iree_hip_graph_update_memcpy_metadata(
        node, &execution_params, kind, dst, (const void*)staging->device_ptr,
        size, dst_span, size);
    if (result != hipSuccess) {
      iree_hip_graph_rollback_memcpy_node_update(node, &update);
      return result;
    }
    iree_host_size_t dst_rows_per_slice = 0;
    result = iree_hip_graph_memcpy_rows_per_slice(
        execution_params.dstPos.y, execution_params.extent.height,
        execution_params.dstPtr.ysize, &dst_rows_per_slice);
    if (result != hipSuccess) {
      iree_hip_graph_rollback_memcpy_node_update(node, &update);
      return result;
    }
    iree_hip_graph_store_memcpy_execution_geometry(
        node, execution_params.dstPtr.pitch, execution_params.extent.width,
        dst_rows_per_slice, execution_params.extent.height,
        execution_params.extent.width, execution_params.extent.height,
        execution_params.extent.depth);
  } else if (kind == hipMemcpyDeviceToHost) {
    if (!post_callback) {
      if (out_update) return hipErrorInvalidValue;
      result =
          iree_hip_graph_add_hidden_memcpy_callback(node, true, &post_callback);
      if (result != hipSuccess) return result;
    }
    update.post_callback_node = post_callback;
    iree_hip_graph_memcpy_callback_data_t* callback_data =
        (iree_hip_graph_memcpy_callback_data_t*)
            post_callback->attrs.host.user_data;
    update.old_post_callback_data = *callback_data;
    if (pre_callback) {
      update.pre_callback_node = pre_callback;
      update.old_pre_callback_data = *(iree_hip_graph_memcpy_callback_data_t*)
                                          pre_callback->attrs.host.user_data;
      iree_hip_graph_disable_memcpy_callback(pre_callback);
    }

    iree_hal_streaming_buffer_t* staging = NULL;
    iree_status_t status = iree_hal_streaming_graph_allocate_host_staging(
        node->graph, size, &staging);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorOutOfMemory;
    }
    iree_host_size_t dst_rows_per_slice = 0;
    result = iree_hip_graph_memcpy_rows_per_slice(
        execution_params.dstPos.y, execution_params.extent.height,
        execution_params.dstPtr.ysize, &dst_rows_per_slice);
    if (result != hipSuccess) return result;
    result = iree_hip_graph_configure_memcpy_callback_data(
        callback_data, &execution_params, dst, staging->host_ptr, size,
        execution_params.dstPtr.pitch, execution_params.extent.width,
        dst_rows_per_slice, execution_params.extent.height);
    if (result != hipSuccess) return result;
    callback_data->hip_params = *params;
    callback_data->hip_params.kind =
        iree_hip_graph_metadata_memcpy_kind(params->kind);

    result = iree_hip_graph_update_memcpy_metadata(
        node, &execution_params, kind, (void*)staging->device_ptr, src, size,
        size, src_span);
    if (result != hipSuccess) {
      iree_hip_graph_rollback_memcpy_node_update(node, &update);
      return result;
    }
    iree_host_size_t src_rows_per_slice = 0;
    result = iree_hip_graph_memcpy_rows_per_slice(
        execution_params.srcPos.y, execution_params.extent.height,
        execution_params.srcPtr.ysize, &src_rows_per_slice);
    if (result != hipSuccess) {
      iree_hip_graph_rollback_memcpy_node_update(node, &update);
      return result;
    }
    iree_hip_graph_store_memcpy_execution_geometry(
        node, execution_params.extent.width, execution_params.srcPtr.pitch,
        execution_params.extent.height, src_rows_per_slice,
        execution_params.extent.width, execution_params.extent.height,
        execution_params.extent.depth);
  } else {
    return hipErrorInvalidValue;
  }

  iree_hip_graph_store_memcpy_query_metadata(node, params);
  node->attrs.memcpy.hip_driver.valid = false;
  if (out_update) {
    *out_update = update;
  } else {
    iree_hip_graph_commit_memcpy_node_update(node, &update);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,
                                                  hipEvent_t* event_out) {
  if (!node || !event_out) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD ||
      !stream_node->attrs.event.event) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *event_out = (hipEvent_t)stream_node->attrs.event.event;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node,
                                                  hipEvent_t event) {
  if (!node || !event) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  iree_hal_streaming_event_t* streaming_event =
      (iree_hal_streaming_event_t*)event;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD ||
      streaming_event->context != stream_node->graph->context) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_event_retain(streaming_event);
  iree_hal_streaming_event_release(stream_node->attrs.event.event);
  stream_node->attrs.event.event = streaming_event;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node,
                                                hipEvent_t* event_out) {
  if (!node || !event_out) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT ||
      !stream_node->attrs.event.event) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *event_out = (hipEvent_t)stream_node->attrs.event.event;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node,
                                                hipEvent_t event) {
  if (!node || !event) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  iree_hal_streaming_event_t* streaming_event =
      (iree_hal_streaming_event_t*)event;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT ||
      streaming_event->context != stream_node->graph->context) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_event_retain(streaming_event);
  iree_hal_streaming_event_release(stream_node->attrs.event.event);
  stream_node->attrs.event.event = streaming_event;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t graphExec,
                                                      hipGraphNode_t node,
                                                      const void* nodeParams) {
  if (!graphExec || !node || !nodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t resolve_result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (resolve_result != hipSuccess) {
    HIP_RETURN_ERROR(resolve_result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hrx_hip_batch_mem_op_node_params_t* params =
      (const hrx_hip_batch_mem_op_node_params_t*)nodeParams;
  iree_host_size_t param_array_size = 0;
  hipError_t validate_result = iree_hip_graph_validate_batch_mem_op_node_params(
      params, &param_array_size);
  if (validate_result != hipSuccess) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(validate_result);
  }
  iree_status_t status = iree_hal_streaming_graph_set_batch_mem_op_node_params(
      stream_node, params, sizeof(*params), params->paramArray,
      param_array_size);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_STATUS(status, hipErrorInvalidValue);
  }
  hipError_t rebuild_result = iree_hip_graph_exec_rebuild(exec);
  iree_hal_streaming_graph_exec_release(exec);
  return rebuild_result;
}

HIPAPI hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t graphExec,
                                                      hipGraphNode_t node,
                                                      hipEvent_t event) {
  if (!graphExec || !node || !event) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_status_t status = iree_hal_streaming_graph_exec_set_event_node_event(
      exec, stream_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD,
      (iree_hal_streaming_event_t*)event);
  iree_hal_streaming_graph_exec_release(exec);
  HIP_RETURN_STATUS(status, hipErrorInvalidValue);
  return hipSuccess;
}

HIPAPI hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t graphExec,
                                                    hipGraphNode_t node,
                                                    hipEvent_t event) {
  if (!graphExec || !node || !event) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_status_t status = iree_hal_streaming_graph_exec_set_event_node_event(
      exec, stream_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT,
      (iree_hal_streaming_event_t*)event);
  iree_hal_streaming_graph_exec_release(exec);
  HIP_RETURN_STATUS(status, hipErrorInvalidValue);
  return hipSuccess;
}

HIPAPI hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec,
                                       unsigned long long* flags) {
  if (!graphExec || !flags) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  *flags = (unsigned long long)iree_hal_streaming_graph_exec_flags(exec);
  iree_hal_streaming_graph_exec_release(exec);
  return hipSuccess;
}

static hipError_t iree_hip_graph_exec_rebuild(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_status_t status =
      iree_hal_streaming_graph_exec_rebuild_from_template(exec);
  if (!iree_status_is_ok(status)) {
    HIP_RETURN_STATUS(status, hipErrorInvalidValue);
  }
  return hipSuccess;
}

static bool iree_hip_graph_node_supports_enable(
    const iree_hal_streaming_graph_node_t* node) {
  switch (node->type) {
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL:
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY:
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET:
      return true;
    default:
      return false;
  }
}

HIPAPI hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec,
                                         hipGraphNode_t hNode,
                                         unsigned int* isEnabled) {
  if (!hGraphExec || !hNode || !isEnabled) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(hGraphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)hNode;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hip_graph_handle_is_live((hipGraph_t)stream_node->graph) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node) ||
      !iree_hip_graph_node_supports_enable(stream_node)) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *isEnabled =
      iree_hal_streaming_graph_exec_node_is_enabled(exec, stream_node) ? 1 : 0;
  iree_hal_streaming_graph_exec_release(exec);
  return hipSuccess;
}

HIPAPI hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec,
                                         hipGraphNode_t hNode,
                                         unsigned int isEnabled) {
  if (!hGraphExec || !hNode) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(hGraphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)hNode;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hip_graph_handle_is_live((hipGraph_t)stream_node->graph) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node) ||
      !iree_hip_graph_node_supports_enable(stream_node)) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_status_t status = iree_hal_streaming_graph_exec_set_node_enabled(
      exec, stream_node, isEnabled != 0);
  iree_hal_streaming_graph_exec_release(exec);
  HIP_RETURN_STATUS(status, hipErrorInvalidValue);
  return hipSuccess;
}

HIPAPI hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_stream_t* stream_obj = NULL;
  result = iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (result != hipSuccess) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(result);
  }
  (void)exec;
  iree_hal_streaming_stream_release(stream_obj);
  iree_hal_streaming_graph_exec_release(exec);
  return hipSuccess;
}

HIPAPI hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t graphExec,
                                                hipGraphNode_t node,
                                                const void* pNodeParams) {
  if (!graphExec || !node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  const hipHostNodeParams* params = (const hipHostNodeParams*)pNodeParams;
  if (!params->fn) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_host_call_node_attrs_t old_attrs =
      stream_node->attrs.host;
  stream_node->attrs.host.fn = (void (*)(void*))params->fn;
  stream_node->attrs.host.user_data = params->userData;
  result = iree_hip_graph_exec_rebuild(exec);
  stream_node->attrs.host = old_attrs;
  iree_hal_streaming_graph_exec_release(exec);
  return result;
}

HIPAPI hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t graphExec,
                                                  hipGraphNode_t node,
                                                  const void* pNodeParams) {
  if (!graphExec || !node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node)) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipMemcpy3DParms* params = (const hipMemcpy3DParms*)pNodeParams;
  hipMemcpyKind kind = hipMemcpyDefault;
  result = iree_hip_graph_validate_memcpy3d_params(
      stream_node->graph->context, params, &kind, NULL, NULL, NULL, NULL, NULL);
  if (result != hipSuccess) {
    if (result == hipErrorInvalidPitchValue) {
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(result);
  }
  if (iree_hip_graph_node_is_visible_memcpy_host_call(stream_node)) {
    void* dst = NULL;
    const void* src = NULL;
    iree_host_size_t size = 0;
    result = iree_hip_graph_validate_memcpy3d_params(
        stream_node->graph->context, params, &kind, &dst, &src, &size, NULL,
        NULL);
    if (result != hipSuccess) {
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(result);
    }
    iree_hip_graph_memcpy_callback_data_t* callback_data =
        (iree_hip_graph_memcpy_callback_data_t*)
            stream_node->attrs.host.user_data;
    const hipMemcpyKind requested_kind =
        iree_hip_graph_metadata_memcpy_kind(params->kind);
    if (requested_kind != callback_data->hip_params.kind) {
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    iree_hip_graph_memcpy_callback_data_t old_callback_data = *callback_data;
    result = iree_hip_graph_set_host_memcpy_callback_params(
        stream_node, params, kind, dst, src, size, &old_callback_data);
    if (result != hipSuccess) {
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(result);
    }
    result = iree_hip_graph_exec_rebuild(exec);
    if (result != hipSuccess) {
      iree_hip_graph_release_memcpy_callback_context_refs(stream_node->graph,
                                                          callback_data);
      *callback_data = old_callback_data;
    } else {
      iree_hip_graph_release_memcpy_callback_context_refs(stream_node->graph,
                                                          &old_callback_data);
    }
    iree_hal_streaming_graph_exec_release(exec);
    return result;
  }
  if (stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipMemcpyKind requested_kind =
      iree_hip_graph_metadata_memcpy_kind(params->kind);
  if (requested_kind != stream_node->attrs.memcpy.hip_kind) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hip_graph_memcpy_node_update_t update;
  result = iree_hip_graph_set_memcpy_node_params(stream_node, params, &update);
  if (result != hipSuccess) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(result);
  }
  result = iree_hip_graph_exec_rebuild(exec);
  if (result != hipSuccess) {
    iree_hip_graph_rollback_memcpy_node_update(stream_node, &update);
  } else {
    iree_hip_graph_commit_memcpy_node_update(stream_node, &update);
  }
  iree_hal_streaming_graph_exec_release(exec);
  return result;
}

HIPAPI hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t graphExec,
                                                    hipGraphNode_t node,
                                                    void* dst, const void* src,
                                                    size_t count,
                                                    hipMemcpyKind kind) {
  if (!graphExec || !node || !dst || !src || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemcpy3DParms params = {0};
  params.dstPtr.ptr = dst;
  params.dstPtr.pitch = count;
  params.dstPtr.xsize = count;
  params.dstPtr.ysize = 1;
  params.srcPtr.ptr = (void*)src;
  params.srcPtr.pitch = count;
  params.srcPtr.xsize = count;
  params.srcPtr.ysize = 1;
  params.extent.width = count;
  params.extent.height = 1;
  params.extent.depth = 1;
  params.kind = kind;
  return hipGraphExecMemcpyNodeSetParams(graphExec, node, &params);
}

HIPAPI hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(
    hipGraphExec_t graphExec, hipGraphNode_t node, void* dst,
    const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
  if (!graphExec || !node || !dst || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  void* symbol_ptr = NULL;
  hipError_t result =
      iree_hip_resolve_symbol_copy_range(symbol, count, offset, &symbol_ptr);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipGraphExecMemcpyNodeSetParams1D(graphExec, node, dst, symbol_ptr,
                                           count, kind);
}

HIPAPI hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(
    hipGraphExec_t graphExec, hipGraphNode_t node, const void* symbol,
    const void* src, size_t count, size_t offset, hipMemcpyKind kind) {
  if (!graphExec || !node || !src || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  void* symbol_ptr = NULL;
  hipError_t result =
      iree_hip_resolve_symbol_copy_range(symbol, count, offset, &symbol_ptr);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipGraphExecMemcpyNodeSetParams1D(graphExec, node, symbol_ptr, src,
                                           count, kind);
}

HIPAPI hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t graphExec,
                                                  hipGraphNode_t node,
                                                  const void* pNodeParams) {
  if (!graphExec || !node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_device_size_t element_count = 0;
  result = iree_hip_graph_validate_memset_params(
      stream_node->graph->context, (const hipMemsetParams*)pNodeParams,
      &dst_ref, &element_count);
  if (result != hipSuccess) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(result);
  }
  const hipMemsetParams* params = (const hipMemsetParams*)pNodeParams;
  const bool old_is_2d = stream_node->attrs.memset.hip_height != 1;
  const bool new_is_2d = params->height != 1;
  if (old_is_2d || new_is_2d) {
    if (stream_node->attrs.memset.pattern_size != params->elementSize ||
        stream_node->attrs.memset.count != element_count ||
        stream_node->attrs.memset.hip_width != params->width ||
        stream_node->attrs.memset.hip_height != params->height ||
        stream_node->attrs.memset.hip_pitch != params->pitch) {
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }
  iree_hal_streaming_graph_memset_node_attrs_t old_attrs =
      stream_node->attrs.memset;
  result = iree_hip_graph_set_memset_node_params(stream_node, params);
  if (result != hipSuccess) {
    stream_node->attrs.memset = old_attrs;
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(result);
  }
  result = iree_hip_graph_exec_rebuild(exec);
  stream_node->attrs.memset = old_attrs;
  iree_hal_streaming_graph_exec_release(exec);
  return result;
}

HIPAPI hipError_t
hipGraphExecKernelNodeSetParams(hipGraphExec_t graphExec, hipGraphNode_t node,
                                const hipKernelNodeParams* pNodeParams) {
  if (!graphExec || !node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_kernel_node_attrs_t old_attrs =
      stream_node->attrs.kernel;
  iree_allocator_t host_allocator = stream_node->graph->host_allocator;
  void* old_constants = NULL;
  iree_hal_buffer_ref_t* old_bindings = NULL;
  iree_host_size_t old_bindings_size = 0;
  if (old_attrs.constants_capacity > 0 && old_attrs.constants.data) {
    iree_status_t status = iree_allocator_malloc(
        host_allocator, old_attrs.constants_capacity, &old_constants);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    memcpy(old_constants, old_attrs.constants.data,
           old_attrs.constants_capacity);
  }
  if (old_attrs.binding_capacity > 0 && old_attrs.bindings.values) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            old_attrs.binding_capacity, sizeof(*old_attrs.bindings.values),
            &old_bindings_size))) {
      iree_allocator_free(host_allocator, old_constants);
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    iree_status_t status = iree_allocator_malloc(
        host_allocator, old_bindings_size, (void**)&old_bindings);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_allocator_free(host_allocator, old_constants);
      iree_hal_streaming_graph_exec_release(exec);
      HIP_RETURN_ERROR(hipErrorOutOfMemory);
    }
    memcpy(old_bindings, old_attrs.bindings.values, old_bindings_size);
  }

  result = hipGraphKernelNodeSetParams(node, pNodeParams);
  if (result != hipSuccess) {
    iree_allocator_free(host_allocator, old_bindings);
    iree_allocator_free(host_allocator, old_constants);
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(result);
  }
  result = iree_hip_graph_exec_rebuild(exec);
  stream_node->attrs.kernel = old_attrs;
  if (old_constants) {
    memcpy((void*)old_attrs.constants.data, old_constants,
           old_attrs.constants_capacity);
  }
  if (old_bindings) {
    memcpy((void*)old_attrs.bindings.values, old_bindings, old_bindings_size);
  }
  iree_allocator_free(host_allocator, old_bindings);
  iree_allocator_free(host_allocator, old_constants);
  iree_hal_streaming_graph_exec_release(exec);
  return result;
}

HIPAPI hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t graphExec,
                                                      hipGraphNode_t node,
                                                      hipGraph_t childGraph) {
  if (!graphExec || !node || !childGraph) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_exec_t* exec = NULL;
  hipError_t result = iree_hip_resolve_graph_exec(graphExec, &exec);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hal_streaming_graph_exec_owns_node(exec, stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if ((iree_hal_streaming_graph_t*)childGraph == stream_node->graph) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_t* old_child_graph =
      stream_node->attrs.child_graph.graph;
  iree_hal_streaming_graph_t* new_child_graph =
      (iree_hal_streaming_graph_t*)childGraph;
  if (!old_child_graph ||
      old_child_graph->node_count != new_child_graph->node_count) {
    iree_hal_streaming_graph_exec_release(exec);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_retain(new_child_graph);
  stream_node->attrs.child_graph.graph = new_child_graph;
  result = iree_hip_graph_exec_rebuild(exec);
  stream_node->attrs.child_graph.graph = old_child_graph;
  iree_hal_streaming_graph_release(new_child_graph);
  iree_hal_streaming_graph_exec_release(exec);
  return result;
}

HIPAPI hipError_t hipGraphExecNodeSetParams(hipGraphExec_t graphExec,
                                            hipGraphNode_t node,
                                            const void* nodeParams) {
  if (!graphExec || !node || !nodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipGraphNodeParams* params = (const hipGraphNodeParams*)nodeParams;
  if (params->type == hipGraphNodeTypeMemset) {
    return hipGraphExecMemsetNodeSetParams(graphExec, node, &params->memset);
  }
  if (params->type == hipGraphNodeTypeHost) {
    return hipGraphExecHostNodeSetParams(graphExec, node, &params->host);
  }
  if (params->type == hipGraphNodeTypeMemcpy) {
    return hipGraphExecMemcpyNodeSetParams(graphExec, node,
                                           &params->memcpy.copyParams);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

HIPAPI hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node,
                                            const void* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipHostNodeParams* params = (const hipHostNodeParams*)pNodeParams;
  if (!params->fn) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  stream_node->attrs.host.fn = (void (*)(void*))params->fn;
  stream_node->attrs.host.user_data = params->userData;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,
                                            hipHostNodeParams* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  pNodeParams->fn = (hipHostFn_t)stream_node->attrs.host.fn;
  pNodeParams->userData = stream_node->attrs.host.user_data;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphInstantiateWithParams(hipGraphExec_t* pGraphExec,
                                                hipGraph_t graph,
                                                void* instantiateParams) {
  if (!pGraphExec || !graph || !instantiateParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (!iree_hip_graph_handle_is_live(graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pGraphExec = NULL;
  hipGraphInstantiateParams* params =
      (hipGraphInstantiateParams*)instantiateParams;
  params->errNode_out = NULL;
  params->result_out = hipGraphInstantiateError;
  if (!iree_hip_graph_instantiate_flags_are_supported(params->flags)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t result =
      hipGraphInstantiateWithFlags(pGraphExec, graph, params->flags);
  params->result_out = result == hipSuccess ? hipGraphInstantiateSuccess
                                            : hipGraphInstantiateError;
  return result;
}

HIPAPI hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node,
                                              void* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipKernelNodeParams* params = (hipKernelNodeParams*)pNodeParams;
  params->func = stream_node->attrs.kernel.hip_function;
  params->gridDim = (dim3){stream_node->attrs.kernel.grid_dim[0],
                           stream_node->attrs.kernel.grid_dim[1],
                           stream_node->attrs.kernel.grid_dim[2]};
  params->blockDim = (dim3){stream_node->attrs.kernel.block_dim[0],
                            stream_node->attrs.kernel.block_dim[1],
                            stream_node->attrs.kernel.block_dim[2]};
  params->sharedMemBytes = stream_node->attrs.kernel.shared_memory_bytes;
  params->kernelParams = stream_node->attrs.kernel.hip_kernel_params;
  params->extra = stream_node->attrs.kernel.hip_extra;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                              const void* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipKernelNodeParams* params = (const hipKernelNodeParams*)pNodeParams;

  // Resolve the kernel symbol first so the extra-buffer handling below can key
  // off its reflected parameter metadata.
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t result = iree_hip_resolve_function_symbol(
      stream_node->graph->context, params->func, &symbol);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  if (!params->kernelParams && !params->extra &&
      !iree_hip_symbol_accepts_empty_kernel_params(symbol)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (params->kernelParams && params->extra) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t launch_config_result = iree_hip_validate_launch_configuration(
      stream_node->graph && stream_node->graph->context
          ? stream_node->graph->context->device_entry
          : NULL,
      symbol, params->gridDim.x, params->gridDim.y, params->gridDim.z,
      params->blockDim.x, params->blockDim.y, params->blockDim.z,
      params->sharedMemBytes);
  if (launch_config_result != hipSuccess) {
    HIP_RETURN_ERROR(launch_config_result);
  }

  void* params_ptr = NULL;
  size_t params_size = 0;
  iree_hal_streaming_dispatch_flags_t dispatch_flags =
      IREE_HAL_STREAMING_DISPATCH_FLAG_NONE;
  if (params->extra) {
    hipError_t parse_result =
        iree_hip_parse_launch_extra(params->extra, &params_ptr, &params_size);
    if (parse_result != hipSuccess) {
      HIP_RETURN_ERROR(parse_result);
    }
    // The extra buffer is already laid out in the kernel's native kernarg ABI.
    // Preserve it byte-for-byte; the graph node update stores a private copy.
    dispatch_flags |= IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED;
  } else if (params->kernelParams) {
    params_ptr = params->kernelParams;
    dispatch_flags |= IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY;
  }

  iree_hal_streaming_dispatch_params_t dispatch_params = {
      .grid_dim = {params->gridDim.x, params->gridDim.y, params->gridDim.z},
      .block_dim = {params->blockDim.x, params->blockDim.y, params->blockDim.z},
      .shared_memory_bytes = params->sharedMemBytes,
      .buffer = params_ptr,
      .buffer_size = params_size,
      .flags = dispatch_flags,
  };
  HIP_RETURN_STATUS(iree_hal_streaming_graph_set_kernel_node_params(
                        stream_node, symbol, &dispatch_params),
                    hipErrorInvalidValue);
  stream_node->attrs.kernel.hip_function = params->func;
  stream_node->attrs.kernel.hip_kernel_params = params->kernelParams;
  stream_node->attrs.kernel.hip_extra = params->extra;
  return hipSuccess;
}

static bool iree_hip_access_property_is_valid(hipAccessProperty property) {
  return property == hipAccessPropertyNormal ||
         property == hipAccessPropertyStreaming ||
         property == hipAccessPropertyPersisting;
}

static hipError_t iree_hip_validate_kernel_node_attribute(
    iree_hal_streaming_graph_node_t* node) {
  if (!iree_hip_graph_node_is_active(node) ||
      node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t iree_hip_validate_access_policy_window(
    const hipAccessPolicyWindow* window) {
  int max_window_size = 0;
  hipError_t result = hipDeviceGetAttribute(
      &max_window_size, hipDeviceAttributeAccessPolicyMaxWindowSize, 0);
  if (result != hipSuccess) {
    return result;
  }
  if (max_window_size < 0 || window->num_bytes > (size_t)max_window_size ||
      !(window->hitRatio >= 0.0f && window->hitRatio <= 1.0f) ||
      !iree_hip_access_property_is_valid(window->hitProp) ||
      !iree_hip_access_property_is_valid(window->missProp) ||
      window->missProp == hipAccessPropertyPersisting) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

HIPAPI hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc,
                                                   hipGraphNode_t hDst) {
  if (!hSrc || !hDst) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* src_node =
      (iree_hal_streaming_graph_node_t*)hSrc;
  iree_hal_streaming_graph_node_t* dst_node =
      (iree_hal_streaming_graph_node_t*)hDst;
  hipError_t result = iree_hip_validate_kernel_node_attribute(src_node);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  result = iree_hip_validate_kernel_node_attribute(dst_node);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }

  dst_node->attrs.kernel.access_policy_window_base_ptr =
      src_node->attrs.kernel.access_policy_window_base_ptr;
  dst_node->attrs.kernel.access_policy_window_num_bytes =
      src_node->attrs.kernel.access_policy_window_num_bytes;
  dst_node->attrs.kernel.access_policy_window_hit_ratio =
      src_node->attrs.kernel.access_policy_window_hit_ratio;
  dst_node->attrs.kernel.access_policy_window_hit_property =
      src_node->attrs.kernel.access_policy_window_hit_property;
  dst_node->attrs.kernel.access_policy_window_miss_property =
      src_node->attrs.kernel.access_policy_window_miss_property;
  dst_node->attrs.kernel.cooperative = src_node->attrs.kernel.cooperative;
  dst_node->attrs.kernel.priority = src_node->attrs.kernel.priority;
  return hipSuccess;
}

HIPAPI hipError_t
hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                               hipKernelNodeAttrValue* value) {
  if (!hNode || !value) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* node =
      (iree_hal_streaming_graph_node_t*)hNode;
  hipError_t result = iree_hip_validate_kernel_node_attribute(node);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }

  memset(value, 0, sizeof(*value));
  switch (attr) {
    case hipKernelNodeAttributeAccessPolicyWindow:
      value->accessPolicyWindow.base_ptr =
          node->attrs.kernel.access_policy_window_base_ptr;
      value->accessPolicyWindow.num_bytes =
          node->attrs.kernel.access_policy_window_num_bytes;
      value->accessPolicyWindow.hitRatio =
          node->attrs.kernel.access_policy_window_hit_ratio;
      value->accessPolicyWindow.hitProp =
          (hipAccessProperty)
              node->attrs.kernel.access_policy_window_hit_property;
      value->accessPolicyWindow.missProp =
          (hipAccessProperty)
              node->attrs.kernel.access_policy_window_miss_property;
      return hipSuccess;
    case hipKernelNodeAttributeCooperative:
      value->cooperative = node->attrs.kernel.cooperative;
      return hipSuccess;
    case hipKernelNodeAttributePriority:
      value->priority = node->attrs.kernel.priority;
      return hipSuccess;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
}

HIPAPI hipError_t
hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                               const hipKernelNodeAttrValue* value) {
  if (!hNode || !value) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* node =
      (iree_hal_streaming_graph_node_t*)hNode;
  hipError_t result = iree_hip_validate_kernel_node_attribute(node);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }

  switch (attr) {
    case hipKernelNodeAttributeAccessPolicyWindow:
      result =
          iree_hip_validate_access_policy_window(&value->accessPolicyWindow);
      if (result != hipSuccess) {
        HIP_RETURN_ERROR(result);
      }
      node->attrs.kernel.access_policy_window_base_ptr =
          value->accessPolicyWindow.base_ptr;
      node->attrs.kernel.access_policy_window_num_bytes =
          value->accessPolicyWindow.num_bytes;
      node->attrs.kernel.access_policy_window_hit_ratio =
          value->accessPolicyWindow.hitRatio;
      node->attrs.kernel.access_policy_window_hit_property =
          (uint32_t)value->accessPolicyWindow.hitProp;
      node->attrs.kernel.access_policy_window_miss_property =
          (uint32_t)value->accessPolicyWindow.missProp;
      return hipSuccess;
    case hipKernelNodeAttributeCooperative:
      node->attrs.kernel.cooperative = value->cooperative;
      return hipSuccess;
    case hipKernelNodeAttributePriority:
      node->attrs.kernel.priority = value->priority;
      return hipSuccess;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
}

HIPAPI hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node,
                                                hipMemAllocNodeParams* params) {
  if (!node || !params) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC ||
      !stream_node->attrs.mem_alloc.params) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *params = *(const hipMemAllocNodeParams*)stream_node->attrs.mem_alloc.params;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node,
                                               void* dptr_out) {
  if (!node || !dptr_out) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *(void**)dptr_out = stream_node->attrs.mem_free.dptr;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node,
                                              void* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hip_graph_handle_is_live((hipGraph_t)stream_node->graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemcpy3DParms* params = (hipMemcpy3DParms*)pNodeParams;
  memset(params, 0, sizeof(*params));
  if (iree_hip_graph_node_is_visible_memcpy_host_call(stream_node)) {
    const iree_hip_graph_memcpy_callback_data_t* callback_data =
        (const iree_hip_graph_memcpy_callback_data_t*)
            stream_node->attrs.host.user_data;
    *params = callback_data->hip_params;
    return hipSuccess;
  }
  if (stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  params->dstPtr.ptr = stream_node->attrs.memcpy.hip_dst;
  params->dstArray = (hipArray_t)stream_node->attrs.memcpy.hip_dst_array;
  params->dstPtr.pitch = stream_node->attrs.memcpy.hip_dst_pitch;
  params->dstPtr.xsize = stream_node->attrs.memcpy.hip_dst_xsize;
  params->dstPtr.ysize = stream_node->attrs.memcpy.hip_dst_ysize;
  params->dstPos.x = stream_node->attrs.memcpy.hip_dst_position_x;
  params->dstPos.y = stream_node->attrs.memcpy.hip_dst_position_y;
  params->dstPos.z = stream_node->attrs.memcpy.hip_dst_position_z;
  params->srcPtr.ptr = (void*)stream_node->attrs.memcpy.hip_src;
  params->srcArray = (hipArray_t)stream_node->attrs.memcpy.hip_src_array;
  params->srcPtr.pitch = stream_node->attrs.memcpy.hip_src_pitch;
  params->srcPtr.xsize = stream_node->attrs.memcpy.hip_src_xsize;
  params->srcPtr.ysize = stream_node->attrs.memcpy.hip_src_ysize;
  params->srcPos.x = stream_node->attrs.memcpy.hip_src_position_x;
  params->srcPos.y = stream_node->attrs.memcpy.hip_src_position_y;
  params->srcPos.z = stream_node->attrs.memcpy.hip_src_position_z;
  params->extent.width = stream_node->attrs.memcpy.hip_extent_width;
  params->extent.height = stream_node->attrs.memcpy.hip_extent_height;
  params->extent.depth = stream_node->attrs.memcpy.hip_extent_depth;
  params->kind = (hipMemcpyKind)stream_node->attrs.memcpy.hip_kind;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node,
                                              const void* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hip_graph_handle_is_live((hipGraph_t)stream_node->graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipMemcpy3DParms* params = (const hipMemcpy3DParms*)pNodeParams;
  hipError_t result = hipSuccess;
  if (iree_hip_graph_node_is_visible_memcpy_host_call(stream_node)) {
    hipMemcpyKind kind = hipMemcpyDefault;
    void* dst = NULL;
    const void* src = NULL;
    iree_host_size_t size = 0;
    result = iree_hip_graph_validate_memcpy3d_params(
        stream_node->graph->context, params, &kind, &dst, &src, &size, NULL,
        NULL);
    if (result == hipSuccess) {
      result = iree_hip_graph_set_host_memcpy_callback_params(
          stream_node, params, kind, dst, src, size, NULL);
    }
  } else {
    result = iree_hip_graph_set_memcpy_node_params(stream_node, params, NULL);
  }
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst,
                                                const void* src, size_t count,
                                                hipMemcpyKind kind) {
  if (!node || !dst || !src || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemcpy3DParms params = {0};
  params.dstPtr.ptr = dst;
  params.dstPtr.pitch = count;
  params.dstPtr.xsize = count;
  params.dstPtr.ysize = 1;
  params.srcPtr.ptr = (void*)src;
  params.srcPtr.pitch = count;
  params.srcPtr.xsize = count;
  params.srcPtr.ysize = 1;
  params.extent.width = count;
  params.extent.height = 1;
  params.extent.depth = 1;
  params.kind = kind;
  return hipGraphMemcpyNodeSetParams(node, &params);
}

HIPAPI hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(
    hipGraphNode_t node, void* dst, const void* symbol, size_t count,
    size_t offset, hipMemcpyKind kind) {
  if (!node || !dst || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  void* symbol_ptr = NULL;
  hipError_t result =
      iree_hip_resolve_symbol_copy_range(symbol, count, offset, &symbol_ptr);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipGraphMemcpyNodeSetParams1D(node, dst, symbol_ptr, count, kind);
}

HIPAPI hipError_t hipGraphMemcpyNodeSetParamsToSymbol(
    hipGraphNode_t node, const void* symbol, const void* src, size_t count,
    size_t offset, hipMemcpyKind kind) {
  if (!node || !src || count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  void* symbol_ptr = NULL;
  hipError_t result =
      iree_hip_resolve_symbol_copy_range(symbol, count, offset, &symbol_ptr);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipGraphMemcpyNodeSetParams1D(node, symbol_ptr, src, count, kind);
}

HIPAPI hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,
                                              void* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)node;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemsetParams* params = (hipMemsetParams*)pNodeParams;
  memset(params, 0, sizeof(*params));
  params->dst = stream_node->attrs.memset.hip_dst;
  params->elementSize = stream_node->attrs.memset.pattern_size;
  params->height = stream_node->attrs.memset.hip_height;
  params->pitch = stream_node->attrs.memset.hip_pitch;
  params->value = stream_node->attrs.memset.pattern;
  params->width = stream_node->attrs.memset.hip_width;
  return hipSuccess;
}

HIPAPI hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,
                                              const void* pNodeParams) {
  if (!node || !pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t result = iree_hip_graph_set_memset_node_params(
      (iree_hal_streaming_graph_node_t*)node,
      (const hipMemsetParams*)pNodeParams);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipGraphNodeSetParams(hipGraphNode_t node,
                                        const void* nodeParams) {
  if (!node || !nodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const hipGraphNodeParams* params = (const hipGraphNodeParams*)nodeParams;
  switch (params->type) {
    case hipGraphNodeTypeMemset:
      return hipGraphMemsetNodeSetParams(node, &params->memset);
    case hipGraphNodeTypeMemcpy:
      return hipGraphMemcpyNodeSetParams(node, &params->memcpy.copyParams);
    case hipGraphNodeTypeKernel:
      return hipGraphKernelNodeSetParams(node, &params->kernel);
    case hipGraphNodeTypeHost:
      return hipGraphHostNodeSetParams(node, &params->host);
    default:
      HIP_RETURN_ERROR(hipErrorNotSupported);
  }
}

typedef struct iree_hip_driver_memcpy3d_t {
  size_t srcXInBytes;
  size_t srcY;
  size_t srcZ;
  size_t srcLOD;
  hipMemoryType srcMemoryType;
  const void* srcHost;
  hipDeviceptr_t srcDevice;
  hipArray_t srcArray;
  size_t srcPitch;
  size_t srcHeight;
  size_t dstXInBytes;
  size_t dstY;
  size_t dstZ;
  size_t dstLOD;
  hipMemoryType dstMemoryType;
  void* dstHost;
  hipDeviceptr_t dstDevice;
  hipArray_t dstArray;
  size_t dstPitch;
  size_t dstHeight;
  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
} iree_hip_driver_memcpy3d_t;

static hipError_t iree_hip_graph_convert_driver_memcpy3d_params(
    const iree_hip_driver_memcpy3d_t* src, hipMemcpy3DParms* dst) {
  if (!src || !dst || src->WidthInBytes == 0 || src->Height == 0 ||
      src->Depth == 0) {
    return hipErrorInvalidValue;
  }
  if (src->srcMemoryType == hipMemoryTypeArray ||
      src->dstMemoryType == hipMemoryTypeArray) {
    return hipErrorInvalidValue;
  }
  if (src->srcLOD != 0 || src->dstLOD != 0) {
    return hipErrorInvalidValue;
  }
  if ((src->srcMemoryType == hipMemoryTypeHost && src->srcZ != 0) ||
      (src->dstMemoryType == hipMemoryTypeHost && src->dstZ != 0)) {
    return hipErrorInvalidValue;
  }
  if (src->srcPitch == 0 || src->dstPitch == 0 ||
      src->srcPitch < src->WidthInBytes || src->dstPitch < src->WidthInBytes) {
    return hipErrorInvalidValue;
  }
  int max_pitch = 0;
  if (hipDeviceGetAttribute(&max_pitch, hipDeviceAttributeMaxPitch, 0) ==
          hipSuccess &&
      max_pitch > 0 &&
      (src->srcPitch >= (size_t)max_pitch ||
       src->dstPitch >= (size_t)max_pitch)) {
    return hipErrorInvalidValue;
  }

  memset(dst, 0, sizeof(*dst));
  dst->srcPos.x = src->srcXInBytes;
  dst->srcPos.y = src->srcY;
  dst->srcPos.z = src->srcZ;
  dst->dstPos.x = src->dstXInBytes;
  dst->dstPos.y = src->dstY;
  dst->dstPos.z = src->dstZ;
  dst->extent.width = src->WidthInBytes;
  dst->extent.height = src->Height;
  dst->extent.depth = src->Depth;

  bool use_default_kind = false;
  if (src->srcMemoryType == hipMemoryTypeHost) {
    dst->srcPtr.ptr = (void*)src->srcHost;
    dst->kind = hipMemcpyHostToHost;
  } else if (src->srcMemoryType == hipMemoryTypeDevice) {
    dst->srcPtr.ptr = (void*)src->srcDevice;
    dst->kind = hipMemcpyDeviceToDevice;
  } else if (src->srcMemoryType == hipMemoryTypeUnified) {
    dst->srcPtr.ptr = (void*)src->srcDevice;
    dst->kind = hipMemcpyDefault;
    use_default_kind = true;
  } else {
    return hipErrorInvalidValue;
  }
  dst->srcPtr.pitch = src->srcPitch;
  dst->srcPtr.xsize = src->WidthInBytes;
  dst->srcPtr.ysize = src->srcHeight ? src->srcHeight : src->Height;

  if (src->dstMemoryType == hipMemoryTypeHost) {
    dst->dstPtr.ptr = src->dstHost;
    dst->kind = dst->kind == hipMemcpyDeviceToDevice ? hipMemcpyDeviceToHost
                                                     : hipMemcpyHostToHost;
  } else if (src->dstMemoryType == hipMemoryTypeDevice) {
    dst->dstPtr.ptr = (void*)src->dstDevice;
    dst->kind = dst->kind == hipMemcpyHostToHost ? hipMemcpyHostToDevice
                                                 : hipMemcpyDeviceToDevice;
  } else if (src->dstMemoryType == hipMemoryTypeUnified) {
    dst->dstPtr.ptr = (void*)src->dstDevice;
    dst->kind = hipMemcpyDefault;
    use_default_kind = true;
  } else {
    return hipErrorInvalidValue;
  }
  dst->dstPtr.pitch = src->dstPitch;
  dst->dstPtr.xsize = src->WidthInBytes;
  dst->dstPtr.ysize = src->dstHeight ? src->dstHeight : src->Height;
  if (use_default_kind) {
    dst->kind = hipMemcpyDefault;
  }
  return hipSuccess;
}

HIPAPI hipError_t hipDrvMemcpy3D(const iree_hip_driver_memcpy3d_t* pCopy) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hipMemcpy3DParms params;
  hipError_t result =
      iree_hip_graph_convert_driver_memcpy3d_params(pCopy, &params);
  if (result == hipSuccess) {
    result = hipMemcpy3D(&params);
  }
  IREE_TRACE_ZONE_END(z0);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipDrvMemcpy3DAsync(const iree_hip_driver_memcpy3d_t* pCopy,
                                      hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hipMemcpy3DParms params;
  hipError_t result =
      iree_hip_graph_convert_driver_memcpy3d_params(pCopy, &params);
  if (result == hipSuccess) {
    result = hipMemcpy3DAsync(&params, stream);
  }
  IREE_TRACE_ZONE_END(z0);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

typedef struct iree_hip_graph_driver_memcpy_callback_data_t {
  iree_hip_graph_memcpy_callback_data_t memcpy;
  iree_hip_driver_memcpy3d_t driver;
} iree_hip_graph_driver_memcpy_callback_data_t;

static void iree_hip_graph_store_driver_memcpy3d_metadata(
    iree_hal_streaming_graph_node_t* node,
    const iree_hip_driver_memcpy3d_t* params) {
  if (!node || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY ||
      !params) {
    return;
  }
  iree_hal_streaming_graph_memcpy_driver_node_attrs_t* attrs =
      &node->attrs.memcpy.hip_driver;
  memset(attrs, 0, sizeof(*attrs));
  attrs->valid = true;
  attrs->src_x_in_bytes = params->srcXInBytes;
  attrs->src_y = params->srcY;
  attrs->src_z = params->srcZ;
  attrs->src_lod = params->srcLOD;
  attrs->src_memory_type = params->srcMemoryType;
  attrs->src_host = params->srcHost;
  attrs->src_device = (iree_hal_streaming_deviceptr_t)params->srcDevice;
  attrs->src_array = params->srcArray;
  attrs->src_pitch = params->srcPitch;
  attrs->src_height = params->srcHeight;
  attrs->dst_x_in_bytes = params->dstXInBytes;
  attrs->dst_y = params->dstY;
  attrs->dst_z = params->dstZ;
  attrs->dst_lod = params->dstLOD;
  attrs->dst_memory_type = params->dstMemoryType;
  attrs->dst_host = params->dstHost;
  attrs->dst_device = (iree_hal_streaming_deviceptr_t)params->dstDevice;
  attrs->dst_array = params->dstArray;
  attrs->dst_pitch = params->dstPitch;
  attrs->dst_height = params->dstHeight;
  attrs->width_in_bytes = params->WidthInBytes;
  attrs->height = params->Height;
  attrs->depth = params->Depth;
}

static void iree_hip_graph_copy_driver_memcpy3d_metadata(
    const iree_hal_streaming_graph_memcpy_driver_node_attrs_t* src,
    iree_hip_driver_memcpy3d_t* dst) {
  memset(dst, 0, sizeof(*dst));
  dst->srcXInBytes = src->src_x_in_bytes;
  dst->srcY = src->src_y;
  dst->srcZ = src->src_z;
  dst->srcLOD = src->src_lod;
  dst->srcMemoryType = (hipMemoryType)src->src_memory_type;
  dst->srcHost = src->src_host;
  dst->srcDevice = (hipDeviceptr_t)src->src_device;
  dst->srcArray = (hipArray_t)src->src_array;
  dst->srcPitch = src->src_pitch;
  dst->srcHeight = src->src_height;
  dst->dstXInBytes = src->dst_x_in_bytes;
  dst->dstY = src->dst_y;
  dst->dstZ = src->dst_z;
  dst->dstLOD = src->dst_lod;
  dst->dstMemoryType = (hipMemoryType)src->dst_memory_type;
  dst->dstHost = src->dst_host;
  dst->dstDevice = (hipDeviceptr_t)src->dst_device;
  dst->dstArray = (hipArray_t)src->dst_array;
  dst->dstPitch = src->dst_pitch;
  dst->dstHeight = src->dst_height;
  dst->WidthInBytes = src->width_in_bytes;
  dst->Height = src->height;
  dst->Depth = src->depth;
}

static hipError_t iree_hip_graph_store_driver_host_memcpy_metadata(
    iree_hal_streaming_graph_node_t* node,
    const iree_hip_driver_memcpy3d_t* params) {
  if (!iree_hip_graph_node_is_memcpy_host_call(node) || !params) {
    return hipErrorInvalidValue;
  }
  iree_hip_graph_driver_memcpy_callback_data_t* callback_data =
      (iree_hip_graph_driver_memcpy_callback_data_t*)node->attrs.host.user_data;
  if (node->attrs.host.user_data_size < sizeof(*callback_data)) {
    iree_hip_graph_memcpy_callback_data_t old_memcpy =
        *(iree_hip_graph_memcpy_callback_data_t*)node->attrs.host.user_data;
    iree_status_t status = iree_arena_allocate(
        &node->graph->arena, sizeof(*callback_data), (void**)&callback_data);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorOutOfMemory;
    }
    memset(callback_data, 0, sizeof(*callback_data));
    callback_data->memcpy = old_memcpy;
    node->attrs.host.user_data = callback_data;
    node->attrs.host.user_data_size = sizeof(*callback_data);
  }
  callback_data->driver = *params;
  return hipSuccess;
}

static hipError_t iree_hip_graph_copy_driver_host_memcpy_metadata(
    const iree_hal_streaming_graph_node_t* node,
    iree_hip_driver_memcpy3d_t* params) {
  if (!iree_hip_graph_node_is_memcpy_host_call(node) || !params ||
      node->attrs.host.user_data_size <
          sizeof(iree_hip_graph_driver_memcpy_callback_data_t)) {
    return hipErrorInvalidValue;
  }
  const iree_hip_graph_driver_memcpy_callback_data_t* callback_data =
      (const iree_hip_graph_driver_memcpy_callback_data_t*)
          node->attrs.host.user_data;
  *params = callback_data->driver;
  return hipSuccess;
}

static hipError_t iree_hip_graph_set_host_memcpy_callback_params(
    iree_hal_streaming_graph_node_t* node, const hipMemcpy3DParms* params,
    hipMemcpyKind kind, void* dst, const void* src, iree_host_size_t size,
    iree_hip_graph_memcpy_callback_data_t* out_old_callback_data) {
  if (!iree_hip_graph_node_is_visible_memcpy_host_call(node) ||
      (kind != hipMemcpyHostToHost && kind != hipMemcpyDeviceToDevice)) {
    return hipErrorInvalidValue;
  }
  iree_host_size_t src_rows_per_slice = 0;
  hipError_t result = iree_hip_graph_memcpy_rows_per_slice(
      params->srcPos.y, params->extent.height, params->srcPtr.ysize,
      &src_rows_per_slice);
  if (result != hipSuccess) return result;
  iree_host_size_t dst_rows_per_slice = 0;
  result = iree_hip_graph_memcpy_rows_per_slice(
      params->dstPos.y, params->extent.height, params->dstPtr.ysize,
      &dst_rows_per_slice);
  if (result != hipSuccess) return result;
  iree_hip_graph_memcpy_callback_data_t* callback_data =
      (iree_hip_graph_memcpy_callback_data_t*)node->attrs.host.user_data;
  iree_hip_graph_memcpy_callback_data_t old_callback_data = *callback_data;
  if (out_old_callback_data) *out_old_callback_data = old_callback_data;
  result = iree_hip_graph_configure_memcpy_callback_data(
      callback_data, params, dst, src, size, params->dstPtr.pitch,
      params->srcPtr.pitch, dst_rows_per_slice, src_rows_per_slice);
  if (result != hipSuccess) return result;
  if (kind != hipMemcpyDeviceToDevice) {
    if (!out_old_callback_data) {
      iree_hip_graph_release_memcpy_callback_context_refs(node->graph,
                                                          &old_callback_data);
    }
    return hipSuccess;
  }

  iree_host_size_t dst_span = 0;
  result = iree_hip_graph_memcpy3d_span_bytes(
      params->dstPtr.pitch, dst_rows_per_slice, params->extent.width,
      params->extent.height, params->extent.depth, &dst_span);
  if (result != hipSuccess) {
    *callback_data = old_callback_data;
    return result;
  }
  iree_host_size_t src_span = 0;
  result = iree_hip_graph_memcpy3d_span_bytes(
      params->srcPtr.pitch, src_rows_per_slice, params->extent.width,
      params->extent.height, params->extent.depth, &src_span);
  if (result != hipSuccess) {
    *callback_data = old_callback_data;
    return result;
  }

  iree_hal_streaming_context_t* dst_context = NULL;
  iree_hal_streaming_buffer_ref_t dst_ref = {0};
  iree_status_t dst_status =
      iree_hal_streaming_memory_lookup_range_across_contexts(
          (iree_hal_streaming_deviceptr_t)dst, dst_span, &dst_context,
          &dst_ref);
  iree_hal_streaming_context_t* src_context = NULL;
  iree_hal_streaming_buffer_ref_t src_ref = {0};
  iree_status_t src_status =
      iree_hal_streaming_memory_lookup_range_across_contexts(
          (iree_hal_streaming_deviceptr_t)src, src_span, &src_context,
          &src_ref);
  const bool valid =
      iree_status_is_ok(dst_status) && iree_status_is_ok(src_status) &&
      (dst_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      (src_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  if (!valid) {
    result = hipErrorInvalidValue;
  }
  if (result == hipSuccess) {
    result = iree_hip_graph_retain_remote_context(node->graph, dst_context);
  }
  bool retained_dst_context = result == hipSuccess;
  if (result == hipSuccess) {
    result = iree_hip_graph_retain_remote_context(node->graph, src_context);
  }
  bool retained_src_context = result == hipSuccess;
  if (result == hipSuccess) {
    callback_data->dst_context = dst_context;
    callback_data->src_context = src_context;
  } else {
    if (retained_src_context) {
      iree_hip_graph_release_remote_context(node->graph, src_context);
    }
    if (retained_dst_context) {
      iree_hip_graph_release_remote_context(node->graph, dst_context);
    }
    *callback_data = old_callback_data;
  }

  if (iree_status_is_ok(dst_status)) {
    iree_hal_streaming_context_release(dst_context);
  } else {
    iree_status_ignore(dst_status);
  }
  if (iree_status_is_ok(src_status)) {
    iree_hal_streaming_context_release(src_context);
  } else {
    iree_status_ignore(src_status);
  }
  if (result == hipSuccess && !out_old_callback_data) {
    iree_hip_graph_release_memcpy_callback_context_refs(node->graph,
                                                        &old_callback_data);
  }
  return result;
}

static hipError_t iree_hip_graph_infer_driver_memcpy3d_params(
    const iree_hal_streaming_graph_memcpy_node_attrs_t* src,
    iree_hip_driver_memcpy3d_t* dst) {
  memset(dst, 0, sizeof(*dst));
  dst->srcXInBytes = src->hip_src_position_x;
  dst->srcY = src->hip_src_position_y;
  dst->srcZ = src->hip_src_position_z;
  dst->srcPitch = src->hip_src_pitch;
  dst->srcHeight = src->hip_src_ysize;
  dst->dstXInBytes = src->hip_dst_position_x;
  dst->dstY = src->hip_dst_position_y;
  dst->dstZ = src->hip_dst_position_z;
  dst->dstPitch = src->hip_dst_pitch;
  dst->dstHeight = src->hip_dst_ysize;
  dst->WidthInBytes = src->hip_extent_width;
  dst->Height = src->hip_extent_height;
  dst->Depth = src->hip_extent_depth;

  switch ((hipMemcpyKind)src->hip_kind) {
    case hipMemcpyHostToHost:
      dst->srcMemoryType = hipMemoryTypeHost;
      dst->srcHost = src->hip_src;
      dst->dstMemoryType = hipMemoryTypeHost;
      dst->dstHost = src->hip_dst;
      return hipSuccess;
    case hipMemcpyHostToDevice:
      dst->srcMemoryType = hipMemoryTypeHost;
      dst->srcHost = src->hip_src;
      dst->dstMemoryType = hipMemoryTypeDevice;
      dst->dstDevice = (hipDeviceptr_t)src->hip_dst;
      return hipSuccess;
    case hipMemcpyDeviceToHost:
      dst->srcMemoryType = hipMemoryTypeDevice;
      dst->srcDevice = (hipDeviceptr_t)src->hip_src;
      dst->dstMemoryType = hipMemoryTypeHost;
      dst->dstHost = src->hip_dst;
      return hipSuccess;
    case hipMemcpyDeviceToDevice:
    case hipMemcpyDeviceToDeviceNoCU:
      dst->srcMemoryType = hipMemoryTypeDevice;
      dst->srcDevice = (hipDeviceptr_t)src->hip_src;
      dst->dstMemoryType = hipMemoryTypeDevice;
      dst->dstDevice = (hipDeviceptr_t)src->hip_dst;
      return hipSuccess;
    case hipMemcpyDefault:
      dst->srcMemoryType = hipMemoryTypeUnified;
      dst->srcDevice = (hipDeviceptr_t)src->hip_src;
      dst->dstMemoryType = hipMemoryTypeUnified;
      dst->dstDevice = (hipDeviceptr_t)src->hip_dst;
      return hipSuccess;
    default:
      return hipErrorInvalidMemcpyDirection;
  }
}

HIPAPI hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* pGraphNode,
                                           hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies,
                                           const void* pCopyParams, void* ctx) {
  (void)ctx;
  if (!pCopyParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemcpy3DParms params = {0};
  hipError_t result = iree_hip_graph_convert_driver_memcpy3d_params(
      (const iree_hip_driver_memcpy3d_t*)pCopyParams, &params);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  result = hipGraphAddMemcpyNode(pGraphNode, graph, pDependencies,
                                 numDependencies, &params);
  if (result == hipSuccess) {
    iree_hal_streaming_graph_node_t* stream_node =
        (iree_hal_streaming_graph_node_t*)*pGraphNode;
    if (stream_node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
      iree_hip_graph_store_driver_memcpy3d_metadata(
          stream_node, (const iree_hip_driver_memcpy3d_t*)pCopyParams);
    } else if (iree_hip_graph_node_is_memcpy_host_call(stream_node)) {
      result = iree_hip_graph_store_driver_host_memcpy_metadata(
          stream_node, (const iree_hip_driver_memcpy3d_t*)pCopyParams);
    }
  }
  return result;
}

HIPAPI hipError_t hipDrvGraphMemcpyNodeGetParams(
    hipGraphNode_t hNode, iree_hip_driver_memcpy3d_t* nodeParams) {
  if (!hNode || !nodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)hNode;
  if (!iree_hip_graph_node_is_active(stream_node) ||
      !iree_hip_graph_handle_is_live((hipGraph_t)stream_node->graph)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (iree_hip_graph_node_is_memcpy_host_call(stream_node)) {
    hipError_t result = iree_hip_graph_copy_driver_host_memcpy_metadata(
        stream_node, nodeParams);
    if (result != hipSuccess) {
      HIP_RETURN_ERROR(result);
    }
    return hipSuccess;
  }
  if (stream_node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  const iree_hal_streaming_graph_memcpy_node_attrs_t* attrs =
      &stream_node->attrs.memcpy;
  if (attrs->hip_driver.valid) {
    iree_hip_graph_copy_driver_memcpy3d_metadata(&attrs->hip_driver,
                                                 nodeParams);
    return hipSuccess;
  }
  hipError_t result =
      iree_hip_graph_infer_driver_memcpy3d_params(attrs, nodeParams);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipSuccess;
}

HIPAPI hipError_t hipDrvGraphMemcpyNodeSetParams(
    hipGraphNode_t hNode, const iree_hip_driver_memcpy3d_t* nodeParams) {
  if (!hNode || !nodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemcpy3DParms params = {0};
  hipError_t result =
      iree_hip_graph_convert_driver_memcpy3d_params(nodeParams, &params);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  iree_hal_streaming_graph_node_t* stream_node =
      (iree_hal_streaming_graph_node_t*)hNode;
  if (iree_hip_graph_node_is_active(stream_node) &&
      iree_hip_graph_handle_is_live((hipGraph_t)stream_node->graph) &&
      iree_hip_graph_node_is_memcpy_host_call(stream_node)) {
    hipMemcpyKind kind = hipMemcpyDefault;
    void* dst = NULL;
    const void* src = NULL;
    iree_host_size_t size = 0;
    result = iree_hip_graph_validate_memcpy3d_params(
        stream_node->graph->context, &params, &kind, &dst, &src, &size, NULL,
        NULL);
    if (result != hipSuccess) {
      HIP_RETURN_ERROR(result);
    }
    iree_hip_graph_memcpy_callback_data_t* callback_data =
        (iree_hip_graph_memcpy_callback_data_t*)
            stream_node->attrs.host.user_data;
    iree_hip_graph_memcpy_callback_data_t old_callback_data;
    result = iree_hip_graph_set_host_memcpy_callback_params(
        stream_node, &params, kind, dst, src, size, &old_callback_data);
    if (result != hipSuccess) {
      HIP_RETURN_ERROR(result);
    }
    result = iree_hip_graph_store_driver_host_memcpy_metadata(stream_node,
                                                              nodeParams);
    if (result != hipSuccess) {
      iree_hip_graph_release_memcpy_callback_context_refs(stream_node->graph,
                                                          callback_data);
      *callback_data = old_callback_data;
      HIP_RETURN_ERROR(result);
    }
    iree_hip_graph_release_memcpy_callback_context_refs(stream_node->graph,
                                                        &old_callback_data);
    return hipSuccess;
  }
  result = hipGraphMemcpyNodeSetParams(hNode, &params);
  if (result == hipSuccess) {
    if (stream_node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
      iree_hip_graph_store_driver_memcpy3d_metadata(stream_node, nodeParams);
    } else {
      result = iree_hip_graph_store_driver_host_memcpy_metadata(stream_node,
                                                                nodeParams);
    }
  }
  return result;
}

HIPAPI hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t* pGraphNode,
                                            hipGraph_t graph,
                                            const hipGraphNode_t* pDependencies,
                                            size_t numDependencies, void* dptr,
                                            void* ctx) {
  (void)ctx;
  return hipGraphAddMemFreeNode(pGraphNode, graph, pDependencies,
                                numDependencies, dptr);
}

HIPAPI hipError_t hipDrvGraphExecMemcpyNodeSetParams(hipGraphExec_t graphExec,
                                                     hipGraphNode_t node,
                                                     const void* pNodeParams,
                                                     void* ctx) {
  (void)ctx;
  if (!pNodeParams) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipMemcpy3DParms params = {0};
  hipError_t result = iree_hip_graph_convert_driver_memcpy3d_params(
      (const iree_hip_driver_memcpy3d_t*)pNodeParams, &params);
  if (result != hipSuccess) {
    HIP_RETURN_ERROR(result);
  }
  return hipGraphExecMemcpyNodeSetParams(graphExec, node, &params);
}

HIPAPI hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode,
                                           hipGraph_t hGraph,
                                           const hipGraphNode_t* dependencies,
                                           size_t numDependencies,
                                           const hipMemsetParams* memsetParams,
                                           hipCtx_t ctx) {
  (void)ctx;
  return hipGraphAddMemsetNode(phGraphNode, hGraph, dependencies,
                               numDependencies, memsetParams);
}

HIPAPI hipError_t hipDrvGraphExecMemsetNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  (void)ctx;
  return hipGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams);
}

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

// Begins capturing operations into a graph.
//
// Parameters:
//  - stream: [IN] Stream to capture operations from.
//  - mode: [IN] Capture mode (affects cross-stream dependencies).
//
// Returns:
//  - hipSuccess: Capture started successfully.
//  - hipErrorInvalidValue: Invalid stream or mode.
//  - hipErrorStreamCaptureInvalidated: Stream already capturing.
//  - hipErrorStreamCaptureImplicit: Implicit capture not supported.
//
// Synchronization: Subsequent stream operations are captured.
//
// Capture modes:
// - hipStreamCaptureModeGlobal: Capture all work.
// - hipStreamCaptureModeThreadLocal: Thread-local capture.
// - hipStreamCaptureModeRelaxed: Relaxed ordering.
//
// Capture behavior:
// - Operations recorded to graph instead of executing.
// - Continues until hipStreamEndCapture().
// - Creates graph from recorded operations.
// - Stream cannot be synchronized during capture.
//
// Captured operations:
// - Kernel launches.
// - Memory copies and sets.
// - Event records and waits.
// - Child stream operations (mode dependent).
//
// Restrictions during capture:
// - No hipStreamSynchronize() on capturing stream.
// - No hipDeviceSynchronize().
// - No blocking operations.
// - Limited cross-stream dependencies.
//
// Cross-stream capture:
// - Global mode: Captures work from other streams.
// - ThreadLocal mode: Only current thread's work.
// - Relaxed mode: Best-effort capture.
//
// Error handling:
// - Invalid operations invalidate capture.
// - Check hipStreamEndCapture() for errors.
// - Query status with hipStreamIsCapturing().
//
// Multi-GPU:
// - Can capture operations for multiple devices.
// - Device switches recorded in graph.
//
// Warning: Stream must be idle before starting capture.
// Pending operations may cause undefined behavior.
//
// See also: hipStreamEndCapture, hipStreamIsCapturing,
//           hipGraphCreate, hipGraphLaunch.
HIPAPI hipError_t hipStreamBeginCapture(hipStream_t stream,
                                        hipStreamCaptureMode mode) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream || stream == hipStreamLegacy) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  // Map HIP capture mode to internal mode.
  iree_hal_streaming_capture_mode_t capture_mode;
  switch (mode) {
    case hipStreamCaptureModeGlobal:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
      break;
    case hipStreamCaptureModeThreadLocal:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL;
      break;
    case hipStreamCaptureModeRelaxed:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Explicit stream handles remain valid when another device is current. The
  // registry lookup also retains the stream across a concurrent destroy.
  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t result = iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_capture_status_t status_internal =
      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  iree_status_t status =
      iree_hal_streaming_capture_status(stream_obj, &status_internal, NULL);
  if (!iree_status_is_ok(status)) {
    result = iree_status_to_fixed_hip_result(status, hipErrorUnknown);
  } else if (status_internal != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    result = hipErrorIllegalState;
  } else {
    status = iree_hal_streaming_begin_capture(stream_obj, capture_mode);
    result = iree_status_to_fixed_hip_result(status, hipErrorUnknown);
  }
  iree_hal_streaming_stream_release(stream_obj);

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

HIPAPI hipError_t hipStreamBeginCaptureToGraph(
    hipStream_t stream, hipGraph_t graph, const hipGraphNode_t* dependencies,
    const void* dependencyData, size_t numDependencies,
    hipStreamCaptureMode mode) {
  (void)dependencyData;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream || !graph || (numDependencies > 0 && !dependencies)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (stream == hipStreamLegacy) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_hal_streaming_capture_mode_t capture_mode;
  switch (mode) {
    case hipStreamCaptureModeGlobal:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
      break;
    case hipStreamCaptureModeThreadLocal:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL;
      break;
    case hipStreamCaptureModeRelaxed:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (!iree_hip_graph_handle_is_live(graph)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  for (size_t i = 0; i < numDependencies; ++i) {
    iree_hal_streaming_graph_node_t* dependency =
        (iree_hal_streaming_graph_node_t*)dependencies[i];
    if (!dependency || dependency->graph != stream_graph) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }

  // Retain explicit streams while capture state is inspected and updated so
  // that a concurrent destroy cannot invalidate the handle mid-operation.
  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t result = iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_capture_status_t status_internal =
      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  iree_status_t status =
      iree_hal_streaming_capture_status(stream_obj, &status_internal, NULL);
  if (!iree_status_is_ok(status)) {
    result = iree_status_to_fixed_hip_result(status, hipErrorUnknown);
  } else if (status_internal != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    result = hipErrorIllegalState;
  } else {
    status = iree_hal_streaming_begin_capture_to_graph(
        stream_obj, stream_graph,
        (iree_hal_streaming_graph_node_t**)dependencies, numDependencies,
        capture_mode);
    result = iree_status_to_fixed_hip_result(status, hipErrorUnknown);
  }
  iree_hal_streaming_stream_release(stream_obj);

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Ends stream capture and returns the captured graph.
//
// Parameters:
//  - stream: [IN] Stream to end capture on.
//  - pGraph: [OUT] Pointer to receive the captured graph.
//
// Returns:
//  - hipSuccess: Capture ended successfully.
//  - hipErrorInvalidValue: Invalid stream or pGraph is NULL.
//  - hipErrorStreamCaptureInvalidated: Capture was invalidated.
//  - hipErrorStreamCaptureUnmatched: Unmatched begin/end.
//  - hipErrorStreamCaptureWrongThread: Wrong thread for thread-local.
//
// Synchronization: Stream returns to normal execution mode.
//
// Capture completion:
// - Stops recording operations to graph.
// - Returns complete graph of captured work.
// - Stream resumes normal execution.
// - Graph ready for instantiation.
//
// Graph contents:
// - All operations between begin/end.
// - Preserved dependencies and ordering.
// - Cross-stream work if mode allows.
// - Event synchronization captured.
//
// Error conditions:
// - Invalid operations during capture.
// - Stream synchronization attempted.
// - Blocking operations encountered.
// - Resource allocation failures.
//
// Graph validation:
// - Check for cycles (must be DAG).
// - Verify all dependencies resolved.
// - Ensure device compatibility.
//
// Post-capture:
// - Graph can be instantiated immediately.
// - Stream can begin new capture.
// - Graph independent of stream lifetime.
//
// Multi-GPU:
// - Captured graph may span devices.
// - Device switches preserved in graph.
//
// Warning: Ensure capture was started with hipStreamBeginCapture.
// Mismatched begin/end causes undefined behavior.
//
// See also: hipStreamBeginCapture, hipGraphInstantiate,
//           hipStreamIsCapturing, hipGraphCreate.
HIPAPI hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorIllegalState);
  }

  if (stream == hipStreamLegacy) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result =
      iree_hip_resolve_registered_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  iree_hal_streaming_graph_t* graph = NULL;
  if (pGraph) {
    *pGraph = NULL;
  }
  if (!pGraph) {
    iree_hal_streaming_capture_status_t capture_status =
        IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
    iree_status_t capture_status_result =
        iree_hal_streaming_capture_status(stream_obj, &capture_status, NULL);
    if (!iree_status_is_ok(capture_status_result)) {
      iree_hal_streaming_stream_release(stream_obj);
      HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(z0, capture_status_result,
                                              hipErrorUnknown);
    }
    if (capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED) {
      iree_hal_streaming_stream_release(stream_obj);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorIllegalState);
    }
  }

  iree_status_t status =
      iree_hal_streaming_end_capture(stream_obj, pGraph ? &graph : NULL);

  if (iree_status_is_ok(status)) {
    if (pGraph) {
      if (!iree_hip_live_graph_register((hipGraph_t)graph)) {
        iree_hal_streaming_graph_release(graph);
        iree_hal_streaming_stream_release(stream_obj);
        IREE_TRACE_ZONE_END(z0);
        HIP_RETURN_ERROR(hipErrorOutOfMemory);
      }
      *pGraph = (hipGraph_t)graph;
    }
  } else {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    hipError_t result = hipErrorInvalidValue;
    switch (status_code) {
      case IREE_STATUS_ABORTED:
        result = hipErrorStreamCaptureUnjoined;
        break;
      case IREE_STATUS_DATA_LOSS:
        result = hipErrorStreamCaptureInvalidated;
        break;
      case IREE_STATUS_FAILED_PRECONDITION:
        result = hipErrorIllegalState;
        break;
      case IREE_STATUS_NOT_FOUND:
        result = hipErrorStreamCaptureUnmatched;
        break;
      case IREE_STATUS_PERMISSION_DENIED:
        result = hipErrorStreamCaptureWrongThread;
        break;
      default:
        result = hipErrorInvalidValue;
        break;
    }
    iree_hal_streaming_stream_release(stream_obj);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  iree_hal_streaming_stream_release(stream_obj);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Queries if a stream is currently capturing.
//
// Parameters:
//  - stream: [IN] Stream to query.
//  - pCaptureStatus: [OUT] Pointer to receive capture status.
//
// Returns:
//  - hipSuccess: Status queried successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorStreamCaptureImplicit: Implicit capture query.
//
// Synchronization: This operation is immediate.
//
// Capture status values:
// - hipStreamCaptureStatusNone: Not capturing.
// - hipStreamCaptureStatusActive: Currently capturing.
// - hipStreamCaptureStatusInvalidated: Capture invalidated.
//
// Status meanings:
// - None: Stream in normal execution mode.
// - Active: Between begin and end capture.
// - Invalidated: Error occurred during capture.
//
// Query behavior:
// - Non-blocking status check.
// - Safe to call anytime.
// - Works with NULL stream.
//
// Use cases:
// - Check before synchronization.
// - Verify capture state.
// - Error detection during capture.
// - Conditional code paths.
//
// Invalidation causes:
// - Invalid API calls during capture.
// - Resource allocation failures.
// - Unsupported operations attempted.
//
// Multi-GPU:
// - Status is per-stream.
// - Cross-device captures tracked.
//
// See also: hipStreamBeginCapture, hipStreamEndCapture,
//           hipStreamGetCaptureInfo.
HIPAPI hipError_t hipStreamIsCapturing(hipStream_t stream,
                                       hipStreamCaptureStatus* pCaptureStatus) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pCaptureStatus) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (!stream) {
    // Use default stream.
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      *pCaptureStatus = hipStreamCaptureStatusNone;
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  bool is_capturing = false;
  iree_status_t status =
      iree_hal_streaming_is_capturing(stream_obj, &is_capturing);

  if (iree_status_is_ok(status)) {
    *pCaptureStatus = is_capturing ? hipStreamCaptureStatusActive
                                   : hipStreamCaptureStatusNone;
  } else {
    *pCaptureStatus = hipStreamCaptureStatusNone;
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets detailed information about stream capture.
//
// Parameters:
//  - stream: [IN] Stream to query.
//  - pCaptureStatus: [OUT] Pointer to receive capture status.
//  - pId: [OUT] Optional pointer to receive capture ID (can be NULL).
//
// Returns:
//  - hipSuccess: Information retrieved successfully.
//  - hipErrorInvalidValue: pCaptureStatus is NULL.
//
// Synchronization: This operation is immediate.
//
// Extended capture information:
// - Status: Current capture state.
// - ID: Unique capture session identifier.
// - ID remains constant during capture.
// - ID changes with each new capture.
//
// Capture ID usage:
// - Track capture sessions.
// - Match related captures.
// - Debug capture issues.
// - Correlate with events.
//
// Status values:
// - None: Not capturing.
// - Active: Currently capturing.
// - Invalidated: Capture failed.
//
// Multi-stream capture:
// - ID shared across captured streams.
// - Helps identify related captures.
// - Cross-stream dependencies tracked.
//
// Multi-GPU:
// - Each device maintains capture IDs.
// - Cross-device captures share ID.
//
// See also: hipStreamIsCapturing, hipStreamBeginCapture,
//           hipStreamEndCapture.
HIPAPI hipError_t hipStreamGetCaptureInfo(
    hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pCaptureStatus) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  bool is_implicit = false;
  hipError_t implicit_result =
      iree_hip_capture_info_query_is_implicit(stream, &is_implicit);
  if (implicit_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(implicit_result);
  }
  if (is_implicit) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  if (!stream) {
    // Use default stream.
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      *pCaptureStatus = hipStreamCaptureStatusNone;
      if (pId) *pId = 0;
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  iree_hal_streaming_capture_status_t status_internal;
  unsigned long long capture_id;
  iree_status_t status = iree_hal_streaming_capture_status(
      stream_obj, &status_internal, &capture_id);

  if (iree_status_is_ok(status)) {
    // Map internal status to HIP status.
    switch (status_internal) {
      case IREE_HAL_STREAMING_CAPTURE_STATUS_NONE:
        *pCaptureStatus = hipStreamCaptureStatusNone;
        break;
      case IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE:
        *pCaptureStatus = hipStreamCaptureStatusActive;
        break;
      case IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED:
        *pCaptureStatus = hipStreamCaptureStatusInvalidated;
        break;
      default:
        *pCaptureStatus = hipStreamCaptureStatusNone;
        break;
    }
    if (pId) {
      *pId = capture_id;
    }
  } else {
    *pCaptureStatus = hipStreamCaptureStatusNone;
    if (pId) *pId = 0;
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets extended stream capture information (v2).
//
// Parameters:
//  - stream: [IN] Stream to query.
//  - captureStatus_out: [OUT] Capture status.
//  - id_out: [OUT] Capture ID (optional, can be NULL).
//  - graph_out: [OUT] Graph being captured (optional, can be NULL).
//  - dependencies_out: [OUT] Dependencies array (optional, can be NULL).
//  - numDependencies_out: [OUT] Number of dependencies (optional, can be NULL).
//
// Returns:
//  - hipSuccess: Information retrieved successfully.
//  - hipErrorInvalidValue: captureStatus_out is NULL.
//  - hipErrorStreamCaptureImplicit: Stream has capture dependencies.
HIPAPI hipError_t hipStreamGetCaptureInfo_v2(
    hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out, hipGraph_t* graph_out,
    const hipGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!captureStatus_out) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  bool is_implicit = false;
  hipError_t implicit_result =
      iree_hip_capture_info_query_is_implicit(stream, &is_implicit);
  if (implicit_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(implicit_result);
  }
  if (is_implicit) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureImplicit);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  if (!stream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      *captureStatus_out = hipStreamCaptureStatusNone;
      if (id_out) *id_out = 0;
      if (graph_out) *graph_out = NULL;
      if (dependencies_out) *dependencies_out = NULL;
      if (numDependencies_out) *numDependencies_out = 0;
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    stream_obj = context->default_stream;
  } else {
    hipError_t init_result = iree_hip_resolve_stream(stream, &stream_obj);
    if (init_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(init_result);
    }
  }

  iree_slim_mutex_lock(&stream_obj->mutex);
  const iree_hal_streaming_capture_status_t capture_status =
      stream_obj->capture_status;
  switch (capture_status) {
    case IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE:
      *captureStatus_out = hipStreamCaptureStatusActive;
      break;
    case IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED:
      *captureStatus_out = hipStreamCaptureStatusInvalidated;
      break;
    case IREE_HAL_STREAMING_CAPTURE_STATUS_NONE:
    default:
      *captureStatus_out = hipStreamCaptureStatusNone;
      break;
  }
  if (id_out) *id_out = stream_obj->capture_id;
  if (graph_out) {
    *graph_out = capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE
                     ? NULL
                     : (hipGraph_t)stream_obj->capture_graph;
  }
  if (dependencies_out) {
    *dependencies_out =
        capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE
            ? NULL
            : (const hipGraphNode_t*)stream_obj->capture_dependencies;
  }
  if (numDependencies_out) {
    *numDependencies_out =
        capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE
            ? 0
            : stream_obj->capture_dependency_count;
  }
  iree_slim_mutex_unlock(&stream_obj->mutex);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Updates dependencies for the next captured node.
//
// Parameters:
//  - stream: [IN] Stream currently capturing.
//  - dependencies: [IN] Array of nodes for next operation to depend on.
//  - numDependencies: [IN] Number of dependencies.
//  - flags: [IN] Update flags (must be 0).
//
// Returns:
//  - hipSuccess: Dependencies updated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorStreamCaptureInvalidated: Stream not capturing.
//
// Synchronization: Affects next captured operation.
//
// Dependency injection:
// - Modifies dependencies for next node.
// - Does not affect already captured nodes.
// - Creates edges from specified nodes.
// - Enables complex graph patterns.
//
// Use cases:
// - Connect independent capture sequences.
// - Create custom synchronization points.
// - Merge parallel capture paths.
// - Fix missing dependencies.
//
// Capture flow:
// - Call during active capture.
// - Next operation depends on specified nodes.
// - Subsequent operations follow normal flow.
// - One-time effect per call.
//
// Dependency rules:
// - Nodes must be from same capture.
// - Cannot create cycles.
// - Dependencies validated at capture.
//
// Advanced patterns:
// - Fork-join parallelism.
// - Conditional dependencies.
// - Dynamic graph construction.
//
// Multi-GPU:
// - Dependencies can cross devices.
// - Ensures proper synchronization.
//
// Warning: Invalid dependencies may invalidate entire capture.
// Verify nodes are from current capture session.
//
// See also: hipStreamBeginCapture, hipGraphAddEmptyNode,
//           hipStreamEndCapture.
HIPAPI hipError_t hipStreamUpdateCaptureDependencies(
    hipStream_t stream, hipGraphNode_t* dependencies, size_t numDependencies,
    unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream || (numDependencies > 0 && !dependencies)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (stream == hipStreamLegacy) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t init_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_capture_dependencies_mode_t mode;
  switch ((hipStreamUpdateCaptureDependenciesFlags)flags) {
    case hipStreamAddCaptureDependencies:
      mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD;
      break;
    case hipStreamSetCaptureDependencies:
      mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_status_t status = iree_hal_streaming_update_capture_dependencies(
      stream_obj, (iree_hal_streaming_graph_node_t**)dependencies,
      numDependencies, mode);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(status_code == IREE_STATUS_FAILED_PRECONDITION
                         ? hipErrorIllegalState
                         : hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Memory pool management
//===----------------------------------------------------------------------===//

// Creates a memory pool for stream-ordered allocations.
//
// Parameters:
//  - pool: [OUT] Pointer to receive the created pool handle.
//  - poolProps: [IN] Pool properties structure.
//
// Returns:
//  - hipSuccess: Pool created successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorNotSupported: Pool type not supported.
//  - hipErrorMemoryAllocation: Insufficient resources.
//
// Synchronization: This operation is synchronous.
//
// Pool properties (hipMemPoolProps):
// - allocType: Allocation type (hipMemAllocationTypePinned, etc.).
// - handleTypes: Export handle types supported.
// - location: Memory location hints.
// - reserved: Reserved for future use.
//
// Pool characteristics:
// - Stream-ordered allocation/deallocation.
// - Automatic memory reuse.
// - Reduced allocation overhead.
// - Better memory utilization.
//
// Allocation behavior:
// - Memory allocated on-demand.
// - Freed memory returned to pool.
// - Pool manages fragmentation.
// - Supports suballocation.
//
// Performance benefits:
// - Faster than hipMalloc/hipFree.
// - Reduced kernel launch overhead.
// - Better memory locality.
// - Efficient resource usage.
//
// Pool sharing:
// - Can be shared between streams.
// - Supports IPC export/import.
// - Multi-process capable.
//
// Multi-GPU:
// - Pool associated with specific device.
// - Cross-device access configurable.
// - Peer access rules apply.
//
// Warning: Pool must be destroyed when no longer needed.
// Active allocations prevent pool destruction.
//
// See also: hipMemPoolDestroy, hipMallocFromPoolAsync,
//           hipMemPoolSetAttribute, hipDeviceSetMemPool.
HIPAPI hipError_t hipMemPoolCreate(hipMemPool_t* pool,
                                   const hipMemPoolProps* poolProps) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !poolProps) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pool = NULL;

  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  hipError_t props_result = iree_hip_validate_mem_pool_props(poolProps);
  if (props_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(props_result);
  }
  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(poolProps->location.id);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  hrx_mem_pool_props_t props = {
      .alloc_handle_type =
          iree_hip_mem_handle_type_to_uint(poolProps->handleTypes),
      .location_type =
          iree_hip_mem_location_type_to_uint(poolProps->location.type),
      .location_id = poolProps->location.id,
      .max_size = poolProps->maxSize,
  };

  hrx_mem_pool_t mem_pool = NULL;
  iree_status_t status =
      HRX_CALL(hrx_mem_pool_create(device->hrx_device, &props, &mem_pool));

  hipError_t result = iree_status_to_hip_result(status);
  if (result == hipSuccess) {
    result = iree_hip_mem_pool_create_handle(mem_pool, poolProps->location.id,
                                             poolProps->allocType,
                                             /*is_default=*/false, pool);
    if (result != hipSuccess) {
      hrx_mem_pool_release(mem_pool);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Destroys a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle to destroy.
//
// Returns:
//  - hipSuccess: Pool destroyed successfully.
//  - hipErrorInvalidValue: pool is NULL or invalid.
//
// Synchronization: Waits for all allocations to be freed.
//
// Destruction behavior:
// - Releases all pool resources.
// - Invalidates pool handle.
// - Pending operations must complete.
// - Active allocations must be freed.
//
// Resource cleanup:
// - Returns memory to system.
// - Closes IPC handles.
// - Releases device resources.
//
// Warning: Ensure all allocations from pool are freed.
// Destroying pool with active allocations causes errors.
//
// See also: hipMemPoolCreate, hipFreeAsync,
//           hipMemPoolTrimTo.
HIPAPI hipError_t hipMemPoolDestroy(hipMemPool_t pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  struct hipMemPool_st* removed_pool = NULL;
  if (!iree_hip_mem_pool_registry_remove(pool, &removed_pool)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (removed_pool->allocation_type == hipMemAllocationTypePinned) {
    iree_hal_streaming_device_t* device =
        iree_hal_streaming_device_entry(removed_pool->device_ordinal);
    if (device) {
      iree_hal_streaming_device_reset_mem_pool_if_current(
          device, removed_pool->hrx_pool);
    }
  }
  iree_hip_mem_pool_release(removed_pool);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets an attribute of a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - attr: [IN] Attribute to set (hipMemPool_attribute enum).
//  - value: [IN] Pointer to new attribute value.
//
// Returns:
//  - hipSuccess: Attribute set successfully.
//  - hipErrorInvalidValue: Invalid pool, attribute, or value.
//  - hipErrorNotSupported: Attribute cannot be modified.
//
// Synchronization: This operation is synchronous.
//
// Settable attributes:
// - hipMemPoolAttrReleaseThreshold: Memory release threshold.
// - hipMemPoolAttrReuseFollowEventDependencies: Event-based reuse.
// - hipMemPoolAttrReuseAllowOpportunistic: Opportunistic reuse.
// - hipMemPoolAttrReuseAllowInternalDependencies: Internal reuse.
//
// Attribute effects:
// - ReleaseThreshold: Bytes to retain before OS release.
// - ReuseFollowEventDependencies: Honor event dependencies.
// - ReuseAllowOpportunistic: Aggressive memory reuse.
// - ReuseAllowInternalDependencies: Reuse within operations.
//
// Performance tuning:
// - Adjust threshold for memory vs performance.
// - Enable opportunistic reuse for throughput.
// - Disable dependencies for deterministic behavior.
//
// Multi-GPU:
// - Attributes apply to pool on all devices.
// - May affect cross-device synchronization.
//
// Warning: Changing attributes affects all future allocations.
// Existing allocations retain original behavior.
//
// See also: hipMemPoolGetAttribute, hipMemPoolCreate,
//           hipMallocFromPoolAsync.
HIPAPI hipError_t hipMemPoolSetAttribute(hipMemPool_t pool,
                                         hipMemPool_attribute attr,
                                         void* value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  uint64_t attr_value = 0;
  switch (attr) {
    case hipMemPoolAttrReleaseThreshold:
    case hipMemPoolAttrReservedMemHigh:
    case hipMemPoolAttrUsedMemHigh:
      attr_value = *(uint64_t*)value;
      break;
    case hipMemPoolAttrReuseFollowEventDependencies:
    case hipMemPoolAttrReuseAllowOpportunistic:
    case hipMemPoolAttrReuseAllowInternalDependencies:
      attr_value = *(int*)value;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t result = iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  hrx_mem_pool_t hrx_pool = NULL;
  result = iree_hip_mem_pool_retain_backend(pool_handle, &hrx_pool);
  hrx_mem_pool_attr_t hrx_attr = iree_hip_mempool_attr_to_hrx(attr);
  if (result == hipSuccess) {
    iree_status_t status =
        HRX_CALL(hrx_mem_pool_set_attribute(hrx_pool, hrx_attr, attr_value));
    result = iree_status_to_hip_result(status);
  }
  hrx_mem_pool_release(hrx_pool);
  iree_hip_mem_pool_release(pool_handle);

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Gets an attribute of a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - attr: [IN] Attribute to query (hipMemPool_attribute enum).
//  - value: [OUT] Pointer to receive attribute value.
//
// Returns:
//  - hipSuccess: Attribute retrieved successfully.
//  - hipErrorInvalidValue: Invalid pool, attribute, or value.
//
// Synchronization: This operation is immediate.
//
// Queryable attributes:
// - hipMemPoolAttrReleaseThreshold: Current release threshold.
// - hipMemPoolAttrReservedMemCurrent: Currently reserved memory.
// - hipMemPoolAttrReservedMemHigh: Peak reserved memory.
// - hipMemPoolAttrUsedMemCurrent: Currently used memory.
// - hipMemPoolAttrUsedMemHigh: Peak used memory.
// - hipMemPoolAttrReuseFollowEventDependencies: Event reuse setting.
// - hipMemPoolAttrReuseAllowOpportunistic: Opportunistic reuse.
// - hipMemPoolAttrReuseAllowInternalDependencies: Internal reuse.
//
// Statistics:
// - Reserved: Memory held by pool from OS.
// - Used: Memory allocated to application.
// - High watermarks: Peak usage tracking.
//
// Performance monitoring:
// - Track memory efficiency.
// - Identify memory pressure.
// - Optimize pool configuration.
//
// See also: hipMemPoolSetAttribute, hipMemPoolCreate,
//           hipMemPoolTrimTo.
HIPAPI hipError_t hipMemPoolGetAttribute(hipMemPool_t pool,
                                         hipMemPool_attribute attr,
                                         void* value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t result = iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  hrx_mem_pool_t hrx_pool = NULL;
  result = iree_hip_mem_pool_retain_backend(pool_handle, &hrx_pool);

  uint64_t attr_value = 0;
  hrx_mem_pool_attr_t hrx_attr = iree_hip_mempool_attr_to_hrx(attr);
  iree_status_t status = iree_ok_status();
  if (result == hipSuccess) {
    status =
        HRX_CALL(hrx_mem_pool_get_attribute(hrx_pool, hrx_attr, &attr_value));
  }

  if (result == hipSuccess && iree_status_is_ok(status)) {
    switch (attr) {
      case hipMemPoolAttrReleaseThreshold:
      case hipMemPoolAttrReservedMemCurrent:
      case hipMemPoolAttrReservedMemHigh:
      case hipMemPoolAttrUsedMemCurrent:
      case hipMemPoolAttrUsedMemHigh:
        *(size_t*)value = (size_t)attr_value;
        break;
      case hipMemPoolAttrReuseFollowEventDependencies:
      case hipMemPoolAttrReuseAllowOpportunistic:
      case hipMemPoolAttrReuseAllowInternalDependencies:
        *(int*)value = (int)attr_value;
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
        break;
    }
  }

  if (result == hipSuccess) result = iree_status_to_hip_result(status);
  hrx_mem_pool_release(hrx_pool);
  iree_hip_mem_pool_release(pool_handle);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets memory access permissions for a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - map: [IN] Array of access descriptors.
//  - count: [IN] Number of access descriptors.
//
// Returns:
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidDevice: Invalid or unsupported device access.
//
// See also: hipMemPoolGetAccess.
HIPAPI hipError_t hipMemPoolSetAccess(hipMemPool_t pool,
                                      const hipMemAccessDesc* map,
                                      size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || (count > 0 && !map)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t result = iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  int device_count = 0;
  result = hipGetDeviceCount(&device_count);
  if (result == hipSuccess && count > (size_t)device_count) {
    result = hipErrorInvalidDevice;
  }
  for (size_t i = 0; result == hipSuccess && i < count; ++i) {
    if (map[i].location.type != hipMemLocationTypeDevice ||
        (map[i].flags != hipMemAccessFlagsProtNone &&
         map[i].flags != hipMemAccessFlagsProtRead &&
         map[i].flags != hipMemAccessFlagsProtReadWrite)) {
      result = hipErrorInvalidValue;
    } else if (map[i].location.id < 0 || map[i].location.id >= device_count) {
      result = hipErrorInvalidDevice;
    } else if (map[i].location.id != pool_handle->device_ordinal) {
      // Cross-device pool mappings require direct peer access. The streaming
      // runtime does not advertise a peer path without a backend capability
      // source that can establish that mapping.
      result = hipErrorNotSupported;
    } else if (map[i].flags != hipMemAccessFlagsProtReadWrite) {
      result = hipErrorInvalidDevice;
    }
  }
  iree_hip_mem_pool_release(pool_handle);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Gets memory access permissions for a memory pool.
//
// Parameters:
//  - flags: [OUT] Pointer to receive access flags.
//  - pool: [IN] Memory pool handle.
//  - location: [IN] Memory location to query.
//
// Returns:
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorInvalidDevice: Invalid or unsupported device access.
//
// See also: hipMemPoolSetAccess.
HIPAPI hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags,
                                      hipMemPool_t pool,
                                      hipMemLocation* location) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!flags || !pool || !location) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t result = iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }

  if (location->type != hipMemLocationTypeDevice) {
    result = hipErrorInvalidValue;
  } else {
    int device_count = 0;
    result = hipGetDeviceCount(&device_count);
    if (result == hipSuccess &&
        (location->id < 0 || location->id >= device_count)) {
      result = hipErrorInvalidValue;
    }
    if (result == hipSuccess && location->id != pool_handle->device_ordinal) {
      result = hipErrorNotSupported;
    }
    if (result == hipSuccess) {
      *flags = hipMemAccessFlagsProtReadWrite;
    }
  }
  iree_hip_mem_pool_release(pool_handle);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Trims a memory pool to specified size.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - minBytesToKeep: [IN] Minimum bytes to retain.
//
// Returns:
//  - hipSuccess: Pool trimmed successfully.
//  - hipErrorInvalidValue: Invalid pool.
//
// Releases unused memory back to the system.
//
// See also: hipMemPoolGetAttribute.
HIPAPI hipError_t hipMemPoolTrimTo(hipMemPool_t pool, size_t minBytesToKeep) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t result = iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(result);
  }
  hrx_mem_pool_t hrx_pool = NULL;
  result = iree_hip_mem_pool_retain_backend(pool_handle, &hrx_pool);
  if (result == hipSuccess) {
    iree_status_t status =
        iree_hal_streaming_memory_release_completed_async_frees_from_pool(
            hrx_pool);
    status = iree_status_join(
        status, HRX_CALL(hrx_mem_pool_trim(hrx_pool, minBytesToKeep)));
    result = iree_status_to_hip_result(status);
  }
  hrx_mem_pool_release(hrx_pool);
  iree_hip_mem_pool_release(pool_handle);

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Exports a memory pool to a shareable handle.
//
// Parameters:
//  - handle_out: [OUT] Pointer to receive handle.
//  - pool: [IN] Memory pool to export.
//  - handleType: [IN] Type of handle to create.
//  - flags: [IN] Export flags.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolImportFromShareableHandle.
HIPAPI hipError_t hipMemPoolExportToShareableHandle(
    void* handle_out, hipMemPool_t pool, hipMemAllocationHandleType handleType,
    unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Imports a memory pool from a shareable handle.
//
// Parameters:
//  - pool_out: [OUT] Pointer to receive pool.
//  - handle: [IN] Shareable handle.
//  - handleType: [IN] Type of handle.
//  - flags: [IN] Import flags.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolExportToShareableHandle.
HIPAPI hipError_t hipMemPoolImportFromShareableHandle(
    hipMemPool_t* pool_out, void* handle, hipMemAllocationHandleType handleType,
    unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Exports a pointer from a memory pool for IPC.
//
// Parameters:
//  - shareData_out: [OUT] Export data structure.
//  - ptr: [IN] Pointer to export.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolImportPointer.
HIPAPI hipError_t
hipMemPoolExportPointer(hipMemPoolPtrExportData* shareData_out, void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Imports a pointer into a memory pool from IPC.
//
// Parameters:
//  - ptr_out: [OUT] Pointer to receive imported pointer.
//  - pool: [IN] Target memory pool.
//  - shareData: [IN] Import data structure.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolExportPointer.
HIPAPI hipError_t hipMemPoolImportPointer(void** ptr_out, hipMemPool_t pool,
                                          hipMemPoolPtrExportData* shareData) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Sets the default memory pool for a device.
//
// Parameters:
//  - device: [IN] Device ID.
//  - pool: [IN] Memory pool to set as default.
//
// Returns:
//  - hipSuccess: Default pool set.
//  - hipErrorInvalidDevice: Invalid device.
//
// See also: hipDeviceGetMemPool, hipMallocAsync.
HIPAPI hipError_t hipDeviceSetMemPool(int device, hipMemPool_t pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  iree_status_t status =
      iree_hal_streaming_device_ensure_default_mem_pool(device_obj);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(iree_status_to_hip_result(status));
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t result = iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (result == hipSuccess &&
      (pool_handle->device_ordinal != device ||
       pool_handle->allocation_type != hipMemAllocationTypePinned)) {
    result = hipErrorInvalidValue;
  }
  hrx_mem_pool_t hrx_pool = NULL;
  if (result == hipSuccess) {
    result = iree_hip_mem_pool_retain_backend(pool_handle, &hrx_pool);
  }
  if (result == hipSuccess) {
    iree_hal_streaming_device_set_mem_pool(device_obj, hrx_pool);
  }
  hrx_mem_pool_release(hrx_pool);
  iree_hip_mem_pool_release(pool_handle);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Gets the current memory pool for a device.
//
// Parameters:
//  - pool: [OUT] Pointer to receive pool handle.
//  - device: [IN] Device ID.
//
// Returns:
//  - hipSuccess: Pool retrieved.
//  - hipErrorInvalidValue: pool is NULL.
//  - hipErrorInvalidDevice: Invalid device.
//
// See also: hipDeviceSetMemPool.
HIPAPI hipError_t hipDeviceGetMemPool(hipMemPool_t* pool, int device) {
  if (!pool) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  iree_status_t status =
      iree_hal_streaming_device_ensure_default_mem_pool(device_obj);
  if (!iree_status_is_ok(status)) {
    HIP_RETURN_ERROR(iree_status_to_hip_result(status));
  }
  HIP_RETURN_ERROR(iree_hip_mem_pool_get_current(device_obj, device, pool));
}

// Gets the default memory pool for a device.
//
// Parameters:
//  - pool_out: [OUT] Pointer to receive default pool.
//  - device: [IN] Device ID.
//
// Returns:
//  - hipSuccess: Default pool retrieved.
//  - hipErrorInvalidValue: pool_out is NULL.
//  - hipErrorInvalidDevice: Invalid device.
//
// Default pool is used by hipMallocAsync when no pool specified.
//
// See also: hipDeviceSetMemPool, hipMallocAsync.
HIPAPI hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* pool_out,
                                             int device) {
  if (!pool_out) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  iree_status_t status =
      iree_hal_streaming_device_ensure_default_mem_pool(device_obj);
  if (!iree_status_is_ok(status)) {
    HIP_RETURN_ERROR(iree_status_to_hip_result(status));
  }
  HIP_RETURN_ERROR(iree_hip_mem_pool_get_default(
      device, hipMemAllocationTypePinned, pool_out));
}

HIPAPI hipError_t hipMemGetMemPool(hipMemPool_t* pool, hipMemLocation* location,
                                   hipMemAllocationType type) {
  if (!pool || !location ||
      !iree_hip_mem_pool_allocation_type_is_supported(type)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *pool = NULL;

  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) HIP_RETURN_ERROR(init_result);
  if (location->type != hipMemLocationTypeDevice || location->id < 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(location->id);
  if (!device) HIP_RETURN_ERROR(hipErrorInvalidValue);
  iree_status_t status =
      iree_hal_streaming_device_ensure_default_mem_pool(device);
  if (!iree_status_is_ok(status)) {
    HIP_RETURN_ERROR(iree_status_to_hip_result(status));
  }

  if (type == hipMemAllocationTypePinned) {
    HIP_RETURN_ERROR(iree_hip_mem_pool_get_current(device, location->id, pool));
  }
  HIP_RETURN_ERROR(iree_hip_mem_pool_get_managed_selection(location->id, pool));
}

HIPAPI hipError_t hipMemSetMemPool(hipMemLocation* location,
                                   hipMemAllocationType type,
                                   hipMemPool_t pool) {
  if (!location || !pool ||
      !iree_hip_mem_pool_allocation_type_is_supported(type)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t init_result = iree_hip_ensure_initialized();
  if (init_result != hipSuccess) HIP_RETURN_ERROR(init_result);
  if (location->type != hipMemLocationTypeDevice || location->id < 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(location->id);
  if (!device) HIP_RETURN_ERROR(hipErrorInvalidValue);

  iree_status_t status =
      iree_hal_streaming_device_ensure_default_mem_pool(device);
  if (!iree_status_is_ok(status)) {
    HIP_RETURN_ERROR(iree_status_to_hip_result(status));
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t result = iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (result != hipSuccess) HIP_RETURN_ERROR(result);
  if (pool_handle->device_ordinal != location->id ||
      pool_handle->allocation_type != type) {
    iree_hip_mem_pool_release(pool_handle);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (type == hipMemAllocationTypePinned) {
    hrx_mem_pool_t hrx_pool = NULL;
    result = iree_hip_mem_pool_retain_backend(pool_handle, &hrx_pool);
    if (result == hipSuccess) {
      iree_hal_streaming_device_set_mem_pool(device, hrx_pool);
    }
    hrx_mem_pool_release(hrx_pool);
  } else {
    result = iree_hip_mem_pool_set_managed_selection(location->id, pool_handle);
  }
  iree_hip_mem_pool_release(pool_handle);
  HIP_RETURN_ERROR(result);
}

// Allocates memory asynchronously from the default pool.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive allocated memory.
//  - size: [IN] Size in bytes to allocate.
//  - stream: [IN] Stream for ordered allocation.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Stream-ordered allocation from device's default pool.
// Memory lifetime tied to stream operations.
//
// See also: hipFreeAsync, hipMallocFromPoolAsync.
HIPAPI hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG("[HIP_API] hipMallocAsync: size=%zu stream=%p\n", size,
                (void*)stream);

  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *ptr = NULL;

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t resolve_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (resolve_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(resolve_result);
  }

  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    if (size == 0) {
      *ptr = NULL;
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    hipMemAllocNodeParams params;
    memset(&params, 0, sizeof(params));
    params.bytesize = size;
    params.poolProps.allocType = hipMemAllocationTypePinned;
    params.poolProps.location.type = hipMemLocationTypeDevice;
    params.poolProps.location.id = (int)stream_obj->context->device_ordinal;

    hipGraphNode_t node = NULL;
    hipError_t result = hipGraphAddMemAllocNode(
        &node, (hipGraph_t)stream_obj->capture_graph,
        (const hipGraphNode_t*)stream_obj->capture_dependencies,
        stream_obj->capture_dependency_count, &params);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_capture_set_last_node(
            stream_obj, (iree_hal_streaming_graph_node_t*)node),
        hipErrorInvalidValue);
    *ptr = params.dptr;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  if (size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  // Allocate eagerly from the stream's selected pool. A pending free on this
  // stream can reuse its backing only after every preceding operation.
  hrx_mem_pool_t pool = NULL;
  hipError_t result = iree_hip_current_mem_pool(stream_obj->context, &pool);
  if (result == hipSuccess) {
    result = iree_hip_malloc_from_pool(stream_obj->context, pool, size,
                                       stream_obj, ptr);
  }
  hrx_mem_pool_release(pool);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Allocates memory asynchronously from a specific pool.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive allocated memory.
//  - size: [IN] Size in bytes to allocate.
//  - pool: [IN] Memory pool to allocate from.
//  - stream: [IN] Stream for ordered allocation.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Stream-ordered allocation with explicit pool selection.
//
// See also: hipFreeAsync, hipMallocAsync, hipMemPoolCreate.
HIPAPI hipError_t hipMallocFromPoolAsync(void** ptr, size_t size,
                                         hipMemPool_t pool,
                                         hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  HIP_DEBUG_LOG(
      "[HIP_API] hipMallocFromPoolAsync: size=%zu pool=%p stream=%p\n", size,
      (void*)pool, (void*)stream);

  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *ptr = NULL;
  if (!pool) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t stream_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (stream_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(stream_result);
  }
  if (size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  struct hipMemPool_st* pool_handle = NULL;
  hipError_t pool_result =
      iree_hip_mem_pool_registry_acquire(pool, &pool_handle);
  if (pool_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(pool_result);
  }

  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    iree_hip_mem_pool_release(pool_handle);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorStreamCaptureUnsupported);
  }

  // The allocation's backing is materialized before this call returns. A pool
  // owned by another device therefore does not require foreign-stream work to
  // make the returned address available, but it cannot reuse a free cached by
  // that stream's context.
  iree_hal_streaming_context_t* allocation_context = stream_obj->context;
  iree_hal_streaming_stream_t* allocation_stream = stream_obj;
  if (pool_handle->device_ordinal != (int)stream_obj->context->device_ordinal) {
    iree_hal_streaming_device_t* pool_device =
        iree_hal_streaming_device_entry(pool_handle->device_ordinal);
    if (!pool_device) {
      pool_result = hipErrorInvalidDevice;
    } else {
      iree_status_t status =
          iree_hal_streaming_device_get_or_create_primary_context(
              pool_device, &allocation_context);
      if (!iree_status_is_ok(status)) {
        pool_result = iree_status_to_hip_result(status);
      } else {
        allocation_stream = NULL;
      }
    }
  }

  hrx_mem_pool_t hrx_pool = NULL;
  if (pool_result == hipSuccess) {
    pool_result = iree_hip_mem_pool_retain_backend(pool_handle, &hrx_pool);
  }
  if (pool_result == hipSuccess) {
    pool_result = iree_hip_malloc_from_pool(allocation_context, hrx_pool, size,
                                            allocation_stream, ptr);
  }
  hrx_mem_pool_release(hrx_pool);
  iree_hip_mem_pool_release(pool_handle);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(pool_result);
}

// Frees memory asynchronously.
//
// Parameters:
//  - ptr: [IN] Pointer to free (can be NULL).
//  - stream: [IN] Stream for ordered deallocation.
//
// Returns:
//  - hipSuccess: Memory freed successfully.
//  - hipErrorInvalidValue: Invalid stream.
//
// Stream-ordered deallocation. Memory returned to pool
// when stream reaches this operation.
//
// See also: hipMallocAsync, hipMallocFromPoolAsync.
HIPAPI hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t* stream_obj = NULL;
  hipError_t resolve_result = iree_hip_resolve_stream(stream, &stream_obj);
  if (resolve_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(resolve_result);
  }

  if (stream_obj->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    if (!ptr) {
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    hipGraphNode_t node = NULL;
    hipError_t result = hipGraphAddMemFreeNode(
        &node, (hipGraph_t)stream_obj->capture_graph,
        (const hipGraphNode_t*)stream_obj->capture_dependencies,
        stream_obj->capture_dependency_count, ptr);
    if (result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(result);
    }
    HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_capture_set_last_node(
            stream_obj, (iree_hal_streaming_graph_node_t*)node),
        hipErrorInvalidValue);
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }

  iree_status_t status = iree_hal_streaming_memory_free_device_async(
      stream_obj->context, (iree_hal_streaming_deviceptr_t)ptr, stream_obj);
  hipError_t result;
  if (iree_status_is_ok(status)) {
    result = hipSuccess;
  } else if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
    iree_status_free(status);
    result = hipErrorInvalidValue;
  } else {
    result = iree_status_to_hip_result(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// Gets a string describing the error code.
//
// Parameters:
//  - error: [IN] Error code to get description for.
//
// Returns: Pointer to a null-terminated string describing the error.
//
// Synchronization: This operation is synchronous.
//
// String behavior:
// - Returns a static string, do not free.
//===----------------------------------------------------------------------===//
// Virtual memory management
//===----------------------------------------------------------------------===//

static bool iree_hip_mem_allocation_handle_type_is_valid(
    hipMemAllocationHandleType handle_type) {
  switch (handle_type) {
    case hipMemHandleTypeNone:
    case hipMemHandleTypePosixFileDescriptor:
    case hipMemHandleTypeWin32:
    case hipMemHandleTypeWin32Kmt:
      return true;
    default:
      return false;
  }
}

static hipError_t iree_hip_validate_mem_allocation_prop(
    const hipMemAllocationProp* prop) {
  if (!prop) return hipErrorInvalidValue;
  if (prop->type != hipMemAllocationTypePinned) {
    return hipErrorInvalidValue;
  }
  if (!iree_hip_mem_allocation_handle_type_is_valid(
          prop->requestedHandleType)) {
    return hipErrorInvalidValue;
  }
  if (prop->location.type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }

  int device_count = 0;
  hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) return count_result;
  if (prop->location.id < 0 || prop->location.id >= device_count) {
    return hipErrorInvalidDevice;
  }
  return hipSuccess;
}

static const size_t iree_hip_vmm_allocation_granularity = 64 * 1024;

static bool iree_hip_size_is_vmm_granularity_multiple(size_t size) {
  return size != 0 && (size % iree_hip_vmm_allocation_granularity) == 0;
}

static bool iree_hip_is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// Reserves virtual address space.
HIPAPI hipError_t hipMemAddressReserve(void** ptr, size_t size,
                                       size_t alignment, void* addr,
                                       unsigned long long flags) {
  if (!ptr || addr || flags != 0 ||
      !iree_hip_size_is_vmm_granularity_multiple(size)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (alignment == 0) {
    alignment = iree_hip_vmm_allocation_granularity;
  }
  if (!iree_hip_is_power_of_two(alignment)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }

  *ptr = NULL;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Frees reserved virtual address space.
HIPAPI hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  if (!devPtr || !iree_hip_size_is_vmm_granularity_multiple(size)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Creates a generic allocation handle.
HIPAPI hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle,
                               size_t size, const hipMemAllocationProp* prop,
                               unsigned long long flags) {
  if (!handle || flags != 0 ||
      !iree_hip_size_is_vmm_granularity_multiple(size)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t prop_result = iree_hip_validate_mem_allocation_prop(prop);
  if (prop_result != hipSuccess) {
    HIP_RETURN_ERROR(prop_result);
  }

  *handle = NULL;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Releases a generic allocation handle.
HIPAPI hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
  if (!handle) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Maps virtual memory to a physical allocation.
HIPAPI hipError_t hipMemMap(void* ptr, size_t size, size_t offset,
                            hipMemGenericAllocationHandle_t handle,
                            unsigned long long flags) {
  if (!ptr || flags != 0 || offset != 0 || !handle ||
      !iree_hip_size_is_vmm_granularity_multiple(size)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Unmaps virtual memory.
HIPAPI hipError_t hipMemUnmap(void* ptr, size_t size) {
  if (!ptr || !iree_hip_size_is_vmm_granularity_multiple(size)) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Sets memory access permissions.
HIPAPI hipError_t hipMemSetAccess(void* ptr, size_t size,
                                  const hipMemAccessDesc* desc, size_t count) {
  if (!ptr || !iree_hip_size_is_vmm_granularity_multiple(size) || !desc ||
      count == 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  for (size_t i = 0; i < count; ++i) {
    if (desc[i].location.type != hipMemLocationTypeDevice ||
        desc[i].location.id < 0) {
      HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
    switch (desc[i].flags) {
      case hipMemAccessFlagsProtNone:
      case hipMemAccessFlagsProtRead:
      case hipMemAccessFlagsProtReadWrite:
        break;
      default:
        HIP_RETURN_ERROR(hipErrorInvalidValue);
    }
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Gets memory access permissions.
HIPAPI hipError_t hipMemGetAccess(unsigned long long* flags,
                                  const hipMemLocation* location, void* ptr) {
  if (!flags || !location || !ptr ||
      location->type != hipMemLocationTypeDevice || location->id < 0) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Gets allocation granularity.
HIPAPI hipError_t hipMemGetAllocationGranularity(
    size_t* granularity, const hipMemAllocationProp* prop,
    hipMemAllocationGranularity_flags option) {
  if (!granularity) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  hipError_t prop_result = iree_hip_validate_mem_allocation_prop(prop);
  if (prop_result != hipSuccess) {
    HIP_RETURN_ERROR(prop_result);
  }
  switch (option) {
    case hipMemAllocationGranularityMinimum:
    case hipMemAllocationGranularityRecommended:
      *granularity = iree_hip_vmm_allocation_granularity;
      return hipSuccess;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
}

// Gets properties from allocation handle.
HIPAPI hipError_t hipMemGetAllocationPropertiesFromHandle(
    hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle) {
  if (!prop || !handle) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Exports an allocation handle to a shareable handle.
HIPAPI hipError_t hipMemExportToShareableHandle(
    void* shareableHandle, hipMemGenericAllocationHandle_t handle,
    hipMemAllocationHandleType handleType, unsigned long long flags) {
  (void)shareableHandle;
  (void)handle;
  (void)handleType;
  (void)flags;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Imports an allocation handle from a shareable handle.
HIPAPI hipError_t hipMemImportFromShareableHandle(
    hipMemGenericAllocationHandle_t* handle, void* osHandle,
    hipMemAllocationHandleType shHandleType) {
  (void)handle;
  (void)osHandle;
  (void)shHandleType;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Retains an allocation handle from an address.
HIPAPI hipError_t hipMemRetainAllocationHandle(
    hipMemGenericAllocationHandle_t* handle, void* addr) {
  (void)handle;
  (void)addr;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// - String remains valid for program lifetime.
// - Returns "unknown error" for unrecognized codes.
// - String is in English.
//
// Usage pattern:
// ```c
// hipError_t err = hipMalloc(&ptr, size);
// if (err != hipSuccess) {
//   printf("HIP error: %s\n", hipGetErrorString(err));
// }
// ```
//
// See also: hipGetErrorName, hipGetLastError.
HIPAPI const char* hipGetErrorString(hipError_t error) {
  switch (error) {
    case hipSuccess:
      return "hipSuccess";
    case hipErrorInvalidValue:
      return "hipErrorInvalidValue";
    case hipErrorOutOfMemory:
      return "hipErrorOutOfMemory";
    case hipErrorNotInitialized:
      return "hipErrorNotInitialized";
    case hipErrorDeinitialized:
      return "hipErrorDeinitialized";
    case hipErrorProfilerDisabled:
      return "hipErrorProfilerDisabled";
    case hipErrorProfilerNotInitialized:
      return "hipErrorProfilerNotInitialized";
    case hipErrorProfilerAlreadyStarted:
      return "hipErrorProfilerAlreadyStarted";
    case hipErrorProfilerAlreadyStopped:
      return "hipErrorProfilerAlreadyStopped";
    case hipErrorInvalidConfiguration:
      return "hipErrorInvalidConfiguration";
    case hipErrorInvalidSymbol:
      return "hipErrorInvalidSymbol";
    case hipErrorInvalidDevicePointer:
      return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection:
      return "hipErrorInvalidMemcpyDirection";
    case hipErrorInsufficientDriver:
      return "hipErrorInsufficientDriver";
    case hipErrorMissingConfiguration:
      return "hipErrorMissingConfiguration";
    case hipErrorPriorLaunchFailure:
      return "hipErrorPriorLaunchFailure";
    case hipErrorInvalidDeviceFunction:
      return "hipErrorInvalidDeviceFunction";
    case hipErrorNoDevice:
      return "hipErrorNoDevice";
    case hipErrorInvalidDevice:
      return "hipErrorInvalidDevice";
    case hipErrorInvalidImage:
      return "hipErrorInvalidImage";
    case hipErrorInvalidContext:
      return "hipErrorInvalidContext";
    case hipErrorContextAlreadyCurrent:
      return "hipErrorContextAlreadyCurrent";
    case hipErrorMapFailed:
      return "hipErrorMapFailed";
    case hipErrorUnmapFailed:
      return "hipErrorUnmapFailed";
    case hipErrorArrayIsMapped:
      return "hipErrorArrayIsMapped";
    case hipErrorAlreadyMapped:
      return "hipErrorAlreadyMapped";
    case hipErrorNoBinaryForGpu:
      return "hipErrorNoBinaryForGpu";
    case hipErrorAlreadyAcquired:
      return "hipErrorAlreadyAcquired";
    case hipErrorNotMapped:
      return "hipErrorNotMapped";
    case hipErrorNotMappedAsArray:
      return "hipErrorNotMappedAsArray";
    case hipErrorNotMappedAsPointer:
      return "hipErrorNotMappedAsPointer";
    case hipErrorECCNotCorrectable:
      return "hipErrorECCNotCorrectable";
    case hipErrorUnsupportedLimit:
      return "hipErrorUnsupportedLimit";
    case hipErrorContextAlreadyInUse:
      return "hipErrorContextAlreadyInUse";
    case hipErrorPeerAccessUnsupported:
      return "hipErrorPeerAccessUnsupported";
    case hipErrorInvalidKernelFile:
      return "hipErrorInvalidKernelFile";
    case hipErrorInvalidGraphicsContext:
      return "hipErrorInvalidGraphicsContext";
    case hipErrorInvalidSource:
      return "hipErrorInvalidSource";
    case hipErrorFileNotFound:
      return "hipErrorFileNotFound";
    case hipErrorSharedObjectSymbolNotFound:
      return "hipErrorSharedObjectSymbolNotFound";
    case hipErrorSharedObjectInitFailed:
      return "hipErrorSharedObjectInitFailed";
    case hipErrorOperatingSystem:
      return "hipErrorOperatingSystem";
    case hipErrorInvalidHandle:
      return "hipErrorInvalidHandle";
    case hipErrorNotFound:
      return "hipErrorNotFound";
    case hipErrorNotReady:
      return "hipErrorNotReady";
    case hipErrorIllegalAddress:
      return "hipErrorIllegalAddress";
    case hipErrorLaunchOutOfResources:
      return "hipErrorLaunchOutOfResources";
    case hipErrorLaunchTimeOut:
      return "hipErrorLaunchTimeOut";
    case hipErrorPeerAccessAlreadyEnabled:
      return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled:
      return "hipErrorPeerAccessNotEnabled";
    case hipErrorSetOnActiveProcess:
      return "hipErrorSetOnActiveProcess";
    case hipErrorContextIsDestroyed:
      return "hipErrorContextIsDestroyed";
    case hipErrorAssert:
      return "hipErrorAssert";
    case hipErrorHostMemoryAlreadyRegistered:
      return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
      return "hipErrorHostMemoryNotRegistered";
    case hipErrorLaunchFailure:
      return "hipErrorLaunchFailure";
    case hipErrorCooperativeLaunchTooLarge:
      return "hipErrorCooperativeLaunchTooLarge";
    case hipErrorNotSupported:
      return "hipErrorNotSupported";
    case hipErrorStreamCaptureUnsupported:
      return "hipErrorStreamCaptureUnsupported";
    case hipErrorStreamCaptureInvalidated:
      return "hipErrorStreamCaptureInvalidated";
    case hipErrorStreamCaptureMerge:
      return "hipErrorStreamCaptureMerge";
    case hipErrorStreamCaptureUnmatched:
      return "hipErrorStreamCaptureUnmatched";
    case hipErrorStreamCaptureUnjoined:
      return "hipErrorStreamCaptureUnjoined";
    case hipErrorStreamCaptureIsolation:
      return "hipErrorStreamCaptureIsolation";
    case hipErrorStreamCaptureImplicit:
      return "hipErrorStreamCaptureImplicit";
    case hipErrorCapturedEvent:
      return "hipErrorCapturedEvent";
    case hipErrorStreamCaptureWrongThread:
      return "hipErrorStreamCaptureWrongThread";
    case hipErrorGraphExecUpdateFailure:
      return "hipErrorGraphExecUpdateFailure";
    case hipErrorUnknown:
    default:
      return "hipErrorUnknown";
  }
}

HIPAPI const char* hipGetErrorName(hipError_t error) {
  // Return the same as hipGetErrorString for simplicity.
  return hipGetErrorString(error);
}

// Handle scoped to THIS shared object (the HIP shim), resolved once. See the
// declaration in binding_internal.h and the note on hipGetProcAddress() for why
// the process-global scope is insufficient.
static void* iree_hip_self_dl_handle_cached = NULL;
static iree_once_flag iree_hip_self_dl_handle_once = IREE_ONCE_FLAG_INIT;
static void iree_hip_init_self_dl_handle(void) {
  Dl_info info;
  // dladdr() on a symbol we define yields the path to this library; reopening
  // it with RTLD_NOLOAD returns a handle to the already-resident module (the
  // extra reference intentionally pins the always-loaded HIP runtime).
  if (dladdr((void*)&hipGetProcAddress, &info) != 0 && info.dli_fname) {
    iree_hip_self_dl_handle_cached =
        dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
  }
}

void* iree_hip_self_dl_handle(void) {
  iree_call_once(&iree_hip_self_dl_handle_once, iree_hip_init_self_dl_handle);
  return iree_hip_self_dl_handle_cached;
}

HIPAPI hipError_t hipGetProcAddress(const char* symbol, void** pfn,
                                    int hipVersion, uint64_t flags,
                                    void* symbolStatus) {
  (void)hipVersion;
  (void)flags;
  if (!symbol || !pfn) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Resolve symbols against this library, not the process-global scope. A
  // consumer may dlopen us with RTLD_LOCAL (Triton's AMD backend does exactly
  // this), so our symbols never enter the global namespace and a
  // dlsym(dlopen(NULL), ...) lookup would spuriously fail with
  // hipErrorNotFound. Fall back to the global scope only if the self-handle
  // could not be established.
  void* handle = iree_hip_self_dl_handle();
  bool close_handle = false;
  if (!handle) {
    handle = dlopen(NULL, RTLD_LAZY);
    close_handle = handle != NULL;
  }
  if (!handle) {
    *pfn = NULL;
    if (symbolStatus) *(int*)symbolStatus = 1;
    HIP_RETURN_ERROR(hipErrorSharedObjectInitFailed);
  }
  *pfn = dlsym(handle, symbol);
  if (close_handle) dlclose(handle);
  if (symbolStatus) *(int*)symbolStatus = *pfn ? 0 : 1;
  HIP_RETURN_ERROR(*pfn ? hipSuccess : hipErrorNotFound);
}

// Driver API version of hipGetErrorString.
//
// Parameters:
//  - hipError: [IN] HIP error code to get string for.
//  - errorString: [OUT] Pointer to receive the error string.
//
// Returns:
//  - hipSuccess: Error string retrieved successfully.
//  - hipErrorInvalidValue: errorString is NULL.
//
// Synchronization: This operation is synchronous.
//
// Note: Unlike hipGetErrorString, this returns the string via output pointer.
//       This is the driver API equivalent for compatibility with CUDA driver
//       API.
HIPAPI hipError_t hipDrvGetErrorString(hipError_t hipError,
                                       const char** errorString) {
  if (!errorString) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *errorString = hipGetErrorString(hipError);
  HIP_RETURN_ERROR(hipSuccess);
}

// Driver API version of hipGetErrorName.
//
// Parameters:
//  - hipError: [IN] HIP error code to get name for.
//  - errorName: [OUT] Pointer to receive the error name.
//
// Returns:
//  - hipSuccess: Error name retrieved successfully.
//  - hipErrorInvalidValue: errorName is NULL.
//
// Synchronization: This operation is synchronous.
//
// Note: Unlike hipGetErrorName, this returns the name via output pointer.
//       This is the driver API equivalent for compatibility with CUDA driver
//       API.
HIPAPI hipError_t hipDrvGetErrorName(hipError_t hipError,
                                     const char** errorName) {
  if (!errorName) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *errorName = hipGetErrorName(hipError);
  HIP_RETURN_ERROR(hipSuccess);
}

// Gets and clears the last error from HIP runtime calls.
//
// Parameters: None.
//
// Returns: The last error code set by any HIP runtime call in this thread.
//
// Synchronization: This operation is synchronous.
//
// Error behavior:
// - Returns the last error from this thread.
// - Clears the error after returning it.
// - Returns hipSuccess if no error has occurred.
// - Each thread has its own error state.
//
// Usage pattern:
// ```c
// hipMalloc(&ptr, size);
// hipError_t err = hipGetLastError();
// if (err != hipSuccess) {
//   printf("Error: %s\n", hipGetErrorString(err));
// }
// ```
//
// Warning: This function clears the error. Use hipPeekAtLastError() to
// check without clearing.
//
// See also: hipPeekAtLastError, hipGetErrorString, hipGetErrorName.
HIPAPI hipError_t hipGetLastError(void) {
  return iree_hip_thread_error_get_and_clear();
}

// Gets the last error and clears it.
//
// Parameters: None.
//
// Returns: The last error code set by any HIP runtime call in this thread,
//          then resets the stored error code to hipSuccess.
//
// Synchronization: This operation is synchronous.
//
// Error behavior:
// - Returns the last error from this thread.
// - Clears the error after returning it.
// - Returns hipSuccess if no error has occurred.
// - Each thread has its own error state.
//
// Note: This is an alias for hipGetLastError with identical behavior.
//
// See also: hipGetLastError, hipPeekAtLastError, hipGetErrorString.
HIPAPI hipError_t hipExtGetLastError(void) {
  return iree_hip_thread_error_get_and_clear();
}

// Gets the last error without clearing it.
//
// Parameters: None.
//
// Returns: The last error code set by any HIP runtime call in this thread.
//
// Synchronization: This operation is synchronous.
//
// Error behavior:
// - Returns the last error from this thread.
// - Does NOT clear the error.
// - Returns hipSuccess if no error has occurred.
// - Error remains set for subsequent calls.
//
// See also: hipGetLastError, hipGetErrorString.
HIPAPI hipError_t hipPeekAtLastError(void) {
  return iree_hip_thread_error_peek();
}

//===----------------------------------------------------------------------===//
// Clang Support
//===----------------------------------------------------------------------===//
// clang++ emits code using these __ functions instead of just using the HIP
// APIs directly. It'd be cleaner if it didn't but it does allow us to hook
// things a bit more directly.

typedef struct uint3 {
  union {
    struct {
      uint32_t x;
      uint32_t y;
      uint32_t z;
    };
    uint32_t data[3];
  };
} uint3;

// Registers a fat binary containing device code with the HIP runtime.
//
// Parameters:
//  - data: [IN] Pointer to the fat binary data containing device code.
//              This data is compiler-generated and embedded in the executable.
//              Must remain valid for the lifetime of the registration.
//
// Returns:
//  - Opaque handle to the registered module (actually void** but treated as
//    opaque). This handle is passed to subsequent registration functions
//    (__hipRegisterFunction, __hipRegisterVar) to associate symbols with
//    this module. Returns NULL if registration fails.
//
// Calling context:
// - Called automatically during program startup (before main() or during
//   dlopen) by compiler-generated initialization code.
// - Called once per translation unit that contains __global__ functions or
//   __device__ variables.
// - Thread-safe: Can be called concurrently from multiple threads during
//   dynamic library loading.
//
// Lifetime:
// - The fat binary data must remain valid until __hipUnregisterFatBinary.
// - The returned handle is valid until __hipUnregisterFatBinary is called.
//
// Implementation notes:
// - The fat binary contains all kernels and device variables from a
//   compilation unit.
// - May contain code for multiple GPU architectures.
// - The runtime selects appropriate code for the current device.
// Fat binary wrapper structure passed by the HIP runtime.
// This matches the __CudaFatBinaryWrapper / __hipFatBinaryWrapper structure
// generated by clang for HIP programs.
typedef struct __hipFatBinaryWrapper_t {
  unsigned int magic;    // Magic number (0x48495046 = "HIPF" or 0xBA55FACE)
  unsigned int version;  // Version of the fat binary wrapper
  const void* binary;    // Pointer to the actual fat binary data (CCOB/bundle)
  const void* reserved;  // Reserved for future use
} __hipFatBinaryWrapper;

#define HIP_FAT_BINARY_MAGIC_OLD 0xBA55FACE
#define HIP_FAT_BINARY_MAGIC_NEW 0x48495046  // "HIPF"

HIPAPI void** __hipRegisterFatBinary(const void* data) {
  HIP_DEBUG_LOG("[HIP_API] __hipRegisterFatBinary(%p)\n", data);
  if (!data) return NULL;

  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) return NULL;

  // The data passed is a fat binary wrapper structure, not the binary itself.
  // We need to extract the actual binary pointer from the wrapper.
  const __hipFatBinaryWrapper* wrapper = (const __hipFatBinaryWrapper*)data;

  // Check for the fat binary wrapper magic.
  const void* binary_data = data;
  if (wrapper->magic == HIP_FAT_BINARY_MAGIC_OLD ||
      wrapper->magic == HIP_FAT_BINARY_MAGIC_NEW) {
    // It's a fat binary wrapper - extract the binary pointer.
    binary_data = wrapper->binary;
  }
  // Otherwise, assume data points directly to the binary.

  if (!binary_data) return NULL;

  iree_hal_streaming_module_registration_t* module = NULL;
  iree_status_t status =
      iree_hal_streaming_global_symbol_registry_register_module(
          registry, binary_data, &module);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return NULL;
  }

  // Return the module handle cast to void**.
  // The HIP runtime expects a void** but we just return our opaque handle.
  return (void**)module;
}

// Unregisters a fat binary previously registered with __hipRegisterFatBinary.
//
// Parameters:
//  - modules: [IN] Handle returned from __hipRegisterFatBinary.
//
// Calling context:
// - Called automatically during program shutdown (after main() or during
//   dlclose) by compiler-generated finalization code.
// - Must be called after all kernels from this module have completed.
// - Thread-safe: Serialized with other registration/unregistration calls.
//
// Behavior:
// - Removes all functions and variables associated with this module.
// - Frees any device memory allocated for this module.
// - After this call, all device functions and variables from this module
//   become invalid.
//
// WARNING: Calling kernels from an unregistered module results in undefined
// behavior.
HIPAPI void __hipUnregisterFatBinary(void** modules) {
  if (!modules) return;

  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) return;

  iree_status_t status =
      iree_hal_streaming_global_symbol_registry_unregister_module(
          registry, (iree_hal_streaming_module_registration_t*)modules);
  iree_status_ignore(status);
}

// Registers a __global__ function (kernel) with the HIP runtime.
//
// Parameters:
//  - modules: [IN] Handle from __hipRegisterFatBinary for the containing
//                  module.
//  - hostFunction: [IN] Host-side function pointer (the stub function).
//                       Used as the key for kernel launches.
//  - deviceFunction: [IN] Device function name in the fat binary (mangled).
//                         Must match the symbol name in the device code.
//  - deviceName: [IN] Human-readable kernel name (typically unmangled).
//                     Used for profiling and debugging.
//  - threadLimit: [IN] Maximum threads per block (0 = no limit).
//  - tid: [UNUSED] Thread index remapping (legacy, pass NULL).
//  - bid: [UNUSED] Block index remapping (legacy, pass NULL).
//  - blockDim: [UNUSED] Static block dimensions (legacy, pass NULL).
//  - gridDim: [UNUSED] Static grid dimensions (legacy, pass NULL).
//  - wSize: [UNUSED] Static shared memory size (legacy, pass NULL).
//
// Calling context:
// - Called automatically during program startup after __hipRegisterFatBinary.
// - Called once per __global__ function in the module.
// - Thread-safe: Protected by internal mutex.
//
// Lifetime:
// - The deviceFunction and deviceName strings are string literals in the
//   executable's .rodata section and remain valid for the program lifetime.
// - Registration remains valid until __hipUnregisterFatBinary.
//
// Usage:
// - After registration, kernels are launched using hipLaunchKernel with
//   the hostFunction pointer as the function argument.
HIPAPI void __hipRegisterFunction(void** modules, const void* hostFunction,
                                  char* deviceFunction, const char* deviceName,
                                  unsigned int threadLimit, uint3* tid,
                                  uint3* bid, dim3* blockDim, dim3* gridDim,
                                  int* wSize) {
  if (!modules || !hostFunction || !deviceName) return;

  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) return;

  const uint32_t shared_size = wSize ? *wSize : 0;

  iree_status_t status =
      iree_hal_streaming_global_symbol_registry_insert_function(
          registry, (iree_hal_streaming_module_registration_t*)modules,
          (void*)hostFunction, deviceName, threadLimit, shared_size);
  iree_status_ignore(status);
}

// Registers a __managed__ variable with the HIP runtime.
//
// Parameters:
//  - hipModule: [IN] Handle from __hipRegisterFatBinary for the containing
//                    module.
//  - pointer: [IN/OUT] Pointer to the host-side variable pointer.
//                      Will be updated to point to managed memory.
//  - init_value: [IN] Initial value to copy (can be NULL).
//  - name: [IN] Variable name in device code.
//  - size: [IN] Size of the variable in bytes.
//  - align: [IN] Required alignment in bytes.
//
// Calling context:
// - Called during program startup after __hipRegisterFatBinary.
// - Managed memory allows automatic migration between host and device.
//
// Lifetime:
// - Managed memory persists until module unregistration.
// - The name string must remain valid (typically a string literal).
//
// NOTE: Managed variables are accessible from both host and device code
// without explicit memory transfers. They should not be used in modern code.
HIPAPI void __hipRegisterManagedVar(void* hipModule, void** pointer,
                                    void* init_value, const char* name,
                                    size_t size, unsigned align) {
  if (!hipModule || !pointer || !name) return;

  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) return;

  void* managed_pointer = init_value ? init_value : *pointer;
  if (managed_pointer) {
    *pointer = managed_pointer;
  }

  iree_status_t status =
      iree_hal_streaming_global_symbol_registry_insert_managed_variable(
          registry, (iree_hal_streaming_module_registration_t*)hipModule,
          managed_pointer ? managed_pointer : pointer, name, size, align);
  if (iree_status_is_ok(status) && (void*)pointer != managed_pointer) {
    status = iree_hal_streaming_global_symbol_registry_insert_managed_variable(
        registry, (iree_hal_streaming_module_registration_t*)hipModule, pointer,
        name, size, align);
  }
  iree_status_ignore(status);
}

HIPAPI void __hipRegisterTexture(void** modules, void* var, char* hostVar,
                                 char* deviceVar, int type, int norm, int ext) {
  (void)modules;
  (void)var;
  (void)hostVar;
  (void)deviceVar;
  (void)type;
  (void)norm;
  (void)ext;
}

// Registers a __device__ or __constant__ variable with the HIP runtime.
//
// Parameters:
//  - modules: [IN] Handle from __hipRegisterFatBinary for the containing
//                  module.
//  - var: [IN] Host-side shadow variable address.
//  - hostVar: [IN] Host variable name (typically same as deviceVar).
//  - deviceVar: [IN] Device variable name (for symbol lookup).
//  - ext: [IN] 1 if externally visible, 0 otherwise.
//  - size: [IN] Size of the variable in bytes.
//  - constant: [IN] 1 if __constant__ memory, 0 if __device__ memory.
//  - global: [IN] 1 if globally visible, 0 otherwise.
//
// Calling context:
// - Called during program startup after __hipRegisterFatBinary.
// - Called once per __device__ or __constant__ variable in the module.
//
// Lifetime:
// - The variable remains valid until __hipUnregisterFatBinary.
HIPAPI void __hipRegisterVar(void** modules, void* var, char* hostVar,
                             char* deviceVar, int ext, size_t size,
                             int constant, int global) {
  if (!modules || !var || !deviceVar) return;

  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) return;

  // Register the variable with a default alignment of 8 bytes.
  // The actual alignment is determined by the device variable type.
  iree_status_t status =
      iree_hal_streaming_global_symbol_registry_insert_variable(
          registry, (iree_hal_streaming_module_registration_t*)modules, var,
          deviceVar, size, 8);
  iree_status_ignore(status);
}

// We make the assumption here that the stack depth is always 0 or 1.
// Clang never emits more than one push per stub call and each stub call pops
// exactly once. The whole push/pop design is terrible, anyway, and unless we
// find something that actually uses the stack we keep things simple.
typedef struct iree_hip_call_configuration_t {
  dim3 grid_dim;
  dim3 block_dim;
  size_t shared_mem;
  hipStream_t stream;
  bool valid;  // true if configuration has been pushed
} iree_hip_call_configuration_t;
static IREE_THREAD_LOCAL iree_hip_call_configuration_t iree_hip_call_config = {
    0};

// Pushes kernel launch configuration onto the call stack.
//
// Parameters:
//  - gridDim: [IN] Grid dimensions (number of blocks).
//  - blockDim: [IN] Block dimensions (threads per block).
//  - sharedMem: [IN] Dynamic shared memory size in bytes.
//  - stream: [IN] Stream to launch on (NULL = default stream).
//
// Returns:
//  - hipSuccess: Configuration pushed successfully.
//  - hipErrorInvalidConfiguration: Invalid launch configuration.
//
// Calling context:
// - Used by <<<>>> kernel launch syntax in HIP/CUDA.
// - Must be immediately followed by the kernel call.
// - Thread-local: Each thread has its own configuration stack.
//
// Usage:
// - Compiler transforms kernel<<<grid,block,shared,stream>>>(...) into:
//   __hipPushCallConfiguration(grid, block, shared, stream);
//   kernel(...);
//
// NOTE: The pushed configuration is consumed by the next kernel launch
// on the current thread.
HIPAPI hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                             size_t sharedMem,
                                             hipStream_t stream) {
  // Store the configuration in thread-local storage.
  // This will be consumed by the next kernel launch on this thread.
  iree_hip_call_config.grid_dim = gridDim;
  iree_hip_call_config.block_dim = blockDim;
  iree_hip_call_config.shared_mem = sharedMem;
  iree_hip_call_config.stream = stream;
  iree_hip_call_config.valid = true;
  return hipSuccess;
}

// Pops kernel launch configuration from the call stack.
//
// Parameters:
//  - gridDim: [OUT] Grid dimensions.
//  - blockDim: [OUT] Block dimensions.
//  - sharedMem: [OUT] Dynamic shared memory size.
//  - stream: [OUT] Stream for launch.
//
// Returns:
//  - hipSuccess: Configuration popped successfully.
//  - hipErrorInvalidValue: No configuration on stack.
//
// Calling context:
// - Used internally by kernel stub functions.
// - Retrieves configuration pushed by __hipPushCallConfiguration.
//
// NOTE: This is typically called from compiler-generated kernel stubs
// to retrieve launch parameters.
HIPAPI hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                            size_t* sharedMem,
                                            hipStream_t* stream) {
  // Check if configuration has been pushed.
  if (IREE_UNLIKELY(!iree_hip_call_config.valid)) {
    HIP_RETURN_ERROR(hipErrorInvalidConfiguration);
  }

  // Return the pushed configuration.
  if (gridDim) *gridDim = iree_hip_call_config.grid_dim;
  if (blockDim) *blockDim = iree_hip_call_config.block_dim;
  if (sharedMem) *sharedMem = iree_hip_call_config.shared_mem;
  if (stream) *stream = iree_hip_call_config.stream;

  // Mark configuration as consumed.
  iree_hip_call_config.valid = false;
  return hipSuccess;
}
