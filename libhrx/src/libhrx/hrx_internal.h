// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_INTERNAL_H_
#define HRX_INTERNAL_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/device_group.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/hal/pool.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree_hal_compat.h"

#ifdef __cplusplus
extern "C" {
#ifndef _Static_assert
#define _Static_assert static_assert
#endif
#endif

//===----------------------------------------------------------------------===//
// Tracing helpers
//===----------------------------------------------------------------------===//

#define HRX_TRACE_ZONE_BEGIN(zone_id, name_literal)                            \
  IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal)
#define HRX_TRACE_ZONE_END(zone_id) IREE_TRACE_ZONE_END(zone_id)
#define HRX_TRACE_ZONE_APPEND_BYTES(zone_id, byte_count)                       \
  IREE_TRACE_ZONE_APPEND_VALUE_I64(zone_id, (int64_t)(byte_count))
#define HRX_RETURN_AND_END_ZONE(zone_id, expr)                                 \
  do {                                                                         \
    hrx_status_t hrx_status__ = (expr);                                        \
    HRX_TRACE_ZONE_END(zone_id);                                               \
    return hrx_status__;                                                       \
  } while (0)
#define HRX_RETURN_VOID_AND_END_ZONE(zone_id)                                  \
  do {                                                                         \
    HRX_TRACE_ZONE_END(zone_id);                                               \
    return;                                                                    \
  } while (0)

//===----------------------------------------------------------------------===//
// Enum value compatibility asserts
//
// HRX enums match IREE values by convention. These asserts guarantee it.
//===----------------------------------------------------------------------===//

#define HRX_STATIC_ASSERT_ENUM_EQ(lhs, rhs, message) \
  _Static_assert((int)(lhs) == (int)(rhs), message)

HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_OK, IREE_STATUS_OK, "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_CANCELLED, IREE_STATUS_CANCELLED,
                          "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_UNKNOWN, IREE_STATUS_UNKNOWN,
                          "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_INVALID_ARGUMENT,
                          IREE_STATUS_INVALID_ARGUMENT, "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_DEADLINE_EXCEEDED,
                          IREE_STATUS_DEADLINE_EXCEEDED, "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_NOT_FOUND, IREE_STATUS_NOT_FOUND,
                          "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_ALREADY_EXISTS,
                          IREE_STATUS_ALREADY_EXISTS, "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_OUT_OF_MEMORY,
                          IREE_STATUS_RESOURCE_EXHAUSTED, "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_OUT_OF_RANGE, IREE_STATUS_OUT_OF_RANGE,
                          "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_UNIMPLEMENTED, IREE_STATUS_UNIMPLEMENTED,
                          "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_INTERNAL, IREE_STATUS_INTERNAL,
                          "status mismatch");
HRX_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_UNAVAILABLE, IREE_STATUS_UNAVAILABLE,
                          "status mismatch");

// Memory type bitfield.
_Static_assert(HRX_MEMORY_TYPE_NONE == IREE_HAL_MEMORY_TYPE_NONE,
               "memory type mismatch");
_Static_assert(HRX_MEMORY_TYPE_OPTIMAL == IREE_HAL_MEMORY_TYPE_OPTIMAL,
               "memory type mismatch");
_Static_assert(HRX_MEMORY_TYPE_HOST_VISIBLE ==
                   IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
               "memory type mismatch");
_Static_assert(HRX_MEMORY_TYPE_HOST_COHERENT ==
                   IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
               "memory type mismatch");
_Static_assert(HRX_MEMORY_TYPE_HOST_CACHED == IREE_HAL_MEMORY_TYPE_HOST_CACHED,
               "memory type mismatch");
_Static_assert(HRX_MEMORY_TYPE_HOST_LOCAL == IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
               "memory type mismatch");
_Static_assert(HRX_MEMORY_TYPE_DEVICE_VISIBLE ==
                   IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
               "memory type mismatch");
_Static_assert(HRX_MEMORY_TYPE_DEVICE_LOCAL ==
                   IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
               "memory type mismatch");

// Memory access bitfield.
_Static_assert(HRX_MEMORY_ACCESS_NONE == IREE_HAL_MEMORY_ACCESS_NONE,
               "memory access mismatch");
_Static_assert(HRX_MEMORY_ACCESS_READ == IREE_HAL_MEMORY_ACCESS_READ,
               "memory access mismatch");
_Static_assert(HRX_MEMORY_ACCESS_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
               "memory access mismatch");
_Static_assert(HRX_MEMORY_ACCESS_DISCARD == IREE_HAL_MEMORY_ACCESS_DISCARD,
               "memory access mismatch");
_Static_assert(HRX_MEMORY_ACCESS_ALL == IREE_HAL_MEMORY_ACCESS_ALL,
               "memory access mismatch");

// Buffer usage bitfield.
_Static_assert(HRX_BUFFER_USAGE_NONE == IREE_HAL_BUFFER_USAGE_NONE,
               "buffer usage mismatch");
_Static_assert(HRX_BUFFER_USAGE_TRANSFER == IREE_HAL_BUFFER_USAGE_TRANSFER,
               "buffer usage mismatch");
_Static_assert(HRX_BUFFER_USAGE_DISPATCH_STORAGE ==
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
               "buffer usage mismatch");
_Static_assert(HRX_BUFFER_USAGE_DEFAULT == IREE_HAL_BUFFER_USAGE_DEFAULT,
               "buffer usage mismatch");

// Buffer view metadata.
_Static_assert(HRX_ELEMENT_TYPE_NONE == IREE_HAL_ELEMENT_TYPE_NONE,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_OPAQUE_8 == IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_BOOL_8 == IREE_HAL_ELEMENT_TYPE_BOOL_8,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_INT_16 == IREE_HAL_ELEMENT_TYPE_INT_16,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_UINT_8 == IREE_HAL_ELEMENT_TYPE_UINT_8,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_INT_32 == IREE_HAL_ELEMENT_TYPE_INT_32,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_INT_64 == IREE_HAL_ELEMENT_TYPE_INT_64,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_SINT_8 == IREE_HAL_ELEMENT_TYPE_SINT_8,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_SINT_16 == IREE_HAL_ELEMENT_TYPE_SINT_16,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_SINT_32 == IREE_HAL_ELEMENT_TYPE_SINT_32,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_SINT_64 == IREE_HAL_ELEMENT_TYPE_SINT_64,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_FLOAT_16 == IREE_HAL_ELEMENT_TYPE_FLOAT_16,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_FLOAT_32 == IREE_HAL_ELEMENT_TYPE_FLOAT_32,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_FLOAT_64 == IREE_HAL_ELEMENT_TYPE_FLOAT_64,
               "element type mismatch");
_Static_assert(HRX_ELEMENT_TYPE_BFLOAT_16 == IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
               "element type mismatch");
_Static_assert(HRX_ENCODING_TYPE_OPAQUE == IREE_HAL_ENCODING_TYPE_OPAQUE,
               "encoding type mismatch");
_Static_assert(HRX_ENCODING_TYPE_DENSE_ROW_MAJOR ==
                   IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
               "encoding type mismatch");

// Memory protection bitfield.
_Static_assert(HRX_MEMORY_PROTECTION_NONE == IREE_HAL_MEMORY_PROTECTION_NONE,
               "memory protection mismatch");
_Static_assert(HRX_MEMORY_PROTECTION_READ == IREE_HAL_MEMORY_PROTECTION_READ,
               "memory protection mismatch");
_Static_assert(HRX_MEMORY_PROTECTION_WRITE == IREE_HAL_MEMORY_PROTECTION_WRITE,
               "memory protection mismatch");

// Map flags reuse memory access values.
_Static_assert(HRX_MAP_READ == IREE_HAL_MEMORY_ACCESS_READ,
               "map flags mismatch");
_Static_assert(HRX_MAP_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
               "map flags mismatch");
_Static_assert(HRX_MAP_DISCARD == IREE_HAL_MEMORY_ACCESS_DISCARD,
               "map flags mismatch");

#undef HRX_STATIC_ASSERT_ENUM_EQ

//===----------------------------------------------------------------------===//
// Internal types backing opaque handles
//===----------------------------------------------------------------------===//

#define HRX_MAX_DEVICES 64

// Status payload (allocated on error, NULL = OK).
typedef struct hrx_status_s {
  hrx_status_code_t code;
  char *message;
} hrx_status_s;

// Allocator (wraps iree_hal_allocator_t, owned by device).
typedef struct hrx_allocator_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_allocator_t *hal_allocator;
  hrx_device_t device;
} hrx_allocator_s;

// Device instance.
typedef struct hrx_device_s {
  iree_atomic_ref_count_t ref_count;
  hrx_accelerator_type_t type;
  int ordinal;
  iree_hal_device_t *hal_device;
  iree_hal_device_group_t *hal_device_group;
  bool profiling_active;
  hrx_allocator_s allocator; // Inline, owned by device.
  char name[128];
  char architecture[64];
} hrx_device_s;

// Timeline semaphore.
typedef struct hrx_semaphore_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_semaphore_t *hal_semaphore;
  hrx_device_t device;
} hrx_semaphore_s;

// Stream with pending command buffer.
typedef struct hrx_stream_s {
  iree_atomic_ref_count_t ref_count;
  hrx_device_t device;
  hrx_semaphore_t semaphore;
  uint64_t timepoint;
  iree_hal_command_buffer_t *pending_cb;
  bool has_pending_work;
  uint32_t flags;
} hrx_stream_s;

// Buffer allocation.
typedef struct hrx_buffer_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_buffer_t *hal_buffer;
  iree_hal_pool_t *hal_pool;
  hrx_device_t device;
  hrx_memory_type_t mem_type;
  size_t size;
  iree_hal_buffer_mapping_t mapping;
  bool is_mapped;
  void *mapped_ptr;
} hrx_buffer_s;

// Loaded VM module with a context containing HAL + bytecode modules.
typedef struct hrx_module_s {
  iree_atomic_ref_count_t ref_count;
  hrx_device_t device;
  iree_vm_module_t *bytecode_module;
  iree_vm_module_t *hal_module;
  iree_vm_context_t *context;
} hrx_module_s;

// Resolved VM function retained with its parent module.
typedef struct hrx_function_s {
  iree_atomic_ref_count_t ref_count;
  hrx_module_t module;
  iree_vm_function_t vm_function;
} hrx_function_s;

// Growable VM argument/result list.
typedef struct hrx_value_list_s {
  iree_atomic_ref_count_t ref_count;
  iree_vm_list_t *vm_list;
} hrx_value_list_s;

// Timeline fence wrapper.
typedef struct hrx_fence_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_fence_t *hal_fence;
} hrx_fence_s;

// Buffer view wrapper.
typedef struct hrx_buffer_view_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_buffer_view_t *hal_buffer_view;
} hrx_buffer_view_s;

// HAL executable wrapper for direct queue/stream dispatch.
typedef struct hrx_executable_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_executable_cache_t *hal_executable_cache;
  iree_hal_executable_t *hal_executable;
  hrx_device_t device;
} hrx_executable_s;

typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

typedef struct hrx_cpu_state_t {
  hrx_device_s devices[HRX_MAX_DEVICES];
  int device_count;
  bool initialized;
  iree_hal_driver_t *driver;
} hrx_cpu_state_t;

typedef struct hrx_gpu_state_t {
  hrx_device_s devices[HRX_MAX_DEVICES];
  int device_count;
  bool initialized;
  iree_hal_driver_t *driver;
} hrx_gpu_state_t;

typedef struct hrx_shared_state_t {
  iree_vm_instance_t *vm_instance;
  iree_async_proactor_pool_t *proactor_pool;
  iree_allocator_t host_allocator;
  int init_count;
  bool shared_initialized;
} hrx_shared_state_t;

// Access global state (defined in runtime.c).
hrx_shared_state_t *hrx_get_shared_state(void);
hrx_gpu_state_t *hrx_get_gpu_state(void);
hrx_cpu_state_t *hrx_get_cpu_state(void);

// Ensure shared infrastructure is created (idempotent).
hrx_status_t hrx_ensure_shared_state(void);

// Convert iree_status_t to hrx_status_t.
hrx_status_t hrx_status_from_iree(iree_status_t iree_status);

// Convert hrx_status_t back to iree_status_t and consume the hrx status.
iree_status_t hrx_status_to_iree(hrx_status_t status);

iree_status_t hrx_iree_exact_pool_create(iree_hal_allocator_t *allocator,
                                         iree_hal_buffer_params_t params,
                                         iree_hal_pool_t **out_pool);

#ifdef __cplusplus
}
#endif

#endif // HRX_INTERNAL_H_
