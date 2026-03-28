// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_INTERNAL_H_
#define PYRE_INTERNAL_H_

#include "pyre_runtime.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/task/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Enum value compatibility asserts
//
// Pyre enums match IREE values by convention. These asserts guarantee it.
//===----------------------------------------------------------------------===//

_Static_assert(PYRE_STATUS_OK == IREE_STATUS_OK, "status mismatch");
_Static_assert(PYRE_STATUS_CANCELLED == IREE_STATUS_CANCELLED, "status mismatch");
_Static_assert(PYRE_STATUS_UNKNOWN == IREE_STATUS_UNKNOWN, "status mismatch");
_Static_assert(PYRE_STATUS_INVALID_ARGUMENT == IREE_STATUS_INVALID_ARGUMENT,
               "status mismatch");
_Static_assert(PYRE_STATUS_DEADLINE_EXCEEDED == IREE_STATUS_DEADLINE_EXCEEDED,
               "status mismatch");
_Static_assert(PYRE_STATUS_NOT_FOUND == IREE_STATUS_NOT_FOUND, "status mismatch");
_Static_assert(PYRE_STATUS_ALREADY_EXISTS == IREE_STATUS_ALREADY_EXISTS,
               "status mismatch");
_Static_assert(PYRE_STATUS_OUT_OF_MEMORY == IREE_STATUS_RESOURCE_EXHAUSTED,
               "status mismatch");
_Static_assert(PYRE_STATUS_OUT_OF_RANGE == IREE_STATUS_OUT_OF_RANGE,
               "status mismatch");
_Static_assert(PYRE_STATUS_UNIMPLEMENTED == IREE_STATUS_UNIMPLEMENTED,
               "status mismatch");
_Static_assert(PYRE_STATUS_INTERNAL == IREE_STATUS_INTERNAL, "status mismatch");
_Static_assert(PYRE_STATUS_UNAVAILABLE == IREE_STATUS_UNAVAILABLE,
               "status mismatch");

// Memory type bitfield.
_Static_assert(PYRE_MEMORY_TYPE_NONE == IREE_HAL_MEMORY_TYPE_NONE,
               "memory type mismatch");
_Static_assert(PYRE_MEMORY_TYPE_OPTIMAL == IREE_HAL_MEMORY_TYPE_OPTIMAL,
               "memory type mismatch");
_Static_assert(PYRE_MEMORY_TYPE_HOST_VISIBLE == IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
               "memory type mismatch");
_Static_assert(PYRE_MEMORY_TYPE_HOST_COHERENT == IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
               "memory type mismatch");
_Static_assert(PYRE_MEMORY_TYPE_HOST_CACHED == IREE_HAL_MEMORY_TYPE_HOST_CACHED,
               "memory type mismatch");
_Static_assert(PYRE_MEMORY_TYPE_HOST_LOCAL == IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
               "memory type mismatch");
_Static_assert(PYRE_MEMORY_TYPE_DEVICE_VISIBLE == IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
               "memory type mismatch");
_Static_assert(PYRE_MEMORY_TYPE_DEVICE_LOCAL == IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
               "memory type mismatch");

// Memory access bitfield.
_Static_assert(PYRE_MEMORY_ACCESS_NONE == IREE_HAL_MEMORY_ACCESS_NONE,
               "memory access mismatch");
_Static_assert(PYRE_MEMORY_ACCESS_READ == IREE_HAL_MEMORY_ACCESS_READ,
               "memory access mismatch");
_Static_assert(PYRE_MEMORY_ACCESS_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
               "memory access mismatch");
_Static_assert(PYRE_MEMORY_ACCESS_DISCARD == IREE_HAL_MEMORY_ACCESS_DISCARD,
               "memory access mismatch");
_Static_assert(PYRE_MEMORY_ACCESS_ALL == IREE_HAL_MEMORY_ACCESS_ALL,
               "memory access mismatch");

// Buffer usage bitfield.
_Static_assert(PYRE_BUFFER_USAGE_NONE == IREE_HAL_BUFFER_USAGE_NONE,
               "buffer usage mismatch");
_Static_assert(PYRE_BUFFER_USAGE_TRANSFER == IREE_HAL_BUFFER_USAGE_TRANSFER,
               "buffer usage mismatch");
_Static_assert(PYRE_BUFFER_USAGE_DISPATCH_STORAGE ==
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
               "buffer usage mismatch");
_Static_assert(PYRE_BUFFER_USAGE_DEFAULT == IREE_HAL_BUFFER_USAGE_DEFAULT,
               "buffer usage mismatch");

// Memory protection bitfield.
_Static_assert(PYRE_MEMORY_PROTECTION_NONE == IREE_HAL_MEMORY_PROTECTION_NONE,
               "memory protection mismatch");
_Static_assert(PYRE_MEMORY_PROTECTION_READ == IREE_HAL_MEMORY_PROTECTION_READ,
               "memory protection mismatch");
_Static_assert(PYRE_MEMORY_PROTECTION_WRITE == IREE_HAL_MEMORY_PROTECTION_WRITE,
               "memory protection mismatch");

// Map flags reuse memory access values.
_Static_assert(PYRE_MAP_READ == IREE_HAL_MEMORY_ACCESS_READ,
               "map flags mismatch");
_Static_assert(PYRE_MAP_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
               "map flags mismatch");
_Static_assert(PYRE_MAP_DISCARD == IREE_HAL_MEMORY_ACCESS_DISCARD,
               "map flags mismatch");

//===----------------------------------------------------------------------===//
// Internal types backing opaque handles
//===----------------------------------------------------------------------===//

#define PYRE_MAX_DEVICES 64

// Status payload (allocated on error, NULL = OK).
typedef struct pyre_status_s {
  pyre_status_code_t code;
  char* message;
} pyre_status_s;

// Allocator (wraps iree_hal_allocator_t, owned by device).
typedef struct pyre_allocator_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_allocator_t* hal_allocator;
  pyre_device_t device;
} pyre_allocator_s;

// Device instance.
typedef struct pyre_device_s {
  iree_atomic_ref_count_t ref_count;
  pyre_accelerator_type_t type;
  int ordinal;
  iree_hal_device_t* hal_device;
  pyre_allocator_s allocator;  // Inline, owned by device.
  char name[128];
  char architecture[64];
} pyre_device_s;

// Timeline semaphore.
typedef struct pyre_semaphore_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_semaphore_t* hal_semaphore;
  pyre_device_t device;
} pyre_semaphore_s;

// Stream with pending command buffer.
typedef struct pyre_stream_s {
  iree_atomic_ref_count_t ref_count;
  pyre_device_t device;
  pyre_semaphore_t semaphore;
  uint64_t timepoint;
  iree_hal_command_buffer_t* pending_cb;
  bool has_pending_work;
  uint32_t flags;
} pyre_stream_s;

// Buffer allocation.
typedef struct pyre_buffer_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_buffer_t* hal_buffer;
  pyre_device_t device;
  pyre_memory_type_t mem_type;
  size_t size;
  void* mapped_ptr;
} pyre_buffer_s;

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

typedef struct pyre_cpu_state_t {
  pyre_device_s devices[PYRE_MAX_DEVICES];
  int device_count;
  bool initialized;
  iree_hal_driver_t* driver;
} pyre_cpu_state_t;

typedef struct pyre_gpu_state_t {
  pyre_device_s devices[PYRE_MAX_DEVICES];
  int device_count;
  bool initialized;
  iree_hal_driver_t* driver;
} pyre_gpu_state_t;

typedef struct pyre_shared_state_t {
  iree_vm_instance_t* vm_instance;
  iree_async_proactor_pool_t* proactor_pool;
  iree_allocator_t host_allocator;
  int init_count;
  bool shared_initialized;
} pyre_shared_state_t;

// Access global state (defined in runtime.c).
pyre_shared_state_t* pyre_get_shared_state(void);
pyre_gpu_state_t* pyre_get_gpu_state(void);
pyre_cpu_state_t* pyre_get_cpu_state(void);

// Ensure shared infrastructure is created (idempotent).
pyre_status_t pyre_ensure_shared_state(void);

// Convert iree_status_t to pyre_status_t.
pyre_status_t pyre_status_from_iree(iree_status_t iree_status);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_INTERNAL_H_
