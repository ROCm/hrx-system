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
// Internal types backing opaque handles
//===----------------------------------------------------------------------===//

#define PYRE_MAX_DEVICES 64

// Status payload (allocated on error, NULL = OK).
typedef struct pyre_status_s {
  pyre_status_code_t code;
  char* message;
} pyre_status_s;

// Device instance.
typedef struct pyre_device_s {
  iree_atomic_ref_count_t ref_count;
  pyre_accelerator_type_t type;
  int ordinal;
  iree_hal_device_t* hal_device;
  iree_hal_allocator_t* allocator;
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

// Map pyre_status_code_t to iree_status_code_t equivalent.
pyre_status_t pyre_status_from_iree(iree_status_t iree_status);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_INTERNAL_H_
