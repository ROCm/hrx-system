// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_RUNTIME_H_
#define PYRE_RUNTIME_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Export macros
//===----------------------------------------------------------------------===//

#if defined(PYRE_BUILDING_SHARED)
#define PYRE_API __attribute__((visibility("default")))
#elif defined(PYRE_STATIC)
#define PYRE_API
#else
#define PYRE_API
#endif

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

#define PYRE_VERSION_MAJOR 0
#define PYRE_VERSION_MINOR 1
#define PYRE_VERSION_PATCH 0

PYRE_API void pyre_runtime_version(int* major, int* minor, int* patch);

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

typedef enum pyre_status_code_t {
  PYRE_STATUS_OK = 0,
  PYRE_STATUS_INVALID_ARGUMENT,
  PYRE_STATUS_NOT_FOUND,
  PYRE_STATUS_ALREADY_EXISTS,
  PYRE_STATUS_OUT_OF_RANGE,
  PYRE_STATUS_UNIMPLEMENTED,
  PYRE_STATUS_INTERNAL,
  PYRE_STATUS_UNAVAILABLE,
  PYRE_STATUS_OUT_OF_MEMORY,
  PYRE_STATUS_DEADLINE_EXCEEDED,
} pyre_status_code_t;

// Status is an opaque pointer. NULL = OK, non-NULL = error with payload.
typedef struct pyre_status_s* pyre_status_t;

static inline bool pyre_status_is_ok(pyre_status_t status) {
  return status == NULL;
}

static inline pyre_status_t pyre_ok_status(void) { return NULL; }

PYRE_API pyre_status_t pyre_make_status(pyre_status_code_t code,
                                        const char* message);

PYRE_API pyre_status_code_t pyre_status_code(pyre_status_t status);

PYRE_API pyre_status_t pyre_status_to_string(pyre_status_t status,
                                             char** out_message,
                                             size_t* out_length);

PYRE_API void pyre_status_free_message(char* message);

PYRE_API void pyre_status_ignore(pyre_status_t status);

//===----------------------------------------------------------------------===//
// Opaque handle types
//===----------------------------------------------------------------------===//

typedef struct pyre_device_s* pyre_device_t;
typedef struct pyre_semaphore_s* pyre_semaphore_t;
typedef struct pyre_stream_s* pyre_stream_t;
typedef struct pyre_buffer_s* pyre_buffer_t;
typedef struct pyre_module_s* pyre_module_t;
typedef struct pyre_executable_s* pyre_executable_t;

//===----------------------------------------------------------------------===//
// Enums and flags
//===----------------------------------------------------------------------===//

typedef enum pyre_accelerator_type_t {
  PYRE_ACCELERATOR_GPU = 0,
  PYRE_ACCELERATOR_CPU = 1,
} pyre_accelerator_type_t;

typedef enum pyre_device_property_t {
  PYRE_DEVICE_PROPERTY_NAME = 0,
  PYRE_DEVICE_PROPERTY_ARCHITECTURE,
  PYRE_DEVICE_PROPERTY_TOTAL_MEMORY,
  PYRE_DEVICE_PROPERTY_COMPUTE_UNITS,
  PYRE_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE,
} pyre_device_property_t;

typedef enum pyre_memory_type_t {
  PYRE_MEMORY_DEVICE_LOCAL = 0,
  PYRE_MEMORY_HOST_VISIBLE = 1,
  PYRE_MEMORY_HOST_LOCAL = 2,
} pyre_memory_type_t;

typedef enum pyre_map_flags_t {
  PYRE_MAP_READ = 1 << 0,
  PYRE_MAP_WRITE = 1 << 1,
  PYRE_MAP_DISCARD = 1 << 2,
} pyre_map_flags_t;

typedef enum pyre_dispatch_flags_t {
  PYRE_DISPATCH_FLAG_NONE = 0,
  PYRE_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS = 1 << 0,
  PYRE_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION = 1 << 1,
} pyre_dispatch_flags_t;

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

typedef struct pyre_timeline_point_t {
  pyre_semaphore_t semaphore;
  uint64_t value;
} pyre_timeline_point_t;

typedef struct pyre_semaphore_list_t {
  pyre_semaphore_t* semaphores;
  uint64_t* values;
  size_t count;
} pyre_semaphore_list_t;

typedef uint64_t pyre_queue_affinity_t;

//===----------------------------------------------------------------------===//
// GPU accelerator lifecycle
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_gpu_initialize(uint32_t flags);
PYRE_API pyre_status_t pyre_gpu_shutdown(void);
PYRE_API pyre_status_t pyre_gpu_device_count(int* count);
PYRE_API pyre_status_t pyre_gpu_device_get(int index, pyre_device_t* device);

//===----------------------------------------------------------------------===//
// CPU accelerator lifecycle
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_cpu_initialize(uint32_t flags);
PYRE_API pyre_status_t pyre_cpu_shutdown(void);
PYRE_API pyre_status_t pyre_cpu_device_count(int* count);
PYRE_API pyre_status_t pyre_cpu_device_get(int index, pyre_device_t* device);

//===----------------------------------------------------------------------===//
// Device operations (generic, works for any accelerator type)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_device_get_property(pyre_device_t device,
                                                pyre_device_property_t prop,
                                                void* value,
                                                size_t value_size);

PYRE_API pyre_status_t pyre_device_synchronize(pyre_device_t device);

PYRE_API pyre_status_t pyre_device_get_type(pyre_device_t device,
                                            pyre_accelerator_type_t* type);

PYRE_API pyre_status_t pyre_device_retain(pyre_device_t device);
PYRE_API pyre_status_t pyre_device_release(pyre_device_t device);

//===----------------------------------------------------------------------===//
// Semaphores (timeline synchronization primitives)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_semaphore_create(pyre_device_t device,
                                             uint64_t initial_value,
                                             pyre_semaphore_t* semaphore);

PYRE_API pyre_status_t pyre_semaphore_retain(pyre_semaphore_t semaphore);
PYRE_API pyre_status_t pyre_semaphore_release(pyre_semaphore_t semaphore);

PYRE_API pyre_status_t pyre_semaphore_query(pyre_semaphore_t semaphore,
                                            uint64_t* value);

PYRE_API pyre_status_t pyre_semaphore_wait(pyre_semaphore_t semaphore,
                                           uint64_t value,
                                           uint64_t timeout_ns);

PYRE_API pyre_status_t pyre_semaphore_signal(pyre_semaphore_t semaphore,
                                             uint64_t value);

//===----------------------------------------------------------------------===//
// Streams (high-level execution contexts)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_stream_create(pyre_device_t device, uint32_t flags,
                                          pyre_stream_t* stream);

PYRE_API pyre_status_t pyre_stream_retain(pyre_stream_t stream);
PYRE_API pyre_status_t pyre_stream_release(pyre_stream_t stream);

PYRE_API pyre_status_t pyre_stream_synchronize(pyre_stream_t stream);

PYRE_API pyre_status_t pyre_stream_query(pyre_stream_t stream, bool* complete);

PYRE_API pyre_status_t pyre_stream_flush(pyre_stream_t stream);

PYRE_API pyre_status_t pyre_stream_get_semaphore(
    pyre_stream_t stream, pyre_semaphore_t* semaphore);

PYRE_API pyre_status_t pyre_stream_get_timeline_position(
    pyre_stream_t stream, pyre_timeline_point_t* position);

PYRE_API pyre_status_t pyre_stream_wait_on(pyre_stream_t stream,
                                           pyre_timeline_point_t position);

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_buffer_allocate(pyre_stream_t stream, size_t size,
                                            pyre_memory_type_t mem_type,
                                            pyre_buffer_t* buffer);

PYRE_API pyre_status_t pyre_buffer_retain(pyre_buffer_t buffer);
PYRE_API pyre_status_t pyre_buffer_release(pyre_buffer_t buffer);

PYRE_API pyre_status_t pyre_buffer_map(pyre_buffer_t buffer,
                                       pyre_map_flags_t flags, size_t offset,
                                       size_t size, void** mapped_ptr);

PYRE_API pyre_status_t pyre_buffer_unmap(pyre_buffer_t buffer);

PYRE_API pyre_status_t pyre_buffer_get_device_ptr(pyre_buffer_t buffer,
                                                  void** device_ptr);

//===----------------------------------------------------------------------===//
// Stream operations (batched via pending command buffer)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_stream_fill_buffer(pyre_stream_t stream,
                                               pyre_buffer_t buffer,
                                               size_t offset, size_t size,
                                               const void* pattern,
                                               size_t pattern_size);

PYRE_API pyre_status_t pyre_stream_copy_buffer(
    pyre_stream_t stream, pyre_buffer_t src, size_t src_offset,
    pyre_buffer_t dst, size_t dst_offset, size_t size);

PYRE_API pyre_status_t pyre_stream_update_buffer(pyre_stream_t stream,
                                                 const void* host_data,
                                                 size_t host_data_size,
                                                 pyre_buffer_t dst,
                                                 size_t dst_offset);

//===----------------------------------------------------------------------===//
// Direct queue operations (single-op, no command buffer)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_queue_fill(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores, pyre_buffer_t buffer,
    size_t offset, size_t size, const void* pattern, size_t pattern_size);

PYRE_API pyre_status_t pyre_queue_copy(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores, pyre_buffer_t src,
    size_t src_offset, pyre_buffer_t dst, size_t dst_offset, size_t size);

PYRE_API pyre_status_t pyre_queue_barrier(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_RUNTIME_H_
