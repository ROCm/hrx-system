// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_BASE_H_
#define HRX_BASE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Export macros
//===----------------------------------------------------------------------===//

#if defined(_WIN32) && defined(HRX_BUILDING_SHARED)
#define HRX_API __declspec(dllexport)
#elif defined(_WIN32) && !defined(HRX_STATIC)
#define HRX_API __declspec(dllimport)
#elif defined(HRX_BUILDING_SHARED)
#define HRX_API __attribute__((visibility("default")))
#elif defined(HRX_STATIC)
#define HRX_API
#else
#define HRX_API
#endif

//===----------------------------------------------------------------------===//
// Host allocator (type + system getter only — functions after Status)
//
// Two-word value type, layout-compatible with iree_allocator_t.
// Used for host-side memory allocation (not device memory).
// hrx_host_allocator_system() returns the process-wide system allocator,
// backed by mimalloc when available.
//===----------------------------------------------------------------------===//

typedef struct hrx_host_allocator_t {
  // Borrowed allocator state, kept alive until all owned resources are
  // released.
  void* self;
  // Allocator control function; layout-compatible with iree_allocator_t::ctl.
  void* ctl;
} hrx_host_allocator_t;

// Returns the process-wide system allocator. Valid as soon as libhrx is loaded
// (no runtime or device group required).
HRX_API hrx_host_allocator_t hrx_host_allocator_system(void);

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

#define HRX_VERSION_MAJOR 0
#define HRX_VERSION_MINOR 1
#define HRX_VERSION_PATCH 0

HRX_API void hrx_runtime_version(int* major, int* minor, int* patch);

//===----------------------------------------------------------------------===//
// Status
//
// Values match iree_status_code_t. Verified by static_assert in
// implementation.
//===----------------------------------------------------------------------===//

typedef enum hrx_status_code_t {
  HRX_STATUS_OK = 0,
  HRX_STATUS_CANCELLED = 1,
  HRX_STATUS_UNKNOWN = 2,
  HRX_STATUS_INVALID_ARGUMENT = 3,
  HRX_STATUS_DEADLINE_EXCEEDED = 4,
  HRX_STATUS_NOT_FOUND = 5,
  HRX_STATUS_ALREADY_EXISTS = 6,
  HRX_STATUS_PERMISSION_DENIED = 7,
  HRX_STATUS_OUT_OF_MEMORY = 8,  // IREE: RESOURCE_EXHAUSTED
  HRX_STATUS_FAILED_PRECONDITION = 9,
  HRX_STATUS_ABORTED = 10,
  HRX_STATUS_OUT_OF_RANGE = 11,
  HRX_STATUS_UNIMPLEMENTED = 12,
  HRX_STATUS_INTERNAL = 13,
  HRX_STATUS_UNAVAILABLE = 14,
  HRX_STATUS_DATA_LOSS = 15,
} hrx_status_code_t;

// Status is an opaque pointer. NULL = OK, non-NULL = error with payload.
typedef struct hrx_status_s* hrx_status_t;

static inline bool hrx_status_is_ok(hrx_status_t status) {
  return status == NULL;
}

static inline hrx_status_t hrx_ok_status(void) { return NULL; }

HRX_API hrx_status_t hrx_make_status(hrx_status_code_t code,
                                     const char* message);

HRX_API hrx_status_code_t hrx_status_code(hrx_status_t status);

HRX_API hrx_status_t hrx_status_to_string(hrx_status_t status,
                                          char** out_message,
                                          size_t* out_length);

HRX_API void hrx_status_free_message(char* message);

HRX_API void hrx_status_ignore(hrx_status_t status);

//===----------------------------------------------------------------------===//
// Host allocator functions
//===----------------------------------------------------------------------===//

// Allocates zeroed memory.
HRX_API hrx_status_t hrx_host_allocator_malloc(hrx_host_allocator_t allocator,
                                               size_t byte_length,
                                               void** out_ptr);

// Allocates uninitialized memory. Only use when immediately overwriting.
HRX_API hrx_status_t hrx_host_allocator_malloc_uninitialized(
    hrx_host_allocator_t allocator, size_t byte_length, void** out_ptr);

// Reallocates to |byte_length|. Extended bytes are undefined.
HRX_API hrx_status_t hrx_host_allocator_realloc(hrx_host_allocator_t allocator,
                                                size_t byte_length,
                                                void** inout_ptr);

// Duplicates a byte block.
HRX_API hrx_status_t hrx_host_allocator_clone(hrx_host_allocator_t allocator,
                                              const void* src,
                                              size_t byte_length,
                                              void** out_ptr);

// Frees memory. NULL is a no-op.
HRX_API void hrx_host_allocator_free(hrx_host_allocator_t allocator, void* ptr);

// Allocates zeroed, aligned memory. The byte at |offset| within the
// allocation will be aligned to at least |min_alignment|.
HRX_API hrx_status_t hrx_host_allocator_malloc_aligned(
    hrx_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** out_ptr);

// Reallocates aligned memory.
HRX_API hrx_status_t hrx_host_allocator_realloc_aligned(
    hrx_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** inout_ptr);

// Frees memory from hrx_host_allocator_malloc_aligned. NULL is a no-op.
HRX_API void hrx_host_allocator_free_aligned(hrx_host_allocator_t allocator,
                                             void* ptr);

// Borrowed string storage valid only for the duration documented by the
// containing API.
typedef struct hrx_string_view_t {
  // Borrowed string data; not necessarily NUL terminated.
  const char* data;
  // String length in bytes.
  size_t size;
} hrx_string_view_t;

// Borrowed byte storage valid only for the duration documented by the
// containing API.
typedef struct hrx_const_byte_span_t {
  // Borrowed byte data, or NULL when |data_length| is zero.
  const void* data;
  // Byte length of |data|.
  size_t data_length;
} hrx_const_byte_span_t;

#ifdef __cplusplus
}
#endif

#endif  // HRX_BASE_H_
