// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_MEMORY_FILE_H_
#define IREE_HAL_UTILS_MEMORY_FILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_memory_file_t
//===----------------------------------------------------------------------===//

// Flags controlling memory file construction.
enum iree_hal_memory_file_flag_bits_t {
  IREE_HAL_MEMORY_FILE_FLAG_NONE = 0u,

  // Requires the host allocation to be imported as a device-visible storage
  // buffer. Construction fails when the allocator cannot provide the storage
  // buffer instead of returning a host-only file.
  IREE_HAL_MEMORY_FILE_FLAG_REQUIRE_DEVICE_VISIBLE_STORAGE = 1u << 0,
};
typedef uint32_t iree_hal_memory_file_flags_t;

// Creates a file backed by |handle| without copying the data.
// Only supports file handles of IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION.
// If |device_allocator| is provided the memory will be imported as a
// device-accessible storage buffer when possible. Import failure leaves the
// file host-only: it still supports synchronous I/O and can be used by devices
// with an explicit staging path. When
// IREE_HAL_MEMORY_FILE_FLAG_REQUIRE_DEVICE_VISIBLE_STORAGE is set, import
// failure returns IREE_STATUS_UNAVAILABLE instead.
//
// The file retains |handle| and therefore keeps its host allocation live until
// all file references and accepted queue operations are released. It does not
// synchronize concurrent host access. Callers must use queue semaphore
// dependencies to order host producers before reads and wait for queue write
// completion before accessing written bytes.
IREE_API_EXPORT iree_status_t iree_hal_memory_file_wrap(
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_memory_file_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file);

// Returns true if |file| is a memory file created by
// iree_hal_memory_file_wrap.
IREE_API_EXPORT bool iree_hal_memory_file_isa(iree_hal_file_t* file);

// Returns the host byte span backing |file|.
IREE_API_EXPORT iree_status_t iree_hal_memory_file_contents(
    iree_hal_file_t* file, iree_byte_span_t* out_contents);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_MEMORY_FILE_H_
