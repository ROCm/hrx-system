// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_FILE_H_
#define IREE_HAL_FILE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"
#include "iree/io/file_handle.h"

typedef struct iree_async_file_t iree_async_file_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitfield specifying how a file should be opened and the access allowed.
enum iree_hal_file_mode_bits_t {
  // Opens the file if it exists on the file system.
  IREE_HAL_FILE_MODE_OPEN = 1u << 0,
};
typedef uint32_t iree_hal_file_mode_t;

// Flags for controlling imported file handle implementation details.
enum iree_hal_external_file_flag_bits_t {
  IREE_HAL_EXTERNAL_FILE_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_external_file_flags_t;

//===----------------------------------------------------------------------===//
// iree_hal_file_t
//===----------------------------------------------------------------------===//

// A file handle usable with device transfer operations.
// Production queue read/write paths expect files to expose a backend-supported
// asynchronous or device-visible representation. Examples include native async
// file handles imported into a proactor, storage-buffer-backed memory files,
// and platform-specific direct-storage objects.
//
// Files are used for bulk data upload and download and on some implementations
// may have hardware-optimized transfer paths.
//
// Implementations with support:
//  CPU: file descriptors/HANDLEs
//  CUDA: cuFile
//    https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html
//  Direct3D: IDStorageFileX
//    https://learn.microsoft.com/en-us/gaming/gdk/_content/gc/system/overviews/directstorage/directstorage-overview
//  Metal: MTLIOFileHandle
//    https://developer.apple.com/documentation/metal/resource_loading?language=objc
//
// Some implementations may allow additional non-native contents to be wrapped
// in file handles to provide implementation-controlled transfer even if not
// hardware-accelerated. See iree_hal_file_import for more information.
typedef struct iree_hal_file_t iree_hal_file_t;

// TODO(benvanik): support opening files from paths.
// IREE_API_EXPORT iree_status_t iree_hal_file_open(
//     iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
//     iree_hal_file_mode_t mode, iree_hal_memory_access_t access,
//     iree_string_view_t path, iree_hal_file_t** out_file);

// Imports an external |handle| for use on |device|.
//
// Access checks will be performed against the provided |access| bits and
// callers must ensure the access is accurate (don't allow writes to read-only
// mapped memory, etc).
//
// The provided |handle| is retained by the imported file. Callers may release
// their reference after this call returns. Any release callback attached to the
// I/O handle will run after the imported file and all queued operations release
// their references.
//
// |out_file| must be released by the caller.
// Fails with IREE_STATUS_UNAVAILABLE if the allocator cannot import the file.
// This may be due to unavailable device/platform capabilities or the properties
// of the external file handle.
IREE_API_EXPORT iree_status_t iree_hal_file_import(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file);

// Retains the given |file| for the caller.
IREE_API_EXPORT void iree_hal_file_retain(iree_hal_file_t* file);

// Releases the given |file| from the caller.
IREE_API_EXPORT void iree_hal_file_release(iree_hal_file_t* file);

// Returns the memory access allowed to the file.
// This may be more strict than the original file handle backing the resource
// if for example we want to prevent particular users from mutating the file.
IREE_API_EXPORT iree_hal_memory_access_t
iree_hal_file_allowed_access(iree_hal_file_t* file);

// Returns the immutable logical length of the accessible file range in bytes.
// This may be a portion of the original storage backing the imported handle.
// Queue and synchronous operations cannot access or extend beyond this range.
IREE_API_EXPORT uint64_t iree_hal_file_length(iree_hal_file_t* file);

// Returns an optional device-accessible storage buffer representing the file.
// Available if the implementation is able to perform import/address-space
// mapping/etc such that device-side transfers can directly access the resources
// as if they were a normal device buffer. When present the buffer covers the
// entire logical file range, carries access and usage bits compatible with the
// file, and keeps the backing storage live independently of the file object.
IREE_API_EXPORT iree_hal_buffer_t* iree_hal_file_storage_buffer(
    iree_hal_file_t* file);

// TODO(benvanik): truncate/extend? (both can be tricky with async)

// Returns the proactor-managed async file handle, or NULL if the file does not
// expose async proactor I/O. NULL does not imply the file is unusable by queue
// operations: memory-backed files may expose a storage buffer instead. Native
// platform files intended for async queue transfers must be opened in a
// platform async mode, such as IREE_IO_FILE_MODE_ASYNC.
// The returned handle is owned by the file and valid for the file's lifetime.
IREE_API_EXPORT iree_async_file_t* iree_hal_file_async_handle(
    iree_hal_file_t* file);

// Validates that |file| allows the given |required_access|.
// Returns IREE_STATUS_PERMISSION_DENIED if the access is not allowed.
IREE_API_EXPORT iree_status_t iree_hal_file_validate_access(
    iree_hal_file_t* file, iree_hal_memory_access_t required_access);

// Validates that the byte range [|byte_offset|, |byte_offset| + |byte_length|)
// is contained within the logical file range. Empty ranges are valid when
// |byte_offset| is at or before the end of the file.
// Returns IREE_STATUS_OUT_OF_RANGE if the range is invalid or overflows.
IREE_API_EXPORT iree_status_t iree_hal_file_validate_range(
    iree_hal_file_t* file, uint64_t byte_offset, uint64_t byte_length);

// Returns true if the host-side iree_hal_file_read and iree_hal_file_write APIs
// are available for use on the file. Synchronous I/O support is optional,
// backend-varying, and independent of whether queue read/write operations can
// use the file.
IREE_API_EXPORT bool iree_hal_file_supports_synchronous_io(
    iree_hal_file_t* file);

// Synchronously reads a segment of |file| into |buffer|.
// Blocks the caller until completed. The file must allow read access and the
// buffer must allow write access. Only available if
// iree_hal_file_supports_synchronous_io is true.
IREE_API_EXPORT iree_status_t iree_hal_file_read(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length);

// Synchronously writes a segment of |buffer| into |file|.
// Blocks the caller until completed. The buffer must allow read access and the
// file must allow write access. Only available if
// iree_hal_file_supports_synchronous_io is true.
IREE_API_EXPORT iree_status_t iree_hal_file_write(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length);

//===----------------------------------------------------------------------===//
// iree_hal_file_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_file_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_file_t* IREE_RESTRICT file);

  iree_hal_memory_access_t(IREE_API_PTR* allowed_access)(iree_hal_file_t* file);

  uint64_t(IREE_API_PTR* length)(iree_hal_file_t* file);

  iree_hal_buffer_t*(IREE_API_PTR* storage_buffer)(iree_hal_file_t* file);

  iree_async_file_t*(IREE_API_PTR* async_handle)(iree_hal_file_t* file);

  bool(IREE_API_PTR* supports_synchronous_io)(iree_hal_file_t* file);
  iree_status_t(IREE_API_PTR* read)(iree_hal_file_t* file, uint64_t file_offset,
                                    iree_hal_buffer_t* buffer,
                                    iree_device_size_t buffer_offset,
                                    iree_device_size_t length);
  iree_status_t(IREE_API_PTR* write)(iree_hal_file_t* file,
                                     uint64_t file_offset,
                                     iree_hal_buffer_t* buffer,
                                     iree_device_size_t buffer_offset,
                                     iree_device_size_t length);
} iree_hal_file_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_file_vtable_t);

IREE_API_EXPORT void iree_hal_file_destroy(iree_hal_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_FILE_H_
