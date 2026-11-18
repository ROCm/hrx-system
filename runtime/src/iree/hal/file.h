// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_FILE_H_
#define IREE_HAL_FILE_H_

#include <stdint.h>

#include "iree/async/file.h"
#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;
typedef struct iree_hal_pool_t iree_hal_pool_t;

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
// Current CPU implementations use file descriptors or HANDLEs.
//
// Some implementations may allow additional non-native contents to be wrapped
// in file handles to provide implementation-controlled transfer even if not
// hardware-accelerated. See iree_hal_file_import for more information.
typedef struct iree_hal_file_t iree_hal_file_t;

// Imports an external |handle| for use on |device|.
// The device must outlive the returned file and accepted operations using it;
// the file may borrow the device's native resources and progress services.
//
// |storage_pool| optionally supplies the contract and native route used to
// expose memory-backed files as device-visible storage buffers. A NULL pool or
// one that does not admit the handle's external storage class leaves the file
// on its native I/O path. Once a pool admits the class, native attachment is a
// selected route and attachment failures are returned to the caller. The pool
// must outlive the returned file and all accepted operations using it.
//
// |queue_family_affinity| specifies the queue families that may access any
// device-visible resources created for the imported file. It is a resource
// accessibility domain and does not select a queue or establish ordering.
// IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY makes the file available to every queue
// family owned by |device|.
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
// Fails with IREE_STATUS_UNAVAILABLE if the selected implementation cannot
// import the file.
// This may be due to unavailable device/platform capabilities or the properties
// of the external file handle.
IREE_API_EXPORT iree_status_t iree_hal_file_import(
    iree_hal_device_t* device, iree_hal_pool_t* storage_pool,
    iree_hal_queue_family_affinity_t queue_family_affinity,
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

// Returns immutable native host-transfer requirements retained at import.
// The result is borrowed for the file's lifetime. Memory-backed files are
// byte-granular. Device-copy routes have their own buffer requirements.
// This accessor performs no native queries, allocations, locking, or lazy
// initialization. A file whose requirements cannot be established fails import.
IREE_API_EXPORT const iree_async_file_requirements_t*
iree_hal_file_requirements(iree_hal_file_t* file);

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

// Decodes one published little-endian device record. LINEAR ignores the suffix
// words, and zero row length ignores all other fields. Empty records only set
// out_transfer->row_length to zero; all remaining fields are unspecified.
// Nonempty records initialize every field. Invalid forms or offsets that the
// host's iree_device_size_t cannot represent fail without producing a transfer.
// Record storage must be readable for the fields selected by its form.
IREE_API_EXPORT iree_status_t iree_hal_file_transfer_record_decode(
    const iree_hal_file_transfer_record_t* record,
    iree_hal_file_transfer_t* out_transfer);

// Validates binding indices and addressed extents without querying resources.
// The enclosing operation first qualifies each binding's resource and window
// and resolves all buffer lengths (no WHOLE_BUFFER). This checks only the
// transfer's relative geometry against those windows, in constant time without
// expanding rows. Empty transfers ignore indices, offsets, and strides. Unused
// dimensions ignore their strides. Overlapping regions remain the caller's
// synchronization responsibility.
IREE_API_EXPORT iree_status_t
iree_hal_file_transfer_validate(iree_hal_file_binding_table_t files,
                                iree_hal_buffer_binding_table_t buffers,
                                const iree_hal_file_transfer_t* transfer);

// Validates a remaining native transfer and selects its next request length.
// |requirements| must come from a successful requirements query.
// |remaining_length| is the logical transfer remainder, which may exceed the
// capacity of a staging buffer. |buffer| supplies storage for at least
// min(|remaining_length|, |maximum_length|) bytes; reads require writable
// storage. A split request preserves buffer, offset, and length alignment for
// the following request. Empty transfers return zero without checking
// alignment. An unaligned request (including a suffix after a short native
// completion) fails without submitting I/O. RESOURCE_EXHAUSTED means
// |maximum_length| cannot hold an aligned chunk. Failure sets |out_length| to
// zero. Successful preparation performs no I/O, allocation, locking, or hidden
// staging; failures may allocate diagnostic status storage.
IREE_API_EXPORT iree_status_t iree_hal_file_prepare_transfer(
    const iree_async_file_transfer_requirements_t* requirements,
    uint64_t file_offset, const void* buffer, uint64_t remaining_length,
    iree_host_size_t maximum_length, iree_host_size_t* out_length);

// Returns true if the host-side iree_hal_file_read and iree_hal_file_write APIs
// are available for use on the file. Synchronous I/O support is optional,
// backend-varying, and independent of whether queue read/write operations can
// use the file.
IREE_API_EXPORT bool iree_hal_file_supports_synchronous_io(
    iree_hal_file_t* file);

// Synchronously reads a segment of |file| into |target|.
// Blocks the caller until the entire span is filled or an error occurs. The
// file must allow read access. The caller supplies writable host storage and
// owns its synchronization, cache maintenance and lifetime through this call.
// The buffer address, offset and length must satisfy the file's retained read
// requirements; incompatible requests fail with INVALID_ARGUMENT.
// An error may leave a partially filled span. Empty spans are valid, including
// at the end of the file. Only available if
// iree_hal_file_supports_synchronous_io is true.
IREE_API_EXPORT iree_status_t iree_hal_file_read(iree_hal_file_t* file,
                                                 uint64_t file_offset,
                                                 iree_byte_span_t target);

// Synchronously writes |source| into a segment of |file|.
// Blocks the caller until the entire span is written or an error occurs. The
// file must allow write access. The caller supplies readable host storage and
// owns its synchronization, cache maintenance and lifetime through this call.
// The buffer address, offset and length must satisfy the file's retained write
// requirements; incompatible requests fail with INVALID_ARGUMENT.
// An error may leave a partially written file range. Empty spans are valid,
// including at the end of the file. Only available if
// iree_hal_file_supports_synchronous_io is true.
IREE_API_EXPORT iree_status_t iree_hal_file_write(
    iree_hal_file_t* file, uint64_t file_offset, iree_const_byte_span_t source);

//===----------------------------------------------------------------------===//
// iree_hal_file_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_file_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_file_t* IREE_RESTRICT file);

  iree_hal_memory_access_t(IREE_API_PTR* allowed_access)(iree_hal_file_t* file);

  uint64_t(IREE_API_PTR* length)(iree_hal_file_t* file);

  const iree_async_file_requirements_t*(IREE_API_PTR* requirements)(
      iree_hal_file_t* file);

  iree_hal_buffer_t*(IREE_API_PTR* storage_buffer)(iree_hal_file_t* file);

  iree_async_file_t*(IREE_API_PTR* async_handle)(iree_hal_file_t* file);

  bool(IREE_API_PTR* supports_synchronous_io)(iree_hal_file_t* file);
  iree_status_t(IREE_API_PTR* read)(iree_hal_file_t* file, uint64_t file_offset,
                                    iree_byte_span_t target);
  iree_status_t(IREE_API_PTR* write)(iree_hal_file_t* file,
                                     uint64_t file_offset,
                                     iree_const_byte_span_t source);
} iree_hal_file_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_file_vtable_t);

IREE_API_EXPORT void iree_hal_file_destroy(iree_hal_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_FILE_H_
