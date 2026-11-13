// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_contents.h"

#if IREE_FILE_IO_ENABLE
#include <errno.h>

#include "iree/io/stdio_util.h"
#endif  // IREE_FILE_IO_ENABLE

//===----------------------------------------------------------------------===//
// iree_io_file_contents_t
//===----------------------------------------------------------------------===//

// We could take alignment as an arg, but roughly page aligned should be
// acceptable for all uses - if someone cares about memory usage they won't
// be using preloading.
#define IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT 4096

IREE_API_EXPORT void iree_io_file_contents_free(
    iree_io_file_contents_t* contents) {
  if (!contents) {
    return;
  }
  iree_allocator_t allocator = contents->allocator;
  iree_io_file_mapping_release(contents->mapping);
  iree_allocator_free(allocator, contents);
}

static iree_status_t iree_io_file_contents_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  if (command == IREE_ALLOCATOR_COMMAND_TRIM) {
    // All owned storage belongs to the still-live file contents.
    return iree_ok_status();
  }
  if (command != IREE_ALLOCATOR_COMMAND_FREE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "file contents deallocator must only be used to "
                            "deallocate file contents");
  }
  iree_io_file_contents_t* contents = (iree_io_file_contents_t*)self;
  if (contents->buffer.data != *inout_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only the file contents buffer is valid");
  }
  iree_io_file_contents_free(contents);
  return iree_ok_status();
}

IREE_API_EXPORT iree_allocator_t
iree_io_file_contents_deallocator(iree_io_file_contents_t* contents) {
  iree_allocator_t allocator = {
      .self = contents,
      .ctl = iree_io_file_contents_allocator_ctl,
  };
  return allocator;
}

#if IREE_FILE_IO_ENABLE

IREE_API_EXPORT iree_status_t iree_io_file_contents_read_stdin(
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fix stdin mode to binary on Windows to match Unix behavior.
  IREE_IO_SET_BINARY_MODE(stdin);

  iree_host_size_t capacity = 4096;
  iree_host_size_t total_size = 0;
  iree_host_size_t data_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_io_file_contents_t), &total_size,
              IREE_STRUCT_FIELD_ALIGNED(capacity, uint8_t,
                                        IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT,
                                        &data_offset)));
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&contents));
  contents->buffer.data = (void*)((uint8_t*)contents + data_offset);

  iree_host_size_t size = 0;
  for (int c = getchar(); c != EOF; c = getchar()) {
    if (size >= capacity - /*NUL*/ 1) {
      // NOTE: if we realloc we may end up with a new alignment and need to move
      // the data around.
      uintptr_t old_offset =
          (uintptr_t)contents->buffer.data - (uintptr_t)contents;
      iree_host_size_t new_capacity = capacity * 2;
      iree_host_size_t new_total_size = 0;
      iree_host_size_t new_data_offset = 0;
      iree_status_t status = IREE_STRUCT_LAYOUT(
          sizeof(iree_io_file_contents_t), &new_total_size,
          IREE_STRUCT_FIELD_ALIGNED(new_capacity, uint8_t,
                                    IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT,
                                    &new_data_offset));
      iree_io_file_contents_t* new_contents = contents;
      if (iree_status_is_ok(status)) {
        status = iree_allocator_realloc(host_allocator, new_total_size,
                                        (void**)&new_contents);
      }
      if (!iree_status_is_ok(status)) {
        iree_allocator_free(host_allocator, contents);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
      contents = new_contents;
      uint8_t* old_data = (uint8_t*)new_contents + old_offset;
      uint8_t* new_data = (uint8_t*)new_contents + new_data_offset;
      if (new_data != old_data) {
        // Alignment changed; move the data with safety for overlapping.
        memmove(new_data, old_data, size);
      }
      contents->buffer.data = new_data;
      capacity = new_capacity;
    }
    contents->buffer.data[size++] = c;
  }

  contents->allocator = host_allocator;
  contents->buffer.data[size] = 0;  // NUL
  contents->buffer.data_length = size;
  *out_contents = contents;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Returns the size, in bytes, of the |file|. Must be seekable.
static uint64_t iree_io_stdio_file_length(FILE* file) {
  IREE_ASSERT_ARGUMENT(file);

  // Capture original offset so we can return to it.
  int64_t origin = iree_ftell(file);
  if (origin == -1) {
    return 0;
  }

  // Seek to the end of the file.
  if (iree_fseek(file, 0, SEEK_END) != 0) {
    return 0;
  }

  // Query the position, telling us the total file length in bytes.
  int64_t length = iree_ftell(file);
  if (length == -1) {
    return 0;
  }

  // Seek back to the file origin.
  if (iree_fseek(file, origin, SEEK_SET) != 0) {
    return 0;
  }

  return (uint64_t)length;
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_read(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Open the file for reading.
  FILE* file = NULL;
  iree_status_t status =
      iree_io_stdio_file_open(path, "rb", host_allocator, &file);

  // Query the file size.
  iree_host_size_t file_size = 0;
  if (iree_status_is_ok(status)) {
    uint64_t file_size_u64 = iree_io_stdio_file_length(file);
    if (file_size_u64 < IREE_HOST_SIZE_MAX) {
      file_size = (iree_host_size_t)file_size_u64;
    } else {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "file size %" PRIu64
                                " exceeds host size maximum of %" PRIhsz
                                " bytes and cannot be read into memory",
                                file_size_u64, IREE_HOST_SIZE_MAX);
    }
  }

  // Allocate the file contents struct and its embedded buffer data.
  // We compute total size with alignment padding and allocate +1 bytes to force
  // a trailing \0 in case this is used as a cstring.
  iree_host_size_t file_total_size = 0;
  iree_host_size_t file_data_offset = 0;
  iree_io_file_contents_t* contents = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_io_file_contents_t), &file_total_size,
        IREE_STRUCT_FIELD_ALIGNED(file_size + /*NUL*/ 1, uint8_t,
                                  IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT,
                                  &file_data_offset));
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, file_total_size,
                                   (void**)&contents);
  }
  if (iree_status_is_ok(status)) {
    contents->allocator = host_allocator;
    contents->buffer = iree_make_byte_span(
        (void*)((uint8_t*)contents + file_data_offset), file_size);

    // Add trailing NUL to make the contents C-string compatible.
    contents->buffer.data[file_size] = 0;  // NUL
  }

  // Attempt to read the file into memory, chunking into ~2GB segments.
  // Several implementations of read have limits at ~2GB even on 64-bit systems.
  iree_host_size_t bytes_read = 0;
  while (iree_status_is_ok(status) && bytes_read < file_size) {
    const iree_host_size_t chunk_size =
        iree_min(file_size - bytes_read, INT_MAX);
    if (fread(contents->buffer.data + bytes_read, 1, chunk_size, file) !=
        chunk_size) {
      status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "unable to read %" PRIhsz " chunk bytes",
                                chunk_size);
      break;
    }
    bytes_read += chunk_size;
  }

  if (file) {
    fclose(file);
  }

  if (iree_status_is_ok(status)) {
    *out_contents = contents;
  } else {
    iree_io_file_contents_free(contents);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_map(
    iree_string_view_t path, iree_io_file_access_t access,
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_file_mode_t mode = IREE_IO_FILE_MODE_READ |
                             IREE_IO_FILE_MODE_SHARE_READ |
                             IREE_IO_FILE_MODE_SHARE_WRITE;
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
    mode |= IREE_IO_FILE_MODE_WRITE;
  }

  iree_io_file_handle_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_open(mode, path, host_allocator, &handle));

  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*contents), (void**)&contents);
  if (iree_status_is_ok(status)) {
    contents->allocator = host_allocator;
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_io_file_map_view(handle, access, 0, IREE_HOST_SIZE_MAX,
                              IREE_IO_FILE_MAPPING_FLAG_PRIVATE |
                                  IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS,
                              host_allocator, &contents->mapping);
  }

  iree_io_file_handle_release(handle);

  if (iree_status_is_ok(status)) {
    if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
      contents->buffer = iree_io_file_mapping_contents_rw(contents->mapping);
    } else {
      contents->const_buffer =
          iree_io_file_mapping_contents_ro(contents->mapping);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_contents = contents;
  } else {
    iree_io_file_contents_free(contents);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_write(
    iree_string_view_t path, iree_const_byte_span_t contents,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Open file for overwriting (if it exists).
  FILE* file = NULL;
  iree_status_t status =
      iree_io_stdio_file_open(path, "wb", host_allocator, &file);

  // Write all contents (if any) to the file. Chunk writes to avoid platform
  // stdio limits and retain the exact progress if the backing store fills.
  iree_host_size_t bytes_written = 0;
  while (iree_status_is_ok(status) && bytes_written < contents.data_length) {
    const iree_host_size_t chunk_size =
        iree_min(contents.data_length - bytes_written, INT_MAX);
    errno = 0;
    const iree_host_size_t write_size =
        fwrite(contents.data + bytes_written, 1, chunk_size, file);
    bytes_written += write_size;
    if (write_size != chunk_size) {
      const int write_error = errno;
      if (write_error != 0) {
        status = iree_make_status(iree_status_code_from_errno(write_error),
                                  "unable to write file '%.*s' after %" PRIhsz
                                  "/%" PRIhsz " bytes (%d: %s)",
                                  (int)path.size, path.data, bytes_written,
                                  contents.data_length, write_error,
                                  strerror(write_error));
      } else {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "short write to file '%.*s' after %" PRIhsz
                                  "/%" PRIhsz " bytes without a stdio error",
                                  (int)path.size, path.data, bytes_written,
                                  contents.data_length);
      }
    }
  }

  // Closing publishes any writes still buffered by stdio. Preserve a prior
  // write failure as the primary status while retaining a distinct close
  // failure as additional diagnostic context.
  if (file) {
    errno = 0;
    if (fclose(file) != 0) {
      const int close_error = errno;
      iree_status_t close_status =
          close_error != 0
              ? iree_make_status(iree_status_code_from_errno(close_error),
                                 "unable to close file '%.*s' (%d: %s)",
                                 (int)path.size, path.data, close_error,
                                 strerror(close_error))
              : iree_make_status(IREE_STATUS_DATA_LOSS,
                                 "unable to close file '%.*s' without a "
                                 "stdio error",
                                 (int)path.size, path.data);
      status = iree_status_join(status, close_status);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

IREE_API_EXPORT iree_status_t iree_io_file_contents_read_stdin(
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_read(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_map(
    iree_string_view_t path, iree_io_file_access_t access,
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_write(
    iree_string_view_t path, iree_const_byte_span_t contents,
    iree_allocator_t host_allocator) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

#endif  // IREE_FILE_IO_ENABLE
