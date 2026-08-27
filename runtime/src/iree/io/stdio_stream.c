// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/stdio_stream.h"

#include <sys/stat.h>
#include <sys/types.h>

#include "iree/base/internal/path.h"
#include "iree/io/stdio_util.h"

//===----------------------------------------------------------------------===//
// iree_io_stdio_stream_t
//===----------------------------------------------------------------------===//

typedef struct iree_io_stdio_stream_t {
  iree_io_stream_t base;
  iree_allocator_t host_allocator;
  FILE* handle;
  bool owns_handle;
} iree_io_stdio_stream_t;

static const iree_io_stream_vtable_t iree_io_stdio_stream_vtable;

static iree_io_stdio_stream_t* iree_io_stdio_stream_cast(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  return (iree_io_stdio_stream_t*)base_stream;
}

IREE_API_EXPORT iree_status_t iree_io_stdio_stream_wrap(
    iree_io_stream_mode_t mode, FILE* handle, bool owns_handle,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stdio_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));
  iree_atomic_ref_count_init(&stream->base.ref_count);
  stream->base.vtable = &iree_io_stdio_stream_vtable;
  stream->base.mode = mode;
  stream->host_allocator = host_allocator;
  stream->handle = handle;
  stream->owns_handle = owns_handle;

  *out_stream = &stream->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#if IREE_FILE_IO_ENABLE

IREE_API_EXPORT iree_status_t
iree_io_stdio_file_open(iree_string_view_t path, const char* mode,
                        iree_allocator_t host_allocator, FILE** out_file) {
  IREE_ASSERT_ARGUMENT(mode);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  if (iree_string_view_is_empty(path)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file path must not be empty");
  }
  if (!path.data) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file path storage must not be NULL");
  }
  if (memchr(path.data, 0, path.size) != NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file path contains an embedded NUL character");
  }

  iree_status_t status = iree_ok_status();
  FILE* file = NULL;
  int open_error = 0;
#if defined(IREE_PLATFORM_WINDOWS)
  wchar_t* win32_path = NULL;
  status = iree_file_path_to_win32(path, host_allocator, &win32_path);
  if (iree_status_is_ok(status)) {
    const iree_host_size_t mode_length = strlen(mode);
    wchar_t mode_wide[16] = {0};
    if (mode_length >= IREE_ARRAYSIZE(mode_wide)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stdio mode is too long");
    } else {
      for (iree_host_size_t i = 0; i < mode_length; ++i) {
        mode_wide[i] = (wchar_t)mode[i];
      }
      file = _wfopen(win32_path, mode_wide);
      if (file == NULL) open_error = errno;
    }
  }
  iree_allocator_free(host_allocator, win32_path);
#else
  char* path_cstring = NULL;
  status = iree_allocator_malloc(host_allocator, path.size + 1,
                                 (void**)&path_cstring);
  if (iree_status_is_ok(status)) {
    iree_string_view_to_cstring(path, path_cstring, path.size + 1);
    file = fopen(path_cstring, mode);
    if (file == NULL) open_error = errno;
  }
  iree_allocator_free(host_allocator, path_cstring);
#endif  // IREE_PLATFORM_WINDOWS

  if (iree_status_is_ok(status) && file == NULL) {
    if (open_error == 0) {
      status = iree_make_status(IREE_STATUS_UNKNOWN,
                                "unable to open file `%.*s` with mode `%s`",
                                (int)path.size, path.data, mode);
    } else {
      status = iree_make_status(
          iree_status_code_from_errno(open_error),
          "unable to open file `%.*s` with mode `%s` (%d: %s)", (int)path.size,
          path.data, mode, open_error, strerror(open_error));
    }
  }

  if (iree_status_is_ok(status)) {
    *out_file = file;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Populates the |out_fopen_mode| string for use with fopen-like calls based on
// the iree_io_stdio_stream_mode_t bitmap.
//
// NOTE: not all implementations support all mode flags and this may have
// different behavior. We should paper over it here but don't today given the
// limited usage of this and our intent to rewrite it all using
// platform-optimal APIs instead of stdio.
static void iree_io_map_stdio_fopen_mode(iree_io_stdio_stream_mode_t stdio_mode,
                                         char out_fopen_mode[16]) {
  memset(out_fopen_mode, 0, 16);

  if (iree_all_bits_set(stdio_mode, IREE_IO_STDIO_STREAM_MODE_READ |
                                        IREE_IO_STDIO_STREAM_MODE_WRITE |
                                        IREE_IO_STDIO_STREAM_MODE_APPEND)) {
    strcat(out_fopen_mode, "a+");
  } else if (iree_all_bits_set(stdio_mode,
                               IREE_IO_STDIO_STREAM_MODE_READ |
                                   IREE_IO_STDIO_STREAM_MODE_WRITE |
                                   IREE_IO_STDIO_STREAM_MODE_DISCARD)) {
    strcat(out_fopen_mode, "w+");
  } else if (iree_all_bits_set(stdio_mode,
                               IREE_IO_STDIO_STREAM_MODE_READ |
                                   IREE_IO_STDIO_STREAM_MODE_WRITE)) {
    strcat(out_fopen_mode, "r+");
  } else if (iree_all_bits_set(stdio_mode,
                               IREE_IO_STDIO_STREAM_MODE_WRITE |
                                   IREE_IO_STDIO_STREAM_MODE_APPEND)) {
    strcat(out_fopen_mode, "a");
  } else if (iree_all_bits_set(stdio_mode, IREE_IO_STDIO_STREAM_MODE_WRITE)) {
    strcat(out_fopen_mode, "w");
  } else if (iree_all_bits_set(stdio_mode, IREE_IO_STDIO_STREAM_MODE_READ)) {
    strcat(out_fopen_mode, "r");
  }
  if (iree_all_bits_set(stdio_mode, IREE_IO_STDIO_STREAM_MODE_WRITE) &&
      !iree_all_bits_set(stdio_mode, IREE_IO_STDIO_STREAM_MODE_DISCARD)) {
    // If writable and not discard then the file must not exist.
    // TODO(benvanik): actually observe this; the C11 spec says `x` is supported
    // but at least on MSVC's CRT it isn't. We can emulate this with stat and
    // such but today we don't have any uses that require it.
    // strcat(out_fopen_mode, "x");
  }
  // Force binary mode (avoid Windows CRLF expansion).
  strcat(out_fopen_mode, "b");
}

IREE_API_EXPORT iree_status_t iree_io_stdio_stream_open(
    iree_io_stdio_stream_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_io_stream_mode_t stream_mode = IREE_IO_STREAM_MODE_SEEKABLE;
  if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_READ)) {
    stream_mode |= IREE_IO_STREAM_MODE_READABLE;
  }
  if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_WRITE)) {
    stream_mode |= IREE_IO_STREAM_MODE_WRITABLE;
  }

  char fopen_mode[16] = {0};
  iree_io_map_stdio_fopen_mode(mode, fopen_mode);

  FILE* handle = NULL;
  iree_status_t status =
      iree_io_stdio_file_open(path, fopen_mode, host_allocator, &handle);

  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_stdio_stream_wrap(
        stream_mode, handle, /*owns_handle=*/true, host_allocator, &stream);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
    if (handle) {
      fclose(handle);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_stdio_stream_open_fd(
    iree_io_stdio_stream_mode_t mode, int fd, iree_allocator_t host_allocator,
    iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Duplicate the file descriptor so that we have our own copy of the seek
  // position. The initial position will be preserved.
  int dup_fd = iree_dup(fd);
  if (dup_fd == -1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_stdio_status(
        "unable to duplicate file descriptor; possibly out of file descriptors "
        "(see ulimit)");
  }

  iree_io_stream_mode_t stream_mode = IREE_IO_STREAM_MODE_SEEKABLE;
  if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_READ)) {
    stream_mode |= IREE_IO_STREAM_MODE_READABLE;
  }
  if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_WRITE)) {
    stream_mode |= IREE_IO_STREAM_MODE_WRITABLE;
  }

  char fopen_mode[16] = {0};
  iree_io_map_stdio_fopen_mode(mode, fopen_mode);

  // NOTE: after this point the file handle is associated with dup_fd and
  // anything we do to it (like closing) will apply to the dup_fd.
  iree_status_t status = iree_ok_status();
  FILE* handle = iree_fdopen(dup_fd, fopen_mode);
  if (handle == NULL) {
    status = iree_make_stdio_statusf(
        "unable to open file descriptor with mode %d", mode);
  }

  // Ownership of the handle (and the dup_fd backing it) is transferred.
  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_stdio_stream_wrap(
        stream_mode, handle, /*owns_handle=*/true, host_allocator, &stream);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    if (stream) {
      // NOTE: closes the file handle/dup_fd.
      iree_io_stream_release(stream);
    } else if (handle) {
      // NOTE: closes the dup_fd.
      fclose(handle);
    } else if (dup_fd >= 0) {
      iree_close(dup_fd);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

IREE_API_EXPORT iree_status_t
iree_io_stdio_file_open(iree_string_view_t path, const char* mode,
                        iree_allocator_t host_allocator, FILE** out_file) {
  IREE_ASSERT_ARGUMENT(mode);
  IREE_ASSERT_ARGUMENT(out_file);
  (void)path;
  (void)host_allocator;
  *out_file = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_stdio_stream_open(
    iree_io_stdio_stream_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_stdio_stream_open_fd(
    iree_io_stdio_stream_mode_t mode, int fd, iree_allocator_t host_allocator,
    iree_io_stream_t** out_stream) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

#endif  // IREE_FILE_IO_ENABLE

static void iree_io_stdio_stream_destroy(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  iree_allocator_t host_allocator = stream->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  fflush(stream->handle);
  if (stream->owns_handle) {
    fclose(stream->handle);
  }

  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

static iree_io_stream_pos_t iree_io_stdio_stream_offset(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  int64_t pos = iree_ftell(stream->handle);
  if (pos == -1) return 0;
  return (iree_io_stream_pos_t)pos;
}

static iree_io_stream_pos_t iree_io_stdio_stream_length(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);

  // Capture original offset so we can return to it.
  int64_t origin = iree_ftell(stream->handle);
  if (origin == -1) return 0;

  // Seek to the end of the file.
  if (iree_fseek(stream->handle, 0, SEEK_END) != 0) return 0;

  // Query the position, telling us the total file length in bytes.
  int64_t length = iree_ftell(stream->handle);
  if (length == -1) return 0;

  // Seek back to the file origin.
  if (iree_fseek(stream->handle, origin, SEEK_SET) != 0) return 0;

  return (iree_io_stream_pos_t)length;
}

static iree_status_t iree_io_stdio_stream_seek(
    iree_io_stream_t* base_stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t seek_offset) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  int origin = 0;
  switch (seek_mode) {
    case IREE_IO_STREAM_SEEK_SET:
      origin = SEEK_SET;
      break;
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      origin = SEEK_CUR;
      break;
    case IREE_IO_STREAM_SEEK_FROM_END:
      origin = SEEK_END;
      break;
    default:
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid seek mode");
      break;
  }

  if (iree_status_is_ok(status)) {
    if (iree_fseek(stream->handle, seek_offset, origin) != 0) {
      status = iree_make_stdio_status("failed to seek");
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_read(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_capacity,
    void* buffer, iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  if (out_buffer_length) *out_buffer_length = 0;
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // Read in ~2GB chunks - even platforms with 64-bit support sometimes don't
  // like read lengths >2GB and there's not really any benefit to doing 12GB
  // reads in one go anyway.
  iree_host_size_t bytes_read = 0;
  while (bytes_read < buffer_capacity) {
    const iree_host_size_t chunk_size =
        iree_min(buffer_capacity - bytes_read, INT_MAX);
    const iree_host_size_t read_size =
        fread((uint8_t*)buffer + bytes_read, 1, chunk_size, stream->handle);
    if (read_size != chunk_size) {
      // Failed to read chunk - may have reached EOF.
      if (feof(stream->handle)) {
        if (out_buffer_length) {
          // Ok to hit EOF; just return what's valid.
          bytes_read += read_size;
        } else {
          status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "end-of-file encountered during read");
        }
      } else {
        status = iree_make_stdio_status("read failed");
      }
      break;
    }
    bytes_read += read_size;
  }

  if (out_buffer_length) {
    *out_buffer_length = bytes_read;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_write(iree_io_stream_t* base_stream,
                                                iree_host_size_t buffer_length,
                                                const void* buffer) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // Write in ~2GB chunks - even platforms with 64-bit support sometimes don't
  // like write lengths >2GB and there's not really any benefit to doing 12GB
  // writes in one go anyway.
  iree_host_size_t bytes_written = 0;
  while (bytes_written < buffer_length) {
    iree_host_size_t chunk_size =
        iree_min(buffer_length - bytes_written, INT_MAX);
    iree_host_size_t write_size =
        fwrite((uint8_t*)buffer + bytes_written, 1, chunk_size, stream->handle);
    if (write_size != chunk_size) {
      // Failed to write chunk; likely exhausted disk space.
      status = iree_make_stdio_status(
          "write failed, possibly out of disk space or device lost");
      break;
    }
    bytes_written += write_size;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_fill(
    iree_io_stream_t* base_stream, iree_io_stream_pos_t count,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(pattern);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // There's not an stdio API for filling contents. When using platform APIs we
  // can extend files when the pattern is zeros to quickly do things but here
  // we just bash fwrite. We could buffer up a reasonable size (4096 etc) of the
  // pattern repeating but this shouldn't be performance critical.
  for (iree_io_stream_pos_t i = 0; i < count; ++i) {
    if (fwrite(pattern, pattern_length, 1, stream->handle) != 1) {
      status = iree_make_stdio_status(
          "write failed, possibly out of disk space or device lost");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_map_read(
    iree_io_stream_t* stream, iree_host_size_t length,
    iree_const_byte_span_t* out_span) {
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "stdio streams do not support mapping");
}

static iree_status_t iree_io_stdio_stream_map_write(
    iree_io_stream_t* stream, iree_host_size_t length,
    iree_byte_span_t* out_span) {
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "stdio streams do not support mapping");
}

static const iree_io_stream_vtable_t iree_io_stdio_stream_vtable = {
    .destroy = iree_io_stdio_stream_destroy,
    .offset = iree_io_stdio_stream_offset,
    .length = iree_io_stdio_stream_length,
    .seek = iree_io_stdio_stream_seek,
    .read = iree_io_stdio_stream_read,
    .write = iree_io_stdio_stream_write,
    .fill = iree_io_stdio_stream_fill,
    .map_read = iree_io_stdio_stream_map_read,
    .map_write = iree_io_stdio_stream_map_write,
};
