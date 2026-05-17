// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/file_index.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/path.h"

#if IREE_FILE_IO_ENABLE
#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <errno.h>
#include <sys/stat.h>
#endif  // IREE_PLATFORM_WINDOWS
#endif  // IREE_FILE_IO_ENABLE

typedef uint8_t iree_hal_remote_file_index_entry_kind_t;

enum iree_hal_remote_file_index_entry_kind_e {
  IREE_HAL_REMOTE_FILE_INDEX_ENTRY_FILE = 0,
  IREE_HAL_REMOTE_FILE_INDEX_ENTRY_DIRECTORY = 1,
};

typedef struct iree_hal_remote_file_index_entry_t {
  // Client-visible logical exact name or directory prefix.
  iree_string_view_t logical_name;
  // Server-local normalized host file path or directory root.
  iree_string_view_t host_path;
  // Entry kind controlling exact or prefix matching.
  iree_hal_remote_file_index_entry_kind_t kind;
  // Read/write access bits granted by this entry.
  iree_hal_memory_access_t allowed_access;
} iree_hal_remote_file_index_entry_t;

struct iree_hal_remote_file_index_t {
  // Reference count for shared server/tool ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for entries and copied strings.
  iree_allocator_t host_allocator;
  // Number of initialized entries.
  iree_host_size_t entry_count;
  // Capacity of |entries|.
  iree_host_size_t entry_capacity;
  // Dynamically-sized allow-list entries.
  iree_hal_remote_file_index_entry_t* entries;
};

static bool iree_hal_remote_file_index_access_is_valid(
    iree_hal_memory_access_t access) {
  return access != 0 && (access & ~(IREE_HAL_MEMORY_ACCESS_READ |
                                    IREE_HAL_MEMORY_ACCESS_WRITE)) == 0;
}

static iree_status_t iree_hal_remote_file_index_dup_string(
    iree_string_view_t value, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  out_value->data = NULL;
  out_value->size = 0;

  iree_host_size_t allocation_size = 0;
  iree_status_t status =
      iree_host_size_checked_add(value.size, 1, &allocation_size)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "file index string length overflow");

  char* storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, allocation_size,
                                   (void**)&storage);
  }
  if (iree_status_is_ok(status)) {
    memcpy(storage, value.data, value.size);
    storage[value.size] = 0;
    *out_value = iree_make_string_view(storage, value.size);
  }
  return status;
}

static iree_status_t iree_hal_remote_file_index_dup_path(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_string_view_t* out_path) {
  iree_status_t status =
      iree_hal_remote_file_index_dup_string(path, host_allocator, out_path);
  if (iree_status_is_ok(status)) {
    iree_host_size_t new_size =
        iree_file_path_canonicalize((char*)out_path->data, out_path->size);
    *out_path = iree_make_string_view(out_path->data, new_size);
  }
  return status;
}

static iree_status_t iree_hal_remote_file_index_dup_logical_name(
    iree_string_view_t logical_name,
    iree_hal_remote_file_index_entry_kind_t kind,
    iree_allocator_t host_allocator, iree_string_view_t* out_logical_name) {
  out_logical_name->data = NULL;
  out_logical_name->size = 0;

  bool append_slash = kind == IREE_HAL_REMOTE_FILE_INDEX_ENTRY_DIRECTORY &&
                      (iree_string_view_is_empty(logical_name) ||
                       logical_name.data[logical_name.size - 1] != '/');
  iree_host_size_t allocation_size = 0;
  iree_status_t status =
      iree_host_size_checked_add(logical_name.size, append_slash ? 2 : 1,
                                 &allocation_size)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "logical file name length overflow");

  char* storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, allocation_size,
                                   (void**)&storage);
  }
  if (iree_status_is_ok(status)) {
    memcpy(storage, logical_name.data, logical_name.size);
    iree_host_size_t logical_name_size = logical_name.size;
    if (append_slash) storage[logical_name_size++] = '/';
    storage[logical_name_size] = 0;
    *out_logical_name = iree_make_string_view(storage, logical_name_size);
  }
  return status;
}

static void iree_hal_remote_file_index_entry_deinitialize(
    iree_hal_remote_file_index_entry_t* entry,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, (void*)entry->logical_name.data);
  iree_allocator_free(host_allocator, (void*)entry->host_path.data);
  memset(entry, 0, sizeof(*entry));
}

static void iree_hal_remote_file_index_destroy(
    iree_hal_remote_file_index_t* file_index) {
  iree_allocator_t host_allocator = file_index->host_allocator;
  for (iree_host_size_t i = 0; i < file_index->entry_count; ++i) {
    iree_hal_remote_file_index_entry_deinitialize(&file_index->entries[i],
                                                  host_allocator);
  }
  iree_allocator_free(host_allocator, file_index->entries);
  iree_allocator_free(host_allocator, file_index);
}

IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_create(
    iree_allocator_t host_allocator,
    iree_hal_remote_file_index_t** out_file_index) {
  IREE_ASSERT_ARGUMENT(out_file_index);
  *out_file_index = NULL;

  iree_hal_remote_file_index_t* file_index = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*file_index), (void**)&file_index);
  if (iree_status_is_ok(status)) {
    memset(file_index, 0, sizeof(*file_index));
    iree_atomic_ref_count_init(&file_index->ref_count);
    file_index->host_allocator = host_allocator;
    *out_file_index = file_index;
  }
  return status;
}

IREE_API_EXPORT void iree_hal_remote_file_index_retain(
    iree_hal_remote_file_index_t* file_index) {
  if (IREE_LIKELY(file_index)) {
    iree_atomic_ref_count_inc(&file_index->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_remote_file_index_release(
    iree_hal_remote_file_index_t* file_index) {
  if (IREE_LIKELY(file_index) &&
      iree_atomic_ref_count_dec(&file_index->ref_count) == 1) {
    iree_hal_remote_file_index_destroy(file_index);
  }
}

static iree_status_t iree_hal_remote_file_index_reserve(
    iree_hal_remote_file_index_t* file_index,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= file_index->entry_capacity) return iree_ok_status();
  return iree_allocator_grow_array(file_index->host_allocator, minimum_capacity,
                                   sizeof(*file_index->entries),
                                   &file_index->entry_capacity,
                                   (void**)&file_index->entries);
}

static iree_status_t iree_hal_remote_file_index_check_duplicate(
    const iree_hal_remote_file_index_t* file_index,
    iree_string_view_t logical_name,
    iree_hal_remote_file_index_entry_kind_t kind) {
  for (iree_host_size_t i = 0; i < file_index->entry_count; ++i) {
    const iree_hal_remote_file_index_entry_t* entry = &file_index->entries[i];
    if (entry->kind == kind &&
        iree_string_view_equal(entry->logical_name, logical_name)) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "remote file entry '%.*s' already exists",
                              (int)logical_name.size, logical_name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_file_index_allow_entry(
    iree_hal_remote_file_index_t* file_index,
    iree_hal_remote_file_index_entry_kind_t kind,
    iree_string_view_t logical_name, iree_string_view_t host_path,
    iree_hal_memory_access_t allowed_access) {
  IREE_ASSERT_ARGUMENT(file_index);

  iree_status_t status = iree_ok_status();
  iree_hal_remote_file_index_entry_t new_entry;
  memset(&new_entry, 0, sizeof(new_entry));

  if (iree_string_view_is_empty(logical_name)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote file logical name must not be empty");
  } else if (iree_string_view_is_empty(host_path)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote file host path must not be empty");
  } else if (!iree_hal_remote_file_index_access_is_valid(allowed_access)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote file access must contain read and/or write bits only");
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_file_index_dup_logical_name(
        logical_name, kind, file_index->host_allocator,
        &new_entry.logical_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_file_index_dup_path(
        host_path, file_index->host_allocator, &new_entry.host_path);
  }
  if (iree_status_is_ok(status)) {
    new_entry.kind = kind;
    new_entry.allowed_access = allowed_access;
    status = iree_hal_remote_file_index_check_duplicate(
        file_index, new_entry.logical_name, kind);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_file_index_reserve(file_index,
                                                file_index->entry_count + 1);
  }
  if (iree_status_is_ok(status)) {
    file_index->entries[file_index->entry_count++] = new_entry;
  } else {
    iree_hal_remote_file_index_entry_deinitialize(&new_entry,
                                                  file_index->host_allocator);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_allow_file(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_name,
    iree_string_view_t host_path, iree_hal_memory_access_t allowed_access) {
  return iree_hal_remote_file_index_allow_entry(
      file_index, IREE_HAL_REMOTE_FILE_INDEX_ENTRY_FILE, logical_name,
      host_path, allowed_access);
}

IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_allow_directory(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_prefix,
    iree_string_view_t host_path, iree_hal_memory_access_t allowed_access) {
  return iree_hal_remote_file_index_allow_entry(
      file_index, IREE_HAL_REMOTE_FILE_INDEX_ENTRY_DIRECTORY, logical_prefix,
      host_path, allowed_access);
}

#if IREE_FILE_IO_ENABLE

static iree_status_t iree_hal_remote_file_index_path_to_cstring(
    iree_string_view_t path, char* storage, iree_host_size_t storage_capacity) {
  if (path.size >= storage_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "path length %" PRIhsz
                            " exceeds maximum character length of %" PRIhsz,
                            path.size, storage_capacity - 1);
  }
  iree_string_view_to_cstring(path, storage, storage_capacity);
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_file_index_stat_path(
    iree_string_view_t host_path,
    iree_hal_remote_file_index_entry_kind_t* out_kind) {
  *out_kind = IREE_HAL_REMOTE_FILE_INDEX_ENTRY_FILE;

  char host_path_storage[IREE_MAX_PATH + 1];
  IREE_RETURN_IF_ERROR(iree_hal_remote_file_index_path_to_cstring(
      host_path, host_path_storage, IREE_ARRAYSIZE(host_path_storage)));

#if defined(IREE_PLATFORM_WINDOWS)
  DWORD attributes = GetFileAttributesA(host_path_storage);
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to stat remote file path '%.*s'",
                            (int)host_path.size, host_path.data);
  }
  *out_kind = (attributes & FILE_ATTRIBUTE_DIRECTORY)
                  ? IREE_HAL_REMOTE_FILE_INDEX_ENTRY_DIRECTORY
                  : IREE_HAL_REMOTE_FILE_INDEX_ENTRY_FILE;
  return iree_ok_status();
#else
  struct stat file_stat;
  if (stat(host_path_storage, &file_stat) != 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to stat remote file path '%.*s'",
                            (int)host_path.size, host_path.data);
  }
  if (S_ISDIR(file_stat.st_mode)) {
    *out_kind = IREE_HAL_REMOTE_FILE_INDEX_ENTRY_DIRECTORY;
    return iree_ok_status();
  }
  if (S_ISREG(file_stat.st_mode)) {
    *out_kind = IREE_HAL_REMOTE_FILE_INDEX_ENTRY_FILE;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "remote file path '%.*s' is not a regular file or "
                          "directory",
                          (int)host_path.size, host_path.data);
#endif  // IREE_PLATFORM_WINDOWS
}

#endif  // IREE_FILE_IO_ENABLE

IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_allow_path(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_name,
    iree_string_view_t host_path, iree_hal_memory_access_t allowed_access) {
#if IREE_FILE_IO_ENABLE
  iree_hal_remote_file_index_entry_kind_t kind =
      IREE_HAL_REMOTE_FILE_INDEX_ENTRY_FILE;
  iree_status_t status = iree_hal_remote_file_index_stat_path(host_path, &kind);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_file_index_allow_entry(
        file_index, kind, logical_name, host_path, allowed_access);
  }
  return status;
#else
  (void)file_index;
  (void)logical_name;
  (void)host_path;
  (void)allowed_access;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
#endif  // IREE_FILE_IO_ENABLE
}

static const iree_hal_remote_file_index_entry_t*
iree_hal_remote_file_index_find_entry(
    const iree_hal_remote_file_index_t* file_index,
    iree_string_view_t logical_name, iree_string_view_t* out_suffix) {
  *out_suffix = iree_string_view_empty();
  const iree_hal_remote_file_index_entry_t* best_entry = NULL;
  for (iree_host_size_t i = 0; i < file_index->entry_count; ++i) {
    const iree_hal_remote_file_index_entry_t* entry = &file_index->entries[i];
    if (entry->kind == IREE_HAL_REMOTE_FILE_INDEX_ENTRY_FILE) {
      if (iree_string_view_equal(entry->logical_name, logical_name)) {
        *out_suffix = iree_string_view_empty();
        return entry;
      }
      continue;
    }
    if (logical_name.size < entry->logical_name.size ||
        memcmp(logical_name.data, entry->logical_name.data,
               entry->logical_name.size) != 0) {
      continue;
    }
    if (!best_entry ||
        entry->logical_name.size > best_entry->logical_name.size) {
      best_entry = entry;
      *out_suffix = iree_string_view_substr(
          logical_name, entry->logical_name.size, IREE_STRING_VIEW_NPOS);
    }
  }
  return best_entry;
}

static iree_io_file_mode_t iree_hal_remote_file_index_io_mode(
    iree_hal_memory_access_t access) {
  iree_io_file_mode_t mode =
      IREE_IO_FILE_MODE_ASYNC | IREE_IO_FILE_MODE_SHARE_READ;
  if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
    mode |= IREE_IO_FILE_MODE_READ;
  }
  if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    mode |= IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SHARE_WRITE;
  }
  return mode;
}

static iree_status_t iree_hal_remote_file_index_resolve_host_path(
    const iree_hal_remote_file_index_entry_t* entry, iree_string_view_t suffix,
    iree_allocator_t host_allocator, char** out_host_path) {
  *out_host_path = NULL;
  iree_status_t status = iree_ok_status();
  if (entry->kind == IREE_HAL_REMOTE_FILE_INDEX_ENTRY_DIRECTORY) {
    if (!iree_file_path_is_portable_relative(suffix)) {
      status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "remote file path suffix '%.*s' escapes the "
                                "allowed logical directory",
                                (int)suffix.size, suffix.data);
    } else {
      status = iree_file_path_join(entry->host_path, suffix, host_allocator,
                                   out_host_path);
    }
  } else {
    iree_string_view_t host_path_view = iree_string_view_empty();
    status = iree_hal_remote_file_index_dup_path(
        entry->host_path, host_allocator, &host_path_view);
    if (iree_status_is_ok(status)) {
      *out_host_path = (char*)host_path_view.data;
    }
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_open(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_name,
    iree_hal_memory_access_t requested_access, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_handle,
    iree_hal_memory_access_t* out_granted_access) {
  IREE_ASSERT_ARGUMENT(out_handle);
  IREE_ASSERT_ARGUMENT(out_granted_access);
  *out_handle = NULL;
  *out_granted_access = 0;

  iree_status_t status = iree_ok_status();
  if (!file_index) {
    status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "remote server has no file index configured");
  } else if (!iree_hal_remote_file_index_access_is_valid(requested_access)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote file open access must contain read "
                              "and/or write bits only");
  }

  iree_string_view_t suffix = iree_string_view_empty();
  const iree_hal_remote_file_index_entry_t* entry = NULL;
  if (iree_status_is_ok(status)) {
    entry = iree_hal_remote_file_index_find_entry(file_index, logical_name,
                                                  &suffix);
    if (!entry) {
      status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "remote file '%.*s' is not allowed",
                                (int)logical_name.size, logical_name.data);
    }
  }
  if (iree_status_is_ok(status) &&
      (requested_access & ~entry->allowed_access) != 0) {
    status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "remote file '%.*s' requested access 0x%x "
                              "exceeds allowed access 0x%x",
                              (int)logical_name.size, logical_name.data,
                              requested_access, entry->allowed_access);
  }

  char* host_path = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_file_index_resolve_host_path(
        entry, suffix, host_allocator, &host_path);
  }

  iree_io_file_handle_t* handle = NULL;
  if (iree_status_is_ok(status)) {
    iree_string_view_t host_path_view = iree_make_cstring_view(host_path);
    status = iree_io_file_handle_open(
        iree_hal_remote_file_index_io_mode(requested_access), host_path_view,
        host_allocator, &handle);
  }

  if (iree_status_is_ok(status)) {
    *out_handle = handle;
    *out_granted_access = requested_access;
  }
  iree_allocator_free(host_allocator, host_path);
  return status;
}
