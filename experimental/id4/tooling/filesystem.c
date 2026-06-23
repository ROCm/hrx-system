// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/filesystem.h"

#include <errno.h>
#include <string.h>

#if defined(IREE_PLATFORM_WINDOWS)
#include <direct.h>
#endif  // IREE_PLATFORM_WINDOWS
#include <sys/stat.h>

static bool id4_tooling_path_is_directory(const char* path) {
#if defined(IREE_PLATFORM_WINDOWS)
  struct _stat file_stat;
  return _stat(path, &file_stat) == 0 && (file_stat.st_mode & _S_IFDIR) != 0;
#else
  struct stat file_stat;
  return stat(path, &file_stat) == 0 && S_ISDIR(file_stat.st_mode);
#endif  // IREE_PLATFORM_WINDOWS
}

static iree_status_t id4_tooling_dup_cstring(iree_string_view_t value,
                                             iree_allocator_t host_allocator,
                                             char** out_string) {
  *out_string = NULL;
  if (value.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "path is too large");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, value.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, value.data, value.size);
  storage[value.size] = 0;
  *out_string = storage;
  return iree_ok_status();
}

iree_status_t id4_tooling_ensure_directory(iree_string_view_t directory,
                                           iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "directory path is required");
  }
  char* directory_path = NULL;
  IREE_RETURN_IF_ERROR(
      id4_tooling_dup_cstring(directory, host_allocator, &directory_path));
#if defined(IREE_PLATFORM_WINDOWS)
  const int result = _mkdir(directory_path);
#else
  const int result = mkdir(directory_path, 0777);
#endif  // IREE_PLATFORM_WINDOWS
  const int saved_errno = errno;
  const bool directory_exists =
      saved_errno == EEXIST && id4_tooling_path_is_directory(directory_path);
  iree_allocator_free(host_allocator, directory_path);
  if (result == 0 || directory_exists) return iree_ok_status();
  return iree_make_status(iree_status_code_from_errno(saved_errno),
                          "failed to create directory (%d)", saved_errno);
}

iree_status_t id4_tooling_format_child_path(iree_string_view_t directory,
                                            iree_string_view_t file_name,
                                            iree_allocator_t host_allocator,
                                            iree_string_view_t* out_path) {
  IREE_ASSERT_ARGUMENT(out_path);
  *out_path = iree_string_view_empty();
  if (iree_string_view_is_empty(directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "directory path is required");
  }
  if (iree_string_view_is_empty(file_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "child file name is required");
  }
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status = iree_string_builder_append_string(&builder, directory);
  if (iree_status_is_ok(status) && directory.data[directory.size - 1] != '/') {
    status = iree_string_builder_append_cstring(&builder, "/");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, file_name);
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t path_size = iree_string_builder_size(&builder);
    char* storage = iree_string_builder_take_storage(&builder);
    *out_path = iree_make_string_view(storage, path_size);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

void id4_tooling_free_path(iree_string_view_t* path,
                           iree_allocator_t host_allocator) {
  if (!path) return;
  iree_allocator_free(host_allocator, (void*)path->data);
  *path = iree_string_view_empty();
}
