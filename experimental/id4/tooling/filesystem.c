// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/filesystem.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#if defined(IREE_PLATFORM_WINDOWS)
#include <direct.h>
#include <windows.h>
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

static bool id4_tooling_path_is_separator(char c) {
#if defined(IREE_PLATFORM_WINDOWS)
  return c == '/' || c == '\\';
#else
  return c == '/';
#endif  // IREE_PLATFORM_WINDOWS
}

static iree_status_t id4_tooling_create_single_directory(const char* path) {
#if defined(IREE_PLATFORM_WINDOWS)
  const int result = _mkdir(path);
#else
  const int result = mkdir(path, 0777);
#endif  // IREE_PLATFORM_WINDOWS
  const int saved_errno = errno;
  if (result == 0 ||
      (saved_errno == EEXIST && id4_tooling_path_is_directory(path))) {
    return iree_ok_status();
  }
  return iree_make_status(iree_status_code_from_errno(saved_errno),
                          "failed to create directory `%s` (%d)", path,
                          saved_errno);
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

  iree_host_size_t path_length = directory.size;
  while (path_length > 1 &&
         id4_tooling_path_is_separator(directory_path[path_length - 1])) {
    directory_path[--path_length] = 0;
  }

  iree_host_size_t first_component_position = 0;
#if defined(IREE_PLATFORM_WINDOWS)
  if (path_length >= 2 && directory_path[1] == ':') {
    first_component_position =
        path_length >= 3 && id4_tooling_path_is_separator(directory_path[2])
            ? 3
            : 2;
  } else if (path_length >= 2 &&
             id4_tooling_path_is_separator(directory_path[0]) &&
             id4_tooling_path_is_separator(directory_path[1])) {
    iree_host_size_t server_end = 2;
    while (server_end < path_length &&
           !id4_tooling_path_is_separator(directory_path[server_end])) {
      ++server_end;
    }
    if (server_end == path_length) {
      iree_allocator_free(host_allocator, directory_path);
      return iree_ok_status();
    }
    iree_host_size_t share_end = server_end + 1;
    while (share_end < path_length &&
           !id4_tooling_path_is_separator(directory_path[share_end])) {
      ++share_end;
    }
    first_component_position =
        share_end == path_length ? path_length : share_end + 1;
  }
#endif  // IREE_PLATFORM_WINDOWS

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = first_component_position;
       i <= path_length && iree_status_is_ok(status); ++i) {
    if (i != path_length && !id4_tooling_path_is_separator(directory_path[i])) {
      continue;
    }
    if (i == 0 || id4_tooling_path_is_separator(directory_path[i - 1])) {
      continue;
    }
    const char saved_character = directory_path[i];
    directory_path[i] = 0;
    if (directory_path[i - 1] != ':') {
      status = id4_tooling_create_single_directory(directory_path);
    }
    directory_path[i] = saved_character;
  }
  iree_allocator_free(host_allocator, directory_path);
  return status;
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

iree_status_t id4_tooling_replace_file(iree_string_view_t source_path,
                                       iree_string_view_t target_path,
                                       iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(source_path) ||
      iree_string_view_is_empty(target_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source and target file paths are required");
  }
  char* source_cstring = NULL;
  IREE_RETURN_IF_ERROR(
      id4_tooling_dup_cstring(source_path, host_allocator, &source_cstring));
  char* target_cstring = NULL;
  iree_status_t status =
      id4_tooling_dup_cstring(target_path, host_allocator, &target_cstring);
  if (iree_status_is_ok(status)) {
#if defined(IREE_PLATFORM_WINDOWS)
    if (!MoveFileExA(source_cstring, target_cstring,
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "failed to publish file `%s` as `%s` (Windows error %lu)",
          source_cstring, target_cstring, (unsigned long)GetLastError());
    }
#else
    if (rename(source_cstring, target_cstring) != 0) {
      const int saved_errno = errno;
      status = iree_make_status(iree_status_code_from_errno(saved_errno),
                                "failed to publish file `%s` as `%s` (%d)",
                                source_cstring, target_cstring, saved_errno);
    }
#endif  // IREE_PLATFORM_WINDOWS
  }
  iree_allocator_free(host_allocator, target_cstring);
  iree_allocator_free(host_allocator, source_cstring);
  return status;
}
