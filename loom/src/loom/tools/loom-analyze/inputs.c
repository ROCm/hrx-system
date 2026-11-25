// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-analyze/inputs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif  // IREE_PLATFORM_WINDOWS

void loom_analyze_input_list_initialize(iree_allocator_t allocator,
                                        loom_analyze_input_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  out_list->allocator = allocator;
}

void loom_analyze_input_list_deinitialize(loom_analyze_input_list_t* list) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_allocator_free(list->allocator, list->paths[i]);
  }
  iree_allocator_free(list->allocator, list->paths);
  memset(list, 0, sizeof(*list));
}

// Matches the normalized ('/'-separated) form of |path| against |excludes|.
static bool loom_analyze_input_is_excluded(const char* path,
                                           const iree_string_view_t* excludes,
                                           iree_host_size_t exclude_count) {
  if (exclude_count == 0) {
    return false;
  }
  char normalized[4096];
  iree_host_size_t length = strlen(path);
  if (length >= sizeof(normalized)) {
    length = sizeof(normalized) - 1;
  }
  for (iree_host_size_t i = 0; i < length; ++i) {
    normalized[i] = path[i] == '\\' ? '/' : path[i];
  }
  normalized[length] = 0;
  const iree_string_view_t value = iree_make_string_view(normalized, length);
  for (iree_host_size_t i = 0; i < exclude_count; ++i) {
    if (iree_string_view_match_pattern(value, excludes[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_analyze_input_list_append(
    loom_analyze_input_list_t* list, const char* path,
    iree_host_size_t length) {
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        list->allocator, list->count + 1, sizeof(char*), &list->capacity,
        (void**)&list->paths));
  }
  char* copy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(list->allocator, length + 1, (void**)&copy));
  memcpy(copy, path, length);
  copy[length] = 0;
  list->paths[list->count++] = copy;
  return iree_ok_status();
}

static bool loom_analyze_input_has_loom_extension(const char* name) {
  const iree_host_size_t length = strlen(name);
  return length > 5 && strcmp(name + length - 5, ".loom") == 0;
}

// Joins |directory| and |name| into |buffer|. Returns false when the result
// does not fit.
static bool loom_analyze_input_join(const char* directory, const char* name,
                                    char* buffer,
                                    iree_host_size_t buffer_size) {
  const iree_host_size_t directory_length = strlen(directory);
  const bool needs_separator = directory_length > 0 &&
                               directory[directory_length - 1] != '/' &&
                               directory[directory_length - 1] != '\\';
  const int written = snprintf(buffer, buffer_size, "%s%s%s", directory,
                               needs_separator ? "/" : "", name);
  return written >= 0 && (iree_host_size_t)written < buffer_size;
}

typedef enum loom_analyze_path_kind_e {
  LOOM_ANALYZE_PATH_MISSING = 0,
  LOOM_ANALYZE_PATH_FILE = 1,
  LOOM_ANALYZE_PATH_DIRECTORY = 2,
} loom_analyze_path_kind_t;

static loom_analyze_path_kind_t loom_analyze_input_path_kind(const char* path) {
#if defined(IREE_PLATFORM_WINDOWS)
  const DWORD attributes = GetFileAttributesA(path);
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    return LOOM_ANALYZE_PATH_MISSING;
  }
  return (attributes & FILE_ATTRIBUTE_DIRECTORY) ? LOOM_ANALYZE_PATH_DIRECTORY
                                                 : LOOM_ANALYZE_PATH_FILE;
#else
  struct stat info;
  if (stat(path, &info) != 0) {
    return LOOM_ANALYZE_PATH_MISSING;
  }
  return S_ISDIR(info.st_mode) ? LOOM_ANALYZE_PATH_DIRECTORY
                               : LOOM_ANALYZE_PATH_FILE;
#endif  // IREE_PLATFORM_WINDOWS
}

static iree_status_t loom_analyze_input_list_add_directory(
    loom_analyze_input_list_t* list, const char* directory,
    const iree_string_view_t* excludes, iree_host_size_t exclude_count,
    int depth) {
  // Guards against symlink cycles; real source trees are far shallower.
  if (depth > 64) {
    return iree_ok_status();
  }
  char child[4096];
#if defined(IREE_PLATFORM_WINDOWS)
  char pattern[4096];
  if (!loom_analyze_input_join(directory, "*", pattern, sizeof(pattern))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "path too long: %s",
                            directory);
  }
  WIN32_FIND_DATAA entry;
  HANDLE handle = FindFirstFileA(pattern, &entry);
  if (handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unable to list directory '%s'", directory);
  }
  iree_status_t status = iree_ok_status();
  do {
    const char* name = entry.cFileName;
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) {
      continue;
    }
    if (!loom_analyze_input_join(directory, name, child, sizeof(child))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE, "path too long: %s",
                                directory);
      break;
    }
    if (loom_analyze_input_is_excluded(child, excludes, exclude_count)) {
      continue;
    }
    if (entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      status = loom_analyze_input_list_add_directory(list, child, excludes,
                                                     exclude_count, depth + 1);
    } else if (loom_analyze_input_has_loom_extension(name)) {
      status = loom_analyze_input_list_append(list, child, strlen(child));
    }
  } while (iree_status_is_ok(status) && FindNextFileA(handle, &entry));
  FindClose(handle);
  return status;
#else
  DIR* handle = opendir(directory);
  if (!handle) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unable to list directory '%s'", directory);
  }
  iree_status_t status = iree_ok_status();
  struct dirent* entry = NULL;
  while (iree_status_is_ok(status) && (entry = readdir(handle)) != NULL) {
    const char* name = entry->d_name;
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) {
      continue;
    }
    if (!loom_analyze_input_join(directory, name, child, sizeof(child))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE, "path too long: %s",
                                directory);
      break;
    }
    if (loom_analyze_input_is_excluded(child, excludes, exclude_count)) {
      continue;
    }
    const loom_analyze_path_kind_t kind = loom_analyze_input_path_kind(child);
    if (kind == LOOM_ANALYZE_PATH_DIRECTORY) {
      status = loom_analyze_input_list_add_directory(list, child, excludes,
                                                     exclude_count, depth + 1);
    } else if (kind == LOOM_ANALYZE_PATH_FILE &&
               loom_analyze_input_has_loom_extension(name)) {
      status = loom_analyze_input_list_append(list, child, strlen(child));
    }
  }
  closedir(handle);
  return status;
#endif  // IREE_PLATFORM_WINDOWS
}

iree_status_t loom_analyze_input_list_add_path(
    loom_analyze_input_list_t* list, iree_string_view_t path,
    const iree_string_view_t* excludes, iree_host_size_t exclude_count) {
  char buffer[4096];
  if (path.size >= sizeof(buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "path too long: %.*s",
                            (int)path.size, path.data);
  }
  memcpy(buffer, path.data, path.size);
  buffer[path.size] = 0;
  switch (loom_analyze_input_path_kind(buffer)) {
    case LOOM_ANALYZE_PATH_MISSING:
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "input path '%s' does not exist", buffer);
    case LOOM_ANALYZE_PATH_DIRECTORY:
      return loom_analyze_input_list_add_directory(list, buffer, excludes,
                                                   exclude_count, 0);
    case LOOM_ANALYZE_PATH_FILE:
      if (loom_analyze_input_is_excluded(buffer, excludes, exclude_count)) {
        return iree_ok_status();
      }
      return loom_analyze_input_list_append(list, buffer, path.size);
  }
  return iree_ok_status();
}

static int loom_analyze_compare_paths(const void* lhs, const void* rhs) {
  return strcmp(*(char* const*)lhs, *(char* const*)rhs);
}

void loom_analyze_input_list_finalize(loom_analyze_input_list_t* list) {
  if (list->count < 2) {
    return;
  }
  qsort(list->paths, list->count, sizeof(list->paths[0]),
        loom_analyze_compare_paths);
  iree_host_size_t kept = 1;
  for (iree_host_size_t i = 1; i < list->count; ++i) {
    if (strcmp(list->paths[i], list->paths[kept - 1]) == 0) {
      iree_allocator_free(list->allocator, list->paths[i]);
      continue;
    }
    list->paths[kept++] = list->paths[i];
  }
  list->count = kept;
}
