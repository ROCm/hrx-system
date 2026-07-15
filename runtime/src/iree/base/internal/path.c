// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/path.h"

#include <stddef.h>
#include <string.h>

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#endif  // IREE_PLATFORM_WINDOWS

static iree_status_t iree_string_view_dup(iree_string_view_t value,
                                          iree_allocator_t allocator,
                                          char** out_buffer) {
  iree_host_size_t alloc_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &alloc_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "string length overflow");
  }
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, alloc_size, (void**)&buffer));
  memcpy(buffer, value.data, value.size);
  buffer[value.size] = 0;  // NUL
  *out_buffer = buffer;
  return iree_ok_status();
}

static iree_status_t iree_string_view_cat(iree_string_view_t lhs,
                                          iree_string_view_t rhs,
                                          iree_allocator_t allocator,
                                          char** out_buffer) {
  // Allocate storage buffer with NUL character.
  iree_host_size_t total_length = 0;
  iree_host_size_t alloc_size = 0;
  if (!iree_host_size_checked_add(lhs.size, rhs.size, &total_length) ||
      !iree_host_size_checked_add(total_length, 1, &alloc_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "string length overflow");
  }
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, alloc_size, (void**)&buffer));

  // Copy both parts.
  memcpy(buffer, lhs.data, lhs.size);
  memcpy(buffer + lhs.size, rhs.data, rhs.size);

  buffer[total_length] = 0;  // NUL
  *out_buffer = buffer;
  return iree_ok_status();
}

static iree_status_t iree_string_view_join(iree_host_size_t part_count,
                                           const iree_string_view_t* parts,
                                           iree_string_view_t separator,
                                           iree_allocator_t allocator,
                                           char** out_buffer) {
  // Compute total output size in characters.
  iree_host_size_t total_length = 0;
  for (iree_host_size_t i = 0; i < part_count; ++i) {
    if (!iree_host_size_checked_add(total_length, parts[i].size,
                                    &total_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "join length overflow");
    }
  }
  if (part_count > 0) {
    iree_host_size_t separator_total = 0;
    if (!iree_host_size_checked_mul(separator.size, part_count - 1,
                                    &separator_total) ||
        !iree_host_size_checked_add(total_length, separator_total,
                                    &total_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "join length overflow");
    }
  }

  // Allocate storage buffer with NUL character.
  iree_host_size_t alloc_size = 0;
  if (!iree_host_size_checked_add(total_length, 1, &alloc_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation size overflow");
  }
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, alloc_size, (void**)&buffer));

  // Append each part and a separator between each.
  char* p = buffer;
  for (iree_host_size_t i = 0; i < part_count; ++i) {
    memcpy(p, parts[i].data, parts[i].size);
    p += parts[i].size;
    if (i != part_count - 1) {
      memcpy(p, separator.data, separator.size);
      p += separator.size;
    }
  }

  buffer[total_length] = 0;  // NUL
  *out_buffer = buffer;
  return iree_ok_status();
}

#if defined(IREE_PLATFORM_WINDOWS)
iree_host_size_t iree_file_path_canonicalize(char* path,
                                             iree_host_size_t path_length) {
  char* p = path;
  iree_host_size_t new_length = path_length;

  // Replace `/` with `\`.
  for (iree_host_size_t i = 0; i < new_length; ++i) {
    if (p[i] == '/') p[i] = '\\';
  }

  // Replace `\\` with `\`.
  if (new_length > 1) {
    for (iree_host_size_t i = 0; i < new_length - 1; ++i) {
      if (p[i] == '\\' && p[i + 1] == '\\') {
        memmove(&p[i + 1], &p[i + 2], new_length - i - 2);
        --new_length;
        --i;
      }
    }
  }

  path[new_length] = 0;  // NUL
  return new_length;
}
#else
iree_host_size_t iree_file_path_canonicalize(char* path,
                                             iree_host_size_t path_length) {
  char* p = path;
  iree_host_size_t new_length = path_length;

  // Replace `//` with `/`.
  if (new_length > 1) {
    for (iree_host_size_t i = 0; i < new_length - 1; ++i) {
      if (p[i] == '/' && p[i + 1] == '/') {
        memmove(&p[i + 1], &p[i + 2], new_length - i - 2);
        --new_length;
        --i;
      }
    }
  }

  path[new_length] = 0;  // NUL
  return new_length;
}
#endif  // IREE_PLATFORM_WINDOWS

iree_status_t iree_file_path_join(iree_string_view_t lhs,
                                  iree_string_view_t rhs,
                                  iree_allocator_t allocator, char** out_path) {
  if (iree_string_view_is_empty(lhs)) {
    return iree_string_view_dup(rhs, allocator, out_path);
  }
  if (iree_string_view_is_empty(rhs)) {
    return iree_string_view_dup(lhs, allocator, out_path);
  }
  if (lhs.data[lhs.size - 1] == '/') {
    if (rhs.data[0] == '/') {
      return iree_string_view_cat(
          lhs, iree_string_view_substr(rhs, 1, IREE_STRING_VIEW_NPOS),
          allocator, out_path);
    }
  } else {
    if (rhs.data[0] != '/') {
      iree_string_view_t parts[2] = {lhs, rhs};
      return iree_string_view_join(IREE_ARRAYSIZE(parts), parts,
                                   iree_make_cstring_view("/"), allocator,
                                   out_path);
    }
  }
  return iree_string_view_cat(lhs, rhs, allocator, out_path);
}

void iree_file_path_split(iree_string_view_t path,
                          iree_string_view_t* out_dirname,
                          iree_string_view_t* out_basename) {
  iree_host_size_t pos = iree_string_view_find_last_of(
      path, iree_make_cstring_view("/"), IREE_STRING_VIEW_NPOS);
  if (pos == IREE_STRING_VIEW_NPOS) {
    // No '/' in path.
    *out_dirname = iree_string_view_empty();
    *out_basename = path;
  } else if (pos == 0) {
    // Single leading '/' in path.
    *out_dirname = iree_string_view_substr(path, 0, 1);
    *out_basename = iree_string_view_substr(path, 1, IREE_STRING_VIEW_NPOS);
  } else {
    *out_dirname = iree_string_view_substr(path, 0, pos);
    *out_basename =
        iree_string_view_substr(path, pos + 1, IREE_STRING_VIEW_NPOS);
  }
}

iree_string_view_t iree_file_path_dirname(iree_string_view_t path) {
  iree_string_view_t dirname = iree_string_view_empty();
  iree_string_view_t basename = iree_string_view_empty();
  iree_file_path_split(path, &dirname, &basename);
  return dirname;
}

iree_string_view_t iree_file_path_basename(iree_string_view_t path) {
  iree_string_view_t dirname = iree_string_view_empty();
  iree_string_view_t basename = iree_string_view_empty();
  iree_file_path_split(path, &dirname, &basename);
  return basename;
}

void iree_file_path_split_basename(iree_string_view_t path,
                                   iree_string_view_t* out_stem,
                                   iree_string_view_t* out_extension) {
  path = iree_file_path_basename(path);
  iree_host_size_t pos = iree_string_view_find_last_of(
      path, iree_make_cstring_view("."), IREE_STRING_VIEW_NPOS);
  if (pos == IREE_STRING_VIEW_NPOS) {
    *out_stem = path;
    *out_extension = iree_string_view_empty();
  } else {
    *out_stem = iree_string_view_substr(path, 0, pos);
    *out_extension =
        iree_string_view_substr(path, pos + 1, IREE_STRING_VIEW_NPOS);
  }
}

iree_string_view_t iree_file_path_stem(iree_string_view_t path) {
  iree_string_view_t stem = iree_string_view_empty();
  iree_string_view_t extension = iree_string_view_empty();
  iree_file_path_split_basename(path, &stem, &extension);
  return stem;
}

iree_string_view_t iree_file_path_extension(iree_string_view_t path) {
  iree_string_view_t stem = iree_string_view_empty();
  iree_string_view_t extension = iree_string_view_empty();
  iree_file_path_split_basename(path, &stem, &extension);
  return extension;
}

static bool iree_file_path_has_versioned_so_suffix(iree_string_view_t path) {
  iree_string_view_t basename = iree_file_path_basename(path);
  if (basename.size < 6) return false;  // "x.so.1"

  for (iree_host_size_t i = 1; i + 4 <= basename.size; ++i) {
    if (memcmp(basename.data + i, ".so.", 4) != 0) continue;

    bool needs_digit = true;
    for (iree_host_size_t j = i + 4; j < basename.size; ++j) {
      char c = basename.data[j];
      if (c >= '0' && c <= '9') {
        needs_digit = false;
      } else if (c == '.' && !needs_digit) {
        needs_digit = true;
      } else {
        return false;
      }
    }
    return !needs_digit;
  }

  return false;
}

// We could limit this to only those libraries supported on the current
// platform. For now most of the libraries we produce follow the CMake defaults
// below, while system libraries may use ELF soname suffixes such as `.so.1`.
bool iree_file_path_is_dynamic_library(iree_string_view_t path) {
  iree_string_view_t ext = iree_file_path_extension(path);
  return iree_string_view_equal(ext, IREE_SV("dll")) ||
         iree_string_view_equal(ext, IREE_SV("dylib")) ||
         iree_string_view_equal(ext, IREE_SV("so")) ||
         iree_string_view_equal(ext, IREE_SV("sos")) ||
         iree_file_path_has_versioned_so_suffix(path);
}

#if defined(IREE_PLATFORM_WINDOWS)

static bool iree_file_path_has_win32_namespace_prefix(
    const wchar_t* path, iree_host_size_t path_length) {
  return path_length >= 4 && path[0] == L'\\' && path[1] == L'\\' &&
         (path[2] == L'?' || path[2] == L'.') && path[3] == L'\\';
}

iree_status_t iree_file_path_to_win32(iree_string_view_t path,
                                      iree_allocator_t allocator,
                                      wchar_t** out_path) {
  IREE_ASSERT_ARGUMENT(out_path);
  *out_path = NULL;

  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file path must not be empty");
  }
  if (path.size >= IREE_MAX_PATH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "path length %" PRIhsz
                            " exceeds maximum character length of %d",
                            path.size, IREE_MAX_PATH);
  }
  if (memchr(path.data, 0, path.size) != NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file path contains an embedded NUL character");
  }

  // A UTF-8 byte sequence never expands to more UTF-16 code units than its
  // byte length. IREE_MAX_PATH bounds this temporary stack allocation.
  wchar_t* utf16_path =
      iree_alloca((path.size + /*NUL=*/1) * sizeof(*utf16_path));
  int utf16_length =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data,
                          (int)path.size, utf16_path, (int)path.size);
  if (utf16_length == 0) {
    DWORD error = GetLastError();
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file path is not valid UTF-8 (Win32 error %lu)",
                            (unsigned long)error);
  }
  utf16_path[utf16_length] = L'\0';

  // Win32 namespace paths already carry their intended interpretation. In
  // particular, \\.\ names devices and must not be made filesystem-absolute.
  if (iree_file_path_has_win32_namespace_prefix(utf16_path, utf16_length)) {
    iree_host_size_t allocation_size = 0;
    if (!iree_host_size_checked_mul((iree_host_size_t)utf16_length + 1,
                                    sizeof(*utf16_path), &allocation_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Win32 path allocation size overflow");
    }
    wchar_t* native_path = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, allocation_size,
                                               (void**)&native_path));
    memcpy(native_path, utf16_path, allocation_size);
    *out_path = native_path;
    return iree_ok_status();
  }

  // Extended-length paths must be absolute. Ask Windows to resolve drive-
  // relative, root-relative, and ordinary relative paths using its native
  // rules before adding the namespace prefix.
  DWORD absolute_capacity = GetFullPathNameW(utf16_path, 0, NULL, NULL);
  if (absolute_capacity == 0) {
    DWORD error = GetLastError();
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "failed to resolve Win32 file path (error %lu)",
                            (unsigned long)error);
  }

  // Six spare code units accommodate the longer UNC prefix transformation:
  // `\\server\share` becomes `\\?\UNC\server\share`.
  iree_host_size_t native_capacity = 0;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add((iree_host_size_t)absolute_capacity, 6,
                                  &native_capacity) ||
      !iree_host_size_checked_mul(native_capacity, sizeof(wchar_t),
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Win32 path allocation size overflow");
  }

  wchar_t* native_path = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, allocation_size, (void**)&native_path));
  wchar_t* absolute_path = native_path + 6;
  DWORD absolute_length =
      GetFullPathNameW(utf16_path, absolute_capacity, absolute_path, NULL);

  iree_status_t status = iree_ok_status();
  if (absolute_length == 0) {
    DWORD error = GetLastError();
    status = iree_make_status(iree_status_code_from_win32_error(error),
                              "failed to resolve Win32 file path (error %lu)",
                              (unsigned long)error);
  } else if (absolute_length >= absolute_capacity) {
    status = iree_make_status(
        IREE_STATUS_ABORTED,
        "current directory changed while resolving a relative file path");
  }

  if (iree_status_is_ok(status)) {
    if (absolute_length >= 2 && absolute_path[0] == L'\\' &&
        absolute_path[1] == L'\\') {
      // The UNC payload already begins at offset 8 because |absolute_path|
      // begins at offset 6 and its two leading separators are discarded.
      static const wchar_t kUncPrefix[] = L"\\\\?\\UNC\\";
      memcpy(native_path, kUncPrefix,
             (IREE_ARRAYSIZE(kUncPrefix) - 1) * sizeof(wchar_t));
    } else {
      static const wchar_t kExtendedPrefix[] = L"\\\\?\\";
      memmove(native_path + IREE_ARRAYSIZE(kExtendedPrefix) - 1, absolute_path,
              ((iree_host_size_t)absolute_length + 1) * sizeof(wchar_t));
      memcpy(native_path, kExtendedPrefix,
             (IREE_ARRAYSIZE(kExtendedPrefix) - 1) * sizeof(wchar_t));
    }
    *out_path = native_path;
  } else {
    iree_allocator_free(allocator, native_path);
  }
  return status;
}

#endif  // IREE_PLATFORM_WINDOWS

void iree_uri_split(iree_string_view_t uri, iree_string_view_t* out_schema,
                    iree_string_view_t* out_path,
                    iree_string_view_t* out_params) {
  *out_schema = iree_string_view_empty();
  *out_path = iree_string_view_empty();
  *out_params = iree_string_view_empty();
  if (iree_string_view_is_empty(uri)) return;

  // Split on `schema` `:` (anything).
  iree_string_view_t rhs = iree_string_view_empty();
  iree_string_view_split(uri, ':', out_schema, &rhs);

  // Strip leading // from the remaining string. The // isn't required but does
  // make things more URI-like.
  if (!iree_string_view_consume_prefix(&rhs, IREE_SV("//"))) {
    rhs = iree_string_view_strip_prefix(rhs, IREE_SV("/"));
  }

  // Split on `path` `?` `params.
  iree_string_view_split(rhs, '?', out_path, out_params);
}

iree_string_view_t iree_uri_schema(iree_string_view_t uri) {
  iree_string_view_t schema, path, params;
  iree_uri_split(uri, &schema, &path, &params);
  return schema;
}

iree_string_view_t iree_uri_path(iree_string_view_t uri) {
  iree_string_view_t schema, path, params;
  iree_uri_split(uri, &schema, &path, &params);
  return path;
}

iree_string_view_t iree_uri_params(iree_string_view_t uri) {
  iree_string_view_t schema, path, params;
  iree_uri_split(uri, &schema, &path, &params);
  return params;
}

bool iree_uri_split_params(iree_string_view_t params, iree_host_size_t capacity,
                           iree_host_size_t* out_count,
                           iree_string_pair_t* out_params) {
  // Cleanup string to remove leading/trailing junk.
  params = iree_string_view_strip_prefix(params, IREE_SV("&"));
  params = iree_string_view_strip_suffix(params, IREE_SV("&"));
  params = iree_string_view_trim(params);

  // Scan once to count; URI parsing should not be on a critical path.
  iree_host_size_t required_capacity =
      iree_string_view_is_empty(params) ? 0 : 1;
  for (iree_host_size_t i = 0; i < params.size; ++i) {
    if (params.data[i] == '&') ++required_capacity;
  }
  *out_count = required_capacity;
  if (capacity < required_capacity) return false;
  if (!out_params) return true;

  // Parse each param into a key=value pair.
  iree_string_view_t remaining = params;
  iree_host_size_t count = 0;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t key_value;
    iree_string_view_split(remaining, '&', &key_value, &remaining);
    iree_string_pair_t* pair = &out_params[count++];
    iree_string_view_split(key_value, '=', &pair->key, &pair->value);
  }

  return true;
}
