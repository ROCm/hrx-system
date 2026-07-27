// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/io/source_path.h"

#include <string.h>

typedef struct loom_tooling_source_prefix_map_t {
  iree_string_view_t old_prefix;
  iree_string_view_t new_prefix;
} loom_tooling_source_prefix_map_t;

void loom_tooling_source_path_options_initialize(
    loom_tooling_source_path_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_tooling_source_path_options_t){
      .prefix_maps = iree_string_view_list_empty(),
  };
}

static iree_status_t loom_tooling_source_prefix_map_parse(
    iree_string_view_t value, loom_tooling_source_prefix_map_t* out_map) {
  iree_host_size_t separator_index = iree_string_view_find_char(value, '=', 0);
  if (separator_index == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid source prefix map '%.*s'; expected <old>=<new>",
        (int)value.size, value.data);
  }
  if (out_map != NULL) {
    *out_map = (loom_tooling_source_prefix_map_t){
        .old_prefix = iree_string_view_substr(value, 0, separator_index),
        .new_prefix = iree_string_view_substr(value, separator_index + 1,
                                              IREE_STRING_VIEW_NPOS),
    };
  }
  return iree_ok_status();
}

#if defined(IREE_PLATFORM_WINDOWS)
static bool loom_tooling_source_path_is_separator(char c) {
  return c == '/' || c == '\\';
}
#endif  // defined(IREE_PLATFORM_WINDOWS)

static bool loom_tooling_source_path_char_equal(char lhs, char rhs) {
#if defined(IREE_PLATFORM_WINDOWS)
  if (loom_tooling_source_path_is_separator(lhs) &&
      loom_tooling_source_path_is_separator(rhs)) {
    return true;
  }
  if (lhs >= 'A' && lhs <= 'Z') lhs = (char)(lhs - 'A' + 'a');
  if (rhs >= 'A' && rhs <= 'Z') rhs = (char)(rhs - 'A' + 'a');
#endif  // defined(IREE_PLATFORM_WINDOWS)
  return lhs == rhs;
}

static iree_status_t loom_tooling_source_prefix_map_validate(
    iree_string_view_t value) {
  return loom_tooling_source_prefix_map_parse(value, NULL);
}

static bool loom_tooling_source_path_starts_with(iree_string_view_t path,
                                                 iree_string_view_t prefix) {
  if (prefix.size > path.size) {
    return false;
  }
  for (iree_host_size_t i = 0; i < prefix.size; ++i) {
    if (!loom_tooling_source_path_char_equal(path.data[i], prefix.data[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_tooling_source_path_join_remap(
    iree_string_view_t new_prefix, iree_string_view_t suffix,
    iree_allocator_t allocator, iree_string_view_t* out_source_path,
    char** out_source_path_storage) {
  *out_source_path = iree_string_view_empty();
  *out_source_path_storage = NULL;

  iree_host_size_t output_size = 0;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(new_prefix.size, suffix.size, &output_size) ||
      !iree_host_size_checked_add(output_size, 1, &allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source prefix map output length overflow");
  }
  if (output_size == 0) {
    return iree_ok_status();
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, allocation_size, (void**)&storage));
  memcpy(storage, new_prefix.data, new_prefix.size);
  memcpy(storage + new_prefix.size, suffix.data, suffix.size);
  storage[output_size] = '\0';

  *out_source_path = iree_make_string_view(storage, output_size);
  *out_source_path_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_tooling_source_path_remap(
    iree_string_view_t source_path,
    const loom_tooling_source_path_options_t* options,
    iree_allocator_t allocator, iree_string_view_t* out_source_path,
    char** out_source_path_storage) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_source_path);
  IREE_ASSERT_ARGUMENT(out_source_path_storage);
  *out_source_path = source_path;
  *out_source_path_storage = NULL;
  if (options->prefix_maps.count == 0) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < options->prefix_maps.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_source_prefix_map_validate(
        options->prefix_maps.values[i]));
  }

  for (iree_host_size_t i = options->prefix_maps.count; i > 0; --i) {
    loom_tooling_source_prefix_map_t map = {0};
    IREE_RETURN_IF_ERROR(loom_tooling_source_prefix_map_parse(
        options->prefix_maps.values[i - 1], &map));
    if (!loom_tooling_source_path_starts_with(source_path, map.old_prefix)) {
      continue;
    }
    const iree_string_view_t suffix = iree_string_view_substr(
        source_path, map.old_prefix.size, IREE_STRING_VIEW_NPOS);
    return loom_tooling_source_path_join_remap(map.new_prefix, suffix,
                                               allocator, out_source_path,
                                               out_source_path_storage);
  }

  return iree_ok_status();
}
