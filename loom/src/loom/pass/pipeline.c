// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/pipeline.h"

#include <stdint.h>
#include <string.h>

iree_status_t loom_pass_pipeline_consume_entry(
    iree_string_view_t* pipeline, loom_pass_pipeline_entry_spec_t* out_entry,
    bool* out_has_entry) {
  memset(out_entry, 0, sizeof(*out_entry));
  *out_has_entry = false;

  iree_string_view_t remaining = iree_string_view_trim(*pipeline);
  if (iree_string_view_is_empty(remaining)) {
    *pipeline = remaining;
    return iree_ok_status();
  }

  iree_host_size_t separator_position = IREE_STRING_VIEW_NPOS;
  uint32_t brace_depth = 0;
  for (iree_host_size_t i = 0; i < remaining.size; ++i) {
    char c = remaining.data[i];
    if (c == '{') {
      if (brace_depth != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "nested pass option dictionaries are not "
                                "supported");
      }
      ++brace_depth;
    } else if (c == '}') {
      if (brace_depth == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unexpected '}' in pass pipeline");
      }
      --brace_depth;
    } else if (c == ',' && brace_depth == 0) {
      separator_position = i;
      break;
    }
  }
  if (brace_depth != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated pass option dictionary");
  }

  iree_string_view_t entry = remaining;
  if (separator_position != IREE_STRING_VIEW_NPOS) {
    entry = iree_string_view_substr(remaining, 0, separator_position);
    iree_string_view_t next = iree_string_view_substr(
        remaining, separator_position + 1, IREE_STRING_VIEW_NPOS);
    if (iree_string_view_is_empty(iree_string_view_trim(next))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty pass pipeline entry");
    }
    *pipeline = next;
  } else {
    *pipeline = iree_string_view_empty();
  }
  entry = iree_string_view_trim(entry);
  if (iree_string_view_is_empty(entry)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty pass pipeline entry");
  }

  iree_host_size_t open_position = iree_string_view_find_char(entry, '{', 0);
  iree_host_size_t close_position = iree_string_view_find_char(entry, '}', 0);
  if (open_position == IREE_STRING_VIEW_NPOS) {
    if (close_position != IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unexpected '}' in pass pipeline entry");
    }
    out_entry->name = entry;
    *out_has_entry = true;
    return iree_ok_status();
  }

  if (close_position != entry.size - 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass option dictionary must terminate the pass "
                            "pipeline entry");
  }

  out_entry->name =
      iree_string_view_trim(iree_string_view_substr(entry, 0, open_position));
  out_entry->options = iree_string_view_trim(iree_string_view_substr(
      entry, open_position + 1, close_position - open_position - 1));
  if (iree_string_view_is_empty(out_entry->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass name required before option dictionary");
  }
  *out_has_entry = true;
  return iree_ok_status();
}

iree_status_t loom_pass_options_parse(iree_string_view_t pass_name,
                                      iree_string_view_t options,
                                      loom_pass_option_parse_callback_t parse) {
  if (iree_string_view_is_empty(iree_string_view_trim(options))) {
    return iree_ok_status();
  }

  iree_string_view_t remaining = options;
  while (!iree_string_view_is_empty(iree_string_view_trim(remaining))) {
    iree_string_view_t assignment = iree_string_view_empty();
    intptr_t separator_position =
        iree_string_view_split(remaining, ',', &assignment, &remaining);
    if (separator_position >= 0 &&
        iree_string_view_is_empty(iree_string_view_trim(remaining))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty option assignment in pass '%.*s'",
                              (int)pass_name.size, pass_name.data);
    }
    assignment = iree_string_view_trim(assignment);
    if (iree_string_view_is_empty(assignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty option assignment in pass '%.*s'",
                              (int)pass_name.size, pass_name.data);
    }

    iree_string_view_t name = iree_string_view_empty();
    iree_string_view_t value = iree_string_view_empty();
    if (iree_string_view_split(assignment, '=', &name, &value) < 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' must use name=value syntax",
          (int)pass_name.size, pass_name.data, (int)assignment.size,
          assignment.data);
    }

    name = iree_string_view_trim(name);
    value = iree_string_view_trim(value);
    if (iree_string_view_is_empty(name) || iree_string_view_is_empty(value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option assignments require non-empty names and values",
          (int)pass_name.size, pass_name.data);
    }
    IREE_RETURN_IF_ERROR(parse.fn(parse.user_data, name, value));
  }
  return iree_ok_status();
}

iree_status_t loom_pass_option_parse_uint32(iree_string_view_t pass_name,
                                            iree_string_view_t option_name,
                                            iree_string_view_t option_value,
                                            uint32_t* out_value) {
  option_value = iree_string_view_trim(option_value);
  if (iree_string_view_is_empty(option_value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' option '%.*s' expected a uint32 value, got '%.*s'",
        (int)pass_name.size, pass_name.data, (int)option_name.size,
        option_name.data, (int)option_value.size, option_value.data);
  }
  uint32_t value = 0;
  for (iree_host_size_t i = 0; i < option_value.size; ++i) {
    char c = option_value.data[i];
    if (c < '0' || c > '9') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' expected a uint32 value, got '%.*s'",
          (int)pass_name.size, pass_name.data, (int)option_name.size,
          option_name.data, (int)option_value.size, option_value.data);
    }
    uint32_t digit = (uint32_t)(c - '0');
    if (value > (UINT32_MAX - digit) / 10u) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' value '%.*s' is outside the uint32 range",
          (int)pass_name.size, pass_name.data, (int)option_name.size,
          option_name.data, (int)option_value.size, option_value.data);
    }
    value = value * 10u + digit;
  }
  *out_value = value;
  return iree_ok_status();
}
