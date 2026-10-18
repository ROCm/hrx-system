// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/io/module_format.h"

#include <string.h>

#include "loom/format/bytecode/format.h"

const char* loom_module_format_name(loom_module_format_t format) {
  switch (format) {
    case LOOM_MODULE_FORMAT_AUTO:
      return "auto";
    case LOOM_MODULE_FORMAT_TEXT:
      return "text";
    case LOOM_MODULE_FORMAT_BYTECODE:
      return "bc";
  }
  return "unknown";
}

iree_status_t loom_module_format_parse(iree_string_view_t value,
                                       bool allow_auto,
                                       loom_module_format_t* out_format) {
  if (allow_auto &&
      iree_string_view_equal(value, iree_make_cstring_view("auto"))) {
    *out_format = LOOM_MODULE_FORMAT_AUTO;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, iree_make_cstring_view("text"))) {
    *out_format = LOOM_MODULE_FORMAT_TEXT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, iree_make_cstring_view("bc")) ||
      iree_string_view_equal(value, iree_make_cstring_view("bytecode"))) {
    *out_format = LOOM_MODULE_FORMAT_BYTECODE;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unsupported format '%.*s'; expected %stext, bc, or bytecode",
      (int)value.size, value.data, allow_auto ? "auto, " : "");
}

loom_module_format_t loom_module_format_detect_input(
    iree_const_byte_span_t input) {
  if (input.data_length >= LOOM_BYTECODE_MAGIC_LENGTH &&
      memcmp(input.data, LOOM_BYTECODE_MAGIC, LOOM_BYTECODE_MAGIC_LENGTH) ==
          0) {
    return LOOM_MODULE_FORMAT_BYTECODE;
  }
  return LOOM_MODULE_FORMAT_TEXT;
}
