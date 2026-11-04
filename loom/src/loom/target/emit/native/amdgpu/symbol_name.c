// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/symbol_name.h"

static bool loom_amdgpu_symbol_name_start_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '$';
}

static bool loom_amdgpu_symbol_name_continue_char(char c) {
  return loom_amdgpu_symbol_name_start_char(c) || (c >= '0' && c <= '9') ||
         c == '.';
}

iree_status_t loom_amdgpu_symbol_name_validate(iree_string_view_t symbol) {
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU artifact symbol name is required");
  }
  if (!loom_amdgpu_symbol_name_start_char(symbol.data[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU artifact symbol '%.*s' has invalid first character '%c'",
        (int)symbol.size, symbol.data, symbol.data[0]);
  }
  for (iree_host_size_t i = 1; i < symbol.size; ++i) {
    if (!loom_amdgpu_symbol_name_continue_char(symbol.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU artifact symbol '%.*s' contains invalid character '%c'",
          (int)symbol.size, symbol.data, symbol.data[i]);
    }
  }
  return iree_ok_status();
}
