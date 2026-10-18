// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line module format spelling and bytecode detection.

#ifndef LOOM_TOOLING_IO_MODULE_FORMAT_H_
#define LOOM_TOOLING_IO_MODULE_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supported external module formats.
typedef enum loom_module_format_e {
  LOOM_MODULE_FORMAT_AUTO = 0,
  LOOM_MODULE_FORMAT_TEXT = 1,
  LOOM_MODULE_FORMAT_BYTECODE = 2,
} loom_module_format_t;

// Returns the stable flag spelling for |format|.
const char* loom_module_format_name(loom_module_format_t format);

// Parses a format flag value: auto, text, bc, or bytecode.
iree_status_t loom_module_format_parse(iree_string_view_t value,
                                       bool allow_auto,
                                       loom_module_format_t* out_format);

// Detects the input format from bytes using the bytecode file magic.
loom_module_format_t loom_module_format_detect_input(
    iree_const_byte_span_t input);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_IO_MODULE_FORMAT_H_
