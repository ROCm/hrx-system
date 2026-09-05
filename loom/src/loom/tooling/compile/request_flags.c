// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/request_flags.h"

#include "iree/base/tooling/flags.h"

IREE_FLAG(string, product, "",
          "Optional product: 'kernel', 'command', or 'module'. Selected roots "
          "infer their product and an explicit value asserts that result.");
IREE_FLAG(string, format, "",
          "Optional exact configured artifact format. Omit this to select the "
          "canonical format for the inferred product and target.");
IREE_FLAG(string, target, "",
          "Optional compilation target in family:selector form. Selected "
          "kernel roots are specialized to that exact configured profile; "
          "authored targets remain compatibility requirements.");

loom_compile_request_options_t loom_compile_request_options_from_flags(
    iree_string_view_list_t roots) {
  return (loom_compile_request_options_t){
      .roots = roots,
      .product = iree_make_cstring_view(FLAG_product),
      .format = iree_make_cstring_view(FLAG_format),
      .target = iree_make_cstring_view(FLAG_target),
  };
}
