// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Auto-generated executable target registration for CTS.
// Source template: runtime/src/iree/hal/cts/util/testdata_target.cc.tpl

#include "iree/hal/cts/util/registry.h"
#include "{HEADER_PATH}"

namespace iree::hal::cts {

static iree_const_byte_span_t Get{TARGET_FUNC_NAME}ExecutableData(
    iree_string_view_t file_name) {
  const iree_file_toc_t* toc = {IDENTIFIER}_create();
  for (size_t i = 0; toc[i].name != nullptr; ++i) {
    if (iree_string_view_equal(file_name,
                               iree_make_cstring_view(toc[i].name))) {
      return iree_make_const_byte_span(
          reinterpret_cast<const uint8_t*>(toc[i].data), toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

static bool {TARGET_VAR_NAME}_registered_ =
    (CtsRegistry::RegisterExecutableTarget(
         "{BACKEND_NAME}",
         {"{TARGET_NAME}", "{TARGET_FAMILY}", "{TARGET_KEY}",
          Get{TARGET_FUNC_NAME}ExecutableData}),
     true);

}  // namespace iree::hal::cts
