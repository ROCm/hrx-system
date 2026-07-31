// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_version.h"

const loom_function_version_type_t loom_target_function_version_type = {
    .name = IREE_SVL("target"),
};

loom_target_function_version_t* loom_target_function_version_list_find(
    const loom_function_version_list_t* list, loom_func_like_t function) {
  return loom_target_function_version_cast(
      loom_function_version_list_find(list, function));
}
