// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/function_version.h"

loom_function_version_t* loom_function_version_list_find(
    const loom_function_version_list_t* list, loom_func_like_t function) {
  if (list == NULL || function.op == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    loom_function_version_t* version = list->values[i];
    if (version != NULL && version->function.op == function.op &&
        version->function.vtable == function.vtable) {
      return version;
    }
  }
  return NULL;
}

void loom_function_version_update(loom_function_version_t* version,
                                  loom_func_like_t function) {
  IREE_ASSERT_ARGUMENT(version);
  IREE_ASSERT(function.op != NULL);
  version->function = function;
}
