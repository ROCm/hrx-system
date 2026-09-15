// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "util/provider.h"

int amdf_cts_provider_initialize(int* argument_count, char*** argument_values) {
  (void)argument_count;
  (void)argument_values;
  return 1;
}

int amdf_cts_provider_deinitialize(void) { return 1; }

amdf_query_api_fn_t amdf_cts_provider_query_api(void) {
  return &amdf_query_api;
}
