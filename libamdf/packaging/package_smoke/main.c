// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/amdf.h"

int main(void) {
  const amdf_api_t* api = 0;
  const amdf_status_t status =
      amdf_query_api(AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api);
  return amdf_status_is_ok(status) && api != 0 &&
                 api->abi_version == AMDF_ABI_VERSION_1
             ? 0
             : 1;
}
