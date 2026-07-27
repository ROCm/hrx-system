// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/stable_id.h"

uint64_t loom_stable_id_from_string(iree_string_view_t key) {
  uint64_t value = UINT64_C(0xCBF29CE484222325);
  for (iree_host_size_t i = 0; i < key.size; ++i) {
    value ^= (uint8_t)key.data[i];
    value *= UINT64_C(0x100000001B3);
  }
  value &= UINT64_C(0x7FFFFFFFFFFFFFFF);
  return value == LOOM_STABLE_ID_NONE ? UINT64_C(1) : value;
}
