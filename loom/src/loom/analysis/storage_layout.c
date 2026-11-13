// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_layout.h"

bool loom_storage_layout_append(uint64_t byte_length, uint64_t byte_alignment,
                                uint64_t* inout_byte_extent,
                                uint64_t* out_byte_offset) {
  IREE_ASSERT_ARGUMENT(inout_byte_extent);
  IREE_ASSERT(byte_alignment != 0 &&
              iree_is_power_of_two_uint64(byte_alignment));
  if (out_byte_offset != NULL) {
    *out_byte_offset = 0;
  }

  uint64_t byte_offset = 0;
  if (!iree_checked_align_u64(*inout_byte_extent, byte_alignment,
                              &byte_offset)) {
    return false;
  }
  uint64_t next_byte_extent = 0;
  if (!iree_checked_add_u64(byte_offset, byte_length, &next_byte_extent)) {
    return false;
  }
  if (out_byte_offset != NULL) {
    *out_byte_offset = byte_offset;
  }
  *inout_byte_extent = next_byte_extent;
  return true;
}
