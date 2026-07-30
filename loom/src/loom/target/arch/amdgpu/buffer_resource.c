// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/buffer_resource.h"

static const loom_amdgpu_buffer_resource_record_encoding_info_t kBufferResourceRecordEncodingInfos
    [LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_COUNT] = {
        [LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_NONE] = {0},
        [LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_BASE48_NUM_RECORDS32_LEGACY_FORMAT] =
            {
                .pointer_high_mask = UINT32_C(0x0000ffff),
                .raw_bounds_checked_word3_control =
                    LOOM_AMDGPU_BUFFER_RESOURCE_BASE48_LEGACY_RAW_WORD3,
                .num_records_word1_bit_count = 0,
            },
        [LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_BASE48_NUM_RECORDS32_UNIFIED_FORMAT] =
            {
                .pointer_high_mask = UINT32_C(0x0000ffff),
                .raw_bounds_checked_word3_control =
                    LOOM_AMDGPU_BUFFER_RESOURCE_BASE48_UNIFIED_RAW_WORD3,
                .num_records_word1_bit_count = 0,
            },
        [LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_BASE57_NUM_RECORDS45] =
            {
                .pointer_high_mask = UINT32_C(0x00000000),
                .raw_bounds_checked_word3_control =
                    LOOM_AMDGPU_BUFFER_RESOURCE_BASE57_RAW_WORD3,
                .num_records_word1_bit_count = 7,
            },
};

const loom_amdgpu_buffer_resource_record_encoding_info_t*
loom_amdgpu_buffer_resource_record_encoding_info(
    loom_amdgpu_buffer_resource_record_encoding_t record_encoding) {
  IREE_ASSERT(
      record_encoding > LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_NONE &&
      record_encoding < LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_COUNT);
  return &kBufferResourceRecordEncodingInfos[record_encoding];
}
