// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/records/target_records.h"

#include "iree/testing/gtest.h"

namespace {

static void ExpectWorkgroupStorageLimit(iree_string_view_t processor_name,
                                        uint64_t expected_byte_limit) {
  const loom_amdgpu_target_record_info_t* info =
      loom_amdgpu_target_record_info_for_processor(processor_name);
  ASSERT_NE(info, nullptr);
  ASSERT_NE(info->bundle, nullptr);
  ASSERT_NE(info->bundle->snapshot, nullptr);
  EXPECT_EQ(info->bundle->snapshot->max_workgroup_storage_bytes,
            expected_byte_limit);
}

TEST(AmdgpuTargetRecordsTest, ModelsGfx125xWorkgroupStorageLimit) {
  ExpectWorkgroupStorageLimit(IREE_SV("gfx1250"), 320u * 1024u);
  ExpectWorkgroupStorageLimit(IREE_SV("gfx12-5-generic"), 320u * 1024u);
}

TEST(AmdgpuTargetRecordsTest, PreservesDefaultWorkgroupStorageLimit) {
  ExpectWorkgroupStorageLimit(IREE_SV("gfx1200"), 64u * 1024u);
}

}  // namespace
