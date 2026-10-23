// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/cts/testdata.h"

#include <algorithm>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

iree_hal_amd_gpu_target_identity_t Identity(const char* key) {
  iree_hal_amd_gpu_target_identity_t identity;
  IREE_CHECK_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
      iree_make_cstring_view(key), &identity));
  return identity;
}

TEST(TestdataTest, PrefersExactIdentityRegardlessOfRegistrationOrder) {
  std::vector<iree::hal::cts::ExecutableTarget> targets = {
      {"generic", "amdgpu", "gfx11-generic", nullptr},
      {"exact", "amdgpu", "gfx1100", nullptr},
      {"other", "amdgpu", "gfx90a", nullptr},
  };
  for (size_t i = 0; i < targets.size(); ++i) {
    const auto* target = SelectExecutableTarget(targets, Identity("gfx1100"));
    ASSERT_NE(target, nullptr);
    EXPECT_STREQ(target->target_key, "gfx1100");
    target = SelectExecutableTarget(targets, Identity("gfx1151"));
    ASSERT_NE(target, nullptr);
    EXPECT_STREQ(target->target_key, "gfx11-generic");
    EXPECT_EQ(SelectExecutableTarget(targets, Identity("gfx950")), nullptr);
    std::rotate(targets.begin(), targets.begin() + 1, targets.end());
  }
}

TEST(TestdataTest, PreservesOverlayIdentityAndFeatureCompatibility) {
  const std::vector<iree::hal::cts::ExecutableTarget> targets = {
      {"generic", "amdgpu", "gfx12-5-generic", nullptr},
      {"a0", "amdgpu", "gfx1250-a0", nullptr},
      {"b0", "amdgpu", "gfx1250", nullptr},
      {"xnack", "amdgpu", "gfx942:xnack+", nullptr},
  };
  for (const char* key : {"gfx1250-a0", "gfx1250"}) {
    const auto* target = SelectExecutableTarget(targets, Identity(key));
    ASSERT_NE(target, nullptr);
    EXPECT_STREQ(target->target_key, key);
  }
  EXPECT_EQ(SelectExecutableTarget(targets, Identity("gfx942:xnack-")),
            nullptr);
  const auto* target =
      SelectExecutableTarget(targets, Identity("gfx942:xnack+"));
  ASSERT_NE(target, nullptr);
  EXPECT_STREQ(target->target_key, "gfx942:xnack+");
}

TEST(TestdataTest, ResolvesPhysicalRevisionAndReportsMissingPrograms) {
  iree::hal::cts::CtsRegistry::RegisterExecutableTarget(
      "lookup_fixture",
      {"a0", "amdgpu", "gfx1250-a0", [](iree_string_view_t name) {
         return iree_string_view_equal(name, IREE_SV("program.bin"))
                    ? iree_make_const_byte_span("a0", 2)
                    : iree_const_byte_span_empty();
       }});
  iree::hal::cts::CtsRegistry::RegisterExecutableTarget(
      "lookup_fixture",
      {"b0", "amdgpu", "gfx1250", [](iree_string_view_t name) {
         return iree_string_view_equal(name, IREE_SV("program.bin"))
                    ? iree_make_const_byte_span("b0", 2)
                    : iree_const_byte_span_empty();
       }});
  amdf_gpu_endpoint_info_t endpoint = {.gfx_ip = {12, 5, 0}};
  for (uint32_t revision = 0; revision < 2; ++revision) {
    endpoint.asic_revision = revision;
    iree_const_byte_span_t data;
    IREE_ASSERT_OK(LookupExecutableData("lookup_fixture", endpoint,
                                        IREE_SV("program.bin"), &data));
    ASSERT_EQ(data.data_length, 2u);
    EXPECT_EQ(data.data[0], revision ? 'b' : 'a');
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                          LookupExecutableData("lookup_fixture", endpoint,
                                               IREE_SV("absent.bin"), &data));
    EXPECT_EQ(data.data, nullptr);
  }
  iree_const_byte_span_t data;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        LookupExecutableData("absent_collection", endpoint,
                                             IREE_SV("program.bin"), &data));
  EXPECT_EQ(data.data, nullptr);
}

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
