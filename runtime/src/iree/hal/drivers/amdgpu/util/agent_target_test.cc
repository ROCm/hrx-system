// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/agent_target.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static std::string FormatTargetId(
    const iree_hal_amdgpu_agent_isa_target_t& isa_target) {
  char buffer[128] = {0};
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_format_artifact_key(
      &isa_target.identity, sizeof(buffer), buffer,
      /*out_buffer_length=*/nullptr));
  return buffer;
}

static iree_status_t InitializeSingleTarget(
    hsa_agent_t agent, hsa_isa_t isa, iree_string_view_t isa_name,
    uint32_t asic_revision, iree_hal_amdgpu_agent_target_t* out_target) {
  const iree_hal_amdgpu_agent_isa_value_t isa_value = {
      /*isa=*/isa,
      /*name=*/isa_name,
  };
  return iree_hal_amdgpu_agent_target_initialize(
      agent, 1, &isa_value, asic_revision, iree_allocator_system(), out_target);
}

TEST(AgentTargetTest, OwnsParsedIdentity) {
  std::string isa_name = "amdgcn-amd-amdhsa--gfx942:xnack-:sramecc+";
  iree_hal_amdgpu_agent_target_t target;
  IREE_ASSERT_OK(InitializeSingleTarget(
      hsa_agent_t{42}, hsa_isa_t{7},
      iree_make_string_view(isa_name.data(), isa_name.size()),
      /*asic_revision=*/0, &target));
  isa_name.assign(isa_name.size(), 'x');

  EXPECT_EQ(target.agent.handle, 42u);
  EXPECT_EQ(target.isa_count, 1u);
  EXPECT_EQ(target.primary_isa.isa.handle, 7u);
  EXPECT_EQ(FormatTargetId(target.primary_isa), "gfx942:sramecc+:xnack-");
  iree_hal_amdgpu_agent_target_deinitialize(&target);
}

TEST(AgentTargetTest, OwnsMultipleIsasInReportedOrder) {
  std::string primary_name = "amdgcn-amd-amdhsa--gfx942:xnack-:sramecc+";
  std::string fallback_name = "amdgcn-amd-amdhsa--gfx9-4-generic";
  const iree_hal_amdgpu_agent_isa_value_t isa_values[] = {
      {
          /*isa=*/hsa_isa_t{7},
          /*name=*/
          iree_make_string_view(primary_name.data(), primary_name.size()),
      },
      {
          /*isa=*/hsa_isa_t{8},
          /*name=*/
          iree_make_string_view(fallback_name.data(), fallback_name.size()),
      },
  };
  iree_hal_amdgpu_agent_target_t target;
  IREE_ASSERT_OK(iree_hal_amdgpu_agent_target_initialize(
      hsa_agent_t{42}, IREE_ARRAYSIZE(isa_values), isa_values,
      /*asic_revision=*/0, iree_allocator_system(), &target));
  primary_name.assign(primary_name.size(), 'x');
  fallback_name.assign(fallback_name.size(), 'x');

  ASSERT_EQ(target.isa_count, 2u);
  EXPECT_EQ(iree_hal_amdgpu_agent_target_isa_at(&target, 0)->isa.handle, 7u);
  EXPECT_EQ(iree_hal_amdgpu_agent_target_isa_at(&target, 1)->isa.handle, 8u);
  EXPECT_EQ(FormatTargetId(*iree_hal_amdgpu_agent_target_isa_at(&target, 0)),
            "gfx942:sramecc+:xnack-");
  EXPECT_EQ(FormatTargetId(*iree_hal_amdgpu_agent_target_isa_at(&target, 1)),
            "gfx9-4-generic");
  EXPECT_EQ(iree_hal_amdgpu_agent_target_isa_at(&target, 2), nullptr);
  iree_hal_amdgpu_agent_target_deinitialize(&target);
}

TEST(AgentTargetTest, FindsCompatibleAlternateIsa) {
  const iree_hal_amdgpu_agent_isa_value_t isa_values[] = {
      {
          /*isa=*/hsa_isa_t{7},
          /*name=*/IREE_SV("amdgcn-amd-amdhsa--gfx942:xnack-:sramecc+"),
      },
      {
          /*isa=*/hsa_isa_t{8},
          /*name=*/IREE_SV("amdgcn-amd-amdhsa--gfx942:xnack+:sramecc+"),
      },
  };
  iree_hal_amdgpu_agent_target_t target;
  IREE_ASSERT_OK(iree_hal_amdgpu_agent_target_initialize(
      hsa_agent_t{42}, IREE_ARRAYSIZE(isa_values), isa_values,
      /*asic_revision=*/0, iree_allocator_system(), &target));
  iree_hal_amdgpu_target_identity_t requested_target;
  IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_parse_artifact_key(
      IREE_SV("gfx942:sramecc+:xnack+"), &requested_target));

  const iree_hal_amdgpu_agent_isa_target_t* compatible_isa =
      iree_hal_amdgpu_agent_target_find_compatible_isa(&target,
                                                       &requested_target);
  ASSERT_NE(compatible_isa, nullptr);
  EXPECT_EQ(compatible_isa->isa.handle, 8u);
  iree_hal_amdgpu_agent_target_deinitialize(&target);
}

TEST(AgentTargetTest, ResolvesPhysicalTargets) {
  iree_hal_amdgpu_agent_target_t a0_target;
  IREE_ASSERT_OK(InitializeSingleTarget(hsa_agent_t{42}, hsa_isa_t{7},
                                        IREE_SV("amdgcn-amd-amdhsa--gfx1250"),
                                        /*asic_revision=*/0, &a0_target));
  EXPECT_EQ(FormatTargetId(a0_target.primary_isa), "gfx1250-a0");
  iree_hal_amdgpu_agent_target_deinitialize(&a0_target);

  iree_hal_amdgpu_agent_target_t b0_target;
  IREE_ASSERT_OK(InitializeSingleTarget(hsa_agent_t{42}, hsa_isa_t{7},
                                        IREE_SV("amdgcn-amd-amdhsa--gfx1250"),
                                        /*asic_revision=*/1, &b0_target));
  EXPECT_EQ(FormatTargetId(b0_target.primary_isa), "gfx1250");
  iree_hal_amdgpu_agent_target_deinitialize(&b0_target);
}

TEST(AgentTargetTest, RejectsUnknownPhysicalRevision) {
  iree_hal_amdgpu_agent_target_t target;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      InitializeSingleTarget(hsa_agent_t{42}, hsa_isa_t{7},
                             IREE_SV("amdgcn-amd-amdhsa--gfx1250"),
                             /*asic_revision=*/2, &target));
}

TEST(AgentTargetTest, RejectsEmptyAndOversizedNames) {
  iree_hal_amdgpu_agent_target_t target;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        InitializeSingleTarget(hsa_agent_t{42}, hsa_isa_t{7},
                                               iree_string_view_empty(),
                                               /*asic_revision=*/0, &target));

  std::string oversized_name(sizeof(target.primary_isa.isa_name_storage), 'x');
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      InitializeSingleTarget(
          hsa_agent_t{42}, hsa_isa_t{7},
          iree_make_string_view(oversized_name.data(), oversized_name.size()),
          /*asic_revision=*/0, &target));
}

}  // namespace
}  // namespace iree::hal::amdgpu
