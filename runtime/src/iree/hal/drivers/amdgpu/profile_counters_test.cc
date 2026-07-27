// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_counters.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_gfxip_version_t GfxIpFromProcessor(
    iree_string_view_t processor) {
  iree_hal_amdgpu_target_id_t target_id = {};
  IREE_EXPECT_OK(iree_hal_amdgpu_target_id_parse(
      processor, IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY,
      &target_id));
  return target_id.version;
}

static iree_hal_amdgpu_aqlprofile_pmc_event_t ResolveCounter(
    iree_string_view_t name, iree_string_view_t processor) {
  iree_hal_amdgpu_aqlprofile_pmc_event_t event = {};
  IREE_EXPECT_OK(iree_hal_amdgpu_profile_counter_resolve_named_event(
      name, GfxIpFromProcessor(processor), &event));
  return event;
}

static iree_status_code_t ResolveCounterStatus(iree_string_view_t name,
                                               iree_string_view_t processor) {
  iree_hal_amdgpu_aqlprofile_pmc_event_t event = {};
  iree_status_t status = iree_hal_amdgpu_profile_counter_resolve_named_event(
      name, GfxIpFromProcessor(processor), &event);
  iree_status_code_t status_code = iree_status_code(status);
  iree_status_free(status);
  return status_code;
}

TEST(ProfileCountersTest, MapsGfx942ThroughGfx940CounterFamily) {
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx940_event =
      ResolveCounter(IREE_SV("SQ_INSTS_VMEM_RD"), IREE_SV("gfx940"));
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx941_event =
      ResolveCounter(IREE_SV("SQ_INSTS_VMEM_RD"), IREE_SV("gfx941"));
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx942_event =
      ResolveCounter(IREE_SV("SQ_INSTS_VMEM_RD"), IREE_SV("gfx942"));

  EXPECT_EQ(gfx940_event.event_id, 58u);
  EXPECT_EQ(gfx941_event.event_id, 58u);
  EXPECT_EQ(gfx942_event.event_id, 58u);
  EXPECT_EQ(gfx942_event.block_name, IREE_HAL_AMDGPU_AQLPROFILE_BLOCK_NAME_SQ);

  iree_hal_amdgpu_aqlprofile_pmc_event_t tcc_event =
      ResolveCounter(IREE_SV("TCC_EA0_RDREQ_DRAM"), IREE_SV("gfx942"));
  EXPECT_EQ(tcc_event.event_id, 102u);
  EXPECT_EQ(tcc_event.block_name, IREE_HAL_AMDGPU_AQLPROFILE_BLOCK_NAME_TCC);
}

TEST(ProfileCountersTest, KeepsGfx90aAndGfx940Separate) {
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx90a_vmem_read =
      ResolveCounter(IREE_SV("SQ_INSTS_VMEM_RD"), IREE_SV("gfx90a"));
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx940_vmem_read =
      ResolveCounter(IREE_SV("SQ_INSTS_VMEM_RD"), IREE_SV("gfx940"));
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx942_vmem_read =
      ResolveCounter(IREE_SV("SQ_INSTS_VMEM_RD"), IREE_SV("gfx942"));

  EXPECT_EQ(gfx90a_vmem_read.event_id, 54u);
  EXPECT_EQ(gfx940_vmem_read.event_id, 58u);
  EXPECT_EQ(gfx942_vmem_read.event_id, 58u);

  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx90a_tcp_read =
      ResolveCounter(IREE_SV("TCP_TOTAL_READ"), IREE_SV("gfx90a"));
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx940_tcp_read =
      ResolveCounter(IREE_SV("TCP_TOTAL_READ"), IREE_SV("gfx940"));
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx942_tcp_read =
      ResolveCounter(IREE_SV("TCP_TOTAL_READ"), IREE_SV("gfx942"));

  EXPECT_EQ(gfx90a_tcp_read.event_id, 30u);
  EXPECT_EQ(gfx940_tcp_read.event_id, 28u);
  EXPECT_EQ(gfx942_tcp_read.event_id, 28u);
}

TEST(ProfileCountersTest, UsesRocprofilerGfx10AndGfx11SqValuIds) {
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx1010_event =
      ResolveCounter(IREE_SV("SQ_INSTS_VALU"), IREE_SV("gfx1010"));
  iree_hal_amdgpu_aqlprofile_pmc_event_t gfx1100_event =
      ResolveCounter(IREE_SV("SQ_INSTS_VALU"), IREE_SV("gfx1100"));

  EXPECT_EQ(gfx1010_event.event_id, 64u);
  EXPECT_EQ(gfx1100_event.event_id, 62u);
}

TEST(ProfileCountersTest, RejectsUnsupportedExactTargets) {
  EXPECT_EQ(ResolveCounterStatus(IREE_SV("SQ_WAVES"), IREE_SV("gfx950")),
            IREE_STATUS_UNIMPLEMENTED);
  EXPECT_EQ(ResolveCounterStatus(IREE_SV("SQ_WAVES"), IREE_SV("gfx1103")),
            IREE_STATUS_UNIMPLEMENTED);
  EXPECT_EQ(ResolveCounterStatus(IREE_SV("SQ_WAVES"), IREE_SV("gfx1200")),
            IREE_STATUS_UNIMPLEMENTED);
}

TEST(ProfileCountersTest, RejectsUnsupportedCounterOnSupportedTarget) {
  EXPECT_EQ(ResolveCounterStatus(IREE_SV("SQ_WAVES_EQ_64"), IREE_SV("gfx1100")),
            IREE_STATUS_UNIMPLEMENTED);
}

TEST(ProfileCountersTest, SupportedNamesUseRocprofilerNames) {
  std::string supported_names(
      iree_hal_amdgpu_profile_counter_supported_names().data,
      iree_hal_amdgpu_profile_counter_supported_names().size);

  EXPECT_NE(supported_names.find("SQ_WAVES_EQ_64"), std::string::npos);
  EXPECT_EQ(supported_names.find("SQ_WAVES_32"), std::string::npos);
  EXPECT_EQ(supported_names.find("SQ_WAVES_64"), std::string::npos);
}

}  // namespace
}  // namespace iree::hal::amdgpu
