// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/abi/tsan.h"

#include <cstddef>
#include <cstdint>

#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/testing/gtest.h"

namespace {

TEST(AmdgpuTsanAbiTest, ConfigConstantsMatchHalTsanAbi) {
  EXPECT_STREQ(LOOM_AMDGPU_TSAN_CONFIG_GLOBAL_NAME,
               IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME);
  EXPECT_EQ(LOOM_AMDGPU_TSAN_CONFIG_BYTE_LENGTH,
            sizeof(iree_hal_amdgpu_tsan_config_t));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_CONFIG_FLAGS_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_config_t, flags));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_CONFIG_SHADOW_BASE_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_config_t, shadow_base));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_CONFIG_SHADOW_SIZE_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_config_t, shadow_size));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_CONFIG_FLAG_ENABLED),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_ENABLED));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_SHADOW),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_SHADOW));
}

TEST(AmdgpuTsanAbiTest, ShadowConstantsMatchHalTsanAbi) {
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_HISTORY_BYTE_LENGTH),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_HISTORY_BYTE_LENGTH));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_HISTORY_COUNT),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_HISTORY_COUNT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_CELL_BYTE_LENGTH),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_CELL_BYTE_LENGTH));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_BYTE_LENGTH),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_BYTE_LENGTH));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_EPOCH_OFFSET),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_EPOCH_OFFSET));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_GENERATION_TAG_OFFSET),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_GENERATION_TAG_OFFSET));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_RESERVED_OFFSET),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_RESERVED_OFFSET));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_MARKER),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_MARKER));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_CHUNK_STATE_OFFSET),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_CHUNK_STATE_OFFSET));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_SHADOW_CHUNK_APPLICATION_BYTE_LENGTH),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_APPLICATION_BYTE_LENGTH));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_SHADOW_CHUNK_APPLICATION_BYTE_SHIFT),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_APPLICATION_BYTE_SHIFT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BIT_COUNT),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BIT_COUNT));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_WORD_BYTE_LENGTH),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_STATE_WORD_BYTE_LENGTH));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_CHUNKS_PER_STATE_WORD),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNKS_PER_STATE_WORD));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BITMAP_ALIGNMENT),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BITMAP_ALIGNMENT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_UNINITIALIZED),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_STATE_UNINITIALIZED));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_CLAIMED),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_STATE_CLAIMED));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_READY),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_CHUNK_STATE_READY));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_SHIFT),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_SHIFT));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_GENERATION_WINDOW_SHIFT),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_GENERATION_WINDOW_SHIFT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_GENERATION_WINDOW_CAPACITY),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_GENERATION_WINDOW_CAPACITY));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC));
}

TEST(AmdgpuTsanAbiTest, ShadowLayoutMarkerCannotAliasPayload) {
  const uint32_t marker = LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_MARKER;
  const uint32_t chunk_state_mask =
      (1u << LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BIT_COUNT) - 1u;
  EXPECT_EQ(marker & chunk_state_mask, 2u);

  const uint32_t access_kind_mask =
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT) - 1u;
  EXPECT_GT(marker & access_kind_mask,
            static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC));
}

TEST(AmdgpuTsanAbiTest, ReportConstantsMatchHalTsanAbi) {
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION,
            IREE_HAL_AMDGPU_TSAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH,
            sizeof(iree_hal_amdgpu_tsan_report_t));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t, record_length));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t, abi_version));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t, check_kind));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t, flags));
  EXPECT_EQ(
      LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET,
      offsetof(iree_hal_amdgpu_tsan_report_t, detail.data_race.memory_space));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_ACCESS_KIND_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_access_kind));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_PRIOR_ACCESS_KIND_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.prior_access_kind));
  EXPECT_EQ(
      LOOM_AMDGPU_TSAN_REPORT_ACCESS_SIZE_OFFSET,
      offsetof(iree_hal_amdgpu_tsan_report_t, detail.data_race.access_size));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_SITE_ID_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_site_id));
  EXPECT_EQ(
      LOOM_AMDGPU_TSAN_REPORT_PRIOR_SITE_ID_OFFSET,
      offsetof(iree_hal_amdgpu_tsan_report_t, detail.data_race.prior_site_id));
  EXPECT_EQ(
      LOOM_AMDGPU_TSAN_REPORT_MEMORY_ADDRESS_OFFSET,
      offsetof(iree_hal_amdgpu_tsan_report_t, detail.data_race.memory_address));
  EXPECT_EQ(
      LOOM_AMDGPU_TSAN_REPORT_SHADOW_ADDRESS_OFFSET,
      offsetof(iree_hal_amdgpu_tsan_report_t, detail.data_race.shadow_address));
  EXPECT_EQ(
      LOOM_AMDGPU_TSAN_REPORT_SHADOW_VALUE_OFFSET,
      offsetof(iree_hal_amdgpu_tsan_report_t, detail.data_race.shadow_value));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_X_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_workgroup_id[0]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Y_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_workgroup_id[1]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Z_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_workgroup_id[2]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_X_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_workitem_id[0]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Y_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_workitem_id[1]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Z_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.current_workitem_id[2]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_X_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.prior_workgroup_id[0]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Y_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.prior_workgroup_id[1]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Z_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.prior_workgroup_id[2]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_X_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.prior_workitem_id[0]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Y_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.prior_workitem_id[1]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Z_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.data_race.prior_workitem_id[2]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_SIZE_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.shadow_size));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_REQUIRED_SHADOW_END_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.required_shadow_end));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_SHADOW_STRIDE_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.workgroup_shadow_stride));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ORDINAL_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.workgroup_ordinal));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_LOCAL_MEMORY_SIZE_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.workgroup_local_memory_size));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_CELL_SIZE_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.shadow_cell_size));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_HISTORY_COUNT_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.shadow_history_count));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_X_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.current_workgroup_id[0]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_Y_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.current_workgroup_id[1]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_Z_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.current_workgroup_id[2]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_X_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.current_workitem_id[0]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_Y_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.current_workitem_id[1]));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_Z_OFFSET,
            offsetof(iree_hal_amdgpu_tsan_report_t,
                     detail.shadow_capacity.current_workitem_id[2]));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_CHECK_KIND_UNKNOWN),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_CHECK_KIND_UNKNOWN));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_CHECK_KIND_DATA_RACE),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_CHECK_KIND_DATA_RACE));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_CHECK_KIND_SHADOW_CAPACITY),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_CHECK_KIND_SHADOW_CAPACITY));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_MEMORY_SPACE_UNKNOWN),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_UNKNOWN));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_MEMORY_SPACE_PRIVATE),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_PRIVATE));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_ACCESS_KIND_READ),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_ACCESS_KIND_WRITE),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_ACCESS_KIND_ATOMIC),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_ATOMIC));
  EXPECT_EQ(static_cast<uint32_t>(LOOM_AMDGPU_TSAN_REPORT_FLAG_NONE),
            static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_NONE));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_ATOMIC),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_CURRENT_ATOMIC));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_ATOMIC),
      static_cast<uint32_t>(IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_PRIOR_ATOMIC));
  EXPECT_EQ(
      static_cast<uint32_t>(LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR),
      static_cast<uint32_t>(
          IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR));
  EXPECT_EQ(static_cast<uint32_t>(
                LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR),
            static_cast<uint32_t>(
                IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR));
}

}  // namespace
