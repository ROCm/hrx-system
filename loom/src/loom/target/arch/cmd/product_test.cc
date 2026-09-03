// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/product.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_byte_sequence_t* MakeProgramData() {
  static const uint8_t kData[] = {0x4C, 0x4F, 0x4F, 0x4D};
  iree_byte_span_t data = iree_byte_span_empty();
  data.data_length = sizeof(kData);
  IREE_CHECK_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), data.data_length, (void**)&data.data));
  memcpy(data.data, kData, sizeof(kData));
  iree_byte_sequence_t* contents = nullptr;
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(
      &data, iree_allocator_system(), &contents));
  return contents;
}

static loom_cmd_program_artifact_set_t MakeArtifactSet(
    uint32_t entry_requirement_ordinal) {
  loom_cmd_program_artifact_set_t artifact_set = {
      /*.programs=*/{},
      /*.entries=*/{},
      /*.entry_requirement_index_storage=*/nullptr,
      /*.string_storage=*/nullptr,
      /*.host_allocator=*/iree_allocator_system(),
  };
  IREE_CHECK_OK(iree_allocator_malloc_array(
      artifact_set.host_allocator, 1, sizeof(*artifact_set.programs.values),
      (void**)&artifact_set.programs.values));
  artifact_set.programs.count = 1;
  IREE_CHECK_OK(iree_allocator_malloc_array(
      artifact_set.host_allocator, 1, sizeof(*artifact_set.entries.values),
      (void**)&artifact_set.entries.values));
  artifact_set.entries.count = 1;
  IREE_CHECK_OK(iree_allocator_malloc_array(
      artifact_set.host_allocator, 1,
      sizeof(*artifact_set.entry_requirement_index_storage),
      (void**)&artifact_set.entry_requirement_index_storage));
  artifact_set.entry_requirement_index_storage[0] = entry_requirement_ordinal;
  artifact_set.programs.values[0] = {
      /*.symbol=*/IREE_SV("decode"),
      /*.data=*/MakeProgramData(),
      /*.entry_requirement_indices=*/
      artifact_set.entry_requirement_index_storage,
      /*.entry_requirement_count=*/1,
  };
  artifact_set.entries.values[0] = {
      /*.symbol=*/IREE_SV("attention"),
  };
  return artifact_set;
}

TEST(CmdProductTest, MovesCompleteArtifactSet) {
  loom_cmd_program_artifact_set_t artifact_set = MakeArtifactSet(0);
  loom_product_t* product = nullptr;
  IREE_ASSERT_OK(loom_cmd_product_create(&artifact_set, iree_allocator_system(),
                                         &product));

  EXPECT_EQ(artifact_set.programs.values, nullptr);
  EXPECT_EQ(artifact_set.programs.count, 0u);
  EXPECT_TRUE(loom_product_isa(product, &loom_cmd_product_descriptor));
  EXPECT_TRUE(iree_string_view_equal(loom_cmd_product_operation.name,
                                     IREE_SV("command")));
  EXPECT_TRUE(iree_string_view_equal(loom_cmd_product_format.name,
                                     IREE_SV("loom-command")));
  EXPECT_EQ(loom_product_export_count(product), 1u);
  EXPECT_EQ(loom_product_requirement_count(product), 1u);

  const loom_product_artifact_t* artifact =
      loom_product_artifact_at(product, 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      artifact->role, IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_COMMAND_PROGRAM)));
  EXPECT_TRUE(iree_string_view_equal(
      artifact->format, IREE_SV(LOOM_CMD_PRODUCT_FORMAT_LOOM_COMMAND)));
  EXPECT_TRUE(iree_string_view_equal(artifact->identifier, IREE_SV("decode")));

  const loom_cmd_program_artifact_set_t* retained_artifact_set =
      loom_cmd_product_artifact_set(product);
  ASSERT_NE(retained_artifact_set, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      retained_artifact_set->programs.values[0].symbol, IREE_SV("decode")));
  EXPECT_EQ(
      retained_artifact_set->programs.values[0].entry_requirement_indices[0],
      0u);
  EXPECT_TRUE(iree_string_view_equal(
      retained_artifact_set->entries.values[0].symbol, IREE_SV("attention")));
  IREE_EXPECT_OK(
      loom_product_format_validate_product(&loom_cmd_product_format, product));

  loom_product_release(product);
}

TEST(CmdProductTest, RejectsInvalidEntryProjectionWithoutTakingOwnership) {
  loom_cmd_program_artifact_set_t artifact_set = MakeArtifactSet(1);
  loom_product_t* product = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_product_create(
                            &artifact_set, iree_allocator_system(), &product));
  EXPECT_EQ(product, nullptr);
  EXPECT_NE(artifact_set.programs.values, nullptr);
  loom_cmd_program_artifact_set_deinitialize(&artifact_set);
}

}  // namespace
}  // namespace loom
