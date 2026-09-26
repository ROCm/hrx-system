// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

namespace loom {
namespace {

TEST(Aie2pLeafObjectTest, EmitsPreparedProgramRepeatedlyWithoutMutation) {
  const loom_aie2p_instruction_id_t nop =
      loom_aie2p_encoding_find_instruction(IREE_SV("NOP"));
  ASSERT_NE(nop, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  uint64_t nop_value = UINT64_MAX;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
      nop, /*field_values=*/nullptr, /*field_value_count=*/0, &nop_value));

  const loom_aie2p_bundle_format_id_t nop_format =
      loom_aie2p_encoding_find_bundle_format(IREE_SV("I16_NOP"));
  ASSERT_NE(nop_format, LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
  loom_aie2p_bundle_format_info_t nop_format_info;
  ASSERT_TRUE(loom_aie2p_encoding_query_bundle_format_info(nop_format,
                                                           &nop_format_info));
  ASSERT_EQ(nop_format_info.bit_count, 16u);

  const loom_aie2p_planned_slot_t slot = {
      /*.encoded_slot=*/{LOOM_AIE2P_SLOT_NOP, nop_value},
      /*.scheduled_packet_index=*/UINT32_MAX,
      /*.flags=*/LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
  };
  const loom_aie2p_planned_bundle_t bundle = {
      /*.issue_cycle=*/0,
      /*.block_index=*/0,
      /*.logical_issue_cycle=*/0,
      /*.byte_offset=*/0,
      /*.slot_start=*/0,
      /*.format=*/nop_format,
      /*.slot_count=*/1,
  };
  const uint32_t block_byte_offset = 0;
  const loom_aie2p_leaf_resource_import_t resource_import = {
      /*.index=*/3,
      /*.extent=*/256,
      /*.cache_swizzle_stride=*/64,
      /*.physical_register=*/7,
      /*.physical_register_count=*/2,
      /*.extent_physical_register=*/UINT32_MAX,
      /*.descriptor_register_class_id=*/5,
      /*.extent_descriptor_register_class_id=*/0,
      /*.extent_physical_register_count=*/0,
      /*.flags=*/LOOM_AIE2P_LEAF_RESOURCE_FLAG_STATIC_EXTENT |
          LOOM_AIE2P_LEAF_RESOURCE_FLAG_CACHE_SWIZZLE_STRIDE,
      /*.import_kind=*/LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING,
      /*.source_type_kind=*/LOOM_TYPE_BUFFER,
  };
  loom_aie2p_leaf_program_plan_t plan = {
      /*.function_name=*/IREE_SV("kernel"),
      /*.storage_requirements=*/{},
      /*.spill=*/{16, 16},
      /*.resource_imports=*/&resource_import,
      /*.resource_import_count=*/1,
      /*.block_byte_offsets=*/&block_byte_offset,
      /*.block_count=*/1,
      /*.bundles=*/&bundle,
      /*.bundle_count=*/1,
      /*.issue_cycle_count=*/1,
      /*.slots=*/&slot,
      /*.slot_count=*/1,
      /*.branch_fixups=*/nullptr,
      /*.branch_fixup_count=*/0,
      /*.storage_fixups=*/nullptr,
      /*.storage_fixup_count=*/0,
      /*.encoded_byte_length=*/2,
      /*.register_writes=*/{},
  };
  plan.storage_requirements[LOOM_STORAGE_SPACE_SCRATCH] = {64, 16};

  const loom_aie2p_leaf_program_plan_t original_plan = plan;
  const loom_aie2p_planned_slot_t original_slot = slot;
  const loom_aie2p_planned_bundle_t original_bundle = bundle;
  const loom_aie2p_leaf_resource_import_t original_resource_import =
      resource_import;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_aie2p_leaf_contribution_t first = {};
  loom_aie2p_leaf_contribution_t second = {};
  IREE_ASSERT_OK(loom_aie2p_leaf_object_emit(&plan, &arena, &first));
  IREE_ASSERT_OK(loom_aie2p_leaf_object_emit(&plan, &arena, &second));

  ASSERT_EQ(first.object.section_count, 2u);
  ASSERT_EQ(first.object.symbol_count, 2u);
  ASSERT_EQ(first.object.fixup_count, 0u);
  EXPECT_TRUE(iree_string_view_equal(first.object.sections[0].section_name,
                                     IREE_SV(".text.kernel")));
  EXPECT_TRUE(iree_string_view_equal(first.object.sections[1].section_name,
                                     IREE_SV(".storage.kernel.scratch")));
  EXPECT_TRUE(
      iree_string_view_equal(first.object.symbols[0].name, IREE_SV("kernel")));
  EXPECT_TRUE(iree_string_view_equal(first.object.symbols[1].name,
                                     IREE_SV("kernel.scratch")));
  ASSERT_EQ(first.object.sections[0].contents.data_length, 2u);
  EXPECT_EQ(first.object.sections[0].contents.data[0], 0u);
  EXPECT_EQ(first.object.sections[0].contents.data[1], 0u);
  EXPECT_EQ(first.object.sections[1].zero_fill_length, 64u);
  EXPECT_EQ(first.object.sections[1].contribution_alignment, 16u);

  EXPECT_EQ(first.realization.storage_domain_count, 1u);
  EXPECT_EQ(first.realization.scratch.byte_length, 64u);
  EXPECT_EQ(first.realization.scratch.minimum_alignment, 16u);
  EXPECT_EQ(first.realization.spill.byte_length, 16u);
  EXPECT_EQ(first.realization.resource_import_count, 1u);
  EXPECT_EQ(first.realization.resource_imports[0].index, 3u);
  EXPECT_EQ(first.realization.resource_imports[0].extent, 256u);
  EXPECT_EQ(first.realization.resource_imports[0].cache_swizzle_stride, 64u);
  EXPECT_TRUE(iree_all_bits_set(
      first.realization.capability_flags,
      LOOM_AIE2P_LEAF_CAPABILITY_FLAG_RESOURCE_IMPORTS |
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE |
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_MATERIALIZED_SPILLS));

  ASSERT_EQ(second.object.section_count, first.object.section_count);
  ASSERT_EQ(second.object.symbol_count, first.object.symbol_count);
  EXPECT_NE(second.object.sections, first.object.sections);
  EXPECT_NE(second.object.sections[0].contents.data,
            first.object.sections[0].contents.data);
  EXPECT_EQ(0, std::memcmp(second.object.sections[0].contents.data,
                           first.object.sections[0].contents.data, 2));
  EXPECT_NE(second.realization.resource_imports,
            first.realization.resource_imports);
  EXPECT_EQ(second.realization.resource_imports[0].index,
            first.realization.resource_imports[0].index);
  EXPECT_EQ(second.realization.resource_imports[0].extent,
            first.realization.resource_imports[0].extent);

  EXPECT_EQ(plan.function_name.data, original_plan.function_name.data);
  EXPECT_EQ(plan.function_name.size, original_plan.function_name.size);
  EXPECT_EQ(plan.resource_imports, original_plan.resource_imports);
  EXPECT_EQ(plan.bundles, original_plan.bundles);
  EXPECT_EQ(plan.slots, original_plan.slots);
  EXPECT_EQ(plan.encoded_byte_length, original_plan.encoded_byte_length);
  EXPECT_EQ(slot.encoded_slot.value, original_slot.encoded_slot.value);
  EXPECT_EQ(bundle.format, original_bundle.format);
  EXPECT_EQ(resource_import.extent, original_resource_import.extent);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
