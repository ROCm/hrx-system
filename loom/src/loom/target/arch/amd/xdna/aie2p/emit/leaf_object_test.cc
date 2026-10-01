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
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

namespace loom {
namespace {

TEST(Aie2pLeafObjectTest, EmitsPreparedProgramRepeatedlyWithoutMutation) {
  const loom_aie2p_instruction_id_t movxm =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVXM"));
  const loom_aie2p_encoding_field_id_t destination =
      loom_aie2p_encoding_find_field(IREE_SV("dst"));
  const loom_aie2p_encoding_field_id_t immediate =
      loom_aie2p_encoding_find_field(IREE_SV("i"));
  ASSERT_NE(movxm, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  const loom_aie2p_encoding_field_value_t movxm_fields[] = {
      {destination, 0},
      {immediate, 0},
  };
  uint64_t movxm_value = UINT64_MAX;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
      movxm, movxm_fields, IREE_ARRAYSIZE(movxm_fields), &movxm_value));

  const loom_aie2p_bundle_format_id_t movxm_format =
      loom_aie2p_encoding_find_bundle_format(IREE_SV("I48_LNG"));
  ASSERT_NE(movxm_format, LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
  loom_aie2p_bundle_format_info_t movxm_format_info;
  ASSERT_TRUE(loom_aie2p_encoding_query_bundle_format_info(movxm_format,
                                                           &movxm_format_info));
  ASSERT_EQ(movxm_format_info.bit_count, 48u);

  const loom_aie2p_planned_slot_t slot = {
      /*.encoded_slot=*/{LOOM_AIE2P_SLOT_LNG, movxm_value},
      /*.scheduled_packet_index=*/0,
      /*.flags=*/LOOM_AIE2P_PLANNED_SLOT_FLAG_READ_ONLY_DATA_ADDRESS,
  };
  const loom_aie2p_planned_bundle_t bundle = {
      /*.issue_cycle=*/0,
      /*.block_index=*/0,
      /*.logical_issue_cycle=*/0,
      /*.byte_offset=*/0,
      /*.slot_start=*/0,
      /*.format=*/movxm_format,
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
  const uint8_t table_contents[] = {0x11, 0x22, 0x33, 0x44, 0x55};
  const loom_aie2p_leaf_read_only_data_t read_only_data = {
      /*.name=*/IREE_SV("table"),
      /*.contents=*/
      iree_make_const_byte_span(table_contents, sizeof(table_contents)),
      /*.minimum_alignment=*/32,
  };
  const loom_aie2p_planned_read_only_data_fixup_t read_only_data_fixup = {
      /*.bundle_index=*/0,
      /*.read_only_data_ordinal=*/0,
  };
  loom_aie2p_leaf_program_plan_t plan = {
      /*.function_name=*/IREE_SV("kernel"),
      /*.storage_requirements=*/{},
      /*.spill=*/{16, 16},
      /*.resource_imports=*/&resource_import,
      /*.resource_import_count=*/1,
      /*.read_only_data=*/&read_only_data,
      /*.read_only_data_count=*/1,
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
      /*.read_only_data_fixups=*/&read_only_data_fixup,
      /*.read_only_data_fixup_count=*/1,
      /*.encoded_byte_length=*/6,
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

  ASSERT_EQ(first.object.section_count, 3u);
  ASSERT_EQ(first.object.symbol_count, 3u);
  ASSERT_EQ(first.object.fixup_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(first.object.sections[0].section_name,
                                     IREE_SV(".text.kernel")));
  EXPECT_TRUE(iree_string_view_equal(first.object.sections[1].section_name,
                                     IREE_SV(".rodata.table")));
  EXPECT_TRUE(iree_string_view_equal(first.object.sections[2].section_name,
                                     IREE_SV(".storage.kernel.scratch")));
  EXPECT_TRUE(
      iree_string_view_equal(first.object.symbols[0].name, IREE_SV("kernel")));
  EXPECT_TRUE(
      iree_string_view_equal(first.object.symbols[1].name, IREE_SV("table")));
  EXPECT_TRUE(iree_string_view_equal(first.object.symbols[2].name,
                                     IREE_SV("kernel.scratch")));
  ASSERT_EQ(first.object.sections[0].contents.data_length, 6u);
  ASSERT_EQ(first.object.sections[1].contents.data_length,
            sizeof(table_contents));
  EXPECT_NE(first.object.sections[1].contents.data, table_contents);
  EXPECT_EQ(0, std::memcmp(first.object.sections[1].contents.data,
                           table_contents, sizeof(table_contents)));
  EXPECT_EQ(first.object.sections[1].storage,
            LOOM_NATIVE_SECTION_STORAGE_CONTENTS);
  EXPECT_EQ(first.object.sections[1].access, LOOM_NATIVE_SECTION_ACCESS_READ);
  EXPECT_EQ(first.object.sections[1].contribution_alignment, 32u);
  EXPECT_EQ(first.object.sections[2].reservation_length, 64u);
  EXPECT_EQ(first.object.sections[2].contribution_alignment, 16u);
  EXPECT_EQ(first.object.fixups[0].section_contribution_index, 0u);
  EXPECT_EQ(first.object.fixups[0].section_offset, 0u);
  EXPECT_EQ(first.object.fixups[0].relocation_kind,
            LOOM_AIE2P_NATIVE_RELOCATION_KIND_LOCAL_ADDRESS_ABSOLUTE);
  EXPECT_EQ(first.object.fixups[0].target_symbol_index, 1u);
  EXPECT_EQ(first.object.fixups[0].addend, 0);

  ASSERT_EQ(first.realization.read_only_data_count, 1u);
  EXPECT_EQ(first.realization.read_only_data[0].section_contribution_index, 1u);
  EXPECT_EQ(first.realization.read_only_data[0].symbol_index, 1u);
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
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS |
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_READ_ONLY_DATA |
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE |
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_MATERIALIZED_SPILLS));

  ASSERT_EQ(second.object.section_count, first.object.section_count);
  ASSERT_EQ(second.object.symbol_count, first.object.symbol_count);
  EXPECT_NE(second.object.sections, first.object.sections);
  EXPECT_NE(second.object.sections[0].contents.data,
            first.object.sections[0].contents.data);
  EXPECT_EQ(0, std::memcmp(second.object.sections[0].contents.data,
                           first.object.sections[0].contents.data, 6));
  EXPECT_NE(second.object.sections[1].contents.data,
            first.object.sections[1].contents.data);
  EXPECT_EQ(0, std::memcmp(second.object.sections[1].contents.data,
                           first.object.sections[1].contents.data,
                           sizeof(table_contents)));
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
