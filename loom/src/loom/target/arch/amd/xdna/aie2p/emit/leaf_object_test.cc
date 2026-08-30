// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"

namespace loom {
namespace {

struct CompiledLeaf {
  loom_module_t* module = nullptr;
  loom_low_emission_frame_t frame = {};
  loom_aie2p_bundle_plan_t plan = {};
  loom_native_object_contribution_t object = {};
};

class Aie2pLeafObjectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_aie2p_low_descriptor_registry_initialize(&registry_);
    iree_arena_initialize(&block_pool_, &planning_arena_);
    iree_arena_initialize(&block_pool_, &object_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&object_arena_);
    iree_arena_deinitialize(&planning_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t CompileSource(std::string_view source, CompiledLeaf* out_leaf) {
    *out_leaf = CompiledLeaf{};
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &parse_options.low_asm_environment);
    IREE_RETURN_IF_ERROR(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("aie2p_leaf_object_test.loom"), &context_,
                        &block_pool_, &parse_options, &out_leaf->module));
    if (out_leaf->module == nullptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P Low source failed to parse");
    }

    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &registry_.registry;
    verify_options.provider_list = loom_low_verify_provider_list_empty();
    verify_options.max_errors = 20;
    loom_low_verify_scratch_t verify_scratch =
        loom_low_verify_scratch_for_module(out_leaf->module);
    loom_low_verify_result_t verify_result = {};
    IREE_RETURN_IF_ERROR(loom_low_verify_module(
        out_leaf->module, &verify_options, &verify_scratch, &verify_result));
    if (verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P Low source failed verification");
    }

    loom_block_t* module_block = loom_module_block(out_leaf->module);
    if (module_block->op_count != 1 ||
        !loom_low_func_def_isa(loom_block_op(module_block, 0))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P test source has no Low function");
    }
    loom_op_t* function_op = loom_block_op(module_block, 0);
    loom_low_emission_frame_options_t frame_options = {};
    frame_options.descriptor_registry = &registry_.registry;
    frame_options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
        out_leaf->module, function_op, &frame_options, &planning_arena_,
        &out_leaf->frame));
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_build(
        &out_leaf->frame, &planning_arena_, &out_leaf->plan));
    return loom_aie2p_leaf_object_emit(&out_leaf->plan, &object_arena_,
                                       &out_leaf->object);
  }

  iree_status_t CompileVectorAdd(std::string_view vector_shape,
                                 std::string_view add_mnemonic,
                                 CompiledLeaf* out_leaf) {
    std::string source =
        "low.func.def target<amd.xdna.aie2p.core> @vector_add(\n"
        "    %lhs_ptr: reg<aie2p.ep>, %rhs_ptr: reg<aie2p.ep>, "
        "%out_ptr: reg<aie2p.ep>) asm {\n"
        "  %lhs = vlda.512.";
    source.append(vector_shape);
    source.append(" %lhs_ptr, 0\n  %rhs = vldb.512.");
    source.append(vector_shape);
    source.append(" %rhs_ptr, 0\n  %sum = ");
    source.append(add_mnemonic);
    source.append(" %rhs, %lhs\n  vst.512.");
    source.append(vector_shape);
    source.append(" %sum, %out_ptr, 0\n  return\n}\n");
    return CompileSource(source, out_leaf);
  }

  iree_status_t CompileResourceVectorAdd(CompiledLeaf* out_leaf) {
    return CompileSource(
        "low.func.def target<amd.xdna.aie2p.core> @vector_add_memory() asm {\n"
        "  %lhs_ptr = resource<native_pointer> {index = 0, source_type = "
        "buffer} : reg<aie2p.ep>\n"
        "  %rhs_ptr = resource<native_pointer> {index = 1, source_type = "
        "buffer} : reg<aie2p.ep>\n"
        "  %out_ptr = resource<native_pointer> {index = 2, source_type = "
        "buffer} : reg<aie2p.ep>\n"
        "  %lhs = vlda.512.i32x16 %lhs_ptr, 0\n"
        "  %rhs = vlda.512.i32x16 %rhs_ptr, 0\n"
        "  %sum = vadd.32 %lhs, %rhs\n"
        "  vst.512.i32x16 %sum, %out_ptr, 0\n"
        "  return\n"
        "}\n",
        out_leaf);
  }

  void ResetLeaf(CompiledLeaf* leaf) {
    if (leaf->module != nullptr) {
      loom_module_free(leaf->module);
      leaf->module = nullptr;
    }
    iree_arena_reset(&object_arena_);
    iree_arena_reset(&planning_arena_);
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t registry_ = {};
  iree_arena_allocator_t planning_arena_ = {};
  iree_arena_allocator_t object_arena_ = {};
};

TEST_F(Aie2pLeafObjectTest, LowFunctionsReproduceRetainedVectorLeaves) {
  struct TestCase {
    std::string_view vector_shape;
    std::string_view mnemonic;
    std::array<uint8_t, 32> expected;
  };
  const TestCase test_cases[] = {
      {
          "i32x16",
          "vadd.32",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x2d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
      {
          "i16x32",
          "vadd.16",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x1d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
      {
          "i8x64",
          "vadd.8",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x0d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.mnemonic);
    CompiledLeaf leaf;
    IREE_ASSERT_OK(
        CompileVectorAdd(test_case.vector_shape, test_case.mnemonic, &leaf));

    ASSERT_EQ(leaf.frame.schedule.issue_group_count, 4u);
    EXPECT_EQ(leaf.frame.schedule.issue_groups[0].issue_cycle, 0u);
    EXPECT_EQ(leaf.frame.schedule.issue_groups[1].issue_cycle, 7u);
    EXPECT_EQ(leaf.frame.schedule.issue_groups[2].issue_cycle, 9u);
    EXPECT_EQ(leaf.frame.schedule.issue_groups[3].issue_cycle, 10u);
    ASSERT_EQ(leaf.plan.bundle_count, 11u);
    ASSERT_EQ(leaf.plan.slot_count, 12u);
    ASSERT_EQ(leaf.plan.encoded_byte_length, 32u);
    EXPECT_EQ(leaf.plan.bundles[0].slot_count, 2u);

    iree_host_size_t synthetic_nop_count = 0;
    iree_host_size_t structural_control_count = 0;
    for (iree_host_size_t i = 0; i < leaf.plan.slot_count; ++i) {
      const loom_aie2p_planned_slot_t* slot = &leaf.plan.slots[i];
      if (iree_any_bit_set(slot->flags,
                           LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP)) {
        ++synthetic_nop_count;
      }
      if (iree_any_bit_set(slot->flags,
                           LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL)) {
        ++structural_control_count;
        EXPECT_EQ(leaf.plan.bundles[5].slot_start, i);
        EXPECT_NE(slot->scheduled_packet_index,
                  LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE);
      }
    }
    EXPECT_EQ(synthetic_nop_count, 7u);
    EXPECT_EQ(structural_control_count, 1u);

    ASSERT_EQ(leaf.object.section_count, 1u);
    ASSERT_EQ(leaf.object.symbol_count, 1u);
    ASSERT_EQ(leaf.object.fixup_count, 0u);
    const loom_native_section_contribution_t* section =
        &leaf.object.sections[0];
    EXPECT_TRUE(iree_string_view_equal(section->section_name,
                                       IREE_SV(".text.vector_add")));
    EXPECT_EQ(section->contribution_alignment, 16u);
    EXPECT_EQ(section->section_type, LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS);
    EXPECT_EQ(section->section_flags,
              LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                  LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR);
    EXPECT_TRUE(iree_string_view_equal(leaf.object.symbols[0].name,
                                       IREE_SV("vector_add")));
    EXPECT_EQ(leaf.object.symbols[0].size, 32u);

    loom_module_free(leaf.module);
    leaf.module = nullptr;
    ASSERT_EQ(section->contents.data_length, test_case.expected.size());
    EXPECT_EQ(0, memcmp(section->contents.data, test_case.expected.data(),
                        test_case.expected.size()));
    ResetLeaf(&leaf);
  }
}

TEST_F(Aie2pLeafObjectTest, ResourceImportsAnchorRegistersWithoutEmittingCode) {
  constexpr std::array<uint8_t, 32> kExpected = {
      0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x2d,
      0x01, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00,
  };
  CompiledLeaf leaf;
  IREE_ASSERT_OK(CompileResourceVectorAdd(&leaf));

  ASSERT_EQ(leaf.frame.schedule.issue_group_count, 4u);
  EXPECT_EQ(leaf.frame.schedule.issue_groups[0].issue_cycle, 0u);
  EXPECT_EQ(leaf.frame.schedule.issue_groups[1].issue_cycle, 7u);
  EXPECT_EQ(leaf.frame.schedule.issue_groups[2].issue_cycle, 9u);
  EXPECT_EQ(leaf.frame.schedule.issue_groups[3].issue_cycle, 10u);

  iree_host_size_t resource_packet_count = 0;
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(&leaf.frame.schedule);
       ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&leaf.frame.schedule, packet_index);
    if (loom_low_resource_isa(packet.node->op)) ++resource_packet_count;
  }
  EXPECT_EQ(resource_packet_count, 3u);
  const loom_low_packet_view_t lhs_load =
      loom_low_packet_at(&leaf.frame.schedule, 3);
  const loom_low_packet_view_t rhs_load =
      loom_low_packet_at(&leaf.frame.schedule, 4);
  ASSERT_NE(lhs_load.descriptor, nullptr);
  ASSERT_NE(rhs_load.descriptor, nullptr);
  EXPECT_EQ(lhs_load.node->issue_cycle, 0u);
  EXPECT_EQ(rhs_load.node->issue_cycle, 0u);
  EXPECT_EQ(lhs_load.node->source_descriptor, lhs_load.descriptor);
  EXPECT_EQ(rhs_load.node->source_descriptor, lhs_load.descriptor);
  EXPECT_NE(rhs_load.descriptor, lhs_load.descriptor);

  ASSERT_EQ(leaf.plan.bundle_count, 11u);
  ASSERT_EQ(leaf.plan.slot_count, 12u);
  ASSERT_EQ(leaf.plan.encoded_byte_length, kExpected.size());
  EXPECT_EQ(leaf.plan.bundles[0].slot_count, 2u);
  for (iree_host_size_t slot_index = 0; slot_index < leaf.plan.slot_count;
       ++slot_index) {
    const uint32_t packet_index =
        leaf.plan.slots[slot_index].scheduled_packet_index;
    if (packet_index == LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) continue;
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&leaf.frame.schedule, packet_index);
    EXPECT_FALSE(loom_low_resource_isa(packet.node->op));
  }
  ASSERT_EQ(leaf.object.section_count, 1u);
  ASSERT_EQ(leaf.object.symbol_count, 1u);
  EXPECT_EQ(leaf.object.symbols[0].size, kExpected.size());
  const loom_native_section_contribution_t* section = &leaf.object.sections[0];
  ASSERT_EQ(section->contents.data_length, kExpected.size());
  EXPECT_EQ(0,
            memcmp(section->contents.data, kExpected.data(), kExpected.size()));

  ResetLeaf(&leaf);
}

TEST_F(Aie2pLeafObjectTest, ReturnFallsBackAfterAnOccupiedAluCycle) {
  CompiledLeaf leaf;
  IREE_ASSERT_OK(
      CompileSource("low.func.def target<amd.xdna.aie2p.core> @scalar_add(\n"
                    "    %value: reg<aie2p.er>) -> (reg<aie2p.er>) asm {\n"
                    "  %sum = add %value, 1\n"
                    "  return %sum\n"
                    "}\n",
                    &leaf));

  ASSERT_EQ(leaf.frame.schedule.issue_group_count, 2u);
  EXPECT_EQ(leaf.frame.schedule.issue_groups[0].issue_cycle, 0u);
  EXPECT_EQ(leaf.frame.schedule.issue_groups[1].issue_cycle, 1u);
  ASSERT_EQ(leaf.plan.bundle_count, 8u);

  iree_host_size_t synthetic_nop_count = 0;
  iree_host_size_t structural_control_count = 0;
  for (iree_host_size_t i = 0; i < leaf.plan.slot_count; ++i) {
    const loom_aie2p_planned_slot_t* slot = &leaf.plan.slots[i];
    if (iree_any_bit_set(slot->flags,
                         LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP)) {
      ++synthetic_nop_count;
    }
    if (iree_any_bit_set(slot->flags,
                         LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL)) {
      ++structural_control_count;
      EXPECT_EQ(leaf.plan.bundles[2].slot_start, i);
    }
  }
  EXPECT_EQ(synthetic_nop_count, 6u);
  EXPECT_EQ(structural_control_count, 1u);
  EXPECT_EQ(leaf.object.section_count, 1u);
  EXPECT_EQ(leaf.object.symbol_count, 1u);

  ResetLeaf(&leaf);
}

TEST_F(Aie2pLeafObjectTest, MaterializesFixedRegisterCopiesAsScalarMoves) {
  CompiledLeaf leaf;
  IREE_ASSERT_OK(CompileSource(
      "low.func.def target<amd.xdna.aie2p.core> @insert_lane(\n"
      "    %vector: reg<aie2p.vec256 x2>, %index: reg<aie2p.er>,\n"
      "    %value: reg<aie2p.er>) -> (reg<aie2p.vec256 x2>) asm {\n"
      "  %fixed_index = copy %index {detached = true} : "
      "reg<aie2p.er> -> reg<aie2p.mr29_insert>\n"
      "  %result = vinsert.8.reg %vector, %fixed_index, %value\n"
      "  return %result\n"
      "}\n",
      &leaf));

  ASSERT_EQ(leaf.frame.allocation.materialized_copy_count, 1u);
  const loom_aie2p_instruction_id_t scalar_move =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOV_alu_mv_mv_mv_scl"));
  ASSERT_NE(scalar_move, LOOM_AIE2P_INSTRUCTION_ID_INVALID);

  iree_host_size_t structural_move_count = 0;
  for (iree_host_size_t i = 0; i < leaf.plan.slot_count; ++i) {
    const loom_aie2p_planned_slot_t* slot = &leaf.plan.slots[i];
    if (!iree_any_bit_set(slot->flags,
                          LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE)) {
      continue;
    }
    ++structural_move_count;
    EXPECT_EQ(slot->encoded_slot.slot, LOOM_AIE2P_SLOT_MV);
    ASSERT_NE(slot->scheduled_packet_index, LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE);
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&leaf.frame.schedule, slot->scheduled_packet_index);
    ASSERT_TRUE(loom_low_copy_isa(packet.node->op));

    std::array<loom_aie2p_instruction_id_t, 16> candidates;
    const iree_host_size_t candidate_count =
        loom_aie2p_encoding_query_instruction_candidates(
            slot->encoded_slot.slot, slot->encoded_slot.value,
            candidates.size(), candidates.data());
    ASSERT_LE(candidate_count, candidates.size());
    EXPECT_NE(std::find(candidates.begin(),
                        candidates.begin() + candidate_count, scalar_move),
              candidates.begin() + candidate_count);

    const loom_aie2p_planned_bundle_t* owning_bundle = nullptr;
    for (iree_host_size_t bundle_index = 0;
         bundle_index < leaf.plan.bundle_count; ++bundle_index) {
      const loom_aie2p_planned_bundle_t* bundle =
          &leaf.plan.bundles[bundle_index];
      if (i >= bundle->slot_start &&
          i < bundle->slot_start + bundle->slot_count) {
        owning_bundle = bundle;
        break;
      }
    }
    ASSERT_NE(owning_bundle, nullptr);
    EXPECT_EQ(owning_bundle->logical_issue_cycle, packet.node->issue_cycle);
  }
  EXPECT_EQ(structural_move_count, 1u);

  ResetLeaf(&leaf);
}

}  // namespace
}  // namespace loom
