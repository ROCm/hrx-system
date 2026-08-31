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
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

namespace loom {
namespace {

struct CompiledLeaf {
  loom_module_t* module = nullptr;
  loom_low_emission_frame_t frame = {};
  loom_aie2p_bundle_plan_t plan = {};
  loom_aie2p_leaf_contribution_t contribution = {};
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
    frame_options.schedule_structural_models =
        loom_aie2p_low_structural_schedule_models();
    frame_options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
        out_leaf->module, function_op, &frame_options, &planning_arena_,
        &out_leaf->frame));
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_build(
        &out_leaf->frame, &planning_arena_, &out_leaf->plan));
    return loom_aie2p_leaf_object_emit(&out_leaf->plan, &object_arena_,
                                       &out_leaf->contribution);
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
        "low.func.def target<amd.xdna.aie2p.core> @vector_add_memory("
        "%extent: reg<aie2p.er>) asm {\n"
        "  %lhs_ptr = resource<native_pointer> extent(%extent) {index = 0, "
        "source_type = buffer} : reg<aie2p.ep>\n"
        "  %rhs_ptr = resource<native_pointer> {index = 1, source_type = "
        "buffer, extent = 1024, cache_swizzle_stride = 64} : reg<aie2p.ep>\n"
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

TEST_F(Aie2pLeafObjectTest, LowFunctionsEmitOptimizedVectorLeaves) {
  // RET advances over the five-bundle delay window so the final store occupies
  // its last delay slot. The independently retained physical sequence is 32
  // bytes because it leaves one trailing NOP after the store; Low needs 30.
  struct TestCase {
    std::string_view vector_shape;
    std::string_view mnemonic;
    std::array<uint8_t, 30> expected;
  };
  const TestCase test_cases[] = {
      {
          "i32x16",
          "vadd.32",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x00, 0x00,
           0x78, 0x2d, 0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a},
      },
      {
          "i16x32",
          "vadd.16",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x00, 0x00,
           0x78, 0x1d, 0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a},
      },
      {
          "i8x64",
          "vadd.8",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x00, 0x00,
           0x78, 0x0d, 0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a},
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
    ASSERT_EQ(leaf.plan.bundle_count, 10u);
    ASSERT_EQ(leaf.plan.slot_count, 11u);
    ASSERT_EQ(leaf.plan.encoded_byte_length, 30u);
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
        EXPECT_EQ(leaf.plan.bundles[4].slot_start, i);
        EXPECT_NE(slot->scheduled_packet_index,
                  LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE);
      }
    }
    EXPECT_EQ(synthetic_nop_count, 6u);
    EXPECT_EQ(structural_control_count, 1u);

    ASSERT_EQ(leaf.contribution.object.section_count, 1u);
    ASSERT_EQ(leaf.contribution.object.symbol_count, 1u);
    ASSERT_EQ(leaf.contribution.object.fixup_count, 0u);
    const loom_native_section_contribution_t* section =
        &leaf.contribution.object.sections[0];
    EXPECT_TRUE(iree_string_view_equal(section->section_name,
                                       IREE_SV(".text.vector_add")));
    EXPECT_EQ(section->contribution_alignment, 16u);
    EXPECT_EQ(section->section_type, LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS);
    EXPECT_EQ(section->section_flags,
              LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                  LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR);
    EXPECT_TRUE(iree_string_view_equal(leaf.contribution.object.symbols[0].name,
                                       IREE_SV("vector_add")));
    EXPECT_EQ(leaf.contribution.object.symbols[0].size, 30u);

    const loom_aie2p_leaf_realization_t& realization =
        leaf.contribution.realization;
    EXPECT_TRUE(iree_string_view_equal(realization.target_identity,
                                       LOOM_AIE2P_LEAF_TARGET_IDENTITY));
    EXPECT_TRUE(iree_string_view_equal(realization.abi_identity,
                                       LOOM_AIE2P_LEAF_ABI_IDENTITY));
    EXPECT_EQ(realization.entry_symbol_index, 0u);
    EXPECT_EQ(realization.elf_machine, LOOM_XDNA_ELF_MACHINE_AIE);
    EXPECT_EQ(realization.target_generation, LOOM_XDNA_TARGET_GENERATION_AIE2P);
    EXPECT_EQ(realization.elf_flags, LOOM_XDNA_ELF_AIE2P_FLAGS);
    EXPECT_EQ(realization.capability_flags, 0u);
    EXPECT_EQ(realization.code.byte_length, 30u);
    EXPECT_EQ(realization.code.minimum_alignment, 16u);
    EXPECT_EQ(realization.read_only_data.byte_length, 0u);
    EXPECT_EQ(realization.initialized_data.byte_length, 0u);
    EXPECT_EQ(realization.zero_fill.byte_length, 0u);
    EXPECT_EQ(realization.stack.byte_length, 0u);
    EXPECT_EQ(realization.scratch.byte_length, 0u);
    EXPECT_EQ(realization.private_storage.byte_length, 0u);
    EXPECT_EQ(realization.workgroup_storage.byte_length, 0u);
    EXPECT_EQ(realization.spill.byte_length, 0u);
    EXPECT_EQ(realization.resource_import_count, 0u);

    loom_module_free(leaf.module);
    leaf.module = nullptr;
    ASSERT_EQ(section->contents.data_length, test_case.expected.size());
    for (iree_host_size_t i = 0; i < test_case.expected.size(); ++i) {
      EXPECT_EQ(section->contents.data[i], test_case.expected[i]) << i;
    }
    ResetLeaf(&leaf);
  }
}

TEST_F(Aie2pLeafObjectTest, ResourceImportsAnchorRegistersWithoutEmittingCode) {
  constexpr std::array<uint8_t, 30> kExpected = {
      0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x00, 0x00,
      0x78, 0x2d, 0x01, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a,
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

  ASSERT_EQ(leaf.plan.bundle_count, 10u);
  ASSERT_EQ(leaf.plan.slot_count, 11u);
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
  ASSERT_EQ(leaf.contribution.object.section_count, 1u);
  ASSERT_EQ(leaf.contribution.object.symbol_count, 1u);
  EXPECT_EQ(leaf.contribution.object.symbols[0].size, kExpected.size());
  const loom_native_section_contribution_t* section =
      &leaf.contribution.object.sections[0];
  ASSERT_EQ(section->contents.data_length, kExpected.size());
  for (iree_host_size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(section->contents.data[i], kExpected[i]) << i;
  }

  const loom_aie2p_leaf_realization_t& realization =
      leaf.contribution.realization;
  EXPECT_EQ(realization.capability_flags,
            LOOM_AIE2P_LEAF_CAPABILITY_FLAG_RESOURCE_IMPORTS);
  ASSERT_EQ(realization.resource_import_count, 3u);
  for (iree_host_size_t i = 0; i < realization.resource_import_count; ++i) {
    const loom_aie2p_leaf_resource_import_t& resource =
        realization.resource_imports[i];
    EXPECT_EQ(resource.index, i);
    EXPECT_EQ(resource.import_kind,
              LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER);
    EXPECT_EQ(resource.source_type_kind, LOOM_TYPE_BUFFER);
    EXPECT_EQ(resource.descriptor_register_class_id,
              AIE2P_CORE_REG_CLASS_ID_AIE2P_EP);
    EXPECT_EQ(resource.physical_register_count, 1u);
    if (i == 0) {
      EXPECT_NE(resource.extent_physical_register, UINT32_MAX);
      EXPECT_EQ(resource.extent_descriptor_register_class_id,
                AIE2P_CORE_REG_CLASS_ID_AIE2P_ER);
      EXPECT_EQ(resource.extent_physical_register_count, 1u);
      EXPECT_EQ(resource.flags, LOOM_AIE2P_LEAF_RESOURCE_FLAG_DYNAMIC_EXTENT);
    } else {
      EXPECT_EQ(resource.extent_physical_register, UINT32_MAX);
      EXPECT_EQ(resource.extent_descriptor_register_class_id, 0u);
      EXPECT_EQ(resource.extent_physical_register_count, 0u);
    }
    if (i == 1) {
      EXPECT_EQ(resource.flags,
                LOOM_AIE2P_LEAF_RESOURCE_FLAG_STATIC_EXTENT |
                    LOOM_AIE2P_LEAF_RESOURCE_FLAG_CACHE_SWIZZLE_STRIDE);
      EXPECT_EQ(resource.extent, 1024u);
      EXPECT_EQ(resource.cache_swizzle_stride, 64u);
    } else {
      EXPECT_EQ(resource.extent, 0u);
      EXPECT_EQ(resource.cache_swizzle_stride, 0u);
    }
  }
  EXPECT_NE(realization.resource_imports[0].physical_register,
            realization.resource_imports[1].physical_register);
  EXPECT_NE(realization.resource_imports[0].physical_register,
            realization.resource_imports[2].physical_register);
  EXPECT_NE(realization.resource_imports[1].physical_register,
            realization.resource_imports[2].physical_register);

  ResetLeaf(&leaf);
}

TEST_F(Aie2pLeafObjectTest, RetainsExactFunctionStorageRequirements) {
  CompiledLeaf leaf;
  IREE_ASSERT_OK(CompileSource(
      "low.func.def target<amd.xdna.aie2p.core> @local_storage() asm {\n"
      "  %stack = storage {byte_alignment = 4, byte_length = 8} "
      ": low.storage<stack>\n"
      "  %scratch = storage {byte_alignment = 16, byte_length = "
      "17} : low.storage<scratch>\n"
      "  %private0 = storage {byte_alignment = 4, byte_length = "
      "4} : low.storage<private>\n"
      "  %private1 = storage {byte_alignment = 16, byte_length = "
      "8} : low.storage<private>\n"
      "  %workgroup = storage {byte_alignment = 32, byte_length "
      "= 64} : low.storage<workgroup>\n"
      "  return\n"
      "}\n",
      &leaf));

  const loom_aie2p_leaf_realization_t& realization =
      leaf.contribution.realization;
  EXPECT_EQ(realization.capability_flags,
            LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE);
  EXPECT_EQ(realization.stack.byte_length, 8u);
  EXPECT_EQ(realization.stack.minimum_alignment, 4u);
  EXPECT_EQ(realization.scratch.byte_length, 17u);
  EXPECT_EQ(realization.scratch.minimum_alignment, 16u);
  EXPECT_EQ(realization.private_storage.byte_length, 24u);
  EXPECT_EQ(realization.private_storage.minimum_alignment, 16u);
  EXPECT_EQ(realization.workgroup_storage.byte_length, 64u);
  EXPECT_EQ(realization.workgroup_storage.minimum_alignment, 32u);
  EXPECT_EQ(realization.spill.byte_length, 0u);
  EXPECT_EQ(realization.spill.minimum_alignment, 0u);
  ASSERT_EQ(realization.storage_domain_count, 4u);
  ASSERT_EQ(leaf.contribution.object.section_count, 5u);
  ASSERT_EQ(leaf.contribution.object.symbol_count, 5u);
  for (iree_host_size_t i = 0; i < realization.storage_domain_count; ++i) {
    const loom_aie2p_leaf_storage_domain_t& domain =
        realization.storage_domains[i];
    EXPECT_EQ(domain.storage_space, i);
    EXPECT_EQ(domain.section_contribution_index, i + 1u);
    EXPECT_EQ(domain.symbol_index, i + 1u);
    const loom_native_section_contribution_t& section =
        leaf.contribution.object.sections[i + 1u];
    EXPECT_EQ(section.section_type, LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS);
    EXPECT_EQ(section.section_flags, LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                                         LOOM_NATIVE_ELF_SECTION_FLAG_WRITE);
    EXPECT_EQ(section.contents.data_length, 0u);
    EXPECT_GT(section.zero_fill_length, 0u);
    const loom_native_object_symbol_t& symbol =
        leaf.contribution.object.symbols[i + 1u];
    EXPECT_EQ(symbol.binding, LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL);
    EXPECT_EQ(symbol.visibility, LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN);
    EXPECT_EQ(symbol.kind, LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA);
  }

  ResetLeaf(&leaf);
}

TEST_F(Aie2pLeafObjectTest,
       RetainsAndRelocatesStructuralLocalStorageAddresses) {
  CompiledLeaf leaf;
  IREE_ASSERT_OK(CompileSource(
      "low.func.def target<amd.xdna.aie2p.core> @local_address() asm {\n"
      "  %padding = storage {byte_alignment = 16, byte_length = 32} : "
      "low.storage<workgroup>\n"
      "  %storage = storage {byte_alignment = 64, byte_length = 256} : "
      "low.storage<workgroup>\n"
      "  %view = storage_view %storage {offset = 64, byte_length = 128} : "
      "low.storage<workgroup> -> low.storage<workgroup>\n"
      "  %address = storage_address %view {offset = 16} : "
      "low.storage<workgroup> -> reg<aie2p.ep>\n"
      "  %value = vlda.512.i32x16 %address, 0\n"
      "  return\n"
      "}\n",
      &leaf));

  const loom_low_packet_view_t* address_packet = nullptr;
  loom_low_packet_view_t address_packet_storage = {};
  loom_low_packet_view_t load_packet = {};
  for (iree_host_size_t i = 0; i < loom_low_packet_count(&leaf.frame.schedule);
       ++i) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&leaf.frame.schedule, i);
    if (loom_low_storage_address_isa(packet.node->op)) {
      address_packet_storage = packet;
      address_packet = &address_packet_storage;
    } else if (loom_low_op_isa(packet.node->op)) {
      load_packet = packet;
    }
  }
  ASSERT_NE(address_packet, nullptr);
  EXPECT_EQ(address_packet->node->descriptor, nullptr);
  EXPECT_EQ(address_packet->node->source_descriptor, nullptr);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const loom_low_descriptor_t* materialize_descriptor =
      &descriptor_set->descriptors
           [AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32];
  const loom_low_descriptor_view_t* materialize_view =
      loom_low_descriptor_set_descriptor_view(descriptor_set,
                                              materialize_descriptor);
  ASSERT_NE(address_packet->node->schedule_class, nullptr);
  EXPECT_EQ(
      address_packet->node->schedule_class,
      &descriptor_set->schedule_classes[materialize_view->schedule_class_id]);
  ASSERT_NE(load_packet.node, nullptr);
  EXPECT_GT(load_packet.node->issue_cycle, address_packet->node->issue_cycle);

  ASSERT_EQ(leaf.plan.storage_fixup_count, 1u);
  const loom_aie2p_planned_storage_fixup_t& planned_fixup =
      leaf.plan.storage_fixups[0];
  EXPECT_EQ(planned_fixup.storage_space, LOOM_STORAGE_SPACE_WORKGROUP);
  EXPECT_EQ(planned_fixup.byte_offset, 144u);
  ASSERT_LT(planned_fixup.bundle_index, leaf.plan.bundle_count);
  const loom_aie2p_planned_bundle_t& address_bundle =
      leaf.plan.bundles[planned_fixup.bundle_index];
  iree_host_size_t storage_address_slot_count = 0;
  for (uint8_t i = 0; i < address_bundle.slot_count; ++i) {
    const loom_aie2p_planned_slot_t& slot =
        leaf.plan.slots[address_bundle.slot_start + i];
    if (iree_any_bit_set(
            slot.flags,
            LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_STORAGE_ADDRESS)) {
      ++storage_address_slot_count;
      EXPECT_EQ(slot.scheduled_packet_index, address_packet->packet_index);
    }
  }
  EXPECT_EQ(storage_address_slot_count, 1u);

  const loom_native_object_contribution_t& object = leaf.contribution.object;
  ASSERT_EQ(object.section_count, 2u);
  ASSERT_EQ(object.symbol_count, 2u);
  ASSERT_EQ(object.fixup_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(object.sections[1].section_name,
                             IREE_SV(".storage.local_address.workgroup")));
  EXPECT_EQ(object.sections[1].section_type,
            LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS);
  EXPECT_EQ(object.sections[1].zero_fill_length, 320u);
  EXPECT_EQ(object.sections[1].contribution_alignment, 64u);
  EXPECT_EQ(object.symbols[1].size, 320u);
  EXPECT_EQ(object.fixups[0].section_contribution_index, 0u);
  EXPECT_EQ(object.fixups[0].section_offset, address_bundle.byte_offset);
  EXPECT_EQ(object.fixups[0].relocation_kind,
            LOOM_AIE2P_NATIVE_RELOCATION_KIND_LOCAL_ADDRESS_ABSOLUTE);
  EXPECT_EQ(object.fixups[0].target_symbol_index, 1u);
  EXPECT_EQ(object.fixups[0].addend, 144);
  EXPECT_EQ(leaf.contribution.realization.capability_flags,
            LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS |
                LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE);

  loom_native_section_contribution_assembly_t assembly = {};
  IREE_ASSERT_OK(loom_native_assemble_section_contributions(
      object.sections, object.section_count, &assembly, &object_arena_));
  ASSERT_EQ(assembly.section_count, 2u);
  assembly.sections[0].address = 0;
  assembly.sections[1].address = 0x70000;
  IREE_ASSERT_OK(loom_aie2p_native_object_apply_fixups(&object, &assembly,
                                                       &object_arena_));
  ASSERT_LE(address_bundle.byte_offset + 6u,
            assembly.sections[0].contents.data_length);
  // Pinned Peano independently encodes `movxm p0, #458896` to these bytes.
  constexpr std::array<uint8_t, 6> kMovxmLocalAddress = {
      0x44, 0x20, 0xc1, 0x00, 0x07, 0x00,
  };
  for (iree_host_size_t i = 0; i < kMovxmLocalAddress.size(); ++i) {
    EXPECT_EQ(
        assembly.sections[0].contents.data[address_bundle.byte_offset + i],
        kMovxmLocalAddress[i])
        << i;
  }

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
  ASSERT_EQ(leaf.plan.bundle_count, 7u);

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
      EXPECT_EQ(leaf.plan.bundles[1].slot_start, i);
    }
  }
  EXPECT_EQ(synthetic_nop_count, 5u);
  EXPECT_EQ(structural_control_count, 1u);
  EXPECT_EQ(leaf.contribution.object.section_count, 1u);
  EXPECT_EQ(leaf.contribution.object.symbol_count, 1u);

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

TEST_F(Aie2pLeafObjectTest,
       PlansNeitherFallthroughConditionalAndBackwardBranches) {
  CompiledLeaf leaf;
  IREE_ASSERT_OK(CompileSource(
      "low.func.def target<amd.xdna.aie2p.core> @branch_diamond(\n"
      "    %condition: reg<aie2p.er>) asm {\n"
      "  low.cond_br %condition, ^then, ^else : reg<aie2p.er>\n"
      "^join:\n"
      "  return\n"
      "^then:\n"
      "  low.br ^join\n"
      "^else:\n"
      "  low.br ^join\n"
      "}\n",
      &leaf));

  ASSERT_EQ(leaf.plan.block_count, 4u);
  constexpr std::array<uint32_t, 4> kBlockByteOffsets = {0, 32, 48, 64};
  for (iree_host_size_t i = 0; i < kBlockByteOffsets.size(); ++i) {
    EXPECT_EQ(leaf.plan.block_byte_offsets[i], kBlockByteOffsets[i]);
  }
  EXPECT_EQ(leaf.plan.encoded_byte_length, 80u);
  ASSERT_EQ(leaf.plan.bundle_count, 31u);
  ASSERT_EQ(leaf.plan.branch_fixup_count, 4u);
  constexpr std::array<uint32_t, 4> kFixupBundleIndices = {0, 6, 19, 25};
  constexpr std::array<uint32_t, 4> kTargetBlockIndices = {2, 3, 1, 1};
  constexpr std::array<uint64_t, 4> kFixupByteOffsets = {0, 16, 48, 64};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const std::array<loom_aie2p_instruction_id_t, 4> expected_instructions = {
      descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO]
          .encoding_id,
      descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT]
          .encoding_id,
      descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT]
          .encoding_id,
      descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT]
          .encoding_id,
  };
  for (iree_host_size_t i = 0; i < leaf.plan.branch_fixup_count; ++i) {
    const loom_aie2p_planned_branch_fixup_t& fixup = leaf.plan.branch_fixups[i];
    EXPECT_EQ(fixup.bundle_index, kFixupBundleIndices[i]);
    EXPECT_EQ(fixup.target_block_index, kTargetBlockIndices[i]);
    const loom_aie2p_planned_bundle_t& bundle =
        leaf.plan.bundles[fixup.bundle_index];
    EXPECT_EQ(bundle.byte_offset, kFixupByteOffsets[i]);
    ASSERT_EQ(bundle.slot_count, 1u);
    const loom_aie2p_planned_slot_t& slot = leaf.plan.slots[bundle.slot_start];
    std::array<loom_aie2p_instruction_id_t, 16> candidates;
    const iree_host_size_t candidate_count =
        loom_aie2p_encoding_query_instruction_candidates(
            slot.encoded_slot.slot, slot.encoded_slot.value, candidates.size(),
            candidates.data());
    ASSERT_LE(candidate_count, candidates.size());
    EXPECT_NE(
        std::find(candidates.begin(), candidates.begin() + candidate_count,
                  expected_instructions[i]),
        candidates.begin() + candidate_count);
  }

  const loom_native_object_contribution_t& object = leaf.contribution.object;
  ASSERT_EQ(object.fixup_count, 4u);
  EXPECT_EQ(leaf.contribution.realization.capability_flags,
            LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS);
  constexpr std::array<int64_t, 4> kTargetAddends = {48, 64, 32, 32};
  for (iree_host_size_t i = 0; i < object.fixup_count; ++i) {
    EXPECT_EQ(object.fixups[i].section_contribution_index, 0u);
    EXPECT_EQ(object.fixups[i].section_offset, kFixupByteOffsets[i]);
    EXPECT_EQ(object.fixups[i].relocation_kind,
              LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE);
    EXPECT_EQ(object.fixups[i].target_symbol_index, 0u);
    EXPECT_EQ(object.fixups[i].addend, kTargetAddends[i]);
  }

  ResetLeaf(&leaf);
}

TEST_F(Aie2pLeafObjectTest, SelectsZeroBranchForTrueFallthrough) {
  CompiledLeaf leaf;
  IREE_ASSERT_OK(CompileSource(
      "low.func.def target<amd.xdna.aie2p.core> @true_fallthrough(\n"
      "    %condition: reg<aie2p.er>) asm {\n"
      "  low.cond_br %condition, ^then, ^else : reg<aie2p.er>\n"
      "^then:\n"
      "  low.br ^join\n"
      "^else:\n"
      "  low.br ^join\n"
      "^join:\n"
      "  return\n"
      "}\n",
      &leaf));

  ASSERT_EQ(leaf.plan.block_count, 4u);
  constexpr std::array<uint32_t, 4> kBlockByteOffsets = {0, 16, 32, 32};
  for (iree_host_size_t i = 0; i < kBlockByteOffsets.size(); ++i) {
    EXPECT_EQ(leaf.plan.block_byte_offsets[i], kBlockByteOffsets[i]);
  }
  EXPECT_EQ(leaf.plan.encoded_byte_length, 46u);
  ASSERT_EQ(leaf.plan.branch_fixup_count, 2u);
  EXPECT_EQ(leaf.plan.branch_fixups[0].bundle_index, 0u);
  EXPECT_EQ(leaf.plan.branch_fixups[0].target_block_index, 2u);
  EXPECT_EQ(leaf.plan.branch_fixups[1].bundle_index, 6u);
  EXPECT_EQ(leaf.plan.branch_fixups[1].target_block_index, 3u);

  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const loom_aie2p_instruction_id_t zero_branch =
      descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_BRANCH_ZERO]
          .encoding_id;
  const loom_aie2p_planned_bundle_t& first_bundle = leaf.plan.bundles[0];
  const loom_aie2p_planned_slot_t& first_slot =
      leaf.plan.slots[first_bundle.slot_start];
  std::array<loom_aie2p_instruction_id_t, 16> candidates;
  const iree_host_size_t candidate_count =
      loom_aie2p_encoding_query_instruction_candidates(
          first_slot.encoded_slot.slot, first_slot.encoded_slot.value,
          candidates.size(), candidates.data());
  ASSERT_LE(candidate_count, candidates.size());
  EXPECT_NE(std::find(candidates.begin(), candidates.begin() + candidate_count,
                      zero_branch),
            candidates.begin() + candidate_count);
  ASSERT_EQ(leaf.contribution.object.fixup_count, 2u);
  EXPECT_EQ(leaf.contribution.object.fixups[0].section_offset, 0u);
  EXPECT_EQ(leaf.contribution.object.fixups[0].addend, 32);
  EXPECT_EQ(leaf.contribution.object.fixups[1].section_offset, 16u);
  EXPECT_EQ(leaf.contribution.object.fixups[1].addend, 32);

  ResetLeaf(&leaf);
}

TEST_F(Aie2pLeafObjectTest, MaterializesLoopEdgeSwapBeforeBackwardBranch) {
  CompiledLeaf leaf;
  IREE_ASSERT_OK(CompileSource(
      "low.func.def target<amd.xdna.aie2p.core> @loop_swap(\n"
      "    %condition: reg<aie2p.er>, %first: reg<aie2p.er>,\n"
      "    %second: reg<aie2p.er>) -> (reg<aie2p.er>, reg<aie2p.er>) asm {\n"
      "  low.br ^loop(%first: reg<aie2p.er>, %second: reg<aie2p.er>)\n"
      "^loop(%lhs: reg<aie2p.er>, %rhs: reg<aie2p.er>):\n"
      "  low.cond_br %condition, ^body, ^exit : reg<aie2p.er>\n"
      "^body:\n"
      "  low.br ^loop(%rhs: reg<aie2p.er>, %lhs: reg<aie2p.er>)\n"
      "^exit:\n"
      "  return %lhs, %rhs : reg<aie2p.er>, reg<aie2p.er>\n"
      "}\n",
      &leaf));

  const loom_low_schedule_block_t& body_block = leaf.frame.schedule.blocks[2];
  ASSERT_GT(body_block.scheduled_node_count, 0u);
  const loom_low_packet_view_t body_terminator =
      loom_low_packet_at_block_ordinal(&leaf.frame.schedule, 2,
                                       body_block.scheduled_node_count - 1u);
  ASSERT_TRUE(loom_low_br_isa(body_terminator.node->op));
  const loom_low_allocation_edge_copy_group_t* edge_copy_group =
      loom_low_allocation_find_edge_copy_group_by_source_ordinal(
          &leaf.frame.allocation, body_terminator.node->source_ordinal);
  ASSERT_NE(edge_copy_group, nullptr);
  ASSERT_GT(edge_copy_group->move_group.moves.count, 0u);

  const loom_aie2p_planned_branch_fixup_t* backward_fixup = nullptr;
  for (iree_host_size_t i = 0; i < leaf.plan.branch_fixup_count; ++i) {
    if (leaf.plan.branch_fixups[i].target_block_index == 1u) {
      backward_fixup = &leaf.plan.branch_fixups[i];
      break;
    }
  }
  ASSERT_NE(backward_fixup, nullptr);
  const loom_aie2p_planned_bundle_t& branch_bundle =
      leaf.plan.bundles[backward_fixup->bundle_index];
  EXPECT_EQ(branch_bundle.block_index, 2u);

  iree_host_size_t structural_move_count = 0;
  iree_host_size_t final_move_bundle_index = 0;
  for (iree_host_size_t bundle_index = 0;
       bundle_index < backward_fixup->bundle_index; ++bundle_index) {
    const loom_aie2p_planned_bundle_t& bundle = leaf.plan.bundles[bundle_index];
    if (bundle.block_index != 2u) continue;
    for (uint8_t i = 0; i < bundle.slot_count; ++i) {
      const loom_aie2p_planned_slot_t& slot =
          leaf.plan.slots[bundle.slot_start + i];
      if (!iree_any_bit_set(slot.flags,
                            LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE)) {
        continue;
      }
      ++structural_move_count;
      final_move_bundle_index = bundle_index;
      EXPECT_EQ(slot.scheduled_packet_index, body_terminator.packet_index);
    }
  }
  EXPECT_EQ(structural_move_count, edge_copy_group->move_group.moves.count);
  EXPECT_EQ(final_move_bundle_index + 1u, backward_fixup->bundle_index);

  ResetLeaf(&leaf);
}

}  // namespace
}  // namespace loom
