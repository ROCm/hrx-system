// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_call.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

constexpr uint64_t kDescriptorSetId = UINT64_C(0x123456789abcdef0);

class AmdgpuProgramCallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t dialect_op_count = 0;
    const loom_op_vtable_t* const* dialect_ops =
        loom_low_dialect_vtables(&dialect_op_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, dialect_ops, (uint16_t)dialect_op_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    InitializeDescriptorSet();
    BuildFunction();
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return loom_symbol_ref_t{.module_id = 0, .symbol_id = symbol_id};
  }

  void InitializeDescriptorSet() {
    register_classes_[LOOM_AMDGPU_REG_CLASS_ID_SGPR] = loom_low_reg_class_t{
        .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
        .alloc_unit_bits = 32,
        .allocatable_count = 128,
    };
    register_classes_[LOOM_AMDGPU_REG_CLASS_ID_VGPR] = loom_low_reg_class_t{
        .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
        .alloc_unit_bits = 32,
        .allocatable_count = 256,
    };
    register_classes_[LOOM_AMDGPU_REG_CLASS_ID_SCC] = loom_low_reg_class_t{
        .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                 LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS,
        .alloc_unit_bits = 32,
        .allocatable_count = 1,
        .physical_register_candidate_start = 0,
    };
    register_classes_[LOOM_AMDGPU_REG_CLASS_ID_EXEC] = loom_low_reg_class_t{
        .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                 LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS,
        .alloc_unit_bits = 64,
        .allocatable_count = 1,
        .physical_register_candidate_start = 1,
    };
    register_classes_[LOOM_AMDGPU_REG_CLASS_ID_VCC] = loom_low_reg_class_t{
        .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                 LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS,
        .alloc_unit_bits = 64,
        .allocatable_count = 1,
        .physical_register_candidate_start = 2,
    };
    descriptor_set_ = loom_low_descriptor_set_t{
        .stable_id = kDescriptorSetId,
        .reg_classes = register_classes_,
        .reg_class_count = IREE_ARRAYSIZE(register_classes_),
        .physical_register_candidate_ids = physical_register_candidates_,
        .physical_register_candidate_count =
            IREE_ARRAYSIZE(physical_register_candidates_),
    };
  }

  void BuildFunction() {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    function_ref_ = AddSymbol(IREE_SV("helper"));
    loom_string_id_t descriptor_set_name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder, IREE_SV("amdgpu.test.core"), &descriptor_set_name));
    const loom_type_t argument_types[] = {
        loom_low_register_type(kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                               2),
        loom_low_register_type(kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                               3),
    };
    const loom_type_t result_types[] = {
        loom_low_register_type(kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                               4),
        loom_low_register_type(kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                               1),
    };
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*effect_contract=*/0, /*inline_policy=*/0,
        /*allocation=*/0, /*schedule=*/0, descriptor_set_name,
        /*target=*/loom_symbol_ref_t{}, /*abi=*/0, loom_named_attr_slice_t{},
        loom_named_attr_slice_t{}, LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_t{}, function_ref_, argument_types,
        IREE_ARRAYSIZE(argument_types), result_types,
        IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
        /*tied_result_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op_));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  iree_arena_allocator_t scratch_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* function_op_ = nullptr;
  loom_symbol_ref_t function_ref_ = {};
  loom_low_reg_class_t register_classes_[5] = {};
  const uint16_t physical_register_candidates_[3] = {10, 11, 12};
  loom_low_descriptor_set_t descriptor_set_ = {};
};

TEST_F(AmdgpuProgramCallTest, PublishesCarrierAndClobberContracts) {
  loom_amdgpu_program_callable_t callable = {.function = function_ref_};
  bool prepared = false;
  IREE_ASSERT_OK(loom_amdgpu_program_callable_prepare_abi(
      module_, function_op_, &descriptor_set_, iree_diagnostic_emitter_t{},
      &arena_, &prepared, &callable));
  ASSERT_TRUE(prepared);
  ASSERT_EQ(callable.argument_count, 2u);
  ASSERT_EQ(callable.result_count, 2u);
  EXPECT_EQ(callable.argument_requirements.sgpr_count, 2u);
  EXPECT_EQ(callable.argument_requirements.vgpr_count, 3u);
  EXPECT_EQ(callable.result_requirements.sgpr_count, 1u);
  EXPECT_EQ(callable.result_requirements.vgpr_count, 4u);
  EXPECT_EQ(callable.locations[0].location_base,
            LOOM_AMDGPU_CALL_ABI_SGPR_BASE);
  EXPECT_EQ(callable.locations[1].location_base,
            LOOM_AMDGPU_CALL_ABI_VGPR_BASE);
  EXPECT_EQ(callable.locations[2].location_base,
            LOOM_AMDGPU_CALL_ABI_VGPR_BASE);
  EXPECT_EQ(callable.locations[3].location_base,
            LOOM_AMDGPU_CALL_ABI_SGPR_BASE);
  EXPECT_EQ(callable.transport_clobbers[0].count, 2u);
  EXPECT_EQ(callable.transport_clobbers[1].count, 3u);
  EXPECT_EQ(callable.transport_clobbers[2].count, 4u);
  EXPECT_EQ(callable.transport_clobbers[3].count, 1u);

  const uint32_t physical_extents[] = {8, 6, 0, 0, 0};
  callable.compilation.frame.allocation.physical_extents.ends_by_reg_class =
      physical_extents;
  callable.compilation.frame.allocation.physical_extents.count =
      IREE_ARRAYSIZE(physical_extents);
  IREE_ASSERT_OK(loom_amdgpu_program_callable_prepare_clobbers(
      &descriptor_set_, &arena_, &scratch_arena_, &callable));
  ASSERT_EQ(callable.clobber_count, 2u);
  EXPECT_EQ(callable.clobbers[0].register_class, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  EXPECT_EQ(callable.clobbers[0].location, 0u);
  EXPECT_EQ(callable.clobbers[0].count, 8u);
  EXPECT_EQ(callable.clobbers[1].register_class, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  EXPECT_EQ(callable.clobbers[1].location, 0u);
  EXPECT_EQ(callable.clobbers[1].count, 6u);

  loom_amdgpu_program_call_table_t table = {};
  IREE_ASSERT_OK(loom_amdgpu_program_call_table_initialize(
      &descriptor_set_, module_->symbols.count, &arena_, &scratch_arena_,
      &table));
  callable.call_state.counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD |
                                     LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS;
  callable.call_state.fixed_state_mask = LOOM_AMDGPU_CALL_FIXED_STATE_ALL;
  loom_amdgpu_program_call_table_publish(&table, &callable);
  const loom_low_call_contract_provider_t provider =
      loom_amdgpu_program_call_table_provider(&table);
  EXPECT_EQ(provider.query(provider.user_data, function_ref_),
            &callable.call_contract);
  const loom_low_call_clobber_list_t common =
      provider.query_common_clobbers(provider.user_data, &descriptor_set_);
  ASSERT_EQ(common.count, 3u);
  EXPECT_EQ(common.values[0].register_class, LOOM_AMDGPU_REG_CLASS_ID_SCC);
  EXPECT_EQ(common.values[0].location, 10u);
  EXPECT_EQ(common.values[1].register_class, LOOM_AMDGPU_REG_CLASS_ID_VCC);
  EXPECT_EQ(common.values[1].location, 12u);
  EXPECT_EQ(common.values[2].register_class, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  EXPECT_EQ(common.values[2].location, LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE);
  EXPECT_EQ(common.values[2].count, LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT);
  const loom_amdgpu_call_state_provider_t state_provider =
      loom_amdgpu_program_call_table_state_provider(&table);
  EXPECT_EQ(state_provider.query(state_provider.user_data, function_ref_),
            &callable.call_state);
  EXPECT_EQ(callable.call_state.fixed_state_mask,
            LOOM_AMDGPU_CALL_FIXED_STATE_ALL);
}

TEST_F(AmdgpuProgramCallTest, EmptySignatureHasNoCarrierPointers) {
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  const loom_symbol_ref_t void_ref = AddSymbol(IREE_SV("void_helper"));
  loom_string_id_t descriptor_set_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(
      &builder, IREE_SV("amdgpu.test.core"), &descriptor_set_name));
  loom_op_t* void_op = nullptr;
  IREE_ASSERT_OK(loom_low_func_def_build(
      &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
      /*cc=*/0, /*effect_contract=*/0, /*inline_policy=*/0,
      /*allocation=*/0, /*schedule=*/0, descriptor_set_name,
      /*target=*/loom_symbol_ref_t{}, /*abi=*/0, loom_named_attr_slice_t{},
      loom_named_attr_slice_t{}, LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_t{}, void_ref, /*arg_types=*/nullptr,
      /*arg_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
      /*tied_results=*/nullptr, /*tied_result_count=*/0,
      /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
      &void_op));

  loom_amdgpu_program_callable_t callable = {.function = void_ref};
  bool prepared = false;
  IREE_ASSERT_OK(loom_amdgpu_program_callable_prepare_abi(
      module_, void_op, &descriptor_set_, iree_diagnostic_emitter_t{}, &arena_,
      &prepared, &callable));
  ASSERT_TRUE(prepared);
  EXPECT_EQ(callable.locations, nullptr);
  EXPECT_EQ(callable.call_contract.arguments, nullptr);
  EXPECT_EQ(callable.call_contract.argument_count, 0u);
  EXPECT_EQ(callable.call_contract.results, nullptr);
  EXPECT_EQ(callable.call_contract.result_count, 0u);
}

}  // namespace
}  // namespace loom
