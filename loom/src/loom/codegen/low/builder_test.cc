// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/builder.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class LowBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, /*archive=*/nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Block pool backing the test module arena.
  iree_arena_block_pool_t block_pool_;
  // Finalized context shared by the test module.
  loom_context_t context_;
  // Module owning recursively structural register types.
  loom_module_t* module_ = nullptr;
};

TEST_F(LowBuilderTest, BuildsTypedRegisterFromDescriptorClass) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_type_t value_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  loom_type_t register_type = loom_type_none();
  IREE_ASSERT_OK(loom_low_build_typed_register_type(
      module_, descriptor_set, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32,
      /*unit_count=*/4, value_type, &register_type));

  EXPECT_EQ(loom_low_register_type_descriptor_set_stable_id(register_type),
            descriptor_set->stable_id);
  EXPECT_EQ(loom_low_register_type_class_id(register_type),
            TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  EXPECT_EQ(loom_low_register_type_unit_count(register_type), 4u);
  const loom_type_t* register_value_type =
      loom_type_register_value_type(register_type);
  ASSERT_NE(register_value_type, nullptr);
  EXPECT_TRUE(loom_type_equal(*register_value_type, value_type));
}

TEST_F(LowBuilderTest, RejectsInvalidTypedRegisterCarrier) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_type_t value_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t register_type = value_type;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_build_typed_register_type(
          module_, descriptor_set, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32,
          /*unit_count=*/0, value_type, &register_type));
  EXPECT_TRUE(loom_type_equal(register_type, loom_type_none()));

  register_type = value_type;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_low_build_typed_register_type(
          module_, descriptor_set, descriptor_set->reg_class_count,
          /*unit_count=*/1, value_type, &register_type));
  EXPECT_TRUE(loom_type_equal(register_type, loom_type_none()));
}

TEST_F(LowBuilderTest, WriteSpaceSummarySurvivesCloningRebuildAndErasure) {
  // Specialize the test instruction's memory resource while preserving its
  // complete register, operand, and encoding contracts.
  loom_low_descriptor_set_t descriptors = *loom_test_low_core_descriptor_set();
  std::vector<loom_low_effect_t> effects(
      descriptors.effects, descriptors.effects + descriptors.effect_count);
  const loom_low_descriptor_t* descriptor =
      &descriptors.descriptors[TEST_LOW_CORE_DESCRIPTOR_REF_TEST_STORE_V4I32];
  const auto* layout = loom_low_descriptor_layout(&descriptors, descriptor);
  for (uint16_t i = 0; i < layout->effect_count; ++i) {
    auto& effect = effects[layout->effect_start + i];
    if (effect.kind == LOOM_LOW_EFFECT_KIND_WRITE) {
      effect.memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    }
  }
  descriptors.effects = effects.data();

  loom_builder_t builder = {};
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_type_t pointer_type = loom_type_none();
  loom_type_t value_type = loom_type_none();
  IREE_ASSERT_OK(loom_low_build_register_type(
      &descriptors, TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR, 1, &pointer_type));
  IREE_ASSERT_OK(loom_low_build_register_type(
      &descriptors, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4, &value_type));
  loom_value_id_t operands[2];
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder, loom_module_block(module_), pointer_type, &operands[0]));
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder, loom_module_block(module_), value_type, &operands[1]));
  loom_op_t* store = nullptr;
  IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
      &builder, &descriptors, descriptor, /*access_flags=*/0, operands,
      IREE_ARRAYSIZE(operands), {}, /*result_types=*/nullptr,
      /*result_count=*/0, /*tied_results=*/nullptr, /*tied_result_count=*/0,
      LOOM_LOCATION_UNKNOWN, &store));
  constexpr auto space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
  EXPECT_EQ(store->intrinsic_write_spaces, 1u << space);
  EXPECT_EQ(module_->body->write_space_counts[space], 1u);
  EXPECT_EQ(loom_region_written_memory_spaces(module_->body), 1u << space);

  loom_ir_remap_t remap = {};
  const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
  IREE_ASSERT_OK(loom_ir_remap_initialize(module_, module_, &module_->arena,
                                          &options, &remap));
  loom_op_t* clone = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_op(&builder, store, &remap, &clone));
  EXPECT_EQ(clone->intrinsic_write_spaces, 1u << space);
  EXPECT_EQ(module_->body->write_space_counts[space], 2u);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  EXPECT_EQ(module_->body->write_space_counts[space], 2u);
  IREE_ASSERT_OK(loom_op_erase(module_, store));
  EXPECT_EQ(module_->body->write_space_counts[space], 1u);
  EXPECT_EQ(loom_region_written_memory_spaces(module_->body), 1u << space);
  IREE_ASSERT_OK(loom_op_erase(module_, clone));
  EXPECT_EQ(loom_region_written_memory_spaces(module_->body), 0);
}

TEST_F(LowBuilderTest, SharesPreparedAttributesAcrossInstructions) {
  const auto* descriptors = loom_test_low_core_descriptor_set();
  const auto* descriptor =
      &descriptors->descriptors
           [TEST_LOW_CORE_DESCRIPTOR_REF_TEST_STATE_ASSIGN_I32_IMMEDIATE];
  loom_builder_t builder = {};
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t value_name = 0;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("i32_value"), &value_name));
  const loom_named_attr_t entry = {
      .name_id = value_name,
      .value = loom_attr_i64(7),
  };
  loom_attribute_t attributes = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_, loom_make_named_attr_slice(&entry, 1), &attributes));
  loom_op_t* instructions[2] = {};
  for (auto& instruction : instructions) {
    IREE_ASSERT_OK(loom_low_build_prepared_descriptor_op(
        &builder, descriptors, descriptor, /*access_flags=*/0, nullptr, 0,
        attributes, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &instruction));
    EXPECT_EQ(loom_low_op_attrs(instruction).entries, attributes.dict_entries);
  }
  IREE_ASSERT_OK(loom_op_erase(module_, instructions[0]));
  EXPECT_EQ(loom_low_op_attrs(instructions[1]).entries[0].value.i64, 7);
}

}  // namespace
}  // namespace loom
