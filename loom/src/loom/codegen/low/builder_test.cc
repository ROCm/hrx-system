// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/builder.h"

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

TEST_F(LowBuilderTest, AccessSpacesSurviveCloningRebuildAndErasure) {
  const auto* descriptors = loom_test_low_core_descriptor_set();
  loom_builder_t builder = {};
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_type_t pointer_type = loom_type_none();
  loom_type_t value_type = loom_type_none();
  IREE_ASSERT_OK(loom_low_build_register_type(
      descriptors, TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR, 1, &pointer_type));
  IREE_ASSERT_OK(loom_low_build_register_type(
      descriptors, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4, &value_type));
  loom_value_id_t operands[2];
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder, loom_module_block(module_), pointer_type, &operands[0]));
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder, loom_module_block(module_), value_type, &operands[1]));
  constexpr auto space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
  for (const bool reads : {false, true}) {
    const auto ordinal =
        reads ? TEST_LOW_CORE_DESCRIPTOR_REF_TEST_LOAD_WORKGROUP_V4I32
              : TEST_LOW_CORE_DESCRIPTOR_REF_TEST_STORE_WORKGROUP_V4I32;
    loom_op_t* access = nullptr;
    IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
        &builder, descriptors, &descriptors->descriptors[ordinal],
        /*access_flags=*/0, operands, reads ? 1 : 2, {}, &value_type,
        /*result_count=*/reads ? 1 : 0, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &access));
    EXPECT_EQ(access->intrinsic_read_spaces, reads ? 1u << space : 0);
    EXPECT_EQ(access->intrinsic_write_spaces, reads ? 0 : 1u << space);
    auto* counts = reads ? module_->body->read_space_counts
                         : module_->body->write_space_counts;
    EXPECT_EQ(counts[space], 1u);

    loom_ir_remap_t remap = {};
    const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
    IREE_ASSERT_OK(loom_ir_remap_initialize(module_, module_, &module_->arena,
                                            &options, &remap));
    loom_op_t* clone = nullptr;
    IREE_ASSERT_OK(loom_ir_clone_op(&builder, access, &remap, &clone));
    EXPECT_EQ(clone->intrinsic_read_spaces, access->intrinsic_read_spaces);
    EXPECT_EQ(clone->intrinsic_write_spaces, access->intrinsic_write_spaces);
    EXPECT_EQ(counts[space], 2u);
    IREE_ASSERT_OK(loom_module_compute_uses(module_));
    EXPECT_EQ(counts[space], 2u);
    IREE_ASSERT_OK(loom_op_erase(module_, access));
    EXPECT_EQ(counts[space], 1u);
    IREE_ASSERT_OK(loom_op_erase(module_, clone));
    EXPECT_EQ(loom_region_read_memory_spaces(module_->body), 0);
    EXPECT_EQ(loom_region_written_memory_spaces(module_->body), 0);
  }
}

}  // namespace
}  // namespace loom
