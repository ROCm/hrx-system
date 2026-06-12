// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_values.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/target/arch/spirv/records/target_records.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/emit/spirv/module_builder.h"

namespace loom {
namespace {

class SpirvModuleValueTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t callee = {/*.module_id=*/0,
                                      /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    body_ = loom_func_like_body(loom_func_like_cast(module_, func_op));
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &body_builder_);
    body_builder_.ip.parent_op = func_op;

    iree_arena_initialize(&block_pool_, &scratch_arena_);
    IREE_ASSERT_OK(loom_spirv_module_builder_initialize(
        &loom_spirv_low_target_bundle_vulkan1_3, iree_allocator_system(),
        &spirv_builder_));
  }

  void TearDown() override {
    loom_spirv_module_builder_deinitialize(&spirv_builder_);
    iree_arena_deinitialize(&scratch_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t BuildValue(int64_t value) {
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_test_constant_build(
        &body_builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_constant_result(op);
  }

  loom_spirv_value_type_t U32ValueType() {
    return (loom_spirv_value_type_t){
        /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
        /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_U32,
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = NULL;
  loom_region_t* body_ = NULL;
  loom_builder_t body_builder_;
  iree_arena_allocator_t scratch_arena_;
  loom_spirv_module_builder_t spirv_builder_ = {};
};

TEST_F(SpirvModuleValueTableTest, ReservesDefinesAndLooksUpValueRefs) {
  const loom_value_id_t value_id = BuildValue(1);
  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module_, body_, &scratch_arena_, &value_domain));

  loom_spirv_module_value_table_t table = {};
  IREE_ASSERT_OK(loom_spirv_module_value_table_initialize(&value_domain, &table,
                                                          &scratch_arena_));

  uint32_t first_result_id = 0;
  first_result_id = loom_spirv_module_value_table_reserve(
      &table, &spirv_builder_, value_id, /*type_id=*/7, U32ValueType());
  EXPECT_NE(first_result_id, 0u);
  EXPECT_TRUE(loom_spirv_module_value_table_exists(&table, value_id));

  uint32_t second_result_id = 0;
  second_result_id = loom_spirv_module_value_table_reserve(
      &table, &spirv_builder_, value_id, /*type_id=*/7, U32ValueType());
  EXPECT_EQ(second_result_id, first_result_id);

  loom_spirv_module_value_table_define(&table, value_id,
                                       (loom_spirv_module_value_ref_t){
                                           /*.id=*/first_result_id,
                                           /*.type_id=*/7,
                                           /*.value_type=*/U32ValueType(),
                                       });
  loom_spirv_module_value_ref_t value_ref =
      loom_spirv_module_value_table_lookup(&table, value_id);
  EXPECT_EQ(value_ref.id, first_result_id);
  EXPECT_EQ(value_ref.type_id, 7u);
  EXPECT_TRUE(
      loom_spirv_value_type_equal(value_ref.value_type, U32ValueType()));

  loom_local_value_domain_release(&value_domain);
}

}  // namespace
}  // namespace loom
