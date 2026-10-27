// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/representation_projection.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/facts_builder.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/target_records.h"

namespace loom {
namespace {

class LowRepresentationProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("representation_projection"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    descriptors_ = loom_low_descriptor_registry_lookup(
        &registry_.registry, IREE_SV("test.low.core"));
    ASSERT_NE(descriptors_, nullptr);
    loom_target_facts_builder_initialize(
        &loom_test_target_fact_type,
        loom_test_target_bundles.values[LOOM_TEST_TARGET_KIND_LOW_CORE],
        &target_facts_);
    const iree_string_view_t classes[] = {IREE_SV("test.i32"),
                                          IREE_SV("test.f32")};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(classes); ++i) {
      uint16_t class_id = 0;
      ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
          descriptors_, classes[i], &class_id, nullptr));
      IREE_ASSERT_OK(loom_low_build_register_type(descriptors_, class_id, 1,
                                                  &argument_types_[i]));
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_func_like_t BuildFunction(iree_string_view_t name, bool declaration,
                                 uint32_t operation_count = 0) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t representation = LOOM_STRING_ID_INVALID;
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    IREE_CHECK_OK(loom_module_intern_string(module_, IREE_SV("test.low.core"),
                                            &representation));
    const loom_symbol_ref_t symbol = {0, symbol_id};
    const loom_type_t result_types[] = {argument_types_[1], argument_types_[0]};
    loom_op_t* op = nullptr;
    if (declaration) {
      IREE_CHECK_OK(loom_low_func_decl_build(
          &builder, 0, 0, 0, 0, 0, 0, 0, 0, 0, LOOM_STRING_ID_INVALID,
          LOOM_STRING_ID_INVALID, representation, loom_symbol_ref_null(), 0, {},
          {}, LOOM_STRING_ID_INVALID, {}, symbol, argument_types_, 2,
          result_types, 2, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
    } else {
      IREE_CHECK_OK(loom_low_func_def_build(
          &builder, 0, 0, 0, 0, 0, 0, 0, 0, representation,
          loom_symbol_ref_null(), 0, {}, {}, LOOM_STRING_ID_INVALID, {}, symbol,
          argument_types_, 2, result_types, 2, nullptr, 0, nullptr, 0,
          LOOM_LOCATION_UNKNOWN, &op));
      loom_region_t* body = loom_low_func_def_body(op);
      loom_builder_initialize(module_, &module_->arena,
                              loom_region_entry_block(body), &builder);
      loom_builder_enter_region(&builder, op, body);
      uint16_t argument_count = 0;
      const loom_value_id_t* arguments = loom_func_like_arg_ids(
          loom_func_like_cast(module_, op), &argument_count);
      loom_value_id_t value = arguments[0];
      const uint32_t ordinal = loom_low_descriptor_set_lookup_descriptor(
          descriptors_, IREE_SV("test.add.i32"));
      const loom_low_descriptor_t* descriptor =
          loom_low_descriptor_set_descriptor_at(descriptors_, ordinal);
      for (uint32_t i = 0; i < operation_count; ++i) {
        const loom_value_id_t operands[] = {value, arguments[0]};
        loom_op_t* packet = nullptr;
        IREE_CHECK_OK(loom_low_build_resolved_descriptor_op(
            &builder, descriptors_, descriptor, 0, operands, 2, {},
            &argument_types_[0], 1, nullptr, 0, LOOM_LOCATION_UNKNOWN,
            &packet));
        value = loom_op_const_results(packet)[0];
      }
      const loom_value_id_t results[] = {arguments[1], value};
      loom_op_t* return_op = nullptr;
      IREE_CHECK_OK(loom_low_return_build(&builder, results, 2,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
    }
    return loom_func_like_cast(module_, op);
  }

  iree_host_size_t PlanAndCheck(loom_func_like_t function) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    const loom_low_representation_projection_plan_t* plan = nullptr;
    IREE_CHECK_OK(loom_low_plan_function_representation(
        module_, function, &target_facts_, &registry_.registry, {}, &arena,
        &plan));
    EXPECT_NE(plan, nullptr);
    if (plan == nullptr) {
      iree_arena_deinitialize(&arena);
      return 0;
    }
    EXPECT_EQ(loom_low_representation_projection_descriptors(plan),
              descriptors_);
    EXPECT_EQ(loom_low_representation_projection_target_facts(plan),
              &target_facts_);
    for (uint16_t i = 0; i < 2; ++i) {
      EXPECT_TRUE(loom_type_equal(
          loom_low_representation_projection_argument_type(plan, i),
          argument_types_[i]));
      EXPECT_TRUE(loom_type_equal(
          loom_low_representation_projection_result_type(plan, i),
          argument_types_[1 - i]));
    }
    const iree_host_size_t bytes = arena.used_allocation_size;
    bool changed = true;
    IREE_CHECK_OK(
        loom_low_apply_function_representation(module_, plan, &changed));
    EXPECT_FALSE(changed);
    iree_arena_deinitialize(&arena);
    return bytes;
  }

  // Storage shared by the module and temporary projection plans.
  iree_arena_block_pool_t block_pool_;
  // Low dialect operation context.
  loom_context_t context_;
  // Owns the valid definitions and declaration under projection.
  loom_module_t* module_ = nullptr;
  // Resolves the synthetic exact representation.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Borrowed exact descriptor set used by every function.
  const loom_low_descriptor_set_t* descriptors_ = nullptr;
  // Selected target preserving the authored representation.
  loom_target_facts_t target_facts_ = {};
  // Distinct input carriers; results deliberately reverse their order.
  loom_type_t argument_types_[2];
};

TEST_F(LowRepresentationProjectionTest, IdentityStorageDependsOnlyOnSignature) {
  const loom_func_like_t empty = BuildFunction(IREE_SV("empty"), false);
  const loom_func_like_t body = BuildFunction(IREE_SV("body"), false, 128);
  const loom_func_like_t declaration =
      BuildFunction(IREE_SV("declaration"), true);
  const iree_host_size_t signature_bytes = PlanAndCheck(empty);
  EXPECT_GT(signature_bytes, 0u);
  EXPECT_EQ(PlanAndCheck(body), signature_bytes);
  EXPECT_EQ(PlanAndCheck(declaration), signature_bytes);
}

}  // namespace
}  // namespace loom
