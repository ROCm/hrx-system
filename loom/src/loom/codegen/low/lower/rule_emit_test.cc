// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_emit.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/bindings.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/low/storage.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower/registry.h"
#include "loom/target/test/target_records.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

loom_low_lower_emit_ref_storage_t PackEmitRef(
    loom_low_lower_emit_ref_t emit_ref, uint8_t descriptor_bit_count) {
  const uint32_t encoded_descriptor_ref =
      emit_ref.descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE
          ? 0u
          : (uint32_t)emit_ref.descriptor_ref + 1u;
  const uint32_t packed_ref =
      encoded_descriptor_ref |
      ((uint32_t)emit_ref.emit_index << descriptor_bit_count);
  return {
      .bytes =
          {
              (uint8_t)packed_ref,
              (uint8_t)(packed_ref >> 8),
              (uint8_t)(packed_ref >> 16),
          },
  };
}

class LowLowerRuleEmitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("lower_rule_emit_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    target_facts_.fact_type = &loom_test_target_fact_type;
    target_facts_.storage.bundle = *loom_test_target_bundles.values[1];
    BuildFunction();
    IREE_ASSERT_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
    fact_table_.context.target_facts = &target_facts_;
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function_));
    options_.target_facts = &target_facts_;
    options_.descriptor_registry = &descriptor_registry_.registry;
    options_.policy = loom_test_low_lower_policy();
    options_.fact_table = &fact_table_;
  }

  void TearDown() override {
    loom_low_lower_result_deinitialize(&result_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("emit_add"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
    const loom_type_t argument_types[] = {i32_type, i32_type};
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*linkage=*/0,
        /*retain=*/0,
        /*cc=*/0, /*effect_contract=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*source_signature=*/loom_attr_absent(),
        /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), symbol, argument_types,
        IREE_ARRAYSIZE(argument_types), &f32_type, 1, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function_op));
    function_ = loom_func_like_cast(module_, function_op);
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function_, &argument_count);
    ASSERT_EQ(argument_count, 2u);

    loom_builder_t body_builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &body_builder);
    body_builder.ip.parent_op = function_op;
    loom_op_t* add_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, /*overflow_flags=*/0,
                                          arguments[0], arguments[1], i32_type,
                                          LOOM_LOCATION_UNKNOWN, &add_op));
    loom_op_t* bitcast_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_bitcast_build(
        &body_builder, loom_scalar_addi_result(add_op), i32_type, f32_type,
        LOOM_LOCATION_UNKNOWN, &bitcast_op));
    const loom_value_id_t result = loom_scalar_bitcast_result(bitcast_op);
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &result, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_target_facts_t target_facts_ = {};
  loom_value_fact_table_t fact_table_ = {};
  loom_low_lower_options_t options_ = {};
  loom_low_lower_result_t result_ = {};
};

TEST_F(LowLowerRuleEmitTest, EmitsDescriptorAndRegisterTransferPrograms) {
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  ASSERT_NE(result_.low_func_op, nullptr);

  const loom_func_like_t low_function =
      loom_func_like_cast(module_, result_.low_func_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(low_function, &argument_count);
  ASSERT_EQ(argument_count, 2u);

  loom_op_t* emitted_add = nullptr;
  loom_op_t* emitted_copy = nullptr;
  loom_op_t* emitted_moves[2] = {nullptr, nullptr};
  iree_host_size_t emitted_move_count = 0;
  loom_op_t* emitted_return = nullptr;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(
      loom_region_entry_block(loom_func_like_body(low_function)), op) {
    if (loom_low_op_isa(op) && loom_low_op_descriptor(op) ==
                                   TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32) {
      ASSERT_EQ(emitted_add, nullptr);
      emitted_add = op;
      continue;
    }
    switch (loom_low_register_transfer_kind(op)) {
      case LOOM_LOW_REGISTER_TRANSFER_COPY:
        ASSERT_EQ(emitted_copy, nullptr);
        emitted_copy = op;
        continue;
      case LOOM_LOW_REGISTER_TRANSFER_MOVE:
        ASSERT_LT(emitted_move_count, IREE_ARRAYSIZE(emitted_moves));
        emitted_moves[emitted_move_count++] = op;
        continue;
      case LOOM_LOW_REGISTER_TRANSFER_SLICE:
      case LOOM_LOW_REGISTER_TRANSFER_CONCAT:
      case LOOM_LOW_REGISTER_TRANSFER_NONE:
        break;
    }
    if (loom_low_return_isa(op)) {
      emitted_return = op;
    }
  }
  ASSERT_NE(emitted_add, nullptr);
  ASSERT_NE(emitted_copy, nullptr);
  ASSERT_EQ(emitted_move_count, IREE_ARRAYSIZE(emitted_moves));
  ASSERT_NE(emitted_return, nullptr);

  const loom_value_slice_t operands = loom_low_op_operands(emitted_add);
  ASSERT_EQ(operands.count, 2u);
  EXPECT_EQ(operands.values[0], arguments[0]);
  EXPECT_EQ(operands.values[1], arguments[1]);
  const loom_value_slice_t results = loom_low_op_results(emitted_add);
  ASSERT_EQ(results.count, 1u);
  EXPECT_EQ(loom_low_copy_source(emitted_copy), results.values[0]);
  EXPECT_EQ(loom_low_move_source(emitted_moves[0]),
            loom_low_copy_result(emitted_copy));
  const loom_type_t copy_result_type =
      loom_module_value_type(module_, loom_low_copy_result(emitted_copy));
  EXPECT_FALSE(loom_type_equal(
      copy_result_type, loom_module_value_type(module_, results.values[0])));
  EXPECT_TRUE(loom_type_equal(
      copy_result_type,
      loom_module_value_type(module_, loom_low_move_result(emitted_moves[0]))));
  EXPECT_EQ(loom_low_move_source(emitted_moves[1]),
            loom_low_move_result(emitted_moves[0]));
  const loom_value_slice_t return_values =
      loom_low_return_values(emitted_return);
  ASSERT_EQ(return_values.count, 1u);
  EXPECT_EQ(return_values.values[0], loom_low_move_result(emitted_moves[1]));
}

class LowLowerWideRuleEmitTest
    : public LowLowerRuleEmitTest,
      public ::testing::WithParamInterface<loom_low_lower_emit_kind_t> {};

TEST_P(LowLowerWideRuleEmitTest, PlansAndEmitsEveryOperandAndResult) {
  // Model a target instruction returning a value plus six state results. The
  // source operation still has one result; the recipe owns the state tuple.
  auto descriptor_set = *loom_low_descriptor_registry_descriptor_set_at(
      &descriptor_registry_.registry, 0);
  std::vector<loom_low_descriptor_t> descriptors(
      descriptor_set.descriptors,
      descriptor_set.descriptors + descriptor_set.descriptor_ordinal_count);
  std::vector<loom_low_descriptor_layout_t> layouts(
      descriptor_set.descriptor_layouts,
      descriptor_set.descriptor_layouts +
          descriptor_set.descriptor_layout_count);
  std::vector<uint16_t> operand_indices(
      descriptor_set.operand_indices,
      descriptor_set.operand_indices + descriptor_set.operand_index_count);
  std::vector<loom_low_operand_t> operand_rows(
      descriptor_set.operands,
      descriptor_set.operands + descriptor_set.operand_count);
  auto& descriptor = descriptors[TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32];
  const uint16_t result_operand = operand_indices[descriptor.operand_start];
  const uint16_t input_operand = operand_indices[descriptor.operand_start + 1];
  auto layout = layouts[descriptor.layout_index];
  constexpr uint16_t kOperandCount = LOOM_LOW_LOWER_EMIT_MAX_OPERANDS;
  constexpr uint16_t kResultCount = LOOM_LOW_LOWER_EMIT_MAX_RESULTS;
  descriptor.operand_start = static_cast<uint16_t>(operand_indices.size());
  descriptor.operand_count = kOperandCount + kResultCount;
  descriptor.result_count = kResultCount;
  descriptor.layout_index = static_cast<uint16_t>(layouts.size());
  for (uint16_t i = 0; i < descriptor.operand_count; ++i) {
    auto operand =
        operand_rows[i < kResultCount ? result_operand : input_operand];
    operand.source_value_index = i < kResultCount ? i : i - kResultCount;
    operand_indices.push_back(static_cast<uint16_t>(operand_rows.size()));
    operand_rows.push_back(operand);
  }
  layout.minimum_packet_operand_count = kOperandCount;
  layouts.push_back(layout);
  descriptor_set.descriptors = descriptors.data();
  descriptor_set.descriptor_layouts = layouts.data();
  descriptor_set.descriptor_layout_count = layouts.size();
  descriptor_set.operands = operand_rows.data();
  descriptor_set.operand_count = operand_rows.size();
  descriptor_set.operand_indices = operand_indices.data();
  descriptor_set.operand_index_count = operand_indices.size();
  descriptor_set.maximum_descriptor_operand_count = descriptor.operand_count;
  const loom_low_descriptor_set_t* descriptor_sets[] = {&descriptor_set};
  const loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = 1,
  };
  options_.descriptor_registry = &registry;

  auto policy = *options_.policy;
  ASSERT_EQ(policy.contract.rule_sets.count, 1u);
  auto rule_set = *policy.contract.rule_sets.values[0];
  std::vector<loom_low_lower_rule_t> rules(
      rule_set.rules, rule_set.rules + rule_set.rule_count);
  std::vector<loom_low_lower_emit_t> emits(
      rule_set.emits, rule_set.emits + rule_set.emit_count);
  std::vector<loom_low_lower_emit_ref_storage_t> emit_refs(
      rule_set.emit_refs, rule_set.emit_refs + rule_set.emit_ref_count);
  std::vector<loom_low_lower_value_ref_t> value_refs(
      rule_set.value_refs, rule_set.value_refs + rule_set.value_ref_count);
  std::vector<loom_low_lower_value_materializer_t> materializers(
      rule_set.materializers,
      rule_set.materializers + rule_set.materializer_count);
  materializers.push_back({
      .can_materialize =
          [](loom_low_lower_context_t*, const loom_op_t*, loom_value_id_t,
             bool* out_can_materialize) {
            *out_can_materialize = true;
            return iree_ok_status();
          },
      .result_type =
          [](const loom_low_lower_context_t* context, loom_value_id_t value,
             const void* plan) {
            EXPECT_EQ(*static_cast<const loom_value_id_t*>(plan), value);
            return loom_low_lower_value_binding_type(context, value);
          },
      .prepare =
          [](loom_low_lower_context_t* context, loom_value_id_t value,
             const void** out_plan) {
            loom_value_id_t* retained = nullptr;
            IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
                context, sizeof(*retained),
                reinterpret_cast<void**>(&retained)));
            *retained = value;
            *out_plan = retained;
            return iree_ok_status();
          },
      .emit =
          {
              .planned =
                  [](loom_low_lower_context_t* context, const loom_op_t*,
                     loom_value_id_t value, const void* plan,
                     loom_value_id_t* out_value) {
                    EXPECT_EQ(*static_cast<const loom_value_id_t*>(plan),
                              value);
                    *out_value = loom_low_lower_lookup_value(context, value);
                    return iree_ok_status();
                  },
          },
  });
  unsigned replaced_count = 0;
  for (auto& rule : rules) {
    if (rule.emit_count != 1) {
      continue;
    }
    auto emit_ref =
        loom_low_lower_rule_set_emit_ref_at(&rule_set, rule.action.emit_start);
    if (emit_ref.descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE ||
        rule_set.descriptor_refs[emit_ref.descriptor_ref].stable_id !=
            loom_low_descriptor_stable_id(&descriptor_set, &descriptor)) {
      continue;
    }
    auto emit = emits[emit_ref.emit_index];
    ASSERT_EQ(emit.operand_ref_count, 2u);
    ASSERT_EQ(emit.result_ref_count, 1u);
    const auto lhs = value_refs[emit.operand_ref_start];
    const auto rhs = value_refs[emit.operand_ref_start + 1];
    const auto result =
        value_refs[loom_low_lower_rule_emit_result_bind_ref_index(&emit, 0)];
    emit.kind = GetParam();
    emit.flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS |
                 LOOM_LOW_LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE;
    emit.operand_ref_start = static_cast<uint16_t>(value_refs.size());
    emit.operand_ref_count = kOperandCount;
    for (uint16_t i = 0; i < kOperandCount; ++i) {
      value_refs.push_back(i % 2 == 0 ? lhs : rhs);
    }
    value_refs.back().materializer_index =
        static_cast<uint16_t>(materializers.size());
    emit.copy_operand_mask = 1u << (kOperandCount - 1);
    emit.result_bind_ref_start = static_cast<uint16_t>(value_refs.size());
    emit.result_ref_count = kResultCount;
    value_refs.push_back(result);
    for (uint16_t i = 1; i < kResultCount; ++i) {
      value_refs.push_back({
          .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
          .index = static_cast<uint16_t>(i - 1),
      });
    }
    rule.temporary_count = kResultCount - 1;
    rule.action.emit_start = static_cast<uint16_t>(emit_refs.size());
    emit_ref.emit_index = static_cast<uint16_t>(emits.size());
    emits.push_back(emit);
    emit_refs.push_back(
        PackEmitRef(emit_ref, rule_set.emit_ref_descriptor_bit_count));
    ++replaced_count;
  }
  ASSERT_GT(replaced_count, 0u);
  rule_set.materializers = materializers.data();
  rule_set.materializer_count = materializers.size();
  rule_set.rules = rules.data();
  rule_set.emits = emits.data();
  rule_set.emit_count = emits.size();
  rule_set.emit_refs = emit_refs.data();
  rule_set.emit_ref_count = emit_refs.size();
  rule_set.value_refs = value_refs.data();
  rule_set.value_ref_count = value_refs.size();
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};
  policy.contract.rule_sets.values = rule_sets;
  options_.policy = &policy;

  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  const auto low_function = loom_func_like_cast(module_, result_.low_func_op);
  uint16_t argument_count = 0;
  const auto* arguments = loom_func_like_arg_ids(low_function, &argument_count);
  ASSERT_EQ(argument_count, 2u);
  unsigned emitted_count = 0;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(
      loom_region_entry_block(loom_func_like_body(low_function)), op) {
    if (!loom_low_op_isa(op) || loom_low_op_descriptor(op) !=
                                    TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32) {
      continue;
    }
    ++emitted_count;
    const auto operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count, kOperandCount);
    for (uint16_t i = 0; i + 1 < kOperandCount; ++i) {
      EXPECT_EQ(operands.values[i], arguments[i % 2]);
    }
    const auto* copy = loom_value_def_op(
        loom_module_value(module_, operands.values[kOperandCount - 1]));
    ASSERT_TRUE(loom_low_copy_isa(copy));
    EXPECT_EQ(loom_low_copy_source(copy), arguments[0]);
    const auto results = loom_low_op_results(op);
    ASSERT_EQ(results.count, kResultCount);
    for (uint16_t i = 0; i < kResultCount; ++i) {
      EXPECT_TRUE(
          loom_type_equal(loom_module_value_type(module_, results.values[i]),
                          loom_module_value_type(module_, arguments[0])));
    }
  }
  EXPECT_EQ(emitted_count, 1u);
}

INSTANTIATE_TEST_SUITE_P(
    DescriptorForms, LowLowerWideRuleEmitTest,
    ::testing::Values(LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
                      LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE));

}  // namespace
}  // namespace loom
