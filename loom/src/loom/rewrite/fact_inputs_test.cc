// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/fact_propagation.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

static bool CanMaterialize(loom_value_facts_t facts, loom_type_t type) {
  int64_t value = 0;
  return loom_type_is_scalar(type) &&
         loom_value_facts_as_exact_i64(facts, &value);
}

static iree_status_t Materialize(loom_builder_t* builder,
                                 loom_value_facts_t facts, loom_type_t type,
                                 loom_location_id_t location,
                                 loom_value_id_t* out_value) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "expected integer");
  }
  loom_op_t* constant = nullptr;
  IREE_RETURN_IF_ERROR(loom_test_constant_build(builder, loom_attr_i64(value),
                                                type, location, &constant));
  *out_value = loom_test_constant_result(constant);
  return iree_ok_status();
}

enum class MaterializerProbeMode { kIndexOnly, kVector };

struct MaterializerProbeState {
  // Capability policy exercised by the active test.
  MaterializerProbeMode mode;
  // Number of fact/type pairs inspected before mutation.
  uint32_t support_query_count;
  // Number of constants constructed by the selected materializer.
  uint32_t materialization_count;
};

static MaterializerProbeState materializer_probe;

static bool ProbeMaterializerSupport(loom_value_facts_t facts,
                                     loom_type_t type) {
  ++materializer_probe.support_query_count;
  if (!loom_value_facts_is_exact(facts)) {
    return false;
  }
  switch (materializer_probe.mode) {
    case MaterializerProbeMode::kIndexOnly:
      return loom_type_is_scalar(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX;
    case MaterializerProbeMode::kVector:
      return loom_type_is_vector(type);
  }
  return false;
}

static iree_status_t ProbeMaterialize(loom_builder_t* builder,
                                      loom_value_facts_t facts,
                                      loom_type_t type,
                                      loom_location_id_t location,
                                      loom_value_id_t* out_value) {
  ++materializer_probe.materialization_count;
  loom_op_t* constant = nullptr;
  if (loom_type_is_scalar(type) &&
      loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(facts.range_lo), type, location, &constant));
    *out_value = loom_index_constant_result(constant);
    return iree_ok_status();
  }
  if (loom_type_is_vector(type)) {
    const loom_scalar_type_t element_type = loom_type_element_type(type);
    loom_attribute_t value = loom_attr_i64(facts.range_lo);
    if (loom_scalar_type_is_float(element_type)) {
      double exact_value = 0.0;
      if (!loom_value_facts_as_exact_float(element_type, facts, &exact_value)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "expected exact floating-point facts");
      }
      value = loom_attr_f64(exact_value);
    }
    IREE_RETURN_IF_ERROR(
        loom_test_constant_build(builder, value, type, location, &constant));
    *out_value = loom_test_constant_result(constant);
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "probe materializer does not support type kind %u",
                          (unsigned)loom_type_kind(type));
}

enum class EquationOwner {
  kOperation,
  kEntry,
  kRegion,
  kMovedRegion,
  kErasedRegion,
  kCFG,
  kCycle,
  kCounted,
  kCondition
};
enum class TypeMutation { kRemove, kAdd, kCreate, kErase, kReplace };

// The API mutation is the subject: a shaped value contributes a nonnegative
// domain to its same-block extent provider. Each provider has an existing
// equation owner that must add or withdraw that domain before folding a use.
class FactPublicationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    auto register_dialect = [&](loom_dialect_id_t dialect,
                                const loom_op_vtable_t* const* vtables) {
      IREE_ASSERT_OK(loom_context_register_dialect(
          &context_, dialect, vtables, static_cast<uint16_t>(count)));
    };
    register_dialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables(&count));
    register_dialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables(&count));
    register_dialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables(&count));
    register_dialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables(&count));
    register_dialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables(&count));
    register_dialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables(&count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("inverse_mutation"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_pass_value_fact_owner_initialize(&pool_, &owner_);
    loom_pass_value_fact_owner_initialize(&pool_, &oracle_owner_);
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_pass_value_fact_owner_deinitialize(&oracle_owner_);
    loom_pass_value_fact_owner_deinitialize(&owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Storage shared by the module and independently scoped analyses.
  iree_arena_block_pool_t pool_;
  // Mutation scratch and immutable input seed storage.
  iree_arena_allocator_t arena_;
  // Registered semantic operations used by the real fact inference.
  loom_context_t context_;
  // Owned module containing the small equation under mutation.
  loom_module_t* module_ = nullptr;
  // Incrementally maintained production fact scope.
  loom_pass_value_fact_owner_t owner_;
  // Independent fresh inference over the mutated IR.
  loom_pass_value_fact_owner_t oracle_owner_;
  // Mutation API under test, with real constant materialization enabled.
  loom_rewriter_t rewriter_;
};

class FactInputsTest : public FactPublicationTest,
                       public ::testing::WithParamInterface<
                           std::tuple<EquationOwner, TypeMutation>> {};

TEST_F(FactPublicationTest, ExactFoldPreflightsCompleteMaterializerFamily) {
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("preflight"), &name));
  uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));

  const loom_type_t types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_op_t* function_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, {.module_id = 0, .symbol_id = symbol}, nullptr, 0,
      nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
  const loom_func_like_t function = loom_func_like_cast(module_, function_op);
  loom_builder_enter_region(&builder, function_op,
                            loom_func_like_body(function));

  loom_value_id_t sources[IREE_ARRAYSIZE(types)] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(types); ++i) {
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(
        loom_index_constant_build(&builder, loom_attr_i64((int64_t)i + 7),
                                  types[i], LOOM_LOCATION_UNKNOWN, &constant));
    sources[i] = loom_index_constant_result(constant);
  }
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &builder, sources, IREE_ARRAYSIZE(sources), nullptr, 0, types,
      IREE_ARRAYSIZE(types), LOOM_LOCATION_UNKNOWN, &assume));
  const loom_value_slice_t results = loom_index_assume_results(assume);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder, results.values, results.count,
                                     LOOM_LOCATION_UNKNOWN, &use));
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner_, module_, loom_pass_value_fact_scope_function(function), &facts));
  loom_rewriter_attach_value_facts(&rewriter_, facts);
  materializer_probe = {
      .mode = MaterializerProbeMode::kIndexOnly,
      .support_query_count = 0,
      .materialization_count = 0,
  };
  rewriter_.constant_materializer = {
      .supports = ProbeMaterializerSupport,
      .materialize = ProbeMaterialize,
  };

  bool folded = false;
  IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter_, assume, &folded));
  EXPECT_FALSE(folded);
  EXPECT_EQ(materializer_probe.support_query_count, 2u);
  EXPECT_EQ(materializer_probe.materialization_count, 0u);
  EXPECT_FALSE(assume->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_EQ(loom_op_const_operands(use)[0], results.values[0]);
  EXPECT_EQ(loom_op_const_operands(use)[1], results.values[1]);
}

TEST_F(FactPublicationTest, ExactFoldUsesSelectedNonScalarMaterializer) {
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("vector_fold"), &name));
  uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* function_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, {.module_id = 0, .symbol_id = symbol}, nullptr, 0,
      nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
  const loom_func_like_t function = loom_func_like_cast(module_, function_op);
  loom_builder_enter_region(&builder, function_op,
                            loom_func_like_body(function));

  const loom_type_t input_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  const loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_BF16, loom_dim_pack_static(4), 0);
  loom_op_t* input = nullptr;
  IREE_ASSERT_OK(loom_vector_splatf_build(&builder, 0.5, input_type,
                                          LOOM_LOCATION_UNKNOWN, &input));
  loom_op_t* conversion = nullptr;
  IREE_ASSERT_OK(loom_vector_fptrunc_build(
      &builder, 0, loom_vector_splatf_result(input), input_type, result_type,
      LOOM_LOCATION_UNKNOWN, &conversion));
  const loom_value_id_t original_result =
      loom_vector_fptrunc_result(conversion);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder, &original_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner_, module_, loom_pass_value_fact_scope_function(function), &facts));
  loom_rewriter_attach_value_facts(&rewriter_, facts);
  materializer_probe = {
      .mode = MaterializerProbeMode::kVector,
      .support_query_count = 0,
      .materialization_count = 0,
  };
  rewriter_.constant_materializer = {
      .supports = ProbeMaterializerSupport,
      .materialize = ProbeMaterialize,
  };

  bool folded = false;
  IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter_, conversion, &folded));
  EXPECT_TRUE(folded);
  EXPECT_EQ(materializer_probe.support_query_count, 1u);
  EXPECT_EQ(materializer_probe.materialization_count, 1u);
  EXPECT_TRUE(conversion->flags & LOOM_OP_FLAG_DEAD);
  const loom_value_id_t replacement = loom_op_const_operands(use)[0];
  EXPECT_NE(replacement, original_result);
  const loom_op_t* replacement_op =
      loom_value_def_op(loom_module_value(module_, replacement));
  ASSERT_TRUE(loom_test_constant_isa(replacement_op));
  EXPECT_EQ(loom_attr_as_f64(loom_test_constant_value(replacement_op)), 0.5);
  EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, replacement),
                              result_type));
}

TEST_P(FactInputsTest, TypeEdgesRefreshTheirEquationOwnerBeforeFolding) {
  const auto [equation_owner, mutation] = GetParam();
  const bool ordinary = equation_owner == EquationOwner::kOperation;
  const bool moved_region = equation_owner == EquationOwner::kMovedRegion;
  const bool erased_region = equation_owner == EquationOwner::kErasedRegion;
  const bool plain_region =
      equation_owner == EquationOwner::kRegion || moved_region || erased_region;
  const bool cfg = equation_owner == EquationOwner::kCFG;
  const bool cycle = equation_owner == EquationOwner::kCycle;
  const bool counted = equation_owner == EquationOwner::kCounted;
  const bool conditional = equation_owner == EquationOwner::kCondition;
  const bool structured = counted || conditional;
  const bool create = mutation == TypeMutation::kCreate;
  const bool add = mutation == TypeMutation::kAdd;
  const bool erase = mutation == TypeMutation::kErase;
  const bool replace = mutation == TypeMutation::kReplace;
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("domain"), &name));
  uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* function_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, {.module_id = 0, .symbol_id = symbol}, &type, 1,
      nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
  const loom_func_like_t function = loom_func_like_cast(module_, function_op);
  loom_region_t* region = loom_func_like_body(function);
  region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_builder_enter_region(&builder, function_op, region);
  auto constant = [&](int64_t value, loom_type_t constant_type) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder, loom_attr_i64(value),
                                            constant_type,
                                            LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_constant_result(op);
  };
  const auto zero = constant(0, type);
  const auto source = loom_block_arg_id(loom_region_entry_block(region), 0);
  const auto i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_op_t* selector = nullptr;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder, 0, zero, {}, i1,
                                       LOOM_LOCATION_UNKNOWN, &selector));
  loom_value_id_t extent = source;
  loom_op_t* loop = nullptr;
  loom_block_t* latch = nullptr;
  loom_block_t* exit = nullptr;
  loom_op_t* root_terminator = nullptr;
  if (ordinary) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_attrs_build(&builder, 0, zero, {}, type,
                                         LOOM_LOCATION_UNKNOWN, &op));
    extent = loom_test_attrs_result(op);
  } else if (cfg || cycle) {
    loom_block_t* body = nullptr;
    IREE_ASSERT_OK(loom_region_append_block(module_, region, &body));
    IREE_ASSERT_OK(
        loom_builder_define_block_arg(&builder, body, type, &extent));
    IREE_ASSERT_OK(loom_cfg_br_build(&builder, body, &source, 1,
                                     LOOM_LOCATION_UNKNOWN, &root_terminator));
    if (cycle) {
      IREE_ASSERT_OK(loom_region_append_block(module_, region, &latch));
      IREE_ASSERT_OK(loom_region_append_block(module_, region, &exit));
    }
    loom_builder_set_block(&builder, body);
  } else if (plain_region) {
    IREE_ASSERT_OK(loom_test_block_args_build(&builder, &source, 1,
                                              LOOM_LOCATION_UNKNOWN, &loop));
    IREE_ASSERT_OK(loom_test_yield_build(
        &builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &root_terminator));
    loom_builder_enter_region(&builder, loop, loom_test_block_args_body(loop));
    extent = loom_block_arg_id(
        loom_region_entry_block(loom_test_block_args_body(loop)), 0);
  } else if (structured) {
    if (counted) {
      const auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      const auto begin = constant(0, index);
      const auto end = constant(4, index);
      const auto step = constant(1, index);
      IREE_ASSERT_OK(loom_scf_for_build(
          &builder, 0, begin, end, step, &source, 1, nullptr, nullptr, 0,
          LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
          LOOM_LOCATION_UNKNOWN, &loop));
    } else {
      IREE_ASSERT_OK(loom_scf_while_build(&builder, &source, 1, nullptr,
                                          nullptr, 1, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &loop));
    }
    IREE_ASSERT_OK(loom_test_yield_build(
        &builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &root_terminator));
    const auto loop_like = loom_loop_like_cast(module_, loop);
    if (conditional) {
      auto* condition_region = loom_loop_like_condition_region(loop_like);
      loom_builder_enter_region(&builder, loop, condition_region);
      const auto condition_arg =
          loom_block_arg_id(loom_region_entry_block(condition_region), 0);
      loom_op_t* condition_op = nullptr;
      IREE_ASSERT_OK(loom_scf_condition_build(
          &builder, loom_test_attrs_result(selector), &condition_arg, 1,
          LOOM_LOCATION_UNKNOWN, &condition_op));
    }
    auto* body = loom_loop_like_body(loop_like);
    loom_builder_enter_region(&builder, loop, body);
    extent = loom_block_arg_id(loom_region_entry_block(body), counted ? 1 : 0);
  }
  const auto dynamic_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(extent), 0);
  const auto static_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(1), 0);
  loom_op_t* shaped = nullptr;
  loom_op_t* shaped_use = nullptr;
  if (!create) {
    IREE_ASSERT_OK(loom_test_attrs_build(&builder, 0, zero, {},
                                         add ? static_type : dynamic_type,
                                         LOOM_LOCATION_UNKNOWN, &shaped));
    const auto shaped_value = loom_test_attrs_result(shaped);
    IREE_ASSERT_OK(loom_test_use_build(&builder, &shaped_value, 1,
                                       LOOM_LOCATION_UNKNOWN, &shaped_use));
  }
  loom_op_t* comparison = nullptr;
  loom_op_t* use = nullptr;
  auto build_comparison = [&](loom_builder_t* selected_builder) {
    IREE_ASSERT_OK(loom_scalar_cmpi_build(
        selected_builder, LOOM_SCALAR_CMPI_PREDICATE_SGE, extent, zero,
        LOOM_LOCATION_UNKNOWN, &comparison));
    const auto result = loom_scalar_cmpi_result(comparison);
    IREE_ASSERT_OK(loom_test_use_build(selected_builder, &result, 1,
                                       LOOM_LOCATION_UNKNOWN, &use));
  };
  if (!replace) {
    build_comparison(&builder);
  }
  loom_op_t* terminator = nullptr;
  if (cycle) {
    IREE_ASSERT_OK(loom_cfg_cond_br_build(
        &builder, loom_test_attrs_result(selector), latch, exit,
        LOOM_LOCATION_UNKNOWN, &terminator));
    loom_builder_set_block(&builder, latch);
    loom_op_t* branch = nullptr;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder, terminator->parent_block,
                                     &extent, 1, LOOM_LOCATION_UNKNOWN,
                                     &branch));
    loom_builder_set_block(&builder, exit);
    IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &branch));
  } else if (structured) {
    IREE_ASSERT_OK(loom_scf_yield_build(&builder, &extent, 1,
                                        LOOM_LOCATION_UNKNOWN, &terminator));
  } else {
    IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));
  }
  loom_value_fact_table_t inputs;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&inputs, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &inputs, source, loom_value_facts_make(-8, 64, 4)));
  auto scope = loom_pass_value_fact_scope_function(function);
  scope.seed_facts = {&inputs, &source, 1};
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(
      loom_pass_value_fact_owner_acquire(&owner_, module_, scope, &facts));
  loom_rewriter_attach_value_facts(&rewriter_, facts);
  rewriter_.constant_materializer = {
      .supports = CanMaterialize,
      .materialize = Materialize,
  };
  IREE_ASSERT_OK(loom_rewriter_seed_function(&rewriter_, function));
  loom_builder_set_before(&rewriter_.builder, terminator);
  if (create) {
    IREE_ASSERT_OK(loom_test_attrs_build(&rewriter_.builder, 0, zero, {},
                                         dynamic_type, LOOM_LOCATION_UNKNOWN,
                                         &shaped));
    const auto shaped_value = loom_test_attrs_result(shaped);
    IREE_ASSERT_OK(loom_test_use_build(&rewriter_.builder, &shaped_value, 1,
                                       LOOM_LOCATION_UNKNOWN, &shaped_use));
  } else if (erase) {
    loom_op_t* erased[] = {shaped_use, shaped};
    IREE_ASSERT_OK(loom_rewriter_erase_closed_set(&rewriter_, erased, 2));
  } else if (replace) {
    loom_op_t* replacement = nullptr;
    IREE_ASSERT_OK(loom_test_attrs_build(&rewriter_.builder, 0, zero, {}, type,
                                         LOOM_LOCATION_UNKNOWN, &replacement));
    IREE_ASSERT_OK(loom_rewriter_replace_all_uses_with(
        &rewriter_, extent, loom_test_attrs_result(replacement)));
    build_comparison(&rewriter_.builder);
  } else {
    IREE_ASSERT_OK(
        loom_rewriter_set_value_type(&rewriter_, loom_test_attrs_result(shaped),
                                     add ? dynamic_type : static_type));
  }
  if (moved_region) {
    auto* old_entry = loom_region_entry_block(loom_test_block_args_body(loop));
    IREE_ASSERT_OK(loom_rewriter_erase(&rewriter_, root_terminator));
    loom_block_t* moved_entry = nullptr;
    IREE_ASSERT_OK(loom_rewriter_move_region_blocks(
        &rewriter_, loom_test_block_args_body(loop), loop, region, 1,
        function_op, &moved_entry));
    ASSERT_NE(moved_entry, old_entry);
    IREE_ASSERT_OK(loom_rewriter_erase(&rewriter_, loop));
    loom_builder_enter_region(&rewriter_.builder, function_op, region);
    IREE_ASSERT_OK(loom_cfg_br_build(&rewriter_.builder, moved_entry, &source,
                                     1, LOOM_LOCATION_UNKNOWN,
                                     &root_terminator));
    IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter_, region));
  } else if (erased_region) {
    IREE_ASSERT_OK(loom_rewriter_erase(&rewriter_, loop));
    loom_builder_set_before(&rewriter_.builder, root_terminator);
    extent = source;
    IREE_ASSERT_OK(loom_test_attrs_build(
        &rewriter_.builder, 0, zero, {},
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_dynamic(extent), 0),
        LOOM_LOCATION_UNKNOWN, &shaped));
    build_comparison(&rewriter_.builder);
  }
  loom_value_fact_table_t* oracle = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(&oracle_owner_, module_,
                                                    scope, &oracle));
  const auto expected = loom_value_fact_table_lookup(oracle, extent);
  int64_t expected_constant = -1;
  const bool expected_exact = loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(oracle, loom_scalar_cmpi_result(comparison)),
      &expected_constant);
  while (loom_op_t* op = loom_rewriter_pop(&rewriter_)) {
    bool folded = false;
    IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter_, op, &folded));
  }
  const auto actual = loom_value_fact_table_lookup(facts, extent);
  EXPECT_EQ(actual.range_lo, expected.range_lo);
  EXPECT_EQ(actual.range_hi, expected.range_hi);
  EXPECT_EQ(actual.known_divisor, expected.known_divisor);
  const auto* actual_op = loom_value_def_op(
      loom_module_value(module_, loom_op_const_operands(use)[0]));
  if (expected_exact) {
    ASSERT_TRUE(loom_test_constant_isa(actual_op));
    EXPECT_EQ(loom_attr_as_i64(loom_test_constant_value(actual_op)),
              expected_constant);
  } else {
    EXPECT_TRUE(loom_scalar_cmpi_isa(actual_op));
  }
}

TEST_F(FactPublicationTest, ScopeDetachReusesPendingStorage) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(&builder, IREE_SV("scope"), &name));
  loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* function_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(&builder, 0, 0, 0, {0, symbol}, nullptr,
                                      0, nullptr, 0, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &function_op));
  const auto function = loom_func_like_cast(module_, function_op);
  loom_builder_enter_region(&builder, function_op,
                            loom_func_like_body(function));
  loom_op_t* source = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder, loom_attr_i64(1), type,
                                          LOOM_LOCATION_UNKNOWN, &source));
  const auto input = loom_test_constant_result(source);
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder, input, input, type,
                                      LOOM_LOCATION_UNKNOWN, &sum));
  loom_op_t* terminator = nullptr;
  const auto result = loom_test_addi_result(sum);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder, &result, 1, LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner_, module_, loom_pass_value_fact_scope_function(function), &facts));
  loom_rewriter_attach_value_facts(&rewriter_, facts);
  iree_host_size_t warmed_bytes = 0;
  for (int64_t value = 2; value <= 8; ++value) {
    IREE_ASSERT_OK(loom_test_constant_rewrite_value(&rewriter_, source,
                                                    loom_attr_i64(value)));
    if (value % 2 == 0) {
      // Discard a pending transaction, refresh independently, and reattach.
      loom_rewriter_attach_value_facts(&rewriter_, nullptr);
      IREE_ASSERT_OK(loom_value_fact_table_compute(facts, module_, function));
      loom_rewriter_attach_value_facts(&rewriter_, facts);
      IREE_ASSERT_OK(loom_test_constant_rewrite_value(
          &rewriter_, source, loom_attr_i64(value + 1)));
    }
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter_));
    int64_t actual = 0;
    ASSERT_TRUE(loom_value_facts_as_exact_i64(
        loom_rewriter_value_facts(&rewriter_, result), &actual));
    EXPECT_EQ(actual, (value + (value % 2 == 0)) * 2);
    if (value == 2) {
      warmed_bytes = arena_.used_allocation_size;
    } else {
      EXPECT_EQ(arena_.used_allocation_size, warmed_bytes);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    InverseShapeDomains, FactInputsTest,
    ::testing::Combine(
        ::testing::Values(EquationOwner::kOperation, EquationOwner::kEntry,
                          EquationOwner::kRegion, EquationOwner::kMovedRegion,
                          EquationOwner::kErasedRegion, EquationOwner::kCFG,
                          EquationOwner::kCycle, EquationOwner::kCounted,
                          EquationOwner::kCondition),
        ::testing::Values(TypeMutation::kRemove, TypeMutation::kAdd,
                          TypeMutation::kCreate, TypeMutation::kErase,
                          TypeMutation::kReplace)));

}  // namespace
}  // namespace loom
