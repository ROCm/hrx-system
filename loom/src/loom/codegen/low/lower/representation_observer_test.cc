// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/representation_observer.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/source_query.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/target_records.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

enum TestRepresentation : loom_low_representation_id_t {
  kRepresentationFirst = 0,
  kRepresentationSecond = 1,
  kRepresentationConflict = 2,
};

enum TestBoundaryAction : uint16_t {
  kBoundaryScalarConstant = 0,
  kBoundaryScalarAssume = 1,
  kBoundaryFunction = 2,
  kBoundaryVectorExtract = 3,
  kBoundaryVectorAdd = 4,
  kBoundaryVectorSubtract = 5,
  kBoundaryVectorMultiply = 6,
};

static constexpr loom_low_lower_representation_boundary_t kBoundaries[] = {
    {LOOM_OP_SCALAR_CONSTANT, kBoundaryScalarConstant},
    {LOOM_OP_SCALAR_ASSUME, kBoundaryScalarAssume},
    {LOOM_OP_FUNC_DEF, kBoundaryFunction},
    {LOOM_OP_VECTOR_EXTRACT, kBoundaryVectorExtract},
    {LOOM_OP_VECTOR_ADDI, kBoundaryVectorAdd},
    {LOOM_OP_VECTOR_SUBI, kBoundaryVectorSubtract},
    {LOOM_OP_VECTOR_MULI, kBoundaryVectorMultiply},
};
static_assert(LOOM_OP_SCALAR_CONSTANT < LOOM_OP_SCALAR_ASSUME);
static_assert(LOOM_OP_SCALAR_ASSUME < LOOM_OP_FUNC_DEF);
static_assert(LOOM_OP_FUNC_DEF < LOOM_OP_VECTOR_EXTRACT);
static_assert(LOOM_OP_VECTOR_EXTRACT < LOOM_OP_VECTOR_ADDI);
static_assert(LOOM_OP_VECTOR_ADDI < LOOM_OP_VECTOR_SUBI);
static_assert(LOOM_OP_VECTOR_SUBI < LOOM_OP_VECTOR_MULI);

static constexpr loom_low_representation_candidate_t kTieCandidates[] = {
    {kRepresentationFirst, 0, {0, 0}},
    {kRepresentationSecond, 0, {0, 0}},
};

static constexpr loom_low_representation_candidate_t kAddCandidates[] = {
    {kRepresentationFirst, 0, {0, 0}},
    {kRepresentationSecond, 0, {4, 4}},
};

static constexpr loom_low_representation_candidate_t kMultiplyCandidates[] = {
    {kRepresentationFirst, 0, {7, 7}},
    {kRepresentationSecond, 0, {0, 0}},
};

static constexpr loom_low_representation_candidate_t kConflictCandidates[] = {
    {kRepresentationConflict, 0, {0, 0}},
};

class LowLowerRepresentationObserverTest : public ::testing::Test {
 protected:
  struct CapturedValue {
    // Source SSA value queried during low-function preamble emission.
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    // Representation selected for source_value_id.
    loom_low_representation_id_t representation =
        LOOM_LOW_REPRESENTATION_ID_NONE;
  };

  static bool IsRepresentableType(loom_type_t type) {
    if (loom_type_is_scalar(type)) {
      return loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
    }
    return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
           loom_type_is_all_static(type) &&
           loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
           loom_type_dim_static_size_at(type, 0) == 4;
  }

  static bool RelatesValues(void* user_data, loom_low_lower_context_t* context,
                            const loom_op_t* source_op,
                            const loom_value_relation_t* relation) {
    auto* test = static_cast<LowLowerRepresentationObserverTest*>(user_data);
    EXPECT_NE(source_op, nullptr);
    ++test->relation_counts_[relation->kind];
    loom_module_t* module = loom_low_lower_context_module(context);
    const loom_type_t left_type =
        loom_module_value_type(module, relation->source_value_id);
    const loom_type_t right_type =
        loom_module_value_type(module, relation->destination_value_id);
    return loom_type_equal(left_type, right_type) &&
           IsRepresentableType(left_type);
  }

  static void ObserveBoundary(
      void* user_data, uint16_t action, loom_low_lower_context_t* context,
      const loom_op_t* source_op,
      loom_low_lower_representation_recorder_t* recorder) {
    auto* test = static_cast<LowLowerRepresentationObserverTest*>(user_data);
    (void)context;
    const loom_low_representation_candidate_t* candidates = nullptr;
    iree_host_size_t candidate_count = 0;
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    switch (action) {
      case kBoundaryScalarConstant:
        candidates = kAddCandidates;
        candidate_count = IREE_ARRAYSIZE(kAddCandidates);
        source_value_id = loom_op_const_results(source_op)[0];
        break;
      case kBoundaryScalarAssume:
        candidates = kMultiplyCandidates;
        candidate_count = IREE_ARRAYSIZE(kMultiplyCandidates);
        source_value_id = loom_op_const_results(source_op)[0];
        break;
      case kBoundaryFunction:
        ++test->source_function_boundary_count_;
        return;
      case kBoundaryVectorExtract:
        candidates = kTieCandidates;
        candidate_count = IREE_ARRAYSIZE(kTieCandidates);
        source_value_id = loom_vector_extract_source(source_op);
        break;
      case kBoundaryVectorAdd:
        candidates = kAddCandidates;
        candidate_count = IREE_ARRAYSIZE(kAddCandidates);
        source_value_id = loom_vector_addi_result(source_op);
        break;
      case kBoundaryVectorSubtract:
        candidates = kConflictCandidates;
        candidate_count = IREE_ARRAYSIZE(kConflictCandidates);
        source_value_id = loom_vector_subi_result(source_op);
        break;
      case kBoundaryVectorMultiply:
        candidates = kMultiplyCandidates;
        candidate_count = IREE_ARRAYSIZE(kMultiplyCandidates);
        source_value_id = loom_vector_muli_result(source_op);
        break;
      default:
        loom_low_lower_representation_record_failure(
            recorder,
            iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "unknown test representation boundary action"));
        return;
    }
    loom_low_lower_representation_record_candidates(
        recorder, source_value_id, candidates, candidate_count);
  }

  static iree_status_t CaptureRepresentations(
      void* user_data, loom_low_lower_context_t* context) {
    auto* test = static_cast<LowLowerRepresentationObserverTest*>(user_data);
    test->capture_called_ = true;
    loom_target_contract_query_environment_t query_environment = {};
    IREE_RETURN_IF_ERROR(loom_low_lower_source_query_environment_initialize(
        context, loom_low_lower_context_descriptor_set(context),
        &query_environment));
    for (iree_host_size_t i = 0; i < test->captured_value_count_; ++i) {
      CapturedValue* captured = &test->captured_values_[i];
      IREE_RETURN_IF_ERROR(loom_low_lower_representation_lookup(
          context, captured->source_value_id, &captured->representation));
      loom_low_representation_id_t query_representation =
          LOOM_LOW_REPRESENTATION_ID_NONE;
      bool query_plan_available = false;
      IREE_RETURN_IF_ERROR(loom_low_lower_representation_query_lookup(
          &query_environment, captured->source_value_id, &query_representation,
          &query_plan_available));
      if (!query_plan_available ||
          query_representation != captured->representation) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "query scope did not retrieve the retained representation");
      }
    }
    return iree_ok_status();
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    iree_host_size_t scf_vtable_count = 0;
    const loom_op_vtable_t* const* scf_vtables =
        loom_scf_dialect_vtables(&scf_vtable_count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, LOOM_DIALECT_SCF, scf_vtables,
                                      static_cast<uint16_t>(scf_vtable_count)));
    iree_host_size_t scf_semantics_count = 0;
    const loom_op_semantics_t* scf_semantics =
        loom_scf_dialect_op_semantics(&scf_semantics_count);
    ASSERT_EQ(scf_semantics_count, scf_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_SCF, scf_semantics,
        static_cast<uint16_t>(scf_semantics_count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("representation_observer_test"), &block_pool_,
        nullptr, iree_allocator_system(), &module_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    target_facts_.fact_type = &loom_test_target_fact_type;
    target_facts_.storage.bundle = *loom_test_target_bundles.values[1];

    provider_ = (loom_low_lower_representation_provider_t){
        .relation = RelatesValues,
        .observe_boundary = ObserveBoundary,
        .boundaries = kBoundaries,
        .boundary_count = IREE_ARRAYSIZE(kBoundaries),
        .relation_mask = LOOM_VALUE_RELATION_MASK_ALL,
        .user_data = this,
    };
    source_plan_observer_ = (loom_low_lower_source_plan_observer_t){
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = &provider_,
    };
    policy_ = *loom_test_low_lower_policy();
    policy_.source_plan_observer = &source_plan_observer_;
    policy_.emit_preamble.fn = CaptureRepresentations;
    policy_.emit_preamble.user_data = this;
    options_.target_facts = &target_facts_;
    options_.descriptor_registry = &descriptor_registry_.registry;
    options_.policy = &policy_;
    options_.control_flow_lowering =
        LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
  }

  void TearDown() override {
    loom_low_lower_result_deinitialize(&result_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_builder_t BuildFunction(const loom_type_t* argument_types,
                               uint16_t argument_count,
                               loom_type_t result_type) {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &module_builder, IREE_SV("representation_test"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    loom_op_t* function_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        argument_types, argument_count, &result_type, 1, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function_op));
    function_ = loom_func_like_cast(module_, function_op);
    loom_block_t* entry_block =
        loom_region_entry_block(loom_func_like_body(function_));
    loom_builder_t body_builder;
    loom_builder_initialize(module_, &module_->arena, entry_block,
                            &body_builder);
    body_builder.ip.parent_op = function_op;
    return body_builder;
  }

  loom_value_id_t BuildIndexConstant(loom_builder_t* builder, int64_t value) {
    loom_op_t* constant = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &constant));
    return loom_index_constant_result(constant);
  }

  void Capture(loom_value_id_t source_value_id) {
    IREE_ASSERT_LT(captured_value_count_, IREE_ARRAYSIZE(captured_values_));
    captured_values_[captured_value_count_++].source_value_id = source_value_id;
  }

  iree_status_t Lower() {
    IREE_CHECK_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
    fact_table_.context.target_facts = &target_facts_;
    IREE_CHECK_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function_));
    options_.fact_table = &fact_table_;
    return loom_low_lower_function(module_, function_, &options_, &result_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_target_facts_t target_facts_ = {};
  loom_value_fact_table_t fact_table_ = {};
  loom_low_lower_representation_provider_t provider_ = {};
  loom_low_lower_source_plan_observer_t source_plan_observer_ = {};
  loom_low_lower_policy_t policy_ = {};
  loom_low_lower_options_t options_ = {};
  loom_low_lower_result_t result_ = {};
  CapturedValue captured_values_[8];
  iree_host_size_t captured_value_count_ = 0;
  uint32_t relation_counts_[LOOM_VALUE_RELATION_COUNT_] = {};
  uint32_t source_function_boundary_count_ = 0;
  bool capture_called_ = false;
};

TEST_F(LowLowerRepresentationObserverTest,
       AggregatesCostsAcrossRawLoopCarrier) {
  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t argument_types[] = {vector_type, vector_type};
  loom_builder_t builder =
      BuildFunction(argument_types, IREE_ARRAYSIZE(argument_types), i32_type);
  loom_block_t* entry_block = builder.ip.block;
  const loom_value_id_t lhs = loom_block_arg_id(entry_block, 0);
  const loom_value_id_t rhs = loom_block_arg_id(entry_block, 1);

  const loom_value_id_t lower_bound = BuildIndexConstant(&builder, 0);
  const loom_value_id_t upper_bound = BuildIndexConstant(&builder, 2);
  const loom_value_id_t step = BuildIndexConstant(&builder, 1);
  const loom_value_id_t rows = BuildIndexConstant(&builder, 2);
  const loom_value_id_t columns = BuildIndexConstant(&builder, 2);

  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder, /*instance_flags=*/0, lhs,
                                        rhs, vector_type, LOOM_LOCATION_UNKNOWN,
                                        &add));
  const loom_value_id_t seed = loom_vector_addi_result(add);
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 3,
  };
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder, /*build_flags=*/0, lower_bound, upper_bound, step, &seed, 1,
      &tied_result, 1, LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, LOOM_LOCATION_UNKNOWN, &loop));

  const loom_builder_ip_t saved =
      loom_builder_enter_region(&builder, loop, loom_scf_for_body(loop));
  const loom_value_id_t body_argument =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_op_t* body_fragment = nullptr;
  IREE_ASSERT_OK(loom_vector_fragment_build(
      &builder, /*build_flags=*/0, LOOM_VECTOR_ROLE_INIT, body_argument,
      LOOM_VALUE_ID_INVALID, rows, columns, nullptr, 0, nullptr, 0, vector_type,
      LOOM_LOCATION_UNKNOWN, &body_fragment));
  loom_op_t* multiply = nullptr;
  IREE_ASSERT_OK(
      loom_vector_muli_build(&builder, /*instance_flags=*/0,
                             loom_vector_fragment_result(body_fragment), rhs,
                             vector_type, LOOM_LOCATION_UNKNOWN, &multiply));
  const loom_value_id_t next = loom_vector_muli_result(multiply);
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(
      loom_scf_yield_build(&builder, &next, 1, LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder, saved);

  const loom_value_id_t loop_result = loom_scf_for_results(loop).values[0];
  loom_op_t* result_fragment = nullptr;
  IREE_ASSERT_OK(loom_vector_fragment_build(
      &builder, /*build_flags=*/0, LOOM_VECTOR_ROLE_RESULT, loop_result,
      LOOM_VALUE_ID_INVALID, rows, columns, nullptr, 0, nullptr, 0, vector_type,
      LOOM_LOCATION_UNKNOWN, &result_fragment));
  const int64_t static_index = 0;
  loom_op_t* extract = nullptr;
  IREE_ASSERT_OK(loom_vector_extract_build(
      &builder, loom_vector_fragment_result(result_fragment), nullptr, 0,
      &static_index, 1, i32_type, LOOM_LOCATION_UNKNOWN, &extract));
  const loom_value_id_t scalar_result = loom_vector_extract_result(extract);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, &scalar_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  Capture(seed);
  Capture(body_argument);
  Capture(loom_vector_fragment_result(body_fragment));
  Capture(next);
  Capture(loop_result);
  Capture(loom_vector_fragment_result(result_fragment));
  IREE_ASSERT_OK(Lower());

  ASSERT_TRUE(capture_called_);
  EXPECT_EQ(source_function_boundary_count_, 1u);
  ASSERT_EQ(captured_value_count_, 6u);
  for (const CapturedValue& captured : captured_values_) {
    if (captured.source_value_id == LOOM_VALUE_ID_INVALID) continue;
    EXPECT_EQ(captured.representation, kRepresentationSecond);
  }
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_TIED_RESULT], 0u);
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_VALUE_ALIAS], 0u);
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_ELEMENTWISE], 0u);
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_LOOP_CARRIED], 0u);
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_LOOP_BYPASS], 0u);
}

TEST_F(LowLowerRepresentationObserverTest,
       FactIdentityAggregatesBoundaryCosts) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_builder_t builder = BuildFunction(nullptr, 0, i32_type);
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(loom_scalar_constant_build(
      &builder, loom_attr_i64(7), i32_type, LOOM_LOCATION_UNKNOWN, &constant));
  const loom_value_id_t constant_result = loom_scalar_constant_result(constant);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder, &constant_result, 1,
                                          nullptr, 0, &i32_type, 1,
                                          LOOM_LOCATION_UNKNOWN, &assume));
  const loom_value_id_t assumed_result =
      loom_scalar_assume_results(assume).values[0];
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, &assumed_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  Capture(constant_result);
  Capture(assumed_result);
  IREE_ASSERT_OK(Lower());

  ASSERT_TRUE(capture_called_);
  ASSERT_EQ(captured_value_count_, 2u);
  EXPECT_EQ(captured_values_[0].representation, kRepresentationSecond);
  EXPECT_EQ(captured_values_[1].representation, kRepresentationSecond);
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_FACT_IDENTITY], 0u);
}

TEST_F(LowLowerRepresentationObserverTest, SelectPayloadsShareRepresentation) {
  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  const loom_type_t mask_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, loom_dim_pack_static(4), 0);
  const loom_type_t argument_types[] = {vector_type, vector_type, mask_type};
  loom_builder_t builder = BuildFunction(
      argument_types, IREE_ARRAYSIZE(argument_types), vector_type);
  loom_block_t* entry_block = builder.ip.block;
  const loom_value_id_t lhs = loom_block_arg_id(entry_block, 0);
  const loom_value_id_t rhs = loom_block_arg_id(entry_block, 1);
  const loom_value_id_t condition = loom_block_arg_id(entry_block, 2);

  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder, /*instance_flags=*/0, lhs,
                                        rhs, vector_type, LOOM_LOCATION_UNKNOWN,
                                        &add));
  loom_op_t* multiply = nullptr;
  IREE_ASSERT_OK(loom_vector_muli_build(&builder, /*instance_flags=*/0, lhs,
                                        rhs, vector_type, LOOM_LOCATION_UNKNOWN,
                                        &multiply));
  loom_op_t* select = nullptr;
  IREE_ASSERT_OK(loom_vector_select_build(
      &builder, condition, loom_vector_addi_result(add),
      loom_vector_muli_result(multiply), vector_type, LOOM_LOCATION_UNKNOWN,
      &select));
  const loom_value_id_t result = loom_vector_select_result(select);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, &result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  Capture(loom_vector_addi_result(add));
  Capture(loom_vector_muli_result(multiply));
  Capture(result);
  IREE_ASSERT_OK(Lower());

  ASSERT_TRUE(capture_called_);
  ASSERT_EQ(captured_value_count_, 3u);
  for (const CapturedValue& captured : captured_values_) {
    if (captured.source_value_id == LOOM_VALUE_ID_INVALID) continue;
    EXPECT_EQ(captured.representation, kRepresentationSecond);
  }
  EXPECT_EQ(relation_counts_[LOOM_VALUE_RELATION_SELECT_PAYLOAD], 2u);
}

TEST_F(LowLowerRepresentationObserverTest,
       RegionBranchYieldsShareResultRepresentation) {
  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t argument_types[] = {vector_type, vector_type, i1_type};
  loom_builder_t builder = BuildFunction(
      argument_types, IREE_ARRAYSIZE(argument_types), vector_type);
  loom_block_t* entry_block = builder.ip.block;
  const loom_value_id_t lhs = loom_block_arg_id(entry_block, 0);
  const loom_value_id_t rhs = loom_block_arg_id(entry_block, 1);
  const loom_value_id_t condition = loom_block_arg_id(entry_block, 2);

  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(
      &builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition, &vector_type,
      1, nullptr, 0, LOOM_LOCATION_UNKNOWN, &branch));
  const loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, branch, loom_scf_if_then_region(branch));
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder, /*instance_flags=*/0, lhs,
                                        rhs, vector_type, LOOM_LOCATION_UNKNOWN,
                                        &add));
  const loom_value_id_t add_result = loom_vector_addi_result(add);
  loom_op_t* then_yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder, &add_result, 1,
                                      LOOM_LOCATION_UNKNOWN, &then_yield));
  loom_builder_restore(&builder, saved);

  const loom_builder_ip_t saved_else = loom_builder_enter_region(
      &builder, branch, loom_scf_if_else_region(branch));
  loom_op_t* multiply = nullptr;
  IREE_ASSERT_OK(loom_vector_muli_build(&builder, /*instance_flags=*/0, lhs,
                                        rhs, vector_type, LOOM_LOCATION_UNKNOWN,
                                        &multiply));
  const loom_value_id_t multiply_result = loom_vector_muli_result(multiply);
  loom_op_t* else_yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder, &multiply_result, 1,
                                      LOOM_LOCATION_UNKNOWN, &else_yield));
  loom_builder_restore(&builder, saved_else);

  const loom_value_id_t result = loom_scf_if_results(branch).values[0];
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, &result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  Capture(add_result);
  Capture(multiply_result);
  Capture(result);
  IREE_ASSERT_OK(Lower());

  ASSERT_TRUE(capture_called_);
  ASSERT_EQ(captured_value_count_, 3u);
  for (const CapturedValue& captured : captured_values_) {
    if (captured.source_value_id == LOOM_VALUE_ID_INVALID) continue;
    EXPECT_EQ(captured.representation, kRepresentationSecond);
  }
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_REGION_RESULT], 0u);
}

TEST_F(LowLowerRepresentationObserverTest,
       CfgArgumentsShareIncomingRepresentation) {
  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  const loom_type_t argument_types[] = {vector_type, vector_type};
  loom_builder_t builder = BuildFunction(
      argument_types, IREE_ARRAYSIZE(argument_types), vector_type);
  loom_block_t* entry_block = builder.ip.block;
  const loom_value_id_t lhs = loom_block_arg_id(entry_block, 0);
  const loom_value_id_t rhs = loom_block_arg_id(entry_block, 1);

  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder, /*instance_flags=*/0, lhs,
                                        rhs, vector_type, LOOM_LOCATION_UNKNOWN,
                                        &add));
  const loom_value_id_t add_result = loom_vector_addi_result(add);
  loom_region_t* body = loom_func_like_body(function_);
  body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_block_t* join_block = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &join_block));
  loom_value_id_t join_argument = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(&builder, join_block,
                                               vector_type, &join_argument));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder, join_block, &add_result, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));

  loom_builder_set_block(&builder, join_block);
  loom_op_t* multiply = nullptr;
  IREE_ASSERT_OK(loom_vector_muli_build(&builder, /*instance_flags=*/0,
                                        join_argument, rhs, vector_type,
                                        LOOM_LOCATION_UNKNOWN, &multiply));
  const loom_value_id_t result = loom_vector_muli_result(multiply);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, &result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  Capture(add_result);
  Capture(join_argument);
  Capture(result);
  IREE_ASSERT_OK(Lower());

  ASSERT_TRUE(capture_called_);
  ASSERT_EQ(captured_value_count_, 3u);
  for (const CapturedValue& captured : captured_values_) {
    if (captured.source_value_id == LOOM_VALUE_ID_INVALID) continue;
    EXPECT_EQ(captured.representation, kRepresentationSecond);
  }
  EXPECT_GT(relation_counts_[LOOM_VALUE_RELATION_CFG_ARGUMENT], 0u);
}

TEST_F(LowLowerRepresentationObserverTest,
       IncompatibleElementwiseDomainsFailClosed) {
  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  const loom_type_t argument_types[] = {vector_type, vector_type};
  loom_builder_t builder = BuildFunction(
      argument_types, IREE_ARRAYSIZE(argument_types), vector_type);
  loom_block_t* entry_block = builder.ip.block;
  const loom_value_id_t lhs = loom_block_arg_id(entry_block, 0);
  const loom_value_id_t rhs = loom_block_arg_id(entry_block, 1);
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder, /*instance_flags=*/0, lhs,
                                        rhs, vector_type, LOOM_LOCATION_UNKNOWN,
                                        &add));
  loom_op_t* subtract = nullptr;
  IREE_ASSERT_OK(loom_vector_subi_build(
      &builder, /*instance_flags=*/0, loom_vector_addi_result(add), rhs,
      vector_type, LOOM_LOCATION_UNKNOWN, &subtract));
  const loom_value_id_t result = loom_vector_subi_result(subtract);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&builder, &result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, Lower());
  EXPECT_FALSE(capture_called_);
}

}  // namespace
}  // namespace loom
