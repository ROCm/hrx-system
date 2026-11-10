// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/control.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/verify/verify.h"

namespace {

class PipelineControlTest : public ::testing::Test {
 protected:
  virtual bool HasAutonomousPreparation() const { return false; }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("control"), &blocks_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    const loom_type_t observed[] = {loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                    loom_type_buffer()};
    observer_ = Function("observe", observed, 2);
    const auto worker = Function("work", nullptr, 0);
    const loom_type_t captured[] = {loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)};
    const auto autonomous = Function("autonomous", captured, 2);
    capture_observer_ = Function("observe_captures", captured, 2);
    auto* leaf = Pipeline("leaf");
    loom_op_t* strand;
    const int64_t origin = 0, extent = 1;
    IREE_ASSERT_OK(loom_pipeline_strand_build(
        &rewriter_.builder, 0, 0, loom_symbol_ref_null(), nullptr, 0, &origin,
        1, nullptr, 0, &extent, 1, nullptr, 0, &extent, 1,
        LOOM_LOCATION_UNKNOWN, &strand));
    loom_builder_enter_region(&rewriter_.builder, strand,
                              loom_pipeline_strand_body(strand));
    loom_op_t* op;
    IREE_ASSERT_OK(loom_func_call_build(&rewriter_.builder, 0, 0, 0, 0, worker,
                                        nullptr, 0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
    IREE_ASSERT_OK(loom_pipeline_end_build(&rewriter_.builder,
                                           LOOM_LOCATION_UNKNOWN, &op));
    if (HasAutonomousPreparation()) {
      // Declare autonomous input after the consuming instruction strand. Its
      // captures must still prepare before that strand executes.
      auto* builder = &rewriter_.builder;
      loom_builder_enter_region(builder, leaf, loom_pipeline_def_body(leaf));
      const auto value =
          loom_region_entry_block(loom_pipeline_def_body(leaf))->arg_ids[0];
      loom_op_t *constant, *sum;
      IREE_ASSERT_OK(
          loom_index_constant_build(builder, loom_attr_i64(7), captured[0],
                                    LOOM_LOCATION_UNKNOWN, &constant));
      IREE_ASSERT_OK(
          loom_index_add_build(builder, value, loom_op_results(constant)[0],
                               captured[0], LOOM_LOCATION_UNKNOWN, &sum));
      const loom_value_id_t operands[] = {value, loom_op_results(sum)[0]};
      IREE_ASSERT_OK(loom_pipeline_strand_build(
          builder, 0, 0, loom_symbol_ref_null(), nullptr, 0, &origin, 1,
          nullptr, 0, &extent, 1, nullptr, 0, &extent, 1, LOOM_LOCATION_UNKNOWN,
          &strand));
      loom_builder_enter_region(builder, strand,
                                loom_pipeline_strand_body(strand));
      IREE_ASSERT_OK(loom_func_call_build(builder, 0, 0, 0, 0, autonomous,
                                          operands, 2, nullptr, 0, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &op));
      IREE_ASSERT_OK(
          loom_pipeline_end_build(builder, LOOM_LOCATION_UNKNOWN, &op));
      engines_[1] = IREE_HOST_SIZE_MAX;
      engines_[2] = 0;
      engines_[3] = IREE_HOST_SIZE_MAX;
    }
    Finish(leaf);
    auto* middle = Pipeline("middle");
    Compose(middle, leaf, 1);
    Finish(middle);
    auto* root = Pipeline("root");
    Compose(root, middle, 2);
    Finish(root);
    realization_.function = loom_func_like_cast(module_, root);
    Verify();

    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 0));
    loom_type_registry_configure_fact_context(&facts_.context);
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(&facts_, module_, realization_.function));
    realization_.facts = &facts_;
    bool valid = false;
    IREE_ASSERT_OK(loom_pipeline_resources_build(
        module_, realization_.function, &facts_, nullptr, 0, nullptr, 0,
        nullptr, 0, {}, &arena_, &realization_.resources, &valid));
    ASSERT_TRUE(valid);
    IREE_ASSERT_OK(loom_pipeline_invocations_build(
        module_, realization_.function, &facts_, &realization_.resources,
        nullptr, 0, {}, {}, &arena_, &realization_.invocations, &valid));
    ASSERT_TRUE(valid);
    ASSERT_EQ(realization_.invocations.count, 5u);
    IREE_ASSERT_OK(loom_pipeline_invocations_analyze_workers(
        module_, &realization_.invocations, nullptr, {}, &arena_,
        &realization_.workers, &valid));
    ASSERT_TRUE(valid);
    const size_t worker_count = HasAutonomousPreparation() ? 4 : 2;
    ASSERT_EQ(realization_.invocations.worker_count, worker_count);
    realization_.instances = instances_;
    loom_builder_enter_region(&rewriter_.builder, nullptr, module_->body);
    for (size_t i = 0; i < worker_count; ++i) {
      if (engines_[i] == IREE_HOST_SIZE_MAX) {
        continue;
      }
      auto* source = &realization_.workers[i];
      loom_local_value_domain_restore(&source->value_domain);
      IREE_ASSERT_OK(loom_pipeline_worker_materialization_clone(
          &rewriter_, source, Symbol(i ? "second_work" : "first_work"),
          &instances_[i]));
      loom_local_value_domain_release(&source->value_domain);
    }
    loom_pipeline_execution_conflict_t conflict;
    IREE_ASSERT_OK(loom_pipeline_execution_build(&realization_.invocations,
                                                 engines_, 1, &arena_,
                                                 &execution_, &conflict));
    EXPECT_EQ(conflict.second_worker, IREE_HOST_SIZE_MAX);
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  loom_symbol_ref_t Symbol(const char* name) {
    loom_string_id_t string;
    loom_symbol_id_t symbol;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &string));
    IREE_CHECK_OK(loom_module_add_symbol(module_, string, &symbol));
    return {0, symbol};
  }

  loom_symbol_ref_t Function(const char* name, const loom_type_t* types,
                             size_t count) {
    auto* builder = &rewriter_.builder;
    loom_builder_enter_region(builder, nullptr, module_->body);
    const auto symbol = Symbol(name);
    loom_op_t* function;
    IREE_CHECK_OK(loom_func_def_build(
        builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, types,
        count, nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &function));
    loom_builder_enter_region(builder, function, loom_func_def_body(function));
    loom_op_t* op;
    IREE_CHECK_OK(loom_func_return_build(builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &op));
    return symbol;
  }

  loom_op_t* Pipeline(const char* name) {
    auto* builder = &rewriter_.builder;
    loom_builder_enter_region(builder, nullptr, module_->body);
    // A source argument deliberately precedes the separately supplied state.
    const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* definition;
    IREE_CHECK_OK(loom_pipeline_def_build(
        builder, 0, 0, 0, 0, loom_symbol_ref_null(), Symbol(name), nullptr, 0,
        &index, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN, &definition));
    loom_builder_enter_region(builder, definition,
                              loom_pipeline_def_body(definition));
    return definition;
  }

  void Compose(loom_op_t* parent, loom_op_t* child, size_t count) {
    auto* builder = &rewriter_.builder;
    const auto value =
        loom_region_entry_block(loom_pipeline_def_body(parent))->arg_ids[0];
    loom_op_t* serial;
    IREE_ASSERT_OK(
        loom_pipeline_serial_build(builder, LOOM_LOCATION_UNKNOWN, &serial));
    loom_builder_enter_region(builder, serial,
                              loom_pipeline_serial_body(serial));
    for (size_t i = 0; i < count; ++i) {
      loom_op_t* call;
      IREE_ASSERT_OK(loom_pipeline_compose_build(
          builder, loom_pipeline_def_callee(child), nullptr, 0, &value, 1,
          LOOM_LOCATION_UNKNOWN, &call));
    }
    loom_op_t* yield;
    IREE_ASSERT_OK(
        loom_pipeline_yield_build(builder, LOOM_LOCATION_UNKNOWN, &yield));
  }

  void Finish(loom_op_t* pipeline) {
    auto* builder = &rewriter_.builder;
    loom_builder_enter_region(builder, pipeline,
                              loom_pipeline_def_body(pipeline));
    loom_op_t* op;
    IREE_ASSERT_OK(
        loom_pipeline_finish_build(builder, LOOM_LOCATION_UNKNOWN, &op));
  }

  void Verify() {
    const loom_verify_options_t options = {
        .sink = {.fn = loom_diagnostic_stderr_sink},
    };
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  static iree_status_t Boundary(
      void* user_data, loom_builder_t* builder, iree_host_size_t engine,
      const loom_pipeline_invocation_schedule_t* scope,
      const loom_pipeline_invocation_t* child, bool before,
      const loom_value_id_t* state) {
    auto* self = static_cast<PipelineControlTest*>(user_data);
    EXPECT_EQ(engine, 0u);
    EXPECT_NE(scope, nullptr);
    EXPECT_NE(child, nullptr);
    ++self->boundary_count_;
    self->entry_count_ += before;
    // Consume both states in reversed order so the final operand wiring proves
    // that source-argument pruning did not shift either boundary operand.
    const loom_value_id_t arguments[] = {state[1], state[0]};
    loom_op_t* call;
    return loom_func_call_build(builder, 0, 0, 0, 0, self->observer_, arguments,
                                2, nullptr, 0, nullptr, 0,
                                LOOM_LOCATION_UNKNOWN, &call);
  }

  static iree_status_t Prepare(void* user_data, loom_builder_t* builder,
                               iree_host_size_t worker,
                               loom_value_slice_t captures) {
    auto* self = static_cast<PipelineControlTest*>(user_data);
    EXPECT_TRUE(worker == 1 || worker == 3);
    EXPECT_EQ(captures.count, 2u);
    ++self->preparation_count_;
    const loom_value_id_t operands[] = {captures.values[1], captures.values[0]};
    loom_op_t* call;
    return loom_func_call_build(builder, 0, 0, 0, 0, self->capture_observer_,
                                operands, 2, nullptr, 0, nullptr, 0,
                                LOOM_LOCATION_UNKNOWN, &call);
  }

  // Common backing for source and retained plans.
  iree_arena_block_pool_t blocks_;
  // Analysis and cloning storage.
  iree_arena_allocator_t arena_;
  // Source dialect registry.
  loom_context_t context_;
  // Original and projected call graphs.
  loom_module_t* module_ = nullptr;
  // Normal construction projection rewriter.
  loom_rewriter_t rewriter_;
  // Immutable source facts.
  loom_value_fact_table_t facts_ = {};
  // Real invocation inventory and analyzed workers.
  loom_pipeline_realization_t realization_ = {};
  // Independent cloned worker bodies.
  loom_pipeline_worker_materialization_t instances_[4] = {};
  // Both serial roles run on one selected engine.
  iree_host_size_t engines_[4] = {0, 0, 0, 0};
  // Retained engine participation and serial scopes.
  loom_pipeline_execution_t execution_ = {};
  // Ordinary operation used to make boundary operands observable in emitted IR.
  loom_symbol_ref_t observer_;
  // Preparation observer makes the projected source capture order visible.
  loom_symbol_ref_t capture_observer_;
  // Number of emitted before/after boundaries.
  size_t boundary_count_ = 0;
  // Number of construction-entry boundaries.
  size_t entry_count_ = 0;
  // Each autonomous occurrence prepares once in its selected engine.
  size_t preparation_count_ = 0;
};

TEST_F(PipelineControlTest, NestedBoundariesKeepTheirProjectedRuntimeState) {
  const loom_type_t types[] = {loom_type_buffer(),
                               loom_type_scalar(LOOM_SCALAR_TYPE_I32)};
  const loom_pipeline_control_options_t options = {
      .worker_engines = engines_,
      .state_types = types,
      .state_count = 2,
      .emit = {.boundary = Boundary, .user_data = this},
  };
  loom_pipeline_control_function_t* functions;
  IREE_ASSERT_OK(loom_pipeline_control_materialize(
      &rewriter_, &realization_, &execution_, &options, &functions));
  EXPECT_EQ(boundary_count_, 8u);
  EXPECT_EQ(entry_count_, 4u);
  for (size_t owner : {0u, 1u, 3u}) {
    const auto& projected = functions[owner];
    ASSERT_EQ(projected.argument_count, 2u);
    EXPECT_EQ(projected.arguments[0], 1u);
    EXPECT_EQ(projected.arguments[1], 2u);
    auto* body = loom_func_like_body(projected.function);
    const auto* entry = loom_region_entry_block(body);
    ASSERT_EQ(entry->arg_count, 2u);
    size_t observations = 0;
    for (iree_host_size_t b = 0; b < body->block_count; ++b) {
      auto* block = body->blocks[b];
      for (auto* op = block->first_op; op; op = op->next_op) {
        if (loom_func_call_isa(op) &&
            loom_func_call_callee(op).symbol_id == observer_.symbol_id) {
          const auto operands = loom_func_call_operands(op);
          ASSERT_EQ(operands.count, 2u);
          EXPECT_EQ(operands.values[0], entry->arg_ids[1]);
          EXPECT_EQ(operands.values[1], entry->arg_ids[0]);
          ++observations;
        }
      }
    }
    EXPECT_EQ(observations, owner == 0 ? 4u : 2u);
  }
  Verify();
}

class PipelinePreparationTest : public PipelineControlTest {
 protected:
  bool HasAutonomousPreparation() const override { return true; }
};

TEST_F(PipelinePreparationTest, CapturesPrepareBeforeConsumersAcrossReentry) {
  const iree_host_size_t preparation[] = {IREE_HOST_SIZE_MAX, 0,
                                          IREE_HOST_SIZE_MAX, 0};
  const loom_pipeline_control_options_t options = {
      .worker_engines = engines_,
      .preparation_engines = preparation,
      .emit = {.prepare_worker = Prepare, .user_data = this},
  };
  loom_pipeline_control_function_t* functions;
  IREE_ASSERT_OK(loom_pipeline_control_materialize(
      &rewriter_, &realization_, &execution_, &options, &functions));
  EXPECT_EQ(preparation_count_, 2u);
  ASSERT_EQ(functions[0].argument_count, 1u);
  EXPECT_EQ(functions[0].arguments[0], 0u);
  for (size_t leaf : {2u, 4u}) {
    auto* body = loom_func_like_body(functions[leaf].function);
    const auto* entry = loom_region_entry_block(body);
    ASSERT_EQ(entry->arg_count, 1u);
    const auto* preparation_call = entry->first_op;
    ASSERT_TRUE(loom_func_call_isa(preparation_call));
    const auto actuals = loom_func_call_operands(preparation_call);
    ASSERT_EQ(actuals.count, 1u);
    EXPECT_EQ(actuals.values[0], entry->arg_ids[0]);
    const auto helper = loom_func_call_callee(preparation_call);
    auto* helper_op = module_->symbols.entries[helper.symbol_id].defining_op;
    auto* prepared = loom_func_def_body(helper_op);
    const auto* prepared_entry = loom_region_entry_block(prepared);
    size_t observed = 0;
    for (size_t b = 0; b < prepared->block_count; ++b) {
      for (auto* op = prepared->blocks[b]->first_op; op; op = op->next_op) {
        if (!loom_func_call_isa(op)) {
          continue;
        }
        EXPECT_EQ(loom_func_call_callee(op).symbol_id,
                  capture_observer_.symbol_id);
        const auto operands = loom_func_call_operands(op);
        ASSERT_EQ(operands.count, 2u);
        EXPECT_EQ(operands.values[1], prepared_entry->arg_ids[0]);
        const auto* sum =
            loom_value_def_op(loom_module_value(module_, operands.values[0]));
        ASSERT_TRUE(loom_index_add_isa(sum));
        EXPECT_EQ(loom_index_add_lhs(sum), prepared_entry->arg_ids[0]);
        const auto* constant = loom_value_def_op(
            loom_module_value(module_, loom_index_add_rhs(sum)));
        ASSERT_TRUE(loom_index_constant_isa(constant));
        EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(constant)), 7);
        ++observed;
      }
    }
    EXPECT_EQ(observed, 1u);
    ASSERT_TRUE(loom_func_call_isa(body->blocks[1]->first_op));
    EXPECT_NE(loom_func_call_callee(body->blocks[1]->first_op).symbol_id,
              capture_observer_.symbol_id);
  }
  Verify();
}

}  // namespace
