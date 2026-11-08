// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/realization.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/target/pass_environment.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

// The callback selects instruction bodies or offloaded mechanics for two
// independent workers. This exercises cloning and version ownership through
// the real occurrence analysis; native artifacts cover physical emission.
class PipelineRealizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_pass_value_fact_owner_initialize(&blocks_, &fact_owner_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&context_));
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    Register(LOOM_DIALECT_PIPELINE, loom_pipeline_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("realization"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    const auto worker = Symbol("worker");
    worker_version_.base.function = Function(&builder_, worker);
    worker_version_.base.type = &loom_target_function_version_type;
    worker_version_.target_context_ordinal = 3;
    const auto source = Symbol("source");
    loom_op_t* definition;
    IREE_ASSERT_OK(loom_pipeline_def_build(
        &builder_, 0, 0, 0, 0, loom_symbol_ref_null(), source, nullptr, 0,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &definition));
    source_version_.base.type = &loom_target_function_version_type;
    source_version_.base.function = loom_func_like_cast(module_, definition);
    source_version_.base.flags = LOOM_FUNCTION_VERSION_FLAG_RETAIN;
    source_version_.target_context_ordinal = 2;
    loom_builder_enter_region(&builder_, definition,
                              loom_pipeline_def_body(definition));
    const int64_t count = 1;
    for (int64_t origin = 0; origin < 2; ++origin) {
      loom_builder_enter_region(&builder_, definition,
                                loom_pipeline_def_body(definition));
      loom_op_t* strand;
      IREE_ASSERT_OK(loom_pipeline_strand_build(
          &builder_, 0, loom_symbol_ref_null(), nullptr, 0, &origin, 1, nullptr,
          0, &count, 1, nullptr, 0, &count, 1, LOOM_LOCATION_UNKNOWN, &strand));
      loom_builder_enter_region(&builder_, strand,
                                loom_pipeline_strand_body(strand));
      loom_op_t* call;
      IREE_ASSERT_OK(loom_func_call_build(&builder_, 0, 0, 0, 0, worker,
                                          nullptr, 0, nullptr, 0, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &call));
      loom_op_t* terminator;
      IREE_ASSERT_OK(loom_pipeline_end_build(&builder_, LOOM_LOCATION_UNKNOWN,
                                             &terminator));
    }
    loom_op_t* terminator;
    loom_builder_enter_region(&builder_, definition,
                              loom_pipeline_def_body(definition));
    IREE_ASSERT_OK(loom_pipeline_finish_build(&builder_, LOOM_LOCATION_UNKNOWN,
                                              &terminator));
    loom_builder_set_block(&builder_, loom_module_block(module_));
    builder_.ip.parent_op = nullptr;
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 0));
    loom_type_registry_configure_fact_context(&facts_.context);
    IREE_ASSERT_OK(loom_value_fact_table_compute(
        &facts_, module_, source_version_.base.function));
    bool valid = false;
    IREE_ASSERT_OK(loom_pipeline_resources_build(
        module_, source_version_.base.function, &facts_, nullptr, 0, nullptr, 0,
        nullptr, 0, {}, &arena_, &resources_, &valid));
    ASSERT_TRUE(valid);
    ASSERT_EQ(resources_.strand_count, 2u);
    IREE_ASSERT_OK(loom_pipeline_invocations_build(
        module_, source_version_.base.function, &facts_, &resources_, nullptr,
        0, {}, {}, &arena_, &invocations_, &valid));
    ASSERT_TRUE(valid);
    IREE_ASSERT_OK(loom_pipeline_invocations_analyze_workers(
        module_, &invocations_, nullptr, {}, &arena_, &source_workers_,
        &valid));
    ASSERT_TRUE(valid);
    loom_function_version_owner_initialize(&arena_, &versions_);
    IREE_ASSERT_OK(
        loom_function_version_owner_append(&versions_, &source_version_.base));
    IREE_ASSERT_OK(
        loom_function_version_owner_append(&versions_, &worker_version_.base));
    capability_ =
        loom_target_pass_capability_make_mutable(nullptr, &versions_, nullptr);
    capabilities_[0] = &capability_.base;
    environment_ = loom_pass_environment_make(capabilities_, 1);
    pass_.arena = &arena_;
    pass_.value_facts = &fact_owner_;
    pass_.environment = &environment_;
    pass_.function_version = &source_version_.base;
    IREE_ASSERT_OK(loom_target_function_version_snapshot_build(
        module_, &versions_.list, &arena_, &snapshot_));
    Verify();
  }

  void TearDown() override {
    loom_pass_value_fact_owner_deinitialize(&fact_owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, dialect, vtables, count));
  }

  loom_symbol_ref_t Symbol(const char* name) {
    loom_string_id_t string;
    loom_symbol_id_t symbol;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &string));
    IREE_CHECK_OK(loom_module_add_symbol(module_, string, &symbol));
    return {0, symbol};
  }

  loom_func_like_t Function(loom_builder_t* builder, loom_symbol_ref_t symbol) {
    loom_builder_set_block(builder, loom_module_block(module_));
    builder->ip.parent_op = nullptr;
    loom_op_t* function;
    IREE_CHECK_OK(loom_func_def_build(
        builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, nullptr,
        0, nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &function));
    loom_builder_enter_region(builder, function, loom_func_def_body(function));
    loom_op_t* terminator;
    IREE_CHECK_OK(loom_func_return_build(builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));
    loom_builder_set_block(builder, loom_module_block(module_));
    builder->ip.parent_op = nullptr;
    return loom_func_like_cast(module_, function);
  }

  void Verify() {
    const loom_verify_options_t options = {
        .sink = {.fn = loom_diagnostic_stderr_sink},
    };
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  static iree_status_t Select(void* data, loom_module_t*,
                              const loom_pipeline_realization_t* realization,
                              bool* valid) {
    auto* self = static_cast<PipelineRealizationTest*>(data);
    EXPECT_EQ(self->emitted_, 0u);
    EXPECT_EQ(realization->instances, nullptr);
    EXPECT_EQ(realization->workers, self->source_workers_);
    ++self->selected_;
    *valid = true;
    return iree_ok_status();
  }

  static bool HasWorkerBody(const void* data, iree_host_size_t index) {
    return static_cast<const PipelineRealizationTest*>(data)->has_body_[index];
  }

  static iree_status_t Worker(void* data, loom_rewriter_t*,
                              const loom_pipeline_realization_t* realization,
                              iree_host_size_t index) {
    auto* self = static_cast<PipelineRealizationTest*>(data);
    EXPECT_EQ(self->selected_, 1u);
    EXPECT_TRUE(self->has_body_[index]);
    const auto* instance = &realization->instances[index].channels;
    EXPECT_TRUE(loom_local_value_domain_is_acquired(&instance->value_domain));
    EXPECT_FALSE(loom_local_value_domain_is_acquired(
        &realization->workers[index].value_domain));
    EXPECT_NE(instance->function.op, realization->workers[index].function.op);
    EXPECT_EQ(realization->worker_versions[index]->base.function.op,
              instance->function.op);
    EXPECT_EQ(realization->worker_versions[index]->target_context_ordinal, 3);
    self->bodies_[index] = instance->function.op;
    ++self->emitted_;
    return iree_ok_status();
  }

  static iree_status_t Entry(void* data, loom_rewriter_t* rewriter,
                             const loom_pipeline_realization_t* realization) {
    auto* self = static_cast<PipelineRealizationTest*>(data);
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(realization->instances[i].channels.function.op,
                self->bodies_[i]);
      if (!self->has_body_[i]) {
        EXPECT_EQ(realization->worker_versions[i], nullptr);
      }
    }
    const auto symbol = loom_func_like_callee(realization->function);
    IREE_RETURN_IF_ERROR(
        loom_rewriter_erase(rewriter, realization->function.op));
    self->source_version_.base.function =
        self->Function(&rewriter->builder, symbol);
    return iree_ok_status();
  }

  void Run() {
    const auto symbols = module_->symbols.count;
    const auto worker = worker_version_.base.function;
    const loom_pipeline_realization_callback_t callback = {
        Select, HasWorkerBody, Worker, Entry, this};
    IREE_ASSERT_OK(loom_pipeline_realize_invocations(
        &pass_, module_, &invocations_, source_workers_, &snapshot_,
        &callback));
    EXPECT_EQ(selected_, 1u);
    EXPECT_EQ(emitted_, size_t(has_body_[0]) + size_t(has_body_[1]));
    EXPECT_EQ(module_->symbols.count, symbols + emitted_);
    EXPECT_EQ(versions_.list.count, 2u + emitted_);
    EXPECT_EQ(worker_version_.base.function.op, worker.op);
    EXPECT_NE(worker.op->parent_block, nullptr);
    Verify();
  }

  // Arena and module own all source and consuming instances throughout the
  // test.
  iree_arena_block_pool_t blocks_;
  // Retained analysis and compiler-version storage.
  iree_arena_allocator_t arena_;
  // Dialect vocabulary for the ordinary source and emitted entry functions.
  loom_context_t context_;
  // Source definitions remain available until every occurrence is consumed.
  loom_module_t* module_ = nullptr;
  // Ordinary source construction helper.
  loom_builder_t builder_;
  // Immutable source facts.
  loom_value_fact_table_t facts_;
  // Immutable source placements.
  loom_pipeline_resources_t resources_;
  // Retained analysis owner with closed source proofs.
  loom_pipeline_invocations_t invocations_;
  // Two source occurrences sharing one immutable worker function.
  loom_pipeline_worker_t* source_workers_ = nullptr;
  // Stable source entry identity.
  loom_target_function_version_t source_version_ = {};
  // Stable source worker identity.
  loom_target_function_version_t worker_version_ = {};
  // Compilation owner that receives emitted worker and entry versions.
  loom_function_version_owner_t versions_;
  // Normal mutable target capability supplied by the compilation owner.
  loom_target_pass_capability_t capability_;
  // Borrowed capability list for the pass frame.
  const loom_pass_environment_capability_t* capabilities_[1];
  // Target capabilities retained for the realization pass.
  loom_pass_environment_t environment_;
  // Normal pass fact cache, invalidated at the consuming rewrite boundary.
  loom_pass_value_fact_owner_t fact_owner_;
  // Normal pass frame consumed by realization.
  loom_pass_t pass_ = {};
  // Source version inventory reused throughout selection and cloning.
  loom_target_function_version_snapshot_t snapshot_ = {};
  // Number of physical selections observed before mutation.
  size_t selected_ = 0;
  // Number of instruction-bearing workers visited during emission.
  size_t emitted_ = 0;
  // Selected instruction-body requirement for each source occurrence.
  bool has_body_[2] = {true, true};
  // Independent consuming bodies; offloaded occurrences remain null.
  loom_op_t* bodies_[2] = {};
};

TEST_F(PipelineRealizationTest, SelectedBodiesOwnIndependentVersions) {
  Run();
  ASSERT_NE(bodies_[0], nullptr);
  ASSERT_NE(bodies_[1], nullptr);
  EXPECT_NE(bodies_[0], bodies_[1]);
}

TEST_F(PipelineRealizationTest, OffloadedWorkersNeedNoBodiesOrVersions) {
  has_body_[0] = has_body_[1] = false;
  Run();
  EXPECT_EQ(bodies_[0], nullptr);
  EXPECT_EQ(bodies_[1], nullptr);
}

TEST_F(PipelineRealizationTest, OmittingBodyPreservesLaterRoleIdentity) {
  has_body_[0] = false;
  Run();
  EXPECT_EQ(bodies_[0], nullptr);
  ASSERT_NE(bodies_[1], nullptr);
  const auto symbol =
      loom_func_like_callee(loom_func_like_cast(module_, bodies_[1]));
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&module_->strings,
                            module_->symbols.entries[symbol.symbol_id].name_id),
      IREE_SV("__pipeline_worker_1")));
}

}  // namespace
}  // namespace loom
