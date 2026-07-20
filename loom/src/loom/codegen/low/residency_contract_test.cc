// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/residency_contract.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/registers.h"
#include "loom/target/reporting/low.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

typedef struct DiagnosticCapture {
  // Number of diagnostic emissions observed.
  uint32_t count;
  // Most recently emitted structured error definition.
  const loom_error_def_t* last_error;
} DiagnosticCapture;

typedef struct ResidencyModelStorage {
  // Stable names indexed by descriptor register class.
  std::vector<iree_string_view_t> resource_names;
  // Dense cliff ranges indexed by descriptor register class.
  std::vector<loom_target_residency_cliff_range_t> cliff_ranges;
  // Single cliff applied to the root value register class.
  loom_target_residency_cliff_t cliff;
  // Model borrowing the vectors and cliff above.
  loom_target_residency_model_t model;
} ResidencyModelStorage;

template <typename T>
static const T* CompileReportRowAt(
    const loom_target_compile_report_row_list_t& row_list,
    iree_host_size_t index) {
  for (const loom_target_compile_report_vec_t* vec = row_list.head; vec != NULL;
       vec = vec->next) {
    if (index < vec->count) {
      const T* rows =
          static_cast<const T*>(loom_target_compile_report_vec_const_rows(vec));
      return &rows[index];
    }
    index -= vec->count;
  }
  return nullptr;
}

static iree_status_t CaptureDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  DiagnosticCapture* capture = static_cast<DiagnosticCapture*>(user_data);
  ++capture->count;
  capture->last_error = emission->error;
  return iree_ok_status();
}

class LowResidencyContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModuleSource(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    loom_low_descriptor_text_asm_environment_initialize(
        &target_registry_.registry, &options.low_asm_environment);
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("residency_contract_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  ModulePtr ParseModule() {
    return ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @nested_candidates(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require 3
  %seed = test.const.i32 7
  %outer = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  %early = test.add.i32 %outer, %input
  %inner = low.residency.candidate 8 1 %outer : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  %late = test.add.i32 %inner, %early
  return %late
}
)");
  }

  ModulePtr ParseRepairModule() {
    return ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @repair_candidate(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require 3
  %seed = test.const.i32 7
  %outer = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  %early = test.add.i32 %outer, %input
  %inner = low.residency.candidate 8 1 %outer : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  %a = test.const.i32 11
  %b = test.const.i32 13
  %middle = test.add.i32 %a, %b
  %late = test.add.i32 %inner, %middle
  %result = test.add.i32 %late, %early
  return %result
}
)");
  }

  ModulePtr ParsePreserveRepairModule() {
    return ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @repair_candidate(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require -1 {preserve = true, projected_baseline = 4}
  %seed = test.const.i32 7
  %outer = low.residency.candidate 7 2 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %early = test.add.i32 %outer, %input
  %inner = low.residency.candidate 8 1 %outer : reg<test.i32> recipe(%seed : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %a = test.const.i32 11
  %b = test.const.i32 13
  %middle = test.add.i32 %a, %b
  %late = test.add.i32 %inner, %middle
  %result = test.add.i32 %late, %early
  return %result
}
)");
  }

  ModulePtr ParsePreserveMinimumRepairModule() {
    return ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @repair_candidate(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require 3 {preserve = true, projected_baseline = 4}
  %seed = test.const.i32 7
  %outer = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  %early = test.add.i32 %outer, %input
  %inner = low.residency.candidate 8 2 %outer : reg<test.i32> recipe(%seed : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %a = test.const.i32 11
  %b = test.const.i32 13
  %middle = test.add.i32 %a, %b
  %late = test.add.i32 %inner, %middle
  %result = test.add.i32 %late, %early
  return %result
}
)");
  }

  loom_op_t* FindLowFunction(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_region_entry_block(module->body), op) {
      if (loom_low_func_def_isa(op)) return op;
    }
    ADD_FAILURE() << "low function not found";
    return nullptr;
  }

  std::vector<loom_op_t*> FindDescriptorPackets(loom_op_t* function_op) {
    std::vector<loom_op_t*> ops;
    loom_op_t* op = nullptr;
    loom_block_for_each_op(
        loom_region_entry_block(loom_low_func_def_body(function_op)), op) {
      if (loom_low_op_isa(op) || loom_low_const_isa(op)) ops.push_back(op);
    }
    return ops;
  }

  void InitializeResidencyModel(loom_module_t* module, loom_op_t* function_op,
                                uint32_t cliff_units,
                                ResidencyModelStorage* storage) {
    std::vector<loom_op_t*> descriptor_ops = FindDescriptorPackets(function_op);
    ASSERT_FALSE(descriptor_ops.empty());
    ASSERT_TRUE(loom_low_const_isa(descriptor_ops[0]));
    const loom_type_t root_type = loom_module_value_type(
        module, loom_low_const_result(descriptor_ops[0]));
    const uint16_t root_resource_id =
        loom_low_register_type_class_id(root_type);
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_lookup(&target_registry_.registry,
                                            IREE_SV("test.low.core"));
    ASSERT_NE(descriptor_set, nullptr);
    ASSERT_LE(descriptor_set->reg_class_count, UINT16_MAX);
    ASSERT_LT(root_resource_id, descriptor_set->reg_class_count);

    storage->resource_names.assign(descriptor_set->reg_class_count,
                                   IREE_SV("register-class"));
    storage->cliff_ranges.assign(descriptor_set->reg_class_count, {});
    storage->cliff = {
        /*.resource_id=*/root_resource_id,
        /*.cliff_units=*/cliff_units,
        /*.tier_before=*/4,
        /*.tier_after=*/2,
    };
    for (uint16_t i = 0; i < descriptor_set->reg_class_count; ++i) {
      storage->cliff_ranges[i].start = i <= root_resource_id ? 0 : 1;
      storage->cliff_ranges[i].count = i == root_resource_id ? 1 : 0;
    }
    storage->model = {
        /*.best_tier=*/4,
        /*.direct_resources=*/
        {
            /*.names=*/storage->resource_names.data(),
            /*.cliffs=*/&storage->cliff,
            /*.cliff_count=*/1,
            /*.cliff_ranges=*/storage->cliff_ranges.data(),
            /*.resource_count=*/
            static_cast<uint16_t>(descriptor_set->reg_class_count),
        },
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(LowResidencyContractTest, ConsumesNestedMarkersIntoExactLeafUses) {
  ModulePtr module = ParseModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  loom_op_t* first_candidate = nullptr;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(
      loom_region_entry_block(loom_low_func_def_body(function_op)), op) {
    if (loom_low_residency_candidate_isa(op)) {
      first_candidate = op;
      break;
    }
  }
  ASSERT_NE(first_candidate, nullptr);
  ASSERT_EQ(loom_low_storage_relation_count(module.get(), first_candidate), 1u);
  loom_low_storage_relation_t relation = {};
  loom_low_storage_relation_get(module.get(), first_candidate, 0, &relation);
  EXPECT_EQ(relation.destination_value_id,
            loom_low_residency_candidate_result(first_candidate));
  EXPECT_EQ(relation.source_value_id,
            loom_low_residency_candidate_source(first_candidate));
  EXPECT_EQ(relation.kind, LOOM_LOW_STORAGE_RELATION_SAME_STORAGE);
  EXPECT_EQ(relation.cause,
            LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_RESIDENCY_CANDIDATE);
  EXPECT_EQ(relation.flags, LOOM_LOW_STORAGE_RELATION_FLAG_HARD);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));

  EXPECT_TRUE(contract.has_requirement);
  EXPECT_EQ(contract.required_tier, 3u);
  EXPECT_TRUE(contract.has_minimum_requirement);
  EXPECT_FALSE(contract.preserves_baseline);
  ASSERT_EQ(contract.candidate_count, 2u);
  EXPECT_EQ(contract.candidates[0].candidate_id, 7u);
  EXPECT_EQ(contract.candidates[0].recompute_cost, 1u);
  EXPECT_EQ(contract.candidates[1].candidate_id, 8u);
  EXPECT_EQ(contract.candidates[1].recompute_cost, 1u);
  EXPECT_EQ(contract.candidates[0].value_id, contract.candidates[1].value_id);
  EXPECT_EQ(contract.candidates[0].use_count, 2u);
  EXPECT_EQ(contract.candidates[1].use_count, 1u);

  std::vector<loom_op_t*> descriptor_ops = FindDescriptorPackets(function_op);
  ASSERT_EQ(descriptor_ops.size(), 3u);
  const loom_value_id_t root_value_id = contract.candidates[0].value_id;
  EXPECT_EQ(loom_op_operands(descriptor_ops[1])[0], root_value_id);
  EXPECT_EQ(loom_op_operands(descriptor_ops[2])[0], root_value_id);

  op = nullptr;
  loom_block_for_each_op(
      loom_region_entry_block(loom_low_func_def_body(function_op)), op) {
    EXPECT_FALSE(loom_low_residency_require_isa(op));
    EXPECT_FALSE(loom_low_residency_candidate_isa(op));
  }
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, IgnoresCandidateProofMetadataAsLeafUses) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @metadata_uses() -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require 3
  %seed = test.const.i32 7
  %outer = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  %inner = low.residency.candidate 8 1 %outer : reg<test.i32> captures(%outer : reg<test.i32>) {sealed = true}
  %result = test.add.i32 %inner, %outer
  return %result
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));

  ASSERT_EQ(contract.candidate_count, 2u);
  EXPECT_EQ(contract.candidates[0].use_count, 2u);
  EXPECT_EQ(contract.candidates[1].use_count, 1u);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       ConsumesDominatingRecipeAcrossNestedCandidateBlocks) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @cross_block_candidates(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require 3
  %seed = test.const.i32 7
  %outer = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  low.br ^body
^body:
  %inner = low.residency.candidate 8 1 %outer : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  %result = test.add.i32 %inner, %input
  return %result
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));

  ASSERT_EQ(contract.candidate_count, 2u);
  EXPECT_EQ(contract.candidates[0].value_id, contract.candidates[1].value_id);
  ASSERT_EQ(contract.candidates[0].materialization_op_count, 1u);
  ASSERT_EQ(contract.candidates[1].materialization_op_count, 1u);
  EXPECT_EQ(contract.candidates[0].materialization_ops[0],
            contract.candidates[1].materialization_ops[0]);
  EXPECT_EQ(contract.candidates[0].use_count, 1u);
  EXPECT_EQ(contract.candidates[1].use_count, 1u);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, DropsCandidateWithNoSurvivingUses) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @stale_candidate() -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require 3
  %seed = test.const.i32 7
  %stale = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {sealed = true}
  return %seed
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));

  EXPECT_TRUE(contract.has_requirement);
  EXPECT_EQ(contract.required_tier, 3u);
  EXPECT_EQ(contract.candidate_count, 0u);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(
      loom_region_entry_block(loom_low_func_def_body(function_op)), op) {
    EXPECT_FALSE(loom_low_residency_require_isa(op));
    EXPECT_FALSE(loom_low_residency_candidate_isa(op));
  }
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, RejectsProducerSliceWithUnrecordedCapture) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @stale_recipe(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require 3
  %root = test.add.i32 %lhs, %rhs
  %placed = low.residency.candidate 7 1 %root : reg<test.i32> captures(%lhs : reg<test.i32>) recipe(%root : reg<test.i32>) {sealed = true}
  return %placed
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_low_residency_contract_consume(
                            module.get(), function_op, &arena, &contract));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       ConsumesUnrolledInstancesWithSharedSourceCandidateIdentity) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @unrolled_instances(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require -1 {preserve = true, projected_baseline = 4}
  %seed = test.const.i32 7
  %placed0 = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %use0 = test.add.i32 %placed0, %input
  %placed1 = low.residency.candidate 7 1 %seed : reg<test.i32> recipe(%seed : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %use1 = test.add.i32 %placed1, %use0
  return %use1
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));

  EXPECT_TRUE(contract.preserves_baseline);
  ASSERT_EQ(contract.candidate_count, 2u);
  EXPECT_EQ(contract.candidates[0].candidate_id, 7u);
  EXPECT_EQ(contract.candidates[1].candidate_id, 7u);
  EXPECT_EQ(contract.candidates[0].value_id, contract.candidates[1].value_id);
  EXPECT_EQ(contract.candidates[0].use_count, 1u);
  EXPECT_EQ(contract.candidates[1].use_count, 1u);
  EXPECT_TRUE(contract.candidates[0].preserves_baseline);
  EXPECT_TRUE(contract.candidates[1].preserves_baseline);
  EXPECT_NE(loom_use_user_op(contract.candidates[0].uses[0]),
            loom_use_user_op(contract.candidates[1].uses[0]));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, EvaluatesExactAllocatedExtent) {
  ModulePtr module = ParseModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));

  static const iree_string_view_t kResourceNames[] = {IREE_SVL("registers")};
  static const loom_target_residency_cliff_t kCliffs[] = {{
      /*.resource_id=*/0,
      /*.cliff_units=*/5,
      /*.tier_before=*/4,
      /*.tier_after=*/2,
  }};
  static const loom_target_residency_cliff_range_t kCliffRanges[] = {{
      /*.start=*/0,
      /*.count=*/1,
  }};
  static const loom_target_residency_model_t kModel = {
      /*.best_tier=*/4,
      /*.direct_resources=*/
      {
          /*.names=*/kResourceNames,
          /*.cliffs=*/kCliffs,
          /*.cliff_count=*/IREE_ARRAYSIZE(kCliffs),
          /*.cliff_ranges=*/kCliffRanges,
          /*.resource_count=*/IREE_ARRAYSIZE(kResourceNames),
      },
  };
  uint32_t assigned_extent = 4;
  loom_low_allocation_table_t allocation = {};
  allocation.assigned_extents.ends_by_reg_class = &assigned_extent;
  allocation.assigned_extents.count = 1;

  uint32_t tier = 0;
  bool satisfied = false;
  IREE_ASSERT_OK(loom_low_residency_contract_evaluate(
      &contract, &kModel, &allocation, &arena, &tier, &satisfied));
  EXPECT_EQ(tier, 4u);
  EXPECT_TRUE(satisfied);

  assigned_extent = 5;
  IREE_ASSERT_OK(loom_low_residency_contract_evaluate(
      &contract, &kModel, &allocation, &arena, &tier, &satisfied));
  EXPECT_EQ(tier, 2u);
  EXPECT_FALSE(satisfied);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, RepairsOnlyRecordedInnerBoundaryUses) {
  ModulePtr module = ParseModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));
  ASSERT_EQ(contract.candidate_count, 2u);
  const loom_value_id_t root_value_id = contract.candidates[0].value_id;

  loom_low_allocation_assignment_t assignment = {
      /*.value_id=*/root_value_id,
  };
  loom_low_allocation_table_t allocation = {};
  allocation.module = module.get();
  allocation.function_op = function_op;
  allocation.assignments = &assignment;
  allocation.assignment_count = 1;
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), function_op, &target_registry_.registry,
      loom_target_selection_empty(), iree_diagnostic_emitter_t{},
      &allocation.target));

  loom_low_allocation_rematerialization_result_t result = {};
  IREE_ASSERT_OK(loom_low_residency_contract_try_repair(
      module.get(), &allocation, &contract,
      LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES, &arena, &result));
  EXPECT_EQ(result.value_id, root_value_id);
  EXPECT_EQ(result.assignment_index, 0u);
  EXPECT_EQ(result.cloned_packet_count, 1u);
  EXPECT_EQ(result.rewritten_operand_count, 1u);

  std::vector<loom_op_t*> descriptor_ops = FindDescriptorPackets(function_op);
  ASSERT_EQ(descriptor_ops.size(), 4u);
  loom_op_t* early_add = descriptor_ops[1];
  loom_op_t* cloned_constant = descriptor_ops[2];
  loom_op_t* late_add = descriptor_ops[3];
  EXPECT_EQ(loom_op_operands(early_add)[0], root_value_id);
  EXPECT_TRUE(loom_low_const_isa(cloned_constant));
  EXPECT_NE(loom_op_operands(late_add)[0], root_value_id);
  EXPECT_EQ(loom_op_operands(late_add)[0],
            loom_low_const_result(cloned_constant));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       RepairsRecordedMultiPacketMaterializationOncePerUseBlock) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @repair_slice(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require -1 {preserve = true, projected_baseline = 4}
  %first = test.add.i32 %input, %input
  %root = test.add.i32 %first, %input
  %placed = low.residency.candidate 7 1 %root : reg<test.i32> captures(%input : reg<test.i32>) recipe(%first : reg<test.i32>, %root : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %padding = test.const.i32 11
  %result = test.add.i32 %placed, %padding
  return %result
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));
  ASSERT_EQ(contract.candidate_count, 1u);
  loom_low_residency_contract_candidate_t& candidate = contract.candidates[0];
  EXPECT_EQ(candidate.materialization_op_count, 2u);
  ASSERT_EQ(candidate.materialization_input_count, 1u);
  EXPECT_EQ(
      candidate.materialization_inputs[0],
      loom_block_arg_id(
          loom_region_entry_block(loom_low_func_def_body(function_op)), 0));
  loom_op_t* original_first = candidate.materialization_ops[0];
  loom_op_t* original_root = candidate.materialization_ops[1];

  loom_low_allocation_assignment_t assignment = {
      /*.value_id=*/candidate.value_id,
  };
  loom_low_allocation_table_t allocation = {};
  allocation.module = module.get();
  allocation.function_op = function_op;
  allocation.assignments = &assignment;
  allocation.assignment_count = 1;
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), function_op, &target_registry_.registry,
      loom_target_selection_empty(), iree_diagnostic_emitter_t{},
      &allocation.target));

  loom_low_allocation_rematerialization_result_t result = {};
  IREE_ASSERT_OK(loom_low_residency_contract_try_repair(
      module.get(), &allocation, &contract,
      LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE, &arena, &result));
  EXPECT_EQ(result.value_id, candidate.value_id);
  EXPECT_EQ(result.assignment_index, 0u);
  EXPECT_EQ(result.cloned_packet_count, 2u);
  EXPECT_EQ(result.rewritten_operand_count, 1u);
  EXPECT_TRUE(candidate.attempted);
  EXPECT_TRUE(candidate.restored);
  EXPECT_TRUE(iree_any_bit_set(original_first->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(original_root->flags, LOOM_OP_FLAG_DEAD));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       RepairsRecordedMultiPacketMaterializationWithoutSourceCaptures) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @repair_constant_slice() -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require -1 {preserve = true, projected_baseline = 4}
  %lhs = test.const.i32 7
  %rhs = test.const.i32 11
  %root = test.add.i32 %lhs, %rhs
  %placed = low.residency.candidate 7 1 %root : reg<test.i32> recipe(%lhs : reg<test.i32>, %rhs : reg<test.i32>, %root : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %padding = test.const.i32 13
  %result = test.add.i32 %placed, %padding
  return %result
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));
  ASSERT_EQ(contract.candidate_count, 1u);
  loom_low_residency_contract_candidate_t& candidate = contract.candidates[0];
  EXPECT_EQ(candidate.materialization_op_count, 3u);
  EXPECT_EQ(candidate.materialization_input_count, 0u);
  loom_op_t* original_lhs = candidate.materialization_ops[0];
  loom_op_t* original_rhs = candidate.materialization_ops[1];
  loom_op_t* original_root = candidate.materialization_ops[2];

  loom_low_allocation_assignment_t assignment = {
      /*.value_id=*/candidate.value_id,
  };
  loom_low_allocation_table_t allocation = {};
  allocation.module = module.get();
  allocation.function_op = function_op;
  allocation.assignments = &assignment;
  allocation.assignment_count = 1;
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), function_op, &target_registry_.registry,
      loom_target_selection_empty(), iree_diagnostic_emitter_t{},
      &allocation.target));

  loom_low_allocation_rematerialization_result_t result = {};
  IREE_ASSERT_OK(loom_low_residency_contract_try_repair(
      module.get(), &allocation, &contract,
      LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE, &arena, &result));
  EXPECT_EQ(result.value_id, candidate.value_id);
  EXPECT_EQ(result.assignment_index, 0u);
  EXPECT_EQ(result.cloned_packet_count, 3u);
  EXPECT_EQ(result.rewritten_operand_count, 1u);
  EXPECT_TRUE(candidate.attempted);
  EXPECT_TRUE(candidate.restored);
  EXPECT_TRUE(iree_any_bit_set(original_lhs->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(original_rhs->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(original_root->flags, LOOM_OP_FLAG_DEAD));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       RepairsTopologicalRecipeIndependentOfPhysicalPacketOrder) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @repair_interleaved_recipe() -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require -1 {preserve = true, projected_baseline = 4}
  %lhs = test.const.i32 7
  %left = test.add.i32 %lhs, %lhs
  %rhs = test.const.i32 11
  %right = test.add.i32 %rhs, %rhs
  %root = test.add.i32 %left, %right
  %placed = low.residency.candidate 7 1 %root : reg<test.i32> recipe(%lhs : reg<test.i32>, %rhs : reg<test.i32>, %left : reg<test.i32>, %right : reg<test.i32>, %root : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %padding = test.const.i32 13
  %result = test.add.i32 %placed, %padding
  return %result
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));
  ASSERT_EQ(contract.candidate_count, 1u);
  loom_low_residency_contract_candidate_t& candidate = contract.candidates[0];
  ASSERT_EQ(candidate.materialization_op_count, 5u);
  EXPECT_GT(candidate.materialization_ops[1]->block_ordinal,
            candidate.materialization_ops[2]->block_ordinal);

  loom_low_allocation_assignment_t assignment = {
      /*.value_id=*/candidate.value_id,
  };
  loom_low_allocation_table_t allocation = {};
  allocation.module = module.get();
  allocation.function_op = function_op;
  allocation.assignments = &assignment;
  allocation.assignment_count = 1;
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), function_op, &target_registry_.registry,
      loom_target_selection_empty(), iree_diagnostic_emitter_t{},
      &allocation.target));

  loom_low_allocation_rematerialization_result_t result = {};
  IREE_ASSERT_OK(loom_low_residency_contract_try_repair(
      module.get(), &allocation, &contract,
      LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE, &arena, &result));
  EXPECT_EQ(result.value_id, candidate.value_id);
  EXPECT_EQ(result.assignment_index, 0u);
  EXPECT_EQ(result.cloned_packet_count, 5u);
  EXPECT_EQ(result.rewritten_operand_count, 1u);
  EXPECT_TRUE(candidate.restored);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       RepairsRecipeWithProducersAcrossDominatedBlocks) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @repair_cross_block_recipe(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require -1 {preserve = true, projected_baseline = 4}
  %unit = test.const.i32 7
  low.br ^body
^body:
  %root = test.add.i32 %input, %unit
  %placed = low.residency.candidate 7 1 %root : reg<test.i32> captures(%input : reg<test.i32>) recipe(%unit : reg<test.i32>, %root : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %padding = test.const.i32 11
  %result = test.add.i32 %placed, %padding
  return %result
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));
  ASSERT_EQ(contract.candidate_count, 1u);
  loom_low_residency_contract_candidate_t& candidate = contract.candidates[0];
  ASSERT_EQ(candidate.materialization_op_count, 2u);
  ASSERT_NE(candidate.materialization_ops[0]->parent_block,
            candidate.materialization_ops[1]->parent_block);
  loom_op_t* original_unit = candidate.materialization_ops[0];
  loom_op_t* original_root = candidate.materialization_ops[1];
  loom_op_t* user_op = loom_use_user_op(candidate.uses[0]);

  loom_low_allocation_assignment_t assignment = {
      /*.value_id=*/candidate.value_id,
  };
  loom_low_allocation_table_t allocation = {};
  allocation.module = module.get();
  allocation.function_op = function_op;
  allocation.assignments = &assignment;
  allocation.assignment_count = 1;
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), function_op, &target_registry_.registry,
      loom_target_selection_empty(), iree_diagnostic_emitter_t{},
      &allocation.target));

  loom_low_allocation_rematerialization_result_t result = {};
  IREE_ASSERT_OK(loom_low_residency_contract_try_repair(
      module.get(), &allocation, &contract,
      LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE, &arena, &result));
  EXPECT_EQ(result.value_id, candidate.value_id);
  EXPECT_EQ(result.assignment_index, 0u);
  EXPECT_EQ(result.cloned_packet_count, 2u);
  EXPECT_EQ(result.rewritten_operand_count, 1u);
  EXPECT_TRUE(candidate.restored);
  EXPECT_TRUE(iree_any_bit_set(original_unit->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(original_root->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_NE(loom_op_operands(user_op)[0], candidate.value_id);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, TerminalRepairRestoresEveryFiniteCandidate) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @many_candidates() asm<test.low.core> {
  low.residency.require 4
  %s0 = test.const.i32 0
  %s1 = test.const.i32 1
  %s2 = test.const.i32 2
  %s3 = test.const.i32 3
  %s4 = test.const.i32 4
  %s5 = test.const.i32 5
  %s6 = test.const.i32 6
  %s7 = test.const.i32 7
  %s8 = test.const.i32 8
  %s9 = test.const.i32 9
  %c0 = low.residency.candidate 0 1 %s0 : reg<test.i32> recipe(%s0 : reg<test.i32>) {sealed = true}
  %u0 = test.add.i32 %c0, %s0
  %c1 = low.residency.candidate 1 1 %s1 : reg<test.i32> recipe(%s1 : reg<test.i32>) {sealed = true}
  %u1 = test.add.i32 %c1, %s1
  %c2 = low.residency.candidate 2 1 %s2 : reg<test.i32> recipe(%s2 : reg<test.i32>) {sealed = true}
  %u2 = test.add.i32 %c2, %s2
  %c3 = low.residency.candidate 3 1 %s3 : reg<test.i32> recipe(%s3 : reg<test.i32>) {sealed = true}
  %u3 = test.add.i32 %c3, %s3
  %c4 = low.residency.candidate 4 1 %s4 : reg<test.i32> recipe(%s4 : reg<test.i32>) {sealed = true}
  %u4 = test.add.i32 %c4, %s4
  %c5 = low.residency.candidate 5 1 %s5 : reg<test.i32> recipe(%s5 : reg<test.i32>) {sealed = true}
  %u5 = test.add.i32 %c5, %s5
  %c6 = low.residency.candidate 6 1 %s6 : reg<test.i32> recipe(%s6 : reg<test.i32>) {sealed = true}
  %u6 = test.add.i32 %c6, %s6
  %c7 = low.residency.candidate 7 1 %s7 : reg<test.i32> recipe(%s7 : reg<test.i32>) {sealed = true}
  %u7 = test.add.i32 %c7, %s7
  %c8 = low.residency.candidate 8 1 %s8 : reg<test.i32> recipe(%s8 : reg<test.i32>) {sealed = true}
  %u8 = test.add.i32 %c8, %s8
  %c9 = low.residency.candidate 9 1 %s9 : reg<test.i32> recipe(%s9 : reg<test.i32>) {sealed = true}
  %u9 = test.add.i32 %c9, %s9
  return
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));
  ASSERT_EQ(contract.candidate_count, 10u);

  std::vector<loom_low_allocation_assignment_t> assignments(
      contract.candidate_count);
  for (iree_host_size_t i = 0; i < contract.candidate_count; ++i) {
    assignments[i].value_id = contract.candidates[i].value_id;
  }
  loom_low_allocation_table_t allocation = {};
  allocation.module = module.get();
  allocation.function_op = function_op;
  allocation.assignments = assignments.data();
  allocation.assignment_count = assignments.size();
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), function_op, &target_registry_.registry,
      loom_target_selection_empty(), iree_diagnostic_emitter_t{},
      &allocation.target));

  loom_low_allocation_rematerialization_result_t result = {};
  uint32_t repaired_candidate_count = 0;
  IREE_ASSERT_OK(loom_low_residency_contract_try_repair_remaining(
      module.get(), &allocation, &contract,
      LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES, &arena, &result,
      &repaired_candidate_count));
  EXPECT_EQ(repaired_candidate_count, 10u);
  EXPECT_EQ(result.cloned_packet_count, 10u);
  EXPECT_EQ(result.rewritten_operand_count, 10u);
  EXPECT_TRUE(loom_low_residency_contract_candidates_exhausted(
      &contract, LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES));
  EXPECT_TRUE(loom_low_residency_contract_candidates_restored(
      &contract, LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       BaselineRepairScopeLeavesNumericOnlyCandidateUntouched) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @mixed_candidates() asm<test.low.core> {
  low.residency.require 3 {preserve = true, projected_baseline = 4}
  %numeric_seed = test.const.i32 7
  %preserve_seed = test.const.i32 11
  %numeric = low.residency.candidate 7 1 %numeric_seed : reg<test.i32> recipe(%numeric_seed : reg<test.i32>) {sealed = true}
  %numeric_use = test.add.i32 %numeric, %numeric_seed
  %preserve = low.residency.candidate 8 2 %preserve_seed : reg<test.i32> recipe(%preserve_seed : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %preserve_use = test.add.i32 %preserve, %preserve_seed
  return
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_residency_contract_t contract = {};
  IREE_ASSERT_OK(loom_low_residency_contract_consume(module.get(), function_op,
                                                     &arena, &contract));
  ASSERT_EQ(contract.candidate_count, 2u);
  ASSERT_FALSE(contract.candidates[0].preserves_baseline);
  ASSERT_TRUE(contract.candidates[1].preserves_baseline);

  loom_low_allocation_assignment_t assignments[2] = {};
  assignments[0].value_id = contract.candidates[0].value_id;
  assignments[1].value_id = contract.candidates[1].value_id;
  loom_low_allocation_table_t allocation = {};
  allocation.module = module.get();
  allocation.function_op = function_op;
  allocation.assignments = assignments;
  allocation.assignment_count = IREE_ARRAYSIZE(assignments);
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), function_op, &target_registry_.registry,
      loom_target_selection_empty(), iree_diagnostic_emitter_t{},
      &allocation.target));

  loom_low_allocation_rematerialization_result_t result = {};
  IREE_ASSERT_OK(loom_low_residency_contract_try_repair(
      module.get(), &allocation, &contract,
      LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE, &arena, &result));
  EXPECT_EQ(result.value_id, contract.candidates[1].value_id);
  EXPECT_FALSE(contract.candidates[0].attempted);
  EXPECT_TRUE(contract.candidates[1].attempted);
  EXPECT_TRUE(contract.candidates[1].restored);
  EXPECT_TRUE(loom_low_residency_contract_candidates_exhausted(
      &contract, LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE));
  EXPECT_TRUE(loom_low_residency_contract_candidates_restored(
      &contract, LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE));
  EXPECT_FALSE(loom_low_residency_contract_candidates_exhausted(
      &contract, LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       SpillFreeFrameConsumesAndReportsSatisfiedContract) {
  ModulePtr module = ParseModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);
  ResidencyModelStorage residency;
  InitializeResidencyModel(module.get(), function_op, /*cliff_units=*/100,
                           &residency);
  DiagnosticCapture diagnostics = {};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t frame_options = {};
  frame_options.descriptor_registry = &target_registry_.registry;
  frame_options.residency_model = &residency.model;
  frame_options.emitter = {CaptureDiagnostic, &diagnostics};
  frame_options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), function_op, &frame_options, &spill_free_options, &arena,
      &frame));

  EXPECT_EQ(diagnostics.count, 0u);
  EXPECT_TRUE(frame.has_residency_contract);
  EXPECT_TRUE(frame.residency_contract_evaluated);
  EXPECT_EQ(frame.required_residency_tier, 3u);
  EXPECT_EQ(frame.allocated_residency_tier, 4u);
  EXPECT_EQ(frame.residency_candidate_count, 2u);
  ASSERT_NE(frame.residency_candidates, nullptr);
  EXPECT_EQ(frame.residency_candidates[0].candidate_id, 7u);
  EXPECT_EQ(frame.residency_candidates[1].candidate_id, 8u);
  EXPECT_FALSE(frame.residency_candidates[0].attempted);
  EXPECT_FALSE(frame.residency_candidates[1].attempted);
  EXPECT_EQ(frame.attempted_residency_candidate_count, 0u);
  EXPECT_EQ(frame.residency_repair_count, 0u);
  EXPECT_EQ(statistics.residency.contract_count, 1u);
  EXPECT_EQ(statistics.residency.candidate_count, 2u);
  EXPECT_EQ(statistics.residency.maximum_projected_required_tier, 3u);
  EXPECT_EQ(statistics.residency.validation_count, 1u);
  EXPECT_EQ(statistics.residency.minimum_observed_allocated_tier, 4u);
  EXPECT_EQ(statistics.residency.maximum_observed_tier_shortfall, 0u);
  EXPECT_EQ(statistics.residency.repair_attempt_count, 0u);
  EXPECT_EQ(statistics.residency.repair_count, 0u);
  EXPECT_EQ(statistics.residency.failure_count, 0u);
  EXPECT_LT(frame.allocation.assigned_extents
                .ends_by_reg_class[residency.cliff.resource_id],
            residency.cliff.cliff_units);

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_low_exact_residency(&report, &frame));
  EXPECT_EQ(report.exact_residency.contract_count, 1u);
  EXPECT_EQ(report.exact_residency.evaluated_contract_count, 1u);
  EXPECT_EQ(report.exact_residency.satisfied_contract_count, 1u);
  EXPECT_EQ(report.exact_residency.repaired_contract_count, 0u);
  ASSERT_EQ(report.source_low_residency_resource_rows.count,
            residency.model.direct_resources.resource_count);
  const auto* cliff_resource = CompileReportRowAt<
      loom_target_compile_report_source_low_residency_resource_row_t>(
      report.source_low_residency_resource_rows, residency.cliff.resource_id);
  ASSERT_NE(cliff_resource, nullptr);
  EXPECT_TRUE(iree_string_view_equal(cliff_resource->phase, IREE_SV("exact")));
  EXPECT_EQ(cliff_resource->resource_id, residency.cliff.resource_id);
  EXPECT_EQ(cliff_resource->units,
            frame.allocation.assigned_extents
                .ends_by_reg_class[residency.cliff.resource_id]);
  EXPECT_EQ(cliff_resource->tier, frame.allocated_residency_tier);
  ASSERT_EQ(report.residency_candidate_rows.count, 2u);
  const auto* candidate =
      CompileReportRowAt<loom_target_compile_report_residency_candidate_row_t>(
          report.residency_candidate_rows, 1);
  ASSERT_NE(candidate, nullptr);
  EXPECT_EQ(candidate->candidate_id, 8u);
  EXPECT_TRUE(
      iree_string_view_equal(candidate->outcome, IREE_SV("not_attempted")));
  loom_target_compile_report_deinitialize(&report);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, SpillFreeFrameRepairsExactResidencyCliff) {
  ModulePtr module = ParseRepairModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);
  ResidencyModelStorage residency;
  InitializeResidencyModel(module.get(), function_op, /*cliff_units=*/4,
                           &residency);
  DiagnosticCapture diagnostics = {};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t frame_options = {};
  frame_options.descriptor_registry = &target_registry_.registry;
  frame_options.residency_model = &residency.model;
  frame_options.emitter = {CaptureDiagnostic, &diagnostics};
  frame_options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), function_op, &frame_options, &spill_free_options, &arena,
      &frame));

  EXPECT_EQ(diagnostics.count, 0u);
  EXPECT_TRUE(frame.has_residency_contract);
  EXPECT_EQ(frame.required_residency_tier, 3u);
  EXPECT_EQ(frame.allocated_residency_tier, 4u);
  EXPECT_EQ(frame.residency_candidate_count, 2u);
  EXPECT_EQ(frame.attempted_residency_candidate_count, 1u);
  EXPECT_EQ(frame.residency_repair_count, 1u);
  ASSERT_NE(frame.residency_candidates, nullptr);
  EXPECT_FALSE(frame.residency_candidates[0].attempted);
  EXPECT_TRUE(frame.residency_candidates[1].attempted);
  EXPECT_TRUE(frame.residency_candidates[1].restored);
  EXPECT_EQ(frame.residency_candidates[1].cloned_packet_count, 1u);
  EXPECT_EQ(frame.residency_candidates[1].rewritten_operand_count, 1u);
  EXPECT_EQ(statistics.residency.contract_count, 1u);
  EXPECT_EQ(statistics.residency.candidate_count, 2u);
  EXPECT_EQ(statistics.residency.maximum_projected_required_tier, 3u);
  EXPECT_EQ(statistics.residency.validation_count, 2u);
  EXPECT_EQ(statistics.residency.minimum_observed_allocated_tier, 2u);
  EXPECT_EQ(statistics.residency.maximum_observed_tier_shortfall, 1u);
  EXPECT_EQ(statistics.residency.repair_attempt_count, 1u);
  EXPECT_EQ(statistics.residency.repair_count, 1u);
  EXPECT_EQ(statistics.residency.failure_count, 0u);
  EXPECT_LT(frame.allocation.assigned_extents
                .ends_by_reg_class[residency.cliff.resource_id],
            residency.cliff.cliff_units);

  std::vector<loom_op_t*> descriptor_ops = FindDescriptorPackets(function_op);
  ASSERT_EQ(descriptor_ops.size(), 8u);
  ASSERT_TRUE(loom_low_const_isa(descriptor_ops[0]));
  ASSERT_TRUE(loom_low_const_isa(descriptor_ops[5]));
  const loom_value_id_t original_seed =
      loom_low_const_result(descriptor_ops[0]);
  EXPECT_EQ(loom_op_operands(descriptor_ops[1])[0], original_seed);
  EXPECT_EQ(loom_op_operands(descriptor_ops[6])[0],
            loom_low_const_result(descriptor_ops[5]));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       SpillFreeFrameRecoversExactBaselineWhenProjectionWasOptimistic) {
  ModulePtr module = ParsePreserveRepairModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);
  ResidencyModelStorage residency;
  InitializeResidencyModel(module.get(), function_op, /*cliff_units=*/2,
                           &residency);
  DiagnosticCapture diagnostics = {};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t frame_options = {};
  frame_options.descriptor_registry = &target_registry_.registry;
  frame_options.residency_model = &residency.model;
  frame_options.emitter = {CaptureDiagnostic, &diagnostics};
  frame_options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), function_op, &frame_options, &spill_free_options, &arena,
      &frame));

  EXPECT_EQ(diagnostics.count, 0u);
  EXPECT_TRUE(frame.has_residency_contract);
  EXPECT_FALSE(frame.has_minimum_residency_requirement);
  EXPECT_TRUE(frame.preserves_residency_baseline);
  EXPECT_TRUE(frame.preserved_residency_baseline_resolved);
  EXPECT_EQ(frame.minimum_residency_tier, 0u);
  EXPECT_EQ(frame.projected_residency_baseline_tier, 4u);
  EXPECT_EQ(frame.required_residency_tier, 2u);
  EXPECT_EQ(frame.allocated_residency_tier, 2u);
  EXPECT_EQ(frame.residency_candidate_count, 2u);
  EXPECT_EQ(frame.attempted_residency_candidate_count, 2u);
  EXPECT_GE(frame.residency_repair_count, 1u);
  ASSERT_NE(frame.residency_candidates, nullptr);
  EXPECT_TRUE(frame.residency_candidates[0].attempted);
  EXPECT_TRUE(frame.residency_candidates[0].restored);
  EXPECT_TRUE(frame.residency_candidates[0].preserves_baseline);
  EXPECT_TRUE(frame.residency_candidates[1].attempted);
  EXPECT_TRUE(frame.residency_candidates[1].restored);
  EXPECT_TRUE(frame.residency_candidates[1].preserves_baseline);
  EXPECT_EQ(statistics.residency.failure_count, 0u);
  EXPECT_EQ(statistics.residency.maximum_projected_required_tier, 4u);
  EXPECT_EQ(statistics.residency.minimum_observed_allocated_tier, 2u);
  EXPECT_EQ(statistics.residency.maximum_observed_tier_shortfall, 2u);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       ExactBaselineRecoveryDoesNotWeakenNumericMinimum) {
  ModulePtr module = ParsePreserveMinimumRepairModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);
  ResidencyModelStorage residency;
  InitializeResidencyModel(module.get(), function_op, /*cliff_units=*/2,
                           &residency);
  DiagnosticCapture diagnostics = {};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t frame_options = {};
  frame_options.descriptor_registry = &target_registry_.registry;
  frame_options.residency_model = &residency.model;
  frame_options.emitter = {CaptureDiagnostic, &diagnostics};
  frame_options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), function_op, &frame_options, &spill_free_options, &arena,
      &frame));

  ASSERT_EQ(diagnostics.count, 1u);
  EXPECT_EQ(diagnostics.last_error, LOOM_ERR_BACKEND_047);
  EXPECT_TRUE(frame.has_residency_contract);
  EXPECT_TRUE(frame.residency_contract_evaluated);
  EXPECT_TRUE(frame.has_minimum_residency_requirement);
  EXPECT_TRUE(frame.preserves_residency_baseline);
  EXPECT_TRUE(frame.preserved_residency_baseline_resolved);
  EXPECT_EQ(frame.minimum_residency_tier, 3u);
  EXPECT_EQ(frame.projected_residency_baseline_tier, 4u);
  EXPECT_EQ(frame.required_residency_tier, 3u);
  EXPECT_EQ(frame.allocated_residency_tier, 2u);
  EXPECT_EQ(frame.residency_candidate_count, 2u);
  EXPECT_EQ(frame.attempted_residency_candidate_count, 2u);
  EXPECT_GE(frame.residency_repair_count, 1u);
  ASSERT_NE(frame.residency_candidates, nullptr);
  EXPECT_TRUE(frame.residency_candidates[0].restored);
  EXPECT_TRUE(frame.residency_candidates[1].restored);
  EXPECT_FALSE(frame.residency_candidates[0].preserves_baseline);
  EXPECT_TRUE(frame.residency_candidates[1].preserves_baseline);
  EXPECT_EQ(statistics.residency.failure_count, 1u);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest,
       PreserveFailsWhenAuthoredPlacementCannotBeRestored) {
  ModulePtr module = ParseModuleSource(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @unrestorable(%input: reg<test.i32>) -> (reg<test.i32>) asm<test.low.core> {
  low.residency.require -1 {preserve = true, projected_baseline = 4}
  %placed = low.residency.candidate 0 1 %input : reg<test.i32> captures(%input : reg<test.i32>) {preserves_baseline = true, sealed = true}
  %seed = test.const.i32 7
  %middle = test.add.i32 %seed, %seed
  %result = test.add.i32 %placed, %middle
  return %result
}
)");
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);
  ResidencyModelStorage residency;
  InitializeResidencyModel(module.get(), function_op, /*cliff_units=*/2,
                           &residency);
  DiagnosticCapture diagnostics = {};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t frame_options = {};
  frame_options.descriptor_registry = &target_registry_.registry;
  frame_options.residency_model = &residency.model;
  frame_options.emitter = {CaptureDiagnostic, &diagnostics};
  frame_options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), function_op, &frame_options, &spill_free_options, &arena,
      &frame));

  ASSERT_EQ(diagnostics.count, 1u);
  EXPECT_EQ(diagnostics.last_error, LOOM_ERR_BACKEND_047);
  EXPECT_TRUE(frame.preserves_residency_baseline);
  EXPECT_FALSE(frame.preserved_residency_baseline_resolved);
  EXPECT_EQ(frame.required_residency_tier, 4u);
  EXPECT_EQ(frame.allocated_residency_tier, 2u);
  EXPECT_EQ(frame.residency_candidate_count, 1u);
  EXPECT_EQ(frame.attempted_residency_candidate_count, 1u);
  EXPECT_EQ(frame.residency_repair_count, 0u);
  ASSERT_NE(frame.residency_candidates, nullptr);
  EXPECT_TRUE(frame.residency_candidates[0].attempted);
  EXPECT_FALSE(frame.residency_candidates[0].restored);
  EXPECT_EQ(frame.residency_candidates[0].cloned_packet_count, 0u);
  EXPECT_EQ(frame.residency_candidates[0].rewritten_operand_count, 0u);
  EXPECT_TRUE(frame.residency_candidates[0].preserves_baseline);
  EXPECT_EQ(statistics.residency.failure_count, 1u);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowResidencyContractTest, SpillFreeFrameDiagnosesExhaustedRepairSet) {
  ModulePtr module = ParseModule();
  loom_op_t* function_op = FindLowFunction(module.get());
  ASSERT_NE(function_op, nullptr);
  ResidencyModelStorage residency;
  InitializeResidencyModel(module.get(), function_op, /*cliff_units=*/2,
                           &residency);
  DiagnosticCapture diagnostics = {};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t frame_options = {};
  frame_options.descriptor_registry = &target_registry_.registry;
  frame_options.residency_model = &residency.model;
  frame_options.emitter = {CaptureDiagnostic, &diagnostics};
  frame_options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), function_op, &frame_options, &spill_free_options, &arena,
      &frame));

  ASSERT_EQ(diagnostics.count, 1u);
  EXPECT_EQ(diagnostics.last_error, LOOM_ERR_BACKEND_047);
  EXPECT_TRUE(frame.has_residency_contract);
  EXPECT_EQ(frame.required_residency_tier, 3u);
  EXPECT_EQ(frame.allocated_residency_tier, 2u);
  EXPECT_EQ(frame.residency_candidate_count, 2u);
  EXPECT_EQ(frame.attempted_residency_candidate_count, 2u);
  EXPECT_EQ(frame.residency_repair_count, 1u);
  EXPECT_EQ(statistics.residency.validation_count, 2u);
  EXPECT_EQ(statistics.residency.maximum_projected_required_tier, 3u);
  EXPECT_EQ(statistics.residency.minimum_observed_allocated_tier, 2u);
  EXPECT_EQ(statistics.residency.maximum_observed_tier_shortfall, 1u);
  EXPECT_EQ(statistics.residency.repair_attempt_count, 2u);
  EXPECT_EQ(statistics.residency.repair_count, 1u);
  EXPECT_EQ(statistics.residency.failure_count, 1u);
  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
