// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks the complete target-module sealing boundary across independent
// clone, target-grouping, materialization, and definition-reuse dimensions.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_version.h"
#include "loom/target/module_sealing.h"
#include "loom/target/provider.h"
#include "loom/target/test/target_records.h"

namespace {

enum class SealingShape {
  kCloneOnly,
  kSharedTarget,
  kDistinctTargets,
  kDistinctAuthoredTargets,
};

static constexpr int64_t kBodyOpCount = 8;

static iree_status_t MaterializeTestTargetDefinition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location,
    loom_op_t** out_target_op) {
  const loom_target_facts_t* facts = resolved_target->facts;
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "test target flags reserve the first 30 bits for common "
                "target facts");
  static_assert(LOOM_TEST_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "test target flags must follow target fact ordinals");
  static_assert(LOOM_TEST_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "test target flags must follow target fact ordinals");
  const auto build_flags =
      static_cast<loom_test_target_build_flags_t>(facts->explicit_fields);

  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_TEST_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.export_plan.export_symbol, &export_symbol));
  }
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_TEST_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.config.contract_set_key, &contract_set_key));
  }

  const loom_target_snapshot_t* snapshot = &facts->storage.snapshot;
  const loom_target_export_plan_t* export_plan = &facts->storage.export_plan;
  const loom_target_config_t* config = &facts->storage.config;
  return loom_test_target_build(
      builder, build_flags,
      static_cast<loom_test_target_kind_t>(facts->selector), symbol,
      snapshot->codegen_format, snapshot->artifact_format,
      snapshot->default_pointer_bitwidth, snapshot->index_bitwidth,
      snapshot->offset_bitwidth, snapshot->max_workgroup_size.x,
      snapshot->max_workgroup_size.y, snapshot->max_workgroup_size.z,
      snapshot->max_flat_workgroup_size, snapshot->max_workgroup_storage_bytes,
      snapshot->subgroup_size, snapshot->max_grid_size.x,
      snapshot->max_grid_size.y, snapshot->max_grid_size.z,
      snapshot->max_flat_grid_size, snapshot->max_workgroup_count.x,
      snapshot->max_workgroup_count.y, snapshot->max_workgroup_count.z,
      snapshot->memory_spaces.generic, snapshot->memory_spaces.global,
      snapshot->memory_spaces.workgroup, snapshot->memory_spaces.constant,
      snapshot->memory_spaces.private_memory, snapshot->memory_spaces.host,
      snapshot->memory_spaces.descriptor, export_plan->abi_kind, export_symbol,
      export_plan->linkage, contract_set_key, config->contract_feature_bits,
      location, out_target_op);
}

static const loom_target_profile_type_t kTestProfileType = {
    /*.name=*/IREE_SVL("module-sealing-benchmark"),
    /*.fact_type=*/&loom_test_target_fact_type,
};

static const loom_target_provider_t kTestProvider = {
    /*.profile_type=*/&kTestProfileType,
    /*.materialize_definition=*/MaterializeTestTargetDefinition,
};

static loom_symbol_ref_t AddSymbol(loom_module_t* module,
                                   loom_builder_t* builder, const char* name) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_CHECK_OK(loom_builder_intern_string(
      builder, iree_make_cstring_view(name), &name_id));
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  return {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
}

static loom_func_like_t AddFunction(loom_module_t* module,
                                    loom_builder_t* module_builder,
                                    loom_type_t value_type, int64_t ordinal) {
  char name[32];
  std::snprintf(name, sizeof(name), "function_%08d", static_cast<int>(ordinal));
  const loom_symbol_ref_t symbol = AddSymbol(module, module_builder, name);
  loom_op_t* function_op = nullptr;
  IREE_CHECK_OK(loom_func_def_build(
      module_builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
      /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      /*export_symbol=*/LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
      symbol, &value_type, /*arg_types_count=*/1, &value_type,
      /*result_count=*/1, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, /*predicates=*/nullptr,
      /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op));
  loom_func_like_t function = loom_func_like_cast(module, function_op);
  if (!loom_func_like_isa(function)) std::abort();

  loom_builder_t body_builder;
  loom_builder_initialize(
      module, &module->arena,
      loom_region_entry_block(loom_func_like_body(function)), &body_builder);
  body_builder.ip.parent_op = function_op;
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  if (argument_count != 1) std::abort();
  loom_value_id_t current_value = arguments[0];
  for (int64_t i = 0; i < kBodyOpCount; ++i) {
    loom_op_t* add_op = nullptr;
    IREE_CHECK_OK(loom_test_addi_build(&body_builder, current_value,
                                       arguments[0], value_type,
                                       LOOM_LOCATION_UNKNOWN, &add_op));
    current_value = loom_test_addi_result(add_op);
  }
  loom_op_t* return_op = nullptr;
  IREE_CHECK_OK(loom_func_return_build(&body_builder, &current_value,
                                       /*values_count=*/1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));
  return function;
}

class ModuleSealingFixture {
 public:
  ModuleSealingFixture(SealingShape shape, int64_t function_count)
      : shape_(shape), function_count_(function_count) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_CHECK_OK(loom_context_finalize(&context_));

    const int64_t target_count = shape == SealingShape::kCloneOnly ? 0
                                 : shape == SealingShape::kSharedTarget
                                     ? 1
                                     : function_count;
    facts_.resize(static_cast<size_t>(target_count));
    for (int64_t i = 0; i < target_count; ++i) {
      loom_target_facts_builder_initialize(
          &loom_test_target_fact_type,
          loom_target_bundle_table_lookup(&loom_test_target_bundles,
                                          LOOM_TEST_TARGET_KIND_LOW_CORE),
          &facts_[i]);
      facts_[i].selector = LOOM_TEST_TARGET_KIND_LOW_CORE;
      facts_[i].explicit_fields = static_cast<loom_target_fact_field_set_t>(i);
    }

    loom_module_size_hints_t hints = {
        /*.value_count=*/0,
        /*.string_count=*/0,
        /*.type_count=*/0,
        /*.encoding_count=*/0,
        /*.symbol_count=*/
        static_cast<iree_host_size_t>(function_count + target_count),
    };
    IREE_CHECK_OK(loom_module_allocate(
        &context_, IREE_SV("module_sealing_benchmark"), &block_pool_, &hints,
        iree_allocator_system(), &source_module_));
    loom_builder_t builder;
    loom_builder_initialize(source_module_, &source_module_->arena,
                            loom_module_block(source_module_), &builder);
    loom_type_t value_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_CHECK_OK(
        loom_module_intern_type(source_module_, value_type, &value_type));

    if (shape == SealingShape::kDistinctAuthoredTargets) {
      for (int64_t i = 0; i < target_count; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "target_%08d", static_cast<int>(i));
        const loom_symbol_ref_t symbol =
            AddSymbol(source_module_, &builder, name);
        loom_op_t* target_op = nullptr;
        const loom_resolved_target_t resolved_target = {
            /*.provider=*/&kTestProvider,
            /*.facts=*/&facts_[i],
        };
        IREE_CHECK_OK(
            MaterializeTestTargetDefinition(&builder, &resolved_target, symbol,
                                            LOOM_LOCATION_UNKNOWN, &target_op));
      }
    }

    functions_.reserve(static_cast<size_t>(function_count));
    for (int64_t i = 0; i < function_count; ++i) {
      functions_.push_back(
          AddFunction(source_module_, &builder, value_type, i));
    }

    if (target_count > 0) {
      versions_.resize(static_cast<size_t>(function_count));
      version_handles_.resize(static_cast<size_t>(function_count));
      for (int64_t i = 0; i < function_count; ++i) {
        const int64_t target_ordinal = target_count == 1 ? 0 : i;
        versions_[i] = loom_target_function_version_t{
            /*.base=*/
            {
                /*.type=*/&loom_target_function_version_type,
                /*.function=*/functions_[i],
            },
            /*.authored_target_name=*/{},
            /*.target_requirement_facts=*/nullptr,
            /*.resolved_target=*/
            {
                /*.provider=*/&kTestProvider,
                /*.facts=*/&facts_[target_ordinal],
            },
            /*.target_context_ordinal=*/
            static_cast<loom_target_context_ordinal_t>(target_ordinal),
            /*.authored_target_is_exact=*/false,
            /*.function_target_facts=*/&facts_[target_ordinal],
        };
        version_handles_[i] = &versions_[i].base;
      }
      version_list_ = {
          /*.values=*/version_handles_.data(),
          /*.count=*/version_handles_.size(),
      };
    }

    loom_module_t* sealed_module = Seal();
    const iree_host_size_t expected_symbol_count =
        source_module_->symbols.count +
        (shape == SealingShape::kDistinctTargets ? target_count : 0) +
        (shape == SealingShape::kSharedTarget ? 1 : 0);
    if (sealed_module->symbols.count != expected_symbol_count) std::abort();
    output_owned_bytes_ = sealed_module->arena.total_allocation_size;
    output_used_bytes_ = sealed_module->arena.used_allocation_size;
    loom_module_free(sealed_module);
  }

  ModuleSealingFixture(const ModuleSealingFixture&) = delete;
  ModuleSealingFixture& operator=(const ModuleSealingFixture&) = delete;

  ~ModuleSealingFixture() {
    loom_module_free(source_module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Seal() {
    loom_module_t* sealed_module = nullptr;
    IREE_CHECK_OK(loom_target_module_seal(
        source_module_, version_list_.count > 0 ? &version_list_ : nullptr,
        &block_pool_, iree_allocator_system(), &sealed_module));
    if (sealed_module == nullptr) std::abort();
    return sealed_module;
  }

  int64_t function_count() const { return function_count_; }

  int64_t target_count() const {
    if (shape_ == SealingShape::kCloneOnly) return 0;
    return shape_ == SealingShape::kSharedTarget ? 1 : function_count_;
  }

  int64_t authored_target_count() const {
    return shape_ == SealingShape::kDistinctAuthoredTargets ? function_count_
                                                            : 0;
  }

  iree_host_size_t source_op_count() const {
    return static_cast<iree_host_size_t>(function_count_ * (kBodyOpCount + 2) +
                                         authored_target_count());
  }

  iree_host_size_t source_value_count() const {
    return source_module_->values.count;
  }

  iree_host_size_t output_owned_bytes() const { return output_owned_bytes_; }

  iree_host_size_t output_used_bytes() const { return output_used_bytes_; }

 private:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_CHECK_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                static_cast<uint16_t>(count)));
  }

  SealingShape shape_;
  int64_t function_count_;
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* source_module_ = nullptr;
  std::vector<loom_target_facts_t> facts_;
  std::vector<loom_func_like_t> functions_;
  std::vector<loom_target_function_version_t> versions_;
  std::vector<loom_function_version_t*> version_handles_;
  loom_function_version_list_t version_list_ = {};
  iree_host_size_t output_owned_bytes_ = 0;
  iree_host_size_t output_used_bytes_ = 0;
};

static void RunSealingBenchmark(benchmark::State& state, SealingShape shape) {
  ModuleSealingFixture fixture(shape, state.range(0));
  for (auto _ : state) {
    loom_module_t* sealed_module = fixture.Seal();
    benchmark::DoNotOptimize(sealed_module);
    loom_module_free(sealed_module);
  }
  state.SetItemsProcessed(state.iterations() * fixture.function_count());
  state.SetComplexityN(fixture.function_count());
  state.counters["authored_targets"] =
      static_cast<double>(fixture.authored_target_count());
  state.counters["functions"] = static_cast<double>(fixture.function_count());
  state.counters["output_owned_bytes"] =
      static_cast<double>(fixture.output_owned_bytes());
  state.counters["output_used_bytes"] =
      static_cast<double>(fixture.output_used_bytes());
  state.counters["source_ops"] = static_cast<double>(fixture.source_op_count());
  state.counters["source_values"] =
      static_cast<double>(fixture.source_value_count());
  state.counters["target_groups"] = static_cast<double>(fixture.target_count());
}

static void BM_CloneOnly(benchmark::State& state) {
  RunSealingBenchmark(state, SealingShape::kCloneOnly);
}

static void BM_SharedTarget(benchmark::State& state) {
  RunSealingBenchmark(state, SealingShape::kSharedTarget);
}

static void BM_DistinctTargets(benchmark::State& state) {
  RunSealingBenchmark(state, SealingShape::kDistinctTargets);
}

static void BM_DistinctAuthoredTargets(benchmark::State& state) {
  RunSealingBenchmark(state, SealingShape::kDistinctAuthoredTargets);
}

static void RegisterScalingArguments(benchmark::Benchmark* value) {
  for (int64_t scale : {1, 16, 64, 512, 4096}) value->Arg(scale);
}

BENCHMARK(BM_CloneOnly)
    ->Apply(RegisterScalingArguments)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();
BENCHMARK(BM_SharedTarget)
    ->Apply(RegisterScalingArguments)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();
BENCHMARK(BM_DistinctTargets)
    ->Apply(RegisterScalingArguments)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();
BENCHMARK(BM_DistinctAuthoredTargets)
    ->Apply(RegisterScalingArguments)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

}  // namespace
