// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/place_loop_invariants.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/source_pressure.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/scf/residency.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/reporting/report.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct DiagnosticCollector {
  int count = 0;
  const loom_error_def_t* last_error = nullptr;
};

static iree_status_t CollectDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticCollector*>(user_data);
  ++collector->count;
  collector->last_error = emission->error;
  return iree_ok_status();
}

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

static const loom_target_residency_model_t* SelectResidencyModel(
    void* user_data,
    const loom_target_contract_query_environment_t* environment) {
  (void)environment;
  return static_cast<const loom_target_residency_model_t*>(user_data);
}

static iree_status_t SelectPressureReserve(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    loom_low_lower_pressure_reserve_list_t* out_reserves) {
  (void)environment;
  const auto* reserve =
      static_cast<const loom_low_lower_pressure_reserve_t*>(user_data);
  *out_reserves = {
      /*.values=*/reserve,
      /*.count=*/1,
      /*.flags=*/LOOM_LOW_LOWER_PRESSURE_RESERVE_FLAG_COMPLETE,
  };
  return iree_ok_status();
}

class PlaceLoopInvariantsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &descriptor_registry_.registry, IREE_SV("test.low.core"));
    IREE_ASSERT_NE(descriptor_set_, nullptr);
    resource_names_.resize(descriptor_set_->reg_class_count,
                           IREE_SVL("test_registers"));
    cliff_ranges_.resize(descriptor_set_->reg_class_count);

    policy_ = *loom_test_low_lower_policy();
    policy_.residency_model = {
        /*.fn=*/SelectResidencyModel,
        /*.user_data=*/&residency_model_,
    };
    policy_entry_ = {
        /*.contract_set_key=*/IREE_SVL("test.low.core"),
        /*.policy=*/&policy_,
    };
    loom_low_lower_policy_registry_initialize_from_entries(&policy_registry_,
                                                           &policy_entry_, 1);
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

  void SetCliff(iree_string_view_t register_class_name, uint32_t cliff_units) {
    uint16_t resource_id = 0;
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptor_set_, register_class_name, &resource_id, nullptr));
    cliff_ = {
        /*.resource_id=*/resource_id,
        /*.cliff_units=*/cliff_units,
        /*.tier_before=*/4,
        /*.tier_after=*/2,
    };
    for (iree_host_size_t i = 0; i < cliff_ranges_.size(); ++i) {
      cliff_ranges_[i] = {
          /*.start=*/i <= resource_id ? 0u : 1u,
          /*.count=*/i == resource_id ? 1u : 0u,
      };
    }
    residency_model_ = {
        /*.best_tier=*/4,
        /*.direct_resources=*/
        {
            /*.names=*/resource_names_.data(),
            /*.cliffs=*/&cliff_,
            /*.cliff_count=*/1,
            /*.cliff_ranges=*/cliff_ranges_.data(),
            /*.resource_count=*/(uint16_t)resource_names_.size(),
        },
    };
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("place_loop_invariants_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return loom_func_like_cast(module,
                               module->symbols.entries[symbol_id].defining_op);
  }

  iree_status_t Run(loom_module_t* module,
                    DiagnosticCollector* collector = nullptr,
                    loom_target_compile_report_t* compile_report = nullptr) {
    iree_arena_allocator_t instance_arena;
    iree_arena_initialize(&block_pool_, &instance_arena);
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    const loom_pass_info_t* pass_info = loom_place_loop_invariants_pass_info();
    std::vector<uint8_t> statistics(pass_info->statistic_layout->storage_size,
                                    0);
    loom_low_pass_environment_storage_t environment_storage;
    loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            &descriptor_registry_.registry, &policy_registry_, nullptr, nullptr,
            nullptr, compile_report, loom_target_selection_empty(),
            loom_symbol_ref_null(), &environment_storage);
    loom_pass_t pass = {};
    pass.info = pass_info;
    pass.module_run = loom_place_loop_invariants_run;
    pass.instance_arena = &instance_arena;
    pass.arena = &instance_arena;
    pass.statistic_storage = statistics.data();
    pass.environment = &environment;
    pass.value_facts = &value_facts;
    if (collector != nullptr) {
      pass.diagnostic_emitter = {
          /*.fn=*/CollectDiagnostic,
          /*.user_data=*/collector,
      };
    }
    const iree_status_t status = loom_place_loop_invariants_run(&pass, module);
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&instance_arena);
    return status;
  }

  static iree_host_size_t CountDirectCmp(const loom_region_t* region) {
    iree_host_size_t count = 0;
    const loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_test_cmp_isa(op)) ++count;
      }
    }
    return count;
  }

  static iree_host_size_t CountDirectAddf(const loom_region_t* region) {
    iree_host_size_t count = 0;
    const loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_scalar_addf_isa(op)) ++count;
      }
    }
    return count;
  }

  static iree_host_size_t CountDirectCmpf(const loom_region_t* region) {
    iree_host_size_t count = 0;
    const loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_scalar_cmpf_isa(op)) ++count;
      }
    }
    return count;
  }

  static loom_op_t* FindDirectFor(loom_region_t* region) {
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_scf_for_isa(op)) return op;
      }
    }
    return nullptr;
  }

  static loom_op_t* FindDirectIf(loom_region_t* region) {
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_scf_if_isa(op)) return op;
      }
    }
    return nullptr;
  }

  static loom_op_t* FindDirectResidencyRequirement(loom_region_t* region) {
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_scf_residency_require_isa(op)) return op;
      }
    }
    return nullptr;
  }

  static loom_op_t* FindDirectResidencyCandidate(loom_region_t* region) {
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_scf_residency_candidate_isa(op)) return op;
      }
    }
    return nullptr;
  }

  static iree_host_size_t CountDirectResidencyCandidates(
      loom_region_t* region) {
    iree_host_size_t count = 0;
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_scf_residency_candidate_isa(op)) ++count;
      }
    }
    return count;
  }

  loom_low_source_pressure_t AnalyzePressure(loom_module_t* module,
                                             loom_func_like_t function,
                                             const loom_region_t* region,
                                             iree_arena_allocator_t* arena) {
    const loom_target_contract_query_environment_t environment = {
        /*.module=*/module,
        /*.function=*/function,
        /*.bundle=*/{},         /*.target_data=*/{},
        /*.target_ref=*/{},     /*.descriptor_set=*/descriptor_set_,
    };
    const loom_low_source_pressure_options_t options = {
        /*.region=*/region,
    };
    loom_low_source_pressure_t pressure = {};
    IREE_CHECK_OK(loom_low_source_pressure_analyze(&environment, &policy_,
                                                   &options, arena, &pressure));
    return pressure;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_low_lower_policy_t policy_ = {};
  loom_low_lower_policy_registry_entry_t policy_entry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
  std::vector<iree_string_view_t> resource_names_;
  std::vector<loom_target_residency_cliff_range_t> cliff_ranges_;
  loom_target_residency_cliff_t cliff_ = {};
  loom_target_residency_model_t residency_model_ = {};
};

TEST_F(PlaceLoopInvariantsTest, HoistsInvariantAndConsumesPreservePolicy) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %seed: i32, %x: i32, %y: i32) -> (i32) {
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : i32) -> (i32) residency(preserve) {
    %less = test.cmp lt, %x, %y : i32
    %next = scf.select %less, %acc, %x : i32
    scf.yield %next : i32
  }
  func.return %result : i32
}
)");
  const loom_func_like_t source_function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_op_t* source_loop =
      loom_region_entry_block(loom_func_like_body(source_function))->first_op;
  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(&block_pool_, &analysis_arena);
  const loom_low_source_pressure_t baseline =
      AnalyzePressure(module.get(), source_function,
                      loom_scf_for_body(source_loop), &analysis_arena);
  EXPECT_EQ(baseline.peak_direct_resource_units[0], 4u);
  iree_arena_deinitialize(&analysis_arena);
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  ASSERT_EQ(CountDirectCmp(body), 1u);
  loom_op_t* requirement = FindDirectResidencyRequirement(body);
  ASSERT_NE(requirement, nullptr);
  EXPECT_EQ(loom_scf_residency_require_minimum(requirement), -1);
  EXPECT_TRUE(loom_scf_residency_require_preserve(requirement));
  EXPECT_EQ(loom_scf_residency_require_projected_baseline(requirement), 4);
  loom_op_t* candidate = FindDirectResidencyCandidate(body);
  ASSERT_NE(candidate, nullptr);
  EXPECT_EQ(loom_scf_residency_candidate_candidate_id(candidate), 0);
  EXPECT_EQ(loom_scf_residency_candidate_recompute_cost(candidate), 1);
  EXPECT_TRUE(loom_scf_residency_candidate_preserves_baseline(candidate));
  IREE_EXPECT_OK(loom_scf_residency_candidate_validate_proven_source(
      module.get(), candidate));
  EXPECT_TRUE(loom_test_cmp_isa(loom_value_def_op(loom_module_value(
      module.get(), loom_scf_residency_candidate_source(candidate)))));
  EXPECT_GT(loom_module_value(module.get(),
                              loom_scf_residency_candidate_result(candidate))
                ->use_count,
            0u);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_TRUE(loom_scf_for_isa(loop));
  EXPECT_EQ(CountDirectCmp(loom_scf_for_body(loop)), 0u);
  EXPECT_TRUE(loom_attr_is_absent(
      loom_op_attrs(loop)[loom_scf_for_residency_policy_ATTR_INDEX]));
}

TEST_F(PlaceLoopInvariantsTest,
       AutomaticModeRejectsOnlyCumulativeCliffCrossingMove) {
  SetCliff(IREE_SV("test.f32"), 7);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %seed: f32, %x: f32, %y: f32, %z: f32) -> (f32) {
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : f32) -> (f32) {
    %t0 = scalar.remf %acc, %z : f32
    %t1 = scalar.remf %t0, %z : f32
    %p = scalar.addf %x, %y : f32
    %q = scalar.addf %x, %y : f32
    test.use %t1 : f32
    test.use %p : f32
    test.use %q : f32
    test.use %x : f32
    test.use %y : f32
    scf.yield %t1 : f32
  }
func.return %result : f32
}
)");
  const loom_func_like_t source_function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_op_t* source_loop =
      loom_region_entry_block(loom_func_like_body(source_function))->first_op;
  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(&block_pool_, &analysis_arena);
  const loom_low_source_pressure_t baseline =
      AnalyzePressure(module.get(), source_function,
                      loom_scf_for_body(source_loop), &analysis_arena);
  uint16_t f32_resource_id = 0;
  ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set_, IREE_SV("test.f32"), &f32_resource_id, nullptr));
  EXPECT_EQ(baseline.peak_direct_resource_units[f32_resource_id], 5u);
  iree_arena_deinitialize(&analysis_arena);
  loom_target_compile_report_t compile_report;
  loom_target_compile_report_initialize(&compile_report,
                                        iree_allocator_system());
  compile_report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  IREE_ASSERT_OK(Run(module.get(), nullptr, &compile_report));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  EXPECT_EQ(CountDirectAddf(body), 1u);
  loom_op_t* requirement = FindDirectResidencyRequirement(body);
  ASSERT_NE(requirement, nullptr);
  EXPECT_EQ(loom_scf_residency_require_minimum(requirement), -1);
  EXPECT_TRUE(loom_scf_residency_require_preserve(requirement));
  EXPECT_EQ(loom_scf_residency_require_projected_baseline(requirement), 4);
  EXPECT_EQ(CountDirectResidencyCandidates(body), 1u);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_TRUE(loom_scf_for_isa(loop));
  EXPECT_EQ(CountDirectAddf(loom_scf_for_body(loop)), 1u);

  ASSERT_EQ(compile_report.source_low_residency_rows.count, 1u);
  const auto* residency_rows =
      static_cast<const loom_target_compile_report_source_low_residency_row_t*>(
          loom_target_compile_report_vec_const_rows(
              compile_report.source_low_residency_rows.head));
  ASSERT_NE(residency_rows, nullptr);
  const auto& residency = residency_rows[0];
  EXPECT_TRUE(
      iree_string_view_equal(residency.function_name, IREE_SV("kernel")));
  EXPECT_TRUE(iree_string_view_equal(residency.policy, IREE_SV("automatic")));
  EXPECT_TRUE(iree_string_view_equal(residency.outcome, IREE_SV("selected")));
  EXPECT_TRUE(
      iree_string_view_equal(residency.reason, IREE_SV("residency_cliff")));
  EXPECT_EQ(residency.baseline_tier, 4u);
  EXPECT_EQ(residency.selected_tier, 4u);
  EXPECT_EQ(residency.required_tier, 4u);
  EXPECT_EQ(residency.candidate_count, 2u);
  EXPECT_EQ(residency.selected_count, 1u);
  EXPECT_EQ(residency.rejected_count, 1u);
  EXPECT_EQ(residency.reserve_count, 0u);
  EXPECT_FALSE(residency.projection_complete);
  ASSERT_EQ(compile_report.residency_candidate_rows.count, 2u);
  const auto* selected_candidate =
      CompileReportRowAt<loom_target_compile_report_residency_candidate_row_t>(
          compile_report.residency_candidate_rows, 0);
  const auto* retained_candidate =
      CompileReportRowAt<loom_target_compile_report_residency_candidate_row_t>(
          compile_report.residency_candidate_rows, 1);
  ASSERT_NE(selected_candidate, nullptr);
  ASSERT_NE(retained_candidate, nullptr);
  EXPECT_EQ(selected_candidate->candidate_id, 0u);
  EXPECT_EQ(retained_candidate->candidate_id, 1u);
  EXPECT_TRUE(iree_string_view_equal(selected_candidate->source_op_name,
                                     IREE_SV("scalar.addf")));
  EXPECT_TRUE(iree_string_view_equal(retained_candidate->source_op_name,
                                     IREE_SV("scalar.addf")));
  EXPECT_TRUE(
      iree_string_view_equal(selected_candidate->stage, IREE_SV("source")));
  EXPECT_TRUE(
      iree_string_view_equal(selected_candidate->outcome, IREE_SV("selected")));
  EXPECT_TRUE(
      iree_string_view_equal(retained_candidate->outcome, IREE_SV("retained")));
  EXPECT_EQ(selected_candidate->projected_recompute_cost, 1u);
  EXPECT_EQ(retained_candidate->projected_recompute_cost, 1u);
  EXPECT_TRUE(selected_candidate->preserves_baseline);
  EXPECT_TRUE(retained_candidate->preserves_baseline);
  ASSERT_EQ(compile_report.source_low_residency_resource_rows.count,
            2u * descriptor_set_->reg_class_count);
  const loom_target_compile_report_source_low_residency_resource_row_t*
      baseline_f32 = nullptr;
  const loom_target_compile_report_source_low_residency_resource_row_t*
      selected_f32 = nullptr;
  for (iree_host_size_t i = 0;
       i < compile_report.source_low_residency_resource_rows.count; ++i) {
    const auto* resource = CompileReportRowAt<
        loom_target_compile_report_source_low_residency_resource_row_t>(
        compile_report.source_low_residency_resource_rows, i);
    ASSERT_NE(resource, nullptr);
    if (resource->resource_id != f32_resource_id) continue;
    if (iree_string_view_equal(resource->phase, IREE_SV("baseline"))) {
      baseline_f32 = resource;
    } else if (iree_string_view_equal(resource->phase, IREE_SV("selected"))) {
      selected_f32 = resource;
    }
  }
  ASSERT_NE(baseline_f32, nullptr);
  ASSERT_NE(selected_f32, nullptr);
  EXPECT_EQ(baseline_f32->tier, 4u);
  EXPECT_EQ(selected_f32->tier, 4u);
  EXPECT_EQ(baseline_f32->next_worse_tier, 2u);
  EXPECT_EQ(selected_f32->next_worse_tier, 2u);
  EXPECT_EQ(baseline_f32->next_worse_cliff_units, 7u);
  EXPECT_EQ(selected_f32->next_worse_cliff_units, 7u);
  EXPECT_TRUE(
      iree_string_view_equal(baseline_f32->resource_kind, IREE_SV("direct")));
  EXPECT_EQ(compile_report.source_low_residency_reserve_rows.count, 0u);
  loom_target_compile_report_deinitialize(&compile_report);
}

TEST_F(PlaceLoopInvariantsTest, ReportsNamedPressureReserveContributions) {
  SetCliff(IREE_SV("test.f32"), 64);
  uint16_t f32_resource_id = 0;
  ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set_, IREE_SV("test.f32"), &f32_resource_id, nullptr));
  std::vector<uint64_t> reserve_units(descriptor_set_->reg_class_count, 0);
  reserve_units[f32_resource_id] = 2;
  loom_low_lower_pressure_reserve_t reserve = {
      /*.name=*/IREE_SVL("target_abi"),
      /*.direct_resource_units=*/reserve_units.data(),
      /*.direct_resource_count=*/reserve_units.size(),
  };
  policy_.pressure_reserves = {
      /*.fn=*/SelectPressureReserve,
      /*.user_data=*/&reserve,
  };

  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %seed: f32, %x: f32, %y: f32) -> (f32) {
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : f32) -> (f32) {
    %sum = scalar.addf %x, %y : f32
    test.use %sum : f32
    scf.yield %acc : f32
  }
  func.return %result : f32
}
)");
  loom_target_compile_report_t compile_report;
  loom_target_compile_report_initialize(&compile_report,
                                        iree_allocator_system());
  compile_report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  IREE_ASSERT_OK(Run(module.get(), nullptr, &compile_report));

  ASSERT_EQ(compile_report.source_low_residency_rows.count, 1u);
  const auto* placement =
      CompileReportRowAt<loom_target_compile_report_source_low_residency_row_t>(
          compile_report.source_low_residency_rows, 0);
  ASSERT_NE(placement, nullptr);
  EXPECT_EQ(placement->reserve_count, 1u);
  EXPECT_TRUE(placement->projection_complete);
  ASSERT_EQ(compile_report.source_low_residency_reserve_rows.count, 1u);
  const auto* reserve_row = CompileReportRowAt<
      loom_target_compile_report_source_low_residency_reserve_row_t>(
      compile_report.source_low_residency_reserve_rows, 0);
  ASSERT_NE(reserve_row, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(reserve_row->function_name, IREE_SV("kernel")));
  EXPECT_TRUE(
      iree_string_view_equal(reserve_row->reserve_name, IREE_SV("target_abi")));
  EXPECT_EQ(reserve_row->resource_id, f32_resource_id);
  EXPECT_EQ(reserve_row->units, 2u);

  loom_target_compile_report_deinitialize(&compile_report);
}

TEST_F(PlaceLoopInvariantsTest, NumericMinimumCanTradeDownFromAuthoredTier) {
  SetCliff(IREE_SV("test.f32"), 7);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %seed: f32, %x: f32, %y: f32, %z: f32) -> (f32) {
  %minimum = scalar.constant 2 : index
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : f32) -> (f32) residency(%minimum) {
    %t0 = scalar.remf %acc, %z : f32
    %t1 = scalar.remf %t0, %z : f32
    %p = scalar.addf %x, %y : f32
    %q = scalar.addf %x, %y : f32
    test.use %t1 : f32
    test.use %p : f32
    test.use %q : f32
    test.use %x : f32
    test.use %y : f32
    scf.yield %t1 : f32
  }
  func.return %result : f32
}
)");
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  EXPECT_EQ(CountDirectAddf(body), 2u);
  EXPECT_EQ(CountDirectResidencyCandidates(body), 2u);
  loom_op_t* requirement = FindDirectResidencyRequirement(body);
  ASSERT_NE(requirement, nullptr);
  EXPECT_EQ(loom_scf_residency_require_minimum(requirement), 2);
  EXPECT_FALSE(loom_scf_residency_require_preserve(requirement));
  EXPECT_EQ(loom_scf_residency_require_projected_baseline(requirement), 0);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(CountDirectAddf(loom_scf_for_body(loop)), 0u);
}

TEST_F(PlaceLoopInvariantsTest, MovesDependencyClosureAcrossTransientCliff) {
  SetCliff(IREE_SV("test.f32"), 6);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %seed: f32, %x: f32, %y: f32, %z: f32) -> (f32) {
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : f32) -> (f32) residency(preserve) {
    %t0 = scalar.remf %acc, %z : f32
    %t1 = scalar.remf %t0, %z : f32
    test.use %t1 : f32
    test.use %x, %y : f32, f32
    %provider = scalar.addf %x, %y : f32
    %consumer = scalar.cmpf olt, %provider, %z : f32
    test.use %consumer : i1
    scf.yield %t1 : f32
  }
  func.return %result : f32
}
)");
  const loom_func_like_t source_function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* source_body = loom_func_like_body(source_function);
  loom_op_t* source_loop = FindDirectFor(source_body);
  ASSERT_NE(source_loop, nullptr);
  loom_block_t* loop_block =
      loom_region_entry_block(loom_scf_for_body(source_loop));
  loom_op_t* provider = nullptr;
  loom_op_t* consumer = nullptr;
  loom_op_t* nested_op = nullptr;
  loom_block_for_each_op(loop_block, nested_op) {
    if (loom_scalar_addf_isa(nested_op)) provider = nested_op;
    if (loom_scalar_cmpf_isa(nested_op)) consumer = nested_op;
  }
  ASSERT_NE(provider, nullptr);
  ASSERT_NE(consumer, nullptr);
  loom_op_t* provider_restore_before = provider->next_op;
  loom_op_t* consumer_restore_before = consumer->next_op;

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(&block_pool_, &analysis_arena);
  const loom_low_source_pressure_t baseline = AnalyzePressure(
      module.get(), source_function, source_body, &analysis_arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module.get(), &analysis_arena));
  IREE_ASSERT_OK(loom_rewriter_move_before(&rewriter, provider, source_loop));
  const loom_low_source_pressure_t provider_only = AnalyzePressure(
      module.get(), source_function, source_body, &analysis_arena);
  IREE_ASSERT_OK(loom_rewriter_move_before(&rewriter, consumer, source_loop));
  const loom_low_source_pressure_t complete_closure = AnalyzePressure(
      module.get(), source_function, source_body, &analysis_arena);
  uint16_t f32_resource_id = 0;
  ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set_, IREE_SV("test.f32"), &f32_resource_id, nullptr));
  EXPECT_EQ(baseline.peak_direct_resource_units[f32_resource_id], 5u);
  EXPECT_EQ(provider_only.peak_direct_resource_units[f32_resource_id], 6u);
  EXPECT_EQ(complete_closure.peak_direct_resource_units[f32_resource_id], 5u);
  EXPECT_EQ(baseline.minimum_tier, 4u);
  EXPECT_EQ(provider_only.minimum_tier, 2u);
  EXPECT_EQ(complete_closure.minimum_tier, 4u);
  IREE_ASSERT_OK(
      loom_rewriter_move_before(&rewriter, consumer, consumer_restore_before));
  IREE_ASSERT_OK(
      loom_rewriter_move_before(&rewriter, provider, provider_restore_before));
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&analysis_arena);

  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  EXPECT_EQ(CountDirectAddf(body), 1u);
  EXPECT_EQ(CountDirectCmpf(body), 1u);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(CountDirectAddf(loom_scf_for_body(loop)), 0u);
  EXPECT_EQ(CountDirectCmpf(loom_scf_for_body(loop)), 0u);
}

TEST_F(PlaceLoopInvariantsTest, PlacesInnerInvariantAtOuterLoopBoundary) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %seed: i32, %x: i32) -> (i32) {
  %result = scf.for %outer_iv = [%start to %end step %step](%outer_acc = %seed : i32) -> (i32) residency(preserve) {
    scf.for %inner_iv = [%start to %end step %step] residency(preserve) {
      %less = test.cmp lt, %outer_acc, %x : i32
      test.use %less : i1
      scf.yield
    }
    scf.yield %outer_acc : i32
  }
  func.return %result : i32
}
)");
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* function_body = loom_func_like_body(function);
  EXPECT_EQ(CountDirectCmp(function_body), 0u);
  loom_op_t* outer_loop = FindDirectFor(function_body);
  ASSERT_NE(outer_loop, nullptr);
  loom_region_t* outer_body = loom_scf_for_body(outer_loop);
  EXPECT_EQ(CountDirectCmp(outer_body), 1u);
  loom_op_t* inner_loop = FindDirectFor(outer_body);
  ASSERT_NE(inner_loop, nullptr);
  EXPECT_EQ(CountDirectCmp(loom_scf_for_body(inner_loop)), 0u);
}

TEST_F(PlaceLoopInvariantsTest, PlacesIndependentSiblingLoopClosures) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %x: i32, %y: i32) {
  scf.for %first_iv = [%start to %end step %step] residency(preserve) {
    %first = test.cmp lt, %x, %y : i32
    test.use %first : i1
    scf.yield
  }
  scf.for %second_iv = [%start to %end step %step] residency(preserve) {
    %second = test.cmp gt, %x, %y : i32
    test.use %second : i1
    scf.yield
  }
  func.return
}
)");
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* function_body = loom_func_like_body(function);
  EXPECT_EQ(CountDirectCmp(function_body), 2u);
  loom_block_t* block = loom_region_entry_block(function_body);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(block, op) {
    if (loom_scf_for_isa(op)) {
      EXPECT_EQ(CountDirectCmp(loom_scf_for_body(op)), 0u);
    }
  }
}

TEST_F(PlaceLoopInvariantsTest, KeepsConditionalInvariantInItsControlPath) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%condition: i1, %start: index, %end: index, %step: index, %x: i32, %y: i32) {
  scf.if %condition {
    scf.for %iv = [%start to %end step %step] residency(preserve) {
      %less = test.cmp lt, %x, %y : i32
      test.use %less : i1
      scf.yield
    }
    scf.yield
  }
  func.return
}
)");
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* function_body = loom_func_like_body(function);
  EXPECT_EQ(CountDirectCmp(function_body), 0u);
  loom_op_t* if_op = FindDirectIf(function_body);
  ASSERT_NE(if_op, nullptr);
  loom_region_t* then_region = loom_scf_if_then_region(if_op);
  EXPECT_EQ(CountDirectCmp(then_region), 1u);
  loom_op_t* loop = FindDirectFor(then_region);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(CountDirectCmp(loom_scf_for_body(loop)), 0u);
}

TEST_F(PlaceLoopInvariantsTest, MinimumPolicyPreservesUnrollContract) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%end: index, %x: i32, %y: i32) {
  %start = scalar.constant 0 : index
  %step = scalar.constant 1 : index
  %factor = scalar.constant 2 : index
  %minimum = scalar.constant 4 : index
  scf.for %iv = [%start to %end step %step] unroll(%factor) schedule(interleaved) residency(%minimum) {
    %less = test.cmp lt, %x, %y : i32
    test.use %less : i1
    scf.yield
  }
  func.return
}
)");
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_NE(loop, nullptr);
  EXPECT_TRUE(loom_scf_for_unroll_factor_is_present(loop));
  EXPECT_EQ(loom_scf_for_unroll_schedule(loop),
            LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED);
  EXPECT_FALSE(loom_scf_for_residency_minimum_is_present(loop));
  EXPECT_TRUE(loom_attr_is_absent(
      loom_op_attrs(loop)[loom_scf_for_residency_policy_ATTR_INDEX]));
  EXPECT_EQ(CountDirectCmp(body), 1u);
  EXPECT_EQ(CountDirectCmp(loom_scf_for_body(loop)), 0u);
  loom_op_t* requirement = FindDirectResidencyRequirement(body);
  ASSERT_NE(requirement, nullptr);
  EXPECT_EQ(loom_scf_residency_require_minimum(requirement), 4);
  EXPECT_FALSE(loom_scf_residency_require_preserve(requirement));
}

TEST_F(PlaceLoopInvariantsTest, ResolvesMinimumThroughExactValueFacts) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%end: index) {
  %start = scalar.constant 0 : index
  %step = scalar.constant 1 : index
  %two = scalar.constant 2 : index
  %minimum = scalar.addi %two, %two : index
  scf.for %iv = [%start to %end step %step] residency(%minimum) {
    scf.yield
  }
  func.return
}
)");
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_NE(loop, nullptr);
  EXPECT_FALSE(loom_scf_for_residency_minimum_is_present(loop));
  loom_op_t* requirement = FindDirectResidencyRequirement(body);
  ASSERT_NE(requirement, nullptr);
  EXPECT_EQ(loom_scf_residency_require_minimum(requirement), 4);
}

TEST_F(PlaceLoopInvariantsTest, RejectsUnresolvedMinimumValueFacts) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %minimum: index) {
  scf.for %iv = [%start to %end step %step] residency(%minimum) {
    scf.yield
  }
  func.return
}
)");
  DiagnosticCollector collector;
  IREE_ASSERT_OK(Run(module.get(), &collector));

  ASSERT_EQ(collector.count, 1);
  EXPECT_EQ(collector.last_error, LOOM_ERR_LOWERING_049);
  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_op_t* loop = FindDirectFor(loom_func_like_body(function));
  ASSERT_NE(loop, nullptr);
  EXPECT_TRUE(loom_scf_for_residency_minimum_is_present(loop));
}

TEST_F(PlaceLoopInvariantsTest, RejectsNegativeExactMinimum) {
  SetCliff(IREE_SV("test.i32"), 64);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index) {
  %minimum = scalar.constant -1 : index
  scf.for %iv = [%start to %end step %step] residency(%minimum) {
    scf.yield
  }
  func.return
}
)");
  DiagnosticCollector collector;
  IREE_ASSERT_OK(Run(module.get(), &collector));

  ASSERT_EQ(collector.count, 1);
  EXPECT_EQ(collector.last_error, LOOM_ERR_LOWERING_049);
  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_op_t* loop = FindDirectFor(loom_func_like_body(function));
  ASSERT_NE(loop, nullptr);
  EXPECT_TRUE(loom_scf_for_residency_minimum_is_present(loop));
}

TEST_F(PlaceLoopInvariantsTest,
       PreserveRecordsProjectedKernelBaselineForExactRecovery) {
  SetCliff(IREE_SV("test.f32"), 5);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %x: f32, %y: f32) {
  scf.for %iv = [%start to %end step %step] residency(preserve) {
    %sum = scalar.addf %x, %y : f32
    test.use %sum : f32
    scf.yield
  }
  %c0 = scalar.constant 0.0 : f32
  %c1 = scalar.constant 1.0 : f32
  %c2 = scalar.constant 2.0 : f32
  %c3 = scalar.constant 3.0 : f32
  %c4 = scalar.constant 4.0 : f32
  test.use %c0, %c1, %c2, %c3, %c4 : f32, f32, f32, f32, f32
  func.return
}
)");
  IREE_ASSERT_OK(Run(module.get()));

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  loom_op_t* requirement = FindDirectResidencyRequirement(body);
  ASSERT_NE(requirement, nullptr);
  EXPECT_EQ(loom_scf_residency_require_minimum(requirement), -1);
  EXPECT_TRUE(loom_scf_residency_require_preserve(requirement));
  EXPECT_EQ(loom_scf_residency_require_projected_baseline(requirement), 2);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(CountDirectAddf(body), 1u);
  EXPECT_EQ(CountDirectAddf(loom_scf_for_body(loop)), 0u);
}

TEST_F(PlaceLoopInvariantsTest,
       DiagnosesMinimumAfterLegalPlacementSearchIsExhausted) {
  SetCliff(IREE_SV("test.f32"), 3);
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %seed: f32, %x: f32, %y: f32) -> (f32) {
  %minimum = scalar.constant 4 : index
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : f32) -> (f32) residency(%minimum) {
    %sum = scalar.addf %x, %y : f32
    test.use %sum : f32
    scf.yield %acc : f32
  }
  func.return %result : f32
}
)");
  DiagnosticCollector collector;
  IREE_ASSERT_OK(Run(module.get(), &collector));
  ASSERT_EQ(collector.count, 1);
  EXPECT_EQ(collector.last_error, LOOM_ERR_LOWERING_049);

  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_op_t* loop = FindDirectFor(loom_func_like_body(function));
  ASSERT_NE(loop, nullptr);
  EXPECT_TRUE(loom_scf_for_residency_minimum_is_present(loop));
  EXPECT_EQ(FindDirectResidencyRequirement(loom_func_like_body(function)),
            nullptr);
}

TEST_F(PlaceLoopInvariantsTest, RequiresModelForAuthoredPolicy) {
  policy_.residency_model = {};
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index) {
  scf.for %iv = [%start to %end step %step] residency(preserve) {
    scf.yield
  }
  func.return
}
)");
  loom_target_compile_report_t compile_report;
  loom_target_compile_report_initialize(&compile_report,
                                        iree_allocator_system());
  compile_report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  DiagnosticCollector collector;
  IREE_ASSERT_OK(Run(module.get(), &collector, &compile_report));
  ASSERT_EQ(collector.count, 1);
  EXPECT_EQ(collector.last_error, LOOM_ERR_LOWERING_049);

  ASSERT_EQ(compile_report.source_low_residency_rows.count, 1u);
  const auto* residency_rows =
      static_cast<const loom_target_compile_report_source_low_residency_row_t*>(
          loom_target_compile_report_vec_const_rows(
              compile_report.source_low_residency_rows.head));
  ASSERT_NE(residency_rows, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(residency_rows[0].policy, IREE_SV("preserve")));
  EXPECT_TRUE(
      iree_string_view_equal(residency_rows[0].outcome, IREE_SV("rejected")));
  EXPECT_TRUE(iree_string_view_equal(residency_rows[0].reason,
                                     IREE_SV("missing_residency_model")));
  loom_target_compile_report_deinitialize(&compile_report);
}

TEST_F(PlaceLoopInvariantsTest, AutomaticPlacementDoesNotRequireModel) {
  policy_.residency_model = {};
  ModulePtr module = Parse(R"(
test.target<low_core> @test_target
func.def target(@test_target) @kernel(%start: index, %end: index, %step: index, %x: i32, %y: i32) {
  scf.for %iv = [%start to %end step %step] {
    %less = test.cmp lt, %x, %y : i32
    test.use %less : i1
    scf.yield
  }
  func.return
}
)");
  DiagnosticCollector collector;
  IREE_ASSERT_OK(Run(module.get(), &collector));

  EXPECT_EQ(collector.count, 0);
  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel"));
  loom_region_t* body = loom_func_like_body(function);
  EXPECT_EQ(CountDirectCmp(body), 1u);
  loom_op_t* loop = FindDirectFor(body);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(CountDirectCmp(loom_scf_for_body(loop)), 0u);
  EXPECT_EQ(FindDirectResidencyRequirement(body), nullptr);
}

}  // namespace
}  // namespace loom
