// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_pressure.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const iree_string_view_t kDirectResourceNames[] = {
    IREE_SVL("integer_registers"),
    IREE_SVL("float_registers"),
};

static const loom_target_residency_cliff_range_t kDirectCliffRanges[] = {
    {/*.start=*/0, /*.count=*/0},
    {/*.start=*/0, /*.count=*/0},
};

static const loom_target_residency_derived_member_t kDerivedMembers[] = {
    {
        /*.resource_id=*/0,
        /*.direct_resource_id=*/0,
        /*.contribution_granularity=*/1,
    },
    {
        /*.resource_id=*/0,
        /*.direct_resource_id=*/1,
        /*.contribution_granularity=*/1,
    },
};

static const loom_target_residency_cliff_t kDerivedCliffs[] = {
    {
        /*.resource_id=*/0,
        /*.cliff_units=*/6,
        /*.tier_before=*/4,
        /*.tier_after=*/2,
    },
};

static const loom_target_residency_derived_resource_t kDerivedResources[] = {
    {
        /*.name=*/IREE_SVL("shared_register_pool"),
        /*.pool_units=*/64,
        /*.allocation_granularity=*/1,
        /*.member_start=*/0,
        /*.member_count=*/2,
        /*.cliff_start=*/0,
        /*.cliff_count=*/1,
    },
};

static const loom_target_residency_model_t kResidencyModel = {
    /*.best_tier=*/4,
    /*.direct_resources=*/
    {
        /*.names=*/kDirectResourceNames,
        /*.cliffs=*/nullptr,
        /*.cliff_count=*/0,
        /*.cliff_ranges=*/kDirectCliffRanges,
        /*.resource_count=*/IREE_ARRAYSIZE(kDirectResourceNames),
    },
    /*.derived_resources=*/
    {
        /*.resources=*/kDerivedResources,
        /*.resource_count=*/IREE_ARRAYSIZE(kDerivedResources),
        /*.members=*/kDerivedMembers,
        /*.member_count=*/IREE_ARRAYSIZE(kDerivedMembers),
        /*.cliffs=*/kDerivedCliffs,
        /*.cliff_count=*/IREE_ARRAYSIZE(kDerivedCliffs),
    },
};

static iree_status_t MapContractValue(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  const loom_type_t type =
      loom_module_value_type(environment->module, source_value_id);
  if (!loom_type_is_scalar(type)) return iree_ok_status();
  if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(0, 1);
  } else if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(1, 1);
  }
  return iree_ok_status();
}

static const loom_target_residency_model_t* ResidencyModel(
    void* user_data,
    const loom_target_contract_query_environment_t* environment) {
  (void)user_data;
  (void)environment;
  return &kResidencyModel;
}

static const loom_low_lower_policy_t kPolicy = {
    /*.name=*/IREE_SVL("source-pressure-test"),
    /*.error_catalog=*/{},
    /*.map_type=*/{},
    /*.source_type_supported=*/{},
    /*.map_value=*/{},
    /*.map_contract_value=*/{/*.fn=*/MapContractValue, /*.user_data=*/nullptr},
    /*.pressure_reserves=*/{},
    /*.residency_model=*/{/*.fn=*/ResidencyModel, /*.user_data=*/nullptr},
};

static iree_status_t PressureReserves(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    loom_low_lower_pressure_reserve_list_t* out_reserves) {
  (void)user_data;
  (void)environment;
  static const uint64_t kTargetAbiUnits[] = {2, 4};
  static const loom_low_lower_pressure_reserve_t kReserves[] = {{
      /*.name=*/IREE_SVL("target_abi"),
      /*.direct_resource_units=*/kTargetAbiUnits,
      /*.direct_resource_count=*/IREE_ARRAYSIZE(kTargetAbiUnits),
  }};
  *out_reserves = {
      /*.values=*/kReserves,
      /*.count=*/IREE_ARRAYSIZE(kReserves),
      /*.flags=*/LOOM_LOW_LOWER_PRESSURE_RESERVE_FLAG_COMPLETE,
  };
  return iree_ok_status();
}

class SourcePressureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    descriptor_set_.reg_class_count = 2;
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
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

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("source_pressure_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[symbol_id].defining_op);
    IREE_ASSERT_NE(function.op, nullptr);
    return function;
  }

  loom_low_source_pressure_t Analyze(
      loom_module_t* module, iree_string_view_t function_name,
      const loom_low_source_pressure_options_t* options = nullptr,
      const loom_region_t* region = nullptr,
      const loom_low_lower_policy_t* policy = &kPolicy) {
    const loom_target_contract_query_environment_t environment = {
        /*.module=*/module,
        /*.function=*/FindFunction(module, function_name),
        /*.bundle=*/{},
        /*.target_data=*/{},
        /*.target_ref=*/{},
        /*.descriptor_set=*/&descriptor_set_,
    };
    loom_low_source_pressure_options_t scoped_options =
        options != nullptr ? *options
                           : loom_low_source_pressure_options_empty();
    scoped_options.region = region;
    loom_low_source_pressure_t pressure;
    IREE_CHECK_OK(loom_low_source_pressure_analyze(
        &environment, policy, &scoped_options, &analysis_arena_, &pressure));
    return pressure;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_low_descriptor_set_t descriptor_set_ = {};
};

TEST_F(SourcePressureTest, AlternativeBranchesUseFeasiblePathMaximum) {
  ModulePtr module = ParseModule(R"(
func.def @branches(%condition: i1) {
  scf.if %condition {
    %i0 = scalar.constant 1 : i32
    %i1 = scalar.constant 2 : i32
    %ir0 = scalar.addi %i0, %i1 : i32
    %ir1 = scalar.addi %i0, %i1 : i32
    %ir2 = scalar.addi %i0, %i1 : i32
    %is0 = scalar.addi %ir0, %ir1 : i32
    %is1 = scalar.addi %is0, %ir2 : i32
    scf.yield
  } else {
    %f0 = scalar.constant 1.0 : f32
    %f1 = scalar.constant 2.0 : f32
    %fr0 = scalar.addf %f0, %f1 : f32
    %fr1 = scalar.addf %f0, %f1 : f32
    %fr2 = scalar.addf %f0, %f1 : f32
    %fs0 = scalar.addf %fr0, %fr1 : f32
    %fs1 = scalar.addf %fs0, %fr2 : f32
    scf.yield
  }
  func.return
}
)");
  const loom_low_source_pressure_options_t options = {
      /*.region=*/{},
      /*.reserves=*/{},
      /*.reserve_count=*/{},
      /*.flags=*/LOOM_LOW_SOURCE_PRESSURE_OPTION_FLAG_RESERVES_COMPLETE,
  };
  const loom_low_source_pressure_t pressure =
      Analyze(module.get(), IREE_SV("branches"), &options);

  ASSERT_TRUE(loom_low_source_pressure_projection_complete(&pressure));
  ASSERT_EQ(pressure.direct_resource_count, 2u);
  EXPECT_EQ(pressure.peak_direct_resource_units[0], 5u);
  EXPECT_EQ(pressure.peak_direct_resource_units[1], 5u);
  EXPECT_EQ(pressure.minimum_tier, 4u);
  ASSERT_NE(pressure.minimum_tier_direct_resource_units, nullptr);
  EXPECT_LT(pressure.minimum_tier_direct_resource_units[0] +
                pressure.minimum_tier_direct_resource_units[1],
            6u);
}

TEST_F(SourcePressureTest, JoinLiveValuesContributeOnEveryBranchPath) {
  ModulePtr module = ParseModule(R"(
func.def @branches(%condition: i1, %i_live: i32, %f_live: f32) -> (i32, f32) {
  scf.if %condition {
    %i0 = scalar.constant 1 : i32
    %i1 = scalar.constant 2 : i32
    %ir0 = scalar.addi %i0, %i1 : i32
    %ir1 = scalar.addi %i0, %i1 : i32
    %ir2 = scalar.addi %i0, %i1 : i32
    %is0 = scalar.addi %ir0, %ir1 : i32
    %is1 = scalar.addi %is0, %ir2 : i32
    scf.yield
  } else {
    %f0 = scalar.constant 1.0 : f32
    %f1 = scalar.constant 2.0 : f32
    %fr0 = scalar.addf %f0, %f1 : f32
    %fr1 = scalar.addf %f0, %f1 : f32
    %fr2 = scalar.addf %f0, %f1 : f32
    %fs0 = scalar.addf %fr0, %fr1 : f32
    %fs1 = scalar.addf %fs0, %fr2 : f32
    scf.yield
  }
  func.return %i_live, %f_live : i32, f32
}
)");
  const loom_low_source_pressure_options_t options = {
      /*.region=*/{},
      /*.reserves=*/{},
      /*.reserve_count=*/{},
      /*.flags=*/LOOM_LOW_SOURCE_PRESSURE_OPTION_FLAG_RESERVES_COMPLETE,
  };
  const loom_low_source_pressure_t pressure =
      Analyze(module.get(), IREE_SV("branches"), &options);

  ASSERT_TRUE(loom_low_source_pressure_projection_complete(&pressure));
  EXPECT_EQ(pressure.minimum_tier, 2u);
  ASSERT_NE(pressure.minimum_tier_direct_resource_units, nullptr);
  EXPECT_GT(pressure.minimum_tier_direct_resource_units[0], 0u);
  EXPECT_GT(pressure.minimum_tier_direct_resource_units[1], 0u);
  EXPECT_GE(pressure.minimum_tier_direct_resource_units[0] +
                pressure.minimum_tier_direct_resource_units[1],
            6u);
}

TEST_F(SourcePressureTest, SimultaneousClassesCrossDerivedResourceCliff) {
  ModulePtr module = ParseModule(R"(
func.def @overlap(%i0: i32, %i1: i32, %f0: f32, %f1: f32) -> (i32, f32) {
  %ir0 = scalar.addi %i0, %i1 : i32
  %ir1 = scalar.addi %i0, %i1 : i32
  %is = scalar.addi %ir0, %ir1 : i32
  %fr0 = scalar.addf %f0, %f1 : f32
  %fr1 = scalar.addf %f0, %f1 : f32
  %fs = scalar.addf %fr0, %fr1 : f32
  func.return %is, %fs : i32, f32
}
)");
  const loom_low_source_pressure_options_t options = {
      /*.region=*/{},
      /*.reserves=*/{},
      /*.reserve_count=*/{},
      /*.flags=*/LOOM_LOW_SOURCE_PRESSURE_OPTION_FLAG_RESERVES_COMPLETE,
  };
  const loom_low_source_pressure_t pressure =
      Analyze(module.get(), IREE_SV("overlap"), &options);

  ASSERT_EQ(pressure.minimum_tier, 2u);
  ASSERT_NE(pressure.minimum_tier_direct_resource_units, nullptr);
  EXPECT_GE(pressure.minimum_tier_direct_resource_units[0] +
                pressure.minimum_tier_direct_resource_units[1],
            4u);
}

TEST_F(SourcePressureTest, LoopCarriedAndLiveThroughValuesContribute) {
  ModulePtr module = ParseModule(R"(
func.def @loop(%start: index, %end: index, %step: index, %seed: i32, %live: i32) -> (i32) {
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : i32) -> (i32) {
    %next = scalar.addi %acc, %live : i32
    scf.yield %next : i32
  }
  func.return %result : i32
}
)");
  const loom_low_source_pressure_t pressure =
      Analyze(module.get(), IREE_SV("loop"));

  ASSERT_EQ(pressure.direct_resource_count, 2u);
  EXPECT_GE(pressure.peak_direct_resource_units[0], 2u);
  EXPECT_EQ(pressure.mapped_value_count, 5u);
  EXPECT_EQ(pressure.live_segment_count, 6u);
  EXPECT_EQ(pressure.transient_segment_count, 1u);
}

TEST_F(SourcePressureTest, NamedReservesApplyAtEveryProgramPoint) {
  ModulePtr module = ParseModule(R"(
func.def @empty() {
  func.return
}
)");
  const uint64_t abi_units[] = {2, 4};
  const loom_low_lower_pressure_reserve_t reserves[] = {
      {
          /*.name=*/IREE_SVL("target_abi"),
          /*.direct_resource_units=*/abi_units,
          /*.direct_resource_count=*/IREE_ARRAYSIZE(abi_units),
      },
  };
  const loom_low_source_pressure_options_t options = {
      /*.region=*/{},
      /*.reserves=*/reserves,
      /*.reserve_count=*/IREE_ARRAYSIZE(reserves),
      /*.flags=*/LOOM_LOW_SOURCE_PRESSURE_OPTION_FLAG_RESERVES_COMPLETE,
  };
  const loom_low_source_pressure_t pressure =
      Analyze(module.get(), IREE_SV("empty"), &options);

  EXPECT_TRUE(loom_low_source_pressure_projection_complete(&pressure));
  EXPECT_EQ(pressure.minimum_tier, 2u);
  EXPECT_EQ(pressure.minimum_tier_point, 0u);
  ASSERT_EQ(pressure.reserve_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(pressure.reserves[0].name, IREE_SV("target_abi")));
  EXPECT_EQ(pressure.reserved_direct_resource_units[0], 2u);
  EXPECT_EQ(pressure.reserved_direct_resource_units[1], 4u);
}

TEST_F(SourcePressureTest, LoweringPolicySuppliesCompleteNamedReserves) {
  ModulePtr module = ParseModule(R"(
func.def @empty() {
  func.return
}
)");
  loom_low_lower_policy_t policy = kPolicy;
  policy.pressure_reserves = {
      /*.fn=*/PressureReserves,
      /*.user_data=*/nullptr,
  };
  const uint64_t allocator_units[] = {1, 1};
  const loom_low_lower_pressure_reserve_t caller_reserves[] = {{
      /*.name=*/IREE_SVL("allocator_fragmentation"),
      /*.direct_resource_units=*/allocator_units,
      /*.direct_resource_count=*/IREE_ARRAYSIZE(allocator_units),
  }};
  const loom_low_source_pressure_options_t options = {
      /*.region=*/{},
      /*.reserves=*/caller_reserves,
      /*.reserve_count=*/IREE_ARRAYSIZE(caller_reserves),
  };
  const loom_low_source_pressure_t pressure =
      Analyze(module.get(), IREE_SV("empty"), &options, nullptr, &policy);

  EXPECT_TRUE(loom_low_source_pressure_projection_complete(&pressure));
  EXPECT_EQ(pressure.minimum_tier, 2u);
  ASSERT_EQ(pressure.reserve_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(pressure.reserves[0].name, IREE_SV("target_abi")));
  EXPECT_TRUE(iree_string_view_equal(pressure.reserves[1].name,
                                     IREE_SV("allocator_fragmentation")));
  EXPECT_EQ(pressure.reserved_direct_resource_units[0], 3u);
  EXPECT_EQ(pressure.reserved_direct_resource_units[1], 5u);
}

TEST_F(SourcePressureTest, ScopedRegionDoesNotInheritSiblingMinimum) {
  ModulePtr module = ParseModule(R"(
func.def @scopes(%start: index, %end: index, %step: index, %seed: i32, %live: i32) -> (i32) {
  %result = scf.for %iv = [%start to %end step %step](%acc = %seed : i32) -> (i32) {
    %next = scalar.addi %acc, %live : i32
    scf.yield %next : i32
  }
  %f0 = scalar.constant 1.0 : f32
  %f1 = scalar.constant 2.0 : f32
  %fr0 = scalar.addf %f0, %f1 : f32
  %fr1 = scalar.addf %f0, %f1 : f32
  %fr2 = scalar.addf %f0, %f1 : f32
  %fr3 = scalar.addf %f0, %f1 : f32
  %fs0 = scalar.addf %fr0, %fr1 : f32
  %fs1 = scalar.addf %fr2, %fr3 : f32
  %fs2 = scalar.addf %fs0, %fs1 : f32
  func.return %result : i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("scopes"));
  loom_op_t* loop_op =
      loom_region_entry_block(loom_func_like_body(function))->first_op;
  ASSERT_TRUE(loom_scf_for_isa(loop_op));

  const loom_target_contract_query_environment_t environment = {
      /*.module=*/module.get(),
      /*.function=*/function,
      /*.bundle=*/{},
      /*.target_data=*/{},
      /*.target_ref=*/{},
      /*.descriptor_set=*/&descriptor_set_,
  };
  const loom_region_t* regions[] = {
      nullptr,
      loom_scf_for_body(loop_op),
  };
  loom_low_source_pressure_t pressures[IREE_ARRAYSIZE(regions)] = {};
  IREE_ASSERT_OK(loom_low_source_pressure_analyze_regions(
      &environment, &kPolicy, nullptr, regions, IREE_ARRAYSIZE(regions),
      &analysis_arena_, pressures));

  EXPECT_EQ(pressures[0].minimum_tier, 2u);
  EXPECT_EQ(pressures[1].minimum_tier, 4u);
  EXPECT_EQ(pressures[0].reserved_direct_resource_units,
            pressures[1].reserved_direct_resource_units);
}

}  // namespace
}  // namespace loom
