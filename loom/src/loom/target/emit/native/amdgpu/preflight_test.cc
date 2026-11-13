// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/preflight.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/facts_builder.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;

const loom_low_descriptor_set_t* LookupAmdgpuCdna3DescriptorSet() {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  return loom_low_descriptor_registry_lookup(&registry.registry,
                                             IREE_SV("amdgpu.cdna3.core"));
}

const loom_low_descriptor_set_t* LookupAmdgpuGfx125xDescriptorSet() {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  return loom_low_descriptor_registry_lookup(
      &registry.registry, IREE_SV("amdgpu.rdna4.gfx125x.core"));
}

class AmdgpuNativePreflightTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &table_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&table_arena_);
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

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return loom_symbol_ref_t{
        .module_id = 0,
        .symbol_id = symbol_id,
    };
  }

  void BuildLowFunction(const loom_low_descriptor_set_t* descriptor_set) {
    const loom_amdgpu_target_record_info_t* target_record =
        loom_amdgpu_target_record_default_info_for_descriptor_set(
            descriptor_set->descriptor_set_ordinal);
    IREE_ASSERT(target_record != nullptr);
    const loom_amdgpu_target_info_t* target_info =
        loom_amdgpu_target_info_find_target_by_kind(target_record->target_kind);
    IREE_ASSERT(target_info != nullptr);
    target_facts_ = {};
    loom_target_facts_builder_initialize(&loom_amdgpu_target_fact_type,
                                         target_record->bundle,
                                         &target_facts_.base);
    IREE_ASSERT_LE(target_info->target_kind, UINT8_MAX);
    target_facts_.base.selector = (uint8_t)target_info->target_kind;
    loom_amdgpu_target_identity_initialize(target_info,
                                           &target_facts_.identity);
    loom_amdgpu_target_properties_resolve(&target_facts_.identity,
                                          &target_facts_.base.storage.bundle,
                                          &target_facts_.properties);

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    const loom_symbol_ref_t target_ref = AddSymbol(IREE_SV("target"));
    const loom_symbol_ref_t callee_ref = AddSymbol(IREE_SV("preflight"));
    loom_string_id_t representation_contract = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &module_builder,
        loom_low_descriptor_set_string(descriptor_set,
                                       descriptor_set->key_string_ref),
        &representation_contract));
    IREE_ASSERT_OK(loom_low_func_def_build(
        &module_builder, LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET,
        /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*effect_contract=*/0, /*temperature=*/0, /*inline_policy=*/0,
        /*allocation=*/0, /*schedule=*/0,
        /*descriptor_set=*/representation_contract, target_ref, /*abi=*/0,
        loom_named_attr_slice_t{}, loom_named_attr_slice_t{},
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_t{}, callee_ref,
        /*arg_types=*/nullptr,
        /*arg_types_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function_op_));
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_low_func_def_body(function_op_)),
        &body_builder_);
  }

  loom_low_resolved_target_t ResolvedTarget(
      const loom_low_descriptor_set_t* descriptor_set) const {
    const loom_target_bundle_t* target_bundle =
        loom_target_facts_bundle(&target_facts_.base);
    return loom_low_resolved_target_t{
        .target_facts = &target_facts_.base,
        .target_name = loom_target_facts_identity_name(&target_facts_.base),
        .descriptor_set_key = loom_low_descriptor_set_string(
            descriptor_set, descriptor_set->key_string_ref),
        .feature_bits = target_bundle->config->contract_feature_bits,
        .descriptor_set = descriptor_set,
    };
  }

  loom_low_schedule_table_t Schedule(
      const loom_low_descriptor_set_t* descriptor_set) {
    loom_low_schedule_table_t schedule = {
        .module = module_,
        .function_op = function_op_,
        .target = ResolvedTarget(descriptor_set),
    };
    return schedule;
  }

  loom_low_allocation_table_t Allocation(
      const loom_low_descriptor_set_t* descriptor_set) {
    physical_extents_.assign(descriptor_set->reg_class_count, 0);
    loom_low_allocation_table_t allocation = {
        .module = module_,
        .function_op = function_op_,
        .target = ResolvedTarget(descriptor_set),
    };
    allocation.physical_extents.ends_by_reg_class = physical_extents_.data();
    allocation.physical_extents.count = physical_extents_.size();
    return allocation;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t table_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* function_op_ = nullptr;
  loom_builder_t body_builder_;
  // Complete AMDGPU facts matching the selected descriptor set.
  loom_amdgpu_target_facts_t target_facts_ = {};
  // Allocation-owned extents, including move scratch and excluding fixed state.
  std::vector<uint32_t> physical_extents_;
};

TEST_F(AmdgpuNativePreflightTest, RetainsAccumulatorAndMoveScratchExtents) {
  const loom_low_descriptor_set_t* descriptor_set =
      LookupAmdgpuCdna3DescriptorSet();
  if (descriptor_set == nullptr) {
    GTEST_SKIP() << "amdgpu.cdna3.core is not linked in this build";
  }
  BuildLowFunction(descriptor_set);
  const uint16_t agpr_reg_class_id =
      loom_amdgpu_agpr_reg_class_id(descriptor_set);
  ASSERT_NE(agpr_reg_class_id, UINT8_MAX);

  loom_low_schedule_table_t schedule = Schedule(descriptor_set);

  loom_low_allocation_assignment_t assignment = {
      .value_id = 0,
      .descriptor_reg_class_id = agpr_reg_class_id,
      .unit_count = 4,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_count = 4,
  };

  loom_low_allocation_table_t allocation = Allocation(descriptor_set);
  allocation.assignments = &assignment;
  allocation.assignment_count = 1;
  // A structural move uses a4 beyond the four-register SSA assignment.
  physical_extents_[agpr_reg_class_id] = 5;

  DiagnosticEmissionCapture capture;
  loom_amdgpu_native_preflight_options_t options = {
      .emitter = capture.emitter(),
  };
  loom_amdgpu_native_preflight_t preflight = {};
  IREE_ASSERT_OK(loom_amdgpu_native_preflight_analyze(&schedule, &allocation,
                                                      &options, &preflight));

  EXPECT_EQ(preflight.error_count, 0u);
  EXPECT_TRUE(capture.emissions.empty());
  EXPECT_EQ(preflight.next_free_agpr, 5u);
}

TEST_F(AmdgpuNativePreflightTest, UsesCompleteRetainedRegisterExtents) {
  const loom_low_descriptor_set_t* descriptor_set =
      LookupAmdgpuGfx125xDescriptorSet();
  ASSERT_NE(descriptor_set, nullptr);
  BuildLowFunction(descriptor_set);

  loom_low_schedule_table_t schedule = Schedule(descriptor_set);
  loom_low_allocation_assignment_t assignments[2] = {};
  for (loom_low_allocation_assignment_t& assignment : assignments) {
    assignment.descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
    assignment.unit_count = 1;
    assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    assignment.location_count = 1;
  }
  assignments[0].value_id = 0;
  assignments[0].location_base = 5;
  assignments[1].value_id = 1;
  assignments[1].location_base = 114;

  loom_low_allocation_table_t allocation = Allocation(descriptor_set);
  allocation.assignments = assignments;
  allocation.assignment_count = IREE_ARRAYSIZE(assignments);
  physical_extents_[LOOM_AMDGPU_REG_CLASS_ID_SGPR] = 6;
  // Parallel moves use v8 even though no SSA assignment names that temporary.
  physical_extents_[LOOM_AMDGPU_REG_CLASS_ID_VGPR] = 9;

  loom_amdgpu_native_preflight_t preflight = {};
  IREE_ASSERT_OK(loom_amdgpu_native_preflight_analyze(
      &schedule, &allocation, /*options=*/nullptr, &preflight));

  EXPECT_EQ(preflight.error_count, 0u);
  EXPECT_EQ(preflight.next_free_sgpr, 6u);
  EXPECT_EQ(preflight.next_free_vgpr, 9u);
}

}  // namespace
}  // namespace loom
