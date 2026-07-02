// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

class LowAllocationMaterializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    descriptor_registry_.descriptor_set_providers = kDescriptorSetProviders;
    descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kDescriptorSetProviders);
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

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry_, &options.low_asm_environment);
    IREE_CHECK_OK(
        loom_text_parse(iree_make_cstring_view(source),
                        IREE_SV("allocation_materialization_test.loom"),
                        &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_op_t* FindLowFunction(loom_module_t* module,
                             iree_string_view_t function_name) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_region_entry_block(module->body), op) {
      if (!loom_low_func_def_isa(op)) continue;
      if (iree_string_view_equal(loom_low_diagnostic_function_name(module, op),
                                 function_name)) {
        return op;
      }
    }
    IREE_ASSERT(false, "low function not found");
    return nullptr;
  }

  loom_value_id_t FindValueByName(loom_module_t* module,
                                  iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < module->values.count; ++i) {
      if (iree_string_view_equal(
              loom_low_diagnostic_value_name(module, (loom_value_id_t)i),
              name)) {
        return (loom_value_id_t)i;
      }
    }
    IREE_ASSERT(false, "value name not found");
    return LOOM_VALUE_ID_INVALID;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_low_descriptor_registry_t descriptor_registry_ = {};
};

TEST_F(LowAllocationMaterializationTest, RecomputesTrafficAfterSliceRewrite) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @stale_slice_plan(%wide: reg<test.i32 x4>) -> (reg<test.i32>) asm<test.low.core> {
  %lane = slice %wide[1] : reg<test.i32 x4> -> reg<test.i32>
  return %lane : reg<test.i32>
}
)");
  loom_op_t* function_op =
      FindLowFunction(module.get(), IREE_SV("stale_slice_plan"));
  const loom_value_id_t wide = FindValueByName(module.get(), IREE_SV("wide"));
  const loom_value_id_t lane = FindValueByName(module.get(), IREE_SV("lane"));
  const loom_liveness_value_class_t register_class = {
      /*.type_kind=*/LOOM_TYPE_REGISTER,
  };
  const loom_low_allocation_assignment_t assignments[] = {
      {
          /*.value_id=*/wide,
          /*.value_class=*/register_class,
          /*.descriptor_reg_class_id=*/0,
          /*.start_point=*/0,
          /*.end_point=*/4,
          /*.unit_count=*/4,
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
          /*.location_base=*/0,
          /*.location_count=*/4,
          /*.unit_end_point_start=*/0,
      },
      {
          /*.value_id=*/lane,
          /*.value_class=*/register_class,
          /*.descriptor_reg_class_id=*/0,
          /*.start_point=*/1,
          /*.end_point=*/3,
          /*.unit_count=*/1,
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
          /*.location_base=*/1,
          /*.location_count=*/1,
          /*.unit_end_point_start=*/0,
      },
  };
  const loom_low_allocation_spill_plan_t spill_plans[] = {
      {
          /*.value_id=*/wide,
          /*.assignment_index=*/0,
          /*.slot_index=*/0,
          /*.slot_space=*/LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE,
          /*.byte_size=*/16,
          /*.byte_alignment=*/4,
          /*.store_count=*/1,
          /*.reload_count=*/1,
      },
      {
          /*.value_id=*/lane,
          /*.assignment_index=*/1,
          /*.slot_index=*/1,
          /*.slot_space=*/LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE,
          /*.byte_size=*/4,
          /*.byte_alignment=*/4,
          /*.store_count=*/1,
          /*.reload_count=*/1,
      },
  };
  loom_low_allocation_table_t table = {};
  table.module = module.get();
  table.function_op = function_op;
  table.assignments = assignments;
  table.assignment_count = IREE_ARRAYSIZE(assignments);
  table.spill_plans = spill_plans;
  table.spill_plan_count = IREE_ARRAYSIZE(spill_plans);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_allocation_materialization_result_t result = {};
  loom_low_allocation_materialization_options_t options = {};
  options.has_supported_storage_spaces = true;
  options.supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_PRIVATE;
  options.record_materialized_spills = true;
  IREE_ASSERT_OK(loom_low_allocation_materialize_spills(
      module.get(), &table, &options, &arena, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.storage_count, 2u);
  EXPECT_EQ(result.storage_bytes, 20u);
  EXPECT_EQ(result.spill_count, 1u);
  EXPECT_EQ(result.spill_bytes, 16u);
  EXPECT_EQ(result.reload_count, 1u);
  EXPECT_EQ(result.reload_bytes, 16u);
  ASSERT_EQ(result.materialized_spill_count, 2u);
  ASSERT_NE(result.materialized_spills, nullptr);
  EXPECT_EQ(result.materialized_spills[0].value_id, wide);
  EXPECT_EQ(result.materialized_spills[0].store_count, 1u);
  EXPECT_EQ(result.materialized_spills[0].reload_count, 1u);
  EXPECT_EQ(result.materialized_spills[1].value_id, lane);
  EXPECT_EQ(result.materialized_spills[1].store_count, 0u);
  EXPECT_EQ(result.materialized_spills[1].reload_count, 0u);

  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
