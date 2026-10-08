// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_traffic.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

class LowAllocationSpillTrafficTest : public ::testing::Test {
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
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("spill_traffic_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
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

  loom_op_t* FindLowFunction(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(loom_low_func_def_isa(op));
    return op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_low_descriptor_registry_t descriptor_registry_ = {};
};

TEST_F(LowAllocationSpillTrafficTest, DetectsMaterializedSpillTraffic) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

low.func.def target<test.low.core>(@test_target) @roundtrip(%input: reg<test.i32>, %other: reg<test.i32>) -> (reg<test.i32>) {
  %storage = low.storage.reserve {byte_alignment = 4, byte_length = 4} : low.storage<scratch>
  low.spill %input, %storage : reg<test.i32>, low.storage<scratch>
  %reload = low.reload %storage : low.storage<scratch> -> reg<test.i32>
  low.return %reload : reg<test.i32>
}
)");
  const loom_value_id_t input = FindValueByName(module.get(), IREE_SV("input"));
  const loom_value_id_t other = FindValueByName(module.get(), IREE_SV("other"));
  const loom_value_id_t storage =
      FindValueByName(module.get(), IREE_SV("storage"));
  const loom_value_id_t reload =
      FindValueByName(module.get(), IREE_SV("reload"));
  const loom_region_t* function_region = loom_low_func_def_body(
      FindLowFunction(module.get(), IREE_SV("roundtrip")));

  EXPECT_EQ(
      loom_low_allocation_spill_register_requirement_for_value(
          module.get(), function_region, input),
      LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC);
  EXPECT_EQ(
      loom_low_allocation_spill_register_requirement_for_value(
          module.get(), function_region, reload),
      LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC);
  EXPECT_EQ(loom_low_allocation_spill_register_requirement_for_value(
                module.get(), function_region, other),
            LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NONE);
  EXPECT_EQ(loom_low_allocation_spill_register_requirement_for_value(
                module.get(), function_region, storage),
            LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NONE);
  EXPECT_EQ(loom_low_allocation_spill_register_requirement_for_value(
                module.get(), function_region, LOOM_VALUE_ID_INVALID),
            LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NONE);
}

TEST_F(LowAllocationSpillTrafficTest, DetectsNestedRegionArguments) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

low.func.def target<test.low.core>(@test_target) @structured(%condition: reg<test.i32>, %lower: reg<test.i32>, %upper: reg<test.i32>, %step: reg<test.i32>, %seed: reg<test.i32>) -> (reg<test.i32>) asm {
  low.br ^body(%seed: reg<test.i32>)
^body(%forwarded: reg<test.i32>):
  %for_result = low.scf.for signed [%lower to %upper step %step] iter_args(%forwarded: reg<test.i32>) -> (reg<test.i32>) do(%for_iv: reg<test.i32>, %for_state: reg<test.i32>) {
    %for_next = test.add.i32 %for_state, %for_iv
    low.scf.yield %for_next : reg<test.i32>
  }
  %while_result = low.scf.while(%while_before = %for_result : reg<test.i32>) -> (reg<test.i32>) {
    low.scf.condition %condition, %while_before : reg<test.i32>, reg<test.i32>
  } do(%while_body: reg<test.i32>) {
    %while_next = test.add.i32 %while_body, %step
    low.scf.yield %while_next : reg<test.i32>
  }
  return %while_result
}
)");
  const loom_region_t* function_region = loom_low_func_def_body(
      FindLowFunction(module.get(), IREE_SV("structured")));

  const iree_string_view_t nested_argument_names[] = {
      IREE_SV("for_iv"),
      IREE_SV("for_state"),
      IREE_SV("while_before"),
      IREE_SV("while_body"),
  };
  for (iree_string_view_t name : nested_argument_names) {
    EXPECT_EQ(
        loom_low_allocation_spill_register_requirement_for_value(
            module.get(), function_region, FindValueByName(module.get(), name)),
        LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NESTED_REGION_ARGUMENT);
  }

  const iree_string_view_t supported_value_names[] = {
      IREE_SV("forwarded"),  IREE_SV("for_next"),     IREE_SV("for_result"),
      IREE_SV("while_next"), IREE_SV("while_result"),
  };
  for (iree_string_view_t name : supported_value_names) {
    EXPECT_EQ(
        loom_low_allocation_spill_register_requirement_for_value(
            module.get(), function_region, FindValueByName(module.get(), name)),
        LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NONE);
  }
}

}  // namespace
}  // namespace loom
