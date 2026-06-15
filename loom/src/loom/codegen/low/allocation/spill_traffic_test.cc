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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_low_descriptor_registry_t descriptor_registry_ = {};
};

TEST_F(LowAllocationSpillTrafficTest, DetectsMaterializedSpillTraffic) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @roundtrip(%input: reg<test.i32>, %other: reg<test.i32>) -> (reg<test.i32>) {
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

  EXPECT_TRUE(
      loom_low_allocation_spill_traffic_value_requires_register_location(
          module.get(), input));
  EXPECT_TRUE(
      loom_low_allocation_spill_traffic_value_requires_register_location(
          module.get(), reload));
  EXPECT_FALSE(
      loom_low_allocation_spill_traffic_value_requires_register_location(
          module.get(), other));
  EXPECT_FALSE(
      loom_low_allocation_spill_traffic_value_requires_register_location(
          module.get(), storage));
  EXPECT_FALSE(
      loom_low_allocation_spill_traffic_value_requires_register_location(
          module.get(), LOOM_VALUE_ID_INVALID));

  loom_liveness_interval_t interval = {};
  interval.value_id = reload;
  EXPECT_TRUE(
      loom_low_allocation_spill_traffic_interval_requires_register_location(
          module.get(), &interval));
}

}  // namespace
}  // namespace loom
