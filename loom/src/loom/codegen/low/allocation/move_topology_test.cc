// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_topology.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
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

class LowAllocationMoveTopologyTest : public ::testing::Test {
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
    options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry_, &options.low_asm_environment);
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("move_topology_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_op_t* FindLowFunction(loom_module_t* module, iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(loom_low_func_def_isa(op));
    return op;
  }

  const loom_op_t* FindFirstOp(loom_region_t* body,
                               bool (*predicate)(const loom_op_t*)) {
    loom_block_t* block = NULL;
    loom_region_for_each_block(body, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (predicate(op)) {
          return op;
        }
      }
    }
    return NULL;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_low_descriptor_registry_t descriptor_registry_ = {};
};

static const char kBranchOnlyConcatFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @branch_only_concat(%cond: reg<test.i32>, %lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32 x2>) asm<test.low.core> {
  %copy = copy %lhs : reg<test.i32> -> reg<test.i32>
  %pair = concat(%copy, %rhs) : (reg<test.i32>, reg<test.i32>) -> reg<test.i32 x2>
  low.cond_br %cond, ^then, ^else : reg<test.i32>
^then:
  low.br ^join(%pair: reg<test.i32 x2>)
^else:
  low.br ^join(%pair: reg<test.i32 x2>)
^join(%state: reg<test.i32 x2>):
  return %state
}
)";

static const char kMaterializedConcatFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @materialized_concat(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32 x2>) asm<test.low.core> {
  %pair = concat(%lhs, %rhs) : (reg<test.i32>, reg<test.i32>) -> reg<test.i32 x2>
  return %pair
}
)";

TEST_F(LowAllocationMoveTopologyTest, ClassifiesPacketMoveOps) {
  ModulePtr module = ParseModule(kBranchOnlyConcatFunction);
  ASSERT_NE(module.get(), nullptr);
  loom_op_t* function_op =
      FindLowFunction(module.get(), IREE_SV("branch_only_concat"));
  loom_region_t* body = loom_low_func_def_body(function_op);

  const loom_op_t* copy_op = FindFirstOp(body, loom_low_copy_isa);
  const loom_op_t* concat_op = FindFirstOp(body, loom_low_concat_isa);
  const loom_op_t* return_op = FindFirstOp(body, loom_low_return_isa);

  ASSERT_NE(copy_op, nullptr);
  ASSERT_NE(concat_op, nullptr);
  ASSERT_NE(return_op, nullptr);
  EXPECT_TRUE(loom_low_allocation_move_topology_op_has_packet_moves(copy_op));
  EXPECT_TRUE(loom_low_allocation_move_topology_op_has_packet_moves(concat_op));
  EXPECT_FALSE(
      loom_low_allocation_move_topology_op_has_packet_moves(return_op));
}

TEST_F(LowAllocationMoveTopologyTest, ClassifiesConcatMaterialization) {
  ModulePtr branch_module = ParseModule(kBranchOnlyConcatFunction);
  ASSERT_NE(branch_module.get(), nullptr);
  loom_op_t* branch_function =
      FindLowFunction(branch_module.get(), IREE_SV("branch_only_concat"));
  loom_region_t* branch_body = loom_low_func_def_body(branch_function);
  const loom_op_t* branch_concat =
      FindFirstOp(branch_body, loom_low_concat_isa);
  ASSERT_NE(branch_concat, nullptr);

  ModulePtr materialized_module = ParseModule(kMaterializedConcatFunction);
  ASSERT_NE(materialized_module.get(), nullptr);
  loom_op_t* materialized_function = FindLowFunction(
      materialized_module.get(), IREE_SV("materialized_concat"));
  loom_region_t* materialized_body =
      loom_low_func_def_body(materialized_function);
  const loom_op_t* materialized_concat =
      FindFirstOp(materialized_body, loom_low_concat_isa);
  ASSERT_NE(materialized_concat, nullptr);

  EXPECT_FALSE(
      loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
          branch_module.get(), branch_concat));
  EXPECT_TRUE(
      loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
          materialized_module.get(), materialized_concat));
}

}  // namespace
}  // namespace loom
