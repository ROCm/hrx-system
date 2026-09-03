// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_program.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SourceProgramTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

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
                                  IREE_SV("source_program_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return loom_func_like_cast(module,
                               module->symbols.entries[symbol_id].defining_op);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
};

TEST_F(SourceProgramTest, RetainsPreorderAndNestedSubtreeLimits) {
  ModulePtr module = ParseModule(R"(
func.def @nested(%condition: i1, %arg: i32) -> (i32) {
  %before = scalar.addi %arg, %arg : i32
  %selected = scf.if %condition -> (i32) {
    %then_value = scalar.addi %before, %arg : i32
    scf.yield %then_value : i32
  } else {
    %else_value = scalar.subi %before, %arg : i32
    scf.yield %else_value : i32
  }
  %after = scalar.addi %selected, %arg : i32
  func.return %after : i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("nested"));
  ASSERT_TRUE(loom_func_like_isa(function));
  const loom_region_t* body = loom_func_like_body(function);

  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
      module.get(), body, &analysis_arena_, &value_domain));
  loom_source_program_t program = {};
  IREE_ASSERT_OK(loom_source_program_build(module.get(), function.op, body,
                                           &value_domain, &analysis_arena_,
                                           &program));

  ASSERT_EQ(program.region_count, 3u);
  ASSERT_EQ(program.block_count, 3u);
  ASSERT_EQ(program.operation_count, 8u);
  ASSERT_EQ(program.node_count, 11u);

  const loom_source_program_node_kind_t expected_kinds[] = {
      LOOM_SOURCE_PROGRAM_NODE_BLOCK,     LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION, LOOM_SOURCE_PROGRAM_NODE_BLOCK,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION, LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_BLOCK,     LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION, LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION,
  };
  const loom_op_kind_t expected_op_kinds[] = {
      LOOM_OP_SCALAR_ADDI, LOOM_OP_SCF_IF,      LOOM_OP_SCALAR_ADDI,
      LOOM_OP_SCF_YIELD,   LOOM_OP_SCALAR_SUBI, LOOM_OP_SCF_YIELD,
      LOOM_OP_SCALAR_ADDI, LOOM_OP_FUNC_RETURN,
  };
  iree_host_size_t operation_index = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_kinds); ++i) {
    const loom_source_program_node_t* node = &program.nodes[i];
    ASSERT_EQ(node->kind, expected_kinds[i]);
    if (node->kind == LOOM_SOURCE_PROGRAM_NODE_OPERATION) {
      EXPECT_EQ(loom_source_program_node_operation(node)->kind,
                expected_op_kinds[operation_index++]);
    }
  }

  EXPECT_TRUE(iree_any_bit_set(program.nodes[0].flags,
                               LOOM_SOURCE_PROGRAM_NODE_ROOT_ENTRY_BLOCK));
  EXPECT_EQ(program.nodes[0].subtree_limit, program.node_count);
  EXPECT_EQ(program.nodes[1].subtree_limit, 2u);
  EXPECT_EQ(program.nodes[2].subtree_limit, 9u);
  EXPECT_EQ(program.nodes[3].subtree_limit, 6u);
  EXPECT_EQ(program.nodes[6].subtree_limit, 9u);
  EXPECT_EQ(program.nodes[9].subtree_limit, 10u);
  EXPECT_EQ(program.nodes[10].subtree_limit, 11u);
  EXPECT_EQ(program.nodes[0].region_depth, 0u);
  EXPECT_EQ(program.nodes[3].region_depth, 1u);
  EXPECT_EQ(program.nodes[6].region_depth, 1u);
  EXPECT_EQ(program.nodes[0].context_op, function.op);
  EXPECT_EQ(program.nodes[3].context_op,
            loom_source_program_node_operation(&program.nodes[2]));
  EXPECT_EQ(program.nodes[6].context_op,
            loom_source_program_node_operation(&program.nodes[2]));

  loom_local_value_domain_release(&value_domain);
}

}  // namespace
}  // namespace loom
