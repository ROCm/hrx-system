// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/call_graph.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class CallGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t test_vtable_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 test_vtables,
                                                 (uint16_t)test_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
    iree_arena_initialize(&block_pool_, &graph_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&graph_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Creates a function with a body, returns its symbol_id and
  // a builder positioned at the body block.
  struct FuncInfo {
    loom_symbol_id_t symbol_id;
    loom_op_t* func_op;
    loom_region_t* body;
  };

  enum class CallKind {
    kSemantic,
    kLowInternal,
    kLowInvoke,
  };

  loom_symbol_ref_t add_symbol(const char* name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &module_builder_, iree_make_cstring_view(name), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  }

  FuncInfo create_func(const char* name) {
    loom_symbol_ref_t callee = add_symbol(name);
    loom_op_t* func_op = NULL;
    IREE_CHECK_OK(loom_test_func_build(&module_builder_, 0, 0, 0, callee, NULL,
                                       0, NULL, 0, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func_like = loom_func_like_cast(module_, func_op);
    loom_region_t* body = loom_func_like_body(func_like);
    return {(uint16_t)callee.symbol_id, func_op, body};
  }

  void add_call(FuncInfo& caller, uint16_t callee_symbol_id,
                CallKind kind = CallKind::kSemantic) {
    loom_builder_t body_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(caller.body),
                            &body_builder);
    body_builder.ip.parent_op = caller.func_op;
    loom_symbol_ref_t callee_ref = {.module_id = 0,
                                    .symbol_id = callee_symbol_id};
    loom_op_t* call_op = NULL;
    switch (kind) {
      case CallKind::kSemantic: {
        IREE_CHECK_OK(loom_test_invoke_build(&body_builder, callee_ref, NULL, 0,
                                             NULL, 0, NULL, 0,
                                             LOOM_LOCATION_UNKNOWN, &call_op));
        break;
      }
      case CallKind::kLowInternal: {
        IREE_CHECK_OK(loom_test_low_call_build(
            &body_builder, callee_ref, NULL, 0, NULL, 0, NULL, 0,
            LOOM_LOCATION_UNKNOWN, &call_op));
        break;
      }
      case CallKind::kLowInvoke: {
        IREE_CHECK_OK(loom_test_low_invoke_build(
            &body_builder, callee_ref, NULL, 0, NULL, 0, NULL, 0,
            LOOM_LOCATION_UNKNOWN, &call_op));
        break;
      }
    }
  }

  void finalize() { IREE_ASSERT_OK(loom_module_compute_uses(module_)); }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_;
  iree_arena_allocator_t graph_arena_;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_F(CallGraphTest, SingleFunctionNoCalls) {
  auto f = create_func("foo");
  (void)f;
  finalize();

  loom_call_graph_t graph;
  IREE_ASSERT_OK(loom_call_graph_build(module_, &graph_arena_, &graph));

  EXPECT_EQ(graph.node_count, 1u);
  EXPECT_FALSE(loom_call_graph_is_recursive(&graph, f.symbol_id));

  const loom_call_graph_node_t* node =
      loom_call_graph_node(&graph, f.symbol_id);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->callee_count, 0u);
}

TEST_F(CallGraphTest, LinearChain) {
  // A -> B -> C
  auto a = create_func("a");
  auto b = create_func("b");
  auto c = create_func("c");
  add_call(a, b.symbol_id);
  add_call(b, c.symbol_id);
  finalize();

  loom_call_graph_t graph;
  IREE_ASSERT_OK(loom_call_graph_build(module_, &graph_arena_, &graph));

  EXPECT_EQ(graph.node_count, 3u);
  EXPECT_FALSE(loom_call_graph_is_recursive(&graph, a.symbol_id));
  EXPECT_FALSE(loom_call_graph_is_recursive(&graph, b.symbol_id));
  EXPECT_FALSE(loom_call_graph_is_recursive(&graph, c.symbol_id));

  // Each should have the expected callee count.
  const loom_call_graph_node_t* na = loom_call_graph_node(&graph, a.symbol_id);
  const loom_call_graph_node_t* nb = loom_call_graph_node(&graph, b.symbol_id);
  const loom_call_graph_node_t* nc = loom_call_graph_node(&graph, c.symbol_id);
  ASSERT_NE(na, nullptr);
  ASSERT_NE(nb, nullptr);
  ASSERT_NE(nc, nullptr);
  EXPECT_EQ(na->callee_count, 1u);
  EXPECT_EQ(nb->callee_count, 1u);
  EXPECT_EQ(nc->callee_count, 0u);

  // All in different SCCs.
  EXPECT_NE(na->scc_id, nb->scc_id);
  EXPECT_NE(nb->scc_id, nc->scc_id);
}

TEST_F(CallGraphTest, SelfRecursion) {
  auto f = create_func("self");
  add_call(f, f.symbol_id);
  finalize();

  loom_call_graph_t graph;
  IREE_ASSERT_OK(loom_call_graph_build(module_, &graph_arena_, &graph));

  EXPECT_EQ(graph.node_count, 1u);
  EXPECT_TRUE(loom_call_graph_is_recursive(&graph, f.symbol_id));
}

TEST_F(CallGraphTest, MutualRecursion) {
  // A -> B, B -> A
  auto a = create_func("a");
  auto b = create_func("b");
  add_call(a, b.symbol_id);
  add_call(b, a.symbol_id);
  finalize();

  loom_call_graph_t graph;
  IREE_ASSERT_OK(loom_call_graph_build(module_, &graph_arena_, &graph));

  EXPECT_EQ(graph.node_count, 2u);
  EXPECT_TRUE(loom_call_graph_is_recursive(&graph, a.symbol_id));
  EXPECT_TRUE(loom_call_graph_is_recursive(&graph, b.symbol_id));

  // Same SCC.
  const loom_call_graph_node_t* na = loom_call_graph_node(&graph, a.symbol_id);
  const loom_call_graph_node_t* nb = loom_call_graph_node(&graph, b.symbol_id);
  ASSERT_NE(na, nullptr);
  ASSERT_NE(nb, nullptr);
  EXPECT_EQ(na->scc_id, nb->scc_id);
}

TEST_F(CallGraphTest, DisconnectedFunctions) {
  auto a = create_func("a");
  auto b = create_func("b");
  (void)a;
  (void)b;
  finalize();

  loom_call_graph_t graph;
  IREE_ASSERT_OK(loom_call_graph_build(module_, &graph_arena_, &graph));

  EXPECT_EQ(graph.node_count, 2u);
  EXPECT_FALSE(loom_call_graph_is_recursive(&graph, a.symbol_id));
  EXPECT_FALSE(loom_call_graph_is_recursive(&graph, b.symbol_id));
}

TEST_F(CallGraphTest, LowInternalCallEdge) {
  auto a = create_func("low_a");
  auto b = create_func("low_b");
  add_call(a, b.symbol_id, CallKind::kLowInternal);
  finalize();

  loom_call_graph_t graph;
  IREE_ASSERT_OK(loom_call_graph_build(module_, &graph_arena_, &graph));

  const loom_call_graph_node_t* node =
      loom_call_graph_node(&graph, a.symbol_id);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->callee_count, 1u);
  EXPECT_EQ(node->callees[0], b.symbol_id);
}

TEST_F(CallGraphTest, LowInvokeCallEdge) {
  auto a = create_func("invoke_a");
  auto b = create_func("invoke_b");
  add_call(a, b.symbol_id, CallKind::kLowInvoke);
  finalize();

  loom_call_graph_t graph;
  IREE_ASSERT_OK(loom_call_graph_build(module_, &graph_arena_, &graph));

  const loom_call_graph_node_t* node =
      loom_call_graph_node(&graph, a.symbol_id);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->callee_count, 1u);
  EXPECT_EQ(node->callees[0], b.symbol_id);
}

}  // namespace
}  // namespace loom
