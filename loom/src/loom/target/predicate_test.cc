// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/predicate.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/verify.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TargetPredicateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_PASS, loom_pass_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_target_pass_predicate_provider_storage_initialize(&block_pool_,
                                                           &predicate_storage_);
    predicate_provider_ =
        loom_target_pass_predicate_provider(&predicate_storage_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("target_predicate_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    if (module == nullptr) {
      IREE_CHECK_OK(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                     "target predicate test parse failed"));
    }
    return ModulePtr(module);
  }

  loom_symbol_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return &module->symbols.entries[symbol_id];
  }

  loom_op_t* FindPipeline(loom_module_t* module, iree_string_view_t name) {
    loom_symbol_t* symbol = FindSymbol(module, name);
    IREE_ASSERT(symbol->defining_op != nullptr);
    IREE_ASSERT(loom_pass_pipeline_isa(symbol->defining_op));
    return symbol->defining_op;
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    loom_symbol_t* symbol = FindSymbol(module, name);
    IREE_ASSERT(symbol->defining_op != nullptr);
    return loom_func_like_cast(module, symbol->defining_op);
  }

  loom_op_t* FirstWhere(loom_op_t* pipeline_op) {
    loom_block_t* pipeline_body =
        loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
    IREE_ASSERT(pipeline_body != nullptr);
    loom_op_t* for_op = pipeline_body->first_op;
    IREE_ASSERT(loom_pass_for_isa(for_op));
    loom_block_t* for_body =
        loom_region_entry_block(loom_pass_for_body(for_op));
    IREE_ASSERT(for_body != nullptr);
    IREE_ASSERT(loom_pass_where_isa(for_body->first_op));
    return for_body->first_op;
  }

  iree_status_t VerifyPipeline(loom_module_t* module,
                               iree_string_view_t pipeline_name) {
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool_, &scratch_arena);
    const loom_pass_verify_options_t options = {
        /*.registry=*/{},
        /*.environment=*/{},
        /*.predicate_provider=*/predicate_provider_,
    };
    iree_status_t status = loom_pass_verify_pipeline_op(
        module, FindPipeline(module, pipeline_name), &options, &scratch_arena);
    iree_arena_deinitialize(&scratch_arena);
    return status;
  }

  bool Evaluate(loom_module_t* module, loom_op_t* where_op,
                iree_string_view_t function_name) {
    loom_symbol_t* symbol = FindSymbol(module, function_name);
    loom_func_like_t function = FindFunction(module, function_name);
    bool match = false;
    const loom_pass_predicate_evaluate_context_t context = {
        /*.pipeline_module=*/module,
        /*.where_op=*/where_op,
        /*.anchor_kind=*/LOOM_PASS_FUNCTION,
        /*.predicate=*/IREE_SV("target"),
        /*.target_module=*/module,
        /*.symbol=*/symbol,
        /*.function=*/function,
    };
    IREE_CHECK_OK(predicate_provider_.evaluate(predicate_provider_.user_data,
                                               &context, &match));
    return match;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_pass_predicate_provider_storage_t predicate_storage_;
  loom_pass_predicate_provider_t predicate_provider_;
};

TEST_F(TargetPredicateTest, VerifiesTargetPredicateAttrs) {
  ModulePtr module = ParseModule(R"(
pass.pipeline<module> @pipeline pipeline {
  for func {
    where target(target = "test_target", target_op = "test.target", codegen = "low_native", abi = "object_function") {
    }
  }
}
)");

  IREE_ASSERT_OK(VerifyPipeline(module.get(), IREE_SV("pipeline")));
}

TEST_F(TargetPredicateTest, MatchesResolvedTargetContract) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target
test.target<quirky> @other_target

pass.pipeline<module> @pipeline pipeline {
  for func {
    where target(target = "@test_target", target_op = "test.target", bundle = "test_target", snapshot = "test_target", codegen = "low_native", artifact_format = "elf", abi = "object_function", config = "test_target", contract = "test.low.core") {
    }
  }
}

func.def target(@test_target) abi(object_function) @matched() {
  func.return
}

func.def target(@other_target) abi(object_function) @rejected() {
  func.return
}
)");

  loom_op_t* where_op =
      FirstWhere(FindPipeline(module.get(), IREE_SV("pipeline")));
  EXPECT_TRUE(Evaluate(module.get(), where_op, IREE_SV("matched")));
  EXPECT_FALSE(Evaluate(module.get(), where_op, IREE_SV("rejected")));
}

}  // namespace
}  // namespace loom
