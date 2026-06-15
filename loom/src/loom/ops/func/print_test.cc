// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"

namespace loom {
namespace {

class FuncPrinterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t func_count = 0;
    const loom_op_vtable_t* const* func_vtables =
        loom_func_dialect_vtables(&func_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, func_vtables, (uint16_t)func_count));

    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  std::string PrintOp(loom_op_t* op) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status = loom_text_print_operation_to_builder(
        module_, op, &builder, LOOM_TEXT_PRINT_DEFAULT);
    std::string result;
    if (iree_status_is_ok(status)) {
      result = std::string(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    }
    IREE_EXPECT_OK(status);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  loom_value_id_t DefineValue(loom_type_t type) {
    loom_value_id_t id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &id));
    return id;
  }

  loom_string_id_t Intern(const char* string) {
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(string), &id));
    return id;
  }

  loom_symbol_ref_t MakeSymbol(const char* name) {
    loom_string_id_t name_id = Intern(name);
    uint16_t symbol_id = (uint16_t)module_->symbols.count;
    EXPECT_LT(symbol_id, (uint16_t)module_->symbols.capacity);
    loom_symbol_t* symbol = &module_->symbols.entries[symbol_id];
    memset(symbol, 0, sizeof(*symbol));
    symbol->name_id = name_id;
    module_->symbols.count++;
    return {0, symbol_id};
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(FuncPrinterTest, Call) {
  loom_symbol_ref_t callee = MakeSymbol("callee");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = DefineValue(f32);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_func_call_build(&builder_, 0, 0, 0, 0, callee, &input, 1,
                                      result_types, 1, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(PrintOp(op), "%1 = func.call @callee(%0) : (f32) -> (f32)\n");
}

TEST_F(FuncPrinterTest, Apply) {
  loom_string_id_t contract = Intern("my.template");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = DefineValue(f32);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_func_apply_build(&builder_, 0, contract, &input, 1, 0, 0,
                                       result_types, 1, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(PrintOp(op), "%1 = func.apply<my.template>(%0) : (f32) -> (f32)\n");
}

TEST_F(FuncPrinterTest, Return) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = DefineValue(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_func_return_build(&builder_, &input, 1, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(PrintOp(op), "func.return %0 : f32\n");
}

TEST_F(FuncPrinterTest, EmptyReturn) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_func_return_build(&builder_, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(PrintOp(op), "func.return\n");
}

TEST_F(FuncPrinterTest, Definition) {
  loom_symbol_ref_t callee = MakeSymbol("entry");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY, 1, 0, 0, 0, 0,
      loom_symbol_ref_null(), 0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), callee, arg_types,
      1, result_types, 1, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(PrintOp(op),
            "func.def public @entry(%0: f32) -> (f32) {\n"
            "}\n");
}

TEST_F(FuncPrinterTest, TemplateOpRef) {
  loom_symbol_ref_t callee = MakeSymbol("vnni_q8");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_string_id_t implements_id = Intern("tile.contract");

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_func_template_build(
      &builder_, 0, implements_id, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
      /*priority=*/0, callee, arg_types, 1, result_types, 1, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &op));

  EXPECT_NE(PrintOp(op).find("func.template<tile.contract>"),
            std::string::npos);
}

TEST_F(FuncPrinterTest, TemplateWithPriority) {
  loom_symbol_ref_t callee = MakeSymbol("fast_matmul");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_string_id_t implements_id = Intern("tile.contract");

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_func_template_build(
      &builder_, LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_PRIORITY, implements_id, 0,
      0, 0, 0, 0, loom_symbol_ref_null(), /*priority=*/10, callee, arg_types, 1,
      result_types, 1, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));

  EXPECT_NE(PrintOp(op).find("priority(10)"), std::string::npos);
}

TEST_F(FuncPrinterTest, TemplateWithTarget) {
  loom_symbol_ref_t target = MakeSymbol("gfx1100");
  loom_symbol_ref_t callee = MakeSymbol("gfx11_matmul");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_string_id_t implements_id = Intern("tile.contract");

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_func_template_build(
      &builder_, LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_TARGET, implements_id, 0, 0,
      0, 0, 0, target, /*priority=*/0, callee, arg_types, 1, result_types, 1,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));

  EXPECT_NE(PrintOp(op).find("target(@gfx1100)"), std::string::npos);
}

}  // namespace
}  // namespace loom
