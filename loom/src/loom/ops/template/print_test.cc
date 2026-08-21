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
#include "loom/ops/template/ops.h"

namespace loom {
namespace {

class TemplatePrinterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_template_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEMPLATE, vtables, (uint16_t)vtable_count));

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

  loom_symbol_ref_t MakeSymbol(const char* name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  loom_op_t* BuildDefinition(loom_template_def_build_flags_t build_flags,
                             loom_symbol_ref_t family,
                             loom_symbol_ref_t implementation,
                             loom_symbol_ref_t target, int64_t priority) {
    const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_template_def_build(
        &builder_, build_flags, family, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*purity=*/0, /*temperature=*/0, target,
        loom_parameterized_attr_array_empty(), priority, implementation, &f32,
        1, &f32, 1, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(TemplatePrinterTest, Apply) {
  const loom_symbol_ref_t family = MakeSymbol("my.template");
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t input = DefineValue(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_template_apply_build(
      &builder_, 0, family, &input, 1, /*purity=*/0, /*temperature=*/0, &f32, 1,
      NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(PrintOp(op),
            "%1 = template.apply<@my.template>(%0) : (f32) -> (f32)\n");
}

TEST_F(TemplatePrinterTest, Return) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t input = DefineValue(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_template_return_build(&builder_, &input, 1,
                                            LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(PrintOp(op), "template.return %0 : f32\n");
}

TEST_F(TemplatePrinterTest, DefinitionNamesFamily) {
  loom_op_t* op =
      BuildDefinition(/*build_flags=*/0, MakeSymbol("tile.contract"),
                      MakeSymbol("vnni_q8"), loom_symbol_ref_null(), 0);
  EXPECT_NE(PrintOp(op).find("template.def<@tile.contract>"),
            std::string::npos);
}

TEST_F(TemplatePrinterTest, DefinitionWithPriority) {
  loom_op_t* op = BuildDefinition(
      LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_PRIORITY, MakeSymbol("tile.contract"),
      MakeSymbol("fast_matmul"), loom_symbol_ref_null(), 10);
  EXPECT_NE(PrintOp(op).find("priority(10)"), std::string::npos);
}

TEST_F(TemplatePrinterTest, DefinitionWithTarget) {
  loom_op_t* op = BuildDefinition(
      LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_TARGET, MakeSymbol("tile.contract"),
      MakeSymbol("gfx11_matmul"), MakeSymbol("gfx1100"), 0);
  EXPECT_NE(PrintOp(op).find("target(@gfx1100)"), std::string::npos);
}

}  // namespace
}  // namespace loom
