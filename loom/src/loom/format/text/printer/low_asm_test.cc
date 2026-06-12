// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/alt_descriptors.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

static const loom_low_descriptor_set_provider_t
    kLowAsmPrinterTestDescriptorSetProviders[] = {
        loom_test_low_core_descriptor_set,
        loom_test_low_alt_descriptor_set,
};

class LowAsmPrinterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    {
      iree_host_size_t count = 0;
      const loom_op_vtable_t* const* vtables =
          loom_test_dialect_vtables(&count);
      IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                   vtables, (uint16_t)count));
    }
    {
      iree_host_size_t count = 0;
      const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
      IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                   vtables, (uint16_t)count));
    }
    low_descriptor_registry_ = {};
    low_descriptor_registry_.descriptor_set_providers =
        kLowAsmPrinterTestDescriptorSetProviders;
    low_descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kLowAsmPrinterTestDescriptorSetProviders);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseOk(const char* source) {
    loom_text_low_asm_environment_t environment = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_, &environment);
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/100,
        /*.low_asm_environment=*/environment,
    };
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  std::string PrintModule(loom_module_t* module,
                          iree_string_view_t descriptor_set_key) {
    loom_text_low_asm_environment_t environment = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_, &environment);
    loom_text_print_options_t options = {
        /*.flags=*/LOOM_TEXT_PRINT_DEFAULT,
        /*.low_asm_environment=*/environment,
        /*.low_asm_descriptor_set_key=*/descriptor_set_key,
    };
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status = loom_text_print_module_to_builder_with_options(
        module, &builder, &options);
    std::string result;
    if (iree_status_is_ok(status)) {
      result = std::string(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    }
    IREE_EXPECT_OK(status);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  iree_status_t PrintModuleStatus(loom_module_t* module,
                                  iree_string_view_t descriptor_set_key,
                                  bool configure_environment) {
    loom_text_low_asm_environment_t environment = {};
    if (configure_environment) {
      loom_low_descriptor_text_asm_environment_initialize(
          &low_descriptor_registry_, &environment);
    }
    loom_text_print_options_t options = {
        /*.flags=*/LOOM_TEXT_PRINT_DEFAULT,
        /*.low_asm_environment=*/environment,
        /*.low_asm_descriptor_set_key=*/descriptor_set_key,
    };
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status = loom_text_print_module_to_builder_with_options(
        module, &builder, &options);
    iree_string_builder_deinitialize(&builder);
    return status;
  }

  // Block pool backing parser and module arenas in each test.
  iree_arena_block_pool_t block_pool_;
  // Dialect registry used by parser calls.
  loom_context_t context_;
  // Low descriptor registry exposed to parser and printer low asm hooks.
  loom_low_descriptor_registry_t low_descriptor_registry_;
};

TEST_F(LowAsmPrinterTest, PrintsDescriptorBackedPacketRegion) {
  const char* source =
      "test.low_asm_region asm<test.low.core> {\n"
      "  %c0 = test.const.i32 7\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "  %spv = OpIAdd %sum, %c0\n"
      "  %call = test.call.i32 %spv {callee = 4}\n"
      "  return %call\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, IREE_SV("test.low.core")), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsExplicitAmbiguousResultType) {
  const char* source =
      "test.low_asm_region asm<test.low.core> {\n"
      "  %c0 = test.const.i32 7\n"
      "  %amb = test.ambiguous : reg<test.i64>\n"
      "  %tied = test.tied.any %c0\n"
      "  return %amb, %tied\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, IREE_SV("test.low.core")), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsStructuralIntrinsics) {
  const char* source =
      "test.low_asm_region asm<test.low.core> {\n"
      "  %state = resource<vm_state> {index = 0, source_type = i64} : "
      "reg<test.i64>\n"
      "  %arg0 = live_in<test.arg0> : reg<test.i32>\n"
      "  %pair = concat(%arg0, %arg0) : (reg<test.i32>, reg<test.i32>) -> "
      "reg<test.i32 x2>\n"
      "  %lane = slice %pair[1] : reg<test.i32 x2> -> reg<test.i32>\n"
      "  %copied = copy %lane : reg<test.i32> -> reg<test.i32>\n"
      "  %storage = storage {byte_alignment = 4, byte_length = 16} : "
      "low.storage<workgroup>\n"
      "  %window = storage_view %storage {offset = 4, byte_length = 8} : "
      "low.storage<workgroup> -> low.storage<workgroup>\n"
      "  %addr = storage_address %window : "
      "low.storage<workgroup> -> reg<test.i32>\n"
      "  return %copied\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, IREE_SV("test.low.core")), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, RejectsMissingPrintEnvironment) {
  loom_module_t* module = ParseOk(
      "test.low_asm_region asm<test.low.core> {\n"
      "  return\n"
      "}\n");
  ASSERT_NE(module, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        PrintModuleStatus(module, IREE_SV("test.low.core"),
                                          /*configure_environment=*/false));
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, RejectsDescriptorWithoutAsmFormInSelectedSet) {
  loom_module_t* module = ParseOk(
      "test.low_asm_region asm<test.low.core> {\n"
      "  %c0 = test.const.i32 7\n"
      "  return %c0\n"
      "}\n");
  ASSERT_NE(module, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        PrintModuleStatus(module, IREE_SV("test.low.alt"),
                                          /*configure_environment=*/true));
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
