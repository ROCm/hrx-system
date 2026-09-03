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

  std::string PrintModule(
      loom_module_t* module,
      loom_text_print_flags_t flags = LOOM_TEXT_PRINT_DEFAULT |
                                      LOOM_TEXT_PRINT_PREFER_LOW_ASM) {
    loom_text_low_asm_environment_t environment = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_, &environment);
    loom_text_print_options_t options = {
        /*.flags=*/flags,
        /*.low_asm_environment=*/environment,
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

  iree_status_t PrintModuleStatus(
      loom_module_t* module, bool configure_environment,
      loom_text_print_flags_t flags = LOOM_TEXT_PRINT_DEFAULT) {
    loom_text_low_asm_environment_t environment = {};
    if (configure_environment) {
      loom_low_descriptor_text_asm_environment_initialize(
          &low_descriptor_registry_, &environment);
    }
    loom_text_print_options_t options = {
        /*.flags=*/flags,
        /*.low_asm_environment=*/environment,
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
      "low.func.def target<test.low.core> @packet() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "  %spv = OpIAdd %sum, %c0\n"
      "  %call = test.call.i32 %spv {callee = 4}\n"
      "  return %call\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, CanonicalizesVerticalSourceGrouping) {
  const char* source =
      "low.func.def target<test.low.core> @grouped() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "\n"
      "  // arithmetic\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "\n"
      "  return %sum\n"
      "}\n";
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @grouped() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "\n"
      "\n"
      "  // arithmetic\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "\n"
      "\n"
      "  return %sum\n"
      "}\n");
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PreservesAuthoredAsmChoice) {
  const char* source =
      "low.func.def target<test.low.core> @assembly("
      "%value: reg<test.i32>) -> (reg<test.i32>) asm {\n"
      "  return %value\n"
      "}\n"
      "\n"
      "low.func.def target<test.low.core> @generic("
      "%value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  low.return %value : reg<test.i32>\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, LOOM_TEXT_PRINT_DEFAULT |
                                    LOOM_TEXT_PRINT_PRESERVE_LOW_ASM |
                                    LOOM_TEXT_PRINT_PREFER_LOW_ASM |
                                    LOOM_TEXT_PRINT_REQUIRE_LOW_ASM),
            source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PreservesNestedExplicitAsmChoice) {
  const char* source =
      "low.func.def target<test.low.core> @select("
      "%condition: reg<test.i32>, %then_value: reg<test.i32>, "
      "%else_value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %result = low.scf.if %condition -> (reg<test.i32>) asm {\n"
      "    low.scf.yield %then_value : reg<test.i32>\n"
      "  } else {\n"
      "    low.scf.yield %else_value : reg<test.i32>\n"
      "  }\n"
      "  low.return %result : reg<test.i32>\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, LOOM_TEXT_PRINT_DEFAULT |
                                    LOOM_TEXT_PRINT_PRESERVE_LOW_ASM),
            source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsCanonicalHintAmongTargetPackets) {
  const char* source =
      "low.func.def target<test.low.core> @hint() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  low.schedule.fence\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "  return %sum\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsExplicitAmbiguousResultType) {
  const char* source =
      "low.func.def target<test.low.core> @ambiguous() -> "
      "(reg<test.i64>, reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %amb = test.ambiguous : reg<test.i64>\n"
      "  %tied = test.tied.any %c0\n"
      "  return %amb, %tied\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PreservesSemanticRegisterResultTypes) {
  const char* source =
      "low.func.def target<test.low.core> @typed(%arg: "
      "reg<test.i32 : i32>) -> (reg<test.i32 : i32>, "
      "reg<test.i32 : i32>) asm {\n"
      "  %value = test.const.i32 7 : reg<test.i32 : i32>\n"
      "  %tied = test.tied.any %arg\n"
      "  return %value, %tied\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, ElidesOnlyExactSemanticResultValueType) {
  const char* source =
      "low.func.def target<test.low.core> @typed_packet("
      "%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> "
      "(reg<test.i32 : i32>, reg<test.i32>, reg<test.i32 : f32>) asm {\n"
      "  %typed = OpIAdd %lhs, %rhs\n"
      "  %carrier = OpIAdd %lhs, %rhs : reg<test.i32>\n"
      "  %alternate = OpIAdd %lhs, %rhs : reg<test.i32 : f32>\n"
      "  return %typed, %carrier, %alternate\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsZeroImmediateConstAndOperandlessOp) {
  const char* source =
      "low.func.def target<test.low.core> @zero_immediate() -> "
      "(reg<test.i32>, reg<test.i64>) asm {\n"
      "  %zero = test.const.zero.i32\n"
      "  %value = test.ambiguous : reg<test.i64>\n"
      "  return %zero, %value\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsStructuralIntrinsics) {
  const char* source =
      "low.func.def target<test.low.core> @structural() -> "
      "(reg<test.i32>) asm {\n"
      "  %binding = resource<hal_binding> {index = 0, source_type = "
      "hal.buffer} : reg<test.ptr>\n"
      "  %arg0 = live_in<test.arg0> {} : reg<test.i32>\n"
      "  %pair = concat(%arg0, %arg0) : (reg<test.i32>, reg<test.i32>) -> "
      "reg<test.i32 x2>\n"
      "  %lane = slice %pair[1] : reg<test.i32 x2> -> reg<test.i32>\n"
      "  %copied = copy %lane : reg<test.i32> -> reg<test.i32>\n"
      "  %moved = move %copied : reg<test.i32> -> reg<test.i32>\n"
      "  %storage = storage {byte_alignment = 4, byte_length = 16} : "
      "low.storage<workgroup>\n"
      "  %window = storage_view %storage {offset = 4, byte_length = 8} : "
      "low.storage<workgroup> -> low.storage<workgroup>\n"
      "  %addr = storage_address %window : "
      "low.storage<workgroup> -> reg<test.i32>\n"
      "  return %moved\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsCanonicalStructuralCall) {
  const char* source =
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.decl target<test.low.core>(@test_target) "
      "@callee(%arg: reg<test.i32>) -> "
      "(reg<test.i32>)\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) "
      "@caller(%arg: reg<test.i32>) -> "
      "(reg<test.i32>) asm {\n"
      "  %result = low.func.call @callee(%arg) : (reg<test.i32>) -> "
      "(reg<test.i32>)\n"
      "  return %result\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, NestedRegionsInheritRepresentationContract) {
  const char* source =
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) @select("
      "%condition: reg<test.i32>, %then_value: reg<test.i32>, "
      "%else_value: reg<test.i32>) -> (reg<test.i32>) "
      "asm {\n"
      "  %result = low.scf.if %condition -> (reg<test.i32>) {\n"
      "    low.scf.yield %then_value : reg<test.i32>\n"
      "  } else {\n"
      "    low.scf.yield %else_value : reg<test.i32>\n"
      "  }\n"
      "  return %result\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, LOOM_TEXT_PRINT_DEFAULT |
                                    LOOM_TEXT_PRINT_REQUIRE_LOW_ASM),
            source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, FunctionRepresentationContractSelectsDescriptorSet) {
  const char* source =
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.alt>(@test_target) @constant() -> "
      "(reg<test.i32>) asm {\n"
      "  %value = test.alt.const.i32 11\n"
      "  return %value\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest,
       TargetlessFunctionRepresentationContractSelectsDescriptorSet) {
  const char* source =
      "low.func.def target<test.low.alt> @constant() -> "
      "(reg<test.i32>) asm {\n"
      "  %value = test.alt.const.i32 11\n"
      "  return %value\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, PrintsMixedFunctionRepresentationContracts) {
  const char* source =
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) "
      "@core() -> (reg<test.i32>) asm {\n"
      "  %value = test.const.i32 7\n"
      "  return %value\n"
      "}\n"
      "\n"
      "low.func.def target<test.low.alt>(@test_target) "
      "@alt() -> (reg<test.i32>) asm {\n"
      "  %value = test.alt.const.i32 11\n"
      "  return %value\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, LOOM_TEXT_PRINT_DEFAULT |
                                    LOOM_TEXT_PRINT_PREFER_LOW_ASM),
            source);
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, RejectsMissingPrintEnvironment) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @empty() asm {\n"
      "  return\n"
      "}\n");
  ASSERT_NE(module, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      PrintModuleStatus(
          module, /*configure_environment=*/false,
          LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM));
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, RequiredOptionalLowAsmRejectsCanonicalFallback) {
  const char* source =
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) "
      "@spill(%value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %slot = low.storage.reserve "
      "{byte_alignment = 4, byte_length = 4} : low.storage<private>\n"
      "  low.spill %value, %slot : reg<test.i32>, low.storage<private>\n"
      "  low.return %value : reg<test.i32>\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      PrintModuleStatus(
          module, /*configure_environment=*/true,
          LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM));
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest,
       PreferredLowAsmFallsBackWhenPacketDescriptionRejectsOperation) {
  const char* source =
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) "
      "@invalid_tie(%src: reg<test.i32>) -> (reg<test.i64>) {\n"
      "  %changed = low.op<test.pass.any>(%src) : "
      "(reg<test.i32>) -> %src as reg<test.i64>\n"
      "  low.return %changed : reg<test.i64>\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module), source);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrintModuleStatus(
          module, /*configure_environment=*/true,
          LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM));
  loom_module_free(module);
}

TEST_F(LowAsmPrinterTest, RequiredLowAsmUsesFunctionRepresentationContract) {
  const char* source =
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) "
      "@add(%lhs: reg<test.i32>, "
      "%rhs: reg<test.i32>) -> (reg<test.i32>) asm {\n"
      "  %sum = low.op<test.add.i32>(%lhs, %rhs) "
      "memory_access([0, 3, 7, -1, 35, 64, 0, 16, 0, 0, 0, 0, 0]) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  return %sum\n"
      "}\n";
  loom_module_t* module = ParseOk(source);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(PrintModule(module, LOOM_TEXT_PRINT_DEFAULT |
                                    LOOM_TEXT_PRINT_REQUIRE_LOW_ASM),
            source);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
