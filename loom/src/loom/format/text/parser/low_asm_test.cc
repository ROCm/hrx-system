// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/registers.h"
#include "loom/target/test/alt_descriptors.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectU32Param;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

static const loom_low_descriptor_set_provider_t
    kLowAsmParserTestDescriptorSetProviders[] = {
        loom_test_low_core_descriptor_set,
        loom_test_low_alt_descriptor_set,
};

class LowAsmParserTest : public ::testing::Test {
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
        kLowAsmParserTestDescriptorSetProviders;
    low_descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kLowAsmParserTestDescriptorSetProviders);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Parse(const char* source, bool enable_low_asm,
                      loom_module_t** out_module) {
    capture_.Reset();
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = capture_.sink();
    options.max_errors = 100;
    if (enable_low_asm) {
      loom_low_descriptor_text_asm_environment_initialize(
          &low_descriptor_registry_, &options.low_asm_environment);
    }
    return loom_text_parse(iree_make_cstring_view(source),
                           iree_make_cstring_view("test.loom"), &context_,
                           &block_pool_, &options, out_module);
  }

  loom_module_t* ParseOk(const char* source) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(Parse(source, /*enable_low_asm=*/true, &module));
    if (!capture_.diagnostics.empty()) {
      std::string msg = "Expected no diagnostics but got " +
                        std::to_string(capture_.diagnostics.size()) + ":\n";
      for (size_t i = 0; i < capture_.diagnostics.size(); ++i) {
        const auto& diagnostic = capture_.diagnostics[i];
        msg += "  [" + std::to_string(i) + "] " +
               (diagnostic.error ? diagnostic.error->summary : "(null)") +
               " line=" + std::to_string(diagnostic.origin_line) +
               " col=" + std::to_string(diagnostic.origin_column);
        for (size_t j = 0; j < diagnostic.params.size(); ++j) {
          if (diagnostic.params[j].kind == LOOM_PARAM_STRING) {
            msg += " p" + std::to_string(j) + "='" +
                   std::string(diagnostic.params[j].string.data,
                               diagnostic.params[j].string.size) +
                   "'";
          }
        }
        msg += "\n";
      }
      ADD_FAILURE() << msg;
    }
    EXPECT_NE(module, nullptr);
    return module;
  }

  const std::vector<CapturedDiagnostic>& ParseExpectErrors(const char* source) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(Parse(source, /*enable_low_asm=*/true, &module));
    EXPECT_EQ(module, nullptr);
    EXPECT_GT(capture_.diagnostics.size(), 0u);
    for (const CapturedDiagnostic& diagnostic : capture_.diagnostics) {
      EXPECT_EQ(diagnostic.origin.provenance,
                LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
      EXPECT_EQ(diagnostic.source_location.provenance,
                LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
    }
    return capture_.diagnostics;
  }

  // Block pool backing parser and module arenas in each test.
  iree_arena_block_pool_t block_pool_;
  // Dialect registry used by parser calls.
  loom_context_t context_;
  // Low descriptor registry exposed to the low asm parser.
  loom_low_descriptor_registry_t low_descriptor_registry_;
  // Diagnostic capture sink populated by parse helpers.
  DiagnosticCapture capture_;
};

static loom_block_t* GetEntryBlock(loom_region_t* region) {
  if (!region || region->block_count == 0) {
    return nullptr;
  }
  return loom_region_entry_block(region);
}

static std::string StringFromId(const loom_module_t* module,
                                loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return "";
  }
  iree_string_view_t value = module->strings.entries[string_id];
  return std::string(value.data, value.size);
}

static const loom_named_attr_t* FindNamedAttr(const loom_module_t* module,
                                              loom_named_attr_slice_t attrs,
                                              iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id >= module->strings.count) {
      continue;
    }
    if (iree_string_view_equal(
            module->strings.entries[attrs.entries[i].name_id], name)) {
      return &attrs.entries[i];
    }
  }
  return nullptr;
}

static void ExpectTestLowCoreRegisterType(loom_type_t type,
                                          uint16_t register_class_id,
                                          uint32_t unit_count) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  ASSERT_TRUE(loom_type_is_register(type));
  EXPECT_EQ(loom_low_register_type_descriptor_set_stable_id(type),
            descriptor_set->stable_id);
  EXPECT_EQ(loom_low_register_type_class_id(type), register_class_id);
  EXPECT_EQ(loom_low_register_type_unit_count(type), unit_count);
}

static void ExpectDescriptorOrdinal(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t actual_descriptor_ordinal, iree_string_view_t descriptor_key) {
  const uint32_t expected_descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, descriptor_key);
  ASSERT_NE(expected_descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  EXPECT_EQ(actual_descriptor_ordinal, expected_descriptor_ordinal);
}

static void ExpectFunctionPacketIdentity(
    const loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t const_key, iree_string_view_t op_key) {
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  const loom_func_like_t function = loom_func_like_cast(module, function_op);
  ASSERT_TRUE(loom_func_like_isa(function));
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  ASSERT_LT(contract_id, module->strings.count);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[contract_id],
      loom_low_descriptor_set_string(descriptor_set,
                                     descriptor_set->key_string_offset)));

  loom_region_t* body_region = loom_low_func_def_body(function_op);
  ASSERT_NE(body_region, nullptr);
  loom_block_t* body = loom_region_entry_block(body_region);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->op_count, 3u);
  const loom_op_t* const_op = loom_block_op(body, 0);
  const loom_op_t* packet_op = loom_block_op(body, 1);
  ASSERT_TRUE(loom_low_const_isa(const_op));
  ASSERT_TRUE(loom_low_op_isa(packet_op));

  ExpectDescriptorOrdinal(descriptor_set, loom_low_const_descriptor(const_op),
                          const_key);
  ExpectDescriptorOrdinal(descriptor_set, loom_low_op_descriptor(packet_op),
                          op_key);
  EXPECT_EQ(const_op->traits, LOOM_TRAIT_PURE);
  EXPECT_EQ(packet_op->traits, LOOM_TRAIT_PURE);

  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_const_result(const_op));
  ASSERT_TRUE(loom_type_is_register(result_type));
  EXPECT_EQ(loom_low_register_type_descriptor_set_stable_id(result_type),
            descriptor_set->stable_id);
}

TEST_F(LowAsmParserTest, ResolvesPacketIdentityPerFunctionContract) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @core() -> "
      "(reg<test.i32>) asm {\n"
      "  %value = test.const.i32 7\n"
      "  %sum = test.add.i32 %value, %value\n"
      "  return %sum\n"
      "}\n"
      "\n"
      "low.func.def target<test.low.alt> @alt() -> "
      "(reg<test.i32>) asm {\n"
      "  %value = test.alt.const.i32 11\n"
      "  %negated = test.alt.neg.i32 %value\n"
      "  return %negated\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 2u);
  ExpectFunctionPacketIdentity(module, loom_block_op(module_block, 0),
                               loom_test_low_core_descriptor_set(),
                               IREE_SV("test.const.i32"),
                               IREE_SV("test.add.i32"));
  ExpectFunctionPacketIdentity(module, loom_block_op(module_block, 1),
                               loom_test_low_alt_descriptor_set(),
                               IREE_SV("test.alt.const.i32"),
                               IREE_SV("test.alt.neg.i32"));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, RecordsExplicitAsmSourceSyntax) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @assembly("
      "%value: reg<test.i32>) -> (reg<test.i32>) asm {\n"
      "  return %value\n"
      "}\n"
      "\n"
      "low.func.def target<test.low.core> @generic("
      "%value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  low.return %value : reg<test.i32>\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 2u);
  loom_region_t* assembly_body =
      loom_low_func_def_body(loom_block_op(module_block, 0));
  loom_region_t* generic_body =
      loom_low_func_def_body(loom_block_op(module_block, 1));
  ASSERT_NE(assembly_body, nullptr);
  ASSERT_NE(generic_body, nullptr);
  EXPECT_TRUE(iree_any_bit_set(assembly_body->source_flags,
                               LOOM_REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM));
  EXPECT_FALSE(iree_any_bit_set(generic_body->source_flags,
                                LOOM_REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, ParsesStructuralRegisterValueTypes) {
  loom_module_t* module = ParseOk(
      "low.func.decl target<test.low.core> @typed("
      "%arg: reg<test.i32 : i32>) -> "
      "(reg<test.i32 x4 : vector<4xi32>>, "
      "reg<test.ptr : kernel.async.token>)\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_decl_isa(function_op));
  loom_func_like_t function = loom_func_like_cast(module, function_op);

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  ASSERT_EQ(argument_count, 1u);
  loom_type_t argument_type = loom_module_value_type(module, argument_ids[0]);
  ExpectTestLowCoreRegisterType(argument_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
  const loom_type_t* argument_value_type =
      loom_type_register_value_type(argument_type);
  ASSERT_NE(argument_value_type, nullptr);
  EXPECT_TRUE(loom_type_equal(*argument_value_type,
                              loom_type_scalar(LOOM_SCALAR_TYPE_I32)));

  ASSERT_EQ(function_op->result_count, 2u);
  loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(function_op)[0]);
  ExpectTestLowCoreRegisterType(result_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4);
  const loom_type_t* result_value_type =
      loom_type_register_value_type(result_type);
  ASSERT_NE(result_value_type, nullptr);
  EXPECT_TRUE(loom_type_equal(
      *result_value_type,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0)));

  loom_type_t dialect_result_type =
      loom_module_value_type(module, loom_op_const_results(function_op)[1]);
  ExpectTestLowCoreRegisterType(dialect_result_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR, 1);
  const loom_type_t* dialect_value_type =
      loom_type_register_value_type(dialect_result_type);
  ASSERT_NE(dialect_value_type, nullptr);
  ASSERT_TRUE(loom_type_is_dialect(*dialect_value_type));
  EXPECT_EQ(loom_type_dialect_param_count(*dialect_value_type), 0u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[loom_type_dialect_name_id(*dialect_value_type)],
      IREE_SV("kernel.async.token")));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, InfersExactSemanticResultValueType) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @typed_packet("
      "%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> "
      "(reg<test.i32 : i32>) asm {\n"
      "  %result = OpIAdd %lhs, %rhs\n"
      "  return %result\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_op_t* function_op = loom_block_op(loom_module_block(module), 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  loom_block_t* entry = GetEntryBlock(loom_low_func_def_body(function_op));
  ASSERT_NE(entry, nullptr);
  loom_op_t* packet_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_low_op_isa(packet_op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_low_op_results(packet_op).values[0]);
  ExpectTestLowCoreRegisterType(result_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
  const loom_type_t* value_type = loom_type_register_value_type(result_type);
  ASSERT_NE(value_type, nullptr);
  EXPECT_TRUE(
      loom_type_equal(*value_type, loom_type_scalar(LOOM_SCALAR_TYPE_I32)));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, BuildsCanonicalLowOps) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @packet() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "  %spv = OpIAdd %sum, %c0\n"
      "  %call = test.call.i32 %spv {callee = 4}\n"
      "  return %call\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));

  loom_region_t* region = loom_low_func_def_body(function_op);
  ASSERT_NE(region, nullptr);
  loom_block_t* entry = GetEntryBlock(region);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 5u);

  loom_op_t* const_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_low_const_isa(const_op));
  ExpectDescriptorOrdinal(loom_test_low_core_descriptor_set(),
                          loom_low_const_descriptor(const_op),
                          IREE_SV("test.const.i32"));
  loom_named_attr_slice_t const_attrs = loom_low_const_attrs(const_op);
  const loom_named_attr_t* i32_value =
      FindNamedAttr(module, const_attrs, IREE_SV("i32_value"));
  ASSERT_NE(i32_value, nullptr);
  ASSERT_EQ(i32_value->value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(i32_value->value.i64, 7);

  loom_type_t const_type =
      loom_module_value_type(module, loom_low_const_result(const_op));
  ExpectTestLowCoreRegisterType(const_type, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32,
                                1);

  loom_op_t* add_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_low_op_isa(add_op));
  ExpectDescriptorOrdinal(loom_test_low_core_descriptor_set(),
                          loom_low_op_descriptor(add_op),
                          IREE_SV("test.add.i32"));
  loom_value_slice_t add_operands = loom_low_op_operands(add_op);
  ASSERT_EQ(add_operands.count, 2u);
  EXPECT_EQ(add_operands.values[0], loom_low_const_result(const_op));
  EXPECT_EQ(add_operands.values[1], loom_low_const_result(const_op));

  loom_op_t* spv_op = loom_block_op(entry, 2);
  ASSERT_TRUE(loom_low_op_isa(spv_op));
  ExpectDescriptorOrdinal(loom_test_low_core_descriptor_set(),
                          loom_low_op_descriptor(spv_op),
                          IREE_SV("test.spv.op_iadd.i32"));
  loom_value_slice_t spv_operands = loom_low_op_operands(spv_op);
  ASSERT_EQ(spv_operands.count, 2u);
  EXPECT_EQ(spv_operands.values[0], loom_low_op_results(add_op).values[0]);
  EXPECT_EQ(spv_operands.values[1], loom_low_const_result(const_op));

  loom_op_t* call_op = loom_block_op(entry, 3);
  ASSERT_TRUE(loom_low_op_isa(call_op));
  ExpectDescriptorOrdinal(loom_test_low_core_descriptor_set(),
                          loom_low_op_descriptor(call_op),
                          IREE_SV("test.call.i32"));
  loom_named_attr_slice_t call_attrs = loom_low_op_attrs(call_op);
  const loom_named_attr_t* callee_ordinal =
      FindNamedAttr(module, call_attrs, IREE_SV("callee_ordinal"));
  ASSERT_NE(callee_ordinal, nullptr);
  ASSERT_EQ(callee_ordinal->value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(callee_ordinal->value.i64, 4);

  loom_op_t* return_op = loom_block_op(entry, 4);
  ASSERT_TRUE(loom_low_return_isa(return_op));
  loom_value_slice_t return_values = loom_low_return_values(return_op);
  ASSERT_EQ(return_values.count, 1u);
  EXPECT_EQ(return_values.values[0], loom_low_op_results(call_op).values[0]);

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, BuildsCanonicalHintAmongTargetPackets) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @hint() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  low.schedule.fence\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "  return %sum\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));

  loom_region_t* region = loom_low_func_def_body(function_op);
  ASSERT_NE(region, nullptr);
  loom_block_t* entry = GetEntryBlock(region);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 4u);
  EXPECT_TRUE(loom_low_const_isa(loom_block_op(entry, 0)));
  EXPECT_TRUE(loom_low_schedule_fence_isa(loom_block_op(entry, 1)));
  EXPECT_TRUE(loom_low_op_isa(loom_block_op(entry, 2)));
  EXPECT_TRUE(loom_low_return_isa(loom_block_op(entry, 3)));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, SelectsDescriptorSet) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.alt> @negate() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.alt.const.i32 5\n"
      "  %neg = test.alt.neg.i32 %c0\n"
      "  return %neg\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));

  loom_region_t* region = loom_low_func_def_body(function_op);
  ASSERT_NE(region, nullptr);
  loom_block_t* entry = GetEntryBlock(region);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 3u);

  loom_op_t* const_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_low_const_isa(const_op));
  ExpectDescriptorOrdinal(loom_test_low_alt_descriptor_set(),
                          loom_low_const_descriptor(const_op),
                          IREE_SV("test.alt.const.i32"));

  loom_op_t* neg_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_low_op_isa(neg_op));
  ExpectDescriptorOrdinal(loom_test_low_alt_descriptor_set(),
                          loom_low_op_descriptor(neg_op),
                          IREE_SV("test.alt.neg.i32"));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, FunctionRepresentationContractSelectsDescriptorSet) {
  loom_module_t* module = ParseOk(
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.alt>(@test_target) @constant() -> "
      "(reg<test.i32>) asm {\n"
      "  %value = test.alt.const.i32 11\n"
      "  return %value\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 2u);
  loom_op_t* function_op = loom_block_op(module_block, 1);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  loom_func_like_t function = loom_func_like_cast(module, function_op);
  ASSERT_TRUE(loom_func_like_isa(function));
  EXPECT_EQ(StringFromId(module, loom_func_like_repr_contract(function)),
            "test.low.alt");

  loom_block_t* entry = GetEntryBlock(loom_low_func_def_body(function_op));
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 2u);
  loom_op_t* const_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_low_const_isa(const_op));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_alt_descriptor_set();
  ExpectDescriptorOrdinal(descriptor_set, loom_low_const_descriptor(const_op),
                          IREE_SV("test.alt.const.i32"));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_const_result(const_op));
  ASSERT_TRUE(loom_type_is_register(result_type));
  EXPECT_EQ(loom_low_register_type_descriptor_set_stable_id(result_type),
            descriptor_set->stable_id);

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, RejectsMissingFunctionRepresentationContracts) {
  const char* sources[] = {
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target(@test_target) @empty() {\n"
      "  low.return\n"
      "}\n",
      "test.target<low_core> @test_target\n"
      "\n"
      "low.kernel.def target(@test_target) @empty() {\n"
      "  low.return\n"
      "}\n",
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.decl target(@test_target) @external()\n",
  };
  for (const char* source : sources) {
    SCOPED_TRACE(source);
    const auto& diagnostics = ParseExpectErrors(source);
    const CapturedDiagnostic* diagnostic = FindDiagnostic(
        capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 3));
    ASSERT_NE(diagnostic, nullptr);
    EXPECT_EQ(GetStringParam(*diagnostic, 0), "(");
    EXPECT_EQ(GetStringParam(*diagnostic, 1), "'<'");
    (void)diagnostics;
  }
}

TEST_F(LowAsmParserTest, RejectsRedundantFunctionAsmContract) {
  const auto& diagnostics = ParseExpectErrors(
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) @constant() -> "
      "(reg<test.i32>) asm<test.low.core> {\n"
      "  %value = test.const.i32 7\n"
      "  return %value\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 3));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "<");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "'{'");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, RejectsUnavailableFunctionRepresentationContract) {
  const auto& diagnostics = ParseExpectErrors(
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.missing>(@test_target) @empty() asm {\n"
      "  return\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0),
            "unknown Low representation contract");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, BuildsStructuralCopy) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @copy() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %copy = copy %c0 : reg<test.i32> -> reg<test.i32>\n"
      "  return %copy\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  loom_region_t* region = loom_low_func_def_body(function_op);
  loom_block_t* entry = GetEntryBlock(region);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 3u);

  loom_op_t* const_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_low_const_isa(const_op));

  loom_op_t* copy_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_low_copy_isa(copy_op));
  EXPECT_EQ(loom_low_copy_source(copy_op), loom_low_const_result(const_op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_low_copy_result(copy_op));
  ExpectTestLowCoreRegisterType(result_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, BuildsStructuralMove) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @move() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %moved = move %c0 : reg<test.i32> -> reg<test.i32>\n"
      "  return %moved\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_op_t* function_op = loom_block_op(loom_module_block(module), 0);
  loom_block_t* entry = GetEntryBlock(loom_low_func_def_body(function_op));
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 3u);
  loom_op_t* const_op = loom_block_op(entry, 0);
  loom_op_t* move_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_low_const_isa(const_op));
  ASSERT_TRUE(loom_low_move_isa(move_op));
  EXPECT_EQ(loom_low_move_source(move_op), loom_low_const_result(const_op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_low_move_result(move_op));
  ExpectTestLowCoreRegisterType(result_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, RejectsAmbiguousInferredResultType) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @ambiguous() asm {\n"
      "  %amb = test.ambiguous\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0),
            "result type annotation is required for one of: "
            "reg<test.i32> | reg<test.i64>");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, AcceptsExplicitAmbiguousResultType) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @ambiguous() -> "
      "(reg<test.i64>) asm {\n"
      "  %amb = test.ambiguous : reg<test.i64>\n"
      "  return %amb\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  loom_region_t* region = loom_low_func_def_body(function_op);
  loom_block_t* entry = GetEntryBlock(region);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 2u);

  loom_op_t* ambiguous_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_low_op_isa(ambiguous_op));
  loom_type_t result_type = loom_module_value_type(
      module, loom_low_op_results(ambiguous_op).values[0]);
  ExpectTestLowCoreRegisterType(result_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_I64, 1);

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, DistinguishesZeroImmediateConstFromOperandlessOp) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @zero_immediate() -> "
      "(reg<test.i32>, reg<test.i64>) asm {\n"
      "  %zero = test.const.zero.i32\n"
      "  %value = test.ambiguous : reg<test.i64>\n"
      "  return %zero, %value\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  loom_region_t* region = loom_low_func_def_body(function_op);
  loom_block_t* entry = GetEntryBlock(region);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 3u);

  loom_op_t* zero_op = loom_block_op(entry, 0);
  ASSERT_TRUE(loom_low_const_isa(zero_op));
  ExpectDescriptorOrdinal(loom_test_low_core_descriptor_set(),
                          loom_low_const_descriptor(zero_op),
                          IREE_SV("test.const.zero.i32"));
  EXPECT_EQ(loom_low_const_attrs(zero_op).count, 0u);

  loom_op_t* value_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_low_op_isa(value_op));
  ExpectDescriptorOrdinal(loom_test_low_core_descriptor_set(),
                          loom_low_op_descriptor(value_op),
                          IREE_SV("test.ambiguous"));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, RejectsInvalidExplicitResultType) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @invalid_type() asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %bad = test.add.i32 %c0, %c0 : reg<test.i64>\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0),
            "result type annotation must be one of: reg<test.i32>");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, InfersTiedResultTypeFromOperand) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @tied() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %tied = test.tied.any %c0\n"
      "  return %tied\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  loom_region_t* region = loom_low_func_def_body(function_op);
  loom_block_t* entry = GetEntryBlock(region);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 3u);

  loom_op_t* tied_op = loom_block_op(entry, 1);
  ASSERT_TRUE(loom_low_op_isa(tied_op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_low_op_results(tied_op).values[0]);
  ExpectTestLowCoreRegisterType(result_type,
                                TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
  ASSERT_EQ(tied_op->tied_result_count, 1u);
  const loom_tied_result_t* tied_results = loom_op_tied_results(tied_op);
  EXPECT_EQ(tied_results[0].result_index, 0u);
  EXPECT_EQ(tied_results[0].operand_index, 0u);

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, RejectsExplicitTiedResultMismatch) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @tied_mismatch() asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %bad = test.tied.any %c0 : reg<test.i64>\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0),
            "result type annotation must match tied operand type");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, RequiresEnvironmentAtFunctionBoundary) {
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      Parse("low.func.def target<test.low.core> @empty() asm {\n"
            "  return\n"
            "}\n",
            /*enable_low_asm=*/false, &module));
  EXPECT_EQ(module, nullptr);
  ASSERT_FALSE(capture_.diagnostics.empty());
  ExpectError(capture_.diagnostics[0],
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  EXPECT_EQ(GetStringParam(capture_.diagnostics[0], 0),
            "Low representation environment is not configured");
}

TEST_F(LowAsmParserTest, RejectsUnknownMnemonic) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @unknown_mnemonic() asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %bad = test.missing.i32 %c0\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "unknown low asm mnemonic");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, RejectsResultCountMismatch) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @result_count() asm {\n"
      "  %lhs, %rhs = test.const.i32 7\n"
      "}\n");
  ExpectError(diagnostics[0],
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 9));
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "test.const.i32");
  ExpectU32Param(diagnostics[0], 1, 1u);
  ExpectU32Param(diagnostics[0], 2, 2u);
}

TEST_F(LowAsmParserTest, RejectsOperandCountMismatch) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @operand_count() asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %sum = test.add.i32 %c0\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 10));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "test.add.i32");
  ExpectU32Param(*diagnostic, 1, 2u);
  ExpectU32Param(*diagnostic, 2, 1u);
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, RejectsUnexpectedNamedImmediate) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @named_immediate() asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  %call = test.call.i32 %c0 {callee_ordinal = 4}\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "unexpected named immediate");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, BuildsControlFlowBlocks) {
  loom_module_t* module = ParseOk(
      "low.func.def target<test.low.core> @control_flow() -> "
      "(reg<test.i32>) asm {\n"
      "  %cond = test.const.i32 1\n"
      "  low.cond_br %cond, ^then, ^else : reg<test.i32>\n"
      "^then:\n"
      "  low.br ^join(%cond: reg<test.i32>)\n"
      "^else:\n"
      "  low.br ^join(%cond: reg<test.i32>)\n"
      "^join(%result: reg<test.i32>):\n"
      "  return %result\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* module_block = loom_module_block(module);
  ASSERT_EQ(module_block->op_count, 1u);
  loom_op_t* function_op = loom_block_op(module_block, 0);
  ASSERT_TRUE(loom_low_func_def_isa(function_op));
  loom_region_t* region = loom_low_func_def_body(function_op);
  ASSERT_NE(region, nullptr);
  ASSERT_EQ(region->block_count, 4u);

  loom_block_t* entry = loom_region_block(region, 0);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->op_count, 2u);
  ASSERT_TRUE(loom_low_const_isa(loom_block_op(entry, 0)));
  ASSERT_TRUE(loom_low_cond_br_isa(loom_block_op(entry, 1)));

  loom_block_t* then_block = loom_region_block(region, 1);
  ASSERT_NE(then_block, nullptr);
  ASSERT_EQ(then_block->op_count, 1u);
  ASSERT_TRUE(loom_low_br_isa(loom_block_op(then_block, 0)));

  loom_block_t* else_block = loom_region_block(region, 2);
  ASSERT_NE(else_block, nullptr);
  ASSERT_EQ(else_block->op_count, 1u);
  ASSERT_TRUE(loom_low_br_isa(loom_block_op(else_block, 0)));

  loom_block_t* join_block = loom_region_block(region, 3);
  ASSERT_NE(join_block, nullptr);
  ASSERT_EQ(join_block->arg_count, 1u);
  ASSERT_EQ(join_block->op_count, 1u);
  ASSERT_TRUE(loom_low_return_isa(loom_block_op(join_block, 0)));

  loom_module_free(module);
}

TEST_F(LowAsmParserTest, RejectsTrailingTokenAfterLocation) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @trailing_token() asm {\n"
      "  %c0 = test.const.i32 7 loc(\"test.loom\":1:1) extra\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "unexpected token after packet");
  (void)diagnostics;
}

TEST_F(LowAsmParserTest, RejectsExtraReturnTypeAnnotation) {
  const auto& diagnostics = ParseExpectErrors(
      "low.func.def target<test.low.core> @return_type() -> "
      "(reg<test.i32>) asm {\n"
      "  %c0 = test.const.i32 7\n"
      "  return %c0 : reg<test.i32>, reg<test.i32>\n"
      "}\n");
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture_, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0),
            "return type annotation count does not match value count");
  (void)diagnostics;
}

}  // namespace
}  // namespace loom
