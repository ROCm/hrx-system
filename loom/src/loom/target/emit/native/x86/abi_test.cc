// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/abi.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/x86/descriptors/avx10_2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx2_features_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_bf16_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_features_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_vnni_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_int16_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_int8_descriptors.h"
#include "loom/target/arch/x86/descriptors/packed_dot_descriptors.h"
#include "loom/target/arch/x86/descriptors/scalar_descriptors.h"
#include "loom/target/arch/x86/descriptors/simd128_descriptors.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using DescriptorSetProvider = const loom_low_descriptor_set_t* (*)(void);
using ModulePtr = ::loom::testing::ModulePtr;

struct ProfileExpectation {
  const char* name;
  const char* descriptor_key;
  const char* carrier;
  DescriptorSetProvider provider;
  uint16_t vector_register_class;
  uint32_t vector_register_count;
  uint32_t mask_register_count;
};

static const ProfileExpectation kProfiles[] = {
    {/*.name=*/"scalar",
     /*.descriptor_key=*/"x86.scalar.core",
     /*.carrier=*/"gpr64",
     /*.provider=*/loom_x86_scalar_core_descriptor_set,
     /*.vector_register_class=*/LOOM_LOW_REG_CLASS_NONE,
     /*.vector_register_count=*/0,
     /*.mask_register_count=*/0},
    {.name = "simd128",
     .descriptor_key = "x86.simd128.core",
     .carrier = "xmm",
     .provider = loom_x86_simd128_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_XMM,
     .vector_register_count = 16,
     .mask_register_count = 0},
    {.name = "avx2",
     .descriptor_key = "x86.avx2.core",
     .carrier = "ymm",
     .provider = loom_x86_avx2_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_YMM,
     .vector_register_count = 16,
     .mask_register_count = 0},
    {.name = "avx2_packed_dot",
     .descriptor_key = "x86.avx2_features.core",
     .carrier = "ymm",
     .provider = loom_x86_avx2_features_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_YMM,
     .vector_register_count = 16,
     .mask_register_count = 0},
    {.name = "avx512",
     .descriptor_key = "x86.avx512.core",
     .carrier = "zmm",
     .provider = loom_x86_avx512_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_ZMM,
     .vector_register_count = 32,
     .mask_register_count = 8},
    {.name = "packed_dot",
     .descriptor_key = "x86.packed_dot.core",
     .carrier = "zmm",
     .provider = loom_x86_packed_dot_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_ZMM,
     .vector_register_count = 32,
     .mask_register_count = 0},
    {.name = "avx512_packed_dot",
     .descriptor_key = "x86.avx512_features.core",
     .carrier = "zmm",
     .provider = loom_x86_avx512_features_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_ZMM,
     .vector_register_count = 32,
     .mask_register_count = 8},
    {.name = "avx512_vnni",
     .descriptor_key = "x86.avx512_vnni.core",
     .carrier = "zmm",
     .provider = loom_x86_avx512_vnni_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_ZMM,
     .vector_register_count = 32,
     .mask_register_count = 0},
    {.name = "avx512_bf16",
     .descriptor_key = "x86.avx512_bf16.core",
     .carrier = "zmm",
     .provider = loom_x86_avx512_bf16_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_ZMM,
     .vector_register_count = 32,
     .mask_register_count = 0},
    {.name = "avx_vnni",
     .descriptor_key = "x86.avx_vnni.core",
     .carrier = "ymm",
     .provider = loom_x86_avx_vnni_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_YMM,
     .vector_register_count = 16,
     .mask_register_count = 0},
    {.name = "avx_vnni_int8",
     .descriptor_key = "x86.avx_vnni_int8.core",
     .carrier = "ymm",
     .provider = loom_x86_avx_vnni_int8_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_YMM,
     .vector_register_count = 16,
     .mask_register_count = 0},
    {.name = "avx_vnni_int16",
     .descriptor_key = "x86.avx_vnni_int16.core",
     .carrier = "ymm",
     .provider = loom_x86_avx_vnni_int16_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_YMM,
     .vector_register_count = 16,
     .mask_register_count = 0},
    {.name = "avx10_2",
     .descriptor_key = "x86.avx10_2.core",
     .carrier = "zmm",
     .provider = loom_x86_avx10_2_core_descriptor_set,
     .vector_register_class = LOOM_X86_REGISTER_CLASS_ZMM,
     .vector_register_count = 32,
     .mask_register_count = 0},
};

static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
    loom_x86_scalar_core_descriptor_set,
    loom_x86_simd128_core_descriptor_set,
    loom_x86_avx2_core_descriptor_set,
    loom_x86_avx2_features_core_descriptor_set,
    loom_x86_avx512_core_descriptor_set,
    loom_x86_packed_dot_core_descriptor_set,
    loom_x86_avx512_features_core_descriptor_set,
    loom_x86_avx512_vnni_core_descriptor_set,
    loom_x86_avx512_bf16_core_descriptor_set,
    loom_x86_avx_vnni_core_descriptor_set,
    loom_x86_avx_vnni_int8_core_descriptor_set,
    loom_x86_avx_vnni_int16_core_descriptor_set,
    loom_x86_avx10_2_core_descriptor_set,
};

struct PreparedAbi {
  loom_x86_function_abi_t abi = {};
  bool supported = false;
  iree_string_view_t constraint = {};
};

class X86FunctionAbiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    descriptor_registry_.descriptor_set_providers = kDescriptorSetProviders;
    descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kDescriptorSetProviders);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
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

  ModulePtr Parse(const std::string& source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry_, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("x86_abi_test.loom"), &context_, &block_pool_,
                        &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(loom_module_t* module, const char* name) {
    const loom_string_id_t name_id =
        loom_module_lookup_string(module, iree_make_cstring_view(name));
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[symbol_id].defining_op);
    IREE_ASSERT(loom_func_like_isa(function));
    return function;
  }

  PreparedAbi Prepare(loom_module_t* module, const char* function_name,
                      DescriptorSetProvider descriptor_set_provider) {
    loom_low_resolved_target_t target = {};
    target.descriptor_set = descriptor_set_provider();
    PreparedAbi prepared;
    IREE_CHECK_OK(loom_x86_function_abi_prepare(
        module, FindFunction(module, function_name), &target, &analysis_arena_,
        &prepared.supported, &prepared.constraint, &prepared.abi));
    return prepared;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_low_descriptor_registry_t descriptor_registry_ = {};
};

TEST_F(X86FunctionAbiTest, CommonClobbersCoverEveryDescriptorProfile) {
  for (const ProfileExpectation& profile : kProfiles) {
    SCOPED_TRACE(profile.name);
    const loom_low_descriptor_set_t* descriptor_set = profile.provider();
    const loom_low_call_clobber_list_t clobbers =
        loom_x86_function_common_call_clobbers(nullptr, descriptor_set);
    const iree_host_size_t expected_row_count =
        (profile.vector_register_count != 0 ? 1 : 0) +
        (profile.mask_register_count != 0 ? 1 : 0);
    ASSERT_EQ(clobbers.count, expected_row_count);
    if (profile.vector_register_count == 0) {
      EXPECT_EQ(clobbers.values, nullptr);
      continue;
    }

    const loom_low_call_clobber_t& vector_clobber = clobbers.values[0];
    EXPECT_EQ(vector_clobber.register_class, profile.vector_register_class);
    EXPECT_EQ(vector_clobber.location, 0u);
    EXPECT_EQ(vector_clobber.count, profile.vector_register_count);
    ASSERT_LT(vector_clobber.register_class, descriptor_set->reg_class_count);
    EXPECT_EQ(descriptor_set->reg_classes[vector_clobber.register_class]
                  .allocatable_count,
              vector_clobber.count);

    if (profile.mask_register_count != 0) {
      const loom_low_call_clobber_t& mask_clobber = clobbers.values[1];
      EXPECT_EQ(mask_clobber.register_class, LOOM_X86_REGISTER_CLASS_K);
      EXPECT_EQ(mask_clobber.location, 0u);
      EXPECT_EQ(mask_clobber.count, profile.mask_register_count);
      ASSERT_LT(mask_clobber.register_class, descriptor_set->reg_class_count);
      EXPECT_EQ(descriptor_set->reg_classes[mask_clobber.register_class]
                    .allocatable_count,
                mask_clobber.count);
    }
  }
}

static void ExpectSingleValuePlan(const PreparedAbi& prepared,
                                  uint16_t register_class, uint16_t byte_length,
                                  uint8_t byte_alignment,
                                  loom_x86_call_abi_value_action_t action) {
  ASSERT_EQ(prepared.abi.call_contract.argument_count, 1u);
  ASSERT_EQ(prepared.abi.call_contract.result_count, 1u);
  const loom_low_allocation_abi_location_t& argument =
      prepared.abi.call_contract.arguments[0];
  EXPECT_EQ(argument.location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(argument.descriptor_reg_class_id, register_class);
  EXPECT_EQ(argument.location_base,
            register_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
                    register_class == LOOM_X86_REGISTER_CLASS_GPR64
                ? 7u
                : 0u);
  const loom_low_allocation_abi_location_t& result =
      prepared.abi.call_contract.results[0];
  EXPECT_EQ(result.location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(result.descriptor_reg_class_id, register_class);
  EXPECT_EQ(result.location_base, 0u);
  const bool has_upper_vector_state =
      register_class == LOOM_X86_REGISTER_CLASS_YMM ||
      register_class == LOOM_X86_REGISTER_CLASS_ZMM;
  EXPECT_EQ(prepared.abi.has_upper_vector_register_argument,
            has_upper_vector_state);
  EXPECT_EQ(prepared.abi.has_upper_vector_result, has_upper_vector_state);
  for (const loom_x86_abi_value_t* value :
       {prepared.abi.arguments, prepared.abi.results}) {
    EXPECT_EQ(value->stack_offset, UINT32_MAX);
    EXPECT_EQ(value->byte_length, byte_length);
    EXPECT_EQ(value->byte_alignment, byte_alignment);
    EXPECT_EQ(value->action, action);
  }
}

TEST_F(X86FunctionAbiTest, ScalarLogicalTypesUsePlatformClasses) {
  struct ScalarCase {
    const char* logical_type;
    const char* carrier;
    uint16_t register_class;
    uint16_t byte_length;
    uint8_t byte_alignment;
    loom_x86_call_abi_value_action_t action;
  };
  static const ScalarCase cases[] = {
      {"i1", "gpr32", LOOM_X86_REGISTER_CLASS_GPR32, 1, 1,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I1},
      {"i8", "gpr32", LOOM_X86_REGISTER_CLASS_GPR32, 1, 1,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I8},
      {"f8E4M3", "gpr32", LOOM_X86_REGISTER_CLASS_GPR32, 1, 1,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I8},
      {"f8E5M2", "gpr32", LOOM_X86_REGISTER_CLASS_GPR32, 1, 1,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I8},
      {"i16", "gpr32", LOOM_X86_REGISTER_CLASS_GPR32, 2, 2,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I16},
      {"f16", "gpr32", LOOM_X86_REGISTER_CLASS_XMM, 2, 2,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"f16", "xmm", LOOM_X86_REGISTER_CLASS_XMM, 2, 2,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"bf16", "gpr32", LOOM_X86_REGISTER_CLASS_XMM, 2, 2,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"bf16", "xmm", LOOM_X86_REGISTER_CLASS_XMM, 2, 2,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"i32", "gpr32", LOOM_X86_REGISTER_CLASS_GPR32, 4, 4,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"i64", "gpr64", LOOM_X86_REGISTER_CLASS_GPR64, 8, 8,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"index", "gpr64", LOOM_X86_REGISTER_CLASS_GPR64, 8, 8,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"offset", "gpr64", LOOM_X86_REGISTER_CLASS_GPR64, 8, 8,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"buffer", "gpr64", LOOM_X86_REGISTER_CLASS_GPR64, 8, 8,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"f32", "xmm", LOOM_X86_REGISTER_CLASS_XMM, 4, 4,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
      {"f64", "xmm", LOOM_X86_REGISTER_CLASS_XMM, 8, 8,
       LOOM_X86_CALL_ABI_VALUE_ACTION_NONE},
  };

  std::string source;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(cases); ++i) {
    source +=
        "low.func.decl target<x86.avx512.core> abi(object_function) "
        "abi_layout({signature = (";
    source += cases[i].logical_type;
    source += ") -> (";
    source += cases[i].logical_type;
    source += ")}) @case" + std::to_string(i) + "(%value: reg<x86.";
    source += cases[i].carrier;
    source += ">) -> (reg<x86.";
    source += cases[i].carrier;
    source += ">)\n";
  }
  ModulePtr module = Parse(source);

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(cases); ++i) {
    SCOPED_TRACE(cases[i].logical_type);
    const std::string name = "case" + std::to_string(i);
    const PreparedAbi prepared = Prepare(module.get(), name.c_str(),
                                         loom_x86_avx512_core_descriptor_set);
    EXPECT_TRUE(prepared.supported);
    ExpectSingleValuePlan(prepared, cases[i].register_class,
                          cases[i].byte_length, cases[i].byte_alignment,
                          cases[i].action);
  }
}

TEST_F(X86FunctionAbiTest, VectorLogicalTypesCoverEveryWidthFamily) {
  struct VectorCase {
    std::string logical_type;
    const char* carrier;
    uint16_t register_class;
    uint16_t byte_length;
  };
  std::vector<VectorCase> cases;
  static const struct {
    const char* element_type;
    uint16_t element_bits;
  } elements[] = {
      {"i8", 8},    {"f8E4M3", 8}, {"f8E5M2", 8}, {"i16", 16}, {"f16", 16},
      {"bf16", 16}, {"i32", 32},   {"f32", 32},   {"i64", 64}, {"f64", 64},
  };
  static const struct {
    uint16_t bits;
    const char* carrier;
    uint16_t register_class;
  } widths[] = {
      {64, "xmm", LOOM_X86_REGISTER_CLASS_XMM},
      {128, "xmm", LOOM_X86_REGISTER_CLASS_XMM},
      {256, "ymm", LOOM_X86_REGISTER_CLASS_YMM},
      {512, "zmm", LOOM_X86_REGISTER_CLASS_ZMM},
  };
  for (const auto& element : elements) {
    for (const auto& width : widths) {
      cases.push_back({"vector<" +
                           std::to_string(width.bits / element.element_bits) +
                           "x" + element.element_type + ">",
                       width.carrier, width.register_class,
                       static_cast<uint16_t>(width.bits / 8)});
    }
  }
  for (uint16_t lane_count : {2, 4, 8, 16, 32, 64}) {
    const uint16_t register_class =
        lane_count <= 16   ? LOOM_X86_REGISTER_CLASS_XMM
        : lane_count == 32 ? LOOM_X86_REGISTER_CLASS_YMM
                           : LOOM_X86_REGISTER_CLASS_ZMM;
    const char* carrier = lane_count <= 16   ? "xmm"
                          : lane_count == 32 ? "ymm"
                                             : "zmm";
    const uint16_t byte_length = lane_count <= 16 ? 16 : lane_count;
    cases.push_back({"vector<" + std::to_string(lane_count) + "xi1>", carrier,
                     register_class, byte_length});
  }

  std::string source;
  for (iree_host_size_t i = 0; i < cases.size(); ++i) {
    source +=
        "low.func.decl target<x86.avx512.core> abi(object_function) "
        "abi_layout({signature = (" +
        cases[i].logical_type + ") -> (" + cases[i].logical_type + ")}) @case" +
        std::to_string(i) + "(%value: reg<x86." + cases[i].carrier +
        ">) -> (reg<x86." + cases[i].carrier + ">)\n";
  }
  ModulePtr module = Parse(source);

  for (iree_host_size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(cases[i].logical_type);
    const std::string name = "case" + std::to_string(i);
    const PreparedAbi prepared = Prepare(module.get(), name.c_str(),
                                         loom_x86_avx512_core_descriptor_set);
    EXPECT_TRUE(prepared.supported);
    ExpectSingleValuePlan(prepared, cases[i].register_class,
                          cases[i].byte_length,
                          static_cast<uint8_t>(cases[i].byte_length),
                          LOOM_X86_CALL_ABI_VALUE_ACTION_NONE);
  }
}

TEST_F(X86FunctionAbiTest, EveryDescriptorProfileAdmitsItsWidestCarrier) {
  std::string source;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kProfiles); ++i) {
    source += "low.func.decl target<";
    source += kProfiles[i].descriptor_key;
    source += "> @profile" + std::to_string(i) + "(%value: reg<x86.";
    source += kProfiles[i].carrier;
    source += ">) -> (reg<x86.";
    source += kProfiles[i].carrier;
    source += ">)\n";
  }
  ModulePtr module = Parse(source);

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kProfiles); ++i) {
    SCOPED_TRACE(kProfiles[i].name);
    const std::string name = "profile" + std::to_string(i);
    const PreparedAbi prepared =
        Prepare(module.get(), name.c_str(), kProfiles[i].provider);
    ASSERT_TRUE(prepared.supported);
    ASSERT_EQ(prepared.abi.call_contract.clobbers.count, 2u);
    EXPECT_EQ(prepared.abi.call_contract.clobbers.values[0].register_class,
              LOOM_X86_REGISTER_CLASS_GPR64);
    EXPECT_EQ(prepared.abi.call_contract.clobbers.values[0].location, 0u);
    EXPECT_EQ(prepared.abi.call_contract.clobbers.values[0].count, 3u);
    EXPECT_EQ(prepared.abi.call_contract.clobbers.values[1].register_class,
              LOOM_X86_REGISTER_CLASS_GPR64);
    EXPECT_EQ(prepared.abi.call_contract.clobbers.values[1].location, 6u);
    EXPECT_EQ(prepared.abi.call_contract.clobbers.values[1].count, 6u);
    const uint16_t register_class =
        kProfiles[i].vector_register_class == LOOM_LOW_REG_CLASS_NONE
            ? LOOM_X86_REGISTER_CLASS_GPR64
            : kProfiles[i].vector_register_class;
    const uint16_t byte_length =
        register_class == LOOM_X86_REGISTER_CLASS_GPR64 ? 8
        : register_class == LOOM_X86_REGISTER_CLASS_XMM ? 16
        : register_class == LOOM_X86_REGISTER_CLASS_YMM ? 32
                                                        : 64;
    ExpectSingleValuePlan(prepared, register_class, byte_length,
                          static_cast<uint8_t>(byte_length),
                          LOOM_X86_CALL_ABI_VALUE_ACTION_NONE);
  }
}

TEST_F(X86FunctionAbiTest, IntegerAndSseBanksOverflowIndependently) {
  ModulePtr module = Parse(R"(
low.func.decl target<x86.avx2.core> abi(object_function) abi_layout({signature = (buffer, vector<8xi32>, i64, vector<8xi32>, i64, vector<8xi32>, i64, vector<8xi32>, i64, vector<8xi32>, i64, vector<8xi32>, i64, vector<8xi32>, i64, vector<8xi32>, vector<8xi32>) -> ()}) @mixed(%output: reg<x86.gpr64>, %v0: reg<x86.ymm>, %g0: reg<x86.gpr64>, %v1: reg<x86.ymm>, %g1: reg<x86.gpr64>, %v2: reg<x86.ymm>, %g2: reg<x86.gpr64>, %v3: reg<x86.ymm>, %g3: reg<x86.gpr64>, %v4: reg<x86.ymm>, %g4: reg<x86.gpr64>, %v5: reg<x86.ymm>, %g5: reg<x86.gpr64>, %v6: reg<x86.ymm>, %g6: reg<x86.gpr64>, %v7: reg<x86.ymm>, %v8: reg<x86.ymm>)
)");
  const PreparedAbi prepared =
      Prepare(module.get(), "mixed", loom_x86_avx2_core_descriptor_set);
  ASSERT_TRUE(prepared.supported);
  ASSERT_EQ(prepared.abi.call_contract.argument_count, 17u);
  static const uint16_t integer_indices[] = {0, 2, 4, 6, 8, 10};
  static const uint8_t integer_registers[] = {7, 6, 2, 1, 8, 9};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(integer_indices); ++i) {
    const loom_low_allocation_abi_location_t& location =
        prepared.abi.call_contract.arguments[integer_indices[i]];
    EXPECT_EQ(location.location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    EXPECT_EQ(location.descriptor_reg_class_id, LOOM_X86_REGISTER_CLASS_GPR64);
    EXPECT_EQ(location.location_base, integer_registers[i]);
  }
  for (uint16_t i = 0; i < 8; ++i) {
    const loom_low_allocation_abi_location_t& location =
        prepared.abi.call_contract.arguments[1 + 2 * i];
    EXPECT_EQ(location.location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    EXPECT_EQ(location.descriptor_reg_class_id, LOOM_X86_REGISTER_CLASS_YMM);
    EXPECT_EQ(location.location_base, i);
  }
  for (uint16_t index : {12, 14, 16}) {
    EXPECT_EQ(prepared.abi.call_contract.arguments[index].location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED);
  }
  EXPECT_EQ(prepared.abi.arguments[12].stack_offset, 0u);
  EXPECT_EQ(prepared.abi.arguments[14].stack_offset, 8u);
  EXPECT_EQ(prepared.abi.arguments[16].stack_offset, 32u);
  EXPECT_EQ(prepared.abi.stack_argument_bytes, 64u);
  EXPECT_EQ(prepared.abi.stack_argument_alignment, 32u);
  EXPECT_TRUE(prepared.abi.has_simd_stack_argument);
  EXPECT_TRUE(prepared.abi.has_upper_vector_register_argument);
  EXPECT_FALSE(prepared.abi.has_upper_vector_result);
}

TEST_F(X86FunctionAbiTest, HalTaskLayoutDoesNotReplacePhysicalSignature) {
  ModulePtr module = Parse(R"(
low.func.decl target<x86.scalar.core> abi(hal_kernel) abi_layout({offsets = [0], signature = (buffer) -> ()}) @kernel(%environment: reg<x86.gpr64>, %dispatch_state: reg<x86.gpr64>, %workgroup_state: reg<x86.gpr64>) -> (reg<x86.gpr32>)
)");
  const PreparedAbi prepared =
      Prepare(module.get(), "kernel", loom_x86_scalar_core_descriptor_set);
  ASSERT_TRUE(prepared.supported);
  ASSERT_EQ(prepared.abi.call_contract.argument_count, 3u);
  static const uint32_t argument_registers[] = {7, 6, 2};
  for (uint16_t i = 0; i < 3; ++i) {
    const loom_low_allocation_abi_location_t& location =
        prepared.abi.call_contract.arguments[i];
    EXPECT_EQ(location.location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    EXPECT_EQ(location.descriptor_reg_class_id, LOOM_X86_REGISTER_CLASS_GPR64);
    EXPECT_EQ(location.location_base, argument_registers[i]);
  }
  ASSERT_EQ(prepared.abi.call_contract.result_count, 1u);
  EXPECT_EQ(prepared.abi.call_contract.results[0].descriptor_reg_class_id,
            LOOM_X86_REGISTER_CLASS_GPR32);
  EXPECT_EQ(prepared.abi.call_contract.results[0].location_base, 0u);
}

TEST_F(X86FunctionAbiTest, StackRowsRetainExactScalarAndVectorWidths) {
  struct StackCase {
    const char* name;
    const char* descriptor_key;
    const char* logical_type;
    const char* carrier;
    DescriptorSetProvider provider;
    uint32_t stack_bytes;
    uint8_t stack_alignment;
  };
  static const StackCase cases[] = {
      {"f16", "x86.avx2.core", "f16", "gpr32",
       loom_x86_avx2_core_descriptor_set, 8, 16},
      {"f16_xmm", "x86.avx512.core", "f16", "xmm",
       loom_x86_avx512_core_descriptor_set, 8, 16},
      {"f32", "x86.avx2.core", "f32", "xmm", loom_x86_avx2_core_descriptor_set,
       8, 16},
      {"f64", "x86.avx2.core", "f64", "xmm", loom_x86_avx2_core_descriptor_set,
       8, 16},
      {"xmm64", "x86.avx512.core", "vector<4xf16>", "xmm",
       loom_x86_avx512_core_descriptor_set, 8, 16},
      {"xmm", "x86.simd128.core", "vector<4xi32>", "xmm",
       loom_x86_simd128_core_descriptor_set, 16, 16},
      {"ymm", "x86.avx2.core", "vector<8xi32>", "ymm",
       loom_x86_avx2_core_descriptor_set, 32, 32},
      {"zmm", "x86.avx512.core", "vector<16xi32>", "zmm",
       loom_x86_avx512_core_descriptor_set, 64, 64},
  };
  std::string source;
  for (const StackCase& test_case : cases) {
    source += "low.func.decl target<";
    source += test_case.descriptor_key;
    source += "> abi(object_function) abi_layout({signature = (";
    for (int i = 0; i < 9; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += test_case.logical_type;
    }
    source += ") -> ()}) @";
    source += test_case.name;
    source += "(";
    for (int i = 0; i < 9; ++i) {
      if (i != 0) {
        source += ", ";
      }
      source += "%v" + std::to_string(i) + ": reg<x86.";
      source += test_case.carrier;
      source += ">";
    }
    source += ")\n";
  }
  ModulePtr module = Parse(source);

  for (const StackCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const PreparedAbi prepared =
        Prepare(module.get(), test_case.name, test_case.provider);
    ASSERT_TRUE(prepared.supported);
    EXPECT_EQ(prepared.abi.call_contract.arguments[8].location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED);
    EXPECT_EQ(prepared.abi.arguments[8].stack_offset, 0u);
    EXPECT_EQ(prepared.abi.stack_argument_bytes, test_case.stack_bytes);
    EXPECT_EQ(prepared.abi.stack_argument_alignment, test_case.stack_alignment);
    EXPECT_TRUE(prepared.abi.has_simd_stack_argument);
  }
}

TEST_F(X86FunctionAbiTest, LowXmmVectorResultRetainsLogicalWidth) {
  ModulePtr module = Parse(R"(
low.func.def target<x86.avx512.core> abi(object_function) abi_layout({signature = (vector<4xf16>, vector<4xf16>, vector<4xf16>) -> (vector<4xf16>, vector<4xf16>, vector<4xf16>)}) @results(%a: reg<x86.xmm>, %b: reg<x86.xmm>, %c: reg<x86.xmm>) -> (reg<x86.xmm>, reg<x86.xmm>, reg<x86.xmm>) asm {
  return %a, %b, %c
}
)");
  const PreparedAbi prepared =
      Prepare(module.get(), "results", loom_x86_avx512_core_descriptor_set);
  ASSERT_TRUE(prepared.supported);
  ASSERT_EQ(prepared.abi.call_contract.result_count, 3u);
  EXPECT_EQ(prepared.abi.call_contract.results[0].location_base, 0u);
  EXPECT_EQ(prepared.abi.call_contract.results[1].location_base, 1u);
  EXPECT_EQ(prepared.abi.call_contract.results[2].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED);
  EXPECT_EQ(prepared.abi.results[2].stack_offset, 0u);
  EXPECT_EQ(prepared.abi.results[2].byte_length, 8u);
  EXPECT_EQ(prepared.abi.results[2].byte_alignment, 8u);
  EXPECT_EQ(prepared.abi.call_storage_bytes, 8u);
  EXPECT_EQ(prepared.abi.call_storage_alignment, 16u);
}

TEST_F(X86FunctionAbiTest, PrivateResultsUseIndependentBanksAndOverflow) {
  ModulePtr module = Parse(R"(
low.func.def target<x86.avx2.core> abi(object_function) abi_layout({signature = (i1, f16, i64, vector<8xi32>, i16, f32, vector<8xi32>) -> (i1, f16, i64, vector<8xi32>, i16, f32, vector<8xi32>)}) @mixed_results(%b0: reg<x86.gpr32>, %h0: reg<x86.gpr32>, %g1: reg<x86.gpr64>, %v1: reg<x86.ymm>, %g2: reg<x86.gpr32>, %f2: reg<x86.xmm>, %v2: reg<x86.ymm>) -> (reg<x86.gpr32>, reg<x86.gpr32>, reg<x86.gpr64>, reg<x86.ymm>, reg<x86.gpr32>, reg<x86.xmm>, reg<x86.ymm>) asm {
  return %b0, %h0, %g1, %v1, %g2, %f2, %v2
}
)");
  const PreparedAbi prepared =
      Prepare(module.get(), "mixed_results", loom_x86_avx2_core_descriptor_set);
  ASSERT_TRUE(prepared.supported);
  ASSERT_EQ(prepared.abi.call_contract.result_count, 7u);

  static const struct {
    uint16_t index;
    uint16_t register_class;
    uint32_t location;
  } register_results[] = {
      {0, LOOM_X86_REGISTER_CLASS_GPR32, 0},
      {1, LOOM_X86_REGISTER_CLASS_XMM, 0},
      {2, LOOM_X86_REGISTER_CLASS_GPR64, 2},
      {3, LOOM_X86_REGISTER_CLASS_YMM, 1},
  };
  for (const auto& result : register_results) {
    const loom_low_allocation_abi_location_t& location =
        prepared.abi.call_contract.results[result.index];
    EXPECT_EQ(location.location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    EXPECT_EQ(location.descriptor_reg_class_id, result.register_class);
    EXPECT_EQ(location.location_base, result.location);
    EXPECT_EQ(prepared.abi.results[result.index].stack_offset, UINT32_MAX);
  }

  for (uint16_t i : {4, 5, 6}) {
    EXPECT_EQ(prepared.abi.call_contract.results[i].location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED);
  }
  EXPECT_EQ(prepared.abi.results[4].stack_offset, 0u);
  EXPECT_EQ(prepared.abi.results[4].byte_length, 2u);
  EXPECT_EQ(prepared.abi.results[4].action,
            LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I16);
  EXPECT_EQ(prepared.abi.results[5].stack_offset, 8u);
  EXPECT_EQ(prepared.abi.results[5].byte_length, 4u);
  EXPECT_EQ(prepared.abi.results[6].stack_offset, 32u);
  EXPECT_EQ(prepared.abi.results[6].byte_length, 32u);
  EXPECT_EQ(prepared.abi.stack_argument_bytes, 0u);
  EXPECT_EQ(prepared.abi.call_storage_bytes, 64u);
  EXPECT_EQ(prepared.abi.call_storage_alignment, 32u);
  EXPECT_TRUE(prepared.abi.has_indirect_results);
  EXPECT_TRUE(prepared.abi.has_upper_vector_result);
}

TEST_F(X86FunctionAbiTest, EveryProfileSupportsPrivateResultOverflow) {
  std::string source;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kProfiles); ++i) {
    source += "low.func.def target<";
    source += kProfiles[i].descriptor_key;
    source += "> @profile" + std::to_string(i);
    source += "(%a: reg<x86.";
    source += kProfiles[i].carrier;
    source += ">, %b: reg<x86.";
    source += kProfiles[i].carrier;
    source += ">, %c: reg<x86.";
    source += kProfiles[i].carrier;
    source += ">) -> (reg<x86.";
    source += kProfiles[i].carrier;
    source += ">, reg<x86.";
    source += kProfiles[i].carrier;
    source += ">, reg<x86.";
    source += kProfiles[i].carrier;
    source += ">) asm {\n  return %a, %b, %c\n}\n";
  }
  ModulePtr module = Parse(source);

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kProfiles); ++i) {
    SCOPED_TRACE(kProfiles[i].name);
    const std::string name = "profile" + std::to_string(i);
    const PreparedAbi prepared =
        Prepare(module.get(), name.c_str(), kProfiles[i].provider);
    ASSERT_TRUE(prepared.supported);
    ASSERT_EQ(prepared.abi.call_contract.result_count, 3u);
    const uint16_t register_class =
        kProfiles[i].vector_register_class == LOOM_LOW_REG_CLASS_NONE
            ? LOOM_X86_REGISTER_CLASS_GPR64
            : kProfiles[i].vector_register_class;
    const uint32_t byte_length =
        register_class == LOOM_X86_REGISTER_CLASS_GPR64 ? 8
        : register_class == LOOM_X86_REGISTER_CLASS_XMM ? 16
        : register_class == LOOM_X86_REGISTER_CLASS_YMM ? 32
                                                        : 64;
    EXPECT_EQ(prepared.abi.call_contract.results[0].location_base, 0u);
    EXPECT_EQ(prepared.abi.call_contract.results[1].location_base,
              register_class == LOOM_X86_REGISTER_CLASS_GPR64 ? 2u : 1u);
    EXPECT_EQ(prepared.abi.call_contract.results[2].location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED);
    EXPECT_EQ(prepared.abi.results[2].stack_offset, 0u);
    EXPECT_EQ(prepared.abi.call_storage_bytes, byte_length);
    EXPECT_EQ(prepared.abi.call_storage_alignment,
              static_cast<uint8_t>(byte_length < 16 ? 16 : byte_length));
    EXPECT_TRUE(prepared.abi.has_indirect_results);
  }
}

TEST_F(X86FunctionAbiTest, RejectsUnavailableAndMismatchedBoundaries) {
  ModulePtr module = Parse(R"(
low.func.decl target<x86.avx512.core> @unavailable(%value: reg<x86.xmm>)
low.func.decl target<x86.scalar.core> abi_layout({signature = (i64) -> ()}) @mismatch(%value: reg<x86.gpr32>)
low.func.decl target<x86.avx512.core> abi_layout({signature = (vector<3xi1>) -> ()}) @unsupported_predicate(%value: reg<x86.xmm>)
low.func.decl target<x86.scalar.core> @multiple_results() -> (reg<x86.gpr64>, reg<x86.gpr64>)
low.func.def public target<x86.scalar.core> @public_results(%a: reg<x86.gpr64>, %b: reg<x86.gpr64>) -> (reg<x86.gpr64>, reg<x86.gpr64>) asm {
  return %a, %b
}
)");
  const PreparedAbi unavailable =
      Prepare(module.get(), "unavailable", loom_x86_scalar_core_descriptor_set);
  EXPECT_FALSE(unavailable.supported);
  EXPECT_TRUE(iree_string_view_equal(
      unavailable.constraint,
      IREE_SV("native x86 ABI register class is unavailable in this profile")));
  const PreparedAbi mismatch =
      Prepare(module.get(), "mismatch", loom_x86_scalar_core_descriptor_set);
  EXPECT_FALSE(mismatch.supported);
  EXPECT_TRUE(iree_string_view_equal(
      mismatch.constraint,
      IREE_SV("native x86 argument logical type and Low carrier disagree")));
  const PreparedAbi unsupported_predicate =
      Prepare(module.get(), "unsupported_predicate",
              loom_x86_avx512_core_descriptor_set);
  EXPECT_FALSE(unsupported_predicate.supported);
  EXPECT_TRUE(iree_string_view_equal(
      unsupported_predicate.constraint,
      IREE_SV("native x86 argument logical type and Low carrier disagree")));
  const PreparedAbi multiple_results = Prepare(
      module.get(), "multiple_results", loom_x86_scalar_core_descriptor_set);
  EXPECT_FALSE(multiple_results.supported);
  EXPECT_TRUE(iree_string_view_equal(
      multiple_results.constraint,
      IREE_SV("native x86 public and imported functions require one platform "
              "result")));
  const PreparedAbi public_results = Prepare(
      module.get(), "public_results", loom_x86_scalar_core_descriptor_set);
  EXPECT_FALSE(public_results.supported);
  EXPECT_TRUE(iree_string_view_equal(
      public_results.constraint,
      IREE_SV("native x86 public and imported functions require one platform "
              "result")));
}

}  // namespace
}  // namespace loom
