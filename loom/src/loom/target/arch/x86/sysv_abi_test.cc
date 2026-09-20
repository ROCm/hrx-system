// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/sysv_abi.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/x86/descriptors/avx2_descriptors.h"
#include "loom/target/arch/x86/descriptors/scalar_descriptors.h"
#include "loom/target/arch/x86/register_classes.h"

namespace loom {
namespace {

class X86AbiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_type_t MakeRegisterType(const loom_low_descriptor_set_t* descriptor_set,
                               loom_x86_register_class_t register_class) {
    uint16_t register_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_CHECK_OK(loom_x86_descriptor_set_register_class_id(
        descriptor_set, register_class, &register_class_id));
    loom_type_t type = loom_type_none();
    IREE_CHECK_OK(loom_low_build_register_type(descriptor_set,
                                               register_class_id, 1, &type));
    return type;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

TEST_F(X86AbiTest, ClassifiesRegisterAndStackIntegerArguments) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_scalar_core_descriptor_set();
  const loom_type_t gpr32 =
      MakeRegisterType(descriptor_set, LOOM_X86_REGISTER_CLASS_GPR32);
  const loom_type_t gpr64 =
      MakeRegisterType(descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  const loom_type_t argument_types[] = {
      gpr32, gpr64, gpr32, gpr64, gpr32, gpr64, gpr32, gpr64,
  };
  const loom_type_t result_types[] = {gpr64};

  loom_x86_sysv_abi_layout_t layout = {0};
  bool supported = false;
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_build(
      descriptor_set, argument_types, IREE_ARRAYSIZE(argument_types),
      result_types, IREE_ARRAYSIZE(result_types), &arena_, &layout,
      &supported));

  ASSERT_TRUE(supported);
  ASSERT_EQ(layout.argument_count, IREE_ARRAYSIZE(argument_types));
  const int64_t expected_locations[] = {
      LOOM_X86_SYSV_GPR_RDI,
      LOOM_X86_SYSV_GPR_RSI,
      LOOM_X86_SYSV_GPR_RDX,
      LOOM_X86_SYSV_GPR_RCX,
      LOOM_X86_SYSV_GPR_R8,
      LOOM_X86_SYSV_GPR_R9,
      loom_x86_sysv_abi_stack_location(0),
      loom_x86_sysv_abi_stack_location(8),
  };
  for (iree_host_size_t i = 0; i < layout.argument_count; ++i) {
    EXPECT_EQ(layout.argument_locations[i], expected_locations[i]);
  }
  EXPECT_EQ(layout.stack_argument_bytes, 16u);
  ASSERT_EQ(layout.result_count, 1u);
  EXPECT_EQ(layout.result_locations[0], LOOM_X86_SYSV_GPR_RAX);
}

TEST_F(X86AbiTest, SupportsVoidLeafSignatureWithoutStorage) {
  loom_x86_sysv_abi_layout_t layout = {0};
  bool supported = false;
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_build(
      loom_x86_scalar_core_descriptor_set(), nullptr, 0, nullptr, 0, &arena_,
      &layout, &supported));
  EXPECT_TRUE(supported);
  EXPECT_EQ(layout.argument_locations, nullptr);
  EXPECT_EQ(layout.argument_count, 0u);
  EXPECT_EQ(layout.result_locations, nullptr);
  EXPECT_EQ(layout.result_count, 0u);
  EXPECT_EQ(layout.stack_argument_bytes, 0u);
}

TEST_F(X86AbiTest, RejectsUnimplementedSignatureClasses) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx2_core_descriptor_set();
  const loom_type_t gpr64 =
      MakeRegisterType(descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  const loom_type_t xmm =
      MakeRegisterType(descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);

  loom_x86_sysv_abi_layout_t layout = {0};
  bool supported = true;
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_build(
      descriptor_set, &xmm, 1, &gpr64, 1, &arena_, &layout, &supported));
  EXPECT_FALSE(supported);

  const loom_type_t result_types[] = {gpr64, gpr64};
  supported = true;
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_build(
      descriptor_set, &gpr64, 1, result_types, IREE_ARRAYSIZE(result_types),
      &arena_, &layout, &supported));
  EXPECT_FALSE(supported);
}

TEST_F(X86AbiTest, AttributeRoundTripsCanonicalLayout) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_scalar_core_descriptor_set();
  const loom_type_t gpr64 =
      MakeRegisterType(descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  const loom_type_t argument_types[] = {gpr64, gpr64, gpr64, gpr64,
                                        gpr64, gpr64, gpr64, gpr64};
  const loom_type_t result_types[] = {gpr64};
  loom_x86_sysv_abi_layout_t layout = {0};
  bool supported = false;
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_build(
      descriptor_set, argument_types, IREE_ARRAYSIZE(argument_types),
      result_types, IREE_ARRAYSIZE(result_types), &arena_, &layout,
      &supported));
  ASSERT_TRUE(supported);

  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_make_attr(module_, &layout, &attr));
  loom_x86_sysv_abi_layout_t parsed = {0};
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_parse(
      module_, descriptor_set, loom_attr_as_dict(attr), argument_types,
      IREE_ARRAYSIZE(argument_types), result_types,
      IREE_ARRAYSIZE(result_types), &arena_, &parsed));

  ASSERT_EQ(parsed.argument_count, IREE_ARRAYSIZE(argument_types));
  EXPECT_EQ(parsed.argument_locations[6], loom_x86_sysv_abi_stack_location(0));
  EXPECT_EQ(parsed.argument_locations[7], loom_x86_sysv_abi_stack_location(8));
  EXPECT_EQ(parsed.stack_argument_bytes, 16u);
  ASSERT_EQ(parsed.result_count, 1u);
  EXPECT_EQ(parsed.result_locations[0], LOOM_X86_SYSV_GPR_RAX);
}

TEST_F(X86AbiTest, RejectsAuthoredNoncanonicalLocation) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_scalar_core_descriptor_set();
  const loom_type_t gpr64 =
      MakeRegisterType(descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  loom_x86_sysv_abi_layout_t layout = {0};
  bool supported = false;
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_build(
      descriptor_set, &gpr64, 1, &gpr64, 1, &arena_, &layout, &supported));
  ASSERT_TRUE(supported);
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_x86_sysv_abi_layout_make_attr(module_, &layout, &attr));

  loom_named_attr_slice_t attrs = loom_attr_as_dict(attr);
  ASSERT_EQ(attrs.count, 4u);
  loom_named_attr_t authored_entries[4];
  std::memcpy(authored_entries, attrs.entries, sizeof(authored_entries));
  const loom_string_id_t argument_locations_key =
      loom_module_lookup_string(module_, IREE_SV("argument_locations"));
  int64_t authored_argument_location = LOOM_X86_SYSV_GPR_RAX;
  bool replaced = false;
  for (loom_named_attr_t& entry : authored_entries) {
    if (entry.name_id == argument_locations_key) {
      entry.value = loom_attr_i64_array(&authored_argument_location, 1);
      replaced = true;
    }
  }
  ASSERT_TRUE(replaced);

  loom_x86_sysv_abi_layout_t parsed = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_x86_sysv_abi_layout_parse(
          module_, descriptor_set,
          loom_make_named_attr_slice(authored_entries,
                                     IREE_ARRAYSIZE(authored_entries)),
          &gpr64, 1, &gpr64, 1, &arena_, &parsed));
}

TEST(X86AbiRegisterSetsTest, ClassifiesVolatileAndPreservedGprs) {
  for (uint32_t i = 0; i < 16; ++i) {
    EXPECT_FALSE(loom_x86_sysv_gpr_is_caller_saved(i) &&
                 loom_x86_sysv_gpr_is_callee_saved(i));
  }
  EXPECT_TRUE(loom_x86_sysv_gpr_is_caller_saved(LOOM_X86_SYSV_GPR_RAX));
  EXPECT_TRUE(loom_x86_sysv_gpr_is_caller_saved(LOOM_X86_SYSV_GPR_R11));
  EXPECT_TRUE(loom_x86_sysv_gpr_is_callee_saved(LOOM_X86_SYSV_GPR_RBX));
  EXPECT_TRUE(loom_x86_sysv_gpr_is_callee_saved(LOOM_X86_SYSV_GPR_R15));
  EXPECT_FALSE(loom_x86_sysv_gpr_is_caller_saved(LOOM_X86_SYSV_GPR_RSP));
  EXPECT_TRUE(loom_x86_sysv_gpr_is_callee_saved(LOOM_X86_SYSV_GPR_RSP));
}

}  // namespace
}  // namespace loom
