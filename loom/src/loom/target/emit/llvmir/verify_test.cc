// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/verify.h"

#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/llvmir/builder.h"
#include "loom/target/emit/llvmir/module.h"

namespace loom {
namespace {

using ModulePtr =
    std::unique_ptr<loom_llvmir_module_t, void (*)(loom_llvmir_module_t*)>;

iree_status_t BuildPtrToAddrModule(
    const loom_llvmir_target_config_t* target_config, uint32_t address_space,
    uint32_t result_bit_width, loom_llvmir_module_t** out_module) {
  loom_llvmir_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_allocate(
      target_config, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, address_space, &pointer_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_integer_type(
      module, result_bit_width, &result_type));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("ptr_to_addr");
  function_desc.return_type = void_type;
  function_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_add_function(module, &function_desc, &function));

  loom_llvmir_parameter_desc_t pointer_desc = {};
  pointer_desc.type_id = pointer_type;
  pointer_desc.name = IREE_SV("pointer");
  loom_llvmir_value_id_t pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function, &pointer_desc, &pointer));

  loom_llvmir_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &block));
  loom_llvmir_cast_desc_t cast_desc = {};
  cast_desc.result_name = IREE_SV("address");
  cast_desc.result_type = result_type;
  cast_desc.op = LOOM_LLVMIR_CAST_PTR_TO_ADDR;
  cast_desc.value = pointer;
  loom_llvmir_value_id_t address = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(block, &cast_desc, &address));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret_void(block));

  *out_module = module_ptr.release();
  return iree_ok_status();
}

TEST(LlvmIrVerifyTest, RejectsFloatBinopOnIntegerType) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(
      loom_llvmir_module_allocate(NULL, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_integer_type(module_ptr.get(), 32, &i32_type));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("bad_binop");
  function_desc.return_type = i32_type;
  function_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_function_t* function = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));

  loom_llvmir_parameter_desc_t lhs_desc = {};
  lhs_desc.type_id = i32_type;
  lhs_desc.name = IREE_SV("lhs");
  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_function_add_parameter(function, &lhs_desc, &lhs));

  loom_llvmir_parameter_desc_t rhs_desc = {};
  rhs_desc.type_id = i32_type;
  rhs_desc.name = IREE_SV("rhs");
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_function_add_parameter(function, &rhs_desc, &rhs));

  loom_llvmir_block_t* block = NULL;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &block));

  loom_llvmir_binop_desc_t binop_desc = {};
  binop_desc.result_name = IREE_SV("wrong");
  binop_desc.result_type = i32_type;
  binop_desc.op = LOOM_LLVMIR_BINOP_FADD;
  binop_desc.lhs = lhs;
  binop_desc.rhs = rhs;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_build_binop(block, &binop_desc, &result));
  IREE_ASSERT_OK(loom_llvmir_build_ret(block, result));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_llvmir_verify_module(module_ptr.get()));
}

TEST(LlvmIrVerifyTest, RejectsIntegerFlagsOnFloatBinop) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(
      loom_llvmir_module_allocate(NULL, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_module_get_float_type(
      module_ptr.get(), LOOM_LLVMIR_FLOAT_F32, &f32_type));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("bad_flags");
  function_desc.return_type = f32_type;
  function_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_function_t* function = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));

  loom_llvmir_parameter_desc_t lhs_desc = {};
  lhs_desc.type_id = f32_type;
  lhs_desc.name = IREE_SV("lhs");
  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_function_add_parameter(function, &lhs_desc, &lhs));

  loom_llvmir_parameter_desc_t rhs_desc = {};
  rhs_desc.type_id = f32_type;
  rhs_desc.name = IREE_SV("rhs");
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_function_add_parameter(function, &rhs_desc, &rhs));

  loom_llvmir_block_t* block = NULL;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &block));

  loom_llvmir_binop_desc_t binop_desc = {};
  binop_desc.result_name = IREE_SV("wrong");
  binop_desc.result_type = f32_type;
  binop_desc.op = LOOM_LLVMIR_BINOP_FADD;
  binop_desc.lhs = lhs;
  binop_desc.rhs = rhs;
  binop_desc.integer_flags = LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_build_binop(block, &binop_desc, &result));
  IREE_ASSERT_OK(loom_llvmir_build_ret(block, result));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_llvmir_verify_module(module_ptr.get()));
}

TEST(LlvmIrVerifyTest, AcceptsPtrToAddrMatchingDefaultAddressWidth) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(BuildPtrToAddrModule(
      /*target_config=*/NULL, /*address_space=*/0, /*result_bit_width=*/64,
      &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  IREE_EXPECT_OK(loom_llvmir_verify_module(module_ptr.get()));
}

TEST(LlvmIrVerifyTest, RejectsPtrToAddrWithWrongDefaultAddressWidth) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(BuildPtrToAddrModule(
      /*target_config=*/NULL, /*address_space=*/0, /*result_bit_width=*/32,
      &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_llvmir_verify_module(module_ptr.get()));
}

TEST(LlvmIrVerifyTest, AcceptsPtrToAddrMatchingAddressSpaceDataLayoutWidth) {
  loom_llvmir_target_config_t target_config = {};
  target_config.data_layout =
      IREE_SV("e-p:64:64-p1:64:64-p7:160:256:256:32-p8:128:128:128:48");
  target_config.default_pointer_bitwidth = 64;
  target_config.index_bitwidth = 32;
  target_config.offset_bitwidth = 64;
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(BuildPtrToAddrModule(&target_config, /*address_space=*/7,
                                      /*result_bit_width=*/32, &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  IREE_EXPECT_OK(loom_llvmir_verify_module(module_ptr.get()));
}

TEST(LlvmIrVerifyTest, RejectsPtrToAddrWithWrongAddressSpaceDataLayoutWidth) {
  loom_llvmir_target_config_t target_config = {};
  target_config.data_layout =
      IREE_SV("e-p:64:64-p1:64:64-p7:160:256:256:32-p8:128:128:128:48");
  target_config.default_pointer_bitwidth = 64;
  target_config.index_bitwidth = 32;
  target_config.offset_bitwidth = 64;
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(BuildPtrToAddrModule(&target_config, /*address_space=*/7,
                                      /*result_bit_width=*/64, &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_llvmir_verify_module(module_ptr.get()));
}

}  // namespace
}  // namespace loom
