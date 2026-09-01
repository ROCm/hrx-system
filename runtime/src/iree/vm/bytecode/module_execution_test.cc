// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/inspection.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/module_test_types.h"
#include "iree/vm/bytecode/wire/core/abi.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/float.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/ref.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/core/stack.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "iree/vm/process.h"

namespace iree::vm::bytecode::testing {
namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

TEST(VMBytecodeModuleTest, CapturesAssertionDiagnosticBeforeUnwind) {
  std::vector<uint8_t> image = BuildRefModuleImage();
  MutableFunctionImage function = FindFunctionImage(&image, 1);
  ASSERT_NE(function.row, nullptr);
  ASSERT_NE(function.bytecode, nullptr);
  function.row->value_register_count_u16 = 1;
  function.row->ref_register_count_u16 = 1;

  uint8_t* cursor = function.bytecode;
  const uint8_t* const end =
      function.bytecode + function.row->bytecode_length_u32;
  const iree_vm_isa_control_block_record_t block = {
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}};
  std::memcpy(cursor, &block, sizeof(block));
  cursor += sizeof(block);
  const iree_vm_isa_constant_zero_record_t zero = {
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 0, 0};
  std::memcpy(cursor, &zero, sizeof(zero));
  cursor += sizeof(zero);
  const iree_vm_isa_control_assert_record_t assertion = {
      IREE_VM_ISA_CORE_OPCODE_CONTROL_ASSERT, 0, 0, 0};
  std::memcpy(cursor, &assertion, sizeof(assertion));
  cursor += sizeof(assertion);
  while (cursor < end - sizeof(iree_vm_isa_control_return_record_t)) {
    std::memcpy(cursor, &zero, sizeof(zero));
    cursor += sizeof(zero);
  }
  const iree_vm_isa_control_return_record_t return_record = {
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}};
  std::memcpy(cursor, &return_record, sizeof(return_record));
  cursor += sizeof(return_record);
  ASSERT_EQ(cursor, end);

  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("assertion"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  iree_vm_environment_free(environment);

  iree_vm_ref_type_t buffer_type = nullptr;
  IREE_ASSERT_OK(iree_vm_module_ref_type_by_ordinal(module, 0, &buffer_type));
  const iree_vm_ref_types_t vm_types = {buffer_type};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                        iree_vm_variant_span_empty(),
                                        iree_allocator_system(), &process));
  iree_vm_function_t fail = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(process, IREE_SV("assertion"),
                                                 IREE_SV("fail"), &fail));

  constexpr char kMessage[] = "assertion diagnostic";
  iree_vm_buffer_t* message_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_const_byte_span(kMessage, sizeof(kMessage) - 1),
      /*minimum_alignment=*/1, iree_allocator_system(), &message_buffer));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&vm_types, &message_buffer),
  };
  iree_status_t status_value = iree_vm_invoke(
      invocation, fail, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_empty());
  iree::Status status(std::move(status_value));
  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_NE(status.ToString().find(kMessage), std::string::npos)
      << status.ToString();
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
}

TEST(VMBytecodeModuleTest, ExecutesScalarStateInstructions) {
  std::vector<uint8_t> image = BuildScalarStateModuleImage();
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("scalar_state"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  iree_vm_environment_free(environment);

  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                        iree_vm_variant_span_empty(),
                                        iree_allocator_system(), &process));
  iree_vm_function_t run = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("scalar_state"), IREE_SV("run"), &run));

  iree_vm_variant_t results[5] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, run, iree_vm_variant_span_empty(),
                                iree_vm_variant_span_from_array(results)));
  const int64_t expected[] = {
      0,
      INT64_C(2309737967),
      INT64_C(81985529216486895),
      INT64_C(-81985529216486896),
      INT64_C(3735928559),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    int64_t value = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[i], &value));
    EXPECT_EQ(value, expected[i]);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
}

TEST(VMBytecodeModuleTest, ExecutesTypedRefOwnershipExactly) {
  std::vector<uint8_t> image = BuildRefModuleImage();
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("refs"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  iree_vm_environment_free(environment);

  iree_vm_ref_type_t buffer_type = nullptr;
  IREE_ASSERT_OK(iree_vm_module_ref_type_by_ordinal(module, 0, &buffer_type));
  const iree_vm_ref_types_t vm_types = {buffer_type};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                        iree_vm_variant_span_empty(),
                                        iree_allocator_system(), &process));
  iree_vm_function_t exercise = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("refs"), IREE_SV("exercise"), &exercise));
  iree_vm_function_t fail = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(process, IREE_SV("refs"),
                                                 IREE_SV("fail"), &fail));

  auto expect_comparisons = [](iree_vm_variant_t* results, int32_t is_null,
                               int32_t equals_null) {
    int32_t actual = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &actual));
    EXPECT_EQ(actual, is_null);
    IREE_ASSERT_OK(iree_vm_i32_from_variant(results[1], &actual));
    EXPECT_EQ(actual, equals_null);
    IREE_ASSERT_OK(iree_vm_i32_from_variant(results[2], &actual));
    EXPECT_EQ(actual, 1);
  };

  iree_vm_variant_t null_arguments[] = {iree_vm_variant_null()};
  iree_vm_variant_t null_results[4] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, exercise,
                                iree_vm_variant_span_from_array(null_arguments),
                                iree_vm_variant_span_from_array(null_results)));
  EXPECT_TRUE(iree_vm_variant_is_empty(null_arguments[0]));
  expect_comparisons(null_results, 1, 1);
  EXPECT_TRUE(iree_vm_variant_is_null(null_results[3]));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(null_results));

  uint8_t storage = 0;
  int borrowed_release_count = 0;
  iree_vm_buffer_t* borrowed_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ, iree_make_byte_span(&storage, 1),
      {CountBufferRelease, &borrowed_release_count}, iree_allocator_system(),
      &borrowed_buffer));
  iree_vm_variant_t borrowed_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&vm_types, borrowed_buffer),
  };
  iree_vm_variant_t borrowed_results[4] = {};
  IREE_ASSERT_OK(iree_vm_invoke(
      invocation, exercise, iree_vm_variant_span_from_array(borrowed_arguments),
      iree_vm_variant_span_from_array(borrowed_results)));
  EXPECT_TRUE(iree_vm_variant_is_empty(borrowed_arguments[0]));
  expect_comparisons(borrowed_results, 0, 0);
  EXPECT_TRUE(iree_vm_variant_ref_isa(borrowed_results[3], buffer_type));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(borrowed_results));
  EXPECT_EQ(borrowed_release_count, 0);
  iree_vm_buffer_release(borrowed_buffer);
  EXPECT_EQ(borrowed_release_count, 1);

  int moved_release_count = 0;
  iree_vm_buffer_t* moved_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                     iree_make_byte_span(&storage, 1),
                                     {CountBufferRelease, &moved_release_count},
                                     iree_allocator_system(), &moved_buffer));
  iree_vm_variant_t moved_arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&vm_types, &moved_buffer),
  };
  EXPECT_EQ(moved_buffer, nullptr);
  iree_vm_variant_t moved_results[4] = {};
  IREE_ASSERT_OK(iree_vm_invoke(
      invocation, exercise, iree_vm_variant_span_from_array(moved_arguments),
      iree_vm_variant_span_from_array(moved_results)));
  EXPECT_TRUE(iree_vm_variant_is_empty(moved_arguments[0]));
  expect_comparisons(moved_results, 0, 0);
  EXPECT_EQ(moved_release_count, 0);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(moved_results));
  EXPECT_EQ(moved_release_count, 1);

  int failure_release_count = 0;
  iree_vm_buffer_t* failure_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ, iree_make_byte_span(&storage, 1),
      {CountBufferRelease, &failure_release_count}, iree_allocator_system(),
      &failure_buffer));
  iree_vm_variant_t failure_arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&vm_types, &failure_buffer),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invoke(invocation, fail,
                     iree_vm_variant_span_from_array(failure_arguments),
                     iree_vm_variant_span_empty()));
  EXPECT_TRUE(iree_vm_variant_is_empty(failure_arguments[0]));
  EXPECT_EQ(failure_release_count, 1);

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
}

TEST(VMBytecodeModuleTest, ExecutesRefABIAndGlobalOwnershipExactly) {
  std::vector<uint8_t> image = BuildRefStateModuleImage();
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(
      environment, &kRefStateTestTypeTable));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("ref_state"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  iree_vm_environment_free(environment);

  iree_vm_ref_type_t buffer_type = nullptr;
  IREE_ASSERT_OK(iree_vm_module_ref_type_by_ordinal(module, 0, &buffer_type));
  const iree_vm_ref_types_t vm_types = {buffer_type};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));

  uint8_t immutable_storage = 0;
  uint8_t mutable_storage = 0;
  int immutable_release_count = 0;
  int mutable_release_count = 0;
  iree_vm_buffer_t* immutable_buffer = nullptr;
  iree_vm_buffer_t* mutable_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span(&immutable_storage, sizeof(immutable_storage)),
      {CountBufferRelease, &immutable_release_count}, iree_allocator_system(),
      &immutable_buffer));
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span(&mutable_storage, sizeof(mutable_storage)),
      {CountBufferRelease, &mutable_release_count}, iree_allocator_system(),
      &mutable_buffer));

  iree_vm_variant_t invalid_process_arguments[] = {
      iree_vm_variant_null(),
      iree_vm_buffer_variant_from_ptr_borrowed(&vm_types, mutable_buffer),
  };
  iree_vm_process_t* process = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_process_create(
          program, invocation,
          iree_vm_variant_span_from_array(invalid_process_arguments),
          iree_allocator_system(), &process));
  EXPECT_EQ(process, nullptr);
  EXPECT_TRUE(iree_vm_variant_is_empty(invalid_process_arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(invalid_process_arguments[1]));
  EXPECT_EQ(immutable_release_count, 0);
  EXPECT_EQ(mutable_release_count, 0);

  iree_vm_variant_t process_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&vm_types, immutable_buffer),
      iree_vm_buffer_variant_from_ptr_borrowed(&vm_types, mutable_buffer),
  };
  IREE_ASSERT_OK(iree_vm_process_create(
      program, invocation, iree_vm_variant_span_from_array(process_arguments),
      iree_allocator_system(), &process));
  EXPECT_TRUE(iree_vm_variant_is_empty(process_arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(process_arguments[1]));

  iree_vm_function_t read = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(process, IREE_SV("ref_state"),
                                                 IREE_SV("read"), &read));
  iree_vm_variant_t read_results[2] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, read, iree_vm_variant_span_empty(),
                                iree_vm_variant_span_from_array(read_results)));
  iree_vm_buffer_t* read_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
      &vm_types, read_results[0], &read_buffer));
  EXPECT_EQ(read_buffer, immutable_buffer);
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
      &vm_types, read_results[1], &read_buffer));
  EXPECT_EQ(read_buffer, mutable_buffer);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(read_results));

  int wrong_type_destruction_count = 0;
  RefStateTestObject wrong_type_object = {};
  iree_vm_ref_object_initialize(&wrong_type_object.ref_object);
  wrong_type_object.destruction_count = &wrong_type_destruction_count;
  iree_vm_function_t wrong_store = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("ref_state"), IREE_SV("wrong_store"), &wrong_store));
  iree_vm_variant_t wrong_argument[] = {iree_vm_variant_from_ptr_borrowed(
      &wrong_type_object, &kRefStateTestObjectType)};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invoke(invocation, wrong_store,
                     iree_vm_variant_span_from_array(wrong_argument),
                     iree_vm_variant_span_empty()));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_argument[0]));
  EXPECT_EQ(wrong_type_destruction_count, 0);

  read_results[0] = iree_vm_variant_empty();
  read_results[1] = iree_vm_variant_empty();
  IREE_ASSERT_OK(iree_vm_invoke(invocation, read, iree_vm_variant_span_empty(),
                                iree_vm_variant_span_from_array(read_results)));
  read_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
      &vm_types, read_results[1], &read_buffer));
  EXPECT_EQ(read_buffer, mutable_buffer);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(read_results));

  std::array<iree_vm_variant_t, 17> overflow_arguments;
  for (iree_host_size_t i = 0; i < overflow_arguments.size() - 1; ++i) {
    overflow_arguments[i] = iree_vm_variant_null();
  }
  overflow_arguments.back() =
      iree_vm_buffer_variant_from_ptr_borrowed(&vm_types, immutable_buffer);
  std::array<iree_vm_variant_t, 17> overflow_results = {};
  iree_vm_function_t overflow = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("ref_state"), IREE_SV("overflow"), &overflow));
  IREE_ASSERT_OK(
      iree_vm_invoke(invocation, overflow,
                     iree_vm_variant_span_from_ptr(overflow_arguments.data(),
                                                   overflow_arguments.size()),
                     iree_vm_variant_span_from_ptr(overflow_results.data(),
                                                   overflow_results.size())));
  for (const iree_vm_variant_t argument : overflow_arguments) {
    EXPECT_TRUE(iree_vm_variant_is_empty(argument));
  }
  for (iree_host_size_t i = 0; i < overflow_results.size() - 1; ++i) {
    EXPECT_TRUE(iree_vm_variant_is_null(overflow_results[i]));
  }
  read_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
      &vm_types, overflow_results.back(), &read_buffer));
  EXPECT_EQ(read_buffer, immutable_buffer);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_ptr(
      overflow_results.data(), overflow_results.size()));
  EXPECT_EQ(immutable_release_count, 0);
  EXPECT_EQ(mutable_release_count, 0);

  iree_vm_process_release(process);
  iree_vm_invocation_free(invocation);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
  EXPECT_EQ(immutable_release_count, 0);
  EXPECT_EQ(mutable_release_count, 0);
  iree_vm_buffer_release(immutable_buffer);
  iree_vm_buffer_release(mutable_buffer);
  EXPECT_EQ(immutable_release_count, 1);
  EXPECT_EQ(mutable_release_count, 1);
  iree_vm_ref_object_release(&wrong_type_object, &kRefStateTestObjectType);
  EXPECT_EQ(wrong_type_destruction_count, 1);
}

TEST(VMBytecodeModuleTest, PreservesIntegerSourcesAcrossDestinationAliasing) {
  auto expect_result = [](uint8_t opcode, uint8_t dst, uint8_t lhs_or_src,
                          uint8_t rhs_or_zero, int32_t argument,
                          int32_t expected) {
    std::vector<uint8_t> image = BuildOwnershipModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 1);
    ASSERT_NE(function.row, nullptr);
    constexpr uint32_t kArithmeticInstructionOffset = 8;
    function.bytecode[kArithmeticInstructionOffset + 0] = opcode;
    function.bytecode[kArithmeticInstructionOffset + 1] = dst;
    function.bytecode[kArithmeticInstructionOffset + 2] = lhs_or_src;
    function.bytecode[kArithmeticInstructionOffset + 3] = rhs_or_zero;

    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    iree_vm_module_t* module = nullptr;
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment, IREE_SV("aliasing"),
        {iree_make_const_byte_span(image.data(), image.size()),
         iree_allocator_null()},
        iree_allocator_system(), &module));
    iree_vm_environment_free(environment);

    iree_vm_program_t* program = nullptr;
    IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                          iree_allocator_system(), &program));
    iree_vm_invocation_t* invocation = nullptr;
    IREE_ASSERT_OK(iree_vm_invocation_allocate(
        kInvocationStorageSize, iree_allocator_system(), &invocation));
    iree_vm_process_t* process = nullptr;
    IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process));
    iree_vm_function_t run = iree_vm_function_null();
    IREE_ASSERT_OK(iree_vm_process_lookup_function(process, IREE_SV("aliasing"),
                                                   IREE_SV("run"), &run));

    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(argument)};
    iree_vm_variant_t results[2] = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation, run,
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    int32_t actual = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &actual));
    EXPECT_EQ(actual, expected);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

    iree_vm_invocation_free(invocation);
    iree_vm_process_release(process);
    iree_vm_program_release(program);
    iree_vm_module_release(module);
  };

  // Binary destinations may alias either source register.
  expect_result(IREE_VM_ISA_CORE_OPCODE_INTEGER_ADD_I32, 0, 0, 1, 35, 42);
  expect_result(IREE_VM_ISA_CORE_OPCODE_INTEGER_SUB_I32, 0, 1, 0, 10, -3);
  // Unary destinations may alias their source, including modular INT_MIN.
  expect_result(IREE_VM_ISA_CORE_OPCODE_INTEGER_NEG_I32, 0, 0, 0, INT32_MIN,
                INT32_MIN);
}

TEST(VMBytecodeModuleTest, PreservesResultsOnIntegerDivisionFailure) {
  const auto expect_failure = [](uint8_t opcode, int16_t divisor,
                                 int32_t dividend,
                                 iree_status_code_t expected_code) {
    std::vector<uint8_t> image = BuildOwnershipModuleImage();
    const MutableFunctionImage initializer = FindFunctionImage(&image, 0);
    ASSERT_NE(initializer.row, nullptr);
    constexpr uint32_t kInitializerImmediateOffset = 6;
    initializer.bytecode[kInitializerImmediateOffset + 0] =
        static_cast<uint8_t>(divisor);
    initializer.bytecode[kInitializerImmediateOffset + 1] =
        static_cast<uint8_t>(static_cast<uint16_t>(divisor) >> 8);

    const MutableFunctionImage function = FindFunctionImage(&image, 1);
    ASSERT_NE(function.row, nullptr);
    constexpr uint32_t kArithmeticInstructionOffset = 8;
    function.bytecode[kArithmeticInstructionOffset + 0] = opcode;
    function.bytecode[kArithmeticInstructionOffset + 1] = 0;
    function.bytecode[kArithmeticInstructionOffset + 2] = 0;
    function.bytecode[kArithmeticInstructionOffset + 3] = 1;

    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    iree_vm_module_t* module = nullptr;
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment, IREE_SV("integer_division_failure"),
        {iree_make_const_byte_span(image.data(), image.size()),
         iree_allocator_null()},
        iree_allocator_system(), &module));
    iree_vm_environment_free(environment);

    iree_vm_program_t* program = nullptr;
    IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                          iree_allocator_system(), &program));
    iree_vm_invocation_t* invocation = nullptr;
    IREE_ASSERT_OK(iree_vm_invocation_allocate(
        kInvocationStorageSize, iree_allocator_system(), &invocation));
    iree_vm_process_t* process = nullptr;
    IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process));
    iree_vm_function_t run = iree_vm_function_null();
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        process, IREE_SV("integer_division_failure"), IREE_SV("run"), &run));

    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(dividend)};
    iree_vm_variant_t results[] = {
        iree_vm_variant_from_i32(123),
        iree_vm_variant_from_i64(456),
    };
    const std::array<uint8_t, sizeof(results)> untouched_results = [&] {
      std::array<uint8_t, sizeof(results)> bytes;
      std::memcpy(bytes.data(), results, sizeof(results));
      return bytes;
    }();
    IREE_EXPECT_STATUS_IS(
        expected_code,
        iree_vm_invoke(invocation, run,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results)));
    EXPECT_EQ(std::memcmp(results, untouched_results.data(), sizeof(results)),
              0);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

    iree_vm_invocation_free(invocation);
    iree_vm_process_release(process);
    iree_vm_program_release(program);
    iree_vm_module_release(module);
  };

  expect_failure(IREE_VM_ISA_CORE_OPCODE_INTEGER_DIV_U32, 0, 1,
                 IREE_STATUS_INVALID_ARGUMENT);
  expect_failure(IREE_VM_ISA_CORE_OPCODE_INTEGER_DIV_S32, -1, INT32_MIN,
                 IREE_STATUS_OUT_OF_RANGE);
}

TEST(VMBytecodeModuleTest, ExecutesFunctionGlobalsAndOverflowABI) {
  std::vector<uint8_t> image = BuildFunctionStateModuleImage();
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("function_state"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  iree_vm_environment_free(environment);

  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  iree_vm_export_t callback_export = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_export(module, IREE_SV("callback"),
                                              &callback_export));
  iree_vm_function_ref_t callback = iree_vm_function_ref_null();
  IREE_ASSERT_OK(
      iree_vm_function_ref_from_export(program, callback_export, &callback));

  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));
  iree_vm_variant_t process_arguments[] = {
      iree_vm_variant_from_function_ref(callback),
      iree_vm_variant_from_function_ref(callback),
  };
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(
      program, invocation, iree_vm_variant_span_from_array(process_arguments),
      iree_allocator_system(), &process));

  iree_vm_function_t read = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("function_state"), IREE_SV("read"), &read));
  iree_vm_variant_t read_results[2] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, read, iree_vm_variant_span_empty(),
                                iree_vm_variant_span_from_array(read_results)));
  for (iree_vm_variant_t result : read_results) {
    iree_vm_function_ref_t actual = iree_vm_function_ref_null();
    IREE_ASSERT_OK(iree_vm_function_ref_from_variant(result, &actual));
    EXPECT_EQ(actual.program_bits, callback.program_bits);
    EXPECT_EQ(actual.target_bits, callback.target_bits);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(read_results));

  iree_vm_function_t overflow = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("function_state"), IREE_SV("overflow"), &overflow));
  std::array<iree_vm_variant_t, 18> overflow_arguments;
  std::array<iree_vm_variant_t, 18> overflow_results = {};
  for (iree_vm_variant_t& argument : overflow_arguments) {
    argument = iree_vm_variant_from_function_ref(callback);
  }
  IREE_ASSERT_OK(
      iree_vm_invoke(invocation, overflow,
                     iree_vm_variant_span_from_ptr(overflow_arguments.data(),
                                                   overflow_arguments.size()),
                     iree_vm_variant_span_from_ptr(overflow_results.data(),
                                                   overflow_results.size())));
  for (iree_vm_variant_t result : overflow_results) {
    iree_vm_function_ref_t actual = iree_vm_function_ref_null();
    IREE_ASSERT_OK(iree_vm_function_ref_from_variant(result, &actual));
    EXPECT_EQ(actual.program_bits, callback.program_bits);
    EXPECT_EQ(actual.target_bits, callback.target_bits);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_ptr(
      overflow_results.data(), overflow_results.size()));

  iree_vm_function_t bad_store = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("function_state"), IREE_SV("bad_store"), &bad_store));
  iree_vm_invocation_t* failure_invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &failure_invocation));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_invoke(failure_invocation, bad_store,
                                       iree_vm_variant_span_empty(),
                                       iree_vm_variant_span_empty()));
  iree_vm_invocation_free(failure_invocation);

  read_results[0] = iree_vm_variant_empty();
  read_results[1] = iree_vm_variant_empty();
  IREE_ASSERT_OK(iree_vm_invoke(invocation, read, iree_vm_variant_span_empty(),
                                iree_vm_variant_span_from_array(read_results)));
  iree_vm_function_ref_t mutable_after_failure = iree_vm_function_ref_null();
  IREE_ASSERT_OK(iree_vm_function_ref_from_variant(read_results[1],
                                                   &mutable_after_failure));
  EXPECT_EQ(mutable_after_failure.program_bits, callback.program_bits);
  EXPECT_EQ(mutable_after_failure.target_bits, callback.target_bits);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(read_results));

  iree_vm_function_t bad_result = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("function_state"), IREE_SV("bad_result"), &bad_result));
  const iree_vm_variant_t sentinel = iree_vm_variant_from_i32(123);
  iree_vm_variant_t bad_results[] = {sentinel};
  failure_invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &failure_invocation));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_invoke(failure_invocation, bad_result,
                     iree_vm_variant_span_empty(),
                     iree_vm_variant_span_from_array(bad_results)));
  iree_vm_invocation_free(failure_invocation);
  EXPECT_EQ(bad_results[0].payload, sentinel.payload);
  EXPECT_EQ(bad_results[0].metadata, sentinel.metadata);

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
}

TEST(VMBytecodeModuleTest, ExecutesCompleteLaunchConfiguration) {
  std::vector<uint8_t> image = BuildLaunchConfigModuleImage();
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("launch"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  iree_vm_environment_free(environment);

  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                        iree_vm_variant_span_empty(),
                                        iree_allocator_system(), &process));
  iree_vm_function_t decode = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(process, IREE_SV("launch"),
                                                 IREE_SV("decode"), &decode));

  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(64),
      iree_vm_variant_from_bf16_bits(0x4000),
  };
  iree_vm_variant_t results[11] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, decode,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  const int64_t expected[] = {128, 1, 1, 1, 1, 1, 1, 1, 1, 32, 256};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    int64_t value = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[i], &value));
    EXPECT_EQ(value, expected[i]);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

  arguments[0] = iree_vm_variant_from_i32(64);
  arguments[1] = iree_vm_variant_from_bf16_bits(0x7FC1);
  for (iree_vm_variant_t& result : results) {
    result = iree_vm_variant_from_i64(0x1234);
  }
  const std::array<iree_vm_variant_t, 11> untouched_results = {
      results[0], results[1], results[2], results[3], results[4],  results[5],
      results[6], results[7], results[8], results[9], results[10],
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invoke(invocation, decode,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));
  EXPECT_EQ(std::memcmp(results, untouched_results.data(), sizeof(results)), 0);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
}

TEST(VMBytecodeModuleTest, ExecutesExactValueOverflowTransactionally) {
  std::vector<uint8_t> image = BuildValueOverflowModuleImage();
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("overflow"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  iree_vm_environment_free(environment);

  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                        iree_vm_variant_span_empty(),
                                        iree_allocator_system(), &process));

  const std::array<int64_t, 18> expected = {
      0,
      1,
      -1,
      INT64_MIN,
      INT64_MAX,
      17,
      -29,
      INT64_C(0x0123456789ABCDEF),
      -INT64_C(0x0123456789ABCDEF),
      101,
      202,
      303,
      404,
      505,
      606,
      707,
      INT64_C(0x13579BDF2468ACE0),
      -INT64_C(0x13579BDF2468ACE0),
  };
  std::array<iree_vm_variant_t, 18> arguments = {};
  for (iree_host_size_t i = 0; i < arguments.size(); ++i) {
    arguments[i] = iree_vm_variant_from_i64(expected[i]);
  }
  std::array<iree_vm_variant_t, 18> results = {};
  iree_vm_function_t identity = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("overflow"), IREE_SV("identity"), &identity));
  IREE_ASSERT_OK(iree_vm_invoke(
      invocation, identity,
      iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
      iree_vm_variant_span_from_ptr(results.data(), results.size())));
  for (iree_host_size_t i = 0; i < results.size(); ++i) {
    int64_t actual = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[i], &actual));
    EXPECT_EQ(actual, expected[i]);
  }
  iree_vm_variant_span_reset(
      iree_vm_variant_span_from_ptr(results.data(), results.size()));

  for (iree_host_size_t i = 0; i < arguments.size(); ++i) {
    arguments[i] = iree_vm_variant_from_i64(expected[i]);
    results[i] = iree_vm_variant_from_i64(INT64_C(0x123456789ABCDEF));
  }
  arguments[0] = iree_vm_variant_from_i64(INT64_C(0x7FC00000));
  const std::array<iree_vm_variant_t, 18> untouched_results = results;
  iree_vm_function_t fail_after_store = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(process, IREE_SV("overflow"),
                                                 IREE_SV("fail_after_store"),
                                                 &fail_after_store));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invoke(
          invocation, fail_after_store,
          iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
          iree_vm_variant_span_from_ptr(results.data(), results.size())));
  EXPECT_EQ(
      std::memcmp(results.data(), untouched_results.data(), sizeof(results)),
      0);
  iree_vm_variant_span_reset(
      iree_vm_variant_span_from_ptr(results.data(), results.size()));

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
