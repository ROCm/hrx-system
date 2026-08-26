// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module.h"

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
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "iree/vm/process.h"

namespace iree::vm::bytecode::testing {
namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

struct CountingAllocator {
  // Allocator performing the actual memory operations.
  iree_allocator_t delegate;
  // Number of allocation-like commands forwarded.
  iree_host_size_t allocation_count;
  // Number of free commands forwarded.
  iree_host_size_t free_count;
};

iree_status_t CountingAllocatorControl(void* self,
                                       iree_allocator_command_t command,
                                       const void* params, void** inout_ptr) {
  auto* allocator = static_cast<CountingAllocator*>(self);
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      ++allocator->allocation_count;
      break;
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++allocator->free_count;
      break;
    default:
      break;
  }
  return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                 inout_ptr);
}

iree_allocator_t MakeCountingAllocator(CountingAllocator* allocator) {
  return iree_allocator_t{allocator, CountingAllocatorControl};
}

std::string_view ToStringView(iree_string_view_t value) {
  return std::string_view(value.data, value.size);
}

TEST(VMBytecodeModuleTest, RejectsBeforeTakingImageStorageOwnership) {
  std::vector<uint8_t> image = BuildOwnershipModuleImage();
  uint8_t* functions =
      FindSectionPayload(&image, IREE_VM_BYTECODE_SECTION_FUNCTIONS);
  ASSERT_NE(functions, nullptr);
  const size_t bytecode_offset =
      sizeof(iree_vm_bytecode_v0_functions_header_t) +
      2 * sizeof(iree_vm_bytecode_v0_function_row_t);
  functions[bytecode_offset + 4] = IREE_VM_ISA_CORE_OPCODE_INTEGER_DIV_S32;

  CountingAllocator image_allocator = {iree_allocator_system(), 0, 0};
  void* owned_image = nullptr;
  IREE_ASSERT_OK(iree_allocator_clone(
      MakeCountingAllocator(&image_allocator),
      iree_make_const_byte_span(image.data(), image.size()), &owned_image));
  ASSERT_EQ(image_allocator.allocation_count, 1u);

  CountingAllocator module_allocator = {iree_allocator_system(), 0, 0};
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_vm_bytecode_module_create(
          environment, IREE_SV("ownership"),
          {iree_make_const_byte_span(owned_image, image.size()),
           MakeCountingAllocator(&image_allocator)},
          MakeCountingAllocator(&module_allocator), &module));
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(module_allocator.allocation_count, 0u);
  EXPECT_EQ(image_allocator.free_count, 0u);

  iree_allocator_free(MakeCountingAllocator(&image_allocator), owned_image);
  EXPECT_EQ(image_allocator.free_count, 1u);
  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest,
     RunsInitializationReflectionAndEscapedRodataLifetime) {
  std::vector<uint8_t> image = BuildOwnershipModuleImage();
  CountingAllocator image_allocator = {iree_allocator_system(), 0, 0};
  void* owned_image = nullptr;
  IREE_ASSERT_OK(iree_allocator_clone(
      MakeCountingAllocator(&image_allocator),
      iree_make_const_byte_span(image.data(), image.size()), &owned_image));

  CountingAllocator module_allocator = {iree_allocator_system(), 0, 0};
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("ownership"),
      {iree_make_const_byte_span(owned_image, image.size()),
       MakeCountingAllocator(&image_allocator)},
      MakeCountingAllocator(&module_allocator), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(module_allocator.allocation_count, 1u);
  EXPECT_EQ(image_allocator.free_count, 0u);
  iree_vm_environment_free(environment);

  EXPECT_EQ(ToStringView(iree_vm_module_name(module)), "ownership");
  EXPECT_EQ(iree_vm_module_export_count(module), 2u);
  EXPECT_EQ(iree_vm_module_function_count(module), 2u);
  EXPECT_EQ(iree_vm_module_ref_type_count(module), 1u);
  iree_vm_ref_type_t buffer_type = nullptr;
  IREE_ASSERT_OK(iree_vm_module_ref_type_by_ordinal(module, 0, &buffer_type));
  iree_vm_ref_types_t vm_types = {buffer_type};

  iree_vm_export_t run_export = {};
  IREE_ASSERT_OK(
      iree_vm_module_lookup_export(module, IREE_SV("run"), &run_export));
  iree_host_size_t required_description_size = 0;
  IREE_ASSERT_OK(iree_vm_export_query_description(
      run_export, iree_byte_span_empty(), &required_description_size, nullptr));
  alignas(max_align_t) std::array<uint8_t, 1024> description_storage = {};
  ASSERT_LE(required_description_size, description_storage.size());
  iree_vm_export_description_t description = {};
  IREE_ASSERT_OK(iree_vm_export_query_description(
      run_export,
      iree_make_byte_span(description_storage.data(),
                          required_description_size),
      &required_description_size, &description));
  EXPECT_EQ(ToStringView(description.name), "run");
  EXPECT_EQ(ToStringView(description.documentation),
            "Adds the process seed and returns image rodata.");
  EXPECT_EQ(ToStringView(description.authored_type),
            "(i32) -> (i32, !vm.ref<vm, buffer>)");
  ASSERT_EQ(description.arguments.count, 1u);
  ASSERT_EQ(description.results.count, 2u);
  EXPECT_EQ(ToStringView(description.arguments.data[0].name), "value");
  EXPECT_EQ(ToStringView(description.results.data[0].name), "sum");
  EXPECT_EQ(ToStringView(description.results.data[1].name), "payload");
  EXPECT_EQ(description.arguments.data[0].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_SCALAR);
  EXPECT_EQ(description.arguments.data[0].type.value.scalar,
            IREE_VM_SCALAR_TYPE_I32);
  EXPECT_EQ(description.results.data[1].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_REF);
  EXPECT_EQ(description.results.data[1].type.value.ref, buffer_type);

  bool found = false;
  iree_vm_metadata_value_t metadata_value = {};
  IREE_ASSERT_OK(iree_vm_module_try_lookup_metadata(
      module, IREE_SV("model.kind"), &found, &metadata_value));
  ASSERT_TRUE(found);
  iree_string_view_t metadata_string = iree_string_view_empty();
  IREE_ASSERT_OK(iree_vm_string_view_from_metadata_value(metadata_value,
                                                         &metadata_string));
  EXPECT_EQ(ToStringView(metadata_string), "ownership");
  IREE_ASSERT_OK(iree_vm_export_try_lookup_metadata(
      run_export, IREE_SV("result.note"), &found, &metadata_value));
  ASSERT_TRUE(found);
  IREE_ASSERT_OK(iree_vm_string_view_from_metadata_value(metadata_value,
                                                         &metadata_string));
  EXPECT_EQ(ToStringView(metadata_string), "stable");

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
  IREE_ASSERT_OK(iree_vm_process_lookup_function(process, IREE_SV("ownership"),
                                                 IREE_SV("run"), &run));

  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(35)};
  iree_vm_variant_t results[2] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, run,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  int32_t sum = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &sum));
  EXPECT_EQ(sum, 42);
  iree_vm_buffer_t* escaped_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_move(&vm_types, &results[1],
                                                      &escaped_buffer));
  ASSERT_NE(escaped_buffer, nullptr);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
  EXPECT_EQ(module_allocator.free_count, 0u);
  EXPECT_EQ(image_allocator.free_count, 0u);

  iree_const_byte_span_t payload = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      escaped_buffer, 0, iree_vm_buffer_length(escaped_buffer), &payload));
  EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(payload.data),
                             payload.data_length),
            "loom-vm-v1");
  iree_vm_buffer_release(escaped_buffer);
  EXPECT_EQ(module_allocator.free_count, 1u);
  EXPECT_EQ(image_allocator.free_count, 1u);
}

TEST(VMBytecodeModuleTest,
     InspectionReflectsTypesWithoutProvidersAndCannotLink) {
  std::vector<uint8_t> image = BuildHALInspectionModuleImage();
  CountingAllocator image_allocator = {iree_allocator_system(), 0, 0};
  void* owned_image = nullptr;
  IREE_ASSERT_OK(iree_allocator_clone(
      MakeCountingAllocator(&image_allocator),
      iree_make_const_byte_span(image.data(), image.size()), &owned_image));

  CountingAllocator module_allocator = {iree_allocator_system(), 0, 0};
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create_for_inspection(
      IREE_SV("hal_inspection"),
      {iree_make_const_byte_span(owned_image, image.size()),
       MakeCountingAllocator(&image_allocator)},
      MakeCountingAllocator(&module_allocator), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(module_allocator.allocation_count, 1u);
  EXPECT_EQ(image_allocator.free_count, 0u);

  EXPECT_EQ(ToStringView(iree_vm_module_name(module)), "hal_inspection");
  EXPECT_EQ(iree_vm_module_ref_type_count(module), 1u);
  EXPECT_EQ(iree_vm_module_export_count(module), 1u);
  EXPECT_EQ(iree_vm_module_function_count(module), 1u);

  iree_vm_ref_type_t device_group_type = nullptr;
  IREE_ASSERT_OK(
      iree_vm_module_ref_type_by_ordinal(module, 0, &device_group_type));
  ASSERT_NE(device_group_type, nullptr);
  const iree_vm_ref_type_key_t type_key =
      iree_vm_ref_type_key(device_group_type);
  EXPECT_EQ(ToStringView(type_key.namespace_name), "hal");
  EXPECT_EQ(ToStringView(type_key.type_name), "device_group");

  iree_vm_export_t device_count_export = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_export(module, IREE_SV("device_count"),
                                              &device_count_export));
  iree_host_size_t description_size = 0;
  IREE_ASSERT_OK(iree_vm_export_query_description(
      device_count_export, iree_byte_span_empty(), &description_size, NULL));
  alignas(max_align_t) std::array<uint8_t, 256> description_storage = {};
  ASSERT_LE(description_size, description_storage.size());
  iree_vm_export_description_t description = {};
  IREE_ASSERT_OK(iree_vm_export_query_description(
      device_count_export,
      iree_make_byte_span(description_storage.data(), description_size),
      &description_size, &description));
  ASSERT_EQ(description.arguments.count, 1u);
  ASSERT_EQ(description.results.count, 1u);
  EXPECT_EQ(description.arguments.data[0].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_REF);
  EXPECT_EQ(description.arguments.data[0].type.value.ref, device_group_type);
  EXPECT_EQ(description.results.data[0].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_SCALAR);
  EXPECT_EQ(description.results.data[0].type.value.scalar,
            IREE_VM_SCALAR_TYPE_I64);
  EXPECT_TRUE(ToStringView(description.documentation).empty());
  EXPECT_TRUE(ToStringView(description.authored_type).empty());

  iree_vm_program_t* program =
      reinterpret_cast<iree_vm_program_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_program_create({module, iree_vm_module_span_empty()},
                             iree_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);

  iree_vm_module_release(module);
  EXPECT_EQ(module_allocator.free_count, 1u);
  EXPECT_EQ(image_allocator.free_count, 1u);
}

TEST(VMBytecodeModuleTest, InspectionAndExecutionShareStructuralDiagnostics) {
  std::vector<uint8_t> image = BuildOwnershipModuleImage();
  image[0] ^= 0xFF;

  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* executable_module =
      reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  iree_status_t executable_status = iree_vm_bytecode_module_create(
      environment, IREE_SV("malformed"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &executable_module);
  EXPECT_EQ(executable_module, nullptr);

  iree_vm_module_t* inspection_module =
      reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  iree_status_t inspection_status =
      iree_vm_bytecode_module_create_for_inspection(
          IREE_SV("malformed"),
          {iree_make_const_byte_span(image.data(), image.size()),
           iree_allocator_null()},
          iree_allocator_system(), &inspection_module);
  EXPECT_EQ(inspection_module, nullptr);

  EXPECT_EQ(iree_status_code(executable_status),
            iree_status_code(inspection_status));
  EXPECT_EQ(ToStringView(iree_status_message(executable_status)),
            ToStringView(iree_status_message(inspection_status)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, executable_status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, inspection_status);
  iree_vm_environment_free(environment);
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

TEST(VMBytecodeModuleTest, RejectsMalformedScalarStateInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  auto expect_rejected = [&](std::vector<uint8_t>& image) {
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_scalar_state"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  struct ByteMutation {
    // Function-table ordinal containing the byte to replace.
    uint32_t function_ordinal;
    // Byte offset from the function's first instruction.
    uint32_t byte_offset;
    // Replacement byte value.
    uint8_t value;
  };
  const ByteMutation mutations[] = {
      // An 8- or 12-byte record cannot begin in the final four-byte slot.
      {1, 64, IREE_VM_ISA_CORE_OPCODE_CONSTANT_I32},
      {1, 64, IREE_VM_ISA_CORE_OPCODE_CONSTANT_I64},
      {1, 64, IREE_VM_ISA_CORE_OPCODE_VALUE_SELECT},
      // Every reserved byte and register operand is verified at module load.
      {1, 6, 1},
      {1, 10, 1},
      {1, 18, 1},
      {1, 45, 1},
      {1, 5, 6},
      {1, 9, 6},
      {1, 17, 6},
      {1, 29, 6},
      {1, 33, 6},
      {1, 37, 6},
      {1, 41, 6},
      {1, 42, 6},
      {1, 43, 6},
      {1, 44, 6},
      {1, 57, 6},
      {1, 61, 6},
      // Pool and global ordinals are checked against their exact partitions.
      {1, 30, 2},
      {1, 34, 2},
      {1, 38, 2},
      {1, 58, 2},
      {1, 62, 2},
      {1, 36, IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_LOAD},
      {1, 56, IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_STORE},
      // A canonical return is legal only as the final instruction.
      {1, 4, IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN},
      // A valid non-return final record is diagnosed as fallthrough.
      {1, 64, IREE_VM_ISA_CORE_OPCODE_CONSTANT_S16},
  };
  for (const ByteMutation& mutation : mutations) {
    std::vector<uint8_t> image = BuildScalarStateModuleImage();
    const MutableFunctionImage function =
        FindFunctionImage(&image, mutation.function_ordinal);
    ASSERT_NE(function.row, nullptr);
    ASSERT_LT(mutation.byte_offset, function.row->bytecode_length_u32);
    function.bytecode[mutation.byte_offset] = mutation.value;
    expect_rejected(image);
  }

  struct ValueFormatMutation {
    // Opcode selecting the shared value-record verification form.
    uint8_t opcode;
    // Byte offset within the four-byte instruction record to corrupt.
    uint8_t field_offset;
    // Replacement byte value.
    uint8_t value;
  };
  const ValueFormatMutation value_format_mutations[] = {
      // Binary records verify all three value-register ordinals.
      {IREE_VM_ISA_CORE_OPCODE_INTEGER_ADD_I64, 1, 6},
      {IREE_VM_ISA_CORE_OPCODE_INTEGER_ADD_I64, 2, 6},
      {IREE_VM_ISA_CORE_OPCODE_INTEGER_ADD_I64, 3, 6},
      // Unary records verify both registers and require canonical padding.
      {IREE_VM_ISA_CORE_OPCODE_INTEGER_NEG_I64, 1, 6},
      {IREE_VM_ISA_CORE_OPCODE_INTEGER_NEG_I64, 2, 6},
      {IREE_VM_ISA_CORE_OPCODE_INTEGER_NEG_I64, 3, 1},
  };
  for (const ValueFormatMutation& mutation : value_format_mutations) {
    std::vector<uint8_t> image = BuildScalarStateModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 1);
    ASSERT_NE(function.row, nullptr);
    constexpr uint32_t kFirstBodyInstructionOffset = 4;
    function.bytecode[kFirstBodyInstructionOffset] = mutation.opcode;
    function.bytecode[kFirstBodyInstructionOffset + mutation.field_offset] =
        mutation.value;
    expect_rejected(image);
  }

  std::vector<uint8_t> partitioned_image = BuildScalarStateModuleImage();
  auto* globals = reinterpret_cast<iree_vm_bytecode_v0_globals_header_t*>(
      FindSectionPayload(&partitioned_image, IREE_VM_BYTECODE_SECTION_GLOBALS));
  ASSERT_NE(globals, nullptr);
  globals->immutable_value_count_u32 = 1;
  expect_rejected(partitioned_image);

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest,
     RejectsMalformedIntegerComparisonAndAddressingInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const auto expect_rejected = [&](const auto& record) {
    std::vector<uint8_t> image = BuildOwnershipModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 1);
    ASSERT_NE(function.row, nullptr);
    constexpr uint32_t kInstructionOffset = 8;
    ASSERT_LE(kInstructionOffset + sizeof(record),
              function.row->bytecode_length_u32 -
                  sizeof(iree_vm_isa_control_return_record_t));
    std::memcpy(function.bytecode + kInstructionOffset, &record,
                sizeof(record));

    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_integer"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  iree_vm_isa_integer_compare_i32_record_t compare_i32 = {};
  compare_i32.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_INTEGER_COMPARE_I32;
  compare_i32.dst_v8 = 0;
  compare_i32.lhs_v8 = 0;
  compare_i32.rhs_v8 = 1;
  compare_i32.predicate_u8 = IREE_VM_ISA_INTEGER_COMPARE_UGE + 1;
  expect_rejected(compare_i32);
  compare_i32.predicate_u8 = IREE_VM_ISA_INTEGER_COMPARE_EQ;
  compare_i32.zero_padding_u8[0] = 1;
  expect_rejected(compare_i32);
  compare_i32.zero_padding_u8[0] = 0;
  compare_i32.dst_v8 = 2;
  expect_rejected(compare_i32);

  iree_vm_isa_integer_lea_i64_record_t lea_i64 = {};
  lea_i64.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_INTEGER_LEA_I64;
  lea_i64.dst_v8 = 0;
  lea_i64.base_v8 = 0;
  lea_i64.index_v8 = 1;
  lea_i64.zero_padding_u8 = 1;
  expect_rejected(lea_i64);
  lea_i64.zero_padding_u8 = 0;
  lea_i64.index_v8 = 2;
  expect_rejected(lea_i64);

  iree_vm_isa_integer_ceildiv_pow2_u32_record_t ceildiv_u32 = {};
  ceildiv_u32.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_INTEGER_CEILDIV_POW2_U32;
  ceildiv_u32.dst_v8 = 0;
  ceildiv_u32.src_v8 = 1;
  ceildiv_u32.log2_u8 = 32;
  expect_rejected(ceildiv_u32);

  iree_vm_isa_integer_ceildiv_pow2_u64_record_t ceildiv_u64 = {};
  ceildiv_u64.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_INTEGER_CEILDIV_POW2_U64;
  ceildiv_u64.dst_v8 = 0;
  ceildiv_u64.src_v8 = 1;
  ceildiv_u64.log2_u8 = 64;
  expect_rejected(ceildiv_u64);

  iree_vm_environment_free(environment);
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

}  // namespace
}  // namespace iree::vm::bytecode::testing
