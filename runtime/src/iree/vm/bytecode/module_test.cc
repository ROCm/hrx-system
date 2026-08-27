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

void CountBufferRelease(void* user_data, iree_byte_span_t storage) {
  (void)storage;
  ++*static_cast<int*>(user_data);
}

struct RefStateTestObject {
  // Required offset-zero VM-visible ownership prefix.
  iree_vm_ref_object_t ref_object;
  // Incremented when the final owner is released.
  int* destruction_count;
};

struct RefStateTestTypes {
  // Type deliberately incompatible with vm.buffer.
  iree_vm_ref_type_t object;
};

extern const iree_vm_ref_type_table_t kRefStateTestTypeTable;

void DestroyRefStateTestObject(void* object) {
  auto* test_object = static_cast<RefStateTestObject*>(object);
  ++*test_object->destruction_count;
}

const iree_vm_ref_type_descriptor_t kRefStateTestObjectType = {
    DestroyRefStateTestObject,
    &kRefStateTestTypeTable,
    IREE_SV("object"),
};
const RefStateTestTypes kRefStateTestTypes = {
    &kRefStateTestObjectType,
};
const iree_vm_ref_type_table_t kRefStateTestTypeTable = {
    sizeof(kRefStateTestTypeTable),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("zz_test"),
    {&kRefStateTestTypes, 1},
};

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
  // Core 0.0 deliberately leaves opcode 0x0F unassigned.
  functions[bytecode_offset + 4] = 0x0F;

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
      IREE_STATUS_INVALID_ARGUMENT,
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

TEST(VMBytecodeModuleTest, RejectsMalformedDirectControlFlow) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const auto expect_rejected = [&](const auto& mutate) {
    std::vector<uint8_t> image = BuildSwitchInspectionModuleImage();
    MutableFunctionImage function = FindFunctionImage(&image, 0);
    ASSERT_NE(function.row, nullptr);
    ASSERT_NE(function.bytecode, nullptr);
    mutate(function);
    iree_vm_module_t* module =
        reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_control"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
  };

  constexpr uint32_t kSwitchOffset = sizeof(iree_vm_isa_control_block_record_t);
  constexpr uint32_t kSecondBlockOffset =
      kSwitchOffset + sizeof(iree_vm_isa_control_switch_record_t);
  constexpr uint32_t kReturnOffset =
      kSecondBlockOffset + sizeof(iree_vm_isa_control_block_record_t);

  // Block and return delimiters require their canonical zero padding.
  expect_rejected(
      [](MutableFunctionImage function) { function.bytecode[1] = 1; });
  expect_rejected([&](MutableFunctionImage function) {
    function.bytecode[kReturnOffset + 1] = 1;
  });

  // The declared count must exactly match decoded control.block records.
  expect_rejected(
      [](MutableFunctionImage function) { function.row->block_count_u32 = 3; });

  // Function-owned switch targets must name decoded block boundaries.
  expect_rejected([](MutableFunctionImage function) {
    auto* switch_targets =
        reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
            function.row + 1);
    switch_targets[0] = 2;
  });

  // A direct target cannot masquerade as a block by landing on an opcode-like
  // byte in the middle of a decoded record.
  expect_rejected([&](MutableFunctionImage function) {
    function.row->block_count_u32 = 1;
    auto* switch_targets =
        reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
            function.row + 1);
    switch_targets[0] = 0;
    const iree_vm_isa_constant_i32_record_t constant = {
        IREE_VM_ISA_CORE_OPCODE_CONSTANT_I32, 0, 0,
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK};
    std::memcpy(function.bytecode + kSwitchOffset, &constant, sizeof(constant));
    const iree_vm_isa_control_branch_s16_record_t branch = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S16, 0, -2};
    std::memcpy(function.bytecode + kSecondBlockOffset, &branch,
                sizeof(branch));
  });

  // Wide conditional branches carry their own canonical padding field.
  expect_rejected([&](MutableFunctionImage function) {
    const iree_vm_isa_control_branch_if_s32_record_t branch = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S32, 0, 1, 0};
    std::memcpy(function.bytecode + kSwitchOffset, &branch, sizeof(branch));
  });

  // Yield uses the same verified direct-target domain as a wide branch while
  // requiring all reserved bytes to remain zero.
  expect_rejected([&](MutableFunctionImage function) {
    const iree_vm_isa_control_yield_s32_record_t yield = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_YIELD_S32, {1, 0, 0}, 0};
    std::memcpy(function.bytecode + kSwitchOffset, &yield, sizeof(yield));
  });
  expect_rejected([&](MutableFunctionImage function) {
    const iree_vm_isa_control_yield_s32_record_t yield = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_YIELD_S32, {0, 0, 0}, INT32_MAX};
    std::memcpy(function.bytecode + kSwitchOffset, &yield, sizeof(yield));
  });
  expect_rejected([&](MutableFunctionImage function) {
    const iree_vm_isa_control_yield_s32_record_t yield = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_YIELD_S32, {0, 0, 0}, 1};
    std::memcpy(function.bytecode + kSwitchOffset, &yield, sizeof(yield));
  });

  expect_rejected([&](MutableFunctionImage function) {
    function.row->block_count_u32 = 1;
    auto* switch_targets =
        reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
            function.row + 1);
    switch_targets[0] = 0;
    const iree_vm_isa_control_branch_s16_record_t branch = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S16, 0, INT16_MAX};
    std::memcpy(function.bytecode + kSecondBlockOffset, &branch,
                sizeof(branch));
  });

  expect_rejected([&](MutableFunctionImage function) {
    function.row->block_count_u32 = 1;
    auto* switch_targets =
        reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
            function.row + 1);
    switch_targets[0] = 0;
    const iree_vm_isa_control_branch_s16_record_t branch = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S16, 1, -4};
    std::memcpy(function.bytecode + kSecondBlockOffset, &branch,
                sizeof(branch));
  });

  expect_rejected([&](MutableFunctionImage function) {
    function.row->block_count_u32 = 1;
    auto* switch_targets =
        reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
            function.row + 1);
    switch_targets[0] = 0;
    const iree_vm_isa_control_branch_s32_record_t branch = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S32, {1, 0, 0}, -5};
    std::memcpy(function.bytecode + kSecondBlockOffset, &branch,
                sizeof(branch));
  });

  expect_rejected([&](MutableFunctionImage function) {
    function.row->block_count_u32 = 1;
    auto* switch_targets =
        reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
            function.row + 1);
    switch_targets[0] = 0;
    const iree_vm_isa_control_branch_if_s16_record_t branch = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S16, 1, -4};
    std::memcpy(function.bytecode + kSecondBlockOffset, &branch,
                sizeof(branch));
  });

  // Conditional control records require a sequential default record.
  expect_rejected([&](MutableFunctionImage function) {
    const iree_vm_isa_control_branch_if_s16_record_t branch = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S16, 0, -5};
    std::memcpy(function.bytecode + kReturnOffset, &branch, sizeof(branch));
  });

  expect_rejected([&](MutableFunctionImage function) {
    function.row->block_count_u32 = 1;
    auto* switch_targets =
        reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
            function.row + 1);
    switch_targets[0] = 0;
    const iree_vm_isa_constant_i32_record_t constant = {
        IREE_VM_ISA_CORE_OPCODE_CONSTANT_I32, 0, 0, 0};
    std::memcpy(function.bytecode + kSwitchOffset, &constant, sizeof(constant));
    const iree_vm_isa_control_switch_record_t switch_record = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_SWITCH, 0, 1, 0};
    std::memcpy(function.bytecode + kSecondBlockOffset, &switch_record,
                sizeof(switch_record));
  });

  // The instruction-local switch slice must fit its function-owned table.
  expect_rejected([&](MutableFunctionImage function) {
    auto* switch_record =
        reinterpret_cast<iree_vm_isa_control_switch_record_t*>(
            function.bytecode + kSwitchOffset);
    switch_record->target_count_u16 = 2;
  });
  expect_rejected([&](MutableFunctionImage function) {
    auto* switch_record =
        reinterpret_cast<iree_vm_isa_control_switch_record_t*>(
            function.bytecode + kSwitchOffset);
    switch_record->target_base_u32 = 2;
  });

  // The selector must name a declared value register.
  expect_rejected([&](MutableFunctionImage function) {
    auto* switch_record =
        reinterpret_cast<iree_vm_isa_control_switch_record_t*>(
            function.bytecode + kSwitchOffset);
    switch_record->selector_v8 = 1;
  });

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, RejectsMalformedProgramFailure) {
  constexpr uint32_t kAssertOffset = sizeof(iree_vm_isa_control_block_record_t);
  constexpr uint32_t kBranchOffset =
      kAssertOffset + sizeof(iree_vm_isa_control_assert_record_t);
  constexpr uint32_t kSecondBlockOffset =
      kBranchOffset + sizeof(iree_vm_isa_control_branch_s16_record_t);
  constexpr uint32_t kFinalOffset =
      kSecondBlockOffset + sizeof(iree_vm_isa_control_block_record_t);

  const auto build_assert_image = [&]() {
    std::vector<uint8_t> image = BuildSwitchInspectionModuleImage();
    MutableFunctionImage function = FindFunctionImage(&image, 0);
    const iree_vm_isa_control_assert_record_t assert_record = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_ASSERT, 0, {0, 0}};
    std::memcpy(function.bytecode + kAssertOffset, &assert_record,
                sizeof(assert_record));
    const iree_vm_isa_control_branch_s16_record_t branch_record = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S16, 0, 0};
    std::memcpy(function.bytecode + kBranchOffset, &branch_record,
                sizeof(branch_record));
    return image;
  };
  const auto build_fail_image = [&]() {
    std::vector<uint8_t> image = BuildSwitchInspectionModuleImage();
    MutableFunctionImage function = FindFunctionImage(&image, 0);
    function.row->ref_register_count_u16 = 1;
    const iree_vm_isa_control_fail_record_t fail_record = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_FAIL,
        IREE_VM_ISA_CONTROL_STATUS_INVALID_ARGUMENT, 0, 0};
    std::memcpy(function.bytecode + kFinalOffset, &fail_record,
                sizeof(fail_record));
    return image;
  };

  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const auto expect_accepted = [&](std::vector<uint8_t> image) {
    iree_vm_module_t* module = nullptr;
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment, IREE_SV("valid_failure"),
        {iree_make_const_byte_span(image.data(), image.size()),
         iree_allocator_null()},
        iree_allocator_system(), &module));
    iree_vm_module_release(module);
  };
  const auto expect_rejected = [&](std::vector<uint8_t> image,
                                   const auto& mutate) {
    MutableFunctionImage function = FindFunctionImage(&image, 0);
    mutate(function);
    iree_vm_module_t* module =
        reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_failure"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
  };

  expect_accepted(build_assert_image());
  expect_accepted(build_fail_image());

  expect_rejected(build_assert_image(), [&](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_assert_record_t*>(
        function.bytecode + kAssertOffset);
    record->condition_v8 = 1;
  });
  expect_rejected(build_assert_image(), [&](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_assert_record_t*>(
        function.bytecode + kAssertOffset);
    record->zero_padding_u8[0] = 1;
  });
  expect_rejected(build_assert_image(), [&](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_assert_record_t*>(
        function.bytecode + kAssertOffset);
    record->zero_padding_u8[1] = 1;
  });
  expect_rejected(build_assert_image(), [&](MutableFunctionImage function) {
    const iree_vm_isa_control_assert_record_t record = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_ASSERT, 0, {0, 0}};
    std::memcpy(function.bytecode + kFinalOffset, &record, sizeof(record));
  });

  for (uint8_t status :
       {uint8_t{0}, uint8_t{17}, static_cast<uint8_t>(UINT8_MAX)}) {
    expect_rejected(build_fail_image(), [&](MutableFunctionImage function) {
      auto* record = reinterpret_cast<iree_vm_isa_control_fail_record_t*>(
          function.bytecode + kFinalOffset);
      record->status_u8 = status;
    });
  }
  expect_rejected(build_fail_image(), [&](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_fail_record_t*>(
        function.bytecode + kFinalOffset);
    record->message_r8_nullable = 1;
  });
  expect_rejected(build_fail_image(), [&](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_fail_record_t*>(
        function.bytecode + kFinalOffset);
    record->zero_padding_u8 = 1;
  });

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

  const uint8_t binary_value_opcodes[] = {
      IREE_VM_ISA_CORE_OPCODE_INTEGER_DIV_S32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_DIV_S64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_DIV_U32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_DIV_U64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_REM_S32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_REM_S64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_REM_U32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_REM_U64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_SHIFT_LEFT_I32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_SHIFT_LEFT_I64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_SHIFT_RIGHT_S32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_SHIFT_RIGHT_S64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_SHIFT_RIGHT_U32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_SHIFT_RIGHT_U64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_ROTATE_LEFT_I32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_ROTATE_LEFT_I64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_ROTATE_RIGHT_I32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_ROTATE_RIGHT_I64,
  };
  for (uint8_t opcode : binary_value_opcodes) {
    for (uint8_t field_offset = 1; field_offset < 4; ++field_offset) {
      std::vector<uint8_t> image = BuildScalarStateModuleImage();
      const MutableFunctionImage function = FindFunctionImage(&image, 1);
      ASSERT_NE(function.row, nullptr);
      constexpr uint32_t kFirstBodyInstructionOffset = 4;
      function.bytecode[kFirstBodyInstructionOffset] = opcode;
      function.bytecode[kFirstBodyInstructionOffset + field_offset] = 6;
      expect_rejected(image);
    }
  }

  const uint8_t bit_count_opcodes[] = {
      IREE_VM_ISA_CORE_OPCODE_INTEGER_COUNT_LEADING_ZEROS_I32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_COUNT_LEADING_ZEROS_I64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_COUNT_TRAILING_ZEROS_I32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_COUNT_TRAILING_ZEROS_I64,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_POPCOUNT_I32,
      IREE_VM_ISA_CORE_OPCODE_INTEGER_POPCOUNT_I64,
  };
  for (uint8_t opcode : bit_count_opcodes) {
    for (uint8_t field_offset = 1; field_offset < 3; ++field_offset) {
      std::vector<uint8_t> image = BuildScalarStateModuleImage();
      const MutableFunctionImage function = FindFunctionImage(&image, 1);
      ASSERT_NE(function.row, nullptr);
      constexpr uint32_t kFirstBodyInstructionOffset = 4;
      function.bytecode[kFirstBodyInstructionOffset] = opcode;
      function.bytecode[kFirstBodyInstructionOffset + field_offset] = 6;
      expect_rejected(image);
    }

    std::vector<uint8_t> image = BuildScalarStateModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 1);
    ASSERT_NE(function.row, nullptr);
    constexpr uint32_t kFirstBodyInstructionOffset = 4;
    function.bytecode[kFirstBodyInstructionOffset] = opcode;
    function.bytecode[kFirstBodyInstructionOffset + 3] = 1;
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

TEST(VMBytecodeModuleTest, RejectsMalformedValueABIInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const auto expect_rejected = [&](const auto& mutate) {
    std::vector<uint8_t> image = BuildValueOverflowModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 0);
    ASSERT_NE(function.row, nullptr);
    mutate(function);

    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_value_abi"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  expect_rejected([](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_value_abi_argument_load_record_t*>(
            function.bytecode + sizeof(iree_vm_isa_control_block_record_t));
    record->dst_v8 = 18;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_value_abi_argument_load_record_t*>(
            function.bytecode + sizeof(iree_vm_isa_control_block_record_t));
    record->slot_u16 = 2;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_value_abi_result_store_record_t*>(
            function.bytecode + sizeof(iree_vm_isa_control_block_record_t) +
            2 * sizeof(iree_vm_isa_value_abi_argument_load_record_t));
    record->src_v8 = 18;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_value_abi_result_store_record_t*>(
            function.bytecode + sizeof(iree_vm_isa_control_block_record_t) +
            2 * sizeof(iree_vm_isa_value_abi_argument_load_record_t));
    record->slot_u16 = 2;
  });

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, RejectsMalformedRefABIAndGlobalInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(
      environment, &kRefStateTestTypeTable));
  const auto expect_rejected = [&](const std::vector<uint8_t>& image) {
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_ref_state"),
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
      // Each ref ABI record validates its register and overflow slot.
      {2, 5, 18},
      {2, 6, 1},
      {2, 9, 18},
      {2, 10, 1},
      {2, 13, 18},
      {2, 14, 1},
      // Global records validate both registers and immutable partitions.
      {0, 5, 2},
      {0, 6, 1},
      {0, 9, 2},
      {0, 10, 0},
      {0, 10, 2},
      {1, 5, 2},
      {1, 6, 1},
      {1, 9, 2},
      {1, 10, 0},
      {1, 10, 2},
  };
  for (const ByteMutation& mutation : mutations) {
    std::vector<uint8_t> image = BuildRefStateModuleImage();
    const MutableFunctionImage function =
        FindFunctionImage(&image, mutation.function_ordinal);
    ASSERT_NE(function.row, nullptr);
    ASSERT_LT(mutation.byte_offset, function.row->bytecode_length_u32);
    function.bytecode[mutation.byte_offset] = mutation.value;
    expect_rejected(image);
  }

  // Overflow ref arguments/results and borrowed direct arguments share the
  // caller's local ref storage. Both contributions must fit.
  const auto expect_call_packet_rejected = [&](uint32_t local_ref_count,
                                               uint16_t direct_ref_move_mask) {
    std::vector<uint8_t> image = BuildRefStateModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 0);
    ASSERT_NE(function.row, nullptr);
    function.row->flags_u16 |= IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL;
    function.row->ref_register_count_u16 = 16;
    function.row->local_ref_count_u32 = local_ref_count;
    const iree_vm_isa_control_call_record_t call = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_CALL,
        IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL, 2, direct_ref_move_mask, 0};
    std::memcpy(function.bytecode + sizeof(iree_vm_isa_control_block_record_t),
                &call, sizeof(call));
    expect_rejected(image);
  };
  // Two argument and result overflow slots do not fit in one local slot.
  expect_call_packet_rejected(1, UINT16_MAX);
  // Sixteen borrowed direct arguments require scratch beyond the two overflow
  // slots that otherwise fit.
  expect_call_packet_rejected(2, 0);

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

TEST(VMBytecodeModuleTest, RejectsMalformedIntegerBitstreamInstructions) {
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
            environment, IREE_SV("malformed_bitstream"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  iree_vm_isa_integer_bitstream_pack_record_t pack = {};
  pack.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_INTEGER_BITSTREAM_PACK;
  pack.result_base_v8 = 1;
  pack.source_base_v8 = 0;
  pack.field_width_u8 = 8;
  pack.source_count_u8 = 1;
  pack.result_count_u8 = 1;
  pack.source_width_u8 = 8;
  pack.result_width_u8 = 8;

  pack.source_count_u8 = 0;
  expect_rejected(pack);
  pack.source_count_u8 = 1;
  pack.result_count_u8 = 0;
  expect_rejected(pack);
  pack.result_count_u8 = 1;
  pack.source_base_v8 = 2;
  expect_rejected(pack);
  pack.source_base_v8 = 0;
  pack.result_base_v8 = 2;
  expect_rejected(pack);
  pack.result_base_v8 = 1;
  pack.field_width_u8 = 0;
  expect_rejected(pack);
  pack.field_width_u8 = 65;
  expect_rejected(pack);
  pack.field_width_u8 = 8;
  pack.source_width_u8 = 24;
  expect_rejected(pack);
  pack.source_width_u8 = 8;
  pack.result_width_u8 = 24;
  expect_rejected(pack);
  pack.result_width_u8 = 8;
  pack.field_width_u8 = 4;
  expect_rejected(pack);
  pack.field_width_u8 = 9;
  expect_rejected(pack);
  pack.field_width_u8 = 64;
  pack.source_count_u8 = 2;
  pack.result_count_u8 = 2;
  pack.source_width_u8 = 64;
  pack.result_width_u8 = 64;
  pack.result_base_v8 = 0;
  expect_rejected(pack);

  iree_vm_isa_integer_bitstream_unpack_u_record_t unpack = {};
  unpack.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_INTEGER_BITSTREAM_UNPACK_U;
  unpack.result_base_v8 = 1;
  unpack.source_base_v8 = 0;
  unpack.field_width_u8 = 4;
  unpack.source_count_u8 = 1;
  unpack.result_count_u8 = 1;
  unpack.source_width_u8 = 8;
  unpack.result_width_u8 = 8;
  expect_rejected(unpack);
  unpack.field_width_u8 = 9;
  expect_rejected(unpack);

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, RejectsMalformedStackInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const auto expect_rejected = [&](const auto& record) {
    std::vector<uint8_t> image = BuildScalarStateModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 1);
    ASSERT_NE(function.row, nullptr);
    function.row->local_byte_length_u16 = 64;
    function.row->ref_register_count_u16 = 2;

    uint8_t* cursor = function.bytecode;
    const uint8_t* const end =
        function.bytecode + function.row->bytecode_length_u32;
    const iree_vm_isa_control_block_record_t block = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}};
    std::memcpy(cursor, &block, sizeof(block));
    cursor += sizeof(block);
    ASSERT_LE(sizeof(record), static_cast<size_t>(end - cursor));
    std::memcpy(cursor, &record, sizeof(record));
    cursor += sizeof(record);
    const iree_vm_isa_constant_zero_record_t zero = {
        IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 0, 0};
    while (cursor < end - sizeof(iree_vm_isa_control_return_record_t)) {
      std::memcpy(cursor, &zero, sizeof(zero));
      cursor += sizeof(zero);
    }
    const iree_vm_isa_control_return_record_t return_record = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}};
    std::memcpy(cursor, &return_record, sizeof(return_record));
    cursor += sizeof(return_record);
    ASSERT_EQ(cursor, end);

    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_stack"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  iree_vm_isa_stack_load_record_t load = {};
  load.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_LOAD;
  load.format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X1;
  load.zero_padding_u8[0] = 1;
  expect_rejected(load);
  load.zero_padding_u8[0] = 0;
  load.format_u8 = 0x10;
  expect_rejected(load);
  load.format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X1;
  load.base_u16 = 57;
  expect_rejected(load);
  load.base_u16 = 0;
  load.dst_v8 = 6;
  expect_rejected(load);

  iree_vm_isa_stack_store_record_t store = {};
  store.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_STORE;
  store.format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X1;
  store.zero_padding_u16 = 1;
  expect_rejected(store);
  store.zero_padding_u16 = 0;
  store.src_v8 = 6;
  expect_rejected(store);

  iree_vm_isa_stack_load_indexed_record_t load_indexed = {};
  load_indexed.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_LOAD_INDEXED;
  load_indexed.scale_u8 = 1;
  load_indexed.format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X1;
  load_indexed.scale_u8 = 0;
  expect_rejected(load_indexed);
  load_indexed.scale_u8 = 1;
  load_indexed.zero_padding_u8 = 1;
  expect_rejected(load_indexed);
  load_indexed.zero_padding_u8 = 0;
  load_indexed.index_v8 = 6;
  expect_rejected(load_indexed);

  iree_vm_isa_stack_store_indexed_record_t store_indexed = {};
  store_indexed.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_STORE_INDEXED;
  store_indexed.scale_u8 = 1;
  store_indexed.format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X1;
  store_indexed.zero_padding_u8 = 1;
  expect_rejected(store_indexed);
  store_indexed.zero_padding_u8 = 0;
  store_indexed.src_v8 = 6;
  expect_rejected(store_indexed);

  iree_vm_isa_stack_fill_record_t fill = {};
  fill.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_FILL;
  fill.length_u16 = 8;
  fill.pattern_width_u8 = 8;
  fill.zero_padding_u8 = 1;
  expect_rejected(fill);
  fill.zero_padding_u8 = 0;
  fill.pattern_width_u8 = 3;
  expect_rejected(fill);
  fill.pattern_width_u8 = 8;
  fill.target_base_u16 = 60;
  expect_rejected(fill);
  fill.target_base_u16 = 0;
  fill.pattern_v8 = 6;
  expect_rejected(fill);

  iree_vm_isa_stack_copy_record_t copy = {};
  copy.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_COPY;
  copy.source_u16 = 8;
  copy.length_u16 = 8;
  copy.zero_padding_u8 = 1;
  expect_rejected(copy);
  copy.zero_padding_u8 = 0;
  copy.target_u16 = 60;
  expect_rejected(copy);
  copy.target_u16 = 0;
  copy.source_u16 = 60;
  expect_rejected(copy);

  iree_vm_isa_stack_compare_record_t compare = {};
  compare.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_COMPARE;
  compare.rhs_u16 = 8;
  compare.length_u16 = 8;
  compare.dst_v8 = 6;
  expect_rejected(compare);
  compare.dst_v8 = 0;
  compare.lhs_u16 = 60;
  expect_rejected(compare);
  compare.lhs_u16 = 0;
  compare.rhs_u16 = 60;
  expect_rejected(compare);

  iree_vm_isa_stack_copy_rodata_record_t copy_rodata = {};
  copy_rodata.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_COPY_RODATA;
  copy_rodata.length_u16 = 4;
  copy_rodata.zero_padding_u8 = 1;
  expect_rejected(copy_rodata);
  copy_rodata.zero_padding_u8 = 0;
  copy_rodata.rodata_u16 = 1;
  expect_rejected(copy_rodata);
  copy_rodata.rodata_u16 = 0;
  copy_rodata.source_offset_u32 = 8;
  expect_rejected(copy_rodata);
  copy_rodata.source_offset_u32 = 0;
  copy_rodata.target_u16 = 62;
  expect_rejected(copy_rodata);

  iree_vm_isa_stack_copy_from_buffer_record_t copy_from_buffer = {};
  copy_from_buffer.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_COPY_FROM_BUFFER;
  copy_from_buffer.length_u16 = 4;
  copy_from_buffer.zero_padding_u8 = 1;
  expect_rejected(copy_from_buffer);
  copy_from_buffer.zero_padding_u8 = 0;
  copy_from_buffer.target_u16 = 62;
  expect_rejected(copy_from_buffer);
  copy_from_buffer.target_u16 = 0;
  copy_from_buffer.buffer_r8 = 2;
  expect_rejected(copy_from_buffer);
  copy_from_buffer.buffer_r8 = 0;
  copy_from_buffer.source_offset_v8 = 6;
  expect_rejected(copy_from_buffer);

  iree_vm_isa_stack_copy_to_buffer_record_t copy_to_buffer = {};
  copy_to_buffer.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_COPY_TO_BUFFER;
  copy_to_buffer.length_u16 = 4;
  copy_to_buffer.zero_padding_u8 = 1;
  expect_rejected(copy_to_buffer);
  copy_to_buffer.zero_padding_u8 = 0;
  copy_to_buffer.source_u16 = 62;
  expect_rejected(copy_to_buffer);
  copy_to_buffer.source_u16 = 0;
  copy_to_buffer.buffer_r8 = 2;
  expect_rejected(copy_to_buffer);
  copy_to_buffer.buffer_r8 = 0;
  copy_to_buffer.target_offset_v8 = 6;
  expect_rejected(copy_to_buffer);

  iree_vm_isa_stack_const_s16_i32_record_t const_i32 = {};
  const_i32.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_CONST_S16_I32;
  const_i32.count_u16 = 2;
  const_i32.zero_padding_u8 = 1;
  expect_rejected(const_i32);
  const_i32.zero_padding_u8 = 0;
  const_i32.target_u16 = 2;
  expect_rejected(const_i32);
  const_i32.target_u16 = 60;
  expect_rejected(const_i32);

  iree_vm_isa_stack_const_s16_i64_record_t const_i64 = {};
  const_i64.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_CONST_S16_I64;
  const_i64.count_u16 = 2;
  const_i64.zero_padding_u8 = 1;
  expect_rejected(const_i64);
  const_i64.zero_padding_u8 = 0;
  const_i64.target_u16 = 4;
  expect_rejected(const_i64);
  const_i64.target_u16 = 56;
  expect_rejected(const_i64);

  iree_vm_isa_stack_pack_i32_u16_x8_record_t pack_i32 = {};
  pack_i32.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I32_U16_X8;
  pack_i32.zero_padding_u8 = 1;
  expect_rejected(pack_i32);
  pack_i32.zero_padding_u8 = 0;
  pack_i32.target_u16 = 2;
  expect_rejected(pack_i32);
  pack_i32.target_u16 = 36;
  expect_rejected(pack_i32);

  iree_vm_isa_stack_pack_i64_u32_x8_record_t pack_i64 = {};
  pack_i64.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I64_U32_X8;
  pack_i64.zero_padding_u8 = 1;
  expect_rejected(pack_i64);
  pack_i64.zero_padding_u8 = 0;
  pack_i64.target_u16 = 4;
  expect_rejected(pack_i64);
  pack_i64.target_u16 = 8;
  expect_rejected(pack_i64);

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, RejectsMalformedRefInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const auto expect_rejected = [&](const auto& record) {
    static_assert(sizeof(record) == 4, "ref records must remain four bytes");
    std::vector<uint8_t> image = BuildOwnershipModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 1);
    ASSERT_NE(function.row, nullptr);
    function.row->ref_register_count_u16 = 3;
    function.row->local_ref_count_u32 = 2;
    constexpr uint32_t kInstructionOffset = 8;
    std::memcpy(function.bytecode + kInstructionOffset, &record,
                sizeof(record));

    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_ref"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  expect_rejected(
      iree_vm_isa_ref_null_record_t{IREE_VM_ISA_CORE_OPCODE_REF_NULL, 3, 0});
  expect_rejected(iree_vm_isa_ref_compare_null_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_COMPARE_NULL, 0, 0, 1});
  expect_rejected(iree_vm_isa_ref_compare_eq_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_COMPARE_EQ, 0, 0, 3});
  expect_rejected(iree_vm_isa_ref_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_RETAIN, 0, 3, 0});
  expect_rejected(
      iree_vm_isa_ref_move_record_t{IREE_VM_ISA_CORE_OPCODE_REF_MOVE, 1, 1, 0});
  expect_rejected(iree_vm_isa_ref_discard_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_DISCARD, 0, 1});
  expect_rejected(iree_vm_isa_ref_stack_load_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_LOAD_RETAIN, 0, 2});
  expect_rejected(iree_vm_isa_ref_stack_load_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_LOAD_MOVE, 3, 0});
  expect_rejected(iree_vm_isa_ref_stack_store_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_RETAIN, 0, 2});
  expect_rejected(iree_vm_isa_ref_stack_store_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_MOVE, 3, 0});
  expect_rejected(iree_vm_isa_ref_stack_discard_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_DISCARD, 1, 0});

  std::vector<uint8_t> image = BuildOwnershipModuleImage();
  const MutableFunctionImage function = FindFunctionImage(&image, 1);
  ASSERT_NE(function.row, nullptr);
  function.row->local_ref_count_u32 = (uint32_t)UINT16_MAX + 2u;
  iree_vm_module_t* module = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_vm_bytecode_module_create(
          environment, IREE_SV("ref_slot_overflow"),
          {iree_make_const_byte_span(image.data(), image.size()),
           iree_allocator_null()},
          iree_allocator_system(), &module));
  EXPECT_EQ(module, nullptr);
  iree_vm_module_release(module);

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, RejectsMalformedFunctionInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));

  const auto expect_image_status = [&](const std::vector<uint8_t>& image,
                                       iree_status_code_t expected_code) {
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        expected_code,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_function"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  std::vector<uint8_t> valid_image = BuildFunctionModuleImage();
  iree_vm_module_t* valid_module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("function"),
      {iree_make_const_byte_span(valid_image.data(), valid_image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &valid_module));
  iree_vm_module_release(valid_module);

  const auto expect_rejected = [&](const auto& mutate) {
    std::vector<uint8_t> image = BuildFunctionModuleImage();
    const MutableFunctionImage function = FindFunctionImage(&image, 0);
    ASSERT_NE(function.row, nullptr);
    mutate(function);
    expect_image_status(image, IREE_STATUS_INVALID_ARGUMENT);
  };

  constexpr uint32_t kNullOffset = 4;
  constexpr uint32_t kCompareNullOffset = 8;
  constexpr uint32_t kCopyOffset = 12;
  constexpr uint32_t kLocalAddressOffset = 16;
  constexpr uint32_t kImportAddressOffset = 24;
  constexpr uint32_t kImportResolvedOffset = 32;
  constexpr uint32_t kStackStoreOffset = 36;
  constexpr uint32_t kStackLoadOffset = 40;
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_null_record_t*>(
        function.bytecode + kNullOffset);
    record->dst_f8 = 5;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_compare_null_record_t*>(
        function.bytecode + kCompareNullOffset);
    record->dst_v8 = 2;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_copy_record_t*>(
        function.bytecode + kCopyOffset);
    record->src_f8 = 5;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kLocalAddressOffset);
    record->target_kind_u8 = UINT8_MAX;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kLocalAddressOffset);
    record->zero_padding_u8 = 1;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kLocalAddressOffset);
    record->target_ordinal_u16 = 2;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kLocalAddressOffset);
    record->callable_type_ordinal_u16 = 2;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kLocalAddressOffset);
    record->callable_type_ordinal_u16 = 1;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kImportAddressOffset);
    record->target_kind_u8 = IREE_VM_ISA_CONTROL_CALL_TARGET_REQUIRED_IMPORT;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kImportAddressOffset);
    record->target_ordinal_u16 = 1;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kImportAddressOffset);
    record->callable_type_ordinal_u16 = 1;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_import_resolved_record_t*>(
        function.bytecode + kImportResolvedOffset);
    record->import_ordinal_u16 = 1;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_stack_store_record_t*>(
        function.bytecode + kStackStoreOffset);
    record->local_ordinal_u16 = 1;
  });
  expect_rejected([](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_func_stack_load_record_t*>(
        function.bytecode + kStackLoadOffset);
    record->dst_f8 = 5;
  });

  // func.import.resolved accepts only optional imports, even when a required
  // import with the same symbol and callable contract is otherwise valid.
  {
    std::vector<uint8_t> image = BuildFunctionModuleImage();
    auto* imports = reinterpret_cast<iree_vm_bytecode_v0_imports_header_t*>(
        FindSectionPayload(&image, IREE_VM_BYTECODE_SECTION_IMPORTS));
    ASSERT_NE(imports, nullptr);
    auto* groups =
        reinterpret_cast<iree_vm_bytecode_v0_import_group_row_t*>(imports + 1);
    auto* entries = reinterpret_cast<iree_vm_bytecode_v0_import_entry_row_t*>(
        groups + imports->group_count_u32);
    entries[0].flags_u16 = 0;
    const MutableFunctionImage function = FindFunctionImage(&image, 0);
    ASSERT_NE(function.row, nullptr);
    auto* address = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
        function.bytecode + kImportAddressOffset);
    address->target_kind_u8 = IREE_VM_ISA_CONTROL_CALL_TARGET_REQUIRED_IMPORT;

    expect_image_status(image, IREE_STATUS_INVALID_ARGUMENT);
  }

  // Direct function-local ordinals cannot address beyond their u16 domain.
  {
    std::vector<uint8_t> image = BuildFunctionModuleImage();
    MutableFunctionImage function = FindFunctionImage(&image, 0);
    ASSERT_NE(function.row, nullptr);
    function.row->local_function_count_u32 = (uint32_t)UINT16_MAX + 2u;
    expect_image_status(image, IREE_STATUS_UNIMPLEMENTED);
  }

  iree_vm_environment_free(environment);
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

TEST(VMBytecodeModuleTest, RejectsMalformedFunctionStateInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const auto expect_rejected = [&](std::vector<uint8_t>& image) {
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_function_state"),
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
      {1, 5, 2},    // Immutable store source register.
      {1, 6, 1},    // Immutable store global partition.
      {1, 10, 0},   // Mutable store global partition.
      {2, 5, 2},    // Immutable load destination register.
      {2, 6, 1},    // Immutable load global partition.
      {2, 10, 0},   // Mutable load global partition.
      {4, 5, 18},   // Function argument overflow destination register.
      {4, 6, 2},    // Function argument overflow slot.
      {4, 13, 18},  // Function result overflow source register.
      {4, 14, 2},   // Function result overflow slot.
  };
  for (const ByteMutation& mutation : mutations) {
    std::vector<uint8_t> image = BuildFunctionStateModuleImage();
    const MutableFunctionImage function =
        FindFunctionImage(&image, mutation.function_ordinal);
    ASSERT_NE(function.row, nullptr);
    ASSERT_LT(mutation.byte_offset, function.row->bytecode_length_u32);
    function.bytecode[mutation.byte_offset] = mutation.value;
    expect_rejected(image);
  }

  std::vector<uint8_t> image = BuildFunctionStateModuleImage();
  auto* globals = reinterpret_cast<iree_vm_bytecode_v0_globals_header_t*>(
      FindSectionPayload(&image, IREE_VM_BYTECODE_SECTION_GLOBALS));
  ASSERT_NE(globals, nullptr);
  globals->immutable_function_count_u32 = 3;
  expect_rejected(image);

  image = BuildFunctionStateModuleImage();
  globals = reinterpret_cast<iree_vm_bytecode_v0_globals_header_t*>(
      FindSectionPayload(&image, IREE_VM_BYTECODE_SECTION_GLOBALS));
  ASSERT_NE(globals, nullptr);
  auto* descriptors =
      reinterpret_cast<iree_vm_bytecode_v0_global_function_descriptor_row_t*>(
          globals + 1);
  descriptors[0].callable_type_ordinal_u16 = 7;
  expect_rejected(image);

  image = BuildFunctionStateModuleImage();
  globals = reinterpret_cast<iree_vm_bytecode_v0_globals_header_t*>(
      FindSectionPayload(&image, IREE_VM_BYTECODE_SECTION_GLOBALS));
  ASSERT_NE(globals, nullptr);
  descriptors =
      reinterpret_cast<iree_vm_bytecode_v0_global_function_descriptor_row_t*>(
          globals + 1);
  descriptors[0].flags_u16 = 2;
  expect_rejected(image);

  // Function overflow arguments and results share the caller's function-local
  // storage after the direct register prefix.
  image = BuildFunctionStateModuleImage();
  MutableFunctionImage function = FindFunctionImage(&image, 1);
  ASSERT_NE(function.row, nullptr);
  function.row->flags_u16 |= IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL;
  function.row->function_register_count_u16 = 16;
  // The 18-argument/result target needs four overflow slots.
  function.row->local_function_count_u32 = 3;
  const iree_vm_isa_control_call_record_t call = {
      IREE_VM_ISA_CORE_OPCODE_CONTROL_CALL,
      IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL, 4, 0, 0};
  std::memcpy(function.bytecode + sizeof(iree_vm_isa_control_block_record_t),
              &call, sizeof(call));
  expect_rejected(image);

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, RejectsMalformedConversionInstructions) {
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
            environment, IREE_SV("malformed_conversion"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  iree_vm_isa_conversion_integer_record_t integer = {
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_INTEGER, 0, 1,
      IREE_VM_ISA_INTEGER_CONVERT_I64_TO_I32 + 1};
  expect_rejected(integer);
  integer.selector_u8 = IREE_VM_ISA_INTEGER_CONVERT_S8_TO_I32;
  integer.dst_v8 = 2;
  expect_rejected(integer);
  integer.dst_v8 = 0;
  integer.src_v8 = 2;
  expect_rejected(integer);

  expect_rejected(iree_vm_isa_conversion_float_extend_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_EXTEND, 0, 1,
      IREE_VM_ISA_FLOAT_EXTEND_BF16_TO_F32 + 1});
  expect_rejected(iree_vm_isa_conversion_float_truncate_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_TRUNCATE, 0, 1,
      IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_BF16 + 1});
  expect_rejected(iree_vm_isa_conversion_float_width_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_WIDTH, 0, 1,
      IREE_VM_ISA_FLOAT_WIDTH_F64_TO_F32 + 1});
  expect_rejected(iree_vm_isa_conversion_integer_to_float_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_INTEGER_TO_FLOAT, 0, 1,
      IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_BF16 + 1});
  expect_rejected(iree_vm_isa_conversion_float_to_integer_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_TO_INTEGER, 0, 1,
      IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_U64 + 1});

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, RejectsMalformedFloatInstructions) {
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
            environment, IREE_SV("malformed_float"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  iree_vm_isa_float_minmax_f32_record_t minmax = {};
  minmax.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_MINMAX_F32;
  minmax.dst_v8 = 0;
  minmax.lhs_v8 = 0;
  minmax.rhs_v8 = 1;
  minmax.selector_u8 = IREE_VM_ISA_FLOAT_MINMAX_MAXNUM + 1;
  expect_rejected(minmax);
  minmax.selector_u8 = IREE_VM_ISA_FLOAT_MINMAX_MINIMUM;
  minmax.zero_padding_u8[0] = 1;
  expect_rejected(minmax);
  minmax.zero_padding_u8[0] = 0;
  minmax.lhs_v8 = 2;
  expect_rejected(minmax);

  iree_vm_isa_float_compare_f64_record_t compare = {};
  compare.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_COMPARE_F64;
  compare.dst_v8 = 0;
  compare.lhs_v8 = 0;
  compare.rhs_v8 = 1;
  compare.predicate_u8 = IREE_VM_ISA_FLOAT_COMPARE_UNO + 1;
  expect_rejected(compare);
  compare.predicate_u8 = IREE_VM_ISA_FLOAT_COMPARE_OEQ;
  compare.zero_padding_u8[2] = 1;
  expect_rejected(compare);
  compare.zero_padding_u8[2] = 0;
  compare.dst_v8 = 2;
  expect_rejected(compare);

  iree_vm_isa_float_classify_f32_record_t classify = {};
  classify.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_CLASSIFY_F32;
  classify.dst_v8 = 0;
  classify.src_v8 = 1;
  classify.selector_u8 = IREE_VM_ISA_FLOAT_CLASSIFY_ISFINITE + 1;
  expect_rejected(classify);
  classify.selector_u8 = IREE_VM_ISA_FLOAT_CLASSIFY_ISNAN;
  classify.src_v8 = 2;
  expect_rejected(classify);

  iree_vm_isa_float_clamp_f64_record_t clamp = {};
  clamp.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_CLAMP_F64;
  clamp.dst_v8 = 0;
  clamp.value_v8 = 0;
  clamp.lower_v8 = 0;
  clamp.upper_v8 = 1;
  clamp.mode_u8 = IREE_VM_ISA_FLOAT_CLAMP_IEEE + 1;
  expect_rejected(clamp);
  clamp.mode_u8 = IREE_VM_ISA_FLOAT_CLAMP_ORDERED;
  clamp.zero_padding_u16 = 1;
  expect_rejected(clamp);
  clamp.zero_padding_u16 = 0;
  clamp.upper_v8 = 2;
  expect_rejected(clamp);

  iree_vm_isa_float_math_unary_f32_record_t unary_f32 = {};
  unary_f32.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_MATH_UNARY_F32;
  unary_f32.dst_v8 = 0;
  unary_f32.src_v8 = 1;
  unary_f32.selector_u8 = IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH + 1;
  expect_rejected(unary_f32);

  iree_vm_isa_float_math_unary_f64_record_t unary_f64 = {};
  unary_f64.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_MATH_UNARY_F64;
  unary_f64.dst_v8 = 0;
  unary_f64.src_v8 = 2;
  unary_f64.selector_u8 = IREE_VM_ISA_FLOAT_MATH_UNARY_EXP;
  expect_rejected(unary_f64);

  iree_vm_isa_float_math_binary_f32_record_t binary_f32 = {};
  binary_f32.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_MATH_BINARY_F32;
  binary_f32.dst_v8 = 0;
  binary_f32.lhs_v8 = 0;
  binary_f32.rhs_v8 = 1;
  binary_f32.selector_u8 = IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC + 1;
  expect_rejected(binary_f32);

  iree_vm_isa_float_math_binary_f64_record_t binary_f64 = {};
  binary_f64.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_MATH_BINARY_F64;
  binary_f64.dst_v8 = 0;
  binary_f64.lhs_v8 = 0;
  binary_f64.rhs_v8 = 1;
  binary_f64.selector_u8 = IREE_VM_ISA_FLOAT_MATH_BINARY_POW;
  binary_f64.zero_padding_u8[1] = 1;
  expect_rejected(binary_f64);
  binary_f64.zero_padding_u8[1] = 0;
  binary_f64.rhs_v8 = 2;
  expect_rejected(binary_f64);

  iree_vm_isa_float_math_ternary_f32_record_t ternary_f32 = {};
  ternary_f32.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_MATH_TERNARY_F32;
  ternary_f32.dst_v8 = 0;
  ternary_f32.a_v8 = 0;
  ternary_f32.b_v8 = 0;
  ternary_f32.c_v8 = 1;
  ternary_f32.selector_u8 = IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA + 1;
  expect_rejected(ternary_f32);

  iree_vm_isa_float_math_ternary_f64_record_t ternary_f64 = {};
  ternary_f64.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FLOAT_MATH_TERNARY_F64;
  ternary_f64.dst_v8 = 0;
  ternary_f64.a_v8 = 0;
  ternary_f64.b_v8 = 0;
  ternary_f64.c_v8 = 1;
  ternary_f64.selector_u8 = IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA;
  ternary_f64.zero_padding_u16 = 1;
  expect_rejected(ternary_f64);
  ternary_f64.zero_padding_u16 = 0;
  ternary_f64.a_v8 = 2;
  expect_rejected(ternary_f64);

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
