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
  alignas(iree_max_align_t) std::array<uint8_t, 1024> description_storage = {};
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

TEST(VMBytecodeModuleTest, PreservesRodataAlignmentInOneModuleAllocation) {
  constexpr iree_host_size_t kRodataAlignment = 64;
  const std::vector<uint8_t> image =
      BuildOwnershipModuleImageWithRodataAlignment(kRodataAlignment);

  std::vector<uint8_t> storage(image.size() + kRodataAlignment);
  const uintptr_t storage_base = reinterpret_cast<uintptr_t>(storage.data());
  iree_host_size_t image_offset =
      (IREE_VM_BYTECODE_IMAGE_ALIGNMENT -
       (storage_base & (IREE_VM_BYTECODE_IMAGE_ALIGNMENT - 1))) &
      (IREE_VM_BYTECODE_IMAGE_ALIGNMENT - 1);
  if (iree_host_ptr_has_alignment(storage.data() + image_offset,
                                  kRodataAlignment)) {
    image_offset += IREE_VM_BYTECODE_IMAGE_ALIGNMENT;
  }
  ASSERT_LE(image_offset + image.size(), storage.size());
  uint8_t* image_data = storage.data() + image_offset;
  ASSERT_TRUE(iree_host_ptr_has_alignment(image_data,
                                          IREE_VM_BYTECODE_IMAGE_ALIGNMENT));
  ASSERT_FALSE(iree_host_ptr_has_alignment(image_data, kRodataAlignment));
  std::memcpy(image_data, image.data(), image.size());

  CountingAllocator module_allocator = {iree_allocator_system(), 0, 0};
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("aligned_rodata"),
      {iree_make_const_byte_span(image_data, image.size()),
       iree_allocator_null()},
      MakeCountingAllocator(&module_allocator), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(module_allocator.allocation_count, 1u);

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
  iree_vm_function_t run = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("aligned_rodata"), IREE_SV("run"), &run));

  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(35)};
  iree_vm_variant_t results[2] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, run,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  iree_vm_buffer_t* escaped_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_move(&vm_types, &results[1],
                                                      &escaped_buffer));
  ASSERT_NE(escaped_buffer, nullptr);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

  iree_vm_invocation_free(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
  iree_vm_environment_free(environment);
  EXPECT_EQ(module_allocator.free_count, 0u);

  iree_const_byte_span_t payload = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      escaped_buffer, 0, iree_vm_buffer_length(escaped_buffer), &payload));
  EXPECT_TRUE(iree_host_ptr_has_alignment(payload.data, kRodataAlignment));
  EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(payload.data),
                             payload.data_length),
            "loom-vm-v1");
  iree_vm_buffer_release(escaped_buffer);
  EXPECT_EQ(module_allocator.free_count, 1u);
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
  alignas(iree_max_align_t) std::array<uint8_t, 256> description_storage = {};
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

TEST(VMBytecodeModuleTest, VerificationDoesNotRequireRefTypeProviders) {
  std::vector<uint8_t> image = BuildBufferModuleImage();
  IREE_ASSERT_OK(iree_vm_bytecode_module_verify(
      iree_make_const_byte_span(image.data(), image.size()),
      iree_allocator_system()));
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

}  // namespace
}  // namespace iree::vm::bytecode::testing
