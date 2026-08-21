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
#include "iree/vm/bytecode/module_test_data.h"
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

uint8_t* FindSectionPayload(std::vector<uint8_t>* image,
                            uint16_t section_type) {
  auto* header =
      reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(image->data());
  auto* rows = reinterpret_cast<iree_vm_bytecode_v0_section_directory_row_t*>(
      header + 1);
  size_t offset = sizeof(*header) + header->section_count_u16 * sizeof(*rows);
  for (uint16_t i = 0; i < header->section_count_u16; ++i) {
    offset = (offset + IREE_VM_BYTECODE_SECTION_ALIGNMENT - 1) &
             ~(IREE_VM_BYTECODE_SECTION_ALIGNMENT - 1);
    if (rows[i].section_type_u16 == section_type) {
      return image->data() + offset;
    }
    offset += static_cast<size_t>(rows[i].byte_length_u64);
  }
  return nullptr;
}

TEST(VMBytecodeModuleTest, RejectsBeforeTakingImageStorageOwnership) {
  std::vector<uint8_t> image = BuildOwnershipModuleImage();
  uint8_t* functions =
      FindSectionPayload(&image, IREE_VM_BYTECODE_SECTION_FUNCTIONS);
  ASSERT_NE(functions, nullptr);
  const size_t bytecode_offset =
      sizeof(iree_vm_bytecode_v0_functions_header_t) +
      2 * sizeof(iree_vm_bytecode_v0_function_row_t);
  functions[bytecode_offset + 4] = IREE_VM_ISA_CORE_OPCODE_INTEGER_ADD_I64;

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
