// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/execution_testdata.h"
#include "iree/vm/bytecode/image.h"
#include "iree/vm/bytecode/launch_config_testdata.h"
#include "iree/vm/sync.h"
#include "iree/vm/test_allocator.h"

namespace {

using iree::vm::testing::CountingAllocator;

iree_const_byte_span_t GetExecutionFixture() {
  const iree_file_toc_t* files = iree_vm_bytecode_execution_testdata_create();
  return iree_make_const_byte_span(files[0].data, files[0].size);
}

using BytecodeModuleCreateFn = decltype(&iree_vm_bytecode_module_create);

void LoadAndInvokeLaunchConfigFixture(BytecodeModuleCreateFn create_module) {
  const iree_file_toc_t* files =
      iree_vm_bytecode_launch_config_testdata_create();
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(
      create_module(environment, IREE_SV("launch"),
                    {iree_make_const_byte_span(files[0].data, files[0].size),
                     iree_allocator_null()},
                    iree_allocator_system(), &module));
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                        iree_allocator_system(), &program));
  alignas(iree_max_align_t) uint8_t invocation_storage[16 * 1024] = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(invocation_storage, sizeof(invocation_storage)),
      &invocation));
  iree_vm_process_t* process = nullptr;
  IREE_ASSERT_OK(iree_vm_process_create(program, invocation,
                                        iree_vm_variant_span_empty(),
                                        iree_allocator_system(), &process));
  iree_vm_function_t function = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process, IREE_SV("launch"), IREE_SV("launch_config"), &function));

  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(7)};
  iree_vm_variant_t results[1] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation, function,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  int32_t result = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &result));
  EXPECT_EQ(result, -7);

  iree_vm_process_release(process);
  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, LoadsAndInvokesVerifiedLaunchConfigFixture) {
  LoadAndInvokeLaunchConfigFixture(iree_vm_bytecode_module_create);
}

TEST(VMBytecodeModuleTest, LoadsAndInvokesTrustedLaunchConfigFixture) {
  LoadAndInvokeLaunchConfigFixture(iree_vm_bytecode_module_create_trusted);
}

TEST(VMBytecodeModuleTest, TrustedCreationOmitsSemanticVerification) {
  const iree_file_toc_t* files =
      iree_vm_bytecode_launch_config_testdata_create();
  const iree_const_byte_span_t contents =
      iree_make_const_byte_span(files[0].data, files[0].size);
  iree_vm_bytecode_module_plan_t plan;
  IREE_ASSERT_OK(iree_vm_bytecode_module_plan_build(contents, &plan));

  std::vector<uint64_t> storage((contents.data_length + 7) / 8);
  std::memcpy(storage.data(), contents.data, contents.data_length);
  const iree_host_size_t function_offset =
      reinterpret_cast<const uint8_t*>(&plan.layout.functions.rows[1]) -
      contents.data;
  auto* function = reinterpret_cast<iree_vm_bytecode_v0_function_row_t*>(
      reinterpret_cast<uint8_t*>(storage.data()) + function_offset);
  ++function->block_count_u32;
  const iree_vm_bytecode_module_storage_t module_storage = {
      iree_make_const_byte_span(storage.data(), contents.data_length),
      iree_allocator_null(),
  };

  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_bytecode_module_create(
                            environment, IREE_SV("verified"), module_storage,
                            iree_allocator_system(), &module));
  IREE_ASSERT_OK(iree_vm_bytecode_module_create_trusted(
      environment, IREE_SV("trusted"), module_storage, iree_allocator_system(),
      &module));

  iree_vm_module_release(module);
  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleTest, TransfersImageAndSlabToEscapedRodata) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));

  CountingAllocator storage_state;
  const iree_allocator_t storage_allocator = storage_state.allocator();
  void* storage_data = nullptr;
  IREE_ASSERT_OK(iree_allocator_clone(storage_allocator, GetExecutionFixture(),
                                      &storage_data));

  CountingAllocator slab_state;
  const iree_allocator_t slab_allocator = slab_state.allocator();
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("lifetime.test"),
      {iree_make_const_byte_span(storage_data,
                                 GetExecutionFixture().data_length),
       storage_allocator},
      slab_allocator, &module));
  EXPECT_EQ(storage_state.allocation_count(), 1u);
  EXPECT_EQ(storage_state.free_count(), 0u);
  EXPECT_EQ(slab_state.allocation_count(), 1u);
  EXPECT_EQ(slab_state.free_count(), 0u);

  iree_vm_bytecode_image_t* image = iree_vm_bytecode_image_from_module(module);
  ASSERT_EQ(image->layout.rodata.count, 1u);
  iree_vm_buffer_t* escaped_buffer = &image->rodata_roots[0];
  iree_vm_buffer_retain(escaped_buffer);

  iree_vm_module_release(module);
  iree_vm_environment_free(environment);
  EXPECT_EQ(storage_state.free_count(), 0u);
  EXPECT_EQ(slab_state.free_count(), 0u);

  iree_const_byte_span_t contents = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      escaped_buffer, 0, iree_vm_buffer_length(escaped_buffer), &contents));
  ASSERT_EQ(contents.data_length, 10u);
  EXPECT_EQ(std::memcmp(contents.data, "loom-vm-v1", contents.data_length), 0);

  iree_vm_buffer_release(escaped_buffer);
  EXPECT_EQ(storage_state.free_count(), 1u);
  EXPECT_EQ(slab_state.free_count(), 1u);
}

void ExpectInvalidImageStorageRemainsWithCaller(
    BytecodeModuleCreateFn create_module) {
  CountingAllocator storage_state;
  const iree_allocator_t storage_allocator = storage_state.allocator();
  void* storage_data = nullptr;
  IREE_ASSERT_OK(iree_allocator_clone(storage_allocator, GetExecutionFixture(),
                                      &storage_data));
  static_cast<uint8_t*>(storage_data)[0] ^= 1;

  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      create_module(environment, IREE_SV("invalid.test"),
                    {iree_make_const_byte_span(
                         storage_data, GetExecutionFixture().data_length),
                     storage_allocator},
                    iree_allocator_system(), &module));
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(storage_state.free_count(), 0u);

  iree_allocator_free(storage_allocator, storage_data);
  iree_vm_environment_free(environment);
  EXPECT_EQ(storage_state.free_count(), 1u);
}

TEST(VMBytecodeModuleTest, VerifiedFailureLeavesImageStorageWithCaller) {
  ExpectInvalidImageStorageRemainsWithCaller(iree_vm_bytecode_module_create);
}

TEST(VMBytecodeModuleTest, TrustedFailureLeavesImageStorageWithCaller) {
  ExpectInvalidImageStorageRemainsWithCaller(
      iree_vm_bytecode_module_create_trusted);
}

}  // namespace
