// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SubmissionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    function_.metadata = &metadata_;
    function_.kernarg = &layout_;
    iree_hal_amd_gpu_capture_pool_initialize(1024, iree_allocator_system(),
                                             &pool_);
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_barrier_profile_initialize({11, 0, 0}, &profile_));
  }
  void TearDown() override {
    iree_hal_amd_gpu_capture_pool_deinitialize(&pool_);
  }
  iree_status_t Prepare(
      iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags = 0,
      iree_const_byte_span_t constants = iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_t bindings = iree_hal_buffer_ref_list_empty()) {
    return iree_hal_amd_gpu_pm4_dispatch_prepare(
        &function_, 65536, config, constants, bindings, flags, &dispatch_);
  }
  // Normalized, trusted image metadata; public launch inputs remain unchecked.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Prepared argument shape, without packing records in geometry-only tests.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {.alignment = 8};
  // Executable-owned function consumed by public dispatch admission.
  iree_hal_amd_gpu_pm4_function_t function_ = {};
  // Admitted geometry output.
  iree_hal_amd_gpu_pm4_dispatch_t dispatch_ = {};
  // Real reusable capture storage.
  iree_hal_amd_gpu_capture_pool_t pool_;
  // Physical cache policy used to capture public barriers.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile_;
};

TEST_F(SubmissionTest, DefaultGeometryUsesRequiredMetadata) {
  auto config = iree_hal_make_static_dispatch_config(7, 3, 2);
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 1u);
  metadata_.has_required_workgroup_size = true;
  metadata_.required_workgroup_size[0] = 32;
  metadata_.required_workgroup_size[1] = 2;
  metadata_.required_workgroup_size[2] = 1;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 32u);
  EXPECT_EQ(dispatch_.workgroup_size[1], 2u);
  EXPECT_EQ(dispatch_.workgroup_count[0], 7u);
  config.workgroup_size[0] = 16;
  config.workgroup_size[1] = 2;
  config.workgroup_size[2] = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(SubmissionTest, ExactItemsAdmitPartialGroupsAndUniformRequirements) {
  iree_hal_dispatch_config_t config = {};
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 4;
  config.workgroup_size[2] = 1;
  config.workitem_count[0] = 65;
  config.workitem_count[1] = 8;
  config.workitem_count[2] = 1;
  const auto flags = IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.workgroup_count[0], 3u);
  EXPECT_EQ(dispatch_.workgroup_count[1], 2u);
  EXPECT_EQ(dispatch_.partial_size[0], 1u);
  EXPECT_EQ(dispatch_.partial_size[1], 4u);
  EXPECT_EQ(dispatch_.partial_size[2], 1u);
  metadata_.uniform_workgroup_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config, flags));
  config.workitem_count[0] = 64;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.partial_size[0], 0u);
  EXPECT_EQ(dispatch_.partial_size[1], 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(SubmissionTest, GeometryAndLdsRespectExecutableLimits) {
  auto config = iree_hal_make_static_dispatch_config(5, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  metadata_.max_flat_workgroup_size = 16;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_flat_workgroup_size = 32;
  metadata_.max_workgroup_count[0] = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_workgroup_count[0] = 5;
  metadata_.group_segment_fixed_size = 32768;
  config.dynamic_workgroup_local_memory = 32768;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.group_segment_byte_length, 65536u);
  ++config.dynamic_workgroup_local_memory;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
  config.dynamic_workgroup_local_memory = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
}

TEST_F(SubmissionTest, UnprovisionedRuntimeServicesFailBeforeCapture) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  function_.launch.requirements =
      IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  function_.launch.requirements = 0;
  layout_.requirements = IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_HOSTCALL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  layout_.requirements = 0;
  metadata_.has_workgroup_cluster_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
}

TEST_F(SubmissionTest, DirectArgumentsRespectDenseAndNativeExtents) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  std::array<uint8_t, 24> constants = {};
  layout_.byte_length = 24;
  layout_.native_byte_length = 12;
  layout_.constant_byte_length = 8;
  IREE_ASSERT_OK(Prepare(config, 0, {constants.data(), 8}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, 0, {constants.data(), 12}));
  const auto native = IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 12}));
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 24}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, native, {constants.data(), 11}));
}

// The executable is an ownership dependency; all capture/refcount logic is
// production code. No native driver is needed to observe its final release.
struct Executable {
  // Refcounted public object borrowed by the capture.
  iree_hal_executable_t base;
  // Final destructor observation, joined before this stack object disappears.
  bool destroyed = false;
  static void Destroy(iree_hal_executable_t* base) {
    reinterpret_cast<Executable*>(base)->destroyed = true;
  }
};

TEST_F(SubmissionTest, CallerAndGpuOwnershipJoinBeforeReturningCapture) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_semaphore_create(
      proactor, 0, 4, iree_allocator_system(), &semaphore));
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  std::array<uint8_t, 16> storage = {};
  bool buffer_destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, storage.size(),
      iree_make_byte_span(storage.data(), storage.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &buffer_destroyed},
      iree_allocator_system(), &buffer));
  layout_.byte_length = 16;
  layout_.native_byte_length = 12;
  layout_.constant_byte_length = 4;
  layout_.binding_count = 1;
  uint64_t wait_value = 3, signal_value = 4;
  auto reference = iree_hal_make_buffer_ref(buffer, 4, 8);
  uint32_t constant = 0x12345678;
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &pool_, {1, &semaphore, &wait_value}, {1, &semaphore, &signal_value},
      &executable.base, &function_, {},
      iree_make_const_byte_span(&constant, sizeof(constant)), {1, &reference},
      &profile_, nullptr, 0, &capture));
  ASSERT_EQ(capture->allocation_size, pool_.block_size);
  const auto tail = iree_hal_amd_gpu_pm4_submission_tail_storage(capture);
  ASSERT_GT(tail.data_length, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(tail.data) % iree_max_align_t, 0u);
  std::memset(tail.data, 0xA5, tail.data_length);
  constant = 0;
  reference = {};
  wait_value = signal_value = 99;
  EXPECT_EQ(capture->waits[0].value, 3u);
  EXPECT_EQ(capture->signals.payload_values[0], 4u);
  EXPECT_EQ(capture->bindings.values[0].offset, 4u);
  EXPECT_EQ(capture->bindings.values[0].length, 8u);
  uint32_t copied = 0;
  std::memcpy(&copied, capture->constants.data, sizeof(copied));
  EXPECT_EQ(copied, 0x12345678u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(capture->signals.payload_values) %
                alignof(uint64_t),
            0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(capture->bindings.values) %
                alignof(iree_hal_buffer_ref_t),
            0u);
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_executable_release(&executable.base);
  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(semaphore);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_FALSE(executable.destroyed);
  EXPECT_FALSE(buffer_destroyed);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_TRUE(executable.destroyed);
  EXPECT_TRUE(buffer_destroyed);
  iree_async_proactor_release(proactor);
}

TEST_F(SubmissionTest, OversizedCaptureUsesCompleteOwnedStorage) {
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  layout_.byte_length = layout_.native_byte_length = 4096;
  std::vector<uint8_t> constants(4096, 0x35);
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &pool_, {}, {}, &executable.base, &function_, {},
      iree_make_const_byte_span(constants.data(), constants.size()), {},
      &profile_, &barriers, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
      &capture));
  EXPECT_EQ(capture->allocation_size, sizeof(*capture) + constants.size());
  EXPECT_EQ(iree_hal_amd_gpu_pm4_submission_tail_storage(capture).data_length,
            0u);
  EXPECT_EQ(capture->before.cache_control, 0u);
  EXPECT_EQ(capture->after.cache_control, 0u);
  constants.assign(constants.size(), 0xAB);
  for (iree_host_size_t i = 0; i < capture->constants.data_length; ++i) {
    ASSERT_EQ(capture->constants.data[i], 0x35);
  }
  iree_hal_executable_release(&executable.base);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_TRUE(executable.destroyed);
}

TEST_F(SubmissionTest, TailStoragePreservesArgumentsAtBlockBoundaries) {
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  const iree_host_size_t capacity =
      pool_.block_size - sizeof(iree_hal_amd_gpu_pm4_submission_t);
  const std::array<iree_host_size_t, 8> sizes = {
      0, 1, 15, 16, 17, capacity - 1, capacity, capacity + 1};
  for (iree_host_size_t size : sizes) {
    layout_.byte_length = layout_.native_byte_length = size;
    std::vector<uint8_t> constants(size, 0x35);
    iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
        &pool_, {}, {}, &executable.base, &function_, {},
        iree_make_const_byte_span(constants.data(), constants.size()), {},
        &profile_, nullptr, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
        &capture));
    const auto tail = iree_hal_amd_gpu_pm4_submission_tail_storage(capture);
    if (size >= capacity - 1) {
      EXPECT_EQ(tail.data_length, 0u);
    } else {
      ASSERT_GT(tail.data_length, 0u);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(tail.data) % iree_max_align_t, 0u);
      std::memset(tail.data, 0xA5, tail.data_length);
    }
    for (iree_host_size_t i = 0; i < size; ++i) {
      EXPECT_EQ(capture->constants.data[i], 0x35);
    }
    EXPECT_EQ(capture->allocation_size,
              size <= capacity ? pool_.block_size : sizeof(*capture) + size);
    iree_hal_amd_gpu_pm4_submission_release(capture);
  }
  iree_hal_executable_release(&executable.base);
  EXPECT_TRUE(executable.destroyed);
}

TEST_F(SubmissionTest, CaptureOverflowRetainsNoInputs) {
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_gpu_pm4_submission_create(
          &pool_, {IREE_HOST_SIZE_MAX, nullptr, nullptr}, {}, nullptr, nullptr,
          {}, {}, {}, &profile_, nullptr, 0, &capture));
  EXPECT_EQ(capture, nullptr);
}

TEST_F(SubmissionTest, PreparedAddressChecksDomainUsageAndCommitment) {
  const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
  const iree_hal_buffer_binding_layout_t binding_layout = {
      sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
  iree_hal_memory_contract_t* contract = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_contract_create(
      this, 2, &binding_layout, iree_allocator_system(), &contract));
  contract->scopes[1].interfaces = 1u
                                   << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  contract->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
  contract->scopes[1].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  iree_hal_queue_family_t family = {
      .memory = {.domain = this, .queue_scope_id = 0},
  };
  iree_hal_buffer_native_binding_t binding = {.device_address = 0x1000};
  iree_hal_buffer_t buffer = {};
  const iree_hal_buffer_vtable_t vtable = {};
  iree_hal_buffer_initialize({}, &buffer, 64, 0, 64,
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                             IREE_HAL_MEMORY_ACCESS_READ,
                             IREE_HAL_BUFFER_USAGE_STORAGE, &vtable, &buffer);
  buffer.memory = {
      .contract = contract,
      .bindings = &binding,
      .binding_offset = 8,
  };
  auto reference = iree_hal_make_buffer_ref(&buffer, 4, 16);
  uint64_t address = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_resolve_address(
      &family, reference, IREE_HAL_MEMORY_ACCESS_READ, &address));
  EXPECT_EQ(address, 0x100Cu);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_amd_gpu_pm4_submission_resolve_address(
          &family, reference, IREE_HAL_MEMORY_ACCESS_WRITE, &address));
  buffer.allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_pm4_submission_resolve_address(
                            &family, reference, 0, &address));
  buffer.allowed_usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  family.memory.domain = &family;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_pm4_submission_resolve_address(
                            &family, reference, 0, &address));
  family.memory.domain = this;
  binding.device_address = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amd_gpu_pm4_submission_resolve_address(
                            &family, reference, 0, &address));
  binding.device_address = 0x1000;
  reference.offset = 60;
  layout_.binding_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Prepare(iree_hal_make_static_dispatch_config(1, 1, 1),
                                0, {}, {1, &reference}));
  iree_hal_memory_contract_release(contract);
}

}  // namespace
