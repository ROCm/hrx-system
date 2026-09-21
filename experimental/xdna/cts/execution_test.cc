// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "amdf/gpu.h"
#include "experimental/xdna/cts/execution_fixture.h"
#include "iree/testing/gtest.h"

namespace iree::experimental::xdna::testing {

class XdnaConcurrentQueuesTest : public XdnaExecutionFixture {};

TEST_F(XdnaConcurrentQueuesTest, PipelinesDistinctCommandsAndReusesSlots) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_, 1, 2));

  // A: (lhs, rhs) -> intermediate; B: (intermediate, rhs) -> lhs. Both
  // immutable commands share one context and instruction allocation.
  const ResolvedBindings consumer_bindings = {
      resolved_bindings_[2], resolved_bindings_[1], resolved_bindings_[0]};
  iree_hal_amd_xdna_executable_storage_t storage = {};
  storage.memory = first_.instructions.memory;
  storage.memory_byte_offset = first_.byte_length / 2;
  storage.mapping = iree_make_byte_span(
      first_.instructions.pointer + storage.memory_byte_offset,
      first_.byte_length - storage.memory_byte_offset);
  ASSERT_EQ(api_->memory_query_address(storage.memory, 0,
                                       AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
                                       &storage.device_address),
            AMDF_STATUS_OK);
  storage.device_address += storage.memory_byte_offset;
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_load(executable_, entry_ordinal_,
                                                   1, &storage));
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_bind(
      executable_, entry_ordinal_, 1, &storage, consumer_bindings.size(),
      consumer_bindings.data()));
  amdf_xdna_kernel_command_t consumer_command = {};
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_invocation(
      executable_, entry_ordinal_, 1, &storage, &consumer_command));
  first_.original_instructions.assign(
      first_.instructions.pointer,
      first_.instructions.pointer + first_.byte_length);
  ASSERT_EQ(api_->host_mapping_cache_control(first_.instructions.mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             first_.byte_length),
            AMDF_STATUS_OK);

  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    std::array<BindingValues, 3> expected;
    BindingValues poisoned;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = static_cast<uint32_t>(i * 2 + iteration * 2 + 3);
      expected[2][i] = expected[0][i] * expected[1][i];
      poisoned[i] = ~expected[2][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
    uint64_t last_submission = 0;
    for (const auto* command : {&first_.command, &consumer_command}) {
      amdf_xdna_kernel_queue_submission_info_t submit = {};
      submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
      submit.structure_size = sizeof(submit);
      submit.command_count = 1;
      submit.commands = command;
      ASSERT_EQ(xdna_api_->kernel_queue_submit(first_.queue, &submit,
                                               &last_submission),
                AMDF_STATUS_OK);
    }
    ASSERT_EQ(api_->kernel_queue_wait(first_.queue, last_submission,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    ASSERT_EQ(api_->kernel_queue_query_status(first_.queue, &status),
              AMDF_STATUS_OK);
    ASSERT_EQ(status.retired_submission, last_submission);
    ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = expected[2][i] * expected[1][i];
    }
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
    ASSERT_NO_FATAL_FAILURE(VerifyInstructions(first_));
  }
}

TEST_F(XdnaConcurrentQueuesTest,
       RecreatesDefaultQueuesWithInstructionsRetained) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(AMDF_MEMORY_PROFILE_ROLE_CREATE));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  const ResolvedBindings consumer_bindings = {
      resolved_bindings_[2], resolved_bindings_[1], resolved_bindings_[0]};
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(consumer_bindings, &second_));

  // Replace the serial fixture's queues with two default-sized windows while
  // their independently scoped instructions and shared data remain live.
  for (auto* execution : {&first_, &second_}) {
    amdf_kernel_queue_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->kernel_queue_query_info(execution->queue, &info),
              AMDF_STATUS_OK);
    const auto status = api_->kernel_queue_destroy(execution->queue);
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      execution->queue = nullptr;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    amdf_xdna_kernel_queue_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.queue_family_ordinal = info.queue_family_ordinal;
    ASSERT_EQ(xdna_api_->kernel_queue_create(execution->context, &create,
                                             &execution->queue),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->kernel_queue_query_info(execution->queue, &info),
              AMDF_STATUS_OK);
    ASSERT_NE(info.maximum_pending_submission_count, 0u);
  }

  // The windows coexist; the data dependency between contexts is explicit.
  // Full-window publication and wraparound have dedicated queue CTS coverage.
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    std::array<BindingValues, 3> expected;
    BindingValues poisoned;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = static_cast<uint32_t>(i * 2 + iteration * 2 + 3);
      expected[2][i] = expected[0][i] * expected[1][i];
      poisoned[i] = ~expected[2][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
    ASSERT_NO_FATAL_FAILURE(RunExecution(second_));
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = expected[2][i] * expected[1][i];
    }
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  }
}

class XdnaExecutionTest
    : public XdnaExecutionFixture,
      public ::testing::WithParamInterface<amdf_memory_profile_roles_t> {};

TEST_P(XdnaExecutionTest, ReusesImmutableInstructionsWithChangingInputs) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(GetParam()));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    std::array<BindingValues, 3> expected;
    BindingValues poisoned;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = kValues[(i * 3 + iteration + 5) % kElementCount];
      expected[2][i] = expected[0][i] * expected[1][i];
      // Every output must change; neither zero-fill nor stale output can pass.
      poisoned[i] = ~expected[2][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  }
}

TEST_P(XdnaExecutionTest, SharesDataAcrossIndependentContextLifetimes) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(GetParam()));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  // A: (lhs, rhs) -> intermediate; B: (intermediate, rhs) -> lhs.
  const ResolvedBindings consumer_bindings = {
      resolved_bindings_[2], resolved_bindings_[1], resolved_bindings_[0]};
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(consumer_bindings, &second_));
  ASSERT_NE(first_.context, second_.context);
  ASSERT_NE(first_.instructions.memory, second_.instructions.memory);

  // Fixed full-array backing forces both contexts onto the same hardware even
  // though each image needs only one logical column. Other providers still
  // exercise independent context lifetimes without claiming forced overlap.
  amdf_xdna_device_info_t device_info = {};
  device_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
  device_info.structure_size = sizeof(device_info);
  ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
            AMDF_STATUS_OK);
  if (device_info.placement_modes & AMDF_XDNA_PLACEMENT_MODE_FIXED_FULL_ARRAY) {
    for (const auto* execution : {&first_, &second_}) {
      amdf_xdna_context_placement_info_t placement = {};
      placement.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_PLACEMENT_INFO;
      placement.structure_size = sizeof(placement);
      ASSERT_EQ(xdna_api_->context_query_placement_info(execution->context,
                                                        &placement),
                AMDF_STATUS_OK);
      ASSERT_EQ(placement.column_origin, device_info.array.column_origin);
      ASSERT_EQ(placement.column_count, device_info.array.column_count);
    }
  }

  std::array<amdf_memory_info_t, 3> original_info = {};
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    auto& info = original_info[ordinal];
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(bindings_[ordinal].storage.memory, &info),
              AMDF_STATUS_OK);
  }

  std::array<BindingValues, 3> expected;
  BindingValues poisoned;
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      // Odd nonunit factors keep the second computation distinct modulo 2^32.
      expected[1][i] = static_cast<uint32_t>(i * 2 + iteration * 2 + 3);
      expected[2][i] = expected[0][i] * expected[1][i];
      poisoned[i] = ~expected[2][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
    // The consumer sees the producer's bytes directly. There is no CPU payload
    // access or cache transition between these fully retired finite commands.
    ASSERT_NO_FATAL_FAILURE(RunExecution(second_));
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = expected[2][i] * expected[1][i];
    }
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  }
  ASSERT_NO_FATAL_FAILURE(DestroyExecution(&first_));

  // Shared backing borrows the ordinary device, not the departed producer.
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(bindings_[ordinal].storage.memory, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(info.byte_length, original_info[ordinal].byte_length);
    ASSERT_EQ(info.source_byte_offset,
              original_info[ordinal].source_byte_offset);
    ASSERT_TRUE(amdf_physical_memory_id_is_equal(
        &info.physical_backing_id,
        &original_info[ordinal].physical_backing_id));
    uint64_t address = 0;
    ASSERT_EQ(
        api_->memory_query_address(bindings_[ordinal].storage.memory, 0,
                                   AMDF_MEMORY_ADDRESS_XDNA_DMA, &address),
        AMDF_STATUS_OK);
    ASSERT_EQ(address + kBindingByteOffset,
              resolved_bindings_[ordinal].device_address);
  }
  for (uint32_t iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[1][i] += 2;
      expected[0][i] = expected[2][i] * expected[1][i];
      poisoned[i] = ~expected[0][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, poisoned));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(RunExecution(second_));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  }
}

TEST_P(XdnaExecutionTest, ReestablishesStateAcrossFullWidthContextSwitches) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings(GetParam()));
  if (IsSkipped()) {
    return;
  }
  amdf_xdna_device_info_t device_info = {};
  device_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
  device_info.structure_size = sizeof(device_info);
  ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
            AMDF_STATUS_OK);
  const uint32_t column_count = device_info.array.column_count;
  if (column_count < device_info.context.minimum_column_count ||
      column_count > device_info.context.maximum_column_count ||
      (column_count - device_info.context.minimum_column_count) %
              device_info.context.column_count_granularity !=
          0) {
    GTEST_SKIP() << "full-array logical contexts are unavailable";
  }
  ASSERT_NO_FATAL_FAILURE(
      PrepareExecution(resolved_bindings_, &first_, column_count));
  // Full-width contexts cannot satisfy this interleave by occupying disjoint
  // physical columns. A produces (lhs * rhs); B consumes it without a host
  // copy.
  const ResolvedBindings consumer_bindings = {
      resolved_bindings_[2], resolved_bindings_[1], resolved_bindings_[0]};
  ASSERT_NO_FATAL_FAILURE(
      PrepareExecution(consumer_bindings, &second_, column_count));
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    std::array<BindingValues, 3> expected;
    BindingValues poisoned;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = static_cast<uint32_t>(i * 2 + iteration * 2 + 3);
      expected[2][i] = expected[0][i] * expected[1][i];
      poisoned[i] = ~expected[2][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
    // Native context lifetime does not preserve array configuration. Reuse
    // the same complete command bytes after every switch; preparation and
    // relocation still happen only once for each instruction allocation.
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
    ASSERT_NO_FATAL_FAILURE(RunExecution(second_));
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = expected[2][i] * expected[1][i];
    }
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
    ASSERT_NO_FATAL_FAILURE(VerifyInstructions(first_));
    ASSERT_NO_FATAL_FAILURE(VerifyInstructions(second_));
  }
}

using XdnaSharedMappingTest = XdnaSharedMappingFixture;

TEST_F(XdnaSharedMappingTest, ExecutesAfterOriginatorMappingCloses) {
  ASSERT_NO_FATAL_FAILURE(CreateSharedBindings());
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  std::array<BindingValues, 3> expected;
  BindingValues poisoned;
  for (size_t i = 0; i < kElementCount; ++i) {
    expected[0][i] = kValues[i];
    expected[1][i] = kValues[(i * 3 + 5) % kElementCount];
    expected[2][i] = expected[0][i] * expected[1][i];
    poisoned[i] = ~expected[2][i];
  }
  ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
  ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
  ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
  ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
  ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
}

// Exercises the queue visibility contract with real producers and consumers.
// COPY_DATA uses TC L2; it does not qualify shader-side PROGRAM transitions.
class XdnaPoolVisibilityTest : public XdnaExecutionTest {
 protected:
  static constexpr uint64_t kStagingByteLength = 4096;
  static constexpr uint64_t kReadbackByteOffset = 1024;
  static constexpr uint64_t kCompletionByteOffset = 2048;
  static constexpr uint64_t kRingByteLength = 4096;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaExecutionTest::SetUp());
    const void* extension = nullptr;
    const auto extension_status =
        api_->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                              AMDF_GPU_EXTENSION_VERSION_LATEST, &extension);
    if (extension_status ==
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "the GPU provider is not enabled";
    }
    ASSERT_EQ(extension_status, AMDF_STATUS_OK);
    gpu_api_ = static_cast<const amdf_gpu_api_t*>(extension);
    uint32_t count = 0;
    ASSERT_EQ(api_->endpoint_enumerate(instance_, 0, nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> endpoints(count);
    ASSERT_EQ(
        api_->endpoint_enumerate(instance_, count, endpoints.data(), &count),
        AMDF_STATUS_OK);
    amdf_endpoint_t* gpu_endpoint = nullptr;
    for (const auto& summary : endpoints) {
      if (summary.engine_kind != AMDF_ENGINE_KIND_GPU) {
        continue;
      }
      amdf_endpoint_t* endpoint = nullptr;
      ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint),
                AMDF_STATUS_OK);
      amdf_endpoint_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->endpoint_query_info(endpoint, &info), AMDF_STATUS_OK);
      for (uint32_t ordinal = 0; ordinal < info.queue_family_count; ++ordinal) {
        amdf_queue_family_info_t family = {};
        family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
        family.structure_size = sizeof(family);
        ASSERT_EQ(
            api_->endpoint_query_queue_family_info(endpoint, ordinal, &family),
            AMDF_STATUS_OK);
        const bool user_publication =
            (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) !=
                0 &&
            (family.user_queue_capabilities &
             AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) != 0 &&
            family.maximum_ring_byte_length >= kRingByteLength;
        const bool kernel_publication =
            (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) !=
            0;
        if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
            family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 &&
            (family.format_features &
             AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR) != 0 &&
            (user_publication || kernel_publication)) {
          gpu_family_ = family;
          gpu_publication_mode_ = user_publication
                                      ? AMDF_QUEUE_PUBLICATION_MODE_USER
                                      : AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
          gpu_endpoint = endpoint;
          break;
        }
      }
      if (gpu_endpoint) {
        break;
      }
    }
    if (!gpu_endpoint) {
      GTEST_SKIP() << "GPU PM4 publication with GCR is not advertised";
    }
    ASSERT_EQ(GetCtsDeviceCache().GetGpuDevice(gpu_endpoint, &gpu_device_),
              AMDF_STATUS_OK);
    pool_accesses_[0] = memory_access_;
    pool_accesses_[0].requirements.address_kinds =
        UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    pool_accesses_[1].device = gpu_device_;
    pool_accesses_[1].requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    pool_accesses_[1].requirements.flags =
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    pool_accesses_[1].requirements.address_kinds = UINT64_C(1)
                                                   << AMDF_MEMORY_ADDRESS_GPU;
  }

  void TearDown() override {
    if (gpu_mapping_) {
      ASSERT_EQ(api_->user_queue_mapping_destroy(gpu_mapping_), AMDF_STATUS_OK);
      gpu_mapping_ = nullptr;
    }
    if (gpu_queue_) {
      const auto status = api_->user_queue_destroy(gpu_queue_);
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        gpu_queue_ = nullptr;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    if (gpu_kernel_queue_) {
      const auto status = api_->kernel_queue_destroy(gpu_kernel_queue_);
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        gpu_kernel_queue_ = nullptr;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&gpu_commands_));
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&staging_));
    XdnaExecutionTest::TearDown();
  }

  void FindProfile(uint32_t access_count,
                   const amdf_memory_device_access_t* accesses,
                   amdf_memory_profile_roles_t role,
                   amdf_memory_profile_t* out_profile) {
    out_profile->ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ordinal = 0;; ++ordinal) {
      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      std::array<amdf_memory_access_capabilities_t, 2> capabilities = {};
      for (auto& capability : capabilities) {
        capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capability.structure_size = sizeof(capability);
      }
      const auto status = api_->memory_scope_query_device_profile(
          system_scope_, ordinal, access_count, accesses, &profile,
          capabilities.data());
      if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) {
        break;
      }
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile.roles & role) != 0 &&
          (profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) != 0 &&
          (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
        *out_profile = profile;
        return;
      }
    }
  }

  void CreatePool() {
    amdf_memory_profile_t profile = {};
    ASSERT_NO_FATAL_FAILURE(FindProfile(
        pool_accesses_.size(), pool_accesses_.data(), GetParam(), &profile));
    if (profile.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      GTEST_SKIP() << "joint memory role " << GetParam()
                   << " is not advertised";
    }
    ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    std::array<amdf_memory_profile_site_t, 3> sites = {};
    sites[0].kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    sites[0].value.device.queue_family_ordinal = queue_family_ordinal_;
    sites[1].kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    sites[1].value.device.access_ordinal = 1;
    sites[1].value.device.queue_family_ordinal = gpu_family_.ordinal;
    sites[2].kind = AMDF_MEMORY_SITE_KIND_HOST;
    sites[2].value.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    amdf_memory_profile_pair_query_t query = {};
    query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
    query.structure_size = sizeof(query);
    query.memory_profile_ordinal = profile.ordinal;
    query.access_count = pool_accesses_.size();
    query.accesses = pool_accesses_.data();
    query.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    query.registered_host_cacheability =
        profile.registration.registered_host_cacheability;
    // All recipes are obtained before any pool backing exists, and retained
    // across independent allocations, queue submissions and data generations.
    std::array<amdf_memory_pair_info_t, 3> pairs = {};
    const uint32_t edges[][2] = {{2, 0}, {1, 0}, {0, 1}};
    for (size_t i = 0; i < pairs.size(); ++i) {
      query.producer = sites[edges[i][0]];
      query.consumer = sites[edges[i][1]];
      pairs[i].type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
      pairs[i].structure_size = sizeof(pairs[i]);
      ASSERT_EQ(
          api_->memory_scope_query_pair_info(system_scope_, &query, &pairs[i]),
          AMDF_STATUS_OK);
      ASSERT_NE(pairs[i].flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
    }
    const auto& publish = pairs[0].release;
    ASSERT_EQ(publish.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    ASSERT_EQ(publish.executor, AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
    ASSERT_EQ(pairs[0].acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pairs[1].acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pairs[2].release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    gpu_release_ = pairs[1].release;
    gpu_acquire_ = pairs[2].acquire;
    ASSERT_EQ(gpu_release_.operation, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
    ASSERT_EQ(gpu_acquire_.operation, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);

    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      auto& storage = bindings_[ordinal].storage;
      amdf_memory_create_info_t create = {};
      create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      create.structure_size = sizeof(create);
      create.memory_profile_ordinal = profile.ordinal;
      create.access_count = query.access_count;
      create.accesses = query.accesses;
      create.required_flags = query.required_flags;
      create.byte_length = kBindingStorageByteLength;
      create.minimum_alignment = kBindingByteLength;
      create.registered_host_cacheability = query.registered_host_cacheability;
      if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        ASSERT_NO_FATAL_FAILURE(PrepareRegistration(
            profile, &bindings_[ordinal].caller_storage, &create));
      }
      ASSERT_EQ(api_->memory_create(system_scope_, &create, &storage.memory),
                AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(MapMemory(create.byte_length, &storage));
      std::memset(storage.pointer, kGuardValue, create.byte_length);
      ASSERT_EQ(
          api_->host_mapping_cache_control(
              storage.mapping, publish.host_operation, 0, create.byte_length),
          AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(WrapBinding(ordinal, storage.memory, 0));
      ASSERT_EQ(
          api_->memory_query_address(storage.memory, 1, AMDF_MEMORY_ADDRESS_GPU,
                                     &gpu_addresses_[ordinal]),
          AMDF_STATUS_OK);
    }
  }

  void CreateGpuQueue() {
    amdf_memory_profile_t profile = {};
    ASSERT_NO_FATAL_FAILURE(FindProfile(
        1, &pool_accesses_[1], AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
    ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = profile.ordinal;
    create.access_count = 1;
    create.accesses = &pool_accesses_[1];
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length = kStagingByteLength;
    ASSERT_EQ(api_->memory_create(system_scope_, &create, &staging_.memory),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(MapMemory(create.byte_length, &staging_));
    ASSERT_EQ(
        api_->memory_query_address(staging_.memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                   &staging_address_),
        AMDF_STATUS_OK);
    if (gpu_publication_mode_ == AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
      amdf_gpu_kernel_queue_create_info_t queue = {};
      queue.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO;
      queue.structure_size = sizeof(queue);
      queue.queue_family_ordinal = gpu_family_.ordinal;
      ASSERT_EQ(gpu_api_->kernel_queue_create(gpu_device_, &queue,
                                              &gpu_kernel_queue_),
                AMDF_STATUS_OK);
      auto command_access = pool_accesses_[1];
      command_access.requirements.access |= AMDF_MEMORY_ACCESS_EXECUTE;
      ASSERT_NO_FATAL_FAILURE(FindProfile(
          1, &command_access, AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
      ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
      create.memory_profile_ordinal = profile.ordinal;
      create.accesses = &command_access;
      create.byte_length = kRingByteLength;
      ASSERT_EQ(
          api_->memory_create(system_scope_, &create, &gpu_commands_.memory),
          AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(MapMemory(create.byte_length, &gpu_commands_));
      return;
    }
    amdf_gpu_user_queue_create_info_t queue = {};
    queue.type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO;
    queue.structure_size = sizeof(queue);
    queue.queue_family_ordinal = gpu_family_.ordinal;
    queue.priority = AMDF_QUEUE_PRIORITY_NORMAL;
    queue.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
    queue.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
    queue.ring_byte_length =
        std::max(gpu_family_.minimum_ring_byte_length, kRingByteLength);
    ASSERT_EQ(gpu_api_->user_queue_create(gpu_device_, &queue, &gpu_queue_),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->user_queue_map(gpu_queue_, nullptr, &gpu_mapping_),
              AMDF_STATUS_OK);
    gpu_mapping_info_.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
    gpu_mapping_info_.structure_size = sizeof(gpu_mapping_info_);
    ASSERT_EQ(
        api_->user_queue_mapping_query_info(gpu_mapping_, &gpu_mapping_info_),
        AMDF_STATUS_OK);
    ASSERT_EQ(gpu_mapping_info_.index_bits, 64u);
    ASSERT_EQ(gpu_mapping_info_.doorbell_bits, 64u);
  }

  static uint32_t Pm4Header(uint32_t opcode, uint32_t count) {
    return (3u << 30) | (opcode << 8) | ((count - 2) << 16);
  }

  static void AppendGpuTransition(const amdf_cache_transition_t& transition,
                                  std::vector<uint32_t>* words) {
    ASSERT_EQ(transition.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(transition.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    // CS_PARTIAL_FLUSH followed by a conservative full-range ACQUIRE_MEM GCR
    // implements both system release and system acquire on this format.
    constexpr uint32_t kGcr = (3 << 0) | (1 << 4) | (1 << 5) | (1 << 7) |
                              (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15);
    words->insert(words->end(),
                  {Pm4Header(0x46, 2), 7 | (4 << 8), Pm4Header(0x58, 8), 0,
                   UINT32_MAX, 0xff, 0, 0, 0x0a, kGcr});
  }

  static void AppendGpuCopy(uint64_t source, uint64_t target,
                            size_t byte_length, std::vector<uint32_t>* words) {
    for (size_t offset = 0; offset < byte_length; offset += sizeof(uint32_t)) {
      // COPY_DATA between TC L2 addresses, with write confirmation.
      words->insert(words->end(),
                    {Pm4Header(0x40, 6), 2 | (2 << 8) | (1 << 20),
                     static_cast<uint32_t>(source + offset),
                     static_cast<uint32_t>((source + offset) >> 32),
                     static_cast<uint32_t>(target + offset),
                     static_cast<uint32_t>((target + offset) >> 32)});
    }
  }

  void RunGpu(std::vector<uint32_t> words, uint32_t completion_value) {
    if (gpu_kernel_queue_) {
      const size_t byte_length = words.size() * sizeof(uint32_t);
      ASSERT_LE(byte_length, kRingByteLength);
      std::memcpy(gpu_commands_.pointer, words.data(), byte_length);
      ASSERT_EQ(api_->host_mapping_cache_control(
                    gpu_commands_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                    byte_length),
                AMDF_STATUS_OK);
      amdf_gpu_kernel_command_t command = {};
      command.memory = gpu_commands_.memory;
      command.byte_length = byte_length;
      amdf_gpu_kernel_queue_submission_info_t submit = {};
      submit.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO;
      submit.structure_size = sizeof(submit);
      submit.command_count = 1;
      submit.commands = &command;
      uint64_t submission = 0;
      ASSERT_EQ(gpu_api_->kernel_queue_submit(gpu_kernel_queue_, &submit,
                                              &submission),
                AMDF_STATUS_OK);
      ASSERT_EQ(api_->kernel_queue_wait(gpu_kernel_queue_, submission,
                                        AMDF_TIMEOUT_INFINITE, 0),
                AMDF_STATUS_OK);
      return;
    }
    auto* completion = reinterpret_cast<volatile uint32_t*>(
        staging_.pointer + kCompletionByteOffset);
    ASSERT_NE(*completion, completion_value);
    const uint64_t completion_address =
        staging_address_ + kCompletionByteOffset;
    // WRITE_DATA confirms a separate coherent completion line after the
    // ordered copies/barriers. Ring consumption alone is not completion.
    words.insert(words.end(), {Pm4Header(0x37, 5), (2 << 8) | (1 << 20),
                               static_cast<uint32_t>(completion_address),
                               static_cast<uint32_t>(completion_address >> 32),
                               completion_value});
    auto* ring = reinterpret_cast<uint32_t*>(
        static_cast<uintptr_t>(gpu_mapping_info_.ring_address));
    const uint64_t capacity =
        gpu_mapping_info_.ring_byte_length / sizeof(*ring);
    std::vector<uint32_t> publication;
    auto append_padding = [&](size_t count) {
      publication.push_back(Pm4Header(0x10, count));
      publication.resize(publication.size() + count - 1, 0);
    };
    // Preserve packet boundaries across ring wrap and leave space for a
    // complete type-3 NOP instead of stranding one dword at the tail.
    for (size_t i = 0; i < words.size();) {
      const size_t count = ((words[i] >> 16) & 0x3fff) + 2;
      const size_t tail =
          capacity - (published_index_ + publication.size()) % capacity;
      if (tail < count || tail == count + 1) {
        append_padding(tail);
      }
      publication.insert(publication.end(), words.begin() + i,
                         words.begin() + i + count);
      i += count;
    }
    size_t padding = 8 - publication.size() % 8;
    if (padding == 1) {
      padding += 8;
    }
    append_padding(padding);
    // All prior batches have retired. The publication still reserves the
    // native PM4 empty/full discriminator by remaining smaller than the ring.
    ASSERT_LT(publication.size(), capacity);
    for (size_t i = 0; i < publication.size(); ++i) {
      ring[(published_index_ + i) % capacity] = publication[i];
    }
    published_index_ += publication.size();
    auto* write_index = reinterpret_cast<volatile uint64_t*>(
        static_cast<uintptr_t>(gpu_mapping_info_.write_index_address));
    auto* doorbell = reinterpret_cast<volatile uint64_t*>(
        static_cast<uintptr_t>(gpu_mapping_info_.doorbell_address));
    std::atomic_thread_fence(std::memory_order_release);
    *write_index = published_index_;
    std::atomic_thread_fence(std::memory_order_release);
    *doorbell = published_index_;
    ASSERT_EQ(api_->user_queue_wait_consumed(gpu_queue_, published_index_,
                                             AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    while (*completion != completion_value) {
      amdf_user_queue_status_t status = {};
      status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
      status.structure_size = sizeof(status);
      ASSERT_EQ(api_->user_queue_query_status(gpu_queue_, &status),
                AMDF_STATUS_OK);
      ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
      std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  // GPU extension table and device borrowed from the shared CTS provider.
  const amdf_gpu_api_t* gpu_api_ = nullptr;
  // Ordinary GPU address domain retained by the CTS cache.
  amdf_device_t* gpu_device_ = nullptr;
  // Exact PM4 family used by qualification and publication.
  amdf_queue_family_info_t gpu_family_ = {};
  // Publication strategy selected from the complete family capabilities.
  amdf_queue_publication_modes_t gpu_publication_mode_ = 0;
  // Complete pool contract, ordered XDNA then GPU.
  std::array<amdf_memory_device_access_t, 2> pool_accesses_ = {};
  // GPU release selected before any backing exists.
  amdf_cache_transition_t gpu_release_ = {};
  // GPU acquire selected before any backing exists.
  amdf_cache_transition_t gpu_acquire_ = {};
  // Cold GPU addresses corresponding to the three XDNA bindings.
  std::array<uint64_t, 3> gpu_addresses_ = {};
  // GPU-only host staging, readback and a separate completion cache line.
  MappedMemory staging_;
  // GPU address of staging_; never used by XDNA.
  uint64_t staging_address_ = 0;
  // Case-owned GPU queue, destroyed before any reachable backing.
  amdf_user_queue_t* gpu_queue_ = nullptr;
  // Native kernel publication when the family has no user producer.
  amdf_kernel_queue_t* gpu_kernel_queue_ = nullptr;
  // Caller-owned command storage, changed only after its submission retires.
  MappedMemory gpu_commands_;
  // Host producer mapping, destroyed before the queue.
  amdf_user_queue_mapping_t* gpu_mapping_ = nullptr;
  // Cold publication operands for the host producer.
  amdf_user_queue_mapping_info_t gpu_mapping_info_ = {};
  // Monotonic dword index; each batch completes before the next overwrites it.
  uint64_t published_index_ = 0;
};

TEST_P(XdnaPoolVisibilityTest, ReplaysQualifiedGpuXdnaGpuTransitions) {
  ASSERT_NO_FATAL_FAILURE(CreatePool());
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(CreateGpuQueue());
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  std::vector<uint32_t> ingress;
  std::vector<uint32_t> egress;
  ASSERT_NO_FATAL_FAILURE(AppendGpuTransition(gpu_acquire_, &egress));
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    AppendGpuCopy(staging_address_ + ordinal * kBindingByteLength,
                  gpu_addresses_[ordinal] + kBindingByteOffset,
                  kBindingByteLength, &ingress);
    AppendGpuCopy(gpu_addresses_[ordinal],
                  staging_address_ + kReadbackByteOffset +
                      ordinal * kBindingStorageByteLength,
                  kBindingStorageByteLength, &egress);
  }
  ASSERT_NO_FATAL_FAILURE(AppendGpuTransition(gpu_release_, &ingress));
  // Retire the GPU readback writes before its host-visible completion marker.
  ASSERT_NO_FATAL_FAILURE(AppendGpuTransition(gpu_release_, &egress));
  for (uint32_t iteration = 0; iteration < 8; ++iteration) {
    SCOPED_TRACE(iteration);
    std::memset(staging_.pointer, 0, kStagingByteLength);
    std::array<BindingValues, 3> expected;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = kValues[(i * 3 + iteration + 5) % kElementCount];
      expected[2][i] = expected[0][i] * expected[1][i];
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        iree_unaligned_store_le_u32(
            staging_.pointer + ordinal * kBindingByteLength + i * 4,
            ordinal == 2 ? ~expected[2][i] : expected[ordinal][i]);
      }
    }
    ASSERT_EQ(api_->host_mapping_cache_control(staging_.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, kStagingByteLength),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(RunGpu(ingress, iteration * 2 + 1));
    // Independent time-sliced commands establish their own tile state. The
    // immutable combined range needs no repeated preparation or relocation.
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
    ASSERT_NO_FATAL_FAILURE(RunGpu(egress, iteration * 2 + 2));
    // The CPU has not touched or maintained the shared payload since setup.
    // Readback covers guards as well as products; poison prevents stale output
    // from passing and GPU writes make the consumer's L2 lines nontrivial.
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            staging_.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
            kReadbackByteOffset, bindings_.size() * kBindingStorageByteLength),
        AMDF_STATUS_OK);
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      SCOPED_TRACE(ordinal);
      const auto* readback = staging_.pointer + kReadbackByteOffset +
                             ordinal * kBindingStorageByteLength;
      for (size_t i = 0; i < kBindingByteLength; ++i) {
        ASSERT_EQ(readback[i], kGuardValue) << "prefix " << i;
        ASSERT_EQ(readback[2 * kBindingByteLength + i], kGuardValue)
            << "suffix " << i;
      }
      for (size_t i = 0; i < kElementCount; ++i) {
        ASSERT_EQ(
            iree_unaligned_load_le_u32(readback + kBindingByteOffset + i * 4),
            expected[ordinal][i])
            << "element " << i;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    PoolBacking, XdnaPoolVisibilityTest,
    ::testing::Values(AMDF_MEMORY_PROFILE_ROLE_CREATE,
                      AMDF_MEMORY_PROFILE_ROLE_REGISTER),
    [](const ::testing::TestParamInfo<amdf_memory_profile_roles_t>& info) {
      return info.param == AMDF_MEMORY_PROFILE_ROLE_CREATE ? "Allocated"
                                                           : "Registered";
    });

INSTANTIATE_TEST_SUITE_P(
    MemoryBacking, XdnaExecutionTest,
    ::testing::Values(AMDF_MEMORY_PROFILE_ROLE_CREATE,
                      AMDF_MEMORY_PROFILE_ROLE_REGISTER,
                      AMDF_MEMORY_PROFILE_ROLE_IMPORT),
    [](const ::testing::TestParamInfo<amdf_memory_profile_roles_t>& info) {
      switch (info.param) {
        case AMDF_MEMORY_PROFILE_ROLE_CREATE:
          return "Allocated";
        case AMDF_MEMORY_PROFILE_ROLE_REGISTER:
          return "Registered";
        default:
          return "Imported";
      }
    });

}  // namespace iree::experimental::xdna::testing
