// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/api_experimental.h"

#include <array>
#include <cstdint>
#include <thread>
#include <vector>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class AmdgpuExperimentalApiTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping live tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping live tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  class TestLogicalDevice {
   public:
    ~TestLogicalDevice() {
      iree_hal_device_release(device_);
      iree_hal_device_group_release(device_group_);
    }

    iree_status_t Initialize(
        const iree_hal_amdgpu_logical_device_options_t* options,
        const iree_hal_amdgpu_libhsa_t* libhsa = &libhsa_) {
      IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator_));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
          IREE_SV("amdgpu"), options, libhsa, &topology_,
          create_context_.params(), host_allocator_, &device_));
      return iree_hal_device_group_create_from_device(
          device_, create_context_.frontier_tracker(), host_allocator_,
          &device_group_);
    }

    iree_hal_device_t* device() const { return device_; }

    iree_hal_amdgpu_logical_device_t* logical_device() const {
      return (iree_hal_amdgpu_logical_device_t*)device_;
    }

   private:
    iree::hal::cts::DeviceCreateContext create_context_;
    iree_hal_device_t* device_ = nullptr;
    iree_hal_device_group_t* device_group_ = nullptr;
  };

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t AmdgpuExperimentalApiTest::host_allocator_;
iree_hal_amdgpu_libhsa_t AmdgpuExperimentalApiTest::libhsa_;
iree_hal_amdgpu_topology_t AmdgpuExperimentalApiTest::topology_;

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
static hsa_status_t HSA_API RejectQueueCuMask(const hsa_queue_t* queue,
                                              uint32_t mask_bit_count,
                                              const uint32_t* mask) {
  (void)queue;
  (void)mask_bit_count;
  (void)mask;
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

static decltype(iree_hal_amdgpu_libhsa_t::hsa_agent_get_info)
    native_aql_agent_get_info_delegate = nullptr;

static hsa_status_t HSA_API NativeAqlAgentGetInfo(hsa_agent_t agent,
                                                  hsa_agent_info_t attribute,
                                                  void* value) {
  if (attribute == (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PM4_EMULATION) {
    *static_cast<bool*>(value) = false;
    return HSA_STATUS_SUCCESS;
  }
  return native_aql_agent_get_info_delegate(agent, attribute, value);
}

class ScopedNativeAqlLibhsa {
 public:
  ScopedNativeAqlLibhsa() = default;

  ~ScopedNativeAqlLibhsa() {
    if (!initialized_) return;
    native_aql_agent_get_info_delegate = previous_agent_get_info_delegate_;
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* source_libhsa) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_libhsa_copy(source_libhsa, &libhsa_));
    previous_agent_get_info_delegate_ = native_aql_agent_get_info_delegate;
    native_aql_agent_get_info_delegate = libhsa_.hsa_agent_get_info;
    libhsa_.hsa_agent_get_info = NativeAqlAgentGetInfo;
    initialized_ = true;
    return iree_ok_status();
  }

  const iree_hal_amdgpu_libhsa_t* libhsa() const { return &libhsa_; }

  ScopedNativeAqlLibhsa(const ScopedNativeAqlLibhsa&) = delete;
  ScopedNativeAqlLibhsa& operator=(const ScopedNativeAqlLibhsa&) = delete;

 private:
  bool initialized_ = false;
  iree_hal_amdgpu_libhsa_t libhsa_ = {};
  decltype(iree_hal_amdgpu_libhsa_t::hsa_agent_get_info)
      previous_agent_get_info_delegate_ = nullptr;
};

class ScopedAgentTargetVersionOverride {
 public:
  ScopedAgentTargetVersionOverride(
      iree_hal_amdgpu_physical_device_t* physical_device,
      iree_hal_amdgpu_gfxip_version_t version)
      : physical_device_(physical_device),
        original_agent_target_(physical_device->agent_target),
        agent_target_(*original_agent_target_) {
    agent_target_.primary_isa.identity.version = version;
    physical_device_->agent_target = &agent_target_;
  }

  ~ScopedAgentTargetVersionOverride() {
    physical_device_->agent_target = original_agent_target_;
  }

  ScopedAgentTargetVersionOverride(const ScopedAgentTargetVersionOverride&) =
      delete;
  ScopedAgentTargetVersionOverride& operator=(
      const ScopedAgentTargetVersionOverride&) = delete;

 private:
  iree_hal_amdgpu_physical_device_t* physical_device_;
  const iree_hal_amdgpu_agent_target_t* original_agent_target_;
  iree_hal_amdgpu_agent_target_t agent_target_;
};

static uint32_t queue_create_probe_call_count = 0;

static hsa_status_t HSA_API
ProbeQueueCreate(hsa_agent_t, uint32_t, hsa_queue_type32_t,
                 void (*)(hsa_status_t, hsa_queue_t*, void*), void*, uint32_t,
                 uint32_t, hsa_queue_t** queue) {
  ++queue_create_probe_call_count;
  *queue = nullptr;
  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

class ScopedQueueCreateProbe {
 public:
  explicit ScopedQueueCreateProbe(
      iree_hal_amdgpu_logical_device_t* logical_device)
      : libhsa_(&logical_device->system->libhsa),
        original_queue_create_(libhsa_->hsa_queue_create) {
    queue_create_probe_call_count = 0;
    libhsa_->hsa_queue_create = ProbeQueueCreate;
  }

  ~ScopedQueueCreateProbe() {
    libhsa_->hsa_queue_create = original_queue_create_;
  }

  ScopedQueueCreateProbe(const ScopedQueueCreateProbe&) = delete;
  ScopedQueueCreateProbe& operator=(const ScopedQueueCreateProbe&) = delete;

  uint32_t call_count() const { return queue_create_probe_call_count; }
  void ResetCallCount() { queue_create_probe_call_count = 0; }

 private:
  iree_hal_amdgpu_libhsa_t* libhsa_;
  decltype(iree_hal_amdgpu_libhsa_t::hsa_queue_create) original_queue_create_;
};
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

iree_status_t SubmitFillAndWait(iree_hal_device_t* device,
                                iree_hal_queue_affinity_t queue_affinity) {
  const iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), buffer_params, sizeof(uint32_t),
      &buffer);
  iree_hal_semaphore_t* semaphore = nullptr;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_create(device, queue_affinity, /*initial_value=*/0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
  }
  uint64_t payload_value = 1;
  const iree_hal_semaphore_list_t signal_semaphore_list = {
      /*.count=*/1,
      /*.semaphores=*/&semaphore,
      /*.payload_values=*/&payload_value,
  };
  if (iree_status_is_ok(status)) {
    const uint32_t pattern = 0xA2A2A2A2u;
    status = iree_hal_device_queue_fill(
        device, queue_affinity, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, buffer, /*target_offset=*/0, sizeof(pattern),
        &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, payload_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  return status;
}

static std::vector<uint32_t> MakeFullNativeMask(uint32_t compute_unit_count) {
  const iree_host_size_t word_count = (compute_unit_count + 31u) / 32u;
  std::vector<uint32_t> mask(word_count, UINT32_MAX);
  const uint32_t tail_bit_count = compute_unit_count % 32u;
  if (tail_bit_count != 0) {
    mask.back() = (UINT32_C(1) << tail_bit_count) - 1u;
  }
  return mask;
}

TEST(AmdgpuExperimentalApiBoundaryTest, RejectsNonNativeDevice) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = IREE_SV("mock");
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));

  const uint32_t mask = 1;
  iree_hal_queue_affinity_t queue_affinity = 1;
  iree_hal_amdgpu_experimental_execution_queue_topology_t topology = {
      /*.first_private_physical_queue_ordinal=*/1,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amdgpu_experimental_execution_queue_query(
                            device, /*physical_device_ordinal=*/0, &topology));
  EXPECT_EQ(topology.first_private_physical_queue_ordinal, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amdgpu_experimental_execution_queue_configure(
                            device, /*physical_device_ordinal=*/0,
                            /*physical_queue_ordinal=*/0,
                            /*mask_bit_count=*/32, &mask, &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);

  iree_hal_device_release(device);
}

TEST_F(AmdgpuExperimentalApiTest,
       ConfiguresAndSubmitsToCallerSelectedSparseProductionQueues) {
  iree_hal_amdgpu_aql_queue_execution_mode_t execution_mode;
  IREE_ASSERT_OK(iree_hal_amdgpu_query_aql_queue_execution_mode(
      &libhsa_, topology_.gpu_agents[0], &execution_mode));
  if (execution_mode != IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_NATIVE) {
    GTEST_SKIP() << "fixed-mask queues require native GPU-consumed AQL";
  }

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 2;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options));

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_GT(logical_device->physical_device_count, 0u);
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  ASSERT_GT(physical_device->compute_unit_count, 0u);
  ASSERT_EQ(physical_device->host_queue_capacity,
            physical_device->host_queue_ordinary_count + 2);

  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.device(), /*physical_device_ordinal=*/0, &queue_topology));
  EXPECT_EQ(queue_topology.first_private_physical_queue_ordinal,
            physical_device->host_queue_ordinary_count);
  EXPECT_EQ(queue_topology.private_physical_queue_count, 2u);
  EXPECT_EQ(queue_topology.native_compute_unit_count,
            physical_device->compute_unit_count);
  EXPECT_TRUE(queue_topology.native_compute_unit_mask_alignment == 1u ||
              queue_topology.native_compute_unit_mask_alignment == 2u);
  EXPECT_GT(queue_topology.native_compute_unit_mask_partition_count, 0u);

  const iree_host_size_t mask_word_count =
      (physical_device->compute_unit_count + 31u) / 32u;
  std::vector<uint32_t> mask(mask_word_count, UINT32_MAX);
  const uint32_t tail_bit_count = physical_device->compute_unit_count % 32u;
  if (tail_bit_count != 0) {
    mask.back() = (UINT32_C(1) << tail_bit_count) - 1u;
  }
  const iree_host_size_t mask_bit_count = mask_word_count * 32u;

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.device(), /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/
      queue_topology.first_private_physical_queue_ordinal + 1, mask_bit_count,
      mask.data(), &queue_affinity));
  const iree_host_size_t first_private_ordinal =
      physical_device->host_queue_ordinary_count;
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, first_private_ordinal));
  EXPECT_TRUE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, first_private_ordinal + 1));
  EXPECT_EQ(physical_device->host_queue_count,
            physical_device->host_queue_ordinary_count + 1);

  iree_hal_queue_affinity_t expected_affinity = 0;
  const iree_hal_amdgpu_queue_affinity_domain_t affinity_domain = {
      /*.supported_affinity=*/logical_device->queue_affinity_mask,
      /*.physical_device_count=*/logical_device->physical_device_count,
      /*.queue_count_per_physical_device=*/
      logical_device->system->topology.gpu_agent_queue_count,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_for_physical_queue(
      affinity_domain, /*physical_device_ordinal=*/0, first_private_ordinal + 1,
      &expected_affinity));
  EXPECT_EQ(queue_affinity, expected_affinity);
  IREE_ASSERT_OK(SubmitFillAndWait(test_device.device(), queue_affinity));
  iree_hal_amdgpu_host_queue_t* ordinary_queue =
      &physical_device->host_queues[0];
  iree_hal_amdgpu_host_queue_t* private_queue =
      &physical_device->host_queues[first_private_ordinal + 1];
  const uint64_t ordinary_epoch_before_any =
      ordinary_queue->notification_ring.epoch.next_submission;
  const uint64_t private_epoch_before_any =
      private_queue->notification_ring.epoch.next_submission;
  IREE_ASSERT_OK(
      SubmitFillAndWait(test_device.device(), IREE_HAL_QUEUE_AFFINITY_ANY));
  EXPECT_GT(ordinary_queue->notification_ring.epoch.next_submission,
            ordinary_epoch_before_any);
  EXPECT_EQ(private_queue->notification_ring.epoch.next_submission,
            private_epoch_before_any);
  IREE_EXPECT_OK(
      iree_hal_device_queue_flush(test_device.device(), queue_affinity));
  IREE_EXPECT_OK(iree_hal_device_queue_flush(test_device.device(),
                                             IREE_HAL_QUEUE_AFFINITY_ANY));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      iree_hal_amdgpu_experimental_execution_queue_configure(
          test_device.device(), /*physical_device_ordinal=*/0,
          /*physical_queue_ordinal=*/
          queue_topology.first_private_physical_queue_ordinal + 1,
          mask_bit_count, mask.data(), &queue_affinity));

  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.device(), /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/
      queue_topology.first_private_physical_queue_ordinal, mask_bit_count,
      mask.data(), &queue_affinity));
  IREE_ASSERT_OK(SubmitFillAndWait(test_device.device(), queue_affinity));
  EXPECT_TRUE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, first_private_ordinal));
  EXPECT_TRUE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, first_private_ordinal + 1));
  EXPECT_EQ(physical_device->host_queue_count,
            physical_device->host_queue_capacity);
}

TEST_F(AmdgpuExperimentalApiTest,
       RejectsMasksThatLeaveHardwarePartitionsUnconfined) {
  iree_hal_amdgpu_aql_queue_execution_mode_t execution_mode;
  IREE_ASSERT_OK(iree_hal_amdgpu_query_aql_queue_execution_mode(
      &libhsa_, topology_.gpu_agents[0], &execution_mode));
  if (execution_mode != IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_NATIVE) {
    GTEST_SKIP() << "fixed-mask queues require native GPU-consumed AQL";
  }

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 1;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options));

  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.device(), /*physical_device_ordinal=*/0, &queue_topology));
  if (queue_topology.native_compute_unit_mask_partition_count == 1) {
    GTEST_SKIP() << "device has no interleaved CU-mask partitions";
  }
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];
  const iree_host_size_t mask_word_count =
      (queue_topology.native_compute_unit_count + 31u) / 32u;
  std::vector<uint32_t> invalid_mask(mask_word_count, 0);
  invalid_mask[0] = 1;
  iree_hal_queue_affinity_t queue_affinity = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_experimental_execution_queue_configure(
          test_device.device(), /*physical_device_ordinal=*/0,
          queue_topology.first_private_physical_queue_ordinal,
          mask_word_count * 32u, invalid_mask.data(), &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, queue_topology.first_private_physical_queue_ordinal));

  std::vector<uint32_t> valid_mask(mask_word_count, 0);
  for (uint32_t partition = 0;
       partition < queue_topology.native_compute_unit_mask_partition_count;
       ++partition) {
    valid_mask[partition / 32u] |= UINT32_C(1) << (partition % 32u);
  }
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.device(), /*physical_device_ordinal=*/0,
      queue_topology.first_private_physical_queue_ordinal,
      mask_word_count * 32u, valid_mask.data(), &queue_affinity));
  EXPECT_NE(queue_affinity, 0u);
  IREE_EXPECT_OK(SubmitFillAndWait(test_device.device(), queue_affinity));
}

TEST_F(AmdgpuExperimentalApiTest,
       RejectsMisalignedNativeComputeUnitGroupsBeforeQueueCreation) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  GTEST_SKIP() << "deterministic HSA queue injection requires dynamic libhsa";
#else
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 1;
  ScopedNativeAqlLibhsa native_aql_libhsa;
  IREE_ASSERT_OK(native_aql_libhsa.Initialize(&libhsa_));
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options, native_aql_libhsa.libhsa()));

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  const iree_hal_amdgpu_gfxip_version_t gfx10_version = {
      /*.major=*/10,
      /*.minor=*/0,
      /*.stepping=*/0,
  };
  ScopedAgentTargetVersionOverride target_override(physical_device,
                                                   gfx10_version);

  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.device(), /*physical_device_ordinal=*/0, &queue_topology));
  ASSERT_EQ(queue_topology.native_compute_unit_mask_alignment, 2u);
  if (queue_topology.native_compute_unit_mask_partition_count >=
      queue_topology.native_compute_unit_count) {
    GTEST_SKIP() << "device is too small to clear one bit while preserving "
                    "every hardware partition";
  }

  std::vector<uint32_t> aligned_mask =
      MakeFullNativeMask(queue_topology.native_compute_unit_count);
  std::vector<uint32_t> misaligned_mask = aligned_mask;
  misaligned_mask[0] &= ~UINT32_C(1);
  const iree_host_size_t private_queue_ordinal =
      queue_topology.first_private_physical_queue_ordinal;
  const iree_host_size_t initial_host_queue_count =
      physical_device->host_queue_count;
  ScopedQueueCreateProbe queue_create_probe(logical_device);

  iree_hal_queue_affinity_t queue_affinity = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_experimental_execution_queue_configure(
                            test_device.device(), /*physical_device_ordinal=*/0,
                            private_queue_ordinal, misaligned_mask.size() * 32u,
                            misaligned_mask.data(), &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);
  EXPECT_EQ(queue_create_probe.call_count(), 0u);
  EXPECT_EQ(physical_device->host_queue_count, initial_host_queue_count);
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, private_queue_ordinal));
  EXPECT_EQ(
      iree_atomic_load(&logical_device->configured_execution_queue_affinity,
                       iree_memory_order_acquire),
      0);

  queue_create_probe.ResetCallCount();
  queue_affinity = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amdgpu_experimental_execution_queue_configure(
                            test_device.device(), /*physical_device_ordinal=*/0,
                            private_queue_ordinal, aligned_mask.size() * 32u,
                            aligned_mask.data(), &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);
  EXPECT_EQ(queue_create_probe.call_count(), 1u);
  EXPECT_EQ(physical_device->host_queue_count, initial_host_queue_count);
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, private_queue_ordinal));
  EXPECT_EQ(
      iree_atomic_load(&logical_device->configured_execution_queue_affinity,
                       iree_memory_order_acquire),
      0);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

TEST_F(AmdgpuExperimentalApiTest,
       RejectsUndefinedMaskAlignmentBeforeQueueCreation) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  GTEST_SKIP() << "deterministic HSA queue injection requires dynamic libhsa";
#else
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 1;
  ScopedNativeAqlLibhsa native_aql_libhsa;
  IREE_ASSERT_OK(native_aql_libhsa.Initialize(&libhsa_));
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options, native_aql_libhsa.libhsa()));

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  const iree_hal_amdgpu_gfxip_version_t undefined_version = {
      /*.major=*/13,
      /*.minor=*/0,
      /*.stepping=*/0,
  };
  ScopedAgentTargetVersionOverride target_override(physical_device,
                                                   undefined_version);
  ScopedQueueCreateProbe queue_create_probe(logical_device);

  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology = {
      /*.first_private_physical_queue_ordinal=*/1,
      /*.private_physical_queue_count=*/1,
      /*.native_compute_unit_count=*/1,
      /*.native_compute_unit_mask_alignment=*/1,
      /*.native_compute_unit_mask_partition_count=*/1,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_experimental_execution_queue_query(
                            test_device.device(), /*physical_device_ordinal=*/0,
                            &queue_topology));
  EXPECT_EQ(queue_topology.first_private_physical_queue_ordinal, 0u);
  EXPECT_EQ(queue_topology.private_physical_queue_count, 0u);
  EXPECT_EQ(queue_topology.native_compute_unit_count, 0u);
  EXPECT_EQ(queue_topology.native_compute_unit_mask_alignment, 0u);
  EXPECT_EQ(queue_topology.native_compute_unit_mask_partition_count, 0u);
  EXPECT_EQ(queue_create_probe.call_count(), 0u);

  std::vector<uint32_t> mask =
      MakeFullNativeMask(physical_device->compute_unit_count);
  const iree_host_size_t private_queue_ordinal =
      physical_device->host_queue_ordinary_count;
  const iree_host_size_t initial_host_queue_count =
      physical_device->host_queue_count;
  iree_hal_queue_affinity_t queue_affinity = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_experimental_execution_queue_configure(
                            test_device.device(), /*physical_device_ordinal=*/0,
                            private_queue_ordinal, mask.size() * 32u,
                            mask.data(), &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);
  EXPECT_EQ(queue_create_probe.call_count(), 0u);
  EXPECT_EQ(physical_device->host_queue_count, initial_host_queue_count);
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, private_queue_ordinal));
  EXPECT_EQ(
      iree_atomic_load(&logical_device->configured_execution_queue_affinity,
                       iree_memory_order_acquire),
      0);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

TEST_F(AmdgpuExperimentalApiTest,
       SetterFailureDestroysUnpublishedQueueAndPermitsRetry) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  GTEST_SKIP() << "deterministic HSA setter injection requires dynamic libhsa";
#else
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 1;

  iree_hal_amdgpu_libhsa_t hooked_libhsa;
  IREE_ASSERT_OK(iree_hal_amdgpu_libhsa_copy(&libhsa_, &hooked_libhsa));
  hooked_libhsa.hsa_amd_queue_cu_set_mask = RejectQueueCuMask;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options, &hooked_libhsa));
  iree_hal_amdgpu_libhsa_deinitialize(&hooked_libhsa);

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.device(), /*physical_device_ordinal=*/0, &queue_topology));
  std::vector<uint32_t> mask =
      MakeFullNativeMask(queue_topology.native_compute_unit_count);
  iree_hal_queue_affinity_t queue_affinity = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_experimental_execution_queue_configure(
                            test_device.device(), /*physical_device_ordinal=*/0,
                            queue_topology.first_private_physical_queue_ordinal,
                            mask.size() * 32u, mask.data(), &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);
  EXPECT_EQ(physical_device->host_queue_count,
            physical_device->host_queue_ordinary_count);
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, queue_topology.first_private_physical_queue_ordinal));
  EXPECT_EQ(
      iree_atomic_load(&logical_device->configured_execution_queue_affinity,
                       iree_memory_order_acquire),
      0);

  logical_device->system->libhsa.hsa_amd_queue_cu_set_mask =
      libhsa_.hsa_amd_queue_cu_set_mask;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.device(), /*physical_device_ordinal=*/0,
      queue_topology.first_private_physical_queue_ordinal, mask.size() * 32u,
      mask.data(), &queue_affinity));
  EXPECT_NE(queue_affinity, 0u);
  IREE_EXPECT_OK(SubmitFillAndWait(test_device.device(), queue_affinity));
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

TEST_F(AmdgpuExperimentalApiTest,
       RejectsRuntimeWithoutProvableExactMaskInitialization) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 1;

  iree_hal_amdgpu_libhsa_t inexact_libhsa;
  IREE_ASSERT_OK(iree_hal_amdgpu_libhsa_copy(&libhsa_, &inexact_libhsa));
  inexact_libhsa.exact_queue_cu_mask_supported = false;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options, &inexact_libhsa));
  iree_hal_amdgpu_libhsa_deinitialize(&inexact_libhsa);

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.device(), /*physical_device_ordinal=*/0, &queue_topology));
  std::vector<uint32_t> mask =
      MakeFullNativeMask(queue_topology.native_compute_unit_count);
  iree_hal_queue_affinity_t queue_affinity = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amdgpu_experimental_execution_queue_configure(
                            test_device.device(), /*physical_device_ordinal=*/0,
                            queue_topology.first_private_physical_queue_ordinal,
                            mask.size() * 32u, mask.data(), &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, queue_topology.first_private_physical_queue_ordinal));
}

TEST_F(AmdgpuExperimentalApiTest,
       ProfilingExcludesConfigurationUntilTheSessionEnds) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 1;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options));

  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];
  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.device(), /*physical_device_ordinal=*/0, &queue_topology));
  std::vector<uint32_t> mask =
      MakeFullNativeMask(queue_topology.native_compute_unit_count);

  iree::hal::cts::TestProfileSink sink;
  iree::hal::cts::TestProfileSinkInitialize(&sink);
  iree::hal::cts::DeviceProfilingScope profiling(test_device.device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 iree::hal::cts::TestProfileSinkAsBase(&sink)));

  iree_hal_queue_affinity_t queue_affinity = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amdgpu_experimental_execution_queue_configure(
                            test_device.device(), /*physical_device_ordinal=*/0,
                            queue_topology.first_private_physical_queue_ordinal,
                            mask.size() * 32u, mask.data(), &queue_affinity));
  EXPECT_EQ(queue_affinity, 0u);
  EXPECT_FALSE(iree_hal_amdgpu_physical_device_host_queue_is_initialized(
      physical_device, queue_topology.first_private_physical_queue_ordinal));

  IREE_ASSERT_OK(profiling.End());
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.device(), /*physical_device_ordinal=*/0,
      queue_topology.first_private_physical_queue_ordinal, mask.size() * 32u,
      mask.data(), &queue_affinity));
  EXPECT_NE(queue_affinity, 0u);
  IREE_EXPECT_OK(SubmitFillAndWait(test_device.device(), queue_affinity));
}

TEST_F(AmdgpuExperimentalApiTest,
       ConcurrentConfigurationPublishesExactlyOneQueue) {
  iree_hal_amdgpu_aql_queue_execution_mode_t execution_mode;
  IREE_ASSERT_OK(iree_hal_amdgpu_query_aql_queue_execution_mode(
      &libhsa_, topology_.gpu_agents[0], &execution_mode));
  if (execution_mode != IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_NATIVE) {
    GTEST_SKIP() << "fixed-mask queues require native GPU-consumed AQL";
  }

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.experimental_execution_queue_count = 1;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options));

  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];
  ASSERT_GT(physical_device->compute_unit_count, 0u);
  const iree_host_size_t mask_word_count =
      (physical_device->compute_unit_count + 31u) / 32u;
  std::vector<uint32_t> full_mask(mask_word_count, UINT32_MAX);
  const uint32_t tail_bit_count = physical_device->compute_unit_count % 32u;
  if (tail_bit_count != 0) {
    full_mask.back() = (UINT32_C(1) << tail_bit_count) - 1u;
  }
  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.device(), /*physical_device_ordinal=*/0, &queue_topology));

  std::array<iree_status_t, 2> statuses = {iree_ok_status(), iree_ok_status()};
  std::array<iree_hal_queue_affinity_t, 2> queue_affinities = {0, 0};
  std::array<std::thread, 2> threads;
  for (iree_host_size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i]() {
      statuses[i] = iree_hal_amdgpu_experimental_execution_queue_configure(
          test_device.device(), /*physical_device_ordinal=*/0,
          /*physical_queue_ordinal=*/
          queue_topology.first_private_physical_queue_ordinal,
          mask_word_count * 32u, full_mask.data(), &queue_affinities[i]);
    });
  }
  for (std::thread& thread : threads) thread.join();

  iree_host_size_t success_count = 0;
  iree_host_size_t already_exists_count = 0;
  iree_hal_queue_affinity_t configured_queue_affinity = 0;
  for (iree_host_size_t i = 0; i < statuses.size(); ++i) {
    const iree_status_code_t status_code = iree_status_code(statuses[i]);
    if (status_code == IREE_STATUS_OK) {
      ++success_count;
      configured_queue_affinity = queue_affinities[i];
    } else if (status_code == IREE_STATUS_ALREADY_EXISTS) {
      ++already_exists_count;
      EXPECT_EQ(queue_affinities[i], 0u);
    }
    iree_status_free(statuses[i]);
  }
  EXPECT_EQ(success_count, 1u);
  EXPECT_EQ(already_exists_count, 1u);
  ASSERT_NE(configured_queue_affinity, 0u);
  IREE_EXPECT_OK(
      SubmitFillAndWait(test_device.device(), configured_queue_affinity));
}

}  // namespace
}  // namespace iree::hal::amdgpu
