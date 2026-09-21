// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "build_tools/vulkan/testing/buffer.h"
#include "build_tools/vulkan/testing/compute.h"
#include "build_tools/vulkan/testing/context.h"
#include "build_tools/vulkan/testing/submission.h"
#include "build_tools/vulkan/testing/testdata/transform_spv.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

namespace native = ::iree::vulkan::testing;

// Counts actual caller-supplied HAL allocations across worker threads. Native
// reuse alone would not detect an accidentally retained HAL device or service.
class CountingAllocator {
 public:
  iree_allocator_t value() { return {this, Control}; }
  size_t live_allocations() const {
    return live_allocations_.load(std::memory_order_relaxed);
  }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* state = static_cast<CountingAllocator*>(self);
    const bool had_allocation = (command == IREE_ALLOCATOR_COMMAND_REALLOC ||
                                 command == IREE_ALLOCATOR_COMMAND_FREE) &&
                                *inout_pointer != nullptr;
    const iree_allocator_t delegate = iree_allocator_system();
    iree_status_t status =
        delegate.ctl(delegate.self, command, params, inout_pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE && had_allocation) {
        state->live_allocations_.fetch_sub(1, std::memory_order_relaxed);
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_allocation &&
                 *inout_pointer) {
        state->live_allocations_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return status;
  }

  // Number of successful allocations not yet freed through this allocator.
  std::atomic<size_t> live_allocations_{0};
};

enum class WrapMode { kSupported, kMissingTimeline };

class BorrowedDeviceTest : public ::testing::TestWithParam<WrapMode> {
 protected:
  static constexpr size_t kWordCount = 1024;
  static constexpr size_t kByteLength = kWordCount * sizeof(uint32_t);

  void SetUp() override {
    auto loaded = instance_.Load();
    if (!loaded && !instance_.loader_available()) {
      GTEST_SKIP() << loaded.message();
    }
    ASSERT_TRUE(loaded);
    VkApplicationInfo application = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    const VkResult result = instance_.Create(instance_info);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
      GTEST_SKIP() << "No Vulkan 1.3 implementation available";
    }
    ASSERT_TRUE(native::CheckResult(result, "vkCreateInstance"));

    const auto enumerate = instance_.Get<PFN_vkEnumeratePhysicalDevices>(
        "vkEnumeratePhysicalDevices");
    uint32_t count = 0;
    ASSERT_TRUE(
        native::CheckResult(enumerate(instance_.handle(), &count, nullptr),
                            "vkEnumeratePhysicalDevices"));
    std::vector<VkPhysicalDevice> physical_devices(count);
    ASSERT_TRUE(native::CheckResult(
        enumerate(instance_.handle(), &count, physical_devices.data()),
        "vkEnumeratePhysicalDevices"));
    physical_devices.resize(count);
    for (VkPhysicalDevice candidate : physical_devices) {
      VkPhysicalDeviceProperties properties;
      instance_.Get<PFN_vkGetPhysicalDeviceProperties>(
          "vkGetPhysicalDeviceProperties")(candidate, &properties);
      if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ||
          properties.apiVersion < VK_API_VERSION_1_3) {
        continue;
      }
      VkPhysicalDeviceVulkan13Features features13 = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
      VkPhysicalDeviceVulkan12Features features12 = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      features12.pNext = &features13;
      VkPhysicalDeviceFeatures2 features = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      features.pNext = &features12;
      instance_.Get<PFN_vkGetPhysicalDeviceFeatures2>(
          "vkGetPhysicalDeviceFeatures2")(candidate, &features);
      if (!features12.bufferDeviceAddress || !features12.scalarBlockLayout ||
          !features12.timelineSemaphore || !features13.synchronization2) {
        continue;
      }
      auto query_families =
          instance_.Get<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
              "vkGetPhysicalDeviceQueueFamilyProperties");
      uint32_t family_count = 0;
      query_families(candidate, &family_count, nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      query_families(candidate, &family_count, families.data());
      for (uint32_t i = 0; i < family_count; ++i) {
        if (families[i].queueCount &&
            (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          physical_device_ = candidate;
          queue_family_ = i;
          RecordProperty("vulkan_device", properties.deviceName);
          break;
        }
      }
      if (physical_device_) {
        break;
      }
    }
    if (!physical_device_) {
      GTEST_SKIP()
          << "No Vulkan 1.3 GPU with the HAL baseline and compute queue";
    }

    VkPhysicalDeviceVulkan13Features enabled13 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enabled13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enabled12 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enabled12.pNext = &enabled13;
    enabled12.bufferDeviceAddress = VK_TRUE;
    enabled12.scalarBlockLayout = VK_TRUE;
    enabled12.timelineSemaphore = GetParam() == WrapMode::kSupported;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &enabled12;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    ASSERT_TRUE(device_.Initialize(instance_, physical_device_, device_info));

    // Record the exact enabled-device contract, not physical-device support.
    external_.enabled_features.general =
        IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
        IREE_HAL_VULKAN_FEATURE_ENABLE_SCALAR_BLOCK_LAYOUT |
        IREE_HAL_VULKAN_FEATURE_ENABLE_SYNCHRONIZATION2;
    if (enabled12.timelineSemaphore) {
      external_.enabled_features.general |=
          IREE_HAL_VULKAN_FEATURE_ENABLE_TIMELINE_SEMAPHORES;
    }
    external_.compute_queue_set.queue_family_index = queue_family_;
    external_.compute_queue_set.queue_indices = 1;

    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = kByteLength;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ASSERT_TRUE(buffer_.Initialize(device_, buffer_info,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                   VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT));
    ASSERT_TRUE(buffer_.Map());
    for (size_t i = 0; i < kWordCount; ++i) {
      expected_[i] = 0x9e3779b9u * static_cast<uint32_t>(i + 1);
    }
    std::memcpy(buffer_.mapped(), expected_.data(), kByteLength);
    ASSERT_TRUE(buffer_.Flush());
    const auto& shader = iree_vulkan_test_transform_create()[0];
    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(Parameters)};
    ASSERT_TRUE(compute_.Initialize(
        device_, shader.size, reinterpret_cast<const uint32_t*>(shader.data),
        "main", 1, &range));
    ASSERT_TRUE(submission_.Initialize(device_, queue_family_, 0));
  }

  void TearDown() override {
    ReleaseHal();
    EXPECT_EQ(allocator_.live_allocations(), 0u);
  }

  void RunNative(uint32_t salt, uint32_t count) {
    ASSERT_TRUE(submission_.Begin());
    const VkCommandBuffer command = submission_.command_buffer();
    const auto barrier =
        device_.Get<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    VkMemoryBarrier acquire = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    acquire.srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    acquire.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier(command,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &acquire, 0, nullptr, 0,
            nullptr);
    device_.Get<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(
        command, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.handle());
    // Prefix and suffix guards expose both an incorrect BDA offset and an
    // incorrect shader count. The second dispatch overlaps and extends the
    // first.
    const Parameters parameters = {buffer_.device_address() + 64, salt, count};
    ASSERT_NE(parameters.address, 64u);
    device_.Get<PFN_vkCmdPushConstants>("vkCmdPushConstants")(
        command, compute_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(parameters), &parameters);
    device_.Get<PFN_vkCmdDispatch>("vkCmdDispatch")(command, (count + 63) / 64,
                                                    1, 1);
    VkMemoryBarrier release = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    release.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    release.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &release, 0, nullptr, 0, nullptr);
    ASSERT_TRUE(submission_.SubmitAndWait());
    ASSERT_TRUE(buffer_.Invalidate());
    for (uint32_t i = 0; i < count; ++i) {
      auto& word = expected_[16 + i];
      word = (word ^ salt) * 1664525u + 1013904223u + i;
    }
    const auto* actual = static_cast<const uint32_t*>(buffer_.mapped());
    for (size_t i = 0; i < kWordCount; ++i) {
      ASSERT_EQ(actual[i], expected_[i]) << "word " << i;
    }
  }

  void Wrap() {
    IREE_ASSERT_OK(context_.Initialize(allocator_.value()));
    ASSERT_GT(allocator_.live_allocations(), 0u);
    IREE_ASSERT_OK(iree_hal_vulkan_syms_create(
        reinterpret_cast<void*>(instance_.get_instance_proc_addr()),
        allocator_.value(), &syms_));
    iree_hal_vulkan_device_options_t options;
    iree_hal_vulkan_device_options_initialize(&options);
    iree_status_t status = iree_hal_vulkan_wrap_device(
        iree_make_cstring_view("borrowed-vulkan"), &options, context_.params(),
        syms_, instance_.handle(), physical_device_, device_.handle(),
        &external_, allocator_.value(), &hal_device_);
    if (GetParam() == WrapMode::kMissingTimeline) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);
      ASSERT_EQ(hal_device_, nullptr);
    } else {
      IREE_ASSERT_OK(status);
      ASSERT_NE(hal_device_, nullptr);
      IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
          hal_device_, context_.frontier_tracker(), allocator_.value(),
          &group_));
    }
  }

  void RunHal() {
    cts::Ref<iree_hal_buffer_t> source;
    cts::Ref<iree_hal_buffer_t> target;
    cts::Ref<iree_hal_semaphore_t> semaphore;
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.queue_family_affinity = iree_hal_make_queue_family_affinity(0);
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(hal_device_), params, kByteLength,
        source.out()));
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(hal_device_), params, kByteLength,
        target.out()));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        hal_device_, params.queue_family_affinity, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore.out()));
    iree_hal_semaphore_t* semaphore_handle = semaphore.get();
    uint64_t filled = 1, copied = 2, downloaded = 3;
    const iree_hal_semaphore_list_t fill_signal = {1, &semaphore_handle,
                                                   &filled};
    const iree_hal_semaphore_list_t copy_signal = {1, &semaphore_handle,
                                                   &copied};
    const iree_hal_semaphore_list_t download_signal = {1, &semaphore_handle,
                                                       &downloaded};
    iree_hal_queue_t* queue = iree_hal_device_queue(hal_device_, 0, 0);
    ASSERT_NE(queue, nullptr);
    const uint32_t pattern = 0x91a7b36d;
    std::array<uint32_t, kWordCount> result = {};
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queue, iree_hal_semaphore_list_empty(), fill_signal, source, 0,
        kByteLength, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_copy(queue, fill_signal, copy_signal, source,
                                       0, target, 0, kByteLength,
                                       IREE_HAL_COPY_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_download(queue, copy_signal, download_signal,
                                           target, 0, result.data(),
                                           kByteLength));
    iree_status_t status = iree_hal_semaphore_list_wait(
        download_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    if (!iree_status_is_ok(status)) {
      // A failed completion witness cannot authorize freeing the host
      // destination.
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      std::abort();
    }
    for (size_t i = 0; i < kWordCount; ++i) {
      ASSERT_EQ(result[i], pattern) << "word " << i;
    }
  }

  void ReleaseHal() {
    iree_hal_device_group_release(group_);
    group_ = nullptr;
    iree_hal_device_release(hal_device_);
    hal_device_ = nullptr;
    iree_hal_vulkan_syms_release(syms_);
    syms_ = nullptr;
    context_.Deinitialize();
  }

  // Independent count of caller-owned HAL state, including group and services.
  CountingAllocator allocator_;

 private:
  struct Parameters {
    // Vulkan-local address of the first uint in the active view.
    VkDeviceAddress address;
    // Changing transform constant.
    uint32_t salt;
    // Number of uints accessible through address.
    uint32_t count;
  };
  static_assert(sizeof(Parameters) == 16);
  static_assert(offsetof(Parameters, salt) == 8);
  static_assert(offsetof(Parameters, count) == 12);

  // Outer native ownership survives all HAL setup, execution and destruction.
  native::Instance instance_;
  // Native device, destroyed after every child below.
  native::Device device_;
  // Independently owned BDA payload and mapping.
  native::Buffer buffer_;
  // Original native shader pipeline reused after HAL teardown.
  native::Compute compute_;
  // Original command storage, destroyed before its referenced native resources.
  native::Submission submission_;
  // Physical device selected from the owned instance.
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  // Compute-capable family from the selected physical device.
  uint32_t queue_family_ = 0;
  // Exact enabled-device and queue facts passed to the public wrap API.
  iree_hal_vulkan_external_device_params_t external_ = {};
  // Independent CPU oracle, including the complete allocation's guard words.
  std::array<uint32_t, kWordCount> expected_ = {};
  // Fixture-owned services; no process-level CTS cache participates.
  cts::DeviceCreateContext context_;
  // Owned external symbol wrapper released before native reuse.
  iree_hal_vulkan_syms_t* syms_ = nullptr;
  // Owned HAL wrapper borrowing the native handles.
  iree_hal_device_t* hal_device_ = nullptr;
  // Owned topology group retaining the wrapper during HAL queue work.
  iree_hal_device_group_t* group_ = nullptr;
};

TEST_P(BorrowedDeviceTest, NativeOwnershipSurvivesHalLifetime) {
  ASSERT_NO_FATAL_FAILURE(RunNative(0x163abc, 257));
  ASSERT_NO_FATAL_FAILURE(Wrap());
  if (GetParam() == WrapMode::kSupported) {
    ASSERT_NO_FATAL_FAILURE(RunHal());
  }
  ReleaseHal();
  ASSERT_EQ(allocator_.live_allocations(), 0u);
  ASSERT_NO_FATAL_FAILURE(RunNative(0x92bcde, 511));
}

INSTANTIATE_TEST_SUITE_P(ExternalVulkan, BorrowedDeviceTest,
                         ::testing::Values(WrapMode::kSupported,
                                           WrapMode::kMissingTimeline),
                         [](const ::testing::TestParamInfo<WrapMode>& info) {
                           return info.param == WrapMode::kSupported
                                      ? "RequiredCoreFeatures"
                                      : "MissingTimeline";
                         });

}  // namespace
}  // namespace iree::hal::vulkan
