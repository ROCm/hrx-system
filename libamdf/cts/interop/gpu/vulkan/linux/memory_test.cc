// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "build_tools/vulkan/testing/buffer.h"
#include "build_tools/vulkan/testing/compute.h"
#include "build_tools/vulkan/testing/context.h"
#include "build_tools/vulkan/testing/submission.h"
#include "build_tools/vulkan/testing/testdata/transform_spv.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/gpu/pm4_commands.h"

namespace {

namespace native = ::iree::vulkan::testing;

constexpr uint64_t kByteLength = 65536;
constexpr size_t kWordCount = kByteLength / sizeof(uint32_t);
constexpr size_t kViewWordOffset = 1024;
constexpr size_t kCopyWordCount = 128;
constexpr size_t kCommandWordCapacity = 1024;
constexpr VkBufferUsageFlags kBufferUsage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr const char* kDeviceExtensions[] = {
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
    VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
};

struct Parameters {
  // This import's Vulkan address plus the current logical view offset.
  VkDeviceAddress address;
  // Generation-dependent input to the numerical transform.
  uint32_t salt;
  // Number of uint elements accessible through address.
  uint32_t count;
};
static_assert(sizeof(Parameters) == 16);
static_assert(offsetof(Parameters, address) == 0);
static_assert(offsetof(Parameters, salt) == 8);
static_assert(offsetof(Parameters, count) == 12);

// Owns one imported reference to the DMA-BUF payload. The exported libamdf
// value is borrowed; Vulkan consumes only the duplicate on successful import.
class ImportedBuffer {
 public:
  ImportedBuffer() = default;
  ~ImportedBuffer() { Reset(); }
  ImportedBuffer(const ImportedBuffer&) = delete;
  ImportedBuffer& operator=(const ImportedBuffer&) = delete;

  void Initialize(const native::Device& device,
                  const amdf_external_memory_t& transport,
                  uint64_t allocation_byte_length,
                  VkExternalMemoryFeatureFlags external_features) {
    device_ = &device;
    ASSERT_EQ(transport.type, AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
    ASSERT_EQ(transport.source_byte_offset, 0u);
    ASSERT_EQ(transport.byte_length, kByteLength);
    VkExternalMemoryBufferCreateInfo external = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.pNext = &external;
    buffer_info.size = kByteLength;
    buffer_info.usage = kBufferUsage;
    ASSERT_EQ(device.Get<PFN_vkCreateBuffer>("vkCreateBuffer")(
                  device.handle(), &buffer_info, nullptr, &buffer_),
              VK_SUCCESS);
    VkMemoryDedicatedRequirements dedicated_requirements = {
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 requirements = {
        VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    requirements.pNext = &dedicated_requirements;
    VkBufferMemoryRequirementsInfo2 requirements_info = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};
    requirements_info.buffer = buffer_;
    device.Get<PFN_vkGetBufferMemoryRequirements2>(
        "vkGetBufferMemoryRequirements2")(device.handle(), &requirements_info,
                                          &requirements);
    ASSERT_LE(requirements.memoryRequirements.size, allocation_byte_length);
    VkMemoryFdPropertiesKHR properties = {
        VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    ASSERT_EQ(
        device.Get<PFN_vkGetMemoryFdPropertiesKHR>(
            "vkGetMemoryFdPropertiesKHR")(
            device.handle(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
            transport.payload.file_descriptor, &properties),
        VK_SUCCESS);
    const uint32_t allowed_types =
        properties.memoryTypeBits &
        requirements.memoryRequirements.memoryTypeBits;
    ASSERT_NE(allowed_types, 0u);
    uint32_t memory_type = 0;
    while ((allowed_types & (1u << memory_type)) == 0) {
      ++memory_type;
    }

    VkMemoryDedicatedAllocateInfo dedicated = {
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer_;
    VkMemoryAllocateFlagsInfo flags = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    if ((external_features & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) ||
        dedicated_requirements.requiresDedicatedAllocation) {
      flags.pNext = &dedicated;
    }
    VkImportMemoryFdInfoKHR import = {
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    import.pNext = &flags;
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    import.fd = dup(transport.payload.file_descriptor);
    ASSERT_GE(import.fd, 0);
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.pNext = &import;
    allocation.allocationSize = allocation_byte_length;
    allocation.memoryTypeIndex = memory_type;
    const VkResult result = device.Get<PFN_vkAllocateMemory>(
        "vkAllocateMemory")(device.handle(), &allocation, nullptr, &memory_);
    if (result != VK_SUCCESS) {
      EXPECT_EQ(close(import.fd), 0);
    }
    ASSERT_EQ(result, VK_SUCCESS);
    ASSERT_EQ(device.Get<PFN_vkBindBufferMemory>("vkBindBufferMemory")(
                  device.handle(), buffer_, memory_, 0),
              VK_SUCCESS);
    VkBufferDeviceAddressInfo address = {
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address.buffer = buffer_;
    address_ = device.Get<PFN_vkGetBufferDeviceAddress>(
        "vkGetBufferDeviceAddress")(device.handle(), &address);
    ASSERT_NE(address_, 0u);
    ASSERT_EQ(address_ % sizeof(uint32_t), 0u);
  }

  void Reset() {
    if (buffer_) {
      device_->Get<PFN_vkDestroyBuffer>("vkDestroyBuffer")(
          device_->handle(), std::exchange(buffer_, VK_NULL_HANDLE), nullptr);
    }
    if (memory_) {
      device_->Get<PFN_vkFreeMemory>("vkFreeMemory")(
          device_->handle(), std::exchange(memory_, VK_NULL_HANDLE), nullptr);
    }
    address_ = 0;
  }

  VkBuffer handle() const { return buffer_; }
  VkDeviceAddress address() const { return address_; }

 private:
  // Borrowed parent, outliving all imported aliases.
  const native::Device* device_ = nullptr;
  // Owned external buffer, destroyed before its imported allocation.
  VkBuffer buffer_ = VK_NULL_HANDLE;
  // Owned independent reference to the imported payload.
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  // Address queried for this bound buffer; never copied from a native alias.
  VkDeviceAddress address_ = 0;
};

struct MappedMemory {
  // Owned allocation and its explicit native device access.
  amdf_memory_t* memory = nullptr;
  // Owned host view, released before memory.
  amdf_host_mapping_t* mapping = nullptr;
  // Borrowed coherent host address from mapping.
  uint32_t* host = nullptr;
  // Native GPU address of memory, independent of Vulkan buffer addresses.
  uint64_t address = 0;
};

struct VulkanCandidate {
  // Borrowed device exposed by the owned Vulkan instance.
  VkPhysicalDevice device;
  // DRM identity used to pair the two APIs on the same physical GPU.
  VkPhysicalDeviceDrmPropertiesEXT drm;
  // Compute queue requested when the matched device is created.
  uint32_t queue_family;
  // Exact DMA-BUF import features for kBufferUsage.
  VkExternalMemoryFeatureFlags external_features;
};

class VulkanMemoryInteropTest : public GpuDeviceFixture {
 protected:
  void FindVulkanCandidates() {
    const auto loaded = vulkan_instance_.Load();
    if (!loaded && !vulkan_instance_.loader_available()) {
      GTEST_SKIP() << loaded.message();
    }
    ASSERT_TRUE(loaded);
    VkApplicationInfo application = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &application;
    const VkResult result = vulkan_instance_.Create(create);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
      GTEST_SKIP() << "No Vulkan 1.2 implementation";
    }
    ASSERT_EQ(result, VK_SUCCESS);
    const auto enumerate = vulkan_instance_.Get<PFN_vkEnumeratePhysicalDevices>(
        "vkEnumeratePhysicalDevices");
    uint32_t count = 0;
    ASSERT_EQ(enumerate(vulkan_instance_.handle(), &count, nullptr),
              VK_SUCCESS);
    std::vector<VkPhysicalDevice> devices(count);
    ASSERT_EQ(enumerate(vulkan_instance_.handle(), &count, devices.data()),
              VK_SUCCESS);
    for (VkPhysicalDevice device : devices) {
      const auto enumerate_extensions =
          vulkan_instance_.Get<PFN_vkEnumerateDeviceExtensionProperties>(
              "vkEnumerateDeviceExtensionProperties");
      uint32_t extension_count = 0;
      ASSERT_EQ(
          enumerate_extensions(device, nullptr, &extension_count, nullptr),
          VK_SUCCESS);
      std::vector<VkExtensionProperties> extensions(extension_count);
      ASSERT_EQ(enumerate_extensions(device, nullptr, &extension_count,
                                     extensions.data()),
                VK_SUCCESS);
      const auto supports = [&](const char* name) {
        return std::any_of(extensions.begin(), extensions.end(),
                           [&](const VkExtensionProperties& extension) {
                             return std::strcmp(extension.extensionName,
                                                name) == 0;
                           });
      };
      if (!supports(VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) ||
          !std::all_of(std::begin(kDeviceExtensions),
                       std::end(kDeviceExtensions), supports)) {
        continue;
      }
      VulkanCandidate candidate = {};
      candidate.device = device;
      candidate.drm.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT;
      VkPhysicalDeviceProperties2 properties = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      properties.pNext = &candidate.drm;
      vulkan_instance_.Get<PFN_vkGetPhysicalDeviceProperties2>(
          "vkGetPhysicalDeviceProperties2")(device, &properties);
      if (!candidate.drm.hasRender ||
          properties.properties.apiVersion < VK_API_VERSION_1_2) {
        continue;
      }
      VkPhysicalDeviceVulkan12Features features12 = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      VkPhysicalDeviceFeatures2 features = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      features.pNext = &features12;
      vulkan_instance_.Get<PFN_vkGetPhysicalDeviceFeatures2>(
          "vkGetPhysicalDeviceFeatures2")(device, &features);
      if (!features12.bufferDeviceAddress) {
        continue;
      }
      VkPhysicalDeviceExternalBufferInfo external_query = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
      external_query.usage = kBufferUsage;
      external_query.handleType =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
      VkExternalBufferProperties external_properties = {
          VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
      vulkan_instance_.Get<PFN_vkGetPhysicalDeviceExternalBufferProperties>(
          "vkGetPhysicalDeviceExternalBufferProperties")(
          device, &external_query, &external_properties);
      candidate.external_features =
          external_properties.externalMemoryProperties.externalMemoryFeatures;
      if (!(candidate.external_features &
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
        continue;
      }
      const auto query_families =
          vulkan_instance_.Get<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
              "vkGetPhysicalDeviceQueueFamilyProperties");
      uint32_t family_count = 0;
      query_families(device, &family_count, nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      query_families(device, &family_count, families.data());
      for (uint32_t i = 0; i < family_count; ++i) {
        if (families[i].queueCount &&
            (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          candidate.queue_family = i;
          candidates_.push_back(candidate);
          break;
        }
      }
    }
    if (candidates_.empty()) {
      GTEST_SKIP()
          << "No Vulkan BDA compute device with DMA-BUF/foreign ownership";
    }
  }

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    amdf_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    amdf_status_t status = api_->endpoint_query_info(endpoint, &info);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (info.native_identity.type !=
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    const auto& identity = info.native_identity.value.linux_device;
    const auto candidate =
        std::find_if(candidates_.begin(), candidates_.end(),
                     [&](const VulkanCandidate& value) {
                       return value.drm.renderMajor == identity.major &&
                              value.drm.renderMinor == identity.minor;
                     });
    if (candidate == candidates_.end()) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    for (uint32_t i = 0; i < info.queue_family_count; ++i) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      status = api_->endpoint_query_queue_family_info(endpoint, i, &family);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      constexpr auto kRoles =
          AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL;
      constexpr auto kCacheOperations =
          AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
          family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 &&
          (family.format_features &
           AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR) &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) &&
          (family.roles & kRoles) == kRoles &&
          (family.cache_operations & kCacheOperations) == kCacheOperations &&
          (family.cache_transition_kinds &
           AMDF_CACHE_TRANSITION_KINDS_GLOBAL) &&
          (family.user_queue_capabilities &
           AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) &&
          (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) &&
          (family.priority_capabilities &
           AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL) &&
          family.maximum_ring_byte_length >=
              kCommandWordCapacity * sizeof(uint32_t)) {
        selected_ = *candidate;
        family_ = family;
        *out_matches = true;
        return AMDF_STATUS_OK;
      }
    }
    *out_matches = false;
    return AMDF_STATUS_OK;
  }

  void SetUp() override;
  void TearDown() override;
  void CreateMemory(uint64_t byte_length, MappedMemory& memory);
  void DestroyMemory(MappedMemory& memory);
  void DestroyNativeQueue();
  void RunVulkan(size_t alias, uint32_t salt, uint32_t view, uint32_t count);
  void CheckReadback(size_t alias);
  void BeginVulkan(size_t alias);
  void EndVulkan(size_t alias);
  void RunNative(uint32_t generation);
  void PublishNative();

  // Vulkan parents outlive all children; submission destruction checks pending
  // work before any imported backing can be destroyed.
  native::Instance vulkan_instance_;
  // Owned device with exactly the BDA feature and admitted external extensions.
  native::Device vulkan_device_;
  // Independent imported references; neither keeps the native source owner.
  std::array<ImportedBuffer, 2> aliases_;
  // Owned host-visible destination used only at explicit oracle boundaries.
  native::Buffer readback_;
  // Owned pipeline with the explicit Parameters ABI.
  native::Compute compute_;
  // Owned command storage and completion fence, destroyed before other
  // children.
  native::Submission submission_;
  // Supported physical devices considered before native activation.
  std::vector<VulkanCandidate> candidates_;
  // Native/Vulkan physical-device match and queried import requirements.
  VulkanCandidate selected_ = {};
  // Exact native family admitted before queue creation.
  amdf_queue_family_info_t family_ = {};
  // Original libamdf source and CPU seed view, released during the lifetime
  // test.
  MappedMemory backing_;
  // Separate coherent completion marker; no payload oracle accesses this view.
  MappedMemory control_;
  // Exported source descriptor, released as soon as both imports succeed.
  amdf_external_memory_t exported_ = {};
  // Owned native user queue, destroyed before its referenced memory.
  amdf_user_queue_t* queue_ = nullptr;
  // Owned host publication mapping of queue_.
  amdf_user_queue_mapping_t* queue_mapping_ = nullptr;
  // Queried ring/index/doorbell addresses belonging to queue_mapping_.
  amdf_user_queue_mapping_info_t queue_mapping_info_ = {};
  // Cumulative PM4 word index, including wrap padding.
  uint64_t frontier_ = 0;
  // Published native work whose consumption/completion is not yet established.
  bool native_pending_ = false;
  // CPU reference, updated independently of intermediate GPU observations.
  std::array<uint32_t, kWordCount> expected_ = {};
};

void VulkanMemoryInteropTest::CreateMemory(uint64_t byte_length,
                                           MappedMemory& memory) {
  const amdf_memory_device_access_t access = {
      device_,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  amdf_memory_create_info_t create = {};
  create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create.structure_size = sizeof(create);
  create.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  create.memory_profile_ordinal = FindMemoryProfileOrdinal(
      system_scope_,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT |
          AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      create.required_flags, access.requirements);
  ASSERT_NE(create.memory_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  amdf_memory_profile_t profile = {};
  profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  profile.structure_size = sizeof(profile);
  amdf_memory_access_capabilities_t capabilities = {};
  capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
  capabilities.structure_size = sizeof(capabilities);
  ASSERT_EQ(QueryMemoryProfile(system_scope_, create.memory_profile_ordinal,
                               access.requirements, &profile, &capabilities),
            AMDF_STATUS_OK);
  const auto* support_end =
      profile.external_memory_support + profile.external_memory_support_count;
  const auto* support =
      std::find_if(std::cbegin(profile.external_memory_support), support_end,
                   [](const amdf_external_memory_support_t& value) {
                     return value.type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
                   });
  ASSERT_NE(support, support_end);
  ASSERT_NE(support->flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT, 0u);
  create.access_count = 1;
  create.accesses = &access;
  create.byte_length = byte_length;
  create.minimum_alignment = 4096;
  ASSERT_EQ(api_->memory_create(system_scope_, &create, &memory.memory),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->memory_query_address(
                memory.memory, 0, AMDF_MEMORY_ADDRESS_GPU, &memory.address),
            AMDF_STATUS_OK);
  amdf_memory_map_info_t map = {};
  map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map.structure_size = sizeof(map);
  map.byte_length = byte_length;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  ASSERT_EQ(api_->memory_map(memory.memory, &map, &memory.mapping),
            AMDF_STATUS_OK);
  amdf_host_mapping_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(api_->host_mapping_query_info(memory.mapping, &info),
            AMDF_STATUS_OK);
  memory.host = static_cast<uint32_t*>(info.pointer);
  ASSERT_NE(memory.host, nullptr);
}

void VulkanMemoryInteropTest::SetUp() {
  ASSERT_NO_FATAL_FAILURE(FindVulkanCandidates());
  if (IsSkipped()) {
    return;
  }
  GpuDeviceFixture::SetUp();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }

  VkPhysicalDeviceVulkan12Features enabled = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  enabled.bufferDeviceAddress = VK_TRUE;
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info = {
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = selected_.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.pNext = &enabled;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = std::size(kDeviceExtensions);
  device_info.ppEnabledExtensionNames = kDeviceExtensions;
  ASSERT_TRUE(vulkan_device_.Initialize(vulkan_instance_, selected_.device,
                                        device_info));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kByteLength, backing_));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(4096, control_));
  for (size_t i = 0; i < kWordCount; ++i) {
    expected_[i] = 0x91870000u ^ (static_cast<uint32_t>(i) * 0x10203u);
  }
  std::memcpy(backing_.host, expected_.data(), kByteLength);
  *control_.host = 0;
  ASSERT_EQ(
      api_->host_mapping_cache_control(
          backing_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kByteLength),
      AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(control_.mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             sizeof(uint32_t)),
            AMDF_STATUS_OK);

  amdf_memory_export_info_t export_info = {};
  export_info.type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO;
  export_info.structure_size = sizeof(export_info);
  export_info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  export_info.byte_length = kByteLength;
  ASSERT_EQ(api_->memory_export(backing_.memory, &export_info, &exported_),
            AMDF_STATUS_OK);
  amdf_memory_info_t backing_info = {};
  backing_info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  backing_info.structure_size = sizeof(backing_info);
  ASSERT_EQ(api_->memory_query_info(backing_.memory, &backing_info),
            AMDF_STATUS_OK);
  for (auto& alias : aliases_) {
    ASSERT_NO_FATAL_FAILURE(alias.Initialize(
        vulkan_device_, exported_, backing_info.native_allocation_byte_length,
        selected_.external_features));
  }
  api_->external_memory_release(&exported_);

  VkBufferCreateInfo readback_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  readback_info.size = kByteLength;
  readback_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  ASSERT_TRUE(readback_.Initialize(vulkan_device_, readback_info,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0));
  ASSERT_TRUE(readback_.Map());
  const auto& shader = iree_vulkan_test_transform_create()[0];
  VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(Parameters)};
  ASSERT_TRUE(compute_.Initialize(
      vulkan_device_, shader.size,
      reinterpret_cast<const uint32_t*>(shader.data), "main", 1, &range));
  ASSERT_TRUE(
      submission_.Initialize(vulkan_device_, selected_.queue_family, 0));

  amdf_gpu_user_queue_create_info_t create_queue = {};
  create_queue.type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO;
  create_queue.structure_size = sizeof(create_queue);
  create_queue.queue_family_ordinal = family_.ordinal;
  create_queue.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  create_queue.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  create_queue.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  create_queue.ring_byte_length =
      std::max<uint64_t>(family_.minimum_ring_byte_length,
                         kCommandWordCapacity * sizeof(uint32_t));
  ASSERT_EQ(gpu_api_->user_queue_create(device_, &create_queue, &queue_),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->user_queue_map(queue_, nullptr, &queue_mapping_),
            AMDF_STATUS_OK);
  queue_mapping_info_.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
  queue_mapping_info_.structure_size = sizeof(queue_mapping_info_);
  ASSERT_EQ(
      api_->user_queue_mapping_query_info(queue_mapping_, &queue_mapping_info_),
      AMDF_STATUS_OK);
  ASSERT_EQ(queue_mapping_info_.index_bits, 64u);
  ASSERT_EQ(queue_mapping_info_.doorbell_bits, 64u);
  ASSERT_NE(queue_mapping_info_.ring_address, 0u);
  ASSERT_NE(queue_mapping_info_.write_index_address, 0u);
  ASSERT_NE(queue_mapping_info_.doorbell_address, 0u);
  ASSERT_EQ(queue_mapping_info_.ring_address % sizeof(uint32_t), 0u);
  ASSERT_EQ(queue_mapping_info_.write_index_address % sizeof(uint64_t), 0u);
  ASSERT_EQ(queue_mapping_info_.doorbell_address % sizeof(uint64_t), 0u);
  ASSERT_GE(queue_mapping_info_.ring_byte_length,
            create_queue.ring_byte_length);
  const uint64_t capacity =
      queue_mapping_info_.ring_byte_length / sizeof(uint32_t);
  ASSERT_EQ(capacity & (capacity - 1), 0u);
}

void VulkanMemoryInteropTest::DestroyMemory(MappedMemory& memory) {
  if (memory.mapping) {
    ASSERT_EQ(api_->host_mapping_destroy(memory.mapping), AMDF_STATUS_OK);
    memory.mapping = nullptr;
    memory.host = nullptr;
  }
  if (memory.memory) {
    ASSERT_EQ(api_->memory_destroy(std::exchange(memory.memory, nullptr)),
              AMDF_STATUS_OK);
    memory.address = 0;
  }
}

void VulkanMemoryInteropTest::DestroyNativeQueue() {
  if (native_pending_) {
    ADD_FAILURE()
        << "Native submission has not completed; backing is still reachable";
    std::abort();
  }
  if (queue_mapping_) {
    ASSERT_EQ(api_->user_queue_mapping_destroy(queue_mapping_), AMDF_STATUS_OK);
    queue_mapping_ = nullptr;
  }
  if (queue_) {
    ASSERT_EQ(api_->user_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
  }
}

void VulkanMemoryInteropTest::TearDown() {
  DestroyNativeQueue();
  if (queue_mapping_ || queue_) {
    ADD_FAILURE() << "Native queue teardown failed; backing is still reachable";
    std::abort();
  }
  if (api_) {
    api_->external_memory_release(&exported_);
    DestroyMemory(control_);
    DestroyMemory(backing_);
  }
  GpuDeviceFixture::TearDown();
}

void VulkanMemoryInteropTest::BeginVulkan(size_t alias) {
  ASSERT_TRUE(submission_.Begin());
  VkBufferMemoryBarrier acquire = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  acquire.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT;
  // The native endpoint provides no Vulkan driver UUID. FOREIGN covers the
  // external participant without assuming Vulkan driver provenance.
  acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
  acquire.dstQueueFamilyIndex = selected_.queue_family;
  acquire.buffer = aliases_[alias].handle();
  acquire.size = kByteLength;
  vulkan_device_.Get<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
      submission_.command_buffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, nullptr, 1, &acquire, 0, nullptr);
}

void VulkanMemoryInteropTest::EndVulkan(size_t alias) {
  VkBufferMemoryBarrier release = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  release.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT;
  release.srcQueueFamilyIndex = selected_.queue_family;
  release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
  release.buffer = aliases_[alias].handle();
  release.size = kByteLength;
  vulkan_device_.Get<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
      submission_.command_buffer(),
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &release, 0,
      nullptr);
  ASSERT_TRUE(submission_.SubmitAndWait());
}

void VulkanMemoryInteropTest::RunVulkan(size_t alias, uint32_t salt,
                                        uint32_t view, uint32_t count) {
  ASSERT_NO_FATAL_FAILURE(BeginVulkan(alias));
  const VkCommandBuffer command = submission_.command_buffer();
  vulkan_device_.Get<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(
      command, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.handle());
  const size_t offset = (view + 1) * kViewWordOffset;
  const Parameters parameters = {
      aliases_[alias].address() + offset * sizeof(uint32_t), salt, count};
  vulkan_device_.Get<PFN_vkCmdPushConstants>("vkCmdPushConstants")(
      command, compute_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
      sizeof(parameters), &parameters);
  vulkan_device_.Get<PFN_vkCmdDispatch>("vkCmdDispatch")(
      command, (count + 63) / 64, 1, 1);
  ASSERT_NO_FATAL_FAILURE(EndVulkan(alias));
  for (uint32_t i = 0; i < count; ++i) {
    auto& value = expected_[offset + i];
    value = (value ^ salt) * 1664525u + 1013904223u + i;
  }
}

void VulkanMemoryInteropTest::CheckReadback(size_t alias) {
  ASSERT_NO_FATAL_FAILURE(BeginVulkan(alias));
  const VkCommandBuffer command = submission_.command_buffer();
  const VkBufferCopy copy = {0, 0, kByteLength};
  vulkan_device_.Get<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(
      command, aliases_[alias].handle(), readback_.handle(), 1, &copy);
  VkBufferMemoryBarrier host = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host.buffer = readback_.handle();
  host.size = kByteLength;
  vulkan_device_.Get<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
      command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
      nullptr, 1, &host, 0, nullptr);
  ASSERT_NO_FATAL_FAILURE(EndVulkan(alias));
  ASSERT_TRUE(readback_.Invalidate());
  const auto* actual = static_cast<const uint32_t*>(readback_.mapped());
  for (size_t i = 0; i < kWordCount; ++i) {
    ASSERT_EQ(actual[i], expected_[i]) << "alias " << alias << ", word " << i;
  }
}

void VulkanMemoryInteropTest::PublishNative() {
  native_pending_ = true;
  // PM4 format v1 uses aligned 64-bit release stores on this x86-64 host.
  // In particular the write-only MMIO doorbell never receives a locked RMW.
  __atomic_store_n(
      reinterpret_cast<uint64_t*>(queue_mapping_info_.write_index_address),
      frontier_, __ATOMIC_RELEASE);
  __atomic_store_n(
      reinterpret_cast<uint64_t*>(queue_mapping_info_.doorbell_address),
      frontier_, __ATOMIC_RELEASE);
  ASSERT_EQ(api_->user_queue_wait_consumed(queue_, frontier_,
                                           AMDF_TIMEOUT_INFINITE, 1000000),
            AMDF_STATUS_OK);
}

void VulkanMemoryInteropTest::RunNative(uint32_t generation) {
  std::array<uint32_t, kCommandWordCapacity> words;
  Pm4CommandWriter commands(words.data());
  commands.SystemBarrier();
  for (size_t i = 0; i < kCopyWordCount; ++i) {
    commands.CopyData32(
        backing_.address + (kViewWordOffset + i) * sizeof(uint32_t),
        backing_.address + (2 * kViewWordOffset + i) * sizeof(uint32_t));
  }
  commands.SystemBarrier();
  commands.WriteData32(control_.address, generation);
  commands.PadToEightWords();
  const uint64_t capacity =
      queue_mapping_info_.ring_byte_length / sizeof(uint32_t);
  ASSERT_LT(commands.word_count(), capacity);
  uint64_t position = frontier_ & (capacity - 1);
  auto* ring = reinterpret_cast<uint32_t*>(queue_mapping_info_.ring_address);
  if (commands.word_count() > capacity - position) {
    const uint64_t tail = capacity - position;
    ASSERT_GE(tail, 2u);
    Pm4CommandWriter padding(ring + position);
    padding.Noop(tail);
    frontier_ += tail;
    ASSERT_NO_FATAL_FAILURE(PublishNative());
    // Only NOPs were published, so consumption establishes their completion.
    native_pending_ = false;
    position = 0;
  }
  std::memcpy(ring + position, words.data(),
              commands.word_count() * sizeof(uint32_t));
  frontier_ += commands.word_count();
  ASSERT_NO_FATAL_FAILURE(PublishNative());
  // Payload completion is separate from ring consumption. The system barrier
  // precedes this marker, which lives outside the shared payload/oracle views.
  while (__atomic_load_n(control_.host, __ATOMIC_ACQUIRE) != generation) {
    __builtin_ia32_pause();
  }
  native_pending_ = false;
  std::copy_n(expected_.begin() + kViewWordOffset, kCopyWordCount,
              expected_.begin() + 2 * kViewWordOffset);
}

TEST_F(VulkanMemoryInteropTest, DeferredComputeAndNativeCopiesPreserveAliases) {
  ASSERT_NO_FATAL_FAILURE(CheckReadback(0));
  for (uint32_t generation = 1; generation <= 8; ++generation) {
    SCOPED_TRACE(generation);
    ASSERT_NO_FATAL_FAILURE(RunVulkan(generation % 2, 0x639a0000u + generation,
                                      0, 257 + generation * 37));
    ASSERT_NO_FATAL_FAILURE(RunNative(generation));
    ASSERT_NO_FATAL_FAILURE(RunVulkan((generation + 1) % 2,
                                      0x95240000u + generation, 1,
                                      129 + generation * 19));
  }
  ASSERT_NO_FATAL_FAILURE(CheckReadback(0));
  ASSERT_NO_FATAL_FAILURE(DestroyNativeQueue());
  ASSERT_NO_FATAL_FAILURE(DestroyMemory(control_));
  ASSERT_NO_FATAL_FAILURE(DestroyMemory(backing_));
  // No source allocation, host view, exported FD or native queue remains.
  ASSERT_NO_FATAL_FAILURE(RunVulkan(0, 0x75980001u, 1, 911));
  ASSERT_NO_FATAL_FAILURE(CheckReadback(0));
  aliases_[0].Reset();
  ASSERT_NO_FATAL_FAILURE(RunVulkan(1, 0x75980002u, 0, 877));
  ASSERT_NO_FATAL_FAILURE(CheckReadback(1));
}

}  // namespace
