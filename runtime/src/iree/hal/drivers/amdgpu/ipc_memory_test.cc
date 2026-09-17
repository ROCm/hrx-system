// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/ipc_memory.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

constexpr iree_hal_queue_family_affinity_t kQueueFamilyAffinity0 =
    ((iree_hal_queue_family_affinity_t)1ull) << 0;

static const char* QueryHostIncompatibilityReason(
    const iree_hal_amdgpu_logical_device_options_t* options) {
  switch (iree_hal_amdgpu_logical_device_options_query_host_compatibility(
      options)) {
    case IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_COMPATIBLE:
      return nullptr;
    case IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_INCOMPATIBLE_HOST_TSAN_ASAN:
      return "AMDGPU ASAN is not supported in host ThreadSanitizer builds";
    default:
      return "AMDGPU logical-device options are not compatible with this host";
  }
}

static void NoopReleaseCallback(void*, iree_hal_buffer_t*) {}

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
typedef struct IpcMemoryMockState {
  // Base pointer returned by the attach hook and recognized by pointer info.
  void* mapped_ptr;

  // Full mapped allocation extent reported through pointer info.
  size_t allocation_size;

  // HSA pointer classification reported by pointer info.
  int pointer_type;

  // HSA memory-pool global flags reported by pointer info.
  uint32_t global_flags;

  // Exporting owner deliberately allowed to differ from the destination GPU.
  hsa_agent_t external_owner;

  // Number of calls made to the attach hook.
  uint32_t attach_call_count;

  // Length passed to the attach hook.
  size_t attached_length;

  // Number of destination agents passed to the attach hook.
  uint32_t attach_agent_count;

  // Destination agents passed to the attach hook.
  std::array<hsa_agent_t, IREE_HAL_AMDGPU_MAX_GPU_AGENT> attach_agents;

  // Number of calls made to the pointer-info hook.
  uint32_t pointer_info_call_count;

  // Number of calls made to the detach hook.
  uint32_t detach_call_count;
} IpcMemoryMockState;

static IpcMemoryMockState* g_ipc_memory_mock = nullptr;

static hsa_status_t HSA_API IpcMemoryAttachHook(
    const hsa_amd_ipc_memory_t*, size_t length, uint32_t num_agents,
    const hsa_agent_t* mapping_agents, void** mapped_ptr) {
  IpcMemoryMockState* state = g_ipc_memory_mock;
  if (!state || !mapped_ptr || num_agents == 0 || !mapping_agents ||
      num_agents > state->attach_agents.size()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ++state->attach_call_count;
  state->attached_length = length;
  state->attach_agent_count = num_agents;
  std::copy_n(mapping_agents, num_agents, state->attach_agents.begin());
  *mapped_ptr = state->mapped_ptr;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t HSA_API
IpcMemoryPointerInfoHook(const void* ptr, hsa_amd_pointer_info_t* info,
                         void* (*)(size_t), uint32_t*, hsa_agent_t**) {
  IpcMemoryMockState* state = g_ipc_memory_mock;
  const uintptr_t pointer_value = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t base_value =
      reinterpret_cast<uintptr_t>(state ? state->mapped_ptr : nullptr);
  if (!state || !info || pointer_value < base_value ||
      pointer_value - base_value >= state->allocation_size) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ++state->pointer_info_call_count;
  *info = {};
  info->size = sizeof(*info);
  info->type = static_cast<decltype(info->type)>(state->pointer_type);
  info->agentBaseAddress = state->mapped_ptr;
  info->sizeInBytes = state->allocation_size;
  info->agentOwner = state->external_owner;
  info->global_flags = state->global_flags;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t HSA_API IpcMemoryDetachHook(void* mapped_ptr) {
  IpcMemoryMockState* state = g_ipc_memory_mock;
  if (!state || mapped_ptr != state->mapped_ptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ++state->detach_call_count;
  return HSA_STATUS_SUCCESS;
}

class ScopedIpcMemoryMock {
 public:
  explicit ScopedIpcMemoryMock(IpcMemoryMockState* state) {
    EXPECT_EQ(g_ipc_memory_mock, nullptr);
    g_ipc_memory_mock = state;
  }

  ~ScopedIpcMemoryMock() { g_ipc_memory_mock = nullptr; }

  ScopedIpcMemoryMock(const ScopedIpcMemoryMock&) = delete;
  ScopedIpcMemoryMock& operator=(const ScopedIpcMemoryMock&) = delete;
};
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

class IpcMemoryTest : public ::testing::Test {
 protected:
  class TestLogicalDevice {
   public:
    ~TestLogicalDevice() {
      iree_hal_device_release(base_device_);
      iree_hal_device_group_release(device_group_);
    }

    iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* libhsa,
                             const iree_hal_amdgpu_topology_t* topology,
                             iree_allocator_t host_allocator) {
      iree_hal_amdgpu_logical_device_options_t options;
      iree_hal_amdgpu_logical_device_options_initialize(&options);
      return InitializeWithOptions(&options, libhsa, topology, host_allocator);
    }

    iree_status_t InitializeWithOptions(
        const iree_hal_amdgpu_logical_device_options_t* options,
        const iree_hal_amdgpu_libhsa_t* libhsa,
        const iree_hal_amdgpu_topology_t* topology,
        iree_allocator_t host_allocator) {
      IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
          IREE_SV("amdgpu"), options, libhsa, topology,
          create_context_.params(), host_allocator, &base_device_));
      return iree_hal_device_group_create_from_device(
          base_device_, create_context_.frontier_tracker(), host_allocator,
          &device_group_);
    }

    iree_hal_allocator_t* allocator() const {
      return iree_hal_device_allocator(base_device_);
    }

    iree_hal_device_t* device() const { return base_device_; }

    iree_hal_amdgpu_logical_device_t* logical_device() const {
      return (iree_hal_amdgpu_logical_device_t*)base_device_;
    }

   private:
    // Creation context supplying the proactor pool and frontier tracker.
    iree::hal::cts::DeviceCreateContext create_context_;

    // Test-owned device reference released before the topology-owning group.
    iree_hal_device_t* base_device_ = nullptr;

    // Device group that owns the topology assigned to |base_device_|.
    iree_hal_device_group_t* device_group_ = nullptr;
  };

  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  // Host allocator shared by logical-device fixtures.
  static iree_allocator_t host_allocator_;

  // Dynamically loaded HSA dispatch table shared by the suite.
  static iree_hal_amdgpu_libhsa_t libhsa_;

  // Visible GPU topology shared by the suite.
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t IpcMemoryTest::host_allocator_;
iree_hal_amdgpu_libhsa_t IpcMemoryTest::libhsa_;
iree_hal_amdgpu_topology_t IpcMemoryTest::topology_;

TEST_F(IpcMemoryTest, SharingExportPreservesRequestedExtent) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.quarantine_size = 0;
  if (const char* reason = QueryHostIncompatibilityReason(&options)) {
    GTEST_SKIP() << reason;
  }

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH |
                 IREE_HAL_BUFFER_USAGE_SHARING_EXPORT;
  params.queue_family_affinity = kQueueFamilyAffinity0;

  for (iree_device_size_t requested_size : {1u, 257u}) {
    iree_hal_buffer_params_t resolved_params = {};
    iree_device_size_t resolved_size = 0;
    const iree_hal_buffer_compatibility_t compatibility =
        iree_hal_allocator_query_buffer_compatibility(
            test_device.allocator(), params, requested_size, &resolved_params,
            &resolved_size);
    ASSERT_TRUE(iree_all_bits_set(
        compatibility, IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                           IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE));
    EXPECT_EQ(resolved_size, requested_size);

    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        test_device.allocator(), params, requested_size, &buffer));
    ASSERT_NE(buffer, nullptr);
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
        buffer_guard(buffer, iree_hal_buffer_release);
    EXPECT_EQ(iree_hal_buffer_allocated_buffer(buffer), buffer);
    EXPECT_EQ(iree_hal_buffer_byte_offset(buffer), 0u);
    EXPECT_EQ(iree_hal_buffer_byte_length(buffer), requested_size);
    EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), requested_size);
    EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                  IREE_HAL_BUFFER_USAGE_SHARING_EXPORT));

    iree_hal_external_buffer_t device_allocation = {};
    IREE_ASSERT_OK(iree_hal_allocator_export_buffer(
        test_device.allocator(), buffer,
        IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &device_allocation));
    hsa_amd_pointer_info_t pointer_info = {};
    pointer_info.size = sizeof(pointer_info);
    IREE_ASSERT_OK(iree_hsa_amd_pointer_info(
        IREE_LIBHSA(&libhsa_),
        (const void*)(uintptr_t)device_allocation.handle.device_allocation.ptr,
        &pointer_info, /*alloc=*/NULL, /*num_agents_accessible=*/NULL,
        /*accessible=*/NULL));
    EXPECT_EQ(pointer_info.sizeInBytes, requested_size);
    EXPECT_EQ(pointer_info.global_flags &
                  IREE_HAL_AMDGPU_ATOMIC_MEMORY_POOL_CLASS_FLAGS,
              HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
    EXPECT_FALSE(iree_any_bit_set(iree_hal_buffer_memory_type(buffer),
                                  IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED));

    iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {};
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_memory_export(test_device.allocator(),
                                                     buffer, &descriptor));
    EXPECT_EQ(descriptor.allocation_size,
              static_cast<uint64_t>(requested_size));
  }
}

TEST_F(IpcMemoryTest, ExportRejectsFineAndUncachedMemoryTypes) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  constexpr size_t kAllocationSize = 4096;
  alignas(4096) std::array<uint8_t, kAllocationSize> storage = {};
  const iree_hal_buffer_placement_t placement = {
      /*.device=*/test_device.device(),
      /*.queue_family_affinity=*/kQueueFamilyAffinity0,
      /*.flags=*/IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  const iree_hal_buffer_release_callback_t release_callback = {
      /*.fn=*/NoopReleaseCallback,
  };

  auto expect_rejected = [&](iree_hal_memory_type_t memory_type) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_buffer_create(
        &libhsa_, placement, memory_type, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_SHARING_EXPORT,
        IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE, kAllocationSize,
        kAllocationSize, storage.data(), release_callback, host_allocator_,
        &buffer));
    iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          iree_hal_amdgpu_ipc_memory_export(
                              test_device.allocator(), buffer, &descriptor));
    iree_hal_buffer_release(buffer);
  };

  expect_rejected(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_COHERENT);
  expect_rejected(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED);
}

TEST_F(IpcMemoryTest, ExportRequiresDedicatedAllocatorAllocation) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  constexpr size_t kAllocationSize = 4096;
  alignas(4096) std::array<uint8_t, kAllocationSize> storage = {};
  const iree_hal_buffer_placement_t placement = {
      /*.device=*/test_device.device(),
      /*.queue_family_affinity=*/kQueueFamilyAffinity0,
      /*.flags=*/IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  const iree_hal_buffer_release_callback_t release_callback = {
      /*.fn=*/NoopReleaseCallback,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_create(
      &libhsa_, placement, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_SHARING_EXPORT,
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE, kAllocationSize,
      kAllocationSize, storage.data(), release_callback, host_allocator_,
      &buffer));

  iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_amdgpu_ipc_memory_export(
                            test_device.allocator(), buffer, &descriptor));
  iree_hal_buffer_release(buffer);
}

TEST_F(IpcMemoryTest, ExportRequiresSharingExportUsage) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_family_affinity = kQueueFamilyAffinity0;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      test_device.allocator(), params, /*allocation_size=*/4096, &buffer));

  iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_amdgpu_ipc_memory_export(
                            test_device.allocator(), buffer, &descriptor));

  iree_hal_buffer_release(buffer);
}

TEST_F(IpcMemoryTest, ImportAliasRequiresAttachedIpcBuffer) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_family_affinity = kQueueFamilyAffinity0;
  iree_hal_buffer_t* ordinary_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      test_device.allocator(), params, /*allocation_size=*/4096,
      &ordinary_buffer));

  iree_hal_buffer_t* alias_buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_ipc_memory_import_alias(test_device.allocator(), params,
                                              ordinary_buffer, &alias_buffer));
  EXPECT_EQ(alias_buffer, nullptr);

  iree_hal_buffer_release(ordinary_buffer);
}

TEST_F(IpcMemoryTest, ExportPreservesSubspan) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  constexpr iree_device_size_t kAllocationSize = 4096;
  constexpr iree_device_size_t kViewOffset = 256;
  constexpr iree_device_size_t kViewLength = 1024;
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH |
                 IREE_HAL_BUFFER_USAGE_SHARING_EXPORT;
  params.queue_family_affinity = kQueueFamilyAffinity0;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      test_device.allocator(), params, kAllocationSize, &buffer));

  iree_hal_external_buffer_t root_pointer = {};
  IREE_ASSERT_OK(iree_hal_allocator_export_buffer(
      test_device.allocator(), buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &root_pointer));
  hsa_amd_pointer_info_t pointer_info = {};
  pointer_info.size = sizeof(pointer_info);
  IREE_ASSERT_OK(iree_hsa_amd_pointer_info(
      IREE_LIBHSA(&libhsa_),
      (const void*)(uintptr_t)root_pointer.handle.device_allocation.ptr,
      &pointer_info, /*alloc=*/NULL, /*num_agents_accessible=*/NULL,
      /*accessible=*/NULL));

  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, kViewOffset, kViewLength,
                                         host_allocator_, &view));
  iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_memory_export(test_device.allocator(),
                                                   view, &descriptor));
  EXPECT_EQ(descriptor.allocation_size,
            static_cast<uint64_t>(pointer_info.sizeInBytes));
  EXPECT_EQ(descriptor.byte_offset,
            root_pointer.handle.device_allocation.ptr + kViewOffset -
                reinterpret_cast<uintptr_t>(pointer_info.agentBaseAddress));

  iree_hal_buffer_release(view);
  iree_hal_buffer_release(buffer);
}

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
TEST_F(IpcMemoryTest, ImportDetachesAfterAttachedRangeValidationFailure) {
  constexpr size_t kDescriptorAllocationSize = 4096;
  constexpr uint64_t kByteOffset = 256;
  constexpr iree_device_size_t kViewLength = 1024;
  alignas(4096) std::array<uint8_t, kDescriptorAllocationSize> mapped_storage =
      {};

  IpcMemoryMockState mock_state = {};
  mock_state.mapped_ptr = mapped_storage.data();
  // Keep the imported view pointer inside the reported allocation while making
  // the view itself exceed that allocation. This forces failure after ROCr has
  // successfully attached the mapping.
  mock_state.allocation_size = kByteOffset + 1;
  mock_state.pointer_type = HSA_EXT_POINTER_TYPE_GRAPHICS;
  mock_state.global_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  mock_state.external_owner.handle = UINT64_MAX;
  ScopedIpcMemoryMock mock_scope(&mock_state);

  iree_hal_amdgpu_libhsa_t hooked_libhsa;
  IREE_ASSERT_OK(iree_hal_amdgpu_libhsa_copy(&libhsa_, &hooked_libhsa));
  hooked_libhsa.hsa_amd_ipc_memory_attach = IpcMemoryAttachHook;
  hooked_libhsa.hsa_amd_pointer_info = IpcMemoryPointerInfoHook;
  hooked_libhsa.hsa_amd_ipc_memory_detach = IpcMemoryDetachHook;

  {
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(
        test_device.Initialize(&hooked_libhsa, &topology_, host_allocator_));

    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
    params.queue_family_affinity = kQueueFamilyAffinity0;
    iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {};
    descriptor.allocation_size = kDescriptorAllocationSize;
    descriptor.byte_offset = kByteOffset;

    iree_hal_buffer_t* imported_buffer = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_amdgpu_ipc_memory_import(
            test_device.allocator(), test_device.allocator(),
            kQueueFamilyAffinity0, params, &descriptor, kViewLength,
            iree_hal_buffer_release_callback_null(), &imported_buffer));
    EXPECT_EQ(imported_buffer, nullptr);
    EXPECT_EQ(mock_state.attach_call_count, 1u);
    EXPECT_EQ(mock_state.pointer_info_call_count, 1u);
    EXPECT_EQ(mock_state.detach_call_count, 1u);
  }

  iree_hal_amdgpu_libhsa_deinitialize(&hooked_libhsa);
}

TEST_F(IpcMemoryTest, ImportAcceptsExternalGraphicsOwnerForDestinationGpu) {
  constexpr size_t kAllocationSize = 4096;
  constexpr uint64_t kByteOffset = 256;
  constexpr iree_device_size_t kViewLength = 1024;
  alignas(4096) std::array<uint8_t, kAllocationSize> mapped_storage = {};

  IpcMemoryMockState mock_state = {};
  mock_state.mapped_ptr = mapped_storage.data();
  mock_state.allocation_size = mapped_storage.size();
  mock_state.pointer_type = HSA_EXT_POINTER_TYPE_GRAPHICS;
  mock_state.global_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  mock_state.external_owner.handle = UINT64_MAX;
  ScopedIpcMemoryMock mock_scope(&mock_state);

  iree_hal_amdgpu_libhsa_t hooked_libhsa;
  IREE_ASSERT_OK(iree_hal_amdgpu_libhsa_copy(&libhsa_, &hooked_libhsa));
  hooked_libhsa.hsa_amd_ipc_memory_attach = IpcMemoryAttachHook;
  hooked_libhsa.hsa_amd_pointer_info = IpcMemoryPointerInfoHook;
  hooked_libhsa.hsa_amd_ipc_memory_detach = IpcMemoryDetachHook;

  {
    TestLogicalDevice test_device;
    IREE_ASSERT_OK(
        test_device.Initialize(&hooked_libhsa, &topology_, host_allocator_));

    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
    params.queue_family_affinity = kQueueFamilyAffinity0;

    iree_hal_external_buffer_t direct_buffer = {};
    direct_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
    direct_buffer.size = kViewLength;
    direct_buffer.handle.device_allocation.ptr =
        reinterpret_cast<uintptr_t>(mapped_storage.data()) + kByteOffset;
    iree_hal_buffer_t* rejected_buffer = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_allocator_import_buffer(
            test_device.allocator(), params, &direct_buffer,
            iree_hal_buffer_release_callback_null(), &rejected_buffer));
    EXPECT_EQ(rejected_buffer, nullptr);

    iree_hal_amdgpu_ipc_memory_descriptor_t descriptor = {};
    descriptor.allocation_size = kAllocationSize;
    descriptor.byte_offset = kByteOffset;

    iree_hal_buffer_t* imported_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_memory_import(
        test_device.allocator(), test_device.allocator(), kQueueFamilyAffinity0,
        params, &descriptor, kViewLength,
        iree_hal_buffer_release_callback_null(), &imported_buffer));
    ASSERT_NE(imported_buffer, nullptr);
    EXPECT_EQ(mock_state.attach_call_count, 1u);
    EXPECT_EQ(mock_state.attached_length, kAllocationSize);
    ASSERT_GT(mock_state.attach_agent_count, 0u);
    EXPECT_EQ(mock_state.attach_agents[0].handle,
              topology_.gpu_agents[0].handle);
    EXPECT_EQ(mock_state.pointer_info_call_count, 2u);
    EXPECT_EQ(iree_hal_amdgpu_buffer_device_pointer(imported_buffer),
              mapped_storage.data() + kByteOffset);
    EXPECT_EQ(iree_hal_buffer_allocation_placement(imported_buffer)
                  .queue_family_affinity,
              kQueueFamilyAffinity0);

    EXPECT_EQ(iree_hal_buffer_memory_type(imported_buffer),
              IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
    // Pool class alone cannot identify the exporting pool or prove the atomic
    // capabilities of the route from a destination GPU to that pool.
    EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(imported_buffer),
              IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE);

    TestLogicalDevice alias_device;
    IREE_ASSERT_OK(
        alias_device.Initialize(&hooked_libhsa, &topology_, host_allocator_));

    uint64_t unavailable_agent_handle = UINT64_MAX;
    for (;;) {
      bool handle_is_attached = false;
      for (uint32_t i = 0; i < mock_state.attach_agent_count; ++i) {
        if (mock_state.attach_agents[i].handle == unavailable_agent_handle) {
          handle_is_attached = true;
          break;
        }
      }
      if (!handle_is_attached) break;
      ASSERT_NE(unavailable_agent_handle, 0u);
      --unavailable_agent_handle;
    }
    iree_hal_amdgpu_topology_t* destination_topology =
        &alias_device.logical_device()->system->topology;
    const hsa_agent_t original_agent = destination_topology->gpu_agents[0];
    destination_topology->gpu_agents[0].handle = unavailable_agent_handle;
    iree_hal_buffer_t* rejected_alias = nullptr;
    iree_status_t rejected_status = iree_hal_amdgpu_ipc_memory_import_alias(
        alias_device.allocator(), params, imported_buffer, &rejected_alias);
    destination_topology->gpu_agents[0] = original_agent;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, rejected_status);
    EXPECT_EQ(rejected_alias, nullptr);
    iree_hal_buffer_release(rejected_alias);
    EXPECT_EQ(mock_state.pointer_info_call_count, 2u);

    iree_hal_buffer_t* alias_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_memory_import_alias(
        alias_device.allocator(), params, imported_buffer, &alias_buffer));
    ASSERT_NE(alias_buffer, nullptr);
    EXPECT_EQ(mock_state.attach_call_count, 1u);
    EXPECT_EQ(mock_state.detach_call_count, 0u);
    EXPECT_EQ(mock_state.pointer_info_call_count, 3u);
    EXPECT_EQ(iree_hal_amdgpu_buffer_device_pointer(alias_buffer),
              mapped_storage.data() + kByteOffset);
    EXPECT_EQ(iree_hal_buffer_allocation_placement(alias_buffer).device,
              alias_device.device());
    EXPECT_EQ(iree_hal_buffer_byte_length(alias_buffer), kViewLength);
    EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(alias_buffer),
              IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE);

    // The destination-device alias owns only its local bookkeeping. The
    // original attached buffer remains the sole owner of ROCr detach.
    iree_hal_buffer_release(alias_buffer);
    EXPECT_EQ(mock_state.detach_call_count, 0u);
    iree_hal_buffer_release(imported_buffer);
    EXPECT_EQ(mock_state.detach_call_count, 1u);
  }

  iree_hal_amdgpu_libhsa_deinitialize(&hooked_libhsa);
}
#else
TEST_F(IpcMemoryTest, ImportHooksRequireDynamicLibhsa) {
  GTEST_SKIP() << "IPC import hook coverage requires dynamic libhsa";
}
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

}  // namespace
}  // namespace iree::hal::amdgpu
