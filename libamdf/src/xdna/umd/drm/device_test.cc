// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/drm/device.h"

#include <drm/amdxdna_accel.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include <climits>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/xdna/device_profile.h"
#include "libamdf/src/xdna/umd/context.h"
#include "libamdf/src/xdna/umd/drm/memory.h"

namespace {

// Only descriptive version fields are changed. Every file, metadata query,
// allocation, mapping and release still uses the real native driver.
struct NativeVersionState {
  // Descriptive version value substituted in all three native version fields.
  int value = 0;
  // Number of successful native version queries observed.
  uint32_t query_count = 0;
};

thread_local NativeVersionState* native_version = nullptr;

}  // namespace

extern "C" int __real_ioctl(int descriptor, unsigned long request, ...);

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  const int result = __real_ioctl(descriptor, request, argument);
  if (result == 0 && request == DRM_IOCTL_VERSION &&
      native_version != nullptr) {
    auto* version = static_cast<drm_version*>(argument);
    version->version_major = native_version->value;
    version->version_minor = native_version->value;
    version->version_patchlevel = native_version->value;
    ++native_version->query_count;
  }
  return result;
}

namespace {

enum class ExecutionSupport { kUnavailable, kElfInstructions };

class LinuxXdnaDeviceTest : public ::testing::TestWithParam<ExecutionSupport> {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_platform_instance_create(amdf_allocator_system(), &instance),
              AMDF_STATUS_OK);
    uint32_t count = 0;
    ASSERT_EQ(amdf_platform_endpoint_enumerate(instance, 0, nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> summaries(count);
    ASSERT_EQ(amdf_platform_endpoint_enumerate(instance, count,
                                               summaries.data(), &count),
              AMDF_STATUS_OK);
    for (const auto& summary : summaries) {
      if (summary.engine_kind != AMDF_ENGINE_KIND_XDNA) {
        continue;
      }
      amdf_endpoint_info_t info;
      ASSERT_EQ(
          amdf_platform_endpoint_open(instance, &summary.id, &endpoint, &info),
          AMDF_STATUS_OK);
      if (amdf_xdna_device_profile_initialize(&info, &device_info,
                                              &selected_profile) &&
          profile->dma.address_bit_count != 0 &&
          (GetParam() == ExecutionSupport::kUnavailable ||
           (profile->execution_capabilities &
            AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS) != 0)) {
        break;
      }
      ASSERT_EQ(amdf_platform_endpoint_close(endpoint), AMDF_STATUS_OK);
      endpoint = nullptr;
    }
    if (endpoint == nullptr) {
      GTEST_SKIP()
          << "No XDNA endpoint supports the requested memory/context use";
    }
    if (GetParam() == ExecutionSupport::kUnavailable) {
      // Remove only provider execution support from the discovered hardware
      // profile. Ordinary allocation must not need any instruction execution
      // metadata.
      memory_only_profile = *profile;
      memory_only_profile.execution_capabilities = 0;
      memory_only_profile.bootstrap = nullptr;
      memory_only_profile.firmware_heap_byte_length = 0;
      memory_only_profile.rows = {};
      profile = &memory_only_profile;
    }
  }

  void TearDown() override {
    native_version = nullptr;
    std::cout << "Release host views and memory" << std::endl;
    for (auto* value : mappings) {
      if (value) {
        amdf_xdna_umd_host_mapping_destroy(value);
      }
    }
    if (imported_memory) {
      EXPECT_EQ(amdf_xdna_umd_memory_destroy(imported_memory), AMDF_STATUS_OK);
    }
    if (memory) {
      EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory), AMDF_STATUS_OK);
    }
    if (external_memory.release != nullptr) {
      external_memory.release(external_memory.release_user_data,
                              external_memory.type, external_memory.payload);
    }
    for (auto* value : contexts) {
      if (value) {
        std::cout << "Destroy native context" << std::endl;
        EXPECT_EQ(amdf_xdna_umd_context_destroy(value), AMDF_STATUS_OK);
      }
    }
    if (device) {
      EXPECT_EQ(amdf_xdna_umd_device_destroy(device), AMDF_STATUS_OK);
    }
    if (endpoint) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint), AMDF_STATUS_OK);
    }
    if (instance) {
      EXPECT_EQ(amdf_platform_instance_destroy(instance), AMDF_STATUS_OK);
    }
  }

  // Optional version metadata substitution, retained across assertion exits.
  NativeVersionState version_state;
  // Native instance retained across early assertion exits.
  amdf_platform_instance_t* instance = nullptr;
  // Query endpoint borrowed during native device construction.
  amdf_platform_endpoint_t* endpoint = nullptr;
  // Selected hardware profile with this test's provider execution support.
  amdf_xdna_device_profile_t* profile = &selected_profile;
  // Architecture encodings selected before native activation.
  amdf_xdna_device_profile_t selected_profile = {};
  // Live capabilities retained for every native child.
  amdf_xdna_device_info_t device_info = {};
  // Actual hardware/DMA facts with execution implementation data removed.
  amdf_xdna_device_profile_t memory_only_profile = {};
  // Native device owning one ordinary address and BO namespace.
  amdf_xdna_umd_device_t* device = nullptr;
  // Independent scheduling contexts borrowing the native device.
  amdf_xdna_umd_context_t* contexts[2] = {};
  // Memory retained until every host view has been destroyed.
  amdf_xdna_umd_memory_t* memory = nullptr;
  // Independently acquired import, including any partial preparation state.
  amdf_xdna_umd_memory_t* imported_memory = nullptr;
  // Owned export retained until all native import preparation has completed.
  amdf_external_memory_t external_memory = {};
  // Independent host views into the same native attachment.
  amdf_xdna_umd_host_mapping_t* mappings[2] = {};
};

TEST_P(LinuxXdnaDeviceTest,
       AdmissionUsesNativeFeaturesInsteadOfVersionMetadata) {
  native_version = &version_state;
  for (int version : {0, INT_MAX}) {
    SCOPED_TRACE(version);
    version_state.value = version;
    version_state.query_count = 0;
    amdf_xdna_umd_device_result_t result = {};
    const amdf_status_t status = amdf_xdna_umd_device_create(
        endpoint, profile, amdf_allocator_system(), &device, &result);
    EXPECT_EQ(status, AMDF_STATUS_OK);
    EXPECT_EQ(version_state.query_count, 1u);
    if (!amdf_status_is_ok(status)) {
      continue;
    }
    EXPECT_NE(result.id.words[0] | result.id.words[1], 0u);
    ASSERT_EQ(amdf_xdna_umd_device_destroy(device), AMDF_STATUS_OK);
    device = nullptr;
  }
}

TEST_P(LinuxXdnaDeviceTest, MemoryDoesNotDependOnSchedulingContexts) {
  const bool supports_execution =
      GetParam() == ExecutionSupport::kElfInstructions;
  const auto capabilities = amdf_xdna_umd_query_context_capabilities(profile);
  EXPECT_EQ(capabilities.scheduling_modes,
            supports_execution ? AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED : 0u);
  EXPECT_EQ(
      capabilities.placement_modes,
      supports_execution ? AMDF_XDNA_PLACEMENT_MODE_FIXED_FULL_ARRAY : 0u);
  amdf_xdna_umd_device_result_t device_result = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_create(endpoint, profile, amdf_allocator_system(),
                                  &device, &device_result),
      AMDF_STATUS_OK);
  device_info.array.column_count = device_result.tiles.column_count;
  device_info.array.row_count = device_result.tiles.row_count;
  profile->rows.core_origin = device_result.tiles.core_origin;
  profile->rows.core_count = device_result.tiles.core_count;
  EXPECT_NE(device_result.id.words[0] | device_result.id.words[1], 0u);
  EXPECT_EQ(device_result.placement_modes, capabilities.placement_modes);
  EXPECT_NE(fcntl(device->descriptor, F_GETFD) & FD_CLOEXEC, 0);
  if (supports_execution) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(device->heap.host_pointer) %
                  profile->firmware_heap_byte_length,
              0u);
  } else {
    EXPECT_EQ(device->heap.handle, 0u);
    EXPECT_EQ(device->heap.host_pointer, nullptr);
  }

  amdf_xdna_context_create_info_t create_info = {};
  create_info.acceptable_scheduling_modes =
      AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
  create_info.logical_column_count = 1;
  amdf_xdna_umd_context_result_t results[2] = {};
  const size_t context_count = supports_execution ? 2 : 0;
  for (size_t i = 0; i < context_count; ++i) {
    create_info.physical_column_origin =
        i == 0 ? profile->info->array.column_origin
               : AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    std::cout << "Create native context " << i << std::endl;
    ASSERT_EQ(amdf_xdna_umd_context_create(device, &create_info, &contexts[i],
                                           &results[i]),
              AMDF_STATUS_OK);
    EXPECT_EQ(results[i].physical_column_origin,
              profile->info->array.column_origin);
    EXPECT_EQ(results[i].physical_column_count,
              profile->info->array.column_count);
  }
  if (supports_execution) {
    EXPECT_NE(results[0].id.words[0], results[1].id.words[0]);
  }

  amdf_memory_native_create_info_t memory_create = {
      .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .required_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .byte_length = 4097,
      .minimum_alignment = 4096};
  amdf_xdna_umd_memory_result_t memory_result = {};
  amdf_memory_native_profile_t memory_profile = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_query_memory_profile(device, 0, &memory_profile),
      AMDF_STATUS_OK);
  std::cout << "Create aligned SHARE attachment" << std::endl;
  ASSERT_EQ(
      amdf_xdna_umd_memory_prepare(device, &memory_profile, &memory_create,
                                   &memory, &memory_result),
      AMDF_STATUS_OK);
  EXPECT_GE(memory_result.byte_length, memory_create.byte_length);
  EXPECT_EQ(memory_result.device_address % memory_create.minimum_alignment, 0u);
  EXPECT_EQ(memory_result.address_kinds, memory_profile.address_kinds);
  EXPECT_EQ(memory_result.dma_address,
            memory_result.device_address + profile->dma.byte_offset);
  EXPECT_LE(memory_result.dma_address + memory_result.byte_length - 1,
            (UINT64_C(1) << profile->dma.address_bit_count) - 1);
  amdf_memory_map_info_t map_info = {
      .byte_offset = 1,
      .byte_length = 4096,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE};
  amdf_xdna_umd_host_mapping_result_t views[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(amdf_xdna_umd_memory_map(memory, &memory_profile.host_mapping,
                                       &map_info, &mappings[i], &views[i]),
              AMDF_STATUS_OK);
  }
  EXPECT_EQ(views[0].pointer,
            static_cast<uint8_t*>(memory->buffer.host_pointer) +
                map_info.byte_offset);
  EXPECT_EQ(views[0].pointer, views[1].pointer);
  std::memset(views[0].pointer, 0xA5, map_info.byte_length);
  ASSERT_EQ(amdf_xdna_umd_host_mapping_cache_control(
                mappings[0], AMDF_HOST_CACHE_OPERATION_FLUSH,
                map_info.byte_offset, map_info.byte_length),
            AMDF_STATUS_OK);
  amdf_xdna_umd_host_mapping_destroy(mappings[0]);
  mappings[0] = nullptr;
  std::cout << "First view destroyed; attachment remains mapped" << std::endl;
  struct amdxdna_drm_get_bo_info native_info = {.handle =
                                                    memory->buffer.handle};
  ASSERT_EQ(
      ioctl(device->descriptor, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &native_info),
      0);
  EXPECT_EQ(native_info.xdna_addr, memory_result.device_address);
  EXPECT_EQ(native_info.vaddr,
            reinterpret_cast<uintptr_t>(memory->buffer.host_pointer));
  EXPECT_EQ(static_cast<uint8_t*>(views[1].pointer)[0], 0xA5);
  EXPECT_EQ(static_cast<uint8_t*>(views[1].pointer)[4095], 0xA5);
  ASSERT_EQ(amdf_xdna_umd_host_mapping_cache_control(
                mappings[1], AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                map_info.byte_offset, map_info.byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(static_cast<uint8_t*>(views[1].pointer)[4095], 0xA5);

  amdf_memory_export_info_t export_info = {
      .external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .byte_length = memory_result.byte_length};
  ASSERT_EQ(amdf_xdna_umd_memory_export(memory, &export_info, &external_memory),
            AMDF_STATUS_OK);
  // Native export supplies the payload/release obligation; the public owner
  // qualifies that transport with its validated logical range and identity.
  external_memory.type = export_info.external_memory_type;
  external_memory.byte_length = export_info.byte_length;
  external_memory.source_byte_offset = memory_result.source_byte_offset;
  external_memory.physical_backing_id = memory_result.physical_backing_id;
  amdf_memory_native_profile_t import_profile = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_query_memory_profile(device, 1, &import_profile),
      AMDF_STATUS_OK);
  amdf_memory_native_import_info_t import_info = {
      .device_access = memory_create.device_access};
  amdf_xdna_umd_memory_result_t import_result = {};
  uint32_t release_count = 0;
  amdf_external_memory_t borrowed_external = external_memory;
  borrowed_external.release = [](void* user_data, amdf_external_memory_type_t,
                                 amdf_external_memory_payload_t) {
    ++*static_cast<uint32_t*>(user_data);
  };
  borrowed_external.release_user_data = &release_count;
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_import(
                device, &import_profile, &import_info, &borrowed_external,
                &imported_memory, &import_result),
            AMDF_STATUS_OK);
  EXPECT_EQ(release_count, 0u);
  EXPECT_TRUE(amdf_physical_memory_id_is_equal(
      &import_result.physical_backing_id, &memory_result.physical_backing_id));
  EXPECT_EQ(static_cast<uint8_t*>(imported_memory->buffer.host_pointer)[1],
            0xA5);

  if (supports_execution) {
    std::cout << "Destroy sibling context while memory stays live" << std::endl;
    ASSERT_EQ(amdf_xdna_umd_context_destroy(contexts[0]), AMDF_STATUS_OK);
    contexts[0] = nullptr;
    EXPECT_NE(contexts[1], nullptr);
  }
  EXPECT_EQ(static_cast<uint8_t*>(views[1].pointer)[4095], 0xA5);
}

TEST_P(LinuxXdnaDeviceTest, NativeMetadataSuppliesArrayGeometry) {
  amdf_xdna_device_info_t different_info = *profile->info;
  different_info.array.column_count = 1;
  different_info.array.row_count = 1;
  amdf_xdna_device_profile_t different_profile = *profile;
  different_profile.info = &different_info;
  different_profile.rows = {};
  amdf_xdna_umd_device_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_create(endpoint, &different_profile,
                                  amdf_allocator_system(), &device, &result),
      AMDF_STATUS_OK);
  struct amdxdna_drm_query_aie_metadata metadata = {};
  struct amdxdna_drm_get_info query = {};
  query.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
  query.buffer_size = sizeof(metadata);
  query.buffer = reinterpret_cast<uintptr_t>(&metadata);
  ASSERT_EQ(ioctl(device->descriptor, DRM_IOCTL_AMDXDNA_GET_INFO, &query), 0);
  EXPECT_EQ(result.tiles.column_count, metadata.cols);
  EXPECT_EQ(result.tiles.row_count, metadata.rows);
  EXPECT_EQ(result.tiles.core_origin, metadata.core.row_start);
  EXPECT_EQ(result.tiles.core_count, metadata.core.row_count);
  EXPECT_EQ(result.tiles.memory_origin, metadata.mem.row_start);
  EXPECT_EQ(result.tiles.memory_count, metadata.mem.row_count);
  EXPECT_EQ(result.tiles.shim_origin, metadata.shim.row_start);
  EXPECT_EQ(result.tiles.shim_count, metadata.shim.row_count);
  // The native device borrows the supplied profile through destruction.
  ASSERT_EQ(amdf_xdna_umd_device_destroy(device), AMDF_STATUS_OK);
  device = nullptr;
}

INSTANTIATE_TEST_SUITE_P(
    NativeSupport, LinuxXdnaDeviceTest,
    ::testing::Values(ExecutionSupport::kUnavailable,
                      ExecutionSupport::kElfInstructions),
    [](const ::testing::TestParamInfo<ExecutionSupport>& info) {
      return info.param == ExecutionSupport::kUnavailable ? "MemoryOnly"
                                                          : "ElfInstructions";
    });

}  // namespace
