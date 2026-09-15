// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/memory.h"

#include <drm/amdxdna_accel.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdarg>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/drm/device.h"
#include "libamdf/src/xdna/umd/drm/memory.h"

namespace {

// Only the DRM dependency is modeled. Allocation and imported descriptor
// ownership use the production memory implementation and real host resources.
struct NativeMemoryState {
  // Terminal GEM-close error, or zero for successful local rollback.
  int close_error = ENODEV;
  // Number of native handle release attempts.
  uint32_t close_count = 0;
  // Number of unpublished memory headers returned to the host allocator.
  uint32_t metadata_free_count = 0;
  // Whether the modeled GEM handle remains unreleased.
  bool handle_live = false;
  // Independent backing reference held by a modeled imported GEM handle.
  int backing_descriptor = -1;
};

// Calls outside the fixture retain the real DRM dependency.
thread_local NativeMemoryState* native_memory = nullptr;

}  // namespace

extern "C" int __real_ioctl(int descriptor, unsigned long request, ...);

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_memory == nullptr) {
    return __real_ioctl(descriptor, request, argument);
  }
  EXPECT_EQ(descriptor, 17);
  switch (request) {
    case DRM_IOCTL_AMDXDNA_CREATE_BO: {
      auto* create = static_cast<amdxdna_drm_create_bo*>(argument);
      EXPECT_EQ(create->type, AMDXDNA_BO_SHARE);
      EXPECT_FALSE(native_memory->handle_live);
      if (create->vaddr != 0) {
        const auto* table =
            reinterpret_cast<const amdxdna_drm_va_tbl*>(create->vaddr);
        if (table->dmabuf_fd >= 0) {
          native_memory->backing_descriptor = dup(table->dmabuf_fd);
          EXPECT_GE(native_memory->backing_descriptor, 0);
        }
      }
      create->handle = 0x1234;
      native_memory->handle_live = true;
      return 0;
    }
    case DRM_IOCTL_AMDXDNA_GET_BO_INFO:
      EXPECT_TRUE(native_memory->handle_live);
      errno = EIO;
      return -1;
    case DRM_IOCTL_GEM_CLOSE: {
      auto* release = static_cast<drm_gem_close*>(argument);
      EXPECT_EQ(release->handle, 0x1234u);
      EXPECT_TRUE(native_memory->handle_live);
      ++native_memory->close_count;
      if (native_memory->close_error != 0) {
        errno = native_memory->close_error;
        return -1;
      }
      if (native_memory->backing_descriptor != -1) {
        EXPECT_EQ(close(native_memory->backing_descriptor), 0);
        native_memory->backing_descriptor = -1;
      }
      native_memory->handle_live = false;
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected DRM memory ioctl: " << request;
      errno = ENOTTY;
      return -1;
  }
}

namespace {

class LinuxXdnaMemoryRollbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_memory = &native_;
    device_.descriptor = 17;
    device_.page_size = 4096;
    device_.profile = &endpoint_profile_;
    device_.host_allocator = amdf_allocator_system();
    device_.host_allocator.user_data = &native_;
    device_.host_allocator.free = [](void* user_data, void* allocation) {
      ++static_cast<NativeMemoryState*>(user_data)->metadata_free_count;
      amdf_free(amdf_allocator_system(), allocation);
    };
    std::memset(&result_, 0xA5, sizeof(result_));
  }

  void TearDown() override {
    EXPECT_EQ(native_.close_count, 1u);
    EXPECT_EQ(native_.metadata_free_count, 1u);
    EXPECT_EQ(native_.handle_live, native_.close_error != 0);
    EXPECT_EQ(memory_, nullptr);
    amdf_xdna_umd_memory_result_t original_result;
    std::memset(&original_result, 0xA5, sizeof(original_result));
    EXPECT_EQ(std::memcmp(&result_, &original_result, sizeof(result_)), 0);
    // A failed native close is deliberately not retried. Only the model's
    // independent test-file reference is reclaimed here, not a real GEM handle.
    if (native_.backing_descriptor != -1) {
      EXPECT_EQ(close(native_.backing_descriptor), 0);
    }
    native_memory = nullptr;
  }

  amdf_memory_native_profile_t QueryProfile(uint32_t ordinal) {
    amdf_memory_native_profile_t profile = {};
    EXPECT_EQ(
        amdf_xdna_umd_device_query_memory_profile(&device_, ordinal, &profile),
        AMDF_STATUS_OK);
    return profile;
  }

  void ReleasePreparedState() {
    ASSERT_NE(memory_, nullptr);
    EXPECT_TRUE(native_.handle_live);
    EXPECT_EQ(native_.close_count, 0u);
    EXPECT_EQ(native_.metadata_free_count, 0u);
    const amdf_status_t status = amdf_xdna_umd_memory_destroy(memory_);
    if (native_.close_error != 0) {
      EXPECT_EQ(status, amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO,
                                         native_.close_error));
      EXPECT_EQ(native_.metadata_free_count, 0u);
      amdf_xdna_umd_memory_abandon(memory_);
    } else {
      EXPECT_EQ(status, AMDF_STATUS_OK);
    }
    memory_ = nullptr;
  }

  // Native failure and resource-consumption state.
  NativeMemoryState native_;
  // Complete address-translation limits for the test device.
  amdf_xdna_endpoint_profile_t endpoint_profile_ = {
      .dma = {.address_bit_count = 48},
  };
  // Explicit live native device borrowed by the constructor.
  amdf_xdna_umd_device_t device_ = {};
  // Constructing owner's native state, including failed preparation progress.
  amdf_xdna_umd_memory_t* memory_ = nullptr;
  // Sentinel output unchanged by failed construction.
  amdf_xdna_umd_memory_result_t result_;
};

TEST_F(LinuxXdnaMemoryRollbackTest, RetainsFailedPreparationUntilOwnerCleanup) {
  const amdf_memory_native_profile_t profile = QueryProfile(0);
  const amdf_memory_native_create_info_t create_info = {.byte_length = 4096};
  EXPECT_EQ(amdf_xdna_umd_memory_prepare(&device_, &profile, &create_info,
                                         &memory_, &result_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  ReleasePreparedState();
}

TEST_F(LinuxXdnaMemoryRollbackTest, OwnerCanReleasePartiallyPreparedBuffer) {
  native_.close_error = 0;
  const amdf_memory_native_profile_t profile = QueryProfile(0);
  const amdf_memory_native_create_info_t create_info = {.byte_length = 4096};
  EXPECT_EQ(amdf_xdna_umd_memory_prepare(&device_, &profile, &create_info,
                                         &memory_, &result_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  ReleasePreparedState();
}

TEST_F(LinuxXdnaMemoryRollbackTest, FailedRegistrationLeavesCallerPagesAlone) {
  void* pages = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(pages, MAP_FAILED);
  const amdf_memory_native_profile_t profile = QueryProfile(2);
  const amdf_memory_native_create_info_t create_info = {
      .byte_length = 4096,
      .registered_host_pointer = pages,
  };
  EXPECT_EQ(amdf_xdna_umd_memory_prepare(&device_, &profile, &create_info,
                                         &memory_, &result_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  ReleasePreparedState();
  unsigned char residency = 0;
  EXPECT_EQ(mincore(pages, 4096, &residency), 0);
  EXPECT_EQ(munmap(pages, 4096), 0);
}

TEST_F(LinuxXdnaMemoryRollbackTest,
       FailedImportDoesNotConsumeCallerDescriptor) {
  // File metadata is real; the ioctl model supplies the DMA-BUF/GEM contract.
  const int descriptor = memfd_create("drm-memory-rollback", MFD_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  ASSERT_EQ(ftruncate(descriptor, 4096), 0);
  uint32_t release_count = 0;
  const amdf_external_memory_t external_memory = {
      .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .payload = {.file_descriptor = descriptor},
      .byte_length = 4096,
      .release =
          [](void* user_data, amdf_external_memory_type_t,
             amdf_external_memory_payload_t) {
            ++*static_cast<uint32_t*>(user_data);
          },
      .release_user_data = &release_count,
  };
  const amdf_memory_native_profile_t profile = QueryProfile(1);
  const amdf_memory_native_import_info_t import_info = {};
  EXPECT_EQ(
      amdf_xdna_umd_memory_prepare_import(&device_, &profile, &import_info,
                                          &external_memory, &memory_, &result_),
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  ReleasePreparedState();
  EXPECT_EQ(release_count, 0u);
  EXPECT_GE(fcntl(descriptor, F_GETFD), 0);
  EXPECT_GE(native_.backing_descriptor, 0);
  EXPECT_NE(native_.backing_descriptor, descriptor);
  EXPECT_EQ(close(descriptor), 0);
  EXPECT_GE(fcntl(native_.backing_descriptor, F_GETFD), 0);
}

TEST(LinuxXdnaMemoryPairTest, DescribesOnlyTheExactLocalXdnaSite) {
  const amdf_memory_access_info_t access_info = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
  };
  amdf_queue_family_info_t family = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA,
      .format_version = AMDF_XDNA_QUEUE_FORMAT_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_COMPUTE,
  };
  const amdf_memory_site_query_t query = {
      .access_info = &access_info,
      .queue_family_info = &family,
  };
  amdf_memory_site_description_t description = {};
  ASSERT_EQ(amdf_xdna_umd_memory_describe_site(nullptr, &query, &description),
            AMDF_STATUS_OK);
  EXPECT_EQ(description.capabilities, AMDF_MEMORY_SITE_CAPABILITY_READ |
                                          AMDF_MEMORY_SITE_CAPABILITY_WRITE);
  EXPECT_EQ(description.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(description.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  EXPECT_EQ(description.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(description.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
  EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);

  family.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  std::memset(&description, 0x5A, sizeof(description));
  const amdf_memory_site_description_t original = description;
  EXPECT_EQ(amdf_status_code(amdf_xdna_umd_memory_describe_site(nullptr, &query,
                                                                &description)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&description, &original, sizeof(description)), 0);
}

TEST(LinuxXdnaMemoryProfileTest,
     SeparatesOwnedImportedAndRegisteredHostProfiles) {
  const amdf_xdna_endpoint_profile_t endpoint_profile = {
      .dma = {.byte_offset = UINT32_C(0x80000000), .address_bit_count = 48},
  };
  amdf_xdna_umd_device_t device = {};
  device.page_size = 4096;
  device.profile = &endpoint_profile;
  amdf_memory_native_profile_t profile = {};
  ASSERT_EQ(amdf_xdna_umd_device_query_memory_profile(&device, 0, &profile),
            AMDF_STATUS_OK);

  EXPECT_EQ(profile.ordinal, 0u);
  EXPECT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE |
                               AMDF_MEMORY_PROFILE_ROLE_EXPORT |
                               AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(profile.guaranteed_flags, AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                          AMDF_MEMORY_FLAG_SHAREABLE |
                                          AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  EXPECT_EQ(profile.guaranteed_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(profile.supported_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(profile.device_address.address_bit_count, 48u);
  EXPECT_EQ(profile.device_address.minimum_address, 0u);
  EXPECT_EQ(profile.device_address.maximum_address, (UINT64_C(1) << 48) - 1);
  EXPECT_EQ(profile.allocation.maximum_byte_length,
            (UINT64_C(1) << 48) - UINT64_C(0x80000000));
  EXPECT_EQ(profile.address_kinds,
            (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE) |
                (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA));
  EXPECT_EQ(profile.allocation.minimum_alignment, 4096u);
  EXPECT_EQ(profile.allocation.maximum_alignment, 4096u);
  EXPECT_EQ(profile.host_mapping.byte_offset_granularity, 1u);
  ASSERT_EQ(profile.external_memory_support_count, 1u);
  EXPECT_EQ(profile.external_memory_support[0].type,
            AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
  EXPECT_EQ(profile.external_memory_support[0].flags,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS);
  EXPECT_EQ(profile.external_memory_support[0].source_offset_alignment, 1u);
  EXPECT_EQ(profile.external_memory_support[0].byte_length_alignment, 1u);

  amdf_memory_native_profile_t imported_profile = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_query_memory_profile(&device, 1, &imported_profile),
      AMDF_STATUS_OK);
  EXPECT_EQ(imported_profile.ordinal, 1u);
  EXPECT_EQ(imported_profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(imported_profile.roles, AMDF_MEMORY_PROFILE_ROLE_IMPORT |
                                        AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(imported_profile.guaranteed_flags,
            AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  EXPECT_EQ(imported_profile.import.minimum_alignment, 1u);
  EXPECT_EQ(imported_profile.import.maximum_alignment, 4096u);
  ASSERT_EQ(imported_profile.external_memory_support_count, 1u);
  EXPECT_EQ(imported_profile.external_memory_support[0].flags,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS |
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API);

  amdf_memory_native_profile_t registered_profile = {};
  ASSERT_EQ(amdf_xdna_umd_device_query_memory_profile(&device, 2,
                                                      &registered_profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(registered_profile.ordinal, 2u);
  EXPECT_EQ(registered_profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(registered_profile.roles, AMDF_MEMORY_PROFILE_ROLE_REGISTER |
                                          AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(registered_profile.guaranteed_flags,
            AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  EXPECT_EQ(registered_profile.guaranteed_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(registered_profile.supported_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(registered_profile.registration.byte_length_granularity, 1u);
  EXPECT_EQ(registered_profile.registration.registered_host_pointer_alignment,
            1u);
  EXPECT_EQ(registered_profile.registration.minimum_alignment, 1u);
  EXPECT_EQ(registered_profile.registration.maximum_alignment, 4096u);
  EXPECT_EQ(registered_profile.registration.native_byte_length_granularity,
            4096u);
  EXPECT_EQ(registered_profile.host_mapping.byte_offset_granularity, 1u);
  EXPECT_EQ(registered_profile.external_memory_support_count, 0u);

  profile.ordinal = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(amdf_xdna_umd_device_query_memory_profile(
                &device, 3, &profile)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(profile.ordinal, UINT32_MAX);
}

TEST(LinuxXdnaMemoryProfileTest, RequiresQualifiedDmaAddressFacts) {
  const amdf_xdna_endpoint_profile_t endpoint_profile = {};
  amdf_xdna_umd_device_t device = {};
  device.page_size = 4096;
  device.profile = &endpoint_profile;
  amdf_memory_native_profile_t profile;
  std::memset(&profile, 0xA5, sizeof(profile));
  const amdf_memory_native_profile_t original = profile;
  EXPECT_EQ(amdf_status_code(amdf_xdna_umd_device_query_memory_profile(
                &device, 0, &profile)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(std::memcmp(&profile, &original, sizeof(profile)), 0);
}

TEST(LinuxXdnaMemoryAddressTest, TranslatesCompleteLogicalRanges) {
  const amdf_xdna_endpoint_profile_t profile = {
      .dma = {.byte_offset = UINT32_C(0x80000000), .address_bit_count = 48},
  };
  amdf_xdna_umd_device_t device = {};
  device.profile = &profile;
  amdf_xdna_umd_memory_t memory = {};
  memory.device = &device;
  const uint64_t maximum_address = (UINT64_C(1) << 48) - 1;
  const uint64_t maximum_native_address =
      maximum_address - profile.dma.byte_offset;
  uint64_t address = UINT64_MAX;
  EXPECT_EQ(amdf_linux_xdna_memory_translate_dma_address(
                &memory, 0, maximum_native_address + 1, &address),
            AMDF_STATUS_OK);
  EXPECT_EQ(address, profile.dma.byte_offset);

  memory.buffer.device_address = maximum_native_address - 4095;
  EXPECT_EQ(
      amdf_linux_xdna_memory_translate_dma_address(&memory, 17, 4079, &address),
      AMDF_STATUS_OK);
  EXPECT_EQ(address, maximum_address - 4078);
  EXPECT_EQ(
      amdf_linux_xdna_memory_translate_dma_address(&memory, 4095, 1, &address),
      AMDF_STATUS_OK);
  EXPECT_EQ(address, maximum_address);
}

TEST(LinuxXdnaMemoryAddressTest, RejectsOverflowWithoutPublishingAddress) {
  const amdf_xdna_endpoint_profile_t profile = {
      .dma = {.byte_offset = UINT32_C(0x80000000), .address_bit_count = 48},
  };
  amdf_xdna_umd_device_t device = {};
  device.profile = &profile;
  amdf_xdna_umd_memory_t memory = {};
  memory.device = &device;
  const uint64_t maximum_native_address =
      (UINT64_C(1) << 48) - 1 - profile.dma.byte_offset;
  const struct {
    // Untranslated native buffer base returned by the driver.
    uint64_t native_address;
    // Byte offset of the requested logical range in that buffer.
    uint64_t byte_offset;
    // Nonempty logical extent to translate.
    uint64_t byte_length;
  } cases[] = {
      {maximum_native_address + 1, 0, 1},
      {maximum_native_address, 1, 1},
      {maximum_native_address, 0, 2},
      {0, 0, maximum_native_address + 2},
      {maximum_native_address - 4095, 17, 4080},
      {UINT64_MAX, 0, 1},
      {1, UINT64_MAX, 1},
      {1, 0, UINT64_MAX},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.native_address);
    SCOPED_TRACE(test_case.byte_offset);
    SCOPED_TRACE(test_case.byte_length);
    memory.buffer.device_address = test_case.native_address;
    uint64_t address = UINT64_C(0x12345678);
    EXPECT_EQ(
        amdf_status_code(amdf_linux_xdna_memory_translate_dma_address(
            &memory, test_case.byte_offset, test_case.byte_length, &address)),
        AMDF_STATUS_CODE_OUT_OF_RANGE);
    EXPECT_EQ(address, UINT64_C(0x12345678));
  }
}

}  // namespace
