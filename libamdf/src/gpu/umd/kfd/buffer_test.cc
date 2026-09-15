// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/buffer.h"

#include <errno.h>
#include <linux/kfd_ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/memory.h"

namespace {

// Models only the KFD ioctl dependency; reservations and buffer ownership use
// the production implementation and real host allocation/mapping operations.
struct NativeBufferState {
  // Number of map calls interrupted after completing the device prefix.
  uint32_t map_interruptions = 0;
  // Number of unmap calls interrupted after completing the device prefix.
  uint32_t unmap_interruptions = 0;
  // Number of free calls interrupted before consuming the allocation.
  uint32_t free_interruptions = 0;
  // Persistent map error returned before making native progress, or zero.
  int map_error = 0;
  // Persistent map error after establishing access, or zero.
  int map_completion_error = 0;
  // Maximum mapped prefix before map_completion_error; UINT32_MAX maps all.
  uint32_t map_completion_count = UINT32_MAX;
  // Persistent unmap error before consuming access, or zero.
  int unmap_error = 0;
  // Persistent unmap error after prefix progress or final synchronization.
  int unmap_completion_error = 0;
  // Maximum unmapped prefix before unmap_completion_error.
  uint32_t unmap_completion_count = UINT32_MAX;
  // Exact unique native participants, including the backing owner first.
  std::vector<uint32_t> gpu_ids = {19};
  // Mapping prefix established by the dependency, including pending sync.
  uint32_t mapped_gpu_count = 0;
  // Persistent free error before consuming the allocation, or zero.
  int free_error = 0;
  // Export failure before producing a descriptor, or zero when not exercised.
  int export_error = 0;
  // Number of backing identity export attempts.
  uint32_t export_count = 0;
  // Native allocation address within the real host reservation.
  uintptr_t address = 0;
  // Page-covered length expected at the native allocation boundary.
  size_t byte_length = 4096;
  // Exact allocation arguments before the dependency publishes its handle.
  kfd_ioctl_alloc_memory_of_gpu_args allocation = {};
  // Number of buffer metadata allocations returned to the host allocator.
  uint32_t metadata_free_count = 0;
  // Input map progress observed by every native call.
  std::vector<uint32_t> map_progress;
  // Input unmap progress observed by every native call.
  std::vector<uint32_t> unmap_progress;
  // Number of native free attempts, including interruptions.
  uint32_t free_count = 0;
  // Whether the native allocation has not yet been consumed by free.
  bool allocation_live = false;
  // Whether GPU access or its unmap synchronization remains live.
  bool access_live = false;
};

// Other threads and calls outside a native fixture retain the real ioctl.
thread_local NativeBufferState* native_state = nullptr;

}  // namespace

extern "C" int __real_ioctl(int descriptor, unsigned long request, ...);

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_state == nullptr) {
    return __real_ioctl(descriptor, request, argument);
  }
  EXPECT_EQ(descriptor, 17);
  switch (request) {
    case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: {
      auto* allocate =
          static_cast<kfd_ioctl_alloc_memory_of_gpu_args*>(argument);
      EXPECT_EQ(allocate->gpu_id, 19u);
      EXPECT_EQ(allocate->size, native_state->byte_length);
      EXPECT_FALSE(native_state->allocation_live);
      native_state->allocation = *allocate;
      allocate->handle = 0x1234;
      native_state->address = allocate->va_addr;
      native_state->allocation_live = true;
      return 0;
    }
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU: {
      auto* map = static_cast<kfd_ioctl_map_memory_to_gpu_args*>(argument);
      EXPECT_EQ(map->handle, 0x1234u);
      EXPECT_EQ(map->n_devices, native_state->gpu_ids.size());
      const auto* gpu_ids =
          reinterpret_cast<const uint32_t*>(map->device_ids_array_ptr);
      EXPECT_EQ(std::vector<uint32_t>(gpu_ids, gpu_ids + map->n_devices),
                native_state->gpu_ids);
      native_state->map_progress.push_back(map->n_success);
      if (native_state->map_error != 0) {
        errno = native_state->map_error;
        return -1;
      }
      map->n_success =
          std::min(map->n_devices, native_state->map_completion_count);
      native_state->mapped_gpu_count = map->n_success;
      native_state->access_live = map->n_success != 0;
      if (native_state->map_completion_error != 0) {
        errno = native_state->map_completion_error;
        return -1;
      }
      if (native_state->map_interruptions != 0) {
        --native_state->map_interruptions;
        errno = EINTR;
        return -1;
      }
      return 0;
    }
    case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
      auto* unmap =
          static_cast<kfd_ioctl_unmap_memory_from_gpu_args*>(argument);
      EXPECT_EQ(unmap->handle, 0x1234u);
      EXPECT_EQ(unmap->n_devices, native_state->mapped_gpu_count);
      const auto* gpu_ids =
          reinterpret_cast<const uint32_t*>(unmap->device_ids_array_ptr);
      EXPECT_EQ(std::vector<uint32_t>(gpu_ids, gpu_ids + unmap->n_devices),
                std::vector<uint32_t>(native_state->gpu_ids.begin(),
                                      native_state->gpu_ids.begin() +
                                          native_state->mapped_gpu_count));
      native_state->unmap_progress.push_back(unmap->n_success);
      if (native_state->unmap_error != 0) {
        errno = native_state->unmap_error;
        return -1;
      }
      unmap->n_success =
          std::min(unmap->n_devices, native_state->unmap_completion_count);
      if (native_state->unmap_completion_error != 0) {
        errno = native_state->unmap_completion_error;
        return -1;
      }
      if (native_state->unmap_interruptions != 0) {
        --native_state->unmap_interruptions;
        errno = EINTR;
        return -1;
      }
      native_state->access_live = false;
      return 0;
    }
    case AMDKFD_IOC_FREE_MEMORY_OF_GPU: {
      auto* release = static_cast<kfd_ioctl_free_memory_of_gpu_args*>(argument);
      EXPECT_EQ(release->handle, 0x1234u);
      EXPECT_TRUE(native_state->allocation_live);
      EXPECT_FALSE(native_state->access_live);
      ++native_state->free_count;
      if (native_state->free_error != 0) {
        errno = native_state->free_error;
        return -1;
      }
      if (native_state->free_interruptions != 0) {
        --native_state->free_interruptions;
        errno = EINTR;
        return -1;
      }
      native_state->allocation_live = false;
      return 0;
    }
    case AMDKFD_IOC_EXPORT_DMABUF:
      ++native_state->export_count;
      EXPECT_NE(native_state->export_error, 0);
      errno = native_state->export_error;
      return -1;
    default:
      ADD_FAILURE() << "unexpected KFD buffer ioctl: " << request;
      errno = ENOTTY;
      return -1;
  }
}

namespace {

class KfdBufferNativeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_state = &native_;
    device_.host_allocator = amdf_allocator_system();
    device_.host_allocator.user_data = &native_;
    device_.host_allocator.free = [](void* user_data, void* allocation) {
      ++static_cast<NativeBufferState*>(user_data)->metadata_free_count;
      amdf_free(amdf_allocator_system(), allocation);
    };
    device_.descriptor = 17;
    device_.render_descriptor = memfd_create("amdf-kfd-buffer", MFD_CLOEXEC);
    ASSERT_GE(device_.render_descriptor, 0);
    ASSERT_EQ(ftruncate(device_.render_descriptor, 4096), 0);
    device_.page_size = 4096;
    device_.topology.gpu_id = 19;
    device_.topology.virtual_address.end = UINT64_MAX;
    create_info_.native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
    create_info_.byte_length = 4096;
    create_info_.alignment = 4096;
  }

  void TearDown() override {
    native_.map_interruptions = 0;
    native_.unmap_interruptions = 0;
    native_.free_interruptions = 0;
    if (buffer_ != nullptr) {
      EXPECT_EQ(amdf_gpu_kfd_buffer_destroy(buffer_), AMDF_STATUS_OK);
    }
    EXPECT_FALSE(native_.allocation_live);
    EXPECT_FALSE(native_.access_live);
    EXPECT_EQ(native_.metadata_free_count, expected_metadata_free_count_);
    if (device_.render_descriptor >= 0) {
      EXPECT_EQ(close(device_.render_descriptor), 0);
    }
    native_state = nullptr;
  }

  amdf_status_t Create() {
    return amdf_gpu_kfd_buffer_create(&device_, &create_info_, &buffer_,
                                      &result_);
  }

  amdf_status_t Destroy() {
    const amdf_status_t status = amdf_gpu_kfd_buffer_destroy(buffer_);
    if (amdf_status_is_ok(status)) buffer_ = nullptr;
    return status;
  }

  void ReleaseLeakedReservation() {
    // There is no real kernel allocation in this dependency model. Verify that
    // production rollback preserved its reservation, then reclaim the known
    // test mapping without invoking another native release attempt.
    void* reservation = reinterpret_cast<void*>(native_.address);
    unsigned char residency = 0;
    EXPECT_EQ(mincore(reservation, 4096, &residency), 0);
    EXPECT_EQ(munmap(reservation, 4096), 0);
    native_.allocation_live = false;
    native_.access_live = false;
  }

  // Native dependency state retained until the production object is released.
  NativeBufferState native_;
  // Borrowed device identity and complete CPU/GPU reservation limits.
  amdf_gpu_umd_device_t device_ = {};
  // Ordinary GTT request without a CPU backing mapping.
  amdf_gpu_kfd_buffer_create_info_t create_info_ = {};
  // Published buffer, or NULL before construction and after successful release.
  amdf_gpu_kfd_buffer_t* buffer_ = nullptr;
  // Address information published together with the buffer.
  amdf_gpu_kfd_buffer_result_t result_ = {};
  // One buffer header, plus a memory header when exercising its native owner.
  uint32_t expected_metadata_free_count_ = 1;
};

TEST_F(KfdBufferNativeTest, CompletesInterruptedMapWithNativeProgress) {
  native_.map_interruptions = 2;
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0, 1, 1}));
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.free_count, 0u);
}

TEST_F(KfdBufferNativeTest, MapsTheUniqueGroupInOneNativeOperation) {
  amdf_gpu_umd_device_t peer = device_;
  peer.topology.gpu_id = 23;
  amdf_gpu_umd_device_t* peers[] = {&peer, &device_, &peer};
  create_info_.peer_count = 3;
  create_info_.peer_devices = peers;
  native_.gpu_ids = {19, 23};
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 1u);
}

TEST_F(KfdBufferNativeTest, PartialGroupMapRollsBackOnlyTheMappedPrefix) {
  amdf_gpu_umd_device_t peer = device_;
  peer.topology.gpu_id = 23;
  amdf_gpu_umd_device_t* peers[] = {&peer};
  create_info_.peer_count = 1;
  create_info_.peer_devices = peers;
  native_.gpu_ids = {19, 23};
  native_.map_completion_count = 1;
  native_.map_completion_error = EIO;
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 1u);
}

TEST_F(KfdBufferNativeTest, RegisteredGroupPreservesCallerPagesThroughUnmap) {
  void* pages = mmap(nullptr, 12288, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(pages, MAP_FAILED);
  std::memset(pages, 0xA7, 12288);
  amdf_gpu_umd_device_t peer = device_;
  peer.topology.gpu_id = 23;
  amdf_gpu_umd_device_t* peers[] = {&peer};
  create_info_.peer_count = 1;
  create_info_.peer_devices = peers;
  create_info_.native_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  create_info_.host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED;
  create_info_.host_byte_offset = 2051;
  create_info_.host_pointer = static_cast<uint8_t*>(pages) + 2051;
  create_info_.byte_length = 8192;
  native_.byte_length = 8192;
  native_.gpu_ids = {19, 23};
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.allocation.flags, create_info_.native_flags);
  EXPECT_EQ(native_.allocation.mmap_offset, reinterpret_cast<uintptr_t>(pages));
  EXPECT_EQ(result_.device_address, native_.address + 2051);
  EXPECT_EQ(result_.host_pointer, create_info_.host_pointer);
  native_.unmap_completion_error = EIO;
  EXPECT_EQ(Destroy(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(static_cast<uint8_t*>(pages)[2051], 0xA7);
  // The dependency resolves its injected final-sync failure; the owner resumes
  // from the consumed prefix, without replaying either mapping or allocation.
  native_.unmap_completion_error = 0;
  ASSERT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0, 2}));
  EXPECT_EQ(native_.free_count, 1u);
  std::memset(pages, 0x69, 12288);
  EXPECT_EQ(munmap(pages, 12288), 0);
}

TEST_F(KfdBufferNativeTest, GroupRetainsProgressThroughFinalSynchronization) {
  amdf_gpu_umd_device_t peer = device_;
  peer.topology.gpu_id = 23;
  amdf_gpu_umd_device_t* peers[] = {&peer};
  create_info_.peer_count = 1;
  create_info_.peer_devices = peers;
  native_.gpu_ids = {19, 23};
  native_.map_interruptions = 1;
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0, 2}));
  native_.unmap_completion_count = 1;
  native_.unmap_completion_error = EIO;
  EXPECT_EQ(Destroy(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 0u);
  native_.unmap_completion_count = 2;
  EXPECT_EQ(Destroy(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 0u);
  native_.unmap_completion_error = 0;
  EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0, 1, 2}));
  EXPECT_EQ(native_.free_count, 1u);
}

TEST_F(KfdBufferNativeTest, FailedGroupUnmapPreservesBackingAndReservation) {
  amdf_gpu_umd_device_t peer = device_;
  peer.topology.gpu_id = 23;
  amdf_gpu_umd_device_t* peers[] = {&peer};
  create_info_.peer_count = 1;
  create_info_.peer_devices = peers;
  native_.gpu_ids = {19, 23};
  native_.map_completion_error = EIO;
  native_.unmap_completion_count = 1;
  native_.unmap_completion_error = ENOMEM;
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 1u);
  ReleaseLeakedReservation();
}

TEST_F(KfdBufferNativeTest, ChecksEveryLiveAddressEnvelopeBeforeAllocation) {
  amdf_gpu_umd_device_t peer = device_;
  peer.topology.gpu_id = 23;
  peer.topology.virtual_address.end = 4096;
  amdf_gpu_umd_device_t* peers[] = {&peer};
  create_info_.peer_count = 1;
  create_info_.peer_devices = peers;
  EXPECT_EQ(amdf_status_code(Create()), AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(native_.address, 0u);
  EXPECT_TRUE(native_.map_progress.empty());
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.free_count, 0u);
}

TEST_F(KfdBufferNativeTest, CompletesInterruptedUnmapBeforeFree) {
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  native_.unmap_interruptions = 2;
  EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0, 1, 1}));
  EXPECT_EQ(native_.free_count, 1u);
}

TEST_F(KfdBufferNativeTest, CompletesInterruptedFreeBeforeConsumingHandle) {
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  native_.free_interruptions = 2;
  EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 3u);
}

TEST_F(KfdBufferNativeTest, DoesNotRetryNonInterruptionMapFailure) {
  native_.map_error = ENOMEM;
  std::memset(&result_, 0xA5, sizeof(result_));
  const amdf_gpu_kfd_buffer_result_t original_result = result_;
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(std::memcmp(&result_, &original_result, sizeof(result_)), 0);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.free_count, 1u);
}

TEST_F(KfdBufferNativeTest,
       FailedConstructionReportsUnmapFailureAndLeaksNative) {
  native_.map_completion_error = EIO;
  native_.unmap_error = ENOMEM;
  std::memset(&result_, 0xA5, sizeof(result_));
  const amdf_gpu_kfd_buffer_result_t original_result = result_;
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(std::memcmp(&result_, &original_result, sizeof(result_)), 0);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 1u);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_TRUE(native_.access_live);
  ReleaseLeakedReservation();
}

TEST_F(KfdBufferNativeTest,
       FailedConstructionReportsFreeFailureAndLeaksNative) {
  native_.map_error = EIO;
  native_.free_error = ENOMEM;
  std::memset(&result_, 0xA5, sizeof(result_));
  const amdf_gpu_kfd_buffer_result_t original_result = result_;
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(std::memcmp(&result_, &original_result, sizeof(result_)), 0);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_EQ(native_.metadata_free_count, 1u);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_FALSE(native_.access_live);
  ReleaseLeakedReservation();
}

TEST_F(KfdBufferNativeTest, DiscardConsumesUnpublishedBufferAfterFreeFailure) {
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  native_.free_error = ENOMEM;
  EXPECT_EQ(amdf_gpu_kfd_buffer_discard(buffer_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  buffer_ = nullptr;
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_EQ(native_.metadata_free_count, 1u);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_FALSE(native_.access_live);
  ReleaseLeakedReservation();
}

TEST_F(KfdBufferNativeTest, DestroyFailureRetainsCallerOwnedBuffer) {
  ASSERT_EQ(Create(), AMDF_STATUS_OK);
  native_.free_error = ENOMEM;
  EXPECT_EQ(Destroy(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_NE(buffer_, nullptr);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_EQ(native_.metadata_free_count, 0u);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_FALSE(native_.access_live);
  native_.free_error = 0;
  EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 2u);
}

TEST_F(KfdBufferNativeTest, MemoryOwnerRetainsPartialBufferPreparation) {
  expected_metadata_free_count_ = 2;
  native_.map_completion_error = EIO;
  native_.unmap_error = ENOMEM;
  amdf_memory_native_profile_t profile;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &profile),
            AMDF_STATUS_OK);
  const amdf_memory_native_create_info_t create_info = {
      .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .byte_length = 4096,
      .minimum_alignment = 4096,
  };
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_memory_result_t original_result = result;
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile,
                                        &create_info, &memory, &result),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(native_.metadata_free_count, 0u);
  amdf_gpu_umd_memory_abandon(memory);
  EXPECT_EQ(native_.metadata_free_count, 2u);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_TRUE(native_.access_live);
  ReleaseLeakedReservation();
}

TEST_F(KfdBufferNativeTest, MemoryOwnerRetainsBackingAfterIdentityQueryFails) {
  expected_metadata_free_count_ = 2;
  native_.export_error = EIO;
  native_.free_error = ENOMEM;
  amdf_memory_native_profile_t profile;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &profile),
            AMDF_STATUS_OK);
  const amdf_memory_native_create_info_t create_info = {
      .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .byte_length = 4096,
      .minimum_alignment = 4096,
  };
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_memory_result_t original_result = result;
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile,
                                        &create_info, &memory, &result),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(native_.export_count, 1u);
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(native_.metadata_free_count, 0u);
  amdf_gpu_umd_memory_abandon(memory);
  EXPECT_EQ(native_.metadata_free_count, 2u);
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_FALSE(native_.access_live);
  ReleaseLeakedReservation();
}

TEST(KfdBufferTest, RejectsMalformedConstructionWithoutPublishingOutputs) {
  amdf_gpu_umd_device_t device = {};
  device.page_size = 4096;
  amdf_gpu_kfd_buffer_create_info_t create_info = {};
  create_info.byte_length = 4096;
  create_info.alignment = 4096;

  auto* const sentinel = reinterpret_cast<amdf_gpu_kfd_buffer_t*>(uintptr_t{1});
  amdf_gpu_kfd_buffer_t* buffer = sentinel;
  amdf_gpu_kfd_buffer_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_kfd_buffer_result_t original_result = result;

  create_info.byte_length = 0;
  EXPECT_EQ(amdf_status_code(amdf_gpu_kfd_buffer_create(&device, &create_info,
                                                        &buffer, &result)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(buffer, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);

  create_info.byte_length = 4096;
  create_info.alignment = 6144;
  EXPECT_EQ(amdf_status_code(amdf_gpu_kfd_buffer_create(&device, &create_info,
                                                        &buffer, &result)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(buffer, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);

  create_info.alignment = 4096;
  create_info.host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED;
  EXPECT_EQ(amdf_status_code(amdf_gpu_kfd_buffer_create(&device, &create_info,
                                                        &buffer, &result)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(buffer, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
}

}  // namespace
