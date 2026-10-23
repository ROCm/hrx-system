// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/doorbell.h"

#include <linux/kfd_ioctl.h>
#include <sys/mman.h>

#include <cerrno>
#include <cstdarg>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"

namespace {

// Only the KFD dependency is modeled. Ownership and VA reservations use the
// production implementation; no native GPU or driver is accessed by this test.
struct NativeDoorbellState {
  // Connection lock shared with ordinary memory and native queue lifecycle.
  amdf_gpu_kfd_memory_lock_t memory_lock = {};
  // Allocation arguments captured before the dependency returns its handle.
  kfd_ioctl_alloc_memory_of_gpu_args allocation = {};
  // Allocation failure before acquiring native backing, or zero.
  int allocation_error = 0;
  // Number of allocation attempts interrupted before native ownership exists.
  uint32_t allocation_interruptions = 0;
  // Total native allocation attempts, including interruptions.
  uint32_t allocation_count = 0;
  // Map failure after the reported prefix, or zero.
  int map_error = 0;
  // Prefix reported by map, including on a final synchronization error.
  uint32_t mapped_count = 1;
  // Number of interrupted mapping waits before success.
  uint32_t map_interruptions = 0;
  // Number of interrupted unmapping waits before success.
  uint32_t unmap_interruptions = 0;
  // Terminal unmap error, or zero.
  int unmap_error = 0;
  // Terminal special-allocation release error, or zero.
  int release_error = 0;
  // Whether native backing remains live in the dependency model.
  bool allocated = false;
  // Whether a GPU mapping or its final synchronization remains live.
  bool mapped = false;
  // Map progress supplied to each call, including resumed waits.
  std::vector<uint32_t> map_progress;
  // Unmap progress supplied to each call, including resumed waits.
  std::vector<uint32_t> unmap_progress;
  // Number of native release attempts.
  uint32_t release_count = 0;
  // Number of production metadata allocations freed.
  uint32_t metadata_free_count = 0;
};

thread_local NativeDoorbellState* native_state = nullptr;

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
  EXPECT_EQ(pthread_mutex_lock(&native_state->memory_lock.mutex), 0);
  EXPECT_TRUE(native_state->memory_lock.occupied);
  EXPECT_EQ(pthread_mutex_unlock(&native_state->memory_lock.mutex), 0);
  switch (request) {
    case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: {
      auto* allocate =
          static_cast<kfd_ioctl_alloc_memory_of_gpu_args*>(argument);
      native_state->allocation = *allocate;
      EXPECT_FALSE(native_state->allocated);
      ++native_state->allocation_count;
      if (native_state->allocation_interruptions != 0) {
        --native_state->allocation_interruptions;
        errno = EINTR;
        return -1;
      }
      if (native_state->allocation_error != 0) {
        errno = native_state->allocation_error;
        return -1;
      }
      allocate->handle = 0x1234;
      native_state->allocated = true;
      return 0;
    }
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU: {
      auto* map = static_cast<kfd_ioctl_map_memory_to_gpu_args*>(argument);
      EXPECT_EQ(map->handle, 0x1234u);
      EXPECT_EQ(map->n_devices, 1u);
      EXPECT_EQ(*reinterpret_cast<uint32_t*>(map->device_ids_array_ptr), 19u);
      native_state->map_progress.push_back(map->n_success);
      map->n_success = native_state->mapped_count;
      native_state->mapped = map->n_success != 0;
      if (native_state->map_error != 0) {
        errno = native_state->map_error;
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
      EXPECT_EQ(unmap->n_devices, 1u);
      EXPECT_EQ(*reinterpret_cast<uint32_t*>(unmap->device_ids_array_ptr), 19u);
      native_state->unmap_progress.push_back(unmap->n_success);
      unmap->n_success = 1;
      if (native_state->unmap_error != 0) {
        errno = native_state->unmap_error;
        return -1;
      }
      if (native_state->unmap_interruptions != 0) {
        --native_state->unmap_interruptions;
        errno = EINTR;
        return -1;
      }
      native_state->mapped = false;
      return 0;
    }
    case AMDKFD_IOC_FREE_MEMORY_OF_GPU: {
      const auto* release =
          static_cast<const kfd_ioctl_free_memory_of_gpu_args*>(argument);
      EXPECT_EQ(release->handle, 0x1234u);
      EXPECT_TRUE(native_state->allocated);
      EXPECT_FALSE(native_state->mapped);
      ++native_state->release_count;
      if (native_state->release_error != 0) {
        errno = native_state->release_error;
        return -1;
      }
      native_state->allocated = false;
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected native doorbell operation " << request;
      errno = ENOTTY;
      return -1;
  }
}

namespace {

class KfdDoorbellTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_state = &native_;
    ASSERT_EQ(amdf_gpu_kfd_memory_lock_initialize(&native_.memory_lock), 0);
    device_.host_allocator = amdf_allocator_system();
    device_.host_allocator.user_data = &native_;
    device_.host_allocator.free = [](void* user_data, void* allocation) {
      ++static_cast<NativeDoorbellState*>(user_data)->metadata_free_count;
      amdf_free(amdf_allocator_system(), allocation);
    };
    device_.descriptor = 17;
    device_.memory_lock = &native_.memory_lock;
    device_.page_size = 4096;
    device_.topology.gpu_id = 19;
    device_.topology.virtual_address.end = UINT64_MAX;
  }

  void TearDown() override {
    if (doorbell_ != nullptr) {
      EXPECT_EQ(amdf_gpu_kfd_doorbell_destroy(doorbell_), AMDF_STATUS_OK);
    }
    EXPECT_FALSE(native_.allocated);
    EXPECT_FALSE(native_.mapped);
    EXPECT_EQ(native_.metadata_free_count, 1u);
    native_state = nullptr;
    EXPECT_FALSE(native_.memory_lock.occupied);
    amdf_gpu_kfd_memory_lock_deinitialize(&native_.memory_lock);
  }

  void ReclaimModeledLeak() {
    // No driver backing exists in this model. Verify the production owner
    // retained its actual reservation, then reclaim that test-only interval.
    void* reservation = reinterpret_cast<void*>(native_.allocation.va_addr);
    unsigned char residency[2] = {};
    EXPECT_EQ(mincore(reservation, 8192, residency), 0);
    EXPECT_EQ(munmap(reservation, 8192), 0);
    native_.allocated = false;
    native_.mapped = false;
  }

  // Modeled platform state, never a native GPU connection.
  NativeDoorbellState native_;
  // Exact borrowed device used by the production owner.
  amdf_gpu_umd_device_t device_ = {};
  // Live production owner, or NULL after a consuming operation.
  amdf_gpu_kfd_doorbell_t* doorbell_ = nullptr;
  // Successful producer-local address result.
  uint64_t address_ = 0;
};

TEST_F(KfdDoorbellTest, OwnsUncachedExactDeviceMapping) {
  native_.allocation_interruptions = 2;
  ASSERT_EQ(amdf_gpu_kfd_doorbell_create(&device_, 8192, &doorbell_, &address_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.allocation_count, 3u);
  EXPECT_EQ(address_, native_.allocation.va_addr);
  EXPECT_EQ(native_.allocation.size, 8192u);
  EXPECT_EQ(native_.allocation.gpu_id, 19u);
  EXPECT_EQ(native_.allocation.mmap_offset, 0u);
  EXPECT_EQ(
      native_.allocation.flags,
      KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL | AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
          KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  ASSERT_EQ(amdf_gpu_kfd_doorbell_destroy(doorbell_), AMDF_STATUS_OK);
  doorbell_ = nullptr;
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.release_count, 1u);
}

TEST_F(KfdDoorbellTest, ResumesMapAndUnmapWithCompletedPrefix) {
  native_.map_interruptions = 1;
  native_.unmap_interruptions = 1;
  ASSERT_EQ(amdf_gpu_kfd_doorbell_create(&device_, 8192, &doorbell_, &address_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0, 1}));
  ASSERT_EQ(amdf_gpu_kfd_doorbell_destroy(doorbell_), AMDF_STATUS_OK);
  doorbell_ = nullptr;
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0, 1}));
  EXPECT_EQ(native_.release_count, 1u);
}

TEST_F(KfdDoorbellTest, FailedMapRollsBackCompletedAccessWithoutPublishing) {
  native_.map_error = EIO;
  auto* const sentinel =
      reinterpret_cast<amdf_gpu_kfd_doorbell_t*>(uintptr_t{1});
  auto* output = sentinel;
  uint64_t address = UINT64_C(0xabc123);
  EXPECT_EQ(amdf_gpu_kfd_doorbell_create(&device_, 8192, &output, &address),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(output, sentinel);
  EXPECT_EQ(address, UINT64_C(0xabc123));
  EXPECT_EQ(native_.unmap_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.release_count, 1u);
}

TEST_F(KfdDoorbellTest, AllocationFailureCreatesNoNativeCleanupObligation) {
  native_.allocation_interruptions = 1;
  native_.allocation_error = ENOMEM;
  EXPECT_EQ(amdf_gpu_kfd_doorbell_create(&device_, 8192, &doorbell_, &address_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(native_.allocation_count, 2u);
  EXPECT_EQ(doorbell_, nullptr);
  EXPECT_EQ(address_, 0u);
  EXPECT_TRUE(native_.map_progress.empty());
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.release_count, 0u);
}

TEST_F(KfdDoorbellTest, FailedRollbackPreservesReservationAndReportsCleanup) {
  native_.map_error = ENOMEM;
  native_.unmap_error = EIO;
  EXPECT_EQ(amdf_gpu_kfd_doorbell_create(&device_, 8192, &doorbell_, &address_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(doorbell_, nullptr);
  EXPECT_EQ(address_, 0u);
  EXPECT_EQ(native_.release_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 1u);
  ReclaimModeledLeak();
}

TEST_F(KfdDoorbellTest, TerminalReleaseDoesNotRetrySpecialAllocation) {
  ASSERT_EQ(amdf_gpu_kfd_doorbell_create(&device_, 8192, &doorbell_, &address_),
            AMDF_STATUS_OK);
  native_.release_error = EINTR;
  EXPECT_EQ(amdf_gpu_kfd_doorbell_destroy(doorbell_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EINTR));
  doorbell_ = nullptr;
  EXPECT_EQ(native_.release_count, 1u);
  EXPECT_EQ(native_.metadata_free_count, 1u);
  ReclaimModeledLeak();
}

TEST_F(KfdDoorbellTest, AbandonPreservesNativeAccessAndReservation) {
  ASSERT_EQ(amdf_gpu_kfd_doorbell_create(&device_, 8192, &doorbell_, &address_),
            AMDF_STATUS_OK);
  amdf_gpu_kfd_doorbell_abandon(doorbell_);
  doorbell_ = nullptr;
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(native_.release_count, 0u);
  EXPECT_EQ(native_.metadata_free_count, 1u);
  ReclaimModeledLeak();
}

}  // namespace
