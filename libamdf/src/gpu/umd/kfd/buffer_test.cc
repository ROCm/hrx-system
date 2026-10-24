// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/buffer.h"

#include <drm/drm.h>
#include <errno.h>
#include <linux/kfd_ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/memory.h"

namespace {

// Models native device operations and selected host cleanup failures. Buffer
// ownership, descriptors, and address reservations use production code and real
// host storage; a failed terminal cleanup is reclaimed directly by the fixture.
struct NativeBufferState {
  // Shared native connection lock borrowed by every modeled device handle.
  amdf_gpu_kfd_memory_lock_t memory_lock = {};
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
  // Import failure before native ownership is established.
  int import_error = 0;
  // Fresh DMA-BUF descriptor borrowed by addressless residency preparation.
  int import_descriptor = -1;
  // Exact failure establishing the render-file source, or zero.
  int prime_error = 0;
  // Exact failure releasing the render-file source, or zero.
  int gem_close_error = 0;
  // Whether the render-file GEM reference remains live.
  bool gem_live = false;
  // Number of terminal render-source release attempts.
  uint32_t gem_close_count = 0;
  // Export failure before producing a descriptor, or zero when not exercised.
  int export_error = 0;
  // Number of backing identity export attempts.
  uint32_t export_count = 0;
  // Temporary exported descriptor owned until its consuming close.
  int export_descriptor = -1;
  // Error after consuming the temporary export once its GEM source is acquired.
  int export_close_error = 0;
  // Error retaining the real host reservation during terminal destruction.
  int munmap_error = 0;
  // Native allocation address within the real host reservation.
  uintptr_t address = 0;
  // Page-covered length expected at the native allocation boundary.
  size_t byte_length = 4096;
  // Exact allocation arguments before the dependency publishes its handle.
  kfd_ioctl_alloc_memory_of_gpu_args allocation = {};
  // Input identities observed before each native construction attempt.
  std::vector<kfd_ioctl_alloc_memory_of_gpu_args> allocation_attempts;
  // Per-attempt errors before native ownership is published; zero succeeds.
  std::vector<int> allocation_errors;
  // Optional observation after recording an attempt, under native ownership.
  std::function<void()> observe_allocation;
  // Successful host unmaps at the recorded native allocation address.
  uint32_t reservation_release_count = 0;
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
extern "C" int __real_close(int descriptor);
extern "C" int __real_munmap(void* address, size_t byte_length);

extern "C" int __wrap_close(int descriptor) {
  const int result = __real_close(descriptor);
  if (native_state != nullptr &&
      descriptor == native_state->export_descriptor) {
    native_state->export_descriptor = -1;
    if (result == 0 && native_state->gem_live &&
        native_state->export_close_error != 0) {
      errno = native_state->export_close_error;
      return -1;
    }
  }
  return result;
}

extern "C" int __wrap_munmap(void* address, size_t byte_length) {
  if (native_state != nullptr && native_state->munmap_error != 0 &&
      address == reinterpret_cast<void*>(native_state->address)) {
    errno = native_state->munmap_error;
    return -1;
  }
  const int result = __real_munmap(address, byte_length);
  if (result == 0 && native_state != nullptr &&
      address == reinterpret_cast<void*>(native_state->address)) {
    ++native_state->reservation_release_count;
  }
  return result;
}

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_state == nullptr) {
    return __real_ioctl(descriptor, request, argument);
  }
  if (request == DRM_IOCTL_PRIME_FD_TO_HANDLE) {
    auto* import = static_cast<drm_prime_handle*>(argument);
    EXPECT_EQ(descriptor, native_state->import_descriptor);
    EXPECT_EQ(import->fd, native_state->export_descriptor >= 0
                              ? native_state->export_descriptor
                              : native_state->import_descriptor);
    EXPECT_TRUE(native_state->allocation_live);
    EXPECT_FALSE(native_state->gem_live);
    if (native_state->prime_error != 0) {
      errno = native_state->prime_error;
      return -1;
    }
    native_state->gem_live = true;
    import->handle = 0x7654;
    return 0;
  }
  if (request == DRM_IOCTL_GEM_CLOSE) {
    EXPECT_EQ(descriptor, native_state->import_descriptor);
    EXPECT_EQ(static_cast<drm_gem_close*>(argument)->handle, 0x7654u);
    EXPECT_TRUE(native_state->allocation_live);
    EXPECT_TRUE(native_state->gem_live);
    ++native_state->gem_close_count;
    if (native_state->gem_close_error != 0) {
      errno = native_state->gem_close_error;
      return -1;
    }
    native_state->gem_live = false;
    return 0;
  }
  EXPECT_EQ(descriptor, 17);
  if (request != AMDKFD_IOC_EXPORT_DMABUF) {
    // Native completion includes the flush after KFD drops its own mutex.
    // Every retry and failure path must still exclude another caller here.
    EXPECT_EQ(pthread_mutex_lock(&native_state->memory_lock.mutex), 0);
    EXPECT_TRUE(native_state->memory_lock.occupied);
    EXPECT_EQ(pthread_mutex_unlock(&native_state->memory_lock.mutex), 0);
  }
  switch (request) {
    case AMDKFD_IOC_IMPORT_DMABUF: {
      auto* import = static_cast<kfd_ioctl_import_dmabuf_args*>(argument);
      EXPECT_EQ(import->gpu_id, 19u);
      EXPECT_EQ(import->va_addr, 0u);
      EXPECT_EQ(import->dmabuf_fd,
                static_cast<uint32_t>(native_state->import_descriptor));
      EXPECT_FALSE(native_state->allocation_live);
      if (native_state->import_error != 0) {
        errno = native_state->import_error;
        return -1;
      }
      import->handle = 0x1234;
      native_state->allocation_live = true;
      return 0;
    }
    case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: {
      auto* allocate =
          static_cast<kfd_ioctl_alloc_memory_of_gpu_args*>(argument);
      EXPECT_EQ(allocate->gpu_id, 19u);
      EXPECT_EQ(allocate->size, native_state->byte_length);
      EXPECT_FALSE(native_state->allocation_live);
      native_state->allocation = *allocate;
      native_state->address = allocate->va_addr;
      const size_t attempt = native_state->allocation_attempts.size();
      native_state->allocation_attempts.push_back(*allocate);
      if (native_state->observe_allocation) {
        native_state->observe_allocation();
      }
      if (attempt < native_state->allocation_errors.size() &&
          native_state->allocation_errors[attempt] != 0) {
        errno = native_state->allocation_errors[attempt];
        return -1;
      }
      allocate->handle = 0x1234;
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
      EXPECT_FALSE(native_state->gem_live);
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
    case AMDKFD_IOC_EXPORT_DMABUF: {
      ++native_state->export_count;
      if (native_state->export_error != 0) {
        errno = native_state->export_error;
        return -1;
      }
      native_state->export_descriptor = dup(native_state->import_descriptor);
      EXPECT_GE(native_state->export_descriptor, 0);
      static_cast<kfd_ioctl_export_dmabuf_args*>(argument)->dmabuf_fd =
          native_state->export_descriptor;
      return 0;
    }
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
    ASSERT_EQ(amdf_gpu_kfd_memory_lock_initialize(&native_.memory_lock), 0);
    device_.host_allocator = amdf_allocator_system();
    device_.host_allocator.user_data = &native_;
    device_.host_allocator.free = [](void* user_data, void* allocation) {
      ++static_cast<NativeBufferState*>(user_data)->metadata_free_count;
      amdf_free(amdf_allocator_system(), allocation);
    };
    device_.descriptor = 17;
    device_.memory_lock = &native_.memory_lock;
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
    EXPECT_FALSE(native_.memory_lock.occupied);
    amdf_gpu_kfd_memory_lock_deinitialize(&native_.memory_lock);
  }

  amdf_status_t Create() {
    return amdf_gpu_kfd_buffer_create(&device_, &create_info_, &buffer_,
                                      &result_);
  }

  void UseRegisteredPages(void* pages) {
    create_info_.native_flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                                KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
    create_info_.host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED;
    create_info_.host_pointer = pages;
  }

  amdf_status_t Destroy() {
    const amdf_status_t status = amdf_gpu_kfd_buffer_destroy(buffer_);
    if (amdf_status_is_ok(status)) {
      buffer_ = nullptr;
    }
    return status;
  }

  amdf_status_t PrepareOwnedImport(amdf_gpu_umd_memory_t** memory,
                                   amdf_gpu_umd_memory_result_t* result) {
    device_.native_lifetime = AMDF_NATIVE_LIFETIME_INSTANCE;
    device_.topology.sdma.engine_count = 1;
    native_.import_descriptor = device_.render_descriptor;
    amdf_memory_native_profile_t profile;
    const auto status =
        amdf_gpu_umd_device_query_memory_profile(&device_, 2, &profile);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const amdf_memory_native_import_info_t import = {
        .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .required_flags = AMDF_MEMORY_FLAG_MAPPING_SOURCE,
        .origin = AMDF_MEMORY_NATIVE_IMPORT_ORIGIN_CONSTRUCTION,
        .minimum_alignment = 4096,
    };
    const amdf_external_memory_t exported = {
        .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .payload = {.file_descriptor = native_.import_descriptor},
        .byte_length = 4096,
    };
    return amdf_gpu_umd_memory_prepare_import(&device_, &profile, &import,
                                              &exported, memory, result);
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

TEST_F(KfdBufferNativeTest, ImportsResidencyWithoutAnImplicitDeviceMapping) {
  native_.import_descriptor = device_.render_descriptor;
  ASSERT_EQ(amdf_gpu_kfd_buffer_prepare_import(
                &device_, native_.import_descriptor, 4096, &buffer_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_kfd_buffer_handle(buffer_), 0x1234u);
  EXPECT_EQ(amdf_gpu_kfd_buffer_byte_length(buffer_), 4096u);
  EXPECT_TRUE(native_.map_progress.empty());
  EXPECT_FALSE(native_.access_live);
  EXPECT_EQ(native_.address, 0u);
  ASSERT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_TRUE(native_.unmap_progress.empty());
  EXPECT_EQ(lseek(native_.import_descriptor, 0, SEEK_END), 4096)
      << "preparation and teardown borrow the export descriptor";
}

TEST_F(KfdBufferNativeTest, FailedImportRetainsOnlyItsMetadataOwner) {
  native_.import_descriptor = device_.render_descriptor;
  native_.import_error = ENOMEM;
  EXPECT_EQ(amdf_gpu_kfd_buffer_prepare_import(
                &device_, native_.import_descriptor, 4096, &buffer_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  ASSERT_NE(buffer_, nullptr);
  EXPECT_FALSE(native_.allocation_live);
  ASSERT_EQ(Destroy(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_TRUE(native_.unmap_progress.empty());
}

TEST_F(KfdBufferNativeTest, ImportedSourceReleasesGemBeforeResidency) {
  expected_metadata_free_count_ = 2;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(PrepareOwnedImport(&memory, &result), AMDF_STATUS_OK);
  EXPECT_EQ(result.flags, AMDF_MEMORY_FLAG_MAPPING_SOURCE);
  EXPECT_EQ(result.device_address, 0u);
  EXPECT_EQ(result.byte_length, 4096u);
  EXPECT_TRUE(native_.gem_live);
  EXPECT_TRUE(native_.map_progress.empty());
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(native_.gem_close_count, 1u);
  EXPECT_EQ(native_.free_count, 1u);
}

TEST_F(KfdBufferNativeTest, FailedRenderImportRollsBackKfdOwnership) {
  expected_metadata_free_count_ = 2;
  native_.prime_error = ENOMEM;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original = result;
  EXPECT_EQ(PrepareOwnedImport(&memory, &result),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_FALSE(native_.gem_live);
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(native_.gem_close_count, 0u);
  EXPECT_EQ(native_.free_count, 1u);
}

TEST_F(KfdBufferNativeTest, FailedImportedSourceReleaseRetainsResidency) {
  expected_metadata_free_count_ = 2;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(PrepareOwnedImport(&memory, &result), AMDF_STATUS_OK);
  native_.gem_close_error = EIO;
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_TRUE(native_.allocation_live);
  EXPECT_TRUE(native_.gem_live);
  amdf_gpu_umd_memory_abandon(memory);
  EXPECT_EQ(native_.gem_close_count, 1u);
  EXPECT_EQ(native_.free_count, 0u);
  // These are dependency-model owners, not real native resources. Discard
  // their modeled state after proving that production cleanup retained them.
  native_.allocation_live = false;
  native_.gem_live = false;
}

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

TEST_F(KfdBufferNativeTest, RestartsInvalidatedUserPagesWithoutNewOwnership) {
  alignas(4096) uint8_t pages[4096] = {};
  pages[0] = 0xA7;
  pages[sizeof(pages) - 1] = 0x69;
  UseRegisteredPages(pages);
  native_.allocation_errors = {EAGAIN, EAGAIN, 0};
  EXPECT_EQ(Create(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.allocation_attempts.size(), 3u);
  for (const auto& attempt : native_.allocation_attempts) {
    EXPECT_EQ(attempt.va_addr, native_.address);
    EXPECT_EQ(attempt.mmap_offset, reinterpret_cast<uintptr_t>(pages));
    EXPECT_EQ(attempt.size, sizeof(pages));
    EXPECT_EQ(attempt.gpu_id, device_.topology.gpu_id);
    EXPECT_EQ(attempt.flags, create_info_.native_flags);
    EXPECT_EQ(attempt.handle, 0u);
  }
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  if (buffer_) {
    EXPECT_EQ(result_.host_pointer, pages);
    EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
  }
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_EQ(native_.reservation_release_count, 1u);
  EXPECT_EQ(pages[0], 0xA7);
  EXPECT_EQ(pages[sizeof(pages) - 1], 0x69);
}

TEST_F(KfdBufferNativeTest, UserPageRestartPreservesTerminalFailureAndOutputs) {
  alignas(4096) uint8_t pages[4096] = {};
  UseRegisteredPages(pages);
  native_.allocation_errors = {EAGAIN, ENOMEM};
  std::memset(&result_, 0xA5, sizeof(result_));
  const auto original_result = result_;
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(native_.allocation_attempts.size(), 2u);
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(std::memcmp(&result_, &original_result, sizeof(result_)), 0);
  EXPECT_TRUE(native_.map_progress.empty());
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.reservation_release_count, 1u);
}

TEST_F(KfdBufferNativeTest, NonUserPageAllocationDoesNotRestartEagain) {
  native_.allocation_errors = {EAGAIN, 0};
  expected_metadata_free_count_ = 2;
  for (const auto flags :
       {KFD_IOC_ALLOC_MEM_FLAGS_GTT, KFD_IOC_ALLOC_MEM_FLAGS_VRAM}) {
    SCOPED_TRACE(flags);
    create_info_.native_flags = flags;
    native_.allocation_attempts.clear();
    EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EAGAIN));
    EXPECT_EQ(native_.allocation_attempts.size(), 1u);
    EXPECT_EQ(buffer_, nullptr);
    EXPECT_TRUE(native_.map_progress.empty());
    if (buffer_) {
      EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
    }
    EXPECT_EQ(native_.free_count, 0u);
  }
  EXPECT_EQ(native_.reservation_release_count, 2u);
}

TEST_F(KfdBufferNativeTest, UserPageAllocationDoesNotRestartInterruption) {
  alignas(4096) uint8_t pages[4096] = {};
  UseRegisteredPages(pages);
  native_.allocation_errors = {EINTR, 0};
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EINTR));
  EXPECT_EQ(native_.allocation_attempts.size(), 1u);
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_TRUE(native_.map_progress.empty());
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(native_.reservation_release_count, 1u);
}

TEST_F(KfdBufferNativeTest, UserPageRestartRollsBackLaterMapFailure) {
  alignas(4096) uint8_t pages[4096] = {};
  UseRegisteredPages(pages);
  native_.allocation_errors = {EAGAIN, 0};
  native_.map_error = EIO;
  EXPECT_EQ(Create(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(native_.allocation_attempts.size(), 2u);
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(native_.map_progress, (std::vector<uint32_t>{0}));
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_EQ(native_.reservation_release_count, 1u);
}

TEST_F(KfdBufferNativeTest, UserPageRestartYieldsToEnrolledRelease) {
  alignas(4096) uint8_t pages[4096] = {};
  UseRegisteredPages(pages);
  native_.allocation_errors = {EAGAIN, 0};
  bool release_completed = false;
  std::thread release;
  native_.observe_allocation = [&] {
    if (native_.allocation_attempts.size() == 1) {
      release = std::thread([&] {
        amdf_gpu_kfd_memory_lock_acquire(&native_.memory_lock,
                                         AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE);
        release_completed = true;
        amdf_gpu_kfd_memory_lock_release(&native_.memory_lock);
      });
      // The real native lock, not thread startup, establishes the waiter.
      while (true) {
        EXPECT_EQ(pthread_mutex_lock(&native_.memory_lock.mutex), 0);
        const bool enrolled =
            native_.memory_lock.buckets[AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE]
                .head != nullptr;
        EXPECT_EQ(pthread_mutex_unlock(&native_.memory_lock.mutex), 0);
        if (enrolled) {
          break;
        }
        std::this_thread::yield();
      }
    } else {
      EXPECT_TRUE(release_completed);
    }
  };
  EXPECT_EQ(Create(), AMDF_STATUS_OK);
  release.join();
  EXPECT_TRUE(release_completed);
  EXPECT_EQ(native_.allocation_attempts.size(), 2u);
  if (buffer_) {
    EXPECT_EQ(Destroy(), AMDF_STATUS_OK);
  }
  EXPECT_EQ(native_.free_count, 1u);
  EXPECT_EQ(native_.reservation_release_count, 1u);
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

TEST_F(KfdBufferNativeTest, FailedTransportCloseRetainsAcquiredGemForCleanup) {
  expected_metadata_free_count_ = 2;
  native_.import_descriptor = device_.render_descriptor;
  native_.export_close_error = EIO;
  native_.gem_close_error = ENOMEM;
  amdf_memory_native_profile_t profile;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &profile),
            AMDF_STATUS_OK);
  const amdf_memory_native_create_info_t create_info = {
      .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .required_flags = AMDF_MEMORY_FLAG_MAPPING_SOURCE,
      .byte_length = 4096,
      .minimum_alignment = 4096,
  };
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original_result = result;
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile,
                                        &create_info, &memory, &result),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(native_.export_count, 2u);
  EXPECT_EQ(native_.export_descriptor, -1);
  EXPECT_TRUE(native_.gem_live);
  EXPECT_EQ(native_.gem_close_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(native_.gem_close_count, 1u);
  EXPECT_EQ(native_.free_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_memory_remaining_backing_byte_length(memory), 4096u);
  amdf_gpu_umd_memory_abandon(memory);
  native_.gem_live = false;
  ReleaseLeakedReservation();
}

TEST_F(KfdBufferNativeTest, FailedHostUnmapDistinguishesBackingFromEmptyVa) {
  for (bool host_mapped : {false, true}) {
    create_info_.host_access = host_mapped
                                   ? AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED
                                   : AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE;
    ASSERT_EQ(Create(), AMDF_STATUS_OK);
    native_.munmap_error = EIO;
    EXPECT_EQ(Destroy(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
    EXPECT_FALSE(native_.allocation_live);
    EXPECT_EQ(amdf_gpu_kfd_buffer_remaining_backing_byte_length(buffer_),
              host_mapped ? 4096u : 0u);
    amdf_gpu_kfd_buffer_abandon(buffer_);
    buffer_ = nullptr;
    native_.munmap_error = 0;
    ReleaseLeakedReservation();
  }
  expected_metadata_free_count_ = 2;
}

TEST(KfdBufferTest, RejectsMalformedConstructionWithoutPublishingOutputs) {
  amdf_gpu_umd_device_t device = {.page_size = 4096};
  amdf_gpu_kfd_buffer_create_info_t create_info = {
      .byte_length = 4096,
      .alignment = 4096,
  };

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
