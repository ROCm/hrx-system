// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory_reservation.h"

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <linux/sync_file.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdarg>
#include <cstring>
#include <functional>
#include <semaphore>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

// Only the native ioctl dependency is modeled. Reservation and transport
// ownership, allocation, host mappings, synchronization and cleanup are real.
struct NativeReservationState {
  // Shared native connection lock, including the transport buffer's mutations.
  amdf_gpu_kfd_memory_lock_t* memory_lock = nullptr;
  // Optional native-call gate and order observation, under native ownership.
  std::function<void(unsigned long)> before_mutation;
  // Real memfd supplying the transport's mapped host storage.
  int render_descriptor = -1;
  // Request failing on the selected occurrence, or zero for success.
  unsigned long failure_request = 0;
  // Remaining matching calls before returning EIO.
  uint32_t failure_countdown = 1;
  // Native reservation mutations accepted before any injected error.
  std::vector<drm_amdgpu_gem_va> mutations;
  // Most recently submitted native fence identity.
  uint64_t sequence = 0;
  // Owned native publication context count.
  uint32_t context_count = 0;
  // Live transport KFD allocation count.
  uint32_t allocation_count = 0;
  // Live transport GEM identity count.
  uint32_t gem_count = 0;
  // Real host address occupied by the transport's one-page allocation.
  uintptr_t transport_address = 0;
  // Most recently exported real descriptor; the production path must close it.
  int fence_descriptor = -1;
  // Native fence status: positive completion, zero pending, or negative errno.
  int32_t fence_status = 1;
  // Number of wait calls that clobber their in/out union and return EINTR.
  uint32_t wait_interruptions = 0;
  // Number of readiness calls interrupted after native prefix progress.
  uint32_t ready_interruptions = 0;
  // Progress expected on the next idempotent KFD map after interruption.
  uint32_t ready_progress = 0;
  // Number of idempotent compute-readiness calls after DRM submission.
  uint32_t ready_count = 0;
  // Native successful prefix returned by the idempotent KFD map.
  uint32_t ready_result_count = 1;
};

thread_local NativeReservationState* native_state = nullptr;

void ExpectMutationLocked() {
  EXPECT_EQ(pthread_mutex_lock(&native_state->memory_lock->mutex), 0);
  EXPECT_TRUE(native_state->memory_lock->occupied);
  EXPECT_EQ(pthread_mutex_unlock(&native_state->memory_lock->mutex), 0);
}

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
  auto& state = *native_state;
  if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU ||
      request == DRM_IOCTL_AMDGPU_GEM_VA || request == DRM_IOCTL_AMDGPU_CS ||
      request == DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE ||
      request == DRM_IOCTL_AMDGPU_WAIT_CS || request == SYNC_IOC_FILE_INFO ||
      request == AMDKFD_IOC_MAP_MEMORY_TO_GPU ||
      request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU ||
      request == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
    ExpectMutationLocked();
    if (state.before_mutation) {
      state.before_mutation(request);
    }
  }
  if (request == state.failure_request && --state.failure_countdown == 0) {
    state.failure_request = 0;
    errno = EIO;
    return -1;
  }
  switch (request) {
    case DRM_IOCTL_AMDGPU_INFO: {
      const auto* query = static_cast<drm_amdgpu_info*>(argument);
      auto* info =
          reinterpret_cast<drm_amdgpu_info_hw_ip*>(query->return_pointer);
      EXPECT_EQ(query->query, AMDGPU_INFO_HW_IP_INFO);
      EXPECT_EQ(query->query_hw_ip.type, AMDGPU_HW_IP_DMA);
      info->available_rings = 1u << 2;
      info->ib_size_alignment = 32;
      info->ib_start_alignment = 256;
      return 0;
    }
    case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: {
      auto* allocate =
          static_cast<kfd_ioctl_alloc_memory_of_gpu_args*>(argument);
      EXPECT_EQ(allocate->size, 4096u);
      EXPECT_EQ(allocate->gpu_id, 19u);
      EXPECT_EQ(state.allocation_count++, 0u);
      state.transport_address = allocate->va_addr;
      allocate->handle = 23;
      allocate->mmap_offset = 0;
      return 0;
    }
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU: {
      auto* map = static_cast<kfd_ioctl_map_memory_to_gpu_args*>(argument);
      EXPECT_EQ(map->handle, 23u);
      EXPECT_EQ(map->n_devices, 1u);
      EXPECT_EQ(*reinterpret_cast<uint32_t*>(map->device_ids_array_ptr), 19u);
      if (state.sequence == 0) {
        map->n_success = 1;
        return 0;
      }
      ++state.ready_count;
      EXPECT_EQ(map->n_success, state.ready_progress);
      map->n_success = state.ready_result_count;
      if (state.ready_interruptions != 0) {
        --state.ready_interruptions;
        state.ready_progress = map->n_success;
        errno = EINTR;
        return -1;
      }
      state.ready_progress = 0;
      return 0;
    }
    case AMDKFD_IOC_EXPORT_DMABUF: {
      auto* export_info = static_cast<kfd_ioctl_export_dmabuf_args*>(argument);
      export_info->dmabuf_fd = dup(state.render_descriptor);
      EXPECT_GE(static_cast<int>(export_info->dmabuf_fd), 0);
      return 0;
    }
    case DRM_IOCTL_PRIME_FD_TO_HANDLE: {
      auto* import = static_cast<drm_prime_handle*>(argument);
      EXPECT_GE(fcntl(import->fd, F_GETFD), 0);
      EXPECT_EQ(state.gem_count++, 0u);
      import->handle = 29;
      return 0;
    }
    case DRM_IOCTL_AMDGPU_CTX: {
      auto* context = static_cast<drm_amdgpu_ctx*>(argument);
      if (context->in.op == AMDGPU_CTX_OP_ALLOC_CTX) {
        EXPECT_EQ(state.context_count++, 0u);
        context->out.alloc.ctx_id = 31;
      } else {
        EXPECT_EQ(context->in.op, AMDGPU_CTX_OP_FREE_CTX);
        EXPECT_EQ(context->in.ctx_id, 31u);
        EXPECT_EQ(state.context_count--, 1u);
      }
      return 0;
    }
    case DRM_IOCTL_AMDGPU_GEM_VA:
      state.mutations.push_back(*static_cast<drm_amdgpu_gem_va*>(argument));
      return 0;
    case DRM_IOCTL_AMDGPU_CS: {
      auto* submit = static_cast<drm_amdgpu_cs*>(argument);
      EXPECT_EQ(submit->in.ctx_id, 31u);
      EXPECT_EQ(submit->in.num_chunks, 2u);
      const auto* pointers = reinterpret_cast<uint64_t*>(submit->in.chunks);
      const auto* chunk = reinterpret_cast<drm_amdgpu_cs_chunk*>(pointers[0]);
      const auto* instruction =
          reinterpret_cast<drm_amdgpu_cs_chunk_ib*>(chunk->chunk_data);
      EXPECT_EQ(instruction->va_start, state.transport_address);
      EXPECT_EQ(instruction->ip_type, AMDGPU_HW_IP_DMA);
      EXPECT_EQ(instruction->ring, 2u);
      EXPECT_EQ(instruction->ib_bytes, 32u);
      submit->out.handle = ++state.sequence;
      return 0;
    }
    case DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE: {
      auto* export_info = static_cast<drm_amdgpu_fence_to_handle*>(argument);
      EXPECT_EQ(export_info->in.fence.seq_no, state.sequence);
      EXPECT_EQ(export_info->in.fence.ctx_id, 31u);
      EXPECT_EQ(export_info->in.fence.ip_type, AMDGPU_HW_IP_DMA);
      EXPECT_EQ(export_info->in.fence.ring, 2u);
      EXPECT_EQ(export_info->in.what, AMDGPU_FENCE_TO_HANDLE_GET_SYNC_FILE_FD);
      state.fence_descriptor = dup(state.render_descriptor);
      EXPECT_GE(state.fence_descriptor, 0);
      export_info->out.handle = state.fence_descriptor;
      return 0;
    }
    case DRM_IOCTL_AMDGPU_WAIT_CS: {
      auto* wait = static_cast<drm_amdgpu_wait_cs*>(argument);
      EXPECT_EQ(wait->in.handle, state.sequence);
      EXPECT_EQ(wait->in.timeout, UINT64_MAX);
      EXPECT_EQ(wait->in.ctx_id, 31u);
      EXPECT_EQ(wait->in.ip_type, AMDGPU_HW_IP_DMA);
      EXPECT_EQ(wait->in.ring, 2u);
      if (state.wait_interruptions != 0) {
        --state.wait_interruptions;
        std::memset(wait, 0xa5, sizeof(*wait));
        errno = EINTR;
        return -1;
      }
      wait->out.status = 0;
      return 0;
    }
    case SYNC_IOC_FILE_INFO:
      EXPECT_EQ(descriptor, state.fence_descriptor);
      static_cast<sync_file_info*>(argument)->status = state.fence_status;
      return 0;
    case DRM_IOCTL_GEM_CLOSE:
      EXPECT_EQ(static_cast<drm_gem_close*>(argument)->handle, 29u);
      EXPECT_EQ(state.gem_count--, 1u);
      return 0;
    case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
      auto* unmap =
          static_cast<kfd_ioctl_unmap_memory_from_gpu_args*>(argument);
      EXPECT_EQ(unmap->handle, 23u);
      unmap->n_success = unmap->n_devices;
      return 0;
    }
    case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
      EXPECT_EQ(state.allocation_count--, 1u);
      return 0;
    default:
      ADD_FAILURE() << "Unexpected native publication ioctl: " << request;
      errno = ENOTTY;
      return -1;
  }
}

namespace {

bool IsReserved(uintptr_t address) {
  unsigned char residency = 0;
  return mincore(reinterpret_cast<void*>(address), 4096, &residency) == 0;
}

class KfdMemoryReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_state = &native_;
    ASSERT_EQ(amdf_gpu_kfd_memory_lock_initialize(&memory_lock_), 0);
    native_.memory_lock = &memory_lock_;
    native_.render_descriptor = memfd_create("amdf-publication", MFD_CLOEXEC);
    ASSERT_GE(native_.render_descriptor, 0);
    ASSERT_EQ(ftruncate(native_.render_descriptor, 4096), 0);
    device_.host_allocator = amdf_allocator_system();
    device_.descriptor = native_.render_descriptor;
    device_.render_descriptor = native_.render_descriptor;
    device_.memory_lock = &memory_lock_;
    device_.page_size = 4096;
    device_.topology.gpu_id = 19;
    device_.topology.virtual_address.end = UINT64_C(1) << 48;
    device_.topology.virtual_address.fragment_size = 2 * 1024 * 1024;
  }

  void TearDown() override {
    native_.failure_request = 0;
    native_.fence_status = 1;
    native_.ready_result_count = 1;
    if (reservation_ != nullptr) {
      EXPECT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation_),
                AMDF_STATUS_OK);
    }
    // Failure cases deliberately abandon native ownership. There is no real
    // GPU in this dependency test, so retire those retained host intervals
    // only after checking that production teardown preserved them.
    for (const auto& [address, length] : abandoned_intervals_) {
      EXPECT_EQ(munmap(reinterpret_cast<void*>(address), length), 0);
    }
    EXPECT_FALSE(memory_lock_.occupied);
    amdf_gpu_kfd_memory_lock_deinitialize(&memory_lock_);
    EXPECT_EQ(close(native_.render_descriptor), 0);
    native_state = nullptr;
  }

  void Create(uint64_t byte_length = 32768, uint64_t minimum_alignment = 4096) {
    amdf_gpu_umd_device_t* devices[] = {&device_};
    ASSERT_EQ(amdf_gpu_umd_memory_reservation_create(
                  amdf_allocator_system(), 1, devices, byte_length,
                  minimum_alignment, &reservation_),
              AMDF_STATUS_OK);
    uint32_t owner = 0;
    uint64_t granularity = 0;
    amdf_memory_access_t required_access = 0;
    amdf_gpu_umd_memory_reservation_query_access(
        reservation_, 0, &address_, &owner, &granularity, &required_access);
  }

  amdf_status_t Clear() {
    const amdf_gpu_umd_mapping_update_t update = {
        .target_byte_offset = 4096,
        .byte_length = 4096,
    };
    bool mutation_started = false;
    const auto status = amdf_gpu_umd_memory_reservation_update(
        reservation_, 0, 1, &update, &mutation_started);
    EXPECT_TRUE(mutation_started);
    EXPECT_FALSE(memory_lock_.occupied);
    if (native_.fence_descriptor >= 0) {
      EXPECT_EQ(fcntl(native_.fence_descriptor, F_GETFD), -1);
      EXPECT_EQ(errno, EBADF);
    }
    return status;
  }

  // Shared connection admission, borrowed by every modeled native caller.
  amdf_gpu_kfd_memory_lock_t memory_lock_;
  // Native dependency state for this test only.
  NativeReservationState native_;
  // Concrete native device passed through the production provider.
  amdf_gpu_umd_device_t device_ = {};
  // Reservation consumed by explicit test teardown or fixture teardown.
  amdf_gpu_umd_memory_reservation_t* reservation_ = nullptr;
  // Published reservation base, valid through checked final removal.
  uint64_t address_ = 0;
  // Host intervals retained by consuming native failure, not production owners.
  std::vector<std::pair<uintptr_t, size_t>> abandoned_intervals_;
};

struct ReservationGeometry {
  // Exact usable virtual extent, without physical padding.
  uint64_t byte_length;
  // Caller-requested base alignment in bytes.
  uint64_t minimum_alignment;
  // Native page-table fragment size reported by the device.
  uint32_t fragment_size;
  // Base alignment selected without changing page-level replacement.
  uint64_t expected_alignment;
};

class KfdReservationGeometryTest
    : public KfdMemoryReservationTest,
      public ::testing::WithParamInterface<ReservationGeometry> {};

TEST_P(KfdReservationGeometryTest, PreservesShortExtentAndPageReplacements) {
  const auto geometry = GetParam();
  device_.topology.virtual_address.fragment_size = geometry.fragment_size;
  ASSERT_NO_FATAL_FAILURE(
      Create(geometry.byte_length, geometry.minimum_alignment));
  EXPECT_EQ(address_ % geometry.expected_alignment, 0u);
  uint64_t address = 0;
  uint32_t owner = 0;
  uint64_t granularity = 0;
  amdf_memory_access_t required_access = 0;
  amdf_gpu_umd_memory_reservation_query_access(
      reservation_, 0, &address, &owner, &granularity, &required_access);
  EXPECT_EQ(granularity, device_.page_size);

  const amdf_gpu_umd_mapping_update_t update = {
      .target_byte_offset = geometry.byte_length - device_.page_size,
      .byte_length = device_.page_size,
  };
  bool mutation_started = false;
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_update(reservation_, 0, 1, &update,
                                                   &mutation_started),
            AMDF_STATUS_OK);
  ASSERT_TRUE(mutation_started);
  ASSERT_EQ(native_.mutations.size(), 1u);
  EXPECT_EQ(native_.mutations[0].va_address,
            address_ + geometry.byte_length - device_.page_size);
  EXPECT_EQ(native_.mutations[0].map_size, device_.page_size);
  const auto status = amdf_gpu_umd_memory_reservation_destroy(reservation_);
  reservation_ = nullptr;
  ASSERT_EQ(status, AMDF_STATUS_OK);
  ASSERT_EQ(native_.mutations.size(), 2u);
  EXPECT_EQ(native_.mutations[1].va_address, address_);
  EXPECT_EQ(native_.mutations[1].map_size, geometry.byte_length);
  EXPECT_EQ(native_.allocation_count, 0u);
  EXPECT_EQ(native_.context_count, 0u);
  EXPECT_EQ(native_.gem_count, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    NativePageGeometry, KfdReservationGeometryTest,
    ::testing::Values(ReservationGeometry{3 * 4096, 4096, 65536, 4096},
                      ReservationGeometry{65536 + 4096, 4096, 65536, 65536},
                      ReservationGeometry{2 * 1024 * 1024 + 4096, 4096,
                                          2 * 1024 * 1024, 2 * 1024 * 1024},
                      ReservationGeometry{2 * 1024 * 1024 + 4096,
                                          4 * 1024 * 1024, 2 * 1024 * 1024,
                                          4 * 1024 * 1024}));

TEST_F(KfdMemoryReservationTest, ResumesBothNativeWaitProtocols) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.wait_interruptions = 1;
  native_.ready_interruptions = 1;
  ASSERT_EQ(Clear(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.wait_interruptions, 0u);
  EXPECT_EQ(native_.ready_interruptions, 0u);
  EXPECT_EQ(native_.ready_count, 2u);
  ASSERT_EQ(native_.mutations.size(), 1u);
  EXPECT_EQ(native_.mutations[0].va_address, address_ + 4096);
  EXPECT_EQ(native_.mutations[0].map_size, 4096u);
}

TEST_F(KfdMemoryReservationTest,
       ExistingReservationPublicationOvertakesOtherCallersBackingWork) {
  ASSERT_NO_FATAL_FAILURE(Create());
  NativeReservationState held_native, acquire_native, release_native;
  for (auto* state : {&held_native, &acquire_native, &release_native}) {
    state->memory_lock = &memory_lock_;
    state->render_descriptor = native_.render_descriptor;
  }
  // Separate calling threads model NUMA workers borrowing one connection.
  // Real buffer and reservation implementations select their own priorities.
  const amdf_gpu_kfd_buffer_create_info_t info = {
      .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT,
      .byte_length = 4096,
      .alignment = 4096,
  };
  auto allocate = [&](NativeReservationState* state,
                      amdf_gpu_kfd_buffer_t** out_buffer) {
    native_state = state;
    amdf_gpu_kfd_buffer_result_t result;
    EXPECT_EQ(amdf_gpu_kfd_buffer_create(&device_, &info, out_buffer, &result),
              AMDF_STATUS_OK);
    native_state = nullptr;
  };
  auto release = [&](NativeReservationState* state,
                     amdf_gpu_kfd_buffer_t* buffer) {
    native_state = state;
    EXPECT_EQ(amdf_gpu_kfd_buffer_destroy(buffer), AMDF_STATUS_OK);
    native_state = nullptr;
  };
  amdf_gpu_kfd_buffer_t* retired = nullptr;
  allocate(&release_native, &retired);

  std::vector<char> order;
  std::binary_semaphore entered{0}, resume{0};
  held_native.before_mutation = [&](unsigned long request) {
    if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
      order.push_back('H');
      entered.release();
      resume.acquire();
    }
  };
  acquire_native.before_mutation = [&](unsigned long request) {
    if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
      order.push_back('A');
    }
  };
  release_native.before_mutation = [&](unsigned long request) {
    if (request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU) {
      order.push_back('R');
    }
  };
  native_.before_mutation = [&](unsigned long request) {
    if (request == DRM_IOCTL_AMDGPU_GEM_VA) {
      order.push_back('M');
    } else if (request == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
      order.push_back('P');
    }
  };
  auto await_enrollment = [&](amdf_gpu_kfd_memory_priority_t priority) {
    while (true) {
      EXPECT_EQ(pthread_mutex_lock(&memory_lock_.mutex), 0);
      const bool enrolled = memory_lock_.buckets[priority].head != nullptr;
      EXPECT_EQ(pthread_mutex_unlock(&memory_lock_.mutex), 0);
      if (enrolled) {
        break;
      }
      std::this_thread::yield();
    }
  };
  amdf_gpu_kfd_buffer_t* held = nullptr;
  amdf_gpu_kfd_buffer_t* acquired = nullptr;
  std::thread holding([&] { allocate(&held_native, &held); });
  entered.acquire();
  std::thread acquiring([&] { allocate(&acquire_native, &acquired); });
  await_enrollment(AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
  std::thread releasing([&] { release(&release_native, retired); });
  await_enrollment(AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE);
  std::thread mapping([&] {
    native_state = &native_;
    const amdf_gpu_umd_mapping_update_t updates[] = {
        {.target_byte_offset = 0, .byte_length = 4096},
        {.target_byte_offset = 4096, .byte_length = 4096},
    };
    bool mutation_started = false;
    EXPECT_EQ(amdf_gpu_umd_memory_reservation_update(
                  reservation_, 0, 2, updates, &mutation_started),
              AMDF_STATUS_OK);
    EXPECT_TRUE(mutation_started);
    native_state = nullptr;
  });
  await_enrollment(AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
  resume.release();
  holding.join();
  acquiring.join();
  releasing.join();
  mapping.join();
  EXPECT_EQ(order, (std::vector<char>{'H', 'M', 'M', 'P', 'R', 'A'}));
  EXPECT_EQ(native_.sequence, 1u);
  EXPECT_EQ(native_.ready_count, 1u);
  EXPECT_EQ(release_native.allocation_count, 0u);

  held_native.before_mutation = {};
  acquire_native.before_mutation = {};
  native_.before_mutation = {};
  release(&held_native, held);
  release(&acquire_native, acquired);
  native_state = &native_;
}

TEST_F(KfdMemoryReservationTest, RejectsFailedPublicationFence) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.fence_status = -ECANCELED;
  EXPECT_EQ(Clear(), amdf_linux_error(ECANCELED));
  EXPECT_EQ(native_.ready_count, 0u);
}

TEST_F(KfdMemoryReservationTest, RejectsIncompletePublicationFence) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.fence_status = 0;
  EXPECT_EQ(Clear(), amdf_linux_error(EPROTO));
  EXPECT_EQ(native_.ready_count, 0u);
}

TEST_F(KfdMemoryReservationTest, RejectsIncompleteComputeReadiness) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.ready_result_count = 0;
  EXPECT_EQ(Clear(), amdf_linux_error(EPROTO));
  EXPECT_EQ(native_.ready_count, 1u);
}

class KfdMemoryReservationFailureTest
    : public KfdMemoryReservationTest,
      public ::testing::WithParamInterface<unsigned long> {};

TEST_P(KfdMemoryReservationFailureTest, PropagatesNativePublicationError) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.failure_request = GetParam();
  EXPECT_EQ(Clear(), amdf_linux_error(EIO));
  EXPECT_EQ(native_.ready_count, 0u);
  EXPECT_TRUE(IsReserved(address_));
  EXPECT_EQ(native_.allocation_count, 1u);
}

TEST_P(KfdMemoryReservationFailureTest,
       FailedDetachPreservesAddressAndPublicationOwnership) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.failure_request = GetParam();
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_detach(reservation_, 0),
            amdf_linux_error(EIO));
  EXPECT_EQ(native_.ready_count, 0u);
  EXPECT_TRUE(IsReserved(address_));
  EXPECT_TRUE(IsReserved(native_.transport_address));
  EXPECT_EQ(native_.allocation_count, 1u);
  EXPECT_EQ(native_.gem_count, 1u);
  EXPECT_EQ(native_.context_count, 1u);
}

INSTANTIATE_TEST_SUITE_P(NativeCall, KfdMemoryReservationFailureTest,
                         ::testing::Values(DRM_IOCTL_AMDGPU_GEM_VA,
                                           DRM_IOCTL_AMDGPU_CS,
                                           DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE,
                                           DRM_IOCTL_AMDGPU_WAIT_CS,
                                           SYNC_IOC_FILE_INFO,
                                           AMDKFD_IOC_MAP_MEMORY_TO_GPU));

TEST_F(KfdMemoryReservationTest,
       DetachesEntireGroupAfterPartialVectorFailureWithoutReleasingOwner) {
  ASSERT_NO_FATAL_FAILURE(Create());
  const amdf_gpu_umd_mapping_update_t updates[] = {
      {.target_byte_offset = 4096, .byte_length = 4096},
      {.target_byte_offset = 16384, .byte_length = 4096},
  };
  native_.failure_request = DRM_IOCTL_AMDGPU_GEM_VA;
  native_.failure_countdown = 2;
  bool mutation_started = false;
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_update(reservation_, 0, 2, updates,
                                                   &mutation_started),
            amdf_linux_error(EIO));
  EXPECT_TRUE(mutation_started);
  EXPECT_EQ(native_.mutations.size(), 1u);
  EXPECT_EQ(native_.sequence, 0u);
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_detach(reservation_, 0),
            AMDF_STATUS_OK);
  ASSERT_EQ(native_.mutations.size(), 2u);
  EXPECT_EQ(native_.mutations.back().operation, AMDGPU_VA_OP_CLEAR);
  EXPECT_EQ(native_.mutations.back().va_address, address_);
  EXPECT_EQ(native_.mutations.back().map_size, 32768u);
  EXPECT_EQ(native_.ready_count, 1u);
  EXPECT_EQ(native_.allocation_count, 1u);
  EXPECT_EQ(native_.gem_count, 1u);
  EXPECT_EQ(native_.context_count, 1u);
  EXPECT_TRUE(IsReserved(address_));
  EXPECT_TRUE(IsReserved(native_.transport_address));
  EXPECT_EQ(fcntl(native_.fence_descriptor, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation_),
            AMDF_STATUS_OK);
  reservation_ = nullptr;
  EXPECT_EQ(native_.ready_count, 2u);
  EXPECT_EQ(native_.allocation_count, 0u);
  EXPECT_EQ(native_.gem_count, 0u);
  EXPECT_EQ(native_.context_count, 0u);
  EXPECT_FALSE(IsReserved(address_));
  EXPECT_FALSE(IsReserved(native_.transport_address));
}

TEST_F(KfdMemoryReservationTest, DetachRejectsFailedPublicationFence) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.fence_status = -ECANCELED;
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_detach(reservation_, 0),
            amdf_linux_error(ECANCELED));
  EXPECT_EQ(native_.ready_count, 0u);
  EXPECT_EQ(native_.context_count, 1u);
  EXPECT_TRUE(IsReserved(address_));
}

TEST_F(KfdMemoryReservationTest, FailedFinalClearPreservesAllNativeOwnership) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.fence_status = -EIO;
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation_),
            amdf_linux_error(EIO));
  reservation_ = nullptr;
  EXPECT_EQ(native_.allocation_count, 1u);
  EXPECT_EQ(native_.gem_count, 1u);
  EXPECT_EQ(native_.context_count, 1u);
  EXPECT_TRUE(IsReserved(address_));
  EXPECT_TRUE(IsReserved(native_.transport_address));
  abandoned_intervals_ = {{address_, 32768}, {native_.transport_address, 4096}};
}

TEST_F(KfdMemoryReservationTest, PreparationFailurePreservesOutput) {
  native_.failure_request = DRM_IOCTL_AMDGPU_CTX;
  amdf_gpu_umd_device_t* devices[] = {&device_};
  auto* sentinel = reinterpret_cast<amdf_gpu_umd_memory_reservation_t*>(1);
  auto* output = sentinel;
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_create(
                amdf_allocator_system(), 1, devices, 32768, 4096, &output),
            amdf_linux_error(EIO));
  EXPECT_EQ(output, sentinel);
  EXPECT_EQ(native_.allocation_count, 0u);
  EXPECT_EQ(native_.gem_count, 0u);
  EXPECT_TRUE(native_.mutations.empty());
  EXPECT_FALSE(IsReserved(native_.transport_address));
}

}  // namespace
