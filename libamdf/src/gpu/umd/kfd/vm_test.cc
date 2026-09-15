// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/vm.h"

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <linux/kfd_ioctl.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

namespace {

enum class Operation {
  kQuery,
  kCreateContext,
  kCreateBuffer,
  kQueryMapping,
  kMap,
  kMapGpu,
  kSubmit,
  kWait,
  kAcquire,
  kCloseBuffer,
  kUnmap,
  kFreeContext,
};

// Models native ownership and command dependencies, not the VM acquisition
// implementation. Kernel fence inclusion is exercised by hardware CTS.
class NativeVm {
 public:
  NativeVm() {
    ip.hw_ip_version_major = 6;
    ip.available_rings = 1;
    ip.ib_start_alignment = 256;
    ip.ib_size_alignment = 32;
    std::memset(storage.data(), 0xA5, storage.size());
    topology.gpu_id = 123;
    topology.virtual_address.begin = 4096;
    topology.virtual_address.end = UINT64_C(1) << 48;
  }

  amdf_status_t Acquire() {
    return amdf_gpu_kfd_vm_acquire(11, 12, &topology, storage.size(), &api,
                                   &bootstrap);
  }

  amdf_status_t Release() {
    return amdf_gpu_kfd_vm_bootstrap_release(&bootstrap);
  }

  amdf_status_t Record(Operation operation) {
    operations.push_back(operation);
    return operations.size() == failure_call
               ? amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO)
               : AMDF_STATUS_OK;
  }

  static amdf_status_t Ioctl(void* user_data, int descriptor,
                             unsigned long request, void* arguments) {
    auto& self = *static_cast<NativeVm*>(user_data);
    EXPECT_EQ(descriptor, request == AMDKFD_IOC_ACQUIRE_VM ? 11 : 12);
    Operation operation;
    if (request == DRM_IOCTL_AMDGPU_INFO) {
      operation = Operation::kQuery;
    } else if (request == DRM_IOCTL_AMDGPU_CTX) {
      const auto* context = static_cast<drm_amdgpu_ctx*>(arguments);
      operation = context->in.op == AMDGPU_CTX_OP_ALLOC_CTX
                      ? Operation::kCreateContext
                      : Operation::kFreeContext;
    } else if (request == DRM_IOCTL_AMDGPU_GEM_CREATE) {
      operation = Operation::kCreateBuffer;
    } else if (request == DRM_IOCTL_AMDGPU_GEM_MMAP) {
      operation = Operation::kQueryMapping;
    } else if (request == DRM_IOCTL_AMDGPU_GEM_VA) {
      operation = Operation::kMapGpu;
    } else if (request == DRM_IOCTL_AMDGPU_CS) {
      operation = Operation::kSubmit;
    } else if (request == DRM_IOCTL_AMDGPU_WAIT_CS) {
      operation = Operation::kWait;
    } else if (request == AMDKFD_IOC_ACQUIRE_VM) {
      operation = Operation::kAcquire;
    } else if (request == DRM_IOCTL_GEM_CLOSE) {
      operation = Operation::kCloseBuffer;
    } else {
      ADD_FAILURE() << "Unexpected native ioctl " << request;
      return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOTTY);
    }
    const amdf_status_t status = self.Record(operation);
    if (!amdf_status_is_ok(status)) return status;

    switch (operation) {
      case Operation::kQuery: {
        auto* query = static_cast<drm_amdgpu_info*>(arguments);
        EXPECT_EQ(query->query, AMDGPU_INFO_HW_IP_INFO);
        EXPECT_EQ(query->query_hw_ip.type, AMDGPU_HW_IP_DMA);
        EXPECT_EQ(query->query_hw_ip.ip_instance, 0u);
        EXPECT_EQ(query->return_size, sizeof(self.ip));
        *reinterpret_cast<drm_amdgpu_info_hw_ip*>(query->return_pointer) =
            self.ip;
        break;
      }
      case Operation::kCreateContext: {
        auto* context = static_cast<drm_amdgpu_ctx*>(arguments);
        EXPECT_EQ(context->in.priority, AMDGPU_CTX_PRIORITY_NORMAL);
        EXPECT_FALSE(self.context_owned);
        self.context_owned = true;
        context->out.alloc.ctx_id = 7;
        break;
      }
      case Operation::kCreateBuffer: {
        auto* create = static_cast<drm_amdgpu_gem_create*>(arguments);
        EXPECT_EQ(create->in.bo_size, self.storage.size());
        EXPECT_EQ(create->in.alignment, self.storage.size());
        EXPECT_EQ(create->in.domains, AMDGPU_GEM_DOMAIN_GTT);
        EXPECT_EQ(create->in.domain_flags,
                  AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED);
        EXPECT_FALSE(self.buffer_owned);
        self.buffer_owned = true;
        create->out.handle = 9;
        break;
      }
      case Operation::kQueryMapping: {
        auto* mapping = static_cast<drm_amdgpu_gem_mmap*>(arguments);
        EXPECT_TRUE(self.buffer_owned);
        EXPECT_EQ(mapping->in.handle, 9u);
        mapping->out.addr_ptr = 0x100000;
        break;
      }
      case Operation::kMapGpu: {
        const auto* map = static_cast<drm_amdgpu_gem_va*>(arguments);
        EXPECT_TRUE(self.buffer_owned);
        EXPECT_TRUE(self.mapping_owned);
        EXPECT_EQ(map->handle, 9u);
        EXPECT_EQ(map->operation, AMDGPU_VA_OP_MAP);
        EXPECT_EQ(map->flags, AMDGPU_VM_DELAY_UPDATE | AMDGPU_VM_PAGE_READABLE |
                                  AMDGPU_VM_PAGE_EXECUTABLE);
        EXPECT_EQ(map->va_address,
                  reinterpret_cast<uintptr_t>(self.storage.data()));
        EXPECT_EQ(map->offset_in_bo, 0u);
        EXPECT_EQ(map->map_size, self.storage.size());
        self.gpu_mapping_recorded = true;
        break;
      }
      case Operation::kSubmit: {
        auto* submission = static_cast<drm_amdgpu_cs*>(arguments);
        EXPECT_TRUE(self.context_owned);
        EXPECT_TRUE(self.gpu_mapping_recorded);
        EXPECT_EQ(submission->in.ctx_id, 7u);
        EXPECT_EQ(submission->in.num_chunks, 2u);
        EXPECT_EQ(submission->in.bo_list_handle, 0u);
        const auto* pointers =
            reinterpret_cast<const uint64_t*>(submission->in.chunks);
        const auto* ib_chunk =
            reinterpret_cast<const drm_amdgpu_cs_chunk*>(pointers[0]);
        const auto* bo_chunk =
            reinterpret_cast<const drm_amdgpu_cs_chunk*>(pointers[1]);
        EXPECT_EQ(ib_chunk->chunk_id, AMDGPU_CHUNK_ID_IB);
        EXPECT_EQ(ib_chunk->length_dw, sizeof(drm_amdgpu_cs_chunk_ib) / 4);
        const auto* ib = reinterpret_cast<const drm_amdgpu_cs_chunk_ib*>(
            ib_chunk->chunk_data);
        EXPECT_EQ(ib->va_start,
                  reinterpret_cast<uintptr_t>(self.storage.data()));
        EXPECT_EQ(ib->ib_bytes, self.ip.ib_size_alignment);
        EXPECT_EQ(ib->ip_type, AMDGPU_HW_IP_DMA);
        EXPECT_EQ(ib->ip_instance, 0u);
        EXPECT_EQ(ib->ring, self.expected_ring);
        for (size_t i = 0; i < self.ip.ib_size_alignment; ++i) {
          EXPECT_EQ(self.storage[i], 0u);
        }
        EXPECT_EQ(bo_chunk->chunk_id, AMDGPU_CHUNK_ID_BO_HANDLES);
        const auto* buffers = reinterpret_cast<const drm_amdgpu_bo_list_in*>(
            bo_chunk->chunk_data);
        EXPECT_EQ(buffers->bo_number, 1u);
        EXPECT_EQ(buffers->bo_info_size, sizeof(drm_amdgpu_bo_list_entry));
        const auto* entry = reinterpret_cast<const drm_amdgpu_bo_list_entry*>(
            buffers->bo_info_ptr);
        EXPECT_EQ(entry->bo_handle, 9u);
        self.submitted = true;
        submission->out.handle = 42;
        break;
      }
      case Operation::kWait: {
        auto* wait = static_cast<drm_amdgpu_wait_cs*>(arguments);
        EXPECT_TRUE(self.submitted);
        EXPECT_TRUE(self.buffer_owned);
        EXPECT_EQ(wait->in.handle, 42u);
        EXPECT_EQ(wait->in.timeout, UINT64_MAX);
        EXPECT_EQ(wait->in.ip_type, AMDGPU_HW_IP_DMA);
        EXPECT_EQ(wait->in.ip_instance, 0u);
        EXPECT_EQ(wait->in.ctx_id, 7u);
        EXPECT_EQ(wait->in.ring, self.expected_ring);
        wait->out.status = self.wait_busy ? 1 : 0;
        self.completed = !self.wait_busy;
        break;
      }
      case Operation::kAcquire: {
        const auto* acquire =
            static_cast<kfd_ioctl_acquire_vm_args*>(arguments);
        EXPECT_TRUE(self.completed);
        EXPECT_TRUE(self.buffer_owned);
        EXPECT_TRUE(self.mapping_owned);
        EXPECT_EQ(acquire->drm_fd, 12u);
        EXPECT_EQ(acquire->gpu_id, self.topology.gpu_id);
        self.acquired = true;
        break;
      }
      case Operation::kCloseBuffer:
        EXPECT_TRUE(self.buffer_owned);
        EXPECT_EQ(static_cast<drm_gem_close*>(arguments)->handle, 9u);
        self.buffer_owned = false;
        break;
      case Operation::kFreeContext: {
        const auto* context = static_cast<drm_amdgpu_ctx*>(arguments);
        EXPECT_EQ(context->in.op, AMDGPU_CTX_OP_FREE_CTX);
        EXPECT_EQ(context->in.ctx_id, 7u);
        EXPECT_TRUE(self.context_owned);
        EXPECT_FALSE(self.buffer_owned);
        EXPECT_FALSE(self.mapping_owned);
        self.context_owned = false;
        break;
      }
      default:
        ADD_FAILURE() << "Non-ioctl operation";
        break;
    }
    return AMDF_STATUS_OK;
  }

  static amdf_status_t Map(void* user_data, int descriptor,
                           uint64_t byte_offset, size_t byte_length,
                           void** out_mapping) {
    auto& self = *static_cast<NativeVm*>(user_data);
    EXPECT_EQ(descriptor, 12);
    EXPECT_EQ(byte_offset, 0x100000u);
    EXPECT_EQ(byte_length, self.storage.size());
    const amdf_status_t status = self.Record(Operation::kMap);
    if (!amdf_status_is_ok(status)) return status;
    EXPECT_TRUE(self.buffer_owned);
    EXPECT_FALSE(self.mapping_owned);
    self.mapping_owned = true;
    *out_mapping = self.storage.data();
    return AMDF_STATUS_OK;
  }

  static amdf_status_t Unmap(void* user_data, void* mapping,
                             size_t byte_length) {
    auto& self = *static_cast<NativeVm*>(user_data);
    EXPECT_EQ(mapping, self.storage.data());
    EXPECT_EQ(byte_length, self.storage.size());
    const amdf_status_t status = self.Record(Operation::kUnmap);
    if (!amdf_status_is_ok(status)) return status;
    EXPECT_FALSE(self.buffer_owned);
    EXPECT_TRUE(self.mapping_owned);
    self.mapping_owned = false;
    return AMDF_STATUS_OK;
  }

  // Dependency table borrowed by the acquisition record.
  amdf_gpu_kfd_vm_native_api_t api = {this, Ioctl, Map, Unmap};
  // Mutable native query response.
  drm_amdgpu_info_hw_ip ip = {};
  // Enumerated address range and selected KFD node.
  amdf_gpu_kfd_topology_t topology = {};
  // Initially empty production ownership record.
  amdf_gpu_kfd_vm_bootstrap_t bootstrap = {};
  // Aligned stand-in for the native coherent GTT view.
  alignas(4096) std::array<uint8_t, 4096> storage;
  // Observed dependency calls in execution order.
  std::vector<Operation> operations;
  // One-based call to fail, or zero for success.
  size_t failure_call = 0;
  // First available native DMA ring expected by submit and wait.
  uint32_t expected_ring = 0;
  // Whether the native wait reports incomplete execution.
  bool wait_busy = false;
  // Native context ownership.
  bool context_owned = false;
  // Native GEM handle ownership.
  bool buffer_owned = false;
  // Native host mapping ownership.
  bool mapping_owned = false;
  // Whether a delayed GPU mapping has been recorded.
  bool gpu_mapping_recorded = false;
  // Whether the IB has been submitted.
  bool submitted = false;
  // Whether successful completion has been observed.
  bool completed = false;
  // Whether KFD accepted acquisition after completion.
  bool acquired = false;
};

TEST(KfdVmTest, ReleasesAnUnstartedBootstrapWithoutNativeCalls) {
  amdf_gpu_kfd_vm_bootstrap_t bootstrap = {};
  EXPECT_EQ(amdf_gpu_kfd_vm_bootstrap_release(&bootstrap), AMDF_STATUS_OK);
}

TEST(KfdVmTest, CompletesCommandAndAcquiresBeforeReleasingBootstrap) {
  NativeVm native;
  ASSERT_EQ(native.Acquire(), AMDF_STATUS_OK);
  EXPECT_TRUE(native.acquired);
  EXPECT_TRUE(native.buffer_owned);
  EXPECT_TRUE(native.mapping_owned);
  EXPECT_TRUE(native.context_owned);
  ASSERT_EQ(native.Release(), AMDF_STATUS_OK);
  const std::vector<Operation> expected = {
      Operation::kQuery,        Operation::kCreateContext,
      Operation::kCreateBuffer, Operation::kQueryMapping,
      Operation::kMap,          Operation::kMapGpu,
      Operation::kSubmit,       Operation::kWait,
      Operation::kAcquire,      Operation::kCloseBuffer,
      Operation::kUnmap,        Operation::kFreeContext};
  EXPECT_EQ(native.operations, expected);
  EXPECT_EQ(native.Release(), AMDF_STATUS_OK);
  EXPECT_EQ(native.operations, expected);
}

TEST(KfdVmTest, AdmitsSharedNopFormatsAndSelectsAnAvailableRing) {
  for (uint32_t major = 2; major <= 7; ++major) {
    NativeVm native;
    native.ip.hw_ip_version_major = major;
    native.ip.available_rings = (1u << 3) | (1u << 5);
    native.expected_ring = 3;
    native.ip.ib_size_alignment = 4096;
    ASSERT_EQ(native.Acquire(), AMDF_STATUS_OK) << major;
    ASSERT_EQ(native.Release(), AMDF_STATUS_OK);
  }
}

TEST(KfdVmTest, RejectsUnqualifiedNativeFormatsBeforeAllocating) {
  for (uint32_t variant = 0; variant < 8; ++variant) {
    NativeVm native;
    switch (variant) {
      case 0:
        native.ip.hw_ip_version_major = 1;
        break;
      case 1:
        native.ip.hw_ip_version_major = 8;
        break;
      case 2:
        native.ip.available_rings = 0;
        break;
      case 3:
        native.ip.ib_size_alignment = 0;
        break;
      case 4:
        native.ip.ib_size_alignment = 6;
        break;
      case 5:
        native.ip.ib_size_alignment = 8192;
        break;
      case 6:
        native.ip.ib_start_alignment = 0;
        break;
      case 7:
        native.ip.ib_start_alignment = 8192;
        break;
    }
    EXPECT_EQ(amdf_status_code(native.Acquire()), AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(native.Release(), AMDF_STATUS_OK);
    EXPECT_EQ(native.operations, std::vector<Operation>{Operation::kQuery});
  }
}

TEST(KfdVmTest, PropagatesEachAcquisitionFailureAndReleasesPartialOwnership) {
  for (size_t failure_call = 1; failure_call <= 9; ++failure_call) {
    NativeVm native;
    native.failure_call = failure_call;
    EXPECT_EQ(native.Acquire(), amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO))
        << failure_call;
    EXPECT_EQ(native.operations.size(), failure_call);
    EXPECT_FALSE(native.acquired);
    ASSERT_EQ(native.Release(), AMDF_STATUS_OK);
    EXPECT_FALSE(native.buffer_owned);
    EXPECT_FALSE(native.mapping_owned);
    EXPECT_FALSE(native.context_owned);
    const size_t operation_count = native.operations.size();
    EXPECT_EQ(native.Release(), AMDF_STATUS_OK);
    EXPECT_EQ(native.operations.size(), operation_count);
  }
}

TEST(KfdVmTest, RetainsFailedReleaseWithoutRepeatingCompletedReleases) {
  for (size_t failure_call = 10; failure_call <= 12; ++failure_call) {
    NativeVm native;
    native.failure_call = failure_call;
    ASSERT_EQ(native.Acquire(), AMDF_STATUS_OK);
    EXPECT_EQ(native.Release(),
              amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
    EXPECT_EQ(native.operations.size(), failure_call);
    EXPECT_EQ(native.bootstrap.buffer_handle != 0, failure_call == 10);
    EXPECT_EQ(native.bootstrap.mapping.byte_length != 0, failure_call <= 11);
    EXPECT_NE(native.bootstrap.context_identifier, 0u);
    ASSERT_EQ(native.Release(), AMDF_STATUS_OK);
    EXPECT_FALSE(native.buffer_owned);
    EXPECT_FALSE(native.mapping_owned);
    EXPECT_FALSE(native.context_owned);
    EXPECT_EQ(native.operations.size(), 13u);
  }
}

TEST(KfdVmTest, IncompleteWaitCannotAcquireVm) {
  NativeVm native;
  native.wait_busy = true;
  EXPECT_EQ(native.Acquire(),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ETIME));
  EXPECT_EQ(native.operations.size(), 8u);
  EXPECT_FALSE(native.acquired);
  EXPECT_EQ(native.Release(), AMDF_STATUS_OK);
}

TEST(KfdVmTest, RejectsHostViewOutsideGpuAddressRange) {
  for (uint32_t variant = 0; variant < 3; ++variant) {
    NativeVm native;
    const uintptr_t address =
        reinterpret_cast<uintptr_t>(native.storage.data());
    if (variant == 0) native.topology.virtual_address.begin = address + 1;
    if (variant == 1) native.topology.virtual_address.end = address;
    if (variant == 2) {
      native.topology.virtual_address.end = address + native.storage.size() - 1;
    }
    EXPECT_EQ(amdf_status_code(native.Acquire()), AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(native.operations.size(), 5u);
    EXPECT_EQ(native.Release(), AMDF_STATUS_OK);
  }
}

TEST(KfdVmTest, NativeErrorsPreserveFailedMapOutput) {
  const auto* api = amdf_gpu_kfd_vm_default_native_api();
  drm_amdgpu_info query = {};
  EXPECT_EQ(api->ioctl(api->user_data, -1, DRM_IOCTL_AMDGPU_INFO, &query),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  void* mapping = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(api->map(api->user_data, -1, 0, 4096, &mapping),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_EQ(mapping, reinterpret_cast<void*>(uintptr_t{1}));
}

}  // namespace
