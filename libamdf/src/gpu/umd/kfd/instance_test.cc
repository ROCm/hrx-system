// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/instance.h"

#include <drm/amdgpu_drm.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/vm.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

enum class Operation {
  kNone,
  kOpenKfd,
  kVersion,
  kCreateProcess,
  kOpenRender,
  kDeviceInfo,
  kAcquire,
  kReleaseBootstrap,
  kClose,
};

// Dependency model for descriptor ownership and VM handover. The real instance
// implementation is linked unchanged; vm_test exercises the handover itself.
struct NativeState {
  // Whether host allocations fail before native acquisition.
  bool fail_allocation = false;
  // One operation to fail, then return to normal operation.
  Operation failure = Operation::kNone;
  // Native ABI returned by the descriptor's version query.
  kfd_ioctl_get_version_args version = {1, 19};
  // Memory facts returned by the exact retained render connection.
  drm_amdgpu_info_device device_info = {
      .device_id = 0x150e,
      .virtual_address_offset = UINT64_C(0x10000),
      .virtual_address_max = UINT64_C(1) << 46,
      .virtual_address_alignment = 4096,
  };
  // Render descriptor used for the most recent successful memory query.
  int queried_descriptor = -1;
  // Native operation sequence, including failed operations.
  std::vector<Operation> operations;
  // Descriptors consumed by close, including close errors.
  std::vector<int> closed;
  // Descriptors still owned by the instance under test.
  std::set<int> descriptors;
  // Render files already converted to KFD VMs.
  std::set<int> acquired;
  // Next synthetic descriptor, outside the test process's real descriptor set.
  int next_descriptor = 10000;

  bool Record(Operation operation) {
    operations.push_back(operation);
    if (failure != operation) return true;
    failure = Operation::kNone;
    errno = EIO;
    return false;
  }

  int Open(Operation operation) {
    if (!Record(operation)) return -1;
    descriptors.insert(next_descriptor);
    return next_descriptor++;
  }

  size_t Count(Operation operation) const {
    return std::count(operations.begin(), operations.end(), operation);
  }
};

// Only linked dependency calls from this test thread use the native model.
thread_local NativeState* native_state = nullptr;

class KfdInstanceTest : public ::testing::Test {
 protected:
  void SetUp() override { native_state = &native_; }

  void TearDown() override {
    native_.failure = Operation::kNone;
    if (instance_ != nullptr) {
      EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
    }
    EXPECT_TRUE(native_.descriptors.empty());
    native_state = nullptr;
  }

  amdf_status_t Prepare(
      amdf_native_lifetime_t lifetime = AMDF_NATIVE_LIFETIME_INSTANCE) {
    amdf_allocator_t allocator = {};
    allocator.user_data = &native_;
    allocator.allocate = [](void* user_data, uint64_t byte_length,
                            uint64_t minimum_alignment) -> void* {
      if (static_cast<NativeState*>(user_data)->fail_allocation) return nullptr;
      const amdf_allocator_t system = amdf_allocator_system();
      return system.allocate(system.user_data, byte_length, minimum_alignment);
    };
    allocator.free = [](void*, void* allocation) {
      const amdf_allocator_t system = amdf_allocator_system();
      system.free(system.user_data, allocation);
    };
    return amdf_gpu_umd_instance_prepare(&instance_, lifetime, allocator);
  }

  amdf_status_t PrepareVm(uint32_t gpu_id, int* out_descriptor,
                          amdf_gpu_kfd_topology_t* topology = nullptr) {
    amdf_platform_endpoint_t endpoint = {};
    endpoint.info.pci.device_id = 0x150e;
    amdf_gpu_kfd_topology_t local_topology = {};
    if (topology == nullptr) topology = &local_topology;
    topology->gpu_id = gpu_id;
    return amdf_gpu_kfd_instance_prepare_vm(instance_, &endpoint, topology,
                                            4096, out_descriptor);
  }

  // Native dependencies, alive through instance teardown.
  NativeState native_;
  // Instance-owned native slot, including progress after preparation failure.
  amdf_gpu_umd_instance_t* instance_ = nullptr;
};

TEST_F(KfdInstanceTest, SharesOneContextAndExactPerGpuRenderBindings) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  const int kfd_descriptor = amdf_gpu_kfd_instance_descriptor(instance_);
  int first = -1;
  int second = -1;
  ASSERT_EQ(PrepareVm(11, &first), AMDF_STATUS_OK);
  ASSERT_EQ(PrepareVm(12, &second), AMDF_STATUS_OK);
  ASSERT_NE(first, second);
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  int repeated = -1;
  ASSERT_EQ(PrepareVm(11, &repeated), AMDF_STATUS_OK);
  EXPECT_EQ(repeated, first);
  EXPECT_EQ(amdf_gpu_kfd_instance_descriptor(instance_), kfd_descriptor);
  EXPECT_EQ(native_.Count(Operation::kOpenKfd), 1u);
  EXPECT_EQ(native_.Count(Operation::kCreateProcess), 1u);
  EXPECT_EQ(native_.Count(Operation::kOpenRender), 2u);
  EXPECT_EQ(native_.Count(Operation::kAcquire), 2u);
  EXPECT_EQ(native_.Count(Operation::kDeviceInfo), 3u);
  ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
  instance_ = nullptr;
  EXPECT_EQ(native_.closed, (std::vector<int>{kfd_descriptor, second, first}));
}

TEST_F(KfdInstanceTest, NativeMemoryRefinementPrecedesBootstrapAndPublication) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  amdf_gpu_kfd_topology_t topology = {};
  topology.virtual_address.begin = UINT64_C(0x20000);
  topology.virtual_address.end = UINT64_C(1) << 47;
  topology.virtual_address.alignment = 65536;
  topology.vram.total_byte_length = UINT64_C(8) << 30;
  topology.vram.visible_byte_length = UINT64_C(256) << 20;
  int descriptor = -7;
  ASSERT_EQ(PrepareVm(11, &descriptor, &topology), AMDF_STATUS_OK);
  EXPECT_EQ(native_.queried_descriptor, descriptor);
  EXPECT_EQ(topology.virtual_address.begin,
            native_.device_info.virtual_address_offset);
  EXPECT_EQ(topology.virtual_address.end,
            native_.device_info.virtual_address_max);
  EXPECT_EQ(topology.virtual_address.alignment, 4096u);
  EXPECT_EQ(topology.memory_features, AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY);
  EXPECT_EQ(topology.vram.total_byte_length, UINT64_C(8) << 30);

  // Device recreation refreshes actual facts but never bootstraps an acquired
  // VM again. Neither an older interval nor placement bits survive refinement.
  native_.device_info.virtual_address_max = UINT64_C(1) << 45;
  topology.vram.visible_byte_length = topology.vram.total_byte_length;
  ASSERT_EQ(PrepareVm(11, &descriptor, &topology), AMDF_STATUS_OK);
  EXPECT_EQ(topology.virtual_address.end, UINT64_C(1) << 45);
  EXPECT_EQ(topology.memory_features,
            AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY |
                AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY);
  native_.device_info.ids_flags = AMDGPU_IDS_FLAGS_FUSION;
  ASSERT_EQ(PrepareVm(11, &descriptor, &topology), AMDF_STATUS_OK);
  EXPECT_EQ(topology.memory_features, 0u);
  EXPECT_EQ(native_.Count(Operation::kAcquire), 1u);
  EXPECT_EQ(native_.Count(Operation::kOpenRender), 1u);
}

TEST_F(KfdInstanceTest, FailedMemoryQueryDoesNotAcquireOrPublishFacts) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  native_.failure = Operation::kDeviceInfo;
  amdf_gpu_kfd_topology_t topology = {};
  topology.gpu_id = 11;
  topology.virtual_address.begin = 0x20000;
  const amdf_gpu_kfd_topology_t original = topology;
  int descriptor = -7;
  EXPECT_EQ(PrepareVm(11, &descriptor, &topology), amdf_linux_error(EIO));
  EXPECT_EQ(descriptor, -7);
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  EXPECT_EQ(native_.Count(Operation::kAcquire), 0u);
  EXPECT_EQ(PrepareVm(11, &descriptor, &topology), AMDF_STATUS_OK);
  EXPECT_EQ(native_.Count(Operation::kOpenRender), 1u);
}

TEST_F(KfdInstanceTest, InvalidNativeMemoryFactsNeverReachBootstrap) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  const drm_amdgpu_info_device valid = native_.device_info;
  for (uint32_t variant = 0; variant < 5; ++variant) {
    SCOPED_TRACE(variant);
    native_.device_info = valid;
    switch (variant) {
      case 0:
        native_.device_info.device_id ^= 1;
        break;
      case 1:
        native_.device_info.virtual_address_max =
            native_.device_info.virtual_address_offset;
        break;
      case 2:
        native_.device_info.virtual_address_alignment = 0;
        break;
      case 3:
        native_.device_info.virtual_address_alignment = 6144;
        break;
      case 4:
        native_.device_info.virtual_address_alignment = 65536;
        break;
    }
    amdf_gpu_kfd_topology_t topology = {};
    topology.gpu_id = 11;
    const amdf_gpu_kfd_topology_t original = topology;
    int descriptor = -7;
    EXPECT_EQ(PrepareVm(11, &descriptor, &topology),
              variant == 4 ? amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)
                           : amdf_linux_error(EPROTO));
    EXPECT_EQ(descriptor, -7);
    EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  }
  EXPECT_EQ(native_.Count(Operation::kAcquire), 0u);
  EXPECT_EQ(native_.Count(Operation::kOpenRender), 1u);
}

TEST_F(KfdInstanceTest, ProcessLifetimeUsesPrimaryContextWithoutCreateProcess) {
  native_.version.minor_version = 18;
  ASSERT_EQ(Prepare(AMDF_NATIVE_LIFETIME_PROCESS), AMDF_STATUS_OK);
  EXPECT_EQ(native_.Count(Operation::kCreateProcess), 0u);
}

TEST_F(KfdInstanceTest, ProcessLifetimeRejectsUnsupportedAbiBeforeVmCreation) {
  for (const auto version :
       {kfd_ioctl_get_version_args{1, 17}, kfd_ioctl_get_version_args{2, 0}}) {
    native_.version = version;
    EXPECT_EQ(Prepare(AMDF_NATIVE_LIFETIME_PROCESS),
              amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(native_.Count(Operation::kCreateProcess), 0u);
    EXPECT_EQ(native_.Count(Operation::kOpenRender), 0u);
    EXPECT_EQ(native_.Count(Operation::kAcquire), 0u);
    ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
    instance_ = nullptr;
  }
}

TEST_F(KfdInstanceTest, InstanceLifetimeRejectsOldAbiWithoutChangingPolicy) {
  native_.version.minor_version = 18;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(native_.Count(Operation::kOpenKfd), 1u);
  EXPECT_EQ(native_.Count(Operation::kCreateProcess), 0u);
}

TEST_F(KfdInstanceTest, FailedHostAllocationDoesNotAcquireNativeState) {
  native_.fail_allocation = true;
  EXPECT_EQ(Prepare(),
            amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED));
  EXPECT_EQ(instance_, nullptr);
  EXPECT_TRUE(native_.operations.empty());
  native_.fail_allocation = false;
  EXPECT_EQ(Prepare(), AMDF_STATUS_OK);
}

TEST_F(KfdInstanceTest, FailedBindingAllocationDoesNotPublishBorrow) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  native_.fail_allocation = true;
  int descriptor = -7;
  EXPECT_EQ(PrepareVm(11, &descriptor),
            amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED));
  EXPECT_EQ(descriptor, -7);
  EXPECT_EQ(native_.Count(Operation::kOpenRender), 0u);
  native_.fail_allocation = false;
  EXPECT_EQ(PrepareVm(11, &descriptor), AMDF_STATUS_OK);
}

TEST_F(KfdInstanceTest, FailedOpenRetainsOnlyHostStateForRetry) {
  native_.failure = Operation::kOpenKfd;
  EXPECT_EQ(Prepare(), amdf_linux_error(EIO));
  EXPECT_NE(instance_, nullptr);
  EXPECT_TRUE(native_.descriptors.empty());
  EXPECT_EQ(Prepare(), AMDF_STATUS_OK);
}

TEST_F(KfdInstanceTest, FailedVersionQueryClosesDescriptorBeforeRetry) {
  native_.failure = Operation::kVersion;
  EXPECT_EQ(Prepare(), amdf_linux_error(EIO));
  EXPECT_TRUE(native_.descriptors.empty());
  EXPECT_EQ(Prepare(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.Count(Operation::kOpenKfd), 2u);
}

TEST_F(KfdInstanceTest, FailedContextSelectionRetriesOnSameDescriptor) {
  native_.failure = Operation::kCreateProcess;
  EXPECT_EQ(Prepare(), amdf_linux_error(EIO));
  ASSERT_EQ(native_.descriptors.size(), 1u);
  const int descriptor = *native_.descriptors.begin();
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_kfd_instance_descriptor(instance_), descriptor);
  EXPECT_EQ(native_.Count(Operation::kOpenKfd), 1u);
  EXPECT_EQ(native_.Count(Operation::kCreateProcess), 2u);
}

TEST_F(KfdInstanceTest, FailedRenderOpenDoesNotPublishBorrow) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  native_.failure = Operation::kOpenRender;
  int descriptor = -7;
  EXPECT_EQ(PrepareVm(11, &descriptor), amdf_linux_error(EIO));
  EXPECT_EQ(descriptor, -7);
  EXPECT_EQ(PrepareVm(11, &descriptor), AMDF_STATUS_OK);
  EXPECT_EQ(native_.Count(Operation::kAcquire), 1u);
}

TEST_F(KfdInstanceTest, FailedAcquisitionReleasesProgressBeforeRetry) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  native_.failure = Operation::kAcquire;
  amdf_gpu_kfd_topology_t topology = {};
  topology.gpu_id = 11;
  const amdf_gpu_kfd_topology_t original = topology;
  int descriptor = -7;
  EXPECT_EQ(PrepareVm(11, &descriptor, &topology), amdf_linux_error(EIO));
  EXPECT_EQ(descriptor, -7);
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  native_.operations.clear();
  ASSERT_EQ(PrepareVm(11, &descriptor), AMDF_STATUS_OK);
  EXPECT_EQ(native_.operations,
            (std::vector<Operation>{Operation::kReleaseBootstrap,
                                    Operation::kDeviceInfo, Operation::kAcquire,
                                    Operation::kReleaseBootstrap}));
}

TEST_F(KfdInstanceTest, FailedBootstrapReleaseNeverReacquiresConvertedVm) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  native_.failure = Operation::kReleaseBootstrap;
  amdf_gpu_kfd_topology_t topology = {};
  topology.gpu_id = 11;
  const amdf_gpu_kfd_topology_t original = topology;
  int descriptor = -7;
  EXPECT_EQ(PrepareVm(11, &descriptor, &topology), amdf_linux_error(EIO));
  EXPECT_EQ(descriptor, -7);
  EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  ASSERT_EQ(PrepareVm(11, &descriptor), AMDF_STATUS_OK);
  EXPECT_EQ(native_.Count(Operation::kAcquire), 1u);
  EXPECT_EQ(native_.Count(Operation::kReleaseBootstrap), 2u);
}

TEST_F(KfdInstanceTest,
       FailedTeardownKeepsConnectionsUntilBootstrapIsReleased) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  native_.failure = Operation::kAcquire;
  int descriptor = -7;
  ASSERT_EQ(PrepareVm(11, &descriptor), amdf_linux_error(EIO));
  native_.failure = Operation::kReleaseBootstrap;
  EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), amdf_linux_error(EIO));
  EXPECT_EQ(native_.descriptors.size(), 2u);
  EXPECT_TRUE(native_.closed.empty());
}

TEST_F(KfdInstanceTest, FailedCloseNeverClosesConsumedDescriptorsAgain) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  int descriptor = -7;
  ASSERT_EQ(PrepareVm(11, &descriptor), AMDF_STATUS_OK);
  native_.failure = Operation::kClose;
  EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), amdf_linux_error(EIO));
  EXPECT_EQ(native_.descriptors, (std::set<int>{descriptor}));
  native_.failure = Operation::kClose;
  EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), amdf_linux_error(EIO));
  EXPECT_TRUE(native_.descriptors.empty());
  ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
  instance_ = nullptr;
  EXPECT_EQ(native_.Count(Operation::kClose), 2u);
}

}  // namespace

// Linker wrappers replace only dependencies of the unchanged instance object.
// Calls by the test runtime outside this model continue to the system library.
extern "C" {
int __real_open(const char* path, int flags, ...);
int __real_close(int descriptor);
int __real_ioctl(int descriptor, unsigned long request, ...);

int __wrap_open(const char* path, int flags, ...) {
  if (native_state != nullptr && std::strcmp(path, "/dev/kfd") == 0) {
    EXPECT_EQ(flags, O_RDWR | O_CLOEXEC);
    return native_state->Open(Operation::kOpenKfd);
  }
  mode_t mode = 0;
  if ((flags & O_CREAT) != 0 || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list arguments;
    va_start(arguments, flags);
    mode = va_arg(arguments, mode_t);
    va_end(arguments);
  }
  return __real_open(path, flags, mode);
}

int __wrap_close(int descriptor) {
  if (native_state == nullptr || descriptor < 10000) {
    return __real_close(descriptor);
  }
  EXPECT_EQ(native_state->descriptors.erase(descriptor), 1u);
  native_state->closed.push_back(descriptor);
  return native_state->Record(Operation::kClose) ? 0 : -1;
}

int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_state == nullptr || descriptor < 10000) {
    return __real_ioctl(descriptor, request, argument);
  }
  EXPECT_EQ(native_state->descriptors.count(descriptor), 1u);
  if (request == AMDKFD_IOC_GET_VERSION) {
    if (!native_state->Record(Operation::kVersion)) return -1;
    *static_cast<kfd_ioctl_get_version_args*>(argument) = native_state->version;
    return 0;
  }
  if (request == DRM_IOCTL_AMDGPU_INFO) {
    auto* query = static_cast<drm_amdgpu_info*>(argument);
    EXPECT_EQ(query->query, AMDGPU_INFO_DEV_INFO);
    EXPECT_EQ(query->return_size, sizeof(drm_amdgpu_info_device));
    if (!native_state->Record(Operation::kDeviceInfo)) return -1;
    *reinterpret_cast<drm_amdgpu_info_device*>(query->return_pointer) =
        native_state->device_info;
    native_state->queried_descriptor = descriptor;
    return 0;
  }
  EXPECT_EQ(request, AMDKFD_IOC_CREATE_PROCESS);
  EXPECT_EQ(argument, nullptr);
  return native_state->Record(Operation::kCreateProcess) ? 0 : -1;
}

amdf_status_t __wrap_amdf_linux_endpoint_open_file(
    const amdf_platform_endpoint_t* endpoint, int* out_descriptor,
    amdf_linux_drm_version_t* out_version) {
  EXPECT_NE(endpoint, nullptr);
  EXPECT_EQ(out_version, nullptr);
  const int descriptor = native_state->Open(Operation::kOpenRender);
  if (descriptor < 0) return amdf_linux_error(EIO);
  *out_descriptor = descriptor;
  return AMDF_STATUS_OK;
}

amdf_status_t __wrap_amdf_gpu_kfd_vm_acquire(
    int kfd_descriptor, int render_descriptor,
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    const amdf_gpu_kfd_vm_native_api_t* native_api,
    amdf_gpu_kfd_vm_bootstrap_t* bootstrap) {
  EXPECT_EQ(native_state->descriptors.count(kfd_descriptor), 1u);
  EXPECT_EQ(native_state->descriptors.count(render_descriptor), 1u);
  EXPECT_EQ(native_state->acquired.count(render_descriptor), 0u);
  EXPECT_EQ(native_state->queried_descriptor, render_descriptor);
  EXPECT_EQ(topology->virtual_address.begin,
            native_state->device_info.virtual_address_offset);
  EXPECT_EQ(topology->virtual_address.end,
            native_state->device_info.virtual_address_max);
  EXPECT_EQ(topology->virtual_address.alignment,
            native_state->device_info.virtual_address_alignment);
  EXPECT_EQ(bootstrap->buffer_handle, 0u);
  EXPECT_NE(topology->gpu_id, 0u);
  EXPECT_EQ(page_size, 4096u);
  EXPECT_EQ(native_api, amdf_gpu_kfd_vm_default_native_api());
  bootstrap->render_descriptor = render_descriptor;
  bootstrap->buffer_handle = 9;
  if (!native_state->Record(Operation::kAcquire)) return amdf_linux_error(EIO);
  native_state->acquired.insert(render_descriptor);
  return AMDF_STATUS_OK;
}

amdf_status_t __wrap_amdf_gpu_kfd_vm_bootstrap_release(
    amdf_gpu_kfd_vm_bootstrap_t* bootstrap) {
  if (bootstrap->buffer_handle == 0) return AMDF_STATUS_OK;
  EXPECT_EQ(native_state->descriptors.count(bootstrap->render_descriptor), 1u);
  if (!native_state->Record(Operation::kReleaseBootstrap)) {
    return amdf_linux_error(EIO);
  }
  bootstrap->buffer_handle = 0;
  return AMDF_STATUS_OK;
}
}  // extern "C"
