// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/instance.h"

#include <drm/amdgpu_drm.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <map>
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
  kAllocateSignalPage,
  kMapSignalPage,
  kUnmapSignalPage,
  kFreeSignalPage,
  kCreateMemoryEvent,
  kCreateHardwareEvent,
  kCreateSignalEvent,
  kDestroyEvent,
  kClose,
};

// Dependency model for descriptor ownership and VM handover. The real instance
// implementation is linked unchanged; vm_test exercises the handover itself.
struct NativeState {
  // Whether host allocations fail before native acquisition.
  bool fail_allocation = false;
  // One operation to fail, then return to normal operation.
  Operation failure = Operation::kNone;
  // Error returned at the selected failure boundary.
  int failure_errno = EIO;
  // Native ABI returned by the descriptor's version query.
  kfd_ioctl_get_version_args version = {1, 19};
  // Memory facts returned by the exact retained render connection.
  drm_amdgpu_info_device device_info = {
      .device_id = 0x150e,
      .virtual_address_offset = UINT64_C(0x10000),
      .virtual_address_max = UINT64_C(1) << 46,
      .virtual_address_alignment = 4096,
      .pte_fragment_size = 2 * 1024 * 1024,
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
  // Native event-page and event-ID ownership within this connection.
  struct {
    // GPU backing handle allocated once for the context.
    uint64_t handle = 0;
    // Backing installed by CREATE_EVENT, possibly owned by a foreign client.
    uint64_t installed_handle = 0;
    // Whether the failing signal create installed the page before failing.
    bool install_before_failure = false;
    // Whether a failed map/unmap changed its PTEs before final-sync failure.
    bool mapping_before_failure = false;
    // GPUs currently mapping the page, independent of final-sync success.
    std::set<uint32_t> mapped;
    // Prefix progress passed into each native mapping attempt.
    std::vector<uint32_t> map_progress;
    // Prefix progress passed into each native unmapping attempt.
    std::vector<uint32_t> unmap_progress;
    // Page handles supplied on signal-event creation attempts.
    std::vector<uint64_t> create_offsets;
    // Live event IDs and their native types.
    std::map<uint32_t, uint32_t> ids;
    // Next signal slot/ID supplied by the native model.
    uint32_t next_id = 1;
  } events;

  bool Record(Operation operation) {
    operations.push_back(operation);
    if (failure != operation) {
      return true;
    }
    failure = Operation::kNone;
    errno = failure_errno;
    return false;
  }

  int Open(Operation operation) {
    if (!Record(operation)) {
      return -1;
    }
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
      ReleaseSignals();
      EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
    }
    EXPECT_TRUE(native_.descriptors.empty());
    EXPECT_TRUE(native_.events.ids.empty());
    EXPECT_TRUE(native_.events.mapped.empty());
    native_state = nullptr;
  }

  amdf_status_t Prepare(
      amdf_native_lifetime_t lifetime = AMDF_NATIVE_LIFETIME_INSTANCE) {
    amdf_allocator_t allocator = {
        .user_data = &native_,
        .allocate = [](void* user_data, uint64_t byte_length,
                       uint64_t minimum_alignment) -> void* {
          if (static_cast<NativeState*>(user_data)->fail_allocation) {
            return nullptr;
          }
          const amdf_allocator_t system = amdf_allocator_system();
          return system.allocate(system.user_data, byte_length,
                                 minimum_alignment);
        },
        .free =
            [](void*, void* allocation) {
              const amdf_allocator_t system = amdf_allocator_system();
              system.free(system.user_data, allocation);
            },
    };
    return amdf_gpu_umd_instance_prepare(&instance_, lifetime, allocator);
  }

  amdf_status_t PrepareVm(uint32_t gpu_id, int* out_descriptor,
                          amdf_gpu_kfd_topology_t* topology = nullptr) {
    amdf_platform_endpoint_t endpoint = {};
    endpoint.info.pci.device_id = 0x150e;
    amdf_gpu_kfd_topology_t local_topology = {};
    if (topology == nullptr) {
      topology = &local_topology;
    }
    topology->gpu_id = gpu_id;
    auto* lock = amdf_gpu_kfd_instance_memory_lock(instance_);
    amdf_gpu_kfd_memory_lock_acquire(lock,
                                     AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
    const auto status = amdf_gpu_kfd_instance_prepare_vm(
        instance_, &endpoint, topology, 4096, out_descriptor);
    amdf_gpu_kfd_memory_lock_release(lock);
    return status;
  }

  void PrepareEventVm(uint32_t gpu_id, amdf_gpu_kfd_topology_t* topology) {
    // The real host VA reservation must fit in the modeled GPU aperture.
    native_.device_info.virtual_address_max = UINT64_C(1) << 48;
    ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
    int descriptor = -1;
    ASSERT_EQ(PrepareVm(gpu_id, &descriptor, topology), AMDF_STATUS_OK);
  }

  amdf_status_t CreateEvent(const amdf_gpu_kfd_topology_t& topology,
                            amdf_gpu_kfd_event_info_t* out_info) {
    auto* lock = amdf_gpu_kfd_instance_memory_lock(instance_);
    amdf_gpu_kfd_memory_lock_acquire(lock,
                                     AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
    const auto status = amdf_gpu_kfd_instance_create_event(instance_, &topology,
                                                           4096, out_info);
    amdf_gpu_kfd_memory_lock_release(lock);
    if (amdf_status_is_ok(status)) {
      signals_.push_back(out_info->signal_id);
    }
    return status;
  }

  void ReleaseSignals() {
    for (uint32_t id : signals_) {
      kfd_ioctl_destroy_event_args destroy = {.event_id = id};
      EXPECT_EQ(ioctl(amdf_gpu_kfd_instance_descriptor(instance_),
                      AMDKFD_IOC_DESTROY_EVENT, &destroy),
                0);
    }
    signals_.clear();
  }

  // Native dependencies, alive through instance teardown.
  NativeState native_;
  // Instance-owned native slot, including progress after preparation failure.
  amdf_gpu_umd_instance_t* instance_ = nullptr;
  // Signal IDs transferred out of the instance and owned by this caller.
  std::vector<uint32_t> signals_;
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

TEST_F(KfdInstanceTest, NativeMemoryQueryPrecedesBootstrapAndPublication) {
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
  EXPECT_EQ(topology.virtual_address.fragment_size, 2 * 1024 * 1024u);
  EXPECT_EQ(topology.memory_features, AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY);
  EXPECT_EQ(topology.vram.total_byte_length, UINT64_C(8) << 30);

  // Device recreation refreshes actual facts but never bootstraps an acquired
  // VM again. Neither an older interval nor placement bits survive a new native
  // query.
  native_.device_info.virtual_address_max = UINT64_C(1) << 45;
  native_.device_info.pte_fragment_size = 65536;
  topology.vram.visible_byte_length = topology.vram.total_byte_length;
  ASSERT_EQ(PrepareVm(11, &descriptor, &topology), AMDF_STATUS_OK);
  EXPECT_EQ(topology.virtual_address.end, UINT64_C(1) << 45);
  EXPECT_EQ(topology.virtual_address.fragment_size, 65536u);
  EXPECT_EQ(topology.memory_features,
            AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY |
                AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY);
  native_.device_info.ids_flags =
      AMDGPU_IDS_FLAGS_FUSION |
      (AMDGPU_IDS_FLAGS_MODE_PT << AMDGPU_IDS_FLAGS_MODE_SHIFT);
  ASSERT_EQ(PrepareVm(11, &descriptor, &topology), AMDF_STATUS_OK);
  EXPECT_EQ(topology.memory_features, 0u);
  EXPECT_EQ(native_.Count(Operation::kAcquire), 1u);
  EXPECT_EQ(native_.Count(Operation::kOpenRender), 1u);
}

TEST_F(KfdInstanceTest, FailedMemoryQueryDoesNotAcquireOrPublishFacts) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  native_.failure = Operation::kDeviceInfo;
  amdf_gpu_kfd_topology_t topology = {.gpu_id = 11};
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
  for (uint32_t variant = 0; variant < 7; ++variant) {
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
      case 5:
        native_.device_info.pte_fragment_size = 0;
        break;
      case 6:
        native_.device_info.pte_fragment_size = 6144;
        break;
    }
    amdf_gpu_kfd_topology_t topology = {.gpu_id = 11};
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
  amdf_gpu_kfd_topology_t topology = {.gpu_id = 11};
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
  amdf_gpu_kfd_topology_t topology = {.gpu_id = 11};
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

TEST_F(KfdInstanceTest, EventBackingAndFaultSourcesAreSharedAcrossGpuBindings) {
  amdf_gpu_kfd_topology_t first_topology = {};
  amdf_gpu_kfd_topology_t second_topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &first_topology));
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(12, &second_topology));
  amdf_gpu_kfd_event_info_t first = {};
  amdf_gpu_kfd_event_info_t second = {};
  amdf_gpu_kfd_event_info_t repeated = {};
  ASSERT_EQ(CreateEvent(first_topology, &first), AMDF_STATUS_OK);
  ASSERT_EQ(CreateEvent(second_topology, &second), AMDF_STATUS_OK);
  ASSERT_EQ(CreateEvent(first_topology, &repeated), AMDF_STATUS_OK);
  EXPECT_NE(first.signal_id, second.signal_id);
  EXPECT_EQ(first.memory_exception_id, second.memory_exception_id);
  EXPECT_EQ(first.hardware_exception_id, second.hardware_exception_id);
  EXPECT_EQ(second.mailbox_address - first.mailbox_address, sizeof(uint64_t));
  EXPECT_EQ(native_.Count(Operation::kAllocateSignalPage), 1u);
  EXPECT_EQ(native_.Count(Operation::kMapSignalPage), 2u);
  EXPECT_EQ(native_.Count(Operation::kCreateMemoryEvent), 1u);
  EXPECT_EQ(native_.Count(Operation::kCreateHardwareEvent), 1u);
  EXPECT_EQ(native_.events.create_offsets,
            (std::vector<uint64_t>{native_.events.handle, 0, 0}));
  ReleaseSignals();
  ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
  instance_ = nullptr;
  EXPECT_EQ(native_.Count(Operation::kUnmapSignalPage), 2u);
  // KFD owns installed backing; FREE_MEMORY would fail with EPERM.
  EXPECT_EQ(native_.Count(Operation::kFreeSignalPage), 0u);
}

TEST_F(KfdInstanceTest, EventPreparationFailuresPublishNoSignal) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  amdf_gpu_kfd_event_info_t info = {.mailbox_address = 0x1234};
  const auto original = info;
  for (auto operation :
       {Operation::kAllocateSignalPage, Operation::kMapSignalPage,
        Operation::kCreateMemoryEvent, Operation::kCreateHardwareEvent}) {
    native_.failure = operation;
    EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(EIO));
    EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
  }
  ASSERT_EQ(CreateEvent(topology, &info), AMDF_STATUS_OK);
  EXPECT_EQ(native_.Count(Operation::kAllocateSignalPage), 2u);
  EXPECT_EQ(native_.Count(Operation::kMapSignalPage), 2u);
  EXPECT_EQ(native_.Count(Operation::kCreateMemoryEvent), 2u);
  EXPECT_EQ(native_.Count(Operation::kCreateHardwareEvent), 2u);
}

TEST_F(KfdInstanceTest, MappingRetryPreservesPteProgressThroughFinalSync) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  native_.failure = Operation::kMapSignalPage;
  native_.events.mapping_before_failure = true;
  amdf_gpu_kfd_event_info_t info = {};
  EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(EIO));
  EXPECT_EQ(native_.events.mapped, (std::set<uint32_t>{11}));
  ASSERT_EQ(CreateEvent(topology, &info), AMDF_STATUS_OK);
  EXPECT_EQ(native_.events.map_progress, (std::vector<uint32_t>{0, 1}));
}

TEST_F(KfdInstanceTest, EventOomBeforeInstallationRetriesTheSameBacking) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  native_.failure = Operation::kCreateSignalEvent;
  native_.failure_errno = ENOMEM;
  amdf_gpu_kfd_event_info_t info = {.signal_id = 0x1234};
  EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(ENOMEM));
  EXPECT_EQ(info.signal_id, 0x1234u);
  EXPECT_EQ(native_.events.installed_handle, 0u);
  ASSERT_EQ(CreateEvent(topology, &info), AMDF_STATUS_OK);
  EXPECT_EQ(
      native_.events.create_offsets,
      (std::vector<uint64_t>{native_.events.handle, native_.events.handle}));
  EXPECT_EQ(native_.Count(Operation::kAllocateSignalPage), 1u);
}

TEST_F(KfdInstanceTest, EventOomAfterInstallationRecoversWithoutNewBacking) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  native_.failure = Operation::kCreateSignalEvent;
  native_.failure_errno = ENOMEM;
  native_.events.install_before_failure = true;
  amdf_gpu_kfd_event_info_t info = {.signal_id = 0x1234};
  EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(ENOMEM));
  EXPECT_EQ(info.signal_id, 0x1234u);
  EXPECT_EQ(native_.events.installed_handle, native_.events.handle);
  ASSERT_EQ(CreateEvent(topology, &info), AMDF_STATUS_OK);
  EXPECT_EQ(
      native_.events.create_offsets,
      (std::vector<uint64_t>{native_.events.handle, native_.events.handle, 0}));
  EXPECT_EQ(native_.Count(Operation::kAllocateSignalPage), 1u);
  native_.failure = Operation::kCreateSignalEvent;
  EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(ENOMEM));
  ASSERT_EQ(CreateEvent(topology, &info), AMDF_STATUS_OK);
  EXPECT_EQ(native_.events.create_offsets,
            (std::vector<uint64_t>{native_.events.handle, native_.events.handle,
                                   0, 0, 0}));
}

TEST_F(KfdInstanceTest, ForeignSignalPageIsNeverAdopted) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  // Another primary-context user installed different backing. Its existence
  // never proves that our GPU mailbox addresses refer to its native page.
  native_.events.installed_handle = 0xdead;
  amdf_gpu_kfd_event_info_t info = {.signal_id = 0x1234};
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(EINVAL));
    EXPECT_EQ(info.signal_id, 0x1234u);
  }
  EXPECT_EQ(native_.events.create_offsets,
            (std::vector<uint64_t>{native_.events.handle, native_.events.handle,
                                   native_.events.handle}));
  // The foreign page is not our allocation; uninstalled backing remains
  // freeable.
  ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
  instance_ = nullptr;
  EXPECT_EQ(native_.Count(Operation::kFreeSignalPage), 1u);
}

TEST_F(KfdInstanceTest, UninstalledEventBackingIsReleasedAtContextTeardown) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  native_.failure = Operation::kCreateMemoryEvent;
  amdf_gpu_kfd_event_info_t info = {};
  EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(EIO));
  ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
  instance_ = nullptr;
  EXPECT_EQ(native_.Count(Operation::kFreeSignalPage), 1u);
  EXPECT_EQ(native_.events.handle, 0u);
}

TEST_F(KfdInstanceTest, EventTeardownRetainsNativeOwnerAcrossFailures) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  amdf_gpu_kfd_event_info_t info = {};
  ASSERT_EQ(CreateEvent(topology, &info), AMDF_STATUS_OK);
  ReleaseSignals();
  native_.failure = Operation::kDestroyEvent;
  EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), amdf_linux_error(EIO));
  EXPECT_TRUE(native_.closed.empty());
  EXPECT_EQ(native_.Count(Operation::kUnmapSignalPage), 0u);
  native_.failure = Operation::kUnmapSignalPage;
  native_.events.mapping_before_failure = true;
  EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), amdf_linux_error(EIO));
  EXPECT_TRUE(native_.closed.empty());
  EXPECT_TRUE(native_.events.ids.empty());
  EXPECT_TRUE(native_.events.mapped.empty());
  ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
  instance_ = nullptr;
  EXPECT_EQ(native_.events.unmap_progress, (std::vector<uint32_t>{0, 1}));
}

TEST_F(KfdInstanceTest, FailedEventMappingCanBeTornDownWithoutRecreation) {
  for (bool mapped : {false, true}) {
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
    native_.failure = Operation::kMapSignalPage;
    native_.events.mapping_before_failure = mapped;
    amdf_gpu_kfd_event_info_t info = {};
    EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(EIO));
    ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
    instance_ = nullptr;
    EXPECT_EQ(native_.events.handle, 0u);
    EXPECT_TRUE(native_.events.mapped.empty());
    EXPECT_EQ(native_.Count(Operation::kUnmapSignalPage), mapped ? 1u : 0u);
    native_.operations.clear();
  }
}

TEST_F(KfdInstanceTest, UninstalledBackingReleaseFailureKeepsContextAlive) {
  amdf_gpu_kfd_topology_t topology = {};
  ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
  native_.failure = Operation::kCreateMemoryEvent;
  amdf_gpu_kfd_event_info_t info = {};
  EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(EIO));
  native_.failure = Operation::kFreeSignalPage;
  EXPECT_EQ(amdf_gpu_umd_instance_destroy(instance_), amdf_linux_error(EIO));
  EXPECT_TRUE(native_.closed.empty());
  EXPECT_TRUE(native_.events.mapped.empty());
  EXPECT_NE(native_.events.handle, 0u);
  ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
  instance_ = nullptr;
  EXPECT_EQ(native_.Count(Operation::kUnmapSignalPage), 1u);
  EXPECT_EQ(native_.Count(Operation::kFreeSignalPage), 2u);
}

TEST_F(KfdInstanceTest, AmbiguousInstallationEndsWithItsNativeContext) {
  for (bool installed : {false, true}) {
    amdf_gpu_kfd_topology_t topology = {};
    ASSERT_NO_FATAL_FAILURE(PrepareEventVm(11, &topology));
    native_.failure = Operation::kCreateSignalEvent;
    native_.failure_errno = ENOMEM;
    native_.events.install_before_failure = installed;
    amdf_gpu_kfd_event_info_t info = {};
    EXPECT_EQ(CreateEvent(topology, &info), amdf_linux_error(ENOMEM));
    ASSERT_EQ(amdf_gpu_umd_instance_destroy(instance_), AMDF_STATUS_OK);
    instance_ = nullptr;
    EXPECT_EQ(native_.Count(Operation::kFreeSignalPage), 0u);
    EXPECT_TRUE(native_.events.mapped.empty());
    EXPECT_TRUE(native_.events.ids.empty());
    // A secondary context's close reclaims installed and ordinary allocations.
    native_.events.handle = 0;
    native_.events.installed_handle = 0;
  }
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
    if (!native_state->Record(Operation::kVersion)) {
      return -1;
    }
    *static_cast<kfd_ioctl_get_version_args*>(argument) = native_state->version;
    return 0;
  }
  if (request == DRM_IOCTL_AMDGPU_INFO) {
    auto* query = static_cast<drm_amdgpu_info*>(argument);
    EXPECT_EQ(query->query, AMDGPU_INFO_DEV_INFO);
    EXPECT_EQ(query->return_size, sizeof(drm_amdgpu_info_device));
    if (!native_state->Record(Operation::kDeviceInfo)) {
      return -1;
    }
    *reinterpret_cast<drm_amdgpu_info_device*>(query->return_pointer) =
        native_state->device_info;
    native_state->queried_descriptor = descriptor;
    return 0;
  }
  auto& state = *native_state;
  auto& events = state.events;
  if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
    auto& allocate =
        *static_cast<kfd_ioctl_alloc_memory_of_gpu_args*>(argument);
    EXPECT_EQ(allocate.size, KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t));
    EXPECT_EQ(allocate.va_addr % 4096, 0u);
    EXPECT_EQ(allocate.flags, KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                  KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                                  (UINT32_C(1) << 31));
    if (!state.Record(Operation::kAllocateSignalPage)) {
      return -1;
    }
    EXPECT_EQ(events.handle, 0u);
    events.handle = 0x550000000b;
    allocate.handle = events.handle;
    return 0;
  }
  if (request == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
    auto& map = *static_cast<kfd_ioctl_map_memory_to_gpu_args*>(argument);
    EXPECT_EQ(map.handle, events.handle);
    EXPECT_EQ(map.n_devices, 1u);
    events.map_progress.push_back(map.n_success);
    const bool success = state.Record(Operation::kMapSignalPage);
    if (success || events.mapping_before_failure) {
      const uint32_t gpu_id =
          *reinterpret_cast<const uint32_t*>(map.device_ids_array_ptr);
      if (map.n_success == 0) {
        EXPECT_TRUE(events.mapped.insert(gpu_id).second);
      }
      map.n_success = 1;
    }
    return success ? 0 : -1;
  }
  if (request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU) {
    auto& unmap = *static_cast<kfd_ioctl_unmap_memory_from_gpu_args*>(argument);
    EXPECT_EQ(unmap.handle, events.handle);
    EXPECT_EQ(unmap.n_devices, 1u);
    events.unmap_progress.push_back(unmap.n_success);
    const bool success = state.Record(Operation::kUnmapSignalPage);
    if (success || events.mapping_before_failure) {
      const uint32_t gpu_id =
          *reinterpret_cast<const uint32_t*>(unmap.device_ids_array_ptr);
      if (unmap.n_success == 0) {
        EXPECT_EQ(events.mapped.erase(gpu_id), 1u);
      }
      unmap.n_success = 1;
    }
    return success ? 0 : -1;
  }
  if (request == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
    auto& release = *static_cast<kfd_ioctl_free_memory_of_gpu_args*>(argument);
    EXPECT_EQ(release.handle, events.handle);
    EXPECT_TRUE(events.mapped.empty());
    EXPECT_NE(events.installed_handle, release.handle);
    if (!state.Record(Operation::kFreeSignalPage)) {
      return -1;
    }
    events.handle = 0;
    return 0;
  }
  if (request == AMDKFD_IOC_CREATE_EVENT) {
    auto& create = *static_cast<kfd_ioctl_create_event_args*>(argument);
    if (create.event_type == KFD_IOC_EVENT_SIGNAL) {
      events.create_offsets.push_back(create.event_page_offset);
      const bool success = state.Record(Operation::kCreateSignalEvent);
      if (create.event_page_offset != 0) {
        EXPECT_EQ(create.event_page_offset, events.handle);
        if (events.installed_handle != 0) {
          errno = EINVAL;
          return -1;
        }
        if (success || events.install_before_failure) {
          events.installed_handle = create.event_page_offset;
        }
      }
      if (!success) {
        return -1;
      }
      if (events.installed_handle == 0) {
        errno = ENOMEM;
        return -1;
      }
      EXPECT_EQ(create.auto_reset, 1u);
    } else {
      EXPECT_TRUE(create.event_type == KFD_IOC_EVENT_MEMORY ||
                  create.event_type == KFD_IOC_EVENT_HW_EXCEPTION);
      EXPECT_EQ(create.auto_reset, 0u);
      const auto operation = create.event_type == KFD_IOC_EVENT_MEMORY
                                 ? Operation::kCreateMemoryEvent
                                 : Operation::kCreateHardwareEvent;
      if (!state.Record(operation)) {
        return -1;
      }
    }
    create.event_id = events.next_id++;
    create.event_slot_index = create.event_id;
    create.event_trigger_data = create.event_id;
    events.ids.emplace(create.event_id, create.event_type);
    return 0;
  }
  if (request == AMDKFD_IOC_DESTROY_EVENT) {
    auto& destroy = *static_cast<kfd_ioctl_destroy_event_args*>(argument);
    if (!state.Record(Operation::kDestroyEvent)) {
      return -1;
    }
    EXPECT_EQ(events.ids.erase(destroy.event_id), 1u);
    return 0;
  }
  EXPECT_EQ(request, AMDKFD_IOC_CREATE_PROCESS);
  EXPECT_EQ(argument, nullptr);
  return native_state->Record(Operation::kCreateProcess) ? 0 : -1;
}

amdf_status_t __wrap_amdf_linux_endpoint_open_file(
    const amdf_platform_endpoint_t* endpoint, int* out_descriptor) {
  EXPECT_NE(endpoint, nullptr);
  const int descriptor = native_state->Open(Operation::kOpenRender);
  if (descriptor < 0) {
    return amdf_linux_error(EIO);
  }
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
  EXPECT_EQ(topology->virtual_address.fragment_size,
            native_state->device_info.pte_fragment_size);
  EXPECT_EQ(bootstrap->buffer_handle, 0u);
  EXPECT_NE(topology->gpu_id, 0u);
  EXPECT_EQ(page_size, 4096u);
  EXPECT_EQ(native_api, amdf_gpu_kfd_vm_default_native_api());
  bootstrap->render_descriptor = render_descriptor;
  bootstrap->buffer_handle = 9;
  if (!native_state->Record(Operation::kAcquire)) {
    return amdf_linux_error(EIO);
  }
  native_state->acquired.insert(render_descriptor);
  return AMDF_STATUS_OK;
}

amdf_status_t __wrap_amdf_gpu_kfd_vm_bootstrap_release(
    amdf_gpu_kfd_vm_bootstrap_t* bootstrap) {
  if (bootstrap->buffer_handle == 0) {
    return AMDF_STATUS_OK;
  }
  EXPECT_EQ(native_state->descriptors.count(bootstrap->render_descriptor), 1u);
  if (!native_state->Record(Operation::kReleaseBootstrap)) {
    return amdf_linux_error(EIO);
  }
  bootstrap->buffer_handle = 0;
  return AMDF_STATUS_OK;
}
}  // extern "C"
