// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/net/carrier/rdma/libverbs.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HsaRuntime {
 public:
  ~HsaRuntime() {
    if (initialized_) {
      iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
    }
  }

  iree_status_t Initialize(iree_allocator_t host_allocator) {
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa_);
    initialized_ = iree_status_is_ok(status);
    return status;
  }

  const iree_hal_amdgpu_libhsa_t* get() const { return &libhsa_; }

 private:
  bool initialized_ = false;
  iree_hal_amdgpu_libhsa_t libhsa_ = {};
};

class HsaTopology {
 public:
  ~HsaTopology() {
    if (initialized_) {
      iree_hal_amdgpu_topology_deinitialize(&topology_);
    }
  }

  iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* libhsa) {
    iree_status_t status =
        iree_hal_amdgpu_topology_initialize_with_defaults(libhsa, &topology_);
    initialized_ = iree_status_is_ok(status);
    return status;
  }

  const iree_hal_amdgpu_topology_t* get() const { return &topology_; }

 private:
  bool initialized_ = false;
  iree_hal_amdgpu_topology_t topology_ = {};
};

class HsaAllocation {
 public:
  explicit HsaAllocation(const iree_hal_amdgpu_libhsa_t* libhsa)
      : libhsa_(libhsa) {}

  ~HsaAllocation() { Reset(); }

  iree_status_t Allocate(hsa_amd_memory_pool_t memory_pool, size_t size) {
    Reset();
    void* ptr = nullptr;
    iree_status_t status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(libhsa_), memory_pool, size,
        HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &ptr);
    if (iree_status_is_ok(status)) {
      ptr_ = ptr;
      size_ = size;
    }
    return status;
  }

  void Reset() {
    if (!ptr_) return;
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(libhsa_, ptr_));
    ptr_ = nullptr;
    size_ = 0;
  }

  void* ptr() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  const iree_hal_amdgpu_libhsa_t* libhsa_ = nullptr;
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

class HsaDmaBuf {
 public:
  explicit HsaDmaBuf(const iree_hal_amdgpu_libhsa_t* libhsa)
      : libhsa_(libhsa) {}

  ~HsaDmaBuf() { Reset(); }

  iree_status_t Export(const void* ptr, size_t size) {
    Reset();
    int dmabuf = -1;
    uint64_t offset = 0;
    iree_status_t status = iree_hsa_amd_portable_export_dmabuf(
        IREE_LIBHSA(libhsa_), ptr, size, &dmabuf, &offset);
    if (iree_status_is_ok(status)) {
      fd_ = dmabuf;
      offset_ = offset;
    } else if (dmabuf >= 0) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(
          iree_hsa_amd_portable_close_dmabuf_raw(libhsa_, dmabuf));
    }
    return status;
  }

  void Reset() {
    if (fd_ < 0) return;
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_portable_close_dmabuf_raw(libhsa_, fd_));
    fd_ = -1;
    offset_ = 0;
  }

  int fd() const { return fd_; }
  uint64_t offset() const { return offset_; }

 private:
  const iree_hal_amdgpu_libhsa_t* libhsa_ = nullptr;
  int fd_ = -1;
  uint64_t offset_ = 0;
};

class LibverbsRuntime {
 public:
  ~LibverbsRuntime() {
    if (initialized_) {
      iree_net_libverbs_deinitialize(&libverbs_);
    }
  }

  iree_status_t Initialize(iree_allocator_t host_allocator) {
    iree_status_t status = iree_net_libverbs_initialize(
        iree_string_view_list_empty(), host_allocator, &libverbs_);
    initialized_ = iree_status_is_ok(status);
    return status;
  }

  const iree_net_libverbs_t* get() const { return &libverbs_; }

 private:
  bool initialized_ = false;
  iree_net_libverbs_t libverbs_ = {};
};

class RdmaDeviceList {
 public:
  explicit RdmaDeviceList(const iree_net_libverbs_t* libverbs)
      : libverbs_(libverbs) {}

  ~RdmaDeviceList() {
    if (devices_) {
      libverbs_->ibv_free_device_list(devices_);
    }
  }

  void Initialize() { devices_ = libverbs_->ibv_get_device_list(&count_); }

  struct ibv_device* first() const {
    return count_ > 0 ? devices_[0] : nullptr;
  }
  int count() const { return count_; }

 private:
  const iree_net_libverbs_t* libverbs_ = nullptr;
  struct ibv_device** devices_ = nullptr;
  int count_ = 0;
};

class RdmaContext {
 public:
  explicit RdmaContext(const iree_net_libverbs_t* libverbs)
      : libverbs_(libverbs) {}

  ~RdmaContext() {
    if (context_) {
      libverbs_->ibv_close_device(context_);
    }
  }

  bool Open(struct ibv_device* device) {
    context_ = libverbs_->ibv_open_device(device);
    return context_ != nullptr;
  }

  struct ibv_context* get() const { return context_; }

 private:
  const iree_net_libverbs_t* libverbs_ = nullptr;
  struct ibv_context* context_ = nullptr;
};

class RdmaProtectionDomain {
 public:
  explicit RdmaProtectionDomain(const iree_net_libverbs_t* libverbs)
      : libverbs_(libverbs) {}

  ~RdmaProtectionDomain() { Reset(); }

  bool Allocate(struct ibv_context* context) {
    pd_ = libverbs_->ibv_alloc_pd(context);
    return pd_ != nullptr;
  }

  int Reset() {
    if (!pd_) return 0;
    int result = libverbs_->ibv_dealloc_pd(pd_);
    pd_ = nullptr;
    return result;
  }

  struct ibv_pd* get() const { return pd_; }

 private:
  const iree_net_libverbs_t* libverbs_ = nullptr;
  struct ibv_pd* pd_ = nullptr;
};

class RdmaMemoryRegion {
 public:
  explicit RdmaMemoryRegion(const iree_net_libverbs_t* libverbs)
      : libverbs_(libverbs) {}

  ~RdmaMemoryRegion() { Reset(); }

  struct ibv_mr* RegisterDmaBuf(struct ibv_pd* pd, uint64_t offset,
                                size_t length, uint64_t iova, int fd,
                                int access) {
    Reset();
    mr_ = libverbs_->ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access);
    return mr_;
  }

  int Reset() {
    if (!mr_) return 0;
    int result = libverbs_->ibv_dereg_mr(mr_);
    mr_ = nullptr;
    return result;
  }

 private:
  const iree_net_libverbs_t* libverbs_ = nullptr;
  struct ibv_mr* mr_ = nullptr;
};

TEST(DmaBufRegistrationTest, RegistersAmdgpuMemoryWithRdma) {
  iree_allocator_t host_allocator = iree_allocator_system();

  HsaRuntime hsa;
  iree_status_t status = hsa.Initialize(host_allocator);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    GTEST_SKIP() << "HSA runtime not available";
  }

  HsaTopology topology;
  IREE_ASSERT_OK(topology.Initialize(hsa.get()));
  if (topology.get()->gpu_agent_count == 0) {
    GTEST_SKIP() << "no AMDGPU agents available";
  }

  hsa_amd_memory_pool_t gpu_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      hsa.get(), topology.get()->gpu_agents[0], &gpu_memory_pool));

  constexpr size_t kAllocationSize = 64 * 1024;
  HsaAllocation allocation(hsa.get());
  IREE_ASSERT_OK(allocation.Allocate(gpu_memory_pool, kAllocationSize));

  hsa_amd_pointer_info_t pointer_info = {};
  pointer_info.size = sizeof(pointer_info);
  IREE_ASSERT_OK(iree_hsa_amd_pointer_info(
      IREE_LIBHSA(hsa.get()), allocation.ptr(), &pointer_info, /*alloc=*/NULL,
      /*num_agents_accessible=*/NULL, /*accessible=*/NULL));
  ASSERT_EQ(pointer_info.type, HSA_EXT_POINTER_TYPE_HSA);
  EXPECT_EQ(pointer_info.agentBaseAddress, allocation.ptr());
  EXPECT_GE(pointer_info.sizeInBytes, allocation.size());

  HsaDmaBuf dmabuf(hsa.get());
  status = dmabuf.Export(allocation.ptr(), allocation.size());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    GTEST_SKIP() << "AMDGPU memory dma-buf export not available";
  }
  ASSERT_GE(dmabuf.fd(), 0);

  LibverbsRuntime libverbs;
  status = libverbs.Initialize(host_allocator);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    GTEST_SKIP() << "libibverbs not available";
  }
  if (!iree_net_libverbs_has_dmabuf_mr(libverbs.get())) {
    GTEST_SKIP() << "libibverbs does not export ibv_reg_dmabuf_mr";
  }

  RdmaDeviceList devices(libverbs.get());
  devices.Initialize();
  if (devices.count() == 0) {
    GTEST_SKIP() << "no RDMA devices available";
  }

  RdmaContext context(libverbs.get());
  errno = 0;
  if (!context.Open(devices.first())) {
    const int error = errno;
    GTEST_SKIP() << "could not open RDMA device "
                 << libverbs.get()->ibv_get_device_name(devices.first()) << ": "
                 << (error ? strerror(error) : "unknown error");
  }

  RdmaProtectionDomain protection_domain(libverbs.get());
  errno = 0;
  if (!protection_domain.Allocate(context.get())) {
    const int error = errno;
    GTEST_SKIP() << "could not allocate RDMA protection domain: "
                 << (error ? strerror(error) : "unknown error");
  }

  constexpr int kAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  const uint64_t page_mask = (uint64_t)page_size - 1;
  const uint64_t iova = (uint64_t)(uintptr_t)allocation.ptr();
  if ((iova & page_mask) != (dmabuf.offset() & page_mask)) {
    GTEST_SKIP() << "AMDGPU dma-buf offset does not share the allocation "
                    "pointer page offset";
  }
  RdmaMemoryRegion memory_region(libverbs.get());
  errno = 0;
  struct ibv_mr* mr = memory_region.RegisterDmaBuf(
      protection_domain.get(), dmabuf.offset(), allocation.size(), iova,
      dmabuf.fd(), kAccess);
  if (!mr) {
    const int error = errno;
    if (error == EOPNOTSUPP || error == ENOTSUP || error == ENOSYS ||
        error == ENODEV) {
      GTEST_SKIP() << "RDMA device does not support dma-buf MR registration: "
                   << strerror(error);
    }
    FAIL() << "ibv_reg_dmabuf_mr failed: "
           << (error ? strerror(error) : "unknown error") << " (errno " << error
           << ")";
  }

  EXPECT_EQ(mr->addr, allocation.ptr());
  EXPECT_EQ(mr->length, allocation.size());
  EXPECT_EQ(0, memory_region.Reset());
  EXPECT_EQ(0, protection_domain.Reset());
}

}  // namespace
