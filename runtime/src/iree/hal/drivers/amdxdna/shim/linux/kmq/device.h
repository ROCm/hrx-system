// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef PCIE_DEVICE_LINUX_XDNA_H
#define PCIE_DEVICE_LINUX_XDNA_H

#include <sys/types.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "bo_flags.h"
#include "fence.h"

namespace shim_xdna {
struct pdev;
struct bo;
struct hw_ctx;

enum class power_mode : uint8_t {
  default_mode = 0,
  low,
  medium,
  high,
  turbo,
};

struct pdev {
  mutable std::mutex m_lock;
  mutable int m_dev_fd = -1;
  mutable std::unique_ptr<bo> m_dev_heap_bo;
  // Extra 64 MiB DEV_HEAP chunks after the first. XRT grows the committed
  // heap this way when a DEV BO does not fit (driver returns EAGAIN) up to
  // the 512 MiB NPU4 maximum. Chunks are mapped contiguously so the kernel's
  // heap UVA check accepts them.
  mutable std::vector<std::unique_ptr<bo>> m_dev_heap_chunks;
  void* m_heap_raw = nullptr;
  size_t m_heap_raw_size = 0;
  void* m_heap_base = nullptr;
  size_t m_heap_committed = 0;
  int m_init_errno = 0;
  bool m_is_aie4 = false;
  // Live bytes suballocated from the committed DEV heap (AMDXDNA_BO_DEV),
  // rounded to the driver's 32 KiB allocation quantum (dev_mem_buf_shift=15).
  // The committed heap starts as one 64 MiB chunk and grows, when one DEV BO
  // is larger than that chunk, in 64 MiB steps up to the driver maximum.
  // Maintained by bo alloc/free so the
  // HAL sees real drm_mm occupancy -- HAL instruction BOs plus shim-internal
  // PDI/command BOs -- and can trim retained caches before the heap fills.
  // Excludes the DEV_HEAP container chunks and non-heap SHMEM/CMD BOs.
  // UMQ/AIE4 has no DEV heap, so this stays zero there.
  mutable std::atomic<uint64_t> m_dev_heap_used_bytes{0};

  pdev();
  explicit pdev(const std::filesystem::path& device_path);
  ~pdev();

  int init_errno() const;
  bool is_aie4() const { return m_is_aie4; }
  // Add one 64 MiB DEV_HEAP chunk. Caller holds m_lock. Returns 0, or ENOSPC
  // when the committed heap is already at the 512 MiB driver maximum.
  int expand_dev_heap();
  int open_device(const std::filesystem::path& device_path);
  // Returns 0 on success or the failing errno for recoverable ioctl failures.
  int try_ioctl(unsigned long cmd, void* arg) const;
  int try_mmap(void* addr, size_t len, int prot, int flags, off_t offset,
               void** out_ptr) const;
};

struct device {
  enum class access_mode : uint8_t { exclusive = 0, shared = 1 };
  // One process-wide DEV heap, shared across HAL device instances. The
  // committed heap starts at one 64 MiB chunk and the shim grows it when a
  // single DEV BO is larger than that chunk. The 64 MiB figure kept by the
  // HAL is the retained-cache budget, not that maximum.
  std::shared_ptr<pdev> m_pdev;
  uint32_t n_rows;
  uint32_t n_cols;
  int m_init_errno = 0;

  device(uint32_t n_rows, uint32_t n_cols);
  device(uint32_t n_rows, uint32_t n_cols,
         const std::filesystem::path& device_path);
  device(uint32_t n_rows, uint32_t n_cols, power_mode mode);
  device(uint32_t n_rows, uint32_t n_cols,
         const std::filesystem::path& device_path, power_mode mode);
  ~device();
  static int create(uint32_t n_rows, uint32_t n_cols,
                    const std::filesystem::path& device_path,
                    std::unique_ptr<device>* out_device);
  int init_errno() const;
  bool is_aie4() const { return m_pdev->is_aie4(); }

  const pdev& get_pdev() const;

  // Live occupancy of the committed DEV heap, rounded to the driver's 32 KiB
  // allocation quantum so it matches the real drm_mm footprint the heap allocator
  // enforces, letting callers bound retained command code/context images against
  // measured pressure regardless of BO size mix.
  uint64_t get_dev_heap_used_bytes() const;

  int alloc_bo(uint32_t ctx_id, size_t size, shim_amdxdna_bo_flags flags,
               std::unique_ptr<bo>* out_bo);
  int alloc_bo(size_t size, uint32_t flags, std::unique_ptr<bo>* out_bo);
  int alloc_bo(size_t size, shim_amdxdna_bo_flags flags,
               std::unique_ptr<bo>* out_bo);

  int create_hw_context(const std::vector<uint8_t>& pdi,
                        const std::string& cu_name,
                        const std::map<std::string, uint32_t>& qos,
                        bool* out_context_pool_exhausted,
                        std::unique_ptr<hw_ctx>* out_context);
  int create_hw_context(const std::vector<uint8_t>& pdi,
                        const std::string& cu_name,
                        bool* out_context_pool_exhausted,
                        std::unique_ptr<hw_ctx>* out_context);
  // AIE4 / UMQ: allocate a host queue BO, skip CONFIG_CU when PDI is empty.
  // `partition_cols` is XRT elf.get_partition_size(); 0 means 1 column.
  int create_hw_context(const std::vector<uint8_t>& pdi,
                        const std::string& cu_name, bool umq,
                        uint32_t partition_cols,
                        bool* out_context_pool_exhausted,
                        std::unique_ptr<hw_ctx>* out_context);

  int read_aie_mem(uint16_t col, uint16_t row, uint32_t offset, uint32_t size,
                   std::vector<char>* out_buf);
  int write_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                    const std::vector<char>& buf, size_t* out_size);
  int read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr,
                   uint32_t* out_reg_val);
  int write_aie_reg_checked(uint16_t col, uint16_t row, uint32_t reg_addr,
                            uint32_t reg_val);

  int get_power_mode(power_mode* out_mode) const;
  // Returns 0 on success or the errno from DRM_IOCTL_AMDXDNA_SET_STATE.
  int set_power_mode(power_mode mode) const;

  int create_fence(fence_handle::access_mode,
                   std::unique_ptr<fence_handle>* out_fence);
  int import_fence(pid_t, int, std::unique_ptr<fence_handle>* out_fence);
};

std::filesystem::path find_default_accel_device_path();
// Returns the NPU architecture name ("Phoenix"/"Strix") read from the device's
// sysfs vbnv, or an empty string if no NPU is found. Does not open the device.
std::string query_npu_arch();
// Resolves 0 rows/cols to hardware defaults without constructing a device.
// Returns 0 on success or an errno-style value on open/ioctl/metadata failure.
int resolve_core_grid_size(const std::filesystem::path& device_path,
                           uint32_t requested_rows, uint32_t requested_cols,
                           uint32_t* out_rows, uint32_t* out_cols);
std::string read_sysfs(const std::string& filename);
std::string stringify_power_mode(power_mode mode);

}  // namespace shim_xdna

#endif
