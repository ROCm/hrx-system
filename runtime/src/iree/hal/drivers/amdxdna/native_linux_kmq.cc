// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amdxdna/native.h"
#include "iree/hal/drivers/amdxdna/native_linux_kmq_internal.h"
#include "iree/hal/drivers/amdxdna/shim/linux/kmq/amdxdna_accel.h"
#include "iree/hal/drivers/amdxdna/shim/linux/kmq/bo.h"
#include "iree/hal/drivers/amdxdna/shim/linux/kmq/device.h"
#include "iree/hal/drivers/amdxdna/shim/linux/kmq/hwctx.h"
#include "iree/hal/drivers/amdxdna/shim/linux/kmq/hwq.h"
#include "iree/hal/drivers/amdxdna/shim/linux/kmq/kernel.h"
#include "iree/hal/drivers/amdxdna/util.h"

struct iree_hal_amdxdna_native_device_t {
  iree_allocator_t host_allocator;
  std::unique_ptr<shim_xdna::device> shim_device;
  std::filesystem::path device_path;
  std::string driver_version_storage;
  std::string driver_srcversion_storage;
  iree_hal_amdxdna_native_c_driver_stack_t driver_stack = {};
  iree_hal_amdxdna_native_c_command_chain_status_t command_chain_status =
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_ENABLED_BY_DEFAULT;
  bool supports_command_chain = true;
  // Concurrent hardware-context budget for this part, resolved once at device
  // creation from the NPU architecture (the KMD has no query for it). 0 when
  // the architecture is unrecognized. See query_caps.
  uint32_t hardware_context_budget = 0;
  std::atomic<iree_host_size_t> live_context_image_bytes{0};
  bool is_aie4 = false;
  std::mutex command_pool_mutex;
  std::vector<std::unique_ptr<shim_xdna::kernel>> start_npu_command_pool;

  iree_hal_amdxdna_native_device_t(
      iree_allocator_t host_allocator,
      std::unique_ptr<shim_xdna::device> shim_device,
      std::filesystem::path device_path)
      : host_allocator(host_allocator),
        shim_device(std::move(shim_device)),
        device_path(std::move(device_path)) {}
};

struct iree_hal_amdxdna_native_buffer_t {
  std::unique_ptr<shim_xdna::bo> bo;

  explicit iree_hal_amdxdna_native_buffer_t(std::unique_ptr<shim_xdna::bo> bo)
      : bo(std::move(bo)) {}
};

struct iree_hal_amdxdna_native_queue_t {
  iree_allocator_t host_allocator = iree_allocator_system();
  shim_xdna::hw_q* hwq = nullptr;
};

struct iree_hal_amdxdna_native_context_t {
  std::unique_ptr<shim_xdna::hw_ctx> context;
  iree_hal_amdxdna_native_queue_t queue;

  explicit iree_hal_amdxdna_native_context_t(
      iree_allocator_t host_allocator,
      std::unique_ptr<shim_xdna::hw_ctx> context)
      : context(std::move(context)) {
    queue.host_allocator = host_allocator;
    queue.hwq = this->context->get_hw_queue();
  }
};

struct iree_hal_amdxdna_native_command_t {
  iree_hal_amdxdna_native_device_t* device;
  iree_hal_amdxdna_native_c_command_opcode_t opcode;
  std::unique_ptr<shim_xdna::kernel> kernel;
  bool has_bound_buffers = false;
  // Snapshot of the compiled ERT header. CERT completion overwrites the whole
  // header word with just the terminal state, so reuse must restore opcode,
  // count, and type before flipping state back to NEW.
  uint32_t compiled_header = 0;
  // Full exec-BO copy taken once after compile. CERT's SVA completion write
  // is the header, but restoring only that word still leaves a reuse hole if
  // firmware clobbers payload/cu_mask; XRT keeps the compiled packet intact
  // and only rewrites the header/state on start().
  std::vector<uint8_t> compiled_exec;

  iree_hal_amdxdna_native_command_t(
      iree_hal_amdxdna_native_device_t* device,
      iree_hal_amdxdna_native_c_command_opcode_t opcode,
      std::unique_ptr<shim_xdna::kernel> kernel)
      : device(device), opcode(opcode), kernel(std::move(kernel)) {}
};

struct iree_hal_amdxdna_native_context_ref_t {
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdxdna_native_device_t* device;
  iree_host_size_t pdi_bytes;
  iree_hal_amdxdna_native_context_t* context;
};

void iree_hal_amdxdna_native_submission_destroy(
    iree_hal_amdxdna_native_submission_t* submission);

iree_status_t iree_hal_amdxdna_native_device_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps);

namespace {

constexpr size_t kMaxExecBoSize = 4096;
// Linux KMQ allocates instruction and PDI BOs from one process-wide device
// heap. Keep one command-construction reserve outside retained cache budgets.
constexpr uint32_t kSharedCodeMemoryBytes = 64u * 1024u * 1024u;
// Free window kept outside the retained command caches so a fresh command chain
// *and* a fresh hardware context can be constructed without tripping the device
// heap's ENOSPC limit.
//
// The heap is a real 64 MiB hardware aperture (AIE2_DEVM_SIZE = SZ_64M at
// dev_mem_base 0x4000000), sub-allocated by the driver with best-fit drm_mm at a
// 32 KiB alignment quantum (dev_mem_buf_shift=15 on this NPU4/Strix part). Every
// AMDXDNA_BO_DEV therefore consumes roundup(size, 32 KiB) of heap address space,
// so the aperture holds exactly 2048 32 KiB slots. live_dev_heap_bytes() now
// charges that true 32 KiB quantum, so it is an exact, BO-mix-independent mirror
// of the driver's drm_mm occupancy (the earlier 4 KiB tally undercounted and was
// mix-dependent, which is why the historical reserve had to be inflated to
// 34 MiB). With exact accounting the reserve is a principled margin, not a fudge:
//   - 8 MiB  for a context-create + command-build burst (its transient PDI and
//     small instruction BOs, which are individually <=64 KiB and fit scattered
//     32 KiB holes, so this is a total-footprint reserve, not a contiguity one);
//   - 8 MiB  best-fit fragmentation / large-BO contiguity headroom (an on-device
//     fragmentation probe showed scattered frees can cap a single contiguous
//     alloc even with tens of MiB nominally free).
// Reserving 16 MiB keeps retained real occupancy <= 48 MiB, i.e. >=16 MiB below
// the hard 64 MiB ceiling, and the chain cache self-trims idle entries against
// this reserve so a new build's first-try allocation succeeds. This is the
// proactive alternative to the old "allocation failed, drop idle caches and
// retry" recovery, which we deliberately avoid (it can leave the NPU wedged).
constexpr uint32_t kSharedCodeMemoryMissReserveBytes = 16u * 1024u * 1024u;

// The exec BO is large enough to hold many chain slots, but XRT's runlist
// implementation chunks native command chains at 24 children. Match that
// firmware-facing scheduling unit instead of advertising the larger packet
// capacity: longer KMQ chains are functionally valid in small tests, but full
// decode-style profiling shows worse device/completion latency for the same
// work when a 28-child section is submitted as one chain instead of XRT's 24+4
// split.
constexpr uint32_t kKmqDefaultChainSlots = 24;

constexpr const char* kKnownBadUbuntu617Srcversion = "2DBDA75956CAFA9D029EA89";

constexpr uint32_t kCommandChainFirmwareMinMajor = 1;
constexpr uint32_t kCommandChainFirmwareMinMinor = 1;
constexpr uint32_t kCommandChainFirmwareMinPatch = 0;
constexpr uint32_t kCommandChainFirmwareMinBuild = 0;

// START_NPU commands are used as KMQ chain children with register-map
// arguments on Linux. Reusing only this opcode avoids stale BO binding-table
// state while removing per-dispatch exec-BO create/destroy churn. The pool is
// bounded independently from the parent-chain chunk size: a logical recorded
// runlist can span multiple 24-child parent chains while still needing cached
// child START_NPU commands for the whole group.
constexpr size_t kMaxStartNpuCommandPoolSize = 64;

struct firmware_version_t {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t patch = 0;
  uint32_t build = 0;
  bool valid = false;
};

struct driver_stack_info_t {
  std::string module_version;
  std::string module_srcversion;
  firmware_version_t firmware;
  uint32_t vendor = 0;
  uint32_t device = 0;
  uint32_t revision = 0;
  bool has_vendor = false;
  bool has_device = false;
  bool has_revision = false;
};

std::string read_first_line(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::string line;
  if (file.is_open()) std::getline(file, line);
  return line;
}

bool parse_sysfs_u32(const std::string& value, uint32_t* out_value) {
  if (value.empty()) return false;
  char* end = nullptr;
  errno = 0;
  unsigned long parsed = std::strtoul(value.c_str(), &end, 0);
  if (errno != 0 || end == value.c_str() ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *out_value = static_cast<uint32_t>(parsed);
  return true;
}

std::filesystem::path sysfs_device_path_for_accel_node(
    const std::filesystem::path& device_path) {
  const std::filesystem::path filename = device_path.filename();
  if (filename.empty()) return {};
  std::filesystem::path sysfs_path =
      std::filesystem::path("/sys/class/accel") / filename / "device";
  std::error_code ec;
  if (std::filesystem::exists(sysfs_path, ec) && !ec) return sysfs_path;
  return {};
}

int firmware_compare(const firmware_version_t& lhs, uint32_t major,
                     uint32_t minor, uint32_t patch, uint32_t build) {
  if (lhs.major != major) return lhs.major < major ? -1 : 1;
  if (lhs.minor != minor) return lhs.minor < minor ? -1 : 1;
  if (lhs.patch != patch) return lhs.patch < patch ? -1 : 1;
  if (lhs.build != build) return lhs.build < build ? -1 : 1;
  return 0;
}

bool is_xdna2_pci_revision(const driver_stack_info_t& info) {
  if (!info.has_vendor || !info.has_device || !info.has_revision) return false;
  if (info.vendor != 0x1022) return false;
  switch (info.device) {
    case 0x17f0:
      return info.revision == 0x10 || info.revision == 0x11 ||
             info.revision == 0x20;
    case 0x1502:
      return info.revision == 0x00;
    default:
      return false;
  }
}

bool device_is_aie4(const iree_hal_amdxdna_native_device_t* device) {
  return device && device->is_aie4;
}

firmware_version_t query_firmware_version(const shim_xdna::pdev& pdev) {
  amdxdna_drm_query_firmware_version firmware = {};
  amdxdna_drm_get_info arg = {
      .param = DRM_AMDXDNA_QUERY_FIRMWARE_VERSION,
      .buffer_size = sizeof(firmware),
      .buffer = reinterpret_cast<uintptr_t>(&firmware),
  };
  firmware_version_t result = {};
  if (pdev.try_ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg) != 0) {
    return result;
  }
  result.major = firmware.major;
  result.minor = firmware.minor;
  result.patch = firmware.patch;
  result.build = firmware.build;
  result.valid = true;
  return result;
}

driver_stack_info_t query_driver_stack_info(
    iree_hal_amdxdna_native_device_t* device) {
  driver_stack_info_t info = {};
  info.module_version = read_first_line("/sys/module/amdxdna/version");
  info.module_srcversion = read_first_line("/sys/module/amdxdna/srcversion");
  info.firmware = query_firmware_version(device->shim_device->get_pdev());

  const std::filesystem::path sysfs_device_path =
      sysfs_device_path_for_accel_node(device->device_path);
  if (!sysfs_device_path.empty()) {
    info.has_vendor = parse_sysfs_u32(
        read_first_line(sysfs_device_path / "vendor"), &info.vendor);
    info.has_device = parse_sysfs_u32(
        read_first_line(sysfs_device_path / "device"), &info.device);
    info.has_revision = parse_sysfs_u32(
        read_first_line(sysfs_device_path / "revision"), &info.revision);
  }
  return info;
}

bool has_known_bad_amdxdna_srcversion(const driver_stack_info_t& info) {
  return info.module_srcversion == kKnownBadUbuntu617Srcversion;
}

iree_hal_amdxdna_native_c_command_chain_status_t select_command_chain_status(
    const driver_stack_info_t& info) {
  iree_hal_amdxdna_native_c_command_chain_status_t status =
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_ENABLED_BY_DEFAULT;
  if (has_known_bad_amdxdna_srcversion(info)) {
    status =
        IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_KNOWN_BAD_STACK;
  } else if (is_xdna2_pci_revision(info)) {
    // There is no ordered kernel-module srcversion. Firmware is the ordered
    // local signal; only enable native parent chains by default once firmware
    // reaches the command-chain-capable floor. Older 1.0.x stock firmware has
    // now failed on multiple Ubuntu amdxdna stacks.
    if (info.firmware.valid) {
      if (firmware_compare(info.firmware, kCommandChainFirmwareMinMajor,
                           kCommandChainFirmwareMinMinor,
                           kCommandChainFirmwareMinPatch,
                           kCommandChainFirmwareMinBuild) < 0) {
        status =
            IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_OLD_FIRMWARE;
      }
    } else if (info.module_version.empty()) {
      status =
          IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_UNIDENTIFIED_STACK;
    }
  }

  return status;
}

bool command_chain_enabled(
    iree_hal_amdxdna_native_c_command_chain_status_t status) {
  return status ==
         IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_ENABLED_BY_DEFAULT;
}

void record_driver_stack_info(iree_hal_amdxdna_native_device_t* device,
                              const driver_stack_info_t& info) {
  device->driver_version_storage = info.module_version;
  device->driver_srcversion_storage = info.module_srcversion;
  device->driver_stack = {};
  device->driver_stack.driver_version =
      iree_make_string_view(device->driver_version_storage.data(),
                            device->driver_version_storage.size());
  device->driver_stack.driver_srcversion =
      iree_make_string_view(device->driver_srcversion_storage.data(),
                            device->driver_srcversion_storage.size());
  device->driver_stack.has_firmware_version = info.firmware.valid;
  device->driver_stack.firmware_major = info.firmware.major;
  device->driver_stack.firmware_minor = info.firmware.minor;
  device->driver_stack.firmware_patch = info.firmware.patch;
  device->driver_stack.firmware_build = info.firmware.build;
  device->driver_stack.has_pci_ids =
      info.has_vendor && info.has_device && info.has_revision;
  device->driver_stack.pci_vendor_id = info.vendor;
  device->driver_stack.pci_device_id = info.device;
  device->driver_stack.pci_revision_id = info.revision;
}

std::unique_ptr<shim_xdna::kernel> acquire_start_npu_command_from_pool(
    iree_hal_amdxdna_native_device_t* device) {
  std::lock_guard<std::mutex> lock(device->command_pool_mutex);
  if (device->start_npu_command_pool.empty()) return nullptr;
  std::unique_ptr<shim_xdna::kernel> kernel =
      std::move(device->start_npu_command_pool.back());
  device->start_npu_command_pool.pop_back();
  return kernel;
}

bool return_start_npu_command_to_pool(
    iree_hal_amdxdna_native_device_t* device,
    std::unique_ptr<shim_xdna::kernel> kernel) {
  if (!device || !kernel) return false;
  if (kernel->reset() != 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(device->command_pool_mutex);
  if (device->start_npu_command_pool.size() >= kMaxStartNpuCommandPoolSize) {
    return false;
  }
  device->start_npu_command_pool.push_back(std::move(kernel));
  return true;
}

std::string string_view_to_string(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

iree_status_t parse_power_mode(
    iree_string_view_t power_mode,
    iree_hal_amdxdna_native_c_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_should_set_power_mode = false;
  *out_power_mode = IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT;
  if (iree_string_view_is_empty(power_mode)) return iree_ok_status();

  *out_should_set_power_mode = true;
  if (iree_string_view_equal(power_mode, IREE_SV("default"))) {
    *out_power_mode = IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT;
  } else if (iree_string_view_equal(power_mode, IREE_SV("low"))) {
    *out_power_mode = IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_LOW;
  } else if (iree_string_view_equal(power_mode, IREE_SV("medium"))) {
    *out_power_mode = IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_MEDIUM;
  } else if (iree_string_view_equal(power_mode, IREE_SV("high"))) {
    *out_power_mode = IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_HIGH;
  } else if (iree_string_view_equal(power_mode, IREE_SV("turbo"))) {
    *out_power_mode = IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_TURBO;
  } else {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_power_mode' expected to be default | low | "
        "medium | high | turbo but got '%.*s'",
        static_cast<int>(power_mode.size), power_mode.data);
  }
  return iree_ok_status();
}

shim_xdna::power_mode to_shim_power_mode(
    iree_hal_amdxdna_native_c_power_mode_t power_mode) {
  switch (power_mode) {
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT:
      return shim_xdna::power_mode::default_mode;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_LOW:
      return shim_xdna::power_mode::low;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_MEDIUM:
      return shim_xdna::power_mode::medium;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_HIGH:
      return shim_xdna::power_mode::high;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_TURBO:
      return shim_xdna::power_mode::turbo;
  }
  return shim_xdna::power_mode::default_mode;
}

uint32_t to_shim_buffer_flags(iree_hal_amdxdna_native_buffer_c_type_t type,
                              bool umq) {
  // UMQ/AIE4 has no DEV_HEAP. XRT instruction BOs are CACHEABLE, which
  // xdna-driver maps to AMDXDNA_BO_CMD when heap is absent. HRX maps
  // EXECBUF to CMD; HOST_ONLY stays SHARE for data BOs.
  if (umq) {
    if (type == IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION ||
        type == IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE) {
      return AMDXDNA_BO_FLAGS_EXECBUF;
    }
    return AMDXDNA_BO_FLAGS_HOST_ONLY;
  }
  switch (type) {
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY:
      return AMDXDNA_BO_FLAGS_HOST_ONLY;
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE:
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION:
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CONTROL_PARAMETER:
      return AMDXDNA_BO_FLAGS_CACHEABLE;
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_PDI:
      // LOAD_PDI reads from the device SRAM address space. Match XRT's
      // xrt_core::bo_int::use_type::pdi allocation contract.
      return AMDXDNA_BO_FLAGS_CACHEABLE;
  }
  return AMDXDNA_BO_FLAGS_HOST_ONLY;
}

shim_xdna::direction to_shim_sync_direction(
    iree_hal_amdxdna_native_buffer_sync_direction_t direction) {
  switch (direction) {
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE:
      return shim_xdna::direction::host2device;
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST:
      return shim_xdna::direction::device2host;
  }
  return shim_xdna::direction::host2device;
}

uint32_t to_ert_opcode(iree_hal_amdxdna_native_c_command_opcode_t opcode) {
  switch (opcode) {
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU:
      return ERT_START_CU;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU:
      return ERT_START_NPU;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_DPU:
      return ERT_START_DPU;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF:
      IREE_ASSERT_UNREACHABLE(
          "Linux KMQ rejects PARTIAL_ELF before opcode conversion");
      return ERT_START_NPU;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN:
      return ERT_CMD_CHAIN;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_FULL_ELF:
      return ERT_START_NPU_PREEMPT_ELF;
  }
  return ERT_START_CU;
}

uint32_t chain_slot_capacity(size_t exec_bo_size) {
  const size_t header = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data);
  return exec_bo_size > header
             ? static_cast<uint32_t>((exec_bo_size - header) / sizeof(uint64_t))
             : 1;
}

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command) {
  return reinterpret_cast<ert_packet*>(
      command->kernel->get_exec_buf_bo()->map());
}

iree_status_t validate_device_size_fits_size_t(iree_device_size_t size) {
  if (IREE_UNLIKELY(size > static_cast<iree_device_size_t>(
                               std::numeric_limits<size_t>::max()))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native allocation size is too large");
  }
  return iree_ok_status();
}

}  // namespace

iree_status_code_t iree_hal_amdxdna_native_linux_bo_allocation_status_code(
    int error_number) {
  const int normalized_error = error_number < 0 ? -error_number : error_number;
  // Drivers report a full shared device heap as either EAGAIN or ENOSPC.
  // Treat both as transient so the common HAL can reclaim idle native
  // resources and retry the allocation once.
  if (normalized_error == EAGAIN || normalized_error == ENOSPC) {
    return IREE_STATUS_UNAVAILABLE;
  }
  return iree_status_code_from_errno(normalized_error);
}

bool iree_hal_amdxdna_native_linux_firmware_supports_full_elf(bool has_version,
                                                              uint32_t major,
                                                              uint32_t minor,
                                                              uint32_t patch,
                                                              uint32_t build) {
  if (!has_version) return false;
  // The public release tuple is distinct from the internal firmware protocol
  // tuple. 1.0.20.31 is the oldest upstream release known to carry protocol
  // 6.12/AIE2_PREEMPT. Treat support as monotonic from that release onward.
  constexpr uint32_t kMinMajor = 1;
  constexpr uint32_t kMinMinor = 0;
  constexpr uint32_t kMinPatch = 20;
  constexpr uint32_t kMinBuild = 31;
  if (major != kMinMajor) return major > kMinMajor;
  if (minor != kMinMinor) return minor > kMinMinor;
  if (patch != kMinPatch) return patch > kMinPatch;
  return build >= kMinBuild;
}

iree_status_t iree_hal_amdxdna_native_resolve_device_options(
    const iree_hal_amdxdna_device_params* options,
    iree_hal_amdxdna_device_params* out_options,
    std::string* out_device_path_storage,
    iree_hal_amdxdna_native_c_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_options = *options;

  if (options->n_core_rows < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_rows' expected a non-negative int32_t but "
        "got %d",
        options->n_core_rows);
  }
  if (options->n_core_cols < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_cols' expected a non-negative int32_t but "
        "got %d",
        options->n_core_cols);
  }
  IREE_RETURN_IF_ERROR(parse_power_mode(options->power_mode, out_power_mode,
                                        out_should_set_power_mode));

  std::filesystem::path device_path =
      iree_string_view_is_empty(options->device_path)
          ? shim_xdna::find_default_accel_device_path()
          : std::filesystem::path(string_view_to_string(options->device_path));
  if (::access(device_path.c_str(), R_OK | W_OK) != 0) {
    const int saved_errno = errno;
    return iree_make_status(iree_status_code_from_errno(saved_errno),
                            "unable to access amdxdna device path '%s'",
                            device_path.c_str());
  }

  uint32_t n_core_rows = static_cast<uint32_t>(options->n_core_rows);
  uint32_t n_core_cols = static_cast<uint32_t>(options->n_core_cols);
  const int err = shim_xdna::resolve_core_grid_size(
      device_path, n_core_rows, n_core_cols, &n_core_rows, &n_core_cols);
  if (err != 0) {
    return iree_make_status(
        iree_status_code_from_errno(err),
        "unable to query amdxdna core grid for device path '%s' (requested "
        "rows=%d cols=%d)",
        device_path.c_str(), options->n_core_rows, options->n_core_cols);
  }

  *out_device_path_storage = device_path.string();
  out_options->device_path = iree_make_string_view(
      out_device_path_storage->data(), out_device_path_storage->size());
  out_options->n_core_rows = static_cast<int32_t>(n_core_rows);
  out_options->n_core_cols = static_cast<int32_t>(n_core_cols);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  std::filesystem::path device_path;
  if (!iree_string_view_is_empty(options->device_path)) {
    device_path = string_view_to_string(options->device_path);
  }
  const std::filesystem::path resolved_device_path =
      device_path.empty() ? shim_xdna::find_default_accel_device_path()
                          : device_path;
  std::unique_ptr<shim_xdna::device> shim_device;
  const int err =
      shim_xdna::device::create(static_cast<uint32_t>(options->n_core_rows),
                                static_cast<uint32_t>(options->n_core_cols),
                                resolved_device_path, &shim_device);
  if (err != 0) {
    return iree_make_status(
        iree_status_code_from_errno(err),
        "unable to open amdxdna device path '%.*s' with core grid %" PRIi32
        "x%" PRIi32,
        static_cast<int>(options->device_path.size), options->device_path.data,
        options->n_core_rows, options->n_core_cols);
  }

  iree_hal_amdxdna_native_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*device), reinterpret_cast<void**>(&device)));
  device = new (device) iree_hal_amdxdna_native_device_t(
      host_allocator, std::move(shim_device), resolved_device_path);
  device->is_aie4 = device->shim_device->is_aie4();
  const driver_stack_info_t driver_stack_info = query_driver_stack_info(device);
  record_driver_stack_info(device, driver_stack_info);
  if (device->is_aie4) {
    // ERT_CMD_CHAIN is the AIE2P parent packet. AIE4 uses per-column
    // ert_dpu_data inside one START_DPU command.
    device->command_chain_status =
        IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_UNIDENTIFIED_STACK;
    device->supports_command_chain = false;
  } else {
    device->command_chain_status =
        select_command_chain_status(driver_stack_info);
    device->supports_command_chain =
        command_chain_enabled(device->command_chain_status);
  }
  // Resolve the per-architecture hardware-context budget once. query_npu_arch()
  // reads sysfs, so cache it here rather than on every query_caps() call.
  const std::string npu_arch = shim_xdna::query_npu_arch();
  device->hardware_context_budget =
      iree_hal_amdxdna_hardware_context_budget_for_arch(
          iree_make_string_view(npu_arch.data(), npu_arch.size()));
  *out_device = device;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_device_destroy(
    iree_hal_amdxdna_native_device_t* device) {
  if (!device) return;
  iree_allocator_t host_allocator = device->host_allocator;
  device->~iree_hal_amdxdna_native_device_t();
  iree_allocator_free(host_allocator, device);
}

iree_status_t iree_hal_amdxdna_native_device_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_power_mode_t power_mode) {
  IREE_ASSERT_ARGUMENT(device);
  return iree_hal_amdxdna_status_from_errno(
      device->shim_device->set_power_mode(to_shim_power_mode(power_mode)),
      "amdxdna set power mode failed");
}

bool iree_hal_amdxdna_native_device_supports_partial_elf_dispatch(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_c_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return (caps.dispatch_models &
          IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF) != 0;
}

bool iree_hal_amdxdna_native_device_uses_npu_payload_dispatch(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_c_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return (caps.dispatch_models &
          IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0;
}

iree_hal_amdxdna_native_c_command_opcode_t
iree_hal_amdxdna_native_device_dispatch_opcode(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_c_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU;
  }
  return caps.default_dispatch_opcode;
}

iree_status_t iree_hal_amdxdna_native_device_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_caps);
  iree_hal_amdxdna_native_c_device_caps_t caps = {};
  caps.max_effective_queues = 1;
  caps.max_command_chain_slots =
      device->supports_command_chain
          ? std::min(chain_slot_capacity(kMaxExecBoSize), kKmqDefaultChainSlots)
          : 0;
  // Zero selects the common conservative chain-cache retention budget.
  caps.max_cached_chain_child_commands = 0;
  // Linux keys contexts by PDI+CU and publishes the architecture table. Do
  // not subtract the Windows XRS non-RT reserved slot here; that mapping is
  // Windows MCDM CreateContext / 0xc01e0009, and the Linux create-hwctx
  // pool signal is ENOENT/ENOSPC/EINVAL, not EAGAIN.
  caps.max_hardware_contexts = device->hardware_context_budget;
  caps.max_shared_code_memory_bytes = kSharedCodeMemoryBytes;
  caps.shared_code_memory_miss_reserve_bytes =
      kSharedCodeMemoryMissReserveBytes;
  if (device_is_aie4(device)) {
    // UMQ has no DEV heap. Leave the shared-memory domain disabled so the
    // KMQ 64 MiB budget and 16 MiB reserve are not applied here.
    caps.max_shared_code_memory_bytes = 0;
    caps.shared_code_memory_miss_reserve_bytes = 0;
    caps.context_image_models =
        IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_NONE;
    caps.dispatch_models = IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_DPU;
    caps.max_command_chain_slots = 0;
    caps.default_dispatch_opcode =
        IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_DPU;
  } else {
    caps.context_image_models =
        IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_PDI;
    // START_NPU is used for command-chain children and is correct on Linux KMQ.
    // Do not advertise PARTIAL_ELF here: its resident-instruction path
    // currently produces wrong results for kernels with per-dispatch moving
    // I/O. The command dirty hooks below only sync exec BO mutations; they do
    // not make the PARTIAL_ELF resident-instruction model correct on Linux.
    caps.dispatch_models = IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_CU |
                           IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU;
    if (device->supports_command_chain) {
      caps.dispatch_models |=
          IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_COMMAND_CHAIN;
    }
    caps.default_dispatch_opcode =
        IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU;
    // KMD owns the authoritative firmware-protocol feature mask, but it does
    // not expose that mask through UAPI. Drivers paired with pre-protocol
    // older firmware have accepted opcode 22 into KMQ and left it forever in
    // ERT state NEW instead of returning EOPNOTSUPP. Fail closed below the
    // oldest release known to carry AIE2_PREEMPT.
    if (iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
            device->driver_stack.has_firmware_version,
            device->driver_stack.firmware_major,
            device->driver_stack.firmware_minor,
            device->driver_stack.firmware_patch,
            device->driver_stack.firmware_build)) {
      caps.dispatch_models |= IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_FULL_ELF;
    }
  }
  caps.completion_models =
      IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_SYNCHRONOUS_WAIT |
      IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_NATIVE_FENCE;
  caps.supports_host_buffer_reuse = true;
  caps.native_owns_control_code_publication = false;
  // CERT overwrites the ERT header on AIE4 completion. Issue restores it
  // before EXEC_CMD. The single-command cache will not reuse an in-flight
  // exec BO, so a second overlapped submit allocates another command.
  caps.submit_completion_is_deferred = true;
  caps.supports_external_buffer_import = false;
  caps.supports_external_buffer_export = false;
  caps.supports_real_multi_queue = false;
  caps.command_chain_status = device->command_chain_status;
  caps.driver_stack = device->driver_stack;
  *out_caps = caps;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(size));

  std::unique_ptr<shim_xdna::bo> bo;
  const size_t host_size = static_cast<size_t>(size);
  const uint32_t flags =
      to_shim_buffer_flags(type, device_is_aie4(device));
  const int err = device->shim_device->alloc_bo(host_size, flags, &bo);
  if (err != 0) {
    const int normalized_err = err < 0 ? -err : err;
    return iree_make_status(
        iree_hal_amdxdna_native_linux_bo_allocation_status_code(normalized_err),
        "amdxdna native BO allocation failed: type=%d size=%" PRIu64
        " flags=0x%08x errno %d",
        (int)type, (uint64_t)size, flags, normalized_err);
  }
  // DEV-heap accounting is maintained by the shim bo layer (pdev), which sees
  // both these HAL instruction BOs and the shim-internal PDI/command BOs, so no
  // per-buffer bookkeeping is needed here.
  *out_buffer = new iree_hal_amdxdna_native_buffer_t(std::move(bo));
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_c_context_image_t* image,
    bool* out_context_pool_exhausted,
    iree_hal_amdxdna_native_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(image);
  IREE_ASSERT_ARGUMENT(out_context_pool_exhausted);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context_pool_exhausted = false;
  *out_context = nullptr;
  const bool is_aie4 = device_is_aie4(device);
  if (is_aie4) {
    if (IREE_UNLIKELY(image->type !=
                      IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_NONE)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Linux AIE4 context creation requires an empty context "
          "image (no PDI/xclbin)");
    }
  } else if (IREE_UNLIKELY(image->type !=
                           IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_PDI)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna Linux KMQ context creation requires a "
                            "PDI context image");
  }
  iree_const_byte_span_t pdi = image->pdi;
  if (IREE_UNLIKELY(pdi.data_length != 0 && !pdi.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native context PDI data is NULL");
  }

  std::vector<uint8_t> pdi_vector;
  if (!is_aie4 && pdi.data_length != 0) {
    pdi_vector.assign(pdi.data, pdi.data + pdi.data_length);
  }
  std::string kernel_name_string;
  if (!iree_string_view_is_empty(image->kernel_name)) {
    kernel_name_string.assign(image->kernel_name.data, image->kernel_name.size);
  }

  std::unique_ptr<shim_xdna::hw_ctx> shim_context;
  const int err =
      is_aie4 ? device->shim_device->create_hw_context(
                    pdi_vector, kernel_name_string, /*umq=*/true,
                    image->partition_cols, out_context_pool_exhausted,
                    &shim_context)
              : device->shim_device->create_hw_context(
                    pdi_vector, kernel_name_string, out_context_pool_exhausted,
                    &shim_context);
  if (err != 0) {
    return iree_hal_amdxdna_status_from_errno(
        err, "amdxdna hardware context creation failed");
  }
  *out_context = new iree_hal_amdxdna_native_context_t(device->host_allocator,
                                                       std::move(shim_context));
  return iree_ok_status();
}

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context) {
  delete context;
}

iree_status_t iree_hal_amdxdna_native_context_close_single_aperture_session(
    iree_hal_amdxdna_native_context_t* context) {
  (void)context;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_max_slots);
  iree_hal_amdxdna_native_c_device_caps_t caps;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_device_query_caps(device, &caps));
  *out_max_slots = caps.max_command_chain_slots;
  return iree_ok_status();
}

size_t iree_hal_amdxdna_native_command_arg_binding_capacity() { return 1024; }

void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  delete buffer;
}

iree_status_t iree_hal_amdxdna_native_buffer_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = nullptr;
  if (IREE_UNLIKELY(!buffer || !buffer->bo)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  void* ptr = buffer->bo->map();
  if (IREE_UNLIKELY(!ptr)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not host-mapped");
  }
  *out_ptr = ptr;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_device_size_t size, iree_device_size_t offset) {
  if (IREE_UNLIKELY(!buffer || !buffer->bo)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(size));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(offset));
  return iree_hal_amdxdna_status_from_errno(
      buffer->bo->sync(to_shim_sync_direction(direction),
                       static_cast<size_t>(size), static_cast<size_t>(offset)),
      "amdxdna native buffer sync failed");
}

iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction) {
  if (IREE_UNLIKELY(!buffer || !buffer->bo)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  return iree_hal_amdxdna_status_from_errno(
      buffer->bo->sync(to_shim_sync_direction(direction)),
      "amdxdna native buffer sync failed");
}

iree_status_t iree_hal_amdxdna_native_buffer_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (IREE_UNLIKELY(!buffer || !buffer->bo)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  return iree_ok_status();
}

uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return buffer->bo->get_paddr();
}

iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return static_cast<iree_device_size_t>(buffer->bo->size());
}

iree_status_t iree_hal_amdxdna_native_context_open_cu(
    iree_hal_amdxdna_native_context_t* context, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_c_cu_index_t* out_cu_index) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_cu_index);
  std::string kernel_name_string;
  if (!iree_string_view_is_empty(kernel_name)) {
    kernel_name_string.assign(kernel_name.data, kernel_name.size);
  }
  shim_xdna::cuidx_t cu_index{.index = 0};
  const int err =
      context->context->open_cu_context(kernel_name_string, &cu_index);
  if (err != 0) {
    return iree_hal_amdxdna_status_from_errno(err, "amdxdna CU lookup failed");
  }
  out_cu_index->index = cu_index.index;
  return iree_ok_status();
}

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_queue(
    iree_hal_amdxdna_native_context_t* context) {
  return &context->queue;
}

uint64_t iree_hal_amdxdna_native_queue_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue) {
  return queue->hwq->exec_cmd_count();
}

iree_status_t iree_hal_amdxdna_native_command_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_t** out_command) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command);
  *out_command = nullptr;

  if (opcode ==
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Linux KMQ does not support START_NPU_PARTIAL_ELF; "
        "PARTIAL_ELF dispatch is not advertised");
  }
  if (opcode == IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_DPU &&
      !device_is_aie4(device)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Linux KMQ does not support START_DPU; "
        "AIE4 DPU dispatch requires a UMQ AIE4 device");
  }
  if (opcode == IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN &&
      !device->supports_command_chain) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Linux KMQ command chains disabled by driver-stack policy "
        "(status=%d)",
        static_cast<int>(device->command_chain_status));
  }

  const bool poolable_start_npu =
      opcode == IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  std::unique_ptr<shim_xdna::kernel> kernel;
  if (poolable_start_npu) {
    kernel = acquire_start_npu_command_from_pool(device);
  }
  if (!kernel) {
    kernel = std::make_unique<shim_xdna::kernel>(
        device->shim_device->get_pdev(), to_ert_opcode(opcode));
  }
  if (kernel->init_errno() != 0) {
    const int normalized_err =
        kernel->init_errno() < 0 ? -kernel->init_errno() : kernel->init_errno();
    return iree_make_status(
        iree_status_code_from_errno(normalized_err),
        "amdxdna native command allocation failed: opcode=%d exec_bo_size=%zu "
        "errno %d",
        (int)opcode, (size_t)kMaxExecBoSize, normalized_err);
  }
  *out_command =
      new iree_hal_amdxdna_native_command_t(device, opcode, std::move(kernel));
  return iree_ok_status();
}

void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command) {
  const bool has_command = command != nullptr;
  if (has_command) {
    if (command->opcode == IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU &&
        !command->has_bound_buffers) {
      return_start_npu_command_to_pool(command->device,
                                       std::move(command->kernel));
    }
  }
  delete command;
}

iree_status_t iree_hal_amdxdna_native_command_reset(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  command->has_bound_buffers = false;
  command->compiled_header = 0;
  command->compiled_exec.clear();
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->reset(), "amdxdna native command reset failed");
}

iree_status_t iree_hal_amdxdna_native_command_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_c_cu_index_t cu_index) {
  shim_xdna::cuidx_t shim_cu_index{.index = cu_index.index};
  command->kernel->set_cu_idx(shim_cu_index);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size) {
  if (IREE_UNLIKELY(!control_buffer || control_buffer_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna Linux KMQ instruction buffer is empty");
  }
  if (IREE_UNLIKELY(control_buffer_size >
                    iree_hal_amdxdna_native_buffer_size(control_buffer))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Linux KMQ instruction byte count exceeds BO size");
  }
  if (IREE_UNLIKELY(control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Linux KMQ instruction byte count is not word aligned");
  }
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_ctrl_bo(*control_buffer->bo,
                                   static_cast<size_t>(control_buffer_size)),
      "amdxdna native command control-buffer argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_start_dpu_columns(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size,
    const iree_hal_amdxdna_native_c_dpu_column_t* columns,
    iree_host_size_t column_count) {
  if (IREE_UNLIKELY(!control_buffer || control_buffer_size == 0 || !columns ||
                    column_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna Linux KMQ START_DPU columns are empty");
  }
  if (IREE_UNLIKELY(column_count > std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna Linux KMQ START_DPU column count is too "
                            "large");
  }
  if (IREE_UNLIKELY(control_buffer_size >
                    iree_hal_amdxdna_native_buffer_size(control_buffer))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Linux KMQ instruction byte count exceeds BO size");
  }
  std::vector<uint16_t> uc_index(column_count);
  std::vector<uint32_t> byte_offset(column_count);
  std::vector<uint32_t> byte_size(column_count);
  for (iree_host_size_t i = 0; i < column_count; ++i) {
    uc_index[i] = columns[i].uc_index;
    byte_offset[i] = columns[i].byte_offset;
    byte_size[i] = columns[i].byte_size;
  }
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_dpu_columns(
          *control_buffer->bo, static_cast<size_t>(control_buffer_size),
          uc_index.data(), byte_offset.data(), byte_size.data(),
          static_cast<uint32_t>(column_count)),
      "amdxdna native START_DPU column packet failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_32(value),
      "amdxdna native command u32 argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_64(value),
      "amdxdna native command u64 argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_update_arg_64(
    iree_hal_amdxdna_native_command_t* command, iree_host_size_t arg_index,
    uint64_t value) {
  if (IREE_UNLIKELY(arg_index > std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna native command argument index is too large");
  }
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->update_arg_64(static_cast<uint32_t>(arg_index), value),
      "amdxdna native command u64 argument update failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  iree_status_t status = iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_bo(*buffer->bo),
      "amdxdna native command buffer argument failed");
  if (iree_status_is_ok(status)) command->has_bound_buffers = true;
  return status;
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset) {
  iree_status_t status = iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_bo_at_offset(*buffer->bo, offset),
      "amdxdna native command buffer argument failed");
  if (iree_status_is_ok(status)) command->has_bound_buffers = true;
  return status;
}

iree_status_t iree_hal_amdxdna_native_command_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size) {
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(offset));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(size));
  iree_status_t status = iree_hal_amdxdna_status_from_errno(
      command->kernel->get_exec_buf_bo()->bind_at(position, *buffer->bo,
                                                  static_cast<size_t>(offset),
                                                  static_cast<size_t>(size)),
      "amdxdna native command buffer binding failed");
  if (iree_status_is_ok(status)) command->has_bound_buffers = true;
  return status;
}

iree_status_t iree_hal_amdxdna_native_command_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  (void)command;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna Linux KMQ native commands do not support rebinding prepared "
      "command BO tables");
}

static void iree_hal_amdxdna_native_command_restore_ert_header(
    iree_hal_amdxdna_native_command_t* command) {
  ert_packet* packet = command_packet(command);
  // AIE2P rewrites kernel-arg VAs in this packet on cache reuse, then calls
  // mark_code_dirty. A full snapshot would put the previous dispatch's args
  // back. Only START_DPU's packet is stable across reuse (args live in the
  // instruction BO); CERT clobbers that header.
  if (command->opcode !=
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_DPU) {
    packet->state = ERT_CMD_STATE_NEW;
    return;
  }
  shim_xdna::bo* exec_bo = command->kernel->get_exec_buf_bo();
  void* mapped = exec_bo ? exec_bo->map() : nullptr;
  const size_t exec_size = exec_bo ? exec_bo->size() : 0;
  // XRT caches the compiled header on the first start and writes that word
  // back on every later start. CERT DMA-writes COMPLETED as the whole header
  // (opcode/count become 0), so a state-nibble RMW is not enough. Snapshot
  // the whole exec BO while opcode is still the compiled START_DPU.
  if (command->compiled_exec.empty() && mapped && exec_size != 0 &&
      packet->opcode != 0) {
    command->compiled_header = packet->header;
    command->compiled_exec.resize(exec_size);
    std::memcpy(command->compiled_exec.data(), mapped, exec_size);
  }
  if (!command->compiled_exec.empty() && mapped &&
      command->compiled_exec.size() <= exec_size) {
    std::memcpy(mapped, command->compiled_exec.data(),
                command->compiled_exec.size());
    packet->header =
        (command->compiled_header & ~0xFu) | ERT_CMD_STATE_NEW;
  } else if (command->compiled_header != 0) {
    packet->header =
        (command->compiled_header & ~0xFu) | ERT_CMD_STATE_NEW;
  } else {
    packet->state = ERT_CMD_STATE_NEW;
  }
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  iree_hal_amdxdna_native_command_restore_ert_header(command);
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->get_exec_buf_bo()->sync(
          shim_xdna::direction::host2device),
      "amdxdna native chain command sync failed");
}

iree_status_t iree_hal_amdxdna_native_command_mark_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  iree_hal_amdxdna_native_command_restore_ert_header(command);
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->get_exec_buf_bo()->sync(
          shim_xdna::direction::host2device),
      "amdxdna native command sync failed");
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_mark_chain_dirty(command);
}

iree_status_t iree_hal_amdxdna_native_command_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  if (IREE_UNLIKELY(command->opcode !=
                    IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (IREE_UNLIKELY(command_count > std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command chain is too large");
  }

  shim_xdna::bo* chain_bo = command->kernel->get_exec_buf_bo();
  const size_t chain_bytes = offsetof(ert_packet, data) +
                             sizeof(ert_cmd_chain_data) +
                             command_count * sizeof(uint64_t);
  if (chain_bytes > chain_bo->size()) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna cmd-chain: %" PRIhsz
                            " slots exceed exec buffer (%zu > %zu bytes)",
                            command_count, chain_bytes, chain_bo->size());
  }

  ert_packet* packet = command_packet(command);
  std::memset(packet, 0, chain_bo->size());
  packet->state = ERT_CMD_STATE_NEW;
  packet->opcode = ERT_CMD_CHAIN;
  ert_cmd_chain_data* chain_data =
      reinterpret_cast<ert_cmd_chain_data*>(packet->data);
  chain_data->command_count = static_cast<uint32_t>(command_count);
  chain_data->submit_index = 0;
  chain_data->error_index = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    shim_xdna::bo* child_bo = commands[i]->kernel->get_exec_buf_bo();
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_status_from_errno(
        child_bo->sync(shim_xdna::direction::host2device),
        "amdxdna cmd-chain child command sync failed"));
    chain_data->data[i] = child_bo->get_drm_bo_handle();
  }
  packet->count =
      (sizeof(ert_cmd_chain_data) + command_count * sizeof(uint64_t)) /
      sizeof(uint32_t);
  return iree_ok_status();
}

// Enqueue one command without blocking. issue_command is async: it writes the
// EXEC_CMD ioctl and returns a sequence number, so several commands can be in
// flight on the in-order hardware queue at once.
static iree_status_t iree_hal_amdxdna_native_queue_issue(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  ert_packet* packet = command_packet(command);
  if (command->opcode ==
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_DPU) {
    iree_hal_amdxdna_native_command_restore_ert_header(command);
  } else {
    // AIE2P: remember the compiled header while opcode is intact, then put
    // that word back with state NEW. Do not copy the payload; cache reuse
    // has already written this dispatch's arg VAs.
    if (packet->opcode != 0) {
      command->compiled_header = packet->header;
    }
    if (command->compiled_header != 0) {
      packet->header =
          (command->compiled_header & ~0xFu) | ERT_CMD_STATE_NEW;
    }
  }
  shim_xdna::bo* exec_bo = command->kernel->get_exec_buf_bo();
  // CMD BOs are not host-coherent. Restore writes opcode/count back after
  // CERT clobbered the word; clflush so CERT's SVA read of the ERT BO sees
  // START_DPU rather than the completed header (opcode 0) still in DRAM.
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_status_from_errno(
      exec_bo->sync(shim_xdna::direction::host2device),
      "amdxdna native command exec-buffer sync failed"));
  if (const int err = queue->hwq->issue_command(exec_bo)) {
    return iree_hal_amdxdna_status_from_errno(
        err, "amdxdna native command submit failed");
  }
  return iree_ok_status();
}

// Bound START_DPU WAIT_CMD so a hung AIE4 job cannot pin the host until
// process kill. Device TDR typically fires first (~8s); this is the HAL
// backstop if CERT hangs without a timeout state.
static constexpr uint32_t kAmdxdnaStartDpuWaitTimeoutMs = 20000;

// Block on one already-issued command and translate its terminal ERT state.
static iree_status_t iree_hal_amdxdna_native_queue_wait_issued(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  ert_packet* packet = command_packet(command);
  shim_xdna::bo* exec_bo = command->kernel->get_exec_buf_bo();
  const uint32_t timeout_ms =
      command->opcode == IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_DPU
          ? kAmdxdnaStartDpuWaitTimeoutMs
          : 0;
  const int rc = queue->hwq->wait_command(exec_bo, timeout_ms);
  if (rc < 0) {
    return iree_hal_amdxdna_status_from_errno(
        rc, "amdxdna native command wait failed");
  }
  if (rc == 0) {
    return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                            "amdxdna %.*s timed out",
                            static_cast<int>(label.size), label.data);
  }
  // WAIT_CMD means host-queue read_index advanced. Success still requires
  // CERT to write ERT_CMD_STATE_COMPLETED through completion_signal (SVA);
  // KMD job_complete() does not. TDR writes TIMEOUT/ABORT through the kernel
  // vmap of the same cmd BO. clflush so either writer is visible here.
  (void)exec_bo->sync(shim_xdna::direction::device2host);
  if (packet->state == ERT_CMD_STATE_COMPLETED) return iree_ok_status();
  if (packet->state == ERT_CMD_STATE_TIMEOUT) {
    return iree_make_status(
        IREE_STATUS_DEADLINE_EXCEEDED,
        "amdxdna %.*s timed out: ert state %u health version=%u npu_gen=%u "
        "ctx_error_type=%u",
        static_cast<int>(label.size), label.data, packet->state,
        packet->data[0], packet->data[1], packet->data[4]);
  }

  if (command->opcode ==
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN) {
    ert_cmd_chain_data* chain_data =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s did not complete: ert state %u (error_index %u, "
        "submit_index %u)",
        static_cast<int>(label.size), label.data, packet->state,
        chain_data->error_index, chain_data->submit_index);
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL, "amdxdna %.*s did not complete: ert state %u",
      static_cast<int>(label.size), label.data, packet->state);
}

iree_status_t iree_hal_amdxdna_native_queue_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_queue_issue(queue, command));
  return iree_hal_amdxdna_native_queue_wait_issued(queue, command, label);
}

iree_status_t iree_hal_amdxdna_native_queue_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label) {
  if (command_count == 0) return iree_ok_status();
  // Issue every command before waiting so a command buffer that chunked into
  // several ERT_CMD_CHAINs runs them back-to-back on the in-order queue instead
  // of stalling on a host round-trip between chunks.
  iree_status_t issue_status = iree_ok_status();
  iree_host_size_t issued_count = 0;
  for (; issued_count < command_count; ++issued_count) {
    issue_status =
        iree_hal_amdxdna_native_queue_issue(queue, commands[issued_count]);
    if (!iree_status_is_ok(issue_status)) break;
  }
  if (issued_count == 0) return issue_status;

  // Wait on the final issued chunk first: the queue completes in submit order,
  // so once the final parent has completed every earlier parent has too. This
  // is the only blocking wait on the steady-state success path.
  iree_status_t wait_status = iree_hal_amdxdna_native_queue_wait_issued(
      queue, commands[issued_count - 1], label);
  // Then verify/drain the earlier chunks. On success these are cheap -- the
  // chunks are already complete, so wait_issued returns on the first poll
  // without blocking -- but they still surface an earlier chunk that completed
  // with an error (the final chunk succeeding does not by itself prove every
  // earlier chunk did). On the failure path the same loop keeps their command
  // resources live until the kernel/firmware is done with them.
  for (iree_host_size_t i = 0; i + 1 < issued_count; ++i) {
    iree_status_t earlier_status =
        iree_hal_amdxdna_native_queue_wait_issued(queue, commands[i], label);
    if (!iree_status_is_ok(earlier_status)) {
      iree_status_ignore(wait_status);
      wait_status = earlier_status;
      break;
    }
  }
  if (!iree_status_is_ok(wait_status)) {
    if (!iree_status_is_ok(issue_status)) iree_status_ignore(issue_status);
    return wait_status;
  }
  return issue_status;
}

struct iree_hal_amdxdna_native_submission_t {
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_native_queue_t* queue;
  iree_hal_amdxdna_native_command_t* command;
  iree_hal_amdxdna_native_command_t** commands;
  iree_host_size_t command_count;
  iree_host_size_t issued_count;
  char label[128];
  size_t label_size;
  bool issued;
  bool waited;
  iree_status_t status;
};

static void iree_hal_amdxdna_native_submission_set_label(
    iree_hal_amdxdna_native_submission_t* submission,
    iree_string_view_t label) {
  submission->label_size = std::min(label.size, sizeof(submission->label) - 1);
  if (submission->label_size) {
    std::memcpy(submission->label, label.data, submission->label_size);
  }
  submission->label[submission->label_size] = '\0';
}

iree_status_t iree_hal_amdxdna_native_queue_submit(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(command);
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_submission = nullptr;

  iree_hal_amdxdna_native_submission_t* submission = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*submission), (void**)&submission));
  std::memset(submission, 0, sizeof(*submission));
  submission->host_allocator = queue->host_allocator;
  submission->queue = queue;
  submission->command = command;
  submission->status = iree_ok_status();
  iree_hal_amdxdna_native_submission_set_label(submission, label);

  iree_status_t status = iree_hal_amdxdna_native_queue_issue(queue, command);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(queue->host_allocator, submission);
    return status;
  }
  submission->issued = true;
  *out_submission = submission;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_queue_submit_all(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(commands);
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_submission = nullptr;
  if (command_count == 0) return iree_ok_status();

  iree_hal_amdxdna_native_submission_t* submission = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*submission), (void**)&submission));
  std::memset(submission, 0, sizeof(*submission));
  submission->host_allocator = queue->host_allocator;
  submission->queue = queue;
  submission->command_count = command_count;
  submission->status = iree_ok_status();
  iree_hal_amdxdna_native_submission_set_label(submission, label);

  iree_status_t status = iree_allocator_malloc_array(
      queue->host_allocator, command_count, sizeof(*submission->commands),
      (void**)&submission->commands);
  if (iree_status_is_ok(status)) {
    std::memcpy(submission->commands, commands,
                command_count * sizeof(*submission->commands));
    for (iree_host_size_t i = 0; i < command_count; ++i) {
      status = iree_hal_amdxdna_native_queue_issue(queue, commands[i]);
      if (!iree_status_is_ok(status)) break;
      submission->issued = true;
      submission->issued_count = i + 1;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_submission_destroy(submission);
    return status;
  }
  *out_submission = submission;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_submission_wait(
    iree_hal_amdxdna_native_submission_t* submission, uint64_t timeout_ns) {
  IREE_ASSERT_ARGUMENT(submission);
  // Linux KMQ's wait_command accepts millisecond timeouts and treats 0 as the
  // default/infinite wait used by submit_and_wait. The HAL semaphore layer owns
  // deadline enforcement for now, matching the Windows MCDM async DDI.
  (void)timeout_ns;
  if (!submission->waited) {
    iree_string_view_t label =
        iree_make_string_view(submission->label, submission->label_size);
    if (submission->command_count != 0) {
      if (IREE_UNLIKELY(submission->issued_count == 0)) {
        submission->status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "amdxdna native submit_all wait on a submission that was not "
            "issued");
      } else {
        iree_status_t wait_status = iree_hal_amdxdna_native_queue_wait_issued(
            submission->queue,
            submission->commands[submission->issued_count - 1], label);
        for (iree_host_size_t i = 0; i + 1 < submission->issued_count; ++i) {
          iree_status_t earlier_status =
              iree_hal_amdxdna_native_queue_wait_issued(
                  submission->queue, submission->commands[i], label);
          if (!iree_status_is_ok(earlier_status)) {
            iree_status_ignore(wait_status);
            wait_status = earlier_status;
          }
        }
        submission->status = wait_status;
      }
    } else {
      submission->status = iree_hal_amdxdna_native_queue_wait_issued(
          submission->queue, submission->command, label);
    }
    submission->waited = true;
  }
  return iree_status_clone(submission->status);
}

iree_status_t iree_hal_amdxdna_native_submission_query(
    iree_hal_amdxdna_native_submission_t* submission, bool* out_ready) {
  IREE_ASSERT_ARGUMENT(submission);
  IREE_ASSERT_ARGUMENT(out_ready);
  if (submission->waited) {
    *out_ready = true;
    return iree_ok_status();
  }
  iree_hal_amdxdna_native_command_t* command =
      submission->command_count == 0
          ? submission->command
          : submission->commands[submission->issued_count - 1];
  shim_xdna::bo* exec_bo = command->kernel->get_exec_buf_bo();
  *out_ready = shim_xdna::poll_command(exec_bo) != 0;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_submission_destroy(
    iree_hal_amdxdna_native_submission_t* submission) {
  if (!submission) return;
  if (submission->issued && !submission->waited) {
    iree_status_t wait_status = iree_hal_amdxdna_native_submission_wait(
        submission, /*timeout_ns=*/UINT64_MAX);
    iree_status_ignore(wait_status);
  }
  iree_status_ignore(submission->status);
  iree_allocator_free(submission->host_allocator, submission->commands);
  iree_allocator_free(submission->host_allocator, submission);
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_resolve_options(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_device_params* out_options,
    iree_byte_span_t* out_device_path_storage,
    iree_hal_amdxdna_native_c_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_device_path_storage = iree_byte_span_empty();
  std::string device_path_storage;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_resolve_device_options(
      options, out_options, &device_path_storage, out_power_mode,
      out_should_set_power_mode));
  if (!device_path_storage.empty()) {
    uint8_t* storage = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, device_path_storage.size(), (void**)&storage));
    std::memcpy(storage, device_path_storage.data(),
                device_path_storage.size());
    *out_device_path_storage =
        iree_make_byte_span(storage, device_path_storage.size());
    out_options->device_path =
        iree_make_string_view((const char*)storage, device_path_storage.size());
  }
  return iree_ok_status();
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  return iree_hal_amdxdna_native_device_create(options, host_allocator,
                                               out_device);
}

extern "C" void iree_hal_amdxdna_native_device_c_destroy(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_destroy(device);
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_power_mode_t power_mode) {
  return iree_hal_amdxdna_native_device_set_power_mode(device, power_mode);
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps) {
  return iree_hal_amdxdna_native_device_query_caps(device, out_caps);
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  return iree_hal_amdxdna_native_device_alloc_buffer(device, size, type,
                                                     out_buffer);
}

extern "C" void iree_hal_amdxdna_native_buffer_c_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  iree_hal_amdxdna_native_buffer_destroy(buffer);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr) {
  return iree_hal_amdxdna_native_buffer_map(buffer, out_ptr);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_device_size_t size, iree_device_size_t offset) {
  return iree_hal_amdxdna_native_buffer_sync(buffer, direction, size, offset);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction) {
  return iree_hal_amdxdna_native_buffer_sync_all(buffer, direction);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_ensure_allocated(buffer);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_queue_c_ensure_buffer_allocated(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(queue);
  return iree_hal_amdxdna_native_buffer_ensure_allocated(buffer);
}

extern "C" iree_status_t iree_hal_amdxdna_native_queue_c_prepare_pdi_buffer(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t* out_address) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_address);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_ensure_allocated(buffer));
  *out_address = iree_hal_amdxdna_native_buffer_device_address(buffer);
  return iree_ok_status();
}

extern "C" iree_status_t
iree_hal_amdxdna_native_queue_c_prepare_control_parameter_buffer(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t* out_address) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_address);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_ensure_allocated(buffer));
  *out_address = iree_hal_amdxdna_native_buffer_device_address(buffer);
  return iree_ok_status();
}

extern "C" uint64_t iree_hal_amdxdna_native_buffer_c_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_device_address(buffer);
}

extern "C" iree_device_size_t iree_hal_amdxdna_native_buffer_c_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_size(buffer);
}

extern "C" bool iree_hal_amdxdna_native_buffer_c_is_device_current(
    const iree_hal_amdxdna_native_buffer_t* buffer) {
  (void)buffer;
  return true;
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_create_context_ref(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_c_context_image_t* image,
    bool* out_context_pool_exhausted,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
  *out_context_pool_exhausted = false;
  *out_context_ref = nullptr;
  const iree_host_size_t pdi_bytes = image->pdi.data_length;
  // Charge before creation because the native constructor allocates the PDI BO
  // internally. Undo the charge on failure.
  device->live_context_image_bytes.fetch_add(pdi_bytes,
                                             std::memory_order_relaxed);
  iree_hal_amdxdna_native_context_t* raw_context = nullptr;
  iree_status_t status = iree_hal_amdxdna_native_device_create_context(
      device, image, out_context_pool_exhausted, &raw_context);
  if (!iree_status_is_ok(status)) {
    device->live_context_image_bytes.fetch_sub(pdi_bytes,
                                               std::memory_order_relaxed);
    return status;
  }
  auto* context_ref = new iree_hal_amdxdna_native_context_ref_t();
  iree_atomic_ref_count_init(&context_ref->ref_count);
  context_ref->device = device;
  context_ref->pdi_bytes = pdi_bytes;
  context_ref->context = raw_context;
  *out_context_ref = context_ref;
  return iree_ok_status();
}

extern "C" iree_hal_amdxdna_native_context_ref_t*
iree_hal_amdxdna_native_context_ref_retain(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  if (!context_ref) return nullptr;
  iree_atomic_ref_count_inc(&context_ref->ref_count);
  return context_ref;
}

extern "C" void iree_hal_amdxdna_native_context_ref_release(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  if (!context_ref) return;
  if (iree_atomic_ref_count_dec(&context_ref->ref_count) == 1) {
    iree_hal_amdxdna_native_context_destroy(context_ref->context);
    context_ref->device->live_context_image_bytes.fetch_sub(
        context_ref->pdi_bytes, std::memory_order_relaxed);
    delete context_ref;
  }
}

extern "C" iree_host_size_t
iree_hal_amdxdna_native_device_c_live_context_image_bytes(
    iree_hal_amdxdna_native_device_t* device) {
  return device
             ? device->live_context_image_bytes.load(std::memory_order_relaxed)
             : 0;
}

extern "C" iree_host_size_t
iree_hal_amdxdna_native_device_c_live_dev_heap_bytes(
    iree_hal_amdxdna_native_device_t* device) {
  if (!device || !device->shim_device) return 0;
  // Exact driver-side occupancy of the 64MiB DEV heap: every live
  // AMDXDNA_BO_DEV buffer (HAL instruction BOs plus shim-internal PDI/command
  // BOs), rounded to the driver's 32KiB allocation quantum (dev_mem_buf_shift=15)
  // so it mirrors real drm_mm address-space consumption rather than the
  // mix-dependent 4KiB byte-sum. This supersedes the per-counter reconstruction,
  // which missed the shim-internal BOs.
  return (iree_host_size_t)device->shim_device->get_dev_heap_used_bytes();
}

extern "C" iree_hal_amdxdna_native_context_t*
iree_hal_amdxdna_native_context_ref_borrow(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  return context_ref ? context_ref->context : nullptr;
}

extern "C" iree_status_t iree_hal_amdxdna_native_context_ref_open_cu(
    iree_hal_amdxdna_native_context_ref_t* context_ref,
    iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_c_cu_index_t* out_cu_index) {
  return iree_hal_amdxdna_native_context_open_cu(context_ref->context,
                                                 kernel_name, out_cu_index);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_context_ref_close_single_aperture_session(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  return iree_hal_amdxdna_native_context_close_single_aperture_session(
      context_ref->context);
}

extern "C" iree_hal_amdxdna_native_queue_t*
iree_hal_amdxdna_native_context_ref_queue(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  return iree_hal_amdxdna_native_context_queue(context_ref->context);
}

extern "C" uint64_t iree_hal_amdxdna_native_queue_c_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue) {
  return iree_hal_amdxdna_native_queue_exec_command_count(queue);
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots) {
  return iree_hal_amdxdna_native_device_query_chain_max_slots(device,
                                                              out_max_slots);
}

extern "C" iree_host_size_t
iree_hal_amdxdna_native_command_c_arg_binding_capacity(void) {
  return iree_hal_amdxdna_native_command_arg_binding_capacity();
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_t** out_command) {
  return iree_hal_amdxdna_native_command_create(device, opcode, out_command);
}

extern "C" void iree_hal_amdxdna_native_command_c_destroy(
    iree_hal_amdxdna_native_command_t* command) {
  iree_hal_amdxdna_native_command_destroy(command);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_reset(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_reset(command);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_c_cu_index_t cu_index) {
  return iree_hal_amdxdna_native_command_set_cu_index(command, cu_index);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size) {
  return iree_hal_amdxdna_native_command_add_control_buffer(
      command, control_buffer, control_buffer_size);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_start_dpu_columns(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size,
    const iree_hal_amdxdna_native_c_dpu_column_t* columns,
    iree_host_size_t column_count) {
  return iree_hal_amdxdna_native_command_add_start_dpu_columns(
      command, control_buffer, control_buffer_size, columns, column_count);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  return iree_hal_amdxdna_native_command_add_arg_32(command, value);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  return iree_hal_amdxdna_native_command_add_arg_64(command, value);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_update_arg_64(
    iree_hal_amdxdna_native_command_t* command, iree_host_size_t arg_index,
    uint64_t value) {
  return iree_hal_amdxdna_native_command_update_arg_64(command, arg_index,
                                                       value);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_command_add_buffer_arg(command, buffer);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset) {
  return iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
      command, buffer, offset);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, iree_host_size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size) {
  return iree_hal_amdxdna_native_command_bind_buffer(command, position, buffer,
                                                     offset, size);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_reset_bound_buffers(command);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_mark_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_mark_code_dirty(command);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_command_c_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_mark_chain_code_dirty(command);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  return iree_hal_amdxdna_native_command_prepare_chain(command, commands,
                                                       command_count);
}

extern "C" iree_status_t iree_hal_amdxdna_native_queue_c_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  return iree_hal_amdxdna_native_queue_submit_and_wait(queue, command, label);
}

extern "C" iree_status_t iree_hal_amdxdna_native_queue_c_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label) {
  return iree_hal_amdxdna_native_queue_submit_all_and_wait(
      queue, commands, command_count, label);
}

extern "C" iree_status_t iree_hal_amdxdna_native_queue_c_submit(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission) {
  return iree_hal_amdxdna_native_queue_submit(queue, command, label,
                                              out_submission);
}

extern "C" iree_status_t iree_hal_amdxdna_native_queue_c_submit_all(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission) {
  return iree_hal_amdxdna_native_queue_submit_all(
      queue, commands, command_count, label, out_submission);
}

extern "C" iree_status_t iree_hal_amdxdna_native_submission_c_wait(
    iree_hal_amdxdna_native_submission_t* submission, uint64_t timeout_ns) {
  return iree_hal_amdxdna_native_submission_wait(submission, timeout_ns);
}

extern "C" iree_status_t iree_hal_amdxdna_native_submission_c_query(
    iree_hal_amdxdna_native_submission_t* submission, bool* out_ready) {
  return iree_hal_amdxdna_native_submission_query(submission, out_ready);
}

extern "C" void iree_hal_amdxdna_native_submission_c_destroy(
    iree_hal_amdxdna_native_submission_t* submission) {
  iree_hal_amdxdna_native_submission_destroy(submission);
}
