// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"
#include "iree/hal/drivers/amdxdna/native.h"
#include "iree/hal/drivers/amdxdna/native_windows_mcdm_internal.h"
#include "iree/hal/drivers/amdxdna/shim/ert.h"
#include "iree/hal/drivers/amdxdna/shim/windows/mcdm/context_blob.h"
#include "iree/hal/drivers/amdxdna/shim/windows/mcdm/kmt_api.h"

namespace mcdm = iree::hal::amdxdna::mcdm;

namespace {

constexpr uint64_t kMaxExecBoSize = 4096;

// The working Windows/XRT IREE matmul capture submits START_CU/type CU with
// payload count 0x12: one CU mask plus 17 data words. The XML ABI names fewer
// arguments, but XRT pads the tail with zeros and the driver receives 0x4c
// packet bytes.
constexpr uint32_t kWindowsDpuRegmapWords = 17;
constexpr uint32_t kWindowsDpuInstructionRegWord = 2;
// XRT's module-runlist path presents START_NPU instruction streams in 0x8000
// slots starting at the ABI-negotiated aperture code view. The path-B parent
// chain descriptor is accepted for one packed child, but multi-child START_NPU
// chains fail unless we preserve this slot cadence.
constexpr size_t kWindowsDpuChainCodeAlignment = 0x8000;
constexpr uint64_t kWindowsDpuChainDescriptorApertureOffset = 0x10000;
constexpr size_t kWindowsDpuChainDescriptorHeaderSize = 0x34;
constexpr size_t kWindowsDpuStartNpuChainDescriptorSize = 0x3c;

// Typed layouts for the two path-B chain descriptor formats (firmware ABI;
// offsets recovered from xrt_core). Expressed as structs with static_asserts so
// the magic offsets are explicit and compile-time validated instead of bare
// memcpy offsets. The descriptor block is zero-initialized, so reserved fields
// stay 0.
struct WindowsDpuChainCuDescriptorHeader {
  uint32_t marker;           // +0x00, always 1
  uint32_t reserved0[10];    // +0x04..+0x2b
  uint32_t cu_index;         // +0x2c
  uint32_t copy_word_count;  // +0x30, START_CU packet words copied after header
};
static_assert(sizeof(WindowsDpuChainCuDescriptorHeader) ==
                  kWindowsDpuChainDescriptorHeaderSize,
              "CU chain descriptor header must be 0x34 bytes");
static_assert(offsetof(WindowsDpuChainCuDescriptorHeader, marker) == 0x00,
              "marker must be at +0x00");
static_assert(offsetof(WindowsDpuChainCuDescriptorHeader, cu_index) == 0x2c,
              "cu_index must be at +0x2c");
static_assert(offsetof(WindowsDpuChainCuDescriptorHeader, copy_word_count) ==
                  0x30,
              "copy_word_count must be at +0x30");

struct WindowsDpuChainNpuDescriptor {
  uint32_t marker0;        // [0]  +0x00, always 2
  uint32_t instr_addr_lo;  // [1]  +0x04
  uint32_t instr_addr_hi;  // [2]  +0x08
  uint32_t reserved0[4];   // [3..6]
  uint32_t instr_size;     // [7]  +0x1c
  uint32_t reserved1[4];   // [8..11]
  uint32_t marker1;        // [12] +0x30, always 2
  uint32_t selector;       // [13] +0x34
  uint32_t selector_hi;    // [14] +0x38
};
static_assert(sizeof(WindowsDpuChainNpuDescriptor) ==
                  kWindowsDpuStartNpuChainDescriptorSize,
              "START_NPU chain descriptor must be 0x3c bytes");
static_assert(offsetof(WindowsDpuChainNpuDescriptor, marker0) == 0x00,
              "marker0 must be at word [0]");
static_assert(offsetof(WindowsDpuChainNpuDescriptor, instr_addr_lo) == 0x04,
              "instr_addr_lo must be at word [1]");
static_assert(offsetof(WindowsDpuChainNpuDescriptor, instr_addr_hi) == 0x08,
              "instr_addr_hi must be at word [2]");
static_assert(offsetof(WindowsDpuChainNpuDescriptor, instr_size) == 0x1c,
              "instr_size must be at word [7]");
static_assert(offsetof(WindowsDpuChainNpuDescriptor, marker1) == 0x30,
              "marker1 must be at word [12]");
static_assert(offsetof(WindowsDpuChainNpuDescriptor, selector) == 0x34,
              "selector must be at word [13]");
// XRT runlists are logically unbounded but internally submitted in fixed-size
// ERT_CMD_CHAIN chunks. XRT 2.19 hardwires that native submit chunk size to 24,
// so the Windows MCDM shim uses the same value for now. The recovered path-B
// descriptor envelope has observed headroom up to 34 children, but 34 is not
// the default until we can prove it is compatible with XRT's intended contract.
// This is a per-native-submit chunk size, not a logical command-chain limit;
// larger logical chains are split by direct_command_buffer.c before reaching
// this layer.
constexpr size_t kWindowsDpuRunlistSubmitSize = 24;
constexpr uint64_t kWindowsDpuPathBExecBoSize = 0x1000;
// Retain the complete FLM chain working set. A 1024-child budget avoids cache
// reconstruction seen with the common 896-child default; 1280 showed no
// further throughput benefit in the validating workload.
constexpr uint32_t kWindowsChainCacheChildCommandBudget = 1024;

struct BoundBuffer {
  size_t position = 0;
  iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
  iree_device_size_t offset = 0;
  iree_device_size_t size = 0;
};

iree_string_view_t normalize_cu_name(iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    if (name.data[i] == ':') {
      return iree_make_string_view(name.data, i);
    }
  }
  return name;
}

uint64_t windows_dpu_pathb_chain_exec_bo_size() {
  return sizeof(ert_packet) + sizeof(ert_cmd_chain_data) +
         kWindowsDpuRunlistSubmitSize * sizeof(uint64_t);
}

uint32_t chain_slot_capacity(size_t exec_bo_size) {
  const size_t header = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data);
  return exec_bo_size > header
             ? static_cast<uint32_t>((exec_bo_size - header) / sizeof(uint64_t))
             : 1;
}

iree_status_t status_from_mcdm_error(const char* label,
                                      const mcdm::Error& error) {
  return iree_make_status(IREE_STATUS_INTERNAL, "%s: %s", label,
                           mcdm::ErrorMessage(&error));
}

iree_status_t validate_device_size_fits_u64(iree_device_size_t size) {
  if (IREE_UNLIKELY(size > std::numeric_limits<uint64_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native allocation size is too large");
  }
  return iree_ok_status();
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

mcdm::BufferKind to_mcdm_buffer_kind(
    iree_hal_amdxdna_native_buffer_c_type_t type) {
  switch (type) {
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY:
      return mcdm::BufferKind::host_only;
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE:
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION:
      return mcdm::BufferKind::cacheable;
  }
  return mcdm::BufferKind::host_only;
}

struct WindowsMcdmOpcodeHandler {
  iree_hal_amdxdna_native_c_command_opcode_t opcode;
  const char* name;
  uint32_t ert_opcode;
  uint32_t initial_packet_word_count = 1;
  bool is_chain = false;
  bool uses_regmap_args = false;
  bool uses_partial_elf = false;
  bool accepts_control_buffer = false;

  // XRT writes partial-ELF ERT packets through their persistent CPU mapping
  // and submits them directly. The packet is host-side command metadata, not
  // executable code; a release fence orders its stores before KMT consumes it.
  // Model data and command-aperture code retain their explicit publication.
  bool skips_exec_buffer_sync() const { return uses_partial_elf; }
};

const WindowsMcdmOpcodeHandler& windows_mcdm_opcode_handler(
    iree_hal_amdxdna_native_c_command_opcode_t opcode) {
  static constexpr WindowsMcdmOpcodeHandler kStartCu = {
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU,
      "start_cu",
      ERT_START_CU,
  };
  static constexpr WindowsMcdmOpcodeHandler kStartNpu = {
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU,
      "start_npu",
      // XRT's non-ELF DPU/TXN path submits DPU kernels as START_CU packets
      // with type ERT_CU. The NPU operation selector is arg0 in the xclbin XML
      // register map, not the ERT packet opcode.
      ERT_START_CU,
      1 + kWindowsDpuRegmapWords,
      false,
      true,
      false,
      true,
  };
  static constexpr WindowsMcdmOpcodeHandler kStartNpuPartialElf = {
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF,
      "start_npu_partial_elf",
      ERT_START_NPU,
      1 + sizeof(ert_npu_data) / sizeof(uint32_t) + 2,
      false,
      false,
      true,
      true,
  };
  static constexpr WindowsMcdmOpcodeHandler kCommandChain = {
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN,
      "command_chain",
      ERT_CMD_CHAIN,
      1,
      true,
  };
  switch (opcode) {
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU:
      return kStartCu;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU:
      return kStartNpu;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF:
      return kStartNpuPartialElf;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN:
      return kCommandChain;
  }
  return kStartCu;
}

ert_start_kernel_cmd* command_start_packet(
    iree_hal_amdxdna_native_command_t* command);

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command);

uint32_t first_set_bit(uint32_t value) {
  for (uint32_t i = 0; i < 32; ++i) {
    if (value & (uint32_t{1} << i)) return i;
  }
  return 0;
}

size_t align_up_size(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

bool try_align_up_size(size_t value, size_t alignment, size_t* out_value) {
  if (!out_value || alignment == 0 ||
      (alignment & (alignment - 1)) != 0 ||
      value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return false;
  }
  *out_value = align_up_size(value, alignment);
  return true;
}

iree_status_t from_c_command_opcode(
    iree_hal_amdxdna_native_c_command_opcode_t opcode,
    iree_hal_amdxdna_native_c_command_opcode_t* out_opcode) {
  switch (opcode) {
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU:
      *out_opcode = IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU:
      *out_opcode = IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF:
      *out_opcode =
          IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN:
      *out_opcode = IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown amdxdna native command opcode");
}

iree_status_t from_c_context_image_type(
    iree_hal_amdxdna_native_c_context_image_type_t type,
    iree_hal_amdxdna_native_c_context_image_type_t* out_type) {
  switch (type) {
    case IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_PDI:
      *out_type = IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_PDI;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_XCLBIN:
      *out_type = IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_XCLBIN;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown amdxdna native context image type");
}

iree_status_t to_native_sync_direction(
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_hal_amdxdna_native_buffer_sync_direction_t* out_direction) {
  switch (direction) {
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE:
      *out_direction = IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST:
      *out_direction = IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown amdxdna native buffer sync direction");
  }
}

iree_status_t to_native_buffer_type(
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_c_type_t* out_type) {
  switch (type) {
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY:
      *out_type = IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE:
      *out_type = IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION:
      *out_type = IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown amdxdna native buffer type");
  }
}

void close_mcdm_adapter_handle(const mcdm::KmtApi& api,
                               mcdm::Adapter* adapter) {
  if (!adapter || !adapter->handle || !api.close_adapter) return;
  D3DKMT_CLOSEADAPTER close = {};
  close.hAdapter = adapter->handle;
  api.close_adapter(&close);
  adapter->handle = 0;
}

}  // namespace

struct iree_hal_amdxdna_native_device_t {
  iree_allocator_t host_allocator;
  mcdm::KmtApi api;
  mcdm::Device device;
  bool pathb_context_ready = false;
  std::mutex pathb_context_mutex;
  std::condition_variable pathb_context_cv;
  size_t pathb_active_submission_count = 0;
  iree_hal_amdxdna_native_context_t* pathb_active_context = nullptr;
};

struct iree_hal_amdxdna_native_buffer_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  mcdm::Buffer buffer;
  iree_hal_amdxdna_native_buffer_c_type_t type =
      IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY;
  bool deferred = false;
  uint8_t* deferred_storage = nullptr;
  iree_host_size_t deferred_storage_size = 0;
  // D3DKMT Lock2 mappings may be replaced after a Path-B submit. Keep the
  // public host mapping stable for HOST_ONLY buffers and copy through the
  // current native mapping at explicit synchronization points.
  uint8_t* host_mirror = nullptr;
  // Eager host allocations expose their KMT Lock2 mapping directly for the
  // allocation lifetime, matching the persistent map contract.
  uint8_t* direct_host_mapping = nullptr;
  bool native_mapping_stale = false;
};

struct iree_hal_amdxdna_native_queue_t {
  iree_hal_amdxdna_native_context_t* context = nullptr;
  uint64_t exec_command_count = 0;
};

struct PathBActiveCodeRange {
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct iree_hal_amdxdna_native_context_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  mcdm::Context context;
  mcdm::CommandAperture command_aperture;
  bool has_command_aperture = false;
  uint64_t pathb_persistent_code_bytes = 0;
  uint64_t pathb_persistent_code_slot_size = 0;
  std::vector<uint8_t> pathb_persistent_code_slots_in_use;
  std::vector<iree_hal_amdxdna_native_command_t*>
      pathb_persistent_code_commands;
  uint64_t pathb_chain_aperture_generation = 1;
  size_t pathb_chain_code_cursor = 0;
  size_t pathb_chain_descriptor_cursor = 0;
  iree_device_size_t pathb_single_code_staged_size = 0;
  uint64_t pathb_single_code_staged_offset = 0;
  std::vector<PathBActiveCodeRange> pathb_active_single_code_ranges;
  std::vector<uint8_t> pathb_completion_slots_in_use;
  size_t pathb_next_completion_slot = 0;
  mcdm::ContextBlobInfo info;
  iree_hal_amdxdna_native_queue_t queue;
};

struct iree_hal_amdxdna_native_command_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  iree_hal_amdxdna_native_c_command_opcode_t opcode;
  iree_hal_amdxdna_native_buffer_t* exec_buffer = nullptr;
  iree_hal_amdxdna_native_buffer_t* control_buffer = nullptr;
  iree_device_size_t control_buffer_size = 0;
  ert_start_kernel_cmd* start_packet = nullptr;
  size_t command_size = 0;
  uint32_t cached_start_header = 0;
  bool cached_start_header_valid = false;
  uint32_t reg_idx = 0;
  uint32_t arg_count = 0;
  bool windows_dpu_regmap_finalized = false;
  bool pathb_code_staged = false;
  iree_device_size_t pathb_code_staged_size = 0;
  uint64_t pathb_single_code_aperture_offset = 0;
  uint64_t pathb_single_code_aperture_capacity = 0;
  iree_hal_amdxdna_native_context_t* pathb_single_code_owner_context = nullptr;
  size_t pathb_single_code_first_slot = 0;
  size_t pathb_single_code_slot_count = 0;
  uint64_t pathb_chain_descriptor_gpu_va = 0;
  uint32_t pathb_chain_descriptor_bytes = 0;
  uint32_t pathb_chain_first_child_opcode = 0;
  uint64_t pathb_chain_code_used_size = 0;
  uint64_t pathb_chain_code_aperture_offset = 0;
  uint64_t pathb_chain_descriptor_aperture_offset = 0;
  uint64_t pathb_chain_aperture_generation = 0;
  bool pathb_chain_prepared_valid = false;
  // CPU-restage flags only. They must not gate device-image installation or
  // opcode-9. Compact Path-B firmware does not retain aperture GPU contents
  // across consumes, so copy+Commit is required before every state-3/chain
  // submit even when the CPU source image is unchanged. Opcode-9 then
  // publishes those slots.
  bool pathb_chain_code_dirty = false;
  bool pathb_chain_descriptor_dirty = false;
  bool pathb_chain_bound_residency_checked = false;
  size_t* pathb_chain_child_code_offsets = nullptr;
  size_t pathb_chain_child_code_offset_count = 0;
  size_t pathb_chain_child_code_offset_capacity = 0;
  iree_hal_amdxdna_native_command_t** chain_children = nullptr;
  size_t chain_child_count = 0;
  size_t chain_child_capacity = 0;
  BoundBuffer* bound_buffers = nullptr;
  size_t bound_buffer_count = 0;
  size_t bound_buffer_capacity = 0;
};

bool iree_hal_amdxdna_native_windows_reserve_code_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t requested_count,
    size_t* out_first_slot) {
  if (!slots_in_use || !out_first_slot || requested_count == 0 ||
      requested_count > slot_capacity) {
    return false;
  }
  for (size_t first = 0; first <= slot_capacity - requested_count; ++first) {
    size_t count = 0;
    while (count < requested_count && !slots_in_use[first + count]) ++count;
    if (count != requested_count) {
      first += count;
      continue;
    }
    std::fill_n(slots_in_use + first, requested_count, uint8_t{1});
    *out_first_slot = first;
    return true;
  }
  return false;
}

bool iree_hal_amdxdna_native_windows_release_code_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t first_slot,
    size_t slot_count) {
  if (!slots_in_use || slot_count == 0 || first_slot > slot_capacity ||
      slot_count > slot_capacity - first_slot) {
    return false;
  }
  for (size_t i = 0; i < slot_count; ++i) {
    if (!slots_in_use[first_slot + i]) return false;
  }
  std::fill_n(slots_in_use + first_slot, slot_count, uint8_t{0});
  return true;
}

size_t iree_hal_amdxdna_native_windows_code_slot_high_watermark(
    const uint8_t* slots_in_use, size_t slot_capacity) {
  if (!slots_in_use) return 0;
  while (slot_capacity > 0 && !slots_in_use[slot_capacity - 1]) {
    --slot_capacity;
  }
  return slot_capacity;
}

void release_command_persistent_code_slots_locked(
    iree_hal_amdxdna_native_command_t* command) {
  iree_hal_amdxdna_native_context_t* context =
      command ? command->pathb_single_code_owner_context : nullptr;
  if (!context) return;
  const bool released = iree_hal_amdxdna_native_windows_release_code_slots(
      context->pathb_persistent_code_slots_in_use.data(),
      context->pathb_persistent_code_slots_in_use.size(),
      command->pathb_single_code_first_slot,
      command->pathb_single_code_slot_count);
  IREE_ASSERT(released);
  auto& commands = context->pathb_persistent_code_commands;
  const auto command_it = std::find(commands.begin(), commands.end(), command);
  IREE_ASSERT(command_it != commands.end());
  if (command_it != commands.end()) commands.erase(command_it);
  context->pathb_persistent_code_bytes =
      iree_hal_amdxdna_native_windows_code_slot_high_watermark(
          context->pathb_persistent_code_slots_in_use.data(),
          context->pathb_persistent_code_slots_in_use.size()) *
      context->pathb_persistent_code_slot_size;
  command->pathb_single_code_owner_context = nullptr;
  command->pathb_single_code_first_slot = 0;
  command->pathb_single_code_slot_count = 0;
  command->pathb_single_code_aperture_offset = 0;
  command->pathb_single_code_aperture_capacity = 0;
}

void release_command_persistent_code_slots(
    iree_hal_amdxdna_native_command_t* command) {
  if (!command) return;
  std::lock_guard<std::mutex> lock(command->device->pathb_context_mutex);
  release_command_persistent_code_slots_locked(command);
}

iree_status_t reserve_command_persistent_code_slots(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, uint64_t required_capacity) {
  iree_hal_amdxdna_native_context_t* context = queue->context;
  std::lock_guard<std::mutex> lock(command->device->pathb_context_mutex);
  if (command->pathb_single_code_owner_context == context &&
      command->pathb_single_code_aperture_capacity >= required_capacity) {
    return iree_ok_status();
  }
  release_command_persistent_code_slots_locked(command);
  const uint64_t slot_size = context->pathb_persistent_code_slot_size;
  if (!slot_size || required_capacity > UINT64_MAX - (slot_size - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna Windows MCDM command slot size overflow");
  }
  const uint64_t requested_count_u64 =
      (required_capacity + slot_size - 1) / slot_size;
  if (requested_count_u64 > std::numeric_limits<size_t>::max()) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna Windows MCDM command slot count overflow");
  }
  const size_t requested_count = static_cast<size_t>(requested_count_u64);
  size_t first_slot = 0;
  if (!iree_hal_amdxdna_native_windows_reserve_code_slots(
          context->pathb_persistent_code_slots_in_use.data(),
          context->pathb_persistent_code_slots_in_use.size(), requested_count,
          &first_slot)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM persistent command code exceeds its aperture "
        "region");
  }
  command->pathb_single_code_owner_context = context;
  command->pathb_single_code_first_slot = first_slot;
  command->pathb_single_code_slot_count = requested_count;
  command->pathb_single_code_aperture_offset =
      context->command_aperture.code_offset + first_slot * slot_size;
  command->pathb_single_code_aperture_capacity = requested_count * slot_size;
  context->pathb_persistent_code_commands.push_back(command);
  context->pathb_persistent_code_bytes =
      iree_hal_amdxdna_native_windows_code_slot_high_watermark(
          context->pathb_persistent_code_slots_in_use.data(),
          context->pathb_persistent_code_slots_in_use.size()) *
      slot_size;
  return iree_ok_status();
}

iree_status_t materialize_deferred_instruction_buffer(
    iree_hal_amdxdna_native_context_t* context,
    iree_hal_amdxdna_native_buffer_t* buffer);
iree_status_t materialize_deferred_buffer(
    iree_hal_amdxdna_native_buffer_t* buffer);
void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer);
iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_device_size_t size, iree_device_size_t offset);
iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction);
uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer);
iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer);
void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command);
iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset);
iree_status_t iree_hal_amdxdna_native_device_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps);

iree_status_t initialize_deferred_buffer_storage(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer || !buffer->deferred) return iree_ok_status();
  if (buffer->buffer.size >
      static_cast<uint64_t>(std::numeric_limits<iree_host_size_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna deferred native allocation is too large");
  }
  const iree_host_size_t storage_size =
      static_cast<iree_host_size_t>(buffer->buffer.size);
  if (storage_size == 0) {
    buffer->buffer.cpu_ptr = nullptr;
    return iree_ok_status();
  }
  void* storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(buffer->device->host_allocator,
                                             storage_size, &storage));
  std::memset(storage, 0, storage_size);
  buffer->deferred_storage = static_cast<uint8_t*>(storage);
  buffer->deferred_storage_size = storage_size;
  buffer->buffer.cpu_ptr = buffer->deferred_storage;
  if (buffer->type == IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY) {
    buffer->host_mirror = buffer->deferred_storage;
  }
  return iree_ok_status();
}

void release_deferred_buffer_storage(iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer || !buffer->deferred_storage) return;
  iree_allocator_free(buffer->device->host_allocator, buffer->deferred_storage);
  buffer->deferred_storage = nullptr;
  buffer->deferred_storage_size = 0;
  if (buffer->deferred) buffer->buffer.cpu_ptr = nullptr;
}

iree_status_t allocate_native_buffer(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(device->host_allocator, sizeof(**out_buffer),
                            reinterpret_cast<void**>(out_buffer)));
  new (*out_buffer) iree_hal_amdxdna_native_buffer_t();
  (*out_buffer)->device = device;
  (*out_buffer)->type = IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY;
  return iree_ok_status();
}

iree_status_t create_native_buffer_from_mcdm(
    iree_hal_amdxdna_native_device_t* device, mcdm::Buffer buffer,
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  IREE_RETURN_IF_ERROR(allocate_native_buffer(device, out_buffer));
  (*out_buffer)->buffer = buffer;
  (*out_buffer)->type = type;
  return iree_ok_status();
}

iree_status_t ensure_host_buffer_mirror(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer ||
      buffer->type != IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY ||
      buffer->host_mirror) {
    return iree_ok_status();
  }
  if (buffer->buffer.size >
      static_cast<uint64_t>(std::numeric_limits<iree_host_size_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna Windows MCDM host mirror is too large");
  }
  const iree_host_size_t size =
      static_cast<iree_host_size_t>(buffer->buffer.size);
  if (size == 0) return iree_ok_status();
  void* storage = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(buffer->device->host_allocator, size, &storage));
  buffer->host_mirror = static_cast<uint8_t*>(storage);
  if (buffer->buffer.cpu_ptr) {
    std::memcpy(buffer->host_mirror, buffer->buffer.cpu_ptr, size);
  } else {
    std::memset(buffer->host_mirror, 0, size);
  }
  return iree_ok_status();
}

iree_status_t create_deferred_native_buffer(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_buffer_c_type_t type, uint64_t size,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  IREE_RETURN_IF_ERROR(allocate_native_buffer(device, out_buffer));
  (*out_buffer)->type = type;
  (*out_buffer)->deferred = true;
  (*out_buffer)->buffer.kind = to_mcdm_buffer_kind(type);
  (*out_buffer)->buffer.size = size;
  iree_status_t status = initialize_deferred_buffer_storage(*out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_buffer_destroy(*out_buffer);
    *out_buffer = nullptr;
    return status;
  }
  return iree_ok_status();
}

iree_status_t create_deferred_native_buffer_with_kind(
    iree_hal_amdxdna_native_device_t* device, mcdm::BufferKind kind,
    uint64_t size, iree_hal_amdxdna_native_buffer_t** out_buffer) {
  IREE_RETURN_IF_ERROR(allocate_native_buffer(device, out_buffer));
  (*out_buffer)->type = IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE;
  (*out_buffer)->deferred = true;
  (*out_buffer)->buffer.kind = kind;
  (*out_buffer)->buffer.size = size;
  iree_status_t status = initialize_deferred_buffer_storage(*out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_buffer_destroy(*out_buffer);
    *out_buffer = nullptr;
    return status;
  }
  return iree_ok_status();
}

namespace {

const WindowsMcdmOpcodeHandler& command_opcode_handler(
    const iree_hal_amdxdna_native_command_t* command) {
  return windows_mcdm_opcode_handler(command->opcode);
}

bool command_is_pathb_chain(const iree_hal_amdxdna_native_command_t* command) {
  return command && command_opcode_handler(command).is_chain;
}

ert_start_kernel_cmd* command_start_packet(
    iree_hal_amdxdna_native_command_t* command) {
  return command->start_packet;
}

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command) {
  return reinterpret_cast<ert_packet*>(command_start_packet(command));
}

void reset_command_packet_for_start(
    iree_hal_amdxdna_native_command_t* command) {
  ert_packet* packet = command_packet(command);
  if (!command->cached_start_header_valid) {
    packet->state = ERT_CMD_STATE_NEW;
    command->cached_start_header = packet->header;
    command->cached_start_header_valid = true;
  }
  packet->header = command->cached_start_header;
  packet->state = ERT_CMD_STATE_NEW;
}

iree_status_t close_pathb_single_code_ranges(
    iree_hal_amdxdna_native_queue_t* queue) {
  if (!queue || !queue->context) {
    return iree_ok_status();
  }
  auto& ranges = queue->context->pathb_active_single_code_ranges;
  while (!ranges.empty()) {
    const PathBActiveCodeRange range = ranges.back();
    mcdm::Error error;
    if (!mcdm::ReleasePathBCodeRange(
            queue->context->device->api, queue->context->device->device,
            &queue->context->context, queue->context->command_aperture,
            range.offset, range.size, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb single-session code release failed",
          error);
    }
    ranges.pop_back();
  }
  return iree_ok_status();
}

iree_status_t close_pathb_single_code_range(
    iree_hal_amdxdna_native_queue_t* queue, uint64_t offset) {
  if (!queue || !queue->context) return iree_ok_status();
  auto& ranges = queue->context->pathb_active_single_code_ranges;
  const auto it = std::find_if(
      ranges.begin(), ranges.end(), [=](const PathBActiveCodeRange& range) {
        return range.offset == offset;
      });
  if (it == ranges.end()) return iree_ok_status();
  mcdm::Error error;
  if (!mcdm::ReleasePathBCodeRange(
          queue->context->device->api, queue->context->device->device,
          &queue->context->context, queue->context->command_aperture, offset,
          it->size, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb single code release failed", error);
  }
  ranges.erase(it);
  return iree_ok_status();
}

void reset_pathb_context_cached_aperture_state(
    iree_hal_amdxdna_native_context_t* context) {
  if (!context) return;
  context->pathb_single_code_staged_size = 0;
  context->pathb_single_code_staged_offset = 0;
  context->pathb_active_single_code_ranges.clear();
}

mcdm::CommandAperture& pathb_chain_aperture(
    iree_hal_amdxdna_native_queue_t* queue) {
  return queue->context->command_aperture;
}

iree_status_t retire_pathb_active_context_locked(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_context_t* next_context) {
  IREE_ASSERT_ARGUMENT(device);
  iree_hal_amdxdna_native_context_t* active = device->pathb_active_context;
  if (!active || active == next_context) return iree_ok_status();
  IREE_RETURN_IF_ERROR(close_pathb_single_code_ranges(&active->queue));
  reset_pathb_context_cached_aperture_state(active);
  if (active->has_command_aperture) {
    mcdm::Error error;
    if (!mcdm::ReleaseCommandApertureGpuMapping(
            active->device->api, active->device->device,
            &active->command_aperture, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM command aperture release failed", error);
    }
  }
  device->pathb_active_context = nullptr;
  return iree_ok_status();
}

iree_status_t activate_pathb_context_for_submit_locked(
    iree_hal_amdxdna_native_queue_t* queue) {
  if (!queue || !queue->context) return iree_ok_status();
  iree_hal_amdxdna_native_context_t* context = queue->context;
  iree_hal_amdxdna_native_device_t* device = context->device;
  if (device->pathb_active_context == context) return iree_ok_status();
  IREE_RETURN_IF_ERROR(retire_pathb_active_context_locked(device, context));
  reset_pathb_context_cached_aperture_state(context);
  if (context->has_command_aperture) {
    mcdm::Error error;
    if (!mcdm::EnsureCommandApertureGpuMapping(
            device->api, device->device, &context->command_aperture, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM command aperture activation failed", error);
    }
  }
  device->pathb_active_context = context;
  return iree_ok_status();
}

iree_status_t ensure_pathb_single_code_range_active(
    iree_hal_amdxdna_native_queue_t* queue, uint64_t offset, uint64_t size,
    bool* out_activated) {
  if (out_activated) *out_activated = false;
  if (!queue || !queue->context) return iree_ok_status();
  auto& ranges = queue->context->pathb_active_single_code_ranges;
  const bool active =
      std::any_of(ranges.begin(), ranges.end(),
                  [=](const PathBActiveCodeRange& range) {
                    return range.offset == offset && range.size == size;
                  });
  if (active) return iree_ok_status();
  mcdm::Error error;
  const mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (!mcdm::AcquirePathBCodeRange(
          queue->context->device->api, queue->context->device->device,
          &queue->context->context, aperture, offset, size, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb single-session code acquire failed", error);
  }
  ranges.push_back({offset, size});
  if (out_activated) *out_activated = true;
  return iree_ok_status();
}

iree_status_t acquire_pathb_code_range(iree_hal_amdxdna_native_queue_t* queue,
                                       uint64_t mapping_offset,
                                       uint64_t code_size) {
  if (!queue || !queue->context || code_size == 0) return iree_ok_status();
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  mcdm::Error error;
  if (!mcdm::AcquirePathBCodeRange(queue->context->device->api,
                                   queue->context->device->device,
                                   &queue->context->context, aperture,
                                   mapping_offset, code_size, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb code range acquire failed", error);
  }
  return iree_ok_status();
}

iree_status_t publish_pathb_code_write(iree_hal_amdxdna_native_queue_t* queue,
                                       uint64_t mapping_offset,
                                       uint64_t code_size) {
  if (!queue || !queue->context || code_size == 0) return iree_ok_status();
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  mcdm::Error error;
  if (!mcdm::PublishPathBCodeWrite(queue->context->device->api,
                                   queue->context->device->device,
                                   &queue->context->context, aperture,
                                   mapping_offset, code_size, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb code write publish failed", error);
  }
  return iree_ok_status();
}

iree_status_t stage_windows_dpu_code_buffer(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.gpu_cpu_ptr || !aperture.gpu_va ||
                    aperture.gpu_va_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires the aperture code BO");
  }
  if (IREE_UNLIKELY(!command->control_buffer ||
                    !command->control_buffer->buffer.cpu_ptr ||
                    command->control_buffer_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU command has no control-code buffer");
  }
  if (IREE_UNLIKELY(command->control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM DPU control-code size is not word aligned");
  }
  const bool control_is_deferred_pathb_instruction =
      command->device &&
      command->control_buffer->type ==
          IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION &&
      command->control_buffer->deferred;
  if (!control_is_deferred_pathb_instruction) {
    IREE_RETURN_IF_ERROR(materialize_deferred_instruction_buffer(
        queue->context, command->control_buffer));
  }
  const bool is_partial_elf = command_opcode_handler(command).uses_partial_elf;
  const uint64_t slot_size =
      mcdm::GetMcdmAbiInfo(command->device->device.mcdm_abi)
          .command_aperture_code_slot_size;
  if (IREE_UNLIKELY(!slot_size || (slot_size & (slot_size - 1)) != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM command aperture has invalid code slot size");
  }
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM command aperture code mapping is unavailable");
  }
  const bool uses_shared_command_code_view =
      mcdm::GetMcdmSubmissionPolicy(command->device->device.mcdm_abi)
          .uses_shared_command_code_view;
  if (uses_shared_command_code_view) {
    release_command_persistent_code_slots(command);
    command->pathb_single_code_aperture_offset = aperture.code_offset;
    command->pathb_single_code_aperture_capacity = aperture.code_size;
  } else {
    const bool retains_slot =
        command->pathb_single_code_owner_context == queue->context &&
        command->pathb_single_code_aperture_capacity >=
            command->control_buffer_size;
    IREE_RETURN_IF_ERROR(reserve_command_persistent_code_slots(
        queue, command, command->control_buffer_size));
    if (!retains_slot) {
      command->pathb_code_staged = false;
      command->pathb_code_staged_size = 0;
    }
  }
  const uint64_t code_offset = command->pathb_single_code_aperture_offset;
  const uint64_t relative_code_offset = code_offset - aperture.code_offset;
  const uint64_t slot_capacity = command->pathb_single_code_aperture_capacity;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    relative_code_offset > aperture.code_size ||
                    slot_capacity >
                        aperture.code_size - relative_code_offset)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM DPU control code exceeds its aperture region");
  }
  uint8_t* const code_cpu_ptr =
      static_cast<uint8_t*>(aperture.code_cpu_ptr) + relative_code_offset;
  const uint64_t code_gpu_va = aperture.code_gpu_va + relative_code_offset;
  auto set_partial_elf_instruction_fields = [&]() -> iree_status_t {
    if (!is_partial_elf) {
      return iree_ok_status();
    }
    ert_npu_data* npu_data = get_ert_npu_data(command->start_packet);
    if (IREE_UNLIKELY(!npu_data)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no NPU data");
    }
    npu_data->instruction_buffer = code_gpu_va;
    npu_data->instruction_buffer_size =
        static_cast<uint32_t>(command->control_buffer_size);
    npu_data->instruction_prop_count = 0;
    return iree_ok_status();
  };
  const bool same_staged_code =
      command->pathb_code_staged &&
      command->pathb_code_staged_size == command->control_buffer_size &&
      (!uses_shared_command_code_view ||
       (queue->context->pathb_single_code_staged_size ==
            command->control_buffer_size &&
        queue->context->pathb_single_code_staged_offset == code_offset &&
        std::memcmp(code_cpu_ptr, command->control_buffer->buffer.cpu_ptr,
                    static_cast<size_t>(command->control_buffer_size)) == 0));
  if (same_staged_code) {
    if (is_partial_elf) {
      IREE_RETURN_IF_ERROR(ensure_pathb_single_code_range_active(
          queue, code_offset, command->control_buffer_size,
          /*out_activated=*/nullptr));
      IREE_RETURN_IF_ERROR(set_partial_elf_instruction_fields());
    }
  } else {
    IREE_RETURN_IF_ERROR(close_pathb_single_code_range(queue, code_offset));
    if (is_partial_elf) {
      command->pathb_code_staged = false;
      command->pathb_code_staged_size = 0;
      IREE_RETURN_IF_ERROR(set_partial_elf_instruction_fields());
      IREE_RETURN_IF_ERROR(ensure_pathb_single_code_range_active(
          queue, code_offset, command->control_buffer_size,
          /*out_activated=*/nullptr));
    } else {
      IREE_RETURN_IF_ERROR(acquire_pathb_code_range(
          queue, code_offset, command->control_buffer_size));
    }
  }
  {
    // Restage from the CPU source of truth on every consume. same_staged_code
    // means the host image is unchanged, not that the previous device image
    // is still resident. Opcode-9 publishes slots; it does not copy bytes.
    if (command->pathb_code_staged &&
        command->pathb_code_staged_size > command->control_buffer_size) {
      const size_t stale_tail_offset =
          static_cast<size_t>(command->control_buffer_size);
      const size_t stale_tail_size = static_cast<size_t>(
          command->pathb_code_staged_size - command->control_buffer_size);
      std::memset(code_cpu_ptr + stale_tail_offset, 0, stale_tail_size);
    }
    std::memcpy(code_cpu_ptr, command->control_buffer->buffer.cpu_ptr,
                static_cast<size_t>(command->control_buffer_size));
  }
  {
    mcdm::Error error;
    if (!mcdm::CommitPathBCodeWrite(
            command->device->api, command->device->device, aperture,
            code_offset, static_cast<uint64_t>(command->control_buffer_size),
            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B single aperture code commit failed",
          error);
    }
    // Compact Commit is CPU clflush. That flush does not snoop the NPU cache,
    // so invalidate the KMT allocation before opcode-9.
    if (!mcdm::SyncCommandApertureCode(
            command->device->api, command->device->device, aperture,
            code_offset, static_cast<uint64_t>(command->control_buffer_size),
            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B single aperture KMT invalidate failed",
          error);
    }
  }
  if (is_partial_elf) {
    mcdm::Error error;
    if (!mcdm::RefreshPathBCodeMappingAfterWrite(
            command->device->api, command->device->device, &aperture, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B single aperture remap failed", error);
    }
  }
  IREE_RETURN_IF_ERROR(publish_pathb_code_write(queue, code_offset,
                                                command->control_buffer_size));
  // Keep the freshly staged control code resident in its command-owned slot.
  // Cached commands can reuse non-overlapping slots until a runlist or context
  // transition releases the active set.
  command->pathb_code_staged = true;
  command->pathb_code_staged_size = command->control_buffer_size;
  queue->context->pathb_single_code_staged_size = command->control_buffer_size;
  queue->context->pathb_single_code_staged_offset = code_offset;
  return is_partial_elf ? iree_ok_status()
                        : set_partial_elf_instruction_fields();
}

iree_status_t check_pkt_count_capacity(
    iree_hal_amdxdna_native_command_t* command, uint32_t bytes) {
  if (!command || !command->start_packet) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native command is not initialized");
  }
  uint32_t next_count = command->start_packet->count + bytes / sizeof(uint32_t);
  if (command->command_size <
      sizeof(command->start_packet->header) +
          static_cast<size_t>(next_count) * sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna native command packet is full");
  }
  return iree_ok_status();
}

iree_status_t inc_pkt_count(iree_hal_amdxdna_native_command_t* command,
                            uint32_t bytes) {
  IREE_RETURN_IF_ERROR(check_pkt_count_capacity(command, bytes));
  command->start_packet->count += bytes / sizeof(uint32_t);
  return iree_ok_status();
}

iree_status_t reserve_bound_buffers(iree_hal_amdxdna_native_command_t* command,
                                    size_t capacity) {
  if (capacity <= command->bound_buffer_capacity) return iree_ok_status();
  size_t new_capacity =
      command->bound_buffer_capacity ? command->bound_buffer_capacity * 2 : 4;
  while (new_capacity < capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      command->device->host_allocator, new_capacity,
      sizeof(*command->bound_buffers), (void**)&command->bound_buffers));
  command->bound_buffer_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t reserve_chain_children(iree_hal_amdxdna_native_command_t* command,
                                     size_t capacity) {
  if (capacity <= command->chain_child_capacity) return iree_ok_status();
  size_t new_capacity =
      command->chain_child_capacity ? command->chain_child_capacity * 2 : 4;
  while (new_capacity < capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      command->device->host_allocator, new_capacity,
      sizeof(*command->chain_children), (void**)&command->chain_children));
  command->chain_child_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t reserve_child_code_offsets(
    iree_hal_amdxdna_native_command_t* command, size_t capacity) {
  if (capacity <= command->pathb_chain_child_code_offset_capacity) {
    return iree_ok_status();
  }
  size_t new_capacity =
      command->pathb_chain_child_code_offset_capacity
          ? command->pathb_chain_child_code_offset_capacity * 2
          : 4;
  while (new_capacity < capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      command->device->host_allocator, new_capacity,
      sizeof(*command->pathb_chain_child_code_offsets),
      (void**)&command->pathb_chain_child_code_offsets));
  command->pathb_chain_child_code_offset_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t bind_buffer_ref(iree_hal_amdxdna_native_command_t* command,
                              size_t position,
                              iree_hal_amdxdna_native_buffer_t* buffer,
                              iree_device_size_t offset,
                              iree_device_size_t size) {
  if (position == 0 && !command_is_pathb_chain(command)) {
    command->bound_buffer_count = 0;
  }
  IREE_RETURN_IF_ERROR(
      reserve_bound_buffers(command, command->bound_buffer_count + 1));
  command->bound_buffers[command->bound_buffer_count++] =
      BoundBuffer{position, buffer, offset, size};
  command->pathb_chain_bound_residency_checked = false;
  return iree_ok_status();
}

bool is_pathb_partial_elf_control_binding(
    iree_hal_amdxdna_native_command_t* command, const BoundBuffer& bound) {
  return command && command->device &&
         command_opcode_handler(command).uses_partial_elf &&
         bound.position == 0 && bound.buffer == command->control_buffer;
}

bool uses_windows_dpu_regmap(iree_hal_amdxdna_native_command_t* command) {
  return command && command_opcode_handler(command).uses_regmap_args;
}

bool uses_partial_elf_npu_packet(iree_hal_amdxdna_native_command_t* command) {
  return command && command_opcode_handler(command).uses_partial_elf;
}

constexpr uint8_t kTxnOpWrite32 = 0;
constexpr uint8_t kTxnOpBlockWrite = 1;
constexpr uint8_t kTxnOpDdrPatch = 129;

uint32_t read_txn_u32(const uint8_t* p) {
  uint32_t value = 0;
  std::memcpy(&value, p, sizeof(value));
  return value;
}

bool get_partial_elf_txn_view(iree_hal_amdxdna_native_command_t* command,
                              const uint8_t** out_bytes, size_t* out_total,
                              uint32_t* out_op_count) {
  if (!command || !command->control_buffer ||
      !command->control_buffer->buffer.cpu_ptr ||
      command->control_buffer_size < 4 * sizeof(uint32_t)) {
    return false;
  }
  const uint8_t* bytes =
      static_cast<const uint8_t*>(command->control_buffer->buffer.cpu_ptr);
  const uint32_t declared_bytes = read_txn_u32(bytes + 12);
  if (declared_bytes < 4 * sizeof(uint32_t) ||
      declared_bytes > command->control_buffer_size) {
    return false;
  }
  *out_bytes = bytes;
  *out_total = declared_bytes;
  *out_op_count = read_txn_u32(bytes + 8);
  return true;
}

uint32_t txn_bd_key(uint32_t location, uint32_t bd_id) {
  const uint32_t col = (location >> 25) & 0x7f;
  const uint32_t row = (location >> 20) & 0x1f;
  return (col << 16) | (row << 8) | bd_id;
}

bool find_partial_elf_bd_ops(const uint8_t* bytes, size_t total,
                             uint32_t op_count, size_t queue_offset,
                             uint32_t key, const uint8_t** out_dma,
                             const uint8_t** out_ddr) {
  *out_dma = nullptr;
  *out_ddr = nullptr;
  size_t offset = 4 * sizeof(uint32_t);
  // A transaction may reprogram the same BD ID multiple times. Resolve each
  // queue push against the most recent programming that precedes it, matching
  // the order in which the transaction executes.
  for (uint32_t i = 0; i < op_count && offset < queue_offset; ++i) {
    const uint32_t op_size =
        iree_hal_amdxdna_txn_op_size(bytes, total, offset);
    if (op_size == 0 || (op_size & 3u) != 0 || offset > total ||
        op_size > total - offset) {
      return false;
    }
    const uint8_t op = bytes[offset];
    if (op == kTxnOpBlockWrite && op_size >= 12 * sizeof(uint32_t)) {
      const uint32_t location = read_txn_u32(bytes + offset + 8);
      const uint32_t bd_id = (location >> 5) & 0xf;
      if (txn_bd_key(location, bd_id) == key) *out_dma = bytes + offset;
    } else if (op == kTxnOpDdrPatch && op_size >= 12 * sizeof(uint32_t)) {
      const uint32_t location = read_txn_u32(bytes + offset + 24);
      const uint32_t bd_id = ((location - 4) >> 5) & 0x1f;
      if (txn_bd_key(location, bd_id) == key) *out_ddr = bytes + offset;
    }
    offset += op_size;
  }
  return true;
}

uint64_t partial_elf_dma_span_words(const uint8_t* dma) {
  auto saturating_add = [](uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs
               ? std::numeric_limits<uint64_t>::max()
               : lhs + rhs;
  };
  auto saturating_mul = [](uint64_t lhs, uint64_t rhs) {
    return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs
               ? std::numeric_limits<uint64_t>::max()
               : lhs * rhs;
  };
  const uint64_t buffer_length = read_txn_u32(dma + 16);
  uint64_t span = buffer_length;
  const uint32_t dim0 = read_txn_u32(dma + 28);
  if (dim0 != 0) {
    const uint64_t dim0_size = (dim0 >> 20) & 0x3ff;
    const uint64_t dim0_stride = (dim0 & 0xfffff) + 1;
    const uint32_t dim1 = read_txn_u32(dma + 32);
    const uint64_t dim1_size = (dim1 >> 20) & 0x3ff;
    const uint64_t dim1_stride = (dim1 & 0xfffff) + 1;
    const uint32_t dim2 = read_txn_u32(dma + 36);
    const uint64_t dim2_size =
        dim0_size && dim1_size ? buffer_length / (dim0_size * dim1_size) : 0;
    const uint64_t dim2_stride = (dim2 & 0xfffff) + 1;
    uint64_t strided_span = 1;
    if (dim0_size > 1) {
      strided_span = saturating_add(
          strided_span, saturating_mul(dim0_size - 1, dim0_stride));
    }
    if (dim1_size > 1) {
      strided_span = saturating_add(
          strided_span, saturating_mul(dim1_size - 1, dim1_stride));
    }
    if (dim2_size > 1) {
      strided_span = saturating_add(
          strided_span, saturating_mul(dim2_size - 1, dim2_stride));
    }
    span = std::max(span, strided_span);
  }
  const uint32_t iter = read_txn_u32(dma + 40);
  const uint64_t iter_size = ((iter >> 20) & 0x3ff) + 1;
  const uint64_t iter_stride = (iter & 0xfffff) + 1;
  if (iter_size > 1) {
    span = saturating_add(span,
                          saturating_mul(iter_size - 1, iter_stride));
  }
  return span;
}

const BoundBuffer* find_bound_buffer_by_position(
    const iree_hal_amdxdna_native_command_t* command, size_t position) {
  for (size_t i = 0; i < command->bound_buffer_count; ++i) {
    if (command->bound_buffers[i].position == position) {
      return &command->bound_buffers[i];
    }
  }
  return nullptr;
}

using CommandOutputRange = iree_hal_amdxdna_native_windows_buffer_range_t;

void collect_all_runtime_bindings(
    iree_hal_amdxdna_native_command_t* command,
    std::vector<CommandOutputRange>* output_ranges) {
  for (size_t i = 0; i < command->bound_buffer_count; ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (!bound.buffer || is_pathb_partial_elf_control_binding(command, bound)) {
      continue;
    }
    output_ranges->push_back({bound.buffer, bound.offset, bound.size});
  }
}

void collect_partial_elf_output_ranges(
    iree_hal_amdxdna_native_command_t* command,
    std::vector<CommandOutputRange>* output_ranges) {
  if (!uses_partial_elf_npu_packet(command)) return;
  const size_t range_base = output_ranges->size();
  auto collect_fallback = [&]() {
    output_ranges->resize(range_base);
    collect_all_runtime_bindings(command, output_ranges);
  };
  const uint8_t* bytes = nullptr;
  size_t total = 0;
  uint32_t op_count = 0;
  if (!get_partial_elf_txn_view(command, &bytes, &total, &op_count)) {
    collect_fallback();
    return;
  }

  size_t output_count = 0;
  size_t offset = 4 * sizeof(uint32_t);
  for (uint32_t i = 0; i < op_count; ++i) {
    const uint32_t op_size =
        iree_hal_amdxdna_txn_op_size(bytes, total, offset);
    if (op_size == 0 || (op_size & 3u) != 0 || offset > total ||
        op_size > total - offset) {
      collect_fallback();
      return;
    }
    if (bytes[offset] == kTxnOpWrite32 && op_size >= 6 * sizeof(uint32_t)) {
      const uint32_t location = read_txn_u32(bytes + offset + 8);
      const uint32_t reg = location & 0xfffff;
      const bool is_queue_push = (reg & 0x1fe00) == 0x1d200;
      const bool is_s2mm = (reg & 0x10) == 0;
      if (is_queue_push && is_s2mm) {
        const uint32_t queue_value = read_txn_u32(bytes + offset + 16);
        const uint32_t key = txn_bd_key(location, queue_value & 0xf);
        const uint8_t* dma = nullptr;
        const uint8_t* ddr = nullptr;
        if (!find_partial_elf_bd_ops(bytes, total, op_count, offset, key, &dma,
                                     &ddr)) {
          collect_fallback();
          return;
        }
        if (dma && ddr) {
          const size_t arg_position =
              static_cast<size_t>(read_txn_u32(ddr + 32)) + 1;
          const BoundBuffer* bound =
              find_bound_buffer_by_position(command, arg_position);
          if (bound && bound->buffer) {
            const uint64_t arg_offset = read_txn_u32(ddr + 40);
            const uint64_t span_words = partial_elf_dma_span_words(dma);
            const uint64_t requested_bytes =
                span_words > std::numeric_limits<uint64_t>::max() / 4
                    ? std::numeric_limits<uint64_t>::max()
                    : span_words * 4;
            if (arg_offset < bound->size) {
              const uint64_t available = bound->size - arg_offset;
              const uint64_t byte_length = std::min(requested_bytes, available);
              output_ranges->push_back(
                  {bound->buffer, bound->offset + arg_offset, byte_length});
              ++output_count;
            }
          }
        }
      }
    }
    offset += op_size;
  }
  if (!output_count) collect_fallback();
}

void mark_runtime_bindings_mapping_stale(
    iree_hal_amdxdna_native_command_t* command) {
  if (!command) return;
  if (command_is_pathb_chain(command)) {
    for (size_t i = 0; i < command->chain_child_count; ++i) {
      mark_runtime_bindings_mapping_stale(command->chain_children[i]);
    }
    return;
  }
  for (size_t i = 0; i < command->bound_buffer_count; ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (bound.buffer && bound.buffer->host_mirror &&
        !is_pathb_partial_elf_control_binding(command, bound)) {
      bound.buffer->native_mapping_stale = true;
    }
  }
}

void collect_command_output_ranges(
    iree_hal_amdxdna_native_command_t* command,
    std::vector<CommandOutputRange>* output_ranges) {
  if (!command) return;
  if (command_is_pathb_chain(command)) {
    for (size_t i = 0; i < command->chain_child_count; ++i) {
      collect_command_output_ranges(command->chain_children[i], output_ranges);
    }
    return;
  }
  if (uses_windows_dpu_regmap(command)) {
    collect_all_runtime_bindings(command, output_ranges);
  } else {
    collect_partial_elf_output_ranges(command, output_ranges);
  }
}

bool command_has_host_mirror(iree_hal_amdxdna_native_command_t* command) {
  if (!command) return false;
  if (command_is_pathb_chain(command)) {
    for (size_t i = 0; i < command->chain_child_count; ++i) {
      if (command_has_host_mirror(command->chain_children[i])) return true;
    }
    return false;
  }
  for (size_t i = 0; i < command->bound_buffer_count; ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (!bound.buffer || is_pathb_partial_elf_control_binding(command, bound)) {
      continue;
    }
    if (bound.buffer->host_mirror) return true;
  }
  return false;
}

iree_status_t refresh_command_output_ranges(
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  bool requires_refresh = false;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (command_has_host_mirror(commands[i])) {
      requires_refresh = true;
      break;
    }
  }
  if (!requires_refresh) return iree_ok_status();

  std::vector<CommandOutputRange> ranges;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    collect_command_output_ranges(commands[i], &ranges);
  }
  const size_t merged_count =
      iree_hal_amdxdna_native_windows_coalesce_buffer_ranges(ranges.data(),
                                                             ranges.size());
  for (size_t i = 0; i < merged_count; ++i) {
    const CommandOutputRange& range = ranges[i];
    auto* buffer = static_cast<iree_hal_amdxdna_native_buffer_t*>(range.buffer);
    // Direct persistent mappings follow the native capability contract:
    // callers synchronize buffers before host reads. Completion only has to
    // refresh compatibility mirrors whose public pointer is separate from the
    // KMT mapping.
    if (!buffer->host_mirror) continue;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync(
        buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST,
        range.length, range.offset));
  }
  return iree_ok_status();
}

iree_status_t refresh_command_output_ranges(
    iree_hal_amdxdna_native_command_t* command) {
  return refresh_command_output_ranges(&command, 1);
}

size_t partial_elf_real_bo_entry_count(
    const iree_hal_amdxdna_native_command_t* command) {
  size_t entry_count = 0;
  for (size_t i = 0; i < command->bound_buffer_count; ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (bound.buffer && bound.position > 0) {
      entry_count = iree_max(entry_count, bound.position);
    }
  }
  return entry_count;
}

iree_status_t maybe_write_partial_elf_bo_table(
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_partial_elf_npu_packet(command)) {
    return iree_ok_status();
  }

  // XRT's module-style command BO carries an out-of-packet table of kernel BO
  // GPU VAs after the 32-byte ERT_START_NPU packet. The inline private packet
  // still advertises only the ERT packet bytes, but the miniport inspects this
  // table for dependency metadata. The negotiated DDI ABI defines its capacity;
  // entries without a bound kernel argument remain zero, matching XRT. This
  // function only stores CPU bytes; the caller must SyncBuffer the table (or
  // the whole child exec BO) before the device reads it.
  std::array<D3DGPU_VIRTUAL_ADDRESS, mcdm::kMaxPathBBoTableEntries>
      real_bo_gpu_vas = {};
  const size_t real_bo_entry_count = partial_elf_real_bo_entry_count(command);
  for (size_t i = 0; i < command->bound_buffer_count; ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (!bound.buffer || bound.position == 0) continue;
    const size_t table_index = bound.position - 1;
    if (table_index >= real_bo_gpu_vas.size()) continue;
    real_bo_gpu_vas[table_index] =
        iree_hal_amdxdna_native_buffer_device_address(bound.buffer) +
        bound.offset;
  }
  mcdm::Error error;
  if (!mcdm::PopulatePathBBoTable(
          command->device->device, command->start_packet, command->command_size,
          real_bo_gpu_vas.data(), real_bo_entry_count, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM PARTIAL_ELF BO table population failed", error);
  }
  return iree_ok_status();
}

iree_status_t write_windows_dpu_regmap_u32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  if (command->reg_idx >= kWindowsDpuRegmapWords) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna Windows MCDM DPU register map is full");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[command->reg_idx++] = value;
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t write_windows_dpu_regmap_u64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  if (command->reg_idx + 1 >= kWindowsDpuRegmapWords) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna Windows MCDM DPU register map is full");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[command->reg_idx++] = static_cast<uint32_t>(value);
  regmap[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
  command->arg_count++;
  return iree_ok_status();
}

void set_windows_dpu_instruction_arg(iree_hal_amdxdna_native_command_t* command,
                                     uint64_t instruction_va,
                                     uint32_t instruction_words) {
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[kWindowsDpuInstructionRegWord] = static_cast<uint32_t>(instruction_va);
  regmap[kWindowsDpuInstructionRegWord + 1] =
      static_cast<uint32_t>(instruction_va >> 32);
  regmap[kWindowsDpuInstructionRegWord + 2] = instruction_words;
}

iree_status_t validate_windows_dpu_regmap_inputs(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  if (IREE_UNLIKELY(!queue || !queue->context ||
                    !queue->context->has_command_aperture)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires a command aperture");
  }
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  queue->context->pathb_single_code_staged_size = 0;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires the aperture code BO");
  }
  if (IREE_UNLIKELY(!command->control_buffer ||
                    !command->control_buffer->buffer.cpu_ptr ||
                    command->control_buffer_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU command has no control-code buffer");
  }
  if (IREE_UNLIKELY(command->control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM DPU control-code size is not word aligned");
  }
  if (IREE_UNLIKELY(command->control_buffer_size > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM DPU control code exceeds aperture code BO");
  }
  // The Windows wrapper DPU ABI carries a TXN selector plus a staged
  // instruction pointer. Data BO VAs after the selector are optional: some
  // START_NPU dispatches have no runtime buffer bindings, while reconfigure
  // and execute packets may have one or more. The rewrite below preserves the
  // first three data VAs if present and leaves absent slots zero-filled.
  const bool has_selector_args =
      command->arg_count >= 1 && command->reg_idx >= 2;
  if (IREE_UNLIKELY(!has_selector_args)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU packet is missing selector/data args");
  }
  return iree_ok_status();
}

iree_status_t rewrite_windows_dpu_regmap_to_instruction(
    iree_hal_amdxdna_native_command_t* command, uint64_t instruction_va) {
  if (command->windows_dpu_regmap_finalized) return iree_ok_status();
  if (IREE_UNLIKELY(command->control_buffer_size / sizeof(uint32_t) >
                    std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Windows MCDM DPU instruction word count is too large");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  const uint64_t selector = static_cast<uint64_t>(regmap[0]) |
                            (static_cast<uint64_t>(regmap[1]) << 32);
  const uint64_t ifm_va = static_cast<uint64_t>(regmap[2]) |
                          (static_cast<uint64_t>(regmap[3]) << 32);
  const uint64_t param_va = static_cast<uint64_t>(regmap[4]) |
                            (static_cast<uint64_t>(regmap[5]) << 32);
  const uint64_t ofm_va = static_cast<uint64_t>(regmap[6]) |
                          (static_cast<uint64_t>(regmap[7]) << 32);
  const uint64_t dummy0_va = static_cast<uint64_t>(regmap[8]) |
                             (static_cast<uint64_t>(regmap[9]) << 32);
  const uint64_t dummy1_va = static_cast<uint64_t>(regmap[10]) |
                             (static_cast<uint64_t>(regmap[11]) << 32);
  const uint64_t dummy2_va = static_cast<uint64_t>(regmap[12]) |
                             (static_cast<uint64_t>(regmap[13]) << 32);
  std::memset(regmap, 0, kWindowsDpuRegmapWords * sizeof(uint32_t));
  regmap[0] = static_cast<uint32_t>(selector);
  regmap[1] = static_cast<uint32_t>(selector >> 32);
  const uint32_t instruction_words =
      static_cast<uint32_t>(command->control_buffer_size / sizeof(uint32_t));
  set_windows_dpu_instruction_arg(command, instruction_va, instruction_words);
  regmap[5] = static_cast<uint32_t>(ifm_va);
  regmap[6] = static_cast<uint32_t>(ifm_va >> 32);
  regmap[7] = static_cast<uint32_t>(param_va);
  regmap[8] = static_cast<uint32_t>(param_va >> 32);
  regmap[9] = static_cast<uint32_t>(ofm_va);
  regmap[10] = static_cast<uint32_t>(ofm_va >> 32);
  regmap[11] = static_cast<uint32_t>(dummy0_va);
  regmap[12] = static_cast<uint32_t>(dummy0_va >> 32);
  regmap[13] = static_cast<uint32_t>(dummy1_va);
  regmap[14] = static_cast<uint32_t>(dummy1_va >> 32);
  regmap[15] = static_cast<uint32_t>(dummy2_va);
  regmap[16] = static_cast<uint32_t>(dummy2_va >> 32);
  command->reg_idx = kWindowsDpuRegmapWords;
  command->windows_dpu_regmap_finalized = true;
  return iree_ok_status();
}

iree_status_t finalize_windows_dpu_regmap(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_windows_dpu_regmap(command)) return iree_ok_status();
  if (command->windows_dpu_regmap_finalized) return iree_ok_status();
  IREE_RETURN_IF_ERROR(validate_windows_dpu_regmap_inputs(queue, command));
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  IREE_RETURN_IF_ERROR(stage_windows_dpu_code_buffer(queue, command));
  uint64_t instruction_va = aperture.code_gpu_va;

  IREE_RETURN_IF_ERROR(
      rewrite_windows_dpu_regmap_to_instruction(command, instruction_va));
  return iree_ok_status();
}

iree_status_t append_pathb_start_cu_chain_descriptor(
    iree_hal_amdxdna_native_command_t* child, uint8_t* descriptor_base,
    size_t descriptor_capacity, size_t* descriptor_used) {
  ert_start_kernel_cmd* start = command_start_packet(child);
  if (IREE_UNLIKELY(start->opcode != ERT_START_CU)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna Windows MCDM path-B chain descriptor only supports "
        "START_CU children");
  }

  const uint32_t extra_cu_masks = start->extra_cu_masks;
  const uint32_t packet_words = 1u + start->count;
  const uint32_t copy_start_word = 2u + extra_cu_masks;
  if (IREE_UNLIKELY(packet_words < copy_start_word)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B chain child START_CU packet is too short");
  }

  const uint32_t* words = reinterpret_cast<const uint32_t*>(start);
  uint32_t cu_index = 0;
  bool found_cu = false;
  for (uint32_t mask_index = 0; mask_index <= extra_cu_masks; ++mask_index) {
    const uint32_t mask = words[1 + mask_index];
    if (!mask) continue;
    cu_index = mask_index * 32u + first_set_bit(mask);
    found_cu = true;
    break;
  }
  if (IREE_UNLIKELY(!found_cu)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B chain child START_CU packet has no CU "
        "mask bit set");
  }

  const uint32_t copy_words = packet_words - copy_start_word;
  const size_t copy_bytes = static_cast<size_t>(copy_words) * sizeof(uint32_t);
  const size_t descriptor_bytes =
      kWindowsDpuChainDescriptorHeaderSize + copy_bytes;
  if (IREE_UNLIKELY(*descriptor_used > descriptor_capacity ||
                    descriptor_bytes >
                        descriptor_capacity - *descriptor_used)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain descriptor block exceeds 0x%zx "
        "bytes",
        descriptor_capacity);
  }

  uint8_t* descriptor = descriptor_base + *descriptor_used;
  std::memset(descriptor, 0, descriptor_bytes);
  WindowsDpuChainCuDescriptorHeader header = {};
  header.marker = 1;
  header.cu_index = cu_index;
  header.copy_word_count = copy_words;
  std::memcpy(descriptor, &header, sizeof(header));
  std::memcpy(descriptor + sizeof(header), words + copy_start_word, copy_bytes);
  *descriptor_used += descriptor_bytes;
  return iree_ok_status();
}

iree_status_t append_pathb_start_npu_chain_descriptor(
    iree_hal_amdxdna_native_command_t* child, uint8_t* descriptor_base,
    size_t descriptor_capacity, size_t* descriptor_used) {
  ert_start_kernel_cmd* start = command_start_packet(child);
  if (IREE_UNLIKELY(start->opcode != ERT_START_NPU)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna Windows MCDM path-B START_NPU chain descriptor received a "
        "non-START_NPU child");
  }
  ert_npu_data* npu_data = get_ert_npu_data(start);
  if (IREE_UNLIKELY(!npu_data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B START_NPU chain child has no NPU data");
  }
  if (IREE_UNLIKELY(*descriptor_used > descriptor_capacity ||
                    kWindowsDpuStartNpuChainDescriptorSize >
                        descriptor_capacity - *descriptor_used)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B START_NPU chain descriptor block exceeds "
        "0x%zx bytes",
        descriptor_capacity);
  }

  const uint32_t* args = get_ert_regmap_begin(start);
  const uint32_t* args_end = get_ert_regmap_end(start);
  const ptrdiff_t arg_words = args_end >= args ? args_end - args : 0;
  const uint32_t selector = arg_words > 0 ? args[0] : 0;
  const uint32_t selector_hi = arg_words > 1 ? args[1] : 0;

  WindowsDpuChainNpuDescriptor desc = {};
  desc.marker0 = 2;
  desc.instr_addr_lo = static_cast<uint32_t>(npu_data->instruction_buffer);
  desc.instr_addr_hi =
      static_cast<uint32_t>(npu_data->instruction_buffer >> 32);
  desc.instr_size = npu_data->instruction_buffer_size;
  desc.marker1 = 2;
  desc.selector = selector;
  desc.selector_hi = selector_hi;

  std::memcpy(descriptor_base + *descriptor_used, &desc, sizeof(desc));
  *descriptor_used += sizeof(desc);
  return iree_ok_status();
}

iree_status_t get_pathb_chain_region_sizes(
    iree_hal_amdxdna_native_command_t* chain_command, size_t* out_code_bytes,
    size_t* out_descriptor_bytes) {
  size_t code_offset = 0;
  size_t descriptor_bytes = 0;
  for (size_t child_index = 0; child_index < chain_command->chain_child_count;
       ++child_index) {
    iree_hal_amdxdna_native_command_t* child =
        chain_command->chain_children[child_index];
    code_offset = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
    code_offset += static_cast<size_t>(child->control_buffer_size);
    ert_start_kernel_cmd* start = command_start_packet(child);
    if (start->opcode == ERT_START_NPU) {
      descriptor_bytes += kWindowsDpuStartNpuChainDescriptorSize;
      continue;
    }
    if (IREE_UNLIKELY(start->opcode != ERT_START_CU)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains only support START_CU or "
          "START_NPU children");
    }
    const uint32_t packet_words = 1u + start->count;
    const uint32_t copy_start_word = 2u + start->extra_cu_masks;
    if (IREE_UNLIKELY(packet_words < copy_start_word)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna Windows MCDM path-B chain child START_CU packet is too "
          "short");
    }
    descriptor_bytes +=
        kWindowsDpuChainDescriptorHeaderSize +
        static_cast<size_t>(packet_words - copy_start_word) * sizeof(uint32_t);
  }
  // XRT's module-runlist aperture layout treats each child instruction stream
  // as occupying a full 0x8000 slot. Reserve through the final slot boundary so
  // batched parent chunks start on the same cadence and opcode-9 marker offsets
  // include the last child slot.
  *out_code_bytes = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
  *out_descriptor_bytes = descriptor_bytes;
  return iree_ok_status();
}

iree_status_t commit_prepared_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command);

iree_status_t prepare_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* chain_command, bool sync_aperture) {
  if (IREE_UNLIKELY(!queue || !queue->context ||
                    !queue->context->has_command_aperture)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires a command aperture");
  }
  if (IREE_UNLIKELY(chain_command->chain_child_count == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM command chain has no child commands");
  }
  mcdm::CommandAperture& aperture = pathb_chain_aperture(queue);
  queue->context->pathb_single_code_staged_size = 0;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires an aperture code BO");
  }
  if (IREE_UNLIKELY(!aperture.gpu_cpu_ptr ||
                    aperture.gpu_va_size <=
                        kWindowsDpuChainDescriptorApertureOffset)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires the locked command "
        "aperture GPU view");
  }

  // Always restage into the existing placement. prepared_valid caches slot
  // offsets, not a sticky device image: compact Path-B firmware drops
  // aperture GPU contents after consume, so copy+Commit must run before the
  // next chain submit.

  const uint64_t code_base_offset =
      chain_command->pathb_chain_code_aperture_offset;
  if (IREE_UNLIKELY(code_base_offset >= aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain code base offset %" PRIu64
        " exceeds aperture code BO (%" PRIu64 " bytes)",
        code_base_offset, aperture.code_size);
  }
  uint8_t* code = static_cast<uint8_t*>(aperture.code_cpu_ptr) +
                  static_cast<size_t>(code_base_offset);
  const size_t code_capacity =
      static_cast<size_t>(aperture.code_size - code_base_offset);
  size_t code_offset = 0;
  size_t code_used = 0;
  IREE_RETURN_IF_ERROR(reserve_child_code_offsets(
      chain_command, chain_command->chain_child_count));
  chain_command->pathb_chain_child_code_offset_count = 0;
  for (size_t child_index = 0; child_index < chain_command->chain_child_count;
       ++child_index) {
    iree_hal_amdxdna_native_command_t* child =
        chain_command->chain_children[child_index];
    if (IREE_UNLIKELY(!child || !child->control_buffer ||
                      !child->control_buffer->buffer.cpu_ptr ||
                      child->control_buffer_size == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM path-B chain child has no control-code "
          "buffer");
    }
    if (IREE_UNLIKELY(child->control_buffer_size % sizeof(uint32_t) != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna Windows MCDM path-B chain child control-code size is not "
          "word aligned");
    }
    const size_t child_code_size =
        static_cast<size_t>(child->control_buffer_size);
    code_offset = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
    if (IREE_UNLIKELY(code_offset > code_capacity ||
                      child_code_size > code_capacity - code_offset)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM path-B chain control code exceeds aperture "
          "code BO (%zu-byte child at offset %zu, capacity %zu)",
          child_code_size, code_offset, code_capacity);
    }
    chain_command->pathb_chain_child_code_offsets
        [chain_command->pathb_chain_child_code_offset_count++] = code_offset;
    code_used = code_offset + child_code_size;
    code_offset = code_used;
  }
  // Do not clear the full padded code range: START_NPU descriptors carry the
  // exact instruction byte count, and XRT leaves the 0x8000-spaced gaps as
  // allocator slack. Clearing those gaps dominated host-side chain prep.

  if (sync_aperture) {
    IREE_RETURN_IF_ERROR(acquire_pathb_code_range(
        queue, aperture.code_offset + code_base_offset,
        align_up_size(code_used, kWindowsDpuChainCodeAlignment)));
  }

  const size_t descriptor_offset =
      chain_command->pathb_chain_descriptor_aperture_offset
          ? static_cast<size_t>(
                chain_command->pathb_chain_descriptor_aperture_offset)
          : align_up_size(
                std::max<size_t>(static_cast<size_t>(
                                     kWindowsDpuChainDescriptorApertureOffset),
                                 static_cast<size_t>(aperture.code_offset +
                                                     code_base_offset) +
                                     code_used),
                0x1000);
  if (IREE_UNLIKELY(descriptor_offset >= aperture.gpu_va_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain descriptor block exceeds command "
        "aperture after %zu bytes of staged code",
        code_used);
  }
  uint8_t* descriptor_base =
      static_cast<uint8_t*>(aperture.gpu_cpu_ptr) + descriptor_offset;
  const size_t descriptor_capacity =
      static_cast<size_t>(aperture.gpu_va_size - descriptor_offset);
  const size_t descriptor_clear_bytes = std::min<size_t>(
      descriptor_capacity,
      chain_command->chain_child_count *
          std::max<size_t>(kWindowsDpuStartNpuChainDescriptorSize,
                           kWindowsDpuChainDescriptorHeaderSize +
                               kWindowsDpuRegmapWords * sizeof(uint32_t)));
  std::memset(descriptor_base, 0, descriptor_clear_bytes);

  size_t descriptor_used = 0;
  chain_command->pathb_chain_descriptor_gpu_va = 0;
  chain_command->pathb_chain_descriptor_bytes = 0;
  chain_command->pathb_chain_first_child_opcode =
      command_packet(chain_command->chain_children[0])->opcode;
  for (size_t child_index = 0; child_index < chain_command->chain_child_count;
       ++child_index) {
    iree_hal_amdxdna_native_command_t* child =
        chain_command->chain_children[child_index];
    reset_command_packet_for_start(child);
    ert_packet* child_packet = command_packet(child);
    if (IREE_UNLIKELY(child_packet->opcode !=
                      chain_command->pathb_chain_first_child_opcode)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains do not support mixed child ERT "
          "opcodes yet");
    }
    const bool is_start_cu_child = uses_windows_dpu_regmap(child);
    const bool is_start_npu_child = uses_partial_elf_npu_packet(child);
    if (IREE_UNLIKELY(!is_start_cu_child && !is_start_npu_child)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains only support DPU commands");
    }
    if (is_start_cu_child && child->windows_dpu_regmap_finalized) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM path-B chain child was already finalized");
    }
    if (IREE_UNLIKELY(!child->control_buffer ||
                      !child->control_buffer->buffer.cpu_ptr ||
                      child->control_buffer_size == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM path-B chain child has no control-code "
          "buffer");
    }
    if (IREE_UNLIKELY(child->control_buffer_size % sizeof(uint32_t) != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna Windows MCDM path-B chain child control-code size is not "
          "word aligned");
    }
    const size_t code_offset =
        chain_command->pathb_chain_child_code_offsets[child_index];
    const uint64_t instruction_va =
        aperture.code_gpu_va + code_base_offset + code_offset;
    if (is_start_cu_child) {
      IREE_RETURN_IF_ERROR(validate_windows_dpu_regmap_inputs(queue, child));
      IREE_RETURN_IF_ERROR(
          rewrite_windows_dpu_regmap_to_instruction(child, instruction_va));
      IREE_RETURN_IF_ERROR(append_pathb_start_cu_chain_descriptor(
          child, descriptor_base, descriptor_capacity, &descriptor_used));
    } else {
      ert_npu_data* npu_data = get_ert_npu_data(command_start_packet(child));
      if (IREE_UNLIKELY(!npu_data)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "amdxdna Windows MCDM PARTIAL_ELF chain child has no NPU data");
      }
      npu_data->instruction_buffer = instruction_va;
      npu_data->instruction_buffer_size =
          static_cast<uint32_t>(child->control_buffer_size);
      npu_data->instruction_prop_count = 0;
      IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(child));
      IREE_RETURN_IF_ERROR(append_pathb_start_npu_chain_descriptor(
          child, descriptor_base, descriptor_capacity, &descriptor_used));
    }
  }
  for (size_t child_index = 0; child_index < chain_command->chain_child_count;
       ++child_index) {
    // reset_command_packet_for_start and the PARTIAL_ELF BO-table rewrite both
    // modify child command BOs. Publish the complete child allocation so a
    // reused chain never exposes a fresh BO table with a stale ERT header.
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        chain_command->chain_children[child_index]->exec_buffer,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
  }
  // Child ERT packets are persistent mapped command metadata. Publish all
  // packet and BO-table stores as one ordered batch before aperture code is
  // committed and the parent chain is submitted.
  std::atomic_thread_fence(std::memory_order_release);

  if (sync_aperture) {
    // CopyAndCommit from each child's control_buffer. A spanning Commit of
    // the code range without that copy would publish leftover aperture bytes.
    IREE_RETURN_IF_ERROR(
        commit_prepared_pathb_chain_code(queue, chain_command));
    IREE_RETURN_IF_ERROR(publish_pathb_code_write(
        queue, aperture.code_offset + code_base_offset,
        align_up_size(code_used, kWindowsDpuChainCodeAlignment)));
    if (descriptor_used) {
      mcdm::Error error;
      if (!mcdm::CommitPathBCodeWrite(
              chain_command->device->api, chain_command->device->device,
              aperture, static_cast<uint64_t>(descriptor_offset),
              static_cast<uint64_t>(descriptor_used), &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B chain descriptor commit failed",
            error);
      }
      if (!mcdm::SyncCommandApertureCode(
              chain_command->device->api, chain_command->device->device,
              aperture, static_cast<uint64_t>(descriptor_offset),
              static_cast<uint64_t>(descriptor_used), &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B chain descriptor KMT invalidate failed",
            error);
      }
    }
  }
  chain_command->pathb_chain_descriptor_gpu_va =
      aperture.protocol_gpu_va + descriptor_offset;
  chain_command->pathb_chain_descriptor_bytes =
      static_cast<uint32_t>(descriptor_used);
  chain_command->pathb_chain_code_used_size =
      align_up_size(code_used, kWindowsDpuChainCodeAlignment);
  chain_command->pathb_chain_prepared_valid = true;
  chain_command->pathb_chain_code_dirty = true;
  chain_command->pathb_chain_descriptor_dirty = true;
  return iree_ok_status();
}

iree_status_t prepare_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* chain_command) {
  return prepare_pathb_chain_code(queue, chain_command, true);
}

iree_status_t sync_prepared_pathb_chain_batch(
    iree_hal_amdxdna_native_queue_t* queue, iree_host_size_t command_count,
    size_t code_offset, size_t code_bytes, size_t descriptor_offset,
    size_t descriptor_bytes) {
  mcdm::CommandAperture& aperture = pathb_chain_aperture(queue);
  mcdm::Error error;
  if (code_bytes) {
    if (!mcdm::PublishPathBCodeWrite(
            queue->context->device->api, queue->context->device->device,
            &queue->context->context, aperture,
            static_cast<uint64_t>(code_offset), code_bytes, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B batch code publish failed", error);
    }
  }
  if (descriptor_bytes) {
    if (!mcdm::CommitPathBCodeWrite(
            queue->context->device->api, queue->context->device->device,
            aperture, static_cast<uint64_t>(descriptor_offset),
            static_cast<uint64_t>(descriptor_bytes), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B batch descriptor commit failed", error);
    }
    if (!mcdm::SyncCommandApertureCode(
            queue->context->device->api, queue->context->device->device,
            aperture, static_cast<uint64_t>(descriptor_offset),
            static_cast<uint64_t>(descriptor_bytes), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B batch descriptor KMT invalidate failed",
          error);
    }
  }
  if ((code_bytes || descriptor_bytes) &&
      !mcdm::RefreshPathBCodeMappingAfterWrite(
          queue->context->device->api, queue->context->device->device,
          &aperture, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM path-B batch aperture refresh failed", error);
  }
  return iree_ok_status();
}

iree_status_t commit_prepared_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  mcdm::CommandAperture& aperture = pathb_chain_aperture(queue);
  if (IREE_UNLIKELY(command->pathb_chain_child_code_offset_count !=
                    command->chain_child_count)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM prepared chain is missing child code offsets");
  }
  mcdm::Error error;
  std::array<mcdm::CpuCopyRange, kWindowsDpuRunlistSubmitSize> ranges = {};
  for (size_t child_index = 0; child_index < command->chain_child_count;
       ++child_index) {
    const iree_hal_amdxdna_native_command_t* child =
        command->chain_children[child_index];
    ranges[child_index].offset =
        aperture.code_offset + command->pathb_chain_code_aperture_offset +
        command->pathb_chain_child_code_offsets[child_index];
    ranges[child_index].source = child->control_buffer->buffer.cpu_ptr;
    ranges[child_index].length = child->control_buffer_size;
  }
  if (!mcdm::CopyAndCommitPathBCodeWrites(aperture, ranges.data(),
                                          command->chain_child_count, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM path-B child code staging failed", error);
  }
  uint64_t span_begin = std::numeric_limits<uint64_t>::max();
  uint64_t span_end = 0;
  for (size_t child_index = 0; child_index < command->chain_child_count;
       ++child_index) {
    if (ranges[child_index].length == 0) continue;
    span_begin = std::min(span_begin, ranges[child_index].offset);
    span_end = std::max(span_end, ranges[child_index].offset +
                                      ranges[child_index].length);
  }
  if (span_begin < span_end &&
      !mcdm::SyncCommandApertureCode(
          command->device->api, command->device->device, aperture, span_begin,
          span_end - span_begin, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM path-B child code KMT invalidate failed", error);
  }
  return iree_ok_status();
}

}  // namespace

size_t iree_hal_amdxdna_native_windows_coalesce_buffer_ranges(
    iree_hal_amdxdna_native_windows_buffer_range_t* ranges,
    size_t range_count) {
  if (!ranges || !range_count) return 0;
  std::sort(
      ranges, ranges + range_count,
      [](const iree_hal_amdxdna_native_windows_buffer_range_t& lhs,
         const iree_hal_amdxdna_native_windows_buffer_range_t& rhs) {
        const uintptr_t lhs_buffer = reinterpret_cast<uintptr_t>(lhs.buffer);
        const uintptr_t rhs_buffer = reinterpret_cast<uintptr_t>(rhs.buffer);
        return lhs_buffer != rhs_buffer ? lhs_buffer < rhs_buffer
                                        : lhs.offset < rhs.offset;
      });
  size_t merged_count = 0;
  for (size_t i = 0; i < range_count; ++i) {
    const auto& range = ranges[i];
    if (!range.buffer || !range.length) continue;
    const uint64_t range_end =
        range.length > std::numeric_limits<uint64_t>::max() - range.offset
            ? std::numeric_limits<uint64_t>::max()
            : range.offset + range.length;
    if (merged_count != 0) {
      auto& previous = ranges[merged_count - 1];
      const uint64_t previous_end =
          previous.length >
                  std::numeric_limits<uint64_t>::max() - previous.offset
              ? std::numeric_limits<uint64_t>::max()
              : previous.offset + previous.length;
      if (previous.buffer == range.buffer && range.offset <= previous_end) {
        previous.length = std::max(previous_end, range_end) - previous.offset;
        continue;
      }
    }
    ranges[merged_count++] = {range.buffer, range.offset,
                              range_end - range.offset};
  }
  return merged_count;
}

bool iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
    uint32_t payload_dword_count, size_t allocation_size,
    size_t* out_packet_bytes) {
  if (!out_packet_bytes || allocation_size < sizeof(uint32_t)) return false;
  const size_t allocation_dword_count = allocation_size / sizeof(uint32_t);
  if (payload_dword_count >= allocation_dword_count) return false;
  *out_packet_bytes =
      (static_cast<size_t>(payload_dword_count) + 1) * sizeof(uint32_t);
  return true;
}

bool iree_hal_amdxdna_native_windows_buffer_requires_context(
    iree_hal_amdxdna_native_buffer_c_type_t type) {
  return type == IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE ||
         type == IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION;
}

bool iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
    const uint8_t* bytes, size_t total, uint32_t op_count, size_t queue_offset,
    uint32_t key, const uint8_t** out_dma, const uint8_t** out_ddr) {
  return find_partial_elf_bd_ops(bytes, total, op_count, queue_offset, key,
                                 out_dma, out_ddr);
}

uint64_t iree_hal_amdxdna_native_windows_partial_elf_dma_span_words(
    const uint8_t* dma) {
  return partial_elf_dma_span_words(dma);
}

iree_status_t materialize_deferred_buffer(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer || !buffer->deferred) return iree_ok_status();
  if (IREE_UNLIKELY(!buffer->device->pathb_context_ready)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM deferred BO materialized before pathb context "
        "setup");
  }

  mcdm::Buffer real_buffer;
  mcdm::Error error;
  if (!mcdm::CreateBuffer(buffer->device->api, buffer->device->device,
                          buffer->buffer.kind, buffer->buffer.size,
                          &real_buffer, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM deferred BO allocation failed", error);
  }
  if (real_buffer.cpu_ptr && buffer->deferred_storage_size > 0) {
    const uint64_t copy_size = std::min<uint64_t>(
        buffer->buffer.size,
        static_cast<uint64_t>(buffer->deferred_storage_size));
    std::memcpy(real_buffer.cpu_ptr, buffer->deferred_storage,
                static_cast<size_t>(copy_size));
    // A sync issued while this buffer was deferred could only order writes to
    // the temporary host storage. Publish the copy into the real KMT mapping
    // before any command can make the new device address visible to firmware.
    if (!mcdm::PublishBufferCpuWrites(real_buffer, /*offset=*/0, copy_size,
                                      &error)) {
      mcdm::DestroyBuffer(buffer->device->api, buffer->device->device,
                          &real_buffer);
      return status_from_mcdm_error(
          "amdxdna Windows MCDM deferred BO publication failed", error);
    }
  }
  const bool transfer_deferred_storage_to_mirror =
      buffer->host_mirror && buffer->host_mirror == buffer->deferred_storage;
  buffer->buffer = real_buffer;
  buffer->deferred = false;
  if (transfer_deferred_storage_to_mirror) {
    buffer->deferred_storage = nullptr;
    buffer->deferred_storage_size = 0;
  } else {
    release_deferred_buffer_storage(buffer);
  }
  return iree_ok_status();
}

iree_status_t materialize_deferred_instruction_buffer(
    iree_hal_amdxdna_native_context_t* context,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  (void)context;
  return materialize_deferred_buffer(buffer);
}

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  iree_hal_amdxdna_native_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*device), reinterpret_cast<void**>(&device)));
  new (device) iree_hal_amdxdna_native_device_t();
  device->host_allocator = host_allocator;

  mcdm::Error error;
  mcdm::Adapter adapter;
  if (!device->api.Load(&error)) {
    iree_status_t status = status_from_mcdm_error(
        "amdxdna Windows MCDM KMT API load failed", error);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }
  // NO XRT warmup: the pure-KMT replay works deterministically without any XRT
  // (3/3 status=0 on clean firmware). A held-open XRT device actually CONFLICTS
  // with our own kick by taking the NPU context. Talk to the driver directly.

  if (!mcdm::FindNpuAdapter(device->api, &adapter, &error)) {
    iree_status_t status = status_from_mcdm_error(
        "amdxdna Windows MCDM adapter discovery failed", error);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }
  if (!mcdm::CreateDevice(device->api, adapter, &device->device, &error)) {
    iree_status_t status = status_from_mcdm_error(
        "amdxdna Windows MCDM device creation failed", error);
    close_mcdm_adapter_handle(device->api, &adapter);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }

  *out_device = device;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_device_destroy(
    iree_hal_amdxdna_native_device_t* device) {
  if (!device) return;
  iree_allocator_t host_allocator = device->host_allocator;
  mcdm::DestroyDevice(device->api, &device->device);
  device->~iree_hal_amdxdna_native_device_t();
  iree_allocator_free(host_allocator, device);
}

iree_status_t iree_hal_amdxdna_native_device_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_power_mode_t power_mode) {
  (void)device;
  if (power_mode == IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna Windows MCDM power-mode control is not implemented");
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
    return IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  }
  return caps.default_dispatch_opcode;
}

iree_status_t iree_hal_amdxdna_native_device_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_caps);
  iree_hal_amdxdna_native_c_device_caps_t caps = {};
  const mcdm::McdmSubmissionPolicy submission_policy =
      mcdm::GetMcdmSubmissionPolicy(device->device.mcdm_abi);
  caps.max_effective_queues = 1;
  const size_t chain_exec_bo_size =
      static_cast<size_t>(windows_dpu_pathb_chain_exec_bo_size());
  caps.max_command_chain_slots = chain_slot_capacity(chain_exec_bo_size);
  caps.max_cached_chain_child_commands =
      kWindowsChainCacheChildCommandBudget;
  caps.context_image_models =
      IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_XCLBIN;
  caps.dispatch_models = IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_CU |
                         IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU |
                         IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF |
                         IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_COMMAND_CHAIN;
  caps.completion_models =
      IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_SYNCHRONOUS_WAIT |
      IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_PROGRESS_FENCE |
      IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_COMPLETION_SLOT;
  caps.supports_host_buffer_reuse =
      mcdm::SupportsHostBufferReuse(device->device.mcdm_abi);
  caps.native_owns_control_code_publication = true;
  // Issue may return before the native completion wait finishes. The HAL
  // retains native resources and keeps cache entries in flight until the
  // completion batch publishes its signal semaphores.
  caps.submit_completion_is_deferred =
      submission_policy.submit_completion_is_deferred;
  caps.supports_external_buffer_import = false;
  caps.supports_external_buffer_export = false;
  caps.supports_real_multi_queue = false;
  caps.default_dispatch_opcode =
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  caps.command_chain_status =
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_ENABLED_BY_DEFAULT;
  *out_caps = caps;
  if (!submission_policy.supports_command_chaining) {
    out_caps->max_command_chain_slots = 0;
    out_caps->dispatch_models &=
        ~IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_COMMAND_CHAIN;
    out_caps->dispatch_models &=
        ~IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF;
    out_caps->command_chain_status =
        IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_KNOWN_BAD_STACK;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));

  if (iree_hal_amdxdna_native_windows_buffer_requires_context(type)) {
    return create_deferred_native_buffer(
        device, type, static_cast<uint64_t>(size), out_buffer);
  }

  mcdm::Buffer buffer;
  mcdm::Error error;
  if (!mcdm::CreateBuffer(device->api, device->device,
                          to_mcdm_buffer_kind(type),
                          static_cast<uint64_t>(size), &buffer, &error)) {
    return status_from_mcdm_error("amdxdna Windows MCDM BO allocation failed",
                                  error);
  }
  return create_native_buffer_from_mcdm(device, buffer, type, out_buffer);
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_c_context_image_t* image,
    iree_hal_amdxdna_native_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(image);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = nullptr;
  if (IREE_UNLIKELY(image->type !=
                    IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_XCLBIN)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM context creation requires an xclbin context "
        "image; compile with --iree-amdaie-amdxdna-emit-context-xclbin=true");
  }
  iree_const_byte_span_t pdi = image->pdi;
  iree_const_byte_span_t xclbin = image->xclbin;
  if (IREE_UNLIKELY(xclbin.data_length == 0 || !xclbin.data)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM context creation requires xclbin data; "
        "compile with --iree-amdaie-amdxdna-emit-context-xclbin=true");
  }

  // Context setup remaps the device's shared command aperture. Serialize the
  // complete retire/create/activate transition so another context cannot
  // submit against an aperture while it is being replaced.
  std::unique_lock<std::mutex> context_lock(device->pathb_context_mutex);
  device->pathb_context_cv.wait(
      context_lock,
      [&]() { return device->pathb_active_submission_count == 0; });
  IREE_RETURN_IF_ERROR(retire_pathb_active_context_locked(device, nullptr));

  mcdm::Error error;
  iree_byte_span_t private_data = iree_byte_span_empty();
  mcdm::ContextBlobInfo info;
  mcdm::Buffer context_private_buffer;
  auto release_context_private_data = [&]() {
    iree_allocator_free(device->host_allocator, private_data.data);
    private_data = iree_byte_span_empty();
    mcdm::DestroyBuffer(device->api, device->device, &context_private_buffer);
    context_private_buffer = mcdm::Buffer();
    mcdm::ContextBlobInfoDeinitialize(&info);
  };
  if (!mcdm::BuildContextPrivateDataForDevice(
          device->api, device->device, xclbin.data, xclbin.data_length,
          GetCurrentProcessId(), device->host_allocator, &private_data, &info,
          &context_private_buffer, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM context blob generation failed", error);
  }
  mcdm::Context context = {};
  if (!mcdm::CreateContext(device->api, device->device, private_data.data,
                           private_data.data_length, &context, &error)) {
    release_context_private_data();
    return status_from_mcdm_error(
        "amdxdna Windows MCDM context creation failed", error);
  }
  context.context_private_buffer = context_private_buffer;
  context_private_buffer = mcdm::Buffer();
  iree_allocator_free(device->host_allocator, private_data.data);
  private_data = iree_byte_span_empty();
  mcdm::CommandAperture command_aperture = {};
  bool has_command_aperture = false;
  if (!mcdm::CreateCommandAperture(device->api, device->device, context,
                                   &command_aperture, &error)) {
    mcdm::DestroyContext(device->api, device->device, &context);
    mcdm::ContextBlobInfoDeinitialize(&info);
    return status_from_mcdm_error(
        "amdxdna Windows MCDM command aperture creation failed", error);
  }
  has_command_aperture = true;
  context.completion_ring.kind = mcdm::BufferKind::cacheable;
  context.completion_ring.size = command_aperture.allocation_size;
  context.completion_ring.mapped_size = command_aperture.allocation_size;
  context.completion_ring.allocation = command_aperture.allocation;
  context.completion_ring.resource = command_aperture.resource;
  context.completion_ring.gpu_va = command_aperture.status_gpu_va;
  context.completion_ring.cpu_ptr = command_aperture.cpu_ptr;
  context.completion_ring_ready = true;
  context.completion_ring_owned = false;
  if (!mcdm::SubmitAndWaitPathBSetup(device->api, device->device, &context,
                                     &command_aperture, pdi.data,
                                     pdi.data_length, &error)) {
    mcdm::DestroyContextWithCommandAperture(device->api, device->device,
                                            &context, &command_aperture);
    mcdm::ContextBlobInfoDeinitialize(&info);
    return status_from_mcdm_error("amdxdna Windows MCDM pathb setup failed",
                                  error);
  }
  device->pathb_context_ready = true;

  iree_hal_amdxdna_native_context_t* native_context = nullptr;
  iree_status_t status =
      iree_allocator_malloc(device->host_allocator, sizeof(*native_context),
                            reinterpret_cast<void**>(&native_context));
  if (!iree_status_is_ok(status)) {
    mcdm::DestroyContextWithCommandAperture(device->api, device->device,
                                            &context, &command_aperture);
    mcdm::ContextBlobInfoDeinitialize(&info);
    return status;
  }
  new (native_context) iree_hal_amdxdna_native_context_t();
  native_context->device = device;
  native_context->context = context;
  native_context->command_aperture = command_aperture;
  native_context->has_command_aperture = has_command_aperture;
  native_context->pathb_persistent_code_slot_size =
      mcdm::GetMcdmAbiInfo(device->device.mcdm_abi)
          .command_aperture_code_slot_size;
  if (IREE_UNLIKELY(!native_context->pathb_persistent_code_slot_size ||
                    native_context->pathb_persistent_code_slot_size >
                        command_aperture.code_size)) {
    native_context->~iree_hal_amdxdna_native_context_t();
    iree_allocator_free(device->host_allocator, native_context);
    mcdm::DestroyContextWithCommandAperture(device->api, device->device,
                                            &context, &command_aperture);
    mcdm::ContextBlobInfoDeinitialize(&info);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM command aperture has no persistent code slots");
  }
  native_context->pathb_persistent_code_slots_in_use.resize(
      static_cast<size_t>(command_aperture.code_size /
                          native_context->pathb_persistent_code_slot_size),
      0);
  native_context->pathb_persistent_code_commands.reserve(
      native_context->pathb_persistent_code_slots_in_use.size());
  native_context->pathb_completion_slots_in_use.resize(
      mcdm::PathBCompletionCapacity(context), 0);
  if (IREE_UNLIKELY(
          native_context->pathb_completion_slots_in_use.empty())) {
    native_context->~iree_hal_amdxdna_native_context_t();
    iree_allocator_free(device->host_allocator, native_context);
    mcdm::DestroyContextWithCommandAperture(device->api, device->device,
                                            &context, &command_aperture);
    mcdm::ContextBlobInfoDeinitialize(&info);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM completion ring has no submission slots");
  }
  native_context->info = info;
  info = mcdm::ContextBlobInfo();
  native_context->queue.context = native_context;
  device->pathb_active_context = native_context;
  *out_context = native_context;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context) {
  if (!context) return;
  iree_hal_amdxdna_native_device_t* device = context->device;
  std::unique_lock<std::mutex> context_lock(device->pathb_context_mutex);
  device->pathb_context_cv.wait(
      context_lock,
      [&]() { return device->pathb_active_submission_count == 0; });
  iree_status_ignore(close_pathb_single_code_ranges(&context->queue));
  if (device->pathb_active_context == context) {
    device->pathb_active_context = nullptr;
  }
  for (iree_hal_amdxdna_native_command_t* command :
       context->pathb_persistent_code_commands) {
    command->pathb_single_code_owner_context = nullptr;
    command->pathb_single_code_first_slot = 0;
    command->pathb_single_code_slot_count = 0;
    command->pathb_single_code_aperture_offset = 0;
    command->pathb_single_code_aperture_capacity = 0;
  }
  context->pathb_persistent_code_commands.clear();
  if (context->has_command_aperture) {
    mcdm::DestroyContextWithCommandAperture(
        context->device->api, context->device->device, &context->context,
        &context->command_aperture);
  } else {
    mcdm::DestroyContext(context->device->api, context->device->device,
                         &context->context);
  }
  mcdm::ContextBlobInfoDeinitialize(&context->info);
  iree_allocator_t host_allocator = context->device->host_allocator;
  context->~iree_hal_amdxdna_native_context_t();
  iree_allocator_free(host_allocator, context);
}

iree_status_t iree_hal_amdxdna_native_context_close_single_aperture_session(
    iree_hal_amdxdna_native_context_t* context) {
  if (!context) return iree_ok_status();
  return close_pathb_single_code_ranges(&context->queue);
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
  if (!buffer) return;
  iree_allocator_t host_allocator = buffer->device->host_allocator;
  if (buffer->host_mirror && buffer->host_mirror != buffer->deferred_storage) {
    iree_allocator_free(host_allocator, buffer->host_mirror);
  }
  buffer->host_mirror = nullptr;
  release_deferred_buffer_storage(buffer);
  if (!buffer->deferred) {
    mcdm::DestroyBuffer(buffer->device->api, buffer->device->device,
                        &buffer->buffer);
  }
  iree_allocator_free(host_allocator, buffer);
}

iree_status_t iree_hal_amdxdna_native_buffer_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = nullptr;
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  if (buffer->type == IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY &&
      !buffer->deferred && !buffer->host_mirror) {
    buffer->direct_host_mapping = static_cast<uint8_t*>(buffer->buffer.cpu_ptr);
  } else {
    IREE_RETURN_IF_ERROR(ensure_host_buffer_mirror(buffer));
  }
  void* host_ptr =
      buffer->host_mirror ? buffer->host_mirror : buffer->buffer.cpu_ptr;
  if (IREE_UNLIKELY(!host_ptr)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not host-mapped");
  }
  *out_ptr = host_ptr;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_device_size_t size, iree_device_size_t offset) {
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(offset));
  if (buffer->deferred) {
    return iree_ok_status();
  }
  const uint64_t sync_offset = static_cast<uint64_t>(offset);
  uint64_t sync_size = static_cast<uint64_t>(size);
  if (sync_offset < buffer->buffer.size) {
    sync_size =
        std::min<uint64_t>(sync_size, buffer->buffer.size - sync_offset);
  } else {
    sync_size = 0;
  }
  if (buffer->type == IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY) {
    mcdm::Error error;
    if (buffer->direct_host_mapping) {
      if (direction == IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE) {
        if (!mcdm::PublishBufferCpuWrites(buffer->buffer, sync_offset,
                                          sync_size, &error)) {
          return status_from_mcdm_error(
              "amdxdna Windows MCDM host BO publication failed", error);
        }
        return iree_ok_status();
      }
      if (!mcdm::InvalidateBufferCpuReads(buffer->buffer, sync_offset,
                                          sync_size, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM host BO cache invalidate failed", error);
      }
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(ensure_host_buffer_mirror(buffer));
    if (direction == IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE) {
      if (buffer->native_mapping_stale) {
        if (!mcdm::RefreshBufferCpuMapping(buffer->device->api,
                                           buffer->device->device,
                                           &buffer->buffer, &error)) {
          return status_from_mcdm_error(
              "amdxdna Windows MCDM host BO CPU mapping refresh failed", error);
        }
        buffer->native_mapping_stale = false;
      }
      if (sync_size) {
        std::memcpy(static_cast<uint8_t*>(buffer->buffer.cpu_ptr) + sync_offset,
                    buffer->host_mirror + sync_offset,
                    static_cast<size_t>(sync_size));
      }
      if (!mcdm::SyncBuffer(buffer->device->api, buffer->device->device,
                            buffer->buffer, sync_offset, sync_size, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM host BO publication failed", error);
      }
      return iree_ok_status();
    }
    if (!mcdm::InvalidateBufferCpuReads(buffer->buffer, sync_offset, sync_size,
                                        &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM host BO cache invalidate failed", error);
    }
    if (buffer->native_mapping_stale) {
      if (!mcdm::RefreshBufferCpuMapping(buffer->device->api,
                                         buffer->device->device,
                                         &buffer->buffer, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM host BO CPU mapping refresh failed", error);
      }
      buffer->native_mapping_stale = false;
    }
    if (sync_size) {
      std::memcpy(buffer->host_mirror + sync_offset,
                  static_cast<uint8_t*>(buffer->buffer.cpu_ptr) + sync_offset,
                  static_cast<size_t>(sync_size));
    }
    return iree_ok_status();
  }
  mcdm::Error error;
  if (direction == IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE) {
    // Cacheable and instruction BOs are written through their Lock2 mapping.
    // Publish the requested range after all packet and BO-table updates, just
    // as XRT's host-only BO sync does before run.start(). A process write
    // barrier alone does not evict dirty cache lines from that mapping.
    if (!mcdm::SyncBuffer(buffer->device->api, buffer->device->device,
                          buffer->buffer, sync_offset, sync_size, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM BO publication failed", error);
    }
    return iree_ok_status();
  }
  if (!mcdm::SyncBuffer(buffer->device->api, buffer->device->device,
                        buffer->buffer, sync_offset, sync_size, &error)) {
    return status_from_mcdm_error("amdxdna Windows MCDM BO sync failed", error);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction) {
  return iree_hal_amdxdna_native_buffer_sync(
      buffer, direction, iree_hal_amdxdna_native_buffer_size(buffer), 0);
}

iree_status_t iree_hal_amdxdna_native_buffer_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return materialize_deferred_buffer(buffer);
}

uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return buffer->buffer.gpu_va;
}

iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return static_cast<iree_device_size_t>(buffer->buffer.size);
}

iree_status_t iree_hal_amdxdna_native_context_open_cu(
    iree_hal_amdxdna_native_context_t* context, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_c_cu_index_t* out_cu_index) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_cu_index);

  const size_t kernel_count = context->info.kernel_name_count;
  if (kernel_count == 0) {
    out_cu_index->index = 0;
    return iree_ok_status();
  }

  iree_string_view_t requested = normalize_cu_name(kernel_name);
  for (size_t i = 0; i < kernel_count; ++i) {
    const char* available_name = mcdm::ContextBlobInfoKernelName(
        &context->info, static_cast<uint32_t>(i));
    if (iree_string_view_equal(requested,
                               iree_make_cstring_view(available_name))) {
      out_cu_index->index = static_cast<uint32_t>(i);
      return iree_ok_status();
    }
  }

  if (kernel_count == 1) {
    out_cu_index->index = 0;
    return iree_ok_status();
  }

  char available[512] = {0};
  size_t available_length = 0;
  for (size_t i = 0; i < kernel_count; ++i) {
    const char* available_name = mcdm::ContextBlobInfoKernelName(
        &context->info, static_cast<uint32_t>(i));
    if (i && available_length < sizeof(available)) {
      available_length +=
          snprintf(available + available_length,
                   sizeof(available) - available_length, "%s", ", ");
    }
    const size_t offset = available_length < sizeof(available)
                              ? available_length
                              : sizeof(available);
    available_length +=
        snprintf(available + offset,
                 available_length < sizeof(available)
                     ? sizeof(available) - available_length
                     : 0,
                 "%s", available_name[0] == 0 ? "<unnamed>" : available_name);
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "amdxdna Windows MCDM context does not contain requested CU '%.*s'; "
      "available CUs: %s",
      static_cast<int>(requested.size), requested.data, available);
}

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_queue(
    iree_hal_amdxdna_native_context_t* context) {
  return &context->queue;
}

uint64_t iree_hal_amdxdna_native_queue_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue) {
  return queue->exec_command_count;
}

iree_status_t iree_hal_amdxdna_native_command_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_t** out_command) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command);
  *out_command = nullptr;

  const WindowsMcdmOpcodeHandler& handler = windows_mcdm_opcode_handler(opcode);
  iree_hal_amdxdna_native_buffer_t* exec_buffer = nullptr;
  iree_status_t status = iree_ok_status();
  uint64_t exec_buffer_size = kMaxExecBoSize;
  if (handler.is_chain) {
    exec_buffer_size = windows_dpu_pathb_chain_exec_bo_size();
  } else {
    exec_buffer_size = kWindowsDpuPathBExecBoSize;
  }
  status = create_deferred_native_buffer_with_kind(
      device, mcdm::BufferKind::execbuf, exec_buffer_size, &exec_buffer);
  if (!iree_status_is_ok(status)) return status;

  iree_hal_amdxdna_native_command_t* command = nullptr;
  status = iree_allocator_malloc(device->host_allocator, sizeof(*command),
                                 reinterpret_cast<void**>(&command));
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_buffer_destroy(exec_buffer);
    return status;
  }
  std::memset(command, 0, sizeof(*command));
  command->device = device;
  command->opcode = opcode;
  command->exec_buffer = exec_buffer;
  command->start_packet =
      reinterpret_cast<ert_start_kernel_cmd*>(exec_buffer->buffer.cpu_ptr);
  command->command_size = static_cast<size_t>(exec_buffer->buffer.size);
  std::memset(command->start_packet, 0, command->command_size);
  command->start_packet->state = ERT_CMD_STATE_NEW;
  command->start_packet->opcode = handler.ert_opcode;
  command->start_packet->type = ERT_CU;
  status = inc_pkt_count(command, sizeof(uint32_t));
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_command_destroy(command);
    return status;
  }
  if (handler.initial_packet_word_count != 1) {
    command->start_packet->count = handler.initial_packet_word_count;
  }
  *out_command = command;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command) {
  if (!command) return;
  release_command_persistent_code_slots(command);
  iree_allocator_free(command->device->host_allocator, command->bound_buffers);
  iree_allocator_free(command->device->host_allocator,
                      command->pathb_chain_child_code_offsets);
  iree_allocator_free(command->device->host_allocator, command->chain_children);
  iree_hal_amdxdna_native_buffer_destroy(command->exec_buffer);
  command->exec_buffer = nullptr;
  command->pathb_chain_child_code_offsets = nullptr;
  command->pathb_chain_child_code_offset_count = 0;
  command->pathb_chain_child_code_offset_capacity = 0;
  command->chain_children = nullptr;
  command->chain_child_count = 0;
  command->chain_child_capacity = 0;
  command->bound_buffers = nullptr;
  command->bound_buffer_count = 0;
  command->bound_buffer_capacity = 0;
  iree_allocator_free(command->device->host_allocator, command);
}

iree_status_t iree_hal_amdxdna_native_command_reset(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  const WindowsMcdmOpcodeHandler& handler = command_opcode_handler(command);
  command->control_buffer = nullptr;
  command->control_buffer_size = 0;
  command->cached_start_header = 0;
  command->cached_start_header_valid = false;
  command->reg_idx = 0;
  command->arg_count = 0;
  command->windows_dpu_regmap_finalized = false;
  command->pathb_code_staged = false;
  command->pathb_code_staged_size = 0;
  command->pathb_chain_descriptor_gpu_va = 0;
  command->pathb_chain_descriptor_bytes = 0;
  command->pathb_chain_first_child_opcode = 0;
  command->pathb_chain_code_used_size = 0;
  command->pathb_chain_code_aperture_offset = 0;
  command->pathb_chain_descriptor_aperture_offset = 0;
  command->pathb_chain_aperture_generation = 0;
  command->pathb_chain_prepared_valid = false;
  command->pathb_chain_code_dirty = false;
  command->pathb_chain_descriptor_dirty = false;
  command->pathb_chain_bound_residency_checked = false;
  command->pathb_chain_child_code_offset_count = 0;
  command->chain_child_count = 0;
  command->bound_buffer_count = 0;

  std::memset(command->start_packet, 0, command->command_size);
  command->start_packet->state = ERT_CMD_STATE_NEW;
  command->start_packet->opcode = handler.ert_opcode;
  command->start_packet->type = ERT_CU;
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(uint32_t)));
  if (handler.initial_packet_word_count != 1) {
    command->start_packet->count = handler.initial_packet_word_count;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_c_cu_index_t cu_index) {
  command->start_packet->cu_mask = 0x1u << cu_index.index;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size) {
  const WindowsMcdmOpcodeHandler& handler = command_opcode_handler(command);
  if (!handler.accepts_control_buffer) {
    if (!handler.is_chain) return iree_ok_status();
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported control buffer command opcode");
  }
  const char* label =
      handler.uses_partial_elf ? "PARTIAL_ELF instruction" : "instruction";
  if (IREE_UNLIKELY(!control_buffer || control_buffer_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna Windows MCDM %s buffer is empty", label);
  }
  if (IREE_UNLIKELY(control_buffer_size >
                    iree_hal_amdxdna_native_buffer_size(control_buffer))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Windows MCDM %s byte count exceeds BO size", label);
  }
  if (IREE_UNLIKELY(control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM %s byte count is not word aligned", label);
  }
  if (IREE_UNLIKELY(control_buffer_size >
                    std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna Windows MCDM %s buffer is too large",
                            label);
  }
  command->control_buffer = control_buffer;
  command->control_buffer_size = control_buffer_size;
  command->pathb_code_staged = false;
  command->pathb_code_staged_size = 0;
  if (!handler.uses_partial_elf) {
    return iree_ok_status();
  }
  ert_npu_data* npu_data = get_ert_npu_data(command->start_packet);
  if (IREE_UNLIKELY(!npu_data)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna Windows MCDM PARTIAL_ELF packet has no NPU data");
  }
  // Path-B exposes the instruction stream through the context instruction BO.
  // The control buffer is host-side source data staged there before submit.
  npu_data->instruction_buffer = 0;
  npu_data->instruction_buffer_size =
      static_cast<uint32_t>(control_buffer_size);
  npu_data->instruction_prop_count = 0;
  return bind_buffer_ref(command, /*position=*/0, control_buffer, /*offset=*/0,
                         control_buffer_size);
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  if (uses_windows_dpu_regmap(command)) {
    if (command->arg_count == 0) {
      return write_windows_dpu_regmap_u64(command, value);
    }
    return write_windows_dpu_regmap_u32(command, value);
  }
  if (uses_partial_elf_npu_packet(command)) {
    if (command->reg_idx >= 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no free arg slots");
    }
    uint32_t* args = get_ert_regmap_begin(command->start_packet);
    args[command->reg_idx++] = value;
    command->arg_count++;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(value)));
  auto args = get_ert_regmap_begin(command->start_packet);
  args[command->reg_idx++] = value;
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  if (uses_windows_dpu_regmap(command)) {
    return write_windows_dpu_regmap_u64(command, value);
  }
  if (uses_partial_elf_npu_packet(command)) {
    if (command->reg_idx + 1 >= 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no free u64 arg slot");
    }
    uint32_t* args = get_ert_regmap_begin(command->start_packet);
    args[command->reg_idx++] = static_cast<uint32_t>(value);
    args[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
    command->arg_count++;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(value)));
  auto args = get_ert_regmap_begin(command->start_packet);
  args[command->reg_idx++] = static_cast<uint32_t>(value);
  args[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_update_arg_64(
    iree_hal_amdxdna_native_command_t* command, iree_host_size_t arg_index,
    uint64_t value) {
  (void)command;
  (void)arg_index;
  (void)value;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna Windows MCDM command arg update is not implemented");
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(command,
                                                                  buffer, 0);
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset) {
  IREE_RETURN_IF_ERROR(materialize_deferred_buffer(buffer));
  if (offset > iree_hal_amdxdna_native_buffer_size(buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer offset too large");
  }
  if (uses_partial_elf_npu_packet(command)) {
    return bind_buffer_ref(
        command, command->arg_count, buffer, offset,
        iree_hal_amdxdna_native_buffer_size(buffer) - offset);
  }
  IREE_RETURN_IF_ERROR(check_pkt_count_capacity(command, sizeof(uint64_t)));
  IREE_RETURN_IF_ERROR(
      bind_buffer_ref(command, command->arg_count, buffer, offset,
                      iree_hal_amdxdna_native_buffer_size(buffer) - offset));
  return iree_hal_amdxdna_native_command_add_arg_64(
      command, iree_hal_amdxdna_native_buffer_device_address(buffer) + offset);
}

iree_status_t iree_hal_amdxdna_native_command_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size) {
  IREE_RETURN_IF_ERROR(materialize_deferred_buffer(buffer));
  const iree_device_size_t buffer_size =
      iree_hal_amdxdna_native_buffer_size(buffer);
  if (offset > buffer_size || size > buffer_size - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer binding range is "
                            "out of bounds (position=%zu offset=%" PRIu64
                            " size=%" PRIu64 " buffer_size=%" PRIu64 ")",
                            position, offset, size, buffer_size);
  }
  return bind_buffer_ref(command, position, buffer, offset, size);
}

iree_status_t iree_hal_amdxdna_native_command_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  command->bound_buffer_count = 0;
  command->pathb_chain_bound_residency_checked = false;
  if (uses_partial_elf_npu_packet(command) && command->control_buffer) {
    IREE_RETURN_IF_ERROR(bind_buffer_ref(command, /*position=*/0,
                                         command->control_buffer, /*offset=*/0,
                                         command->control_buffer_size));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (IREE_UNLIKELY(!command_is_pathb_chain(command))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  command->pathb_chain_prepared_valid = false;
  command->pathb_chain_code_dirty = true;
  command->pathb_chain_descriptor_dirty = true;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (command_is_pathb_chain(command)) {
    command->pathb_chain_code_dirty = true;
    if (!command->pathb_chain_prepared_valid) {
      command->pathb_chain_descriptor_dirty = true;
    }
  } else {
    command->pathb_code_staged = false;
    command->pathb_code_staged_size = 0;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (IREE_UNLIKELY(!command_is_pathb_chain(command))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (!command->pathb_chain_prepared_valid) {
    command->pathb_chain_descriptor_dirty = true;
  }
  command->pathb_chain_code_dirty = true;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  if (IREE_UNLIKELY(!command_is_pathb_chain(command))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (IREE_UNLIKELY(command_count > std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command chain is too large");
  }
  const size_t chain_bytes = offsetof(ert_packet, data) +
                             sizeof(ert_cmd_chain_data) +
                             command_count * sizeof(uint64_t);
  if (chain_bytes > command->command_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna cmd-chain: %" PRIhsz
                            " slots exceed exec buffer (%zu > %zu bytes)",
                            command_count, chain_bytes, command->command_size);
  }

  ert_packet* packet = command_packet(command);
  std::memset(packet, 0, command->command_size);
  command->cached_start_header_valid = false;
  command->chain_child_count = 0;
  command->pathb_chain_child_code_offset_count = 0;
  command->bound_buffer_count = 0;
  IREE_RETURN_IF_ERROR(reserve_chain_children(command, command_count));
  packet->state = ERT_CMD_STATE_NEW;
  packet->opcode = ERT_CMD_CHAIN;
  ert_cmd_chain_data* chain_data =
      reinterpret_cast<ert_cmd_chain_data*>(packet->data);
  chain_data->command_count = static_cast<uint32_t>(command_count);
  chain_data->submit_index = 0;
  chain_data->error_index = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    IREE_RETURN_IF_ERROR(materialize_deferred_buffer(commands[i]->exec_buffer));
    // Materializing a deferred exec BO replaces the temporary host-storage
    // mapping with the real KMT allocation mapping. Refresh the cached packet
    // pointer before any later descriptor construction reads the child.
    commands[i]->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        commands[i]->exec_buffer->buffer.cpu_ptr);
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        commands[i]->exec_buffer,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
    chain_data->data[i] = mcdm::GetPathBChainChildHandle(
        command->device->device, &commands[i]->exec_buffer->buffer);
    command->chain_children[command->chain_child_count++] = commands[i];
    IREE_RETURN_IF_ERROR(bind_buffer_ref(
        command, i, commands[i]->exec_buffer, 0,
        iree_hal_amdxdna_native_buffer_size(commands[i]->exec_buffer)));
  }
  packet->count =
      (sizeof(ert_cmd_chain_data) + command_count * sizeof(uint64_t)) /
      sizeof(uint32_t);
  return iree_ok_status();
}

// Async submit is built on a clean issue/wait split of the path-B submit:
// submit_issue does all pre-dispatch work and issues the command to the HW
// queue (SubmitPathB[Chain] -> fence token); submit_wait blocks on that fence
// and does the post-dispatch aperture + output sync. submit_and_wait composes
// them (unchanged synchronous behavior); the async DDI exposes them separately.
struct iree_hal_amdxdna_native_submission_t {
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_native_queue_t* queue = nullptr;
  iree_hal_amdxdna_native_command_t* command = nullptr;
  iree_hal_amdxdna_native_command_t** commands = nullptr;
  iree_host_size_t command_count = 0;
  iree_host_size_t issued_count = 0;
  char label[128] = {};
  size_t label_size = 0;
  mcdm::PathBPendingSubmit pending = {};
  mcdm::PathBPendingSubmit* pending_batch = nullptr;
  uint32_t completion_slot_offset = 0;
  uint32_t* completion_slot_offsets = nullptr;
  iree_host_size_t completion_slot_count = 0;
  ert_packet* packet = nullptr;
  bool is_pathb_chain = false;
  bool is_pathb_chain_batch = false;
  bool is_pathb_partial_elf = false;
  bool issued = false;
  bool waited = false;
  bool owns_pathb_submission = false;
  iree_status_t status = iree_ok_status();
};

bool iree_hal_amdxdna_native_windows_reserve_completion_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t requested_count,
    size_t start_slot, uint32_t* out_slot_offsets, size_t* out_next_slot) {
  if (!slots_in_use || !out_slot_offsets || requested_count == 0 ||
      requested_count > slot_capacity || start_slot >= slot_capacity ||
      !out_next_slot) {
    return false;
  }
  size_t free_count = 0;
  for (size_t i = 0; i < slot_capacity; ++i) {
    free_count += slots_in_use[i] == 0;
  }
  if (free_count < requested_count) return false;
  size_t reserved_count = 0;
  size_t slot = start_slot;
  for (size_t visited = 0;
       visited < slot_capacity && reserved_count < requested_count;
       ++visited) {
    if (!slots_in_use[slot]) {
      slots_in_use[slot] = 1;
      out_slot_offsets[reserved_count++] =
          static_cast<uint32_t>((slot + 1) * 8);
    }
    slot = (slot + 1) % slot_capacity;
  }
  *out_next_slot = slot;
  return true;
}

bool iree_hal_amdxdna_native_windows_release_completion_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t slot_count,
    const uint32_t* slot_offsets) {
  if (!slots_in_use || !slot_offsets || slot_count == 0) return false;
  for (size_t i = 0; i < slot_count; ++i) {
    const uint32_t offset = slot_offsets[i];
    if (offset < 8 || offset % 8 != 0) return false;
    const size_t slot_index = offset / 8 - 1;
    if (slot_index >= slot_capacity || !slots_in_use[slot_index]) return false;
    for (size_t j = 0; j < i; ++j) {
      if (slot_offsets[j] == offset) return false;
    }
  }
  for (size_t i = 0; i < slot_count; ++i) {
    slots_in_use[slot_offsets[i] / 8 - 1] = 0;
  }
  return true;
}

bool iree_hal_amdxdna_native_windows_completion_slots_are_free(
    const uint8_t* slots_in_use, size_t slot_capacity) {
  if (!slots_in_use && slot_capacity != 0) return false;
  for (size_t i = 0; i < slot_capacity; ++i) {
    if (slots_in_use[i] != 0) return false;
  }
  return true;
}

iree_status_t begin_pathb_submission(
    iree_hal_amdxdna_native_submission_t* submission,
    iree_host_size_t completion_slot_count) {
  iree_hal_amdxdna_native_device_t* device =
      submission->queue->context->device;
  std::unique_lock<std::mutex> lock(device->pathb_context_mutex);
  iree_hal_amdxdna_native_context_t* context = submission->queue->context;
  if (IREE_UNLIKELY(completion_slot_count == 0 ||
                    completion_slot_count >
                        context->pathb_completion_slots_in_use.size())) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM submission requires %" PRIhsz
        " completion slots but the context has %zu",
        completion_slot_count, context->pathb_completion_slots_in_use.size());
  }
  device->pathb_context_cv.wait(lock, [&]() {
    // Command objects and the shared command aperture are mutable staging
    // resources. Keep one native submission in flight until they gain
    // submission-owned snapshots; a batch still issues all of its parents
    // asynchronously using distinct completion slots below.
    return device->pathb_active_submission_count == 0;
  });
  // With a single native submission in flight, its retirement must release
  // every completion slot before the active count reaches zero. Waiting for a
  // leaked slot here would deadlock because no active owner remains to release
  // it or notify this condition variable.
  if (IREE_UNLIKELY(
          !iree_hal_amdxdna_native_windows_completion_slots_are_free(
              context->pathb_completion_slots_in_use.data(),
              context->pathb_completion_slots_in_use.size()))) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna Windows MCDM completion slots remain reserved without an "
        "active submission");
  }
  IREE_RETURN_IF_ERROR(
      activate_pathb_context_for_submit_locked(submission->queue));
  uint32_t* slot_offsets = submission->is_pathb_chain_batch
                               ? submission->completion_slot_offsets
                               : &submission->completion_slot_offset;
  const bool slots_reserved =
      iree_hal_amdxdna_native_windows_reserve_completion_slots(
          context->pathb_completion_slots_in_use.data(),
          context->pathb_completion_slots_in_use.size(), completion_slot_count,
          context->pathb_next_completion_slot, slot_offsets,
          &context->pathb_next_completion_slot);
  IREE_ASSERT(slots_reserved);
  if (IREE_UNLIKELY(!slots_reserved)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna Windows MCDM completion-slot reservation invariant failed");
  }
  submission->completion_slot_count = completion_slot_count;
  ++device->pathb_active_submission_count;
  submission->owns_pathb_submission = true;
  return iree_ok_status();
}

iree_status_t initialize_pathb_completion_slots(
    iree_hal_amdxdna_native_submission_t* submission) {
  const uint32_t* slot_offsets = submission->is_pathb_chain_batch
                                     ? submission->completion_slot_offsets
                                     : &submission->completion_slot_offset;
  iree_hal_amdxdna_native_context_t* context = submission->queue->context;
  mcdm::Error error;
  if (!mcdm::InitializePathBCompletionSlots(
          &context->context, slot_offsets, submission->completion_slot_count,
          &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM completion-slot initialization failed", error);
  }
  return iree_ok_status();
}

void end_pathb_submission(
    iree_hal_amdxdna_native_submission_t* submission) {
  if (!submission || !submission->owns_pathb_submission) return;
  iree_hal_amdxdna_native_device_t* device =
      submission->queue->context->device;
  {
    std::lock_guard<std::mutex> lock(device->pathb_context_mutex);
    submission->owns_pathb_submission = false;
    iree_hal_amdxdna_native_context_t* context = submission->queue->context;
    const uint32_t* slot_offsets = submission->is_pathb_chain_batch
                                       ? submission->completion_slot_offsets
                                       : &submission->completion_slot_offset;
    const bool slots_released =
        iree_hal_amdxdna_native_windows_release_completion_slots(
            context->pathb_completion_slots_in_use.data(),
            context->pathb_completion_slots_in_use.size(),
            submission->completion_slot_count, slot_offsets);
    IREE_ASSERT(slots_released);
    submission->completion_slot_count = 0;
    IREE_ASSERT(device->pathb_active_submission_count > 0);
    --device->pathb_active_submission_count;
  }
  device->pathb_context_cv.notify_all();
}

// Releases a submission gate on pre-issue failures. Once any command reaches
// hardware, native_submission_destroy waits it before releasing the gate.
class PathBSubmissionIssueGuard {
 public:
  explicit PathBSubmissionIssueGuard(
      iree_hal_amdxdna_native_submission_t* submission)
      : submission_(submission) {}
  ~PathBSubmissionIssueGuard() {
    if (!submission_->issued) end_pathb_submission(submission_);
  }

 private:
  iree_hal_amdxdna_native_submission_t* submission_;
};

class PathBSubmissionWaitGuard {
 public:
  explicit PathBSubmissionWaitGuard(
      iree_hal_amdxdna_native_submission_t* submission)
      : submission_(submission) {}
  ~PathBSubmissionWaitGuard() { end_pathb_submission(submission_); }

 private:
  iree_hal_amdxdna_native_submission_t* submission_;
};

void iree_hal_amdxdna_native_submission_destroy(
    iree_hal_amdxdna_native_submission_t* submission);

void set_submission_label(iree_hal_amdxdna_native_submission_t* submission,
                          iree_string_view_t label) {
  submission->label_size = label.size < sizeof(submission->label) - 1
                               ? label.size
                               : sizeof(submission->label) - 1;
  if (submission->label_size) {
    std::memcpy(submission->label, label.data, submission->label_size);
  }
  submission->label[submission->label_size] = '\0';
}

iree_status_t stage_pathb_command_for_submit(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command,
    const WindowsMcdmOpcodeHandler& handler) {
  if (handler.is_chain) {
    IREE_RETURN_IF_ERROR(prepare_pathb_chain_code(queue, command));
  } else {
    IREE_RETURN_IF_ERROR(stage_windows_dpu_code_buffer(queue, command));
  }
  return iree_ok_status();
}

iree_status_t submit_pathb_command_to_kmt(
    iree_hal_amdxdna_native_submission_t* s, uint32_t command_bytes,
    const WindowsMcdmOpcodeHandler& handler, mcdm::Error* error) {
  iree_hal_amdxdna_native_queue_t* queue = s->queue;
  iree_hal_amdxdna_native_command_t* command = s->command;
  ert_packet* packet = command_packet(command);
  if (handler.is_chain) {
    mcdm::PathBChainSubmitInfo chain_info = {};
    chain_info.descriptor_gpu_va = command->pathb_chain_descriptor_gpu_va;
    chain_info.descriptor_bytes = command->pathb_chain_descriptor_bytes;
    chain_info.command_count =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data)->command_count;
    chain_info.first_child_opcode = command->pathb_chain_first_child_opcode;
    if (!mcdm::SubmitPathBChain(
            command->device->api, command->device->device,
            &queue->context->context, command->exec_buffer->buffer, packet,
            command_bytes, chain_info, s->completion_slot_offset,
            &packet->header, &s->pending, error)) {
      mcdm::Error empty_error;
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb chain submit failed",
          error ? *error : empty_error);
    }
    return iree_ok_status();
  }
  if (!mcdm::SubmitPathB(command->device->api, command->device->device,
                         &queue->context->context, command->exec_buffer->buffer,
                         packet, command_bytes, /*command_state=*/3,
                         s->completion_slot_offset, &packet->header,
                         &s->pending, error)) {
    mcdm::Error empty_error;
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb command submit failed",
        error ? *error : empty_error);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_submit_issue(
    iree_hal_amdxdna_native_submission_t* s) {
  iree_hal_amdxdna_native_queue_t* queue = s->queue;
  iree_hal_amdxdna_native_command_t* command = s->command;
  const WindowsMcdmOpcodeHandler& handler = command_opcode_handler(command);
  ert_packet* packet = command_packet(command);
  IREE_RETURN_IF_ERROR(begin_pathb_submission(s, 1));
  PathBSubmissionIssueGuard issue_guard(s);
  IREE_RETURN_IF_ERROR(initialize_pathb_completion_slots(s));
  if (!queue->context->has_command_aperture) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM pathb submit requested without command aperture");
  }
  {
    reset_command_packet_for_start(command);
    IREE_RETURN_IF_ERROR(finalize_windows_dpu_regmap(queue, command));
  }
  const bool is_pathb_chain = handler.is_chain;
  const bool is_pathb_partial_elf = !is_pathb_chain && handler.uses_partial_elf;

  mcdm::Error error;
  {
    for (size_t i = 0; i < command->bound_buffer_count; ++i) {
      const BoundBuffer& bound = command->bound_buffers[i];
      if (!bound.buffer) continue;
      if (is_pathb_partial_elf_control_binding(command, bound)) continue;
      IREE_RETURN_IF_ERROR(materialize_deferred_buffer(bound.buffer));
      char bound_label[32] = {0};
      snprintf(bound_label, sizeof(bound_label), "bound[%zu]", i);
      if (!mcdm::WaitForBufferResidency(
              command->device->api, command->device->device,
              queue->context->context, bound.buffer->buffer, bound_label,
              &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM bound BO residency wait failed", error);
      }
    }
  }
  if (!is_pathb_partial_elf) {
    IREE_RETURN_IF_ERROR(close_pathb_single_code_ranges(queue));
  }
  size_t command_bytes = 0;
  if (!iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
          packet->count, command->exec_buffer->buffer.size, &command_bytes)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Windows MCDM command packet exceeds its execution buffer");
  }
  IREE_RETURN_IF_ERROR(stage_pathb_command_for_submit(queue, command, handler));
  {
    IREE_RETURN_IF_ERROR(materialize_deferred_buffer(command->exec_buffer));
    command->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        command->exec_buffer->buffer.cpu_ptr);
    packet = command_packet(command);
  }
  {
    IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(command));
  }
  if (is_pathb_partial_elf &&
      !mcdm::SyncBuffer(command->device->api, command->device->device,
                        command->exec_buffer->buffer,
                        mcdm::kPathBBoTableOffset, mcdm::kPathBBoTableSize,
                        &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM PARTIAL_ELF BO table publication failed", error);
  }
  // The state-3 packet lives in a normal command BO. Keep its CPU writes
  // ordered before the KMT submit. PARTIAL_ELF packets are copied inline, but
  // their out-of-packet BO table is read from this allocation by the miniport
  // and was published explicitly above.
  {
    if (!handler.skips_exec_buffer_sync()) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
          command->exec_buffer,
          IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
    } else {
      std::atomic_thread_fence(std::memory_order_release);
    }
  }
  IREE_RETURN_IF_ERROR(submit_pathb_command_to_kmt(
      s, static_cast<uint32_t>(command_bytes), handler, &error));
  s->packet = packet;
  s->is_pathb_chain = is_pathb_chain;
  s->is_pathb_partial_elf = is_pathb_partial_elf;
  s->issued = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_submit_all_issue(
    iree_hal_amdxdna_native_submission_t* s) {
  iree_hal_amdxdna_native_queue_t* queue = s->queue;
  iree_hal_amdxdna_native_device_t* device = queue->context->device;
  iree_hal_amdxdna_native_command_t* const* commands = s->commands;
  const iree_host_size_t command_count = s->command_count;

  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (!command_is_pathb_chain(commands[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna native async batch submit currently requires "
          "ERT_CMD_CHAIN parent commands");
    }
  }
  IREE_RETURN_IF_ERROR(begin_pathb_submission(s, command_count));
  PathBSubmissionIssueGuard issue_guard(s);
  IREE_RETURN_IF_ERROR(initialize_pathb_completion_slots(s));
  if (!queue->context->has_command_aperture) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM pathb batch submit requested without command "
        "aperture");
  }

  // Batch issue is no-wait: all parent completions must occupy distinct slots
  // until the collective WaitForPathBSubmits retires them.
  const size_t completion_capacity =
      mcdm::PathBCompletionCapacity(queue->context->context);
  if (IREE_UNLIKELY(command_count > completion_capacity)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM pathb batch submit has %" PRIhsz
        " parents, exceeding the %zu-slot completion ring capacity",
        command_count, completion_capacity);
  }
  IREE_RETURN_IF_ERROR(close_pathb_single_code_ranges(queue));
  mcdm::CommandAperture& aperture = pathb_chain_aperture(queue);
  std::vector<size_t> code_sizes(command_count);
  std::vector<size_t> code_capacities(command_count);
  std::vector<size_t> descriptor_sizes(command_count);
  std::vector<size_t> descriptor_capacities(command_count);
  std::vector<size_t> parent_packet_sizes(command_count);
  size_t chain_code_begin = 0;
  if (IREE_UNLIKELY(!try_align_up_size(
          static_cast<size_t>(queue->context->pathb_persistent_code_bytes),
          0x1000, &chain_code_begin))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Windows MCDM persistent code size cannot be aligned");
  }
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    IREE_RETURN_IF_ERROR(get_pathb_chain_region_sizes(
        commands[i], &code_sizes[i], &descriptor_sizes[i]));
    if (IREE_UNLIKELY(
            !try_align_up_size(code_sizes[i], 0x1000, &code_capacities[i]) ||
            !try_align_up_size(descriptor_sizes[i], 0x1000,
                               &descriptor_capacities[i]))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "amdxdna Windows MCDM chain region size cannot be aligned");
    }
  }

  auto placement_is_current = [&](iree_hal_amdxdna_native_command_t* command) {
    return command->pathb_chain_aperture_generation ==
               queue->context->pathb_chain_aperture_generation &&
           command->pathb_chain_code_aperture_offset >= chain_code_begin;
  };
  auto reset_chain_aperture_generation = [&]() {
    ++queue->context->pathb_chain_aperture_generation;
    if (queue->context->pathb_chain_aperture_generation == 0) {
      queue->context->pathb_chain_aperture_generation = 1;
    }
    queue->context->pathb_chain_code_cursor = chain_code_begin;
    queue->context->pathb_chain_descriptor_cursor =
        static_cast<size_t>(aperture.gpu_va_size);
  };

  if (queue->context->pathb_chain_code_cursor < chain_code_begin ||
      queue->context->pathb_chain_descriptor_cursor == 0) {
    reset_chain_aperture_generation();
  }

  size_t required_code_bytes = 0;
  size_t required_descriptor_bytes = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (placement_is_current(commands[i])) continue;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            required_code_bytes, code_capacities[i], &required_code_bytes) ||
        !iree_host_size_checked_add(required_descriptor_bytes,
                                    descriptor_capacities[i],
                                    &required_descriptor_bytes))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "amdxdna Windows MCDM chain placement size overflows");
    }
  }
  const auto placement_fits = [&]() {
    size_t code_end = 0;
    if (!iree_host_size_checked_add(queue->context->pathb_chain_code_cursor,
                                    required_code_bytes, &code_end)) {
      return false;
    }
    if (code_end > aperture.code_size ||
        required_descriptor_bytes >
            queue->context->pathb_chain_descriptor_cursor) {
      return false;
    }
    const size_t descriptor_begin =
        (queue->context->pathb_chain_descriptor_cursor -
         required_descriptor_bytes) &
        ~size_t{0xFFF};
    return descriptor_begin >= kWindowsDpuChainDescriptorApertureOffset &&
           static_cast<size_t>(aperture.code_offset) + code_end <=
               descriptor_begin;
  };
  if (!placement_fits()) {
    reset_chain_aperture_generation();
    required_code_bytes = 0;
    required_descriptor_bytes = 0;
    for (iree_host_size_t i = 0; i < command_count; ++i) {
      if (IREE_UNLIKELY(!iree_host_size_checked_add(
              required_code_bytes, code_capacities[i],
              &required_code_bytes) ||
          !iree_host_size_checked_add(required_descriptor_bytes,
                                      descriptor_capacities[i],
                                      &required_descriptor_bytes))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM chain placement size overflows");
      }
    }
    if (IREE_UNLIKELY(!placement_fits())) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM path-B batch does not fit in an empty command "
          "aperture generation");
    }
  }

  for (iree_host_size_t i = 0; i < command_count; ++i) {
    iree_hal_amdxdna_native_command_t* command = commands[i];
    if (placement_is_current(command)) continue;
    const size_t code_base =
        align_up_size(queue->context->pathb_chain_code_cursor, 0x1000);
    const size_t descriptor_capacity = descriptor_capacities[i];
    const size_t descriptor_base =
        (queue->context->pathb_chain_descriptor_cursor - descriptor_capacity) &
        ~size_t{0xFFF};
    command->pathb_chain_code_aperture_offset = code_base;
    command->pathb_chain_descriptor_aperture_offset = descriptor_base;
    command->pathb_chain_aperture_generation =
        queue->context->pathb_chain_aperture_generation;
    command->pathb_chain_prepared_valid = false;
    queue->context->pathb_chain_code_cursor =
        code_base + code_capacities[i];
    queue->context->pathb_chain_descriptor_cursor = descriptor_base;
  }

  size_t dirty_code_begin = std::numeric_limits<size_t>::max();
  size_t dirty_code_end = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    dirty_code_begin = std::min(
        dirty_code_begin,
        static_cast<size_t>(commands[i]->pathb_chain_code_aperture_offset));
    dirty_code_end = std::max(
        dirty_code_end,
        static_cast<size_t>(commands[i]->pathb_chain_code_aperture_offset) +
            code_sizes[i]);
  }
  if (dirty_code_begin != std::numeric_limits<size_t>::max()) {
    mcdm::Error acquire_error;
    if (!mcdm::AcquirePathBCodeRange(
            queue->context->device->api, queue->context->device->device,
            &queue->context->context, aperture,
            aperture.code_offset + dirty_code_begin,
            dirty_code_end - dirty_code_begin, &acquire_error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb chain code range acquire failed",
          acquire_error);
    }
  }
  mcdm::Error error;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    {
      reset_command_packet_for_start(command);
      IREE_RETURN_IF_ERROR(finalize_windows_dpu_regmap(queue, command));
    }

    if (!command->pathb_chain_bound_residency_checked) {
      for (size_t i = 0; i < command->bound_buffer_count; ++i) {
        const BoundBuffer& bound = command->bound_buffers[i];
        if (!bound.buffer) continue;
        IREE_RETURN_IF_ERROR(materialize_deferred_buffer(bound.buffer));
        char residency_label[32] = {0};
        snprintf(residency_label, sizeof(residency_label), "batch-bound[%zu]",
                 i);
        if (!mcdm::WaitForBufferResidency(
                command->device->api, command->device->device,
                queue->context->context, bound.buffer->buffer, residency_label,
                &error)) {
          return status_from_mcdm_error(
              "amdxdna Windows MCDM bound BO residency wait failed", error);
        }
      }
      command->pathb_chain_bound_residency_checked = true;
    }

    {
      IREE_RETURN_IF_ERROR(prepare_pathb_chain_code(queue, command, false));
    }
    // CopyAndCommit from each child's control_buffer before opcode-9. Dirty
    // flags may skip a host memcpy optimization later; they must not skip
    // reinstalling the device image dropped after the last consume.
    IREE_RETURN_IF_ERROR(commit_prepared_pathb_chain_code(queue, command));
    {
      IREE_RETURN_IF_ERROR(materialize_deferred_buffer(command->exec_buffer));
      command->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
          command->exec_buffer->buffer.cpu_ptr);
      packet = command_packet(command);
      reset_command_packet_for_start(command);
    }

    // Runlist parents are small host-authored command packets. XRT keeps their
    // Lock2 mapping resident and publishes the packet cache lines directly;
    // avoid a KMT cache operation over the full 4 KiB exec BO for each parent.
    size_t& parent_packet_bytes = parent_packet_sizes[command_index];
    if (!iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
            packet->count, command->exec_buffer->buffer.size,
            &parent_packet_bytes)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "amdxdna Windows MCDM pathb chain parent packet exceeds its "
          "execution buffer");
    }
    if (!mcdm::PublishBufferCpuWrites(command->exec_buffer->buffer,
                                      /*offset=*/0, parent_packet_bytes,
                                      &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb chain parent publication failed", error);
    }
  }
  size_t dirty_descriptor_begin = std::numeric_limits<size_t>::max();
  size_t dirty_descriptor_end = 0;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    const size_t begin =
        static_cast<size_t>(command->pathb_chain_descriptor_aperture_offset);
    const size_t end =
        begin + static_cast<size_t>(command->pathb_chain_descriptor_bytes);
    dirty_descriptor_begin = std::min(dirty_descriptor_begin, begin);
    dirty_descriptor_end = std::max(dirty_descriptor_end, end);
  }
  const size_t descriptor_sync_offset =
      dirty_descriptor_begin == std::numeric_limits<size_t>::max()
          ? 0
          : dirty_descriptor_begin;
  const size_t descriptor_sync_bytes =
      dirty_descriptor_begin == std::numeric_limits<size_t>::max()
          ? 0
          : dirty_descriptor_end - dirty_descriptor_begin;

  // code_bytes=0: child instruction bytes were CopyAndCommit'd above.
  // Opcode-9 is the following per-command loop, which publishes the exact
  // consumed range after the device image is reinstalled.
  IREE_RETURN_IF_ERROR(sync_prepared_pathb_chain_batch(
      queue, command_count, /*code_offset=*/0, /*code_bytes=*/0,
      descriptor_sync_offset, descriptor_sync_bytes));
  // Opcode-9 publishes aperture slots in HW-queue order. It does not copy
  // bytes; CopyAndCommit above reinstalled the device image first.
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    IREE_RETURN_IF_ERROR(publish_pathb_code_write(
        queue,
        aperture.code_offset + command->pathb_chain_code_aperture_offset,
        command->pathb_chain_code_used_size));
    command->pathb_chain_code_dirty = false;
    command->pathb_chain_descriptor_dirty = false;
  }
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    mcdm::PathBChainSubmitInfo chain_info = {};
    chain_info.descriptor_gpu_va = command->pathb_chain_descriptor_gpu_va;
    chain_info.descriptor_bytes = command->pathb_chain_descriptor_bytes;
    chain_info.command_count =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data)->command_count;
    chain_info.first_child_opcode = command->pathb_chain_first_child_opcode;
    if (!mcdm::SubmitPathBChain(
            command->device->api, command->device->device,
            &queue->context->context, command->exec_buffer->buffer, packet,
            static_cast<uint32_t>(parent_packet_sizes[command_index]),
            chain_info, s->completion_slot_offsets[command_index],
            &packet->header, &s->pending_batch[command_index], &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb chain batch submit failed", error);
    }
    s->issued = true;
    s->issued_count = command_index + 1;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_submit_all_wait(
    iree_hal_amdxdna_native_submission_t* s) {
  PathBSubmissionWaitGuard wait_guard(s);
  if (IREE_UNLIKELY(!s->issued || !s->pending_batch || s->issued_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna native submit_all wait on a submission that was not issued");
  }
  iree_hal_amdxdna_native_queue_t* queue = s->queue;
  iree_hal_amdxdna_native_device_t* device = queue->context->device;
  mcdm::Error error;
  if (!mcdm::WaitForPathBSubmits(device->api, device->device,
                                  &queue->context->context, s->pending_batch,
                                  s->issued_count, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb chain batch wait failed", error);
  }
  for (iree_host_size_t command_index = 0; command_index < s->issued_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = s->commands[command_index];
    ert_packet* packet = command_packet(command);
    // The direct command-buffer chain flush invalidates the exact I/O binding
    // ranges once after the whole group completes. Avoid invalidating every
    // bound BO for every native parent chunk here; that duplicates work and was
    // the dominant batched-chain host overhead.
    queue->exec_command_count++;
    if (packet->state == ERT_CMD_STATE_COMPLETED) continue;
    ert_cmd_chain_data* chain_data =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data);
    const mcdm::PathBPendingSubmit& pending =
        s->pending_batch[command_index];
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s batch command %" PRIhsz
        " did not complete: ert state %u (error_index %u, submit_index %u, "
        "fence=%" PRIu64 ", completion_slot=0x%x, code_offset=0x%" PRIx64
        ", code_size=0x%" PRIx64 ", prepared=%u, code_dirty=%u, "
        "descriptor_dirty=%u, generation=%" PRIu64 ")",
        static_cast<int>(s->label_size), s->label, command_index, packet->state,
        chain_data->error_index, chain_data->submit_index, pending.fence_id,
        pending.slot_offset, command->pathb_chain_code_aperture_offset,
        command->pathb_chain_code_used_size,
        command->pathb_chain_prepared_valid ? 1u : 0u,
        command->pathb_chain_code_dirty ? 1u : 0u,
        command->pathb_chain_descriptor_dirty ? 1u : 0u,
        command->pathb_chain_aperture_generation);
  }
  for (iree_host_size_t command_index = 0; command_index < s->issued_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = s->commands[command_index];
    mark_runtime_bindings_mapping_stale(command);
  }
  IREE_RETURN_IF_ERROR(
      refresh_command_output_ranges(s->commands, s->issued_count));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_submit_wait(
    iree_hal_amdxdna_native_submission_t* s) {
  PathBSubmissionWaitGuard wait_guard(s);
  if (IREE_UNLIKELY(!s->issued || !s->packet)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna native submit_wait on a submission that was not issued");
  }
  iree_hal_amdxdna_native_queue_t* queue = s->queue;
  iree_hal_amdxdna_native_command_t* command = s->command;
  ert_packet* packet = s->packet;
  mcdm::Error error;
  if (!mcdm::WaitForPathBSubmits(command->device->api, command->device->device,
                                  &queue->context->context, &s->pending,
                                  /*pending_count=*/1, &error)) {
    return status_from_mcdm_error("amdxdna Windows MCDM pathb wait failed",
                                  error);
  }
  if (!s->is_pathb_partial_elf) {
    if (!mcdm::SubmitPathBApertureSync(
            command->device->api, command->device->device,
            &queue->context->context, queue->context->command_aperture,
            /*offset=*/queue->context->command_aperture.code_offset,
            /*wait_for_cpu=*/true, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb post-dispatch sync failed", error);
    }
  }
  {
    queue->exec_command_count++;
    if (packet->state == ERT_CMD_STATE_COMPLETED) {
      mark_runtime_bindings_mapping_stale(command);
      IREE_RETURN_IF_ERROR(refresh_command_output_ranges(command));
      return iree_ok_status();
    }
    if (command_is_pathb_chain(command)) {
      ert_cmd_chain_data* chain_data =
          reinterpret_cast<ert_cmd_chain_data*>(packet->data);
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "amdxdna %.*s did not complete: ert state %u (error_index %u, "
          "submit_index %u)",
          static_cast<int>(s->label_size), s->label, packet->state,
          chain_data->error_index, chain_data->submit_index);
    }
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s did not complete: ert state %u (fence=%" PRIu64
        ", completion_slot=0x%x, code_offset=0x%" PRIx64
        ", code_capacity=0x%" PRIx64 ", active_code_ranges=%zu)",
        static_cast<int>(s->label_size), s->label, packet->state,
        s->pending.fence_id, s->pending.slot_offset,
        command->pathb_single_code_aperture_offset,
        command->pathb_single_code_aperture_capacity,
        queue->context->pathb_active_single_code_ranges.size());
  }
}

iree_status_t iree_hal_amdxdna_native_queue_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(command);
  iree_hal_amdxdna_native_submission_t submission;
  std::memset(&submission, 0, sizeof(submission));
  submission.host_allocator = queue->context->device->host_allocator;
  submission.queue = queue;
  submission.command = command;
  set_submission_label(&submission, label);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_submit_issue(&submission));
  return iree_hal_amdxdna_native_submit_wait(&submission);
}

iree_status_t iree_hal_amdxdna_native_queue_submit(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(command);
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_submission = nullptr;
  iree_allocator_t host_allocator = queue->context->device->host_allocator;
  iree_hal_amdxdna_native_submission_t* submission = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*submission),
                            reinterpret_cast<void**>(&submission)));
  std::memset(submission, 0, sizeof(*submission));
  submission->host_allocator = host_allocator;
  submission->queue = queue;
  submission->command = command;
  set_submission_label(submission, label);
  iree_status_t status = iree_hal_amdxdna_native_submit_issue(submission);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, submission);
    return status;
  }
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

  iree_allocator_t host_allocator = queue->context->device->host_allocator;
  iree_hal_amdxdna_native_submission_t* submission = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*submission),
                            reinterpret_cast<void**>(&submission)));
  std::memset(submission, 0, sizeof(*submission));
  submission->host_allocator = host_allocator;
  submission->queue = queue;
  submission->command_count = command_count;
  submission->is_pathb_chain_batch = true;
  submission->status = iree_ok_status();
  set_submission_label(submission, label);

  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, command_count, sizeof(*submission->commands),
      reinterpret_cast<void**>(&submission->commands));
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, command_count, sizeof(*submission->pending_batch),
        reinterpret_cast<void**>(&submission->pending_batch));
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, command_count,
        sizeof(*submission->completion_slot_offsets),
        reinterpret_cast<void**>(&submission->completion_slot_offsets));
  }
  if (iree_status_is_ok(status)) {
    std::memcpy(submission->commands, commands,
                command_count * sizeof(*submission->commands));
    std::memset(submission->pending_batch, 0,
                command_count * sizeof(*submission->pending_batch));
    status = iree_hal_amdxdna_native_submit_all_issue(submission);
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
  // The completion queue owns native retirement and always waits indefinitely;
  // API deadlines are handled by its notification wait without releasing
  // command storage that hardware may still reference.
  (void)timeout_ns;
  if (!submission->waited) {
    submission->status =
        submission->is_pathb_chain_batch
            ? iree_hal_amdxdna_native_submit_all_wait(submission)
            : iree_hal_amdxdna_native_submit_wait(submission);
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
  if (submission->is_pathb_chain_batch) {
    *out_ready = true;
    for (iree_host_size_t i = 0; i < submission->issued_count; ++i) {
      if (!mcdm::IsPathBSubmitComplete(submission->queue->context->context,
                                       submission->pending_batch[i])) {
        *out_ready = false;
        break;
      }
    }
    return iree_ok_status();
  }
  // Non-blocking poll of the HW progress fence. Note this reports HW
  // completion; submission_wait is still required to run the device->host
  // output sync.
  *out_ready = mcdm::IsPathBSubmitComplete(submission->queue->context->context,
                                           submission->pending);
  return iree_ok_status();
}

void iree_hal_amdxdna_native_submission_destroy(
    iree_hal_amdxdna_native_submission_t* submission) {
  if (!submission) return;
  // The fence wait is synchronous: if issued-but-not-waited, wait now so the HW
  // is finished touching the command/aperture before the caller frees them.
  if (submission->issued && !submission->waited) {
    submission->status =
        submission->is_pathb_chain_batch
            ? iree_hal_amdxdna_native_submit_all_wait(submission)
            : iree_hal_amdxdna_native_submit_wait(submission);
    submission->waited = true;
  }
  iree_status_ignore(submission->status);
  iree_allocator_free(submission->host_allocator, submission->pending_batch);
  iree_allocator_free(submission->host_allocator,
                      submission->completion_slot_offsets);
  iree_allocator_free(submission->host_allocator, submission->commands);
  iree_allocator_free(submission->host_allocator, submission);
}

iree_status_t iree_hal_amdxdna_native_queue_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(commands);
  if (command_count == 0) return iree_ok_status();

  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (!command_is_pathb_chain(commands[i])) {
      for (iree_host_size_t j = 0; j < command_count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_queue_submit_and_wait(
            queue, commands[j], label));
      }
      return iree_ok_status();
    }
  }

  iree_hal_amdxdna_native_submission_t* submission = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_queue_submit_all(
      queue, commands, command_count, label, &submission));
  iree_status_t status =
      iree_hal_amdxdna_native_submission_wait(submission, UINT64_MAX);
  iree_hal_amdxdna_native_submission_destroy(submission);
  return status;
}

struct iree_hal_amdxdna_native_context_ref_t {
  iree_allocator_t host_allocator;
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdxdna_native_context_t* context;
};

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_resolve_options(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_device_params* out_options,
    iree_byte_span_t* out_device_path_storage,
    iree_hal_amdxdna_native_c_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_device_path_storage = iree_byte_span_empty();
  (void)host_allocator;
  *out_options = *options;
  iree_hal_amdxdna_native_c_power_mode_t native_power_mode;
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
  if (!iree_string_view_is_empty(options->device_path) &&
      !iree_string_view_equal(options->device_path, IREE_SV("default")) &&
      !iree_string_view_equal(options->device_path,
                              IREE_SV("amdxdna://default")) &&
      !iree_string_view_equal(options->device_path, IREE_SV("0"))) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "Windows MCDM amdxdna device path '%.*s' is not available; only the "
        "default adapter is currently supported",
        static_cast<int>(options->device_path.size), options->device_path.data);
  }
  IREE_RETURN_IF_ERROR(parse_power_mode(options->power_mode, &native_power_mode,
                                        out_should_set_power_mode));
  *out_power_mode = native_power_mode;
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
  *out_buffer = nullptr;
  iree_hal_amdxdna_native_buffer_c_type_t native_type;
  IREE_RETURN_IF_ERROR(to_native_buffer_type(type, &native_type));
  iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_alloc_buffer(
      device, size, native_type, &buffer));
  *out_buffer = buffer;
  return iree_ok_status();
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
  iree_hal_amdxdna_native_buffer_sync_direction_t native_direction;
  IREE_RETURN_IF_ERROR(to_native_sync_direction(direction, &native_direction));
  return iree_hal_amdxdna_native_buffer_sync(buffer, native_direction, size,
                                             offset);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction) {
  iree_hal_amdxdna_native_buffer_sync_direction_t native_direction;
  IREE_RETURN_IF_ERROR(to_native_sync_direction(direction, &native_direction));
  return iree_hal_amdxdna_native_buffer_sync_all(buffer, native_direction);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_ensure_allocated(buffer);
}

extern "C" uint64_t iree_hal_amdxdna_native_buffer_c_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_device_address(buffer);
}

extern "C" iree_device_size_t iree_hal_amdxdna_native_buffer_c_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_size(buffer);
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_create_context_ref(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_c_context_image_t* image,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
  *out_context_ref = nullptr;
  iree_hal_amdxdna_native_c_context_image_t native_image;
  IREE_RETURN_IF_ERROR(
      from_c_context_image_type(image->type, &native_image.type));
  native_image.pdi = image->pdi;
  native_image.xclbin = image->xclbin;
  native_image.kernel_name = image->kernel_name;
  iree_hal_amdxdna_native_context_t* raw_context = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_create_context(
      device, &native_image, &raw_context));
  iree_hal_amdxdna_native_context_ref_t* context_ref = nullptr;
  iree_status_t status =
      iree_allocator_malloc(device->host_allocator, sizeof(*context_ref),
                            reinterpret_cast<void**>(&context_ref));
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_context_destroy(raw_context);
    return status;
  }
  std::memset(context_ref, 0, sizeof(*context_ref));
  context_ref->host_allocator = device->host_allocator;
  iree_atomic_ref_count_init(&context_ref->ref_count);
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
    iree_allocator_t host_allocator = context_ref->host_allocator;
    iree_hal_amdxdna_native_context_destroy(context_ref->context);
    iree_allocator_free(host_allocator, context_ref);
  }
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
  iree_hal_amdxdna_native_c_cu_index_t native_cu_index;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_context_open_cu(
      context_ref->context, kernel_name, &native_cu_index));
  out_cu_index->index = native_cu_index.index;
  return iree_ok_status();
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
  *out_command = nullptr;
  iree_hal_amdxdna_native_c_command_opcode_t native_opcode;
  IREE_RETURN_IF_ERROR(from_c_command_opcode(opcode, &native_opcode));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_create(
      device, native_opcode, out_command));
  return iree_ok_status();
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
  iree_hal_amdxdna_native_c_cu_index_t native_cu_index;
  native_cu_index.index = cu_index.index;
  return iree_hal_amdxdna_native_command_set_cu_index(command, native_cu_index);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size) {
  return iree_hal_amdxdna_native_command_add_control_buffer(
      command, control_buffer, control_buffer_size);
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
