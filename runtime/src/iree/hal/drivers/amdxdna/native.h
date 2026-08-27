// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_NATIVE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_NATIVE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// OS-neutral native DDI used by the amdxdna HAL. Platform implementations own
// the concrete handles and keep OS/shim details behind these opaque types.
typedef struct iree_hal_amdxdna_native_device_t
    iree_hal_amdxdna_native_device_t;
typedef struct iree_hal_amdxdna_native_buffer_t
    iree_hal_amdxdna_native_buffer_t;
typedef struct iree_hal_amdxdna_native_context_ref_t
    iree_hal_amdxdna_native_context_ref_t;
typedef struct iree_hal_amdxdna_native_context_t
    iree_hal_amdxdna_native_context_t;
typedef struct iree_hal_amdxdna_native_queue_t iree_hal_amdxdna_native_queue_t;
typedef struct iree_hal_amdxdna_native_command_t
    iree_hal_amdxdna_native_command_t;
typedef struct iree_hal_amdxdna_native_submission_t
    iree_hal_amdxdna_native_submission_t;

typedef enum iree_hal_amdxdna_native_buffer_sync_direction_t {
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE = 0,
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST = 1,
} iree_hal_amdxdna_native_buffer_sync_direction_t;

typedef enum iree_hal_amdxdna_native_buffer_c_type_t {
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY = 0,
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE = 1,
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION = 2,
} iree_hal_amdxdna_native_buffer_c_type_t;

typedef struct iree_hal_amdxdna_native_c_cu_index_t {
  uint32_t index;
} iree_hal_amdxdna_native_c_cu_index_t;

typedef enum iree_hal_amdxdna_native_c_command_opcode_t {
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU = 0,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU = 1,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF = 2,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN = 3,
} iree_hal_amdxdna_native_c_command_opcode_t;

typedef enum iree_hal_amdxdna_native_c_power_mode_t {
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT = 0,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_LOW = 1,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_MEDIUM = 2,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_HIGH = 3,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_TURBO = 4,
} iree_hal_amdxdna_native_c_power_mode_t;

typedef enum iree_hal_amdxdna_native_c_context_image_type_t {
  IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_PDI = 0,
  IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_XCLBIN = 1,
} iree_hal_amdxdna_native_c_context_image_type_t;

enum iree_hal_amdxdna_native_c_context_image_model_bits_t {
  IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_PDI = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_XCLBIN = 1u << 1,
};

enum iree_hal_amdxdna_native_c_dispatch_model_bits_t {
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_CU = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU = 1u << 1,
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF = 1u << 2,
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_COMMAND_CHAIN = 1u << 3,
};

enum iree_hal_amdxdna_native_c_completion_model_bits_t {
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_SYNCHRONOUS_WAIT = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_NATIVE_FENCE = 1u << 1,
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_PROGRESS_FENCE = 1u << 2,
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_COMPLETION_SLOT = 1u << 3,
};

typedef enum iree_hal_amdxdna_native_c_command_chain_status_t {
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_UNKNOWN = 0,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_ENABLED_BY_DEFAULT = 1,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_KNOWN_BAD_STACK = 4,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_OLD_FIRMWARE = 5,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_CHAIN_STATUS_DISABLED_UNIDENTIFIED_STACK =
      6,
} iree_hal_amdxdna_native_c_command_chain_status_t;

typedef struct iree_hal_amdxdna_native_c_driver_stack_t {
  iree_string_view_t driver_version;
  iree_string_view_t driver_srcversion;
  bool has_firmware_version;
  uint32_t firmware_major;
  uint32_t firmware_minor;
  uint32_t firmware_patch;
  uint32_t firmware_build;
  bool has_pci_ids;
  uint32_t pci_vendor_id;
  uint32_t pci_device_id;
  uint32_t pci_revision_id;
} iree_hal_amdxdna_native_c_driver_stack_t;

typedef struct iree_hal_amdxdna_native_c_device_caps_t {
  uint32_t max_effective_queues;
  uint32_t max_command_chain_slots;
  // Maximum number of native child commands that the device-level chain cache
  // may retain. Zero selects the common conservative default.
  uint32_t max_cached_chain_child_commands;
  // Concurrent hardware-context budget for this NPU, i.e. how many native
  // hardware contexts may be kept alive at once before the driver refuses to
  // create more. The amdxdna KMD exposes no query for this ceiling, so backends
  // derive it from the device architecture (see
  // iree_hal_amdxdna_hardware_context_budget_for_arch). 0 means "unknown"; the
  // context cache then falls back to a conservative default.
  uint32_t max_hardware_contexts;
  uint32_t context_image_models;
  uint32_t dispatch_models;
  uint32_t completion_models;
  // Host-only native buffers may be retained and reused after their HAL buffer
  // wrapper is released. False requires immediate native-buffer destruction.
  bool supports_host_buffer_reuse;
  // Native submission owns publication of command control code. Common code
  // must not publish the source control BO when this is true.
  bool native_owns_control_code_publication;
  // Native issue may return before completion. Common code must retain
  // submission resources and defer signal semaphores until native completion.
  // This does not imply multiple hardware queues or concurrent queue access.
  bool submit_completion_is_deferred;
  bool supports_external_buffer_import;
  bool supports_external_buffer_export;
  bool supports_real_multi_queue;
  iree_hal_amdxdna_native_c_command_opcode_t default_dispatch_opcode;
  iree_hal_amdxdna_native_c_command_chain_status_t command_chain_status;
  iree_hal_amdxdna_native_c_driver_stack_t driver_stack;
} iree_hal_amdxdna_native_c_device_caps_t;

// Maps an NPU architecture name (e.g. "Phoenix", "Strix", "Strix Halo",
// "Krackan") to a soft concurrent hardware-context budget: how many native
// contexts the cache aims to keep alive. The KMD exposes no query for the true
// ceiling, which also varies with the part and its array partitioning, so this
// is a target, not a hard limit -- the context cache evicts and retries on
// creation failure to back off to whatever the driver accepts. The budget is
// architecture-keyed (identical on Linux and Windows for a given part). Returns
// 0 for an unknown architecture, signaling the caller to use a default.
static inline uint32_t iree_hal_amdxdna_hardware_context_budget_for_arch(
    iree_string_view_t arch) {
  // Phoenix (npu1).
  if (iree_string_view_starts_with(arch, IREE_SV("Phoenix"))) {
    return 6;
  }
  // Strix / Strix Halo / Krackan (npu2 family).
  if (iree_string_view_starts_with(arch, IREE_SV("Strix")) ||
      iree_string_view_starts_with(arch, IREE_SV("Krackan"))) {
    return 32;
  }
  return 0;  // unknown architecture -> caller falls back to a default
}

// Maps AMD PCI IDs to the same architecture names Linux derives from sysfs.
// revision_id is reserved for future disambiguation; device_id is sufficient
// today for Phoenix (0x1502) vs Strix/Krackan-family (0x17f0) parts.
static inline iree_string_view_t iree_hal_amdxdna_npu_arch_for_pci(
    uint32_t vendor_id, uint32_t device_id, uint32_t revision_id) {
  (void)revision_id;
  if (vendor_id != 0x1022u) return iree_string_view_empty();
  switch (device_id) {
    case 0x1502u:
      return IREE_SV("Phoenix");
    case 0x17f0u:
      return IREE_SV("Strix");
    default:
      return iree_string_view_empty();
  }
}

static inline uint32_t iree_hal_amdxdna_hardware_context_budget_for_pci(
    uint32_t vendor_id, uint32_t device_id, uint32_t revision_id) {
  return iree_hal_amdxdna_hardware_context_budget_for_arch(
      iree_hal_amdxdna_npu_arch_for_pci(vendor_id, device_id, revision_id));
}

typedef struct iree_hal_amdxdna_native_c_context_image_t {
  iree_hal_amdxdna_native_c_context_image_type_t type;
  iree_const_byte_span_t pdi;
  iree_const_byte_span_t xclbin;
  iree_string_view_t kernel_name;
} iree_hal_amdxdna_native_c_context_image_t;

iree_status_t iree_hal_amdxdna_native_device_c_resolve_options(
    const struct iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    struct iree_hal_amdxdna_device_params* out_options,
    iree_byte_span_t* out_device_path_storage,
    iree_hal_amdxdna_native_c_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode);

iree_status_t iree_hal_amdxdna_native_device_c_create(
    const struct iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device);

void iree_hal_amdxdna_native_device_c_destroy(
    iree_hal_amdxdna_native_device_t* device);

iree_status_t iree_hal_amdxdna_native_device_c_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_power_mode_t power_mode);

iree_status_t iree_hal_amdxdna_native_device_c_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps);

iree_status_t iree_hal_amdxdna_native_device_c_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_t** out_buffer);

void iree_hal_amdxdna_native_buffer_c_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_buffer_c_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr);

iree_status_t iree_hal_amdxdna_native_buffer_c_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_device_size_t size, iree_device_size_t offset);

iree_status_t iree_hal_amdxdna_native_buffer_c_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction);

iree_status_t iree_hal_amdxdna_native_buffer_c_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer);

uint64_t iree_hal_amdxdna_native_buffer_c_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_device_size_t iree_hal_amdxdna_native_buffer_c_size(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_device_c_create_context_ref(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_c_context_image_t* image,
    bool* out_context_pool_exhausted,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref);

iree_hal_amdxdna_native_context_ref_t*
iree_hal_amdxdna_native_context_ref_retain(
    iree_hal_amdxdna_native_context_ref_t* context_ref);

void iree_hal_amdxdna_native_context_ref_release(
    iree_hal_amdxdna_native_context_ref_t* context_ref);

iree_hal_amdxdna_native_context_t* iree_hal_amdxdna_native_context_ref_borrow(
    iree_hal_amdxdna_native_context_ref_t* context_ref);

iree_status_t iree_hal_amdxdna_native_context_ref_open_cu(
    iree_hal_amdxdna_native_context_ref_t* context_ref,
    iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_c_cu_index_t* out_cu_index);

iree_status_t iree_hal_amdxdna_native_context_ref_close_single_aperture_session(
    iree_hal_amdxdna_native_context_ref_t* context_ref);

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_ref_queue(
    iree_hal_amdxdna_native_context_ref_t* context_ref);

uint64_t iree_hal_amdxdna_native_queue_c_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue);

iree_status_t iree_hal_amdxdna_native_device_c_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots);

iree_host_size_t iree_hal_amdxdna_native_command_c_arg_binding_capacity(void);

iree_status_t iree_hal_amdxdna_native_command_c_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_t** out_command);

void iree_hal_amdxdna_native_command_c_destroy(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_c_reset(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_c_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_c_cu_index_t cu_index);

iree_status_t iree_hal_amdxdna_native_command_c_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size);

iree_status_t iree_hal_amdxdna_native_command_c_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value);

iree_status_t iree_hal_amdxdna_native_command_c_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value);

iree_status_t iree_hal_amdxdna_native_command_c_update_arg_64(
    iree_hal_amdxdna_native_command_t* command, iree_host_size_t arg_index,
    uint64_t value);

iree_status_t iree_hal_amdxdna_native_command_c_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset);

iree_status_t iree_hal_amdxdna_native_command_c_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, iree_host_size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size);

iree_status_t iree_hal_amdxdna_native_command_c_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_c_mark_code_dirty(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_c_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_c_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count);

iree_status_t iree_hal_amdxdna_native_queue_c_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label);

iree_status_t iree_hal_amdxdna_native_queue_c_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label);

iree_status_t iree_hal_amdxdna_native_queue_c_submit(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission);

iree_status_t iree_hal_amdxdna_native_queue_c_submit_all(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission);

iree_status_t iree_hal_amdxdna_native_submission_c_wait(
    iree_hal_amdxdna_native_submission_t* submission, uint64_t timeout_ns);

iree_status_t iree_hal_amdxdna_native_submission_c_query(
    iree_hal_amdxdna_native_submission_t* submission, bool* out_ready);

void iree_hal_amdxdna_native_submission_c_destroy(
    iree_hal_amdxdna_native_submission_t* submission);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_NATIVE_H_
