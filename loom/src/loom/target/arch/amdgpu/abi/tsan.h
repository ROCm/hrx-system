// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU TSAN feedback ABI constants used by Loom code generation.
//
// This header intentionally mirrors only the device-visible layout facts needed
// by the compiler. It must not include runtime/src/iree/hal/drivers/amdgpu/abi
// headers because their host side includes HSA headers, while the Loom compiler
// should stay independent from HSA except in execution/tooling code.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ABI_TSAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_ABI_TSAN_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION 0u

#define LOOM_AMDGPU_TSAN_CONFIG_GLOBAL_NAME "iree_tsan_config"

enum loom_amdgpu_tsan_config_layout_e {
  // Total byte length of the TSAN runtime config record.
  LOOM_AMDGPU_TSAN_CONFIG_BYTE_LENGTH = 96u,
  // Offset of the TSAN config flags bitfield.
  LOOM_AMDGPU_TSAN_CONFIG_FLAGS_OFFSET = 8u,
  // Offset of the TSAN memory granule shift.
  LOOM_AMDGPU_TSAN_CONFIG_MEMORY_GRANULE_SHIFT_OFFSET = 12u,
  // Offset of the TSAN shadow allocation base pointer.
  LOOM_AMDGPU_TSAN_CONFIG_SHADOW_BASE_OFFSET = 16u,
  // Offset of the TSAN dispatch shadow byte stride.
  LOOM_AMDGPU_TSAN_CONFIG_DISPATCH_SHADOW_STRIDE_OFFSET = 32u,
  // Offset of the TSAN workgroup shadow byte stride.
  LOOM_AMDGPU_TSAN_CONFIG_WORKGROUP_SHADOW_STRIDE_OFFSET = 40u,
  // Offset of the TSAN workgroup capacity.
  LOOM_AMDGPU_TSAN_CONFIG_WORKGROUP_CAPACITY_OFFSET = 48u,
  // Offset of the TSAN shadow entry byte size.
  LOOM_AMDGPU_TSAN_CONFIG_SHADOW_ENTRY_SIZE_OFFSET = 52u,
  // Offset of the host-observed queue AQL ring base pointer.
  LOOM_AMDGPU_TSAN_CONFIG_QUEUE_AQL_BASE_OFFSET = 56u,
  // Offset of the host packet ID queue AQL ring slot mask.
  LOOM_AMDGPU_TSAN_CONFIG_QUEUE_AQL_SLOT_MASK_OFFSET = 64u,
  // Offset of the queue-local TSAN state pointer.
  LOOM_AMDGPU_TSAN_CONFIG_QUEUE_STATE_BASE_OFFSET = 72u,
  // Offset of the queue-local shadow slot count.
  LOOM_AMDGPU_TSAN_CONFIG_SHADOW_SLOT_COUNT_OFFSET = 80u,
};

enum loom_amdgpu_tsan_config_flag_bits_e {
  // TSAN checking is enabled for the owning runtime.
  LOOM_AMDGPU_TSAN_CONFIG_FLAG_ENABLED = 1u << 0,
  // Queue-local TSAN state is available.
  LOOM_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_STATE = 1u << 1,
};

enum loom_amdgpu_tsan_queue_state_layout_e {
  // Total byte length of one queue-local TSAN state record.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_BYTE_LENGTH = 112u,
  // Offset of the queue shadow allocation base pointer.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_SHADOW_BASE_OFFSET = 56u,
  // Offset of the queue shadow allocation byte length.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_SHADOW_SIZE_OFFSET = 64u,
  // Offset of the per-dispatch shadow stride.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_DISPATCH_SHADOW_STRIDE_OFFSET = 72u,
  // Offset of the per-workgroup shadow stride.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_WORKGROUP_SHADOW_STRIDE_OFFSET = 80u,
  // Offset of the queue workgroup capacity.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_WORKGROUP_CAPACITY_OFFSET = 88u,
  // Offset of the queue shadow entry byte size.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_SHADOW_ENTRY_SIZE_OFFSET = 92u,
  // Offset of the queue memory granule shift.
  LOOM_AMDGPU_TSAN_QUEUE_STATE_MEMORY_GRANULE_SHIFT_OFFSET = 96u,
};

enum loom_amdgpu_tsan_shadow_layout_e {
  // Byte length of each TSAN local-memory shadow entry.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_BYTE_LENGTH = 8u,
  // Per-workgroup shadow header byte length.
  LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_BYTE_LENGTH = 8u,
  // Offset of the low 32-bit per-workgroup barrier epoch inside the shadow
  // header. The upper 32 bits of the 8-byte header are reserved.
  LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_EPOCH_OFFSET = 0u,
  // Bit offset of the encoded prior access kind in a shadow entry.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_SHIFT = 0u,
  // Bit count of the encoded prior access kind in a shadow entry.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT = 3u,
  // Bit offset of the encoded prior linear local workitem id.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT = 3u,
  // Bit count of the encoded prior linear local workitem id.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT = 10u,
  // Bit offset of the encoded workgroup barrier epoch.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT = 13u,
  // Bit count of the encoded workgroup barrier epoch.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT = 10u,
  // Bit offset of the encoded dispatch generation low bits.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT = 23u,
  // Bit count of the encoded dispatch generation low bits.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT = 20u,
  // Bit offset of the encoded instrumentation site low bits.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT = 43u,
  // Bit count of the encoded instrumentation site low bits.
  LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT = 21u,
};

// Encoded access kind stored in each TSAN shadow entry.
typedef uint32_t loom_amdgpu_tsan_shadow_access_kind_t;

enum loom_amdgpu_tsan_shadow_access_kind_e {
  // No prior access has been recorded for the shadow entry.
  LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY = 0u,
  // Prior access was a non-atomic read.
  LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ = 1u,
  // Prior access was a non-atomic write.
  LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE = 2u,
  // Prior access was a non-atomic read-modify-write observation.
  LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE = 3u,
  // Prior access was an atomic memory operation.
  LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC = 4u,
};

// TSAN check kind carried by AMDGPU TSAN feedback reports.
typedef uint32_t loom_amdgpu_tsan_check_kind_t;

enum loom_amdgpu_tsan_check_kind_e {
  // Check kind was not provided by the instrumentation site.
  LOOM_AMDGPU_TSAN_CHECK_KIND_UNKNOWN = 0u,
  // Conflicting accesses to the same memory location were observed without
  // intervening synchronization.
  LOOM_AMDGPU_TSAN_CHECK_KIND_DATA_RACE = 1u,
};

// Memory space containing the observed address in AMDGPU TSAN reports.
typedef uint32_t loom_amdgpu_tsan_memory_space_t;

enum loom_amdgpu_tsan_memory_space_e {
  // Memory space was not provided by the instrumentation site.
  LOOM_AMDGPU_TSAN_MEMORY_SPACE_UNKNOWN = 0u,
  // Global device memory.
  LOOM_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL = 1u,
  // Workgroup-shared LDS memory.
  LOOM_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP = 2u,
  // Per-workitem private memory.
  LOOM_AMDGPU_TSAN_MEMORY_SPACE_PRIVATE = 3u,
};

// Access kind participating in an AMDGPU TSAN report.
typedef uint32_t loom_amdgpu_tsan_access_kind_t;

enum loom_amdgpu_tsan_access_kind_e {
  // Access kind was not provided by the instrumentation site.
  LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN = 0u,
  // Instrumented read access.
  LOOM_AMDGPU_TSAN_ACCESS_KIND_READ = 1u,
  // Instrumented write access.
  LOOM_AMDGPU_TSAN_ACCESS_KIND_WRITE = 2u,
  // Instrumented read-modify-write access.
  LOOM_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE = 3u,
  // Instrumented atomic access.
  LOOM_AMDGPU_TSAN_ACCESS_KIND_ATOMIC = 4u,
};

// Bitset of loom_amdgpu_tsan_report_flag_bits_e values.
typedef uint32_t loom_amdgpu_tsan_report_flags_t;

enum loom_amdgpu_tsan_report_flag_bits_e {
  // No report-level flags are set.
  LOOM_AMDGPU_TSAN_REPORT_FLAG_NONE = 0u,
  // Current access was an atomic memory operation.
  LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_ATOMIC = 1u << 0,
  // Prior access was an atomic memory operation.
  LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_ATOMIC = 1u << 1,
  // Prior workitem id is stored as a linear local id in prior_workitem_id_x.
  LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR = 1u << 2,
  // Current workitem id is stored as a linear local id in
  // current_workitem_id_x.
  LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR = 1u << 3,
};

enum loom_amdgpu_tsan_report_layout_e {
  // Total byte length of the TSAN report payload.
  LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH = 120u,
  // Offset of the payload record length.
  LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET = 0u,
  // Offset of the payload ABI version.
  LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET = 4u,
  // Offset of the TSAN check kind.
  LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET = 8u,
  // Offset of the report flags bitfield.
  LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET = 12u,
  // Offset of the reported memory space.
  LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET = 16u,
  // Offset of the reporting workitem access kind.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_ACCESS_KIND_OFFSET = 20u,
  // Offset of the prior workitem access kind.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_ACCESS_KIND_OFFSET = 24u,
  // Offset of the access size in bytes.
  LOOM_AMDGPU_TSAN_REPORT_ACCESS_SIZE_OFFSET = 28u,
  // Offset of the current instrumentation site identifier.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_SITE_ID_OFFSET = 32u,
  // Offset of the prior instrumentation site identifier.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_SITE_ID_OFFSET = 40u,
  // Offset of the address or memory-space-relative byte offset that raced.
  LOOM_AMDGPU_TSAN_REPORT_MEMORY_ADDRESS_OFFSET = 48u,
  // Offset of the shadow address consulted by the check.
  LOOM_AMDGPU_TSAN_REPORT_SHADOW_ADDRESS_OFFSET = 56u,
  // Offset of the shadow value observed by the check.
  LOOM_AMDGPU_TSAN_REPORT_SHADOW_VALUE_OFFSET = 64u,
  // Offset of the current workgroup X coordinate.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_X_OFFSET = 72u,
  // Offset of the current workgroup Y coordinate.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Y_OFFSET = 76u,
  // Offset of the current workgroup Z coordinate.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Z_OFFSET = 80u,
  // Offset of the current workitem X coordinate.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_X_OFFSET = 84u,
  // Offset of the current workitem Y coordinate.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Y_OFFSET = 88u,
  // Offset of the current workitem Z coordinate.
  LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Z_OFFSET = 92u,
  // Offset of the prior workgroup X coordinate.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_X_OFFSET = 96u,
  // Offset of the prior workgroup Y coordinate.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Y_OFFSET = 100u,
  // Offset of the prior workgroup Z coordinate.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Z_OFFSET = 104u,
  // Offset of the prior workitem X coordinate.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_X_OFFSET = 108u,
  // Offset of the prior workitem Y coordinate.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Y_OFFSET = 112u,
  // Offset of the prior workitem Z coordinate.
  LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Z_OFFSET = 116u,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ABI_TSAN_H_
