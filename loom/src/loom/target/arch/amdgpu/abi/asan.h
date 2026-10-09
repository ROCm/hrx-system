// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU ASAN shadow ABI constants used by Loom code generation.
//
// This header intentionally mirrors only the device-visible layout facts needed
// by the compiler. It must not include runtime/src/iree/hal/drivers/amdgpu/abi
// headers because their host side includes HSA headers, while the Loom compiler
// should stay independent from HSA except in execution/tooling code.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ABI_ASAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_ABI_ASAN_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_ASAN_CONFIG_ABI_VERSION 0u

#define LOOM_AMDGPU_ASAN_CONFIG_GLOBAL_NAME "iree_asan_config"

// Marks Loom-native code objects whose dynamic object symbols describe the
// ASAN global layout v0 contract. Each object has its exact logical size,
// begins on a shadow granule, and is followed by a poisoned shadow granule.
#define LOOM_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME \
  "iree_asan_global_layout_v0"

// ABI version for AMDGPU ASAN feedback report payloads.
#define LOOM_AMDGPU_ASAN_REPORT_ABI_VERSION 0u

// Log2 application bytes represented by one shadow byte in the current Loom
// AMDGPU access-check lowering.
#define LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT 3u

// Application bytes represented by one shadow byte in global layout v0.
#define LOOM_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_GRANULE_SIZE \
  (1u << LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT)

// Bitset of loom_amdgpu_asan_config_flag_bits_e values.
typedef uint32_t loom_amdgpu_asan_config_flags_t;

enum loom_amdgpu_asan_config_flag_bits_e {
  // ASAN shadow checking is disabled.
  LOOM_AMDGPU_ASAN_CONFIG_FLAG_NONE = 0u,
  // ASAN shadow checking is enabled for the owning logical device.
  LOOM_AMDGPU_ASAN_CONFIG_FLAG_ENABLED = 1u << 0,
};

// Access kind carried by AMDGPU ASAN feedback reports.
typedef uint32_t loom_amdgpu_asan_access_kind_t;

enum loom_amdgpu_asan_access_kind_e {
  // Access kind was not provided by the instrumentation site.
  LOOM_AMDGPU_ASAN_ACCESS_KIND_UNKNOWN = 0u,
  // Instrumented read access.
  LOOM_AMDGPU_ASAN_ACCESS_KIND_READ = 1u,
  // Instrumented write access.
  LOOM_AMDGPU_ASAN_ACCESS_KIND_WRITE = 2u,
  // Instrumented atomic read-modify-write access.
  LOOM_AMDGPU_ASAN_ACCESS_KIND_ATOMIC = 3u,
};

// Bitset of loom_amdgpu_asan_report_flag_bits_e values.
typedef uint32_t loom_amdgpu_asan_report_flags_t;

enum loom_amdgpu_asan_report_flag_bits_e {
  // No report-level flags are set.
  LOOM_AMDGPU_ASAN_REPORT_FLAG_NONE = 0u,
};

enum loom_amdgpu_asan_config_layout_e {
  // Total byte length of the ASAN configuration record.
  LOOM_AMDGPU_ASAN_CONFIG_BYTE_LENGTH = 96u,
  // Offset of the configuration record length.
  LOOM_AMDGPU_ASAN_CONFIG_RECORD_LENGTH_OFFSET = 0u,
  // Offset of the configuration ABI version.
  LOOM_AMDGPU_ASAN_CONFIG_ABI_VERSION_OFFSET = 4u,
  // Offset of the configuration flags bitfield.
  LOOM_AMDGPU_ASAN_CONFIG_FLAGS_OFFSET = 8u,
  // Offset of the shadow scale shift.
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SCALE_SHIFT_OFFSET = 12u,
  // Offset of the shadow allocation base pointer.
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_BASE_OFFSET = 16u,
  // Offset of the application address window base.
  LOOM_AMDGPU_ASAN_CONFIG_APPLICATION_WINDOW_BASE_OFFSET = 24u,
  // Offset of the application address window byte length.
  LOOM_AMDGPU_ASAN_CONFIG_APPLICATION_WINDOW_SIZE_OFFSET = 32u,
  // Offset of the shadow allocation byte length.
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SIZE_OFFSET = 40u,
  // Offset of the physical shadow slab byte length.
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SLAB_SIZE_OFFSET = 48u,
  // Offset of the first reserved configuration field.
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_0_OFFSET = 56u,
  // Offset of the second reserved configuration field.
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_1_OFFSET = 64u,
  // Offset of the third reserved configuration field.
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_2_OFFSET = 72u,
  // Offset of the fourth reserved configuration field.
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_3_OFFSET = 80u,
  // Offset of the fifth reserved configuration field.
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_4_OFFSET = 88u,
};

enum loom_amdgpu_asan_report_layout_e {
  // Total byte length of the ASAN report payload.
  LOOM_AMDGPU_ASAN_REPORT_BYTE_LENGTH = 64u,
  // Offset of the payload record length.
  LOOM_AMDGPU_ASAN_REPORT_RECORD_LENGTH_OFFSET = 0u,
  // Offset of the payload ABI version.
  LOOM_AMDGPU_ASAN_REPORT_ABI_VERSION_OFFSET = 4u,
  // Offset of the instrumented access kind.
  LOOM_AMDGPU_ASAN_REPORT_ACCESS_KIND_OFFSET = 8u,
  // Offset of the report flags bitfield.
  LOOM_AMDGPU_ASAN_REPORT_FLAGS_OFFSET = 12u,
  // Offset of the application address that failed the check.
  LOOM_AMDGPU_ASAN_REPORT_FAULT_ADDRESS_OFFSET = 16u,
  // Offset of the access size in bytes.
  LOOM_AMDGPU_ASAN_REPORT_ACCESS_SIZE_OFFSET = 24u,
  // Offset of the instrumentation site identifier.
  LOOM_AMDGPU_ASAN_REPORT_SITE_ID_OFFSET = 32u,
  // Offset of the shadow address consulted by the check.
  LOOM_AMDGPU_ASAN_REPORT_SHADOW_ADDRESS_OFFSET = 40u,
  // Offset of the shadow value observed by the check.
  LOOM_AMDGPU_ASAN_REPORT_SHADOW_VALUE_OFFSET = 48u,
  // Offset of the reserved report field.
  LOOM_AMDGPU_ASAN_REPORT_RESERVED_ARRAY_0_OFFSET = 56u,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ABI_ASAN_H_
