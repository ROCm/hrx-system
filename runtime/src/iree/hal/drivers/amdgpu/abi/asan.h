// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-visible ASAN configuration ABI shared by AMDGPU host code and
// instrumented device code.

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_ASAN_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_ASAN_H_

#include "iree/hal/drivers/amdgpu/abi/common.h"

// Name of the executable global containing |iree_hal_amdgpu_asan_config_t|.
#define IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME "iree_asan_config"

// ABI version for |iree_hal_amdgpu_asan_config_t|.
#define IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0 0u

// ABI version for |iree_hal_amdgpu_asan_report_t|.
#define IREE_HAL_AMDGPU_ASAN_REPORT_ABI_VERSION_0 0u

// Bitfield specifying properties of the ASAN configuration.
typedef uint32_t iree_hal_amdgpu_asan_config_flags_t;
enum iree_hal_amdgpu_asan_config_flag_bits_t {
  IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_NONE = 0u,
  // ASAN shadow checking is enabled for the owning logical device.
  IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED = 1u << 0,
};

// ASAN access kind that triggered a report.
typedef uint32_t iree_hal_amdgpu_asan_access_kind_t;
enum iree_hal_amdgpu_asan_access_kind_bits_t {
  // Access kind was not provided by the instrumentation site.
  IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_UNKNOWN = 0u,
  // Instrumented read access.
  IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ = 1u,
  // Instrumented write access.
  IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE = 2u,
  // Instrumented atomic read-modify-write access.
  IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_ATOMIC = 3u,
};

// Bitfield specifying properties of an ASAN report.
typedef uint32_t iree_hal_amdgpu_asan_report_flags_t;
enum iree_hal_amdgpu_asan_report_flag_bits_t {
  IREE_HAL_AMDGPU_ASAN_REPORT_FLAG_NONE = 0u,
};

// Runtime-published ASAN configuration read by instrumented device code.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_asan_config_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Flags describing enabled ASAN runtime facilities.
  iree_hal_amdgpu_asan_config_flags_t flags;
  // Log2 application bytes represented by one shadow byte.
  uint32_t shadow_scale_shift;
  // Device-visible base added to shifted application addresses.
  uint64_t shadow_base;
  // First application address covered by the shadow map.
  uint64_t application_window_base;
  // Number of application bytes covered by the shadow map.
  uint64_t application_window_size;
  // Number of shadow bytes reserved by the shadow map.
  uint64_t shadow_size;
  // Physical shadow slab size used by the runtime mapper.
  uint64_t shadow_slab_size;
  // Reserved for future ASAN runtime state. Must be zero.
  uint64_t reserved[5];
} iree_hal_amdgpu_asan_config_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_asan_config_t) == 96,
                          "ASAN config size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_asan_config_t, shadow_base) == 16,
    "ASAN config shadow fields must follow the fixed header");

// ASAN diagnostic payload carried by feedback packets of kind ASAN.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_asan_report_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Instrumented access kind that triggered the report.
  iree_hal_amdgpu_asan_access_kind_t access_kind;
  // Flags describing optional report fields.
  iree_hal_amdgpu_asan_report_flags_t flags;
  // Application address that failed the ASAN check.
  uint64_t fault_address;
  // Access size in bytes.
  uint64_t access_size;
  // Compiler-assigned instrumentation site identifier.
  uint64_t site_id;
  // Shadow address consulted by the check, or 0 when unavailable.
  uint64_t shadow_address;
  // Shadow value observed by the check, or 0 when unavailable.
  uint64_t shadow_value;
  // Reserved for future report fields. Must be zero.
  uint64_t reserved[1];
} iree_hal_amdgpu_asan_report_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_asan_report_t) == 64,
                          "ASAN report payload size is part of the ABI");

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_ASAN_H_
