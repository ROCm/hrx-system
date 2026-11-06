// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-visible UBSAN report ABI shared by AMDGPU host code and instrumented
// device code.

#ifndef IREE_HAL_DRIVER_AMDGPU_ABI_UBSAN_H_
#define IREE_HAL_DRIVER_AMDGPU_ABI_UBSAN_H_

#include "iree/hal/driver/amdgpu/abi/common.h"

// ABI version for |iree_hal_amdgpu_ubsan_report_t|.
#define IREE_HAL_AMDGPU_UBSAN_REPORT_ABI_VERSION_0 0u

// UBSAN check kind that triggered a report.
typedef uint32_t iree_hal_amdgpu_ubsan_check_kind_t;
enum iree_hal_amdgpu_ubsan_check_kind_bits_t {
  // Check kind was not provided by the instrumentation site.
  IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_UNKNOWN = 0u,
  // Integer arithmetic overflowed its represented domain.
  IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_INTEGER_OVERFLOW = 1u,
  // An integer divisor was zero.
  IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_DIVIDE_BY_ZERO = 2u,
  // A memory address violated its required alignment.
  IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_ALIGNMENT = 3u,
  // A floating-point value violated a not-NaN contract.
  IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_FLOAT_NAN_CONTRACT = 4u,
  // Execution reached a source location declared unreachable.
  IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_UNREACHABLE = 5u,
  // A runtime assertion predicate evaluated to false.
  IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_ASSERTION = 6u,
};

// Bitfield specifying properties of a UBSAN report.
typedef uint32_t iree_hal_amdgpu_ubsan_report_flags_t;
enum iree_hal_amdgpu_ubsan_report_flag_bits_t {
  IREE_HAL_AMDGPU_UBSAN_REPORT_FLAG_NONE = 0u,
};

// UBSAN diagnostic payload carried by feedback packets of kind UBSAN.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_ubsan_report_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Runtime check kind that triggered the report.
  iree_hal_amdgpu_ubsan_check_kind_t check_kind;
  // Flags describing optional report fields.
  iree_hal_amdgpu_ubsan_report_flags_t flags;
  // Compiler-assigned instrumentation site identifier.
  uint64_t site_id;
  // Check-specific first operand or value.
  uint64_t operand0;
  // Check-specific second operand or value.
  uint64_t operand1;
  // Reserved for future report fields. Must be zero.
  uint64_t reserved[3];
} iree_hal_amdgpu_ubsan_report_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_ubsan_report_t) == 64,
                          "UBSAN report payload size is part of the ABI");

#endif  // IREE_HAL_DRIVER_AMDGPU_ABI_UBSAN_H_
