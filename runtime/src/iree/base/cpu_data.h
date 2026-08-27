// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_CPU_DATA_H_
#define IREE_BASE_CPU_DATA_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/schemas/cpu_data.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable CPU architecture identifiers used by serialized CPU data.
typedef uint32_t iree_cpu_architecture_t;
typedef enum iree_cpu_architecture_e {
  // No CPU architecture is known.
  IREE_CPU_ARCHITECTURE_UNKNOWN = 0,
  // 32-bit Arm architecture.
  IREE_CPU_ARCHITECTURE_ARM_32 = 1,
  // 64-bit Arm architecture.
  IREE_CPU_ARCHITECTURE_ARM_64 = 2,
  // 32-bit RISC-V architecture.
  IREE_CPU_ARCHITECTURE_RISCV_32 = 3,
  // 64-bit RISC-V architecture.
  IREE_CPU_ARCHITECTURE_RISCV_64 = 4,
  // 32-bit WebAssembly architecture.
  IREE_CPU_ARCHITECTURE_WASM_32 = 5,
  // 64-bit WebAssembly architecture.
  IREE_CPU_ARCHITECTURE_WASM_64 = 6,
  // 32-bit x86 architecture.
  IREE_CPU_ARCHITECTURE_X86_32 = 7,
  // 64-bit x86 architecture.
  IREE_CPU_ARCHITECTURE_X86_64 = 8,
} iree_cpu_architecture_e;

// Explicit CPU architecture and stable capability fields.
//
// The field interpretation is architecture-specific and defined by
// iree/schemas/cpu_data.h. Zero bits are never assumed to be available.
typedef struct iree_cpu_data_t {
  // Architecture used to interpret |fields|.
  iree_cpu_architecture_t architecture;
  // Stable architecture-specific CPU capability fields.
  uint64_t fields[IREE_CPU_DATA_FIELD_COUNT];
} iree_cpu_data_t;

// Availability of a named CPU feature in explicit CPU data.
typedef uint32_t iree_cpu_feature_availability_t;
typedef enum iree_cpu_feature_availability_e {
  // The feature name is not defined for the CPU architecture.
  IREE_CPU_FEATURE_AVAILABILITY_UNKNOWN = 0,
  // The feature is known but not safe to assume available.
  IREE_CPU_FEATURE_AVAILABILITY_UNAVAILABLE = 1,
  // The feature is available for generated code.
  IREE_CPU_FEATURE_AVAILABILITY_AVAILABLE = 2,
} iree_cpu_feature_availability_e;

// Returns the architecture of the current compilation target.
IREE_API_EXPORT iree_cpu_architecture_t iree_cpu_architecture_host(void);

// Returns the canonical name of |architecture| or an empty string if invalid.
IREE_API_EXPORT iree_string_view_t
iree_cpu_architecture_name(iree_cpu_architecture_t architecture);

// Parses a canonical architecture |name| into |out_architecture|.
//
// Returns false and sets |out_architecture| to UNKNOWN if |name| is unknown.
IREE_API_EXPORT bool iree_cpu_architecture_parse(
    iree_string_view_t name, iree_cpu_architecture_t* out_architecture);

// Returns the number of named features defined for |architecture|.
IREE_API_EXPORT iree_host_size_t
iree_cpu_feature_count(iree_cpu_architecture_t architecture);

// Returns the canonical feature name at |ordinal| or an empty string.
IREE_API_EXPORT iree_string_view_t iree_cpu_feature_name(
    iree_cpu_architecture_t architecture, iree_host_size_t ordinal);

// Looks up a canonical feature |name| in explicit |cpu_data|.
IREE_API_EXPORT iree_cpu_feature_availability_t iree_cpu_data_query_feature(
    const iree_cpu_data_t* cpu_data, iree_string_view_t name);

// Returns true if |available| has all named features in |required|.
//
// Architectures must match. Required bits not assigned to named instruction
// features fail closed instead of interpreting future packed scalar fields as
// feature masks.
IREE_API_EXPORT bool iree_cpu_data_satisfies_features(
    const iree_cpu_data_t* available, const iree_cpu_data_t* required);

// Appends a canonical CPU target key such as `x86_64:+avx:+avx2`.
//
// Only named instruction features are represented in the key. The complete
// fixed-width fields remain authoritative device facts for serialization.
IREE_API_EXPORT iree_status_t iree_cpu_data_append_target_key(
    const iree_cpu_data_t* cpu_data, iree_string_builder_t* builder);

// Parses a CPU |target_key| into explicit CPU data.
//
// Feature suffixes may appear in any order and are normalized by
// iree_cpu_data_append_target_key. Unknown and duplicate features are rejected.
IREE_API_EXPORT iree_status_t iree_cpu_data_parse_target_key(
    iree_string_view_t target_key, iree_cpu_data_t* out_cpu_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_CPU_DATA_H_
