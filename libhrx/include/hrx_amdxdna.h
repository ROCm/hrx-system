// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_AMDXDNA_H_
#define HRX_AMDXDNA_H_

#include "hrx_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HRX_AMDXDNA_EXECUTABLE_CREATE_ABI_VERSION_0 0u

typedef uint32_t hrx_amdxdna_context_mode_t;
enum hrx_amdxdna_context_mode_bits_t {
  HRX_AMDXDNA_CONTEXT_MODE_CREATE = 0u,
  HRX_AMDXDNA_CONTEXT_MODE_REUSE = 1u,
};

// One amdxdna executable run. HRX derives backend relocation metadata from the
// XAie transaction; it is not part of the public API.
typedef struct hrx_amdxdna_executable_run_t {
  uint32_t record_length;
  uint32_t abi_version;
  hrx_const_byte_span_t transaction;
  hrx_const_byte_span_t data_payload;
} hrx_amdxdna_executable_run_t;

// One dispatchable entry point. CREATE selects a PDI from an xclbin; REUSE
// dispatches against a context established by another entry point.
typedef struct hrx_amdxdna_executable_entry_point_t {
  uint32_t record_length;
  uint32_t abi_version;
  hrx_string_view_t name;
  hrx_amdxdna_context_mode_t context_mode;
  uint32_t xclbin_ordinal;
  uint32_t pdi_ordinal;
  uint32_t source_line;
  hrx_string_view_t source_file;
  const hrx_amdxdna_executable_run_t* runs;
  size_t run_count;
} hrx_amdxdna_executable_entry_point_t;

// Generic semantic description of an amdxdna executable. Repeated run and
// entry-point records are walked using each record's |record_length| so future
// ABI versions can append fields without changing the stride seen by HRX.
typedef struct hrx_amdxdna_executable_create_params_t {
  uint32_t record_length;
  uint32_t abi_version;
  uint32_t flags;
  uint32_t reserved;
  const hrx_const_byte_span_t* xclbins;
  size_t xclbin_count;
  const hrx_amdxdna_executable_entry_point_t* entry_points;
  size_t entry_point_count;
} hrx_amdxdna_executable_create_params_t;

// Creates and loads an amdxdna executable while keeping the backend package
// representation private to HRX. All input storage is borrowed for the
// duration of the call and may be released after it returns.
HRX_API hrx_status_t hrx_amdxdna_executable_create(
    hrx_device_t device,
    const hrx_amdxdna_executable_create_params_t* params,
    hrx_executable_t* executable);

#ifdef __cplusplus
}
#endif

#endif  // HRX_AMDXDNA_H_
