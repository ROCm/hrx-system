// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_AMDXDNA_H_
#define HRX_AMDXDNA_H_

#include "hrx_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0 0u
#define HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_1 1u
#define HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0 0u
#define HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_1 1u
#define HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0 0u
#define HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_1 1u
#define HRX_AMDXDNA_CONTROL_PARAMETER_ABI_VERSION_0 0u

// `{0}` is valid C but GCC C++ -Wmissing-field-initializers rejects it when
// later members are omitted. Value-init with `{}` zeroes every field in C++.
#ifdef __cplusplus
#define HRX_AMDXDNA_STRUCT_ZERO_INIT \
  {                                  \
  }
#else
#define HRX_AMDXDNA_STRUCT_ZERO_INIT {0}
#endif

// HAL executable target advertised by amdxdna devices. Pass these to
// hrx_executable_load_data / hrx_executable_load_file. This is the device
// target identity, not the XADX/PDI container format string.
#define HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY "amdxdna"
#define HRX_AMDXDNA_EXECUTABLE_TARGET_KEY "amdxdna"

typedef uint32_t hrx_amdxdna_context_mode_t;
enum hrx_amdxdna_context_mode_bits_t {
  HRX_AMDXDNA_CONTEXT_MODE_CREATE = 0u,
  HRX_AMDXDNA_CONTEXT_MODE_REUSE = 1u,
};

// One relocation of a standalone PDI device address into a transaction.
// |transaction_offset| names the first byte of a little-endian 64-bit address
// in the run transaction. |pdi_ordinal| indexes create_params.pdis.
typedef struct hrx_amdxdna_pdi_relocation_t {
  uint32_t transaction_offset;
  uint32_t pdi_ordinal;
  int64_t addend;
} hrx_amdxdna_pdi_relocation_t;

// One relocation of the internally staged control parameter buffer address
// into a transaction.
typedef struct hrx_amdxdna_control_parameter_relocation_t {
  uint32_t transaction_offset;
  int64_t addend;
} hrx_amdxdna_control_parameter_relocation_t;

typedef uint32_t hrx_amdxdna_control_parameter_kind_t;
enum hrx_amdxdna_control_parameter_kind_bits_t {
  // Firmware propagates the slot to storage read by AIE core code.
  HRX_AMDXDNA_CONTROL_PARAMETER_KIND_CORE = 0u,
  // Firmware uses the slot to update an address-bearing register.
  HRX_AMDXDNA_CONTROL_PARAMETER_KIND_ADDRESS = 1u,
};

// One scalar staged from the dense dispatch constants block into the firmware
// control parameter buffer. |constant_offset| addresses the dispatch constants
// block and |state_table_index| addresses one of 32 four-byte firmware slots.
// |scalar_type| is producer reflection metadata such as "i32" or "bf16"; HRX
// copies |byte_length| raw bytes and does not interpret the scalar value.
typedef struct hrx_amdxdna_control_parameter_t {
  uint32_t record_length;
  uint32_t abi_version;
  hrx_string_view_t name;
  hrx_string_view_t scalar_type;
  uint32_t constant_offset;
  uint32_t state_table_index;
  uint32_t byte_length;
  hrx_amdxdna_control_parameter_kind_t kind;
} hrx_amdxdna_control_parameter_t;

// Returns an initialized v0 control-parameter record.
static inline hrx_amdxdna_control_parameter_t
hrx_amdxdna_control_parameter_default(void) {
  hrx_amdxdna_control_parameter_t parameter = HRX_AMDXDNA_STRUCT_ZERO_INIT;
  parameter.record_length = (uint32_t)sizeof(parameter);
  parameter.abi_version = HRX_AMDXDNA_CONTROL_PARAMETER_ABI_VERSION_0;
  return parameter;
}

// One control-code run in an amdxdna executable. |transaction| contains an
// XAie transaction and |data_payload| optionally contains reconfiguration
// data. HRX derives host-buffer relocation metadata from |transaction|.
// |pdi_relocations| and |control_parameter_relocations| require ABI version 1.
// They are optional and patch runtime-owned addresses used by the transaction.
typedef struct hrx_amdxdna_executable_run_t {
  uint32_t record_length;
  uint32_t abi_version;
  hrx_const_byte_span_t transaction;
  hrx_const_byte_span_t data_payload;
  const hrx_amdxdna_pdi_relocation_t* pdi_relocations;
  size_t pdi_relocation_count;
  const hrx_amdxdna_control_parameter_relocation_t*
      control_parameter_relocations;
  size_t control_parameter_relocation_count;
} hrx_amdxdna_executable_run_t;

// Returns an initialized v0 run record.
static inline hrx_amdxdna_executable_run_t hrx_amdxdna_executable_run_default(
    void) {
  hrx_amdxdna_executable_run_t run = HRX_AMDXDNA_STRUCT_ZERO_INIT;
  run.record_length = (uint32_t)sizeof(run);
  run.abi_version = HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0;
  return run;
}

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
  // Optional scalar layout for the firmware control parameter buffer. Nonempty
  // metadata requires entry-point ABI version 1.
  const hrx_amdxdna_control_parameter_t* control_parameters;
  size_t control_parameter_count;
} hrx_amdxdna_executable_entry_point_t;

// Returns an initialized v0 entry-point record.
static inline hrx_amdxdna_executable_entry_point_t
hrx_amdxdna_executable_entry_point_default(void) {
  hrx_amdxdna_executable_entry_point_t entry_point =
      HRX_AMDXDNA_STRUCT_ZERO_INIT;
  entry_point.record_length = (uint32_t)sizeof(entry_point);
  entry_point.abi_version = HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0;
  entry_point.context_mode = HRX_AMDXDNA_CONTEXT_MODE_CREATE;
  return entry_point;
}

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
  // Standalone PDIs referenced by run pdi_relocations. Nonempty PDI metadata
  // requires create-parameter ABI version 1. The context PDI remains selected
  // from an xclbin by entry_point.pdi_ordinal.
  const hrx_const_byte_span_t* pdis;
  size_t pdi_count;
} hrx_amdxdna_executable_create_params_t;

// Returns initialized v0 executable creation parameters.
static inline hrx_amdxdna_executable_create_params_t
hrx_amdxdna_executable_create_params_default(void) {
  hrx_amdxdna_executable_create_params_t params = HRX_AMDXDNA_STRUCT_ZERO_INIT;
  params.record_length = (uint32_t)sizeof(params);
  params.abi_version = HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0;
  return params;
}

// Serializes a generic amdxdna executable description to the XADX container
// format. The returned bytes are allocated with |host_allocator| and must be
// released with hrx_host_allocator_free using the same allocator. On failure,
// |out_data| is NULL and |out_data_length| is zero.
//
// This entry point is intended for tooling, caching, and artifact creation.
// Runtime users should normally call hrx_amdxdna_executable_create, which keeps
// the serialized representation internal to HRX.
HRX_API hrx_status_t
hrx_amdxdna_xadx_serialize(const hrx_amdxdna_executable_create_params_t* params,
                           hrx_host_allocator_t host_allocator,
                           uint8_t** out_data, size_t* out_data_length);

// Creates and loads an amdxdna executable while keeping the backend package
// representation private to HRX. All input storage is borrowed for the
// duration of the call and may be released after it returns. The load uses
// HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY/KEY.
HRX_API hrx_status_t hrx_amdxdna_executable_create(
    hrx_device_t device, const hrx_amdxdna_executable_create_params_t* params,
    hrx_executable_t* executable);

// Registers a one-run |transaction| and returns an opaque non-zero token. This
// cold-path call hashes, validates, and derives patch metadata. If the entry
// point declares PDI or control-parameter relocations, every variant must
// preserve that relocation schema and its transaction offsets; HRX applies the
// executable's relocations to each variant.
//
// Registration is thread-safe, does not require stream synchronization, and
// does not change what another stream dispatches. The executable must remain
// live for this call and every dispatch. Tokens are owned by the executable and
// must not be used with another executable or export. The bounded LRU cache may
// evict cold variants; a later dispatch then returns NOT_FOUND and the caller
// must register the transaction again.
HRX_API hrx_status_t hrx_amdxdna_executable_cache_transaction(
    hrx_executable_t executable, uint32_t export_ordinal,
    hrx_const_byte_span_t transaction, uint64_t* out_variant);

// Atomically records a dispatch against one cached transaction variant.
// Concurrent streams may dispatch different variants of the same executable.
// A successful call snapshots the variant into stream-owned recording storage,
// so later cache eviction does not affect the recorded dispatch.
HRX_API hrx_status_t hrx_amdxdna_stream_dispatch_variant(
    hrx_stream_t stream, hrx_executable_t executable, uint32_t export_ordinal,
    uint64_t variant, const hrx_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const hrx_buffer_ref_t* bindings, size_t binding_count, uint32_t flags);

#ifdef __cplusplus
}
#endif

#undef HRX_AMDXDNA_STRUCT_ZERO_INIT

#endif  // HRX_AMDXDNA_H_
