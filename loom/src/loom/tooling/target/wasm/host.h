// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Synchronous host execution of ordinary WebAssembly modules.

#ifndef LOOM_TOOLING_TARGET_WASM_HOST_H_
#define LOOM_TOOLING_TARGET_WASM_HOST_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical WebAssembly value types callable through the JavaScript host.
typedef enum loom_wasm_host_value_type_e {
  LOOM_WASM_HOST_VALUE_TYPE_I32 = 0x7F,
  LOOM_WASM_HOST_VALUE_TYPE_I64 = 0x7E,
  LOOM_WASM_HOST_VALUE_TYPE_F32 = 0x7D,
  LOOM_WASM_HOST_VALUE_TYPE_F64 = 0x7C,
} loom_wasm_host_value_type_t;

typedef enum loom_wasm_host_module_flag_bits_e {
  // The loaded module exposes linear memory for root transport.
  LOOM_WASM_HOST_MODULE_FLAG_HAS_MEMORY = 1u << 0,
} loom_wasm_host_module_flag_bits_t;

// Bitset of loom_wasm_host_module_flag_bits_t values.
typedef uint32_t loom_wasm_host_module_flags_t;

// One ordinary module instance retained by the JavaScript host.
typedef struct loom_wasm_host_module_t {
  // Host-owned module instance handle, or zero when empty.
  uint32_t handle;
  // Physical parameter count accepted by the selected export.
  uint32_t parameter_count;
  // Physical result count returned by the selected export.
  uint32_t result_count;
  // Capabilities established from the instantiated module.
  loom_wasm_host_module_flags_t flags;
} loom_wasm_host_module_t;

// One complete caller-owned allocation root transferred around a call.
typedef struct loom_wasm_host_memory_region_t {
  // Nonzero address of the first root byte in the nested module memory.
  // Zero is reserved for a null buffer, including for empty storage roots.
  uint32_t address;
  // Mutable root storage in the caller's WebAssembly memory.
  uint8_t* data;
  // Number of bytes transferred from and back to |data|.
  uint32_t data_length;
} loom_wasm_host_memory_region_t;

// Instantiates |module_data| once and retains |function_export_name|.
//
// The host reads the selected export's physical signature from the canonical
// WebAssembly binary and writes it to the caller-owned type arrays. Parameter
// and result counts must match the expected source arity. v128 parameters and
// results are rejected because JavaScript cannot call them directly. An
// exported WebAssembly memory is retained automatically for region transport.
iree_status_t loom_wasm_host_module_load(
    iree_const_byte_span_t module_data, iree_string_view_t function_export_name,
    uint32_t parameter_count, uint32_t result_count,
    loom_wasm_host_value_type_t* out_parameter_types,
    loom_wasm_host_value_type_t* out_result_types,
    loom_wasm_host_module_t* out_module);

// Releases the host-owned module instance and resets |module|.
// Safe to call on a zero-initialized module.
void loom_wasm_host_module_release(loom_wasm_host_module_t* module);

// Calls the selected export and synchronously returns scalar results and
// mutated allocation roots.
//
// Scalar payloads use their raw low bits in physical signature order; i32/f32
// occupy 32 bits and i64/f64 occupy all 64 bits. Every region describes a
// complete allocation root. When the module exports memory, roots are copied
// into it before the call and copied back afterward, including when the
// function traps.
iree_status_t loom_wasm_host_module_call(
    const loom_wasm_host_module_t* module, const uint64_t* argument_bits,
    uint64_t* result_bits, iree_host_size_t region_count,
    const loom_wasm_host_memory_region_t* regions);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_WASM_HOST_H_
