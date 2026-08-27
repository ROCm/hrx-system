// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Decoded function-symbol header shared by full and selective materializers.

#ifndef LOOM_FORMAT_BYTECODE_FUNCTION_HEADER_H_
#define LOOM_FORMAT_BYTECODE_FUNCTION_HEADER_H_

#include "loom/format/bytecode/reader/source_trivia.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_region_payload_metadata_t
    loom_bytecode_region_payload_metadata_t;

// Common prefix decoded before dispatching one symbol entry.
typedef struct loom_bytecode_symbol_entry_header_t {
  // Source STRINGS ordinal naming the symbol.
  uint32_t name_string_ordinal;
  // Validated wire symbol kind.
  uint8_t kind;
  // Absolute byte offset of the wire kind for diagnostics.
  uint64_t kind_offset;
  // Validated wire visibility.
  uint8_t visibility;
  // Validated wire symbol flags.
  uint16_t flags;
  // Projected import module string, or invalid when not imported.
  loom_string_id_t import_module_id;
  // Projected import symbol string, or invalid when not imported.
  loom_string_id_t import_symbol_id;
} loom_bytecode_symbol_entry_header_t;

// Materialized function-like metadata preceding root-region references.
//
// Arrays and attributes borrow the active materializer arena. Signature values,
// types, strings, and symbol refs belong to the materializer's output module.
// Root-region payload bytes are not read while producing this header.
typedef struct loom_bytecode_function_header_t {
  // Source symbol ordinal being decoded.
  iree_host_size_t source_symbol_ordinal;
  // Output-module symbol ID for the implementation.
  loom_symbol_id_t symbol_id;
  // Borrowed source symbol name.
  iree_string_view_t name;
  // Registered operation metadata for the function-like symbol.
  const loom_op_vtable_t* vtable;
  // Function-like interface implemented by vtable.
  const loom_func_like_vtable_t* func_like;
  // Canonical operation kind corresponding to vtable.
  loom_op_kind_t op_kind;
  // Authored source comments and vertical separation.
  loom_bytecode_source_trivia_t source_trivia;
  // Function calling convention byte.
  uint8_t calling_convention;
  // Function purity byte.
  uint8_t purity;
  // Number of kernel workload arguments at the start of signature_values.
  uint16_t workload_argument_count;
  // Number of ordinary arguments following workload arguments.
  uint16_t argument_count;
  // Number of results following all arguments.
  uint16_t result_count;
  // Number of tied result records.
  uint16_t tied_result_count;
  // Workload arguments, ordinary arguments, and results in wire order.
  loom_value_id_t* signature_values;
  // Tied result records in result order.
  loom_tied_result_t* tied_results;
  // Complete reconstructed operation attribute array.
  loom_attribute_t* attributes;
  // Independently bounded root-region payloads in declared slot order.
  loom_bytecode_region_payload_metadata_t* region_payloads;
  // Number of entries in region_payloads.
  uint8_t region_payload_count;
  // Body entry in region_payloads plus one, or zero when absent.
  uint8_t body_region_payload_ordinal_plus_one;
  // Workload/configuration entry in region_payloads plus one, or zero when
  // absent.
  uint8_t kernel_workload_region_payload_ordinal_plus_one;
} loom_bytecode_function_header_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_FUNCTION_HEADER_H_
