// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven text printer for loom IR.
//
// Walks the format element arrays in each op's vtable (.rodata) to emit
// canonical textual IR. One generic function handles all ops — no per-op
// code. The printer is read-only over the IR (unless location capture
// is enabled, which updates op locations to point at the output).
//
// All output goes through a loom_output_stream_t — zero heap
// allocations in the print path.

#ifndef LOOM_FORMAT_TEXT_PRINTER_PRINTER_H_
#define LOOM_FORMAT_TEXT_PRINTER_PRINTER_H_

#include "iree/base/api.h"
#include "loom/format/text/low_asm.h"
#include "loom/ir/ir.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Flags controlling printer behavior.
enum loom_text_print_flag_bits_e {
  // Update each op's location to point at its byte range in the output.
  LOOM_TEXT_PRINT_CAPTURE_LOCATIONS = 1u << 0,
  // Use encoding aliases (#enc) instead of inlining (#q8_0<block=32>).
  LOOM_TEXT_PRINT_USE_ALIASES = 1u << 1,
  // Pretty-print with 2-space indentation (default on).
  LOOM_TEXT_PRINT_INDENT = 1u << 2,
  // Omit region bodies. Regions print as empty braces: { ... }. Useful
  // for printing function/module declarations without their contents.
  LOOM_TEXT_PRINT_SKIP_REGIONS = 1u << 3,
  // Emit trailing loc() annotations on ops.
  LOOM_TEXT_PRINT_LOCATIONS = 1u << 4,
};
typedef uint32_t loom_text_print_flags_t;

// Default flags for canonical output.
#define LOOM_TEXT_PRINT_DEFAULT \
  (LOOM_TEXT_PRINT_INDENT | LOOM_TEXT_PRINT_USE_ALIASES)

// Options controlling canonical text printing.
typedef struct loom_text_print_options_t {
  // Flag bitset controlling layout and optional annotations.
  loom_text_print_flags_t flags;
  // Optional environment used to print low asm region syntax.
  loom_text_low_asm_environment_t low_asm_environment;
  // Descriptor-set key selected for low asm region syntax in this print.
  iree_string_view_t low_asm_descriptor_set_key;
} loom_text_print_options_t;

// Prints a complete module to canonical text via the output stream.
iree_status_t loom_text_print_module(const loom_module_t* module,
                                     loom_output_stream_t* stream,
                                     loom_text_print_flags_t flags);

// Prints a complete module using explicit printer options.
iree_status_t loom_text_print_module_with_options(
    const loom_module_t* module, loom_output_stream_t* stream,
    const loom_text_print_options_t* options);

// Prints a single operation via the output stream.
iree_status_t loom_text_print_operation(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_output_stream_t* stream,
                                        loom_text_print_flags_t flags);

// Prints a single operation using explicit printer options.
iree_status_t loom_text_print_operation_with_options(
    const loom_module_t* module, const loom_op_t* op,
    loom_output_stream_t* stream, const loom_text_print_options_t* options);

// Prints a type in canonical form to the output stream.
iree_status_t loom_text_print_type(loom_type_t type,
                                   const loom_module_t* module,
                                   loom_output_stream_t* stream);

// Prints a type using explicit printer options.
iree_status_t loom_text_print_type_with_options(
    loom_type_t type, const loom_module_t* module, loom_output_stream_t* stream,
    const loom_text_print_options_t* options);

// Prints a standalone attribute value in canonical form to the output stream.
// Enum values print as raw numeric fallbacks because no operation field
// descriptor is available at this boundary.
iree_status_t loom_text_print_attribute(const loom_attribute_t* attr,
                                        const loom_module_t* module,
                                        loom_output_stream_t* stream);

// Convenience: print module to an iree_string_builder_t.
iree_status_t loom_text_print_module_to_builder(const loom_module_t* module,
                                                iree_string_builder_t* builder,
                                                loom_text_print_flags_t flags);

// Convenience: print module to a builder using explicit printer options.
iree_status_t loom_text_print_module_to_builder_with_options(
    const loom_module_t* module, iree_string_builder_t* builder,
    const loom_text_print_options_t* options);

// Convenience: print single op to an iree_string_builder_t.
iree_status_t loom_text_print_operation_to_builder(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags);

// Convenience: print single op to a builder using explicit printer options.
iree_status_t loom_text_print_operation_to_builder_with_options(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, const loom_text_print_options_t* options);

//===----------------------------------------------------------------------===//
// Field emission callback
//===----------------------------------------------------------------------===//

// Identifies the category of an op field emitted by the text printer.
typedef enum loom_print_field_kind_e {
  LOOM_PRINT_FIELD_OPERAND = 0,
  LOOM_PRINT_FIELD_RESULT = 1,
  LOOM_PRINT_FIELD_ATTR = 2,
  LOOM_PRINT_FIELD_REGION = 3,
  LOOM_PRINT_FIELD_SUCCESSOR = 4,
} loom_print_field_kind_t;

// Identifies a concrete op field emitted by the text printer.
//
// This callback-local representation intentionally carries a full 16-bit field
// index instead of reusing loom_field_ref_t's packed 6-bit constraint-table
// encoding. Wide variadic operands/results still need precise byte spans in
// fallback diagnostics.
typedef struct loom_print_field_ref_t {
  loom_print_field_kind_t kind;
  uint16_t index;
} loom_print_field_ref_t;

static inline loom_print_field_ref_t loom_print_field_ref(
    loom_print_field_kind_t kind, uint16_t index) {
  return (loom_print_field_ref_t){
      /*.kind=*/kind,
      /*.index=*/index,
  };
}

static inline bool loom_print_field_ref_equal(loom_print_field_ref_t lhs,
                                              loom_print_field_ref_t rhs) {
  return lhs.kind == rhs.kind && lhs.index == rhs.index;
}

// Called by the printer for each field it emits in the output.
// Fires for operand %names, result %names, attribute values, successor labels,
// and region bodies — every semantically meaningful token span that corresponds
// to an op field.
// |field_ref| identifies the field category and index. |start| and |end| are
// byte offsets in the output stream.
typedef void (*loom_print_field_fn_t)(void* user_data,
                                      loom_print_field_ref_t field_ref,
                                      iree_host_size_t start,
                                      iree_host_size_t end);

// Bundles a field emission callback with its context.
typedef struct loom_print_field_callback_t {
  loom_print_field_fn_t fn;
  void* user_data;
} loom_print_field_callback_t;

// Prints a single operation to a builder, invoking |callback.fn| for
// each field emitted. Use the callback to record byte ranges of
// specific fields for diagnostic highlighting.
iree_status_t loom_text_print_operation_with_field_callback(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags,
    loom_print_field_callback_t callback);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_PRINTER_H_
