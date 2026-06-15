// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_ATOMS_H_
#define LOOM_FORMAT_TEXT_PRINTER_ATOMS_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/context.h"
#include "loom/ir/ir.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stack buffer size for formatting generated value names (%0, %1$2, ...).
// Explicit names live in the module string table and are streamed directly so
// they are not capped by this fallback buffer.
#define LOOM_VALUE_NAME_BUFFER_SIZE 32

// Prints an SSA value reference without token spacing.
iree_status_t loom_print_value_ref(loom_output_stream_t* stream,
                                   const loom_module_t* module,
                                   loom_value_id_t value_id);

// Emits an SSA value name through the token spacing model.
iree_status_t loom_print_value_name(loom_print_context_t* ctx,
                                    loom_value_id_t value_id);

// Prints a value name and reports its emitted field range.
iree_status_t loom_print_value_name_with_field(
    loom_print_context_t* ctx, loom_value_id_t value_id,
    loom_print_field_ref_t field_ref);

// Prints a canonical attribute payload.
iree_status_t loom_print_attr(loom_output_stream_t* stream,
                              const loom_attribute_t* attr,
                              const loom_module_t* module,
                              const loom_attr_descriptor_t* descriptor);

// Prints |type| using the current contextual printer state.
iree_status_t loom_print_type(loom_print_context_t* ctx, loom_type_t type);

// Prints the canonical type of |value_id|, or <unknown> for malformed IR.
iree_status_t loom_print_value_type(loom_print_context_t* ctx,
                                    loom_value_id_t value_id);

// Prints the canonical result type of |value_id|, or <unknown> for malformed
// IR.
iree_status_t loom_print_result_value_type(loom_print_context_t* ctx,
                                           loom_value_id_t value_id);

// Prints a source location annotation.
iree_status_t loom_print_location(loom_output_stream_t* stream,
                                  const loom_module_t* module,
                                  loom_location_id_t location_id);

// Prints all encoding aliases declared in |module|.
iree_status_t loom_print_encoding_aliases(loom_print_context_t* ctx,
                                          const loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_ATOMS_H_
