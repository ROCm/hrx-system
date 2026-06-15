// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/module_binary.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/emit/wasm/binary_writer.h"
#include "loom/target/emit/wasm/function_body.h"
#include "loom/target/emit/wasm/types.h"
#include "loom/target/registers.h"

enum {
  LOOM_WASM_SECTION_CUSTOM = 0,
  LOOM_WASM_SECTION_TYPE = 1,
  LOOM_WASM_SECTION_FUNCTION = 3,
  LOOM_WASM_SECTION_MEMORY = 5,
  LOOM_WASM_SECTION_EXPORT = 7,
  LOOM_WASM_SECTION_CODE = 10,
};

enum {
  LOOM_WASM_EXPORT_KIND_FUNCTION = 0,
};

enum {
  LOOM_WASM_LIMITS_MIN_ONLY = 0,
};

enum {
  LOOM_WASM_FUNCTION_TYPE = 0x60,
};

enum {
  LOOM_WASM_NAME_SUBSECTION_FUNCTIONS = 1,
};

enum {
  LOOM_WASM_FUNCTION_INDEX_NONE = UINT32_MAX,
};

typedef struct loom_wasm_function_type_t {
  // Arena-owned Wasm value types for function parameters.
  const loom_wasm_value_type_t* parameters;
  // Number of parameter entries.
  uint32_t parameter_count;
  // Arena-owned Wasm value types for function results.
  const loom_wasm_value_type_t* results;
  // Number of result entries.
  uint32_t result_count;
} loom_wasm_function_type_t;

typedef struct loom_wasm_module_function_t {
  // Symbol table entry defining this function.
  const loom_symbol_t* symbol;
  // Symbol reference used by low.func.call to name this function.
  loom_symbol_ref_t callee;
  // Borrowed module symbol name.
  iree_string_view_t name;
  // Borrowed export name, or empty when the function is module-private.
  iree_string_view_t export_name;
  // Wasm function index assigned by module order.
  uint32_t function_index;
  // Wasm type index assigned by signature interning.
  uint32_t type_index;
  // Emission frame produced for this low function.
  loom_low_emission_frame_t frame;
  // Interned Wasm function signature.
  loom_wasm_function_type_t type;
  // Size-prefixed Wasm code-section body bytes.
  loom_wasm_function_body_t body;
} loom_wasm_module_function_t;

typedef struct loom_wasm_module_layout_t {
  // Module being emitted.
  loom_module_t* module;
  // Arena-scoped function records in Wasm function-index order.
  loom_wasm_module_function_t* functions;
  // Number of function records.
  iree_host_size_t function_count;
  // Module symbol-id to Wasm function-index lookup table.
  uint32_t* symbol_function_indices;
  // Arena-scoped unique function signatures.
  loom_wasm_function_type_t* types;
  // Number of unique signatures.
  iree_host_size_t type_count;
  // Number of function exports.
  iree_host_size_t export_count;
  // Structural facts represented in the emitted module binary.
  loom_wasm_module_binary_flags_t flags;
} loom_wasm_module_layout_t;

static iree_status_t loom_wasm_module_write_section(
    loom_wasm_binary_writer_t* module_writer, uint8_t section_id,
    const loom_wasm_binary_writer_t* payload_writer) {
  if (payload_writer->length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm section %" PRIu8 " exceeds u32 size",
                            section_id);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(module_writer, section_id));
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      module_writer, (uint32_t)payload_writer->length));
  return loom_wasm_binary_write_bytes(module_writer, payload_writer->data,
                                      payload_writer->length);
}

static iree_status_t loom_wasm_module_write_name(
    loom_wasm_binary_writer_t* writer, iree_string_view_t name) {
  if (name.size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm name exceeds u32 size");
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(writer, (uint32_t)name.size));
  return loom_wasm_binary_write_bytes(writer, (const uint8_t*)name.data,
                                      name.size);
}

static iree_string_view_t loom_wasm_module_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_wasm_module_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol == NULL) {
    return IREE_SV("<unnamed>");
  }
  iree_string_view_t name =
      loom_wasm_module_string_or_empty(module, symbol->name_id);
  return iree_string_view_is_empty(name) ? IREE_SV("<unnamed>") : name;
}

static iree_string_view_t loom_wasm_module_function_export_name(
    const loom_module_t* module, const loom_symbol_t* symbol,
    const loom_op_t* function_op) {
  loom_func_like_t function =
      loom_func_like_cast(module, (loom_op_t*)function_op);
  if (!loom_func_like_isa(function)) {
    return iree_string_view_empty();
  }
  iree_string_view_t export_name = loom_wasm_module_string_or_empty(
      module, loom_func_like_export_symbol(function));
  if (!iree_string_view_is_empty(export_name)) {
    return export_name;
  }
  if (loom_func_like_visibility(function) != 0 ||
      (symbol->flags & LOOM_SYMBOL_FLAG_PUBLIC) != 0) {
    return loom_wasm_module_symbol_name(module, symbol);
  }
  return iree_string_view_empty();
}

static iree_status_t loom_wasm_module_read_value_type(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, loom_value_id_t value_id,
    loom_wasm_value_type_t* out_value_type) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_low_type_is_register(type)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm signature value %u is not declared as a register value",
        (unsigned)value_id);
  }
  if (loom_low_register_type_descriptor_set_stable_id(type) !=
      descriptor_set->stable_id) {
    iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm signature value %u does not use descriptor set '%.*s'",
        (unsigned)value_id, (int)descriptor_set_key.size,
        descriptor_set_key.data);
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(type);
  if (unit_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm signature value %u declares %u register units",
        (unsigned)value_id, unit_count);
  }
  return loom_wasm_value_type_from_descriptor_register_class(
      loom_low_register_type_class_id(type), out_value_type);
}

static iree_status_t loom_wasm_module_build_value_type_list(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_value_id_t* value_ids, uint32_t value_count,
    iree_arena_allocator_t* arena, const loom_wasm_value_type_t** out_types) {
  *out_types = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_wasm_value_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*types), (void**)&types));
  for (uint32_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_module_read_value_type(
        module, descriptor_set, value_ids[i], &types[i]));
  }
  *out_types = types;
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_build_function_type(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_wasm_function_type_t* out_type) {
  *out_type = (loom_wasm_function_type_t){0};
  loom_func_like_t function =
      loom_func_like_cast(frame->module, (loom_op_t*)frame->function_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm module emission requires a func-like op");
  }

  uint16_t parameter_count = 0;
  const loom_value_id_t* parameters =
      loom_func_like_arg_ids(function, &parameter_count);
  loom_value_slice_t results = loom_low_func_def_results(frame->function_op);

  iree_status_t status = loom_wasm_module_build_value_type_list(
      frame->module, frame->target.descriptor_set, parameters, parameter_count,
      arena, &out_type->parameters);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_build_value_type_list(
        frame->module, frame->target.descriptor_set, results.values,
        results.count, arena, &out_type->results);
  }
  if (iree_status_is_ok(status)) {
    out_type->parameter_count = parameter_count;
    out_type->result_count = results.count;
  }
  return status;
}

static bool loom_wasm_module_function_types_equal(
    const loom_wasm_function_type_t* lhs,
    const loom_wasm_function_type_t* rhs) {
  if (lhs->parameter_count != rhs->parameter_count ||
      lhs->result_count != rhs->result_count) {
    return false;
  }
  if (lhs->parameter_count != 0 &&
      memcmp(lhs->parameters, rhs->parameters,
             lhs->parameter_count * sizeof(*lhs->parameters)) != 0) {
    return false;
  }
  return lhs->result_count == 0 ||
         memcmp(lhs->results, rhs->results,
                lhs->result_count * sizeof(*lhs->results)) == 0;
}

static iree_status_t loom_wasm_module_intern_function_type(
    loom_wasm_module_layout_t* layout, const loom_wasm_function_type_t* type,
    uint32_t* out_type_index) {
  for (iree_host_size_t i = 0; i < layout->type_count; ++i) {
    if (loom_wasm_module_function_types_equal(&layout->types[i], type)) {
      *out_type_index = (uint32_t)i;
      return iree_ok_status();
    }
  }
  if (layout->type_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm type index exceeds u32");
  }
  *out_type_index = (uint32_t)layout->type_count;
  layout->types[layout->type_count++] = *type;
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_build_function_frame(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame) {
  if (options->schedule_strategy !=
      LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm module emission requires source-order low scheduling");
  }
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
      module, function_op, options, arena, out_frame));
  if (out_frame->target.descriptor_set !=
      loom_wasm_core_simd128_descriptor_set()) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm module emission requires descriptor set 'wasm.core.simd128'");
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_prepare_function(
    loom_wasm_module_layout_t* layout, loom_wasm_module_function_t* function,
    loom_module_t* module, const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena) {
  IREE_RETURN_IF_ERROR(loom_wasm_module_build_function_frame(
      module, function->symbol->defining_op, options, arena, &function->frame));
  IREE_RETURN_IF_ERROR(loom_wasm_module_build_function_type(
      &function->frame, arena, &function->type));
  IREE_RETURN_IF_ERROR(loom_wasm_module_intern_function_type(
      layout, &function->type, &function->type_index));
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_collect_functions(
    loom_module_t* module, const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, loom_wasm_module_layout_t* out_layout) {
  *out_layout = (loom_wasm_module_layout_t){
      .module = module,
  };
  if (module->symbols.count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm module emission requires low functions");
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*out_layout->functions),
      (void**)&out_layout->functions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*out_layout->types),
                                                 (void**)&out_layout->types));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, module->symbols.count,
                                sizeof(*out_layout->symbol_function_indices),
                                (void**)&out_layout->symbol_function_indices));

  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    out_layout->symbol_function_indices[symbol_index] =
        LOOM_WASM_FUNCTION_INDEX_NONE;
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_op_t* defining_op = symbol->defining_op;
    if (defining_op == NULL) {
      continue;
    }
    if (loom_low_func_decl_isa(defining_op)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm module imports for low.func.decl are not implemented");
    }
    if (loom_low_kernel_def_isa(defining_op)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm module emission for low.kernel.def is not implemented");
    }
    if (!loom_low_func_def_isa(defining_op)) {
      continue;
    }
    if (out_layout->function_count >= LOOM_WASM_FUNCTION_INDEX_NONE ||
        symbol_index >= LOOM_SYMBOL_ID_INVALID) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm function index exceeds supported range");
    }
    loom_wasm_module_function_t* function =
        &out_layout->functions[out_layout->function_count];
    *function = (loom_wasm_module_function_t){
        .symbol = symbol,
        .callee = {.module_id = 0, .symbol_id = (uint16_t)symbol_index},
        .name = loom_wasm_module_symbol_name(module, symbol),
        .export_name =
            loom_wasm_module_function_export_name(module, symbol, defining_op),
        .function_index = (uint32_t)out_layout->function_count,
    };
    out_layout->symbol_function_indices[symbol_index] =
        function->function_index;
    if (!iree_string_view_is_empty(function->export_name)) {
      ++out_layout->export_count;
    }
    IREE_RETURN_IF_ERROR(loom_wasm_module_prepare_function(
        out_layout, function, module, options, arena));
    ++out_layout->function_count;
  }

  if (out_layout->function_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm module emission requires at least one "
                            "low.func.def");
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_resolve_function_index(
    void* user_data, loom_symbol_ref_t callee, uint32_t* out_function_index) {
  const loom_wasm_module_layout_t* layout =
      (const loom_wasm_module_layout_t*)user_data;
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm direct call uses a non-local callee symbol");
  }
  if (callee.symbol_id >= layout->module->symbols.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm direct call uses an out-of-range callee "
                            "symbol %" PRIu16,
                            callee.symbol_id);
  }
  const uint32_t function_index =
      layout->symbol_function_indices[callee.symbol_id];
  if (function_index != LOOM_WASM_FUNCTION_INDEX_NONE) {
    *out_function_index = function_index;
    return iree_ok_status();
  }
  const iree_string_view_t callee_name = loom_wasm_module_symbol_name(
      layout->module, &layout->module->symbols.entries[callee.symbol_id]);
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "Wasm direct call callee '@%.*s' is not defined in "
                          "this module",
                          (int)callee_name.size, callee_name.data);
}

static iree_status_t loom_wasm_module_emit_function_bodies(
    loom_wasm_module_layout_t* layout, iree_allocator_t allocator) {
  const loom_wasm_function_body_options_t body_options = {
      .resolve_function_index = loom_wasm_module_resolve_function_index,
      .resolve_function_index_user_data = layout,
  };
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    loom_wasm_module_function_t* function = &layout->functions[i];
    IREE_RETURN_IF_ERROR(
        loom_wasm_emit_function_body(&function->frame.allocation, &body_options,
                                     allocator, &function->body));
    if (iree_any_bit_set(function->body.flags,
                         LOOM_WASM_FUNCTION_BODY_FLAG_USES_MEMORY)) {
      layout->flags |= LOOM_WASM_MODULE_BINARY_FLAG_DEFINES_MEMORY;
    }
  }
  return iree_ok_status();
}

static void loom_wasm_module_deinitialize_function_bodies(
    loom_wasm_module_layout_t* layout, iree_allocator_t allocator) {
  if (layout == NULL || layout->functions == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    loom_wasm_function_body_deinitialize(&layout->functions[i].body, allocator);
  }
}

static iree_status_t loom_wasm_module_write_value_type_list(
    const loom_wasm_value_type_t* types, uint32_t type_count,
    loom_wasm_binary_writer_t* writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(writer, type_count));
  for (uint32_t i = 0; i < type_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(writer, types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_type_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer) {
  if (layout->type_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm type count exceeds u32");
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      payload_writer, (uint32_t)layout->type_count));
  for (iree_host_size_t i = 0; i < layout->type_count; ++i) {
    const loom_wasm_function_type_t* type = &layout->types[i];
    IREE_RETURN_IF_ERROR(
        loom_wasm_binary_write_u8(payload_writer, LOOM_WASM_FUNCTION_TYPE));
    IREE_RETURN_IF_ERROR(loom_wasm_module_write_value_type_list(
        type->parameters, type->parameter_count, payload_writer));
    IREE_RETURN_IF_ERROR(loom_wasm_module_write_value_type_list(
        type->results, type->result_count, payload_writer));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_type_section(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_type_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_TYPE, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_function_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer) {
  if (layout->function_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm function count exceeds u32");
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      payload_writer, (uint32_t)layout->function_count));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
        payload_writer, layout->functions[i].type_index));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_function_section(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_function_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_FUNCTION, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_memory_section_payload(
    loom_wasm_binary_writer_t* payload_writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(payload_writer, 1));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(payload_writer, LOOM_WASM_LIMITS_MIN_ONLY));
  return loom_wasm_binary_write_u32_leb(payload_writer, 1);
}

static iree_status_t loom_wasm_module_write_memory_section(
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_memory_section_payload(&payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_MEMORY, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_export_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer) {
  if (layout->export_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm export count exceeds u32");
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      payload_writer, (uint32_t)layout->export_count));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    const loom_wasm_module_function_t* function = &layout->functions[i];
    if (iree_string_view_is_empty(function->export_name)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_wasm_module_write_name(payload_writer, function->export_name));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(
        payload_writer, LOOM_WASM_EXPORT_KIND_FUNCTION));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
        payload_writer, function->function_index));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_export_section(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  if (layout->export_count == 0) {
    return iree_ok_status();
  }
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_export_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_EXPORT, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_code_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer) {
  if (layout->function_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm function body count exceeds u32");
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      payload_writer, (uint32_t)layout->function_count));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    const loom_wasm_function_body_t* body = &layout->functions[i].body;
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_bytes(
        payload_writer, body->data, body->data_length));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_code_section(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_code_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_CODE, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_name_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer, iree_allocator_t allocator) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_module_write_name(payload_writer, IREE_SV("name")));

  loom_wasm_binary_writer_t function_names_writer;
  loom_wasm_binary_writer_initialize(allocator, &function_names_writer);
  iree_status_t status = iree_ok_status();
  if (layout->function_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm name function count exceeds u32");
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_binary_write_u32_leb(&function_names_writer,
                                            (uint32_t)layout->function_count);
  }
  for (iree_host_size_t i = 0;
       i < layout->function_count && iree_status_is_ok(status); ++i) {
    const loom_wasm_module_function_t* function = &layout->functions[i];
    status = loom_wasm_binary_write_u32_leb(&function_names_writer,
                                            function->function_index);
    if (iree_status_is_ok(status)) {
      status =
          loom_wasm_module_write_name(&function_names_writer, function->name);
    }
  }
  if (iree_status_is_ok(status) && function_names_writer.length > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm function-name subsection exceeds u32");
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_binary_write_u8(payload_writer,
                                       LOOM_WASM_NAME_SUBSECTION_FUNCTIONS);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_binary_write_u32_leb(
        payload_writer, (uint32_t)function_names_writer.length);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_wasm_binary_write_bytes(payload_writer, function_names_writer.data,
                                     function_names_writer.length);
  }
  loom_wasm_binary_writer_deinitialize(&function_names_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_name_section(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status = loom_wasm_module_write_name_section_payload(
      layout, &payload_writer, allocator);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_CUSTOM, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_header(
    loom_wasm_binary_writer_t* module_writer) {
  static const uint8_t header[] = {
      0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
  };
  return loom_wasm_binary_write_bytes(module_writer, header, sizeof(header));
}

void loom_wasm_module_binary_deinitialize(loom_wasm_module_binary_t* module,
                                          iree_allocator_t allocator) {
  if (!module) {
    return;
  }
  iree_allocator_free(allocator, module->data);
  *module = (loom_wasm_module_binary_t){0};
}

iree_status_t loom_wasm_emit_module(
    loom_module_t* module, const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    loom_wasm_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->descriptor_registry);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = (loom_wasm_module_binary_t){0};

  loom_wasm_module_layout_t layout = {0};
  iree_status_t status =
      loom_wasm_module_collect_functions(module, options, arena, &layout);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_emit_function_bodies(&layout, allocator);
  }

  loom_wasm_binary_writer_t module_writer;
  loom_wasm_binary_writer_initialize(allocator, &module_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_header(&module_writer);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_wasm_module_write_type_section(&layout, &module_writer, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_function_section(&layout, &module_writer,
                                                     allocator);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(layout.flags,
                       LOOM_WASM_MODULE_BINARY_FLAG_DEFINES_MEMORY)) {
    status = loom_wasm_module_write_memory_section(&module_writer, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_export_section(&layout, &module_writer,
                                                   allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_wasm_module_write_code_section(&layout, &module_writer, allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_wasm_module_write_name_section(&layout, &module_writer, allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_module = (loom_wasm_module_binary_t){
        .data = module_writer.data,
        .data_length = module_writer.length,
        .flags = layout.flags,
    };
    module_writer.data = NULL;
  }

  loom_wasm_binary_writer_deinitialize(&module_writer);
  loom_wasm_module_deinitialize_function_bodies(&layout, allocator);
  if (!iree_status_is_ok(status)) {
    loom_wasm_module_binary_deinitialize(out_module, allocator);
  }
  return status;
}
