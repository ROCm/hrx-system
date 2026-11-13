// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/module_binary.h"

#include <string.h>

#include "loom/target/artifact_diagnostics.h"
#include "loom/target/emit/wasm/binary_writer.h"
#include "loom/target/emit/wasm/function_body.h"
#include "loom/target/emit/wasm/types.h"

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
  LOOM_WASM_EXPORT_KIND_MEMORY = 2,
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

typedef struct loom_wasm_module_layout_t {
  // Caller sink for module-wide format rejection.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // False after body or section rejection; partial bytes are never published.
  bool accepted;
  // Planned functions in WebAssembly function-index order.
  const loom_wasm_function_plan_t* functions;
  // Number of planned functions.
  iree_host_size_t function_count;
  // Planned interned function signatures.
  const loom_wasm_function_type_t* types;
  // Number of planned function signatures.
  iree_host_size_t type_count;
  // Number of planned function exports.
  iree_host_size_t function_export_count;
  // Allocator-owned encoded function bodies.
  loom_wasm_function_body_t* bodies;
  // Structural facts represented in the emitted module binary.
  loom_wasm_module_binary_flags_t flags;
} loom_wasm_module_layout_t;

static iree_status_t loom_wasm_module_write_section(
    loom_wasm_module_layout_t* layout, loom_wasm_binary_writer_t* module_writer,
    uint8_t section_id, iree_string_view_t quantity,
    const loom_wasm_binary_writer_t* payload_writer) {
  if (payload_writer->length > UINT32_MAX) {
    layout->accepted = false;
    return loom_target_artifact_emit_limit(layout->diagnostic_emitter, NULL,
                                           IREE_SV("Wasm"), quantity,
                                           payload_writer->length, UINT32_MAX);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(module_writer, section_id));
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      module_writer, (uint32_t)payload_writer->length));
  return loom_wasm_binary_write_bytes(module_writer, payload_writer->data,
                                      payload_writer->length);
}

static iree_status_t loom_wasm_module_write_name(
    loom_wasm_binary_writer_t* writer, iree_string_view_t name) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(writer, (uint32_t)name.size));
  return loom_wasm_binary_write_bytes(writer, (const uint8_t*)name.data,
                                      name.size);
}

static void loom_wasm_module_layout_deinitialize(
    loom_wasm_module_layout_t* layout, iree_allocator_t allocator) {
  if (layout == NULL) {
    return;
  }
  if (layout->bodies != NULL) {
    for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
      loom_wasm_function_body_deinitialize(&layout->bodies[i], allocator);
    }
    iree_allocator_free(allocator, layout->bodies);
  }
  *layout = (loom_wasm_module_layout_t){0};
}

static iree_status_t loom_wasm_module_layout_initialize(
    const loom_wasm_program_plan_t* plan,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_allocator_t allocator,
    loom_wasm_module_layout_t* out_layout) {
  *out_layout = (loom_wasm_module_layout_t){
      .diagnostic_emitter = diagnostic_emitter,
      .accepted = true,
      .functions = plan->functions,
      .function_count = plan->function_count,
      .types = plan->types,
      .type_count = plan->type_count,
      .function_export_count = plan->export_count,
  };

  const iree_host_size_t body_storage_size =
      plan->function_count * sizeof(*out_layout->bodies);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, body_storage_size,
                                             (void**)&out_layout->bodies));
  memset(out_layout->bodies, 0, body_storage_size);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < plan->function_count && iree_status_is_ok(status) &&
       out_layout->accepted;
       ++i) {
    status = loom_wasm_emit_function_body(
        plan, &plan->functions[i], diagnostic_emitter, allocator,
        &out_layout->accepted, &out_layout->bodies[i]);
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(out_layout->bodies[i].flags,
                         LOOM_WASM_FUNCTION_BODY_FLAG_USES_MEMORY)) {
      out_layout->flags |= LOOM_WASM_MODULE_BINARY_FLAG_DEFINES_MEMORY;
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_wasm_module_layout_deinitialize(out_layout, allocator);
  }
  return status;
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
    loom_wasm_module_layout_t* layout, loom_wasm_binary_writer_t* module_writer,
    iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_type_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        layout, module_writer, LOOM_WASM_SECTION_TYPE,
        IREE_SV("type section byte length"), &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_function_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      payload_writer, (uint32_t)layout->function_count));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
        payload_writer, layout->functions[i].type_index));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_function_section(
    loom_wasm_module_layout_t* layout, loom_wasm_binary_writer_t* module_writer,
    iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_function_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        layout, module_writer, LOOM_WASM_SECTION_FUNCTION,
        IREE_SV("function section byte length"), &payload_writer);
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
    loom_wasm_module_layout_t* layout, loom_wasm_binary_writer_t* module_writer,
    iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_memory_section_payload(&payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        layout, module_writer, LOOM_WASM_SECTION_MEMORY,
        IREE_SV("memory section byte length"), &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_export_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer) {
  const bool exports_memory = iree_any_bit_set(
      layout->flags, LOOM_WASM_MODULE_BINARY_FLAG_DEFINES_MEMORY);
  const uint64_t export_count =
      (uint64_t)layout->function_export_count + (exports_memory ? 1u : 0u);
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(payload_writer, (uint32_t)export_count));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    const loom_wasm_function_plan_t* function = &layout->functions[i];
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
  if (exports_memory) {
    IREE_RETURN_IF_ERROR(
        loom_wasm_module_write_name(payload_writer, IREE_SV("memory")));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(
        payload_writer, LOOM_WASM_EXPORT_KIND_MEMORY));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(payload_writer, 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_export_section(
    loom_wasm_module_layout_t* layout, loom_wasm_binary_writer_t* module_writer,
    iree_allocator_t allocator) {
  if (layout->function_export_count == 0 &&
      !iree_any_bit_set(layout->flags,
                        LOOM_WASM_MODULE_BINARY_FLAG_DEFINES_MEMORY)) {
    return iree_ok_status();
  }
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_export_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        layout, module_writer, LOOM_WASM_SECTION_EXPORT,
        IREE_SV("export section byte length"), &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_code_section_payload(
    const loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      payload_writer, (uint32_t)layout->function_count));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    const loom_wasm_function_body_t* body = &layout->bodies[i];
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_bytes(
        payload_writer, body->data, body->data_length));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_code_section(
    loom_wasm_module_layout_t* layout, loom_wasm_binary_writer_t* module_writer,
    iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_code_section_payload(layout, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        layout, module_writer, LOOM_WASM_SECTION_CODE,
        IREE_SV("code section byte length"), &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_name_section_payload(
    loom_wasm_module_layout_t* layout,
    loom_wasm_binary_writer_t* payload_writer, iree_allocator_t allocator) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_module_write_name(payload_writer, IREE_SV("name")));

  loom_wasm_binary_writer_t function_names_writer;
  loom_wasm_binary_writer_initialize(allocator, &function_names_writer);
  iree_status_t status = loom_wasm_binary_write_u32_leb(
      &function_names_writer, (uint32_t)layout->function_count);
  for (iree_host_size_t i = 0;
       i < layout->function_count && iree_status_is_ok(status); ++i) {
    const loom_wasm_function_plan_t* function = &layout->functions[i];
    status = loom_wasm_binary_write_u32_leb(&function_names_writer,
                                            function->function_index);
    if (iree_status_is_ok(status)) {
      status =
          loom_wasm_module_write_name(&function_names_writer, function->name);
    }
  }
  if (iree_status_is_ok(status) && function_names_writer.length > UINT32_MAX) {
    layout->accepted = false;
    status = loom_target_artifact_emit_limit(
        layout->diagnostic_emitter, NULL, IREE_SV("Wasm"),
        IREE_SV("function-name subsection byte length"),
        function_names_writer.length, UINT32_MAX);
  }
  if (iree_status_is_ok(status) && layout->accepted) {
    status = loom_wasm_binary_write_u8(payload_writer,
                                       LOOM_WASM_NAME_SUBSECTION_FUNCTIONS);
  }
  if (iree_status_is_ok(status) && layout->accepted) {
    status = loom_wasm_binary_write_u32_leb(
        payload_writer, (uint32_t)function_names_writer.length);
  }
  if (iree_status_is_ok(status) && layout->accepted) {
    status =
        loom_wasm_binary_write_bytes(payload_writer, function_names_writer.data,
                                     function_names_writer.length);
  }
  loom_wasm_binary_writer_deinitialize(&function_names_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_name_section(
    loom_wasm_module_layout_t* layout, loom_wasm_binary_writer_t* module_writer,
    iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status = loom_wasm_module_write_name_section_payload(
      layout, &payload_writer, allocator);
  if (iree_status_is_ok(status) && layout->accepted) {
    status = loom_wasm_module_write_section(
        layout, module_writer, LOOM_WASM_SECTION_CUSTOM,
        IREE_SV("name section byte length"), &payload_writer);
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

iree_status_t loom_wasm_program_emit_binary(
    const loom_wasm_program_plan_t* plan,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_allocator_t allocator,
    bool* out_accepted, loom_wasm_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_accepted = false;
  *out_module = (loom_wasm_module_binary_t){0};

  loom_wasm_module_layout_t layout = {0};
  iree_status_t status = loom_wasm_module_layout_initialize(
      plan, diagnostic_emitter, allocator, &layout);

  loom_wasm_binary_writer_t module_writer;
  loom_wasm_binary_writer_initialize(allocator, &module_writer);
  if (iree_status_is_ok(status) && layout.accepted) {
    status = loom_wasm_module_write_header(&module_writer);
  }
  if (iree_status_is_ok(status) && layout.accepted) {
    status =
        loom_wasm_module_write_type_section(&layout, &module_writer, allocator);
  }
  if (iree_status_is_ok(status) && layout.accepted) {
    status = loom_wasm_module_write_function_section(&layout, &module_writer,
                                                     allocator);
  }
  if (iree_status_is_ok(status) && layout.accepted &&
      iree_any_bit_set(layout.flags,
                       LOOM_WASM_MODULE_BINARY_FLAG_DEFINES_MEMORY)) {
    status = loom_wasm_module_write_memory_section(&layout, &module_writer,
                                                   allocator);
  }
  if (iree_status_is_ok(status) && layout.accepted) {
    status = loom_wasm_module_write_export_section(&layout, &module_writer,
                                                   allocator);
  }
  if (iree_status_is_ok(status) && layout.accepted) {
    status =
        loom_wasm_module_write_code_section(&layout, &module_writer, allocator);
  }
  if (iree_status_is_ok(status) && layout.accepted) {
    status =
        loom_wasm_module_write_name_section(&layout, &module_writer, allocator);
  }
  if (iree_status_is_ok(status) && layout.accepted) {
    *out_module = (loom_wasm_module_binary_t){
        .data = module_writer.data,
        .data_length = module_writer.length,
        .flags = layout.flags,
    };
    module_writer.data = NULL;
    *out_accepted = true;
  }

  loom_wasm_binary_writer_deinitialize(&module_writer);
  loom_wasm_module_layout_deinitialize(&layout, allocator);
  if (!iree_status_is_ok(status)) {
    loom_wasm_module_binary_deinitialize(out_module, allocator);
  }
  return status;
}
