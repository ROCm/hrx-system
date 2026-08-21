// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/tooling/dump.h"

#include <string.h>

#include "iree/vm/bytecode/inspection.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/module_storage.h"
#include "iree/vm/bytecode/tooling/disassembler.h"

typedef struct iree_vm_bytecode_tooling_section_descriptor_t {
  // BSTRING offset of the canonical section name.
  uint16_t name_offset;
  // Architectural section type identifier.
  uint16_t section_type;
} iree_vm_bytecode_tooling_section_descriptor_t;

#include "iree/vm/bytecode/tooling/module_tables.c.inc"

static iree_string_view_t iree_vm_bytecode_dump_module_string(uint16_t offset) {
  const uint8_t* value = iree_vm_bytecode_tooling_module_string_table + offset;
  return iree_make_string_view((const char*)value + 1, value[0]);
}

static iree_status_t iree_vm_bytecode_dump_emit(
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder) {
  iree_status_t status = write_callback.fn(write_callback.user_data,
                                           iree_string_builder_view(builder));
  iree_string_builder_reset(builder);
  return status;
}

static iree_status_t iree_vm_bytecode_dump_append_quoted(
    iree_string_builder_t* builder, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '\\': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\\"));
        break;
      }
      case '"': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\""));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\n"));
        break;
      }
      case '\r': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\r"));
        break;
      }
      case '\t': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\t"));
        break;
      }
      default:
        if (c < 0x20 || c == 0x7F) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, "\\x%02X", (unsigned int)c));
        } else {
          char* output = NULL;
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_inline(builder, 1, &output));
          if (output) output[0] = (char)c;
        }
        break;
    }
  }
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t iree_vm_bytecode_dump_append_hex_bytes(
    iree_string_builder_t* builder, iree_const_byte_span_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "0x"));
  for (iree_host_size_t i = 0; i < value.data_length; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%02X", (unsigned int)value.data[i]));
  }
  return iree_ok_status();
}

static const char* iree_vm_bytecode_dump_scalar_name(
    iree_vm_scalar_type_t type) {
  switch (type) {
    case IREE_VM_SCALAR_TYPE_I8:
      return "i8";
    case IREE_VM_SCALAR_TYPE_I16:
      return "i16";
    case IREE_VM_SCALAR_TYPE_I32:
      return "i32";
    case IREE_VM_SCALAR_TYPE_I64:
      return "i64";
    case IREE_VM_SCALAR_TYPE_F8E4M3FN:
      return "f8e4m3fn";
    case IREE_VM_SCALAR_TYPE_F8E5M2:
      return "f8e5m2";
    case IREE_VM_SCALAR_TYPE_F16:
      return "f16";
    case IREE_VM_SCALAR_TYPE_BF16:
      return "bf16";
    case IREE_VM_SCALAR_TYPE_F32:
      return "f32";
    case IREE_VM_SCALAR_TYPE_F64:
      return "f64";
    default:
      return NULL;
  }
}

static iree_status_t iree_vm_bytecode_dump_append_signature_type(
    iree_string_builder_t* builder, iree_vm_signature_type_t type) {
  switch (type.kind) {
    case IREE_VM_SIGNATURE_TYPE_KIND_SCALAR: {
      const char* name = iree_vm_bytecode_dump_scalar_name(type.value.scalar);
      if (name) return iree_string_builder_append_cstring(builder, name);
      return iree_string_builder_append_format(builder, "scalar<%u>",
                                               (unsigned int)type.value.scalar);
    }
    case IREE_VM_SIGNATURE_TYPE_KIND_REF: {
      const iree_vm_ref_type_key_t key = iree_vm_ref_type_key(type.value.ref);
      return iree_string_builder_append_format(
          builder, "ref<%.*s, %.*s>", (int)key.namespace_name.size,
          key.namespace_name.data, (int)key.type_name.size, key.type_name.data);
    }
    case IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION: {
      const iree_string_view_t module_name =
          iree_vm_module_name(type.value.callable.module);
      return iree_string_builder_append_format(
          builder, "func<%.*s:%" PRIhsz ">", (int)module_name.size,
          module_name.data, type.value.callable.ordinal);
    }
    default:
      return iree_string_builder_append_cstring(builder, "invalid");
  }
}

static iree_status_t iree_vm_bytecode_dump_append_field_span(
    iree_string_builder_t* builder, iree_vm_signature_field_span_t fields) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "("));
  for (iree_host_size_t i = 0; i < fields.count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    const iree_vm_signature_field_t* field = &fields.data[i];
    if (!iree_string_view_is_empty(field->name)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%.*s: ", (int)field->name.size, field->name.data));
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_dump_append_signature_type(builder, field->type));
  }
  return iree_string_builder_append_cstring(builder, ")");
}

static iree_status_t iree_vm_bytecode_dump_append_type_span(
    iree_string_builder_t* builder, iree_vm_signature_type_span_t types) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "("));
  for (iree_host_size_t i = 0; i < types.count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_dump_append_signature_type(builder, types.data[i]));
  }
  return iree_string_builder_append_cstring(builder, ")");
}

static iree_status_t iree_vm_bytecode_dump_append_authored_fields(
    const char* field_kind, iree_string_builder_t* builder,
    iree_vm_signature_field_span_t fields,
    iree_vm_bytecode_dump_write_callback_t write_callback) {
  for (iree_host_size_t i = 0; i < fields.count; ++i) {
    const iree_vm_signature_field_t* field = &fields.data[i];
    if (iree_string_view_is_empty(field->authored_type)) continue;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "    %s[%" PRIhsz "].authored_type = ", field_kind, i));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_dump_append_quoted(builder, field->authored_type));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_dump_append_metadata_value(
    iree_string_builder_t* builder, iree_vm_metadata_value_t value) {
  switch (value.type) {
    case IREE_VM_METADATA_VALUE_TYPE_BOOL:
      return iree_string_builder_append_cstring(
          builder, value.data.data[0] ? "bool(true)" : "bool(false)");
    case IREE_VM_METADATA_VALUE_TYPE_I64: {
      const uint64_t bits = iree_unaligned_load_le_u64(value.data.data);
      int64_t signed_value = 0;
      memcpy(&signed_value, &bits, sizeof(signed_value));
      return iree_string_builder_append_format(builder, "i64(%" PRId64 ")",
                                               signed_value);
    }
    case IREE_VM_METADATA_VALUE_TYPE_U64:
      return iree_string_builder_append_format(
          builder, "u64(%" PRIu64 ")",
          iree_unaligned_load_le_u64(value.data.data));
    case IREE_VM_METADATA_VALUE_TYPE_F64:
      return iree_string_builder_append_format(
          builder, "f64(bits=0x%016" PRIX64 ")",
          iree_unaligned_load_le_u64(value.data.data));
    case IREE_VM_METADATA_VALUE_TYPE_UTF8: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, "utf8("));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_append_quoted(
          builder, iree_make_string_view((const char*)value.data.data,
                                         value.data.data_length)));
      return iree_string_builder_append_cstring(builder, ")");
    }
    case IREE_VM_METADATA_VALUE_TYPE_BYTES: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, "bytes("));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_dump_append_hex_bytes(builder, value.data));
      return iree_string_builder_append_cstring(builder, ")");
    }
    default: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "type<%u>(", (unsigned int)value.type));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_dump_append_hex_bytes(builder, value.data));
      return iree_string_builder_append_cstring(builder, ")");
    }
  }
}

static iree_status_t iree_vm_bytecode_dump_append_metadata_entry(
    const char* indentation, iree_vm_metadata_entry_t entry,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, indentation));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_append_quoted(builder, entry.key));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " = "));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_dump_append_metadata_value(builder, entry.value));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_vm_bytecode_dump_emit(write_callback, builder);
}

static iree_status_t iree_vm_bytecode_dump_append_declaration_presentation(
    iree_string_view_t documentation, iree_string_view_t authored_type,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder) {
  if (!iree_string_view_is_empty(documentation)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "    documentation = "));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_dump_append_quoted(builder, documentation));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  }
  if (!iree_string_view_is_empty(authored_type)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "    authored_type = "));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_dump_append_quoted(builder, authored_type));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_dump_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_allocator_t host_allocator, iree_string_builder_t* builder) {
  const iree_vm_callable_type_t callable_type = {module, ordinal};
  iree_host_size_t required_size = 0;
  IREE_RETURN_IF_ERROR(iree_vm_callable_type_query_description(
      callable_type, iree_byte_span_empty(), &required_size, NULL));

  void* storage_data = NULL;
  if (required_size != 0) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(host_allocator, required_size, &storage_data));
  }
  iree_vm_callable_type_description_t description = {0};
  iree_status_t status = iree_vm_callable_type_query_description(
      callable_type, iree_make_byte_span(storage_data, required_size),
      &required_size, &description);
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_format(builder, "  [%" PRIhsz "] ", ordinal);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_bytecode_dump_append_type_span(builder, description.arguments);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, " -> ");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_bytecode_dump_append_type_span(builder, description.results);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(description.flags,
                       IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD)) {
    status = iree_string_builder_append_cstring(builder, " may_yield");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, "\n");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_emit(write_callback, builder);
  }
  iree_allocator_free(host_allocator, storage_data);
  return status;
}

static iree_status_t iree_vm_bytecode_dump_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_allocator_t host_allocator, iree_string_builder_t* builder) {
  iree_vm_import_t import_value = {0};
  IREE_RETURN_IF_ERROR(
      iree_vm_module_import_by_ordinal(module, ordinal, &import_value));
  iree_host_size_t required_size = 0;
  IREE_RETURN_IF_ERROR(iree_vm_import_query_description(
      import_value, iree_byte_span_empty(), &required_size, NULL));

  void* storage_data = NULL;
  if (required_size != 0) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(host_allocator, required_size, &storage_data));
  }
  iree_vm_import_description_t description = {0};
  iree_status_t status = iree_vm_import_query_description(
      import_value, iree_make_byte_span(storage_data, required_size),
      &required_size, &description);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        builder, "  [%" PRIhsz "] @%.*s::%.*s : ", ordinal,
        (int)description.target.module_name.size,
        description.target.module_name.data,
        (int)description.target.export_name.size,
        description.target.export_name.data);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_bytecode_dump_append_field_span(builder, description.arguments);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, " -> ");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_bytecode_dump_append_field_span(builder, description.results);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(description.flags,
                       IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL)) {
    status = iree_string_builder_append_cstring(builder, " optional");
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(description.callable_flags,
                       IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD)) {
    status = iree_string_builder_append_cstring(builder, " may_yield");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, "\n");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_emit(write_callback, builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_append_declaration_presentation(
        description.documentation, description.authored_type, write_callback,
        builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_append_authored_fields(
        "argument", builder, description.arguments, write_callback);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_append_authored_fields(
        "result", builder, description.results, write_callback);
  }
  for (iree_host_size_t i = 0;
       i < iree_vm_import_metadata_count(import_value) &&
       iree_status_is_ok(status);
       ++i) {
    iree_vm_metadata_entry_t entry = {0};
    status = iree_vm_import_metadata_by_ordinal(import_value, i, &entry);
    if (iree_status_is_ok(status)) {
      status = iree_vm_bytecode_dump_append_metadata_entry(
          "    metadata ", entry, write_callback, builder);
    }
  }
  iree_allocator_free(host_allocator, storage_data);
  return status;
}

static iree_status_t iree_vm_bytecode_dump_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_allocator_t host_allocator, iree_string_builder_t* builder) {
  iree_vm_export_t export_value = {0};
  IREE_RETURN_IF_ERROR(
      iree_vm_module_export_by_ordinal(module, ordinal, &export_value));
  iree_vm_module_export_declaration_t declaration = {0};
  IREE_RETURN_IF_ERROR(
      iree_vm_module_query_export(module, ordinal, &declaration));
  iree_host_size_t required_size = 0;
  IREE_RETURN_IF_ERROR(iree_vm_export_query_description(
      export_value, iree_byte_span_empty(), &required_size, NULL));

  void* storage_data = NULL;
  if (required_size != 0) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(host_allocator, required_size, &storage_data));
  }
  iree_vm_export_description_t description = {0};
  iree_status_t status = iree_vm_export_query_description(
      export_value, iree_make_byte_span(storage_data, required_size),
      &required_size, &description);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        builder, "  [%" PRIhsz "] @%.*s -> function[%" PRIhsz "] : ", ordinal,
        (int)description.name.size, description.name.data,
        declaration.function_ordinal);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_bytecode_dump_append_field_span(builder, description.arguments);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, " -> ");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_bytecode_dump_append_field_span(builder, description.results);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(description.callable_flags,
                       IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD)) {
    status = iree_string_builder_append_cstring(builder, " may_yield");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, "\n");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_emit(write_callback, builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_append_declaration_presentation(
        description.documentation, description.authored_type, write_callback,
        builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_append_authored_fields(
        "argument", builder, description.arguments, write_callback);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_dump_append_authored_fields(
        "result", builder, description.results, write_callback);
  }
  for (iree_host_size_t i = 0;
       i < iree_vm_export_metadata_count(export_value) &&
       iree_status_is_ok(status);
       ++i) {
    iree_vm_metadata_entry_t entry = {0};
    status = iree_vm_export_metadata_by_ordinal(export_value, i, &entry);
    if (iree_status_is_ok(status)) {
      status = iree_vm_bytecode_dump_append_metadata_entry(
          "    metadata ", entry, write_callback, builder);
    }
  }
  iree_allocator_free(host_allocator, storage_data);
  return status;
}

static iree_status_t iree_vm_bytecode_dump_append_private_signature_type(
    const iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor,
    iree_string_builder_t* builder) {
  iree_vm_signature_type_t type = {0};
  if (descriptor->kind_u16 >= IREE_VM_BYTECODE_SIGNATURE_KIND_I8 &&
      descriptor->kind_u16 <= IREE_VM_BYTECODE_SIGNATURE_KIND_F64) {
    type.kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR;
    type.value.scalar = (iree_vm_scalar_type_t)descriptor->kind_u16;
  } else if (descriptor->kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
    type.kind = IREE_VM_SIGNATURE_TYPE_KIND_REF;
    type.value.ref = module->resolved_ref_types[descriptor->type_ordinal_u16];
  } else {
    type.kind = IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION;
    type.value.callable.module = &module->base;
    type.value.callable.ordinal = descriptor->type_ordinal_u16;
  }
  return iree_vm_bytecode_dump_append_signature_type(builder, type);
}

static iree_status_t iree_vm_bytecode_dump_append_private_signature_range(
    const iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors,
    uint32_t count, iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "("));
  for (uint32_t i = 0; i < count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_append_private_signature_type(
        module, &descriptors[i], builder));
  }
  return iree_string_builder_append_cstring(builder, ")");
}

static iree_status_t iree_vm_bytecode_dump_function(
    const iree_vm_bytecode_module_t* module, uint32_t ordinal,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder) {
  const iree_vm_bytecode_v0_function_row_t* function =
      &module->layout.functions.rows[ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &module->layout.signatures.rows[function->signature_ordinal_u16];
  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
      iree_vm_bytecode_signature_descriptors(&module->layout.signatures,
                                             function->signature_ordinal_u16);
  const uint32_t argument_count =
      iree_vm_bytecode_signature_argument_count(signature);
  const uint32_t result_count =
      iree_vm_bytecode_signature_result_count(signature);

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  [%" PRIu32 "] signature[%" PRIu16 "] bytecode=[%" PRIu32 ", +%" PRIu32
      ") registers=(%" PRIu16 ", %" PRIu16 ", %" PRIu16 ") locals=(%" PRIu16
      ", %" PRIu32 ", %" PRIu32 ") switch_targets=%" PRIu32,
      ordinal, function->signature_ordinal_u16, function->bytecode_offset_u32,
      function->bytecode_length_u32, function->value_register_count_u16,
      function->ref_register_count_u16, function->function_register_count_u16,
      function->local_byte_length_u16, function->local_ref_count_u32,
      function->local_function_count_u32,
      function->switch_target_entry_count_u32));
  if (iree_any_bit_set(function->flags_u16,
                       IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " may_yield"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "    type = "));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_append_private_signature_range(
      module, descriptors, argument_count, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " -> "));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_append_private_signature_range(
      module, descriptors + argument_count, result_count, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));

  for (uint32_t i = 0; i < function->switch_target_entry_count_u32; ++i) {
    const uint32_t target_word_offset =
        module->layout.functions
            .switch_targets[function->switch_target_base_u32 + i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "    switch_target[%" PRIu32 "] = +%" PRIu64 " (word=%" PRIu32 ")\n", i,
        (uint64_t)target_word_offset * 4u, target_word_offset));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  }

  return iree_vm_bytecode_disassemble_function(
      ordinal,
      iree_make_const_byte_span(module->layout.functions.bytecode_data +
                                    function->bytecode_offset_u32,
                                function->bytecode_length_u32),
      write_callback, builder);
}

static const iree_vm_bytecode_tooling_section_descriptor_t*
iree_vm_bytecode_dump_find_section(uint16_t section_type) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_vm_bytecode_tooling_sections); ++i) {
    if (iree_vm_bytecode_tooling_sections[i].section_type == section_type) {
      return &iree_vm_bytecode_tooling_sections[i];
    }
  }
  return NULL;
}

static iree_status_t iree_vm_bytecode_dump_sections(
    const iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "sections:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));

  iree_host_size_t offset = sizeof(*module->layout.image.header) +
                            module->layout.image.section_count *
                                sizeof(*module->layout.image.sections);
  for (uint16_t i = 0; i < module->layout.image.section_count; ++i) {
    const iree_vm_bytecode_v0_section_directory_row_t* row =
        &module->layout.image.sections[i];
    offset = iree_host_align(offset, IREE_VM_BYTECODE_SECTION_ALIGNMENT);
    const iree_vm_bytecode_tooling_section_descriptor_t* descriptor =
        iree_vm_bytecode_dump_find_section(row->section_type_u16);
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "  [%" PRIu16 "] ", i));
    if (descriptor) {
      const iree_string_view_t name =
          iree_vm_bytecode_dump_module_string(descriptor->name_offset);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%.*s", (int)name.size, name.data));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, "unknown"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " type=0x%04" PRIX16 " flags=0x%04" PRIX16 " bytes=[%" PRIhsz
        ", +%" PRIu64 ")\n",
        row->section_type_u16, row->section_flags_u16, offset,
        row->byte_length_u64));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
    offset += (iree_host_size_t)row->byte_length_u64;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_dump_contents(
    const iree_vm_module_t* base_module,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_allocator_t host_allocator, iree_string_builder_t* builder) {
  const iree_vm_bytecode_module_t* module =
      iree_vm_bytecode_module_cast_const(base_module);
  const iree_vm_bytecode_v0_image_header_t* header =
      module->layout.image.header;

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "vm.module "));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_append_quoted(
      builder, iree_vm_module_name(base_module)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " core=%" PRIu16 ".%" PRIu16 " bytes=%" PRIhsz "\n",
      header->core_major_u16, header->core_required_minor_u16,
      module->image->storage.contents.data_length));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "requirements:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  for (uint16_t i = 0; i < module->layout.requirements.count; ++i) {
    const iree_vm_bytecode_v0_requirement_row_t* requirement =
        &module->layout.requirements.rows[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  page=0x%02" PRIX16 " version=%" PRIu16 ".%" PRIu16 "\n",
        requirement->page_id_u16, requirement->major_u16,
        requirement->required_minor_u16));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_dump_sections(module, write_callback, builder));

  const iree_vm_bytecode_v0_globals_header_t* globals =
      module->layout.globals.header;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "resources: constants=%" PRIu32 " globals=(values=%" PRIu32
      ", refs=%" PRIu32 ", functions=%" PRIu32 ") rodata=%" PRIu32 "\n",
      module->layout.constants.count, globals ? globals->value_count_u32 : 0,
      globals ? globals->ref_count_u32 : 0,
      globals ? globals->function_count_u32 : 0, module->layout.rodata.count));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "module_metadata:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  for (iree_host_size_t i = 0; i < iree_vm_module_metadata_count(base_module);
       ++i) {
    iree_vm_metadata_entry_t entry = {0};
    IREE_RETURN_IF_ERROR(
        iree_vm_module_metadata_by_ordinal(base_module, i, &entry));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_append_metadata_entry(
        "  ", entry, write_callback, builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "ref_types:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  for (iree_host_size_t i = 0; i < iree_vm_module_ref_type_count(base_module);
       ++i) {
    iree_vm_ref_type_t type = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_module_ref_type_by_ordinal(base_module, i, &type));
    const iree_vm_ref_type_key_t key = iree_vm_ref_type_key(type);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  [%" PRIhsz "] ref<%.*s, %.*s>\n", i,
        (int)key.namespace_name.size, key.namespace_name.data,
        (int)key.type_name.size, key.type_name.data));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "callable_types:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  for (uint32_t i = 0; i < module->layout.callable_types.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_callable_type(
        base_module, i, write_callback, host_allocator, builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "imports:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  for (iree_host_size_t i = 0; i < iree_vm_module_import_count(base_module);
       ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_import(
        base_module, i, write_callback, host_allocator, builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "exports:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  for (iree_host_size_t i = 0; i < iree_vm_module_export_count(base_module);
       ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_export(
        base_module, i, write_callback, host_allocator, builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "functions:\n"));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_dump_emit(write_callback, builder));
  for (uint32_t i = 0; i < module->layout.functions.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_dump_function(module, i, write_callback, builder));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_dump(
    iree_string_view_t module_name, iree_const_byte_span_t contents,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_allocator_t host_allocator) {
  if (!write_callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "write callback is required");
  }

  iree_vm_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create_for_inspection(
      module_name,
      (iree_vm_bytecode_module_storage_t){contents, iree_allocator_null()},
      host_allocator, &module));

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status = iree_vm_bytecode_dump_contents(
      module, write_callback, host_allocator, &builder);
  iree_string_builder_deinitialize(&builder);
  iree_vm_module_release(module);
  return status;
}
