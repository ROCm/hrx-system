// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_presentation.h"

#include <string.h>

#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/codegen/low/function.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/emit/vm/module_layout.h"
#include "loom/target/emit/vm/module_types.h"
#include "loom/util/stream.h"

static iree_status_t loom_vm_module_presentation_print_field_type(
    const loom_module_t* module, loom_type_t register_type,
    loom_output_stream_t* stream) {
  const loom_type_t* logical_type =
      loom_type_register_value_type(register_type);
  if (logical_type == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM presentation field register has no logical type");
  }
  return loom_text_print_type(*logical_type, module, stream);
}

static iree_status_t loom_vm_module_presentation_print_signature(
    const loom_module_t* module, loom_type_t signature,
    loom_output_stream_t* stream) {
  const uint16_t argument_count = loom_type_func_arg_count(signature);
  const loom_type_t* argument_types = loom_type_func_arg_types(signature);
  const uint16_t result_count = loom_type_func_result_count(signature);
  const loom_type_t* result_types = loom_type_func_result_types(signature);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '('));
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_print_field_type(
        module, argument_types[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ") -> ("));
  for (uint16_t i = 0; i < result_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_print_field_type(
        module, result_types[i], stream));
  }
  return loom_output_stream_write_char(stream, ')');
}

static iree_status_t loom_vm_module_presentation_measure_field_type(
    const loom_module_t* module, loom_type_t register_type,
    iree_host_size_t* out_length) {
  loom_output_stream_t stream;
  loom_output_stream_null(&stream);
  IREE_RETURN_IF_ERROR(loom_vm_module_presentation_print_field_type(
      module, register_type, &stream));
  *out_length = stream.offset;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_presentation_measure_signature(
    const loom_module_t* module, loom_type_t signature,
    iree_host_size_t* out_length) {
  loom_output_stream_t stream;
  loom_output_stream_null(&stream);
  IREE_RETURN_IF_ERROR(
      loom_vm_module_presentation_print_signature(module, signature, &stream));
  *out_length = stream.offset;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_presentation_allocate_text(
    iree_host_size_t text_length, iree_host_size_t* inout_storage_length) {
  iree_host_size_t storage_length = 0;
  if (!iree_host_size_checked_add(text_length, 1, &storage_length) ||
      !iree_host_size_checked_add(*inout_storage_length, storage_length,
                                  inout_storage_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM presentation text exceeds host size");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_presentation_measure_documentation(
    const loom_module_t* module, const loom_op_t* op,
    iree_host_size_t* out_length) {
  *out_length = 0;
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    if ((i != 0 && !iree_host_size_checked_add(*out_length, 1, out_length)) ||
        !iree_host_size_checked_add(*out_length, comments[i].size,
                                    out_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM declaration documentation exceeds host "
                              "size");
    }
  }
  return iree_ok_status();
}

static void loom_vm_module_presentation_emit_documentation(
    const loom_module_t* module, const loom_op_t* op, char* storage,
    iree_host_size_t storage_capacity, iree_string_view_t* out_text) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    if (i != 0) storage[offset++] = '\n';
    if (comments[i].size != 0) {
      memcpy(storage + offset, comments[i].data, comments[i].size);
    }
    offset += comments[i].size;
  }
  IREE_ASSERT_EQ(offset + 1, storage_capacity);
  storage[offset] = 0;
  *out_text = iree_make_string_view(storage, offset);
}

static iree_status_t loom_vm_module_presentation_emit_field_type(
    const loom_module_t* module, loom_type_t register_type, char* storage,
    iree_host_size_t storage_capacity, iree_string_view_t* out_text) {
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(storage, storage_capacity,
                                              &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_vm_module_presentation_print_field_type(
      module, register_type, &stream);
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(iree_string_builder_size(&builder) + 1, storage_capacity);
    *out_text = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_vm_module_presentation_emit_signature(
    const loom_module_t* module, loom_type_t signature, char* storage,
    iree_host_size_t storage_capacity, iree_string_view_t* out_text) {
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(storage, storage_capacity,
                                              &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status =
      loom_vm_module_presentation_print_signature(module, signature, &stream);
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(iree_string_builder_size(&builder) + 1, storage_capacity);
    *out_text = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

typedef struct loom_vm_module_presentation_declaration_t {
  // Prepared public import declaration or export definition.
  loom_op_t* op;
  // Preserved source-ordered logical callable signature.
  loom_type_t signature;
  // Canonical module-local machine callable type.
  uint16_t callable_type_ordinal;
  // Import or export declaration kind.
  uint16_t kind;
  // Ordinal in the selected public declaration domain.
  uint16_t ordinal;
} loom_vm_module_presentation_declaration_t;

static loom_vm_module_presentation_declaration_t
loom_vm_module_presentation_declaration(const loom_vm_module_layout_t* layout,
                                        uint32_t entry_index) {
  if (entry_index < layout->import_count) {
    const loom_vm_module_import_layout_t* import = layout->imports[entry_index];
    return (loom_vm_module_presentation_declaration_t){
        .op = import->declaration_op,
        .signature = import->logical_signature,
        .callable_type_ordinal = import->callable_type_ordinal,
        .kind = IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT,
        .ordinal = (uint16_t)entry_index,
    };
  }
  const uint32_t export_index = entry_index - (uint32_t)layout->import_count;
  const loom_vm_module_function_layout_t* function =
      layout->exports[export_index];
  return (loom_vm_module_presentation_declaration_t){
      .op = function->function_op,
      .signature = function->logical_signature,
      .callable_type_ordinal = function->callable_type_ordinal,
      .kind = IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_EXPORT,
      .ordinal = (uint16_t)export_index,
  };
}

static iree_string_view_t loom_vm_module_presentation_field_name(
    const loom_module_t* module, loom_named_attr_slice_t names,
    uint16_t field_ordinal) {
  if (names.count == 0) return iree_string_view_empty();
  return module->strings.entries[names.entries[field_ordinal].value.string_id];
}

typedef struct loom_vm_module_presentation_field_counts_t {
  // Number of machine argument fields.
  uint32_t arguments;
  // Number of machine result fields.
  uint32_t results;
} loom_vm_module_presentation_field_counts_t;

static loom_vm_module_presentation_field_counts_t
loom_vm_module_presentation_field_counts(
    const loom_vm_module_layout_t* layout,
    loom_vm_module_presentation_declaration_t declaration) {
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &layout->type_tables.callable_types[declaration.callable_type_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &layout->type_tables.signatures[callable_type->signature_ordinal_u16];
  return (loom_vm_module_presentation_field_counts_t){
      .arguments = (uint32_t)signature->argument_value_count_u16 +
                   signature->argument_ref_count_u16 +
                   signature->argument_function_count_u16,
      .results = (uint32_t)signature->result_value_count_u16 +
                 signature->result_ref_count_u16 +
                 signature->result_function_count_u16,
  };
}

static iree_status_t loom_vm_module_presentation_count_fields(
    const loom_vm_module_layout_t* layout, uint32_t entry_count,
    uint32_t* out_field_count) {
  *out_field_count = 0;
  for (uint32_t i = 0; i < entry_count; ++i) {
    const loom_vm_module_presentation_declaration_t declaration =
        loom_vm_module_presentation_declaration(layout, i);
    const loom_vm_module_presentation_field_counts_t counts =
        loom_vm_module_presentation_field_counts(layout, declaration);
    const uint32_t field_count = counts.arguments + counts.results;
    if (field_count > UINT32_MAX - *out_field_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM presentation field count exceeds u32");
    }
    *out_field_count += field_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_presentation_prepare_fields(
    const loom_module_t* module, const loom_type_t* register_types,
    uint16_t logical_count, loom_named_attr_slice_t names,
    const loom_value_id_t* values, uint16_t value_count,
    loom_vm_module_presentation_field_layout_t* fields, uint32_t machine_count,
    iree_host_size_t* inout_text_storage_length) {
  uint32_t machine_ordinal = 0;
  for (uint16_t i = 0; i < logical_count; ++i) {
    loom_vm_call_abi_register_layout_t register_layout = {0};
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_classify_type(
        module, register_types[i], &register_layout));
    IREE_ASSERT_LE(machine_ordinal, machine_count);
    IREE_ASSERT_LE((uint32_t)register_layout.unit_count,
                   machine_count - machine_ordinal);
    for (uint16_t j = 0; j < register_layout.unit_count; ++j) {
      fields[machine_ordinal + j] =
          (loom_vm_module_presentation_field_layout_t){
              .name_string_ordinal = UINT16_MAX,
              .authored_type_string_ordinal = UINT16_MAX,
          };
    }

    // Presentation follows the machine signature. Anchor source aggregate
    // metadata at its first machine field and leave continuation fields empty.
    loom_vm_module_presentation_field_layout_t* field =
        &fields[machine_ordinal];
    field->register_type = &register_types[i];
    field->name = names.count != 0
                      ? loom_vm_module_presentation_field_name(module, names, i)
                  : value_count == logical_count
                      ? loom_module_value_name(module, values[i])
                      : iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_measure_field_type(
        module, register_types[i], &field->authored_type.size));
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_allocate_text(
        field->authored_type.size, inout_text_storage_length));
    machine_ordinal += register_layout.unit_count;
  }
  IREE_ASSERT_EQ(machine_ordinal, machine_count);
  return iree_ok_status();
}

iree_status_t loom_vm_module_presentation_layout_build(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout) {
  layout->presentation = (loom_vm_module_presentation_layout_t){0};
  const iree_host_size_t host_entry_count =
      layout->import_count + layout->export_count;
  if (host_entry_count == 0) return iree_ok_status();
  if (host_entry_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM presentation entry count exceeds u32");
  }
  const uint32_t entry_count = (uint32_t)host_entry_count;
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_presentation_count_fields(
      layout, entry_count, &field_count));

  loom_vm_module_presentation_entry_layout_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entry_count, sizeof(*entries), (void**)&entries));
  loom_vm_module_presentation_field_layout_t* fields = NULL;
  if (field_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, field_count, sizeof(*fields), (void**)&fields));
  }
  layout->presentation = (loom_vm_module_presentation_layout_t){
      .entries = entries,
      .entry_count = entry_count,
      .fields = fields,
      .field_count = field_count,
  };

  iree_host_size_t text_storage_length = 0;
  uint32_t field_base = 0;
  for (uint32_t i = 0; i < entry_count; ++i) {
    const loom_vm_module_presentation_declaration_t declaration =
        loom_vm_module_presentation_declaration(layout, i);
    const uint16_t argument_count =
        loom_type_func_arg_count(declaration.signature);
    const uint16_t result_count =
        loom_type_func_result_count(declaration.signature);
    const loom_vm_module_presentation_field_counts_t machine_counts =
        loom_vm_module_presentation_field_counts(layout, declaration);
    const uint32_t declaration_field_count =
        machine_counts.arguments + machine_counts.results;
    entries[i] = (loom_vm_module_presentation_entry_layout_t){
        .declaration_kind = declaration.kind,
        .declaration_ordinal = declaration.ordinal,
        .authored_type_string_ordinal = UINT16_MAX,
        .documentation_string_ordinal = UINT16_MAX,
        .field_base = field_base,
    };
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_measure_documentation(
        layout->module, declaration.op, &entries[i].documentation.size));
    if (entries[i].documentation.size != 0) {
      IREE_RETURN_IF_ERROR(loom_vm_module_presentation_allocate_text(
          entries[i].documentation.size, &text_storage_length));
    }
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_measure_signature(
        layout->module, declaration.signature, &entries[i].authored_type.size));
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_allocate_text(
        entries[i].authored_type.size, &text_storage_length));

    const loom_func_like_t function =
        loom_func_like_cast(layout->module, declaration.op);
    uint16_t physical_argument_count = 0;
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function, &physical_argument_count);
    const loom_value_id_t* results = loom_op_const_results(declaration.op);
    const uint16_t physical_result_count = declaration.op->result_count;
    const loom_type_t* argument_types =
        loom_type_func_arg_types(declaration.signature);
    const loom_type_t* result_types =
        loom_type_func_result_types(declaration.signature);
    const loom_named_attr_slice_t abi_layout =
        loom_low_function_def_isa(declaration.op)
            ? loom_low_function_abi_layout(declaration.op)
            : loom_low_func_decl_abi_layout(declaration.op);
    loom_named_attr_slice_t argument_names = loom_named_attr_slice_empty();
    loom_named_attr_slice_t result_names = loom_named_attr_slice_empty();
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_resolve_presentation_names(
        layout->module, abi_layout, argument_count, result_count,
        &argument_names, &result_names));
    loom_vm_module_presentation_field_layout_t* argument_fields =
        machine_counts.arguments != 0 ? &fields[field_base] : NULL;
    loom_vm_module_presentation_field_layout_t* result_fields =
        machine_counts.results != 0
            ? &fields[field_base + machine_counts.arguments]
            : NULL;
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_prepare_fields(
        layout->module, argument_types, argument_count, argument_names,
        arguments, physical_argument_count, argument_fields,
        machine_counts.arguments, &text_storage_length));
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_prepare_fields(
        layout->module, result_types, result_count, result_names, results,
        physical_result_count, result_fields, machine_counts.results,
        &text_storage_length));
    field_base += declaration_field_count;
  }
  IREE_ASSERT_EQ(field_base, field_count);

  char* text_storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, text_storage_length, (void**)&text_storage));
  iree_host_size_t text_offset = 0;
  for (uint32_t i = 0; i < entry_count; ++i) {
    const loom_vm_module_presentation_declaration_t declaration =
        loom_vm_module_presentation_declaration(layout, i);
    loom_vm_module_presentation_entry_layout_t* entry = &entries[i];
    if (entry->documentation.size != 0) {
      const iree_host_size_t documentation_capacity =
          entry->documentation.size + 1;
      loom_vm_module_presentation_emit_documentation(
          layout->module, declaration.op, text_storage + text_offset,
          documentation_capacity, &entry->documentation);
      text_offset += documentation_capacity;
    }
    const iree_host_size_t signature_capacity = entry->authored_type.size + 1;
    IREE_RETURN_IF_ERROR(loom_vm_module_presentation_emit_signature(
        layout->module, declaration.signature, text_storage + text_offset,
        signature_capacity, &entry->authored_type));
    text_offset += signature_capacity;

    const loom_vm_module_presentation_field_counts_t machine_counts =
        loom_vm_module_presentation_field_counts(layout, declaration);
    const uint32_t machine_field_count =
        machine_counts.arguments + machine_counts.results;
    for (uint32_t j = 0; j < machine_field_count; ++j) {
      loom_vm_module_presentation_field_layout_t* field =
          &fields[entry->field_base + j];
      if (field->register_type == NULL) continue;
      const iree_host_size_t field_capacity = field->authored_type.size + 1;
      IREE_RETURN_IF_ERROR(loom_vm_module_presentation_emit_field_type(
          layout->module, *field->register_type, text_storage + text_offset,
          field_capacity, &field->authored_type));
      text_offset += field_capacity;
    }
  }
  IREE_ASSERT_EQ(text_offset, text_storage_length);
  return iree_ok_status();
}

static void loom_vm_module_presentation_resolve_nullable_string(
    const loom_vm_module_type_tables_t* tables, iree_string_view_t value,
    uint16_t* out_ordinal) {
  *out_ordinal = UINT16_MAX;
  if (iree_string_view_is_empty(value)) return;
  const bool resolved = loom_vm_module_type_tables_try_resolve_string_ordinal(
      tables, value, out_ordinal);
  IREE_ASSERT(resolved);
}

void loom_vm_module_presentation_resolve_string_ordinals(
    loom_vm_module_layout_t* layout) {
  const loom_vm_module_type_tables_t* tables = &layout->type_tables;
  for (uint32_t i = 0; i < layout->presentation.entry_count; ++i) {
    loom_vm_module_presentation_entry_layout_t* entry =
        &layout->presentation.entries[i];
    loom_vm_module_presentation_resolve_nullable_string(
        tables, entry->documentation, &entry->documentation_string_ordinal);
    loom_vm_module_presentation_resolve_nullable_string(
        tables, entry->authored_type, &entry->authored_type_string_ordinal);
  }
  for (uint32_t i = 0; i < layout->presentation.field_count; ++i) {
    loom_vm_module_presentation_field_layout_t* field =
        &layout->presentation.fields[i];
    loom_vm_module_presentation_resolve_nullable_string(
        tables, field->name, &field->name_string_ordinal);
    loom_vm_module_presentation_resolve_nullable_string(
        tables, field->authored_type, &field->authored_type_string_ordinal);
  }
}
