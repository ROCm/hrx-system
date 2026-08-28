// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_emitter.h"

#include <string.h>

#include "iree/io/vec_stream.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/format/bytecode/writer/encoder.h"
#include "loom/target/emit/vm/function_encoder.h"
#include "loom/target/emit/vm/module_layout.h"

enum {
  LOOM_VM_MODULE_MAX_SECTION_COUNT = 12,
  LOOM_VM_MODULE_STREAM_BLOCK_SIZE = 32 * 1024,
};

typedef struct loom_vm_module_writer_t {
  // Segmented stream receiving the module image.
  iree_io_stream_t* stream;
  // Page-buffered forward writer over |stream|.
  loom_bytecode_page_writer_t writer;
  // Final section-directory rows in strict section-type order.
  iree_vm_bytecode_v0_section_directory_row_t
      sections[LOOM_VM_MODULE_MAX_SECTION_COUNT];
  // Number of initialized entries in |sections|.
  uint16_t section_count;
  // Absolute stream offset of the Functions section row array.
  uint64_t function_rows_offset;
} loom_vm_module_writer_t;

static iree_status_t loom_vm_module_writer_write_record(
    loom_vm_module_writer_t* writer, const void* record,
    iree_host_size_t record_size) {
  return loom_bytecode_page_writer_write(&writer->writer, record, record_size);
}

static iree_status_t loom_vm_module_writer_begin_section_with_flags(
    loom_vm_module_writer_t* writer, uint16_t section_type,
    uint16_t section_flags, uint32_t payload_alignment,
    uint64_t* out_section_start) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_pad_to_alignment(
      &writer->writer, payload_alignment));
  IREE_ASSERT_LT(writer->section_count, LOOM_VM_MODULE_MAX_SECTION_COUNT);
  writer->sections[writer->section_count] =
      (iree_vm_bytecode_v0_section_directory_row_t){
          .section_type_u16 = section_type,
          .section_flags_u16 = section_flags,
          .payload_alignment_u32 = payload_alignment,
      };
  *out_section_start = writer->writer.total_written;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_writer_begin_section(
    loom_vm_module_writer_t* writer, uint16_t section_type,
    uint32_t payload_alignment, uint64_t* out_section_start) {
  return loom_vm_module_writer_begin_section_with_flags(
      writer, section_type, /*section_flags=*/0, payload_alignment,
      out_section_start);
}

static void loom_vm_module_writer_end_section(loom_vm_module_writer_t* writer,
                                              uint64_t section_start) {
  iree_vm_bytecode_v0_section_directory_row_t* row =
      &writer->sections[writer->section_count++];
  row->byte_length_u64 = writer->writer.total_written - section_start;
  IREE_ASSERT_NE(row->byte_length_u64, 0u);
}

static bool loom_vm_module_has_globals(const loom_vm_module_layout_t* layout) {
  return layout->resources.value_global_count != 0 ||
         layout->resources.ref_global_count != 0 ||
         layout->resources.function_global_count != 0;
}

static iree_status_t loom_vm_module_write_strings(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_type_tables_t* tables = &layout->type_tables;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_STRINGS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_strings_header_t header = {
      .string_count_u32 = tables->string_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));

  uint32_t string_offset = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_u32_le(&writer->writer, string_offset));
  for (uint32_t i = 0; i < tables->string_count; ++i) {
    const iree_string_view_t value = tables->strings[i];
    if (value.size > UINT32_MAX - string_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM string data length exceeds u32");
    }
    string_offset += (uint32_t)value.size;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u32_le(&writer->writer, string_offset));
  }
  for (uint32_t i = 0; i < tables->string_count; ++i) {
    const iree_string_view_t value = tables->strings[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
        &writer->writer, value.data, value.size));
  }
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_ref_types(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_type_tables_t* tables = &layout->type_tables;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_REF_TYPES,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_ref_types_header_t header = {
      .group_count_u32 = tables->ref_type_group_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
      writer, tables->ref_type_groups,
      tables->ref_type_group_count * sizeof(*tables->ref_type_groups)));
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
      writer, tables->ref_type_entries,
      tables->ref_type_entry_count * sizeof(*tables->ref_type_entries)));
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_signatures(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_type_tables_t* tables = &layout->type_tables;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_SIGNATURES,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_signatures_header_t header = {
      .signature_count_u32 = tables->signature_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));

  IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
      writer, tables->signatures,
      tables->signature_count * sizeof(*tables->signatures)));
  if (tables->signature_descriptor_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
        writer, tables->signature_descriptors,
        tables->signature_descriptor_count *
            sizeof(*tables->signature_descriptors)));
  }
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_callable_types(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_type_tables_t* tables = &layout->type_tables;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_callable_types_header_t header = {
      .callable_type_count_u32 = tables->callable_type_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
      writer, tables->callable_types,
      tables->callable_type_count * sizeof(*tables->callable_types)));
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_imports(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_IMPORTS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_imports_header_t header = {
      .group_count_u32 = (uint32_t)layout->import_group_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  for (iree_host_size_t i = 0; i < layout->import_group_count; ++i) {
    const loom_vm_module_import_group_layout_t* group =
        &layout->import_groups[i];
    const iree_vm_bytecode_v0_import_group_row_t row = {
        .module_name_string_u16 =
            group->first_import->module_name_string_ordinal,
        .entry_count_u32 = group->import_count,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  for (iree_host_size_t i = 0; i < layout->import_count; ++i) {
    const loom_vm_module_import_layout_t* import = layout->imports[i];
    const iree_vm_bytecode_v0_import_entry_row_t row = {
        .symbol_name_string_u16 = import->symbol_name_string_ordinal,
        .callable_type_ordinal_u16 = import->callable_type_ordinal,
        .flags_u16 = import->flags,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_exports(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_EXPORTS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_exports_header_t header = {
      .export_count_u32 = (uint32_t)layout->export_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  for (iree_host_size_t i = 0; i < layout->export_count; ++i) {
    const loom_vm_module_function_layout_t* function = layout->exports[i];
    const iree_vm_bytecode_v0_export_row_t row = {
        .name_string_u16 = function->export_name_string_ordinal,
        .callable_type_ordinal_u16 = function->callable_type_ordinal,
        .function_ordinal_u16 = function->function_ordinal,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_functions(
    const loom_vm_module_layout_t* layout,
    const loom_vm_module_emitter_options_t* options,
    iree_arena_allocator_t* scratch_arena, loom_vm_module_writer_t* writer,
    iree_vm_bytecode_v0_function_row_t* function_rows,
    iree_vm_bytecode_v0_switch_target_entry_t* switch_targets,
    bool* out_complete) {
  *out_complete = false;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_FUNCTIONS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_functions_header_t header = {
      .function_count_u32 = (uint32_t)layout->function_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  writer->function_rows_offset = writer->writer.total_written;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_zeros(
      &writer->writer, layout->function_count * sizeof(*function_rows)));
  iree_host_size_t switch_target_byte_length = 0;
  if (!iree_host_size_checked_mul(layout->switch_target_entry_count,
                                  sizeof(*switch_targets),
                                  &switch_target_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM switch-target table exceeds host size");
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_zeros(
      &writer->writer, switch_target_byte_length));

  const uint64_t bytecode_start = writer->writer.total_written;
  const loom_vm_function_encoder_options_t function_options = {
      .module_layout = layout,
      .descriptor_registry = options->descriptor_registry,
      .allocation_budgets = options->allocation_budgets,
      .allocation_budget_count = options->allocation_budget_count,
      .diagnostic_emitter = options->diagnostic_emitter,
  };
  iree_arena_allocator_t function_arena;
  iree_arena_initialize(scratch_arena->block_pool, &function_arena);
  iree_status_t status = iree_ok_status();
  bool functions_complete = true;
  uint32_t switch_target_base = 0;
  for (iree_host_size_t i = 0; i < layout->function_count &&
                               iree_status_is_ok(status) && functions_complete;
       ++i) {
    const uint64_t relative_offset =
        writer->writer.total_written - bytecode_start;
    if (relative_offset > UINT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM function bytecode offsets exceed u32");
      break;
    }
    loom_vm_function_encoding_t encoding = {0};
    status = loom_vm_function_encode(layout->module, &layout->functions[i],
                                     &function_options, &function_arena,
                                     &writer->writer, &encoding);
    if (iree_status_is_ok(status) && encoding.is_complete) {
      if (switch_target_base > layout->switch_target_entry_count ||
          encoding.row.switch_target_entry_count_u32 >
              layout->switch_target_entry_count - switch_target_base) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "VM function switch-target count exceeds module layout");
      }
    }
    if (iree_status_is_ok(status) && encoding.is_complete) {
      encoding.row.bytecode_offset_u32 = (uint32_t)relative_offset;
      encoding.row.switch_target_base_u32 = switch_target_base;
      function_rows[i] = encoding.row;
      if (encoding.row.switch_target_entry_count_u32 != 0) {
        memcpy(switch_targets + switch_target_base, encoding.switch_targets,
               encoding.row.switch_target_entry_count_u32 *
                   sizeof(*switch_targets));
      }
      switch_target_base += encoding.row.switch_target_entry_count_u32;
    } else if (iree_status_is_ok(status)) {
      functions_complete = false;
    }
    iree_arena_reset(&function_arena);
  }
  iree_arena_deinitialize(&function_arena);
  IREE_RETURN_IF_ERROR(status);
  if (!functions_complete) return iree_ok_status();
  if (switch_target_base != layout->switch_target_entry_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "VM module switch-target count changed during emission");
  }
  loom_vm_module_writer_end_section(writer, section_start);
  *out_complete = true;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_globals(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_resource_layout_t* resources = &layout->resources;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_GLOBALS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_globals_header_t header = {
      .value_count_u32 = resources->value_global_count,
      .immutable_value_count_u32 = resources->immutable_value_global_count,
      .ref_count_u32 = resources->ref_global_count,
      .immutable_ref_count_u32 = resources->immutable_ref_global_count,
      .function_count_u32 = resources->function_global_count,
      .immutable_function_count_u32 =
          resources->immutable_function_global_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
      writer, resources->ref_global_descriptors,
      resources->ref_global_count *
          sizeof(*resources->ref_global_descriptors)));
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
      writer, resources->function_global_descriptors,
      resources->function_global_count *
          sizeof(*resources->function_global_descriptors)));
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_constants(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_resource_layout_t* resources = &layout->resources;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_CONSTANTS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
      writer, resources->constant_cells,
      resources->constant_count * sizeof(*resources->constant_cells)));
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_rodata(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_resource_layout_t* resources = &layout->resources;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section(
      writer, IREE_VM_BYTECODE_SECTION_RODATA,
      resources->rodata_section_alignment, &section_start));
  const iree_vm_bytecode_v0_rodata_header_t header = {
      .block_count_u32 = resources->rodata_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  for (uint32_t i = 0; i < resources->rodata_count; ++i) {
    const loom_vm_module_rodata_layout_t* rodata = &resources->rodata[i];
    const iree_vm_bytecode_v0_rodata_block_descriptor_t descriptor = {
        .byte_length_u64 = rodata->contents.data_length,
        .minimum_alignment_u32 = rodata->minimum_alignment,
    };
    IREE_RETURN_IF_ERROR(loom_vm_module_writer_write_record(
        writer, &descriptor, sizeof(descriptor)));
  }
  for (uint32_t i = 0; i < resources->rodata_count; ++i) {
    const loom_vm_module_rodata_layout_t* rodata = &resources->rodata[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_pad_to_alignment(
        &writer->writer, rodata->minimum_alignment));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
        &writer->writer, rodata->contents.data, rodata->contents.data_length));
  }
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_presentation(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_presentation_layout_t* presentation =
      &layout->presentation;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section_with_flags(
      writer, IREE_VM_BYTECODE_SECTION_PRESENTATION,
      IREE_VM_BYTECODE_SECTION_PRESENTATION_REQUIRED_FLAGS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_presentation_header_t header = {
      .entry_count_u32 = presentation->entry_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  for (uint32_t i = 0; i < presentation->entry_count; ++i) {
    const loom_vm_module_presentation_entry_layout_t* entry =
        &presentation->entries[i];
    const iree_vm_bytecode_v0_presentation_entry_row_t row = {
        .declaration_ordinal_u16 = entry->declaration_ordinal,
        .declaration_kind_u16 = entry->declaration_kind,
        .documentation_string_u16 = entry->documentation_string_ordinal,
        .authored_type_string_u16 = entry->authored_type_string_ordinal,
        .field_base_u32 = entry->field_base,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  for (uint32_t i = 0; i < presentation->field_count; ++i) {
    const loom_vm_module_presentation_field_layout_t* field =
        &presentation->fields[i];
    const iree_vm_bytecode_v0_presentation_field_row_t row = {
        .name_string_u16 = field->name_string_ordinal,
        .authored_type_string_u16 = field->authored_type_string_ordinal,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write_metadata(
    const loom_vm_module_layout_t* layout, loom_vm_module_writer_t* writer) {
  const loom_vm_module_metadata_layout_t* metadata = &layout->metadata;
  uint64_t section_start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_writer_begin_section_with_flags(
      writer, IREE_VM_BYTECODE_SECTION_METADATA,
      IREE_VM_BYTECODE_SECTION_METADATA_REQUIRED_FLAGS,
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT, &section_start));
  const iree_vm_bytecode_v0_metadata_header_t header = {
      .module_entry_count_u32 = metadata->module_entry_count,
      .import_scope_count_u32 = metadata->import_scope_count,
      .export_scope_count_u32 = metadata->export_scope_count,
      .total_entry_count_u32 = metadata->total_entry_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  for (uint32_t i = 0; i < metadata->import_scope_count; ++i) {
    const loom_vm_module_metadata_scope_layout_t* scope =
        &metadata->import_scopes[i];
    const iree_vm_bytecode_v0_metadata_scope_row_t row = {
        .declaration_ordinal_u16 = scope->declaration_ordinal,
        .entry_count_u16 = scope->entry_count,
        .entry_base_u32 = scope->entry_base,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  for (uint32_t i = 0; i < metadata->export_scope_count; ++i) {
    const loom_vm_module_metadata_scope_layout_t* scope =
        &metadata->export_scopes[i];
    const iree_vm_bytecode_v0_metadata_scope_row_t row = {
        .declaration_ordinal_u16 = scope->declaration_ordinal,
        .entry_count_u16 = scope->entry_count,
        .entry_base_u32 = scope->entry_base,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  for (uint32_t i = 0; i < metadata->total_entry_count; ++i) {
    const loom_vm_module_metadata_entry_layout_t* entry = &metadata->entries[i];
    const iree_vm_bytecode_v0_metadata_entry_row_t row = {
        .key_string_u16 = entry->key_string_ordinal,
        .value_type_u16 = entry->value_type,
    };
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &row, sizeof(row)));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_pad_to_alignment(
      &writer->writer,
      iree_alignof(iree_vm_bytecode_v0_metadata_value_offset_t)));
  for (uint32_t i = 0; i <= metadata->total_entry_count; ++i) {
    const iree_vm_bytecode_v0_metadata_value_offset_t offset =
        metadata->value_offsets[i];
    IREE_RETURN_IF_ERROR(
        loom_vm_module_writer_write_record(writer, &offset, sizeof(offset)));
  }
  for (uint32_t i = 0; i < metadata->total_entry_count; ++i) {
    const iree_const_byte_span_t value =
        loom_vm_module_metadata_entry_value(&metadata->entries[i]);
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
        &writer->writer, value.data, value.data_length));
  }
  loom_vm_module_writer_end_section(writer, section_start);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_patch_records(
    loom_vm_module_writer_t* writer,
    const iree_vm_bytecode_v0_function_row_t* function_rows,
    iree_host_size_t function_count,
    const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets,
    uint32_t switch_target_count, uint64_t image_length) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_flush(&writer->writer));

  IREE_RETURN_IF_ERROR(
      iree_io_stream_seek(writer->stream, IREE_IO_STREAM_SEEK_SET,
                          (iree_io_stream_pos_t)writer->function_rows_offset));
  loom_bytecode_page_writer_t patch_writer;
  loom_bytecode_page_writer_initialize(&patch_writer, writer->stream);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
      &patch_writer, function_rows, function_count * sizeof(*function_rows)));
  if (switch_target_count != 0) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
        &patch_writer, switch_targets,
        (iree_host_size_t)switch_target_count * sizeof(*switch_targets)));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_flush(&patch_writer));

  IREE_RETURN_IF_ERROR(iree_io_stream_seek(
      writer->stream, IREE_IO_STREAM_SEEK_SET,
      (iree_io_stream_pos_t)sizeof(iree_vm_bytecode_v0_image_header_t)));
  loom_bytecode_page_writer_initialize(&patch_writer, writer->stream);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
      &patch_writer, writer->sections,
      writer->section_count * sizeof(*writer->sections)));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_flush(&patch_writer));

  return iree_io_stream_seek(writer->stream, IREE_IO_STREAM_SEEK_SET,
                             (iree_io_stream_pos_t)image_length);
}

static iree_status_t loom_vm_module_write_image(
    const loom_vm_module_layout_t* layout,
    const loom_vm_module_emitter_options_t* options,
    iree_arena_allocator_t* scratch_arena, loom_vm_module_writer_t* writer,
    bool* out_complete) {
  *out_complete = false;
  uint16_t section_count = 3;
  if (layout->type_tables.string_count != 0) ++section_count;
  if (layout->type_tables.ref_type_entry_count != 0) ++section_count;
  if (layout->import_count != 0) ++section_count;
  if (layout->export_count != 0) ++section_count;
  if (layout->resources.constant_count != 0) ++section_count;
  if (loom_vm_module_has_globals(layout)) ++section_count;
  if (layout->resources.rodata_count != 0) ++section_count;
  if (layout->presentation.entry_count != 0) ++section_count;
  if (layout->metadata.total_entry_count != 0) ++section_count;
  iree_vm_bytecode_v0_image_header_t header = {0};
  memcpy(header.magic_u8, IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES,
         IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_LENGTH);
  header.core_major_u16 = IREE_VM_BYTECODE_CORE_MAJOR;
  header.core_required_minor_u16 = IREE_VM_BYTECODE_CORE_MINOR;
  header.section_count_u16 = section_count;
  IREE_RETURN_IF_ERROR(
      loom_vm_module_writer_write_record(writer, &header, sizeof(header)));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_zeros(
      &writer->writer, section_count * sizeof(*writer->sections)));

  if (layout->type_tables.string_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_strings(layout, writer));
  }
  if (layout->type_tables.ref_type_entry_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_ref_types(layout, writer));
  }
  IREE_RETURN_IF_ERROR(loom_vm_module_write_signatures(layout, writer));
  IREE_RETURN_IF_ERROR(loom_vm_module_write_callable_types(layout, writer));
  if (layout->import_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_imports(layout, writer));
  }
  if (layout->export_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_exports(layout, writer));
  }

  iree_vm_bytecode_v0_function_row_t* function_rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, layout->function_count, sizeof(*function_rows),
      (void**)&function_rows));
  iree_vm_bytecode_v0_switch_target_entry_t* switch_targets = NULL;
  if (layout->switch_target_entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, layout->switch_target_entry_count,
        sizeof(*switch_targets), (void**)&switch_targets));
  }
  bool functions_complete = false;
  IREE_RETURN_IF_ERROR(loom_vm_module_write_functions(
      layout, options, scratch_arena, writer, function_rows, switch_targets,
      &functions_complete));
  if (!functions_complete) return iree_ok_status();
  if (layout->resources.constant_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_constants(layout, writer));
  }
  if (loom_vm_module_has_globals(layout)) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_globals(layout, writer));
  }
  if (layout->resources.rodata_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_rodata(layout, writer));
  }
  if (layout->presentation.entry_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_presentation(layout, writer));
  }
  if (layout->metadata.total_entry_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_module_write_metadata(layout, writer));
  }
  IREE_ASSERT_EQ(writer->section_count, section_count);

  const uint64_t image_length = writer->writer.total_written;
  if (image_length > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM module image exceeds stream offset range");
  }
  IREE_RETURN_IF_ERROR(loom_vm_module_patch_records(
      writer, function_rows, layout->function_count, switch_targets,
      layout->switch_target_entry_count, image_length));
  *out_complete = true;
  return iree_ok_status();
}

iree_status_t loom_vm_emit_module(
    loom_module_t* module, const loom_vm_module_emitter_options_t* options,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t host_allocator,
    iree_byte_sequence_t** out_contents) {
  *out_contents = NULL;
  loom_vm_module_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(
      loom_vm_module_layout_build(module, scratch_arena, &layout));

  loom_vm_module_writer_t writer = {0};
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      LOOM_VM_MODULE_STREAM_BLOCK_SIZE, host_allocator, &writer.stream);
  if (iree_status_is_ok(status)) {
    loom_bytecode_page_writer_initialize(&writer.writer, writer.stream);
  }
  bool is_complete = false;
  if (iree_status_is_ok(status)) {
    status = loom_vm_module_write_image(&layout, options, scratch_arena,
                                        &writer, &is_complete);
  }
  if (iree_status_is_ok(status) && is_complete) {
    status = iree_io_vec_stream_move_contents(writer.stream, out_contents);
  }
  iree_io_stream_release(writer.stream);
  return status;
}
