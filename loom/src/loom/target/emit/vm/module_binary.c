// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_binary.h"

#include "iree/io/vec_stream.h"
#include "iree/vm/bytecode/wire/core.h"

static iree_status_t loom_vm_section_begin(
    iree_io_stream_t* stream, iree_io_stream_pos_t base_offset,
    iree_vm_bytecode_section_type_t type,
    iree_vm_bytecode_v0_section_directory_row_t* row,
    iree_io_stream_pos_t* out_start) {
  const uint8_t zero = 0;
  const iree_io_stream_pos_t padding =
      -(base_offset + iree_io_stream_offset(stream)) &
      (IREE_VM_BYTECODE_IMAGE_ALIGNMENT - 1);
  IREE_RETURN_IF_ERROR(iree_io_stream_fill(stream, padding, &zero, 1));
  *out_start = base_offset + iree_io_stream_offset(stream);
  *row = (iree_vm_bytecode_v0_section_directory_row_t){
      .section_type_u16 = type,
      .payload_alignment_u32 = IREE_VM_BYTECODE_IMAGE_ALIGNMENT,
  };
  return iree_ok_status();
}

// Patches reserved bytes without changing the append position.
static iree_status_t loom_vm_stream_patch(iree_io_stream_t* stream,
                                          iree_io_stream_pos_t offset,
                                          iree_const_byte_span_t contents) {
  const iree_io_stream_pos_t end = iree_io_stream_offset(stream);
  IREE_RETURN_IF_ERROR(
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, offset));
  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(stream, contents.data_length, contents.data));
  return iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, end);
}

static iree_status_t loom_vm_program_write_strings(
    const loom_vm_program_plan_t* plan, iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* row) {
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, 0, IREE_VM_BYTECODE_SECTION_STRINGS, row, &start));
  const iree_vm_bytecode_v0_strings_header_t header = {plan->string_count};
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  uint32_t offset = 0;
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(offset), &offset));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < plan->string_count && iree_status_is_ok(status);
       ++i) {
    offset += (uint32_t)plan->strings[i].size;
    status = iree_io_stream_write(stream, sizeof(offset), &offset);
  }
  for (uint32_t i = 0; i < plan->string_count && iree_status_is_ok(status);
       ++i) {
    status = iree_io_stream_write_string(stream, plan->strings[i]);
  }
  if (iree_status_is_ok(status)) {
    row->byte_length_u64 = iree_io_stream_offset(stream) - start;
  }
  return status;
}

static iree_status_t loom_vm_program_write_reference_types(
    const loom_vm_program_plan_t* plan, iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* row) {
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, 0, IREE_VM_BYTECODE_SECTION_REF_TYPES, row, &start));
  const iree_vm_bytecode_v0_ref_types_header_t header = {
      .group_count_u32 = plan->reference_group_count,
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, plan->reference_group_count * sizeof(plan->reference_groups[0]),
      plan->reference_groups));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, plan->reference_count * sizeof(plan->reference_entries[0]),
      plan->reference_entries));
  row->byte_length_u64 = iree_io_stream_offset(stream) - start;
  return iree_ok_status();
}

static iree_status_t loom_vm_program_write_signatures(
    const loom_vm_program_plan_t* plan, iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* rows) {
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, 0, IREE_VM_BYTECODE_SECTION_SIGNATURES, &rows[0], &start));
  const iree_vm_bytecode_v0_signatures_header_t signatures_header = {
      plan->signature_count};
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(signatures_header),
                                            &signatures_header));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, plan->signature_count * sizeof(plan->signatures[0]),
      plan->signatures));
  if (plan->signature_descriptor_count) {
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(stream,
                             plan->signature_descriptor_count *
                                 sizeof(plan->signature_descriptors[0]),
                             plan->signature_descriptors));
  }
  rows[0].byte_length_u64 = iree_io_stream_offset(stream) - start;

  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, 0, IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, &rows[1], &start));
  const iree_vm_bytecode_v0_callable_types_header_t callables_header = {
      plan->signature_count};
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(callables_header),
                                            &callables_header));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < plan->signature_count && iree_status_is_ok(status);
       ++i) {
    const iree_vm_bytecode_v0_callable_type_row_t callable = {
        .signature_ordinal_u16 = (uint16_t)i,
    };
    status = iree_io_stream_write(stream, sizeof(callable), &callable);
  }
  if (iree_status_is_ok(status)) {
    rows[1].byte_length_u64 = iree_io_stream_offset(stream) - start;
  }
  return status;
}

static iree_status_t loom_vm_program_write_imports(
    const loom_vm_program_plan_t* plan, iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* row) {
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, 0, IREE_VM_BYTECODE_SECTION_IMPORTS, row, &start));
  const iree_vm_bytecode_v0_imports_header_t header = {
      .group_count_u32 = plan->import_group_count,
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, plan->import_group_count * sizeof(plan->import_groups[0]),
      plan->import_groups));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, plan->import_count * sizeof(plan->imports[0]), plan->imports));
  row->byte_length_u64 = iree_io_stream_offset(stream) - start;
  return iree_ok_status();
}

static iree_status_t loom_vm_program_write_exports(
    const loom_vm_program_plan_t* plan, iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* row) {
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, 0, IREE_VM_BYTECODE_SECTION_EXPORTS, row, &start));
  const iree_vm_bytecode_v0_exports_header_t header = {plan->export_count};
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, plan->export_count * sizeof(plan->exports[0]), plan->exports));
  row->byte_length_u64 = iree_io_stream_offset(stream) - start;
  return iree_ok_status();
}

static iree_status_t loom_vm_program_write_function_prefix(
    const loom_vm_program_plan_t* plan, iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* row) {
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, 0, IREE_VM_BYTECODE_SECTION_FUNCTIONS, row, &start));
  const iree_vm_bytecode_v0_functions_header_t header = {
      .function_count_u32 = plan->function_count,
      .maximum_block_count_u32 = plan->maximum_block_count,
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, plan->function_count * sizeof(plan->functions[0]),
      plan->functions));
  row->byte_length_u64 = iree_io_stream_offset(stream) - start +
                         iree_byte_sequence_length(plan->function_bytecode);
  return iree_ok_status();
}

static iree_status_t loom_vm_program_write_rodata(
    const loom_vm_program_plan_t* plan, iree_io_stream_pos_t base_offset,
    iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* row) {
  const uint8_t zero = 0;
  IREE_RETURN_IF_ERROR(
      iree_io_stream_fill(stream,
                          -(base_offset + iree_io_stream_offset(stream)) &
                              (plan->rodata_alignment - 1),
                          &zero, 1));
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, base_offset, IREE_VM_BYTECODE_SECTION_RODATA, row, &start));
  row->payload_alignment_u32 = plan->rodata_alignment;
  const iree_vm_bytecode_v0_rodata_header_t header = {
      .block_count_u32 = plan->rodata_count,
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < plan->rodata_count && iree_status_is_ok(status);
       ++i) {
    const iree_vm_bytecode_v0_rodata_block_descriptor_t descriptor = {
        .byte_length_u64 = plan->rodata[i].contents.data_length,
        .minimum_alignment_u32 = plan->rodata[i].alignment,
    };
    status = iree_io_stream_write(stream, sizeof(descriptor), &descriptor);
  }
  for (uint32_t i = 0; i < plan->rodata_count && iree_status_is_ok(status);
       ++i) {
    const loom_vm_rodata_plan_t* block = &plan->rodata[i];
    status = iree_io_stream_fill(
        stream,
        -(base_offset + iree_io_stream_offset(stream) - start) &
            (block->alignment - 1),
        &zero, 1);
    if (iree_status_is_ok(status)) {
      status = iree_io_stream_write(stream, block->contents.data_length,
                                    block->contents.data);
    }
  }
  if (iree_status_is_ok(status)) {
    row->byte_length_u64 = base_offset + iree_io_stream_offset(stream) - start;
  }
  return status;
}

static iree_status_t loom_vm_program_write_native_calls(
    const loom_vm_program_plan_t* plan, iree_io_stream_pos_t base_offset,
    iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* row) {
  iree_io_stream_pos_t start = 0;
  IREE_RETURN_IF_ERROR(loom_vm_section_begin(
      stream, base_offset, IREE_VM_BYTECODE_SECTION_NATIVE_CALLS, row, &start));
  const iree_vm_bytecode_v0_native_calls_header_t header = {
      .layout_count_u32 = plan->native_layout_count,
      .field_count_u32 = plan->native_field_count,
      .import_count_u32 = plan->native_import_count,
      .callsite_count_u32 = plan->native_call_count,
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  iree_status_t status = iree_ok_status();
  for (const loom_vm_native_layout_plan_t* layout = plan->native_layouts;
       layout && iree_status_is_ok(status); layout = layout->next) {
    status = iree_io_stream_write(stream, sizeof(layout->row), &layout->row);
  }
  for (const loom_vm_native_layout_plan_t* layout = plan->native_layouts;
       layout && iree_status_is_ok(status); layout = layout->next) {
    status = iree_io_stream_write(
        stream, layout->row.field_count_u16 * sizeof(*layout->fields),
        layout->fields);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_write(
        stream, plan->native_import_count * sizeof(*plan->native_imports),
        plan->native_imports);
  }
  for (const loom_vm_native_callsite_plan_t* callsite = plan->native_calls;
       callsite && iree_status_is_ok(status); callsite = callsite->next) {
    status =
        iree_io_stream_write(stream, sizeof(callsite->row), &callsite->row);
  }
  for (const loom_vm_native_callsite_plan_t* callsite = plan->native_calls;
       callsite && iree_status_is_ok(status); callsite = callsite->next) {
    status = iree_io_stream_write(
        stream, callsite->row.word_count_u32 * sizeof(uint32_t),
        callsite->words);
  }
  if (iree_status_is_ok(status)) {
    row->byte_length_u64 = base_offset + iree_io_stream_offset(stream) - start;
  }
  return status;
}

static iree_status_t loom_vm_program_write_prefix(
    const loom_vm_program_plan_t* plan, uint16_t section_count,
    iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* directory,
    uint16_t* out_section_count) {
  const iree_vm_bytecode_v0_image_header_t header = {
      .magic_u8 = {'I', 'R', 'E', 'E', 'V', 'M', 0, 0},
      .core_major_u16 = IREE_VM_BYTECODE_CORE_MAJOR,
      .core_required_minor_u16 = IREE_VM_BYTECODE_CORE_MINOR,
      .section_count_u16 = section_count,
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, section_count * sizeof(directory[0]), directory));

  uint16_t section = 0;
  if (plan->string_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_vm_program_write_strings(plan, stream, &directory[section++]));
  }
  if (plan->reference_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_program_write_reference_types(
        plan, stream, &directory[section++]));
  }
  if (plan->signature_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_vm_program_write_signatures(plan, stream, &directory[section]));
    section += 2;
  }
  if (plan->import_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_vm_program_write_imports(plan, stream, &directory[section++]));
  }
  if (plan->export_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_vm_program_write_exports(plan, stream, &directory[section++]));
  }
  if (plan->function_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_program_write_function_prefix(
        plan, stream, &directory[section++]));
  }
  *out_section_count = section;
  return iree_ok_status();
}

iree_status_t loom_vm_program_emit_binary(const loom_vm_program_plan_t* plan,
                                          iree_allocator_t allocator,
                                          iree_byte_sequence_t** out_binary) {
  *out_binary = NULL;
  const uint16_t section_count =
      (plan->string_count != 0) + (plan->reference_count != 0) +
      2 * (plan->signature_count != 0) + (plan->import_count != 0) +
      (plan->export_count != 0) + (plan->function_count != 0) +
      plan->has_rodata_section + (plan->native_call_count != 0);
  iree_io_stream_t* prefix_stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE, 32 * 1024,
      allocator, &prefix_stream));

  iree_vm_bytecode_v0_section_directory_row_t directory[9] = {0};
  uint16_t section = 0;
  iree_status_t status = loom_vm_program_write_prefix(
      plan, section_count, prefix_stream, directory, &section);

  iree_io_stream_t* suffix_stream = NULL;
  if (iree_status_is_ok(status) &&
      (plan->has_rodata_section || plan->native_call_count)) {
    iree_io_stream_pos_t suffix_base = 0;
    iree_io_stream_t* stream = prefix_stream;
    if (plan->function_count != 0) {
      suffix_base = iree_io_stream_offset(prefix_stream) +
                    iree_byte_sequence_length(plan->function_bytecode);
      status = iree_io_vec_stream_create(
          IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE,
          32 * 1024, allocator, &suffix_stream);
      stream = suffix_stream;
    }
    if (iree_status_is_ok(status) && plan->has_rodata_section) {
      status = loom_vm_program_write_rodata(plan, suffix_base, stream,
                                            &directory[section++]);
    }
    if (iree_status_is_ok(status) && plan->native_call_count) {
      status = loom_vm_program_write_native_calls(plan, suffix_base, stream,
                                                  &directory[section++]);
    }
  }
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(section, section_count);
    status = loom_vm_stream_patch(
        prefix_stream, sizeof(iree_vm_bytecode_v0_image_header_t),
        iree_make_const_byte_span(directory,
                                  section_count * sizeof(directory[0])));
  }

  iree_byte_sequence_t* prefix = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_move_contents(prefix_stream, &prefix);
  }
  iree_byte_sequence_t* suffix = NULL;
  if (iree_status_is_ok(status) && suffix_stream) {
    status = iree_io_vec_stream_move_contents(suffix_stream, &suffix);
  }
  if (iree_status_is_ok(status)) {
    if (plan->function_count != 0) {
      // Function bytes already have their final representation and remain
      // independently owned. Retain them between the writer-owned ranges so
      // artifact construction never flattens or duplicates the instruction
      // stream.
      iree_byte_sequence_t* sequences[3] = {
          prefix,
          plan->function_bytecode,
          suffix,
      };
      const iree_host_size_t sequence_count = suffix ? 3 : 2;
      status = iree_byte_sequence_create_concat(sequence_count, sequences,
                                                allocator, out_binary);
    } else {
      *out_binary = prefix;
      prefix = NULL;
    }
  }
  iree_byte_sequence_release(suffix);
  iree_byte_sequence_release(prefix);
  iree_io_stream_release(suffix_stream);
  iree_io_stream_release(prefix_stream);
  return status;
}
