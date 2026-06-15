// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hsaco.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/descriptor.h"
#include "loom/target/emit/native/elf.h"

//===----------------------------------------------------------------------===//
// AMDGPU code-object layout constants
//===----------------------------------------------------------------------===//

enum {
  LOOM_AMDGPU_HSACO_SECTION_NOTE = 0,
  LOOM_AMDGPU_HSACO_SECTION_DYNSYM = 1,
  LOOM_AMDGPU_HSACO_SECTION_HASH = 2,
  LOOM_AMDGPU_HSACO_SECTION_DYNSTR = 3,
  LOOM_AMDGPU_HSACO_SECTION_RODATA = 4,
  LOOM_AMDGPU_HSACO_SECTION_TEXT = 5,
  LOOM_AMDGPU_HSACO_SECTION_DYNAMIC = 6,
  LOOM_AMDGPU_HSACO_SECTION_SYMTAB = 7,
  LOOM_AMDGPU_HSACO_SECTION_STRTAB = 8,
  LOOM_AMDGPU_HSACO_SECTION_COUNT = 9,
};

enum {
  LOOM_AMDGPU_HSACO_DYN_NULL = 0,
  LOOM_AMDGPU_HSACO_DYN_HASH = 4,
  LOOM_AMDGPU_HSACO_DYN_STRTAB = 5,
  LOOM_AMDGPU_HSACO_DYN_SYMTAB = 6,
  LOOM_AMDGPU_HSACO_DYN_STRSZ = 10,
  LOOM_AMDGPU_HSACO_DYN_SYMENT = 11,
};

enum {
  LOOM_AMDGPU_HSACO_SYMBOL_BIND_LOCAL = 0,
  LOOM_AMDGPU_HSACO_SYMBOL_BIND_GLOBAL = 1,
};

enum {
  LOOM_AMDGPU_HSACO_SYMBOL_TYPE_NONE = 0,
  LOOM_AMDGPU_HSACO_SYMBOL_TYPE_OBJECT = 1,
  LOOM_AMDGPU_HSACO_SYMBOL_TYPE_FUNCTION = 2,
};

enum {
  LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_DEFAULT = 0,
  LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_PROTECTED = 3,
};

#define LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE 16u
#define LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE 24u
#define LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE UINT64_C(64)
#define LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE UINT64_C(56)
#define LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT UINT64_C(7)
#define LOOM_AMDGPU_HSACO_READ_SEGMENT_BASE \
  (LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE +      \
   LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT * \
       LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE)
#define LOOM_AMDGPU_HSACO_TEXT_SEGMENT_BASE UINT64_C(0x1000)
#define LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT UINT64_C(0x1000)
#define LOOM_AMDGPU_HSACO_NOTE_ALIGNMENT UINT64_C(4)
#define LOOM_AMDGPU_HSACO_TEXT_ALIGNMENT UINT64_C(256)
#define LOOM_AMDGPU_HSACO_DYNAMIC_ENTRY_COUNT 6u

typedef struct loom_amdgpu_hsaco_kernel_layout_t {
  // Byte offset of the kernel entry within the `.text` section.
  uint64_t text_offset;
  // Byte offset of the kernel descriptor within the `.rodata` section.
  uint64_t descriptor_offset;
  // Name offset of the kernel entry symbol in `.dynstr` and `.strtab`.
  uint32_t entry_name_offset;
  // Name offset of the descriptor symbol in `.dynstr` and `.strtab`.
  uint32_t descriptor_name_offset;
} loom_amdgpu_hsaco_kernel_layout_t;

typedef struct loom_amdgpu_hsaco_payloads_t {
  // Arena-backed copied metadata kernel rows.
  loom_amdgpu_metadata_kernel_t* metadata_kernels;
  // Number of entries in |metadata_kernels|.
  iree_host_size_t metadata_kernel_count;
  // Arena-backed per-kernel placement rows.
  loom_amdgpu_hsaco_kernel_layout_t* kernel_layouts;
  // ELF section descriptors passed to the generic writer.
  loom_native_elf64le_section_t sections[LOOM_AMDGPU_HSACO_SECTION_COUNT];
  // ELF program headers passed to the generic writer.
  loom_native_elf64le_segment_t segments[7];
  // Byte length of the header-backed read segment.
  uint64_t read_segment_file_size;
} loom_amdgpu_hsaco_payloads_t;

static uint16_t loom_amdgpu_hsaco_section_index(iree_host_size_t index) {
  return (uint16_t)(index + 1u);
}

static bool loom_amdgpu_hsaco_checked_add_uint64(uint64_t lhs, uint64_t rhs,
                                                 uint64_t* out_result) {
  *out_result = lhs + rhs;
  return *out_result >= lhs;
}

static bool loom_amdgpu_hsaco_checked_align_uint64(uint64_t value,
                                                   uint64_t alignment,
                                                   uint64_t* out_result) {
  if (!iree_is_power_of_two_uint64(alignment)) {
    return false;
  }
  if (!loom_amdgpu_hsaco_checked_add_uint64(value, alignment - 1u,
                                            out_result)) {
    return false;
  }
  *out_result &= ~(alignment - 1u);
  return true;
}

static iree_status_t loom_amdgpu_hsaco_align_uint64(uint64_t value,
                                                    uint64_t alignment,
                                                    uint64_t* out_result) {
  if (loom_amdgpu_hsaco_checked_align_uint64(value, alignment, out_result)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AMDGPU HSACO layout alignment overflow");
}

static iree_status_t loom_amdgpu_hsaco_cast_host_size(
    uint64_t value, iree_host_size_t* out_value) {
  if (value > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO payload exceeds host size capacity");
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_allocate_zeroed(
    iree_arena_allocator_t* arena, iree_host_size_t length,
    iree_byte_span_t* out_span) {
  *out_span = iree_make_byte_span(NULL, 0);
  if (length == 0) {
    return iree_ok_status();
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, length, (void**)&data));
  memset(data, 0, length);
  *out_span = iree_make_byte_span(data, length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_append_u32(
    iree_byte_span_t target, iree_host_size_t* inout_offset, uint32_t value) {
  if (*inout_offset > target.data_length ||
      target.data_length - *inout_offset < 4u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO u32 payload write overflow");
  }
  uint8_t* data = target.data + *inout_offset;
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
  *inout_offset += 4u;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_store_u32(iree_byte_span_t target,
                                                 iree_host_size_t byte_offset,
                                                 uint32_t value) {
  if (byte_offset > target.data_length ||
      target.data_length - byte_offset < 4u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO u32 payload store overflow");
  }
  uint8_t* data = target.data + byte_offset;
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_append_u64(
    iree_byte_span_t target, iree_host_size_t* inout_offset, uint64_t value) {
  if (*inout_offset > target.data_length ||
      target.data_length - *inout_offset < 8u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO u64 payload write overflow");
  }
  uint8_t* data = target.data + *inout_offset;
  for (iree_host_size_t i = 0; i < 8u; ++i) {
    data[i] = (uint8_t)(value >> (i * 8u));
  }
  *inout_offset += 8u;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_write_symbol(
    iree_byte_span_t target, iree_host_size_t symbol_index,
    uint32_t name_offset, uint8_t binding, uint8_t type, uint8_t visibility,
    uint16_t section_index, uint64_t value, uint64_t size) {
  const iree_host_size_t byte_offset =
      symbol_index * LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE;
  if (symbol_index > target.data_length / LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE ||
      target.data_length - byte_offset < LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol table write overflow");
  }
  uint8_t* entry = target.data + byte_offset;
  entry[0] = (uint8_t)name_offset;
  entry[1] = (uint8_t)(name_offset >> 8);
  entry[2] = (uint8_t)(name_offset >> 16);
  entry[3] = (uint8_t)(name_offset >> 24);
  entry[4] = (uint8_t)((binding << 4u) | type);
  entry[5] = visibility;
  entry[6] = (uint8_t)section_index;
  entry[7] = (uint8_t)(section_index >> 8);
  for (iree_host_size_t i = 0; i < 8u; ++i) {
    entry[8 + i] = (uint8_t)(value >> (i * 8u));
    entry[16 + i] = (uint8_t)(size >> (i * 8u));
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_hsaco_elf_hash(iree_string_view_t name) {
  uint32_t hash = 0;
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    hash = (hash << 4u) + (uint8_t)name.data[i];
    const uint32_t high = hash & UINT32_C(0xf0000000);
    if (high != 0) {
      hash ^= high >> 24u;
      hash &= ~high;
    }
  }
  return hash;
}

static bool loom_amdgpu_hsaco_symbol_start_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '$';
}

static bool loom_amdgpu_hsaco_symbol_continue_char(char c) {
  return loom_amdgpu_hsaco_symbol_start_char(c) || (c >= '0' && c <= '9') ||
         c == '.';
}

static iree_status_t loom_amdgpu_hsaco_validate_symbol(
    iree_string_view_t symbol, iree_string_view_t field_name) {
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO symbol field '%.*s' is required",
                            (int)field_name.size, field_name.data);
  }
  if (!loom_amdgpu_hsaco_symbol_start_char(symbol.data[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HSACO symbol field '%.*s' has invalid first character '%c'",
        (int)field_name.size, field_name.data, symbol.data[0]);
  }
  for (iree_host_size_t i = 1; i < symbol.size; ++i) {
    if (!loom_amdgpu_hsaco_symbol_continue_char(symbol.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO symbol field '%.*s' contains invalid character '%c'",
          (int)field_name.size, field_name.data, symbol.data[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_validate_target_id(
    iree_string_view_t target, iree_string_view_t processor_name) {
  if (iree_string_view_is_empty(processor_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO processor is required");
  }
  loom_amdgpu_amdhsa_target_id_t target_id = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_parse_amdhsa_target_id(target, &target_id));
  if (!iree_string_view_is_empty(target_id.feature_suffix)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU HSACO target-feature suffixes are not supported yet: '%.*s'",
        (int)target_id.feature_suffix.size, target_id.feature_suffix.data);
  }
  if (!iree_string_view_equal(target_id.processor->name, processor_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HSACO target id '%.*s' selects processor '%.*s' but file "
        "processor is '%.*s'",
        (int)target.size, target.data, (int)target_id.processor->name.size,
        target_id.processor->name.data, (int)processor_name.size,
        processor_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_validate_file(
    const loom_amdgpu_hsaco_file_t* file) {
  if (file == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO file description is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_validate_target_id(file->target, file->processor));
  if (file->kernel_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO requires at least one kernel");
  }
  if (file->kernels == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO kernel array is required");
  }
  if (file->kernel_count > (UINT32_MAX - 1u) / 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO dynamic symbol count exceeds "
                            "32-bit ELF capacity");
  }
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_validate_symbol(
        kernel->metadata.name, IREE_SV("kernel.name")));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hsaco_validate_symbol(kernel->metadata.descriptor_symbol,
                                          IREE_SV("kernel.descriptor_symbol")));
    if (kernel->text.data == NULL || kernel->text.data_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU HSACO kernel '%.*s' has no text bytes",
                              (int)kernel->metadata.name.size,
                              kernel->metadata.name.data);
    }
    if (iree_string_view_equal(kernel->metadata.name,
                               kernel->metadata.descriptor_symbol)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSACO kernel '%.*s' reuses its entry symbol as descriptor "
          "symbol",
          (int)kernel->metadata.name.size, kernel->metadata.name.data);
    }
    for (iree_host_size_t j = i + 1u; j < file->kernel_count; ++j) {
      const loom_amdgpu_hsaco_kernel_t* other_kernel = &file->kernels[j];
      if (iree_string_view_equal(kernel->metadata.name,
                                 other_kernel->metadata.name) ||
          iree_string_view_equal(kernel->metadata.name,
                                 other_kernel->metadata.descriptor_symbol) ||
          iree_string_view_equal(kernel->metadata.descriptor_symbol,
                                 other_kernel->metadata.name) ||
          iree_string_view_equal(kernel->metadata.descriptor_symbol,
                                 other_kernel->metadata.descriptor_symbol)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU HSACO symbol '%.*s' conflicts with another kernel symbol",
            (int)kernel->metadata.name.size, kernel->metadata.name.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_copy_metadata_kernels(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, iree_arena_allocator_t* arena) {
  payloads->metadata_kernel_count = file->kernel_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, file->kernel_count, sizeof(payloads->metadata_kernels[0]),
      (void**)&payloads->metadata_kernels));
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    payloads->metadata_kernels[i] = file->kernels[i].metadata;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_note(
    const loom_amdgpu_hsaco_file_t* file,
    const loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_note, iree_arena_allocator_t* arena) {
  *out_note = iree_make_const_byte_span(NULL, 0);

  const loom_amdgpu_code_object_metadata_t metadata = {
      .target = file->target,
      .kernels = payloads->metadata_kernels,
      .kernel_count = payloads->metadata_kernel_count,
  };

  iree_string_builder_t measure_builder;
  iree_string_builder_initialize(iree_allocator_null(), &measure_builder);
  iree_status_t status =
      loom_amdgpu_metadata_append_elf_note(&metadata, &measure_builder);
  const iree_host_size_t note_size =
      iree_status_is_ok(status) ? iree_string_builder_size(&measure_builder)
                                : 0;
  iree_string_builder_deinitialize(&measure_builder);
  IREE_RETURN_IF_ERROR(status);

  uint8_t* note_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, note_size + 1u, (void**)&note_data));
  iree_string_builder_t note_builder;
  iree_string_builder_initialize_with_storage((char*)note_data, note_size + 1u,
                                              &note_builder);
  status = loom_amdgpu_metadata_append_elf_note(&metadata, &note_builder);
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&note_builder) != note_size) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HSACO metadata note size changed between measure and emit");
  }
  iree_string_builder_deinitialize(&note_builder);
  if (iree_status_is_ok(status)) {
    *out_note = iree_make_const_byte_span(note_data, note_size);
  }
  return status;
}

static iree_status_t loom_amdgpu_hsaco_build_string_tables(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_string_table, iree_arena_allocator_t* arena) {
  *out_string_table = iree_make_const_byte_span(NULL, 0);

  iree_host_size_t string_table_size = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    iree_host_size_t entry_name_size = 0;
    iree_host_size_t descriptor_name_size = 0;
    if (!iree_host_size_checked_add(kernel->metadata.name.size, 1u,
                                    &entry_name_size) ||
        !iree_host_size_checked_add(kernel->metadata.descriptor_symbol.size, 1u,
                                    &descriptor_name_size) ||
        !iree_host_size_checked_add(string_table_size, entry_name_size,
                                    &string_table_size) ||
        !iree_host_size_checked_add(string_table_size, descriptor_name_size,
                                    &string_table_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO string table size overflow");
    }
  }
  if (string_table_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO string table exceeds ELF32 name "
                            "offset capacity");
  }

  iree_byte_span_t string_table = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_allocate_zeroed(
      arena, string_table_size, &string_table));
  iree_host_size_t string_table_offset = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    loom_amdgpu_hsaco_kernel_layout_t* layout = &payloads->kernel_layouts[i];
    layout->entry_name_offset = (uint32_t)string_table_offset;
    memcpy(string_table.data + string_table_offset, kernel->metadata.name.data,
           kernel->metadata.name.size);
    string_table_offset += kernel->metadata.name.size + 1u;
    layout->descriptor_name_offset = (uint32_t)string_table_offset;
    memcpy(string_table.data + string_table_offset,
           kernel->metadata.descriptor_symbol.data,
           kernel->metadata.descriptor_symbol.size);
    string_table_offset += kernel->metadata.descriptor_symbol.size + 1u;
  }
  *out_string_table =
      iree_make_const_byte_span(string_table.data, string_table.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_text(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, iree_const_byte_span_t* out_text,
    iree_arena_allocator_t* arena) {
  *out_text = iree_make_const_byte_span(NULL, 0);

  uint64_t text_size_u64 = 0;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        text_size_u64, LOOM_AMDGPU_HSACO_TEXT_ALIGNMENT, &text_size_u64));
    payloads->kernel_layouts[i].text_offset = text_size_u64;
    if (!loom_amdgpu_hsaco_checked_add_uint64(
            text_size_u64, file->kernels[i].text.data_length, &text_size_u64)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO text section size overflow");
    }
  }

  iree_host_size_t text_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_cast_host_size(text_size_u64, &text_size));
  iree_byte_span_t text = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_allocate_zeroed(arena, text_size, &text));
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    iree_host_size_t text_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_cast_host_size(
        payloads->kernel_layouts[i].text_offset, &text_offset));
    memcpy(text.data + text_offset, kernel->text.data,
           kernel->text.data_length);
  }
  *out_text = iree_make_const_byte_span(text.data, text.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_rodata(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, uint64_t rodata_address,
    uint64_t text_address, iree_host_size_t rodata_size,
    iree_const_byte_span_t* out_rodata, iree_arena_allocator_t* arena) {
  *out_rodata = iree_make_const_byte_span(NULL, 0);

  iree_byte_span_t rodata = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_allocate_zeroed(arena, rodata_size, &rodata));
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_layout_t* layout =
        &payloads->kernel_layouts[i];
    uint64_t entry_address = 0;
    uint64_t descriptor_address = 0;
    if (!loom_amdgpu_hsaco_checked_add_uint64(text_address, layout->text_offset,
                                              &entry_address) ||
        !loom_amdgpu_hsaco_checked_add_uint64(
            rodata_address, layout->descriptor_offset, &descriptor_address)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU HSACO descriptor-to-entry address overflow");
    }
    if (entry_address > INT64_MAX || descriptor_address > INT64_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO descriptor address exceeds signed "
                              "offset capacity");
    }
    const int64_t entry_byte_offset =
        (int64_t)entry_address - (int64_t)descriptor_address;
    loom_amdgpu_kernel_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        file->processor, &payloads->metadata_kernels[i], entry_byte_offset,
        &descriptor));
    descriptor.flags |= file->kernels[i].descriptor_options.flags;
    descriptor.user_sgpr_count =
        iree_max(descriptor.user_sgpr_count,
                 file->kernels[i].descriptor_options.user_sgpr_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_validate_metadata(
        &descriptor, &payloads->metadata_kernels[i]));

    iree_host_size_t descriptor_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_cast_host_size(
        layout->descriptor_offset, &descriptor_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_write(
        &descriptor,
        iree_make_byte_span(rodata.data + descriptor_offset,
                            rodata.data_length - descriptor_offset)));
  }
  *out_rodata = iree_make_const_byte_span(rodata.data, rodata.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_plan_rodata(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads,
    iree_const_byte_span_t* out_placeholder) {
  *out_placeholder = iree_make_const_byte_span(NULL, 0);

  uint64_t rodata_size_u64 = 0;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        rodata_size_u64, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ALIGNMENT,
        &rodata_size_u64));
    payloads->kernel_layouts[i].descriptor_offset = rodata_size_u64;
    if (!loom_amdgpu_hsaco_checked_add_uint64(
            rodata_size_u64, LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH,
            &rodata_size_u64)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO rodata section size overflow");
    }
  }

  iree_host_size_t rodata_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_cast_host_size(rodata_size_u64, &rodata_size));
  *out_placeholder = iree_make_const_byte_span(NULL, rodata_size);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_symbol_table(
    const loom_amdgpu_hsaco_file_t* file,
    const loom_amdgpu_hsaco_payloads_t* payloads, uint64_t rodata_address,
    uint64_t text_address, iree_const_byte_span_t* out_symbol_table,
    iree_arena_allocator_t* arena) {
  *out_symbol_table = iree_make_const_byte_span(NULL, 0);

  iree_host_size_t symbol_count = 0;
  if (!iree_host_size_checked_mul(file->kernel_count, 2u, &symbol_count) ||
      !iree_host_size_checked_add(symbol_count, 1u, &symbol_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol count overflow");
  }
  iree_host_size_t symbol_table_size = 0;
  if (!iree_host_size_checked_mul(symbol_count,
                                  LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
                                  &symbol_table_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol table size overflow");
  }
  iree_byte_span_t symbol_table = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_allocate_zeroed(
      arena, symbol_table_size, &symbol_table));

  iree_host_size_t symbol_index = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const loom_amdgpu_hsaco_kernel_t* kernel = &file->kernels[i];
    const loom_amdgpu_hsaco_kernel_layout_t* layout =
        &payloads->kernel_layouts[i];
    uint64_t entry_address = 0;
    uint64_t descriptor_address = 0;
    if (!loom_amdgpu_hsaco_checked_add_uint64(text_address, layout->text_offset,
                                              &entry_address) ||
        !loom_amdgpu_hsaco_checked_add_uint64(
            rodata_address, layout->descriptor_offset, &descriptor_address)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO symbol address overflow");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_write_symbol(
        symbol_table, symbol_index++, layout->entry_name_offset,
        LOOM_AMDGPU_HSACO_SYMBOL_BIND_GLOBAL,
        LOOM_AMDGPU_HSACO_SYMBOL_TYPE_FUNCTION,
        LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_PROTECTED,
        loom_amdgpu_hsaco_section_index(LOOM_AMDGPU_HSACO_SECTION_TEXT),
        entry_address, kernel->text.data_length));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_write_symbol(
        symbol_table, symbol_index++, layout->descriptor_name_offset,
        LOOM_AMDGPU_HSACO_SYMBOL_BIND_GLOBAL,
        LOOM_AMDGPU_HSACO_SYMBOL_TYPE_OBJECT,
        LOOM_AMDGPU_HSACO_SYMBOL_VISIBILITY_PROTECTED,
        loom_amdgpu_hsaco_section_index(LOOM_AMDGPU_HSACO_SECTION_RODATA),
        descriptor_address, LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH));
  }
  *out_symbol_table =
      iree_make_const_byte_span(symbol_table.data, symbol_table.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_sysv_hash(
    const loom_amdgpu_hsaco_file_t* file, iree_host_size_t symbol_count,
    iree_const_byte_span_t* out_hash, iree_arena_allocator_t* arena) {
  *out_hash = iree_make_const_byte_span(NULL, 0);

  const iree_host_size_t bucket_count = symbol_count;
  iree_host_size_t hash_word_count = 0;
  if (!iree_host_size_checked_add(2u, bucket_count, &hash_word_count) ||
      !iree_host_size_checked_add(hash_word_count, symbol_count,
                                  &hash_word_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO hash table word count overflow");
  }
  iree_host_size_t hash_size = 0;
  if (!iree_host_size_checked_mul(hash_word_count, sizeof(uint32_t),
                                  &hash_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO hash table size overflow");
  }
  iree_byte_span_t hash = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_allocate_zeroed(arena, hash_size, &hash));

  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bucket_count, sizeof(buckets[0]), (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(buckets[0]));
  uint32_t* chains = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_count, sizeof(chains[0]), (void**)&chains));
  memset(chains, 0, symbol_count * sizeof(chains[0]));
  iree_host_size_t write_offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u32(hash, &write_offset,
                                                    (uint32_t)bucket_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u32(hash, &write_offset,
                                                    (uint32_t)symbol_count));
  iree_host_size_t symbol_index = 1u;
  for (iree_host_size_t i = 0; i < file->kernel_count; ++i) {
    const uint32_t entry_bucket =
        loom_amdgpu_hsaco_elf_hash(file->kernels[i].metadata.name) %
        (uint32_t)bucket_count;
    chains[symbol_index] = buckets[entry_bucket];
    buckets[entry_bucket] = (uint32_t)symbol_index++;

    const uint32_t descriptor_bucket =
        loom_amdgpu_hsaco_elf_hash(
            file->kernels[i].metadata.descriptor_symbol) %
        (uint32_t)bucket_count;
    chains[symbol_index] = buckets[descriptor_bucket];
    buckets[descriptor_bucket] = (uint32_t)symbol_index++;
  }
  for (iree_host_size_t i = 0; i < bucket_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_store_u32(
        hash, (2u + i) * sizeof(uint32_t), buckets[i]));
  }
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_store_u32(
        hash, (2u + bucket_count + i) * sizeof(uint32_t), chains[i]));
  }
  *out_hash = iree_make_const_byte_span(hash.data, hash.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_build_dynamic_table(
    const loom_native_elf64le_section_t* sections,
    iree_const_byte_span_t* out_dynamic_table, iree_arena_allocator_t* arena) {
  *out_dynamic_table = iree_make_const_byte_span(NULL, 0);

  iree_byte_span_t dynamic_table = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_allocate_zeroed(
      arena,
      LOOM_AMDGPU_HSACO_DYNAMIC_ENTRY_COUNT * LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE,
      &dynamic_table));
  iree_host_size_t offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_HASH));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset,
      sections[LOOM_AMDGPU_HSACO_SECTION_HASH].address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_STRTAB));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset,
      sections[LOOM_AMDGPU_HSACO_SECTION_DYNSTR].address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_SYMTAB));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset,
      sections[LOOM_AMDGPU_HSACO_SECTION_DYNSYM].address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_STRSZ));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset,
      sections[LOOM_AMDGPU_HSACO_SECTION_DYNSTR].contents.data_length));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_SYMENT));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(
      dynamic_table, &offset, LOOM_AMDGPU_HSACO_DYN_NULL));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_append_u64(dynamic_table, &offset, 0));
  *out_dynamic_table =
      iree_make_const_byte_span(dynamic_table.data, dynamic_table.data_length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_assign_read_addresses(
    loom_native_elf64le_section_t* sections, uint64_t* out_read_end) {
  uint64_t address = LOOM_AMDGPU_HSACO_READ_SEGMENT_BASE;
  for (iree_host_size_t i = LOOM_AMDGPU_HSACO_SECTION_NOTE;
       i <= LOOM_AMDGPU_HSACO_SECTION_RODATA; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
        address, sections[i].alignment, &address));
    sections[i].address = address;
    if (!loom_amdgpu_hsaco_checked_add_uint64(
            address, sections[i].contents.data_length, &address)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU HSACO read segment address overflow");
    }
  }
  *out_read_end = address;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hsaco_prepare_sections(
    const loom_amdgpu_hsaco_file_t* file,
    loom_amdgpu_hsaco_payloads_t* payloads, iree_arena_allocator_t* arena) {
  iree_const_byte_span_t note = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_note(file, payloads, &note, arena));

  iree_host_size_t symbol_count = 0;
  if (!iree_host_size_checked_mul(file->kernel_count, 2u, &symbol_count) ||
      !iree_host_size_checked_add(symbol_count, 1u, &symbol_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol count overflow");
  }

  iree_const_byte_span_t string_table = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_build_string_tables(
      file, payloads, &string_table, arena));

  iree_const_byte_span_t text = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_text(file, payloads, &text, arena));

  iree_const_byte_span_t rodata_placeholder =
      iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_plan_rodata(file, payloads, &rodata_placeholder));

  iree_host_size_t symbol_table_size = 0;
  if (!iree_host_size_checked_mul(symbol_count,
                                  LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
                                  &symbol_table_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO symbol table size overflow");
  }
  iree_const_byte_span_t symbol_table_placeholder =
      iree_make_const_byte_span(NULL, symbol_table_size);

  iree_const_byte_span_t hash = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_sysv_hash(file, symbol_count, &hash, arena));

  iree_const_byte_span_t dynamic_table_placeholder =
      iree_make_const_byte_span(NULL, LOOM_AMDGPU_HSACO_DYNAMIC_ENTRY_COUNT *
                                          LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE);

  loom_native_elf64le_section_t* sections = payloads->sections;
  sections[LOOM_AMDGPU_HSACO_SECTION_NOTE] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".note"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_NOTE,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = LOOM_AMDGPU_HSACO_NOTE_ALIGNMENT,
      .contents = note,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_DYNSYM] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".dynsym"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_DYNSYM,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = 8,
      .entry_size = LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
      .link = loom_amdgpu_hsaco_section_index(LOOM_AMDGPU_HSACO_SECTION_DYNSTR),
      .info = 1,
      .contents = symbol_table_placeholder,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_HASH] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".hash"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_HASH,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = 4,
      .entry_size = sizeof(uint32_t),
      .link = loom_amdgpu_hsaco_section_index(LOOM_AMDGPU_HSACO_SECTION_DYNSYM),
      .contents = hash,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_DYNSTR] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".dynstr"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
               LOOM_NATIVE_ELF_SECTION_FLAG_STRINGS,
      .alignment = 1,
      .contents = string_table,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_RODATA] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".rodata"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = LOOM_AMDGPU_KERNEL_DESCRIPTOR_ALIGNMENT,
      .contents = rodata_placeholder,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_TEXT] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".text"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
               LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
      .contents = text,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".dynamic"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_DYNAMIC,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_WRITE |
               LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
      .entry_size = LOOM_AMDGPU_HSACO_DYN_ENTRY_SIZE,
      .link = loom_amdgpu_hsaco_section_index(LOOM_AMDGPU_HSACO_SECTION_DYNSTR),
      .contents = dynamic_table_placeholder,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_SYMTAB] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".symtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
      .alignment = 8,
      .entry_size = LOOM_AMDGPU_HSACO_SYMBOL_ENTRY_SIZE,
      .link = loom_amdgpu_hsaco_section_index(LOOM_AMDGPU_HSACO_SECTION_STRTAB),
      .info = 1,
      .contents = symbol_table_placeholder,
  };
  sections[LOOM_AMDGPU_HSACO_SECTION_STRTAB] = (loom_native_elf64le_section_t){
      .name = IREE_SV(".strtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_STRINGS,
      .alignment = 1,
      .contents = string_table,
  };

  uint64_t read_end = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_assign_read_addresses(sections, &read_end));
  payloads->read_segment_file_size = read_end;
  uint64_t text_address = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
      iree_max(read_end, LOOM_AMDGPU_HSACO_TEXT_SEGMENT_BASE),
      LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT, &text_address));
  sections[LOOM_AMDGPU_HSACO_SECTION_TEXT].address = text_address;
  uint64_t dynamic_address = 0;
  uint64_t text_end = 0;
  if (!loom_amdgpu_hsaco_checked_add_uint64(text_address, text.data_length,
                                            &text_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO text segment address overflow");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_align_uint64(
      text_end, LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT, &dynamic_address));
  sections[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC].address = dynamic_address;

  iree_const_byte_span_t rodata = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_build_rodata(
      file, payloads, sections[LOOM_AMDGPU_HSACO_SECTION_RODATA].address,
      text_address,
      sections[LOOM_AMDGPU_HSACO_SECTION_RODATA].contents.data_length, &rodata,
      arena));
  sections[LOOM_AMDGPU_HSACO_SECTION_RODATA].contents = rodata;

  iree_const_byte_span_t symbol_table = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_build_symbol_table(
      file, payloads, sections[LOOM_AMDGPU_HSACO_SECTION_RODATA].address,
      text_address, &symbol_table, arena));
  sections[LOOM_AMDGPU_HSACO_SECTION_DYNSYM].contents = symbol_table;
  sections[LOOM_AMDGPU_HSACO_SECTION_SYMTAB].contents = symbol_table;

  iree_const_byte_span_t dynamic_table = iree_make_const_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_build_dynamic_table(sections, &dynamic_table, arena));
  sections[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC].contents = dynamic_table;
  return iree_ok_status();
}

static void loom_amdgpu_hsaco_prepare_segments(
    loom_amdgpu_hsaco_payloads_t* payloads) {
  const loom_native_elf64le_section_t* sections = payloads->sections;
  loom_native_elf64le_segment_t* segments = payloads->segments;
  segments[0] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_PHDR,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .file_offset = LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE,
      .file_size = LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT *
                   LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE,
      .memory_size = LOOM_AMDGPU_HSACO_PROGRAM_HEADER_COUNT *
                     LOOM_AMDGPU_HSACO_PROGRAM_HEADER_SIZE,
      .virtual_address = LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE,
      .physical_address = LOOM_AMDGPU_HSACO_ELF_HEADER_SIZE,
      .alignment = 8,
  };
  segments[1] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .file_offset = 0,
      .file_size = payloads->read_segment_file_size,
      .memory_size = payloads->read_segment_file_size,
      .virtual_address = 0,
      .physical_address = 0,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
  };
  segments[2] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE,
      .first_section = LOOM_AMDGPU_HSACO_SECTION_TEXT,
      .section_count = 1,
      .virtual_address = sections[LOOM_AMDGPU_HSACO_SECTION_TEXT].address,
      .physical_address = sections[LOOM_AMDGPU_HSACO_SECTION_TEXT].address,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
  };
  segments[3] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE,
      .first_section = LOOM_AMDGPU_HSACO_SECTION_DYNAMIC,
      .section_count = 1,
      .virtual_address = sections[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC].address,
      .physical_address = sections[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC].address,
      .alignment = LOOM_AMDGPU_HSACO_SEGMENT_PAGE_ALIGNMENT,
  };
  segments[4] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_DYNAMIC,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE,
      .first_section = LOOM_AMDGPU_HSACO_SECTION_DYNAMIC,
      .section_count = 1,
      .virtual_address = sections[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC].address,
      .physical_address = sections[LOOM_AMDGPU_HSACO_SECTION_DYNAMIC].address,
      .alignment = 8,
  };
  segments[5] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_GNU_STACK,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE,
      .file_offset = 0,
      .file_size = 0,
      .memory_size = 0,
      .virtual_address = 0,
      .physical_address = 0,
      .alignment = 0,
  };
  segments[6] = (loom_native_elf64le_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .first_section = LOOM_AMDGPU_HSACO_SECTION_NOTE,
      .section_count = 1,
      .virtual_address = sections[LOOM_AMDGPU_HSACO_SECTION_NOTE].address,
      .physical_address = sections[LOOM_AMDGPU_HSACO_SECTION_NOTE].address,
      .alignment = 4,
  };
}

iree_status_t loom_amdgpu_hsaco_write_file(
    const loom_amdgpu_hsaco_file_t* file, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_hsaco_validate_file(file));

  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(file->processor, &processor));
  if (processor->elf.machine_flags == 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU HSACO processor '%.*s' has no ELF e_flags mapping",
        (int)file->processor.size, file->processor.data);
  }
  const uint32_t elf_flags =
      processor->elf.machine_flags | processor->elf.feature_flags;

  loom_amdgpu_hsaco_payloads_t payloads = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, file->kernel_count, sizeof(payloads.kernel_layouts[0]),
      (void**)&payloads.kernel_layouts));
  memset(payloads.kernel_layouts, 0,
         file->kernel_count * sizeof(payloads.kernel_layouts[0]));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_copy_metadata_kernels(file, &payloads, scratch_arena));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_prepare_sections(file, &payloads, scratch_arena));
  loom_amdgpu_hsaco_prepare_segments(&payloads);

  const loom_native_elf64le_file_t elf_file = {
      .type = LOOM_NATIVE_ELF_FILE_TYPE_DYN,
      .machine = LOOM_NATIVE_ELF_MACHINE_AMDGPU,
      .os_abi = LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA,
      .abi_version = LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V6,
      .flags = elf_flags,
      .entry = 0,
      .sections = payloads.sections,
      .section_count = IREE_ARRAYSIZE(payloads.sections),
      .segments = payloads.segments,
      .segment_count = IREE_ARRAYSIZE(payloads.segments),
  };
  return loom_native_elf64le_write_file(&elf_file, stream, scratch_arena);
}
