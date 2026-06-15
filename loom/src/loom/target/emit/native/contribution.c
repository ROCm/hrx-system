// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/contribution.h"

#include <inttypes.h>
#include <string.h>

typedef struct loom_native_section_accumulator_t {
  // Section descriptor being assembled.
  loom_native_elf64le_section_t section;
  // Total assembled byte length so far, before final allocation.
  uint64_t next_offset;
} loom_native_section_accumulator_t;

static bool loom_native_contribution_checked_add_uint64(uint64_t lhs,
                                                        uint64_t rhs,
                                                        uint64_t* out_result) {
  *out_result = lhs + rhs;
  return *out_result >= lhs;
}

static bool loom_native_contribution_checked_align_uint64(
    uint64_t value, uint64_t alignment, uint64_t* out_result) {
  if (!loom_native_contribution_checked_add_uint64(value, alignment - 1u,
                                                   out_result)) {
    return false;
  }
  *out_result &= ~(alignment - 1u);
  return true;
}

static uint64_t loom_native_contribution_normalize_alignment(
    uint64_t alignment) {
  return alignment == 0 ? 1u : alignment;
}

static iree_status_t loom_native_contribution_validate_name(
    iree_string_view_t name, iree_host_size_t index) {
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native contribution %" PRIhsz " section name is required", index);
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    if (name.data[i] == '\0') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native contribution %" PRIhsz
                              " section name contains an embedded NUL",
                              index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_native_contribution_validate(
    const loom_native_section_contribution_t* contribution,
    iree_host_size_t index) {
  IREE_RETURN_IF_ERROR(loom_native_contribution_validate_name(
      contribution->section_name, index));
  if (contribution->section_type == LOOM_NATIVE_ELF_SECTION_TYPE_NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native contribution %" PRIhsz " must not use SHT_NULL", index);
  }
  const uint64_t alignment = loom_native_contribution_normalize_alignment(
      contribution->contribution_alignment);
  if (!iree_is_power_of_two_uint64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native contribution %" PRIhsz
                            " alignment must be a power of two",
                            index);
  }
  if (contribution->contents.data == NULL &&
      contribution->contents.data_length != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native contribution %" PRIhsz " has a size but no contents", index);
  }
  return iree_ok_status();
}

static bool loom_native_contribution_sections_match(
    const loom_native_elf64le_section_t* section,
    const loom_native_section_contribution_t* contribution) {
  return section->type == contribution->section_type &&
         section->flags == contribution->section_flags &&
         section->entry_size == contribution->entry_size &&
         section->link == contribution->link &&
         section->info == contribution->info;
}

static iree_status_t loom_native_contribution_find_or_add_section(
    loom_native_section_accumulator_t* accumulators,
    iree_host_size_t* inout_section_count,
    const loom_native_section_contribution_t* contribution,
    iree_host_size_t contribution_index, iree_host_size_t* out_section_index) {
  *out_section_index = 0;
  for (iree_host_size_t i = 0; i < *inout_section_count; ++i) {
    loom_native_section_accumulator_t* accumulator = &accumulators[i];
    if (!iree_string_view_equal(accumulator->section.name,
                                contribution->section_name)) {
      continue;
    }
    if (!loom_native_contribution_sections_match(&accumulator->section,
                                                 contribution)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "native contribution %" PRIhsz
          " has section metadata that conflicts with an earlier `%.*s` "
          "contribution",
          contribution_index, (int)contribution->section_name.size,
          contribution->section_name.data);
    }
    *out_section_index = i;
    return iree_ok_status();
  }

  const iree_host_size_t section_index = *inout_section_count;
  loom_native_section_accumulator_t* accumulator = &accumulators[section_index];
  *accumulator = (loom_native_section_accumulator_t){
      .section =
          {
              .name = contribution->section_name,
              .type = contribution->section_type,
              .flags = contribution->section_flags,
              .address = 0,
              .alignment = loom_native_contribution_normalize_alignment(
                  contribution->contribution_alignment),
              .entry_size = contribution->entry_size,
              .link = contribution->link,
              .info = contribution->info,
              .contents = iree_const_byte_span_empty(),
          },
      .next_offset = 0,
  };
  *inout_section_count = section_index + 1u;
  *out_section_index = section_index;
  return iree_ok_status();
}

static iree_status_t loom_native_contribution_plan_layout(
    const loom_native_section_contribution_t* contributions,
    iree_host_size_t contribution_count,
    loom_native_section_accumulator_t* accumulators,
    iree_host_size_t* out_section_count,
    loom_native_section_contribution_layout_t* contribution_layouts) {
  *out_section_count = 0;
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_native_section_contribution_t* contribution = &contributions[i];
    IREE_RETURN_IF_ERROR(loom_native_contribution_validate(contribution, i));

    iree_host_size_t section_index = 0;
    IREE_RETURN_IF_ERROR(loom_native_contribution_find_or_add_section(
        accumulators, out_section_count, contribution, i, &section_index));

    loom_native_section_accumulator_t* accumulator =
        &accumulators[section_index];
    const uint64_t alignment = loom_native_contribution_normalize_alignment(
        contribution->contribution_alignment);
    if (alignment > accumulator->section.alignment) {
      accumulator->section.alignment = alignment;
    }

    uint64_t section_offset = 0;
    if (!loom_native_contribution_checked_align_uint64(
            accumulator->next_offset, alignment, &section_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native contribution section layout overflow");
    }
    uint64_t next_offset = 0;
    if (!loom_native_contribution_checked_add_uint64(
            section_offset, (uint64_t)contribution->contents.data_length,
            &next_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native contribution section layout overflow");
    }
    accumulator->next_offset = next_offset;
    contribution_layouts[i] = (loom_native_section_contribution_layout_t){
        .section_index = section_index,
        .section_offset = section_offset,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_native_contribution_allocate_output(
    const loom_native_section_accumulator_t* accumulators,
    iree_host_size_t section_count,
    loom_native_elf64le_section_t** out_sections,
    uint8_t*** out_section_contents, iree_arena_allocator_t* arena) {
  *out_sections = NULL;
  *out_section_contents = NULL;
  if (section_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t section_name_bytes = 0;
  for (iree_host_size_t i = 0; i < section_count; ++i) {
    if (!iree_host_size_checked_add(section_name_bytes,
                                    accumulators[i].section.name.size,
                                    &section_name_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native section names are too large");
    }
  }
  char* section_names = NULL;
  if (section_name_bytes != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, section_name_bytes, (void**)&section_names));
  }

  loom_native_elf64le_section_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, section_count, sizeof(*sections), (void**)&sections));
  uint8_t** section_contents = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, section_count,
                                                 sizeof(*section_contents),
                                                 (void**)&section_contents));
  memset(section_contents, 0, section_count * sizeof(*section_contents));

  char* section_name_cursor = section_names;
  for (iree_host_size_t i = 0; i < section_count; ++i) {
    const loom_native_section_accumulator_t* accumulator = &accumulators[i];
    sections[i] = accumulator->section;
    if (accumulator->section.name.size != 0) {
      memcpy(section_name_cursor, accumulator->section.name.data,
             accumulator->section.name.size);
      sections[i].name = iree_make_string_view(section_name_cursor,
                                               accumulator->section.name.size);
      section_name_cursor += accumulator->section.name.size;
    }
    if (accumulator->next_offset == 0) {
      sections[i].contents = iree_const_byte_span_empty();
      continue;
    }
    if (accumulator->next_offset > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native assembled section `%.*s` is too large",
                              (int)accumulator->section.name.size,
                              accumulator->section.name.data);
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, (iree_host_size_t)accumulator->next_offset,
                            (void**)&section_contents[i]));
    memset(section_contents[i], 0, (iree_host_size_t)accumulator->next_offset);
    sections[i].contents = iree_make_const_byte_span(
        section_contents[i], (iree_host_size_t)accumulator->next_offset);
  }
  *out_sections = sections;
  *out_section_contents = section_contents;
  return iree_ok_status();
}

static iree_status_t loom_native_contribution_copy_contents(
    const loom_native_section_contribution_t* contributions,
    iree_host_size_t contribution_count,
    const loom_native_section_contribution_layout_t* contribution_layouts,
    uint8_t** section_contents) {
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_native_section_contribution_t* contribution = &contributions[i];
    if (contribution->contents.data_length == 0) {
      continue;
    }
    const loom_native_section_contribution_layout_t* layout =
        &contribution_layouts[i];
    if (layout->section_offset > IREE_HOST_SIZE_MAX ||
        contribution->contents.data_length >
            IREE_HOST_SIZE_MAX - (iree_host_size_t)layout->section_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native contribution copy range overflows host "
                              "address space");
    }
    uint8_t* target = section_contents[layout->section_index] +
                      (iree_host_size_t)layout->section_offset;
    memcpy(target, contribution->contents.data,
           contribution->contents.data_length);
  }
  return iree_ok_status();
}

iree_status_t loom_native_assemble_section_contributions(
    const loom_native_section_contribution_t* contributions,
    iree_host_size_t contribution_count,
    loom_native_section_contribution_assembly_t* out_assembly,
    iree_arena_allocator_t* arena) {
  *out_assembly = (loom_native_section_contribution_assembly_t){0};
  if (contribution_count == 0) {
    return iree_ok_status();
  }

  loom_native_section_accumulator_t* accumulators = NULL;
  iree_status_t status = iree_arena_allocate_array(
      arena, contribution_count, sizeof(*accumulators), (void**)&accumulators);
  loom_native_section_contribution_layout_t* contribution_layouts = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, contribution_count,
                                       sizeof(*contribution_layouts),
                                       (void**)&contribution_layouts);
  }

  iree_host_size_t section_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_native_contribution_plan_layout(
        contributions, contribution_count, accumulators, &section_count,
        contribution_layouts);
  }
  loom_native_elf64le_section_t* sections = NULL;
  uint8_t** section_contents = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_native_contribution_allocate_output(
        accumulators, section_count, &sections, &section_contents, arena);
  }
  if (iree_status_is_ok(status)) {
    status = loom_native_contribution_copy_contents(
        contributions, contribution_count, contribution_layouts,
        section_contents);
  }

  if (iree_status_is_ok(status)) {
    *out_assembly = (loom_native_section_contribution_assembly_t){
        .sections = sections,
        .section_count = section_count,
        .contribution_layouts = contribution_layouts,
        .contribution_layout_count = contribution_count,
    };
  }
  return status;
}
