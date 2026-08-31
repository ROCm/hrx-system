// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

#include <inttypes.h>

// One contiguous mapping range shared by instruction and bundle fields.
typedef struct loom_aie2p_encoding_bit_range_t {
  // First destination bit in the instruction or bundle.
  uint8_t target_bit;
  // First source bit in the field or slot value.
  uint8_t value_bit;
  // Number of contiguous bits in the range.
  uint8_t bit_count;
} loom_aie2p_encoding_bit_range_t;

// Deduplicated mapping pattern referenced by physical fields.
typedef struct loom_aie2p_encoding_mapping_pattern_t {
  // First row in the compact bit-range table.
  uint8_t bit_range_offset;
  // Number of contiguous ranges in the pattern.
  uint8_t bit_range_count;
  // Number of significant low source bits accepted by the pattern.
  uint8_t value_bit_count;
} loom_aie2p_encoding_mapping_pattern_t;

// Deduplicated instruction field identity and mapping pattern.
typedef struct loom_aie2p_instruction_field_layout_t {
  // Target-owned field identifier.
  uint8_t field_id;
  // Mapping pattern scattering the field value into instruction bits.
  uint8_t mapping_pattern_id;
} loom_aie2p_instruction_field_layout_t;

// Deduplicated instruction layout shared by concrete forms.
typedef struct loom_aie2p_instruction_layout_t {
  // First field-layout identifier in the field-reference table.
  uint16_t field_ref_offset;
  // Number of encoded operand fields.
  uint8_t field_count;
  // Physical slot identifier.
  uint8_t slot_id;
} loom_aie2p_instruction_layout_t;

// Concrete named instruction form and its shared physical layout.
typedef struct loom_aie2p_instruction_form_t {
  // Byte offset into the stable instruction-name table.
  uint16_t name_offset;
  // Deduplicated instruction layout identifier.
  uint16_t layout_id;
  // Byte length of the stable instruction name.
  uint8_t name_length;
  // Number of subsequent bundles in the architectural delay window.
  uint8_t delay_slot_count;
} loom_aie2p_instruction_form_t;

// One physical slot within a bundle format.
typedef struct loom_aie2p_bundle_field_layout_t {
  // Physical slot identifier.
  uint8_t slot_id;
  // Mapping pattern scattering the slot value into bundle bits.
  uint8_t mapping_pattern_id;
} loom_aie2p_bundle_field_layout_t;

// Concrete variable-width bundle format.
typedef struct loom_aie2p_bundle_layout_t {
  // Per-byte bits fixed by the bundle format discriminator.
  uint8_t fixed_mask[LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE];
  // Per-byte required values for bits in |fixed_mask|.
  uint8_t fixed_value[LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE];
  // Byte offset into the stable bundle-name table.
  uint16_t name_offset;
  // First row in the bundle field table.
  uint16_t field_offset;
  // Byte length of the stable bundle name.
  uint8_t name_length;
  // Total number of encoded bits in the bundle.
  uint8_t bit_count;
  // Number of physical slot fields.
  uint8_t field_count;
} loom_aie2p_bundle_layout_t;

// Contiguous search range for instruction forms in one physical slot.
typedef struct loom_aie2p_instruction_search_range_t {
  // First row in the instruction search-order table.
  uint16_t offset;
  // Number of candidate instruction forms for the slot.
  uint16_t count;
} loom_aie2p_instruction_search_range_t;

#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding_tables.inl"

static_assert(sizeof(loom_aie2p_encoding_bit_range_t) == 3,
              "AIE2P bit ranges must remain compact");
static_assert(sizeof(loom_aie2p_encoding_mapping_pattern_t) == 3,
              "AIE2P mapping patterns must remain compact");
static_assert(sizeof(loom_aie2p_instruction_field_layout_t) == 2,
              "AIE2P instruction fields must remain compact");
static_assert(sizeof(loom_aie2p_instruction_layout_t) == 4,
              "AIE2P instruction layouts must remain compact");
static_assert(sizeof(loom_aie2p_instruction_form_t) == 6,
              "AIE2P instruction forms must remain compact");
static_assert(sizeof(loom_aie2p_bundle_field_layout_t) == 2,
              "AIE2P bundle fields must remain compact");
static_assert(sizeof(loom_aie2p_bundle_layout_t) == 40,
              "AIE2P bundle layouts must remain compact");
static_assert(
    sizeof(kLoomAie2pSlotBitCounts) + sizeof(kLoomAie2pEncodingFieldNames) +
            sizeof(kLoomAie2pEncodingBitRanges) +
            sizeof(kLoomAie2pEncodingMappingPatterns) +
            sizeof(kLoomAie2pInstructionFieldLayouts) +
            sizeof(kLoomAie2pInstructionLayoutFieldRefs) +
            sizeof(kLoomAie2pInstructionLayoutFixedMasks) +
            sizeof(kLoomAie2pInstructionLayouts) +
            sizeof(kLoomAie2pInstructionNames) +
            sizeof(kLoomAie2pInstructionFixedValues) +
            sizeof(kLoomAie2pInstructionForms) +
            sizeof(kLoomAie2pBundleFields) + sizeof(kLoomAie2pBundleNames) +
            sizeof(kLoomAie2pBundleLayouts) +
            sizeof(kLoomAie2pInstructionSearchOrder) +
            sizeof(kLoomAie2pInstructionSearchRanges) <=
        64 * 1024,
    "complete AIE2P physical encoding tables must remain within 64 KiB");

static const loom_aie2p_instruction_form_t*
loom_aie2p_encoding_instruction_form(loom_aie2p_instruction_id_t instruction) {
  if (instruction == LOOM_AIE2P_INSTRUCTION_ID_INVALID ||
      instruction >= IREE_ARRAYSIZE(kLoomAie2pInstructionForms)) {
    return NULL;
  }
  return &kLoomAie2pInstructionForms[instruction];
}

static const loom_aie2p_instruction_layout_t*
loom_aie2p_encoding_instruction_layout(
    const loom_aie2p_instruction_form_t* form) {
  return &kLoomAie2pInstructionLayouts[form->layout_id];
}

static const loom_aie2p_bundle_layout_t* loom_aie2p_encoding_bundle_layout(
    loom_aie2p_bundle_format_id_t format) {
  if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID ||
      format >= IREE_ARRAYSIZE(kLoomAie2pBundleLayouts)) {
    return NULL;
  }
  return &kLoomAie2pBundleLayouts[format];
}

static iree_string_view_t loom_aie2p_encoding_instruction_name(
    const loom_aie2p_instruction_form_t* form) {
  return iree_make_string_view(kLoomAie2pInstructionNames + form->name_offset,
                               form->name_length);
}

static iree_string_view_t loom_aie2p_encoding_bundle_name(
    const loom_aie2p_bundle_layout_t* layout) {
  return iree_make_string_view(kLoomAie2pBundleNames + layout->name_offset,
                               layout->name_length);
}

static uint64_t loom_aie2p_encoding_low_bit_mask(uint8_t bit_count) {
  return UINT64_MAX >> (64u - bit_count);
}

static uint64_t loom_aie2p_encoding_scatter_u64(uint64_t container,
                                                uint8_t mapping_pattern_id,
                                                uint64_t value) {
  const loom_aie2p_encoding_mapping_pattern_t* pattern =
      &kLoomAie2pEncodingMappingPatterns[mapping_pattern_id];
  for (uint8_t i = 0; i < pattern->bit_range_count; ++i) {
    const loom_aie2p_encoding_bit_range_t* range =
        &kLoomAie2pEncodingBitRanges[pattern->bit_range_offset + i];
    const uint64_t low_mask =
        loom_aie2p_encoding_low_bit_mask(range->bit_count);
    const uint64_t target_mask = low_mask << range->target_bit;
    const uint64_t range_value = ((value >> range->value_bit) & low_mask)
                                 << range->target_bit;
    container = (container & ~target_mask) | range_value;
  }
  return container;
}

static uint64_t loom_aie2p_encoding_gather_u64(uint64_t container,
                                               uint8_t mapping_pattern_id) {
  const loom_aie2p_encoding_mapping_pattern_t* pattern =
      &kLoomAie2pEncodingMappingPatterns[mapping_pattern_id];
  uint64_t value = 0;
  for (uint8_t i = 0; i < pattern->bit_range_count; ++i) {
    const loom_aie2p_encoding_bit_range_t* range =
        &kLoomAie2pEncodingBitRanges[pattern->bit_range_offset + i];
    const uint64_t low_mask =
        loom_aie2p_encoding_low_bit_mask(range->bit_count);
    const uint64_t range_value = ((container >> range->target_bit) & low_mask)
                                 << range->value_bit;
    value |= range_value;
  }
  return value;
}

static void loom_aie2p_encoding_scatter_bytes(uint8_t* data,
                                              uint8_t mapping_pattern_id,
                                              uint64_t value) {
  const loom_aie2p_encoding_mapping_pattern_t* pattern =
      &kLoomAie2pEncodingMappingPatterns[mapping_pattern_id];
  for (uint8_t i = 0; i < pattern->bit_range_count; ++i) {
    const loom_aie2p_encoding_bit_range_t* range =
        &kLoomAie2pEncodingBitRanges[pattern->bit_range_offset + i];
    for (uint8_t j = 0; j < range->bit_count; ++j) {
      const uint8_t target_bit = range->target_bit + j;
      const uint8_t byte_index = target_bit / 8;
      const uint8_t byte_mask = (uint8_t)(1u << (target_bit % 8));
      if (value & (UINT64_C(1) << (range->value_bit + j))) {
        data[byte_index] |= byte_mask;
      } else {
        data[byte_index] &= (uint8_t)~byte_mask;
      }
    }
  }
}

static uint64_t loom_aie2p_encoding_gather_bytes(const uint8_t* data,
                                                 uint8_t mapping_pattern_id) {
  const loom_aie2p_encoding_mapping_pattern_t* pattern =
      &kLoomAie2pEncodingMappingPatterns[mapping_pattern_id];
  uint64_t value = 0;
  for (uint8_t i = 0; i < pattern->bit_range_count; ++i) {
    const loom_aie2p_encoding_bit_range_t* range =
        &kLoomAie2pEncodingBitRanges[pattern->bit_range_offset + i];
    for (uint8_t j = 0; j < range->bit_count; ++j) {
      const uint8_t target_bit = range->target_bit + j;
      const uint8_t byte_index = target_bit / 8;
      const uint8_t byte_mask = (uint8_t)(1u << (target_bit % 8));
      if (data[byte_index] & byte_mask) {
        value |= UINT64_C(1) << (range->value_bit + j);
      }
    }
  }
  return value;
}

static const loom_aie2p_encoding_field_value_t*
loom_aie2p_encoding_find_field_value(
    uint8_t field_id, const loom_aie2p_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count) {
  for (iree_host_size_t i = 0; i < field_value_count; ++i) {
    if ((uint16_t)field_values[i].field_id == field_id) {
      return &field_values[i];
    }
  }
  return NULL;
}

static const loom_aie2p_encoded_slot_t* loom_aie2p_encoding_find_slot(
    uint8_t slot_id, const loom_aie2p_encoded_slot_t* encoded_slots,
    iree_host_size_t encoded_slot_count) {
  for (iree_host_size_t i = 0; i < encoded_slot_count; ++i) {
    if ((uint16_t)encoded_slots[i].slot == slot_id) return &encoded_slots[i];
  }
  return NULL;
}

static const loom_aie2p_instruction_field_layout_t*
loom_aie2p_encoding_instruction_field(
    const loom_aie2p_instruction_layout_t* layout, uint8_t field_ordinal) {
  const uint8_t field_layout_id =
      kLoomAie2pInstructionLayoutFieldRefs[layout->field_ref_offset +
                                           field_ordinal];
  return &kLoomAie2pInstructionFieldLayouts[field_layout_id];
}

iree_string_view_t loom_aie2p_encoding_llvm_aie_source_commit(void) {
  return kLoomAie2pEncodingLlvmAieSourceCommit;
}

iree_host_size_t loom_aie2p_encoding_instruction_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pInstructionForms) - 1;
}

iree_host_size_t loom_aie2p_encoding_field_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pEncodingFieldNames) - 1;
}

iree_host_size_t loom_aie2p_encoding_bundle_format_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pBundleLayouts) - 1;
}

loom_aie2p_instruction_id_t loom_aie2p_encoding_find_instruction(
    iree_string_view_t name) {
  iree_host_size_t begin = 1;
  iree_host_size_t end = IREE_ARRAYSIZE(kLoomAie2pInstructionForms);
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const iree_string_view_t middle_name = loom_aie2p_encoding_instruction_name(
        &kLoomAie2pInstructionForms[middle]);
    const int comparison = iree_string_view_compare(name, middle_name);
    if (comparison == 0) return (loom_aie2p_instruction_id_t)middle;
    if (comparison < 0) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  return LOOM_AIE2P_INSTRUCTION_ID_INVALID;
}

loom_aie2p_encoding_field_id_t loom_aie2p_encoding_find_field(
    iree_string_view_t name) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(kLoomAie2pEncodingFieldNames);
       ++i) {
    if (iree_string_view_equal(kLoomAie2pEncodingFieldNames[i], name)) {
      return (loom_aie2p_encoding_field_id_t)i;
    }
  }
  return LOOM_AIE2P_ENCODING_FIELD_ID_INVALID;
}

loom_aie2p_bundle_format_id_t loom_aie2p_encoding_find_bundle_format(
    iree_string_view_t name) {
  iree_host_size_t begin = 1;
  iree_host_size_t end = IREE_ARRAYSIZE(kLoomAie2pBundleLayouts);
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const iree_string_view_t middle_name =
        loom_aie2p_encoding_bundle_name(&kLoomAie2pBundleLayouts[middle]);
    const int comparison = iree_string_view_compare(name, middle_name);
    if (comparison == 0) return (loom_aie2p_bundle_format_id_t)middle;
    if (comparison < 0) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  return LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
}

loom_aie2p_bundle_format_id_t loom_aie2p_encoding_find_bundle_format_for_slots(
    const loom_aie2p_slot_t* slots, iree_host_size_t slot_count) {
  if (slot_count == 0 ||
      slot_count > LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT || slots == NULL) {
    return LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
  }
  uint16_t requested_slot_mask = 0;
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    if (slots[i] <= LOOM_AIE2P_SLOT_INVALID ||
        slots[i] >= LOOM_AIE2P_SLOT_COUNT) {
      return LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
    }
    const uint16_t slot_bit = (uint16_t)(1u << slots[i]);
    if (requested_slot_mask & slot_bit) {
      return LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
    }
    requested_slot_mask |= slot_bit;
  }

  for (iree_host_size_t format = 1;
       format < IREE_ARRAYSIZE(kLoomAie2pBundleLayouts); ++format) {
    const loom_aie2p_bundle_layout_t* layout = &kLoomAie2pBundleLayouts[format];
    if (layout->field_count != slot_count) continue;
    uint16_t format_slot_mask = 0;
    for (uint8_t i = 0; i < layout->field_count; ++i) {
      const loom_aie2p_bundle_field_layout_t* field =
          &kLoomAie2pBundleFields[layout->field_offset + i];
      format_slot_mask |= (uint16_t)(1u << field->slot_id);
    }
    if (format_slot_mask == requested_slot_mask) {
      return (loom_aie2p_bundle_format_id_t)format;
    }
  }
  return LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
}

iree_string_view_t loom_aie2p_encoding_field_name(
    loom_aie2p_encoding_field_id_t field) {
  if (field == LOOM_AIE2P_ENCODING_FIELD_ID_INVALID ||
      field >= IREE_ARRAYSIZE(kLoomAie2pEncodingFieldNames)) {
    return iree_string_view_empty();
  }
  return kLoomAie2pEncodingFieldNames[field];
}

static iree_string_view_t loom_aie2p_encoding_diagnostic_field_name(
    uint8_t field) {
  iree_string_view_t name =
      loom_aie2p_encoding_field_name((loom_aie2p_encoding_field_id_t)field);
  return name.size ? name : IREE_SV("<invalid>");
}

bool loom_aie2p_encoding_query_instruction_info(
    loom_aie2p_instruction_id_t instruction,
    loom_aie2p_instruction_info_t* out_info) {
  if (out_info == NULL) return false;
  const loom_aie2p_instruction_form_t* form =
      loom_aie2p_encoding_instruction_form(instruction);
  if (form == NULL) return false;
  const loom_aie2p_instruction_layout_t* layout =
      loom_aie2p_encoding_instruction_layout(form);
  *out_info = (loom_aie2p_instruction_info_t){
      .name = loom_aie2p_encoding_instruction_name(form),
      .slot = (loom_aie2p_slot_t)layout->slot_id,
      .bit_count = kLoomAie2pSlotBitCounts[layout->slot_id],
      .delay_slot_count = form->delay_slot_count,
  };
  return true;
}

bool loom_aie2p_encoding_query_bundle_format_info(
    loom_aie2p_bundle_format_id_t format,
    loom_aie2p_bundle_format_info_t* out_info) {
  if (out_info == NULL) return false;
  const loom_aie2p_bundle_layout_t* layout =
      loom_aie2p_encoding_bundle_layout(format);
  if (layout == NULL) return false;
  *out_info = (loom_aie2p_bundle_format_info_t){
      .name = loom_aie2p_encoding_bundle_name(layout),
      .bit_count = layout->bit_count,
      .slot_count = layout->field_count,
  };
  return true;
}

static uint64_t loom_aie2p_encoding_pack_verified_instruction_fields(
    loom_aie2p_instruction_id_t instruction,
    const loom_aie2p_instruction_layout_t* layout,
    const loom_aie2p_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count) {
  uint64_t encoded_value = kLoomAie2pInstructionFixedValues[instruction];
  for (uint8_t i = 0; i < layout->field_count; ++i) {
    const loom_aie2p_instruction_field_layout_t* field =
        loom_aie2p_encoding_instruction_field(layout, i);
    const loom_aie2p_encoding_field_value_t* field_value =
        loom_aie2p_encoding_find_field_value(field->field_id, field_values,
                                             field_value_count);
    IREE_ASSERT(field_value != NULL &&
                "verified AIE2P instruction field is missing");
    encoded_value = loom_aie2p_encoding_scatter_u64(
        encoded_value, field->mapping_pattern_id, field_value->value);
  }
  return encoded_value;
}

iree_status_t loom_aie2p_encoding_pack_instruction(
    loom_aie2p_instruction_id_t instruction,
    const loom_aie2p_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count, uint64_t* out_value) {
  if (out_value == NULL || (field_value_count != 0 && field_values == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P instruction encoding requires valid field and output storage");
  }
  *out_value = 0;
  const loom_aie2p_instruction_form_t* form =
      loom_aie2p_encoding_instruction_form(instruction);
  if (form == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown AIE2P instruction id %d", instruction);
  }
  const loom_aie2p_instruction_layout_t* layout =
      loom_aie2p_encoding_instruction_layout(form);
  const iree_string_view_t instruction_name =
      loom_aie2p_encoding_instruction_name(form);

  for (iree_host_size_t i = 0; i < field_value_count; ++i) {
    bool expected_field = false;
    for (uint8_t j = 0; j < layout->field_count; ++j) {
      const loom_aie2p_instruction_field_layout_t* field =
          loom_aie2p_encoding_instruction_field(layout, j);
      expected_field |= field->field_id == (uint16_t)field_values[i].field_id;
    }
    if (!expected_field) {
      const iree_string_view_t field_name =
          loom_aie2p_encoding_diagnostic_field_name(
              (uint8_t)field_values[i].field_id);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P instruction %.*s does not encode field %.*s",
          (int)instruction_name.size, instruction_name.data,
          (int)field_name.size, field_name.data);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (field_values[j].field_id == field_values[i].field_id) {
        const iree_string_view_t field_name =
            loom_aie2p_encoding_diagnostic_field_name(
                (uint8_t)field_values[i].field_id);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P instruction %.*s repeats field %.*s",
                                (int)instruction_name.size,
                                instruction_name.data, (int)field_name.size,
                                field_name.data);
      }
    }
  }

  for (uint8_t i = 0; i < layout->field_count; ++i) {
    const loom_aie2p_instruction_field_layout_t* field =
        loom_aie2p_encoding_instruction_field(layout, i);
    const loom_aie2p_encoding_field_value_t* field_value =
        loom_aie2p_encoding_find_field_value(field->field_id, field_values,
                                             field_value_count);
    if (field_value == NULL) {
      const iree_string_view_t field_name =
          loom_aie2p_encoding_diagnostic_field_name(field->field_id);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P instruction %.*s is missing field %.*s",
                              (int)instruction_name.size, instruction_name.data,
                              (int)field_name.size, field_name.data);
    }
    const loom_aie2p_encoding_mapping_pattern_t* pattern =
        &kLoomAie2pEncodingMappingPatterns[field->mapping_pattern_id];
    const uint64_t value_mask =
        loom_aie2p_encoding_low_bit_mask(pattern->value_bit_count);
    if (field_value->value & ~value_mask) {
      const iree_string_view_t field_name =
          loom_aie2p_encoding_diagnostic_field_name(field->field_id);
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AIE2P instruction %.*s field %.*s value 0x%" PRIx64
          " exceeds mask 0x%" PRIx64,
          (int)instruction_name.size, instruction_name.data,
          (int)field_name.size, field_name.data, field_value->value,
          value_mask);
    }
  }
  *out_value = loom_aie2p_encoding_pack_verified_instruction_fields(
      instruction, layout, field_values, field_value_count);
  return iree_ok_status();
}

loom_aie2p_encoded_slot_t loom_aie2p_encoding_pack_verified_instruction(
    loom_aie2p_instruction_id_t instruction,
    const loom_aie2p_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count) {
  const loom_aie2p_instruction_form_t* form =
      &kLoomAie2pInstructionForms[instruction];
  const loom_aie2p_instruction_layout_t* layout =
      loom_aie2p_encoding_instruction_layout(form);
  IREE_ASSERT(field_value_count == layout->field_count &&
              "verified AIE2P instruction field count differs");
  return (loom_aie2p_encoded_slot_t){
      .slot = (loom_aie2p_slot_t)layout->slot_id,
      .value = loom_aie2p_encoding_pack_verified_instruction_fields(
          instruction, layout, field_values, field_value_count),
  };
}

iree_status_t loom_aie2p_encoding_unpack_instruction(
    loom_aie2p_instruction_id_t instruction, uint64_t encoded_value,
    iree_host_size_t field_value_capacity,
    loom_aie2p_encoding_field_value_t* out_field_values,
    iree_host_size_t* out_field_value_count) {
  if (out_field_value_count != NULL) *out_field_value_count = 0;
  const loom_aie2p_instruction_form_t* form =
      loom_aie2p_encoding_instruction_form(instruction);
  if (form == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown AIE2P instruction id %d", instruction);
  }
  const loom_aie2p_instruction_layout_t* layout =
      loom_aie2p_encoding_instruction_layout(form);
  const iree_string_view_t instruction_name =
      loom_aie2p_encoding_instruction_name(form);
  if (out_field_value_count != NULL) {
    *out_field_value_count = layout->field_count;
  }
  if (layout->field_count > field_value_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P instruction %.*s needs %u decoded fields; capacity is %" PRIhsz,
        (int)instruction_name.size, instruction_name.data, layout->field_count,
        field_value_capacity);
  }
  if (layout->field_count != 0 && out_field_values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P decoded field storage is NULL");
  }
  const uint64_t encoding_mask = loom_aie2p_encoding_low_bit_mask(
      kLoomAie2pSlotBitCounts[layout->slot_id]);
  const uint64_t fixed_mask =
      kLoomAie2pInstructionLayoutFixedMasks[form->layout_id];
  const uint64_t fixed_value = kLoomAie2pInstructionFixedValues[instruction];
  if ((encoded_value & ~encoding_mask) != 0 ||
      (encoded_value & fixed_mask) != fixed_value) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoded value 0x%" PRIx64 " does not match AIE2P instruction %.*s",
        encoded_value, (int)instruction_name.size, instruction_name.data);
  }

  for (uint8_t i = 0; i < layout->field_count; ++i) {
    const loom_aie2p_instruction_field_layout_t* field =
        loom_aie2p_encoding_instruction_field(layout, i);
    const uint64_t value = loom_aie2p_encoding_gather_u64(
        encoded_value, field->mapping_pattern_id);
    out_field_values[i] = (loom_aie2p_encoding_field_value_t){
        .field_id = (loom_aie2p_encoding_field_id_t)field->field_id,
        .value = value,
    };
  }
  return iree_ok_status();
}

iree_host_size_t loom_aie2p_encoding_query_instruction_candidates(
    loom_aie2p_slot_t slot, uint64_t encoded_value,
    iree_host_size_t candidate_capacity,
    loom_aie2p_instruction_id_t* out_candidates) {
  if (slot <= LOOM_AIE2P_SLOT_INVALID || slot >= LOOM_AIE2P_SLOT_COUNT) {
    return 0;
  }
  const uint8_t slot_bit_count = kLoomAie2pSlotBitCounts[slot];
  const uint64_t slot_mask = loom_aie2p_encoding_low_bit_mask(slot_bit_count);
  if ((encoded_value & ~slot_mask) != 0) return 0;

  const loom_aie2p_instruction_search_range_t search_range =
      kLoomAie2pInstructionSearchRanges[slot];
  iree_host_size_t candidate_count = 0;
  for (uint16_t i = 0; i < search_range.count; ++i) {
    const loom_aie2p_instruction_id_t instruction =
        (loom_aie2p_instruction_id_t)
            kLoomAie2pInstructionSearchOrder[search_range.offset + i];
    const loom_aie2p_instruction_form_t* form =
        &kLoomAie2pInstructionForms[instruction];
    const loom_aie2p_instruction_layout_t* layout =
        loom_aie2p_encoding_instruction_layout(form);
    const uint64_t fixed_mask =
        kLoomAie2pInstructionLayoutFixedMasks[form->layout_id];
    const uint64_t fixed_value = kLoomAie2pInstructionFixedValues[instruction];
    if ((encoded_value & fixed_mask) != fixed_value) {
      continue;
    }
    if (candidate_count < candidate_capacity && out_candidates != NULL) {
      out_candidates[candidate_count] = instruction;
    }
    ++candidate_count;
  }
  return candidate_count;
}

iree_status_t loom_aie2p_encoding_pack_bundle(
    loom_aie2p_bundle_format_id_t format,
    const loom_aie2p_encoded_slot_t* encoded_slots,
    iree_host_size_t encoded_slot_count,
    loom_aie2p_encoding_packet_t* out_packet) {
  if (out_packet == NULL ||
      (encoded_slot_count != 0 && encoded_slots == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P bundle encoding requires valid slot and output storage");
  }
  *out_packet = (loom_aie2p_encoding_packet_t){0};
  const loom_aie2p_bundle_layout_t* layout =
      loom_aie2p_encoding_bundle_layout(format);
  if (layout == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown AIE2P bundle format id %d", format);
  }
  const iree_string_view_t bundle_name =
      loom_aie2p_encoding_bundle_name(layout);

  for (iree_host_size_t i = 0; i < encoded_slot_count; ++i) {
    bool expected_slot = false;
    for (uint8_t j = 0; j < layout->field_count; ++j) {
      const loom_aie2p_bundle_field_layout_t* field =
          &kLoomAie2pBundleFields[layout->field_offset + j];
      expected_slot |= field->slot_id == (uint16_t)encoded_slots[i].slot;
    }
    if (!expected_slot) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P bundle %.*s does not contain physical slot %d",
          (int)bundle_name.size, bundle_name.data, encoded_slots[i].slot);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (encoded_slots[j].slot == encoded_slots[i].slot) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P bundle %.*s repeats physical slot %d",
                                (int)bundle_name.size, bundle_name.data,
                                encoded_slots[i].slot);
      }
    }
  }

  out_packet->data_length = layout->bit_count / 8;
  memcpy(out_packet->data, layout->fixed_value, out_packet->data_length);
  for (uint8_t i = 0; i < layout->field_count; ++i) {
    const loom_aie2p_bundle_field_layout_t* field =
        &kLoomAie2pBundleFields[layout->field_offset + i];
    const loom_aie2p_encoded_slot_t* encoded_slot =
        loom_aie2p_encoding_find_slot(field->slot_id, encoded_slots,
                                      encoded_slot_count);
    if (encoded_slot == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P bundle %.*s is missing physical slot %u",
                              (int)bundle_name.size, bundle_name.data,
                              field->slot_id);
    }
    const loom_aie2p_encoding_mapping_pattern_t* pattern =
        &kLoomAie2pEncodingMappingPatterns[field->mapping_pattern_id];
    const uint64_t value_mask =
        loom_aie2p_encoding_low_bit_mask(pattern->value_bit_count);
    if (encoded_slot->value & ~value_mask) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P bundle %.*s slot %u value 0x%" PRIx64
                              " exceeds mask 0x%" PRIx64,
                              (int)bundle_name.size, bundle_name.data,
                              field->slot_id, encoded_slot->value, value_mask);
    }
    loom_aie2p_encoding_scatter_bytes(
        out_packet->data, field->mapping_pattern_id, encoded_slot->value);
  }
  return iree_ok_status();
}

static bool loom_aie2p_encoding_bundle_layout_matches(
    const loom_aie2p_bundle_layout_t* layout, iree_const_byte_span_t packet) {
  const iree_host_size_t packet_length = layout->bit_count / 8u;
  if (packet_length > packet.data_length) return false;
  for (iree_host_size_t i = 0; i < packet_length; ++i) {
    if ((packet.data[i] & layout->fixed_mask[i]) != layout->fixed_value[i]) {
      return false;
    }
  }
  return true;
}

static void loom_aie2p_encoding_decode_bundle_layout(
    const loom_aie2p_bundle_layout_t* layout,
    loom_aie2p_bundle_format_id_t format, const uint8_t* packet_data,
    loom_aie2p_decoded_bundle_t* out_bundle) {
  *out_bundle = (loom_aie2p_decoded_bundle_t){
      .format = format,
      .slot_count = layout->field_count,
  };
  for (uint8_t i = 0; i < layout->field_count; ++i) {
    const loom_aie2p_bundle_field_layout_t* field =
        &kLoomAie2pBundleFields[layout->field_offset + i];
    out_bundle->slots[i] = (loom_aie2p_encoded_slot_t){
        .slot = (loom_aie2p_slot_t)field->slot_id,
        .value = loom_aie2p_encoding_gather_bytes(packet_data,
                                                  field->mapping_pattern_id),
    };
  }
}

iree_status_t loom_aie2p_encoding_decode_bundle(
    iree_const_byte_span_t packet, loom_aie2p_decoded_bundle_t* out_bundle) {
  if (out_bundle == NULL || packet.data == NULL || packet.data_length == 0 ||
      packet.data_length > LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P bundle decode requires 1-16 input bytes and "
                            "valid output storage");
  }
  *out_bundle = (loom_aie2p_decoded_bundle_t){0};
  for (iree_host_size_t format = 1;
       format < IREE_ARRAYSIZE(kLoomAie2pBundleLayouts); ++format) {
    const loom_aie2p_bundle_layout_t* layout = &kLoomAie2pBundleLayouts[format];
    if (layout->bit_count / 8u != packet.data_length ||
        !loom_aie2p_encoding_bundle_layout_matches(layout, packet)) {
      continue;
    }
    loom_aie2p_encoding_decode_bundle_layout(
        layout, (loom_aie2p_bundle_format_id_t)format, packet.data, out_bundle);
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no AIE2P bundle format matches %" PRIhsz " bytes",
                          packet.data_length);
}

iree_status_t loom_aie2p_encoding_decode_bundle_prefix(
    iree_const_byte_span_t packet, loom_aie2p_decoded_bundle_t* out_bundle,
    iree_host_size_t* out_packet_length) {
  if (out_bundle == NULL || out_packet_length == NULL || packet.data == NULL ||
      packet.data_length == 0 ||
      packet.data_length > LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P bundle prefix decode requires 1-16 input bytes and valid "
        "output storage");
  }
  *out_bundle = (loom_aie2p_decoded_bundle_t){0};
  *out_packet_length = 0;
  for (iree_host_size_t format = 1;
       format < IREE_ARRAYSIZE(kLoomAie2pBundleLayouts); ++format) {
    const loom_aie2p_bundle_layout_t* layout = &kLoomAie2pBundleLayouts[format];
    if (!loom_aie2p_encoding_bundle_layout_matches(layout, packet)) continue;
    loom_aie2p_encoding_decode_bundle_layout(
        layout, (loom_aie2p_bundle_format_id_t)format, packet.data, out_bundle);
    *out_packet_length = layout->bit_count / 8u;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no AIE2P bundle format matches a %" PRIhsz
                          "-byte prefix",
                          packet.data_length);
}
