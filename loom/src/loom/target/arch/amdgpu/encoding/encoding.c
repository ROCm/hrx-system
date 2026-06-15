// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/encoding/encoding.h"

#include <inttypes.h>

#include "loom/target/arch/amdgpu/target_info.h"

#define LOOM_AMDGPU_ENCODING_TABLE_DECL(descriptor_set_ordinal, table_fn) \
  const loom_amdgpu_encoding_table_t* table_fn(void);
#include "loom/target/arch/amdgpu/encoding/encoding_tables.inl"
#undef LOOM_AMDGPU_ENCODING_TABLE_DECL

static uint64_t loom_amdgpu_encoding_u64_mask(uint8_t bit_count) {
  if (bit_count == 0) {
    return 0;
  }
  if (bit_count >= 64) {
    return UINT64_MAX;
  }
  return (UINT64_C(1) << bit_count) - 1;
}

static uint32_t loom_amdgpu_encoding_u32_mask(uint8_t bit_count) {
  if (bit_count == 0) {
    return 0;
  }
  if (bit_count >= 32) {
    return UINT32_MAX;
  }
  return (UINT32_C(1) << bit_count) - 1;
}

static const loom_amdgpu_encoding_format_layout_t*
loom_amdgpu_encoding_find_format(const loom_amdgpu_encoding_table_t* table,
                                 uint16_t encoding_format) {
  uint32_t low = 0;
  uint32_t high = table->format_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_amdgpu_encoding_format_layout_t* format = &table->formats[mid];
    if (format->format_id == encoding_format) {
      return format;
    }
    if (format->format_id < encoding_format) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return NULL;
}

static const loom_amdgpu_encoding_field_layout_t*
loom_amdgpu_encoding_find_field(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_format_layout_t* format, uint16_t field_id) {
  for (uint16_t i = 0; i < format->field_count; ++i) {
    const loom_amdgpu_encoding_field_layout_t* field =
        &table->fields[format->field_start + i];
    if (field->field_id == field_id) {
      return field;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_encoding_write_bits(
    loom_amdgpu_encoding_packet_t* packet, uint8_t bit_offset,
    uint8_t bit_count, uint64_t value) {
  uint8_t remaining_bits = bit_count;
  uint8_t current_bit_offset = bit_offset;
  uint64_t current_value = value;
  while (remaining_bits > 0) {
    const uint8_t word_index = current_bit_offset / 32;
    if (word_index >= packet->word_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU encoding bit range starting at bit %u exceeds %u-word packet",
          bit_offset, packet->word_count);
    }
    const uint8_t word_bit_offset = current_bit_offset % 32;
    const uint8_t chunk_bit_count =
        (uint8_t)iree_min(remaining_bits, (uint8_t)(32 - word_bit_offset));
    const uint32_t chunk_mask = loom_amdgpu_encoding_u32_mask(chunk_bit_count);
    packet->words[word_index] =
        (packet->words[word_index] & ~(chunk_mask << word_bit_offset)) |
        ((uint32_t)(current_value & chunk_mask) << word_bit_offset);
    current_value >>= chunk_bit_count;
    current_bit_offset += chunk_bit_count;
    remaining_bits -= chunk_bit_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encoding_set_field(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_format_layout_t* format,
    loom_amdgpu_encoding_packet_t* packet, uint16_t field_id, uint64_t value) {
  const loom_amdgpu_encoding_field_layout_t* field =
      loom_amdgpu_encoding_find_field(table, format, field_id);
  if (field == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU encoding format %" PRIu16
                            " has no field id %" PRIu16,
                            format->format_id, field_id);
  }
  if (field->value_bit_count < 64 && (value >> field->value_bit_count) != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU encoding field %" PRIu16 " value 0x%" PRIx64
                            " does not fit %u source bits",
                            field_id, value, field->value_bit_count);
  }
  for (uint8_t i = 0; i < field->range_count; ++i) {
    const loom_amdgpu_encoding_bit_range_t* bit_range =
        &table->bit_ranges[field->range_start + i];
    if (bit_range->padding_bit_count != 0) {
      const uint64_t padding_mask =
          loom_amdgpu_encoding_u64_mask(bit_range->padding_bit_count);
      const uint64_t padding_value =
          (value >> bit_range->source_bit_offset) & padding_mask;
      if (padding_value != bit_range->padding_value) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU encoding field %" PRIu16
                                " requires padding value 0x%" PRIx64
                                " at source bit %u but found 0x%" PRIx64,
                                field_id, bit_range->padding_value,
                                bit_range->source_bit_offset, padding_value);
      }
    }
    const uint8_t value_bit_offset =
        bit_range->source_bit_offset + bit_range->padding_bit_count;
    const uint64_t range_value =
        (value >> value_bit_offset) &
        loom_amdgpu_encoding_u64_mask(bit_range->bit_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_write_bits(
        packet, bit_range->bit_offset, bit_range->bit_count, range_value));
  }
  return iree_ok_status();
}

iree_string_view_t loom_amdgpu_encoding_format_name(uint16_t encoding_format) {
  switch (encoding_format) {
    case LOOM_AMDGPU_ENCODING_FORMAT_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP1:
      return IREE_SV("sop1");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP1_LITERAL:
      return IREE_SV("sop1_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP2:
      return IREE_SV("sop2");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP2_LITERAL:
      return IREE_SV("sop2_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPP:
      return IREE_SV("sopp");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPC:
      return IREE_SV("sopc");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPC_LITERAL:
      return IREE_SV("sopc_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPK:
      return IREE_SV("sopk");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPK_LITERAL:
      return IREE_SV("sopk_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2:
      return IREE_SV("vop2");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL:
      return IREE_SV("vop2_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3:
      return IREE_SV("vop3");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P:
      return IREE_SV("vop3p");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3PX2:
      return IREE_SV("vop3px2");
    case LOOM_AMDGPU_ENCODING_FORMAT_SMEM:
      return IREE_SV("smem");
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
      return IREE_SV("mubuf");
    case LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER:
      return IREE_SV("vbuffer");
    case LOOM_AMDGPU_ENCODING_FORMAT_DS:
      return IREE_SV("ds");
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT:
      return IREE_SV("flat");
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL:
      return IREE_SV("flat_glbl");
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL:
      return IREE_SV("flat_global");
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_SCRATCH:
      return IREE_SV("flat_scratch");
    case LOOM_AMDGPU_ENCODING_FORMAT_VDS:
      return IREE_SV("vds");
    case LOOM_AMDGPU_ENCODING_FORMAT_VFLAT:
      return IREE_SV("vflat");
    case LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL:
      return IREE_SV("vglobal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VSCRATCH:
      return IREE_SV("vscratch");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1:
      return IREE_SV("vop1");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL:
      return IREE_SV("vop1_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP:
      return IREE_SV("vop1_dpp");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP16:
      return IREE_SV("vop1_dpp16");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_SDWA:
      return IREE_SV("vop1_sdwa");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL:
      return IREE_SV("vop3p_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA:
      return IREE_SV("vop3p_mfma");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3_LITERAL:
      return IREE_SV("vop3_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3_SDST:
      return IREE_SV("vop3_sdst");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3_SDST_LITERAL:
      return IREE_SV("vop3_sdst_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY:
      return IREE_SV("vopdxy");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY_LITERAL:
      return IREE_SV("vopdxy_literal");
    default:
      return IREE_SV("unknown");
  }
}

bool loom_amdgpu_encoding_table_has_format(
    const loom_amdgpu_encoding_table_t* table, uint16_t encoding_format) {
  if (table == NULL) {
    return false;
  }
  return loom_amdgpu_encoding_find_format(table, encoding_format) != NULL;
}

bool loom_amdgpu_encoding_field_uses_unified_source(uint16_t field_id) {
  switch (field_id) {
    case LOOM_AMDGPU_ENCODING_FIELD_SRC0:
    case LOOM_AMDGPU_ENCODING_FIELD_SRC1:
    case LOOM_AMDGPU_ENCODING_FIELD_SRC2:
    case LOOM_AMDGPU_ENCODING_FIELD_SCALE_SRC0:
    case LOOM_AMDGPU_ENCODING_FIELD_SCALE_SRC1:
      return true;
    default:
      return false;
  }
}

bool loom_amdgpu_encoding_field_is_src0(uint16_t field_id) {
  return field_id == LOOM_AMDGPU_ENCODING_FIELD_SRC0;
}

bool loom_amdgpu_encoding_field_is_dst_sel(uint16_t field_id) {
  return field_id == LOOM_AMDGPU_ENCODING_FIELD_DST_SEL;
}

bool loom_amdgpu_encoding_field_is_literal(uint16_t field_id) {
  return field_id == LOOM_AMDGPU_ENCODING_FIELD_LITERAL;
}

bool loom_amdgpu_sdwa_dst_selector_writes_subdword(uint32_t selector) {
  enum {
    LOOM_AMDGPU_SDWA_SELECTOR_DWORD = 6,
  };
  return selector != LOOM_AMDGPU_SDWA_SELECTOR_DWORD;
}

uint8_t loom_amdgpu_vgpr_msb_slot_shift(loom_amdgpu_vgpr_msb_slot_t slot) {
  static const uint8_t kSlotShifts[] = {
      [LOOM_AMDGPU_VGPR_MSB_SLOT_NONE] = 0,
      [LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0] = 0,
      [LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1] = 2,
      [LOOM_AMDGPU_VGPR_MSB_SLOT_SRC2] = 4,
      [LOOM_AMDGPU_VGPR_MSB_SLOT_DST] = 6,
  };
  if ((uint32_t)slot >= IREE_ARRAYSIZE(kSlotShifts)) {
    return 0;
  }
  return kSlotShifts[slot];
}

static bool loom_amdgpu_encoding_format_uses_vop_vgpr_msb_slots(
    uint16_t encoding_format_id) {
  switch (encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP16:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_SDWA:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3_LITERAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3PX2:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3_SDST:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3_SDST_LITERAL:
      return true;
    default:
      return false;
  }
}

loom_amdgpu_vgpr_msb_slot_t loom_amdgpu_encoding_vgpr_msb_slot(
    uint16_t encoding_format_id, uint16_t encoding_field_id) {
  if (loom_amdgpu_encoding_format_uses_vop_vgpr_msb_slots(encoding_format_id)) {
    switch (encoding_field_id) {
      case LOOM_AMDGPU_ENCODING_FIELD_SRC0:
      case LOOM_AMDGPU_ENCODING_FIELD_VSRC0:
        return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0;
      case LOOM_AMDGPU_ENCODING_FIELD_SRC1:
      case LOOM_AMDGPU_ENCODING_FIELD_VSRC1:
        return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1;
      case LOOM_AMDGPU_ENCODING_FIELD_SRC2:
      case LOOM_AMDGPU_ENCODING_FIELD_VSRC2:
        return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC2;
      case LOOM_AMDGPU_ENCODING_FIELD_VDST:
        return LOOM_AMDGPU_VGPR_MSB_SLOT_DST;
      default:
        return LOOM_AMDGPU_VGPR_MSB_SLOT_NONE;
    }
  }

  switch (encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_DS:
    case LOOM_AMDGPU_ENCODING_FORMAT_VDS:
      switch (encoding_field_id) {
        case LOOM_AMDGPU_ENCODING_FIELD_ADDR:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0;
        case LOOM_AMDGPU_ENCODING_FIELD_DATA0:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1;
        case LOOM_AMDGPU_ENCODING_FIELD_DATA1:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC2;
        case LOOM_AMDGPU_ENCODING_FIELD_VDST:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_DST;
        default:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_NONE;
      }
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_VFLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VSCRATCH:
      switch (encoding_field_id) {
        case LOOM_AMDGPU_ENCODING_FIELD_ADDR:
        case LOOM_AMDGPU_ENCODING_FIELD_VADDR:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0;
        case LOOM_AMDGPU_ENCODING_FIELD_DATA:
        case LOOM_AMDGPU_ENCODING_FIELD_VDATA:
        case LOOM_AMDGPU_ENCODING_FIELD_VSRC:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1;
        case LOOM_AMDGPU_ENCODING_FIELD_VDST:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_DST;
        default:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_NONE;
      }
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
    case LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER:
      switch (encoding_field_id) {
        case LOOM_AMDGPU_ENCODING_FIELD_VADDR:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0;
        case LOOM_AMDGPU_ENCODING_FIELD_VDATA:
        case LOOM_AMDGPU_ENCODING_FIELD_VDST:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_DST;
        default:
          return LOOM_AMDGPU_VGPR_MSB_SLOT_NONE;
      }
    default:
      return LOOM_AMDGPU_VGPR_MSB_SLOT_NONE;
  }
}

bool loom_amdgpu_encoding_inline_u32_source(
    const loom_amdgpu_encoding_table_t* table, uint32_t value,
    uint16_t* out_source) {
  *out_source = 0;
  if (table == NULL) {
    return false;
  }
  if (value < table->scalar_inline_u32_count) {
    *out_source = (uint16_t)(table->scalar_inline_u32_zero + value);
    return true;
  }
  return false;
}

bool loom_amdgpu_encoding_inline_f32_source(
    const loom_amdgpu_encoding_table_t* table, uint32_t bit_pattern,
    uint16_t* out_source) {
  *out_source = 0;
  if (table == NULL) {
    return false;
  }
  uint16_t low = 0;
  uint16_t high = table->inline_f32_source_count;
  while (low < high) {
    const uint16_t mid = (uint16_t)(low + (high - low) / 2);
    const loom_amdgpu_encoding_inline_f32_source_t* source =
        &table->inline_f32_sources[mid];
    if (source->bit_pattern == bit_pattern) {
      *out_source = source->source;
      return true;
    }
    if (source->bit_pattern < bit_pattern) {
      low = (uint16_t)(mid + 1);
    } else {
      high = mid;
    }
  }
  return false;
}

const loom_amdgpu_encoding_table_t*
loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
    uint16_t descriptor_set_ordinal) {
  const loom_amdgpu_encoding_table_t* const
      tables[LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT] = {
#define LOOM_AMDGPU_ENCODING_TABLE(descriptor_set_ordinal, table_fn) \
  [descriptor_set_ordinal] = table_fn(),
#include "loom/target/arch/amdgpu/encoding/encoding_tables.inl"
#undef LOOM_AMDGPU_ENCODING_TABLE
      };
  if (descriptor_set_ordinal >= IREE_ARRAYSIZE(tables)) {
    return NULL;
  }
  return tables[descriptor_set_ordinal];
}

iree_status_t loom_amdgpu_encoding_pack(
    const loom_amdgpu_encoding_table_t* table, uint16_t encoding_format,
    uint16_t opcode, const loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count,
    loom_amdgpu_encoding_packet_t* out_packet) {
  *out_packet = (loom_amdgpu_encoding_packet_t){0};

  const loom_amdgpu_encoding_format_layout_t* format =
      loom_amdgpu_encoding_find_format(table, encoding_format);
  if (format == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU encoding table '%.*s' has no format id %" PRIu16,
        (int)table->descriptor_set_key.size, table->descriptor_set_key.data,
        encoding_format);
  }
  if (format->word_count > LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU encoding format %" PRIu16
                            " has unsupported word count %" PRIu16,
                            encoding_format, format->word_count);
  }

  out_packet->bit_count = format->bit_count;
  out_packet->word_count = format->word_count;
  for (uint16_t i = 0; i < format->word_count; ++i) {
    out_packet->words[i] = format->identifier_words[i];
  }

  for (iree_host_size_t i = 0; i < field_value_count; ++i) {
    if (opcode != LOOM_AMDGPU_ENCODING_OPCODE_NONE &&
        field_values[i].field_id == LOOM_AMDGPU_ENCODING_FIELD_OP) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU encoding callers must not pass OP as a field value when a "
          "separate opcode is supplied");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (field_values[j].field_id == field_values[i].field_id) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU encoding repeats field id %" PRIu16,
                                field_values[i].field_id);
      }
    }
  }

  if (opcode != LOOM_AMDGPU_ENCODING_OPCODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_set_field(
        table, format, out_packet, LOOM_AMDGPU_ENCODING_FIELD_OP, opcode));
  }
  for (iree_host_size_t i = 0; i < field_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_set_field(
        table, format, out_packet, field_values[i].field_id,
        field_values[i].value));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encoding_pack_u32_source(
    const loom_amdgpu_encoding_table_t* table, uint16_t inline_format,
    uint16_t literal_format, uint16_t opcode, uint16_t destination_field_id,
    uint64_t destination_value, uint16_t source_field_id, uint32_t imm32,
    const loom_amdgpu_encoding_field_value_t* extra_field_values,
    iree_host_size_t extra_field_value_count,
    loom_amdgpu_encoding_packet_t* out_packet) {
  enum {
    LOOM_AMDGPU_ENCODING_U32_SOURCE_FIELD_CAPACITY = 4,
  };
  IREE_ASSERT(extra_field_value_count + 3 <=
              LOOM_AMDGPU_ENCODING_U32_SOURCE_FIELD_CAPACITY);
  loom_amdgpu_encoding_field_value_t
      field_values[LOOM_AMDGPU_ENCODING_U32_SOURCE_FIELD_CAPACITY] = {
          {
              .field_id = destination_field_id,
              .value = destination_value,
          },
          {
              .field_id = source_field_id,
              .value = 0,
          },
      };
  iree_host_size_t field_value_count = 2;
  for (iree_host_size_t i = 0; i < extra_field_value_count; ++i) {
    field_values[field_value_count++] = extra_field_values[i];
  }

  uint16_t source = 0;
  if (loom_amdgpu_encoding_inline_u32_source(table, imm32, &source)) {
    field_values[1].value = source;
    return loom_amdgpu_encoding_pack(table, inline_format, opcode, field_values,
                                     field_value_count, out_packet);
  }

  field_values[1].value = table->source_literal;
  field_values[field_value_count++] = (loom_amdgpu_encoding_field_value_t){
      .field_id = LOOM_AMDGPU_ENCODING_FIELD_LITERAL,
      .value = imm32,
  };
  return loom_amdgpu_encoding_pack(table, literal_format, opcode, field_values,
                                   field_value_count, out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_sopp_simm16(
    const loom_amdgpu_encoding_table_t* table, uint16_t opcode,
    uint16_t immediate, loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU SOPP immediate encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SIMM16,
          .value = immediate,
      },
  };
  return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_SOPP,
                                   opcode, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_sgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint16_t ssrc0,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU s_mov_b32 encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SDST,
          .value = sdst,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SSRC0,
          .value = ssrc0,
      },
  };
  return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_SOP1,
                                   table->s_mov_b32_opcode, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU s_mov_b32 encoding requires an encoding table");
  }
  return loom_amdgpu_encoding_pack_u32_source(
      table, LOOM_AMDGPU_ENCODING_FORMAT_SOP1,
      LOOM_AMDGPU_ENCODING_FORMAT_SOP1_LITERAL, table->s_mov_b32_opcode,
      LOOM_AMDGPU_ENCODING_FIELD_SDST, sdst, LOOM_AMDGPU_ENCODING_FIELD_SSRC0,
      imm32, /*extra_field_values=*/NULL, /*extra_field_value_count=*/0,
      out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_vgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint16_t vsrc0,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU v_mov_b32 encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VDST,
          .value = vdst,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SRC0,
          .value = (uint64_t)table->vector_source_vgpr0 + vsrc0,
      },
  };
  return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1,
                                   table->v_mov_b32_opcode, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU v_mov_b32 encoding requires an encoding table");
  }
  return loom_amdgpu_encoding_pack_u32_source(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1,
      LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL, table->v_mov_b32_opcode,
      LOOM_AMDGPU_ENCODING_FIELD_VDST, vdst, LOOM_AMDGPU_ENCODING_FIELD_SRC0,
      imm32, /*extra_field_values=*/NULL, /*extra_field_value_count=*/0,
      out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_vop2_u32_vgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t opcode, uint16_t vdst,
    uint32_t imm32, uint16_t vsrc1, loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOP2 encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VSRC1,
          .value = vsrc1,
      },
  };
  return loom_amdgpu_encoding_pack_u32_source(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP2,
      LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL, opcode,
      LOOM_AMDGPU_ENCODING_FIELD_VDST, vdst, LOOM_AMDGPU_ENCODING_FIELD_SRC0,
      imm32, field_values, IREE_ARRAYSIZE(field_values), out_packet);
}

static iree_status_t loom_amdgpu_encoding_pack_vopdxy_format(
    const loom_amdgpu_encoding_table_t* table, uint16_t encoding_format,
    const loom_amdgpu_encoding_vopdxy_fields_t* fields,
    const uint32_t* literal_u32, loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL || fields == NULL || out_packet == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPDXY encoding requires a table, fields, "
                            "and an output packet");
  }
  if (((fields->vdst_x ^ fields->vdst_y) & 1u) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPDXY destination registers must have "
                            "opposite parity");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SRCX0,
          .value = fields->src0_x,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VSRCX1,
          .value = fields->vsrc1_x,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_OPY,
          .value = fields->op_y,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_OPX,
          .value = fields->op_x,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SRCY0,
          .value = fields->src0_y,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VSRCY1,
          .value = fields->vsrc1_y,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VDSTY,
          // VOPD stores VDSTY[7:1]; the low bit is implied by VDSTX parity.
          .value = fields->vdst_y & ~UINT16_C(1),
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VDSTX,
          .value = fields->vdst_x,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_LITERAL,
          .value = literal_u32 ? *literal_u32 : 0,
      },
  };
  const iree_host_size_t field_value_count =
      literal_u32 ? IREE_ARRAYSIZE(field_values)
                  : IREE_ARRAYSIZE(field_values) - 1;
  return loom_amdgpu_encoding_pack(table, encoding_format,
                                   LOOM_AMDGPU_ENCODING_OPCODE_NONE,
                                   field_values, field_value_count, out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_vopdxy(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_vopdxy_fields_t* fields,
    loom_amdgpu_encoding_packet_t* out_packet) {
  return loom_amdgpu_encoding_pack_vopdxy_format(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY, fields, /*literal_u32=*/NULL,
      out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_vopdxy_literal(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_vopdxy_fields_t* fields, uint32_t literal_u32,
    loom_amdgpu_encoding_packet_t* out_packet) {
  return loom_amdgpu_encoding_pack_vopdxy_format(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY_LITERAL, fields, &literal_u32,
      out_packet);
}
