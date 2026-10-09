// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#include "iree/base/internal/math.h"

// Allocation move membership and register-part projections are resolved from
// the owned machine tables at generation time.
#include "loom/target/arch/amd/xdna/aie2p/descriptors/register_tables.inl"

#define LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT 16u

uint32_t loom_aie2p_descriptor_register_part_units(uint16_t register_part_id) {
  return register_part_id == LOOM_LOW_REGISTER_PART_NONE
             ? UINT32_MAX
             : kRegisterPartAtomicUnitMasks[register_part_id];
}

uint32_t loom_aie2p_descriptor_select_move(
    loom_aie2p_physical_register_id_t source,
    loom_aie2p_physical_register_id_t destination) {
  const uint32_t candidates =
      kMoveSourceMasks[source] & kMoveDestinationMasks[destination];
  return candidates != 0
             ? kMoveDescriptorOrdinals[iree_math_count_trailing_zeros_u32(
                   candidates)]
             : LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

uint8_t loom_aie2p_descriptor_move_parts(
    loom_aie2p_register_move_t move, loom_aie2p_register_move_t out_parts[2]) {
  const uint8_t source_pair = kMoveScalarPairIndices[move.source];
  const uint8_t destination_pair = kMoveScalarPairIndices[move.destination];
  if (source_pair != 0 && destination_pair != 0) {
    for (uint8_t i = 0; i < 2; ++i) {
      out_parts[i] = (loom_aie2p_register_move_t){
          .source = kMoveScalarPairs[source_pair][i],
          .destination = kMoveScalarPairs[destination_pair][i],
      };
    }
    return 2;
  }
  out_parts[0] = move;
  return 1;
}

static void loom_aie2p_descriptor_append_field(
    loom_aie2p_encoding_field_id_t field_id, uint64_t value,
    iree_host_size_t* field_count,
    loom_aie2p_encoding_field_value_t* field_values) {
  IREE_ASSERT(*field_count < LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT &&
              "verified AIE2P descriptor exceeds field storage");
  field_values[*field_count] = (loom_aie2p_encoding_field_value_t){
      .field_id = field_id,
      .value = value,
  };
  ++*field_count;
}

loom_aie2p_encoded_slot_t loom_aie2p_descriptor_encode(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal,
    const loom_low_allocation_assignment_t* const* operand_assignments,
    const int64_t* immediate_values) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];

  loom_aie2p_encoding_field_value_t
      field_values[LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT];
  iree_host_size_t field_count = 0;
  for (uint16_t i = 0; i < descriptor->encoding_field_value_count; ++i) {
    const loom_low_encoding_field_value_t* field =
        &descriptor_set
             ->encoding_field_values[descriptor->encoding_field_value_start +
                                     i];
    loom_aie2p_descriptor_append_field(field->encoding_field_id, field->value,
                                       &field_count, field_values);
  }

  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (operand->encoding_field_id == 0) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment = operand_assignments[i];
    const uint8_t encoded_register =
        loom_aie2p_machine_adapt_allocated_register(
            operand->encoding_adapter_id,
            (loom_aie2p_physical_register_id_t)assignment->location_base);
    loom_aie2p_descriptor_append_field(operand->encoding_field_id,
                                       encoded_register, &field_count,
                                       field_values);
  }

  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    if (immediate->encoding_field_id == 0) {
      continue;
    }
    const uint64_t encoded_immediate =
        loom_aie2p_machine_encode_verified_immediate(immediate->encoding_id,
                                                     immediate_values[i]);
    loom_aie2p_descriptor_append_field(immediate->encoding_field_id,
                                       encoded_immediate, &field_count,
                                       field_values);
  }

  return loom_aie2p_encoding_pack_verified_instruction(
      descriptor->encoding_id, field_values, field_count);
}
