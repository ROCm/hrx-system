// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

#define LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT 16u

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
        &descriptor_set->operands[descriptor->operand_start + i];
    if (operand->encoding_field_id == 0) continue;
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
    if (immediate->encoding_field_id == 0) continue;
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
