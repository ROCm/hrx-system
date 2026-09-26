// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

static iree_status_t loom_aie2p_native_relocation_add_signed(
    uint64_t value, int64_t addend, uint64_t* out_value) {
  if (addend >= 0) {
    if (!iree_checked_add_u64(value, (uint64_t)addend, out_value)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P relocation target overflows");
    }
    return iree_ok_status();
  }
  const uint64_t magnitude = (uint64_t)(-(addend + 1)) + 1u;
  if (magnitude > value) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P relocation target underflows");
  }
  *out_value = value - magnitude;
  return iree_ok_status();
}

typedef struct loom_aie2p_native_relocation_immediate_t {
  // Instruction field receiving the encoded target value.
  loom_aie2p_encoding_field_id_t field;
  // Machine immediate domain encoding the semantic target value.
  loom_aie2p_immediate_id_t immediate;
  // Number of encoded fields in |instruction|.
  uint8_t field_count;
} loom_aie2p_native_relocation_immediate_t;

static loom_aie2p_machine_form_info_t loom_aie2p_native_relocation_machine_form(
    loom_aie2p_instruction_id_t instruction) {
  loom_aie2p_machine_form_info_t form_info;
  const bool found = loom_aie2p_machine_query_form(
      (loom_aie2p_machine_form_id_t)instruction, &form_info);
  if (!found) {
    IREE_ASSERT_UNREACHABLE(
        "encoding and machine instruction IDs must remain aligned");
    IREE_BUILTIN_UNREACHABLE();
  }
  return form_info;
}

static loom_aie2p_native_relocation_immediate_t
loom_aie2p_native_relocation_immediate(
    loom_aie2p_instruction_id_t instruction) {
  loom_aie2p_instruction_info_t instruction_info;
  const bool instruction_found = loom_aie2p_encoding_query_instruction_info(
      instruction, &instruction_info);
  if (!instruction_found) {
    IREE_ASSERT_UNREACHABLE("relocated instruction ID must be valid");
    IREE_BUILTIN_UNREACHABLE();
  }
  const loom_aie2p_machine_form_info_t form_info =
      loom_aie2p_native_relocation_machine_form(instruction);
  loom_aie2p_native_relocation_immediate_t result = {
      .field_count = instruction_info.field_count,
  };
  const uint8_t operand_count = form_info.output_count + form_info.input_count;
  for (uint8_t i = 0; i < operand_count; ++i) {
    loom_aie2p_machine_operand_info_t operand_info;
    const bool operand_found = loom_aie2p_machine_query_form_operand(
        (loom_aie2p_machine_form_id_t)instruction, i, &operand_info);
    if (!operand_found) {
      IREE_ASSERT_UNREACHABLE("machine-form operand must be valid");
      IREE_BUILTIN_UNREACHABLE();
    }
    if (operand_info.kind != LOOM_AIE2P_MACHINE_OPERAND_KIND_IMMEDIATE) {
      continue;
    }
    if (result.immediate != LOOM_AIE2P_IMMEDIATE_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE(
          "relocated instruction must have exactly one immediate");
      IREE_BUILTIN_UNREACHABLE();
    }
    result.immediate = (loom_aie2p_immediate_id_t)operand_info.type_id;
    result.field = loom_aie2p_encoding_find_field(operand_info.name);
  }
  if (result.immediate == LOOM_AIE2P_IMMEDIATE_ID_INVALID ||
      result.field == LOOM_AIE2P_ENCODING_FIELD_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE(
        "relocated instruction immediate must have an encoded field");
    IREE_BUILTIN_UNREACHABLE();
  }
  return result;
}

static bool loom_aie2p_native_relocation_is_core_branch(
    loom_aie2p_instruction_id_t instruction) {
  const loom_aie2p_machine_form_info_t form_info =
      loom_aie2p_native_relocation_machine_form(instruction);
  switch (form_info.control_flow_kind) {
    case LOOM_AIE2P_CONTROL_FLOW_BRANCH_CONDITIONAL_NONZERO:
    case LOOM_AIE2P_CONTROL_FLOW_BRANCH_CONDITIONAL_ZERO:
    case LOOM_AIE2P_CONTROL_FLOW_BRANCH_DIRECT:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_aie2p_native_relocation_patch_core_branch(
    uint64_t target_address, uint64_t fixup_offset,
    loom_native_elf_section_t* section) {
  const loom_aie2p_slot_t branch_slot = LOOM_AIE2P_SLOT_LNG;
  const loom_aie2p_bundle_format_id_t branch_format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&branch_slot, 1);
  IREE_ASSERT(branch_format != LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID &&
              "generated AIE2P tables must contain the LNG bundle");
  loom_aie2p_bundle_format_info_t branch_format_info;
  const bool found = loom_aie2p_encoding_query_bundle_format_info(
      branch_format, &branch_format_info);
  IREE_ASSERT(found && branch_format_info.bit_count % 8 == 0);
  const iree_host_size_t packet_length = branch_format_info.bit_count / 8u;
  if (fixup_offset > IREE_HOST_SIZE_MAX ||
      (iree_host_size_t)fixup_offset > section->contents.data_length ||
      packet_length >
          section->contents.data_length - (iree_host_size_t)fixup_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P branch fixup lies outside its section");
  }

  uint8_t* packet_bytes =
      (uint8_t*)section->contents.data + (iree_host_size_t)fixup_offset;
  loom_aie2p_decoded_bundle_t decoded_bundle;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_decode_bundle(
      iree_make_const_byte_span(packet_bytes, packet_length), &decoded_bundle));
  if (decoded_bundle.format != branch_format ||
      decoded_bundle.slot_count != 1 ||
      decoded_bundle.slots[0].slot != branch_slot) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P core branch relocation requires one standalone LNG bundle");
  }

  const iree_host_size_t instruction_count =
      loom_aie2p_encoding_instruction_count();
  loom_aie2p_instruction_id_t* instruction_candidates =
      (loom_aie2p_instruction_id_t*)iree_alloca(
          instruction_count * sizeof(*instruction_candidates));
  const iree_host_size_t candidate_count =
      loom_aie2p_encoding_query_instruction_candidates(
          branch_slot, decoded_bundle.slots[0].value, instruction_count,
          instruction_candidates);
  IREE_ASSERT_LE(candidate_count, instruction_count);
  loom_aie2p_instruction_id_t branch_instruction =
      LOOM_AIE2P_INSTRUCTION_ID_INVALID;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    const loom_aie2p_instruction_id_t candidate = instruction_candidates[i];
    if (!loom_aie2p_native_relocation_is_core_branch(candidate)) {
      continue;
    }
    if (branch_instruction != LOOM_AIE2P_INSTRUCTION_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P branch relocation matches multiple instruction forms");
    }
    branch_instruction = candidate;
  }
  if (branch_instruction == LOOM_AIE2P_INSTRUCTION_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P branch relocation does not reference J, JZ, or JNZ");
  }

  const loom_aie2p_native_relocation_immediate_t target_immediate =
      loom_aie2p_native_relocation_immediate(branch_instruction);
  const iree_host_size_t field_capacity = target_immediate.field_count;
  loom_aie2p_encoding_field_value_t* field_values =
      (loom_aie2p_encoding_field_value_t*)iree_alloca(field_capacity *
                                                      sizeof(*field_values));
  iree_host_size_t field_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_unpack_instruction(
      branch_instruction, decoded_bundle.slots[0].value, field_capacity,
      field_values, &field_count));
  if (target_address > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P core branch target exceeds signed range");
  }
  uint64_t encoded_target = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_machine_encode_immediate(
      target_immediate.immediate, (int64_t)target_address, &encoded_target));
  bool target_field_found = false;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (field_values[i].field_id != target_immediate.field) {
      continue;
    }
    field_values[i].value = encoded_target;
    target_field_found = true;
  }
  IREE_ASSERT(target_field_found &&
              "generated branch form must encode cpmaddr");
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      branch_instruction, field_values, field_count,
      &decoded_bundle.slots[0].value));

  loom_aie2p_encoding_packet_t encoded_bundle;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_bundle(
      decoded_bundle.format, decoded_bundle.slots, decoded_bundle.slot_count,
      &encoded_bundle));
  IREE_ASSERT_EQ(encoded_bundle.data_length, packet_length);
  memcpy(packet_bytes, encoded_bundle.data, packet_length);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_relocation_patch_local_address(
    uint64_t target_address, uint64_t fixup_offset,
    loom_native_elf_section_t* section) {
  if (fixup_offset > IREE_HOST_SIZE_MAX ||
      (iree_host_size_t)fixup_offset >= section->contents.data_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P local-address fixup lies outside its section");
  }

  uint8_t* packet_bytes =
      (uint8_t*)section->contents.data + (iree_host_size_t)fixup_offset;
  const iree_host_size_t remaining_length =
      section->contents.data_length - (iree_host_size_t)fixup_offset;
  const iree_host_size_t decode_length = iree_min(
      remaining_length, (iree_host_size_t)LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE);
  loom_aie2p_decoded_bundle_t decoded_bundle;
  iree_host_size_t packet_length = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_decode_bundle_prefix(
      iree_make_const_byte_span(packet_bytes, decode_length), &decoded_bundle,
      &packet_length));

  const iree_host_size_t instruction_count =
      loom_aie2p_encoding_instruction_count();
  loom_aie2p_instruction_id_t* instruction_candidates =
      (loom_aie2p_instruction_id_t*)iree_alloca(
          instruction_count * sizeof(*instruction_candidates));
  const loom_aie2p_instruction_id_t movxm_instruction =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVXM"));
  IREE_ASSERT(movxm_instruction != LOOM_AIE2P_INSTRUCTION_ID_INVALID &&
              "generated AIE2P tables must contain MOVXM");
  loom_aie2p_instruction_id_t address_instruction =
      LOOM_AIE2P_INSTRUCTION_ID_INVALID;
  loom_aie2p_encoded_slot_t* address_slot = NULL;
  for (uint8_t i = 0; i < decoded_bundle.slot_count; ++i) {
    loom_aie2p_encoded_slot_t* slot = &decoded_bundle.slots[i];
    const iree_host_size_t candidate_count =
        loom_aie2p_encoding_query_instruction_candidates(
            slot->slot, slot->value, instruction_count, instruction_candidates);
    IREE_ASSERT_LE(candidate_count, instruction_count);
    for (iree_host_size_t j = 0; j < candidate_count; ++j) {
      const loom_aie2p_instruction_id_t candidate = instruction_candidates[j];
      if (candidate != movxm_instruction) {
        continue;
      }
      if (address_slot != NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P local-address relocation matches multiple MOVXM slots");
      }
      address_slot = slot;
      address_instruction = candidate;
    }
  }
  if (address_slot == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P local-address relocation does not reference MOVXM");
  }

  const loom_aie2p_native_relocation_immediate_t address_immediate =
      loom_aie2p_native_relocation_immediate(address_instruction);
  const iree_host_size_t field_capacity = address_immediate.field_count;
  loom_aie2p_encoding_field_value_t* field_values =
      (loom_aie2p_encoding_field_value_t*)iree_alloca(field_capacity *
                                                      sizeof(*field_values));
  iree_host_size_t field_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_unpack_instruction(
      address_instruction, address_slot->value, field_capacity, field_values,
      &field_count));
  if (target_address > INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P local-memory address exceeds signed MOVXM range");
  }
  uint64_t encoded_target = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_machine_encode_immediate(
      address_immediate.immediate, (int64_t)target_address, &encoded_target));
  bool target_field_found = false;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (field_values[i].field_id != address_immediate.field) {
      continue;
    }
    field_values[i].value = encoded_target;
    target_field_found = true;
  }
  IREE_ASSERT(target_field_found &&
              "generated local-address form must encode i");
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      address_instruction, field_values, field_count, &address_slot->value));

  loom_aie2p_encoding_packet_t encoded_bundle;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_bundle(
      decoded_bundle.format, decoded_bundle.slots, decoded_bundle.slot_count,
      &encoded_bundle));
  IREE_ASSERT_EQ(encoded_bundle.data_length, packet_length);
  memcpy(packet_bytes, encoded_bundle.data, packet_length);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_object_apply_fixups(
    const loom_native_object_contribution_t* object,
    loom_native_section_contribution_assembly_t* assembly,
    iree_arena_allocator_t* scratch_arena) {
  if (object->fixup_count == 0) {
    return iree_ok_status();
  }

  loom_native_object_symbol_layout_t* symbol_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, object->symbol_count, sizeof(*symbol_layouts),
      (void**)&symbol_layouts));
  IREE_RETURN_IF_ERROR(loom_native_object_resolve_symbol_layouts(
      object->symbols, object->symbol_count, assembly->contribution_layouts,
      assembly->contribution_layout_count, symbol_layouts));
  loom_native_object_fixup_layout_t* fixup_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, object->fixup_count, sizeof(*fixup_layouts),
      (void**)&fixup_layouts));
  IREE_RETURN_IF_ERROR(loom_native_object_resolve_fixup_layouts(
      object->fixups, object->fixup_count, object->symbol_count,
      assembly->contribution_layouts, assembly->contribution_layout_count,
      fixup_layouts));

  for (iree_host_size_t i = 0; i < object->fixup_count; ++i) {
    const loom_native_object_fixup_t* fixup = &object->fixups[i];
    const loom_native_object_fixup_layout_t* fixup_layout = &fixup_layouts[i];
    const loom_native_object_symbol_layout_t* target_layout =
        &symbol_layouts[fixup->target_symbol_index];
    if (fixup_layout->section_index >= assembly->section_count ||
        target_layout->section_index >= assembly->section_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P fixup section layout is invalid");
    }
    loom_native_elf_section_t* source_section =
        &assembly->sections[fixup_layout->section_index];
    loom_native_elf_section_t* target_section =
        &assembly->sections[target_layout->section_index];
    uint64_t target_offset = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_relocation_add_signed(
        target_layout->section_offset, fixup->addend, &target_offset));
    uint64_t target_address = 0;
    if (!iree_checked_add_u64(target_section->address, target_offset,
                              &target_address)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P relocation address overflows");
    }
    switch ((loom_aie2p_native_relocation_kind_t)fixup->relocation_kind) {
      case LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE: {
        if (fixup_layout->section_index != target_layout->section_index) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AIE2P core branch target must share its executable section");
        }
        if (source_section->type != LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS ||
            (source_section->flags &
             (LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR)) !=
                (LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                 LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AIE2P core branch fixup requires executable PROGBITS");
        }
        if (target_offset >= target_section->contents.data_length) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "AIE2P core branch target lies outside its code section");
        }
        IREE_RETURN_IF_ERROR(loom_aie2p_native_relocation_patch_core_branch(
            target_address, fixup_layout->section_offset, source_section));
        break;
      }
      case LOOM_AIE2P_NATIVE_RELOCATION_KIND_LOCAL_ADDRESS_ABSOLUTE:
        if (source_section->type != LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS ||
            (source_section->flags &
             (LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR)) !=
                (LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                 LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AIE2P local-address fixup requires executable PROGBITS");
        }
        if ((target_section->type != LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS &&
             target_section->type != LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS) ||
            (target_section->flags & (LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                                      LOOM_NATIVE_ELF_SECTION_FLAG_WRITE)) !=
                (LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                 LOOM_NATIVE_ELF_SECTION_FLAG_WRITE) ||
            iree_any_bit_set(target_section->flags,
                             LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AIE2P local address must target allocated writable data");
        }
        if (target_offset >=
            loom_native_elf_section_byte_length(target_section)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "AIE2P local address lies outside its storage domain");
        }
        IREE_RETURN_IF_ERROR(loom_aie2p_native_relocation_patch_local_address(
            target_address, fixup_layout->section_offset, source_section));
        break;
      case LOOM_AIE2P_NATIVE_RELOCATION_KIND_NONE:
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown AIE2P relocation kind %u",
                                fixup->relocation_kind);
    }
  }
  return iree_ok_status();
}
