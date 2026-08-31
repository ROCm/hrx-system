// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
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
  const uint32_t branch_descriptor_ordinals[] = {
      AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT,
      AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO,
      AIE2P_CORE_DESCRIPTOR_REF_BRANCH_ZERO,
  };
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const loom_low_descriptor_t* branch_descriptor = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(branch_descriptor_ordinals);
       ++i) {
    const loom_low_descriptor_t* candidate_descriptor =
        &descriptor_set->descriptors[branch_descriptor_ordinals[i]];
    for (iree_host_size_t j = 0; j < candidate_count; ++j) {
      if (instruction_candidates[j] != candidate_descriptor->encoding_id) {
        continue;
      }
      if (branch_descriptor != NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P branch relocation matches multiple instruction forms");
      }
      branch_descriptor = candidate_descriptor;
    }
  }
  if (branch_descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P branch relocation does not reference J, JZ, or JNZ");
  }

  const iree_host_size_t field_capacity =
      branch_descriptor->encoding_field_value_count +
      branch_descriptor->operand_count + branch_descriptor->immediate_count;
  loom_aie2p_encoding_field_value_t* field_values =
      (loom_aie2p_encoding_field_value_t*)iree_alloca(field_capacity *
                                                      sizeof(*field_values));
  iree_host_size_t field_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_unpack_instruction(
      branch_descriptor->encoding_id, decoded_bundle.slots[0].value,
      field_capacity, field_values, &field_count));
  IREE_ASSERT_EQ(branch_descriptor->immediate_count, 1u);
  const loom_low_immediate_t* target_immediate =
      &descriptor_set->immediates[branch_descriptor->immediate_start];
  if (target_address > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P core branch target exceeds signed range");
  }
  uint64_t encoded_target = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_machine_encode_immediate(
      (loom_aie2p_immediate_id_t)target_immediate->encoding_id,
      (int64_t)target_address, &encoded_target));
  bool target_field_found = false;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (field_values[i].field_id != target_immediate->encoding_field_id) {
      continue;
    }
    field_values[i].value = encoded_target;
    target_field_found = true;
  }
  IREE_ASSERT(target_field_found &&
              "generated branch descriptor must encode cpmaddr");
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      branch_descriptor->encoding_id, field_values, field_count,
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

  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const loom_low_descriptor_t* address_descriptor =
      &descriptor_set->descriptors
           [AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32];
  const iree_host_size_t instruction_count =
      loom_aie2p_encoding_instruction_count();
  loom_aie2p_instruction_id_t* instruction_candidates =
      (loom_aie2p_instruction_id_t*)iree_alloca(
          instruction_count * sizeof(*instruction_candidates));
  loom_aie2p_encoded_slot_t* address_slot = NULL;
  for (uint8_t i = 0; i < decoded_bundle.slot_count; ++i) {
    loom_aie2p_encoded_slot_t* slot = &decoded_bundle.slots[i];
    const iree_host_size_t candidate_count =
        loom_aie2p_encoding_query_instruction_candidates(
            slot->slot, slot->value, instruction_count, instruction_candidates);
    IREE_ASSERT_LE(candidate_count, instruction_count);
    for (iree_host_size_t j = 0; j < candidate_count; ++j) {
      if (instruction_candidates[j] != address_descriptor->encoding_id) {
        continue;
      }
      if (address_slot != NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P local-address relocation matches multiple MOVXM slots");
      }
      address_slot = slot;
    }
  }
  if (address_slot == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P local-address relocation does not reference MOVXM");
  }

  const iree_host_size_t field_capacity =
      address_descriptor->encoding_field_value_count +
      address_descriptor->operand_count + address_descriptor->immediate_count;
  loom_aie2p_encoding_field_value_t* field_values =
      (loom_aie2p_encoding_field_value_t*)iree_alloca(field_capacity *
                                                      sizeof(*field_values));
  iree_host_size_t field_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_unpack_instruction(
      address_descriptor->encoding_id, address_slot->value, field_capacity,
      field_values, &field_count));
  IREE_ASSERT_EQ(address_descriptor->immediate_count, 1u);
  const loom_low_immediate_t* address_immediate =
      &descriptor_set->immediates[address_descriptor->immediate_start];
  if (target_address > INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P local-memory address exceeds signed MOVXM range");
  }
  uint64_t encoded_target = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_machine_encode_immediate(
      (loom_aie2p_immediate_id_t)address_immediate->encoding_id,
      (int64_t)target_address, &encoded_target));
  bool target_field_found = false;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (field_values[i].field_id != address_immediate->encoding_field_id) {
      continue;
    }
    field_values[i].value = encoded_target;
    target_field_found = true;
  }
  IREE_ASSERT(target_field_found &&
              "generated local-address descriptor must encode i");
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      address_descriptor->encoding_id, field_values, field_count,
      &address_slot->value));

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
  if (object->fixup_count == 0) return iree_ok_status();

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
      case LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE:
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
