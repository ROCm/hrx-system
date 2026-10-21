// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/artifact.h"

#include <inttypes.h>

#include "iree/schemas/xdna_executable.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"
#include "loom/tools/loom-check/compile.h"

enum {
  LOOM_AIE2P_ARTIFACT_CHECK_SECTION_FLAG_ALLOCATE = 2u,
  LOOM_AIE2P_ARTIFACT_CHECK_SECTION_FLAG_EXECUTE = 4u,
  LOOM_AIE2P_ARTIFACT_CHECK_MINIMUM_LOOP_SETUP_DISTANCE = 112u,
};

typedef struct loom_aie2p_artifact_loop_setup_t {
  // Decoded number of hardware-loop body executions.
  uint64_t iteration_count;
  // Relocated architectural loop-start address.
  uint64_t start_address;
  // Relocated architectural loop-end address.
  uint64_t end_address;
  // Section-relative bundle containing LC.
  uint32_t count_bundle_offset;
  // Encoded length of the bundle containing LC.
  uint32_t count_bundle_length;
  // Section-relative bundle containing LS.
  uint32_t start_bundle_offset;
  // Section-relative bundle containing LE.
  uint32_t end_setup_bundle_offset;
  // Encoded length of the bundle containing LE.
  uint32_t end_setup_bundle_length;
  // Whether LC has been decoded.
  bool has_count;
  // Whether LS has been decoded.
  bool has_start;
} loom_aie2p_artifact_loop_setup_t;

static iree_status_t loom_aie2p_artifact_check_unpack_loop_descriptor(
    const loom_aie2p_encoded_slot_t* slot, uint32_t descriptor_ordinal,
    loom_aie2p_instruction_id_t instruction, bool* out_matches,
    uint64_t* out_immediate) {
  *out_matches = false;
  *out_immediate = 0;
  loom_aie2p_instruction_id_t candidates[32];
  const iree_host_size_t candidate_count =
      loom_aie2p_encoding_query_instruction_candidates(
          slot->slot, slot->value, IREE_ARRAYSIZE(candidates), candidates);
  if (candidate_count > IREE_ARRAYSIZE(candidates)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P artifact instruction has too many decode candidates");
  }
  bool has_instruction = false;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    has_instruction |= candidates[i] == instruction;
  }
  if (!has_instruction) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const int64_t zero_immediate[] = {0};
  const loom_aie2p_encoded_slot_t reference = loom_aie2p_descriptor_encode(
      descriptor_set, descriptor_ordinal, /*operand_assignments=*/NULL,
      zero_immediate);
  if (slot->slot != reference.slot) {
    return iree_ok_status();
  }
  loom_aie2p_encoding_field_value_t actual_fields[8];
  loom_aie2p_encoding_field_value_t reference_fields[8];
  iree_host_size_t actual_field_count = 0;
  iree_host_size_t reference_field_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_unpack_instruction(
      instruction, slot->value, IREE_ARRAYSIZE(actual_fields), actual_fields,
      &actual_field_count));
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_unpack_instruction(
      instruction, reference.value, IREE_ARRAYSIZE(reference_fields),
      reference_fields, &reference_field_count));
  if (actual_field_count > IREE_ARRAYSIZE(actual_fields) ||
      reference_field_count > IREE_ARRAYSIZE(reference_fields)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P loop setup has too many encoded fields");
  }
  const loom_aie2p_encoding_field_id_t destination_field =
      loom_aie2p_encoding_find_field(IREE_SV("dst"));
  const loom_aie2p_encoding_field_id_t immediate_field =
      loom_aie2p_encoding_find_field(IREE_SV("i"));
  uint64_t actual_destination = UINT64_MAX;
  uint64_t reference_destination = UINT64_MAX;
  for (iree_host_size_t i = 0; i < actual_field_count; ++i) {
    if (actual_fields[i].field_id == destination_field) {
      actual_destination = actual_fields[i].value;
    } else if (actual_fields[i].field_id == immediate_field) {
      *out_immediate = actual_fields[i].value;
    }
  }
  for (iree_host_size_t i = 0; i < reference_field_count; ++i) {
    if (reference_fields[i].field_id == destination_field) {
      reference_destination = reference_fields[i].value;
    }
  }
  *out_matches = actual_destination != UINT64_MAX &&
                 actual_destination == reference_destination;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_artifact_check_decode_loop_setup_bundle(
    const loom_aie2p_decoded_bundle_t* bundle, uint32_t bundle_offset,
    uint32_t bundle_length, loom_aie2p_artifact_loop_setup_t* setup,
    bool* out_has_end) {
  *out_has_end = false;
  const loom_aie2p_instruction_id_t mova =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVA"));
  const loom_aie2p_instruction_id_t movxm =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVXM"));
  for (uint8_t i = 0; i < bundle->slot_count; ++i) {
    const loom_aie2p_encoded_slot_t* slot = &bundle->slots[i];
    bool matches = false;
    uint64_t immediate = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_artifact_check_unpack_loop_descriptor(
        slot, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_COUNT_SHORT, mova, &matches,
        &immediate));
    if (matches) {
      setup->iteration_count = immediate;
      setup->count_bundle_offset = bundle_offset;
      setup->count_bundle_length = bundle_length;
      setup->has_count = true;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_artifact_check_unpack_loop_descriptor(
        slot, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_START_ADDRESS, movxm,
        &matches, &immediate));
    if (matches) {
      setup->start_address = immediate;
      setup->start_bundle_offset = bundle_offset;
      setup->has_start = true;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_artifact_check_unpack_loop_descriptor(
        slot, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_END_ADDRESS, movxm, &matches,
        &immediate));
    if (matches) {
      setup->end_address = immediate;
      setup->end_setup_bundle_offset = bundle_offset;
      setup->end_setup_bundle_length = bundle_length;
      *out_has_end = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_artifact_check_bundle_has_control(
    const loom_aie2p_decoded_bundle_t* bundle, bool* out_has_control) {
  *out_has_control = false;
  for (uint8_t slot_index = 0; slot_index < bundle->slot_count; ++slot_index) {
    const loom_aie2p_encoded_slot_t* slot = &bundle->slots[slot_index];
    loom_aie2p_instruction_id_t candidates[32];
    const iree_host_size_t candidate_count =
        loom_aie2p_encoding_query_instruction_candidates(
            slot->slot, slot->value, IREE_ARRAYSIZE(candidates), candidates);
    if (candidate_count > IREE_ARRAYSIZE(candidates)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "AIE2P artifact instruction has too many decode candidates");
    }
    for (iree_host_size_t i = 0; i < candidate_count; ++i) {
      loom_aie2p_machine_form_info_t form;
      if (loom_aie2p_machine_query_form(
              (loom_aie2p_machine_form_id_t)candidates[i], &form) &&
          form.control_flow_kind != LOOM_AIE2P_CONTROL_FLOW_NONE) {
        *out_has_control = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static bool loom_aie2p_artifact_check_bundle_is_nop(
    const loom_aie2p_decoded_bundle_t* bundle) {
  for (uint8_t i = 0; i < bundle->slot_count; ++i) {
    const loom_aie2p_encoded_slot_t* slot = &bundle->slots[i];
    const loom_aie2p_encoded_slot_t nop =
        loom_aie2p_bundle_encoding_encode_slot_nop(slot->slot);
    if (slot->value != nop.value) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_aie2p_artifact_check_loop_body_facts(
    iree_const_byte_span_t section, uint64_t section_address,
    const loom_aie2p_artifact_loop_setup_t* setup, bool* out_end_on_bundle,
    bool* out_end_group_full_width, uint32_t* out_trailing_nop_bundle_count,
    uint32_t* out_control_count) {
  *out_end_on_bundle = false;
  *out_end_group_full_width = false;
  *out_trailing_nop_bundle_count = 0;
  *out_control_count = 0;
  iree_host_size_t offset = 0;
  while (offset < section.data_length) {
    loom_aie2p_decoded_bundle_t bundle;
    iree_host_size_t bundle_length = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_encoding_decode_bundle_prefix(
        iree_make_const_byte_span(
            section.data + offset,
            iree_min(section.data_length - offset,
                     (iree_host_size_t)LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE)),
        &bundle, &bundle_length));
    const uint64_t address = section_address + offset;
    *out_end_on_bundle |= address == setup->end_address;
    *out_end_group_full_width |=
        address == setup->end_address &&
        bundle_length == LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE;
    if (address >= setup->start_address && address <= setup->end_address) {
      if (loom_aie2p_artifact_check_bundle_is_nop(&bundle)) {
        ++*out_trailing_nop_bundle_count;
      } else {
        *out_trailing_nop_bundle_count = 0;
      }
      bool has_control = false;
      IREE_RETURN_IF_ERROR(
          loom_aie2p_artifact_check_bundle_has_control(&bundle, &has_control));
      *out_control_count += has_control;
    }
    offset += bundle_length;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_artifact_check_print_hardware_loops_in_section(
    iree_string_view_t section_name, uint64_t section_address,
    iree_const_byte_span_t section, iree_string_builder_t* output) {
  uint32_t loop_count = 0;
  for (uint8_t pass = 0; pass < 2; ++pass) {
    iree_host_size_t offset = 0;
    uint32_t loop_ordinal = 0;
    loom_aie2p_artifact_loop_setup_t setup = {0};
    while (offset < section.data_length) {
      loom_aie2p_decoded_bundle_t bundle;
      iree_host_size_t bundle_length = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_encoding_decode_bundle_prefix(
          iree_make_const_byte_span(
              section.data + offset,
              iree_min(section.data_length - offset,
                       (iree_host_size_t)LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE)),
          &bundle, &bundle_length));
      bool has_end = false;
      IREE_RETURN_IF_ERROR(loom_aie2p_artifact_check_decode_loop_setup_bundle(
          &bundle, (uint32_t)offset, (uint32_t)bundle_length, &setup,
          &has_end));
      if (has_end && setup.has_count && setup.has_start) {
        if (pass == 0) {
          ++loop_count;
        } else {
          bool end_on_bundle = false;
          bool end_group_full_width = false;
          uint32_t trailing_nop_bundle_count = 0;
          uint32_t body_control_count = 0;
          IREE_RETURN_IF_ERROR(loom_aie2p_artifact_check_loop_body_facts(
              section, section_address, &setup, &end_on_bundle,
              &end_group_full_width, &trailing_nop_bundle_count,
              &body_control_count));
          const uint64_t end_setup_address =
              section_address + setup.end_setup_bundle_offset;
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              output,
              "  loop %u: iterations=%" PRIu64
              " count_start_coissued=%s end_setup_next=%s"
              " start_aligned=%s start_after_setup=%s end_aligned=%s"
              " end_on_bundle=%s end_group_full_width=%s"
              " setup_distance_valid=%s trailing_nop_bundles=%u"
              " body_control=%u\n",
              loop_ordinal++, setup.iteration_count,
              setup.count_bundle_offset == setup.start_bundle_offset ? "true"
                                                                     : "false",
              setup.end_setup_bundle_offset ==
                      setup.count_bundle_offset + setup.count_bundle_length
                  ? "true"
                  : "false",
              setup.start_address % 16u == 0 ? "true" : "false",
              setup.start_address ==
                      end_setup_address + setup.end_setup_bundle_length
                  ? "true"
                  : "false",
              setup.end_address % 16u == 0 ? "true" : "false",
              end_on_bundle ? "true" : "false",
              end_group_full_width ? "true" : "false",
              setup.end_address >= end_setup_address &&
                      setup.end_address - end_setup_address >=
                          LOOM_AIE2P_ARTIFACT_CHECK_MINIMUM_LOOP_SETUP_DISTANCE
                  ? "true"
                  : "false",
              trailing_nop_bundle_count, body_control_count));
        }
        setup = (loom_aie2p_artifact_loop_setup_t){0};
      }
      offset += bundle_length;
    }
    if (pass == 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "%.*s: hardware_loops=%u\n", (int)section_name.size,
          section_name.data, loop_count));
    }
  }
  return iree_ok_status();
}

static bool loom_aie2p_artifact_check_matches(
    const loom_check_emit_provider_t* provider, iree_string_view_t name) {
  (void)provider;
  return iree_string_view_equal(name, IREE_SV("aie2p-xdna"));
}

static iree_status_t loom_aie2p_artifact_check_print(
    iree_const_byte_span_t bytes, iree_string_builder_t* output) {
  // The production ELF writer owns layout validity. Read its public wire
  // records independently, so checks cover serialized requirements and fixups.
  const uint32_t header_offset = iree_unaligned_load_le_u32(bytes.data + 28);
  const uint32_t metadata_offset =
      iree_unaligned_load_le_u32(bytes.data + header_offset + 4);
  const uint8_t* metadata = bytes.data + metadata_offset;
  const iree_xdna_elf_header_record_t header =
      iree_xdna_elf_decode_header(metadata);
  const uint32_t entry_offset =
      IREE_XDNA_ELF_HEADER_RECORD_SIZE +
      header.allocation_count * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE +
      header.allocation_use_count * sizeof(uint32_t);
  const uint32_t binding_offset =
      entry_offset + header.entry_count * IREE_XDNA_ELF_ENTRY_RECORD_SIZE;
  const uint32_t relocation_offset =
      binding_offset + header.binding_count * IREE_XDNA_ELF_BINDING_RECORD_SIZE;
  const uint32_t invocation_offset =
      relocation_offset +
      header.relocation_count * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "partition: %u columns, %u rows\nentries: %u\n",
      header.column_count, header.row_count, header.entry_count));
  for (uint32_t i = 0; i < header.entry_count; ++i) {
    const iree_xdna_elf_entry_record_t entry = iree_xdna_elf_decode_entry(
        metadata + entry_offset + i * IREE_XDNA_ELF_ENTRY_RECORD_SIZE);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "entry %u: %u bindings, %u dynamic relocations\n", i,
        entry.binding_count, entry.dynamic_relocation_count));
    for (uint32_t j = 0; j < entry.binding_count; ++j) {
      const iree_xdna_elf_binding_record_t binding =
          iree_xdna_elf_decode_binding(metadata + binding_offset +
                                       (entry.first_binding + j) *
                                           IREE_XDNA_ELF_BINDING_RECORD_SIZE);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output,
          "  binding %u: access=%u bytes=%" PRIu64 " alignment=%" PRIu64
          " offset=[%" PRIu64 ",%" PRIu64 "]\n",
          j, binding.access, binding.minimum_byte_length,
          binding.minimum_alignment, binding.minimum_byte_offset,
          binding.maximum_byte_offset));
    }
    for (uint32_t j = 0; j < entry.dynamic_relocation_count; ++j) {
      const iree_xdna_elf_relocation_record_t relocation =
          iree_xdna_elf_decode_relocation(
              metadata + relocation_offset +
              (entry.first_dynamic_relocation + j) *
                  IREE_XDNA_ELF_RELOCATION_RECORD_SIZE);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output,
          "  relocation %u: binding=%u addend=%" PRId64 " maximum=%" PRIu64
          " alignment=%" PRIu64 "\n",
          j, relocation.source_ordinal, relocation.addend,
          relocation.maximum_value, relocation.alignment));
    }
    for (uint32_t j = 0; j < entry.invocation_count; ++j) {
      const iree_xdna_elf_invocation_record_t invocation =
          iree_xdna_elf_decode_invocation(
              metadata + invocation_offset +
              (entry.first_invocation + j) *
                  IREE_XDNA_ELF_INVOCATION_RECORD_SIZE);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "  invocation %u: bytes=%u next=%u\n", j,
          invocation.byte_length, invocation.next_invocation));
    }
  }
  return iree_ok_status();
}

// Read final ELF sections, including relocated code and uninitialized data.
// This is check-provider output, independent of the compiler's placement plan.
static iree_status_t loom_aie2p_artifact_check_print_sections(
    iree_const_byte_span_t bytes, iree_string_builder_t* output) {
  const uint32_t table_offset = iree_unaligned_load_le_u32(bytes.data + 32);
  const uint16_t record_size = iree_unaligned_load_le_u16(bytes.data + 46);
  const uint16_t record_count = iree_unaligned_load_le_u16(bytes.data + 48);
  const uint16_t names_index = iree_unaligned_load_le_u16(bytes.data + 50);
  const uint8_t* names_header =
      bytes.data + table_offset + names_index * record_size;
  const char* names =
      (const char*)bytes.data + iree_unaligned_load_le_u32(names_header + 16);
  for (uint16_t i = 0; i < record_count; ++i) {
    const uint8_t* header = bytes.data + table_offset + i * record_size;
    const uint32_t flags = iree_unaligned_load_le_u32(header + 8);
    if (!(flags & 2)) {
      continue;  // Only SHF_ALLOC sections occupy native worker memory.
    }
    const uint32_t type = iree_unaligned_load_le_u32(header + 4);
    const uint32_t address = iree_unaligned_load_le_u32(header + 12);
    const uint32_t length = iree_unaligned_load_le_u32(header + 20);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "%s: address=0x%08x bytes=%u %s\n",
        names + iree_unaligned_load_le_u32(header), address, length,
        type == 8 ? "reserved" : "initialized"));
    if (type == 8) {
      continue;
    }
    const uint8_t* contents =
        bytes.data + iree_unaligned_load_le_u32(header + 16);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "  "));
    for (uint32_t j = 0; j < length; ++j) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(output, "%02x", contents[j]));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

// Decodes linked executable sections and reports architectural loop facts.
// Offsets and packet bytes remain incidental; the output captures the final
// state-register programming and the contracts imposed on its repeated range.
static iree_status_t loom_aie2p_artifact_check_print_hardware_loops(
    iree_const_byte_span_t bytes, iree_string_builder_t* output) {
  const uint32_t table_offset = iree_unaligned_load_le_u32(bytes.data + 32);
  const uint16_t record_size = iree_unaligned_load_le_u16(bytes.data + 46);
  const uint16_t record_count = iree_unaligned_load_le_u16(bytes.data + 48);
  const uint16_t names_index = iree_unaligned_load_le_u16(bytes.data + 50);
  const uint8_t* names_header =
      bytes.data + table_offset + names_index * record_size;
  const char* names =
      (const char*)bytes.data + iree_unaligned_load_le_u32(names_header + 16);
  for (uint16_t i = 0; i < record_count; ++i) {
    const uint8_t* header = bytes.data + table_offset + i * record_size;
    const uint32_t type = iree_unaligned_load_le_u32(header + 4);
    const uint32_t flags = iree_unaligned_load_le_u32(header + 8);
    if (type == 8 ||
        !iree_all_bits_set(
            flags, LOOM_AIE2P_ARTIFACT_CHECK_SECTION_FLAG_ALLOCATE |
                       LOOM_AIE2P_ARTIFACT_CHECK_SECTION_FLAG_EXECUTE)) {
      continue;
    }
    const uint32_t address = iree_unaligned_load_le_u32(header + 12);
    const uint32_t offset = iree_unaligned_load_le_u32(header + 16);
    const uint32_t length = iree_unaligned_load_le_u32(header + 20);
    const iree_string_view_t name =
        iree_make_cstring_view(names + iree_unaligned_load_le_u32(header));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_artifact_check_print_hardware_loops_in_section(
            name, address,
            iree_make_const_byte_span(bytes.data + offset, length), output));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_artifact_check_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  iree_string_view_t options = iree_string_view_trim(request->target_options);
  enum {
    OUTPUT_ARTIFACT,
    OUTPUT_SECTIONS,
    OUTPUT_HARDWARE_LOOPS,
    OUTPUT_PIPELINE_REPORT,
  } output = OUTPUT_ARTIFACT;
  bool source_low = false;
  while (!iree_string_view_is_empty(options)) {
    iree_string_view_t option;
    iree_string_view_split(options, ' ', &option, &options);
    options = iree_string_view_trim(options);
    if (iree_string_view_equal(option, IREE_SV("output=sections")) &&
        output == OUTPUT_ARTIFACT) {
      output = OUTPUT_SECTIONS;
    } else if (iree_string_view_equal(option,
                                      IREE_SV("output=hardware-loops")) &&
               output == OUTPUT_ARTIFACT) {
      output = OUTPUT_HARDWARE_LOOPS;
    } else if (iree_string_view_equal(option,
                                      IREE_SV("output=pipeline-report")) &&
               output == OUTPUT_ARTIFACT) {
      output = OUTPUT_PIPELINE_REPORT;
    } else if (iree_string_view_equal(option, IREE_SV("input=source-low")) &&
               !source_low) {
      source_low = true;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid or duplicate aie2p-xdna option '%.*s'",
                              (int)option.size, option.data);
    }
  }
  const loomc_compile_report_options_t report_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(report_options),
      .mode = LOOMC_COMPILE_REPORT_MODE_DETAILS,
      .format = LOOMC_COMPILE_REPORT_FORMAT_TEXT,
  };
  const loom_check_compile_artifact_options_t compile_options = {
      .artifact_format = IREE_SV("xdna"),
      .lower_source_to_low = source_low,
      .report = output == OUTPUT_PIPELINE_REPORT ? &report_options : NULL,
  };
  loomc_source_t* artifact_source = NULL;
  loomc_source_t* report_source = NULL;
  iree_status_t status = loom_check_compile_artifact(
      request, &compile_options, &artifact_source, &report_source);
  const loomc_byte_span_t source_contents =
      loomc_source_contents(artifact_source);
  const iree_const_byte_span_t bytes = iree_make_const_byte_span(
      source_contents.data, source_contents.data_length);
  if (iree_status_is_ok(status) && artifact_source != NULL &&
      output == OUTPUT_PIPELINE_REPORT) {
    // Select the canonical inventory from the public report. Per-function
    // scheduling and compiler allocation statistics are separate contracts.
    const loomc_byte_span_t report_contents =
        loomc_source_contents(report_source);
    iree_string_view_t remaining = iree_make_string_view(
        (const char*)report_contents.data, report_contents.data_length);
    while (iree_status_is_ok(status) && !iree_string_view_is_empty(remaining)) {
      iree_string_view_t line;
      iree_string_view_split(remaining, '\n', &line, &remaining);
      if (iree_string_view_starts_with(line,
                                       IREE_SV("COMPILE-REPORT: pipeline"))) {
        status = iree_string_builder_append_format(
            &request->result->actual_output, "%.*s\n", (int)line.size,
            line.data);
      }
    }
  } else if (iree_status_is_ok(status) && artifact_source != NULL &&
             output == OUTPUT_SECTIONS) {
    status = loom_aie2p_artifact_check_print_sections(
        bytes, &request->result->actual_output);
  } else if (iree_status_is_ok(status) && artifact_source != NULL &&
             output == OUTPUT_HARDWARE_LOOPS) {
    status = loom_aie2p_artifact_check_print_hardware_loops(
        bytes, &request->result->actual_output);
  } else if (iree_status_is_ok(status) && artifact_source != NULL) {
    status =
        loom_aie2p_artifact_check_print(bytes, &request->result->actual_output);
  }
  loomc_source_release(report_source);
  loomc_source_release(artifact_source);
  return status;
}

static iree_status_t loom_aie2p_artifact_check_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "aie2p-xdna\n");
}

const loom_check_emit_provider_t loom_aie2p_artifact_check_emit_provider = {
    .name = IREE_SVL("aie2p-xdna"),
    .match = loom_aie2p_artifact_check_matches,
    .execute = loom_aie2p_artifact_check_execute,
    .append_names = loom_aie2p_artifact_check_append_names,
};
