// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/hardware_loops.h"

#include <inttypes.h>

#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/execute.h"
#include "loomc/iree.h"

enum {
  LOOM_AIE2P_HARDWARE_LOOP_SECTION_FLAG_ALLOCATE = 2u,
  LOOM_AIE2P_HARDWARE_LOOP_SECTION_FLAG_EXECUTE = 4u,
  LOOM_AIE2P_HARDWARE_LOOP_MINIMUM_SETUP_DISTANCE = 112u,
};

typedef struct loom_aie2p_hardware_loop_setup_t {
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
} loom_aie2p_hardware_loop_setup_t;

static iree_status_t loom_aie2p_hardware_loops_unpack_descriptor(
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

static iree_status_t loom_aie2p_hardware_loops_decode_setup_bundle(
    const loom_aie2p_decoded_bundle_t* bundle, uint32_t bundle_offset,
    uint32_t bundle_length, loom_aie2p_hardware_loop_setup_t* setup,
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
    IREE_RETURN_IF_ERROR(loom_aie2p_hardware_loops_unpack_descriptor(
        slot, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_COUNT_SHORT, mova, &matches,
        &immediate));
    if (matches) {
      setup->iteration_count = immediate;
      setup->count_bundle_offset = bundle_offset;
      setup->count_bundle_length = bundle_length;
      setup->has_count = true;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_hardware_loops_unpack_descriptor(
        slot, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_START_ADDRESS, movxm,
        &matches, &immediate));
    if (matches) {
      setup->start_address = immediate;
      setup->start_bundle_offset = bundle_offset;
      setup->has_start = true;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_hardware_loops_unpack_descriptor(
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

static iree_status_t loom_aie2p_hardware_loops_bundle_has_control(
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

static bool loom_aie2p_hardware_loops_bundle_is_nop(
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

static iree_status_t loom_aie2p_hardware_loops_body_facts(
    iree_const_byte_span_t section, uint64_t section_address,
    const loom_aie2p_hardware_loop_setup_t* setup, bool* out_end_on_bundle,
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
      if (loom_aie2p_hardware_loops_bundle_is_nop(&bundle)) {
        ++*out_trailing_nop_bundle_count;
      } else {
        *out_trailing_nop_bundle_count = 0;
      }
      bool has_control = false;
      IREE_RETURN_IF_ERROR(
          loom_aie2p_hardware_loops_bundle_has_control(&bundle, &has_control));
      *out_control_count += has_control;
    }
    offset += bundle_length;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_hardware_loops_print_section(
    iree_string_view_t section_name, uint64_t section_address,
    iree_const_byte_span_t section, iree_string_builder_t* output) {
  uint32_t loop_count = 0;
  for (uint8_t pass = 0; pass < 2; ++pass) {
    iree_host_size_t offset = 0;
    uint32_t loop_ordinal = 0;
    loom_aie2p_hardware_loop_setup_t setup = {0};
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
      IREE_RETURN_IF_ERROR(loom_aie2p_hardware_loops_decode_setup_bundle(
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
          IREE_RETURN_IF_ERROR(loom_aie2p_hardware_loops_body_facts(
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
                          LOOM_AIE2P_HARDWARE_LOOP_MINIMUM_SETUP_DISTANCE
                  ? "true"
                  : "false",
              trailing_nop_bundle_count, body_control_count));
        }
        setup = (loom_aie2p_hardware_loop_setup_t){0};
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

// Decodes linked executable sections and reports architectural loop facts.
// Offsets and packet bytes remain incidental; the output captures the final
// state-register programming and the contracts imposed on its repeated range.
static iree_status_t loom_aie2p_hardware_loops_print(
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
        !iree_all_bits_set(flags,
                           LOOM_AIE2P_HARDWARE_LOOP_SECTION_FLAG_ALLOCATE |
                               LOOM_AIE2P_HARDWARE_LOOP_SECTION_FLAG_EXECUTE)) {
      continue;
    }
    const uint32_t address = iree_unaligned_load_le_u32(header + 12);
    const uint32_t offset = iree_unaligned_load_le_u32(header + 16);
    const uint32_t length = iree_unaligned_load_le_u32(header + 20);
    const iree_string_view_t name =
        iree_make_cstring_view(names + iree_unaligned_load_le_u32(header));
    IREE_RETURN_IF_ERROR(loom_aie2p_hardware_loops_print_section(
        name, address, iree_make_const_byte_span(bytes.data + offset, length),
        output));
  }
  return iree_ok_status();
}

static bool loom_aie2p_hardware_loops_check_matches(
    const loom_check_emit_provider_t* provider, iree_string_view_t name) {
  (void)provider;
  return iree_string_view_equal(name, IREE_SV("aie2p-hardware-loops"));
}

static iree_status_t loom_aie2p_hardware_loops_check_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  iree_string_view_t options = iree_string_view_trim(request->target_options);
  bool source_low = false;
  while (!iree_string_view_is_empty(options)) {
    iree_string_view_t option;
    iree_string_view_split(options, ' ', &option, &options);
    options = iree_string_view_trim(options);
    if (iree_string_view_equal(option, IREE_SV("input=source-low")) &&
        !source_low) {
      source_low = true;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid or duplicate aie2p-hardware-loops option '%.*s'",
          (int)option.size, option.data);
    }
  }

  const loom_check_compile_artifact_options_t compile_options = {
      .artifact_format = IREE_SV("xdna"),
      .lower_source_to_low = source_low,
  };
  loomc_result_t* compile_result = NULL;
  iree_status_t status =
      loom_check_compile_artifact(request, &compile_options, &compile_result);
  const loomc_artifact_t* artifact = NULL;
  if (iree_status_is_ok(status) && compile_result != NULL) {
    status = loom_check_compile_select_artifact(
        compile_result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
        compile_options.artifact_format, &artifact);
  }
  loomc_source_t* artifact_source = NULL;
  if (iree_status_is_ok(status) && artifact != NULL) {
    status = iree_status_from_loomc(loomc_artifact_create_source(
        artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
        loomc_allocator_from_iree(request->host_allocator), &artifact_source));
  }
  loomc_result_release(compile_result);
  if (iree_status_is_ok(status) && artifact_source != NULL) {
    const loomc_byte_span_t source_contents =
        loomc_source_contents(artifact_source);
    status = loom_aie2p_hardware_loops_print(
        iree_make_const_byte_span(source_contents.data,
                                  source_contents.data_length),
        &request->result->actual_output);
  }
  loomc_source_release(artifact_source);
  return status;
}

static iree_status_t loom_aie2p_hardware_loops_check_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "aie2p-hardware-loops\n");
}

static const loom_check_emit_provider_t kAie2pHardwareLoopsCheckEmitProvider = {
    .name = IREE_SVL("aie2p-hardware-loops"),
    .match = loom_aie2p_hardware_loops_check_matches,
    .execute = loom_aie2p_hardware_loops_check_execute,
    .append_names = loom_aie2p_hardware_loops_check_append_names,
};

static const loom_check_emit_provider_t* const
    kAie2pHardwareLoopsCheckEmitProviders[] = {
        &kAie2pHardwareLoopsCheckEmitProvider,
};

const loom_check_provider_t loom_aie2p_hardware_loops_check_provider = {
    .name = IREE_SVL("amd-xdna-aie2p-hardware-loops"),
    .emit_providers = kAie2pHardwareLoopsCheckEmitProviders,
    .emit_provider_count =
        IREE_ARRAYSIZE(kAie2pHardwareLoopsCheckEmitProviders),
};
