// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_emission.h"

#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/kernel_assembly.h"
#include "loom/target/emit/native/amdgpu/kernel_entry.h"

iree_status_t loom_amdgpu_kernel_emission_build(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_kernel_plan_t* plan,
    const loom_amdgpu_data_symbol_directory_t* data_symbols,
    loom_amdgpu_kernel_emission_flags_t flags,
    iree_string_builder_t* target_listing, iree_arena_allocator_t* table_arena,
    iree_diagnostic_emitter_t emitter, bool* out_accepted,
    loom_amdgpu_kernel_emission_result_t* out_result) {
  *out_accepted = false;
  *out_result = (loom_amdgpu_kernel_emission_result_t){0};
  const loom_amdgpu_kernel_record_t* record = &plan->record;
  const loom_amdgpu_packet_plan_t* packet_plan = &plan->function.packet_plan;

  loom_amdgpu_encoded_instruction_stream_t stream = {0};
  const loom_amdgpu_encode_instruction_stream_options_t encode_options = {
      .packet_plan = packet_plan,
      .storage_layout = &plan->function.storage_layout,
      .flags =
          iree_any_bit_set(
              flags, LOOM_AMDGPU_KERNEL_EMISSION_FLAG_CAPTURE_NATIVE_INSERTIONS)
              ? LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS
              : LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_instruction_stream_result_with_options(
          &frame->schedule, &frame->allocation, &encode_options, table_arena,
          &stream));

  const loom_amdgpu_kernel_entry_envelope_t* entry_envelope =
      record->entry_envelope;
  iree_const_byte_span_t kernel_text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* body_text_fixups = NULL;
  bool references_bound = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_data_symbol_directory_bind_fixups(
      data_symbols, &frame->schedule, stream.text_fixups,
      stream.text_fixup_count, table_arena, emitter, &references_bound,
      &body_text_fixups));
  if (!references_bound) {
    return iree_ok_status();
  }
  const loom_amdgpu_hsaco_text_fixup_t* kernel_text_fixups = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_entry_prepend_text(
      entry_envelope, stream.text, body_text_fixups, stream.text_fixup_count,
      table_arena, &kernel_text, &kernel_text_fixups));
  if (target_listing != NULL) {
    if (iree_string_builder_size(target_listing) != 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(target_listing, "\n\n"));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_kernel_assembly(
        &frame->schedule, &frame->allocation, record, packet_plan,
        &stream.layout, target_listing, table_arena));
  }

  *out_result = (loom_amdgpu_kernel_emission_result_t){
      .kernel =
          {
              .metadata = record->metadata,
              .descriptor_options =
                  {
                      .flags = record->descriptor_flags,
                      .minimum_user_sgpr_count = record->user_sgpr_count,
                      .accum_offset = record->accum_offset,
                  },
              .text = kernel_text,
              .text_fixups = kernel_text_fixups,
              .text_fixup_count = stream.text_fixup_count,
          },
      .body = stream,
      .entry_instruction_count = entry_envelope->instruction_count,
      .coissued_instruction_count = packet_plan->vopd_plan.pair_count,
  };
  *out_accepted = true;
  return iree_ok_status();
}
