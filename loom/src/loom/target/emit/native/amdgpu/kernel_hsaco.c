// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"

#include <inttypes.h>

#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/emit/native/amdgpu/kernel_entry.h"
#include "loom/target/emit/native/amdgpu/kernel_record.h"

iree_status_t loom_amdgpu_build_kernel_hsaco_contribution(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* scratch_arena) {
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){0};
  if (options->summary != NULL) {
    *options->summary = (loom_amdgpu_kernel_hsaco_summary_t){0};
  }

  loom_amdgpu_kernel_record_t record = {0};
  const loom_amdgpu_kernel_record_options_t record_options = {
      .abi_layout = options->abi_layout,
      .abi_verify = options->abi_verify,
      .preflight = options->preflight,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build(
      schedule, allocation, &record_options, &record, scratch_arena));

  loom_amdgpu_encoded_instruction_stream_t stream = {0};
  const loom_amdgpu_encode_instruction_stream_options_t encode_options = {
      .packet_plan = options->packet_plan,
      .storage_layout = &record.storage_layout,
      .flags = options->encoding_flags,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_instruction_stream_result_with_options(
          schedule, allocation, &encode_options, &stream, scratch_arena));

  const loom_amdgpu_kernel_entry_envelope_t* entry_envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(
          &record.processor->properties);
  iree_const_byte_span_t kernel_text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* kernel_text_fixups = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_entry_prepend_text(
      entry_envelope, stream.text, stream.text_fixups, stream.text_fixup_count,
      &kernel_text, &kernel_text_fixups, scratch_arena));
  uint64_t kernel_instruction_count = 0;
  if (!iree_checked_add_u64(stream.instruction_count,
                            entry_envelope->instruction_count,
                            &kernel_instruction_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU kernel entry instruction count overflowed");
  }
  const uint64_t coissued_instruction_count =
      options->packet_plan != NULL
          ? (uint64_t)options->packet_plan->vopd_plan.pair_count
          : 0;

  const loom_amdgpu_hsaco_kernel_t kernel = {
      .metadata = record.metadata,
      .descriptor_options =
          {
              .flags = record.descriptor_flags,
              .user_sgpr_count = record.user_sgpr_count,
          },
      .text = kernel_text,
      .text_fixups = kernel_text_fixups,
      .text_fixup_count = stream.text_fixup_count,
  };
  loom_amdgpu_occupancy_target_resources_t target_resources = {0};
  const uint32_t flat_workgroup_size =
      record.metadata.has_required_workgroup_size
          ? record.metadata.max_flat_workgroup_size
          : 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_build_target_resources(
      record.processor, record.metadata.wavefront_size,
      record.metadata.sgpr_count, record.metadata.vgpr_count,
      flat_workgroup_size, record.metadata.group_segment_fixed_size,
      scratch_arena, &target_resources));
  const loom_amdgpu_kernel_hsaco_target_resources_t hsaco_target_resources = {
      .scalar_register_class = target_resources.scalar_register_class,
      .scalar_register_count = target_resources.scalar_register_count,
      .vector_register_class = target_resources.vector_register_class,
      .vector_register_count = target_resources.vector_register_count,
      .wave_size = target_resources.wave_size,
      .max_waves_per_simd = target_resources.max_waves_per_simd,
      .resident_waves_per_simd = target_resources.resident_waves_per_simd,
      .occupancy_percent = target_resources.occupancy_percent,
      .limiting_resource = target_resources.limiting_resource,
      .residency_summary = target_resources.residency_summary,
  };
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){
      .artifact_target_key = record.artifact_target_key,
      .code_object_target_id = record.code_object_target_id,
      .processor = record.processor->name,
      .kernel = kernel,
      .branch_layout = stream.branch_layout,
      .native_insertions = stream.native_insertions,
      .native_insertion_count = stream.native_insertion_count,
      .summary =
          {
              .instruction_count = kernel_instruction_count,
              .body_instruction_count = stream.instruction_count,
              .entry_instruction_count = entry_envelope->instruction_count,
              .coissued_instruction_count = coissued_instruction_count,
              .coissued_component_count = coissued_instruction_count * 2u,
              .text_byte_count = kernel_text.data_length,
              .text_storage_byte_count = kernel_text.data_length,
              .private_segment_fixed_size =
                  record.metadata.private_segment_fixed_size,
              .group_segment_fixed_size =
                  record.metadata.group_segment_fixed_size,
              .target_resources = hsaco_target_resources,
          },
  };

  if (options->summary != NULL) {
    *options->summary = out_contribution->summary;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_write_kernel_hsaco_contributions(
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count,
    const loom_amdgpu_kernel_hsaco_write_options_t* options,
    iree_io_stream_t* stream, iree_arena_allocator_t* scratch_arena) {
  if (contribution_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel HSACO requires at least one contribution");
  }

  const iree_string_view_t artifact_target_key =
      contributions[0].artifact_target_key;
  const iree_string_view_t code_object_target_id =
      contributions[0].code_object_target_id;
  const iree_string_view_t processor = contributions[0].processor;
  loom_amdgpu_hsaco_kernel_t* kernels = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, contribution_count, sizeof(kernels[0]), (void**)&kernels));
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_amdgpu_kernel_hsaco_contribution_t* contribution =
        &contributions[i];
    if (!iree_string_view_equal(contribution->artifact_target_key,
                                artifact_target_key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel contribution %" PRIhsz
          " artifact target '%.*s' does not match batch target '%.*s'",
          i, (int)contribution->artifact_target_key.size,
          contribution->artifact_target_key.data, (int)artifact_target_key.size,
          artifact_target_key.data);
    }
    if (!iree_string_view_equal(contribution->code_object_target_id,
                                code_object_target_id)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel contribution %" PRIhsz
          " code-object target '%.*s' does not match batch target '%.*s'",
          i, (int)contribution->code_object_target_id.size,
          contribution->code_object_target_id.data,
          (int)code_object_target_id.size, code_object_target_id.data);
    }
    if (!iree_string_view_equal(contribution->processor, processor)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel contribution %" PRIhsz
          " processor '%.*s' does not match batch processor '%.*s'",
          i, (int)contribution->processor.size, contribution->processor.data,
          (int)processor.size, processor.data);
    }
    kernels[i] = contribution->kernel;
  }

  const loom_amdgpu_hsaco_file_t file = {
      .target = code_object_target_id,
      .processor = processor,
      .kernels = kernels,
      .kernel_count = contribution_count,
      .data_symbols = options ? options->data_symbols : NULL,
      .data_symbol_count = options ? options->data_symbol_count : 0,
  };
  return loom_amdgpu_hsaco_write_file(&file, stream, scratch_arena);
}

iree_status_t loom_amdgpu_emit_kernel_hsaco(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  loom_amdgpu_kernel_hsaco_options_t contribution_options = *options;
  contribution_options.summary = NULL;
  if (options->summary != NULL) {
    *options->summary = (loom_amdgpu_kernel_hsaco_summary_t){0};
  }

  loom_amdgpu_kernel_hsaco_contribution_t contribution = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_kernel_hsaco_contribution(
      schedule, allocation, &contribution_options, &contribution,
      scratch_arena));
  const loom_amdgpu_kernel_hsaco_write_options_t write_options = {
      .data_symbols = options->data_symbols,
      .data_symbol_count = options->data_symbol_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_write_kernel_hsaco_contributions(
      &contribution, 1, &write_options, stream, scratch_arena));
  if (options->summary != NULL) {
    *options->summary = contribution.summary;
  }
  return iree_ok_status();
}
