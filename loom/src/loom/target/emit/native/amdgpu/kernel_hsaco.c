// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"

#include <inttypes.h>

#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/emit/native/amdgpu/kernel_record.h"

iree_status_t loom_amdgpu_build_kernel_hsaco_contribution(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* scratch_arena) {
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){0};
  if (options != NULL && options->summary != NULL) {
    *options->summary = (loom_amdgpu_kernel_hsaco_summary_t){0};
  }

  loom_amdgpu_kernel_record_t record = {0};
  const loom_amdgpu_kernel_record_options_t record_options = {
      .abi_layout = options ? options->abi_layout : NULL,
      .preflight = options ? options->preflight : NULL,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build(
      schedule, allocation, &record_options, &record, scratch_arena));

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_packet_plan_t* packet_plan =
      options ? options->packet_plan : NULL;
  if (packet_plan != NULL) {
    const loom_amdgpu_encode_instruction_stream_options_t encode_options = {
        .packet_plan = packet_plan,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_instruction_stream_with_options(
        schedule, allocation, &encode_options, &text, scratch_arena));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_instruction_stream(
        schedule, allocation, &text, scratch_arena));
  }

  const loom_amdgpu_hsaco_kernel_t kernel = {
      .metadata = record.metadata,
      .descriptor_options =
          {
              .flags = record.descriptor_flags,
              .user_sgpr_count = record.user_sgpr_count,
          },
      .text = text,
  };
  const uint64_t instruction_count =
      loom_amdgpu_packet_plan_instruction_count(schedule, packet_plan);
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){
      .target = record.target_id,
      .processor = record.processor->name,
      .kernel = kernel,
      .summary =
          {
              .instruction_count = instruction_count,
              .text_byte_count = text.data_length,
              .text_storage_byte_count = text.data_length,
              .private_segment_fixed_size =
                  record.metadata.private_segment_fixed_size,
              .group_segment_fixed_size =
                  record.metadata.group_segment_fixed_size,
          },
  };

  if (options != NULL && options->summary != NULL) {
    *options->summary = out_contribution->summary;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_write_kernel_hsaco_contributions(
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  if (contribution_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel HSACO requires at least one contribution");
  }

  const iree_string_view_t target = contributions[0].target;
  const iree_string_view_t processor = contributions[0].processor;
  loom_amdgpu_hsaco_kernel_t* kernels = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, contribution_count, sizeof(kernels[0]), (void**)&kernels));
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_amdgpu_kernel_hsaco_contribution_t* contribution =
        &contributions[i];
    if (!iree_string_view_equal(contribution->target, target)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel contribution %" PRIhsz
          " target '%.*s' does not match batch target '%.*s'",
          i, (int)contribution->target.size, contribution->target.data,
          (int)target.size, target.data);
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
      .target = target,
      .processor = processor,
      .kernels = kernels,
      .kernel_count = contribution_count,
  };
  return loom_amdgpu_hsaco_write_file(&file, stream, scratch_arena);
}

iree_status_t loom_amdgpu_emit_kernel_hsaco(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  loom_amdgpu_kernel_hsaco_options_t contribution_options =
      options ? *options : (loom_amdgpu_kernel_hsaco_options_t){0};
  contribution_options.summary = NULL;
  if (options != NULL && options->summary != NULL) {
    *options->summary = (loom_amdgpu_kernel_hsaco_summary_t){0};
  }

  loom_amdgpu_kernel_hsaco_contribution_t contribution = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_kernel_hsaco_contribution(
      schedule, allocation, &contribution_options, &contribution,
      scratch_arena));
  IREE_RETURN_IF_ERROR(loom_amdgpu_write_kernel_hsaco_contributions(
      &contribution, 1, stream, scratch_arena));
  if (options != NULL && options->summary != NULL) {
    *options->summary = contribution.summary;
  }
  return iree_ok_status();
}
