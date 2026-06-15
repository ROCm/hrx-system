// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_assembly.h"

#include <inttypes.h>

#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/target/emit/native/amdgpu/kernel_record.h"
#include "loom/target/emit/native/amdgpu/metadata.h"

#define LOOM_AMDGPU_KERNEL_ASSEMBLY_CODE_OBJECT_VERSION 5u

static iree_status_t loom_amdgpu_kernel_assembly_accum_offset(
    uint32_t vgpr_count, uint32_t* out_accum_offset) {
  const uint32_t normalized_vgpr_count = vgpr_count == 0 ? 1u : vgpr_count;
  if (normalized_vgpr_count > UINT32_MAX - 3u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel assembly accum-offset VGPR count overflows");
  }
  *out_accum_offset = (normalized_vgpr_count + 3u) & ~UINT32_C(3);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_assembly_append_metadata(
    const loom_amdgpu_kernel_record_t* record, iree_string_builder_t* builder) {
  const loom_amdgpu_metadata_kernel_t* kernel = &record->metadata;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n.rodata\n.p2align 6\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ".amdhsa_kernel %.*s\n", (int)record->symbol.size,
      record->symbol.data));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  .amdhsa_group_segment_fixed_size %" PRIu32
      "\n"
      "  .amdhsa_private_segment_fixed_size %" PRIu32 "\n",
      kernel->group_segment_fixed_size, kernel->private_segment_fixed_size));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_kernarg_size %" PRIu32 "\n",
      kernel->kernarg_segment_size));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_user_sgpr_count %" PRIu32 "\n",
      record->user_sgpr_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  .amdhsa_user_sgpr_dispatch_ptr %u\n"
      "  .amdhsa_user_sgpr_kernarg_segment_ptr %u\n",
      iree_any_bit_set(record->descriptor_flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR)
          ? 1u
          : 0u,
      iree_any_bit_set(
          record->descriptor_flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
          ? 1u
          : 0u));
  if (loom_amdgpu_processor_kernel_descriptor_has_flags(
          record->processor,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  .amdhsa_wavefront_size32 %u\n",
        kernel->wavefront_size == 32 ? 1u : 0u));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  .amdhsa_system_sgpr_workgroup_id_x %u\n"
      "  .amdhsa_system_sgpr_workgroup_id_y %u\n"
      "  .amdhsa_system_sgpr_workgroup_id_z %u\n",
      iree_any_bit_set(record->descriptor_flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X)
          ? 1u
          : 0u,
      iree_any_bit_set(record->descriptor_flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y)
          ? 1u
          : 0u,
      iree_any_bit_set(record->descriptor_flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z)
          ? 1u
          : 0u));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_system_vgpr_workitem_id %" PRIu32 "\n",
      record->system_vgpr_workitem_id));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_next_free_vgpr %" PRIu32 "\n", kernel->vgpr_count));
  if (loom_amdgpu_processor_kernel_descriptor_has_flags(
          record->processor,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET)) {
    uint32_t accum_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_accum_offset(
        kernel->vgpr_count, &accum_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  .amdhsa_accum_offset %" PRIu32 "\n", accum_offset));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_next_free_sgpr %" PRIu32 "\n", kernel->sgpr_count));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ".end_amdhsa_kernel\n"));

  const loom_amdgpu_code_object_metadata_t metadata = {
      .target = record->target_id,
      .kernels = &record->metadata,
      .kernel_count = 1,
  };
  return loom_amdgpu_metadata_append_assembly(&metadata, builder);
}

static iree_status_t loom_amdgpu_kernel_assembly_emit(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_assembly_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  loom_amdgpu_kernel_record_t record = {0};
  const loom_amdgpu_kernel_record_options_t record_options = {
      .abi_layout = options ? options->abi_layout : NULL,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build(
      schedule, allocation, &record_options, &record, scratch_arena));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ".text\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ".amdgcn_target \"%.*s\"\n", (int)record.target_id.size,
      record.target_id.data));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ".amdhsa_code_object_version %u\n\n",
      LOOM_AMDGPU_KERNEL_ASSEMBLY_CODE_OBJECT_VERSION));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ".protected %.*s\n"
      ".globl %.*s\n"
      ".p2align 8\n"
      ".type %.*s,@function\n"
      "%.*s:\n",
      (int)record.symbol.size, record.symbol.data, (int)record.symbol.size,
      record.symbol.data, (int)record.symbol.size, record.symbol.data,
      (int)record.symbol.size, record.symbol.data));
  const struct loom_amdgpu_packet_plan_t* packet_plan =
      options ? options->packet_plan : NULL;
  if (packet_plan != NULL) {
    const loom_amdgpu_assembly_fragment_options_t assembly_options = {
        .packet_plan = packet_plan,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_assembly_fragment_with_options(
        schedule, allocation, &assembly_options, builder, scratch_arena));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_assembly_fragment(
        schedule, allocation, builder, scratch_arena));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ".Lfunc_end0:\n"
      ".size %.*s, .Lfunc_end0-%.*s\n",
      (int)record.symbol.size, record.symbol.data, (int)record.symbol.size,
      record.symbol.data));
  return loom_amdgpu_kernel_assembly_append_metadata(&record, builder);
}

iree_status_t loom_amdgpu_emit_kernel_assembly(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  return loom_amdgpu_kernel_assembly_emit(schedule, allocation, NULL, builder,
                                          scratch_arena);
}

iree_status_t loom_amdgpu_emit_kernel_assembly_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_assembly_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  if (options == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel assembly options are required");
  }
  return loom_amdgpu_kernel_assembly_emit(schedule, allocation, options,
                                          builder, scratch_arena);
}
