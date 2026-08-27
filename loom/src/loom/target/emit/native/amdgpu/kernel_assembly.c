// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_assembly.h"

#include <inttypes.h>

#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/target/emit/native/amdgpu/kernel_entry.h"
#include "loom/target/emit/native/amdgpu/kernel_record.h"
#include "loom/target/emit/native/amdgpu/metadata.h"

#define LOOM_AMDGPU_KERNEL_ASSEMBLY_CODE_OBJECT_VERSION 6u

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

static uint32_t loom_amdgpu_kernel_assembly_flag(
    const loom_amdgpu_kernel_record_t* record,
    loom_amdgpu_kernel_descriptor_flags_t flag) {
  return iree_any_bit_set(record->descriptor_flags, flag) ? 1u : 0u;
}

static bool loom_amdgpu_kernel_assembly_supports_wgp_mode(
    const loom_amdgpu_processor_info_t* processor) {
  switch (processor->properties.kernel_descriptor.profile) {
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12:
      return true;
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125:
      return false;
  }
  return false;
}

static iree_status_t loom_amdgpu_kernel_assembly_append_metadata(
    const loom_amdgpu_kernel_record_t* record, iree_string_builder_t* builder) {
  const loom_amdgpu_metadata_kernel_t* kernel = &record->metadata;
  const bool has_architected_flat_scratch =
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &record->processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH);
  const bool uses_gfx10_sgpr_encoding =
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &record->processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING);
  const bool has_accum_offset =
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &record->processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET);
  const bool has_dx10_clamp_and_ieee_mode =
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &record->processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE);
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
  if (!has_architected_flat_scratch) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  .amdhsa_user_sgpr_private_segment_buffer %u\n",
        loom_amdgpu_kernel_assembly_flag(
            record,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  .amdhsa_user_sgpr_dispatch_ptr %u\n"
      "  .amdhsa_user_sgpr_queue_ptr %u\n"
      "  .amdhsa_user_sgpr_kernarg_segment_ptr %u\n",
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR),
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_QUEUE_PTR),
      loom_amdgpu_kernel_assembly_flag(
          record,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_user_sgpr_dispatch_id %u\n",
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID)));
  if (!has_architected_flat_scratch) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  .amdhsa_user_sgpr_flat_scratch_init %u\n",
        loom_amdgpu_kernel_assembly_flag(
            record,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT)));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_user_sgpr_private_segment_size %u\n",
      loom_amdgpu_kernel_assembly_flag(
          record,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)));
  if (uses_gfx10_sgpr_encoding) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  .amdhsa_wavefront_size32 %u\n",
        kernel->wavefront_size == 32 ? 1u : 0u));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_uses_dynamic_stack %u\n",
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_%s %u\n",
      has_architected_flat_scratch
          ? "enable_private_segment"
          : "system_sgpr_private_segment_wavefront_offset",
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_PRIVATE_SEGMENT)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  .amdhsa_system_sgpr_workgroup_id_x %u\n"
      "  .amdhsa_system_sgpr_workgroup_id_y %u\n"
      "  .amdhsa_system_sgpr_workgroup_id_z %u\n"
      "  .amdhsa_system_sgpr_workgroup_info %u\n",
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X),
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y),
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z),
      loom_amdgpu_kernel_assembly_flag(
          record, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_INFO)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_system_vgpr_workitem_id %" PRIu32 "\n",
      record->system_vgpr_workitem_id));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_next_free_vgpr %" PRIu32 "\n", kernel->vgpr_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_next_free_sgpr %" PRIu32 "\n", kernel->sgpr_count));
  if (has_accum_offset) {
    uint32_t accum_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_accum_offset(
        kernel->vgpr_count, &accum_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  .amdhsa_accum_offset %" PRIu32 "\n", accum_offset));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "  .amdhsa_reserve_vcc 0\n"));
  if (!has_architected_flat_scratch) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, "  .amdhsa_reserve_flat_scratch 0\n"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "  .amdhsa_float_round_mode_32 0\n"
      "  .amdhsa_float_round_mode_16_64 0\n"
      "  .amdhsa_float_denorm_mode_32 3\n"
      "  .amdhsa_float_denorm_mode_16_64 3\n"));
  if (has_dx10_clamp_and_ieee_mode) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder,
                                           "  .amdhsa_dx10_clamp 1\n"
                                           "  .amdhsa_ieee_mode 1\n"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, "  .amdhsa_fp16_overflow 0\n"));
  if (has_accum_offset) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "  .amdhsa_tg_split 0\n"));
  }
  if (loom_amdgpu_kernel_assembly_supports_wgp_mode(record->processor)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "  .amdhsa_workgroup_processor_mode %u\n", 1u));
  }
  if (uses_gfx10_sgpr_encoding) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder,
                                          "  .amdhsa_memory_ordered %u\n"
                                          "  .amdhsa_forward_progress %u\n",
                                          1u, 1u));
  }
  if (record->processor->properties.kernel_descriptor.profile ==
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder,
                                           "  .amdhsa_shared_vgpr_count 0\n"
                                           "  .amdhsa_inst_pref_size 0\n"));
  }
  if (record->processor->properties.kernel_descriptor.profile ==
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12 ||
      record->processor->properties.kernel_descriptor.profile ==
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder,
        "  .amdhsa_inst_pref_size 0\n"
        "  .amdhsa_round_robin_scheduling 0\n"));
  }
  if (record->processor->properties.kernel_descriptor.profile ==
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, "  .amdhsa_named_barrier_count 0\n"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "  .amdhsa_exception_fp_ieee_invalid_op 0\n"
      "  .amdhsa_exception_fp_denorm_src 0\n"
      "  .amdhsa_exception_fp_ieee_div_zero 0\n"
      "  .amdhsa_exception_fp_ieee_overflow 0\n"
      "  .amdhsa_exception_fp_ieee_underflow 0\n"
      "  .amdhsa_exception_fp_ieee_inexact 0\n"
      "  .amdhsa_exception_int_div_zero 0\n"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ".end_amdhsa_kernel\n"));

  const loom_amdgpu_code_object_metadata_t metadata = {
      .target = record->code_object_target_id,
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
      .abi_layout = options->abi_layout,
      .abi_verify = options->abi_verify,
      .preflight = options->preflight,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build(
      schedule, allocation, &record_options, &record, scratch_arena));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ".text\n"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, ".amdgcn_target \"%.*s\"\n",
                                        (int)record.code_object_target_id.size,
                                        record.code_object_target_id.data));
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
  const loom_amdgpu_kernel_entry_envelope_t* entry_envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(
          &record.processor->properties);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, entry_envelope->assembly));
  const loom_amdgpu_assembly_fragment_options_t assembly_options = {
      .packet_plan = options->packet_plan,
      .storage_layout = &record.storage_layout,
      .branch_layout = options->branch_layout,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_assembly_fragment_with_options(
      schedule, allocation, &assembly_options, builder, scratch_arena));
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
    const loom_amdgpu_kernel_assembly_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  return loom_amdgpu_kernel_assembly_emit(schedule, allocation, options,
                                          builder, scratch_arena);
}
