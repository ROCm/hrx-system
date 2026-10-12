// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_entry.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

// This entry profile uses an unclaused VMEM followed by V_NOP. Replay mode
// establishes the multi-group XNACK behavior assumed by XCNT wait insertion
// before the scheduled body performs any VGPR-MSB transitions.
static const char loom_amdgpu_initial_vmem_replay_entry_assembly[] =
    "  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE\n"
    "  v_nop\n"
    "  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1\n";

static const uint8_t loom_amdgpu_initial_vmem_replay_entry_text[] = {
    0x00, 0x40, 0x17, 0xee, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x7e, 0x41, 0x06, 0x80, 0xb9, 0x01, 0x00, 0x00, 0x00,
};

static const loom_amdgpu_kernel_entry_envelope_t
    loom_amdgpu_initial_vmem_replay_entry_envelope = {
        .assembly =
            {
                .data = loom_amdgpu_initial_vmem_replay_entry_assembly,
                .size =
                    sizeof(loom_amdgpu_initial_vmem_replay_entry_assembly) - 1u,
            },
        .text =
            {
                .data = loom_amdgpu_initial_vmem_replay_entry_text,
                .data_length =
                    sizeof(loom_amdgpu_initial_vmem_replay_entry_text),
            },
        .instruction_count = 3u,
        .minimum_sgpr_count = 2u,
        .minimum_vgpr_count = 1u,
};

// GFX9 SOP2 encodings for the absolute flat-scratch base (MI200 ISA 10.8,
// LLVM AMDGPUUsage "Flat Scratch"):
//   s_add_u32  flat_scratch_lo, s[init],   s[wave_offset]
//   s_addc_u32 flat_scratch_hi, s[init+1], 0
// SALU writes of FLAT_SCRATCH need no wait states before scratch access.
#define LOOM_AMDGPU_KERNEL_ENTRY_SOP2_ENCODING 0x80000000u
#define LOOM_AMDGPU_KERNEL_ENTRY_SOP2_OP_SHIFT 23u
#define LOOM_AMDGPU_KERNEL_ENTRY_SOP2_SDST_SHIFT 16u
#define LOOM_AMDGPU_KERNEL_ENTRY_SOP2_SSRC1_SHIFT 8u
#define LOOM_AMDGPU_KERNEL_ENTRY_GFX9_S_ADD_U32 0u
#define LOOM_AMDGPU_KERNEL_ENTRY_GFX9_S_ADDC_U32 4u
#define LOOM_AMDGPU_KERNEL_ENTRY_GFX9_FLAT_SCRATCH_LO 102u
#define LOOM_AMDGPU_KERNEL_ENTRY_GFX9_FLAT_SCRATCH_HI 103u
#define LOOM_AMDGPU_KERNEL_ENTRY_INLINE_ZERO 128u
#define LOOM_AMDGPU_KERNEL_ENTRY_FLAT_SCRATCH_TEXT_LENGTH 8u
#define LOOM_AMDGPU_KERNEL_ENTRY_FLAT_SCRATCH_ASSEMBLY_CAPACITY 96u

static uint32_t loom_amdgpu_kernel_entry_sop2(uint32_t opcode, uint32_t sdst,
                                              uint32_t ssrc0, uint32_t ssrc1) {
  return LOOM_AMDGPU_KERNEL_ENTRY_SOP2_ENCODING |
         (opcode << LOOM_AMDGPU_KERNEL_ENTRY_SOP2_OP_SHIFT) |
         (sdst << LOOM_AMDGPU_KERNEL_ENTRY_SOP2_SDST_SHIFT) |
         (ssrc1 << LOOM_AMDGPU_KERNEL_ENTRY_SOP2_SSRC1_SHIFT) | ssrc0;
}

static void loom_amdgpu_kernel_entry_store_u32le(uint8_t* bytes,
                                                 uint32_t value) {
  bytes[0] = (uint8_t)value;
  bytes[1] = (uint8_t)(value >> 8);
  bytes[2] = (uint8_t)(value >> 16);
  bytes[3] = (uint8_t)(value >> 24);
}

static const loom_amdgpu_kernel_entry_envelope_t*
loom_amdgpu_kernel_entry_profile_envelope(
    loom_amdgpu_kernel_entry_profile_t profile) {
  static const loom_amdgpu_kernel_entry_envelope_t empty_envelope = {0};
  switch (profile) {
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE:
      return &empty_envelope;
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY:
      return &loom_amdgpu_initial_vmem_replay_entry_envelope;
    default:
      IREE_CHECK_UNREACHABLE("unknown AMDGPU kernel entry profile");
      return &empty_envelope;
  }
}

loom_amdgpu_kernel_entry_plan_t loom_amdgpu_kernel_entry_plan_for_properties(
    const loom_amdgpu_processor_properties_t* properties) {
  return (loom_amdgpu_kernel_entry_plan_t){
      .profile = properties->kernel_entry.profile,
      .flat_scratch_init_sgpr = LOOM_AMDGPU_KERNEL_ENTRY_NO_SGPR,
      .private_segment_wave_offset_sgpr = LOOM_AMDGPU_KERNEL_ENTRY_NO_SGPR,
  };
}

iree_status_t loom_amdgpu_kernel_entry_envelope_build(
    const loom_amdgpu_kernel_entry_plan_t* plan, iree_arena_allocator_t* arena,
    loom_amdgpu_kernel_entry_envelope_t* out_envelope) {
  const loom_amdgpu_kernel_entry_envelope_t* profile_envelope =
      loom_amdgpu_kernel_entry_profile_envelope(plan->profile);
  if (plan->flat_scratch_init_sgpr == LOOM_AMDGPU_KERNEL_ENTRY_NO_SGPR) {
    *out_envelope = *profile_envelope;
    return iree_ok_status();
  }
  const uint32_t init_sgpr = plan->flat_scratch_init_sgpr;
  const uint32_t wave_offset_sgpr = plan->private_segment_wave_offset_sgpr;
  IREE_ASSERT(
      init_sgpr + 1u < LOOM_AMDGPU_KERNEL_ENTRY_GFX9_FLAT_SCRATCH_LO &&
          wave_offset_sgpr < LOOM_AMDGPU_KERNEL_ENTRY_GFX9_FLAT_SCRATCH_LO,
      "flat-scratch launch inputs must be addressable SGPRs");

  const iree_host_size_t text_length =
      profile_envelope->text.data_length +
      LOOM_AMDGPU_KERNEL_ENTRY_FLAT_SCRATCH_TEXT_LENGTH;
  uint8_t* text = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, text_length, (void**)&text));
  if (profile_envelope->text.data_length != 0) {
    memcpy(text, profile_envelope->text.data,
           profile_envelope->text.data_length);
  }
  uint8_t* scratch_text = text + profile_envelope->text.data_length;
  loom_amdgpu_kernel_entry_store_u32le(
      scratch_text, loom_amdgpu_kernel_entry_sop2(
                        LOOM_AMDGPU_KERNEL_ENTRY_GFX9_S_ADD_U32,
                        LOOM_AMDGPU_KERNEL_ENTRY_GFX9_FLAT_SCRATCH_LO,
                        init_sgpr, wave_offset_sgpr));
  loom_amdgpu_kernel_entry_store_u32le(
      scratch_text + 4,
      loom_amdgpu_kernel_entry_sop2(
          LOOM_AMDGPU_KERNEL_ENTRY_GFX9_S_ADDC_U32,
          LOOM_AMDGPU_KERNEL_ENTRY_GFX9_FLAT_SCRATCH_HI, init_sgpr + 1u,
          LOOM_AMDGPU_KERNEL_ENTRY_INLINE_ZERO));

  const iree_host_size_t assembly_capacity =
      profile_envelope->assembly.size +
      LOOM_AMDGPU_KERNEL_ENTRY_FLAT_SCRATCH_ASSEMBLY_CAPACITY;
  char* assembly = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, assembly_capacity, (void**)&assembly));
  if (profile_envelope->assembly.size != 0) {
    memcpy(assembly, profile_envelope->assembly.data,
           profile_envelope->assembly.size);
  }
  const int scratch_assembly_length =
      snprintf(assembly + profile_envelope->assembly.size,
               LOOM_AMDGPU_KERNEL_ENTRY_FLAT_SCRATCH_ASSEMBLY_CAPACITY,
               "  s_add_u32 flat_scratch_lo, s%" PRIu32 ", s%" PRIu32
               "\n"
               "  s_addc_u32 flat_scratch_hi, s%" PRIu32 ", 0\n",
               init_sgpr, wave_offset_sgpr, init_sgpr + 1u);
  IREE_ASSERT(scratch_assembly_length > 0 &&
              scratch_assembly_length <
                  (int)LOOM_AMDGPU_KERNEL_ENTRY_FLAT_SCRATCH_ASSEMBLY_CAPACITY);

  uint32_t minimum_sgpr_count = iree_max(init_sgpr + 2u, wave_offset_sgpr + 1u);
  minimum_sgpr_count =
      iree_max(minimum_sgpr_count, profile_envelope->minimum_sgpr_count);
  *out_envelope = (loom_amdgpu_kernel_entry_envelope_t){
      .assembly = iree_make_string_view(
          assembly, profile_envelope->assembly.size +
                        (iree_host_size_t)scratch_assembly_length),
      .text = iree_make_const_byte_span(text, text_length),
      .instruction_count = profile_envelope->instruction_count + 2u,
      .minimum_sgpr_count = minimum_sgpr_count,
      .minimum_vgpr_count = profile_envelope->minimum_vgpr_count,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_kernel_entry_prepend_text(
    const loom_amdgpu_kernel_entry_envelope_t* envelope,
    iree_const_byte_span_t body_text,
    const loom_amdgpu_hsaco_text_fixup_t* body_fixups,
    iree_host_size_t body_fixup_count, iree_const_byte_span_t* out_text,
    const loom_amdgpu_hsaco_text_fixup_t** out_fixups,
    iree_arena_allocator_t* arena) {
  *out_text = iree_const_byte_span_empty();
  *out_fixups = NULL;
  if (envelope == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel entry envelope is required");
  }
  if (body_text.data_length != 0 && body_text.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel body has no text storage");
  }
  if (body_fixup_count != 0 && body_fixups == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel body has no text fixup storage");
  }
  if (envelope->text.data_length != 0 && envelope->text.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel entry has no text storage");
  }
  if (envelope->text.data_length == 0) {
    *out_text = body_text;
    *out_fixups = body_fixups;
    return iree_ok_status();
  }

  iree_host_size_t text_length = 0;
  if (!iree_host_size_checked_add(envelope->text.data_length,
                                  body_text.data_length, &text_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU kernel entry text length overflowed");
  }
  uint8_t* text = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, text_length, (void**)&text));
  memcpy(text, envelope->text.data, envelope->text.data_length);
  if (body_text.data_length != 0) {
    memcpy(text + envelope->text.data_length, body_text.data,
           body_text.data_length);
  }

  loom_amdgpu_hsaco_text_fixup_t* fixups = NULL;
  if (body_fixup_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, body_fixup_count, sizeof(fixups[0]), (void**)&fixups));
    for (iree_host_size_t i = 0; i < body_fixup_count; ++i) {
      fixups[i] = body_fixups[i];
      if (!iree_checked_add_u64(fixups[i].literal_byte_offset,
                                envelope->text.data_length,
                                &fixups[i].literal_byte_offset) ||
          !iree_checked_add_u64(fixups[i].base_pc_byte_offset,
                                envelope->text.data_length,
                                &fixups[i].base_pc_byte_offset)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU kernel entry text fixup offset overflowed");
      }
    }
  }

  *out_text = iree_make_const_byte_span(text, text_length);
  *out_fixups = fixups;
  return iree_ok_status();
}
