// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_entry.h"

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

static const loom_amdgpu_kernel_entry_envelope_t
    loom_amdgpu_empty_entry_envelope = {0};

iree_status_t loom_amdgpu_kernel_entry_envelope_build(
    const loom_amdgpu_processor_properties_t* properties,
    const loom_amdgpu_hal_kernel_abi_verify_result_t* abi,
    uint32_t private_segment_size, iree_arena_allocator_t* arena,
    const loom_amdgpu_kernel_entry_envelope_t** out_envelope) {
  *out_envelope = &loom_amdgpu_empty_entry_envelope;
  if (private_segment_size != 0 &&
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_USER_FLAT_SCRATCH_INIT)) {
    struct entry_storage_t {
      // Entry record retained by the kernel's native emission record.
      loom_amdgpu_kernel_entry_envelope_t envelope;
      // The two SOP2 instructions initializing the absolute scratch address.
      uint8_t text[8];
      // Assembly for the same instructions; launch SGPR indices fit two digits.
      char assembly[96];
    };
    struct entry_storage_t* storage = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*storage), (void**)&storage));
    const uint32_t init = abi->flat_scratch_sgprs.init;
    const uint32_t offset = abi->flat_scratch_sgprs.wavefront_offset;
    // S_ADD_U32 flat_scratch_lo, init, offset; S_ADDC_U32
    // flat_scratch_hi, init+1, 0. The launch registers die at this entry;
    // allocation may reuse them in the scheduled body.
    const uint8_t text[] = {(uint8_t)init,
                            (uint8_t)offset,
                            0x66,
                            0x80,
                            (uint8_t)(init + 1),
                            0x80,
                            0x67,
                            0x82};
    memcpy(storage->text, text, sizeof(text));
    const int assembly_length =
        snprintf(storage->assembly, sizeof(storage->assembly),
                 "  s_add_u32 flat_scratch_lo, s%u, s%u\n"
                 "  s_addc_u32 flat_scratch_hi, s%u, 0\n",
                 init, offset, init + 1);
    storage->envelope = (loom_amdgpu_kernel_entry_envelope_t){
        .assembly = iree_make_string_view(storage->assembly, assembly_length),
        .text = iree_make_const_byte_span(storage->text, sizeof(storage->text)),
        .instruction_count = 2,
        .minimum_sgpr_count = offset + 1,
    };
    *out_envelope = &storage->envelope;
    return iree_ok_status();
  }
  switch (properties->kernel_entry.profile) {
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE:
      break;
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY:
      *out_envelope = &loom_amdgpu_initial_vmem_replay_entry_envelope;
      break;
    default:
      IREE_CHECK_UNREACHABLE("unknown AMDGPU kernel entry profile");
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_kernel_entry_prepend_text(
    const loom_amdgpu_kernel_entry_envelope_t* envelope,
    iree_const_byte_span_t body_text,
    const loom_amdgpu_hsaco_text_fixup_t* body_fixups,
    iree_host_size_t body_fixup_count, iree_arena_allocator_t* arena,
    iree_const_byte_span_t* out_text,
    const loom_amdgpu_hsaco_text_fixup_t** out_fixups) {
  *out_text = iree_const_byte_span_empty();
  *out_fixups = NULL;
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
    // Encoded body-relative offsets fit the combined text admitted above.
    for (iree_host_size_t i = 0; i < body_fixup_count; ++i) {
      fixups[i] = body_fixups[i];
      fixups[i].literal_byte_offset += envelope->text.data_length;
      fixups[i].base_pc_byte_offset += envelope->text.data_length;
    }
  }

  *out_text = iree_make_const_byte_span(text, text_length);
  *out_fixups = fixups;
  return iree_ok_status();
}
