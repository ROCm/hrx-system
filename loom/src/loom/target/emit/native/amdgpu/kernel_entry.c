// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_entry.h"

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

const loom_amdgpu_kernel_entry_envelope_t*
loom_amdgpu_kernel_entry_envelope_for_properties(
    const loom_amdgpu_processor_properties_t* properties) {
  switch (properties->kernel_entry.profile) {
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE:
      return &loom_amdgpu_empty_entry_envelope;
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY:
      return &loom_amdgpu_initial_vmem_replay_entry_envelope;
    default:
      IREE_CHECK_UNREACHABLE("unknown AMDGPU kernel entry profile");
      return &loom_amdgpu_empty_entry_envelope;
  }
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
