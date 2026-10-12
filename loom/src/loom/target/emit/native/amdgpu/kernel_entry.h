// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned AMDGPU hardware kernel-entry envelopes.
//
// Scheduled target-low instructions describe the semantic kernel body. Some
// processors additionally require native instructions at the hardware entry
// point. This layer owns those instructions, their resource floors, and the
// displacement of body-relative code-object fixups.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ENTRY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ENTRY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/emit/native/amdgpu/text_fixup.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_kernel_entry_envelope_t {
  // Assembly instructions inserted immediately after the entry label.
  iree_string_view_t assembly;
  // Encoded instructions inserted immediately before the kernel body.
  iree_const_byte_span_t text;
  // Number of native instructions in |assembly| and |text|.
  uint32_t instruction_count;
  // Minimum scalar register count required by the entry instructions.
  uint32_t minimum_sgpr_count;
  // Minimum vector register count required by the entry instructions.
  uint32_t minimum_vgpr_count;
} loom_amdgpu_kernel_entry_envelope_t;

// Sentinel for an entry plan without an SGPR input.
#define LOOM_AMDGPU_KERNEL_ENTRY_NO_SGPR UINT32_MAX

// Finalized entry requirements of one kernel, retained by its kernel record
// and consumed identically by assembly and binary emission.
//
// The absolute flat-scratch sequence reads the flat-scratch-init pair and the
// private-segment wave offset, writes FLAT_SCRATCH, and clobbers SCC. None of
// these are live into the scheduled body, so the body allocation may reuse the
// consumed input SGPRs after the entry sequence.
typedef struct loom_amdgpu_kernel_entry_plan_t {
  // Hardware-entry profile selected by the processor.
  loom_amdgpu_kernel_entry_profile_t profile;
  // First SGPR of the flat-scratch-init launch pair, or
  // LOOM_AMDGPU_KERNEL_ENTRY_NO_SGPR when the entry composes no absolute
  // flat-scratch base.
  uint32_t flat_scratch_init_sgpr;
  // System SGPR holding the private-segment wave byte offset. Valid only with
  // |flat_scratch_init_sgpr|.
  uint32_t private_segment_wave_offset_sgpr;
} loom_amdgpu_kernel_entry_plan_t;

// Returns a plan with only |properties|' static entry profile.
loom_amdgpu_kernel_entry_plan_t loom_amdgpu_kernel_entry_plan_for_properties(
    const loom_amdgpu_processor_properties_t* properties);

// Materializes the hardware-entry instructions required by |plan|. Profile
// packets precede the flat-scratch sequence, which precedes every body or
// spill access to private memory. Plan-dependent storage is allocated from
// |arena|; static-only plans borrow immutable storage.
iree_status_t loom_amdgpu_kernel_entry_envelope_build(
    const loom_amdgpu_kernel_entry_plan_t* plan, iree_arena_allocator_t* arena,
    loom_amdgpu_kernel_entry_envelope_t* out_envelope);

// Prepends |envelope| to |body_text| and displaces all body-relative fixups.
// Empty envelopes return the original body storage and fixup array directly.
iree_status_t loom_amdgpu_kernel_entry_prepend_text(
    const loom_amdgpu_kernel_entry_envelope_t* envelope,
    iree_const_byte_span_t body_text,
    const loom_amdgpu_hsaco_text_fixup_t* body_fixups,
    iree_host_size_t body_fixup_count, iree_const_byte_span_t* out_text,
    const loom_amdgpu_hsaco_text_fixup_t** out_fixups,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ENTRY_H_
