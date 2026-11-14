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
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
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

// Builds the entry envelope using the launch layout retained before allocation
// and the final private-storage size, including spills. Static envelopes share
// immutable storage; a scratch initializer is owned by |arena|. Targets without
// entry instructions return an empty record.
iree_status_t loom_amdgpu_kernel_entry_envelope_build(
    const loom_amdgpu_processor_properties_t* properties,
    const loom_amdgpu_hal_kernel_abi_verify_result_t* abi,
    uint32_t private_segment_size, iree_arena_allocator_t* arena,
    const loom_amdgpu_kernel_entry_envelope_t** out_envelope);

// Prepends |envelope| to |body_text| and displaces all body-relative fixups.
// Empty envelopes return the original body storage and fixup array directly.
// The entry builder and native encoder own the input storage and fixup bounds.
// Only combined-storage capacity and allocation can fail here.
iree_status_t loom_amdgpu_kernel_entry_prepend_text(
    const loom_amdgpu_kernel_entry_envelope_t* envelope,
    iree_const_byte_span_t body_text,
    const loom_amdgpu_hsaco_text_fixup_t* body_fixups,
    iree_host_size_t body_fixup_count, iree_arena_allocator_t* arena,
    iree_const_byte_span_t* out_text,
    const loom_amdgpu_hsaco_text_fixup_t** out_fixups);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ENTRY_H_
