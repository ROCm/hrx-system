// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/byte_permute.h"

#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

void loom_amdgpu_select_byte_permute_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_byte_permute_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_byte_permute_plan_t){0};
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT)) {
    out_plan->kind = LOOM_AMDGPU_BYTE_PERMUTE_KIND_LITERAL_SELECTOR;
    out_plan->descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT;
    return;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32)) {
    out_plan->kind = LOOM_AMDGPU_BYTE_PERMUTE_KIND_REGISTER_SELECTOR;
    out_plan->descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32;
  }
}

iree_status_t loom_amdgpu_byte_permute_emitter_initialize(
    loom_low_lower_context_t* context,
    const loom_amdgpu_byte_permute_plan_t* plan,
    loom_amdgpu_byte_permute_emitter_t* out_emitter) {
  IREE_ASSERT_NE(plan->kind, LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE);
  out_emitter->plan = *plan;
  out_emitter->selector_type = loom_type_none();
  out_emitter->selector_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, plan->descriptor_ref, &out_emitter->descriptor));
  if (plan->kind == LOOM_AMDGPU_BYTE_PERMUTE_KIND_REGISTER_SELECTOR) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_type(context, &out_emitter->selector_type));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_byte_permute_emitter_emit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_byte_permute_emitter_t* emitter, loom_value_id_t source0,
    loom_value_id_t source1, uint32_t selector, loom_type_t result_type,
    loom_value_id_t* out_value) {
  switch (emitter->plan.kind) {
    case LOOM_AMDGPU_BYTE_PERMUTE_KIND_LITERAL_SELECTOR:
      return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
          context, source_op, &emitter->descriptor, source0, source1, selector,
          result_type, out_value);
    case LOOM_AMDGPU_BYTE_PERMUTE_KIND_REGISTER_SELECTOR: {
      uint32_t selector_index = 0;
      while (selector_index < emitter->selector_count &&
             emitter->selector_cache[selector_index].immediate != selector) {
        ++selector_index;
      }
      loom_amdgpu_byte_permute_selector_cache_entry_t entry;
      uint32_t promotion_index = selector_index;
      if (selector_index == emitter->selector_count) {
        entry.immediate = selector;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, selector,
            emitter->selector_type, &entry.register_value));
        if (emitter->selector_count < IREE_ARRAYSIZE(emitter->selector_cache)) {
          promotion_index = emitter->selector_count++;
        } else {
          promotion_index = emitter->selector_count - 1u;
        }
      } else {
        entry = emitter->selector_cache[selector_index];
      }
      // Keep reuse bounded by the existing fixed working set. Arbitrary
      // shuffles can use more selector payloads than the cache can hold; an
      // evicted selector is safely rematerialized if it appears again.
      for (uint32_t i = promotion_index; i > 0; --i) {
        emitter->selector_cache[i] = emitter->selector_cache[i - 1u];
      }
      emitter->selector_cache[0] = entry;
      return loom_amdgpu_emit_resolved_vgpr_ternary(
          context, source_op, &emitter->descriptor, source0, source1,
          entry.register_value, result_type, out_value);
    }
    case LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU byte permutation kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
