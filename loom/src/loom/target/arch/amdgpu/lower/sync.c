// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sync.h"

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_kernel_barrier_is_workgroup_acq_rel(
    const loom_op_t* source_op) {
  return loom_kernel_barrier_memory_space(source_op) ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         loom_kernel_barrier_ordering(source_op) ==
             LOOM_ATOMIC_ORDERING_ACQ_REL &&
         loom_kernel_barrier_scope(source_op) == LOOM_ATOMIC_SCOPE_WORKGROUP;
}

static bool loom_amdgpu_kernel_barrier_has_single_wave_workgroup(
    loom_low_lower_context_t* context) {
  loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  if (!loom_kernel_def_isa(source_function.op)) {
    return false;
  }
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle == NULL || bundle->snapshot == NULL ||
      bundle->snapshot->subgroup_size == 0) {
    return false;
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          loom_low_lower_context_module(context), source_function, bundle,
          &flat_workgroup_size)) {
    return false;
  }
  return flat_workgroup_size <= bundle->snapshot->subgroup_size;
}

static iree_status_t loom_amdgpu_select_kernel_barrier_lds_wait(
    loom_low_lower_context_t* context,
    loom_amdgpu_kernel_barrier_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;

  loom_amdgpu_wait_packet_selection_t selection = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_try_select_counter_mask(
      loom_low_lower_context_descriptor_set(context),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS, /*target_count=*/0, &selection,
      &selected));
  if (!selected) {
    return iree_ok_status();
  }

  loom_amdgpu_explicit_packet_immediate_template_t
      immediates[LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < selection.immediate_count; ++i) {
    immediates[i] = (loom_amdgpu_explicit_packet_immediate_template_t){
        .name = selection.immediates[i].name,
        .value = selection.immediates[i].value,
    };
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_explicit_packet_row_plan(
      context, selection.descriptor, immediates, selection.immediate_count,
      &out_plan->wait));
  out_plan->kind = LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_LDS_WAIT;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_barrier_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_kernel_barrier_isa(source_op)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_kernel_barrier_is_workgroup_acq_rel(source_op)) {
    return iree_ok_status();
  }

  loom_amdgpu_kernel_barrier_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));

  if (loom_amdgpu_kernel_barrier_has_single_wave_workgroup(context)) {
    bool selected = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_select_kernel_barrier_lds_wait(context, plan, &selected));
    if (selected) {
      *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      return iree_ok_status();
    }
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_low_lower_context_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_ok_status();
  }

  plan->kind = LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_S_BARRIER;
  *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_kernel_barrier_plan_t* plan) {
  IREE_ASSERT(plan != NULL);
  switch (plan->kind) {
    case LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_S_BARRIER: {
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
          /*operands=*/NULL, /*operand_count=*/0,
          loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
          /*result_count=*/0, &low_op);
    }
    case LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_LDS_WAIT:
      return loom_amdgpu_emit_explicit_packet_plan(context, source_op,
                                                   &plan->wait);
    case LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "AMDGPU kernel barrier plan has no lowering kind");
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_barrier(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_amdgpu_kernel_barrier_is_workgroup_acq_rel(op)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("barrier.workgroup_scope"));
  }

  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
      IREE_SV("descriptor.s_barrier"));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_collective(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  return loom_amdgpu_low_legality_reject(context, op,
                                         IREE_SV("collective.packet_lowering"));
}
