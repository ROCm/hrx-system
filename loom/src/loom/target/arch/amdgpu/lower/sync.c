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
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_kernel_barrier_is_global_memory(
    const loom_op_t* source_op) {
  return loom_kernel_barrier_memory_space(source_op) ==
         LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
}

static bool loom_amdgpu_kernel_barrier_global_ordering_has_acquire(
    const loom_op_t* source_op) {
  const loom_atomic_ordering_t ordering =
      loom_kernel_barrier_ordering(source_op);
  return ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
         ordering == LOOM_ATOMIC_ORDERING_ACQ_REL;
}

static bool loom_amdgpu_kernel_barrier_global_ordering_has_release(
    const loom_op_t* source_op) {
  const loom_atomic_ordering_t ordering =
      loom_kernel_barrier_ordering(source_op);
  return ordering == LOOM_ATOMIC_ORDERING_RELEASE ||
         ordering == LOOM_ATOMIC_ORDERING_ACQ_REL;
}

static bool loom_amdgpu_kernel_barrier_global_ordering_available(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  if (loom_amdgpu_kernel_barrier_global_ordering_has_release(source_op) &&
      !loom_amdgpu_system_memory_release_ordering_available(descriptor_set)) {
    return false;
  }
  return !loom_amdgpu_kernel_barrier_global_ordering_has_acquire(source_op) ||
         loom_amdgpu_system_memory_acquire_ordering_available(descriptor_set);
}

static bool loom_amdgpu_workgroup_is_single_subgroup(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table) {
  uint32_t flat_workgroup_size = 0;
  return loom_amdgpu_required_flat_workgroup_size_from_facts(
             module, function, bundle, fact_table, &flat_workgroup_size) &&
         flat_workgroup_size <= loom_amdgpu_target_wavefront_size(bundle);
}

static bool loom_amdgpu_kernel_barrier_is_memory_order_only(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table, const loom_op_t* source_op) {
  if (loom_amdgpu_kernel_barrier_is_global_memory(source_op)) {
    return false;
  }
  if (loom_kernel_barrier_scope(source_op) == LOOM_ATOMIC_SCOPE_SUBGROUP) {
    return true;
  }
  return loom_kernel_barrier_scope(source_op) == LOOM_ATOMIC_SCOPE_WORKGROUP &&
         loom_amdgpu_workgroup_is_single_subgroup(module, function, bundle,
                                                  fact_table);
}

static iree_status_t loom_amdgpu_select_kernel_barrier_lds_wait(
    loom_low_lower_context_t* context,
    loom_amdgpu_kernel_barrier_plan_t* out_plan) {
  loom_amdgpu_wait_packet_selection_t selection = {0};
  const bool selected = loom_amdgpu_wait_packet_try_select_counter_mask(
      loom_low_lower_context_descriptor_set(context),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_WORKGROUP, /*target_count=*/0, &selection);
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
  return iree_ok_status();
}

bool loom_amdgpu_workgroup_barrier_lowering_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER)) {
    return true;
  }
  return loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER_SIGNAL_ALL) &&
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER_WAIT_ALL);
}

static bool loom_amdgpu_workgroup_memory_wait_lowering_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  loom_amdgpu_wait_packet_selection_t selection = {0};
  return loom_amdgpu_wait_packet_try_select_counter_mask(
      descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_WORKGROUP,
      /*target_count=*/0, &selection);
}

iree_status_t loom_amdgpu_select_workgroup_barrier_plan(
    loom_low_lower_context_t* context,
    loom_amdgpu_kernel_barrier_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_kernel_barrier_plan_t){0};

  const uint32_t barrier_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_low_lower_context_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER);
  if (barrier_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    out_plan->kind = LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_S_BARRIER;
    return iree_ok_status();
  }

  bool signal_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_explicit_packet_plan(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER_SIGNAL_ALL,
      /*immediates=*/NULL, /*immediate_count=*/0, &out_plan->split_signal,
      &signal_present));
  bool wait_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_explicit_packet_plan(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER_WAIT_ALL,
      /*immediates=*/NULL, /*immediate_count=*/0, &out_plan->split_wait,
      &wait_present));
  if (signal_present && wait_present) {
    out_plan->kind = LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_SPLIT_BARRIER;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_barrier_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_kernel_barrier_isa(source_op)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_kernel_barrier_is_global_memory(source_op) &&
      !loom_amdgpu_kernel_barrier_global_ordering_available(
          loom_low_lower_context_descriptor_set(context), source_op)) {
    return iree_ok_status();
  }

  loom_amdgpu_kernel_barrier_plan_t local_plan = {0};
  const bool is_memory_order_only =
      loom_amdgpu_kernel_barrier_is_memory_order_only(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context),
          loom_low_lower_context_fact_table(context), source_op);
  if (is_memory_order_only) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_select_kernel_barrier_lds_wait(context, &local_plan));
  }
  if (local_plan.kind == LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE &&
      (!is_memory_order_only ||
       loom_kernel_barrier_scope(source_op) == LOOM_ATOMIC_SCOPE_WORKGROUP)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_select_workgroup_barrier_plan(context, &local_plan));
  }
  if (local_plan.kind == LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE) {
    return iree_ok_status();
  }

  loom_amdgpu_kernel_barrier_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = local_plan;
  *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_workgroup_barrier_plan(
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
    case LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_SPLIT_BARRIER: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_explicit_packet_plan(
          context, source_op, &plan->split_signal));
      return loom_amdgpu_emit_explicit_packet_plan(context, source_op,
                                                   &plan->split_wait);
    }
    case LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE:
      IREE_ASSERT_UNREACHABLE("unselected AMDGPU kernel barrier plan");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU kernel barrier lowering kind");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_amdgpu_lower_kernel_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_kernel_barrier_plan_t* plan) {
  if (!loom_amdgpu_kernel_barrier_is_global_memory(source_op)) {
    return loom_amdgpu_lower_workgroup_barrier_plan(context, source_op, plan);
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (loom_amdgpu_kernel_barrier_global_ordering_has_release(source_op)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_system_memory_build_release_ordering_scoped(
            builder, descriptor_set, LOOM_CACHE_SCOPE_DEVICE,
            source_op->location));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lower_workgroup_barrier_plan(context, source_op, plan));
  if (loom_amdgpu_kernel_barrier_global_ordering_has_acquire(source_op)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_system_memory_build_acquire_ordering_scoped(
            builder, descriptor_set, LOOM_CACHE_SCOPE_DEVICE,
            source_op->location));
  }
  return iree_ok_status();
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
  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);

  if (loom_amdgpu_kernel_barrier_is_global_memory(op)) {
    if (loom_amdgpu_kernel_barrier_global_ordering_has_release(op) &&
        !loom_amdgpu_system_memory_release_ordering_available(descriptor_set)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.global_memory_release"));
    }
    if (loom_amdgpu_kernel_barrier_global_ordering_has_acquire(op) &&
        !loom_amdgpu_system_memory_acquire_ordering_available(descriptor_set)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.global_memory_acquire"));
    }
    if (!loom_amdgpu_workgroup_barrier_lowering_available(descriptor_set)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.workgroup_barrier"));
    }
    return iree_ok_status();
  }

  const bool is_memory_order_only =
      loom_amdgpu_kernel_barrier_is_memory_order_only(
          loom_target_low_legality_module(context),
          loom_target_low_legality_function(context), bundle,
          loom_target_low_legality_fact_table(context), op);
  if (is_memory_order_only &&
      loom_amdgpu_workgroup_memory_wait_lowering_available(descriptor_set)) {
    return iree_ok_status();
  }
  if (loom_kernel_barrier_scope(op) == LOOM_ATOMIC_SCOPE_SUBGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.workgroup_memory_wait"));
  }

  if (!loom_amdgpu_workgroup_barrier_lowering_available(descriptor_set)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.workgroup_barrier"));
  }
  return iree_ok_status();
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
