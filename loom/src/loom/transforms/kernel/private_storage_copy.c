// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/private_storage_copy.h"

#include "iree/base/checked_math.h"
#include "loom/analysis/loop_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/view/ops.h"

// A resultless loop cannot export its intermediate values. Allow pure
// address computations and projections, while preserving any other effect,
// storage identity, or control transfer through ordinary loop planning.
static bool loom_private_storage_copy_body(const loom_module_t* module,
                                           loom_block_t* block,
                                           const loom_op_t* load,
                                           const loom_op_t* store) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op == load || op == store || loom_scf_yield_isa(op)) {
      continue;
    }
    const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
    if (op->region_count || !(traits & LOOM_TRAIT_PURE) ||
        loom_traits_order_reads(traits) ||
        iree_any_bit_set(traits,
                         LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_TERMINATOR)) {
      return false;
    }
  }
  return true;
}

// Computes the first address in iteration order and its signed byte stride.
static bool loom_private_storage_copy_walk(const loom_symbolic_expr_t* address,
                                           loom_value_id_t induction_variable,
                                           int64_t initial, int64_t step,
                                           uint64_t trips, int64_t* out_start,
                                           int64_t* out_stride) {
  if (trips == 1 && loom_symbolic_expr_is_constant(address)) {
    *out_start = address->constant;
    *out_stride = 0;
    return true;
  }
  if (!loom_symbolic_expr_is_linear(address) || address->term_count != 1 ||
      address->terms[0].value_id != induction_variable) {
    return false;
  }
  const int64_t coefficient = address->terms[0].coefficient;
  return iree_checked_mul_add_i64(address->constant, coefficient, initial,
                                  out_start) &&
         iree_checked_mul_i64(coefficient, step, out_stride);
}

iree_status_t loom_private_storage_record_copy_loop(
    loom_private_storage_plan_t* plan, loom_view_region_table_t* regions,
    const loom_value_fact_table_t* facts, loom_op_t* loop, loom_op_t* load,
    loom_private_storage_allocation_t* source, loom_op_t* store,
    loom_private_storage_allocation_t* target,
    loom_private_storage_copy_t** out_copy) {
  *out_copy = NULL;
  if (!source || !source->selected ||
      loom_traits_have_observable_effects(
          loom_op_effective_traits(plan->module, load)) ||
      loom_traits_have_observable_effects(
          loom_op_effective_traits(plan->module, store)) ||
      !loom_private_storage_copy_body(plan->module, store->parent_block, load,
                                      store)) {
    return iree_ok_status();
  }
  const loom_value_facts_t lower =
      loom_value_fact_table_lookup(facts, loom_scf_for_lower_bound(loop));
  const loom_value_facts_t upper =
      loom_value_fact_table_lookup(facts, loom_scf_for_upper_bound(loop));
  const loom_value_facts_t step =
      loom_value_fact_table_lookup(facts, loom_scf_for_step(loop));
  uint64_t trips = 0;
  if (!loom_value_facts_is_exact(lower) || !loom_value_facts_is_exact(upper) ||
      !loom_value_facts_is_exact(step) || step.range_lo <= 0 ||
      !loom_loop_domain_trip_count(LOOM_LOOP_BOUND_SIGNED, 64, lower.range_lo,
                                   upper.range_lo, step.range_lo, &trips) ||
      trips == 0) {
    return iree_ok_status();
  }
  const loom_type_t type =
      loom_module_value_type(plan->module, loom_view_load_result(load));
  if (!loom_type_is_scalar(type)) {
    return iree_ok_status();
  }
  const int32_t bits = loom_scalar_type_bitwidth(loom_type_element_type(type));
  const int64_t width = bits / 8;
  if (width == 0 || bits % 8 || trips > (uint64_t)(INT64_MAX / width)) {
    return iree_ok_status();
  }
  const int64_t byte_length = (int64_t)trips * width;
  const loom_value_id_t iv =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  const loom_memory_access_t accesses[] = {
      loom_memory_access_cast(plan->module, load),
      loom_memory_access_cast(plan->module, store),
  };
  int64_t starts[2], strides[2];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(accesses); ++i) {
    const loom_memory_access_t access = accesses[i];
    loom_view_region_t region;
    bool derived = false;
    IREE_RETURN_IF_ERROR(loom_view_region_table_derive_element_region(
        regions, loom_memory_access_view(access),
        loom_memory_access_static_indices(access),
        loom_memory_access_dynamic_indices(access), &region, &derived));
    if (!derived || region.static_element_byte_count != width ||
        !loom_private_storage_copy_walk(&region.begin_byte_offset, iv,
                                        lower.range_lo, step.range_lo, trips,
                                        &starts[i], &strides[i])) {
      return iree_ok_status();
    }
    if (trips > 1 && strides[i] != width && strides[i] != -width) {
      return iree_ok_status();
    }
    if (trips > 1 && strides[i] < 0 &&
        !iree_checked_add_i64(starts[i], width - byte_length, &starts[i])) {
      return iree_ok_status();
    }
  }
  if (trips > 1 && strides[0] != strides[1]) {
    return iree_ok_status();
  }
  return loom_private_storage_record_copy(plan, loop, source, starts[0], target,
                                          starts[1], byte_length, out_copy);
}

static loom_private_storage_allocation_t* loom_private_storage_copy_root(
    loom_private_storage_allocation_t* allocation) {
  while (allocation != allocation->copy_component) {
    allocation->copy_component = allocation->copy_component->copy_component;
    allocation = allocation->copy_component;
  }
  return allocation;
}

void loom_private_storage_finalize_copy_components(
    loom_private_storage_plan_t* plan) {
  for (loom_private_storage_allocation_t* allocation = plan->allocations;
       allocation; allocation = allocation->next) {
    allocation->copy_component = loom_private_storage_copy_root(allocation);
    if (!allocation->selected) {
      allocation->copy_component->selected = false;
    }
  }
}

iree_status_t loom_private_storage_record_copy(
    loom_private_storage_plan_t* plan, loom_op_t* op,
    loom_private_storage_allocation_t* source, int64_t source_offset,
    loom_private_storage_allocation_t* target, int64_t target_offset,
    int64_t byte_length, loom_private_storage_copy_t** out_copy) {
  *out_copy = NULL;
  if (!source || !target || !source->selected || !target->selected ||
      source_offset < 0 || target_offset < 0 || byte_length <= 0 ||
      source->byte_count - byte_length < source_offset ||
      target->byte_count - byte_length < target_offset ||
      (source == target && source_offset < target_offset + byte_length &&
       target_offset < source_offset + byte_length)) {
    return iree_ok_status();
  }
  loom_private_storage_copy_t* copy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*copy), (void**)&copy));
  *copy = (loom_private_storage_copy_t){.op = op,
                                        .source = source,
                                        .target = target,
                                        .source_offset = source_offset,
                                        .target_offset = target_offset,
                                        .byte_length = byte_length,
                                        .next = plan->copies};
  plan->copies = copy;
  target->receives_copy = true;
  if (!source->copy_node_ordinal) {
    source->copy_node_ordinal = ++plan->copy_allocation_count;
  }
  if (!target->copy_node_ordinal) {
    target->copy_node_ordinal = ++plan->copy_allocation_count;
  }
  *out_copy = copy;
  // Erasing either endpoint's stores alone could change the other endpoint.
  source->write_only = target->write_only = false;
  source = loom_private_storage_copy_root(source);
  target = loom_private_storage_copy_root(target);
  if (source != target) {
    if (source->copy_rank < target->copy_rank) {
      source->copy_component = target;
    } else {
      target->copy_component = source;
      source->copy_rank += source->copy_rank == target->copy_rank;
    }
  }
  return iree_ok_status();
}

void loom_private_storage_resolve_copy_selection(
    loom_private_storage_plan_t* plan) {
  for (loom_private_storage_copy_t* copy = plan->copies; copy;
       copy = copy->next) {
    if (!copy->source->selected || !copy->target->selected) {
      copy->source->copy_component->selected = false;
    }
  }
  for (loom_private_storage_copy_t* copy = plan->copies; copy;
       copy = copy->next) {
    if (!copy->source->copy_component->selected) {
      copy->source->selected = copy->target->selected = false;
    }
  }
}
