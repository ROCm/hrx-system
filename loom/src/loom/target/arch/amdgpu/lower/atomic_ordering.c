// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/atomic_ordering.h"

#include "loom/ops/atomic.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_coherence.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_atomic_ordering_has_acquire(uint8_t ordering) {
  switch (ordering) {
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_ordering_has_release(uint8_t ordering) {
  switch (ordering) {
    case LOOM_ATOMIC_ORDERING_RELEASE:
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_source_has_acquire_ordering(
    const loom_low_source_memory_access_plan_t* source) {
  return loom_amdgpu_atomic_ordering_has_acquire(source->atomic.ordering) ||
         loom_amdgpu_atomic_ordering_has_acquire(
             source->atomic.failure_ordering);
}

static bool loom_amdgpu_atomic_source_has_release_ordering(
    const loom_low_source_memory_access_plan_t* source) {
  return loom_amdgpu_atomic_ordering_has_release(source->atomic.ordering);
}

static bool loom_amdgpu_atomic_global_ordering_supported(
    const loom_low_descriptor_set_t* descriptor_set, uint8_t ordering) {
  if (ordering != LOOM_ATOMIC_ORDERING_ACQUIRE &&
      ordering != LOOM_ATOMIC_ORDERING_RELEASE &&
      ordering != LOOM_ATOMIC_ORDERING_ACQ_REL &&
      ordering != LOOM_ATOMIC_ORDERING_SEQ_CST) {
    return false;
  }
  return loom_amdgpu_memory_coherence_rule(descriptor_set) != NULL;
}

static bool loom_amdgpu_atomic_memory_space_is_device_visible(
    loom_value_fact_memory_space_t memory_space) {
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
         memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC;
}

bool loom_amdgpu_atomic_scope_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_source_memory_access_plan_t* source,
    loom_type_t value_type) {
  if (source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return source->atomic.scope == LOOM_ATOMIC_SCOPE_WORKGROUP;
  }
  if (!loom_amdgpu_atomic_memory_space_is_device_visible(
          source->memory_space)) {
    return false;
  }
  if (source->atomic.scope == LOOM_ATOMIC_SCOPE_DEVICE) {
    return true;
  }
  const int32_t bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(value_type));
  if (source->atomic.scope != LOOM_ATOMIC_SCOPE_SYSTEM ||
      !loom_type_is_scalar(value_type) ||
      (bit_count != 32 && bit_count != 64)) {
    return false;
  }
  // System updates require a coherence recipe and backing that admits the
  // operation; mapping admission belongs to the runtime. Native floating
  // arithmetic also passes the numerical and memory-domain capability gate;
  // bitwise exchange and compare-exchange preserve either carrier width.
  return loom_amdgpu_memory_coherence_rule(descriptor_set) != NULL;
}

static bool loom_amdgpu_atomic_ordering_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space, uint8_t ordering) {
  if (ordering == LOOM_ATOMIC_ORDERING_RELAXED) {
    return true;
  }
  if (loom_amdgpu_atomic_memory_space_is_device_visible(memory_space)) {
    return loom_amdgpu_atomic_global_ordering_supported(descriptor_set,
                                                        ordering);
  }
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  return ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
         ordering == LOOM_ATOMIC_ORDERING_RELEASE ||
         ordering == LOOM_ATOMIC_ORDERING_ACQ_REL ||
         ordering == LOOM_ATOMIC_ORDERING_SEQ_CST;
}

bool loom_amdgpu_atomic_orderings_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_source_memory_access_plan_t* source) {
  return loom_amdgpu_atomic_ordering_supported(
             descriptor_set, source->memory_space, source->atomic.ordering) &&
         loom_amdgpu_atomic_ordering_supported(descriptor_set,
                                               source->memory_space,
                                               source->atomic.failure_ordering);
}

static bool loom_amdgpu_atomic_append_wait_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t counter_mask,
    loom_amdgpu_atomic_explicit_packet_selection_t* waits,
    iree_host_size_t wait_capacity, iree_host_size_t* inout_wait_count) {
  loom_amdgpu_wait_packet_selection_t selection = {0};
  if (!loom_amdgpu_wait_packet_try_select_counter_mask(
          descriptor_set, counter_mask, /*target_count=*/0, &selection)) {
    return false;
  }
  IREE_ASSERT(*inout_wait_count < wait_capacity);
  IREE_ASSERT(selection.immediate_count <=
              LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY);
  loom_amdgpu_atomic_explicit_packet_selection_t* wait =
      &waits[(*inout_wait_count)++];
  *wait = (loom_amdgpu_atomic_explicit_packet_selection_t){
      .descriptor_ref = selection.descriptor_ref,
      .immediate_count = selection.immediate_count,
  };
  for (iree_host_size_t i = 0; i < selection.immediate_count; ++i) {
    wait->immediates[i] = (loom_amdgpu_explicit_packet_immediate_template_t){
        .name = selection.immediates[i].name,
        .value = selection.immediates[i].value,
    };
  }
  return true;
}

static loom_amdgpu_atomic_explicit_packet_selection_t
loom_amdgpu_atomic_select_cache_packet(
    const loom_amdgpu_memory_coherence_rule_t* rule,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_cache_scope_t scope) {
  loom_amdgpu_atomic_explicit_packet_selection_t packet = {
      .descriptor_ref = descriptor_ref,
  };
  loom_amdgpu_memory_coherence_attr_t
      attrs[LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_CAPACITY];
  packet.immediate_count = loom_amdgpu_memory_coherence_select_attrs(
      rule->cache_attrs[scope == LOOM_CACHE_SCOPE_SYSTEM], scope, attrs);
  for (iree_host_size_t i = 0; i < packet.immediate_count; ++i) {
    packet.immediates[i] = (loom_amdgpu_explicit_packet_immediate_template_t){
        .name = attrs[i].name,
        .value = attrs[i].value,
    };
  }
  return packet;
}

static bool loom_amdgpu_atomic_select_global_release_packets(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_coherence_rule_t* rule, loom_cache_scope_t scope,
    loom_amdgpu_atomic_ordering_selection_t* ordering) {
  for (iree_host_size_t i = 0; i < rule->release_wait_count; ++i) {
    if (!loom_amdgpu_atomic_append_wait_counter_mask(
            descriptor_set, rule->release_wait_masks[i],
            ordering->pre_atomic_packets,
            IREE_ARRAYSIZE(ordering->pre_atomic_packets),
            &ordering->pre_atomic_packet_count)) {
      return false;
    }
  }
  if (rule->writeback != 0 && scope >= rule->writeback_scope) {
    ordering->pre_atomic_packets[ordering->pre_atomic_packet_count++] =
        loom_amdgpu_atomic_select_cache_packet(rule, rule->writeback, scope);
    return loom_amdgpu_atomic_append_wait_counter_mask(
        descriptor_set, rule->writeback_wait_mask, ordering->pre_atomic_packets,
        IREE_ARRAYSIZE(ordering->pre_atomic_packets),
        &ordering->pre_atomic_packet_count);
  }
  return true;
}

static bool loom_amdgpu_atomic_select_global_acquire_waits(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_atomic_ordering_selection_t* ordering,
    loom_amdgpu_atomic_operation_kind_t operation_kind) {
  const uint32_t counter_mask =
      operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE
          ? LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE
          : LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD;
  return loom_amdgpu_atomic_append_wait_counter_mask(
      descriptor_set, counter_mask, ordering->post_atomic_waits,
      IREE_ARRAYSIZE(ordering->post_atomic_waits),
      &ordering->post_atomic_wait_count);
}

static bool loom_amdgpu_atomic_select_global_acquire_cache_controls(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_coherence_rule_t* rule, loom_cache_scope_t scope,
    loom_amdgpu_atomic_ordering_selection_t* ordering) {
  for (iree_host_size_t i = 0; i < rule->invalidate_count; ++i) {
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            rule->invalidates[i])) {
      return false;
    }
    ordering->post_atomic_visibility_packets[i] =
        loom_amdgpu_atomic_select_cache_packet(rule, rule->invalidates[i],
                                               scope);
  }
  ordering->post_atomic_visibility_packet_count = rule->invalidate_count;
  if (rule->invalidate_wait_mask != 0) {
    return loom_amdgpu_atomic_append_wait_counter_mask(
        descriptor_set, rule->invalidate_wait_mask,
        ordering->post_atomic_visibility_packets,
        IREE_ARRAYSIZE(ordering->post_atomic_visibility_packets),
        &ordering->post_atomic_visibility_packet_count);
  }
  return true;
}

bool loom_amdgpu_atomic_select_ordering(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_atomic_operation_kind_t operation_kind,
    loom_amdgpu_atomic_ordering_selection_t* ordering) {
  *ordering = (loom_amdgpu_atomic_ordering_selection_t){0};
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  if (!loom_amdgpu_atomic_memory_space_is_device_visible(
          source->memory_space) ||
      (!loom_amdgpu_atomic_source_has_release_ordering(source) &&
       !loom_amdgpu_atomic_source_has_acquire_ordering(source))) {
    return true;
  }
  if (rule == NULL) {
    return false;
  }

  if (loom_amdgpu_atomic_source_has_release_ordering(source)) {
    if (!loom_amdgpu_atomic_select_global_release_packets(
            descriptor_set, rule,
            loom_amdgpu_memory_coherence_scope(source->atomic.scope),
            ordering)) {
      return false;
    }
  }
  if (loom_amdgpu_atomic_source_has_acquire_ordering(source)) {
    if (!loom_amdgpu_atomic_select_global_acquire_waits(
            descriptor_set, ordering, operation_kind)) {
      return false;
    }
    if (source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC &&
        rule->local_wait_mask != 0 &&
        !loom_amdgpu_atomic_append_wait_counter_mask(
            descriptor_set, rule->local_wait_mask, ordering->post_atomic_waits,
            IREE_ARRAYSIZE(ordering->post_atomic_waits),
            &ordering->post_atomic_wait_count)) {
      return false;
    }
    if (!loom_amdgpu_atomic_select_global_acquire_cache_controls(
            descriptor_set, rule,
            loom_amdgpu_memory_coherence_scope(source->atomic.scope),
            ordering)) {
      return false;
    }
  }
  return true;
}

loom_amdgpu_memory_coherence_attr_t loom_amdgpu_atomic_select_packet_attr(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_source_memory_access_plan_t* source) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  if (!rule || !loom_amdgpu_atomic_memory_space_is_device_visible(
                   source->memory_space)) {
    return (loom_amdgpu_memory_coherence_attr_t){0};
  }
  loom_amdgpu_memory_coherence_attr_t
      attrs[LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_CAPACITY];
  const uint8_t count = loom_amdgpu_memory_coherence_select_attrs(
      rule->atomic_attrs[source->atomic.scope == LOOM_ATOMIC_SCOPE_SYSTEM],
      loom_amdgpu_memory_coherence_scope(source->atomic.scope), attrs);
  // Atomic plans retain one scope field; return control stays in descriptors.
  IREE_ASSERT_LE(count, 1);
  return count ? attrs[0] : (loom_amdgpu_memory_coherence_attr_t){0};
}

static iree_status_t loom_amdgpu_atomic_resolve_explicit_packet_selection(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_explicit_packet_selection_t* selection,
    loom_amdgpu_explicit_packet_plan_t* out_plan) {
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_explicit_packet_plan(
      context, selection->descriptor_ref, selection->immediates,
      selection->immediate_count, out_plan, &present));
  if (!present) {
    IREE_ASSERT_UNREACHABLE(
        "selected AMDGPU explicit atomic ordering packet descriptor");
    IREE_BUILTIN_UNREACHABLE();
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_atomic_resolve_ordering_selection(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_ordering_selection_t* selection,
    loom_amdgpu_atomic_ordering_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_atomic_ordering_plan_t){0};
  out_plan->pre_atomic_packet_count = selection->pre_atomic_packet_count;
  for (iree_host_size_t i = 0; i < selection->pre_atomic_packet_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_atomic_resolve_explicit_packet_selection(
        context, &selection->pre_atomic_packets[i],
        &out_plan->pre_atomic_packets[i]));
  }
  out_plan->post_atomic_wait_count = selection->post_atomic_wait_count;
  for (iree_host_size_t i = 0; i < selection->post_atomic_wait_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_atomic_resolve_explicit_packet_selection(
        context, &selection->post_atomic_waits[i],
        &out_plan->post_atomic_waits[i]));
  }
  out_plan->post_atomic_visibility_packet_count =
      selection->post_atomic_visibility_packet_count;
  for (iree_host_size_t i = 0;
       i < selection->post_atomic_visibility_packet_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_atomic_resolve_explicit_packet_selection(
        context, &selection->post_atomic_visibility_packets[i],
        &out_plan->post_atomic_visibility_packets[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_atomic_ordering_packets(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_packet_plan_t* packets,
    iree_host_size_t packet_count) {
  for (iree_host_size_t i = 0; i < packet_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_explicit_packet_plan(context, source_op, &packets[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_atomic_pre_ordering(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_ordering_plan_t* ordering) {
  return loom_amdgpu_emit_atomic_ordering_packets(
      context, source_op, ordering->pre_atomic_packets,
      ordering->pre_atomic_packet_count);
}

iree_status_t loom_amdgpu_emit_atomic_post_ordering(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_ordering_plan_t* ordering) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_ordering_packets(
      context, source_op, ordering->post_atomic_waits,
      ordering->post_atomic_wait_count));
  return loom_amdgpu_emit_atomic_ordering_packets(
      context, source_op, ordering->post_atomic_visibility_packets,
      ordering->post_atomic_visibility_packet_count);
}
