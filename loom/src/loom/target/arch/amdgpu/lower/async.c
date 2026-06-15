// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/async.h"

#include <stdint.h>

#include "loom/ir/facts.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

static_assert(LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY <=
                  LOOM_AMDGPU_ASYNC_WAIT_IMMEDIATE_CAPACITY,
              "async wait plans must hold any selected wait packet immediate");

typedef uint32_t loom_amdgpu_async_gather_rejection_flags_t;

#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_PLAN ((uint32_t)1u << 0)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_MEMORY_SPACE \
  ((uint32_t)1u << 1)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_ADDRESS ((uint32_t)1u << 2)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_VIEW ((uint32_t)1u << 3)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_MEMORY_SPACE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_OFFSET ((uint32_t)1u << 5)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_SLOT_BASE ((uint32_t)1u << 6)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_PACKET_WIDTH ((uint32_t)1u << 7)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DESCRIPTOR_MISSING \
  ((uint32_t)1u << 8)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_OFFSET_IMMEDIATE ((uint32_t)1u << 9)
#define LOOM_AMDGPU_ASYNC_GATHER_REJECTION_CACHE_POLICY ((uint32_t)1u << 10)

typedef struct loom_amdgpu_async_gather_diagnostic_t {
  // Target-specific rejection bits for async gather selection.
  loom_amdgpu_async_gather_rejection_flags_t rejection_bits;
  // Source memory planning diagnostics when source view decomposition fails.
  loom_low_source_memory_access_diagnostic_t source_diagnostic;
  // Source address planning diagnostics when target address selection fails.
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic;
} loom_amdgpu_async_gather_diagnostic_t;

typedef uint32_t loom_amdgpu_async_wait_rejection_flags_t;

#define LOOM_AMDGPU_ASYNC_WAIT_REJECTION_LOCAL_GROUP ((uint32_t)1u << 0)
#define LOOM_AMDGPU_ASYNC_WAIT_REJECTION_COMPLETED_GROUP ((uint32_t)1u << 1)
#define LOOM_AMDGPU_ASYNC_WAIT_REJECTION_NEWER_GROUPS ((uint32_t)1u << 2)
#define LOOM_AMDGPU_ASYNC_WAIT_REJECTION_GROUP_TOKEN ((uint32_t)1u << 3)
#define LOOM_AMDGPU_ASYNC_WAIT_REJECTION_PACKET_COUNT ((uint32_t)1u << 4)
#define LOOM_AMDGPU_ASYNC_WAIT_REJECTION_DESCRIPTOR ((uint32_t)1u << 5)

typedef struct loom_amdgpu_async_wait_diagnostic_t {
  // Target-specific rejection bits for async wait selection.
  loom_amdgpu_async_wait_rejection_flags_t rejection_bits;
} loom_amdgpu_async_wait_diagnostic_t;

typedef struct loom_amdgpu_async_gather_descriptor_candidate_t {
  // Number of bytes moved by the async packet.
  uint32_t packet_byte_count;
  // Stable descriptor ref selected for the active descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_async_gather_descriptor_candidate_t;

typedef struct loom_amdgpu_async_gather_selection_t {
  // Source global-like view access transferred into LDS.
  loom_low_source_memory_access_plan_t source;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      source_dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Source SSA view value passed to kernel.async.gather.
  loom_value_id_t source_view;
  // Destination LDS view value passed to kernel.async.gather.
  loom_value_id_t dest_view;
  // Static LDS byte offset materialized into M0.
  uint32_t dest_byte_offset;
  // Static global byte offset encoded in the packet immediate.
  int64_t source_immediate_offset;
  // Number of bytes moved by the selected async packet.
  uint32_t packet_byte_count;
  // Stable descriptor ref selected for the active descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_async_gather_selection_t;

static const loom_amdgpu_async_gather_descriptor_candidate_t
    kAmdgpuAsyncGatherDescriptors[] = {
        {
            .packet_byte_count = 4,
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_DWORD_SADDR,
        },
        {
            .packet_byte_count = 12,
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_DWORDX3_SADDR,
        },
        {
            .packet_byte_count = 16,
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_DWORDX4_SADDR,
        },
};

static bool loom_amdgpu_async_gather_exact_i64(loom_value_facts_t facts,
                                               int64_t* out_value) {
  *out_value = 0;
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_amdgpu_async_gather_source_memory_space_is_global_like(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return false;
  }
}

static bool loom_amdgpu_async_gather_select_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t packet_byte_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref,
    uint32_t* out_descriptor_ordinal) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuAsyncGatherDescriptors); ++i) {
    const loom_amdgpu_async_gather_descriptor_candidate_t* candidate =
        &kAmdgpuAsyncGatherDescriptors[i];
    if (candidate->packet_byte_count != packet_byte_count) {
      continue;
    }
    const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set, candidate->descriptor_ref);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    *out_descriptor_ref = candidate->descriptor_ref;
    *out_descriptor_ordinal = descriptor_ordinal;
    return true;
  }
  return false;
}

static bool loom_amdgpu_async_gather_select_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source_view, loom_vector_memory_cache_policy_t cache_policy,
    loom_amdgpu_async_gather_selection_t* selection,
    loom_amdgpu_async_gather_diagnostic_t* diagnostic,
    uint32_t* out_descriptor_ordinal) {
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;

  if (!loom_low_source_memory_access_plan_build_view(
          module, fact_table, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD,
          source_view, cache_policy, &selection->source,
          &diagnostic->source_diagnostic)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_PLAN;
    return false;
  }
  if (!loom_amdgpu_async_gather_source_memory_space_is_global_like(
          selection->source.memory_space)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_MEMORY_SPACE;
    return false;
  }

  loom_amdgpu_memory_access_t access = {
      .source = selection->source,
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
  };
  if (!loom_amdgpu_memory_access_select_dynamic_term_kinds(
          module, /*fact_table=*/NULL, /*view_regions=*/NULL, &access,
          &diagnostic->memory_diagnostic)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_ADDRESS;
    return false;
  }
  for (uint8_t i = 0; i < access.source.dynamic_term_count; ++i) {
    if (access.dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      continue;
    }
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_ADDRESS;
    return false;
  }

  const uint64_t packet_byte_count =
      (uint64_t)selection->source.element_byte_count *
      (uint64_t)selection->source.vector_lane_count;
  if (packet_byte_count > UINT32_MAX) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_PACKET_WIDTH;
    return false;
  }
  selection->packet_byte_count = (uint32_t)packet_byte_count;

  if (!loom_amdgpu_async_gather_select_descriptor(
          descriptor_set, selection->packet_byte_count,
          &selection->descriptor_ref, out_descriptor_ordinal)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    selection->source_dynamic_term_kinds[i] = access.dynamic_term_kinds[i];
  }
  return true;
}

static bool loom_amdgpu_async_gather_select_dest(
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_value_id_t dest_view, loom_amdgpu_async_gather_selection_t* selection,
    loom_amdgpu_async_gather_diagnostic_t* diagnostic) {
  loom_value_fact_view_reference_t dest_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, dest_view),
          &dest_reference)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_VIEW;
    return false;
  }
  if (dest_reference.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_MEMORY_SPACE;
    return false;
  }
  uint64_t lds_root_byte_offset = 0;
  if (!loom_amdgpu_source_lds_layout_lookup_root(fact_table, source_function,
                                                 dest_reference.root_value_id,
                                                 &lds_root_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_SLOT_BASE;
    return false;
  }

  int64_t dest_byte_offset = 0;
  if (!loom_amdgpu_async_gather_exact_i64(dest_reference.base_byte_offset,
                                          &dest_byte_offset) ||
      dest_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_OFFSET;
    return false;
  }
  const uint64_t view_byte_offset = (uint64_t)dest_byte_offset;
  if (lds_root_byte_offset > UINT32_MAX ||
      view_byte_offset > UINT32_MAX - lds_root_byte_offset) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_OFFSET;
    return false;
  }
  selection->dest_byte_offset =
      (uint32_t)(lds_root_byte_offset + view_byte_offset);
  return true;
}

static bool loom_amdgpu_async_gather_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_func_like_t source_function, const loom_op_t* source_op,
    loom_amdgpu_async_gather_selection_t* out_selection,
    loom_amdgpu_async_gather_diagnostic_t* out_diagnostic) {
  *out_selection = (loom_amdgpu_async_gather_selection_t){
      .source_view = LOOM_VALUE_ID_INVALID,
      .dest_view = LOOM_VALUE_ID_INVALID,
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  *out_diagnostic = (loom_amdgpu_async_gather_diagnostic_t){0};

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (source_op->attribute_count < 2 ||
      !loom_vector_memory_cache_policy_from_attrs(loom_op_attrs(source_op)[0],
                                                  loom_op_attrs(source_op)[1],
                                                  &cache_policy)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_CACHE_POLICY;
    return false;
  }

  out_selection->source_view = loom_kernel_async_gather_source(source_op);
  out_selection->dest_view = loom_kernel_async_gather_dest(source_op);
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_async_gather_select_source(
          module, fact_table, descriptor_set, out_selection->source_view,
          cache_policy, out_selection, out_diagnostic, &descriptor_ordinal) ||
      !loom_amdgpu_async_gather_select_dest(fact_table, source_function,
                                            out_selection->dest_view,
                                            out_selection, out_diagnostic)) {
    return false;
  }

  loom_amdgpu_descriptor_offset_immediate_info_t offset_info = {0};
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          &offset_info) ||
      offset_info.unit_byte_count != 1) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_OFFSET_IMMEDIATE;
    return false;
  }
  const int64_t signed_max = offset_info.unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info.unsigned_max;
  if (out_selection->source.static_byte_offset < offset_info.signed_min ||
      out_selection->source.static_byte_offset > signed_max) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_source_memory_offset_fits_u32(&out_selection->source,
                                                 /*static_byte_offset=*/0)) {
    out_diagnostic->memory_diagnostic.rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_ADDRESS;
    return false;
  }
  out_selection->source_immediate_offset =
      out_selection->source.static_byte_offset;

  loom_amdgpu_memory_access_t access = {
      .source = out_selection->source,
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
      .immediate_offset = out_selection->source_immediate_offset,
      .packet_byte_count = out_selection->packet_byte_count,
  };
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    access.dynamic_term_kinds[i] = out_selection->source_dynamic_term_kinds[i];
  }
  if (!loom_amdgpu_memory_cache_policy_can_lower(descriptor_set, &access)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_GATHER_REJECTION_CACHE_POLICY;
    return false;
  }
  return true;
}

static iree_status_t loom_amdgpu_async_gather_resolve_selection(
    loom_low_lower_context_t* context,
    const loom_amdgpu_async_gather_selection_t* selection,
    loom_amdgpu_async_gather_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_async_gather_plan_t){
      .source = selection->source,
      .source_view = selection->source_view,
      .dest_view = selection->dest_view,
      .dest_byte_offset = selection->dest_byte_offset,
      .source_immediate_offset = selection->source_immediate_offset,
      .packet_byte_count = selection->packet_byte_count,
  };
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    out_plan->source_dynamic_term_kinds[i] =
        selection->source_dynamic_term_kinds[i];
  }
  return loom_amdgpu_resolve_descriptor_ref(context, selection->descriptor_ref,
                                            &out_plan->descriptor);
}

iree_status_t loom_amdgpu_select_kernel_async_gather_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_async_gather_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_async_gather_plan_t){0};
  *out_selected = false;
  loom_amdgpu_async_gather_diagnostic_t diagnostic = {0};
  loom_amdgpu_async_gather_selection_t selection = {0};
  if (!loom_amdgpu_async_gather_select(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context),
          loom_low_lower_context_descriptor_set(context),
          loom_low_lower_context_source_function(context), source_op,
          &selection, &diagnostic)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_async_gather_resolve_selection(
      context, &selection, out_plan));
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_async_group_packet_count(const loom_module_t* module,
                                                 const loom_op_t* group_op,
                                                 uint32_t* out_packet_count) {
  *out_packet_count = 0;
  IREE_ASSERT(loom_kernel_async_group_isa(group_op));
  loom_value_slice_t tokens = loom_kernel_async_group_tokens(group_op);
  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    const loom_value_id_t token_id = tokens.values[i];
    if (token_id >= module->values.count) {
      return false;
    }
    const loom_value_t* token = loom_module_value(module, token_id);
    if (loom_value_is_block_arg(token)) {
      return false;
    }
    const loom_op_t* token_op = loom_value_def_op(token);
    if (token_op == NULL || !loom_kernel_async_gather_isa(token_op)) {
      return false;
    }
    ++(*out_packet_count);
  }
  return true;
}

static bool loom_amdgpu_async_wait_stream_counts(
    const loom_module_t* module, const loom_op_t* wait_op,
    uint16_t* out_newer_packet_count, bool* out_needs_wait_packet,
    loom_amdgpu_async_wait_diagnostic_t* diagnostic) {
  *out_newer_packet_count = 0;
  *out_needs_wait_packet = false;

  const loom_value_id_t waited_group_id = loom_kernel_async_wait_group(wait_op);
  if (waited_group_id >= module->values.count ||
      wait_op->parent_block == NULL) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ASYNC_WAIT_REJECTION_LOCAL_GROUP;
    return false;
  }

  const loom_block_t* block = wait_op->parent_block;
  iree_host_size_t waited_group_index = IREE_HOST_SIZE_MAX;
  iree_host_size_t group_count = 0;
  iree_host_size_t completed_group_count = 0;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op == wait_op) {
      break;
    }
    if (loom_kernel_async_group_isa(op)) {
      if (loom_kernel_async_group_group(op) == waited_group_id) {
        waited_group_index = group_count;
      }
      ++group_count;
      continue;
    }
    if (!loom_kernel_async_wait_isa(op)) {
      continue;
    }
    // Legal prior waits complete through the group whose ordinal is implied by
    // the prior stream depth. Invalid waits are rejected at their own op.
    const int64_t newer_groups = loom_kernel_async_wait_newer_groups(op);
    if (newer_groups < 0 || (uint64_t)newer_groups >= group_count) {
      continue;
    }
    const iree_host_size_t completed_index =
        group_count - (iree_host_size_t)newer_groups - 1;
    if (completed_group_count <= completed_index) {
      completed_group_count = completed_index + 1;
    }
  }

  if (waited_group_index == IREE_HOST_SIZE_MAX) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ASYNC_WAIT_REJECTION_LOCAL_GROUP;
    return false;
  }
  if (waited_group_index < completed_group_count) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ASYNC_WAIT_REJECTION_COMPLETED_GROUP;
    return false;
  }

  const int64_t actual_newer_groups =
      loom_kernel_async_wait_newer_groups(wait_op);
  if (actual_newer_groups < 0 ||
      (uint64_t)actual_newer_groups !=
          (uint64_t)(group_count - waited_group_index - 1)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ASYNC_WAIT_REJECTION_NEWER_GROUPS;
    return false;
  }

  uint64_t newer_packet_count = 0;
  uint64_t packet_count_to_wait = 0;
  iree_host_size_t group_index = 0;
  loom_block_for_each_op(block, op) {
    if (op == wait_op) {
      break;
    }
    if (!loom_kernel_async_group_isa(op)) {
      continue;
    }
    uint32_t group_packet_count = 0;
    if (!loom_amdgpu_async_group_packet_count(module, op,
                                              &group_packet_count)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ASYNC_WAIT_REJECTION_GROUP_TOKEN;
      return false;
    }
    if (group_index >= completed_group_count &&
        group_index <= waited_group_index) {
      packet_count_to_wait += group_packet_count;
    } else if (group_index > waited_group_index) {
      newer_packet_count += group_packet_count;
    }
    ++group_index;
  }

  if (newer_packet_count > UINT16_MAX) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ASYNC_WAIT_REJECTION_PACKET_COUNT;
    return false;
  }
  *out_newer_packet_count = (uint16_t)newer_packet_count;
  *out_needs_wait_packet = packet_count_to_wait != 0;
  return true;
}

iree_status_t loom_amdgpu_select_kernel_async_wait_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_async_wait_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_async_wait_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint16_t target_count = 0;
  bool needs_wait_packet = false;
  loom_amdgpu_async_wait_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_async_wait_stream_counts(module, source_op, &target_count,
                                            &needs_wait_packet, &diagnostic)) {
    return iree_ok_status();
  }
  if (!needs_wait_packet) {
    *out_selected = true;
    return iree_ok_status();
  }

  loom_amdgpu_wait_packet_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_select_counter_mask(
      loom_low_lower_context_descriptor_set(context),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD, target_count, &selection));

  loom_amdgpu_async_wait_immediate_t
      immediates[LOOM_AMDGPU_ASYNC_WAIT_IMMEDIATE_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < selection.immediate_count; ++i) {
    immediates[i] = (loom_amdgpu_async_wait_immediate_t){
        .name = selection.immediates[i].name,
        .value = selection.immediates[i].value,
    };
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_explicit_packet_row_plan(
      context, selection.descriptor, immediates, selection.immediate_count,
      &out_plan->wait));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_m0_move(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_async_gather_plan_t* plan, loom_value_id_t* out_low_m0) {
  *out_low_m0 = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_dest_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset(
      context, source_op, LOOM_VALUE_ID_INVALID,
      /*dynamic_index_byte_stride=*/1,
      LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE, plan->dest_byte_offset,
      &low_dest_offset));

  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_descriptor_row_implicit_resource_type(
      context, plan->descriptor.descriptor, &m0_type));
  loom_value_id_t operands[] = {low_dest_offset};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &m0_type,
      1, &low_op));
  *out_low_m0 = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_async_gather(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_async_gather_plan_t* plan) {
  loom_amdgpu_memory_access_t access = {
      .source = plan->source,
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
      .immediate_offset = plan->source_immediate_offset,
      .packet_byte_count = plan->packet_byte_count,
  };
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    access.dynamic_term_kinds[i] = plan->source_dynamic_term_kinds[i];
  }

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
      context, source_op, &access, LOOM_VALUE_ID_INVALID, &low_vaddr));

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source_view, &low_resource));
  loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
      context, source_op, &access, low_resource, &low_saddr));

  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_m0_move(context, source_op, plan, &low_m0));

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, &access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[] = {low_vaddr, low_saddr, low_m0};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op));
  return loom_low_lower_elide_value(context,
                                    loom_kernel_async_gather_token(source_op));
}

iree_status_t loom_amdgpu_lower_kernel_async_wait(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_async_wait_plan_t* plan) {
  return loom_amdgpu_emit_explicit_packet_plan(context, source_op, &plan->wait);
}

static iree_string_view_t loom_amdgpu_async_gather_rejection_key(
    const loom_amdgpu_async_gather_diagnostic_t* diagnostic) {
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_PLAN)) {
    return loom_low_source_memory_access_rejection_key(
        diagnostic->source_diagnostic.rejection_bits);
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_ADDRESS)) {
    return loom_amdgpu_memory_access_rejection_key(
        diagnostic->memory_diagnostic.rejection_bits);
  }
  if (iree_any_bit_set(
          diagnostic->rejection_bits,
          LOOM_AMDGPU_ASYNC_GATHER_REJECTION_SOURCE_MEMORY_SPACE)) {
    return IREE_SV("async_gather.source_memory_space");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_VIEW)) {
    return IREE_SV("async_gather.dest_view");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_MEMORY_SPACE)) {
    return IREE_SV("async_gather.dest_memory_space");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_SLOT_BASE)) {
    return IREE_SV("async_gather.dest_slot_base");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DEST_OFFSET)) {
    return IREE_SV("async_gather.dest_offset");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_PACKET_WIDTH)) {
    return IREE_SV("async_gather.packet_width");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_DESCRIPTOR_MISSING)) {
    return IREE_SV("async_gather.descriptor_missing");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_OFFSET_IMMEDIATE)) {
    return IREE_SV("async_gather.offset_immediate");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_GATHER_REJECTION_CACHE_POLICY)) {
    return IREE_SV("async_gather.cache_policy");
  }
  return IREE_SV("async_gather.shape");
}

static iree_string_view_t loom_amdgpu_async_wait_rejection_key(
    const loom_amdgpu_async_wait_diagnostic_t* diagnostic) {
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_WAIT_REJECTION_LOCAL_GROUP)) {
    return IREE_SV("async_wait.local_group");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_WAIT_REJECTION_COMPLETED_GROUP)) {
    return IREE_SV("async_wait.completed_group");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_WAIT_REJECTION_NEWER_GROUPS)) {
    return IREE_SV("async_wait.newer_groups");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_WAIT_REJECTION_GROUP_TOKEN)) {
    return IREE_SV("async_wait.group_token");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_WAIT_REJECTION_PACKET_COUNT)) {
    return IREE_SV("async_wait.packet_count");
  }
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_ASYNC_WAIT_REJECTION_DESCRIPTOR)) {
    return IREE_SV("async_wait.descriptor");
  }
  return IREE_SV("async_wait.stream");
}

static iree_status_t loom_amdgpu_low_legality_reject_async_transfer(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  return loom_amdgpu_low_legality_reject(context, op,
                                         IREE_SV("async.transfer_packet"));
}

static iree_status_t loom_amdgpu_low_legality_verify_kernel_async_gather(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  loom_amdgpu_async_gather_selection_t selection = {0};
  loom_amdgpu_async_gather_diagnostic_t diagnostic = {0};
  if (loom_amdgpu_async_gather_select(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context),
          loom_target_low_legality_function(context), op, &selection,
          &diagnostic)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(
      context, op, loom_amdgpu_async_gather_rejection_key(&diagnostic));
}

static iree_status_t loom_amdgpu_low_legality_verify_kernel_async_group(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  loom_value_slice_t tokens = loom_kernel_async_group_tokens(op);
  const loom_module_t* module = loom_target_low_legality_module(context);
  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    const loom_value_id_t token_id = tokens.values[i];
    if (token_id >= module->values.count) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("async_group.token_value"));
    }
    const loom_value_t* token = loom_module_value(module, token_id);
    const loom_op_t* defining_op =
        loom_value_is_block_arg(token) ? NULL : loom_value_def_op(token);
    if (defining_op == NULL || !loom_kernel_async_gather_isa(defining_op)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("async_group.local_gather_tokens"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_kernel_async_wait(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  uint16_t target_count = 0;
  bool needs_wait_packet = false;
  loom_amdgpu_async_wait_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_async_wait_stream_counts(
          loom_target_low_legality_module(context), op, &target_count,
          &needs_wait_packet, &diagnostic)) {
    return loom_amdgpu_low_legality_reject(
        context, op, loom_amdgpu_async_wait_rejection_key(&diagnostic));
  }
  if (!needs_wait_packet) {
    return iree_ok_status();
  }
  loom_amdgpu_wait_packet_selection_t selection = {0};
  iree_status_t status = loom_amdgpu_wait_packet_select_counter_mask(
      loom_target_low_legality_descriptor_set(context),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD, target_count, &selection);
  if (iree_status_is_ok(status)) {
    return iree_ok_status();
  }
  iree_status_free(status);
  diagnostic.rejection_bits |= LOOM_AMDGPU_ASYNC_WAIT_REJECTION_DESCRIPTOR;
  return loom_amdgpu_low_legality_reject(
      context, op, loom_amdgpu_async_wait_rejection_key(&diagnostic));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_async(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  switch (op->kind) {
    case LOOM_OP_KERNEL_ASYNC_GROUP:
      return loom_amdgpu_low_legality_verify_kernel_async_group(context, op);
    case LOOM_OP_KERNEL_ASYNC_WAIT:
      return loom_amdgpu_low_legality_verify_kernel_async_wait(context, op);
    case LOOM_OP_KERNEL_ASYNC_GATHER:
      return loom_amdgpu_low_legality_verify_kernel_async_gather(context, op);
    case LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER:
    case LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK:
    case LOOM_OP_KERNEL_ASYNC_COPY:
    case LOOM_OP_KERNEL_ASYNC_COPY_MASK:
    case LOOM_OP_KERNEL_ASYNC_GATHER_MASK:
    case LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS:
    case LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS:
      return loom_amdgpu_low_legality_reject_async_transfer(context, op);
    default:
      IREE_ASSERT_UNREACHABLE("AMDGPU async verifier selected unknown op kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
