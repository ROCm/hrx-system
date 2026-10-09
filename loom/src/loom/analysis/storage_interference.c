// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_interference.h"

#include <string.h>

#include "loom/analysis/control_uniformity.h"
#include "loom/analysis/liveness.h"
#include "loom/analysis/liveness_events.h"
#include "loom/analysis/value_relation.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/util/cfg_postdominance.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/walk.h"

typedef uint8_t loom_storage_interference_entry_flags_t;

#define LOOM_STORAGE_INTERFERENCE_ENTRY_ROOT ((uint8_t)1u << 0)
#define LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE ((uint8_t)1u << 1)
#define LOOM_STORAGE_INTERFERENCE_ENTRY_REPEATED ((uint8_t)1u << 2)
#define LOOM_STORAGE_INTERFERENCE_ENTRY_SINGLE_INSTANCE ((uint8_t)1u << 3)
#define LOOM_STORAGE_INTERFERENCE_ENTRY_WORKGROUP_CUT ((uint8_t)1u << 4)

typedef struct loom_storage_interference_barrier_t {
  // Workgroup acq_rel rendezvous in source traversal order.
  const loom_op_t* op;
  // Retained proof that all workgroup invocations reach the rendezvous.
  bool uniform;
} loom_storage_interference_barrier_t;

typedef struct loom_storage_interference_async_access_t {
  // Buffer or view whose allocation instance is accessed asynchronously.
  loom_value_ordinal_t reference_ordinal;
  // Token result locating the start of the asynchronous transfer.
  loom_value_ordinal_t token_ordinal;
  // Group named by the first known completing wait in the transfer's block.
  loom_value_ordinal_t completion_group_ordinal;
} loom_storage_interference_async_access_t;

typedef struct loom_storage_interference_lifetime_scratch_t {
  // Temporary storage released after the per-root proof bits are retained.
  iree_arena_allocator_t arena;
  // Known asynchronous access extents touching repeated allocations.
  loom_storage_interference_async_access_t* async_accesses;
  // Number of initialized asynchronous extents.
  iree_host_size_t async_access_count;
  // Allocated asynchronous extent capacity.
  iree_host_size_t async_access_capacity;
} loom_storage_interference_lifetime_scratch_t;

typedef struct loom_storage_interference_footprint_t {
  // Memory access and asynchronous completion operations for this root.
  const loom_op_t** operations;
  // Number of initialized operations.
  iree_host_size_t operation_count;
  // Allocated operation pointer capacity.
  iree_host_size_t operation_capacity;
} loom_storage_interference_footprint_t;

typedef struct loom_storage_interference_entry_t {
  // Root, provenance completeness, and instance-lifetime proof bits.
  loom_storage_interference_entry_flags_t flags;
  // Memory space declared by the allocation root.
  loom_value_fact_memory_space_t memory_space;
  // Complete operation footprint for an allocation root.
  loom_storage_interference_footprint_t footprint;
} loom_storage_interference_entry_t;

typedef struct loom_storage_interference_edge_t {
  // Destination local value ordinal receiving the source provenance.
  loom_value_ordinal_t destination_ordinal;
  // Next outgoing provenance edge for the same source value.
  struct loom_storage_interference_edge_t* next;
} loom_storage_interference_edge_t;

typedef struct loom_storage_interference_membership_t {
  // Local ordinal of the buffer.alloca root reaching this value.
  loom_value_ordinal_t root_ordinal;
  // Local ordinal of the value carrying the root provenance.
  loom_value_ordinal_t value_ordinal;
  // Next allocation-root membership for the same value.
  struct loom_storage_interference_membership_t* next_for_value;
  // Next newly discovered membership awaiting propagation.
  struct loom_storage_interference_membership_t* next_pending;
} loom_storage_interference_membership_t;

typedef struct loom_storage_interference_cfg_region_t {
  // Region summarized by the retained graph results.
  const loom_region_t* region;
  // Populated CFG facts borrowed from the value-fact scope.
  const loom_value_fact_cfg_region_t* facts;
  // Postdominance built once from the retained graph.
  loom_cfg_postdominance_t postdominance;
  // Next lazily initialized region summary.
  struct loom_storage_interference_cfg_region_t* next;
} loom_storage_interference_cfg_region_t;

struct loom_storage_interference_t {
  // Module containing the analyzed function.
  const loom_module_t* module;
  // Populated value facts and retained CFG snapshots.
  const loom_value_fact_table_t* fact_table;
  // Active local numbering for the function body.
  const loom_local_value_domain_t* value_domain;
  // Arena owning all analysis results and query summaries.
  iree_arena_allocator_t* arena;
  // Function whose body was analyzed.
  loom_func_like_t function;
  // Root entries indexed by local value ordinal.
  loom_storage_interference_entry_t* entries;
  // Outgoing provenance edges indexed by local value ordinal.
  loom_storage_interference_edge_t** edges;
  // Allocation-root memberships indexed by local value ordinal.
  loom_storage_interference_membership_t** memberships;
  // First newly discovered membership awaiting propagation.
  loom_storage_interference_membership_t* pending_head;
  // Last newly discovered membership awaiting propagation.
  loom_storage_interference_membership_t* pending_tail;
  // Qualifying workgroup lifetime barriers in source order.
  loom_storage_interference_barrier_t* workgroup_barriers;
  // Number of qualifying workgroup barriers.
  iree_host_size_t workgroup_barrier_count;
  // Allocated workgroup barrier record capacity.
  iree_host_size_t workgroup_barrier_capacity;
  // Lazily built postdominance summaries for queried CFG regions.
  loom_storage_interference_cfg_region_t* cfg_regions;
  // Retained workgroup-uniform branch exclusion analysis.
  loom_control_uniformity_info_t control_uniformity;
  // True when an operation may access storage without described operands.
  bool has_unknown_memory_access;
};

static bool loom_storage_interference_value_is_reference(
    const loom_storage_interference_t* analysis, loom_value_id_t value_id) {
  const loom_type_t type = loom_module_value_type(analysis->module, value_id);
  return loom_type_is_buffer(type) || loom_type_is_view(type);
}

static bool loom_storage_interference_resolve_reference_root(
    const loom_storage_interference_t* analysis, loom_value_id_t value_id,
    loom_value_id_t* out_root_value_id) {
  *out_root_value_id = LOOM_VALUE_ID_INVALID;
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(analysis->fact_table, value_id);
  loom_value_fact_buffer_reference_t buffer_reference;
  if (loom_value_facts_query_buffer_reference(&analysis->fact_table->context,
                                              facts, &buffer_reference)) {
    *out_root_value_id = loom_value_fact_buffer_reference_resolve_root_value(
        buffer_reference, value_id);
    return true;
  }
  loom_value_fact_view_reference_t view_reference;
  if (loom_value_facts_query_view_reference(&analysis->fact_table->context,
                                            facts, &view_reference)) {
    *out_root_value_id = loom_value_fact_view_reference_resolve_root_value(
        view_reference, value_id);
    return true;
  }
  return false;
}

static iree_status_t loom_storage_interference_append_edge(
    loom_storage_interference_t* analysis, loom_value_id_t source_value_id,
    loom_value_id_t destination_value_id) {
  if (source_value_id == destination_value_id) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t source_ordinal =
      loom_local_value_domain_try_ordinal(analysis->value_domain,
                                          source_value_id);
  const loom_value_ordinal_t destination_ordinal =
      loom_local_value_domain_try_ordinal(analysis->value_domain,
                                          destination_value_id);
  if (source_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      destination_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  loom_storage_interference_edge_t* edge = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(analysis->arena, sizeof(*edge), (void**)&edge));
  *edge = (loom_storage_interference_edge_t){
      .destination_ordinal = destination_ordinal,
      .next = analysis->edges[source_ordinal],
  };
  analysis->edges[source_ordinal] = edge;
  return iree_ok_status();
}

static iree_status_t loom_storage_interference_add_membership(
    loom_storage_interference_t* analysis, loom_value_ordinal_t value_ordinal,
    loom_value_ordinal_t root_ordinal) {
  for (loom_storage_interference_membership_t* membership =
           analysis->memberships[value_ordinal];
       membership; membership = membership->next_for_value) {
    if (membership->root_ordinal == root_ordinal) {
      return iree_ok_status();
    }
  }

  loom_storage_interference_membership_t* membership = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(analysis->arena, sizeof(*membership),
                                           (void**)&membership));
  *membership = (loom_storage_interference_membership_t){
      .root_ordinal = root_ordinal,
      .value_ordinal = value_ordinal,
      .next_for_value = analysis->memberships[value_ordinal],
  };
  analysis->memberships[value_ordinal] = membership;
  if (analysis->pending_tail) {
    analysis->pending_tail->next_pending = membership;
  } else {
    analysis->pending_head = membership;
  }
  analysis->pending_tail = membership;
  return iree_ok_status();
}

static iree_status_t loom_storage_interference_propagate_memberships(
    loom_storage_interference_t* analysis) {
  while (analysis->pending_head) {
    loom_storage_interference_membership_t* membership = analysis->pending_head;
    analysis->pending_head = membership->next_pending;
    if (!analysis->pending_head) {
      analysis->pending_tail = NULL;
    }
    for (loom_storage_interference_edge_t* edge =
             analysis->edges[membership->value_ordinal];
         edge; edge = edge->next) {
      IREE_RETURN_IF_ERROR(loom_storage_interference_add_membership(
          analysis, edge->destination_ordinal, membership->root_ordinal));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_storage_interference_append_footprint_operation(
    loom_storage_interference_t* analysis,
    loom_storage_interference_entry_t* entry, const loom_op_t* operation) {
  for (iree_host_size_t i = 0; i < entry->footprint.operation_count; ++i) {
    if (entry->footprint.operations[i] == operation) {
      return iree_ok_status();
    }
  }
  if (entry->footprint.operation_count >= entry->footprint.operation_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(analysis->arena, entry->footprint.operation_count,
                              entry->footprint.operation_capacity == 0
                                  ? 8
                                  : entry->footprint.operation_capacity * 2,
                              sizeof(*entry->footprint.operations),
                              &entry->footprint.operation_capacity,
                              (void**)&entry->footprint.operations));
  }
  entry->footprint.operations[entry->footprint.operation_count++] = operation;
  return iree_ok_status();
}

static bool loom_storage_interference_async_transfer_isa(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_KERNEL_ASYNC_COPY:
    case LOOM_OP_KERNEL_ASYNC_COPY_MASK:
    case LOOM_OP_KERNEL_ASYNC_GATHER:
    case LOOM_OP_KERNEL_ASYNC_GATHER_MASK:
    case LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER:
    case LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK:
    case LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS:
    case LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS:
      return true;
    default:
      return false;
  }
}

static const loom_op_t* loom_storage_interference_async_completion(
    const loom_storage_interference_t* analysis, const loom_op_t* producer_op) {
  if (producer_op->result_count != 1 || !producer_op->parent_block) {
    return NULL;
  }
  const loom_value_t* token = loom_module_value(
      analysis->module, loom_op_const_results(producer_op)[0]);
  if (token->use_count != 1) {
    return NULL;
  }
  const loom_op_t* group_op = loom_use_user_op(loom_value_uses(token)[0]);
  if (!loom_kernel_async_group_isa(group_op) ||
      group_op->parent_block != producer_op->parent_block) {
    return NULL;
  }

  for (const loom_op_t* op = group_op->next_op; op; op = op->next_op) {
    if (!loom_kernel_async_wait_isa(op)) {
      continue;
    }
    const loom_value_t* waited_group =
        loom_module_value(analysis->module, loom_kernel_async_wait_group(op));
    if (loom_value_is_block_arg(waited_group)) {
      return NULL;
    }
    const loom_op_t* waited_group_op = loom_value_def_op(waited_group);
    if (!loom_kernel_async_group_isa(waited_group_op) ||
        waited_group_op->parent_block != producer_op->parent_block) {
      return NULL;
    }
    if (waited_group_op->block_ordinal >= group_op->block_ordinal) {
      return op;
    }
  }
  return NULL;
}

static void loom_storage_interference_mark_memberships_incomplete(
    loom_storage_interference_t* analysis,
    const loom_storage_interference_membership_t* memberships) {
  for (const loom_storage_interference_membership_t* membership = memberships;
       membership; membership = membership->next_for_value) {
    analysis->entries[membership->root_ordinal].flags &=
        ~LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE;
  }
}

static iree_status_t loom_storage_interference_record_value_accesses(
    loom_storage_interference_t* analysis, loom_value_ordinal_t value_ordinal,
    loom_storage_interference_lifetime_scratch_t* lifetime) {
  loom_storage_interference_membership_t* memberships =
      analysis->memberships[value_ordinal];
  if (!memberships) {
    return iree_ok_status();
  }
  const loom_value_id_t value_id =
      analysis->value_domain->value_ids[value_ordinal];
  const loom_value_t* value = loom_module_value(analysis->module, value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    // A call can retain a reference or return an alias, including encoding it
    // as a scalar at a Low boundary. Until the callee's access/escape contract
    // is known, only the roots exposed to that call lose their lifetime proof.
    // Purity alone does not describe result aliases.
    // Binding storage into a channel also exposes aliases and asynchronous
    // lifetimes not described by the buffer/view use graph. Channel realization
    // must make those accesses explicit before this analysis can prove reuse.
    if (loom_channel_bind_isa(user_op) ||
        loom_call_like_isa(
            loom_call_like_cast(analysis->module, (loom_op_t*)user_op))) {
      loom_storage_interference_mark_memberships_incomplete(analysis,
                                                            memberships);
      continue;
    }
    const loom_op_vtable_t* vtable = loom_op_vtable(analysis->module, user_op);
    const loom_operand_descriptor_t* descriptor = NULL;
    if (!loom_op_operand_descriptor_at(vtable, user_op,
                                       loom_use_operand_index(*use),
                                       &descriptor, NULL, NULL)) {
      if (loom_traits_may_access_memory(
              loom_op_effective_traits(analysis->module, user_op))) {
        loom_storage_interference_mark_memberships_incomplete(analysis,
                                                              memberships);
      }
      continue;
    }
    if (!iree_any_bit_set(descriptor->flags,
                          LOOM_OPERAND_READS | LOOM_OPERAND_WRITES)) {
      continue;
    }

    const loom_op_t* completion_op = NULL;
    if (loom_storage_interference_async_transfer_isa(user_op)) {
      completion_op =
          loom_storage_interference_async_completion(analysis, user_op);
      if (!completion_op) {
        loom_storage_interference_mark_memberships_incomplete(analysis,
                                                              memberships);
        continue;
      }
    }
    bool accesses_repeated_root = false;
    for (loom_storage_interference_membership_t* membership = memberships;
         membership; membership = membership->next_for_value) {
      loom_storage_interference_entry_t* entry =
          &analysis->entries[membership->root_ordinal];
      accesses_repeated_root |= iree_any_bit_set(
          entry->flags, LOOM_STORAGE_INTERFERENCE_ENTRY_REPEATED);
      IREE_RETURN_IF_ERROR(loom_storage_interference_append_footprint_operation(
          analysis, entry, user_op));
      if (completion_op) {
        IREE_RETURN_IF_ERROR(
            loom_storage_interference_append_footprint_operation(
                analysis, entry, completion_op));
      }
    }
    if (completion_op && accesses_repeated_root) {
      if (lifetime->async_access_count == lifetime->async_access_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            &lifetime->arena, lifetime->async_access_count,
            lifetime->async_access_capacity
                ? lifetime->async_access_capacity * 2
                : 4,
            sizeof(*lifetime->async_accesses), &lifetime->async_access_capacity,
            (void**)&lifetime->async_accesses));
      }
      lifetime->async_accesses[lifetime->async_access_count++] =
          (loom_storage_interference_async_access_t){
              .reference_ordinal = value_ordinal,
              .token_ordinal = loom_local_value_domain_ordinal(
                  analysis->value_domain, loom_op_const_results(user_op)[0]),
              .completion_group_ordinal = loom_local_value_domain_ordinal(
                  analysis->value_domain,
                  loom_kernel_async_wait_group(completion_op)),
          };
    }
  }
  return iree_ok_status();
}

static void loom_storage_interference_append_lifetime_events(
    const loom_storage_interference_t* analysis, loom_value_ordinal_t ordinal,
    uint32_t start_point, uint32_t end_point, loom_liveness_event_t* events,
    iree_host_size_t* event_count, uint32_t* maximum_point) {
  const loom_value_id_t value_id = analysis->value_domain->value_ids[ordinal];
  events[(*event_count)++] = (loom_liveness_event_t){
      .point = start_point,
      .value_id = value_id,
      .value_ordinal = ordinal,
      .value_delta = 1,
  };
  events[(*event_count)++] = (loom_liveness_event_t){
      .point = end_point,
      .value_id = value_id,
      .value_ordinal = ordinal,
      .value_delta = -1,
  };
  *maximum_point = iree_max(*maximum_point, end_point);
}

static iree_status_t loom_storage_interference_append_async_events(
    const loom_storage_interference_t* analysis,
    loom_storage_interference_lifetime_scratch_t* lifetime,
    const loom_liveness_analysis_t* liveness, loom_liveness_event_t* events,
    iree_host_size_t* event_count, uint32_t* maximum_point) {
  if (!lifetime->async_access_count) {
    return iree_ok_status();
  }
  // Retain the first same-block completion boundary for each named group.
  // A recognized transfer completion always names a group defined in that
  // block. Later waits for the same group cannot extend its access lifetime.
  uint32_t* completion_points = NULL;
  const iree_host_size_t value_count = analysis->value_domain->value_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&lifetime->arena, value_count,
                                                 sizeof(*completion_points),
                                                 (void**)&completion_points));
  memset(completion_points, 0xFF, value_count * sizeof(*completion_points));
  for (uint32_t i = 0; i < liveness->operation_count; ++i) {
    const loom_liveness_operation_point_t* point =
        loom_liveness_operation_at(liveness, i);
    if (!loom_kernel_async_wait_isa(point->op)) {
      continue;
    }
    const loom_value_id_t group_id = loom_kernel_async_wait_group(point->op);
    const loom_value_t* group = loom_module_value(analysis->module, group_id);
    if (loom_value_is_block_arg(group) ||
        loom_value_def_op(group)->parent_block != point->op->parent_block) {
      continue;
    }
    const loom_value_ordinal_t group_ordinal =
        loom_local_value_domain_ordinal(analysis->value_domain, group_id);
    if (completion_points[group_ordinal] == UINT32_MAX) {
      completion_points[group_ordinal] = point->end_point;
    }
  }
  for (iree_host_size_t i = 0; i < lifetime->async_access_count; ++i) {
    const loom_storage_interference_async_access_t* access =
        &lifetime->async_accesses[i];
    // A transfer's token is defined immediately after its launch boundary.
    const uint32_t start_point = loom_liveness_interval_for_value_ordinal(
                                     liveness, access->token_ordinal)
                                     ->definition_point -
                                 1;
    const uint32_t end_point =
        completion_points[access->completion_group_ordinal];
    IREE_ASSERT_NE(end_point, UINT32_MAX);
    loom_storage_interference_append_lifetime_events(
        analysis, access->reference_ordinal, start_point, end_point, events,
        event_count, maximum_point);
  }
  return iree_ok_status();
}

static iree_status_t loom_storage_interference_prove_single_instances(
    loom_storage_interference_t* analysis,
    loom_storage_interference_lifetime_scratch_t* lifetime) {
  const iree_host_size_t value_count = analysis->value_domain->value_count;
  bool needs_liveness = false;
  iree_host_size_t workgroup_root_count = 0;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    loom_storage_interference_entry_t* entry = &analysis->entries[i];
    if (iree_all_bits_set(entry->flags,
                          LOOM_STORAGE_INTERFERENCE_ENTRY_REPEATED |
                              LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE)) {
      entry->flags |= LOOM_STORAGE_INTERFERENCE_ENTRY_SINGLE_INSTANCE;
      needs_liveness = true;
      if (entry->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
          entry->footprint.operation_count != 0) {
        ++workgroup_root_count;
      }
    }
  }
  if (!needs_liveness) {
    return iree_ok_status();
  }
  if (workgroup_root_count != 0) {
    loom_target_workgroup_size_t size = {0};
    if (loom_kernel_def_static_workgroup_size_from_facts(
            analysis->module, analysis->function.op, analysis->fact_table,
            &size) &&
        size.x == 1 && size.y == 1 && size.z == 1) {
      // One invocation has no peers whose accesses can outlive its own.
      workgroup_root_count = 0;
    }
  }
  loom_value_ordinal_t* workgroup_roots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &lifetime->arena, workgroup_root_count, sizeof(*workgroup_roots),
      (void**)&workgroup_roots));
  iree_host_size_t root_index = 0;
  for (loom_value_ordinal_t i = 0; workgroup_root_count != 0 && i < value_count;
       ++i) {
    const loom_storage_interference_entry_t* entry = &analysis->entries[i];
    if (iree_all_bits_set(entry->flags,
                          LOOM_STORAGE_INTERFERENCE_ENTRY_REPEATED |
                              LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE) &&
        entry->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
        entry->footprint.operation_count != 0) {
      workgroup_roots[root_index++] = i;
    }
  }

  loom_liveness_analysis_t liveness;
  IREE_RETURN_IF_ERROR(loom_liveness_analyze_local_value_domain(
      analysis->value_domain, loom_liveness_order_empty(), &lifetime->arena,
      &liveness));
  iree_host_size_t segment_count = lifetime->async_access_count;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    if (analysis->memberships[i]) {
      const loom_liveness_segment_range_t range =
          loom_liveness_segment_range_for_value_ordinal(&liveness, i);
      if (!iree_host_size_checked_add(segment_count, range.count,
                                      &segment_count)) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "storage lifetime event count exceeds host size");
      }
    }
  }
  if (segment_count > IREE_HOST_SIZE_MAX / 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "storage lifetime event count exceeds host size");
  }
  loom_liveness_event_t* events = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &lifetime->arena, segment_count * 2, sizeof(*events), (void**)&events));
  iree_host_size_t event_count = 0;
  uint32_t maximum_point = 0;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    if (!analysis->memberships[i]) {
      continue;
    }
    const loom_liveness_segment_range_t range =
        loom_liveness_segment_range_for_value_ordinal(&liveness, i);
    for (uint32_t j = 0; j < range.count; ++j) {
      const loom_liveness_segment_t* segment =
          &liveness.segments[range.start + j];
      loom_storage_interference_append_lifetime_events(
          analysis, i, segment->start_point, segment->end_point, events,
          &event_count, &maximum_point);
    }
  }
  IREE_RETURN_IF_ERROR(loom_storage_interference_append_async_events(
      analysis, lifetime, &liveness, events, &event_count, &maximum_point));
  IREE_RETURN_IF_ERROR(loom_liveness_events_sort(
      events, event_count, maximum_point, &lifetime->arena));
  iree_host_size_t* live_counts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&lifetime->arena, value_count,
                                                 sizeof(*live_counts),
                                                 (void**)&live_counts));
  memset(live_counts, 0, value_count * sizeof(*live_counts));

  iree_host_size_t event_index = 0;
  iree_host_size_t barrier_index = 0;
  for (uint32_t i = 0; i < liveness.operation_count; ++i) {
    const loom_liveness_operation_point_t* point =
        loom_liveness_operation_at(&liveness, i);
    while (event_index < event_count &&
           events[event_index].point <= point->start_point) {
      const loom_liveness_event_t* event = &events[event_index++];
      for (const loom_storage_interference_membership_t* membership =
               analysis->memberships[event->value_ordinal];
           membership; membership = membership->next_for_value) {
        live_counts[membership->root_ordinal] += event->value_delta;
      }
    }
    if (loom_buffer_alloca_isa(point->op)) {
      const loom_value_ordinal_t root = loom_local_value_domain_ordinal(
          analysis->value_domain, loom_buffer_alloca_result(point->op));
      if (iree_any_bit_set(analysis->entries[root].flags,
                           LOOM_STORAGE_INTERFERENCE_ENTRY_REPEATED) &&
          live_counts[root] != 0) {
        analysis->entries[root].flags &=
            ~LOOM_STORAGE_INTERFERENCE_ENTRY_SINGLE_INSTANCE;
      }
    }
    // Both retained operation tables use the same source preorder. Matching
    // that subsequence consumes the walk's barrier qualification without
    // reclassifying control or looking up operation pointers in another index.
    if (barrier_index < analysis->workgroup_barrier_count &&
        point->op == analysis->workgroup_barriers[barrier_index].op) {
      const loom_storage_interference_barrier_t* barrier =
          &analysis->workgroup_barriers[barrier_index++];
      if (!barrier->uniform) {
        continue;
      }
      for (iree_host_size_t j = 0; j < workgroup_root_count; ++j) {
        const loom_value_ordinal_t root = workgroup_roots[j];
        const loom_value_t* value = loom_module_value(
            analysis->module, analysis->value_domain->value_ids[root]);
        if (loom_value_def_op(value)->parent_block == point->op->parent_block &&
            live_counts[root] == 0) {
          analysis->entries[root].flags |=
              LOOM_STORAGE_INTERFERENCE_ENTRY_WORKGROUP_CUT;
        }
      }
    }
  }
  IREE_ASSERT_EQ(barrier_index, analysis->workgroup_barrier_count);
  for (iree_host_size_t i = 0; i < workgroup_root_count; ++i) {
    loom_storage_interference_entry_t* entry =
        &analysis->entries[workgroup_roots[i]];
    if (!iree_any_bit_set(entry->flags,
                          LOOM_STORAGE_INTERFERENCE_ENTRY_WORKGROUP_CUT)) {
      entry->flags &= ~LOOM_STORAGE_INTERFERENCE_ENTRY_SINGLE_INSTANCE;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_storage_interference_append_workgroup_barrier(
    loom_storage_interference_t* analysis, const loom_op_t* barrier_op) {
  if (analysis->workgroup_barrier_count >=
      analysis->workgroup_barrier_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        analysis->arena, analysis->workgroup_barrier_count,
        analysis->workgroup_barrier_capacity == 0
            ? 4
            : analysis->workgroup_barrier_capacity * 2,
        sizeof(*analysis->workgroup_barriers),
        &analysis->workgroup_barrier_capacity,
        (void**)&analysis->workgroup_barriers));
  }
  analysis->workgroup_barriers[analysis->workgroup_barrier_count++] =
      (loom_storage_interference_barrier_t){
          .op = barrier_op,
          .uniform = loom_control_uniformity_prove_execution(
              &analysis->control_uniformity, barrier_op,
              LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP, NULL),
      };
  return iree_ok_status();
}

static bool loom_storage_interference_is_lifetime_barrier(const loom_op_t* op) {
  return loom_kernel_barrier_isa(op) &&
         loom_kernel_barrier_memory_space(op) ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         loom_kernel_barrier_scope(op) == LOOM_ATOMIC_SCOPE_WORKGROUP &&
         loom_kernel_barrier_ordering(op) == LOOM_ATOMIC_ORDERING_ACQ_REL;
}

static iree_status_t loom_storage_interference_walk_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_storage_interference_t* analysis =
      (loom_storage_interference_t*)user_data;
  const loom_op_vtable_t* vtable = loom_op_vtable(analysis->module, op);
  const loom_trait_flags_t traits =
      loom_op_effective_traits(analysis->module, op);
  const bool known_async_stream_effect =
      loom_kernel_async_group_isa(op) || loom_kernel_async_wait_isa(op);
  // Fresh frame storage is reachable by a callee through its operands. The
  // per-value walk handles those exposures after alias propagation; a call
  // without a reference to a root cannot invalidate that root's proof.
  const bool callable_effect =
      loom_call_like_isa(loom_call_like_cast(analysis->module, op));
  if (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS) &&
      !known_async_stream_effect && !callable_effect) {
    analysis->has_unknown_memory_access = true;
  } else if (loom_traits_may_access_memory(traits) &&
             !known_async_stream_effect && !callable_effect) {
    bool described_read = false;
    bool described_write = false;
    if (vtable && vtable->operand_descriptors) {
      const uint8_t descriptor_count =
          loom_op_vtable_operand_descriptor_count(vtable);
      for (uint8_t i = 0; i < descriptor_count; ++i) {
        const loom_operand_flags_t flags = vtable->operand_descriptors[i].flags;
        described_read |= iree_any_bit_set(flags, LOOM_OPERAND_READS);
        described_write |= iree_any_bit_set(flags, LOOM_OPERAND_WRITES);
      }
    }
    const bool missing_read =
        iree_any_bit_set(traits, LOOM_TRAIT_READS_MEMORY) && !described_read;
    const bool missing_write =
        iree_any_bit_set(traits, LOOM_TRAIT_WRITES_MEMORY) && !described_write;
    analysis->has_unknown_memory_access |= missing_read || missing_write;
  }

  if (loom_storage_interference_is_lifetime_barrier(op)) {
    IREE_RETURN_IF_ERROR(
        loom_storage_interference_append_workgroup_barrier(analysis, op));
  }

  const loom_value_relation_mask_t relation_mask =
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_TIED_RESULT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_SELECT_PAYLOAD) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_BYPASS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REGION_RESULT);
  loom_value_relation_iterator_t iterator;
  loom_value_relation_iterator_initialize(analysis->module, op, relation_mask,
                                          &iterator);
  loom_value_relation_t relation;
  while (loom_value_relation_iterator_next(&iterator, &relation)) {
    if (!loom_storage_interference_value_is_reference(
            analysis, relation.source_value_id) ||
        !loom_storage_interference_value_is_reference(
            analysis, relation.destination_value_id)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_storage_interference_append_edge(
        analysis, relation.source_value_id, relation.destination_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_storage_interference_initialize_values(
    loom_storage_interference_t* analysis) {
  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < analysis->value_domain->value_count; ++value_ordinal) {
    const loom_value_id_t value_id =
        analysis->value_domain->value_ids[value_ordinal];
    const loom_value_t* value = loom_module_value(analysis->module, value_id);
    if (!loom_value_is_block_arg(value)) {
      const loom_op_t* defining_op = loom_value_def_op(value);
      if (defining_op && loom_buffer_alloca_isa(defining_op) &&
          loom_buffer_alloca_result(defining_op) == value_id) {
        analysis->entries[value_ordinal] = (loom_storage_interference_entry_t){
            .flags = LOOM_STORAGE_INTERFERENCE_ENTRY_ROOT |
                     LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE,
            .memory_space = loom_buffer_alloca_memory_space(defining_op),
        };
        analysis->entries[value_ordinal].flags |=
            loom_value_fact_table_block_may_repeat(analysis->fact_table,
                                                   defining_op->parent_block)
                ? LOOM_STORAGE_INTERFERENCE_ENTRY_REPEATED
                : LOOM_STORAGE_INTERFERENCE_ENTRY_SINGLE_INSTANCE;
      }
    }

    if (!loom_storage_interference_value_is_reference(analysis, value_id)) {
      continue;
    }
    loom_value_id_t root_value_id = LOOM_VALUE_ID_INVALID;
    if (loom_storage_interference_resolve_reference_root(analysis, value_id,
                                                         &root_value_id)) {
      IREE_RETURN_IF_ERROR(loom_storage_interference_append_edge(
          analysis, root_value_id, value_id));
    }
  }

  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < analysis->value_domain->value_count; ++value_ordinal) {
    if (iree_any_bit_set(analysis->entries[value_ordinal].flags,
                         LOOM_STORAGE_INTERFERENCE_ENTRY_ROOT)) {
      IREE_RETURN_IF_ERROR(loom_storage_interference_add_membership(
          analysis, value_ordinal, value_ordinal));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_storage_interference_analyze_function(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain, loom_func_like_t function,
    iree_arena_allocator_t* arena, loom_storage_interference_t** out_analysis) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  IREE_ASSERT_ARGUMENT(function.op);
  IREE_ASSERT_ARGUMENT(loom_func_like_body(function));
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_analysis);
  *out_analysis = NULL;

  loom_storage_interference_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*analysis), (void**)&analysis));
  *analysis = (loom_storage_interference_t){
      .module = module,
      .fact_table = fact_table,
      .value_domain = value_domain,
      .arena = arena,
      .function = function,
  };
  loom_control_uniformity_info_initialize(module, fact_table, arena,
                                          &analysis->control_uniformity);
  const iree_host_size_t value_count = value_domain->value_count;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, value_count,
                                                   sizeof(*analysis->entries),
                                                   (void**)&analysis->entries));
    memset(analysis->entries, 0, value_count * sizeof(*analysis->entries));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, value_count,
                                                   sizeof(*analysis->edges),
                                                   (void**)&analysis->edges));
    memset(analysis->edges, 0, value_count * sizeof(*analysis->edges));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*analysis->memberships),
        (void**)&analysis->memberships));
    memset(analysis->memberships, 0,
           value_count * sizeof(*analysis->memberships));
  }

  IREE_RETURN_IF_ERROR(loom_storage_interference_initialize_values(analysis));
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, loom_func_like_body(function), LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){
          .fn = loom_storage_interference_walk_op,
          .user_data = analysis,
      },
      &walk_result));
  IREE_ASSERT_EQ(walk_result, LOOM_WALK_CONTINUE);
  IREE_RETURN_IF_ERROR(
      loom_storage_interference_propagate_memberships(analysis));

  if (analysis->has_unknown_memory_access) {
    for (loom_value_ordinal_t value_ordinal = 0;
         value_ordinal < value_domain->value_count; ++value_ordinal) {
      analysis->entries[value_ordinal].flags &=
          ~LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE;
    }
  }
  loom_storage_interference_lifetime_scratch_t lifetime = {0};
  iree_arena_initialize(arena->block_pool, &lifetime.arena);
  iree_status_t status = iree_ok_status();
  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < value_domain->value_count && iree_status_is_ok(status);
       ++value_ordinal) {
    status = loom_storage_interference_record_value_accesses(
        analysis, value_ordinal, &lifetime);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_storage_interference_prove_single_instances(analysis, &lifetime);
  }
  iree_arena_deinitialize(&lifetime.arena);
  if (iree_status_is_ok(status)) {
    *out_analysis = analysis;
  }
  return status;
}

static const loom_storage_interference_entry_t*
loom_storage_interference_lookup_root(
    const loom_storage_interference_t* analysis,
    loom_value_id_t root_value_id) {
  const loom_value_ordinal_t root_ordinal = loom_local_value_domain_try_ordinal(
      analysis->value_domain, root_value_id);
  if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return NULL;
  }
  const loom_storage_interference_entry_t* entry =
      &analysis->entries[root_ordinal];
  return iree_any_bit_set(entry->flags, LOOM_STORAGE_INTERFERENCE_ENTRY_ROOT)
             ? entry
             : NULL;
}

bool loom_storage_interference_root_may_be_accessed(
    const loom_storage_interference_t* analysis,
    loom_value_id_t root_value_id) {
  IREE_ASSERT_ARGUMENT(analysis);
  const loom_storage_interference_entry_t* entry =
      loom_storage_interference_lookup_root(analysis, root_value_id);
  return entry == NULL ||
         !iree_all_bits_set(entry->flags,
                            LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE) ||
         entry->footprint.operation_count != 0;
}

bool loom_storage_interference_root_has_single_live_instance(
    const loom_storage_interference_t* analysis,
    loom_value_id_t root_value_id) {
  const loom_storage_interference_entry_t* entry =
      loom_storage_interference_lookup_root(analysis, root_value_id);
  return entry &&
         iree_any_bit_set(entry->flags,
                          LOOM_STORAGE_INTERFERENCE_ENTRY_SINGLE_INSTANCE);
}

static iree_status_t loom_storage_interference_get_cfg_region(
    loom_storage_interference_t* analysis, const loom_region_t* region,
    loom_storage_interference_cfg_region_t** out_summary) {
  *out_summary = NULL;
  for (loom_storage_interference_cfg_region_t* summary = analysis->cfg_regions;
       summary; summary = summary->next) {
    if (summary->region == region) {
      *out_summary = summary;
      return iree_ok_status();
    }
  }

  loom_storage_interference_cfg_region_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(analysis->arena, sizeof(*summary), (void**)&summary));
  *summary = (loom_storage_interference_cfg_region_t){
      .region = region,
      .facts =
          loom_value_fact_table_lookup_cfg_region(analysis->fact_table, region),
      .next = analysis->cfg_regions,
  };
  if (summary->facts) {
    IREE_RETURN_IF_ERROR(loom_cfg_postdominance_build(
        &summary->facts->graph, analysis->arena, &summary->postdominance));
  }
  analysis->cfg_regions = summary;
  *out_summary = summary;
  return iree_ok_status();
}

static bool loom_storage_interference_block_postdominates(
    const loom_cfg_postdominance_t* postdominance, uint16_t postdominator,
    uint16_t block) {
  if (!postdominance->available) {
    return false;
  }
  if (postdominator == block) {
    return true;
  }
  const loom_cfg_postdominator_t* nodes = postdominance->nodes;
  if (nodes[postdominator].immediate_postdominator ==
          LOOM_CFG_POSTDOMINATOR_INVALID ||
      nodes[block].immediate_postdominator == LOOM_CFG_POSTDOMINATOR_INVALID ||
      nodes[postdominator].depth > nodes[block].depth) {
    return false;
  }
  uint32_t current = block;
  while (nodes[current].depth > nodes[postdominator].depth) {
    current = nodes[current].immediate_postdominator;
    if (current == LOOM_CFG_POSTDOMINATOR_INVALID) {
      return false;
    }
  }
  return current == postdominator;
}

static bool loom_storage_interference_project_to_region(
    const loom_op_t* operation, const loom_region_t* target_region,
    const loom_op_t** out_anchor) {
  *out_anchor = NULL;
  for (const loom_op_t* anchor = operation; anchor;
       anchor = anchor->parent_op) {
    if (!anchor->parent_block) {
      return false;
    }
    if (anchor->parent_block->parent_region == target_region) {
      *out_anchor = anchor;
      return true;
    }
    if (!anchor->parent_op ||
        loom_traits_is_isolated(anchor->parent_op->traits)) {
      return false;
    }
  }
  return false;
}

static iree_status_t loom_storage_interference_prove_order(
    loom_storage_interference_t* analysis, const loom_op_t* before,
    const loom_op_t* after, bool require_after_postdominance,
    bool* out_proven) {
  *out_proven = false;
  if (!before->parent_block || !after->parent_block ||
      before->parent_block->parent_region !=
          after->parent_block->parent_region) {
    return iree_ok_status();
  }
  if (before->parent_block == after->parent_block) {
    *out_proven = before->block_ordinal < after->block_ordinal;
    return iree_ok_status();
  }

  loom_storage_interference_cfg_region_t* summary = NULL;
  IREE_RETURN_IF_ERROR(loom_storage_interference_get_cfg_region(
      analysis, before->parent_block->parent_region, &summary));
  if (!summary->facts || !summary->facts->dominance.available ||
      !summary->postdominance.available) {
    return iree_ok_status();
  }
  const uint16_t before_block = before->parent_block->region_index;
  const uint16_t after_block = after->parent_block->region_index;
  if (before_block >= summary->facts->graph.block_count ||
      after_block >= summary->facts->graph.block_count) {
    return iree_ok_status();
  }
  *out_proven =
      require_after_postdominance
          ? loom_storage_interference_block_postdominates(
                &summary->postdominance, after_block, before_block)
          : loom_cfg_dominance_block_dominates(&summary->facts->dominance,
                                               before_block, after_block);
  return iree_ok_status();
}

static iree_status_t loom_storage_interference_prove_before_barrier(
    loom_storage_interference_t* analysis, const loom_op_t* operation,
    const loom_op_t* barrier, bool* out_proven) {
  *out_proven = false;
  if (!barrier->parent_block) {
    return iree_ok_status();
  }
  const loom_op_t* operation_anchor = NULL;
  if (!loom_storage_interference_project_to_region(
          operation, barrier->parent_block->parent_region, &operation_anchor)) {
    return iree_ok_status();
  }
  return loom_storage_interference_prove_order(
      analysis, operation_anchor, barrier,
      /*require_after_postdominance=*/true, out_proven);
}

static iree_status_t loom_storage_interference_prove_after_barrier(
    loom_storage_interference_t* analysis, const loom_op_t* barrier,
    const loom_op_t* operation, bool* out_proven) {
  *out_proven = false;
  for (const loom_op_t* barrier_anchor = barrier; barrier_anchor;
       barrier_anchor = barrier_anchor->parent_op) {
    if (!barrier_anchor->parent_block) {
      return iree_ok_status();
    }
    const loom_op_t* operation_anchor = NULL;
    if (loom_storage_interference_project_to_region(
            operation, barrier_anchor->parent_block->parent_region,
            &operation_anchor)) {
      iree_status_t status = loom_storage_interference_prove_order(
          analysis, barrier_anchor, operation_anchor,
          /*require_after_postdominance=*/false, out_proven);
      if (iree_status_is_ok(status) && !*out_proven) {
        status = loom_storage_interference_prove_order(
            analysis, barrier_anchor, operation_anchor,
            /*require_after_postdominance=*/true, out_proven);
      }
      return status;
    }
    if (!barrier_anchor->parent_op ||
        loom_traits_is_isolated(barrier_anchor->parent_op->traits)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_storage_interference_operations_share_loop(
    const loom_storage_interference_t* analysis, const loom_op_t* lhs,
    const loom_op_t* rhs) {
  for (const loom_op_t* lhs_ancestor = lhs->parent_op; lhs_ancestor;
       lhs_ancestor = lhs_ancestor->parent_op) {
    if (!loom_loop_like_isa(
            loom_loop_like_cast(analysis->module, (loom_op_t*)lhs_ancestor))) {
      continue;
    }
    for (const loom_op_t* rhs_ancestor = rhs->parent_op; rhs_ancestor;
         rhs_ancestor = rhs_ancestor->parent_op) {
      if (lhs_ancestor == rhs_ancestor) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_storage_interference_operations_share_cfg_cycle(
    const loom_storage_interference_t* analysis, const loom_op_t* lhs,
    const loom_op_t* rhs) {
  for (const loom_op_t* lhs_anchor = lhs; lhs_anchor;
       lhs_anchor = lhs_anchor->parent_op) {
    if (!lhs_anchor->parent_block) {
      return true;
    }
    const loom_region_t* region = lhs_anchor->parent_block->parent_region;
    const loom_op_t* rhs_anchor = NULL;
    if (loom_storage_interference_project_to_region(rhs, region, &rhs_anchor) &&
        (region->block_count > 1 ||
         iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG))) {
      const loom_value_fact_cfg_region_t* facts =
          loom_value_fact_table_lookup_cfg_region(analysis->fact_table, region);
      if (!facts ||
          lhs_anchor->parent_block->region_index >= facts->graph.block_count ||
          rhs_anchor->parent_block->region_index >= facts->graph.block_count) {
        return true;
      }
      const loom_cfg_block_info_t* lhs_info =
          &facts->graph.blocks[lhs_anchor->parent_block->region_index];
      const loom_cfg_block_info_t* rhs_info =
          &facts->graph.blocks[rhs_anchor->parent_block->region_index];
      if (lhs_info->component == rhs_info->component &&
          lhs_info->component_is_cyclic) {
        return true;
      }
    }
    if (!lhs_anchor->parent_op ||
        loom_traits_is_isolated(lhs_anchor->parent_op->traits)) {
      break;
    }
  }
  return false;
}

static bool loom_storage_interference_footprints_cross_cycle(
    const loom_storage_interference_t* analysis,
    const loom_storage_interference_footprint_t* lhs,
    const loom_storage_interference_footprint_t* rhs) {
  for (iree_host_size_t i = 0; i < lhs->operation_count; ++i) {
    for (iree_host_size_t j = 0; j < rhs->operation_count; ++j) {
      if (loom_storage_interference_operations_share_loop(
              analysis, lhs->operations[i], rhs->operations[j]) ||
          loom_storage_interference_operations_share_cfg_cycle(
              analysis, lhs->operations[i], rhs->operations[j])) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t loom_storage_interference_barrier_separates(
    loom_storage_interference_t* analysis,
    const loom_storage_interference_footprint_t* before,
    const loom_op_t* barrier,
    const loom_storage_interference_footprint_t* after, bool* out_separates) {
  *out_separates = false;
  const bool crosses_cycle =
      loom_storage_interference_footprints_cross_cycle(analysis, before, after);
  if (crosses_cycle) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < before->operation_count; ++i) {
    bool proven = false;
    IREE_RETURN_IF_ERROR(loom_storage_interference_prove_before_barrier(
        analysis, before->operations[i], barrier, &proven));
    if (!proven) {
      return iree_ok_status();
    }
  }
  for (iree_host_size_t i = 0; i < after->operation_count; ++i) {
    bool proven = false;
    IREE_RETURN_IF_ERROR(loom_storage_interference_prove_after_barrier(
        analysis, barrier, after->operations[i], &proven));
    if (!proven) {
      return iree_ok_status();
    }
  }
  *out_separates = true;
  return iree_ok_status();
}

iree_status_t loom_storage_interference_prove_workgroup_nonoverlap(
    loom_storage_interference_t* analysis, loom_value_id_t lhs_root_value_id,
    loom_value_id_t rhs_root_value_id, bool* out_proven) {
  IREE_ASSERT_ARGUMENT(analysis);
  IREE_ASSERT_ARGUMENT(out_proven);
  *out_proven = false;
  if (lhs_root_value_id == rhs_root_value_id) {
    return iree_ok_status();
  }
  const loom_storage_interference_entry_t* lhs =
      loom_storage_interference_lookup_root(analysis, lhs_root_value_id);
  const loom_storage_interference_entry_t* rhs =
      loom_storage_interference_lookup_root(analysis, rhs_root_value_id);
  if (!lhs || !rhs ||
      lhs->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      rhs->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      !iree_all_bits_set(lhs->flags,
                         LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE) ||
      !iree_all_bits_set(rhs->flags,
                         LOOM_STORAGE_INTERFERENCE_ENTRY_COMPLETE)) {
    return iree_ok_status();
  }
  if (lhs->footprint.operation_count == 0 ||
      rhs->footprint.operation_count == 0) {
    *out_proven = true;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_control_uniformity_prove_mutually_exclusive_execution(
          &analysis->control_uniformity, lhs->footprint.operation_count,
          lhs->footprint.operations, rhs->footprint.operation_count,
          rhs->footprint.operations, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP,
          out_proven));
  if (*out_proven) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < analysis->workgroup_barrier_count; ++i) {
    if (!analysis->workgroup_barriers[i].uniform) {
      continue;
    }
    const loom_op_t* barrier = analysis->workgroup_barriers[i].op;
    bool separates = false;
    IREE_RETURN_IF_ERROR(loom_storage_interference_barrier_separates(
        analysis, &lhs->footprint, barrier, &rhs->footprint, &separates));
    if (!separates) {
      IREE_RETURN_IF_ERROR(loom_storage_interference_barrier_separates(
          analysis, &rhs->footprint, barrier, &lhs->footprint, &separates));
    }
    if (separates) {
      *out_proven = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}
