// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/low/schedule_scope.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"

iree_status_t loom_low_schedule_scope_builder_append(
    loom_low_schedule_scope_builder_t* builder, const loom_op_t* op,
    loom_low_schedule_control_kind_t kind, uint16_t block_index,
    uint32_t node_index, iree_arena_allocator_t* arena) {
  if (builder->control_count >= UINT32_MAX - 1u) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native scheduling scope index capacity exceeded");
  }
  if (builder->control_count == builder->control_capacity) {
    const iree_host_size_t new_capacity =
        builder->control_capacity ? builder->control_capacity * 2 : 8;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, builder->control_count, new_capacity, sizeof(*builder->controls),
        &builder->control_capacity, (void**)&builder->controls));
  }
  builder->controls[builder->control_count++] = (loom_low_schedule_control_t){
      .op = op,
      .node_index = node_index,
      .block_index = block_index,
      .kind = kind,
      .scope_before = LOOM_LOW_SCHEDULE_SCOPE_UNREACHABLE,
      .scope_after = LOOM_LOW_SCHEDULE_SCOPE_UNREACHABLE,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_scope_emit_error(
    const loom_cfg_graph_t* graph, const loom_op_t* op,
    iree_string_view_t reason_key, const loom_op_t* begin_op,
    iree_diagnostic_emitter_t emitter, loom_low_schedule_scopes_t* scopes) {
  ++scopes->error_count;
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(graph->module, op)),
      loom_param_string(reason_key),
  };
  const loom_diagnostic_related_op_t related = {
      .label = IREE_SV("scope begins here"),
      .op = begin_op,
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_054,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = begin_op ? &related : NULL,
      .related_op_count = begin_op ? 1 : 0,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_low_schedule_scope_builder_finish(
    loom_low_schedule_scope_builder_t* builder, const loom_cfg_graph_t* graph,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_schedule_scopes_t* out_scopes) {
  *out_scopes = (loom_low_schedule_scopes_t){
      .controls = builder->controls,
      .control_count = builder->control_count,
  };
  if (builder->control_count == 0) {
    return iree_ok_status();
  }

  uint32_t* block_entry_scopes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, graph->block_count,
                                                 sizeof(*block_entry_scopes),
                                                 (void**)&block_entry_scopes));
  uint32_t* block_control_starts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count + 1, sizeof(*block_control_starts),
      (void**)&block_control_starts));
  uint32_t control_index = 0;
  for (uint16_t i = 0; i < graph->block_count; ++i) {
    block_entry_scopes[i] = LOOM_LOW_SCHEDULE_SCOPE_UNREACHABLE;
    block_control_starts[i] = control_index;
    while (control_index < builder->control_count &&
           builder->controls[control_index].block_index == i) {
      ++control_index;
    }
  }
  block_control_starts[graph->block_count] = control_index;
  block_entry_scopes[0] = 0;
  out_scopes->block_entry_scopes = block_entry_scopes;

  const iree_host_size_t reachable_count =
      graph->blocks ? graph->reverse_postorder.count : 1;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < reachable_count && iree_status_is_ok(status) &&
       out_scopes->error_count == 0;
       ++i) {
    const uint16_t block_index =
        graph->blocks ? graph->reverse_postorder.values[i] : 0;
    const loom_block_t* block = graph->region->blocks[block_index];
    uint32_t active_scope = block_entry_scopes[block_index];
    const uint32_t control_end = block_control_starts[block_index + 1];
    for (control_index = block_control_starts[block_index];
         control_index < control_end && iree_status_is_ok(status) &&
         out_scopes->error_count == 0;
         ++control_index) {
      loom_low_schedule_control_t* control = &builder->controls[control_index];
      control->scope_before = active_scope;
      if (control->kind == LOOM_LOW_SCHEDULE_CONTROL_BEGIN) {
        active_scope = control_index + 1;
        ++out_scopes->scope_count;
      } else if (active_scope == 0) {
        status = loom_low_schedule_scope_emit_error(graph, control->op,
                                                    IREE_SV("outside-scope"),
                                                    NULL, emitter, out_scopes);
      } else if (control->kind == LOOM_LOW_SCHEDULE_CONTROL_END) {
        active_scope = builder->controls[active_scope - 1].scope_before;
      }
      control->scope_after = active_scope;
    }
    if (!iree_status_is_ok(status) || out_scopes->error_count != 0) {
      break;
    }

    const loom_cfg_block_index_span_t successors =
        graph->blocks ? loom_cfg_graph_successors(graph, block_index)
                      : (loom_cfg_block_index_span_t){0};
    if (successors.count == 0 && active_scope != 0) {
      status = loom_low_schedule_scope_emit_error(
          graph, block->last_op, IREE_SV("unclosed-scope"),
          builder->controls[active_scope - 1].op, emitter, out_scopes);
    }
    for (iree_host_size_t j = 0;
         j < successors.count && iree_status_is_ok(status) &&
         out_scopes->error_count == 0;
         ++j) {
      uint32_t* successor_scope = &block_entry_scopes[successors.values[j]];
      if (*successor_scope == LOOM_LOW_SCHEDULE_SCOPE_UNREACHABLE) {
        *successor_scope = active_scope;
      } else if (*successor_scope != active_scope) {
        const loom_op_t* begin_op =
            active_scope != 0 ? builder->controls[active_scope - 1].op : NULL;
        status = loom_low_schedule_scope_emit_error(
            graph, block->last_op, IREE_SV("inconsistent-join"), begin_op,
            emitter, out_scopes);
      }
    }
  }
  return status;
}
