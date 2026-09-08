// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/composition.h"

#include <stdio.h>
#include <string.h>

#include "loom/analysis/pipeline_firing.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/rewriter.h"

enum { LOOM_AIE2P_PIPELINE_COMPOSITE_BUFFER_ALIGNMENT = 64 };

static iree_status_t loom_aie2p_pipeline_composition_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static bool loom_aie2p_pipeline_composition_symbol_refs_equal(
    loom_symbol_ref_t lhs, loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static bool loom_aie2p_pipeline_composition_flow_is_internal(
    const loom_pipeline_plan_flow_t* flow, uint32_t group_index) {
  return flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE &&
         flow->group_index == group_index;
}

static iree_status_t loom_aie2p_pipeline_composition_validate_group(
    const loom_pipeline_plan_t* plan, const loom_pipeline_firing_plan_t* firing,
    uint32_t group_index) {
  const loom_pipeline_firing_group_t* group_firing =
      &firing->groups[group_index];
  if (group_firing->completion_stage_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P frame-completion stages require a phased worker program");
  }
  const loom_pipeline_plan_group_t* group = &plan->groups[group_index];
  const loom_pipeline_plan_instance_t* instance =
      &plan->instances[group->instance_start];
  if (group_firing->fold_count != 0 &&
      (instance->fold_record_count != group_firing->records_per_frame ||
       instance->fold_output_count != group_firing->fold_count)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P folded worker requires compatible folds on every boundary "
        "output; mixed cadences require a phased worker program");
  }
  for (uint32_t i = 0; i < group_firing->record_stage_count; ++i) {
    const uint32_t stage_index =
        firing->stage_indices[group_firing->stage_start + i];
    const loom_pipeline_plan_stage_t* stage = &plan->stages[stage_index];
    for (uint16_t input_index = 0; input_index < stage->input_count;
         ++input_index) {
      const uint32_t flow_index =
          plan->stage_ports[stage->port_start + input_index].flow_index;
      const loom_pipeline_plan_flow_t* flow = &plan->flows[flow_index];
      if (loom_aie2p_pipeline_composition_flow_is_internal(flow, group_index) &&
          flow->minimum_capacity > 1) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P buffered same-group pipeline flow requires a composite "
            "ring state machine");
      }
    }
  }
  return iree_ok_status();
}

static bool loom_aie2p_pipeline_composition_find_group_port(
    const loom_pipeline_plan_t* plan, uint32_t group_index, uint32_t flow_index,
    uint32_t source_lane, loom_pipeline_plan_group_port_direction_t direction,
    uint32_t* out_port) {
  for (uint32_t i = 0; i < plan->group_port_count; ++i) {
    const loom_pipeline_plan_group_port_t* port = &plan->group_ports[i];
    if (port->group_index != group_index || port->direction != direction ||
        port->source_lane != source_lane) {
      continue;
    }
    if (direction == LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_RECEIVE) {
      if (port->flow_index != flow_index) continue;
    } else if (plan->flows[port->flow_index].storage_flow_index !=
               plan->flows[flow_index].storage_flow_index) {
      continue;
    }
    *out_port = port->port;
    return true;
  }
  return false;
}

static iree_status_t loom_aie2p_pipeline_composition_group_target(
    const loom_module_t* module, const loom_pipeline_plan_t* plan,
    uint32_t group_index, loom_symbol_ref_t* out_target) {
  *out_target = loom_symbol_ref_null();
  for (uint32_t stage_index = 0; stage_index < plan->stage_count;
       ++stage_index) {
    const loom_pipeline_plan_stage_t* stage = &plan->stages[stage_index];
    if (stage->group_index != group_index) continue;
    if (!loom_symbol_ref_is_valid(stage->entry) ||
        stage->entry.module_id != 0 ||
        stage->entry.symbol_id >= module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P composite stage entry is invalid");
    }
    const loom_op_t* entry_op =
        module->symbols.entries[stage->entry.symbol_id].defining_op;
    if (entry_op == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P composite stage entry is undefined");
    }
    const loom_func_like_t entry = loom_func_like_const_cast(module, entry_op);
    if (!loom_func_like_isa(entry)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P composite stage entry is not callable");
    }
    const loom_symbol_ref_t target = loom_func_like_target(entry);
    if (!loom_symbol_ref_is_valid(target)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P composite stage entry requires an exact core target");
    }
    if (!loom_symbol_ref_is_valid(*out_target)) {
      *out_target = target;
    } else if (!loom_aie2p_pipeline_composition_symbol_refs_equal(*out_target,
                                                                  target)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "same-group AIE2P pipeline stages must use one core target");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_composition_add_symbol(
    loom_module_t* module, const loom_pipeline_plan_t* plan,
    uint32_t group_index, iree_arena_allocator_t* arena,
    loom_symbol_ref_t* out_ref) {
  const loom_symbol_ref_t pipeline_ref = loom_func_like_callee(plan->pipeline);
  IREE_ASSERT_EQ(pipeline_ref.module_id, 0u);
  IREE_ASSERT_LT(pipeline_ref.symbol_id, module->symbols.count);
  const loom_string_id_t pipeline_name_id =
      module->symbols.entries[pipeline_ref.symbol_id].name_id;
  IREE_ASSERT_LT(pipeline_name_id, module->strings.count);
  const iree_string_view_t pipeline_name =
      module->strings.entries[pipeline_name_id];
  const iree_string_view_t infix = IREE_SV("$group$");
  iree_host_size_t name_capacity = 0;
  if (!iree_host_size_checked_add(pipeline_name.size, infix.size + 11,
                                  &name_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P composite symbol name overflow");
  }
  char* name_storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, name_capacity, (void**)&name_storage));
  memcpy(name_storage, pipeline_name.data, pipeline_name.size);
  memcpy(name_storage + pipeline_name.size, infix.data, infix.size);
  const int suffix_length =
      snprintf(name_storage + pipeline_name.size + infix.size, 11, "%" PRIu32,
               group_index);
  if (suffix_length < 0 || suffix_length >= 11) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P composite symbol suffix overflow");
  }
  const iree_host_size_t name_length =
      pipeline_name.size + infix.size + (iree_host_size_t)suffix_length;
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_make_string_view(name_storage, name_length), &name_id));
  if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "AIE2P composite symbol already exists");
  }
  out_ref->module_id = 0;
  return loom_module_add_symbol(module, name_id, &out_ref->symbol_id);
}

static uint32_t loom_aie2p_pipeline_composition_group_port_count(
    const loom_pipeline_plan_t* plan, uint32_t group_index) {
  uint32_t port_count = 0;
  for (uint32_t i = 0; i < plan->group_port_count; ++i) {
    const loom_pipeline_plan_group_port_t* port = &plan->group_ports[i];
    if (port->group_index == group_index && port->port >= port_count) {
      port_count = port->port + 1;
    }
  }
  return port_count;
}

static iree_status_t loom_aie2p_pipeline_composition_record_byte_length(
    const loom_pipeline_plan_flow_t* flow, uint64_t* out_byte_length) {
  uint64_t element_count = 0;
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(flow->tile_type));
  uint64_t bit_count = 0;
  if (!loom_type_static_element_count(flow->tile_type, &element_count) ||
      element_bit_count <= 0 ||
      !iree_checked_mul_u64(element_count, (uint64_t)element_bit_count,
                            &bit_count) ||
      (bit_count & 7u) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P composite flow requires a whole-byte static tile type");
  }
  *out_byte_length = bit_count / 8u;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_composition_allocate_flow_buffer(
    loom_builder_t* builder, const loom_pipeline_plan_flow_t* flow,
    loom_location_id_t location, loom_value_id_t* out_buffer) {
  uint64_t byte_length = 0;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_pipeline_composition_record_byte_length(flow, &byte_length));
  if (byte_length > INT64_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P composite flow tile is too large");
  }
  loom_op_t* byte_length_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64((int64_t)byte_length),
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), location, &byte_length_op));
  loom_op_t* alloca_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_buffer_alloca_build(builder, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
                               LOOM_AIE2P_PIPELINE_COMPOSITE_BUFFER_ALIGNMENT,
                               loom_index_constant_result(byte_length_op),
                               loom_type_buffer(), location, &alloca_op));
  *out_buffer = loom_buffer_alloca_result(alloca_op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_composition_build_group(
    loom_module_t* module, const loom_pipeline_plan_t* plan,
    const loom_pipeline_firing_plan_t* firing, uint32_t group_index,
    loom_symbol_ref_t target, loom_rewriter_t* rewriter,
    iree_arena_allocator_t* arena, loom_symbol_ref_t* out_entry,
    loom_op_t** out_function) {
  *out_entry = loom_symbol_ref_null();
  *out_function = NULL;
  const uint32_t port_count =
      loom_aie2p_pipeline_composition_group_port_count(plan, group_index);
  if (port_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P composite worker has too many ports");
  }
  loom_type_t* argument_types = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_allocate_array(
      arena, port_count, sizeof(*argument_types), (void**)&argument_types));
  for (uint32_t i = 0; i < port_count; ++i) {
    argument_types[i] = loom_type_buffer();
  }

  loom_value_id_t* flow_buffers = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_allocate_array(
      arena, plan->flow_count, sizeof(*flow_buffers), (void**)&flow_buffers));
  for (uint32_t i = 0; i < plan->flow_count; ++i) {
    flow_buffers[i] = LOOM_VALUE_ID_INVALID;
  }

  const loom_pipeline_firing_group_t* group_firing =
      &firing->groups[group_index];
  const uint32_t stage_count = group_firing->record_stage_count;
  loom_op_t** call_ops = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_allocate_array(
      arena, stage_count, sizeof(*call_ops), (void**)&call_ops));
  uint32_t maximum_stage_port_count = 0;
  for (uint32_t i = 0; i < stage_count; ++i) {
    const uint32_t stage_index =
        firing->stage_indices[group_firing->stage_start + i];
    const loom_pipeline_plan_stage_t* stage = &plan->stages[stage_index];
    const uint32_t stage_port_count =
        (uint32_t)stage->input_count + stage->output_count;
    maximum_stage_port_count =
        iree_max(maximum_stage_port_count, stage_port_count);
  }
  loom_value_id_t* operands = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_allocate_array(
      arena, maximum_stage_port_count, sizeof(*operands), (void**)&operands));

  loom_symbol_ref_t entry = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_add_symbol(
      module, plan, group_index, arena, &entry));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_builder_set_before(&builder, plan->pipeline.op);
  loom_op_t* function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &builder, LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET,
      /*visibility=*/0, /*retain=*/0, /*cc=*/0, /*purity=*/0,
      /*temperature=*/0, /*inline_policy=*/0, target, /*abi=*/0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), entry, argument_types, port_count,
      /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, plan->pipeline.op->location,
      &function_op));
  *out_function = function_op;
  const loom_func_like_t function = loom_func_like_cast(module, function_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  IREE_ASSERT_EQ(argument_count, port_count);
  loom_builder_enter_region(&builder, function_op,
                            loom_func_like_body(function));

  for (uint32_t i = 0; i < plan->group_port_count; ++i) {
    const loom_pipeline_plan_group_port_t* port = &plan->group_ports[i];
    if (port->group_index != group_index ||
        port->direction != LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND) {
      continue;
    }
    IREE_ASSERT_LT(port->port, port_count);
    const uint32_t storage_flow_index =
        plan->flows[port->flow_index].storage_flow_index;
    IREE_ASSERT_LT(storage_flow_index, plan->flow_count);
    flow_buffers[storage_flow_index] = arguments[port->port];
  }

  uint32_t call_count = 0;
  for (uint32_t i = 0; i < stage_count; ++i) {
    const uint32_t stage_index =
        firing->stage_indices[group_firing->stage_start + i];
    const loom_pipeline_plan_stage_t* stage = &plan->stages[stage_index];
    const uint32_t stage_port_count =
        (uint32_t)stage->input_count + stage->output_count;
    for (uint32_t port_index = 0; port_index < stage_port_count; ++port_index) {
      const loom_pipeline_plan_stage_port_t* stage_port =
          &plan->stage_ports[stage->port_start + port_index];
      const uint32_t flow_index = stage_port->flow_index;
      const loom_pipeline_plan_flow_t* flow = &plan->flows[flow_index];
      if (port_index < stage->input_count &&
          !loom_aie2p_pipeline_composition_flow_is_internal(flow,
                                                            group_index)) {
        uint32_t group_port = 0;
        if (!loom_aie2p_pipeline_composition_find_group_port(
                plan, group_index, flow_index, stage_port->source_lane,
                LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_RECEIVE, &group_port)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AIE2P composite stage input has no physical group port");
        }
        IREE_ASSERT_LT(group_port, port_count);
        operands[port_index] = arguments[group_port];
        continue;
      }
      const uint32_t storage_flow_index = flow->storage_flow_index;
      IREE_ASSERT_LT(storage_flow_index, plan->flow_count);
      loom_value_id_t* flow_buffer = &flow_buffers[storage_flow_index];
      if (*flow_buffer == LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(
            loom_aie2p_pipeline_composition_allocate_flow_buffer(
                &builder, &plan->flows[storage_flow_index],
                plan->pipeline.op->location, flow_buffer));
      }
      operands[port_index] = *flow_buffer;
    }
    loom_op_t* call_op = NULL;
    IREE_RETURN_IF_ERROR(loom_func_call_build(
        &builder, LOOM_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY,
        /*purity=*/0, /*temperature=*/0, LOOM_INLINE_POLICY_INLINE,
        stage->entry, operands, stage_port_count,
        /*result_types=*/NULL, /*result_count=*/0,
        /*tied_results=*/NULL, /*tied_result_count=*/0,
        plan->pipeline.op->location, &call_op));
    IREE_ASSERT_LT(call_count, stage_count);
    call_ops[call_count++] = call_op;
  }
  IREE_ASSERT_EQ(call_count, stage_count);
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_func_return_build(&builder, /*operands=*/NULL, /*operands_count=*/0,
                             plan->pipeline.op->location, &return_op));

  // Compose while stage arguments are ordinary source buffers. Once the
  // standalone stage callables reach Low their target entry buffers become
  // resources, which are resident ABI surfaces and cannot be inlined as local
  // values. The generic callable rewriter handles both linear and arbitrary
  // CFG stage bodies here without weakening that Low invariant.
  for (uint32_t i = 0; i < call_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_callable_inline_direct_call(rewriter, call_ops[i]));
  }
  *out_entry = entry;
  return iree_ok_status();
}

iree_status_t loom_aie2p_pipeline_composition_erase(
    loom_module_t* module, loom_aie2p_pipeline_composition_t* composition) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < composition->group_count; ++i) {
    if (composition->group_functions[i] == NULL) continue;
    status = iree_status_join(
        status, loom_op_erase(module, composition->group_functions[i]));
    composition->group_functions[i] = NULL;
  }
  return status;
}

iree_status_t loom_aie2p_pipeline_composition_materialize(
    loom_module_t* module, const loom_pipeline_plan_t* plan,
    iree_arena_allocator_t* arena,
    loom_aie2p_pipeline_composition_t* out_composition) {
  *out_composition = (loom_aie2p_pipeline_composition_t){0};
  loom_pipeline_firing_plan_t firing = {0};
  IREE_RETURN_IF_ERROR(loom_pipeline_firing_plan_build(plan, arena, &firing));
  loom_symbol_ref_t* instance_entries = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_allocate_array(
      arena, plan->instance_count, sizeof(*instance_entries),
      (void**)&instance_entries));
  loom_op_t** group_functions = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_allocate_array(
      arena, plan->group_count, sizeof(*group_functions),
      (void**)&group_functions));
  if (plan->group_count != 0) {
    memset(group_functions, 0, plan->group_count * sizeof(*group_functions));
  }
  *out_composition = (loom_aie2p_pipeline_composition_t){
      .instance_entries = instance_entries,
      .group_functions = group_functions,
      .group_count = plan->group_count,
  };

  loom_symbol_ref_t* group_targets = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_allocate_array(
      arena, plan->group_count, sizeof(*group_targets),
      (void**)&group_targets));
  for (uint32_t group_index = 0; group_index < plan->group_count;
       ++group_index) {
    group_targets[group_index] = loom_symbol_ref_null();
    if (plan->groups[group_index].stage_count == 1) continue;
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_validate_group(
        plan, &firing, group_index));
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_composition_group_target(
        module, plan, group_index, &group_targets[group_index]));
  }

  loom_rewriter_t rewriter = {0};
  iree_status_t status = loom_rewriter_initialize(&rewriter, module, arena);
  for (uint32_t group_index = 0;
       group_index < plan->group_count && iree_status_is_ok(status);
       ++group_index) {
    const loom_pipeline_plan_group_t* group = &plan->groups[group_index];
    loom_symbol_ref_t entry = loom_symbol_ref_null();
    if (group->stage_count == 1) {
      entry = plan->instances[group->instance_start].entry;
    } else {
      status = loom_aie2p_pipeline_composition_build_group(
          module, plan, &firing, group_index, group_targets[group_index],
          &rewriter, arena, &entry, &group_functions[group_index]);
    }
    if (!iree_status_is_ok(status)) break;
    for (uint32_t lane = 0; lane < group->lane_count; ++lane) {
      instance_entries[group->instance_start + lane] = entry;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, loom_aie2p_pipeline_composition_erase(module, out_composition));
  }
  return status;
}
