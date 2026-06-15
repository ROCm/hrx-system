// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame.h"

#include "loom/codegen/low/addressability.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/low/ops.h"

typedef enum loom_low_emission_frame_failure_e {
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS = 0,
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT = 1,
  LOOM_LOW_EMISSION_FRAME_FAILURE_ADDRESS_STATE_ITERATION_LIMIT = 2,
  LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS = 3,
} loom_low_emission_frame_failure_t;

static iree_status_t loom_low_emission_frame_liveness_order_from_schedule(
    loom_op_t* low_func_op, const loom_low_schedule_table_t* schedule,
    iree_arena_allocator_t* arena, loom_liveness_order_t* out_order) {
  *out_order = loom_liveness_order_empty();

  loom_region_t* body = loom_low_function_body(low_func_op);
  if (schedule->function_op != low_func_op ||
      schedule->block_count != body->block_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low emission-frame schedule must describe the low function body");
  }
  if (schedule->scheduled_node_count != 0 &&
      schedule->scheduled_node_indices == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low emission-frame schedule has packets but no packet index table");
  }

  loom_liveness_block_order_t* block_orders = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, body->block_count, sizeof(*block_orders), (void**)&block_orders));
  iree_host_size_t total_scheduled_nodes = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_low_schedule_block_t* schedule_block =
        &schedule->blocks[block_index];
    if (schedule_block->block != block ||
        schedule_block->scheduled_node_count != block->op_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low emission-frame schedule block %u does not "
                              "match the function body",
                              block_index);
    }
    const loom_op_t** ops = NULL;
    if (schedule_block->scheduled_node_count != 0) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, schedule_block->scheduled_node_count,
                                    sizeof(*ops), (void**)&ops));
    }
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < schedule_block->scheduled_node_count;
         ++scheduled_ordinal) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)schedule_block->scheduled_node_start +
          scheduled_ordinal;
      if (packet_index >= schedule->scheduled_node_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low emission-frame schedule block %u references packet %" PRIhsz
            " outside the schedule",
            block_index, packet_index);
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (node_index >= schedule->node_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low emission-frame schedule packet %" PRIhsz
                                " references node %" PRIu32,
                                packet_index, node_index);
      }
      const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
      if (node->block != block ||
          node->scheduled_ordinal != scheduled_ordinal) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low emission-frame schedule node does not match its block order");
      }
      ops[scheduled_ordinal] = node->op;
      ++total_scheduled_nodes;
    }
    block_orders[block_index] = (loom_liveness_block_order_t){
        .block = block,
        .ops = ops,
        .op_count = schedule_block->scheduled_node_count,
    };
  }
  if (total_scheduled_nodes != schedule->scheduled_node_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low emission-frame schedule has %" PRIhsz
        " scheduled packet(s) but body blocks cover %" PRIhsz,
        schedule->scheduled_node_count, total_scheduled_nodes);
  }
  *out_order = (loom_liveness_order_t){
      .blocks = block_orders,
      .block_count = body->block_count,
  };
  return iree_ok_status();
}

iree_status_t loom_low_emission_frame_build(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }

  *out_frame = (loom_low_emission_frame_t){
      .module = module,
      .function_op = low_func_op,
  };

  loom_low_schedule_options_t schedule_options = {
      .descriptor_registry = options->descriptor_registry,
      .target_selection = options->target_selection,
      .memory_access_table = options->memory_access_table,
      .pressure_cliffs = options->schedule_pressure_cliffs,
      .pair_affinities = options->schedule_pair_affinities,
      .emitter = options->emitter,
      .diagnostic_flags = options->schedule_diagnostic_flags,
      .strategy = options->schedule_strategy,
  };
  IREE_RETURN_IF_ERROR(loom_low_schedule_function(
      module, low_func_op, &schedule_options, arena, &out_frame->schedule));

  loom_low_storage_lease_table_t storage_leases = {0};
  if (options->storage_lease_provider != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_storage_lease_build(
        &out_frame->schedule, options->storage_lease_provider, arena,
        &storage_leases));
  }

  loom_liveness_order_t liveness_order = loom_liveness_order_empty();
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_liveness_order_from_schedule(
      low_func_op, &out_frame->schedule, arena, &liveness_order));
  loom_low_allocation_options_t allocation_options = {
      .liveness_order = liveness_order,
      .descriptor_registry = options->descriptor_registry,
      .target_selection = options->target_selection,
      .budgets = options->allocation_budgets,
      .budget_count = options->allocation_budget_count,
      .fixed_values = options->allocation_fixed_values,
      .fixed_value_count = options->allocation_fixed_value_count,
      .reserved_ranges = options->allocation_reserved_ranges,
      .reserved_range_count = options->allocation_reserved_range_count,
      .storage_leases = storage_leases,
      .emitter = options->emitter,
      .diagnostic_flags = options->allocation_diagnostic_flags,
  };
  IREE_RETURN_IF_ERROR(loom_low_allocate_function(
      module, low_func_op, &allocation_options, arena, &out_frame->allocation));

  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(&out_frame->schedule,
                                                       &out_frame->allocation));
  out_frame->target = out_frame->schedule.target;
  return iree_ok_status();
}

static iree_status_t loom_low_emission_frame_lower_spill_traffic(
    const loom_low_emission_frame_spill_free_options_t* options,
    loom_module_t* module, loom_op_t* low_func_op,
    iree_arena_allocator_t* arena) {
  if (options->lower_spill_traffic == NULL) {
    return iree_ok_status();
  }
  return options->lower_spill_traffic(options->lower_spill_traffic_user_data,
                                      module, low_func_op, arena);
}

static iree_status_t loom_low_emission_frame_materialize_address_state(
    const loom_low_emission_frame_spill_free_options_t* options,
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result) {
  *out_result = (loom_low_emission_frame_materialize_address_state_result_t){0};
  if (options->materialize_address_state == NULL) {
    return iree_ok_status();
  }
  return options->materialize_address_state(
      options->materialize_address_state_user_data, module, low_func_op, frame,
      arena, out_result);
}

static iree_status_t loom_low_emission_frame_validate_final(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* options,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    bool* out_accepted) {
  *out_accepted = false;
  loom_low_addressability_validation_result_t addressability_result = {0};
  IREE_RETURN_IF_ERROR(loom_low_addressability_validate_allocated_packets(
      &frame->schedule, &frame->allocation, frame_options->emitter,
      &addressability_result));
  if (addressability_result.error_count != 0) {
    return iree_ok_status();
  }
  if (options->validate_frame == NULL) {
    *out_accepted = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      options->validate_frame(options->validate_frame_user_data, frame, arena));
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_low_emission_frame_make_spill_assignment_status(
    const loom_low_emission_frame_t* frame) {
  const iree_string_view_t target_key =
      loom_low_diagnostic_target_key(&frame->target);
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low emission frame for target '%.*s' function '@%.*s' still has %zu "
      "spill-slot assignment(s) after spill traffic lowering",
      (int)target_key.size, target_key.data, (int)function_name.size,
      function_name.data, frame->allocation.spill_count);
}

static iree_status_t loom_low_emission_frame_make_iteration_limit_status(
    const loom_low_emission_frame_t* frame, iree_host_size_t iteration_count) {
  const iree_string_view_t target_key =
      loom_low_diagnostic_target_key(&frame->target);
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low emission frame for target '%.*s' function '@%.*s' did not reach a "
      "spill-free final frame after %zu spill materialization iteration(s); "
      "%zu spill plan(s) and %zu spill-slot assignment(s) remain",
      (int)target_key.size, target_key.data, (int)function_name.size,
      function_name.data, iteration_count, frame->allocation.spill_plan_count,
      frame->allocation.spill_count);
}

static iree_status_t
loom_low_emission_frame_make_address_state_iteration_limit_status(
    const loom_low_emission_frame_t* frame, iree_host_size_t iteration_count) {
  const iree_string_view_t target_key =
      loom_low_diagnostic_target_key(&frame->target);
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low emission frame for target '%.*s' function '@%.*s' did not reach a "
      "stable final frame after %zu address-state materialization iteration(s)",
      (int)target_key.size, target_key.data, (int)function_name.size,
      function_name.data, iteration_count);
}

static iree_status_t loom_low_emission_frame_make_no_progress_status(
    const loom_low_emission_frame_t* frame) {
  const iree_string_view_t target_key =
      loom_low_diagnostic_target_key(&frame->target);
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low emission frame for target '%.*s' function '@%.*s' made no spill "
      "materialization progress with %zu pending spill plan(s)",
      (int)target_key.size, target_key.data, (int)function_name.size,
      function_name.data, frame->allocation.spill_plan_count);
}

static iree_string_view_t loom_low_emission_frame_failure_code(
    loom_low_emission_frame_failure_t failure) {
  switch (failure) {
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS:
      return IREE_SV("remaining-spill-assignments");
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT:
      return IREE_SV("spill-materialization-iteration-limit");
    case LOOM_LOW_EMISSION_FRAME_FAILURE_ADDRESS_STATE_ITERATION_LIMIT:
      return IREE_SV("address-state-iteration-limit");
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS:
      return IREE_SV("spill-materialization-no-progress");
    default:
      return IREE_SV("<unknown>");
  }
}

static iree_status_t loom_low_emission_frame_emit_final_failure(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_t* frame,
    loom_low_emission_frame_failure_t failure, iree_host_size_t iteration_count,
    iree_host_size_t iteration_limit) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&frame->target)),
      loom_param_string(loom_low_diagnostic_export_name(&frame->target)),
      loom_param_string(loom_low_diagnostic_config_key(&frame->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(frame->module, frame->function_op)),
      loom_param_string(loom_low_emission_frame_failure_code(failure)),
      loom_param_u64((uint64_t)iteration_count),
      loom_param_u64((uint64_t)iteration_limit),
      loom_param_u64((uint64_t)frame->allocation.spill_plan_count),
      loom_param_u64((uint64_t)frame->allocation.spill_count),
      loom_param_u64((uint64_t)frame->schedule.scheduled_node_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = frame->function_op,
      .error = LOOM_ERR_BACKEND_021,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(frame_options->emitter, &emission);
}

static iree_status_t loom_low_emission_frame_fail_final(
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_t* frame,
    loom_low_emission_frame_failure_t failure, iree_host_size_t iteration_count,
    iree_host_size_t iteration_limit) {
  if (frame_options->emitter.fn != NULL) {
    return loom_low_emission_frame_emit_final_failure(
        frame_options, frame, failure, iteration_count, iteration_limit);
  }
  switch (failure) {
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS:
      return loom_low_emission_frame_make_spill_assignment_status(frame);
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT:
      return loom_low_emission_frame_make_iteration_limit_status(
          frame, iteration_count);
    case LOOM_LOW_EMISSION_FRAME_FAILURE_ADDRESS_STATE_ITERATION_LIMIT:
      return loom_low_emission_frame_make_address_state_iteration_limit_status(
          frame, iteration_count);
    case LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS:
      return loom_low_emission_frame_make_no_progress_status(frame);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "unknown low emission-frame failure");
  }
}

iree_status_t loom_low_emission_frame_build_spill_free(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* frame_options,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame) {
  *out_frame = (loom_low_emission_frame_t){0};
  if (spill_free_options == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "spill-free low emission frame construction requires spill-free "
        "options");
  }

  IREE_RETURN_IF_ERROR(loom_low_emission_frame_lower_spill_traffic(
      spill_free_options, module, low_func_op, arena));
  iree_host_size_t iteration_count = 0;
  iree_host_size_t iteration_limit = 0;
  iree_host_size_t address_state_iteration_count = 0;
  iree_host_size_t address_state_iteration_limit = 0;
  for (;;) {
    loom_low_emission_frame_t frame = {0};
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
        module, low_func_op, frame_options, arena, &frame));
    if (iteration_limit == 0) {
      if (frame.allocation.liveness.value_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low emission frame spill materialization iteration limit "
            "overflows host size");
      }
      iteration_limit = frame.allocation.liveness.value_count + 1;
    }
    if (frame.allocation.spill_plan_count == 0 &&
        frame.allocation.spill_count == 0) {
      if (address_state_iteration_limit == 0) {
        if (frame.schedule.scheduled_node_count == IREE_HOST_SIZE_MAX) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "low emission frame address-state materialization iteration "
              "limit overflows host size");
        }
        address_state_iteration_limit = frame.schedule.scheduled_node_count + 1;
      }
      loom_low_emission_frame_materialize_address_state_result_t
          address_state_result = {0};
      IREE_RETURN_IF_ERROR(loom_low_emission_frame_materialize_address_state(
          spill_free_options, module, low_func_op, &frame, arena,
          &address_state_result));
      if (address_state_result.error_count != 0) {
        return iree_ok_status();
      }
      if (address_state_result.changed) {
        if (address_state_iteration_count >= address_state_iteration_limit) {
          return loom_low_emission_frame_fail_final(
              frame_options, &frame,
              LOOM_LOW_EMISSION_FRAME_FAILURE_ADDRESS_STATE_ITERATION_LIMIT,
              address_state_iteration_count, address_state_iteration_limit);
        }
        ++address_state_iteration_count;
        continue;
      }
      bool accepted = false;
      IREE_RETURN_IF_ERROR(loom_low_emission_frame_validate_final(
          frame_options, spill_free_options, &frame, arena, &accepted));
      if (!accepted) {
        return iree_ok_status();
      }
      *out_frame = frame;
      return iree_ok_status();
    }
    if (frame.allocation.spill_plan_count == 0) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ASSIGNMENTS, iteration_count,
          iteration_limit);
    }
    if (iteration_count >= iteration_limit) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_ITERATION_LIMIT,
          iteration_count, iteration_limit);
    }

    loom_low_allocation_materialization_result_t result = {0};
    loom_low_allocation_materialization_options_t materialization_options =
        spill_free_options->materialization_options;
    materialization_options.allow_existing_storage_traffic = true;
    // The frame loop materializes one complete allocation snapshot at a time;
    // the next iteration accounts for any spill traffic introduced by it.
    materialization_options.max_spill_plan_count = 0;
    if (materialization_options.emitter.fn == NULL) {
      materialization_options.emitter = frame_options->emitter;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_spills(
        module, &frame.allocation, &materialization_options, arena, &result));
    if (result.error_count != 0) {
      return iree_ok_status();
    }
    if (result.storage_count == 0 && result.spill_count == 0 &&
        result.reload_count == 0) {
      return loom_low_emission_frame_fail_final(
          frame_options, &frame,
          LOOM_LOW_EMISSION_FRAME_FAILURE_SPILL_NO_PROGRESS, iteration_count,
          iteration_limit);
    }

    IREE_RETURN_IF_ERROR(loom_low_emission_frame_lower_spill_traffic(
        spill_free_options, module, low_func_op, arena));
    ++iteration_count;
  }
}
