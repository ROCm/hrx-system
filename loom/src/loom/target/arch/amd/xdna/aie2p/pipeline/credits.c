// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/credits.h"

#include "loom/target/arch/amd/xdna/aie2p/pipeline/sram.h"

typedef struct loom_aie2p_native_credit_action_t {
  // Next private protocol restored on the same engine at this boundary.
  struct loom_aie2p_native_credit_action_t* next;
  // Physical protocol whose construction is entering or ending.
  const loom_aie2p_native_channel_t* channel;
  // Writer's admitted aperture for exact residual retirement, otherwise NULL.
  const loom_aie2p_native_channel_access_t* access;
  // Exact unread records to reclaim; unused for quiescent restoration.
  uint32_t residual;
  // Boundary at which this action establishes or retires private credits.
  loom_aie2p_native_credit_phase_t phase;
} loom_aie2p_native_credit_action_t;

typedef struct loom_aie2p_native_credit_engine_t {
  // Selected actions at each construction boundary, in emission order.
  loom_aie2p_native_credit_action_t*
      actions[LOOM_AIE2P_NATIVE_CREDIT_PHASE_COUNT];
} loom_aie2p_native_credit_engine_t;

struct loom_aie2p_native_credit_scope_t {
  // Per-engine actions, allocated only for a scope with private restoration.
  loom_aie2p_native_credit_engine_t* engines;
  // Unassigned restorations, consumed when control routes select their engine.
  loom_aie2p_native_credit_action_t* pending;
  // All participants must rendezvous before any selected reset begins.
  bool quiescence;
  // All participants wait for fresh capacities before entering this child.
  bool initializes;
};

uint32_t loom_aie2p_native_initial_credits(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel) {
  const bool reserve_initial =
      channel->cursor.writer.worker != UINT32_MAX &&
      context->workers[channel->cursor.writer.worker].execution ==
          LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION &&
      channel->cursor.writer.maximum_admissions == 1 &&
      !channel->replay.record_count;
  return (channel->replay.record_count ? channel->replay.credits
                                       : channel->source->capacity) -
         reserve_initial;
}

static bool loom_aie2p_native_credit_residual(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocations_t* invocations, iree_host_size_t index,
    uint32_t* out_residual) {
  const loom_aie2p_native_channel_t* channel = &context->channels[index];
  const loom_aie2p_native_cursor_t* writer = &channel->cursor.writer;
  const loom_aie2p_native_cursor_t* reader = &channel->cursor.reader;
  const loom_pipeline_invocation_t* owner =
      invocations->channels[index].invocation;
  if (channel->replay.record_count || writer->shared ||
      !writer->exact_admissions ||
      context->workers[writer->worker].execution !=
          LOOM_AIE2P_NATIVE_EXECUTION_CORE ||
      invocations->workers[writer->worker].invocation != owner) {
    return false;
  }
  uint32_t reads = 0;
  if (reader->worker != UINT32_MAX) {
    if (reader->shared || !reader->exact_admissions ||
        invocations->workers[reader->worker].invocation != owner) {
      return false;
    }
    reads = reader->maximum_admissions;
  }
  if (writer->maximum_admissions < reads ||
      writer->maximum_admissions - reads > channel->source->capacity) {
    return false;
  }
  *out_residual = writer->maximum_admissions - reads;
  return true;
}

iree_status_t loom_aie2p_native_select_credits(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t* out_control_count) {
  *out_control_count = 0;
  const loom_pipeline_invocations_t* invocations = &realization->invocations;
  if (!context->execution.schedule_count) {
    return iree_ok_status();
  }
  iree_arena_allocator_t* arena = context->pass->arena;
  iree_host_size_t* boundaries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->count, sizeof(*boundaries), (void**)&boundaries));
  boundaries[0] = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 1; i < invocations->count; ++i) {
    const loom_pipeline_invocation_t* invocation = invocations->entries[i];
    boundaries[i] =
        invocation->schedule ? i : boundaries[invocation->parent->ordinal];
  }
  const iree_host_size_t engine_count = context->execution.engine_count;
  for (iree_host_size_t i = 0; i < invocations->channel_count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    const iree_host_size_t boundary =
        boundaries[invocations->channels[i].invocation->ordinal];
    if (boundary == IREE_HOST_SIZE_MAX ||
        (channel->balance.balanced && !channel->initializes_on_entry)) {
      continue;
    }
    uint32_t residual = 0;
    const bool exact =
        !channel->initializes_on_entry &&
        loom_aie2p_native_credit_residual(context, invocations, i, &residual);
    if (exact && !residual) {
      continue;
    }
    if (!context->credit_scopes) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, invocations->count, sizeof(*context->credit_scopes),
          (void**)&context->credit_scopes));
      memset(context->credit_scopes, 0,
             invocations->count * sizeof(*context->credit_scopes));
    }
    struct loom_aie2p_native_credit_scope_t* scope =
        &context->credit_scopes[boundary];
    if (!scope->engines) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, engine_count,
                                                     sizeof(*scope->engines),
                                                     (void**)&scope->engines));
      memset(scope->engines, 0, engine_count * sizeof(*scope->engines));
    }
    const iree_host_size_t worker = channel->cursor.writer.worker;
    const loom_aie2p_native_credit_phase_t phase =
        channel->initializes_on_entry ? LOOM_AIE2P_NATIVE_CREDIT_INITIALIZE
        : exact                       ? LOOM_AIE2P_NATIVE_CREDIT_RETIRE
                                      : LOOM_AIE2P_NATIVE_CREDIT_RESET;
    loom_aie2p_native_credit_action_t** actions =
        exact ? &scope->engines[context->core_engines[worker]].actions[phase]
              : &scope->pending;
    loom_aie2p_native_credit_action_t* action = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*action), (void**)&action));
    *action = (loom_aie2p_native_credit_action_t){
        .next = *actions,
        .channel = channel,
        .phase = phase,
        .access =
            exact ? &context->workers[worker]
                         .channels
                         .accesses[context->workers[worker].channels.indices[i]]
                  : NULL,
        .residual = residual};
    *actions = action;
    scope->quiescence |= !exact && !channel->initializes_on_entry;
    scope->initializes |= channel->initializes_on_entry;
    *out_control_count += exact ? 0 : 2;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_route_credits(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, uint32_t* next_source,
    uint8_t* next_controller, bool* out_valid) {
  *out_valid = false;
  if (!context->credit_scopes) {
    *out_valid = true;
    return iree_ok_status();
  }
  const iree_host_size_t engine_count = context->execution.engine_count;
  for (iree_host_size_t boundary = 1; boundary < realization->invocations.count;
       ++boundary) {
    struct loom_aie2p_native_credit_scope_t* scope =
        &context->credit_scopes[boundary];
    while (scope->pending) {
      loom_aie2p_native_credit_action_t* action = scope->pending;
      const loom_aie2p_native_channel_t* channel = action->channel;
      // The common participant row already includes all concurrent descendants.
      // Any existing participant can restore private state after quiescence;
      // reuse an existing request/reply path before selecting a local path.
      const iree_host_size_t* participants =
          &context->execution.workers[boundary * engine_count];
      iree_host_size_t worker = IREE_HOST_SIZE_MAX;
      uint32_t best = 0;
      loom_aie2p_native_tile_t* tile = context->pool_tiles[channel->pool_index];
      const iree_host_size_t tile_index = tile - context->tiles;
      for (iree_host_size_t engine = 0; engine < engine_count; ++engine) {
        if (participants[engine] == IREE_HOST_SIZE_MAX) {
          continue;
        }
        const loom_aie2p_native_worker_t* controller =
            &context->workers[context->execution.workers[engine]];
        const uint32_t preference =
            controller->control_packets &&
                    controller->control_packets[tile_index] != UINT8_MAX
                ? (controller->control_replies[tile_index] != UINT8_MAX ? 3 : 2)
                : (controller->tile == tile ? 1 : 0);
        if (worker == IREE_HOST_SIZE_MAX || preference > best) {
          worker = participants[engine];
          best = preference;
        }
      }
      if (worker == IREE_HOST_SIZE_MAX) {
        return loom_aie2p_native_reject(
            context, channel->source->binding,
            IREE_SV("a resident participant for private channel restoration"));
      }
      bool valid = false;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_select_control_route(
          context, realization, worker, tile, true, next_source,
          next_controller, &valid));
      if (!valid) {
        return iree_ok_status();
      }
      scope->pending = action->next;
      loom_aie2p_native_credit_engine_t* engine =
          &scope->engines[context->core_engines[worker]];
      action->next = engine->actions[action->phase];
      engine->actions[action->phase] = action;
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

bool loom_aie2p_native_credits_require_quiescence(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocation_t* child) {
  return context->credit_scopes &&
         context->credit_scopes[child->ordinal].quiescence;
}

bool loom_aie2p_native_credits_require_initialization(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocation_t* child) {
  return context->credit_scopes &&
         context->credit_scopes[child->ordinal].initializes;
}

static iree_status_t loom_aie2p_native_reset_credits(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_worker_t* controller,
    const loom_aie2p_native_channel_t* channel) {
  loom_aie2p_worker_builder_t* code = &context->code;
  const loom_aie2p_native_tile_t* tile =
      context->pool_tiles[channel->pool_index];
  const iree_host_size_t tile_index = tile - context->tiles;
  const uint64_t origin =
      (uint64_t)tile->coordinate.column << context->family->column_shift |
      (uint64_t)tile->coordinate.row << context->family->row_shift;
  const uint16_t locks[] = {channel->free_lock, channel->ready_lock};
  const uint32_t values[] = {
      loom_aie2p_native_initial_credits(context, channel), 0};
  uint32_t address = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(locks); ++i) {
    loom_value_id_t value;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, builder,
        loom_xdna_register_field_encode(tile->memory_layout->lock_value,
                                        values[i]),
        &value));
    address = (uint32_t)(loom_xdna_register_field_address(
                             context->family, tile->memory_layout->lock_value,
                             tile->coordinate, &locks[i]) -
                         origin);
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        code, builder, controller->control_packets[tile_index], address, &value,
        1));
  }
  // A read on the same ordered control path acknowledges both writes. Merely
  // enqueueing them before the join would let another core enter too early.
  loom_value_id_t acknowledged[4];
  return loom_aie2p_worker_control_read(
      code, builder, controller->control_packets[tile_index],
      controller->control_replies[tile_index], address & ~15u,
      (address & 15u) / sizeof(uint32_t) + 1, acknowledged);
}

static iree_status_t loom_aie2p_native_retire_credits(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_region_t* region, const loom_aie2p_native_channel_access_t* access,
    uint32_t credits) {
  if (access->window) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_worker_lock(&context->code, builder,
                               AIE2P_CORE_DESCRIPTOR_REF_LOCK_ACQUIRE_IMMEDIATE,
                               access->locks.ready, -(int32_t)credits));
    return loom_aie2p_worker_lock(
        &context->code, builder,
        AIE2P_CORE_DESCRIPTOR_REF_LOCK_RELEASE_IMMEDIATE, access->locks.free,
        credits);
  }
  const uint16_t locks[] = {access->channel->ready_lock,
                            access->channel->free_lock};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(locks); ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_sram_lock(
        context, builder, region, access->channel, locks[i], i == 0, credits));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_emit_credits(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t engine, const loom_pipeline_invocation_t* child,
    loom_aie2p_native_credit_phase_t phase) {
  if (!context->credit_scopes ||
      !context->credit_scopes[child->ordinal].engines) {
    return iree_ok_status();
  }
  const loom_aie2p_native_credit_engine_t* row =
      &context->credit_scopes[child->ordinal].engines[engine];
  const loom_aie2p_native_credit_action_t* actions = row->actions[phase];
  if (!actions) {
    return iree_ok_status();
  }
  loom_aie2p_worker_builder_t* code = &context->code;
  code->worker_index = context->execution.workers[engine];
  const loom_aie2p_native_worker_t* controller =
      &context->workers[code->worker_index];
  loom_builder_t helper;
  loom_symbol_ref_t symbol;
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, NULL, 0, NULL, 0, &helper,
                                                &symbol, &function));
  for (const loom_aie2p_native_credit_action_t* action = actions; action;
       action = action->next) {
    if (phase != LOOM_AIE2P_NATIVE_CREDIT_RETIRE) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_reset_credits(
          context, &helper, controller, action->channel));
    } else {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_retire_credits(
          context, &helper, loom_low_func_def_body(function), action->access,
          action->residual));
    }
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, NULL, 0));
  loom_op_t* call = NULL;
  return loom_low_invoke_build(builder, 0, 0, 0, 0, symbol, NULL, 0, NULL, 0,
                               NULL, 0, LOOM_LOCATION_UNKNOWN, &call);
}
