// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/channel_plan.h"

#include "loom/analysis/value_relation.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/type_registry.h"

typedef struct loom_channel_plan_membership_t {
  // Borrowed local value domain for the immutable source snapshot.
  const loom_local_value_domain_t* domain;
  // Disjoint-set parents for channel membership, indexed by local ordinal.
  loom_value_ordinal_t* parents;
  // Known channel identity for each current representative.
  const loom_channel_identity_t** channels;
  // Borrowed summaries indexed by module symbol ID, or NULL.
  const loom_channel_plan_callable_t* const* callables;
  // Disjoint-set forwarding classes for owned accesses, excluding channels.
  loom_value_ordinal_t* admission_parents;
  // Source admission ordinal for each class, or INVALID for ambiguity.
  loom_value_ordinal_t* admissions;
  // Whether a value receives an owned access through a forwarding edge.
  bool* admission_forwarded;
  // Number of distinct roots in each class, saturated at two for ambiguity.
  uint8_t* admission_origins;
} loom_channel_plan_membership_t;

static_assert(LOOM_OP_CHANNEL_COUNT_ == 10,
              "new channel operations require a retained action kind");

// Classifies one retained action at the collection boundary. Channel-aware
// calls are identified by their resolved summary; all remaining retained
// actions are verified Channel-dialect operations.
static loom_channel_plan_action_kind_t loom_channel_plan_classify_action(
    const loom_op_t* op, const loom_channel_plan_callable_t* callable) {
  if (callable) {
    return LOOM_CHANNEL_PLAN_ACTION_KIND_CALL;
  }
  switch (op->kind) {
    case LOOM_OP_CHANNEL_BIND:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_BIND;
    case LOOM_OP_CHANNEL_RESERVE:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE;
    case LOOM_OP_CHANNEL_ACCEPT:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_ACCEPT;
    case LOOM_OP_CHANNEL_WAIT:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_WAIT;
    case LOOM_OP_CHANNEL_ACQUIRE:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE;
    case LOOM_OP_CHANNEL_PUBLISH:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_PUBLISH;
    case LOOM_OP_CHANNEL_RELEASE:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_RELEASE;
    case LOOM_OP_CHANNEL_FANOUT:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_FANOUT;
    case LOOM_OP_CHANNEL_COPY:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_COPY;
    case LOOM_OP_CHANNEL_SELECT:
      return LOOM_CHANNEL_PLAN_ACTION_KIND_SELECT;
    default:
      IREE_ASSERT_UNREACHABLE("retained action must be a channel op or call");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_channel_plan_tracks_type(loom_type_t type) {
  return loom_channel_type_isa(type) || loom_read_type_isa(type) ||
         loom_write_type_isa(type);
}

static loom_value_ordinal_t loom_channel_plan_admission_representative(
    loom_channel_plan_membership_t* membership, loom_value_ordinal_t ordinal) {
  while (membership->admission_parents[ordinal] != ordinal) {
    membership->admission_parents[ordinal] =
        membership->admission_parents[membership->admission_parents[ordinal]];
    ordinal = membership->admission_parents[ordinal];
  }
  return ordinal;
}

static void loom_channel_plan_forward_admission(
    loom_channel_plan_membership_t* membership, loom_value_id_t source,
    loom_value_id_t destination) {
  membership->admission_forwarded[loom_local_value_domain_ordinal(
      membership->domain, destination)] = true;
  loom_value_ordinal_t lhs = loom_channel_plan_admission_representative(
      membership, loom_local_value_domain_ordinal(membership->domain, source));
  loom_value_ordinal_t rhs = loom_channel_plan_admission_representative(
      membership,
      loom_local_value_domain_ordinal(membership->domain, destination));
  if (lhs == rhs) {
    return;
  }
  if (rhs < lhs) {
    const loom_value_ordinal_t temporary = lhs;
    lhs = rhs;
    rhs = temporary;
  }
  membership->admission_parents[rhs] = lhs;
}

static loom_value_ordinal_t loom_channel_plan_representative(
    loom_channel_plan_membership_t* membership, loom_value_ordinal_t ordinal) {
  while (membership->parents[ordinal] != ordinal) {
    membership->parents[ordinal] =
        membership->parents[membership->parents[ordinal]];
    ordinal = membership->parents[ordinal];
  }
  return ordinal;
}

static bool loom_channel_plan_connect(
    loom_channel_plan_membership_t* membership, loom_value_id_t source,
    loom_value_id_t destination, const loom_op_t* op,
    loom_channel_plan_rejection_t* rejection) {
  loom_value_ordinal_t lhs = loom_channel_plan_representative(
      membership, loom_local_value_domain_ordinal(membership->domain, source));
  loom_value_ordinal_t rhs = loom_channel_plan_representative(
      membership,
      loom_local_value_domain_ordinal(membership->domain, destination));
  if (lhs == rhs) {
    return true;
  }
  const loom_channel_identity_t* lhs_channel = membership->channels[lhs];
  const loom_channel_identity_t* rhs_channel = membership->channels[rhs];
  if (lhs_channel && rhs_channel && lhs_channel != rhs_channel) {
    *rejection = (loom_channel_plan_rejection_t){
        .kind = LOOM_CHANNEL_PLAN_REJECTION_DYNAMIC_CHANNEL,
        .op = op,
        .value_id = destination,
    };
    return false;
  }
  // A stable ordinal order keeps representatives deterministic. Compression
  // turns the completed relation into direct entries before publication.
  if (rhs < lhs) {
    const loom_value_ordinal_t temporary = lhs;
    lhs = rhs;
    rhs = temporary;
  }
  membership->parents[rhs] = lhs;
  membership->channels[lhs] = lhs_channel ? lhs_channel : rhs_channel;
  return true;
}

static bool loom_channel_plan_collect_membership(
    loom_channel_plan_membership_t* membership, const loom_op_t* op,
    const loom_channel_plan_callable_t** out_callable,
    loom_channel_plan_rejection_t* rejection) {
  const loom_module_t* module = membership->domain->module;
  *out_callable = NULL;
  if (loom_channel_acquire_isa(op) || loom_channel_reserve_isa(op) ||
      loom_channel_accept_isa(op)) {
    return loom_channel_plan_connect(membership, loom_op_operands(op)[0],
                                     loom_op_results(op)[0], op, rejection);
  }
  if (loom_channel_fanout_isa(op)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      loom_channel_plan_forward_admission(
          membership, loom_channel_fanout_read(op), loom_op_results(op)[i]);
      if (!loom_channel_plan_connect(membership, loom_channel_fanout_read(op),
                                     loom_op_results(op)[i], op, rejection)) {
        return false;
      }
    }
    return true;
  }

  if (loom_func_call_isa(op)) {
    loom_value_id_t first_channel_value = LOOM_VALUE_ID_INVALID;
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      const loom_value_id_t value = loom_op_operands(op)[i];
      if (loom_channel_plan_tracks_type(
              loom_module_value_type(module, value))) {
        first_channel_value = value;
        break;
      }
    }
    for (uint16_t i = 0;
         i < op->result_count && first_channel_value == LOOM_VALUE_ID_INVALID;
         ++i) {
      const loom_value_id_t value = loom_op_results(op)[i];
      if (loom_channel_plan_tracks_type(
              loom_module_value_type(module, value))) {
        first_channel_value = value;
      }
    }
    if (first_channel_value == LOOM_VALUE_ID_INVALID) {
      return true;
    }
    const loom_symbol_ref_t callee = loom_func_call_callee(op);
    const loom_channel_plan_callable_t* callable =
        membership->callables && callee.module_id == 0
            ? membership->callables[callee.symbol_id]
            : NULL;
    if (!callable) {
      *rejection = (loom_channel_plan_rejection_t){
          .kind = LOOM_CHANNEL_PLAN_REJECTION_OPAQUE_USE,
          .op = op,
          .value_id = first_channel_value,
      };
      return false;
    }
    for (uint16_t i = 0; i < op->result_count; ++i) {
      const uint16_t argument = callable->result_arguments[i];
      if (argument != UINT16_MAX &&
          !loom_channel_plan_connect(membership, loom_op_operands(op)[argument],
                                     loom_op_results(op)[i], op, rejection)) {
        return false;
      }
    }
    *out_callable = callable;
    return true;
  }
  if (loom_op_dialect_id(op->kind) == LOOM_DIALECT_CHANNEL) {
    // Wait/publish/release/copy transport obligations without joining their
    // channels. Select's rank is ordinary data; its read retains membership.
    if (loom_channel_select_isa(op)) {
      return loom_channel_plan_connect(
          membership, loom_channel_select_channel(op),
          loom_channel_select_read(op), op, rejection);
    }
    return true;
  }

  const loom_value_relation_mask_t mask =
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_SELECT_PAYLOAD) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS);
  loom_value_relation_iterator_t iterator;
  loom_value_relation_iterator_initialize(module, op, mask, &iterator);
  loom_value_relation_t relation;
  bool forwards_channel = false;
  while (loom_value_relation_iterator_next(&iterator, &relation)) {
    if (!loom_channel_plan_tracks_type(
            loom_module_value_type(module, relation.destination_value_id))) {
      continue;
    }
    forwards_channel = true;
    const loom_type_t destination_type =
        loom_module_value_type(module, relation.destination_value_id);
    if (loom_read_type_isa(destination_type) ||
        loom_write_type_isa(destination_type)) {
      loom_channel_plan_forward_admission(membership, relation.source_value_id,
                                          relation.destination_value_id);
    }
    if (!loom_channel_plan_connect(membership, relation.source_value_id,
                                   relation.destination_value_id, op,
                                   rejection)) {
      return false;
    }
  }
  if (forwards_channel || loom_func_return_isa(op)) {
    return true;
  }
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_id_t value = loom_op_operands(op)[i];
    if (loom_channel_plan_tracks_type(loom_module_value_type(module, value))) {
      *rejection = (loom_channel_plan_rejection_t){
          .kind = LOOM_CHANNEL_PLAN_REJECTION_OPAQUE_USE,
          .op = op,
          .value_id = value,
      };
      return false;
    }
  }
  return true;
}

iree_status_t loom_channel_plan_build(
    const loom_local_value_domain_t* value_domain,
    const loom_channel_plan_binding_t* bindings, iree_host_size_t binding_count,
    const loom_channel_plan_callable_t* const* callables,
    iree_arena_allocator_t* arena, loom_channel_plan_t* out_plan,
    loom_channel_plan_rejection_t* out_rejection) {
  *out_plan = (loom_channel_plan_t){0};
  *out_rejection = (loom_channel_plan_rejection_t){
      .value_id = LOOM_VALUE_ID_INVALID,
  };
  const loom_module_t* module = value_domain->module;
  const loom_region_t* region = value_domain->region;
  const loom_value_ordinal_t value_count = value_domain->value_count;
  loom_channel_plan_membership_t membership = {
      .domain = value_domain,
      .callables = callables,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, value_count,
                                                 sizeof(*membership.parents),
                                                 (void**)&membership.parents));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, value_count,
                                                 sizeof(*membership.channels),
                                                 (void**)&membership.channels));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*membership.admission_parents),
      (void**)&membership.admission_parents));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*membership.admissions),
      (void**)&membership.admissions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*membership.admission_forwarded),
      (void**)&membership.admission_forwarded));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*membership.admission_origins),
      (void**)&membership.admission_origins));
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    membership.parents[i] = i;
    membership.channels[i] = NULL;
    membership.admission_parents[i] = i;
    membership.admissions[i] = LOOM_VALUE_ORDINAL_INVALID;
    membership.admission_forwarded[i] = false;
    membership.admission_origins[i] = 0;
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    membership.channels[loom_local_value_domain_ordinal(
        value_domain, bindings[i].value_id)] = bindings[i].channel;
  }

  loom_channel_plan_action_t* actions = NULL;
  iree_host_size_t action_count = 0;
  iree_host_size_t action_capacity = 0;
  loom_op_t** returns = NULL;
  iree_host_size_t return_count = 0;
  iree_host_size_t return_capacity = 0;
  for (uint16_t b = 0; b < region->block_count; ++b) {
    for (loom_op_t* op = region->blocks[b]->first_op; op; op = op->next_op) {
      IREE_ASSERT_EQ(op->region_count, 0);
      if (loom_channel_bind_isa(op)) {
        const loom_value_id_t value = loom_channel_bind_result(op);
        const loom_value_ordinal_t representative =
            loom_channel_plan_representative(
                &membership,
                loom_local_value_domain_ordinal(value_domain, value));
        if (membership.channels[representative]) {
          *out_rejection = (loom_channel_plan_rejection_t){
              .kind = LOOM_CHANNEL_PLAN_REJECTION_DYNAMIC_CHANNEL,
              .op = op,
              .value_id = value,
          };
          return iree_ok_status();
        }
        loom_channel_identity_t* identity = NULL;
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(arena, sizeof(*identity), (void**)&identity));
        identity->value_id = value;
        membership.channels[representative] = identity;
      }
      const loom_channel_plan_callable_t* callable = NULL;
      if (!loom_channel_plan_collect_membership(&membership, op, &callable,
                                                out_rejection)) {
        return iree_ok_status();
      }
      if (loom_func_return_isa(op)) {
        if (return_count == return_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              arena, return_count, return_count + 1, sizeof(*returns),
              &return_capacity, (void**)&returns));
        }
        returns[return_count++] = op;
      }
      if (!callable && loom_op_dialect_id(op->kind) != LOOM_DIALECT_CHANNEL) {
        continue;
      }
      if (action_count == action_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, action_count, action_count + 1, sizeof(*actions),
            &action_capacity, (void**)&actions));
      }
      actions[action_count++] = (loom_channel_plan_action_t){
          .op = op,
          .channel = NULL,
          .callable = callable,
          .kind = loom_channel_plan_classify_action(op, callable),
      };
    }
  }

  // Flatten every known identity onto its value before publishing the direct
  // table. Every representative points to itself, so filling other entries
  // cannot disturb identities still needed by later values.
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    membership.parents[i] = loom_channel_plan_representative(&membership, i);
    membership.admission_parents[i] =
        loom_channel_plan_admission_representative(&membership, i);
  }
  // Seed only after all forwarding classes are complete. The local value
  // domain already identifies definitions; no source or use-chain rescan is
  // needed. A class with multiple origins cannot name one admission.
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    if (membership.admission_forwarded[i]) {
      continue;
    }
    const loom_value_id_t value_id = value_domain->value_ids[i];
    const loom_type_t type = loom_module_value_type(module, value_id);
    if (!loom_read_type_isa(type) && !loom_write_type_isa(type)) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, value_id);
    const loom_op_t* op =
        loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
    const bool known =
        i < value_domain->definition_count && op &&
        (loom_channel_acquire_isa(op) || loom_channel_accept_isa(op) ||
         loom_channel_reserve_isa(op));
    const loom_value_ordinal_t representative = membership.admission_parents[i];
    if (known && membership.admission_origins[representative] == 0) {
      membership.admission_origins[representative] = 1;
      membership.admissions[representative] = i;
    } else {
      membership.admission_origins[representative] = 2;
      membership.admissions[representative] = LOOM_VALUE_ORDINAL_INVALID;
    }
  }
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    const loom_value_id_t value = value_domain->value_ids[i];
    const loom_channel_identity_t* channel =
        membership.channels[membership.parents[i]];
    membership.channels[i] = channel;
    const loom_value_ordinal_t admission =
        membership.admissions[membership.admission_parents[i]];
    membership.admissions[i] = admission;
    if (!channel &&
        loom_channel_plan_tracks_type(loom_module_value_type(module, value))) {
      *out_rejection = (loom_channel_plan_rejection_t){
          .kind = LOOM_CHANNEL_PLAN_REJECTION_UNBOUND_ACCESS,
          .value_id = value,
      };
      return iree_ok_status();
    }
  }
  loom_channel_plan_t plan = {
      .value_domain = value_domain,
      .channels = membership.channels,
      .admissions = membership.admissions,
      .value_count = value_count,
      .actions = actions,
      .action_count = action_count,
      .returns = returns,
      .return_count = return_count,
  };
  for (iree_host_size_t i = 0; i < action_count; ++i) {
    if (actions[i].kind == LOOM_CHANNEL_PLAN_ACTION_KIND_CALL) {
      continue;
    }
    loom_op_t* op = actions[i].op;
    actions[i].channel = loom_channel_plan_channel(
        &plan, actions[i].kind == LOOM_CHANNEL_PLAN_ACTION_KIND_BIND
                   ? loom_channel_bind_result(op)
                   : loom_op_operands(op)[0]);
    actions[i].destination_channel =
        actions[i].kind == LOOM_CHANNEL_PLAN_ACTION_KIND_COPY
            ? loom_channel_plan_channel(&plan,
                                        loom_channel_copy_destination(op))
            : NULL;
  }
  *out_plan = plan;
  return iree_ok_status();
}

iree_status_t loom_channel_plan_summarize(
    const loom_channel_plan_t* plan, loom_op_t* function,
    iree_arena_allocator_t* arena, loom_channel_plan_callable_t* out_callable,
    loom_channel_plan_rejection_t* out_rejection) {
  *out_callable = (loom_channel_plan_callable_t){0};
  *out_rejection = (loom_channel_plan_rejection_t){
      .value_id = LOOM_VALUE_ID_INVALID,
  };
  const loom_local_value_domain_t* domain = plan->value_domain;
  const loom_block_t* entry = loom_region_const_entry_block(domain->region);
  uint16_t* formal_arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, plan->value_count,
                                                 sizeof(*formal_arguments),
                                                 (void**)&formal_arguments));
  for (loom_value_ordinal_t i = 0; i < plan->value_count; ++i) {
    formal_arguments[i] = UINT16_MAX;
  }
  for (uint16_t i = 0; i < entry->arg_count; ++i) {
    formal_arguments[loom_local_value_domain_ordinal(domain,
                                                     entry->arg_ids[i])] = i;
  }
  uint16_t* result_arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, function->result_count,
                                                 sizeof(*result_arguments),
                                                 (void**)&result_arguments));
  for (uint16_t i = 0; i < function->result_count; ++i) {
    result_arguments[i] = UINT16_MAX;
  }
  for (iree_host_size_t r = 0; r < plan->return_count; ++r) {
    const loom_op_t* return_op = plan->returns[r];
    for (uint16_t i = 0; i < return_op->operand_count; ++i) {
      const loom_value_id_t value = loom_op_operands(return_op)[i];
      if (!loom_channel_plan_tracks_type(
              loom_module_value_type(domain->module, value))) {
        continue;
      }
      const loom_channel_identity_t* channel =
          loom_channel_plan_channel(plan, value);
      const uint16_t argument =
          formal_arguments[loom_local_value_domain_ordinal(domain,
                                                           channel->value_id)];
      if (argument == UINT16_MAX ||
          (r != 0 && result_arguments[i] != argument)) {
        *out_rejection = (loom_channel_plan_rejection_t){
            .kind = LOOM_CHANNEL_PLAN_REJECTION_DYNAMIC_CHANNEL,
            .op = return_op,
            .value_id = value,
        };
        return iree_ok_status();
      }
      result_arguments[i] = argument;
    }
  }
  *out_callable = (loom_channel_plan_callable_t){
      .function = function,
      .result_arguments = result_arguments,
  };
  return iree_ok_status();
}
