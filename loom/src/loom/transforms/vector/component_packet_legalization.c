// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/component_packet_legalization.h"

#include <string.h>

#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/walk.h"

#define LOOM_VECTOR_COMPONENT_INDEX_INVALID UINT32_MAX

// Keeps static source expansion bounded independently of source program size.
// Larger components retain their authored aggregate and use the ordinary
// reference path instead of multiplying compiler work by packet count.
#define LOOM_VECTOR_COMPONENT_PACKET_OP_LIMIT 64u
#define LOOM_VECTOR_COMPONENT_MEMBER_LIMIT \
  (LOOM_VECTOR_COMPONENT_PACKET_OP_LIMIT / 2u)

typedef struct loom_vector_component_record_t {
  // Authored decomposable operation.
  loom_op_t* op;
  // Union-find parent record.
  uint32_t parent_index;
  // Next member in the union-find root's circular member list.
  uint32_t union_next_member_index;
  // Number of members owned by this union-find root.
  uint32_t union_member_count;
  // Latest block ordinal owned by this union-find root.
  uint64_t union_last_block_ordinal;
  // Compact component ordinal assigned after classification.
  uint32_t component_index;
  // Next component member in dominance order.
  uint32_t next_member_index;
  // Previous component member in dominance order.
  uint32_t previous_member_index;
  // Packet result produced while materializing one chunk.
  loom_value_id_t packet;
  // Logical lane count shared by the component.
  uint32_t lane_count;
  // Target-native packet candidates for this operation, by policy ordinal.
  uint64_t candidate_bits;
  // Operations created while materializing packets and an escaping aggregate.
  uint64_t created_op_count;
  // Authored target-contract result retained for downstream legalization.
  loom_target_contract_query_result_t authored_query_result;
  // True when non-operand references prevent ordinary SSA replacement.
  bool blocked;
  // True when this root contains an operation requiring packet legalization.
  bool demanded;
  // True when the result escapes the selected component.
  bool escapes;
} loom_vector_component_record_t;

typedef struct loom_vector_component_t {
  // First member in dominance order.
  uint32_t first_member_index;
  // Last member in dominance order.
  uint32_t last_member_index;
  // Number of member operations.
  uint32_t member_count;
  // Logical lane count shared by all members.
  uint32_t lane_count;
  // Selected packet lane count, or zero when the component remains authored.
  uint32_t packet_lane_count;
  // Number of packets required by |packet_lane_count|.
  uint32_t packet_count;
  // Target-native packet candidates shared by every member.
  uint64_t candidate_bits;
  // True when a member violates source IR rather than target capability.
  bool blocked;
} loom_vector_component_t;

struct loom_vector_component_plan_t {
  // Active target legalization context.
  loom_target_legalization_context_t* context;
  // Optional observer for completed authored-operation rewrites.
  loom_vector_component_packet_rewrite_callback_t rewrite_callback;
  // Dense one-based record index keyed by local value ordinal.
  uint32_t* value_record_indices;
  // Dominance-ordered decomposable operation records.
  loom_vector_component_record_t* records;
  // Number of initialized records.
  uint32_t record_count;
  // Maximum number of records backed by |records|.
  uint32_t record_capacity;
  // Compact components in first-member order.
  loom_vector_component_t* components;
  // Number of initialized components.
  uint32_t component_count;
  // Number of records in the backward closure of a target rejection.
  uint32_t demanded_record_count;
};

typedef struct loom_vector_component_external_t {
  // Current SSA value captured outside the component.
  loom_value_id_t source;
  // Authored aggregate vector type.
  loom_type_t source_type;
  // Rank-one row-major view used for packet slicing.
  loom_value_id_t linear_source;
  // Slice materialized for the current packet.
  loom_value_id_t packet;
} loom_vector_component_external_t;

static bool loom_vector_component_static_lane_count(loom_type_t type,
                                                    uint32_t* out_lane_count) {
  *out_lane_count = 0;
  uint64_t lane_count = 0;
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type) ||
      !loom_type_static_element_count(type, &lane_count) || lane_count < 1 ||
      lane_count > UINT32_MAX) {
    return false;
  }
  *out_lane_count = (uint32_t)lane_count;
  return true;
}

static loom_type_t loom_vector_component_packet_type(loom_type_t source_type,
                                                     uint32_t lane_count) {
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(source_type),
      loom_dim_pack_static(lane_count),
      loom_type_rank(source_type) == 1 ? source_type.encoding_id : 0);
}

// Describes the complete shape-preserving family without naming operation
// kinds. The decomposable trait establishes one result and per-lane semantics;
// this routine establishes the row-major packet interval shared by its static
// vector operands and result.
static bool loom_vector_component_describe(const loom_module_t* module,
                                           const loom_op_t* op,
                                           uint32_t* out_lane_count) {
  *out_lane_count = 0;
  if (op->result_count != 1 || op->region_count != 0 ||
      op->successor_count != 0 ||
      !iree_all_bits_set(loom_op_effective_traits(module, op),
                         LOOM_TRAIT_DECOMPOSABLE)) {
    return false;
  }
  uint32_t lane_count = 0;
  if (!loom_vector_component_static_lane_count(
          loom_module_value_type(module, loom_op_const_results(op)[0]),
          &lane_count) ||
      lane_count <= 1) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_type_t operand_type =
        loom_module_value_type(module, operands[i]);
    if (!loom_type_is_vector(operand_type)) {
      continue;
    }
    uint32_t operand_lane_count = 0;
    if (!loom_vector_component_static_lane_count(operand_type,
                                                 &operand_lane_count) ||
        operand_lane_count != lane_count) {
      return false;
    }
  }
  *out_lane_count = lane_count;
  return true;
}

static const loom_op_t* loom_vector_component_user_anchor(
    const loom_op_t* user_op, const loom_block_t* source_block) {
  const loom_op_t* anchor = user_op;
  while (anchor != NULL && anchor->parent_block != source_block) {
    anchor = anchor->parent_op;
  }
  return anchor;
}

static uint32_t loom_vector_component_find_root(
    loom_vector_component_plan_t* plan, uint32_t record_index) {
  uint32_t root_index = record_index;
  while (plan->records[root_index].parent_index != root_index) {
    root_index = plan->records[root_index].parent_index;
  }
  while (record_index != root_index) {
    const uint32_t parent_index = plan->records[record_index].parent_index;
    plan->records[record_index].parent_index = root_index;
    record_index = parent_index;
  }
  return root_index;
}

static uint32_t loom_vector_component_record_for_value(
    const loom_vector_component_plan_t* plan, loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal = loom_local_value_domain_try_ordinal(
      plan->context->value_domain, value_id);
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return LOOM_VECTOR_COMPONENT_INDEX_INVALID;
  }
  const uint32_t one_based_index = plan->value_record_indices[ordinal];
  return one_based_index == 0 ? LOOM_VECTOR_COMPONENT_INDEX_INVALID
                              : one_based_index - 1u;
}

static iree_status_t loom_vector_component_query(
    loom_vector_component_plan_t* plan, const loom_op_t* op,
    uint32_t source_lane_count, uint32_t packet_lane_count,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_vector_lane_projection_t projection = {
      .source_lane_count = source_lane_count,
      .projected_lane_count = packet_lane_count,
  };
  return loom_target_legalization_query_contract_with_vector_lane_projection(
      plan->context, op, projection, out_result);
}

static iree_status_t loom_vector_component_candidate_bits(
    loom_vector_component_plan_t* plan, const loom_op_t* op,
    uint32_t source_lane_count, uint64_t* out_candidate_bits) {
  *out_candidate_bits = 0;
  const loom_target_vector_packet_policy_t* policy =
      plan->context->vector_packet_policy;
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t packet_lane_count = policy->native_lane_counts[i];
    if (packet_lane_count == 0 || packet_lane_count >= source_lane_count) {
      continue;
    }
    loom_target_contract_query_result_t packet_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_vector_component_query(
        plan, op, source_lane_count, packet_lane_count, &packet_result));
    if (packet_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      continue;
    }
    const uint32_t tail_lane_count = source_lane_count % packet_lane_count;
    if (tail_lane_count != 0) {
      loom_target_contract_query_result_t tail_result =
          loom_target_contract_query_result_empty();
      IREE_RETURN_IF_ERROR(loom_vector_component_query(
          plan, op, source_lane_count, tail_lane_count, &tail_result));
      if (tail_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
        continue;
      }
    }
    *out_candidate_bits |= (uint64_t)1u << i;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_component_classify_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  (void)walk_context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_vector_component_plan_t* plan = (loom_vector_component_plan_t*)user_data;
  uint32_t lane_count = 0;
  if (!loom_vector_component_describe(plan->context->module, op, &lane_count)) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(plan->record_count, plan->record_capacity);
  const uint32_t record_index = plan->record_count++;
  loom_vector_component_record_t* record = &plan->records[record_index];
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_value_t* result_value =
      loom_module_value(plan->context->module, result);
  *record = (loom_vector_component_record_t){
      .op = op,
      .parent_index = record_index,
      .union_next_member_index = record_index,
      .union_member_count = 1,
      .union_last_block_ordinal = op->block_ordinal,
      .component_index = LOOM_VECTOR_COMPONENT_INDEX_INVALID,
      .next_member_index = LOOM_VECTOR_COMPONENT_INDEX_INVALID,
      .previous_member_index = LOOM_VECTOR_COMPONENT_INDEX_INVALID,
      .packet = LOOM_VALUE_ID_INVALID,
      .lane_count = lane_count,
  };
  IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
      plan->context, op, &record->authored_query_result));
  record->blocked =
      loom_value_has_attribute_uses(result_value) ||
      loom_module_value_has_type_uses(plan->context->module, result) ||
      record->authored_query_result.outcome ==
          LOOM_TARGET_CONTRACT_QUERY_INVALID_IR;
  record->demanded = record->authored_query_result.outcome !=
                         LOOM_TARGET_CONTRACT_QUERY_LEGAL &&
                     record->authored_query_result.outcome !=
                         LOOM_TARGET_CONTRACT_QUERY_INVALID_IR;
  if (record->demanded) {
    ++plan->demanded_record_count;
  }

  const loom_value_ordinal_t result_ordinal =
      loom_local_value_domain_ordinal(plan->context->value_domain, result);
  IREE_ASSERT_EQ(plan->value_record_indices[result_ordinal], 0u);
  plan->value_record_indices[result_ordinal] = record_index + 1u;
  return iree_ok_status();
}

static uint32_t loom_vector_component_operand_root(
    loom_vector_component_plan_t* plan,
    const loom_vector_component_record_t* record, uint16_t operand_index) {
  const loom_value_id_t operand =
      loom_op_const_operands(record->op)[operand_index];
  const uint32_t producer_index =
      loom_vector_component_record_for_value(plan, operand);
  if (producer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    return LOOM_VECTOR_COMPONENT_INDEX_INVALID;
  }
  const loom_vector_component_record_t* producer =
      &plan->records[producer_index];
  if (producer->op->parent_block != record->op->parent_block ||
      producer->lane_count != record->lane_count) {
    return LOOM_VECTOR_COMPONENT_INDEX_INVALID;
  }
  return loom_vector_component_find_root(plan, producer_index);
}

static bool loom_vector_component_root_is_first_operand(
    loom_vector_component_plan_t* plan,
    const loom_vector_component_record_t* record, uint16_t operand_index,
    uint32_t root_index) {
  for (uint16_t i = 0; i < operand_index; ++i) {
    if (loom_vector_component_operand_root(plan, record, i) == root_index) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_component_root_is_selected(
    loom_vector_component_plan_t* plan,
    const loom_vector_component_record_t* record, uint32_t component_root_index,
    uint32_t root_index) {
  if (root_index == component_root_index) {
    return true;
  }
  for (uint16_t i = 0; i < record->op->operand_count; ++i) {
    if (loom_vector_component_operand_root(plan, record, i) == root_index) {
      return true;
    }
  }
  return false;
}

// Returns true when moving |root_index| to the prospective component's final
// operation would place a replacement after an external use. Demand has
// already propagated through the complete producer closure; direct producer
// roots become internal if the merge succeeds.
static bool loom_vector_component_root_has_early_external_use(
    loom_vector_component_plan_t* plan, uint32_t root_index,
    const loom_vector_component_record_t* record, uint32_t component_root_index,
    uint64_t insertion_ordinal) {
  uint32_t member_index = root_index;
  do {
    const loom_vector_component_record_t* member = &plan->records[member_index];
    const loom_value_id_t result = loom_op_const_results(member->op)[0];
    const loom_value_t* value =
        loom_module_value(plan->context->module, result);
    const loom_use_t* use = NULL;
    loom_value_for_each_use(value, use) {
      const loom_op_t* user_op = loom_use_user_op(*use);
      if (user_op->result_count == 1) {
        const uint32_t user_index = loom_vector_component_record_for_value(
            plan, loom_op_const_results(user_op)[0]);
        if (user_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
          const uint32_t user_root_index =
              loom_vector_component_find_root(plan, user_index);
          if (loom_vector_component_root_is_selected(
                  plan, record, component_root_index, user_root_index)) {
            continue;
          }
        }
      }
      const loom_op_t* anchor =
          loom_vector_component_user_anchor(user_op, record->op->parent_block);
      if (anchor != NULL && anchor->block_ordinal < insertion_ordinal) {
        return true;
      }
    }
    member_index = member->union_next_member_index;
  } while (member_index != root_index);
  return false;
}

static void loom_vector_component_merge_roots(
    loom_vector_component_plan_t* plan, uint32_t target_root_index,
    uint32_t source_root_index) {
  loom_vector_component_record_t* target = &plan->records[target_root_index];
  loom_vector_component_record_t* source = &plan->records[source_root_index];
  const uint32_t target_next_index = target->union_next_member_index;
  target->union_next_member_index = source->union_next_member_index;
  source->union_next_member_index = target_next_index;
  source->parent_index = target_root_index;
}

// Propagates target-rejected roots backward through the complete same-lane
// producer closure. This marks the operations eligible to join a demanded
// component without yet choosing widths or moving IR.
static void loom_vector_component_propagate_demand(
    loom_vector_component_plan_t* plan) {
  for (uint32_t record_index = plan->record_count; record_index > 0;
       --record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index - 1u];
    if (!record->demanded) {
      continue;
    }
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_index =
          loom_vector_component_operand_root(plan, record, i);
      if (producer_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID &&
          !plan->records[producer_index].demanded) {
        plan->records[producer_index].demanded = true;
        ++plan->demanded_record_count;
      }
    }
  }
}

// Resolves packet widths only for the backward closure of a target rejection.
// Legal components outside that closure remain authored and pay no projected
// contract queries.
static iree_status_t loom_vector_component_classify_candidates(
    loom_vector_component_plan_t* plan) {
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!record->demanded || record->blocked) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vector_component_candidate_bits(
        plan, record->op, record->lane_count, &record->candidate_bits));
  }
  return iree_ok_status();
}

// Forms demanded components in dominance order. Earlier fanout consumers have
// already joined a producer root when a later sibling considers it, while an
// unselected legal consumer remains an external-use boundary. Legal upstream
// producers join when doing so preserves every use and target-native width.
static void loom_vector_component_form(loom_vector_component_plan_t* plan) {
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    uint32_t component_root_index =
        loom_vector_component_find_root(plan, record_index);
    loom_vector_component_record_t* component_root =
        &plan->records[component_root_index];
    if (!component_root->demanded || component_root->blocked ||
        component_root->candidate_bits == 0) {
      continue;
    }

    uint64_t candidate_bits = component_root->candidate_bits;
    uint32_t member_count = component_root->union_member_count;
    uint64_t insertion_ordinal = component_root->union_last_block_ordinal;
    bool has_producer_root = false;
    bool can_merge = true;
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_root_index =
          loom_vector_component_operand_root(plan, record, i);
      if (producer_root_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
          producer_root_index == component_root_index ||
          !loom_vector_component_root_is_first_operand(plan, record, i,
                                                       producer_root_index)) {
        continue;
      }
      has_producer_root = true;
      const loom_vector_component_record_t* producer_root =
          &plan->records[producer_root_index];
      if (producer_root->blocked ||
          producer_root->union_member_count >
              LOOM_VECTOR_COMPONENT_MEMBER_LIMIT - member_count) {
        can_merge = false;
        break;
      }
      candidate_bits &= producer_root->candidate_bits;
      member_count += producer_root->union_member_count;
      insertion_ordinal =
          iree_max(insertion_ordinal, producer_root->union_last_block_ordinal);
    }
    if (!has_producer_root || !can_merge || candidate_bits == 0) {
      continue;
    }

    if (loom_vector_component_root_has_early_external_use(
            plan, component_root_index, record, component_root_index,
            insertion_ordinal)) {
      continue;
    }
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_root_index =
          loom_vector_component_operand_root(plan, record, i);
      if (producer_root_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
          producer_root_index == component_root_index ||
          !loom_vector_component_root_is_first_operand(plan, record, i,
                                                       producer_root_index)) {
        continue;
      }
      if (loom_vector_component_root_has_early_external_use(
              plan, producer_root_index, record, component_root_index,
              insertion_ordinal)) {
        can_merge = false;
        break;
      }
    }
    if (!can_merge) {
      continue;
    }

    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_root_index =
          loom_vector_component_operand_root(plan, record, i);
      if (producer_root_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
          producer_root_index == component_root_index) {
        continue;
      }
      loom_vector_component_merge_roots(plan, component_root_index,
                                        producer_root_index);
    }
    component_root_index =
        loom_vector_component_find_root(plan, component_root_index);
    component_root = &plan->records[component_root_index];
    component_root->candidate_bits = candidate_bits;
    component_root->union_member_count = member_count;
    component_root->union_last_block_ordinal = insertion_ordinal;
  }
}

static iree_status_t loom_vector_component_compact(
    loom_vector_component_plan_t* plan) {
  if (plan->demanded_record_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->context->arena, plan->demanded_record_count,
      sizeof(*plan->components), (void**)&plan->components));

  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    if (!plan->records[record_index].demanded) {
      continue;
    }
    const uint32_t root_index =
        loom_vector_component_find_root(plan, record_index);
    uint32_t component_index = plan->records[root_index].component_index;
    if (component_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
      component_index = plan->component_count++;
      plan->records[root_index].component_index = component_index;
      plan->components[component_index] = (loom_vector_component_t){
          .first_member_index = record_index,
          .last_member_index = record_index,
          .member_count = 1,
          .lane_count = plan->records[record_index].lane_count,
          .candidate_bits = plan->records[root_index].candidate_bits,
          .blocked = plan->records[root_index].blocked,
      };
    } else {
      loom_vector_component_t* component = &plan->components[component_index];
      loom_vector_component_record_t* previous =
          &plan->records[component->last_member_index];
      previous->next_member_index = record_index;
      plan->records[record_index].previous_member_index =
          component->last_member_index;
      component->last_member_index = record_index;
      ++component->member_count;
    }
    plan->records[record_index].component_index = component_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_component_select(
    loom_vector_component_plan_t* plan, loom_vector_component_t* component) {
  if (component->blocked ||
      plan->context->policy == LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    return iree_ok_status();
  }
  bool needs_legalization = false;
  uint32_t member_index = component->first_member_index;
  while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    const loom_vector_component_record_t* record = &plan->records[member_index];
    needs_legalization |= record->authored_query_result.outcome !=
                          LOOM_TARGET_CONTRACT_QUERY_LEGAL;
    member_index = record->next_member_index;
  }
  if (!needs_legalization) {
    return iree_ok_status();
  }

  const loom_target_vector_packet_policy_t* policy =
      plan->context->vector_packet_policy;
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    if ((component->candidate_bits & ((uint64_t)1u << i)) == 0) {
      continue;
    }
    const uint32_t packet_lane_count = policy->native_lane_counts[i];
    const uint32_t packet_count =
        (component->lane_count - 1u) / packet_lane_count + 1u;
    iree_host_size_t expanded_op_count = 0;
    if (!iree_host_size_checked_mul(component->member_count, packet_count,
                                    &expanded_op_count) ||
        expanded_op_count > LOOM_VECTOR_COMPONENT_PACKET_OP_LIMIT) {
      continue;
    }
    if (packet_lane_count <= component->packet_lane_count) {
      continue;
    }
    component->packet_lane_count = packet_lane_count;
    component->packet_count = packet_count;
  }
  return iree_ok_status();
}

static bool loom_vector_component_record_is_internal_operand(
    const loom_vector_component_plan_t* plan,
    const loom_vector_component_t* component, loom_value_id_t operand,
    uint32_t* out_record_index) {
  *out_record_index = loom_vector_component_record_for_value(plan, operand);
  return *out_record_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID &&
         plan->records[*out_record_index].component_index ==
             plan->records[component->first_member_index].component_index;
}

static iree_status_t loom_vector_component_external_packet(
    loom_vector_component_plan_t* plan,
    loom_vector_component_external_t* values, iree_host_size_t* value_count,
    iree_host_size_t value_capacity, loom_value_id_t source,
    uint32_t source_lane_count, uint32_t lane_offset, uint32_t lane_count,
    loom_location_id_t location, loom_value_id_t* out_packet) {
  for (iree_host_size_t i = 0; i < *value_count; ++i) {
    if (values[i].source == source) {
      if (values[i].packet == LOOM_VALUE_ID_INVALID) {
        break;
      }
      *out_packet = values[i].packet;
      return iree_ok_status();
    }
  }

  iree_host_size_t value_index = 0;
  for (; value_index < *value_count; ++value_index) {
    if (values[value_index].source == source) {
      break;
    }
  }
  if (value_index == *value_count) {
    IREE_ASSERT_LT(*value_count, value_capacity);
    ++*value_count;
    values[value_index] = (loom_vector_component_external_t){
        .source = source,
        .source_type = loom_module_value_type(plan->context->module, source),
        .linear_source = source,
        .packet = LOOM_VALUE_ID_INVALID,
    };
    if (loom_type_rank(values[value_index].source_type) != 1) {
      const loom_type_t linear_type = loom_vector_component_packet_type(
          values[value_index].source_type, source_lane_count);
      loom_op_t* bitcast_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
          &plan->context->rewriter->builder, source,
          values[value_index].source_type, linear_type, location, &bitcast_op));
      values[value_index].linear_source =
          loom_vector_bitcast_result(bitcast_op);
    }
  }
  const int64_t static_offset = lane_offset;
  const loom_type_t packet_type = loom_vector_component_packet_type(
      values[value_index].source_type, lane_count);
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_slice_build(
      &plan->context->rewriter->builder, values[value_index].linear_source,
      /*offsets=*/NULL,
      /*offsets_count=*/0, &static_offset, /*static_offsets_count=*/1,
      packet_type, location, &slice_op));
  values[value_index].packet = loom_vector_slice_result(slice_op);
  *out_packet = values[value_index].packet;
  return iree_ok_status();
}

static bool loom_vector_component_result_escapes(
    const loom_vector_component_plan_t* plan,
    const loom_vector_component_t* component, uint32_t record_index) {
  const loom_vector_component_record_t* record = &plan->records[record_index];
  const loom_value_t* result = loom_module_value(
      plan->context->module, loom_op_const_results(record->op)[0]);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (user_op->result_count != 1) {
      return true;
    }
    const uint32_t user_record_index = loom_vector_component_record_for_value(
        plan, loom_op_const_results(user_op)[0]);
    if (user_record_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
        plan->records[user_record_index].component_index !=
            record->component_index) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_vector_component_materialize(
    loom_vector_component_plan_t* plan, loom_vector_component_t* component,
    uint32_t* out_rewritten_op_count) {
  *out_rewritten_op_count = 0;
  if (component->packet_lane_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t external_capacity = 0;
  iree_host_size_t escape_count = 0;
  uint16_t maximum_operand_count = 0;
  uint32_t member_index = component->first_member_index;
  while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    loom_vector_component_record_t* record = &plan->records[member_index];
    external_capacity += record->op->operand_count;
    maximum_operand_count =
        iree_max(maximum_operand_count, record->op->operand_count);
    record->escapes =
        loom_vector_component_result_escapes(plan, component, member_index);
    escape_count += record->escapes ? 1u : 0u;
    member_index = record->next_member_index;
  }

  loom_vector_component_external_t* external_values = NULL;
  if (external_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->context->arena, external_capacity, sizeof(*external_values),
        (void**)&external_values));
  }
  loom_value_id_t* packet_operands = NULL;
  if (maximum_operand_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->context->arena, maximum_operand_count, sizeof(*packet_operands),
        (void**)&packet_operands));
  }
  iree_host_size_t external_count = 0;
  loom_value_id_t* escape_packets = NULL;
  if (escape_count != 0) {
    // Selection bounds member_count * packet_count to the static expansion
    // limit, and every escaping value is one component member.
    const iree_host_size_t escape_packet_count =
        escape_count * component->packet_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->context->arena, escape_packet_count, sizeof(*escape_packets),
        (void**)&escape_packets));
  }

  loom_rewriter_t* rewriter = plan->context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder,
                          plan->records[component->last_member_index].op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  for (uint32_t packet_index = 0; packet_index < component->packet_count;
       ++packet_index) {
    for (iree_host_size_t i = 0; i < external_count; ++i) {
      external_values[i].packet = LOOM_VALUE_ID_INVALID;
    }
    const uint32_t lane_offset = packet_index * component->packet_lane_count;
    const uint32_t packet_lane_count = iree_min(
        component->lane_count - lane_offset, component->packet_lane_count);
    iree_host_size_t escape_index = 0;
    member_index = component->first_member_index;
    while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
      loom_vector_component_record_t* record = &plan->records[member_index];
      loom_op_t* source_op = record->op;
      const uint64_t created_op_count_before = rewriter->created_op_count;
      const loom_value_id_t* source_operands =
          loom_op_const_operands(source_op);
      for (uint16_t i = 0; i < source_op->operand_count; ++i) {
        const loom_value_id_t source_operand = source_operands[i];
        const loom_type_t operand_type =
            loom_module_value_type(plan->context->module, source_operand);
        uint32_t operand_lane_count = 0;
        if (!loom_vector_component_static_lane_count(operand_type,
                                                     &operand_lane_count) ||
            operand_lane_count != component->lane_count) {
          packet_operands[i] = source_operand;
          continue;
        }
        uint32_t producer_record_index = LOOM_VECTOR_COMPONENT_INDEX_INVALID;
        if (loom_vector_component_record_is_internal_operand(
                plan, component, source_operand, &producer_record_index)) {
          IREE_ASSERT_NE(plan->records[producer_record_index].packet,
                         LOOM_VALUE_ID_INVALID);
          packet_operands[i] = plan->records[producer_record_index].packet;
        } else {
          IREE_RETURN_IF_ERROR(loom_vector_component_external_packet(
              plan, external_values, &external_count, external_capacity,
              source_operand, component->lane_count, lane_offset,
              packet_lane_count, source_op->location, &packet_operands[i]));
        }
      }

      // Allocation links the operation at the insertion point. Resolve and
      // materialize every operand first so each packet definition dominates
      // the cloned consumer in the block order.
      loom_op_t* packet_op = NULL;
      IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
          builder, source_op->kind, source_op->operand_count,
          source_op->result_count, /*region_count=*/0,
          /*tied_result_count=*/0, source_op->attribute_count,
          source_op->location, &packet_op));
      packet_op->instance_flags = source_op->instance_flags;
      if (source_op->operand_count != 0) {
        memcpy(loom_op_operands(packet_op), packet_operands,
               source_op->operand_count * sizeof(*packet_operands));
      }
      if (source_op->attribute_count != 0) {
        memcpy(loom_op_attrs(packet_op), loom_op_const_attrs(source_op),
               source_op->attribute_count * sizeof(loom_attribute_t));
      }
      const loom_type_t source_result_type = loom_module_value_type(
          plan->context->module, loom_op_const_results(source_op)[0]);
      const loom_type_t packet_result_type = loom_vector_component_packet_type(
          source_result_type, packet_lane_count);
      IREE_RETURN_IF_ERROR(loom_builder_define_value(
          builder, packet_result_type, &record->packet));
      loom_op_results(packet_op)[0] = record->packet;
      IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, packet_op));
      if (record->escapes) {
        escape_packets[escape_index * component->packet_count + packet_index] =
            record->packet;
        ++escape_index;
      }
      record->created_op_count +=
          rewriter->created_op_count - created_op_count_before;
      member_index = record->next_member_index;
    }
  }

  iree_host_size_t escape_index = 0;
  member_index = component->first_member_index;
  while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    loom_vector_component_record_t* record = &plan->records[member_index];
    if (record->escapes) {
      const uint64_t created_op_count_before = rewriter->created_op_count;
      loom_op_t* concat_op = NULL;
      const loom_value_id_t source_result =
          loom_op_const_results(record->op)[0];
      const loom_type_t source_result_type =
          loom_module_value_type(plan->context->module, source_result);
      const loom_type_t linear_result_type = loom_vector_component_packet_type(
          source_result_type, component->lane_count);
      IREE_RETURN_IF_ERROR(loom_vector_concat_build(
          builder, /*axis=*/0,
          &escape_packets[escape_index * component->packet_count],
          component->packet_count, linear_result_type, record->op->location,
          &concat_op));
      loom_value_id_t replacement = loom_vector_concat_result(concat_op);
      if (!loom_type_equal(linear_result_type, source_result_type)) {
        loom_op_t* bitcast_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
            builder, replacement, linear_result_type, source_result_type,
            record->op->location, &bitcast_op));
        replacement = loom_vector_bitcast_result(bitcast_op);
      }
      IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
          rewriter, record->op, &replacement, 1, value_checkpoint));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
          rewriter, source_result, replacement));
      record->created_op_count +=
          rewriter->created_op_count - created_op_count_before;
      ++escape_index;
    }
    member_index = record->next_member_index;
  }

  member_index = component->last_member_index;
  while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    loom_vector_component_record_t* record = &plan->records[member_index];
    const uint32_t previous_member_index = record->previous_member_index;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, record->op));
    member_index = previous_member_index;
  }
  if (plan->rewrite_callback.fn != NULL) {
    member_index = component->first_member_index;
    while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
      const loom_vector_component_record_t* record =
          &plan->records[member_index];
      IREE_RETURN_IF_ERROR(plan->rewrite_callback.fn(
          plan->rewrite_callback.user_data, record->op,
          record->created_op_count, /*erased_op_count=*/1));
      member_index = record->next_member_index;
    }
  }
  *out_rewritten_op_count = component->member_count;
  return iree_ok_status();
}

const loom_target_contract_query_result_t*
loom_vector_component_packet_query_cache_lookup(
    const loom_vector_component_packet_query_cache_t* cache,
    const loom_op_t* op) {
  const loom_vector_component_plan_t* plan = cache->plan;
  if (plan == NULL || op->result_count != 1) {
    return NULL;
  }
  const uint32_t record_index = loom_vector_component_record_for_value(
      plan, loom_op_const_results(op)[0]);
  if (record_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    return NULL;
  }
  const loom_vector_component_record_t* record = &plan->records[record_index];
  IREE_ASSERT_EQ(record->op, op);
  return &record->authored_query_result;
}

iree_status_t loom_vector_component_packet_legalize(
    loom_target_legalization_context_t* context, loom_region_t* region,
    loom_vector_component_packet_rewrite_callback_t rewrite_callback,
    loom_vector_component_packet_query_cache_t* out_query_cache,
    uint32_t* out_rewritten_op_count) {
  *out_query_cache = loom_vector_component_packet_query_cache_empty();
  *out_rewritten_op_count = 0;
  const loom_target_vector_packet_policy_t* policy =
      context->vector_packet_policy;
  if (policy->native_lane_count_count == 0) {
    return iree_ok_status();
  }

  const loom_value_ordinal_t value_count = context->value_domain->value_count;
  const loom_value_ordinal_t definition_count =
      context->value_domain->definition_count;
  if (definition_count == 0) {
    return iree_ok_status();
  }
  loom_vector_component_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(context->arena, sizeof(*plan), (void**)&plan));
  *plan = (loom_vector_component_plan_t){
      .context = context,
      .rewrite_callback = rewrite_callback,
      .record_capacity = definition_count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, value_count, sizeof(*plan->value_record_indices),
      (void**)&plan->value_record_indices));
  memset(plan->value_record_indices, 0,
         value_count * sizeof(*plan->value_record_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, definition_count, sizeof(*plan->records),
      (void**)&plan->records));

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(
      loom_walk_region(context->module, region, LOOM_WALK_PRE_ORDER,
                       (loom_walk_callback_t){
                           .fn = loom_vector_component_classify_op,
                           .user_data = plan,
                       },
                       &walk_result));
  out_query_cache->plan = plan;
  if (plan->demanded_record_count == 0) {
    return iree_ok_status();
  }
  loom_vector_component_propagate_demand(plan);
  IREE_RETURN_IF_ERROR(loom_vector_component_classify_candidates(plan));
  loom_vector_component_form(plan);
  IREE_RETURN_IF_ERROR(loom_vector_component_compact(plan));

  for (uint32_t component_index = 0; component_index < plan->component_count;
       ++component_index) {
    IREE_RETURN_IF_ERROR(
        loom_vector_component_select(plan, &plan->components[component_index]));
  }
  for (uint32_t component_index = 0; component_index < plan->component_count;
       ++component_index) {
    uint32_t rewritten_op_count = 0;
    IREE_RETURN_IF_ERROR(loom_vector_component_materialize(
        plan, &plan->components[component_index], &rewritten_op_count));
    *out_rewritten_op_count += rewritten_op_count;
  }
  return iree_ok_status();
}
