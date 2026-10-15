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
#include "loom/target/contract.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/packet_plan.h"

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
  // Candidate widths whose contract requires packet-local producer provenance.
  uint64_t producer_closure_candidate_bits;
  // Operations created while materializing packets and an escaping aggregate.
  uint64_t created_op_count;
  // Authored target-contract result retained for downstream legalization.
  loom_target_contract_query_result_t authored_query_result;
  // True when non-operand references prevent ordinary SSA replacement.
  bool blocked;
  // True when this root contains an operation requiring packet legalization.
  bool demanded;
  // True when every downstream use remains in the component graph or reaches
  // a terminal store in the same block.
  bool packet_terminal_owned;
  // True when the result escapes the selected component.
  bool escapes;
} loom_vector_component_record_t;

typedef struct loom_vector_component_producer_closure_t {
  // Union-find parent record for the packet-local producer closure.
  uint32_t parent_index;
  // Offset of retained cross-component producer indices, or invalid.
  uint32_t retained_producer_offset;
  // Packet results retained for consumers in later closure components.
  loom_value_id_t* retained_packets;
  // Logical lane width of each retained packet.
  uint32_t retained_packet_lane_count;
  // Number of entries in |retained_packets|.
  uint32_t retained_packet_count;
  // True when the operation belongs to this packet-local producer closure.
  bool demanded;
  // True when whole-vector provenance requires the complete producer chain.
  bool provenance_demanded;
  // True when one closure component selected a different packet shape.
  bool selection_mismatch;
} loom_vector_component_producer_closure_t;

typedef enum loom_vector_component_candidate_scope_e {
  LOOM_VECTOR_COMPONENT_CANDIDATE_SCOPE_REJECTED_ROOTS = 0,
  LOOM_VECTOR_COMPONENT_CANDIDATE_SCOPE_LEGAL_PRODUCERS = 1,
} loom_vector_component_candidate_scope_t;

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

typedef struct loom_vector_component_retained_load_t {
  // Native lane candidates admitted by the memory root.
  uint64_t candidate_bits;
  // First component record consuming the root, one-based.
  uint32_t first_consumer_record_index;
} loom_vector_component_retained_load_t;

struct loom_vector_component_plan_t {
  // Active target legalization context.
  loom_target_legalization_context_t* context;
  // Optional observer for completed authored-operation rewrites.
  loom_vector_component_packet_rewrite_callback_t rewrite_callback;
  // Dense one-based record index keyed by local value ordinal.
  uint32_t* value_record_indices;
  // Retained memory roots keyed by local value ordinal.
  loom_vector_component_retained_load_t* retained_loads;
  // Original producer records retained across component rewrites.
  uint32_t* retained_producer_record_indices;
  // Optional state for packet-local producer closures.
  loom_vector_component_producer_closure_t* producer_closures;
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
  // Number of authored operations requiring packet legalization.
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

iree_status_t loom_vector_component_packet_op_has_vector_candidate(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_contract_query_result_t* authored_query_result,
    bool* out_has_candidate) {
  *out_has_candidate = false;
  uint32_t lane_count = 0;
  if (!loom_vector_component_describe(context->module, op, &lane_count)) {
    return iree_ok_status();
  }
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_value_t* result_value = loom_module_value(context->module, result);
  if (loom_value_has_attribute_uses(result_value) ||
      loom_module_value_has_type_uses(context->module, result) ||
      authored_query_result->outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR ||
      (authored_query_result->outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL &&
       loom_target_legalization_op_has_source_vector_carriers(context, op))) {
    return iree_ok_status();
  }
  uint64_t candidate_bits = 0;
  uint64_t producer_closure_candidate_bits = 0;
  IREE_RETURN_IF_ERROR(loom_vector_packet_component_candidate_bits(
      context, op, lane_count, &candidate_bits,
      &producer_closure_candidate_bits));
  for (uint8_t i = 0;
       i < context->vector_packet_policy->native_lane_count_count; ++i) {
    if (context->vector_packet_policy->native_lane_counts[i] > 1u &&
        iree_any_bit_set(candidate_bits, (uint64_t)1u << i)) {
      *out_has_candidate = true;
      break;
    }
  }
  return iree_ok_status();
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
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      ordinal >= plan->context->value_domain->definition_count) {
    return LOOM_VECTOR_COMPONENT_INDEX_INVALID;
  }
  const uint32_t one_based_index = plan->value_record_indices[ordinal];
  return one_based_index == 0 ? LOOM_VECTOR_COMPONENT_INDEX_INVALID
                              : one_based_index - 1u;
}

// Returns the direct same-block, same-lane decomposable producer while the
// authored operand is still present. Component materialization may replace an
// earlier producer use, so cross-component provenance edges are retained
// separately before rewriting begins.
static uint32_t loom_vector_component_direct_producer_record(
    const loom_vector_component_plan_t* plan,
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
  return producer->op->parent_block == record->op->parent_block &&
                 producer->lane_count == record->lane_count
             ? producer_index
             : LOOM_VECTOR_COMPONENT_INDEX_INVALID;
}

static iree_status_t loom_vector_component_classify_op(
    loom_vector_component_plan_t* plan, loom_op_t* op) {
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
                         LOOM_TARGET_CONTRACT_QUERY_INVALID_IR &&
                     (record->authored_query_result.outcome !=
                          LOOM_TARGET_CONTRACT_QUERY_LEGAL ||
                      !loom_target_legalization_op_has_source_vector_carriers(
                          plan->context, op));
  if (record->demanded) {
    ++plan->demanded_record_count;
  }

  const loom_value_ordinal_t result_ordinal =
      loom_local_value_domain_ordinal(plan->context->value_domain, result);
  IREE_ASSERT_EQ(plan->value_record_indices[result_ordinal], 0u);
  plan->value_record_indices[result_ordinal] = record_index + 1u;
  return iree_ok_status();
}

// Classifies the straight-line suffixes whose values can remain packetized all
// the way to memory. Reverse dominance order makes each direct consumer's
// classification available before its producer. Control-flow, callable, and
// other aggregate observations terminate the suffix so their existing carrier
// projection remains authoritative.
static void loom_vector_component_classify_packet_terminal_ownership(
    loom_vector_component_plan_t* plan) {
  for (uint32_t record_index = plan->record_count; record_index > 0;
       --record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index - 1u];
    const loom_value_id_t result = loom_op_const_results(record->op)[0];
    const loom_value_t* result_value =
        loom_module_value(plan->context->module, result);
    bool packet_terminal_owned =
        !record->blocked && result_value->use_count != 0;
    const loom_use_t* use = NULL;
    loom_value_for_each_use(result_value, use) {
      const loom_op_t* user_op = loom_use_user_op(*use);
      if (user_op->parent_block != record->op->parent_block) {
        packet_terminal_owned = false;
        break;
      }
      if (loom_vector_store_isa(user_op) &&
          loom_vector_store_value(user_op) == result) {
        continue;
      }
      if (user_op->result_count != 1) {
        packet_terminal_owned = false;
        break;
      }
      const uint32_t consumer_index = loom_vector_component_record_for_value(
          plan, loom_op_const_results(user_op)[0]);
      if (consumer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
          plan->records[consumer_index].lane_count != record->lane_count ||
          !plan->records[consumer_index].packet_terminal_owned) {
        packet_terminal_owned = false;
        break;
      }
    }
    record->packet_terminal_owned = packet_terminal_owned;
  }
}

static uint32_t loom_vector_component_operand_root(
    loom_vector_component_plan_t* plan,
    const loom_vector_component_record_t* record, uint16_t operand_index) {
  const uint32_t producer_index =
      loom_vector_component_direct_producer_record(plan, record, operand_index);
  if (producer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
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
// already identified every required producer; direct demanded roots become
// internal if the merge succeeds.
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

// Resolves projected packet contracts for one phase of demanded operations.
static iree_status_t loom_vector_component_classify_candidates(
    loom_vector_component_plan_t* plan,
    loom_vector_component_candidate_scope_t scope) {
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    const bool authored_legal = record->authored_query_result.outcome ==
                                LOOM_TARGET_CONTRACT_QUERY_LEGAL;
    const bool scope_is_legal =
        scope == LOOM_VECTOR_COMPONENT_CANDIDATE_SCOPE_LEGAL_PRODUCERS;
    if (!record->demanded || record->blocked ||
        authored_legal != scope_is_legal) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vector_packet_component_candidate_bits(
        plan->context, record->op, record->lane_count, &record->candidate_bits,
        &record->producer_closure_candidate_bits));
  }
  return iree_ok_status();
}

static bool loom_vector_component_producer_closure_demanded(
    const loom_vector_component_plan_t* plan, uint32_t record_index) {
  return plan->producer_closures != NULL &&
         plan->producer_closures[record_index].demanded;
}

static bool loom_vector_component_result_requires_packet_retention(
    const loom_vector_component_plan_t* plan, uint32_t record_index) {
  const loom_op_t* op = plan->records[record_index].op;
  const loom_value_id_t result = loom_op_const_results(op)[0];
  return !loom_target_legalization_has_source_vector_carrier(
      plan->context, loom_module_value_type(plan->context->module, result),
      result);
}

static bool loom_vector_component_has_retained_packet_edge(
    loom_vector_component_plan_t* plan, uint32_t consumer_index,
    uint32_t producer_index) {
  return producer_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID &&
         !plan->records[consumer_index].blocked &&
         !plan->records[producer_index].blocked &&
         loom_vector_component_result_requires_packet_retention(plan,
                                                                producer_index);
}

static bool loom_vector_component_load_is_component_owned(
    const loom_vector_component_plan_t* plan, const loom_op_t* load_op,
    loom_value_id_t result) {
  const loom_value_t* result_value =
      loom_module_value(plan->context->module, result);
  if (loom_value_has_attribute_uses(result_value) ||
      loom_module_value_has_type_uses(plan->context->module, result)) {
    return false;
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result_value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (user_op->result_count != 1) {
      return false;
    }
    const uint32_t user_record_index = loom_vector_component_record_for_value(
        plan, loom_op_const_results(user_op)[0]);
    if (user_record_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
        user_op->parent_block != load_op->parent_block ||
        !plan->records[user_record_index].packet_terminal_owned) {
      return false;
    }
  }
  return true;
}

static loom_op_t* loom_vector_component_retained_load_for_operand(
    const loom_vector_component_plan_t* plan,
    const loom_vector_component_record_t* consumer, uint16_t operand_index,
    loom_value_ordinal_t* out_ordinal) {
  *out_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!iree_any_bit_set(
          plan->context->vector_packet_policy->flags,
          LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_MEMORY_ROOT_COMPONENTS)) {
    return NULL;
  }
  const loom_value_id_t operand =
      loom_op_const_operands(consumer->op)[operand_index];
  const loom_type_t operand_type =
      loom_module_value_type(plan->context->module, operand);
  uint32_t operand_lane_count = 0;
  if (!loom_vector_component_static_lane_count(operand_type,
                                               &operand_lane_count) ||
      operand_lane_count != consumer->lane_count) {
    return NULL;
  }
  const loom_value_t* operand_value =
      loom_module_value(plan->context->module, operand);
  if (loom_value_is_block_arg(operand_value)) {
    return NULL;
  }
  loom_op_t* operand_op = loom_value_def_op(operand_value);
  if (operand_op == NULL || !loom_vector_load_isa(operand_op)) {
    return NULL;
  }
  if (!loom_vector_component_load_is_component_owned(plan, operand_op,
                                                     operand)) {
    return NULL;
  }
  *out_ordinal =
      loom_local_value_domain_try_ordinal(plan->context->value_domain, operand);
  return *out_ordinal == LOOM_VALUE_ORDINAL_INVALID ? NULL : operand_op;
}

// Marks rejected roots that have no independent packet contract. Exact-lane
// provenance contracts must rebuild their producer chain at packet width;
// slicing an authored aggregate would retain an offset into the wide origin
// instead of establishing a complete packet-local origin.
static iree_status_t loom_vector_component_mark_producer_closures(
    loom_vector_component_plan_t* plan) {
  bool has_producer_closure = false;
  bool has_retained_load = false;
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    const loom_vector_component_record_t* record = &plan->records[record_index];
    if (!record->demanded || record->blocked) {
      continue;
    }
    const uint64_t independent_candidate_bits =
        record->candidate_bits & ~record->producer_closure_candidate_bits;
    if (independent_candidate_bits == 0 &&
        record->producer_closure_candidate_bits != 0) {
      has_producer_closure = true;
    }
    has_producer_closure |=
        loom_vector_component_result_requires_packet_retention(plan,
                                                               record_index);
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_index =
          loom_vector_component_direct_producer_record(plan, record, i);
      has_producer_closure |= loom_vector_component_has_retained_packet_edge(
          plan, record_index, producer_index);
      loom_value_ordinal_t load_ordinal = LOOM_VALUE_ORDINAL_INVALID;
      has_retained_load |= loom_vector_component_retained_load_for_operand(
                               plan, record, i, &load_ordinal) != NULL;
    }
  }
  if (!has_producer_closure && !has_retained_load) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->context->arena, plan->record_count,
      sizeof(*plan->producer_closures), (void**)&plan->producer_closures));
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    const loom_vector_component_record_t* record = &plan->records[record_index];
    const uint64_t independent_candidate_bits =
        record->candidate_bits & ~record->producer_closure_candidate_bits;
    const bool provenance_demanded =
        record->demanded && !record->blocked &&
        independent_candidate_bits == 0 &&
        record->producer_closure_candidate_bits != 0;
    plan->producer_closures[record_index] =
        (loom_vector_component_producer_closure_t){
            .parent_index = record_index,
            .retained_producer_offset = LOOM_VECTOR_COMPONENT_INDEX_INVALID,
            .demanded = provenance_demanded,
            .provenance_demanded = provenance_demanded,
        };
  }
  return iree_ok_status();
}

// Carries a packet-local provenance requirement through the complete
// same-lane producer closure. Each producer is projected into the component so
// its packet result establishes a fresh whole-packet origin for the consumer.
static void loom_vector_component_propagate_producer_closures(
    loom_vector_component_plan_t* plan) {
  for (uint32_t record_index = plan->record_count; record_index > 0;
       --record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index - 1u];
    if (!loom_vector_component_producer_closure_demanded(plan,
                                                         record_index - 1u)) {
      continue;
    }
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_index =
          loom_vector_component_operand_root(plan, record, i);
      if (producer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
        continue;
      }
      loom_vector_component_record_t* producer = &plan->records[producer_index];
      if (!producer->demanded) {
        producer->demanded = true;
        ++plan->demanded_record_count;
      }
      plan->producer_closures[producer_index].demanded = true;
      plan->producer_closures[producer_index].provenance_demanded = true;
    }
  }
}

static iree_status_t loom_vector_component_prepare_retained_load_tables(
    loom_vector_component_plan_t* plan) {
  if (plan->retained_loads != NULL) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t definition_count =
      plan->context->value_domain->definition_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->context->arena, definition_count, sizeof(*plan->retained_loads),
      (void**)&plan->retained_loads));
  memset(plan->retained_loads, 0,
         definition_count * sizeof(*plan->retained_loads));
  return iree_ok_status();
}

static void loom_vector_component_enqueue_retained_record(
    loom_vector_component_plan_t* plan, uint32_t record_index,
    uint32_t* worklist, uint32_t* worklist_count) {
  plan->producer_closures[record_index].demanded = true;
  loom_vector_component_record_t* record = &plan->records[record_index];
  if (record->demanded) {
    return;
  }
  record->demanded = true;
  ++plan->demanded_record_count;
  worklist[(*worklist_count)++] = record_index;
}

// Expands from rejected roots through every edge that cannot preserve an
// authored aggregate. A retained result reaches both its direct producers and
// consumers; a retained load reaches all decomposable consumers that share its
// packet tuple. The worklist keeps disconnected legal graphs untouched and
// visits each newly demanded operation once.
static iree_status_t loom_vector_component_expand_retained_packet_closures(
    loom_vector_component_plan_t* plan) {
  if (plan->producer_closures == NULL) {
    return iree_ok_status();
  }

  uint32_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->context->arena, plan->record_count,
                                sizeof(*worklist), (void**)&worklist));
  uint32_t worklist_count = 0;
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    if (plan->records[record_index].demanded) {
      worklist[worklist_count++] = record_index;
    }
  }

  uint32_t worklist_index = 0;
  while (worklist_index < worklist_count) {
    const uint32_t record_index = worklist[worklist_index++];
    loom_vector_component_record_t* record = &plan->records[record_index];
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_index =
          loom_vector_component_direct_producer_record(plan, record, i);
      if (loom_vector_component_has_retained_packet_edge(plan, record_index,
                                                         producer_index)) {
        plan->producer_closures[record_index].demanded = true;
        loom_vector_component_enqueue_retained_record(
            plan, producer_index, worklist, &worklist_count);
      }

      loom_value_ordinal_t load_ordinal = LOOM_VALUE_ORDINAL_INVALID;
      loom_op_t* load_op = loom_vector_component_retained_load_for_operand(
          plan, record, i, &load_ordinal);
      if (load_op == NULL) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_vector_component_prepare_retained_load_tables(plan));
      loom_vector_component_retained_load_t* retained_load =
          &plan->retained_loads[load_ordinal];
      if (retained_load->first_consumer_record_index == 0) {
        retained_load->first_consumer_record_index = record_index + 1u;
        const loom_type_t load_type = loom_module_value_type(
            plan->context->module, loom_op_const_results(load_op)[0]);
        IREE_RETURN_IF_ERROR(loom_vector_packet_load_candidate_bits(
            plan->context, load_op, plan->context->vector_packet_policy,
            load_type, &retained_load->candidate_bits));
      }
      plan->producer_closures[record_index].demanded = true;

      const loom_value_id_t load_result = loom_op_const_results(load_op)[0];
      const loom_value_t* load_value =
          loom_module_value(plan->context->module, load_result);
      const loom_use_t* use = NULL;
      loom_value_for_each_use(load_value, use) {
        const loom_op_t* user_op = loom_use_user_op(*use);
        if (user_op->result_count != 1) {
          continue;
        }
        const uint32_t consumer_index = loom_vector_component_record_for_value(
            plan, loom_op_const_results(user_op)[0]);
        if (consumer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
            plan->records[consumer_index].blocked) {
          continue;
        }
        loom_vector_component_enqueue_retained_record(
            plan, consumer_index, worklist, &worklist_count);
      }
    }

    if (!loom_vector_component_result_requires_packet_retention(plan,
                                                                record_index)) {
      continue;
    }
    const loom_value_id_t result = loom_op_const_results(record->op)[0];
    const loom_value_t* result_value =
        loom_module_value(plan->context->module, result);
    const loom_use_t* use = NULL;
    loom_value_for_each_use(result_value, use) {
      const loom_op_t* user_op = loom_use_user_op(*use);
      if (user_op->result_count != 1) {
        continue;
      }
      const uint32_t consumer_index = loom_vector_component_record_for_value(
          plan, loom_op_const_results(user_op)[0]);
      if (consumer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
          !loom_vector_component_has_retained_packet_edge(plan, consumer_index,
                                                          record_index)) {
        continue;
      }
      plan->producer_closures[record_index].demanded = true;
      loom_vector_component_enqueue_retained_record(plan, consumer_index,
                                                    worklist, &worklist_count);
    }
  }
  return iree_ok_status();
}

// Prefers contracts that consume slices of authored target-native producers.
// Producer replay remains available only when every legal candidate requires
// packet-local provenance.
static void loom_vector_component_select_independent_candidates(
    loom_vector_component_plan_t* plan) {
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!record->demanded ||
        loom_vector_component_producer_closure_demanded(plan, record_index)) {
      continue;
    }
    const uint64_t independent_candidate_bits =
        record->candidate_bits & ~record->producer_closure_candidate_bits;
    if (independent_candidate_bits != 0) {
      record->candidate_bits = independent_candidate_bits;
    }
  }
}

static uint32_t loom_vector_component_find_producer_closure_root(
    loom_vector_component_plan_t* plan, uint32_t record_index) {
  uint32_t root_index = record_index;
  while (plan->producer_closures[root_index].parent_index != root_index) {
    root_index = plan->producer_closures[root_index].parent_index;
  }
  while (record_index != root_index) {
    const uint32_t parent_index =
        plan->producer_closures[record_index].parent_index;
    plan->producer_closures[record_index].parent_index = root_index;
    record_index = parent_index;
  }
  return root_index;
}

// Intersects candidate widths across each connected packet-local producer
// closure. The separate union-find preserves the component partition so early
// aggregate uses can still keep producers and consumers in distinct rewrite
// components.
static void loom_vector_component_constrain_producer_closures(
    loom_vector_component_plan_t* plan) {
  if (plan->producer_closures == NULL) {
    return;
  }
  if (plan->retained_loads != NULL) {
    for (uint32_t record_index = 0; record_index < plan->record_count;
         ++record_index) {
      loom_vector_component_record_t* record = &plan->records[record_index];
      if (!loom_vector_component_producer_closure_demanded(plan,
                                                           record_index)) {
        continue;
      }
      for (uint16_t i = 0; i < record->op->operand_count; ++i) {
        loom_value_ordinal_t load_ordinal = LOOM_VALUE_ORDINAL_INVALID;
        if (loom_vector_component_retained_load_for_operand(
                plan, record, i, &load_ordinal) == NULL ||
            plan->retained_loads[load_ordinal].first_consumer_record_index ==
                0) {
          continue;
        }
        const loom_vector_component_retained_load_t retained_load =
            plan->retained_loads[load_ordinal];
        record->candidate_bits &= retained_load.candidate_bits;
        const uint32_t first_consumer_index =
            retained_load.first_consumer_record_index - 1u;
        const uint32_t consumer_root_index =
            loom_vector_component_find_producer_closure_root(plan,
                                                             record_index);
        const uint32_t first_consumer_root_index =
            loom_vector_component_find_producer_closure_root(
                plan, first_consumer_index);
        plan->producer_closures[first_consumer_root_index].parent_index =
            consumer_root_index;
      }
    }
  }
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!loom_vector_component_producer_closure_demanded(plan, record_index)) {
      continue;
    }
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const uint32_t producer_index =
          loom_vector_component_operand_root(plan, record, i);
      if (producer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID ||
          !loom_vector_component_producer_closure_demanded(plan,
                                                           producer_index)) {
        continue;
      }
      const loom_vector_component_producer_closure_t* consumer_closure =
          &plan->producer_closures[record_index];
      const loom_vector_component_producer_closure_t* producer_closure =
          &plan->producer_closures[producer_index];
      const bool provenance_edge = consumer_closure->provenance_demanded &&
                                   producer_closure->provenance_demanded;
      if (!provenance_edge && !loom_vector_component_has_retained_packet_edge(
                                  plan, record_index, producer_index)) {
        continue;
      }
      const uint32_t consumer_root_index =
          loom_vector_component_find_producer_closure_root(plan, record_index);
      const uint32_t producer_root_index =
          loom_vector_component_find_producer_closure_root(plan,
                                                           producer_index);
      plan->producer_closures[producer_root_index].parent_index =
          consumer_root_index;
    }
  }

  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!loom_vector_component_producer_closure_demanded(plan, record_index)) {
      continue;
    }
    const uint32_t root_index =
        loom_vector_component_find_producer_closure_root(plan, record_index);
    plan->records[root_index].candidate_bits &= record->candidate_bits;
  }
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!loom_vector_component_producer_closure_demanded(plan, record_index)) {
      continue;
    }
    const uint32_t root_index =
        loom_vector_component_find_producer_closure_root(plan, record_index);
    record->candidate_bits = plan->records[root_index].candidate_bits;
  }
}

// Forms demanded components in dominance order. Earlier fanout consumers have
// already joined a producer root when a later sibling considers it. Undemanded
// legal producers remain external target snapshots consumed through slices.
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
      const loom_vector_component_record_t* producer_root =
          &plan->records[producer_root_index];
      if (!producer_root->demanded) {
        continue;
      }
      has_producer_root = true;
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
          !plan->records[producer_root_index].demanded ||
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
          producer_root_index == component_root_index ||
          !plan->records[producer_root_index].demanded) {
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
    needs_legalization |=
        record->authored_query_result.outcome !=
            LOOM_TARGET_CONTRACT_QUERY_LEGAL ||
        loom_vector_component_producer_closure_demanded(plan, member_index);
    member_index = record->next_member_index;
  }
  if (!needs_legalization) {
    return iree_ok_status();
  }

  const loom_target_vector_packet_policy_t* policy =
      plan->context->vector_packet_policy;
  if (iree_any_bit_set(
          policy->flags,
          LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_DYNAMIC_WINDOW_REPLAY)) {
    // A rejected dynamic window owns the source-width producer graph. Eagerly
    // materializing an escaping aggregate here would force the later window to
    // index an SSA packet tuple at runtime. Native target window contracts keep
    // their ordinary component behavior.
    member_index = component->first_member_index;
    while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
      const loom_vector_component_record_t* record =
          &plan->records[member_index];
      const loom_value_id_t result = loom_op_const_results(record->op)[0];
      const loom_value_t* result_value =
          loom_module_value(plan->context->module, result);
      const loom_use_t* use = NULL;
      loom_value_for_each_use(result_value, use) {
        const loom_op_t* user_op = loom_use_user_op(*use);
        if (!loom_vector_slice_isa(user_op) ||
            loom_vector_slice_source(user_op) != result) {
          continue;
        }
        bool dynamic_window_owns_consumer_chain = false;
        IREE_RETURN_IF_ERROR(
            loom_vector_packet_dynamic_window_owns_consumer_chain(
                plan->context, policy, user_op,
                &dynamic_window_owns_consumer_chain));
        if (dynamic_window_owns_consumer_chain) {
          return iree_ok_status();
        }
      }
      member_index = record->next_member_index;
    }
  }

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

// Closure components select atomically. A component rejected by expansion or
// source constraints disables every producer and consumer that would otherwise
// depend on its packet-local provenance.
static void loom_vector_component_reconcile_producer_closure_selections(
    loom_vector_component_plan_t* plan) {
  if (plan->producer_closures == NULL) {
    return;
  }
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!loom_vector_component_producer_closure_demanded(plan, record_index)) {
      continue;
    }
    const uint32_t root_index =
        loom_vector_component_find_producer_closure_root(plan, record_index);
    const loom_vector_component_t* root_component =
        &plan->components[plan->records[root_index].component_index];
    const loom_vector_component_t* component =
        &plan->components[record->component_index];
    if (root_component->packet_lane_count != component->packet_lane_count ||
        root_component->packet_count != component->packet_count) {
      plan->producer_closures[root_index].selection_mismatch = true;
    }
  }
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    const loom_vector_component_record_t* record = &plan->records[record_index];
    if (!loom_vector_component_producer_closure_demanded(plan, record_index)) {
      continue;
    }
    const uint32_t root_index =
        loom_vector_component_find_producer_closure_root(plan, record_index);
    if (plan->producer_closures[root_index].selection_mismatch) {
      loom_vector_component_t* component =
          &plan->components[record->component_index];
      component->packet_lane_count = 0;
      component->packet_count = 0;
    }
  }
}

// Captures the original producer edges of selected provenance closures before
// rewriting begins. Ordinary components continue to query their live authored
// operands and allocate no edge storage.
static iree_status_t loom_vector_component_capture_retained_producers(
    loom_vector_component_plan_t* plan) {
  iree_host_size_t retained_operand_count = 0;
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!loom_vector_component_producer_closure_demanded(plan, record_index) ||
        plan->components[record->component_index].packet_lane_count == 0) {
      continue;
    }
    if (!iree_host_size_checked_add(retained_operand_count,
                                    record->op->operand_count,
                                    &retained_operand_count) ||
        retained_operand_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "retained producer table exceeds uint32_t");
    }
  }
  if (retained_operand_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->context->arena, retained_operand_count,
      sizeof(*plan->retained_producer_record_indices),
      (void**)&plan->retained_producer_record_indices));

  uint32_t retained_operand_offset = 0;
  for (uint32_t record_index = 0; record_index < plan->record_count;
       ++record_index) {
    loom_vector_component_record_t* record = &plan->records[record_index];
    if (!loom_vector_component_producer_closure_demanded(plan, record_index) ||
        plan->components[record->component_index].packet_lane_count == 0) {
      continue;
    }
    plan->producer_closures[record_index].retained_producer_offset =
        retained_operand_offset;
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      plan->retained_producer_record_indices[retained_operand_offset++] =
          loom_vector_component_direct_producer_record(plan, record, i);
    }
  }
  IREE_ASSERT_EQ(retained_operand_offset, retained_operand_count);
  return iree_ok_status();
}

static bool loom_vector_component_record_is_internal_operand(
    const loom_vector_component_plan_t* plan,
    const loom_vector_component_t* component, uint32_t producer_record_index) {
  return producer_record_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID &&
         plan->records[producer_record_index].component_index ==
             plan->records[component->first_member_index].component_index;
}

// Returns a packet retained by an earlier producer component in the same
// packet-local provenance closure. Planning has already synchronized the
// selected widths and materialization follows dominance order.
static bool loom_vector_component_retained_producer_packet(
    const loom_vector_component_plan_t* plan,
    const loom_vector_component_t* component, uint32_t consumer_index,
    uint16_t operand_index, uint32_t packet_index,
    loom_value_id_t* out_packet) {
  if (!loom_vector_component_producer_closure_demanded(plan, consumer_index)) {
    return false;
  }
  const loom_vector_component_record_t* consumer =
      &plan->records[consumer_index];
  const loom_vector_component_producer_closure_t* consumer_closure =
      &plan->producer_closures[consumer_index];
  if (consumer_closure->retained_producer_offset ==
      LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    return false;
  }
  const uint32_t producer_index =
      plan->retained_producer_record_indices
          [consumer_closure->retained_producer_offset + operand_index];
  if (producer_index == LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    return false;
  }
  const loom_vector_component_record_t* producer =
      &plan->records[producer_index];
  if (!loom_vector_component_producer_closure_demanded(plan, producer_index) ||
      producer->component_index == consumer->component_index) {
    return false;
  }
  const loom_vector_component_producer_closure_t* producer_closure =
      &plan->producer_closures[producer_index];
  uint32_t consumer_root_index = consumer_index;
  while (plan->producer_closures[consumer_root_index].parent_index !=
         consumer_root_index) {
    consumer_root_index =
        plan->producer_closures[consumer_root_index].parent_index;
  }
  uint32_t producer_root_index = producer_index;
  while (plan->producer_closures[producer_root_index].parent_index !=
         producer_root_index) {
    producer_root_index =
        plan->producer_closures[producer_root_index].parent_index;
  }
  if (consumer_root_index != producer_root_index) {
    return false;
  }
  IREE_ASSERT(producer_closure->retained_packets != NULL);
  IREE_ASSERT_EQ(producer_closure->retained_packet_lane_count,
                 component->packet_lane_count);
  IREE_ASSERT_EQ(producer_closure->retained_packet_count,
                 component->packet_count);
  IREE_ASSERT_LT(packet_index, producer_closure->retained_packet_count);
  *out_packet = producer_closure->retained_packets[packet_index];
  IREE_ASSERT_NE(*out_packet, LOOM_VALUE_ID_INVALID);
  return true;
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
  const loom_value_t* linear_value = loom_module_value(
      plan->context->module, values[value_index].linear_source);
  const loom_op_t* linear_op = loom_value_is_block_arg(linear_value)
                                   ? NULL
                                   : loom_value_def_op(linear_value);
  if (linear_op != NULL && loom_vector_concat_isa(linear_op) &&
      loom_vector_concat_axis(linear_op) == 0) {
    const loom_value_slice_t inputs = loom_vector_concat_inputs(linear_op);
    uint32_t input_lane_offset = 0;
    iree_host_size_t first_input = IREE_HOST_SIZE_MAX;
    iree_host_size_t input_count = 0;
    const uint32_t lane_end = lane_offset + lane_count;
    for (iree_host_size_t i = 0; i < inputs.count; ++i) {
      uint32_t input_lane_count = 0;
      if (!loom_vector_component_static_lane_count(
              loom_module_value_type(plan->context->module, inputs.values[i]),
              &input_lane_count)) {
        first_input = IREE_HOST_SIZE_MAX;
        break;
      }
      if (input_lane_offset == lane_offset) {
        first_input = i;
      }
      input_lane_offset += input_lane_count;
      if (first_input != IREE_HOST_SIZE_MAX) {
        ++input_count;
        if (input_lane_offset == lane_end) {
          if (input_count == 1) {
            values[value_index].packet = inputs.values[first_input];
          } else {
            loom_op_t* concat_op = NULL;
            IREE_RETURN_IF_ERROR(loom_vector_concat_build(
                &plan->context->rewriter->builder, /*axis=*/0,
                &inputs.values[first_input], input_count, packet_type, location,
                &concat_op));
            values[value_index].packet = loom_vector_concat_result(concat_op);
          }
          *out_packet = values[value_index].packet;
          return iree_ok_status();
        }
        if (input_lane_offset > lane_end) {
          break;
        }
      } else if (input_lane_offset > lane_offset) {
        break;
      }
    }
  }
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

static iree_status_t loom_vector_component_query_packetized_store_roots(
    const loom_vector_component_plan_t* plan,
    const loom_vector_component_t* component, bool* out_has_store_root,
    bool* out_supports_packet_lane_count) {
  *out_has_store_root = false;
  *out_supports_packet_lane_count = true;
  uint32_t member_index = component->first_member_index;
  while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    const loom_vector_component_record_t* record = &plan->records[member_index];
    const loom_value_id_t result = loom_op_const_results(record->op)[0];
    const loom_value_t* value =
        loom_module_value(plan->context->module, result);
    const loom_use_t* use = NULL;
    loom_value_for_each_use(value, use) {
      const loom_op_t* user_op = loom_use_user_op(*use);
      if (loom_vector_store_isa(user_op) &&
          loom_vector_store_value(user_op) == result &&
          loom_vector_component_result_requires_packet_retention(
              plan, member_index)) {
        *out_has_store_root = true;
        bool supported = false;
        IREE_RETURN_IF_ERROR(loom_vector_packet_store_supports_lane_count(
            plan->context, user_op, plan->context->vector_packet_policy,
            component->packet_lane_count, &supported));
        if (!supported) {
          *out_supports_packet_lane_count = false;
          return iree_ok_status();
        }
      }
    }
    member_index = record->next_member_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_component_rewrite_packetized_load_root(
    loom_vector_component_plan_t* plan, loom_op_t* source_op,
    uint32_t maximum_lane_count, bool* out_rewritten) {
  loom_rewriter_t* rewriter = plan->context->rewriter;
  const bool observe_operation_creation =
      rewriter->operation_creation.callback.fn != NULL;
  if (observe_operation_creation) {
    loom_rewriter_set_operation_creation_source(rewriter, source_op);
  }
  iree_status_t status =
      loom_vector_packet_legalize_load_with_maximum_lane_count(
          plan->context, source_op, plan->context->vector_packet_policy,
          maximum_lane_count, out_rewritten);
  if (observe_operation_creation) {
    loom_rewriter_set_operation_creation_source(rewriter, NULL);
  }
  return status;
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_vector_component_rewrite_observed_packetized_load_root(
    loom_vector_component_plan_t* plan, loom_op_t* source_op,
    uint32_t maximum_lane_count, bool* out_rewritten) {
  loom_target_contract_query_result_t authored_query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
      plan->context, source_op, &authored_query_result));
  loom_rewriter_t* rewriter = plan->context->rewriter;
  const uint64_t created_op_count_before = rewriter->created_op_count;
  IREE_RETURN_IF_ERROR(loom_vector_component_rewrite_packetized_load_root(
      plan, source_op, maximum_lane_count, out_rewritten));
  if (!*out_rewritten) {
    return iree_ok_status();
  }
  return plan->rewrite_callback.fn(
      plan->rewrite_callback.user_data, source_op, &authored_query_result,
      rewriter->created_op_count - created_op_count_before,
      /*erased_op_count=*/1);
}

static iree_status_t loom_vector_component_prepare_packetized_load_roots(
    loom_vector_component_plan_t* plan,
    const loom_vector_component_t* component, bool* out_ready,
    uint32_t* out_rewritten_op_count) {
  *out_ready = true;
  *out_rewritten_op_count = 0;
  uint32_t member_index = component->first_member_index;
  while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    const loom_vector_component_record_t* record = &plan->records[member_index];
    const loom_value_id_t* operands = loom_op_const_operands(record->op);
    for (uint16_t i = 0; i < record->op->operand_count; ++i) {
      const loom_value_id_t operand = operands[i];
      uint32_t operand_lane_count = 0;
      if (!loom_vector_component_static_lane_count(
              loom_module_value_type(plan->context->module, operand),
              &operand_lane_count) ||
          operand_lane_count != component->lane_count) {
        continue;
      }
      const uint32_t producer_record_index =
          loom_vector_component_direct_producer_record(plan, record, i);
      if (loom_vector_component_record_is_internal_operand(
              plan, component, producer_record_index)) {
        continue;
      }
      const loom_value_t* operand_value =
          loom_module_value(plan->context->module, operand);
      loom_op_t* operand_op = loom_value_is_block_arg(operand_value)
                                  ? NULL
                                  : loom_value_def_op(operand_value);
      if (operand_op == NULL || !loom_vector_load_isa(operand_op)) {
        continue;
      }
      const loom_value_ordinal_t operand_ordinal =
          loom_local_value_domain_try_ordinal(plan->context->value_domain,
                                              operand);
      const bool retained_load =
          operand_ordinal != LOOM_VALUE_ORDINAL_INVALID &&
          plan->retained_loads != NULL &&
          plan->retained_loads[operand_ordinal].first_consumer_record_index !=
              0;
      if (!retained_load &&
          loom_target_legalization_has_source_vector_carrier(
              plan->context,
              loom_module_value_type(plan->context->module, operand),
              operand)) {
        continue;
      }
      // Oversized loads retain their ordinary memory-selected packet width.
      // A load owned by a straight-line terminal closure instead uses the
      // component width so its packets remain directly consumable downstream.
      const uint32_t maximum_lane_count =
          retained_load ? component->packet_lane_count : 0;

      bool rewritten = false;
      if (plan->rewrite_callback.fn == NULL) {
        IREE_RETURN_IF_ERROR(loom_vector_component_rewrite_packetized_load_root(
            plan, operand_op, maximum_lane_count, &rewritten));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_vector_component_rewrite_observed_packetized_load_root(
                plan, operand_op, maximum_lane_count, &rewritten));
      }
      if (!rewritten) {
        *out_ready = false;
        return iree_ok_status();
      }
      ++*out_rewritten_op_count;
    }
    member_index = record->next_member_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_component_materialize(
    loom_vector_component_plan_t* plan, loom_vector_component_t* component,
    uint32_t* out_rewritten_op_count) {
  *out_rewritten_op_count = 0;
  if (component->packet_lane_count == 0) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(
          plan->context->vector_packet_policy->flags,
          LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_MEMORY_ROOT_COMPONENTS)) {
    bool has_store_root = false;
    bool stores_support_packet_lane_count = false;
    IREE_RETURN_IF_ERROR(loom_vector_component_query_packetized_store_roots(
        plan, component, &has_store_root, &stores_support_packet_lane_count));
    if (has_store_root &&
        (component->lane_count % component->packet_lane_count != 0 ||
         !stores_support_packet_lane_count)) {
      // Mixed-width tails and stores that prefer another packet width remain
      // owned by the store planner, which can select a maximal sequence across
      // the complete producer graph.
      return iree_ok_status();
    }
    bool load_roots_ready = false;
    uint32_t rewritten_load_count = 0;
    IREE_RETURN_IF_ERROR(loom_vector_component_prepare_packetized_load_roots(
        plan, component, &load_roots_ready, &rewritten_load_count));
    *out_rewritten_op_count = rewritten_load_count;
    if (!load_roots_ready) {
      return iree_ok_status();
    }
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
  iree_host_size_t retained_escape_index = 0;
  member_index = component->first_member_index;
  while (member_index != LOOM_VECTOR_COMPONENT_INDEX_INVALID) {
    loom_vector_component_record_t* record = &plan->records[member_index];
    if (record->escapes) {
      if (loom_vector_component_producer_closure_demanded(plan, member_index)) {
        loom_vector_component_producer_closure_t* producer_closure =
            &plan->producer_closures[member_index];
        producer_closure->retained_packets =
            &escape_packets[retained_escape_index * component->packet_count];
        producer_closure->retained_packet_lane_count =
            component->packet_lane_count;
        producer_closure->retained_packet_count = component->packet_count;
      }
      ++retained_escape_index;
    }
    member_index = record->next_member_index;
  }
  IREE_ASSERT_EQ(retained_escape_index, escape_count);

  loom_rewriter_t* rewriter = plan->context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  const bool observe_operation_creation =
      rewriter->operation_creation.callback.fn != NULL;
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
      if (observe_operation_creation) {
        loom_rewriter_set_operation_creation_source(rewriter, source_op);
      }
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
        const uint32_t producer_record_index =
            loom_vector_component_direct_producer_record(plan, record, i);
        if (loom_vector_component_record_is_internal_operand(
                plan, component, producer_record_index)) {
          IREE_ASSERT_NE(plan->records[producer_record_index].packet,
                         LOOM_VALUE_ID_INVALID);
          packet_operands[i] = plan->records[producer_record_index].packet;
        } else if (!loom_vector_component_retained_producer_packet(
                       plan, component, member_index, i, packet_index,
                       &packet_operands[i])) {
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
      if (observe_operation_creation) {
        loom_rewriter_set_operation_creation_source(rewriter, record->op);
      }
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
  if (observe_operation_creation) {
    loom_rewriter_set_operation_creation_source(rewriter, NULL);
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
          &record->authored_query_result, record->created_op_count,
          /*erased_op_count=*/1));
      member_index = record->next_member_index;
    }
  }
  *out_rewritten_op_count += component->member_count;
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
    loom_target_legalization_context_t* context,
    loom_vector_component_packet_rewrite_callback_t rewrite_callback,
    loom_vector_component_packet_query_cache_t* out_query_cache,
    uint32_t* out_rewritten_op_count) {
  *out_query_cache = loom_vector_component_packet_query_cache_empty();
  *out_rewritten_op_count = 0;
  if (context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  const loom_target_vector_packet_policy_t* policy =
      context->vector_packet_policy;
  if (policy->native_lane_count_count == 0) {
    return iree_ok_status();
  }

  if (!iree_any_bit_set(context->value_domain->operation_traits,
                        LOOM_TRAIT_DECOMPOSABLE)) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t definition_count =
      context->value_domain->definition_count;
  if (definition_count == 0) {
    return iree_ok_status();
  }

  uint32_t record_capacity = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < definition_count;
       ++ordinal) {
    const loom_value_t* value = loom_module_value(
        context->module, context->value_domain->value_ids[ordinal]);
    if (loom_value_is_block_arg(value)) {
      continue;
    }
    loom_op_t* op = loom_value_def_op(value);
    uint32_t lane_count = 0;
    if (op != NULL &&
        loom_vector_component_describe(context->module, op, &lane_count)) {
      ++record_capacity;
    }
  }
  if (record_capacity == 0) {
    return iree_ok_status();
  }

  loom_vector_component_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(context->arena, sizeof(*plan), (void**)&plan));
  *plan = (loom_vector_component_plan_t){
      .context = context,
      .rewrite_callback = rewrite_callback,
      .record_capacity = record_capacity,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, definition_count, sizeof(*plan->value_record_indices),
      (void**)&plan->value_record_indices));
  memset(plan->value_record_indices, 0,
         definition_count * sizeof(*plan->value_record_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, record_capacity, sizeof(*plan->records),
      (void**)&plan->records));

  for (loom_value_ordinal_t ordinal = 0; ordinal < definition_count;
       ++ordinal) {
    const loom_value_t* value = loom_module_value(
        context->module, context->value_domain->value_ids[ordinal]);
    if (!loom_value_is_block_arg(value)) {
      loom_op_t* op = loom_value_def_op(value);
      if (op != NULL) {
        IREE_RETURN_IF_ERROR(loom_vector_component_classify_op(plan, op));
      }
    }
  }
  IREE_ASSERT_EQ(plan->record_count, record_capacity);
  out_query_cache->plan = plan;
  if (plan->demanded_record_count == 0) {
    return iree_ok_status();
  }
  loom_vector_component_classify_packet_terminal_ownership(plan);
  IREE_RETURN_IF_ERROR(loom_vector_component_classify_candidates(
      plan, LOOM_VECTOR_COMPONENT_CANDIDATE_SCOPE_REJECTED_ROOTS));
  IREE_RETURN_IF_ERROR(loom_vector_component_mark_producer_closures(plan));
  loom_vector_component_propagate_producer_closures(plan);
  IREE_RETURN_IF_ERROR(
      loom_vector_component_expand_retained_packet_closures(plan));
  IREE_RETURN_IF_ERROR(loom_vector_component_classify_candidates(
      plan, LOOM_VECTOR_COMPONENT_CANDIDATE_SCOPE_LEGAL_PRODUCERS));
  loom_vector_component_select_independent_candidates(plan);
  loom_vector_component_constrain_producer_closures(plan);
  loom_vector_component_form(plan);
  IREE_RETURN_IF_ERROR(loom_vector_component_compact(plan));

  for (uint32_t component_index = 0; component_index < plan->component_count;
       ++component_index) {
    IREE_RETURN_IF_ERROR(
        loom_vector_component_select(plan, &plan->components[component_index]));
  }
  loom_vector_component_reconcile_producer_closure_selections(plan);
  IREE_RETURN_IF_ERROR(loom_vector_component_capture_retained_producers(plan));
  for (uint32_t component_index = 0; component_index < plan->component_count;
       ++component_index) {
    uint32_t rewritten_op_count = 0;
    IREE_RETURN_IF_ERROR(loom_vector_component_materialize(
        plan, &plan->components[component_index], &rewritten_op_count));
    *out_rewritten_op_count += rewritten_op_count;
  }
  return iree_ok_status();
}
