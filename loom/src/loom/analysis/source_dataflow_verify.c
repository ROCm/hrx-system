// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_dataflow_verify.h"

static iree_status_t loom_source_dataflow_invalid_provider(
    const loom_source_dataflow_provider_t* provider, const char* detail) {
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "source dataflow provider '%.*s' %s",
      (int)provider->name.size, provider->name.data, detail);
}

iree_status_t loom_source_dataflow_provider_verify(
    const loom_source_dataflow_provider_t* provider) {
  if (provider == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source dataflow provider is NULL");
  }
  if (iree_string_view_is_empty(provider->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source dataflow provider name is empty");
  }
  if (provider->valid_bits == 0) {
    return loom_source_dataflow_invalid_provider(provider,
                                                 "has no valid evidence bits");
  }
  if (provider->phase_count == 0 ||
      provider->phase_count > LOOM_SOURCE_DATAFLOW_MAX_PHASE_COUNT ||
      provider->phase_bits == NULL || provider->reserved != 0) {
    return loom_source_dataflow_invalid_provider(
        provider, "has an invalid monotone phase declaration");
  }
  loom_source_dataflow_bits_t declared_phase_bits = 0;
  for (uint8_t i = 0; i < provider->phase_count; ++i) {
    if (provider->phase_bits[i] == 0 ||
        (provider->phase_bits[i] & declared_phase_bits) != 0) {
      return loom_source_dataflow_invalid_provider(
          provider, "has empty or overlapping monotone phase bits");
    }
    declared_phase_bits |= provider->phase_bits[i];
  }
  if (declared_phase_bits != provider->valid_bits) {
    return loom_source_dataflow_invalid_provider(
        provider, "phase bits do not partition all valid evidence");
  }
  if ((provider->dialect_count != 0) != (provider->dialects != NULL)) {
    return loom_source_dataflow_invalid_provider(
        provider, "has inconsistent dialect table storage");
  }
  if ((uint16_t)provider->dialect_base_id + provider->dialect_count > 256) {
    return loom_source_dataflow_invalid_provider(provider,
                                                 "dialect range overflows");
  }
  if ((provider->operation_count != 0) != (provider->operations != NULL) ||
      (provider->port_count != 0) != (provider->ports != NULL) ||
      (provider->rule_count != 0) != (provider->rules != NULL) ||
      (provider->predicate_count != 0) != (provider->predicates != NULL) ||
      (provider->flow_rule_count != 0) != (provider->flow_rules != NULL) ||
      (provider->projection_count != 0) != (provider->projections != NULL)) {
    return loom_source_dataflow_invalid_provider(
        provider, "has inconsistent generated pool storage");
  }
  if (provider->predicate_count > 64) {
    return loom_source_dataflow_invalid_provider(
        provider, "registers more than 64 operation predicates");
  }
  for (uint8_t i = 0; i < provider->predicate_count; ++i) {
    if (provider->predicates[i].fn == NULL) {
      return loom_source_dataflow_invalid_provider(
          provider, "registers a NULL operation predicate");
    }
  }
  for (uint8_t dialect_index = 0; dialect_index < provider->dialect_count;
       ++dialect_index) {
    const loom_source_dataflow_dialect_table_t* dialect =
        &provider->dialects[dialect_index];
    if (dialect->operation_count > 256 ||
        (dialect->operation_count != 0 && dialect->operation_indices == NULL)) {
      return loom_source_dataflow_invalid_provider(
          provider, "has an invalid dialect operation table");
    }
    for (uint16_t i = 0; i < dialect->operation_count; ++i) {
      const uint16_t operation_index = dialect->operation_indices[i];
      if (operation_index != LOOM_SOURCE_DATAFLOW_OPERATION_INDEX_NONE &&
          operation_index > provider->operation_count) {
        return loom_source_dataflow_invalid_provider(
            provider, "references an absent operation row");
      }
    }
  }
  for (uint16_t i = 0; i < provider->port_count; ++i) {
    if (provider->ports[i].kind != LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD &&
        provider->ports[i].kind != LOOM_SOURCE_DATAFLOW_PORT_RESULT_FIELD) {
      return loom_source_dataflow_invalid_provider(provider,
                                                   "has an unknown port kind");
    }
  }
  for (uint16_t i = 0; i < provider->rule_count; ++i) {
    const loom_source_dataflow_rule_t* rule = &provider->rules[i];
    if (rule->reserved != 0 || rule->phase >= provider->phase_count ||
        rule->kind > LOOM_SOURCE_DATAFLOW_RULE_ALL ||
        (rule->source_bits & ~provider->valid_bits) != 0 ||
        (rule->target_bits & ~provider->valid_bits) != 0 ||
        rule->predicate_index_plus_one > provider->predicate_count) {
      return loom_source_dataflow_invalid_provider(provider,
                                                   "has an invalid rule row");
    }
    loom_source_dataflow_bits_t available_bits = 0;
    for (uint8_t phase = 0; phase <= rule->phase; ++phase) {
      available_bits |= provider->phase_bits[phase];
    }
    if ((rule->source_bits & ~available_bits) != 0 ||
        (rule->target_bits & ~provider->phase_bits[rule->phase]) != 0) {
      return loom_source_dataflow_invalid_provider(
          provider, "has a rule crossing monotone phase strata");
    }
    if (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_SEED) {
      if (rule->source_bits != 0 || rule->source_port_mask != 0 ||
          rule->target_bits == 0 || rule->target_port_mask == 0) {
        return loom_source_dataflow_invalid_provider(
            provider, "has a malformed seed rule");
      }
    } else {
      if (rule->source_bits == 0 || rule->source_port_mask == 0 ||
          rule->target_bits == 0 || rule->target_port_mask == 0) {
        return loom_source_dataflow_invalid_provider(
            provider, "has a malformed transfer rule");
      }
      if (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_COPY &&
          rule->source_bits != rule->target_bits) {
        return loom_source_dataflow_invalid_provider(
            provider, "has a copy rule that remaps evidence bits");
      }
    }
  }
  for (uint16_t i = 0; i < provider->flow_rule_count; ++i) {
    const loom_source_dataflow_flow_rule_t* rule = &provider->flow_rules[i];
    if (rule->reserved[0] != 0 || rule->reserved[1] != 0 ||
        rule->reserved[2] != 0 || rule->phase >= provider->phase_count ||
        rule->kind == LOOM_SOURCE_DATAFLOW_RULE_SEED ||
        rule->kind > LOOM_SOURCE_DATAFLOW_RULE_ALL || rule->flow_kinds == 0 ||
        (rule->flow_kinds & ~LOOM_SOURCE_PROGRAM_VALUE_FLOW_KIND_MASK) != 0 ||
        rule->directions == 0 ||
        (rule->directions & ~LOOM_SOURCE_DATAFLOW_FLOW_DIRECTION_MASK) != 0 ||
        rule->source_bits == 0 || rule->target_bits == 0) {
      return loom_source_dataflow_invalid_provider(
          provider, "has an invalid source-program flow rule");
    }
    loom_source_dataflow_bits_t available_bits = 0;
    for (uint8_t phase = 0; phase <= rule->phase; ++phase) {
      available_bits |= provider->phase_bits[phase];
    }
    if ((rule->source_bits & ~available_bits) != 0 ||
        (rule->target_bits & ~provider->phase_bits[rule->phase]) != 0 ||
        (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_COPY &&
         rule->source_bits != rule->target_bits)) {
      return loom_source_dataflow_invalid_provider(
          provider, "has a source-program flow rule crossing phase strata");
    }
  }
  for (uint16_t i = 0; i < provider->projection_count; ++i) {
    const loom_source_dataflow_projection_t* projection =
        &provider->projections[i];
    if (projection->reserved[0] != 0 || projection->reserved[1] != 0 ||
        projection->reserved[2] != 0 || projection->phase == 0 ||
        projection->phase >= provider->phase_count ||
        projection->target_bits == 0 ||
        (projection->required_bits & projection->forbidden_bits) != 0 ||
        (projection->required_value_flags &
         projection->forbidden_value_flags) != 0 ||
        ((projection->required_value_flags |
          projection->forbidden_value_flags) &
         ~LOOM_SOURCE_PROGRAM_VALUE_FLAG_MASK) != 0) {
      return loom_source_dataflow_invalid_provider(
          provider, "has an invalid phase-boundary projection");
    }
    loom_source_dataflow_bits_t prior_bits = 0;
    for (uint8_t phase = 0; phase < projection->phase; ++phase) {
      prior_bits |= provider->phase_bits[phase];
    }
    if (((projection->required_bits | projection->forbidden_bits) &
         ~prior_bits) != 0 ||
        (projection->target_bits & ~provider->phase_bits[projection->phase]) !=
            0) {
      return loom_source_dataflow_invalid_provider(
          provider, "has a projection crossing monotone phase strata");
    }
  }
  for (uint16_t i = 0; i < provider->operation_count; ++i) {
    const loom_source_dataflow_operation_t* operation =
        &provider->operations[i];
    if (operation->reserved != 0 ||
        operation->port_count > LOOM_SOURCE_DATAFLOW_MAX_PORT_COUNT ||
        operation->port_start > provider->port_count ||
        operation->port_count > provider->port_count - operation->port_start ||
        operation->rule_start > provider->rule_count ||
        operation->rule_count > provider->rule_count - operation->rule_start) {
      return loom_source_dataflow_invalid_provider(
          provider, "has an invalid operation row span");
    }
    const uint32_t valid_port_mask =
        operation->port_count == 32
            ? UINT32_MAX
            : (((uint32_t)1u << operation->port_count) - 1);
    for (uint8_t j = 0; j < operation->rule_count; ++j) {
      const loom_source_dataflow_rule_t* rule =
          &provider->rules[operation->rule_start + j];
      if (((rule->source_port_mask | rule->target_port_mask) &
           ~valid_port_mask) != 0) {
        return loom_source_dataflow_invalid_provider(
            provider, "has a rule that selects an absent operation port");
      }
    }
  }
  return iree_ok_status();
}
