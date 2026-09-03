// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_representation_verify.h"

static iree_status_t loom_low_source_representation_invalid_provider(
    const loom_low_source_representation_provider_t* provider,
    const char* detail) {
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "source representation provider '%.*s' %s",
      (int)provider->name.size, provider->name.data, detail);
}

static bool loom_low_source_representation_span_is_valid(uint32_t start,
                                                         uint32_t count,
                                                         uint32_t capacity) {
  return start <= capacity && count <= capacity - start;
}

static bool loom_low_source_representation_name_is_valid(
    const loom_low_source_representation_provider_t* provider,
    loom_bstring_table_offset_t offset) {
  loom_bstring_t name = NULL;
  return loom_bstring_table_try_get(&provider->string_table, offset, &name) &&
         loom_bstring_length(name) != 0;
}

static bool loom_low_source_representation_bindings_equal(
    const loom_low_source_representation_provider_t* provider,
    const loom_low_source_representation_candidate_t* left,
    const loom_low_source_representation_candidate_t* right) {
  if (left->binding_count != right->binding_count) return false;
  for (uint16_t i = 0; i < left->binding_count; ++i) {
    const loom_low_source_representation_binding_t left_binding =
        provider->bindings[left->binding_start + i];
    const loom_low_source_representation_binding_t right_binding =
        provider->bindings[right->binding_start + i];
    if (left_binding.representation_index !=
            right_binding.representation_index ||
        left_binding.flags != right_binding.flags) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_source_representation_verify_recipe(
    const loom_low_source_representation_provider_t* provider,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_source_representation_candidate_t* candidate,
    iree_arena_allocator_t* arena) {
  const loom_low_descriptor_recipe_t recipe = {
      .entries = candidate->recipe_entry_count == 0
                     ? NULL
                     : &provider->recipe_entries[candidate->recipe_entry_start],
      .entry_count = candidate->recipe_entry_count,
      .dependencies =
          candidate->recipe_dependency_count == 0
              ? NULL
              : &provider
                     ->recipe_dependencies[candidate->recipe_dependency_start],
      .dependency_count = candidate->recipe_dependency_count,
      .durable_pressure_deltas =
          candidate->durable_pressure_delta_count == 0
              ? NULL
              : &provider->durable_pressure_deltas
                     [candidate->durable_pressure_delta_start],
      .durable_pressure_delta_count = candidate->durable_pressure_delta_count,
  };
  loom_low_descriptor_cost_t cost = {0};
  return loom_low_descriptor_cost_compute(descriptor_set, &recipe, arena,
                                          &cost);
}

iree_status_t loom_low_source_representation_provider_verify(
    const loom_low_source_representation_provider_t* provider,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena) {
  if (provider == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source representation provider is NULL");
  }
  if (descriptor_set == NULL || arena == NULL) {
    return loom_low_source_representation_invalid_provider(
        provider, "has no descriptor set or verification arena");
  }
  if (iree_string_view_is_empty(provider->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source representation provider name is empty");
  }
  if (provider->representation_count == 0 ||
      provider->representations == NULL) {
    return loom_low_source_representation_invalid_provider(
        provider, "has no physical representations");
  }
  if (provider->string_table.data == NULL ||
      provider->string_table.data_length == 0) {
    return loom_low_source_representation_invalid_provider(
        provider, "has no packed string table");
  }
  if ((provider->dialect_count != 0) != (provider->dialects != NULL) ||
      (provider->operation_count != 0) != (provider->operations != NULL) ||
      (provider->group_count != 0) != (provider->groups != NULL) ||
      (provider->port_count != 0) != (provider->ports != NULL) ||
      (provider->candidate_count != 0) != (provider->candidates != NULL) ||
      (provider->binding_count != 0) != (provider->bindings != NULL) ||
      (provider->recipe_entry_count != 0) !=
          (provider->recipe_entries != NULL) ||
      (provider->recipe_dependency_count != 0) !=
          (provider->recipe_dependencies != NULL) ||
      (provider->durable_pressure_delta_count != 0) !=
          (provider->durable_pressure_deltas != NULL) ||
      (provider->predicate_count != 0) != (provider->predicates != NULL)) {
    return loom_low_source_representation_invalid_provider(
        provider, "has inconsistent generated pool storage");
  }
  if (provider->reserved != 0 ||
      (uint16_t)provider->dialect_base_id + provider->dialect_count > 256) {
    return loom_low_source_representation_invalid_provider(
        provider, "has invalid provider metadata");
  }
  if ((provider->target_data_count != 0) != (provider->target_data != NULL) ||
      (provider->target_data_count != 0) !=
          (provider->target_data_stride != 0) ||
      (provider->target_data_count != 0 &&
       provider->target_data_count >
           IREE_HOST_SIZE_MAX / provider->target_data_stride)) {
    return loom_low_source_representation_invalid_provider(
        provider, "has inconsistent target-data storage");
  }

  for (uint16_t i = 0; i < provider->representation_count; ++i) {
    const loom_low_source_representation_t* representation =
        &provider->representations[i];
    if (representation->stable_key == LOOM_LOW_SOURCE_REPRESENTATION_KEY_NONE ||
        representation->reserved != 0 ||
        !loom_low_source_representation_name_is_valid(
            provider, representation->name_string_offset)) {
      return loom_low_source_representation_invalid_provider(
          provider, "has an invalid physical representation row");
    }
    for (uint16_t j = 0; j < i; ++j) {
      if (provider->representations[j].stable_key ==
          representation->stable_key) {
        return loom_low_source_representation_invalid_provider(
            provider, "has duplicate physical representation identities");
      }
    }
  }

  for (uint8_t i = 0; i < provider->predicate_count; ++i) {
    if (provider->predicates[i].fn == NULL) {
      return loom_low_source_representation_invalid_provider(
          provider, "registers a NULL candidate predicate");
    }
  }

  for (uint8_t dialect_index = 0; dialect_index < provider->dialect_count;
       ++dialect_index) {
    const loom_low_source_representation_dialect_table_t* dialect =
        &provider->dialects[dialect_index];
    if (dialect->operation_count > 256 ||
        (dialect->operation_count != 0 && dialect->operation_indices == NULL)) {
      return loom_low_source_representation_invalid_provider(
          provider, "has an invalid dialect operation table");
    }
    for (uint16_t i = 0; i < dialect->operation_count; ++i) {
      const uint16_t operation_index = dialect->operation_indices[i];
      if (operation_index !=
              LOOM_LOW_SOURCE_REPRESENTATION_OPERATION_INDEX_NONE &&
          operation_index > provider->operation_count) {
        return loom_low_source_representation_invalid_provider(
            provider, "references an absent operation row");
      }
    }
  }

  for (uint16_t i = 0; i < provider->operation_count; ++i) {
    const loom_low_source_representation_operation_t* operation =
        &provider->operations[i];
    if (operation->reserved != 0 || operation->group_count == 0 ||
        !loom_low_source_representation_span_is_valid(operation->group_start,
                                                      operation->group_count,
                                                      provider->group_count)) {
      return loom_low_source_representation_invalid_provider(
          provider, "has an invalid operation row");
    }
  }

  for (uint16_t i = 0; i < provider->port_count; ++i) {
    const loom_low_source_representation_port_t* port = &provider->ports[i];
    if ((port->kind != LOOM_LOW_SOURCE_REPRESENTATION_PORT_OPERAND_FIELD &&
         port->kind != LOOM_LOW_SOURCE_REPRESENTATION_PORT_RESULT_FIELD) ||
        port->reserved != 0 || port->extension != 0) {
      return loom_low_source_representation_invalid_provider(
          provider, "has an invalid source value port");
    }
  }

  for (uint16_t i = 0; i < provider->binding_count; ++i) {
    const loom_low_source_representation_binding_t* binding =
        &provider->bindings[i];
    if (binding->representation_index >= provider->representation_count ||
        binding->reserved != 0 ||
        (binding->flags & ~LOOM_LOW_SOURCE_REPRESENTATION_BINDING_FLAG_MASK) !=
            0) {
      return loom_low_source_representation_invalid_provider(
          provider, "has an invalid candidate component binding");
    }
  }

  for (uint16_t i = 0; i < provider->candidate_count; ++i) {
    const loom_low_source_representation_candidate_t* candidate =
        &provider->candidates[i];
    if (candidate->stable_key == 0 || candidate->reserved != 0 ||
        !loom_low_source_representation_name_is_valid(
            provider, candidate->name_string_offset) ||
        candidate->proof < LOOM_LOW_SOURCE_REPRESENTATION_PROOF_EXACT ||
        candidate->proof > LOOM_LOW_SOURCE_REPRESENTATION_PROOF_OPAQUE ||
        candidate->predicate_index_plus_one > provider->predicate_count ||
        !loom_low_source_representation_span_is_valid(
            candidate->binding_start, candidate->binding_count,
            provider->binding_count) ||
        !loom_low_source_representation_span_is_valid(
            candidate->recipe_entry_start, candidate->recipe_entry_count,
            provider->recipe_entry_count) ||
        !loom_low_source_representation_span_is_valid(
            candidate->recipe_dependency_start,
            candidate->recipe_dependency_count,
            provider->recipe_dependency_count) ||
        !loom_low_source_representation_span_is_valid(
            candidate->durable_pressure_delta_start,
            candidate->durable_pressure_delta_count,
            provider->durable_pressure_delta_count) ||
        (candidate->target_data_ordinal !=
             LOOM_LOW_SOURCE_REPRESENTATION_TARGET_DATA_ORDINAL_NONE &&
         candidate->target_data_ordinal >= provider->target_data_count)) {
      return loom_low_source_representation_invalid_provider(
          provider, "has an invalid realization candidate row");
    }
    for (uint16_t j = 0; j < i; ++j) {
      if (provider->candidates[j].stable_key == candidate->stable_key) {
        return loom_low_source_representation_invalid_provider(
            provider, "has duplicate realization candidate identities");
      }
    }
    IREE_RETURN_IF_ERROR(loom_low_source_representation_verify_recipe(
        provider, descriptor_set, candidate, arena));
  }

  for (uint16_t group_index = 0; group_index < provider->group_count;
       ++group_index) {
    const loom_low_source_representation_group_t* group =
        &provider->groups[group_index];
    if (group->stable_key == 0 ||
        !loom_low_source_representation_name_is_valid(
            provider, group->name_string_offset) ||
        group->port_count == 0 || group->candidate_count == 0 ||
        group->component_count == 0 ||
        group->component_count >
            LOOM_LOW_SOURCE_REPRESENTATION_MAX_COMPONENT_COUNT ||
        group->component_count > group->port_count ||
        (group->flags & ~LOOM_LOW_SOURCE_REPRESENTATION_GROUP_FLAG_MASK) != 0 ||
        !loom_low_source_representation_span_is_valid(
            group->port_start, group->port_count, provider->port_count) ||
        !loom_low_source_representation_span_is_valid(
            group->candidate_start, group->candidate_count,
            provider->candidate_count)) {
      return loom_low_source_representation_invalid_provider(
          provider, "has an invalid candidate group row");
    }
    for (uint16_t prior = 0; prior < group_index; ++prior) {
      if (provider->groups[prior].stable_key == group->stable_key) {
        return loom_low_source_representation_invalid_provider(
            provider, "has duplicate candidate group identities");
      }
    }

    uint32_t covered_component_mask = 0;
    for (uint8_t port_ordinal = 0; port_ordinal < group->port_count;
         ++port_ordinal) {
      const loom_low_source_representation_port_t* port =
          &provider->ports[group->port_start + port_ordinal];
      if (port->component_index >= group->component_count) {
        return loom_low_source_representation_invalid_provider(
            provider, "has a port selecting an absent component slot");
      }
      covered_component_mask |= (uint32_t)1u << port->component_index;
    }
    const uint32_t required_component_mask =
        group->component_count == 32
            ? UINT32_MAX
            : ((uint32_t)1u << group->component_count) - 1;
    if (covered_component_mask != required_component_mask) {
      return loom_low_source_representation_invalid_provider(
          provider, "has a component slot without a source value port");
    }

    for (uint8_t candidate_ordinal = 0;
         candidate_ordinal < group->candidate_count; ++candidate_ordinal) {
      const loom_low_source_representation_candidate_t* candidate =
          &provider->candidates[group->candidate_start + candidate_ordinal];
      if (candidate->binding_count != group->component_count) {
        return loom_low_source_representation_invalid_provider(
            provider, "has a candidate with the wrong component arity");
      }
      for (uint8_t prior = 0; prior < candidate_ordinal; ++prior) {
        const loom_low_source_representation_candidate_t* prior_candidate =
            &provider->candidates[group->candidate_start + prior];
        if (candidate->predicate_index_plus_one ==
                prior_candidate->predicate_index_plus_one &&
            loom_low_source_representation_bindings_equal(provider, candidate,
                                                          prior_candidate)) {
          return loom_low_source_representation_invalid_provider(
              provider,
              "has duplicate simultaneously eligible representation tuples");
        }
      }
    }

    for (uint8_t component = 0; component < group->component_count;
         ++component) {
      uint16_t canonical_representation =
          LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE;
      for (uint8_t candidate_ordinal = 0;
           candidate_ordinal < group->candidate_count; ++candidate_ordinal) {
        const loom_low_source_representation_candidate_t* candidate =
            &provider->candidates[group->candidate_start + candidate_ordinal];
        const loom_low_source_representation_binding_t binding =
            provider->bindings[candidate->binding_start + component];
        const bool is_canonical = iree_any_bit_set(
            binding.flags, LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL);
        for (uint8_t prior = 0; prior < candidate_ordinal; ++prior) {
          const loom_low_source_representation_candidate_t* prior_candidate =
              &provider->candidates[group->candidate_start + prior];
          const loom_low_source_representation_binding_t prior_binding =
              provider->bindings[prior_candidate->binding_start + component];
          if (prior_binding.representation_index ==
                  binding.representation_index &&
              iree_any_bit_set(
                  prior_binding.flags,
                  LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL) !=
                  is_canonical) {
            return loom_low_source_representation_invalid_provider(
                provider,
                "classifies one representation inconsistently as canonical");
          }
        }
        if (!is_canonical) continue;
        if (canonical_representation !=
                LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE &&
            canonical_representation != binding.representation_index) {
          return loom_low_source_representation_invalid_provider(
              provider, "has multiple canonical representations in a domain");
        }
        canonical_representation = binding.representation_index;
      }
      if (canonical_representation ==
          LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE) {
        return loom_low_source_representation_invalid_provider(
            provider, "has a candidate domain without a canonical fallback");
      }
    }
  }

  return iree_ok_status();
}
