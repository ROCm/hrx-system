// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_structure.h"

#include "loom/error/error_catalog.h"
#include "loom/ops/op_defs.h"
#include "loom/verify/verify_diagnostics.h"

static void loom_verify_segmented_operand_structure(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, iree_string_view_t op_name) {
  uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (segment_count == 0 || !vtable->operand_descriptors) {
    if (op->operand_count != 0) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
          loom_param_u32(0),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_001, params,
                                  IREE_ARRAYSIZE(params));
    }
    return;
  }

  const uint16_t* segment_counts = loom_op_const_operand_segment_counts(op);
  uint32_t total_count = 0;
  for (uint8_t i = 0; i < segment_count; ++i) {
    const loom_operand_descriptor_t* descriptor =
        &vtable->operand_descriptors[i];
    uint16_t segment_value_count = segment_counts[i];
    total_count += segment_value_count;
    if (iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC)) {
      continue;
    }
    uint16_t expected_count = 1;
    if (iree_any_bit_set(descriptor->flags, LOOM_OPERAND_OPTIONAL)) {
      if (segment_value_count <= 1) continue;
      expected_count = 1;
    } else if (segment_value_count == 1) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(segment_value_count),
        loom_param_u32(expected_count),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_001, params,
                                IREE_ARRAYSIZE(params));
  }
  if (total_count != op->operand_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(op->operand_count),
        loom_param_u32(total_count),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_001, params,
                                IREE_ARRAYSIZE(params));
  }
}

static void loom_verify_emit_symbol_definition_diagnostic(
    loom_verify_state_t* state, const loom_op_t* op, loom_symbol_ref_t ref,
    uint8_t symbol_attr_index, const loom_symbol_t* symbol) {
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_diagnostic_field(
          loom_verify_symbol_name(state, ref), LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
          symbol_attr_index),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("first definition here"),
      .op = symbol->defining_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_SYMBOL_005,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

iree_status_t loom_verify_symbol_definition(loom_verify_state_t* state,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable) {
  if (!vtable->symbol_def || !vtable->attr_descriptors) {
    return iree_ok_status();
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return iree_ok_status();
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[symbol_attr_index];
  if (descriptor->attr_kind != LOOM_ATTR_SYMBOL) return iree_ok_status();
  loom_symbol_ref_t ref =
      loom_attr_as_symbol(loom_op_const_attrs(op)[symbol_attr_index]);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= state->module->symbols.count) {
    return iree_ok_status();
  }
  const loom_symbol_t* symbol = &state->module->symbols.entries[ref.symbol_id];
  if (symbol->defining_op && symbol->defining_op != op) {
    loom_verify_emit_symbol_definition_diagnostic(state, op, ref,
                                                  symbol_attr_index, symbol);
  }
  return iree_ok_status();
}

static bool loom_verify_trait_conflict(loom_trait_flags_t traits,
                                       iree_string_view_t* out_trait_a,
                                       iree_string_view_t* out_trait_b) {
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_PURE)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("PURE");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("UNKNOWN_EFFECTS");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_HINT | LOOM_TRAIT_NON_DETERMINISTIC)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("NON_DETERMINISTIC");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("CONVERGENT");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_PURE | LOOM_TRAIT_NON_DETERMINISTIC)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("NON_DETERMINISTIC");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("UNKNOWN_EFFECTS");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_UNIQUE_IDENTITY)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("UNIQUE_IDENTITY");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("UNKNOWN_EFFECTS");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("UNKNOWN_EFFECTS");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("UNKNOWN_EFFECTS");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_SAFE_TO_SPECULATE |
                                    LOOM_TRAIT_NON_DETERMINISTIC)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("NON_DETERMINISTIC");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_UNIQUE_IDENTITY)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("UNIQUE_IDENTITY");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_HINT)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("HINT");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_CONVERGENT)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("CONVERGENT");
    return true;
  }
  return false;
}

static void loom_verify_op_trait_flags_consistency(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_trait_flags_t traits) {
  iree_string_view_t trait_a = iree_string_view_empty();
  iree_string_view_t trait_b = iree_string_view_empty();
  if (!loom_verify_trait_conflict(traits, &trait_a, &trait_b)) return;
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_string(trait_a),
      loom_param_string(trait_b),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_016, params,
                              IREE_ARRAYSIZE(params));
}

void loom_verify_op_declared_trait_consistency(loom_verify_state_t* state,
                                               const loom_op_t* op,
                                               const loom_op_vtable_t* vtable) {
  loom_verify_op_trait_flags_consistency(state, op, vtable, vtable->traits);
}

void loom_verify_op_effective_trait_consistency(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  loom_trait_flags_t effective_traits =
      loom_op_effective_traits(state->module, op);
  if (effective_traits == vtable->traits) return;
  loom_verify_op_trait_flags_consistency(state, op, vtable, effective_traits);
}

static iree_string_view_t loom_verify_kind_name(loom_verify_state_t* state,
                                                loom_op_kind_t kind) {
  const loom_op_vtable_t* vtable = loom_verify_lookup_vtable(state, kind);
  return vtable ? loom_op_vtable_name(vtable) : IREE_SV("unknown");
}

static iree_string_view_t loom_verify_parent_context_name(
    loom_verify_state_t* state, const loom_op_t* op) {
  if (!op->parent_op) return IREE_SV("module");
  return loom_verify_kind_name(state, op->parent_op->kind);
}

static const loom_op_t* loom_verify_find_ancestor(const loom_op_t* op,
                                                  loom_op_kind_t kind) {
  const loom_op_t* parent = op->parent_op;
  while (parent) {
    if (parent->kind == kind) return parent;
    parent = parent->parent_op;
  }
  return NULL;
}

static void loom_verify_emit_placement_diagnostic(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, iree_string_view_t constraint_kind,
    loom_op_kind_t ancestor_kind, iree_string_view_t actual_ancestor) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_string(constraint_kind),
      loom_param_string(loom_verify_kind_name(state, ancestor_kind)),
      loom_param_string(actual_ancestor),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_029, params,
                              IREE_ARRAYSIZE(params));
}

void loom_verify_op_placement(loom_verify_state_t* state, const loom_op_t* op,
                              const loom_op_vtable_t* vtable) {
  const loom_op_placement_descriptor_t* placement = vtable->placement;
  if (!placement) return;

  for (uint8_t i = 0; i < placement->required_parent_count; ++i) {
    loom_op_kind_t parent_kind = placement->required_parents[i];
    if (op->parent_op && op->parent_op->kind == parent_kind) continue;
    loom_verify_emit_placement_diagnostic(
        state, op, vtable, IREE_SV("parent"), parent_kind,
        loom_verify_parent_context_name(state, op));
  }
  for (uint8_t i = 0; i < placement->required_ancestor_count; ++i) {
    loom_op_kind_t ancestor_kind = placement->required_ancestors[i];
    if (loom_verify_find_ancestor(op, ancestor_kind)) continue;
    loom_verify_emit_placement_diagnostic(
        state, op, vtable, IREE_SV("required"), ancestor_kind,
        loom_verify_parent_context_name(state, op));
  }
  for (uint8_t i = 0; i < placement->forbidden_ancestor_count; ++i) {
    loom_op_kind_t ancestor_kind = placement->forbidden_ancestors[i];
    const loom_op_t* ancestor = loom_verify_find_ancestor(op, ancestor_kind);
    if (!ancestor) continue;
    loom_verify_emit_placement_diagnostic(
        state, op, vtable, IREE_SV("forbidden"), ancestor_kind,
        loom_verify_kind_name(state, ancestor->kind));
  }
}

static uint8_t loom_verify_required_region_count(
    const loom_op_vtable_t* vtable) {
  if (!vtable->region_descriptors) return vtable->region_count;
  bool has_variadic_regions =
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS);
  uint8_t fixed_region_count = has_variadic_regions && vtable->region_count > 0
                                   ? (uint8_t)(vtable->region_count - 1)
                                   : vtable->region_count;
  for (uint8_t i = 0; i < fixed_region_count; ++i) {
    const loom_region_descriptor_t* region_descriptor =
        loom_op_vtable_region_descriptor(vtable, i);
    if (!region_descriptor) return fixed_region_count;
    if (iree_any_bit_set(region_descriptor->flags, LOOM_REGION_OPTIONAL)) {
      return i;
    }
  }
  return fixed_region_count;
}

void loom_verify_func_purity_body_effects(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable) {
  const loom_func_like_vtable_t* func_vtable = vtable->func_like;
  if (!func_vtable) return;
  if (func_vtable->purity_attr_index == LOOM_ATTR_INDEX_NONE) return;
  if (func_vtable->body_region_index == LOOM_REGION_INDEX_NONE) return;
  if (func_vtable->body_region_index >= op->region_count) return;
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  if (loom_attr_as_enum(attrs[func_vtable->purity_attr_index]) == 0) return;
  loom_region_t* body = loom_op_regions(op)[func_vtable->body_region_index];
  if (!loom_region_has_read_effects(body) &&
      !loom_region_has_write_effects(body) &&
      !loom_region_has_convergent_effects(body)) {
    return;
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_u32(body ? body->read_effect_count : 0),
      loom_param_u32(body ? body->write_effect_count : 0),
      loom_param_u32(body ? body->convergent_effect_count : 0),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_017, params,
                              IREE_ARRAYSIZE(params));
}
void loom_verify_op_structure(loom_verify_state_t* state, const loom_op_t* op,
                              const loom_op_vtable_t* vtable) {
  iree_string_view_t op_name = loom_op_vtable_name(vtable);

  // Check operand count.
  bool has_segmented_operands = loom_op_vtable_has_segmented_operands(vtable);
  bool has_variadic_operands =
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_OPERANDS);
  if (has_segmented_operands) {
    loom_verify_segmented_operand_structure(state, op, vtable, op_name);
  } else if (has_variadic_operands) {
    if (op->operand_count < vtable->fixed_operand_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
          loom_param_u32(vtable->fixed_operand_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_001, params,
                                  IREE_ARRAYSIZE(params));
    }
  } else {
    const uint8_t operand_descriptor_count =
        loom_op_vtable_operand_descriptor_count(vtable);
    if (op->operand_count < vtable->fixed_operand_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
          loom_param_u32(vtable->fixed_operand_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_001, params,
                                  IREE_ARRAYSIZE(params));
    } else if (op->operand_count > operand_descriptor_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
          loom_param_u32(operand_descriptor_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_001, params,
                                  IREE_ARRAYSIZE(params));
    }
  }

  // Check result count.
  bool has_variadic_results =
      (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
  if (has_variadic_results) {
    if (op->result_count < vtable->fixed_result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
          loom_param_u32(vtable->fixed_result_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_002, params,
                                  IREE_ARRAYSIZE(params));
    }
  } else {
    if (op->result_count != vtable->fixed_result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
          loom_param_u32(vtable->fixed_result_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_002, params,
                                  IREE_ARRAYSIZE(params));
    }
  }

  // Check attribute count.
  if (op->attribute_count != vtable->attribute_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(op->attribute_count),
        loom_param_u32(vtable->attribute_count),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_003, params,
                                IREE_ARRAYSIZE(params));
  }

  // Check region count.
  bool has_variadic_regions =
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS);
  uint8_t minimum_region_count = loom_verify_required_region_count(vtable);
  bool region_count_matches =
      has_variadic_regions ? op->region_count >= minimum_region_count
                           : op->region_count >= minimum_region_count &&
                                 op->region_count <= vtable->region_count;
  if (!region_count_matches) {
    uint8_t expected_region_count = op->region_count < minimum_region_count
                                        ? minimum_region_count
                                        : vtable->region_count;
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(op->region_count),
        loom_param_u32(expected_region_count),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_004, params,
                                IREE_ARRAYSIZE(params));
  }
}

static bool loom_verify_region_directly_contains_block(
    const loom_region_t* region, const loom_block_t* target) {
  if (!region || !target) return false;
  for (uint16_t i = 0; i < region->block_count; ++i) {
    if (loom_region_const_block(region, i) == target) return true;
  }
  return false;
}

void loom_verify_successor_targets(loom_verify_state_t* state,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable) {
  if (op->successor_count == 0) return;
  const loom_region_t* parent_region =
      op->parent_block ? op->parent_block->parent_region : NULL;
  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  loom_block_t* const* successors = loom_op_const_successors(op);
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    loom_diagnostic_field_ref_t successor_ref =
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, i);
    if (!successors[i]) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_with_field_ref(loom_param_u32(i), successor_ref),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_023, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }
    if (successors[i]->parent_region != parent_region ||
        !loom_verify_region_directly_contains_block(parent_region,
                                                    successors[i])) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_with_field_ref(loom_param_u32(i), successor_ref),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_024, params,
                                  IREE_ARRAYSIZE(params));
    }
  }
}

// Resolves a field reference to its author-facing DSL field name when available
static bool loom_verify_attr_kind_matches_descriptor(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor) {
  if (descriptor->attr_kind == LOOM_ATTR_ANY) {
    return attr.kind > LOOM_ATTR_ABSENT && attr.kind < LOOM_ATTR_ANY;
  }
  return attr.kind == descriptor->attr_kind;
}

static void loom_verify_predicate_list_attr(loom_verify_state_t* state,
                                            const loom_op_t* op,
                                            iree_string_view_t name,
                                            uint8_t attr_index,
                                            loom_attribute_t attr) {
  if (attr.kind != LOOM_ATTR_PREDICATE_LIST) return;
  loom_diagnostic_param_t attr_name_param =
      loom_verify_param_string_for_diagnostic_field(
          name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index);
  if (attr.count > 0 && !attr.predicate_list) {
    loom_diagnostic_param_t params[] = {
        attr_name_param,
        loom_param_u32(attr.count),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_019, params,
                                IREE_ARRAYSIZE(params));
    return;
  }
  for (uint16_t predicate_index = 0; predicate_index < attr.count;
       ++predicate_index) {
    const loom_predicate_t* predicate = &attr.predicate_list[predicate_index];
    const char* predicate_name = loom_predicate_kind_name(predicate->kind);
    if (!predicate_name) {
      loom_diagnostic_param_t params[] = {
          attr_name_param,
          loom_param_u32(predicate_index),
          loom_param_u32(predicate->kind),
          loom_param_u32(LOOM_PREDICATE_COUNT_),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_020, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }

    uint8_t expected_argument_count =
        loom_predicate_kind_argument_count(predicate->kind);
    if (predicate->arg_count != expected_argument_count) {
      loom_diagnostic_param_t params[] = {
          attr_name_param,
          loom_param_u32(predicate_index),
          loom_param_string(iree_make_cstring_view(predicate_name)),
          loom_param_u32(expected_argument_count),
          loom_param_u32(predicate->arg_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_021, params,
                                  IREE_ARRAYSIZE(params));
    }

    uint8_t argument_count = predicate->arg_count;
    if (argument_count > IREE_ARRAYSIZE(predicate->arg_tags)) {
      argument_count = (uint8_t)IREE_ARRAYSIZE(predicate->arg_tags);
    }
    for (uint8_t argument_index = 0; argument_index < argument_count;
         ++argument_index) {
      uint8_t tag = predicate->arg_tags[argument_index];
      if (tag > LOOM_PRED_ARG_NONE && tag < LOOM_PRED_ARG_COUNT_) continue;
      loom_diagnostic_param_t params[] = {
          attr_name_param,
          loom_param_u32(predicate_index),
          loom_param_u32(argument_index),
          loom_param_u32(tag),
          loom_param_u32(LOOM_PRED_ARG_COUNT_),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_022, params,
                                  IREE_ARRAYSIZE(params));
    }
  }
}

void loom_verify_type_constraints(loom_verify_state_t* state,
                                  const loom_op_t* op,
                                  const loom_op_vtable_t* vtable) {
  // Check operand type constraints.
  if (vtable->operand_descriptors) {
    bool has_segmented = loom_op_vtable_has_segmented_operands(vtable);
    bool has_variadic = iree_any_bit_set(vtable->vtable_flags,
                                         LOOM_OP_VTABLE_VARIADIC_OPERANDS);
    uint8_t descriptor_count = loom_op_vtable_operand_descriptor_count(vtable);
    for (uint8_t i = 0; i < descriptor_count; ++i) {
      loom_type_constraint_t constraint =
          (loom_type_constraint_t)vtable->operand_descriptors[i]
              .type_constraint;
      if (constraint == LOOM_TYPE_CONSTRAINT_ANY) continue;

      loom_value_slice_t operand_span = {0};
      bool operand_is_variadic = false;
      if (has_segmented) {
        operand_span = loom_op_operand_field_span(vtable, op, i);
        operand_is_variadic = iree_any_bit_set(
            vtable->operand_descriptors[i].flags, LOOM_OPERAND_VARIADIC);
      } else {
        if (i >= op->operand_count) continue;
        uint16_t start = i;
        uint16_t end = (has_variadic && i == vtable->fixed_operand_count)
                           ? op->operand_count
                           : (uint16_t)(i + 1);
        operand_span = (loom_value_slice_t){
            .values = loom_op_operands(op) + start,
            .count = (uint16_t)(end - start),
        };
        operand_is_variadic = has_variadic && i == vtable->fixed_operand_count;
      }
      for (uint16_t j = 0; j < operand_span.count; ++j) {
        loom_value_id_t value_id = operand_span.values[j];
        loom_type_t type = loom_verify_value_type(state, value_id);
        if (!loom_type_satisfies_constraint(type, constraint)) {
          uint8_t operand_ref = LOOM_FIELD_REF(LOOM_FIELD_OPERAND, i);
          uint16_t element_offset = j;
          char operand_name_buffer[64];
          iree_string_view_t operand_name =
              operand_is_variadic
                  ? loom_verify_indexed_field_name(
                        vtable, operand_ref, element_offset,
                        operand_name_buffer, sizeof(operand_name_buffer))
                  : loom_verify_field_name(vtable, operand_ref,
                                           operand_name_buffer,
                                           sizeof(operand_name_buffer));
          loom_diagnostic_param_t operand_param =
              operand_is_variadic
                  ? loom_verify_param_string_for_indexed_field(
                        operand_name, operand_ref, element_offset)
                  : loom_verify_param_string_for_field(operand_name,
                                                       operand_ref);
          loom_diagnostic_param_t params[] = {
              operand_param,
              loom_param_type(type),
              loom_param_string(iree_make_cstring_view(
                  loom_type_constraint_name(constraint))),
          };
          loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_003, params,
                                      IREE_ARRAYSIZE(params));
        }
      }
    }
  }

  // Check result type constraints.
  if (vtable->result_descriptors) {
    bool has_variadic =
        (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
    uint8_t descriptor_count =
        vtable->fixed_result_count + (has_variadic ? 1 : 0);
    for (uint8_t i = 0; i < descriptor_count && i < op->result_count; ++i) {
      loom_type_constraint_t constraint =
          (loom_type_constraint_t)vtable->result_descriptors[i].type_constraint;
      if (constraint == LOOM_TYPE_CONSTRAINT_ANY) continue;

      uint16_t start = i;
      uint16_t end = (has_variadic && i == vtable->fixed_result_count)
                         ? op->result_count
                         : (uint16_t)(i + 1);
      for (uint16_t j = start; j < end; ++j) {
        loom_value_id_t value_id = loom_op_const_results(op)[j];
        loom_type_t type = loom_verify_value_type(state, value_id);
        if (!loom_type_satisfies_constraint(type, constraint)) {
          uint8_t result_ref = LOOM_FIELD_REF(LOOM_FIELD_RESULT, i);
          uint16_t element_offset = (uint16_t)(j - i);
          bool result_is_variadic =
              has_variadic && i == vtable->fixed_result_count;
          char result_name_buffer[64];
          iree_string_view_t result_name =
              result_is_variadic
                  ? loom_verify_indexed_field_name(
                        vtable, result_ref, element_offset, result_name_buffer,
                        sizeof(result_name_buffer))
                  : loom_verify_field_name(vtable, result_ref,
                                           result_name_buffer,
                                           sizeof(result_name_buffer));
          loom_diagnostic_param_t result_param =
              result_is_variadic
                  ? loom_verify_param_string_for_indexed_field(
                        result_name, result_ref, element_offset)
                  : loom_verify_param_string_for_field(result_name, result_ref);
          loom_diagnostic_param_t params[] = {
              result_param,
              loom_param_type(type),
              loom_param_string(iree_make_cstring_view(
                  loom_type_constraint_name(constraint))),
          };
          loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_004, params,
                                      IREE_ARRAYSIZE(params));
        }
      }
    }
  }

  // Check attribute kinds and enum value ranges.
  if (vtable->attr_descriptors) {
    const loom_attribute_t* attrs = loom_op_attrs(op);
    for (uint8_t i = 0; i < vtable->attribute_count && i < op->attribute_count;
         ++i) {
      const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
      bool optional = (descriptor->flags & LOOM_ATTR_OPTIONAL) != 0;
      if (optional && loom_attr_is_absent(attrs[i])) continue;
      iree_string_view_t attr_name = loom_bstring_view(descriptor->name);
      if (!loom_verify_attr_kind_matches_descriptor(attrs[i], descriptor)) {
        loom_diagnostic_param_t params[] = {
            loom_verify_param_string_for_diagnostic_field(
                attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
            loom_param_u32(attrs[i].kind),
            loom_param_u32(descriptor->attr_kind),
        };
        loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_005, params,
                                    IREE_ARRAYSIZE(params));
      }
      if (attrs[i].kind == LOOM_ATTR_ENUM && descriptor->enum_case_count > 0 &&
          (descriptor->flags & LOOM_ATTR_OPEN_ENUM) == 0) {
        uint8_t case_index = (uint8_t)attrs[i].raw;
        if (case_index >= descriptor->enum_case_count) {
          loom_diagnostic_param_t params[] = {
              loom_verify_param_string_for_diagnostic_field(
                  attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
              loom_param_u32(case_index),
              loom_param_u32(descriptor->enum_case_count),
          };
          loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_010, params,
                                      IREE_ARRAYSIZE(params));
        }
      }
      loom_verify_predicate_list_attr(state, op, attr_name, i, attrs[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// Operand dictionary representation checks
//===----------------------------------------------------------------------===//

static iree_string_view_t loom_verify_attr_descriptor_name(
    const loom_op_vtable_t* vtable, uint16_t attr_index) {
  if (!vtable->attr_descriptors || attr_index >= vtable->attribute_count) {
    return IREE_SV("operand dictionary names");
  }
  return loom_bstring_view(vtable->attr_descriptors[attr_index].name);
}

static void loom_verify_emit_operand_dict_count_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint16_t attr_index, uint16_t names_count,
    uint16_t operand_count) {
  iree_string_view_t attr_name =
      loom_verify_attr_descriptor_name(vtable, attr_index);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_diagnostic_field(
          attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index),
      loom_param_u32(names_count),
      loom_param_string(IREE_SV("operand dictionary operands")),
      loom_param_u32(operand_count),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_013, params,
                              IREE_ARRAYSIZE(params));
}

static void loom_verify_emit_operand_dict_attr_violation(
    loom_verify_state_t* state, const loom_op_t* op,
    iree_string_view_t attr_name, uint16_t attr_index, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_diagnostic_field(
          attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_014, params,
                              IREE_ARRAYSIZE(params));
}

void loom_verify_operand_dicts(loom_verify_state_t* state, const loom_op_t* op,
                               const loom_op_vtable_t* vtable) {
  if (!vtable->format_elements) return;
  for (uint16_t element_index = 0; element_index < vtable->format_element_count;
       ++element_index) {
    const loom_format_element_t* element =
        &vtable->format_elements[element_index];
    if (element->kind != LOOM_FORMAT_KIND_OPERAND_DICT) continue;

    loom_value_slice_t operand_span =
        loom_op_operand_field_span(vtable, op, element->field_index);
    uint16_t attr_index = element->data;
    if (attr_index >= op->attribute_count) {
      continue;
    }
    uint16_t operand_count = operand_span.count;
    loom_attribute_t names_attr = loom_op_attrs(op)[attr_index];
    if (loom_attr_is_absent(names_attr)) {
      if (operand_count != 0) {
        loom_verify_emit_operand_dict_count_mismatch(
            state, op, vtable, attr_index, 0, operand_count);
      }
      continue;
    }
    if (names_attr.kind != LOOM_ATTR_DICT) continue;
    if (names_attr.count != operand_count) {
      loom_verify_emit_operand_dict_count_mismatch(
          state, op, vtable, attr_index, names_attr.count, operand_count);
      continue;
    }
    if (names_attr.count == 0) continue;

    iree_string_view_t attr_name =
        loom_verify_attr_descriptor_name(vtable, attr_index);
    if (!names_attr.dict_entries) {
      loom_verify_emit_operand_dict_attr_violation(
          state, op, attr_name, attr_index, names_attr.count,
          IREE_SV("non-null dictionary entries"));
      continue;
    }

    for (uint16_t i = 0; i < names_attr.count; ++i) {
      const loom_named_attr_t* entry = &names_attr.dict_entries[i];
      if (entry->name_id == LOOM_STRING_ID_INVALID ||
          entry->name_id >= state->module->strings.count) {
        loom_verify_emit_operand_dict_attr_violation(
            state, op, attr_name, attr_index, entry->name_id,
            IREE_SV("interned key string id"));
        continue;
      }
      iree_string_view_t key_name =
          state->module->strings.entries[entry->name_id];
      if (i > 0) {
        const loom_named_attr_t* previous_entry =
            &names_attr.dict_entries[i - 1];
        if (previous_entry->name_id < state->module->strings.count) {
          iree_string_view_t previous_key_name =
              state->module->strings.entries[previous_entry->name_id];
          if (iree_string_view_compare(previous_key_name, key_name) >= 0) {
            loom_verify_emit_operand_dict_attr_violation(
                state, op, attr_name, attr_index, i,
                IREE_SV("canonical sorted unique keys"));
          }
        }
      }
      if (entry->value.kind != LOOM_ATTR_I64) {
        loom_diagnostic_param_t params[] = {
            loom_verify_param_string_for_diagnostic_field(
                key_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index),
            loom_param_u32(entry->value.kind),
            loom_param_u32(LOOM_ATTR_I64),
        };
        loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_005, params,
                                    IREE_ARRAYSIZE(params));
        continue;
      }
      int64_t ordinal = entry->value.i64;
      if (ordinal < 0 || ordinal >= operand_count) {
        loom_verify_emit_operand_dict_attr_violation(
            state, op, key_name, attr_index, ordinal,
            IREE_SV("operand ordinal in range"));
        continue;
      }
      for (uint16_t j = 0; j < i; ++j) {
        const loom_named_attr_t* previous_entry = &names_attr.dict_entries[j];
        if (previous_entry->value.kind == LOOM_ATTR_I64 &&
            previous_entry->value.i64 == ordinal) {
          loom_verify_emit_operand_dict_attr_violation(
              state, op, key_name, attr_index, ordinal,
              IREE_SV("unique operand ordinal"));
          break;
        }
      }
    }
  }
}

typedef enum loom_verify_type_malformation_e {
  LOOM_VERIFY_TYPE_MALFORMATION_NONE = 0,
  LOOM_VERIFY_TYPE_MALFORMATION_TYPE_KIND_OUT_OF_RANGE = 1,
  LOOM_VERIFY_TYPE_MALFORMATION_ENCODING_ROLE_OUT_OF_RANGE = 2,
  LOOM_VERIFY_TYPE_MALFORMATION_VECTOR_RANK_ZERO = 3,
  LOOM_VERIFY_TYPE_MALFORMATION_VECTOR_ENCODING_ATTACHMENT = 4,
} loom_verify_type_malformation_t;

static iree_string_view_t loom_verify_type_malformation_code(
    loom_verify_type_malformation_t malformation) {
  switch (malformation) {
    case LOOM_VERIFY_TYPE_MALFORMATION_TYPE_KIND_OUT_OF_RANGE:
      return IREE_SV("type_kind_out_of_range");
    case LOOM_VERIFY_TYPE_MALFORMATION_ENCODING_ROLE_OUT_OF_RANGE:
      return IREE_SV("encoding_role_out_of_range");
    case LOOM_VERIFY_TYPE_MALFORMATION_VECTOR_RANK_ZERO:
      return IREE_SV("vector_rank_zero");
    case LOOM_VERIFY_TYPE_MALFORMATION_VECTOR_ENCODING_ATTACHMENT:
      return IREE_SV("vector_encoding_attachment");
    case LOOM_VERIFY_TYPE_MALFORMATION_NONE:
    default:
      return IREE_SV("unknown");
  }
}

static loom_verify_type_malformation_t
loom_verify_type_well_formed_malformation(loom_type_t type) {
  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) {
    return LOOM_VERIFY_TYPE_MALFORMATION_TYPE_KIND_OUT_OF_RANGE;
  }
  switch (kind) {
    case LOOM_TYPE_ENCODING:
      if (!loom_encoding_role_is_valid(loom_type_encoding_role(type))) {
        return LOOM_VERIFY_TYPE_MALFORMATION_ENCODING_ROLE_OUT_OF_RANGE;
      }
      break;
    case LOOM_TYPE_VECTOR:
      if (loom_type_rank(type) == 0) {
        return LOOM_VERIFY_TYPE_MALFORMATION_VECTOR_RANK_ZERO;
      }
      if (type.encoding_id != 0 || type.encoding_flags != 0) {
        return LOOM_VERIFY_TYPE_MALFORMATION_VECTOR_ENCODING_ATTACHMENT;
      }
      break;
    default:
      break;
  }
  return LOOM_VERIFY_TYPE_MALFORMATION_NONE;
}

static void loom_verify_emit_type_well_formed_diagnostic(
    loom_verify_state_t* state, const loom_op_t* op, loom_type_t type,
    iree_string_view_t field_name, loom_diagnostic_field_ref_t field_ref,
    loom_verify_type_malformation_t malformation) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(field_name), field_ref),
      loom_param_type(type),
      loom_param_string(loom_verify_type_malformation_code(malformation)),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_010, params,
                              IREE_ARRAYSIZE(params));
}

static void loom_verify_value_type_well_formed(
    loom_verify_state_t* state, const loom_op_t* op, loom_value_id_t value_id,
    const loom_op_vtable_t* vtable, uint8_t category, uint16_t value_index,
    loom_diagnostic_field_ref_t field_ref) {
  if (value_id == LOOM_VALUE_ID_INVALID) return;
  if (value_id >= state->module->values.count) return;
  loom_type_t type = loom_module_value_type(state->module, value_id);
  loom_verify_type_malformation_t malformation =
      loom_verify_type_well_formed_malformation(type);
  if (malformation == LOOM_VERIFY_TYPE_MALFORMATION_NONE) return;
  char name_buffer[64];
  iree_string_view_t field_name = loom_verify_value_field_name(
      vtable, op, category, value_index, name_buffer, sizeof(name_buffer));
  loom_verify_emit_type_well_formed_diagnostic(state, op, type, field_name,
                                               field_ref, malformation);
}

void loom_verify_op_type_well_formedness(loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_verify_value_type_well_formed(
        state, op, operands[i], vtable, LOOM_FIELD_OPERAND, i,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_verify_value_type_well_formed(
        state, op, results[i], vtable, LOOM_FIELD_RESULT, i,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, i));
  }
}

void loom_verify_block_arg_type_well_formedness(loom_verify_state_t* state,
                                                const loom_block_t* block) {
  char name_buffer[32];
  for (uint16_t a = 0; a < block->arg_count; ++a) {
    loom_value_id_t arg_id = loom_block_arg_id(block, a);
    if (arg_id == LOOM_VALUE_ID_INVALID) continue;
    if (arg_id >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, arg_id);
    loom_verify_type_malformation_t malformation =
        loom_verify_type_well_formed_malformation(type);
    if (malformation == LOOM_VERIFY_TYPE_MALFORMATION_NONE) continue;
    iree_snprintf(name_buffer, sizeof(name_buffer), "block arg %u", a);
    loom_verify_emit_type_well_formed_diagnostic(
        state, NULL, type, iree_make_cstring_view(name_buffer),
        loom_diagnostic_field_ref_none(), malformation);
  }
}

//===----------------------------------------------------------------------===//
// SSA encoding reference validation
//===----------------------------------------------------------------------===//

static bool loom_verify_op_result_contains_value(const loom_op_t* op,
                                                 loom_value_id_t value_id) {
  if (!op) return false;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == value_id) return true;
  }
  return false;
}

static bool loom_verify_value_is_named_placeholder(
    const loom_verify_state_t* state, loom_value_id_t value_id) {
  if (value_id >= state->module->values.count) return false;
  const loom_value_t* value = &state->module->values.entries[value_id];
  if (loom_value_is_block_arg(value)) return false;
  return value->name_id != LOOM_STRING_ID_INVALID &&
         loom_def_op(value->def) == NULL;
}

static bool loom_verify_op_allows_declaration_local_encoding_refs(
    const loom_op_vtable_t* vtable) {
  return vtable && vtable->symbol_def &&
         loom_symbol_definition_implements(vtable->symbol_def,
                                           LOOM_SYMBOL_INTERFACE_GLOBAL) &&
         iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
}

// Validates a single SSA encoding reference embedded in a value's type.
// If the type carries LOOM_ENCODING_FLAG_SSA, the encoding_id is a
// value_id that must be in range and have type LOOM_TYPE_ENCODING. It must
// also be defined in scope unless the reference is to a sibling result in the
// current op type annotation or to a declaration-local global placeholder.
static void loom_verify_encoding_ref(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_type_t type,
    iree_string_view_t field_name, loom_diagnostic_field_ref_t field_ref,
    bool allow_current_op_results, bool allow_declaration_placeholders) {
  if (!loom_type_has_ssa_encoding(type)) return;
  uint16_t encoding_value_id = loom_type_encoding_value_id(type);
  if (encoding_value_id >= state->module->values.count) {
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(loom_param_string(field_name), field_ref),
        loom_param_u32(encoding_value_id),
        loom_param_u32((uint32_t)state->module->values.count),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_ENCODING_003, params,
                                IREE_ARRAYSIZE(params));
    return;
  }
  if (!loom_bitset_test(state->defined_bits, state->defined_bits_length,
                        encoding_value_id)) {
    bool allowed_current_result =
        allow_current_op_results &&
        loom_verify_op_result_contains_value(op, encoding_value_id);
    bool allowed_declaration_placeholder =
        allow_declaration_placeholders &&
        loom_verify_op_allows_declaration_local_encoding_refs(vtable) &&
        loom_verify_value_is_named_placeholder(state, encoding_value_id);
    if (!allowed_current_result && !allowed_declaration_placeholder) {
      iree_string_view_t value_name =
          loom_verify_value_name(state, encoding_value_id);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(loom_param_string(field_name), field_ref),
          loom_param_string(value_name),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_ENCODING_004, params,
                                  IREE_ARRAYSIZE(params));
      return;
    }
  }
  loom_type_t encoding_type =
      loom_module_value_type(state->module, encoding_value_id);
  if (!loom_type_is_encoding(encoding_type)) {
    iree_string_view_t value_name =
        loom_verify_value_name(state, encoding_value_id);
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(loom_param_string(field_name), field_ref),
        loom_param_string(value_name),
        loom_param_type(encoding_type),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_ENCODING_005, params,
                                IREE_ARRAYSIZE(params));
  }
}

// Checks SSA encoding references in all operand and result types of an op.
void loom_verify_encoding_refs(loom_verify_state_t* state, const loom_op_t* op,
                               const loom_op_vtable_t* vtable) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID) continue;
    if (operands[i] >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, operands[i]);
    if (!loom_type_has_ssa_encoding(type)) continue;
    char name_buffer[64];
    iree_string_view_t name = loom_verify_value_field_name(
        vtable, op, LOOM_FIELD_OPERAND, i, name_buffer, sizeof(name_buffer));
    loom_verify_encoding_ref(
        state, op, vtable, type, name,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i),
        /*allow_current_op_results=*/false,
        /*allow_declaration_placeholders=*/false);
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) continue;
    if (results[i] >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, results[i]);
    if (!loom_type_has_ssa_encoding(type)) continue;
    char name_buffer[64];
    iree_string_view_t name = loom_verify_value_field_name(
        vtable, op, LOOM_FIELD_RESULT, i, name_buffer, sizeof(name_buffer));
    loom_verify_encoding_ref(
        state, op, vtable, type, name,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, i),
        /*allow_current_op_results=*/true,
        /*allow_declaration_placeholders=*/true);
  }
}

// Checks SSA encoding references in block argument types. Called when
// entering a block, after block args are defined but before ops are
// verified. The encoding value must already be visible (from an
// enclosing scope or earlier in this scope).
void loom_verify_block_arg_encoding_refs(loom_verify_state_t* state,
                                         const loom_block_t* block) {
  char name_buffer[32];
  for (uint16_t a = 0; a < block->arg_count; ++a) {
    loom_value_id_t arg_id = loom_block_arg_id(block, a);
    if (arg_id == LOOM_VALUE_ID_INVALID) continue;
    if (arg_id >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, arg_id);
    if (!loom_type_has_ssa_encoding(type)) continue;
    iree_snprintf(name_buffer, sizeof(name_buffer), "block arg %u", a);
    loom_verify_encoding_ref(state, NULL, NULL, type,
                             iree_make_cstring_view(name_buffer),
                             loom_diagnostic_field_ref_none(),
                             /*allow_current_op_results=*/false,
                             /*allow_declaration_placeholders=*/false);
  }
}

void loom_verify_symbol_references(loom_verify_state_t* state,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable) {
  if (!vtable->attr_descriptors) return;
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < vtable->attribute_count && i < op->attribute_count;
       ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (descriptor->attr_kind != LOOM_ATTR_SYMBOL) continue;
    if ((descriptor->flags & LOOM_ATTR_OPTIONAL) &&
        loom_attr_is_absent(attrs[i])) {
      continue;
    }
    loom_symbol_ref_t ref = loom_attr_as_symbol(attrs[i]);
    if (!loom_symbol_ref_is_valid(ref)) continue;

    if (ref.module_id != 0) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(ref.module_id),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i)),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_SYMBOL_004, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }

    if (ref.symbol_id >= state->module->symbols.count) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(ref.symbol_id),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i)),
          loom_param_u32((uint32_t)state->module->symbols.count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_SYMBOL_001, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }

    const loom_symbol_t* symbol =
        &state->module->symbols.entries[ref.symbol_id];
    if (symbol->definition == NULL || symbol->defining_op == NULL) {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_diagnostic_field(
              loom_verify_symbol_name(state, ref),
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_SYMBOL_002, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }

    if (descriptor->symbol_ref &&
        !loom_symbol_implements(symbol, descriptor->symbol_ref->interfaces)) {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_diagnostic_field(
              loom_verify_symbol_name(state, ref),
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
          loom_param_string(loom_verify_symbol_definition_name(symbol)),
          loom_param_string(
              loom_symbol_reference_descriptor_name(descriptor->symbol_ref)),
      };
      loom_diagnostic_related_op_t related_ops[] = {{
          .label = IREE_SV("defined here"),
          .op = symbol->defining_op,
      }};
      loom_diagnostic_emission_t emission = {
          .op = op,
          .error = LOOM_ERR_SYMBOL_003,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
          .related_ops = related_ops,
          .related_op_count = IREE_ARRAYSIZE(related_ops),
      };
      loom_verify_emit_diagnostic(state, &emission);
    }
  }
}

//===----------------------------------------------------------------------===//
// Region structure checks
//===----------------------------------------------------------------------===//

static const loom_op_t* loom_verify_block_last_live_op(
    const loom_block_t* block) {
  return block->last_op;
}

static bool loom_verify_op_is_terminator(loom_verify_state_t* state,
                                         const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_verify_lookup_vtable(state, op->kind);
  return vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR);
}

static bool loom_verify_region_terminator_matches(
    const loom_region_descriptor_t* region_descriptor,
    const loom_op_t* terminator) {
  if (!terminator) return false;
  if (region_descriptor->terminator == LOOM_OP_KIND_UNKNOWN) return true;
  return terminator->kind == region_descriptor->terminator;
}

static bool loom_verify_region_has_cfg_successors(const loom_region_t* region) {
  if (!region) return false;
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return true;
  }
  for (uint16_t b = 0; b < region->block_count; ++b) {
    const loom_block_t* block = loom_region_const_block(region, b);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->successor_count > 0) return true;
    }
  }
  return false;
}

bool loom_verify_region_entry_yield(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t region_index,
    uint16_t* out_yield_count, const loom_value_id_t** out_yield_operands) {
  *out_yield_count = 0;
  if (out_yield_operands) {
    *out_yield_operands = NULL;
  }
  if (region_index >= op->region_count) {
    return false;
  }
  const loom_region_descriptor_t* region_descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  if (!region_descriptor) return false;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return false;

  const loom_block_t* entry = loom_region_const_entry_block(region);
  const loom_op_t* terminator = loom_verify_block_last_live_op(entry);
  if (terminator && loom_verify_op_is_terminator(state, terminator) &&
      loom_verify_region_terminator_matches(region_descriptor, terminator)) {
    *out_yield_count = terminator->operand_count;
    if (out_yield_operands) {
      *out_yield_operands = loom_op_const_operands(terminator);
    }
    return true;
  }
  return false;
}

static iree_string_view_t loom_verify_op_kind_name(loom_verify_state_t* state,
                                                   loom_op_kind_t kind) {
  const loom_op_vtable_t* vtable = loom_verify_lookup_vtable(state, kind);
  if (!vtable) return IREE_SV("<unknown op>");
  return loom_op_vtable_name(vtable);
}

static void loom_verify_region_terminator_kind(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t region_index,
    bool region_uses_cfg_successors,
    const loom_region_descriptor_t* region_descriptor,
    const loom_op_t* terminator_op) {
  if (!terminator_op || region_uses_cfg_successors ||
      loom_verify_region_terminator_matches(region_descriptor, terminator_op)) {
    return;
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_u32(region_index),
      loom_param_string(
          loom_verify_op_kind_name(state, region_descriptor->terminator)),
      loom_param_string(loom_verify_op_kind_name(state, terminator_op->kind)),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_018, params,
                              IREE_ARRAYSIZE(params));
}

void loom_verify_region_structure(loom_verify_state_t* state,
                                  const loom_op_t* op,
                                  const loom_op_vtable_t* vtable) {
  if (!vtable->region_descriptors) return;
  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    const loom_region_descriptor_t* region_descriptor =
        loom_op_vtable_region_descriptor(vtable, i);
    if (!region_descriptor) return;
    loom_region_t* region = regions[i];
    if (!region) {
      if (iree_any_bit_set(region_descriptor->flags, LOOM_REGION_OPTIONAL)) {
        continue;
      }
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->region_count),
          loom_param_u32(vtable->region_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_004, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }
    if ((region_descriptor->flags & LOOM_REGION_SINGLE_BLOCK) &&
        region->block_count != 1) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(i),
          loom_param_u32(region->block_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_006, params,
                                  IREE_ARRAYSIZE(params));
    }
    bool region_uses_cfg_successors =
        loom_verify_region_has_cfg_successors(region);
    for (uint16_t b = 0; b < region->block_count; ++b) {
      const loom_block_t* block = loom_region_const_block(region, b);
      const loom_op_t* terminator_op = NULL;
      const loom_op_t* current_op = NULL;
      loom_block_for_each_op(block, current_op) {
        if (terminator_op) {
          const loom_op_vtable_t* current_vtable =
              loom_verify_lookup_vtable(state, current_op->kind);
          iree_string_view_t current_name =
              current_vtable ? loom_op_vtable_name(current_vtable)
                             : IREE_SV("<unknown op>");
          loom_diagnostic_param_t params[] = {
              loom_param_string(current_name),
          };
          loom_diagnostic_related_op_t related_ops[] = {{
              .label = IREE_SV("terminator here"),
              .op = terminator_op,
          }};
          loom_diagnostic_emission_t emission = {
              .op = current_op,
              .error = LOOM_ERR_STRUCTURE_012,
              .params = params,
              .param_count = IREE_ARRAYSIZE(params),
              .related_ops = related_ops,
              .related_op_count = IREE_ARRAYSIZE(related_ops),
          };
          loom_verify_emit_diagnostic(state, &emission);
          continue;
        }
        if (loom_verify_op_is_terminator(state, current_op)) {
          terminator_op = current_op;
        }
      }
      if (!terminator_op && !region_uses_cfg_successors) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(op_name),
            loom_param_u32(i),
        };
        loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_005, params,
                                    IREE_ARRAYSIZE(params));
      }
      loom_verify_region_terminator_kind(state, op, vtable, i,
                                         region_uses_cfg_successors,
                                         region_descriptor, terminator_op);
    }
  }
}
