// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_value_types.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/value_replacement.h"
#include "loom/ops/op_defs.h"
#include "loom/verify/verify_diagnostics.h"

static bool loom_verify_type_representation_equal(loom_type_t expected,
                                                  loom_type_t actual) {
  return expected.header == actual.header &&
         expected.encoding_id == actual.encoding_id &&
         expected.encoding_flags == actual.encoding_flags &&
         expected.dims[0] == actual.dims[0] &&
         expected.dims[1] == actual.dims[1];
}

static bool loom_verify_bounded_type_equal_after_remap(
    const loom_module_t* module, const loom_type_value_remap_t* remap,
    loom_type_t expected, loom_type_t actual) {
  return loom_type_may_reference_values(expected)
             ? loom_type_equal_after_value_remap(module, expected, actual,
                                                 remap)
             : loom_type_equal(expected, actual);
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static void
loom_verify_func_like_emit_exit_count_mismatch(loom_verify_state_t* state,
                                               const loom_op_t* exit_op,
                                               uint16_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_u32(exit_op->operand_count),
      loom_param_u32(expected_count),
  };
  loom_verify_emit_structured(state, exit_op, LOOM_ERR_STRUCTURE_008, params,
                              IREE_ARRAYSIZE(params));
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static void
loom_verify_func_like_emit_exit_type_mismatch(loom_verify_state_t* state,
                                              const loom_op_t* exit_op,
                                              loom_type_t actual,
                                              loom_type_t expected) {
  loom_diagnostic_param_t params[] = {loom_param_type(actual),
                                      loom_param_type(expected)};
  loom_verify_emit_structured(state, exit_op, LOOM_ERR_TYPE_009, params,
                              IREE_ARRAYSIZE(params));
}

// Recursive containers can revisit shared canonical children through many
// roots. Keep their traversal state out of the common bounded-type frame.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_verify_func_like_exit_types_with_lookup(
    loom_verify_state_t* state, const loom_op_t* exit_op,
    const loom_type_value_remap_t* remap, uint16_t start) {
  loom_type_remap_lookup_t lookup;
  loom_type_remap_lookup_initialize(state->module, remap, &lookup);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = start; i < remap->count && iree_status_is_ok(status); ++i) {
    const loom_type_t actual =
        loom_module_value_type(state->module, remap->target_values[i]);
    const loom_type_t expected =
        loom_module_value_type(state->module, remap->source_values[i]);
    bool equal = loom_verify_type_representation_equal(expected, actual);
    if (!equal && loom_type_remap_requires_lookup(expected)) {
      status = loom_type_remap_lookup_equal(&lookup, expected, actual, &equal);
    } else if (!equal) {
      equal = loom_verify_bounded_type_equal_after_remap(state->module, remap,
                                                         expected, actual);
    }
    if (iree_status_is_ok(status) && !equal) {
      loom_verify_func_like_emit_exit_type_mismatch(state, exit_op, actual,
                                                    expected);
      if (loom_verify_at_error_limit(state)) {
        break;
      }
    }
  }
  loom_type_remap_lookup_deinitialize(&lookup);
  return status;
}

iree_status_t loom_verify_func_like_exit(loom_verify_state_t* state,
                                         const loom_op_t* func_op,
                                         const loom_op_t* exit_op) {
  if (exit_op->operand_count != func_op->result_count) {
    loom_verify_func_like_emit_exit_count_mismatch(state, exit_op,
                                                   func_op->result_count);
    return iree_ok_status();
  }
  if (func_op->result_count == 0) {
    return iree_ok_status();
  }

  const loom_value_id_t* expected_values = loom_op_const_results(func_op);
  const loom_value_id_t* actual_values = loom_op_const_operands(exit_op);
  const loom_type_value_remap_t remap = {
      .source_values = expected_values,
      .target_values = actual_values,
      .count = func_op->result_count,
      .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
  };
  for (uint16_t i = 0; i < func_op->result_count; ++i) {
    const loom_type_t actual =
        loom_module_value_type(state->module, actual_values[i]);
    const loom_type_t expected =
        loom_module_value_type(state->module, expected_values[i]);
    // Identical immutable representations need no structural or mapped walk.
    if (loom_verify_type_representation_equal(expected, actual)) {
      continue;
    }
    if (loom_type_remap_requires_lookup(expected)) {
      return loom_verify_func_like_exit_types_with_lookup(state, exit_op,
                                                          &remap, i);
    }
    if (loom_verify_bounded_type_equal_after_remap(state->module, &remap,
                                                   expected, actual)) {
      continue;
    }
    loom_verify_func_like_emit_exit_type_mismatch(state, exit_op, actual,
                                                  expected);
    if (loom_verify_at_error_limit(state)) {
      break;
    }
  }
  return iree_ok_status();
}

// Loop results describe one recurring type scheme. Unique result definitions
// map to each entry's definitions; repeated initial operands never choose which
// carried extent or encoding a body value depends on.
void loom_verify_loop_entry_types(loom_verify_state_t* state,
                                  const loom_op_t* op,
                                  const loom_loop_like_vtable_t* loop) {
  const uint8_t regions[] = {loop->body_region_index,
                             loop->condition_region_index};
  for (uint8_t r = 0; r < IREE_ARRAYSIZE(regions); ++r) {
    if (regions[r] == LOOM_REGION_INDEX_NONE) {
      continue;
    }
    const loom_block_t* entry =
        loom_region_const_entry_block(loom_op_regions(op)[regions[r]]);
    const bool has_induction_variable =
        regions[r] == loop->body_region_index &&
        loop->iv_block_arg_index != LOOM_BLOCK_ARG_INDEX_NONE;
    const uint16_t offset = has_induction_variable ? 1 : 0;
    const uint32_t expected_count = (uint32_t)op->result_count + offset;
    if (entry->arg_count != expected_count) {
      loom_diagnostic_param_t params[] = {loom_param_u32(entry->arg_count),
                                          loom_param_u32(expected_count)};
      loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_007, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }
    if (offset) {
      const loom_loop_like_t reference = {(loom_op_t*)op, loop};
      const loom_type_t actual =
          loom_module_value_type(state->module, entry->arg_ids[0]);
      const loom_type_t expected = loom_module_value_type(
          state->module, loom_loop_like_lower_bound(reference));
      if (!loom_type_equal(actual, expected)) {
        loom_diagnostic_param_t params[] = {loom_param_u32(0),
                                            loom_param_type(actual),
                                            loom_param_type(expected)};
        loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_013, params,
                                    IREE_ARRAYSIZE(params));
      }
    }
    const loom_type_value_remap_t remap = {
        .source_values = loom_op_const_results(op),
        .target_values = offset ? entry->arg_ids + offset : entry->arg_ids,
        .count = op->result_count,
        .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
    };
    for (uint16_t i = 0; i < op->result_count; ++i) {
      const loom_type_t actual =
          loom_module_value_type(state->module, remap.target_values[i]);
      const loom_type_t expected =
          loom_module_value_type(state->module, remap.source_values[i]);
      if (loom_type_equal_after_value_remap(state->module, expected, actual,
                                            &remap)) {
        continue;
      }
      loom_diagnostic_param_t params[] = {loom_param_u32(i + offset),
                                          loom_param_type(actual),
                                          loom_param_type(expected)};
      loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_013, params,
                                  IREE_ARRAYSIZE(params));
    }
  }
}

//===----------------------------------------------------------------------===//
// SSA references carried by value types
//===----------------------------------------------------------------------===//

// Field names are diagnostic-only work. Keep formatting out of the valid path,
// including variadic results and block arguments whose names need an ordinal.
static loom_diagnostic_param_t loom_verify_type_ref_field_param(
    const loom_op_t* op, const loom_op_vtable_t* vtable, uint16_t field_index,
    bool is_result, char* buffer, iree_host_size_t buffer_size) {
  if (!vtable) {
    iree_snprintf(buffer, buffer_size, "block arg %u", field_index);
    return loom_param_string(iree_make_cstring_view(buffer));
  }
  const uint8_t category = is_result ? LOOM_FIELD_RESULT : LOOM_FIELD_OPERAND;
  const loom_diagnostic_field_kind_t kind =
      is_result ? LOOM_DIAGNOSTIC_FIELD_RESULT : LOOM_DIAGNOSTIC_FIELD_OPERAND;
  const iree_string_view_t field_name = loom_verify_value_field_name(
      vtable, op, category, field_index, buffer, buffer_size);
  return loom_param_with_field_ref(
      loom_param_string(field_name),
      loom_diagnostic_field_ref(kind, field_index));
}

static bool loom_verify_op_allows_declaration_local_refs(
    const loom_op_vtable_t* vtable) {
  return vtable->symbol_def &&
         loom_symbol_definition_implements(vtable->symbol_def,
                                           LOOM_SYMBOL_INTERFACE_GLOBAL) &&
         iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
}

// Definition-site references may name co-results or global declaration-local
// placeholders. The constructor retains result ownership, so co-reference
// checks never search the result array for each referenced value.
static bool loom_verify_definition_ref_is_visible(
    const loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_value_id_t value_id,
    bool allows_local_definitions) {
  if (loom_verify_value_is_visible(state, value_id)) {
    return true;
  }
  if (!allows_local_definitions) {
    return false;
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op == op ||
         (!defining_op && value->name_id != LOOM_STRING_ID_INVALID &&
          loom_verify_op_allows_declaration_local_refs(vtable));
}

// Validates a single SSA encoding reference embedded in a value's type.
// If the type carries LOOM_ENCODING_FLAG_SSA, the encoding_id is a
// value_id that must be in range and have type LOOM_TYPE_ENCODING. It must
// also be defined in scope unless the reference is to a sibling result in the
// current op type annotation or to a declaration-local global placeholder.
static void loom_verify_encoding_ref(loom_verify_state_t* state,
                                     const loom_op_t* op,
                                     const loom_op_vtable_t* vtable,
                                     loom_type_t type, uint16_t field_index,
                                     bool is_result) {
  if (!loom_type_has_ssa_encoding(type)) {
    return;
  }
  uint16_t encoding_value_id = loom_type_encoding_value_id(type);
  if (encoding_value_id >= state->module->values.count) {
    char name_buffer[64];
    loom_diagnostic_param_t params[] = {
        loom_verify_type_ref_field_param(op, vtable, field_index, is_result,
                                         name_buffer, sizeof(name_buffer)),
        loom_param_u32(encoding_value_id),
        loom_param_u32((uint32_t)state->module->values.count),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_ENCODING_003, params,
                                IREE_ARRAYSIZE(params));
    return;
  }
  if (!loom_verify_definition_ref_is_visible(state, op, vtable,
                                             encoding_value_id, is_result)) {
    iree_string_view_t value_name =
        loom_verify_value_name(state, encoding_value_id);
    char name_buffer[64];
    loom_diagnostic_param_t params[] = {
        loom_verify_type_ref_field_param(op, vtable, field_index, is_result,
                                         name_buffer, sizeof(name_buffer)),
        loom_param_string(value_name),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_ENCODING_004, params,
                                IREE_ARRAYSIZE(params));
    return;
  }
  loom_type_t encoding_type =
      loom_module_value_type(state->module, encoding_value_id);
  if (!loom_type_is_encoding(encoding_type)) {
    iree_string_view_t value_name =
        loom_verify_value_name(state, encoding_value_id);
    char name_buffer[64];
    loom_diagnostic_param_t params[] = {
        loom_verify_type_ref_field_param(op, vtable, field_index, is_result,
                                         name_buffer, sizeof(name_buffer)),
        loom_param_string(value_name),
        loom_param_type(encoding_type),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_ENCODING_005, params,
                                IREE_ARRAYSIZE(params));
  }
}

// The type-use producer retains all nested references. Verify each definition's
// outgoing edges once instead of traversing its type tree at every operand use.
static void loom_verify_defined_type_refs(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_value_id_t value_id, loom_type_t type,
    uint16_t field_index, bool is_result) {
  loom_verify_encoding_ref(state, op, vtable, type, field_index, is_result);
  const loom_value_id_t direct_encoding =
      loom_type_has_ssa_encoding(type) ? loom_type_encoding_value_id(type)
                                       : LOOM_VALUE_ID_INVALID;
  loom_type_use_iterator_t dependencies;
  loom_module_value_type_dependencies(state->module, value_id, &dependencies);
  for (loom_value_id_t referenced_id =
           loom_type_dependencies_next(&dependencies);
       referenced_id != LOOM_VALUE_ID_INVALID;
       referenced_id = loom_type_dependencies_next(&dependencies)) {
    if (referenced_id == direct_encoding ||
        loom_verify_definition_ref_is_visible(state, op, vtable, referenced_id,
                                              is_result)) {
      continue;
    }
    char name_buffer[64];
    loom_diagnostic_param_t params[] = {
        loom_verify_type_ref_field_param(op, vtable, field_index, is_result,
                                         name_buffer, sizeof(name_buffer)),
        loom_param_string(loom_verify_value_name(state, referenced_id)),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_016, params,
                                IREE_ARRAYSIZE(params));
    if (loom_verify_at_error_limit(state)) {
      return;
    }
  }
}

void loom_verify_value_type_refs(loom_verify_state_t* state,
                                 const loom_op_t* op,
                                 const loom_op_vtable_t* vtable) {
  const bool defines_arguments = loom_verify_has_func_signature_scope(vtable) &&
                                 loom_op_vtable_owns_operands(vtable);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID ||
        operands[i] >= state->module->values.count) {
      continue;
    }
    loom_type_t type = loom_module_value_type(state->module, operands[i]);
    if ((!loom_type_has_ssa_encoding(type) && !defines_arguments) ||
        !loom_type_may_reference_values(type)) {
      continue;
    }
    if (defines_arguments) {
      loom_verify_defined_type_refs(state, op, vtable, operands[i], type, i,
                                    /*is_result=*/false);
    } else {
      loom_verify_encoding_ref(state, op, vtable, type, i,
                               /*is_result=*/false);
    }
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        results[i] >= state->module->values.count) {
      continue;
    }
    loom_type_t type = loom_module_value_type(state->module, results[i]);
    if (!loom_type_may_reference_values(type)) {
      continue;
    }
    loom_verify_defined_type_refs(state, op, vtable, results[i], type, i,
                                  /*is_result=*/true);
  }
}

// Checks SSA references in block argument types after all arguments are defined
// but before operations are verified. Referenced values must already be visible
// in the current scope. Block arguments carry no source location; the owning
// operation anchors diagnostics at their containing scope.
void loom_verify_block_arg_type_refs(loom_verify_state_t* state,
                                     const loom_block_t* block,
                                     const loom_op_t* owner) {
  for (uint16_t a = 0; a < block->arg_count; ++a) {
    loom_value_id_t arg_id = loom_block_arg_id(block, a);
    if (arg_id == LOOM_VALUE_ID_INVALID ||
        arg_id >= state->module->values.count) {
      continue;
    }
    loom_type_t type = loom_module_value_type(state->module, arg_id);
    if (!loom_type_may_reference_values(type)) {
      continue;
    }
    loom_verify_defined_type_refs(state, owner, NULL, arg_id, type, a,
                                  /*is_result=*/false);
  }
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static void
loom_verify_emit_attribute_ref_not_visible(loom_verify_state_t* state,
                                           const loom_op_t* op,
                                           const loom_op_vtable_t* vtable,
                                           uint8_t attribute_index,
                                           loom_value_id_t value_id) {
  char name_buffer[32];
  iree_string_view_t field_name;
  if (vtable->attr_descriptors && attribute_index < vtable->attribute_count) {
    field_name =
        loom_bstring_view(vtable->attr_descriptors[attribute_index].name);
  } else {
    // Structural verification diagnoses undescribed slots separately.
    iree_snprintf(name_buffer, sizeof(name_buffer), "attribute %u",
                  attribute_index);
    field_name = iree_make_cstring_view(name_buffer);
  }
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_diagnostic_field(
          field_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attribute_index),
      loom_param_string(loom_verify_value_name(state, value_id)),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_017, params,
                              IREE_ARRAYSIZE(params));
}

// Attributes retain their nested type and predicate references at construction.
// Signature predicates may describe the declaration's own results or global
// shape placeholders; ordinary attributes require a dominating definition.
void loom_verify_attribute_value_refs(loom_verify_state_t* state,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable) {
  const bool allows_local_definitions =
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
  const uint32_t* attribute_owners = loom_op_attribute_owners(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (!attribute_owners[i]) {
      continue;
    }
    loom_type_use_iterator_t dependencies;
    loom_attribute_dependencies_begin(&state->module->type_uses, op, i,
                                      &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      if (loom_verify_definition_ref_is_visible(state, op, vtable, provider,
                                                allows_local_definitions)) {
        continue;
      }
      loom_verify_emit_attribute_ref_not_visible(state, op, vtable, i,
                                                 provider);
      if (loom_verify_at_error_limit(state)) {
        return;
      }
    }
  }
}
