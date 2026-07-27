// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_constraints.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/scalar_type.h"
#include "loom/target/registers.h"
#include "loom/verify/verify_diagnostics.h"
#include "loom/verify/verify_structure.h"

static bool loom_constraint_property_equals(
    loom_type_t a, loom_type_t b, loom_constraint_property_t property) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_TYPE:
      return memcmp(&a, &b, sizeof(loom_type_t)) == 0;
    case LOOM_PROPERTY_KIND:
      return loom_type_kind(a) == loom_type_kind(b);
    case LOOM_PROPERTY_ELEMENT_TYPE:
      return loom_type_element_type(a) == loom_type_element_type(b);
    case LOOM_PROPERTY_ENCODING:
      return loom_type_encoding_equals(a, b);
    case LOOM_PROPERTY_SHAPE:
      return loom_type_shape_equals(a, b);
    case LOOM_PROPERTY_RANK:
      return loom_type_rank(a) == loom_type_rank(b);
    case LOOM_PROPERTY_REGISTER_CLASS:
      return loom_type_is_register(a) && loom_type_is_register(b) &&
             loom_type_register_payload0(a) == loom_type_register_payload0(b) &&
             loom_low_register_type_class_id(a) ==
                 loom_low_register_type_class_id(b);
    case LOOM_PROPERTY_REGISTER_UNIT_COUNT:
      return loom_type_is_register(a) && loom_type_is_register(b) &&
             loom_low_register_type_unit_count(a) ==
                 loom_low_register_type_unit_count(b);
    default:
      return false;
  }
}

// Default error defs for PAIRWISE_EQ, indexed by property.
static const loom_error_def_t* loom_pairwise_eq_default_error(
    loom_constraint_property_t property) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_TYPE:
      return LOOM_ERR_TYPE_001;
    case LOOM_PROPERTY_KIND:
      return LOOM_ERR_TYPE_001;
    case LOOM_PROPERTY_ELEMENT_TYPE:
      return LOOM_ERR_TYPE_002;
    case LOOM_PROPERTY_ENCODING:
      return LOOM_ERR_ENCODING_001;
    case LOOM_PROPERTY_SHAPE:
      return LOOM_ERR_SHAPE_002;
    case LOOM_PROPERTY_RANK:
      return LOOM_ERR_SHAPE_001;
    case LOOM_PROPERTY_REGISTER_CLASS:
    case LOOM_PROPERTY_REGISTER_UNIT_COUNT:
      return LOOM_ERR_TYPE_001;
    default:
      return LOOM_ERR_TYPE_001;
  }
}

static const loom_error_def_t* loom_verify_constraint_error_or(
    const loom_constraint_t* constraint,
    const loom_error_def_t* default_error) {
  const loom_error_def_t* error =
      loom_error_def_lookup_ref((loom_error_ref_t)constraint->error_ref);
  return error ? error : default_error;
}

// Builds diagnostic params for a pairwise property mismatch.
// Different properties produce different param schemas:
//   TYPE/KIND: (name_a, type_a, name_b, type_b)
//   ELEMENT_TYPE: (name_a, element_type_a, name_b, element_type_b)
//   ENCODING: (name_a, name_b)
//   SHAPE: (name_a, name_b)
//   RANK: (name_a, rank_a, name_b, rank_b)
static void loom_verify_emit_pairwise_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint,
    loom_type_t type_a, loom_type_t type_b, uint8_t ref_a, uint8_t ref_b) {
  char name_a_buffer[32];
  char name_b_buffer[32];
  iree_string_view_t name_a = loom_verify_field_name(
      vtable, ref_a, name_a_buffer, sizeof(name_a_buffer));
  iree_string_view_t name_b = loom_verify_field_name(
      vtable, ref_b, name_b_buffer, sizeof(name_b_buffer));
  const loom_error_def_t* error = loom_verify_constraint_error_or(
      constraint, loom_pairwise_eq_default_error(constraint->property));

  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_TYPE: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_type(type_a),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_type(type_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    case LOOM_PROPERTY_REGISTER_CLASS:
    case LOOM_PROPERTY_REGISTER_UNIT_COUNT:
    case LOOM_PROPERTY_KIND: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_type(type_a),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_type(type_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    case LOOM_PROPERTY_ELEMENT_TYPE: {
      loom_type_t element_a = loom_type_scalar(loom_type_element_type(type_a));
      loom_type_t element_b = loom_type_scalar(loom_type_element_type(type_b));
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_type(element_a),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_type(element_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    case LOOM_PROPERTY_ENCODING:
    case LOOM_PROPERTY_SHAPE: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_verify_param_string_for_field(name_b, ref_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 2 ? error->param_count : 2);
      break;
    }
    case LOOM_PROPERTY_RANK: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_i64(loom_type_rank(type_a)),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_i64(loom_type_rank(type_b)),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    default:
      break;
  }
}

static void loom_verify_emit_indexed_pairwise_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_error_def_t* error,
    uint8_t ref_a, bool a_is_indexed, uint16_t a_element_index,
    loom_type_t type_a, uint8_t ref_b, uint16_t b_element_index,
    loom_type_t type_b) {
  char name_a_buffer[32];
  char name_b_buffer[32];
  iree_string_view_t name_a =
      a_is_indexed
          ? loom_verify_indexed_field_name(vtable, ref_a, a_element_index,
                                           name_a_buffer, sizeof(name_a_buffer))
          : loom_verify_field_name(vtable, ref_a, name_a_buffer,
                                   sizeof(name_a_buffer));
  iree_string_view_t name_b = loom_verify_indexed_field_name(
      vtable, ref_b, b_element_index, name_b_buffer, sizeof(name_b_buffer));
  loom_diagnostic_param_t params[] = {
      a_is_indexed ? loom_verify_param_string_for_indexed_field(name_a, ref_a,
                                                                a_element_index)
                   : loom_verify_param_string_for_field(name_a, ref_a),
      loom_param_type(type_a),
      loom_verify_param_string_for_indexed_field(name_b, ref_b,
                                                 b_element_index),
      loom_param_type(type_b),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 4 ? error->param_count : 4);
}

//===----------------------------------------------------------------------===//
// Constraint relation handlers
//===----------------------------------------------------------------------===//
//
// One handler per loom_constraint_relation_e value. All handlers share
// the same signature so they can be dispatched through the table at
// the bottom of this section. Handlers must validate their own
// constraint args (arg_count, field categories) and silently return on
// malformed inputs — structural verification runs first, so any
// malformed op has already been diagnosed by the time semantic
// constraints are evaluated.
//
// To add a new relation:
//   1. Add the LOOM_RELATION_* enum value in op_defs.h with a doc
//      comment describing the check.
//   2. Add the name string in loom_constraint_relation_name (op_defs.c).
//   3. Add a handler here following the same pattern.
//   4. Add the handler to kVerifyRelationFns below.
//   5. Add the corresponding Constraint constructor in dsl.py and the
//      mapping in c_tables.py CONSTRAINT_MAP.

// PAIRWISE_EQ: every element of every listed field has the same
// property as the first element of the first field. Variadic fields
// are walked elementwise. Args: 1+ value fields. Stops on the first
// mismatch to avoid cascading errors.
static void loom_verify_relation_pairwise_eq(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  loom_value_id_t first_id =
      loom_verify_resolve_value_field(op, vtable, constraint->args[0]);
  if (first_id == LOOM_VALUE_ID_INVALID) return;
  loom_type_t first_type = loom_verify_value_type(state, first_id);
  uint8_t first_ref = constraint->args[0];
  bool first_is_variadic =
      loom_verify_is_variadic_field(vtable, constraint->args[0]);

  // PAIRWISE_EQ stores the type-mismatch error as a compact ref on the
  // constraint (Python's SameType, SameElementType, etc.), so honor it when
  // present and fall back to the property's default otherwise.
  const loom_error_def_t* type_error = loom_verify_constraint_error_or(
      constraint, loom_pairwise_eq_default_error(constraint->property));

  // Check remaining elements within the first arg's variadic range.
  if (first_is_variadic) {
    uint16_t count = 0;
    const loom_value_id_t* values =
        loom_verify_resolve_variadic_field(op, vtable, first_ref, &count);
    for (uint16_t i = 1; i < count; ++i) {
      loom_type_t other_type = loom_verify_value_type(state, values[i]);
      if (loom_constraint_property_equals(first_type, other_type,
                                          constraint->property)) {
        continue;
      }
      loom_verify_emit_indexed_pairwise_mismatch(
          state, op, vtable, type_error, first_ref, /*a_is_indexed=*/true, 0,
          first_type, first_ref, i, other_type);
      return;
    }
  }

  // Check subsequent constraint args against the reference type.
  for (uint8_t i = 1; i < constraint->arg_count; ++i) {
    uint8_t arg_ref = constraint->args[i];
    if (!loom_verify_is_variadic_field(vtable, arg_ref)) {
      loom_value_id_t other_id =
          loom_verify_resolve_value_field(op, vtable, arg_ref);
      loom_type_t other_type = loom_verify_value_type(state, other_id);
      if (loom_constraint_property_equals(first_type, other_type,
                                          constraint->property)) {
        continue;
      }
      loom_verify_emit_pairwise_mismatch(state, op, vtable, constraint,
                                         first_type, other_type, first_ref,
                                         arg_ref);
      return;
    }
    uint16_t count = 0;
    const loom_value_id_t* values =
        loom_verify_resolve_variadic_field(op, vtable, arg_ref, &count);
    for (uint16_t j = 0; j < count; ++j) {
      loom_type_t other_type = loom_verify_value_type(state, values[j]);
      if (loom_constraint_property_equals(first_type, other_type,
                                          constraint->property)) {
        continue;
      }
      loom_verify_emit_indexed_pairwise_mismatch(
          state, op, vtable, type_error, first_ref,
          /*a_is_indexed=*/first_is_variadic, 0, first_type, arg_ref, j,
          other_type);
      return;
    }
  }
}

// ALL_SAME: every element of a single variadic value field shares the
// same property as the first element. Args: 1 variadic value field.
// Stops on the first mismatch.
static void loom_verify_relation_all_same(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  uint16_t count = 0;
  const loom_value_id_t* values = loom_verify_resolve_variadic_field(
      op, vtable, constraint->args[0], &count);
  if (count <= 1) return;
  loom_type_t first_type = loom_verify_value_type(state, values[0]);
  for (uint16_t i = 1; i < count; ++i) {
    loom_type_t other_type = loom_verify_value_type(state, values[i]);
    if (loom_constraint_property_equals(first_type, other_type,
                                        constraint->property)) {
      continue;
    }
    const loom_error_def_t* error =
        loom_verify_constraint_error_or(constraint, LOOM_ERR_SHAPE_003);
    loom_diagnostic_param_t params[] = {
        loom_param_type(first_type),
        loom_param_u32(i),
        loom_param_type(other_type),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
    return;
  }
}

// FIELD_SATISFIES: every element of every listed value field satisfies
// the type constraint stored in the constraint property slot. Args:
// 1+ value fields.
static void loom_verify_relation_field_satisfies(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  loom_type_constraint_t expected =
      (loom_type_constraint_t)constraint->property;
  if (expected >= LOOM_TYPE_CONSTRAINT_COUNT_) return;

  for (uint8_t i = 0; i < constraint->arg_count; ++i) {
    uint8_t field_ref = constraint->args[i];
    if (!loom_verify_is_variadic_field(vtable, field_ref)) {
      loom_value_id_t value_id =
          loom_verify_resolve_value_field(op, vtable, field_ref);
      if (value_id == LOOM_VALUE_ID_INVALID) continue;
      loom_type_t value_type = loom_verify_value_type(state, value_id);
      if (loom_type_satisfies_constraint(value_type, expected)) continue;

      char name_buffer[32];
      iree_string_view_t field_name = loom_verify_field_name(
          vtable, field_ref, name_buffer, sizeof(name_buffer));
      const loom_error_def_t* error =
          LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
              ? LOOM_ERR_TYPE_004
              : LOOM_ERR_TYPE_003;
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(field_name, field_ref),
          loom_param_type(value_type),
          loom_param_string(
              iree_make_cstring_view(loom_type_constraint_name(expected))),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 3 ? error->param_count : 3);
      return;
    }

    uint16_t count = 0;
    const loom_value_id_t* values =
        loom_verify_resolve_variadic_field(op, vtable, field_ref, &count);
    for (uint16_t j = 0; j < count; ++j) {
      loom_type_t value_type = loom_verify_value_type(state, values[j]);
      if (loom_type_satisfies_constraint(value_type, expected)) continue;

      char name_buffer[32];
      iree_string_view_t field_name = loom_verify_indexed_field_name(
          vtable, field_ref, j, name_buffer, sizeof(name_buffer));
      const loom_error_def_t* error =
          LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
              ? LOOM_ERR_TYPE_004
              : LOOM_ERR_TYPE_003;
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_indexed_field(field_name, field_ref, j),
          loom_param_type(value_type),
          loom_param_string(
              iree_make_cstring_view(loom_type_constraint_name(expected))),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 3 ? error->param_count : 3);
      return;
    }
  }
}

// REGION_ARGS_SATISFY: every entry block argument of a region satisfies the
// type constraint stored in the constraint property slot. Args: (region field).
static void loom_verify_relation_region_args_satisfy(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  loom_type_constraint_t expected =
      (loom_type_constraint_t)constraint->property;
  if (expected >= LOOM_TYPE_CONSTRAINT_COUNT_) return;
  uint8_t region_ref = constraint->args[0];
  if (LOOM_FIELD_REF_CATEGORY(region_ref) != LOOM_FIELD_REGION) return;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(region_ref);
  if (region_index >= op->region_count) return;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return;
  loom_block_t* entry = loom_region_entry_block(region);

  for (uint16_t i = 0; i < entry->arg_count; ++i) {
    loom_type_t argument_type =
        loom_verify_value_type(state, loom_block_arg_id(entry, i));
    if (loom_type_satisfies_constraint(argument_type, expected)) continue;

    char region_name_buffer[32];
    char argument_name_buffer[64];
    iree_string_view_t region_name = loom_verify_field_name(
        vtable, region_ref, region_name_buffer, sizeof(region_name_buffer));
    iree_snprintf(argument_name_buffer, sizeof(argument_name_buffer),
                  "%.*s.args[%u]", (int)region_name.size, region_name.data, i);
    const loom_error_def_t* error =
        loom_verify_constraint_error_or(constraint, LOOM_ERR_TYPE_014);
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(
            iree_make_cstring_view(argument_name_buffer), region_ref),
        loom_param_type(argument_type),
        loom_param_string(
            iree_make_cstring_view(loom_type_constraint_name(expected))),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
    return;
  }
}

typedef struct loom_verify_value_span_t {
  const loom_value_id_t* values;
  uint16_t count;
} loom_verify_value_span_t;

static bool loom_verify_region_entry_args(const loom_op_t* op,
                                          loom_field_ref_t region_ref,
                                          loom_verify_value_span_t* out_span) {
  *out_span = (loom_verify_value_span_t){0};
  if (LOOM_FIELD_REF_CATEGORY(region_ref) != LOOM_FIELD_REGION) return false;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(region_ref);
  if (region_index >= op->region_count) return false;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return false;
  loom_block_t* entry = loom_region_entry_block(region);
  out_span->values = entry->arg_ids;
  out_span->count = entry->arg_count;
  return true;
}

static bool loom_verify_query_element_bit_width(loom_type_t type,
                                                int32_t* out_bit_width) {
  if (!loom_type_is_scalar(type) && !loom_type_is_shaped(type)) return false;
  int32_t bit_width = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_width <= 0) return false;
  *out_bit_width = bit_width;
  return true;
}

static bool loom_verify_resolve_i64_attr_field(const loom_op_t* op,
                                               uint8_t attr_ref,
                                               int64_t* out_value) {
  if (LOOM_FIELD_REF_CATEGORY(attr_ref) != LOOM_FIELD_ATTR) return false;
  uint8_t attr_index = LOOM_FIELD_REF_INDEX(attr_ref);
  if (attr_index >= op->attribute_count) return false;
  loom_attribute_t attr = loom_op_attrs(op)[attr_index];
  if (attr.kind != LOOM_ATTR_I64) return false;
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_verify_resolve_attr_field(const loom_op_t* op,
                                           uint8_t attr_ref,
                                           loom_attribute_t* out_attr) {
  if (LOOM_FIELD_REF_CATEGORY(attr_ref) != LOOM_FIELD_ATTR) return false;
  uint8_t attr_index = LOOM_FIELD_REF_INDEX(attr_ref);
  if (attr_index >= op->attribute_count) return false;
  *out_attr = loom_op_attrs(op)[attr_index];
  return true;
}

static void loom_verify_emit_i64_attr_constraint(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t attr_ref, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  char attr_name_buffer[32];
  iree_string_view_t attr_name = loom_verify_field_name(
      vtable, attr_ref, attr_name_buffer, sizeof(attr_name_buffer));
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(attr_name, attr_ref),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_014, params,
                              IREE_ARRAYSIZE(params));
}

static void loom_verify_emit_attr_kind_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t attr_ref,
    const loom_error_def_t* error, loom_attr_kind_t actual_kind,
    loom_attr_kind_t expected_kind) {
  char attr_name_buffer[32];
  iree_string_view_t attr_name = loom_verify_field_name(
      vtable, attr_ref, attr_name_buffer, sizeof(attr_name_buffer));
  if (!error) error = LOOM_ERR_TYPE_005;
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(attr_name, attr_ref),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

static void loom_verify_emit_value_field_constraint(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t field_ref, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  char field_name_buffer[32];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  const loom_error_def_t* error =
      LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
          ? LOOM_ERR_TYPE_004
          : LOOM_ERR_TYPE_003;
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(field_name, field_ref),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

static bool loom_verify_attr_literal_fits_scalar_type(
    loom_attribute_t attr, loom_scalar_type_t scalar_type,
    int64_t* out_actual_value) {
  if (attr.kind != LOOM_ATTR_I64) {
    return true;
  }
  int64_t lo = 0;
  int64_t hi = 0;
  if (!loom_scalar_type_integer_domain(scalar_type, &lo, &hi)) {
    return true;
  }
  const int64_t value = loom_attr_as_i64(attr);
  *out_actual_value = value;
  return value >= lo && value <= hi;
}

// ATTR_MATCHES_ELEMENT_TYPE: an attribute literal payload kind matches the
// scalar element type of a value field and, for integer-like payloads, is
// representable in that element type's value domain. Args: (attr field, value
// field).
static void loom_verify_relation_attr_matches_element_type(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  if (constraint->property != LOOM_PROPERTY_ELEMENT_TYPE) return;
  uint8_t attr_ref = constraint->args[0];
  uint8_t field_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, field_ref)) return;

  loom_attribute_t attr = {0};
  loom_value_id_t value_id =
      loom_verify_resolve_value_field(op, vtable, field_ref);
  if (!loom_verify_resolve_attr_field(op, attr_ref, &attr) ||
      value_id == LOOM_VALUE_ID_INVALID || loom_attr_is_absent(attr)) {
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  if (!loom_type_is_scalar(value_type) && !loom_type_is_shaped(value_type)) {
    return;
  }

  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (loom_attr_matches_scalar_type(attr, loom_type_element_type(value_type),
                                    &expected_kind)) {
    int64_t actual_value = 0;
    if (loom_verify_attr_literal_fits_scalar_type(
            attr, loom_type_element_type(value_type), &actual_value)) {
      return;
    }
    loom_verify_emit_i64_attr_constraint(state, op, vtable, attr_ref,
                                         actual_value,
                                         IREE_SV("element type domain"));
    return;
  }
  loom_verify_emit_attr_kind_mismatch(
      state, op, vtable, attr_ref,
      loom_verify_constraint_error_or(constraint, NULL), attr.kind,
      expected_kind);
}

// ATTR_I64_PREDICATE: an i64 attribute satisfies a predicate stored in the
// property slot. Args: (i64 attr field).
static void loom_verify_relation_attr_i64_predicate(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  uint8_t attr_ref = constraint->args[0];
  int64_t value = 0;
  if (!loom_verify_resolve_i64_attr_field(op, attr_ref, &value)) return;

  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_BIT_WIDTH_POSITIVE:
      if (value > 0) return;
      loom_verify_emit_i64_attr_constraint(state, op, vtable, attr_ref, value,
                                           IREE_SV("positive bit width"));
      return;
    default:
      return;
  }
}

// ELEMENT_WIDTH_ORDER: the scalar or shaped element bit width of the first
// value field is strictly greater or less than the second value field. Args:
// (checked value field, reference value field).
static void loom_verify_relation_element_width_order(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t field_ref = constraint->args[0];
  uint8_t reference_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, field_ref) ||
      loom_verify_is_variadic_field(vtable, reference_ref)) {
    return;
  }

  loom_value_id_t value_id =
      loom_verify_resolve_value_field(op, vtable, field_ref);
  loom_value_id_t reference_id =
      loom_verify_resolve_value_field(op, vtable, reference_ref);
  if (value_id == LOOM_VALUE_ID_INVALID ||
      reference_id == LOOM_VALUE_ID_INVALID) {
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  loom_type_t reference_type = loom_verify_value_type(state, reference_id);
  int32_t value_bit_width = 0;
  int32_t reference_bit_width = 0;
  if (!loom_verify_query_element_bit_width(value_type, &value_bit_width) ||
      !loom_verify_query_element_bit_width(reference_type,
                                           &reference_bit_width)) {
    return;
  }

  const char* relation_text = NULL;
  bool relation_matches = false;
  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_ELEMENT_WIDTH_GREATER_THAN:
      relation_text = "greater than";
      relation_matches = value_bit_width > reference_bit_width;
      break;
    case LOOM_PROPERTY_ELEMENT_WIDTH_LESS_THAN:
      relation_text = "less than";
      relation_matches = value_bit_width < reference_bit_width;
      break;
    default:
      return;
  }
  if (relation_matches) return;

  char field_name_buffer[32];
  char reference_name_buffer[32];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  iree_string_view_t reference_name =
      loom_verify_field_name(vtable, reference_ref, reference_name_buffer,
                             sizeof(reference_name_buffer));

  char expected_buffer[96];
  iree_snprintf(expected_buffer, sizeof(expected_buffer),
                "element bit width %s %.*s", relation_text,
                (int)reference_name.size, reference_name.data);
  const loom_error_def_t* error =
      LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
          ? LOOM_ERR_TYPE_004
          : LOOM_ERR_TYPE_003;
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(field_name, field_ref),
      loom_param_type(value_type),
      loom_param_string(iree_make_cstring_view(expected_buffer)),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

// ELEMENT_WIDTH_AT_LEAST_ATTR: the scalar or shaped element bit width of the
// first value field is at least the i64 attribute value. Args: (checked value
// field, i64 attr field).
static void loom_verify_relation_element_width_at_least_attr(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t field_ref = constraint->args[0];
  uint8_t attr_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, field_ref)) return;

  loom_value_id_t value_id =
      loom_verify_resolve_value_field(op, vtable, field_ref);
  int64_t required_bit_width = 0;
  if (value_id == LOOM_VALUE_ID_INVALID ||
      !loom_verify_resolve_i64_attr_field(op, attr_ref, &required_bit_width) ||
      required_bit_width <= 0) {
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  int32_t element_bit_width = 0;
  if (!loom_verify_query_element_bit_width(value_type, &element_bit_width)) {
    return;
  }
  if ((int64_t)element_bit_width >= required_bit_width) return;

  char field_name_buffer[32];
  char attr_name_buffer[32];
  char expected_buffer[96];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  iree_string_view_t attr_name = loom_verify_field_name(
      vtable, attr_ref, attr_name_buffer, sizeof(attr_name_buffer));
  iree_snprintf(expected_buffer, sizeof(expected_buffer),
                "element bit width at least %.*s", (int)attr_name.size,
                attr_name.data);
  const loom_error_def_t* error =
      LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
          ? LOOM_ERR_TYPE_004
          : LOOM_ERR_TYPE_003;
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(field_name, field_ref),
      loom_param_type(value_type),
      loom_param_string(iree_make_cstring_view(expected_buffer)),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

// BIT_RANGE_WITHIN_ELEMENT_WIDTH: a bit range described by offset and width
// attributes fits within a scalar or shaped element width. Args: (checked value
// field, offset i64 attr field, width i64 attr field).
static void loom_verify_relation_bit_range_within_element_width(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 3) return;
  uint8_t field_ref = constraint->args[0];
  uint8_t offset_ref = constraint->args[1];
  uint8_t width_ref = constraint->args[2];
  if (loom_verify_is_variadic_field(vtable, field_ref)) return;

  loom_value_id_t value_id =
      loom_verify_resolve_value_field(op, vtable, field_ref);
  int64_t offset = 0;
  int64_t width = 0;
  if (value_id == LOOM_VALUE_ID_INVALID ||
      !loom_verify_resolve_i64_attr_field(op, offset_ref, &offset) ||
      !loom_verify_resolve_i64_attr_field(op, width_ref, &width)) {
    return;
  }

  if (offset < 0) {
    loom_verify_emit_i64_attr_constraint(state, op, vtable, offset_ref, offset,
                                         IREE_SV("non-negative bit offset"));
    return;
  }
  if (width <= 0) {
    loom_verify_emit_i64_attr_constraint(state, op, vtable, width_ref, width,
                                         IREE_SV("positive bitfield width"));
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  int32_t element_bit_width = 0;
  if (!loom_verify_query_element_bit_width(value_type, &element_bit_width)) {
    return;
  }
  if (offset <= element_bit_width && width <= element_bit_width - offset) {
    return;
  }
  loom_verify_emit_i64_attr_constraint(
      state, op, vtable, width_ref, width,
      IREE_SV("bitfield range within storage element width"));
}

typedef struct loom_verify_total_bit_count_expr_t {
  // Product of all static dimensions and the element bit width.
  uint64_t static_factor;
  // Sorted dynamic dimension value IDs participating in the product.
  loom_value_id_t dynamic_dims[LOOM_TYPE_MAX_RANK];
  // Number of entries used in dynamic_dims.
  uint8_t dynamic_dim_count;
} loom_verify_total_bit_count_expr_t;

static void loom_verify_sort_dynamic_dims(loom_value_id_t* dims,
                                          uint8_t dim_count) {
  for (uint8_t i = 1; i < dim_count; ++i) {
    loom_value_id_t value_id = dims[i];
    uint8_t j = i;
    while (j > 0 && dims[j - 1] > value_id) {
      dims[j] = dims[j - 1];
      --j;
    }
    dims[j] = value_id;
  }
}

static bool loom_verify_total_bit_count_expr(
    loom_type_t type, int32_t element_width,
    loom_verify_total_bit_count_expr_t* out_expr) {
  *out_expr = (loom_verify_total_bit_count_expr_t){
      .static_factor = (uint64_t)element_width,
  };
  if (element_width <= 0) return false;
  if (loom_type_is_scalar(type)) return true;
  if (!loom_type_is_shaped(type)) return false;

  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) {
      if (out_expr->dynamic_dim_count >=
          IREE_ARRAYSIZE(out_expr->dynamic_dims)) {
        return false;
      }
      loom_value_id_t dimension_value_id = loom_type_dim_value_id_at(type, i);
      if (dimension_value_id == LOOM_VALUE_ID_INVALID) return false;
      out_expr->dynamic_dims[out_expr->dynamic_dim_count++] =
          dimension_value_id;
      continue;
    }

    int64_t dimension_size = loom_type_dim_static_size_at(type, i);
    if (dimension_size < 0) return false;
    if (dimension_size == 0) {
      out_expr->static_factor = 0;
      out_expr->dynamic_dim_count = 0;
      return true;
    }
    uint64_t dimension_size_u64 = (uint64_t)dimension_size;
    if (out_expr->static_factor > UINT64_MAX / dimension_size_u64) {
      return false;
    }
    out_expr->static_factor *= dimension_size_u64;
  }

  loom_verify_sort_dynamic_dims(out_expr->dynamic_dims,
                                out_expr->dynamic_dim_count);
  return true;
}

static bool loom_verify_total_bit_count_expr_equal(
    const loom_verify_total_bit_count_expr_t* lhs,
    const loom_verify_total_bit_count_expr_t* rhs) {
  if (lhs->static_factor != rhs->static_factor ||
      lhs->dynamic_dim_count != rhs->dynamic_dim_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->dynamic_dim_count; ++i) {
    if (lhs->dynamic_dims[i] != rhs->dynamic_dims[i]) return false;
  }
  return true;
}

// TOTAL_BIT_COUNT_EQUAL: two value fields must have the same total bit count
// when expressed as element-bit-width times element count. Dynamic dimensions
// are compared by SSA identity after sorting, so vector<[%m]x2xi8> and
// vector<[%m]xi16> prove equal without needing arithmetic facts. Args:
// (lhs value field, rhs value field).
static void loom_verify_relation_total_bit_count_equal(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t lhs_ref = constraint->args[0];
  uint8_t rhs_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, lhs_ref) ||
      loom_verify_is_variadic_field(vtable, rhs_ref)) {
    return;
  }

  loom_value_id_t lhs_value_id =
      loom_verify_resolve_value_field(op, vtable, lhs_ref);
  loom_value_id_t rhs_value_id =
      loom_verify_resolve_value_field(op, vtable, rhs_ref);
  if (lhs_value_id == LOOM_VALUE_ID_INVALID ||
      rhs_value_id == LOOM_VALUE_ID_INVALID) {
    return;
  }

  loom_type_t lhs_type = loom_verify_value_type(state, lhs_value_id);
  loom_type_t rhs_type = loom_verify_value_type(state, rhs_value_id);
  int32_t lhs_width = 0;
  int32_t rhs_width = 0;
  if (!loom_verify_query_element_bit_width(lhs_type, &lhs_width) ||
      !loom_verify_query_element_bit_width(rhs_type, &rhs_width)) {
    return;
  }

  if (lhs_width == rhs_width && loom_type_shape_equals(lhs_type, rhs_type)) {
    return;
  }

  loom_verify_total_bit_count_expr_t lhs_bit_count = {0};
  loom_verify_total_bit_count_expr_t rhs_bit_count = {0};
  bool counts_match =
      loom_verify_total_bit_count_expr(lhs_type, lhs_width, &lhs_bit_count) &&
      loom_verify_total_bit_count_expr(rhs_type, rhs_width, &rhs_bit_count) &&
      loom_verify_total_bit_count_expr_equal(&lhs_bit_count, &rhs_bit_count);
  if (counts_match) return;

  char lhs_name_buffer[32];
  char expected_buffer[96];
  iree_string_view_t lhs_name = loom_verify_field_name(
      vtable, lhs_ref, lhs_name_buffer, sizeof(lhs_name_buffer));
  iree_snprintf(expected_buffer, sizeof(expected_buffer),
                "provably same total bit count as %.*s", (int)lhs_name.size,
                lhs_name.data);
  loom_verify_emit_value_field_constraint(
      state, op, vtable, rhs_ref, rhs_type,
      iree_make_cstring_view(expected_buffer));
}

static bool loom_verify_static_bit_count(loom_type_t type,
                                         int64_t bit_width_per_element,
                                         bool* out_is_static,
                                         uint64_t* out_bit_count) {
  *out_is_static = false;
  *out_bit_count = 0;
  if (bit_width_per_element < 0) return false;

  uint64_t element_count = 0;
  if (loom_type_is_scalar(type)) {
    element_count = 1;
  } else if (!loom_type_static_element_count(type, &element_count)) {
    return true;
  }

  *out_is_static = true;
  if (bit_width_per_element == 0) return true;
  uint64_t bit_width = (uint64_t)bit_width_per_element;
  if (element_count > UINT64_MAX / bit_width) return false;
  *out_bit_count = element_count * bit_width;
  return true;
}

static iree_string_view_t loom_verify_payload_bit_count_mismatch_constraint(
    loom_constraint_property_t property) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_PACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE:
      return IREE_SV(
          "packed payload bit count equal to result storage bit count");
    case LOOM_PROPERTY_UNPACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE:
      return IREE_SV(
          "unpacked payload bit count equal to source storage bit count");
    default:
      return IREE_SV("payload bit count equal to storage bit count");
  }
}

// PAYLOAD_BIT_COUNT_MATCHES_STORAGE: a payload field with a fixed element bit
// width stored in an i64 attr must have the same static total bit count as a
// storage value field. Dynamic shapes are accepted here because the relation is
// only intended to catch concrete bitstream-size mistakes without requiring a
// symbolic arithmetic solver. Args: (payload value field, width i64 attr field,
// storage value field, diagnostic value field).
static void loom_verify_relation_payload_bit_count_matches_storage(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 4) return;
  uint8_t payload_ref = constraint->args[0];
  uint8_t width_ref = constraint->args[1];
  uint8_t storage_ref = constraint->args[2];
  uint8_t diagnostic_ref = constraint->args[3];
  if (loom_verify_is_variadic_field(vtable, payload_ref) ||
      loom_verify_is_variadic_field(vtable, storage_ref) ||
      loom_verify_is_variadic_field(vtable, diagnostic_ref)) {
    return;
  }

  loom_value_id_t payload_value_id =
      loom_verify_resolve_value_field(op, vtable, payload_ref);
  loom_value_id_t storage_value_id =
      loom_verify_resolve_value_field(op, vtable, storage_ref);
  loom_value_id_t diagnostic_value_id =
      loom_verify_resolve_value_field(op, vtable, diagnostic_ref);
  int64_t payload_width = 0;
  if (payload_value_id == LOOM_VALUE_ID_INVALID ||
      storage_value_id == LOOM_VALUE_ID_INVALID ||
      diagnostic_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_verify_resolve_i64_attr_field(op, width_ref, &payload_width) ||
      payload_width <= 0) {
    return;
  }

  loom_type_t payload_type = loom_verify_value_type(state, payload_value_id);
  loom_type_t storage_type = loom_verify_value_type(state, storage_value_id);
  loom_type_t diagnostic_type =
      loom_verify_value_type(state, diagnostic_value_id);
  int32_t storage_width = 0;
  if (!loom_verify_query_element_bit_width(storage_type, &storage_width)) {
    return;
  }

  bool payload_bit_count_is_static = false;
  uint64_t payload_bit_count = 0;
  if (!loom_verify_static_bit_count(payload_type, payload_width,
                                    &payload_bit_count_is_static,
                                    &payload_bit_count)) {
    loom_verify_emit_value_field_constraint(
        state, op, vtable, payload_ref, payload_type,
        IREE_SV("representable static payload bit count"));
    return;
  }

  bool storage_bit_count_is_static = false;
  uint64_t storage_bit_count = 0;
  if (!loom_verify_static_bit_count(storage_type, storage_width,
                                    &storage_bit_count_is_static,
                                    &storage_bit_count)) {
    loom_verify_emit_value_field_constraint(
        state, op, vtable, storage_ref, storage_type,
        IREE_SV("representable static storage bit count"));
    return;
  }

  if (!payload_bit_count_is_static || !storage_bit_count_is_static) return;
  if (payload_bit_count == storage_bit_count) return;

  loom_verify_emit_value_field_constraint(
      state, op, vtable, diagnostic_ref, diagnostic_type,
      loom_verify_payload_bit_count_mismatch_constraint(constraint->property));
}

static iree_string_view_t loom_verify_grouped_last_axis_divisibility_constraint(
    int64_t group_size) {
  switch (group_size) {
    case 2:
      return IREE_SV("last axis extent divisible by 2");
    case 4:
      return IREE_SV("last axis extent divisible by 4");
    case 8:
      return IREE_SV("last axis extent divisible by 8");
    default:
      return IREE_SV("last axis extent divisible by group size");
  }
}

static iree_string_view_t loom_verify_grouped_last_axis_result_constraint(
    int64_t group_size) {
  switch (group_size) {
    case 2:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by 2");
    case 4:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by 4");
    case 8:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by 8");
    default:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by group "
          "size");
  }
}

// LAST_AXIS_GROUPED_BY: result vector shape equals source vector shape with
// the last axis divided by a small static group size stored in the property
// slot. Args: (source vector field, result vector field).
static void loom_verify_relation_last_axis_grouped_by(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2 || constraint->property == 0) return;
  uint8_t source_ref = constraint->args[0];
  uint8_t result_ref = constraint->args[1];
  loom_value_id_t source_id =
      loom_verify_resolve_value_field(op, vtable, source_ref);
  loom_value_id_t result_id =
      loom_verify_resolve_value_field(op, vtable, result_ref);
  if (source_id == LOOM_VALUE_ID_INVALID ||
      result_id == LOOM_VALUE_ID_INVALID) {
    return;
  }

  loom_type_t source_type = loom_verify_value_type(state, source_id);
  loom_type_t result_type = loom_verify_value_type(state, result_id);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return;
  }

  char source_name_buffer[32];
  char result_name_buffer[32];
  iree_string_view_t source_name = loom_verify_field_name(
      vtable, source_ref, source_name_buffer, sizeof(source_name_buffer));
  iree_string_view_t result_name = loom_verify_field_name(
      vtable, result_ref, result_name_buffer, sizeof(result_name_buffer));

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank == 0 || result_rank == 0) return;
  if (result_rank != source_rank) {
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(result_name, result_ref),
        loom_param_i64(result_rank),
        loom_verify_param_string_for_field(source_name, source_ref),
        loom_param_i64(source_rank),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_SHAPE_001, params,
                                IREE_ARRAYSIZE(params));
    return;
  }

  uint8_t grouped_axis = source_rank - 1;
  for (uint8_t axis = 0; axis < grouped_axis; ++axis) {
    if (loom_type_dim(source_type, axis) == loom_type_dim(result_type, axis)) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(result_name, result_ref),
        loom_verify_param_string_for_field(source_name, source_ref),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_SHAPE_002, params,
                                IREE_ARRAYSIZE(params));
    return;
  }

  if (loom_type_dim_is_dynamic_at(source_type, grouped_axis)) return;

  int64_t source_axis_size =
      loom_type_dim_static_size_at(source_type, grouped_axis);
  int64_t group_size = constraint->property;
  if ((source_axis_size % group_size) != 0) {
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(source_name, source_ref),
        loom_param_type(source_type),
        loom_param_string(
            loom_verify_grouped_last_axis_divisibility_constraint(group_size)),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_003, params,
                                IREE_ARRAYSIZE(params));
    return;
  }

  if (loom_type_dim_is_dynamic_at(result_type, grouped_axis)) return;

  int64_t result_axis_size =
      loom_type_dim_static_size_at(result_type, grouped_axis);
  if (result_axis_size == source_axis_size / group_size) return;

  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(result_name, result_ref),
      loom_param_type(result_type),
      loom_param_string(
          loom_verify_grouped_last_axis_result_constraint(group_size)),
  };
  loom_verify_emit_structured(state, op, LOOM_ERR_TYPE_004, params,
                              IREE_ARRAYSIZE(params));
}

// COUNT_MATCHES_RANK: a variadic value field's element count equals
// the rank of a shaped value field. Args: (shaped value field,
// variadic value field).
static void loom_verify_relation_count_matches_rank(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  loom_type_t shaped_type = loom_verify_value_type(
      state, loom_verify_resolve_value_field(op, vtable, constraint->args[0]));
  uint16_t variadic_count =
      loom_verify_variadic_count(op, vtable, constraint->args[1]);
  uint8_t rank = loom_type_rank(shaped_type);
  if (variadic_count == rank) return;
  char name_buffer[32];
  iree_string_view_t operand_name = loom_verify_field_name(
      vtable, constraint->args[0], name_buffer, sizeof(name_buffer));
  const loom_error_def_t* error =
      loom_verify_constraint_error_or(constraint, LOOM_ERR_SUBRANGE_001);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(operand_name, constraint->args[0]),
      loom_param_u32(variadic_count),
      loom_param_i64(rank),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

// COUNT_MATCHES_STATIC_ELEMENT_COUNT: a variadic value field's element count
// equals the static element count of a shaped value field. Args: (shaped value
// field, variadic value field).
static void loom_verify_relation_count_matches_static_element_count(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t shaped_ref = constraint->args[0];
  uint8_t values_ref = constraint->args[1];
  loom_type_t shaped_type = loom_verify_value_type(
      state, loom_verify_resolve_value_field(op, vtable, shaped_ref));
  if (!loom_type_is_shaped(shaped_type) ||
      !loom_type_is_all_static(shaped_type)) {
    return;
  }

  uint64_t expected_count = 0;
  if (!loom_type_static_element_count(shaped_type, &expected_count)) {
    char shaped_name_buffer[32];
    iree_string_view_t shaped_name = loom_verify_field_name(
        vtable, shaped_ref, shaped_name_buffer, sizeof(shaped_name_buffer));
    const loom_error_def_t* error =
        LOOM_FIELD_REF_CATEGORY(shaped_ref) == LOOM_FIELD_RESULT
            ? LOOM_ERR_TYPE_004
            : LOOM_ERR_TYPE_003;
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(shaped_name, shaped_ref),
        loom_param_type(shaped_type),
        loom_param_string(IREE_SV("representable static element count")),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
    return;
  }

  uint16_t value_count = loom_verify_variadic_count(op, vtable, values_ref);
  if (value_count == expected_count) return;

  char values_name_buffer[32];
  char shaped_name_buffer[32];
  char expected_name_buffer[64];
  iree_string_view_t values_name = loom_verify_field_name(
      vtable, values_ref, values_name_buffer, sizeof(values_name_buffer));
  iree_string_view_t shaped_name = loom_verify_field_name(
      vtable, shaped_ref, shaped_name_buffer, sizeof(shaped_name_buffer));
  iree_snprintf(expected_name_buffer, sizeof(expected_name_buffer),
                "%.*s static element count", (int)shaped_name.size,
                shaped_name.data);
  uint32_t expected_count_param =
      expected_count > UINT32_MAX ? UINT32_MAX : (uint32_t)expected_count;
  const loom_error_def_t* error =
      loom_verify_constraint_error_or(constraint, LOOM_ERR_STRUCTURE_013);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(values_name, values_ref),
      loom_param_u32(value_count),
      loom_verify_param_string_for_field(
          iree_make_cstring_view(expected_name_buffer), shaped_ref),
      loom_param_u32(expected_count_param),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 4 ? error->param_count : 4);
}

// ATTR_IN_RANGE_RANK: an i64 attribute's value falls within
// [0, rank) of a shaped value field. Args: (shaped value field,
// i64 attr field).
static void loom_verify_relation_attr_in_range_rank(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  loom_type_t shaped_type = loom_verify_value_type(
      state, loom_verify_resolve_value_field(op, vtable, constraint->args[0]));
  uint8_t attr_index = LOOM_FIELD_REF_INDEX(constraint->args[1]);
  if (attr_index >= op->attribute_count) return;
  int64_t dim_index = loom_attr_as_i64(loom_op_attrs(op)[attr_index]);
  uint8_t rank = loom_type_rank(shaped_type);
  if (dim_index >= 0 && dim_index < rank) return;
  const loom_error_def_t* error =
      loom_verify_constraint_error_or(constraint, LOOM_ERR_SUBRANGE_002);
  loom_diagnostic_param_t params[] = {
      loom_param_i64(dim_index),
      loom_param_i64(rank),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 2 ? error->param_count : 2);
}

// REGION_ARG_COUNT: a region's entry block argument count matches the element
// count of a variadic value field or another region's entry block args. Args:
// (region field, variadic value field | region field).
static void loom_verify_relation_region_arg_count(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(constraint->args[0]);
  if (region_index >= op->region_count) return;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return;
  uint16_t block_arg_count = loom_region_entry_arg_count(region);
  uint16_t expected_count = 0;
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) == LOOM_FIELD_REGION) {
    loom_verify_value_span_t reference_args = {0};
    if (!loom_verify_region_entry_args(op, constraint->args[1],
                                       &reference_args)) {
      return;
    }
    expected_count = reference_args.count;
  } else {
    expected_count =
        loom_verify_variadic_count(op, vtable, constraint->args[1]);
  }
  if (block_arg_count == expected_count) return;
  const loom_error_def_t* error =
      loom_verify_constraint_error_or(constraint, LOOM_ERR_STRUCTURE_007);
  loom_diagnostic_param_t params[] = {
      loom_param_u32(block_arg_count),
      loom_param_u32(expected_count),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 2 ? error->param_count : 2);
}

// REGION_ARG_MATCH: each region entry block argument's property matches the
// corresponding element of a variadic value field or another region's entry
// block args at the same position. Args: (region field, variadic value field |
// region field).
static void loom_verify_relation_region_arg_match(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  loom_verify_value_span_t args = {0};
  if (!loom_verify_region_entry_args(op, constraint->args[0], &args)) return;
  loom_verify_value_span_t inputs = {0};
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) == LOOM_FIELD_REGION) {
    if (!loom_verify_region_entry_args(op, constraint->args[1], &inputs)) {
      return;
    }
  } else {
    inputs.values = loom_verify_resolve_variadic_field(
        op, vtable, constraint->args[1], &inputs.count);
    if (!inputs.values) return;
  }
  uint16_t check_count = args.count < inputs.count ? args.count : inputs.count;
  for (uint16_t i = 0; i < check_count; ++i) {
    loom_type_t block_arg_type = loom_verify_value_type(state, args.values[i]);
    loom_type_t input_type = loom_verify_value_type(state, inputs.values[i]);
    if (loom_constraint_property_equals(block_arg_type, input_type,
                                        constraint->property)) {
      continue;
    }
    const loom_error_def_t* error =
        loom_verify_constraint_error_or(constraint, LOOM_ERR_TYPE_008);
    loom_type_t expected_type =
        constraint->property == LOOM_PROPERTY_ELEMENT_TYPE
            ? loom_type_scalar(loom_type_element_type(input_type))
            : input_type;
    loom_diagnostic_param_t params[] = {
        loom_param_u32(i),
        loom_param_type(block_arg_type),
        loom_param_type(expected_type),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
  }
}

// YIELD_COUNT: a region's terminator (yield) operand count matches
// the element count of a variadic value field. Args: (region field,
// variadic value field).
static void loom_verify_relation_yield_count(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint16_t yield_count = 0;
  if (!loom_verify_region_entry_yield(state, op, vtable,
                                      LOOM_FIELD_REF_INDEX(constraint->args[0]),
                                      &yield_count, NULL)) {
    return;
  }
  uint16_t result_count =
      loom_verify_variadic_count(op, vtable, constraint->args[1]);
  // A non-variadic result counts as a single element for the purposes
  // of yield-count comparisons (the field still names a value).
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) == LOOM_FIELD_RESULT &&
      LOOM_FIELD_REF_INDEX(constraint->args[1]) < vtable->fixed_result_count) {
    result_count = 1;
  }
  if (yield_count == result_count) return;
  const loom_error_def_t* error =
      loom_verify_constraint_error_or(constraint, LOOM_ERR_STRUCTURE_008);
  loom_diagnostic_param_t params[] = {
      loom_param_u32(yield_count),
      loom_param_u32(result_count),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 2 ? error->param_count : 2);
}

// YIELD_MATCH: each region terminator (yield) operand's property
// matches the corresponding element of a variadic result field at the
// same position. Args: (region field, variadic result field).
static void loom_verify_relation_yield_match(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  // Yields forward into result values — only result-side fields are
  // valid as the second arg.
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) != LOOM_FIELD_RESULT) {
    return;
  }
  uint16_t yield_count = 0;
  const loom_value_id_t* yield_operands = NULL;
  if (!loom_verify_region_entry_yield(state, op, vtable,
                                      LOOM_FIELD_REF_INDEX(constraint->args[0]),
                                      &yield_count, &yield_operands)) {
    return;
  }
  uint16_t result_count = 0;
  const loom_value_id_t* result_values = loom_verify_resolve_variadic_field(
      op, vtable, constraint->args[1], &result_count);
  if (!result_values) return;
  uint16_t check_count =
      yield_count < result_count ? yield_count : result_count;
  loom_type_value_remap_t yield_remap = {
      .source_values = result_values,
      .target_values = yield_operands,
      .count = check_count,
  };
  for (uint16_t i = 0; i < check_count; ++i) {
    loom_type_t yield_type = loom_verify_value_type(state, yield_operands[i]);
    loom_type_t result_type = loom_verify_value_type(state, result_values[i]);
    bool matched = constraint->property == LOOM_PROPERTY_TYPE
                       ? loom_type_equal_after_value_remap(
                             result_type, yield_type, &yield_remap)
                       : loom_constraint_property_equals(
                             yield_type, result_type, constraint->property);
    if (matched) {
      continue;
    }
    const loom_error_def_t* error =
        loom_verify_constraint_error_or(constraint, LOOM_ERR_TYPE_009);
    loom_type_t expected_type =
        constraint->property == LOOM_PROPERTY_TYPE
            ? result_type
            : loom_type_scalar(loom_type_element_type(result_type));
    loom_diagnostic_param_t params[] = {
        loom_param_type(yield_type),
        loom_param_type(expected_type),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 2 ? error->param_count : 2);
  }
}

// VARIADIC_MATCH: two variadic value fields agree position-by-position
// on count and per-element property. Args: (variadic value field,
// variadic value field). Stops at the count check on a mismatch to
// avoid cascading per-position errors that obscure the root cause.
static void loom_verify_relation_variadic_match(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t ref_a = constraint->args[0];
  uint8_t ref_b = constraint->args[1];
  uint16_t count_a = 0;
  uint16_t count_b = 0;
  const loom_value_id_t* values_a =
      loom_verify_resolve_variadic_field(op, vtable, ref_a, &count_a);
  const loom_value_id_t* values_b =
      loom_verify_resolve_variadic_field(op, vtable, ref_b, &count_b);
  if (!values_a || !values_b) return;

  if (count_a != count_b) {
    char name_a_buffer[32];
    char name_b_buffer[32];
    iree_string_view_t name_a = loom_verify_field_name(
        vtable, ref_a, name_a_buffer, sizeof(name_a_buffer));
    iree_string_view_t name_b = loom_verify_field_name(
        vtable, ref_b, name_b_buffer, sizeof(name_b_buffer));
    const loom_error_def_t* error =
        loom_verify_constraint_error_or(constraint, LOOM_ERR_STRUCTURE_013);
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(name_a, ref_a),
        loom_param_u32(count_a),
        loom_verify_param_string_for_field(name_b, ref_b),
        loom_param_u32(count_b),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 4 ? error->param_count : 4);
    return;
  }

  // Per-position type mismatches use the property's default type
  // error, NOT the constraint's error_ref: VARIADIC_MATCH stores the count
  // error (ERR_STRUCTURE_013) on the constraint, which is the wrong
  // shape (STRING/U32/STRING/U32) for type-mismatch params.
  const loom_error_def_t* type_error =
      loom_pairwise_eq_default_error(constraint->property);
  loom_type_value_remap_t value_remap = {
      .source_values = values_b,
      .target_values = values_a,
      .count = count_a,
  };
  for (uint16_t i = 0; i < count_a; ++i) {
    loom_type_t type_a = loom_verify_value_type(state, values_a[i]);
    loom_type_t type_b = loom_verify_value_type(state, values_b[i]);
    bool matched =
        constraint->property == LOOM_PROPERTY_TYPE
            ? loom_type_equal_after_value_remap(type_b, type_a, &value_remap)
            : loom_constraint_property_equals(type_a, type_b,
                                              constraint->property);
    if (matched) {
      continue;
    }
    loom_verify_emit_indexed_pairwise_mismatch(state, op, vtable, type_error,
                                               ref_a, /*a_is_indexed=*/true, i,
                                               type_a, ref_b, i, type_b);
  }
}

// REGISTER_UNIT_COUNT_SUM: register unit counts in the first variadic register
// field sum to the register unit count of the second register field. Args:
// (summed register value field, result register value field).
static void loom_verify_relation_register_unit_count_sum(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) {
    return;
  }
  uint8_t sources_ref = constraint->args[0];
  uint8_t result_ref = constraint->args[1];
  if (!loom_verify_is_variadic_field(vtable, sources_ref) ||
      loom_verify_is_variadic_field(vtable, result_ref)) {
    return;
  }

  uint16_t source_count = 0;
  const loom_value_id_t* source_values = loom_verify_resolve_variadic_field(
      op, vtable, sources_ref, &source_count);
  loom_value_id_t result_value =
      loom_verify_resolve_value_field(op, vtable, result_ref);
  if (!source_values || result_value == LOOM_VALUE_ID_INVALID) {
    return;
  }

  uint64_t source_unit_count = 0;
  for (uint16_t i = 0; i < source_count; ++i) {
    loom_type_t source_type = loom_verify_value_type(state, source_values[i]);
    if (!loom_type_is_register(source_type)) {
      return;
    }
    source_unit_count += loom_low_register_type_unit_count(source_type);
  }

  loom_type_t result_type = loom_verify_value_type(state, result_value);
  if (!loom_type_is_register(result_type)) {
    return;
  }
  if (source_unit_count == loom_low_register_type_unit_count(result_type)) {
    return;
  }

  char source_name_buffer[32];
  char expected_buffer[96];
  iree_string_view_t source_name = loom_verify_field_name(
      vtable, sources_ref, source_name_buffer, sizeof(source_name_buffer));
  iree_snprintf(expected_buffer, sizeof(expected_buffer),
                "register unit count equal to sum of %.*s",
                (int)source_name.size, source_name.data);
  loom_verify_emit_value_field_constraint(
      state, op, vtable, result_ref, result_type,
      iree_make_cstring_view(expected_buffer));
}

// Dispatch table for semantic constraint relations. Indexed by the
// loom_constraint_relation_t enum value. Adding a relation requires
// (a) updating the enum in op_defs.h, (b) adding a handler above, and
// (c) appending the handler here. The static_assert below ensures
// every enum value has a handler.
typedef void (*loom_verify_relation_fn_t)(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          const loom_constraint_t* constraint);

static const loom_verify_relation_fn_t kVerifyRelationFns[] = {
    [LOOM_RELATION_PAIRWISE_EQ] = loom_verify_relation_pairwise_eq,
    [LOOM_RELATION_ALL_SAME] = loom_verify_relation_all_same,
    [LOOM_RELATION_FIELD_SATISFIES] = loom_verify_relation_field_satisfies,
    [LOOM_RELATION_REGION_ARGS_SATISFY] =
        loom_verify_relation_region_args_satisfy,
    [LOOM_RELATION_ATTR_I64_PREDICATE] =
        loom_verify_relation_attr_i64_predicate,
    [LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE] =
        loom_verify_relation_attr_matches_element_type,
    [LOOM_RELATION_ELEMENT_WIDTH_ORDER] =
        loom_verify_relation_element_width_order,
    [LOOM_RELATION_ELEMENT_WIDTH_AT_LEAST_ATTR] =
        loom_verify_relation_element_width_at_least_attr,
    [LOOM_RELATION_BIT_RANGE_WITHIN_ELEMENT_WIDTH] =
        loom_verify_relation_bit_range_within_element_width,
    [LOOM_RELATION_TOTAL_BIT_COUNT_EQUAL] =
        loom_verify_relation_total_bit_count_equal,
    [LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE] =
        loom_verify_relation_payload_bit_count_matches_storage,
    [LOOM_RELATION_COUNT_MATCHES_RANK] =
        loom_verify_relation_count_matches_rank,
    [LOOM_RELATION_COUNT_MATCHES_STATIC_ELEMENT_COUNT] =
        loom_verify_relation_count_matches_static_element_count,
    [LOOM_RELATION_ATTR_IN_RANGE_RANK] =
        loom_verify_relation_attr_in_range_rank,
    [LOOM_RELATION_REGION_ARG_COUNT] = loom_verify_relation_region_arg_count,
    [LOOM_RELATION_REGION_ARG_MATCH] = loom_verify_relation_region_arg_match,
    [LOOM_RELATION_YIELD_COUNT] = loom_verify_relation_yield_count,
    [LOOM_RELATION_YIELD_MATCH] = loom_verify_relation_yield_match,
    [LOOM_RELATION_VARIADIC_MATCH] = loom_verify_relation_variadic_match,
    [LOOM_RELATION_LAST_AXIS_GROUPED_BY] =
        loom_verify_relation_last_axis_grouped_by,
    [LOOM_RELATION_REGISTER_UNIT_COUNT_SUM] =
        loom_verify_relation_register_unit_count_sum,
};
static_assert(IREE_ARRAYSIZE(kVerifyRelationFns) == LOOM_RELATION_COUNT_,
              "verify relation dispatch table out of sync with enum");

static void loom_verify_semantic_constraint(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->relation >= LOOM_RELATION_COUNT_) return;
  kVerifyRelationFns[constraint->relation](state, op, vtable, constraint);
}

void loom_verify_semantic_constraints(loom_verify_state_t* state,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable) {
  if (!vtable->constraints || vtable->constraint_count == 0) return;
  for (uint8_t i = 0; i < vtable->constraint_count; ++i) {
    if (loom_verify_at_error_limit(state)) return;
    loom_verify_semantic_constraint(state, op, vtable, &vtable->constraints[i]);
  }
}
