// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/value_materializer.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/base/internal/prng.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/numpy_io.h"
#include "loom/util/math.h"

enum {
  LOOM_TESTBENCH_MAX_SHAPE_RANK = 15,
};

typedef enum loom_testbench_generator_kind_e {
  LOOM_TESTBENCH_GENERATOR_IOTA = 0,
  LOOM_TESTBENCH_GENERATOR_FILL = 1,
  LOOM_TESTBENCH_GENERATOR_RANDOM_UNIFORM = 2,
} loom_testbench_generator_kind_t;

typedef struct loom_testbench_generator_state_t {
  // Generated value kind.
  loom_testbench_generator_kind_t kind;
  // Scalar element type being generated.
  loom_scalar_type_t scalar_type;
  // Offset, fill payload, or random lower-bound payload.
  loom_attribute_t first_attr;
  // Step, fill payload, or random upper-bound payload.
  loom_attribute_t second_attr;
  // Deterministic random state for RANDOM_UNIFORM.
  iree_prng_xoroshiro128_state_t prng_state;
} loom_testbench_generator_state_t;

void loom_testbench_value_materializer_options_initialize(
    loom_testbench_value_materializer_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->host_allocator = iree_allocator_system();
}

static iree_status_t loom_testbench_value_table_add_slot_capacity(
    iree_host_size_t count, iree_host_size_t addend,
    iree_host_size_t* out_count) {
  if (!iree_host_size_checked_add(count, addend, out_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "testbench value slot count overflow");
  }
  return iree_ok_status();
}

typedef iree_status_t (*loom_testbench_case_value_callback_fn_t)(
    void* user_data, loom_value_id_t value_id);

typedef struct loom_testbench_case_value_callback_t {
  // Function invoked for each value ID referenced by a case plan.
  loom_testbench_case_value_callback_fn_t fn;
  // Opaque caller-owned context passed to |fn|.
  void* user_data;
} loom_testbench_case_value_callback_t;

static iree_status_t loom_testbench_case_value_callback_invoke(
    loom_testbench_case_value_callback_t callback, loom_value_id_t value_id) {
  return callback.fn(callback.user_data, value_id);
}

static iree_status_t loom_testbench_case_plan_walk_type_dynamic_dimensions(
    loom_type_t type, loom_testbench_case_value_callback_t callback) {
  if (!loom_type_is_shaped(type)) {
    return iree_ok_status();
  }
  iree_host_size_t rank = loom_type_rank(type);
  for (iree_host_size_t dim_index = 0; dim_index < rank; ++dim_index) {
    if (!loom_type_dim_is_dynamic_at(type, dim_index)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
        callback, loom_type_dim_value_id_at(type, dim_index)));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_plan_walk_values(
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_value_callback_t callback) {
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
        callback, case_plan->parameters[parameter_index].value_id));
  }
  for (iree_host_size_t source_index = 0;
       source_index < case_plan->value_source_count; ++source_index) {
    const loom_testbench_value_source_plan_t* source =
        &case_plan->value_sources[source_index];
    IREE_RETURN_IF_ERROR(
        loom_testbench_case_value_callback_invoke(callback, source->value_id));
    if (source->kind == LOOM_TESTBENCH_VALUE_SOURCE_RANDOM_UNIFORM) {
      IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
          callback, source->random_uniform.seed_value_id));
    }
    IREE_RETURN_IF_ERROR(loom_testbench_case_plan_walk_type_dynamic_dimensions(
        source->type, callback));
  }
  for (iree_host_size_t file_write_index = 0;
       file_write_index < case_plan->file_write_count; ++file_write_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
        callback, case_plan->file_writes[file_write_index].value_id));
  }
  for (iree_host_size_t invocation_index = 0;
       invocation_index < case_plan->invocation_count; ++invocation_index) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[invocation_index];
    for (iree_host_size_t input_index = 0;
         input_index < invocation->input_count; ++input_index) {
      IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
          callback, invocation->input_value_ids[input_index]));
    }
    for (iree_host_size_t result_index = 0;
         result_index < invocation->result_count; ++result_index) {
      IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
          callback, invocation->result_value_ids[result_index]));
    }
  }
  for (iree_host_size_t expectation_index = 0;
       expectation_index < case_plan->expectation_count; ++expectation_index) {
    const loom_testbench_expectation_plan_t* expectation =
        &case_plan->expectations[expectation_index];
    IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
        callback, expectation->actual_value_id));
    if (expectation->expected_value_id != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
          callback, expectation->expected_value_id));
    }
    if (expectation->kind != LOOM_TESTBENCH_EXPECTATION_SHAPE) {
      continue;
    }
    for (iree_host_size_t dim_index = 0;
         dim_index < expectation->shape.dimension_value_count; ++dim_index) {
      IREE_RETURN_IF_ERROR(loom_testbench_case_value_callback_invoke(
          callback, expectation->shape.dimension_value_ids[dim_index]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_value_table_count_slot(
    void* user_data, loom_value_id_t value_id) {
  (void)value_id;
  iree_host_size_t* count = (iree_host_size_t*)user_data;
  return loom_testbench_value_table_add_slot_capacity(*count, 1, count);
}

static iree_status_t loom_testbench_value_table_max_slot_count(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t* out_max_slot_count) {
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_case_plan_walk_values(
      case_plan, (loom_testbench_case_value_callback_t){
                     .fn = loom_testbench_value_table_count_slot,
                     .user_data = &count,
                 }));
  *out_max_slot_count = count;
  return iree_ok_status();
}

static iree_host_size_t loom_testbench_value_table_lower_bound(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    bool* out_found) {
  iree_host_size_t lo = 0;
  iree_host_size_t hi = table->slot_count;
  while (lo < hi) {
    iree_host_size_t mid = lo + (hi - lo) / 2;
    if (table->slots[mid].value_id < value_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  *out_found = lo < table->slot_count && table->slots[lo].value_id == value_id;
  return lo;
}

static const loom_testbench_value_slot_t*
loom_testbench_value_table_lookup_slot(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id) {
  bool found = false;
  iree_host_size_t slot_index =
      loom_testbench_value_table_lower_bound(table, value_id, &found);
  return found ? &table->slots[slot_index] : NULL;
}

static loom_testbench_value_slot_t* loom_testbench_value_table_lookup_slot_mut(
    loom_testbench_value_table_t* table, loom_value_id_t value_id) {
  bool found = false;
  iree_host_size_t slot_index =
      loom_testbench_value_table_lower_bound(table, value_id, &found);
  return found ? &table->slots[slot_index] : NULL;
}

static iree_status_t loom_testbench_value_table_include_value_callback(
    void* user_data, loom_value_id_t value_id) {
  loom_testbench_value_table_t* table =
      (loom_testbench_value_table_t*)user_data;
  if (value_id >= table->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value ID %u is outside the module value table",
                            (unsigned)value_id);
  }
  bool found = false;
  iree_host_size_t slot_index =
      loom_testbench_value_table_lower_bound(table, value_id, &found);
  if (found) {
    return iree_ok_status();
  }
  IREE_ASSERT(table->slot_count < table->slot_capacity,
              "testbench value slot capacity must cover the case plan");
  memmove(&table->slots[slot_index + 1], &table->slots[slot_index],
          (table->slot_count - slot_index) * sizeof(*table->slots));
  table->slots[slot_index] = (loom_testbench_value_slot_t){
      .value_id = value_id,
      .flags = 0,
      .variant = iree_vm_variant_empty(),
  };
  ++table->slot_count;
  return iree_ok_status();
}

static iree_status_t loom_testbench_value_table_populate_slots(
    loom_testbench_value_table_t* table) {
  return loom_testbench_case_plan_walk_values(
      table->case_plan,
      (loom_testbench_case_value_callback_t){
          .fn = loom_testbench_value_table_include_value_callback,
          .user_data = table,
      });
}

iree_status_t loom_testbench_value_table_initialize(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_allocator_t host_allocator, loom_testbench_value_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));
  out_table->module = module;
  out_table->case_plan = case_plan;
  out_table->host_allocator = host_allocator;

  iree_host_size_t max_slot_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_testbench_value_table_max_slot_count(case_plan, &max_slot_count));
  if (max_slot_count == 0) {
    return iree_ok_status();
  }

  iree_status_t status = iree_allocator_malloc(
      host_allocator, max_slot_count * sizeof(*out_table->slots),
      (void**)&out_table->slots);
  if (iree_status_is_ok(status)) {
    out_table->slot_capacity = max_slot_count;
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_value_table_populate_slots(out_table);
  }
  if (!iree_status_is_ok(status)) {
    loom_testbench_value_table_deinitialize(out_table);
  }
  return status;
}

void loom_testbench_value_table_deinitialize(
    loom_testbench_value_table_t* table) {
  if (!table) {
    return;
  }
  loom_testbench_value_table_reset(table);
  iree_allocator_free(table->host_allocator, table->slots);
  memset(table, 0, sizeof(*table));
}

void loom_testbench_value_table_reset(loom_testbench_value_table_t* table) {
  if (!table->slots) {
    return;
  }
  for (iree_host_size_t slot_index = 0; slot_index < table->slot_count;
       ++slot_index) {
    loom_testbench_value_slot_t* slot = &table->slots[slot_index];
    if (iree_any_bit_set(slot->flags,
                         LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED)) {
      iree_vm_variant_reset(&slot->variant);
      slot->variant = iree_vm_variant_empty();
      slot->flags &= ~LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED;
    }
  }
}

bool loom_testbench_value_table_contains(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id) {
  const loom_testbench_value_slot_t* slot =
      loom_testbench_value_table_lookup_slot(table, value_id);
  return slot &&
         iree_any_bit_set(slot->flags, LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED);
}

iree_status_t loom_testbench_value_table_assign_move(
    loom_testbench_value_table_t* table, loom_value_id_t value_id,
    iree_vm_variant_t* variant) {
  loom_testbench_value_slot_t* slot =
      loom_testbench_value_table_lookup_slot_mut(table, value_id);
  if (!slot) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value ID %u is outside the case value table",
                            (unsigned)value_id);
  }
  if (iree_any_bit_set(slot->flags, LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED)) {
    iree_vm_variant_reset(&slot->variant);
  }
  slot->variant = *variant;
  slot->flags |= LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED;
  *variant = iree_vm_variant_empty();
  return iree_ok_status();
}

static iree_status_t loom_testbench_value_table_set_value(
    loom_testbench_value_table_t* table, loom_value_id_t value_id,
    iree_vm_value_t value) {
  iree_vm_variant_t variant = iree_vm_make_variant_value(value);
  return loom_testbench_value_table_assign_move(table, value_id, &variant);
}

iree_status_t loom_testbench_value_table_lookup_borrow(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    const iree_vm_variant_t** out_variant) {
  *out_variant = NULL;
  const loom_testbench_value_slot_t* slot =
      loom_testbench_value_table_lookup_slot(table, value_id);
  if (!slot ||
      !iree_any_bit_set(slot->flags, LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "value ID %u has not been materialized",
                            (unsigned)value_id);
  }
  *out_variant = &slot->variant;
  return iree_ok_status();
}

iree_status_t loom_testbench_value_table_lookup_retain(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    iree_vm_variant_t* out_variant) {
  *out_variant = iree_vm_variant_empty();
  const iree_vm_variant_t* borrowed_variant = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_borrow(
      table, value_id, &borrowed_variant));
  *out_variant = *borrowed_variant;
  if (iree_vm_variant_is_ref(*out_variant)) {
    iree_vm_ref_retain_inplace(&out_variant->ref);
  }
  return iree_ok_status();
}

static bool loom_testbench_attr_as_i64(loom_attribute_t attr,
                                       int64_t* out_value) {
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_testbench_attr_as_f64(loom_attribute_t attr,
                                       double* out_value) {
  if (attr.kind == LOOM_ATTR_F64) {
    *out_value = loom_attr_as_f64(attr);
    return true;
  }
  if (attr.kind == LOOM_ATTR_I64) {
    *out_value = (double)loom_attr_as_i64(attr);
    return true;
  }
  return false;
}

static iree_status_t loom_testbench_attr_to_vm_value(
    loom_attribute_t attr, loom_scalar_type_t scalar_type,
    iree_vm_value_t* out_value) {
  loom_attr_kind_t expected_kind = LOOM_ATTR_ABSENT;
  if (!loom_attr_matches_scalar_type(attr, scalar_type, &expected_kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "literal attribute kind %u does not match scalar type %s",
        (unsigned)attr.kind, loom_scalar_type_name(scalar_type));
  }

  int64_t integer_value = 0;
  double floating_value = 0.0;
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I1:
      if (attr.kind == LOOM_ATTR_BOOL) {
        *out_value = iree_vm_value_make_i8((int8_t)(attr.raw ? 1 : 0));
        return iree_ok_status();
      }
      if (loom_testbench_attr_as_i64(attr, &integer_value)) {
        *out_value = iree_vm_value_make_i8((int8_t)(integer_value != 0));
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_I8:
      if (loom_testbench_attr_as_i64(attr, &integer_value) &&
          integer_value >= INT8_MIN && integer_value <= INT8_MAX) {
        *out_value = iree_vm_value_make_i8((int8_t)integer_value);
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_I16:
      if (loom_testbench_attr_as_i64(attr, &integer_value) &&
          integer_value >= INT16_MIN && integer_value <= INT16_MAX) {
        *out_value = iree_vm_value_make_i16((int16_t)integer_value);
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_I32:
      if (loom_testbench_attr_as_i64(attr, &integer_value) &&
          integer_value >= INT32_MIN && integer_value <= INT32_MAX) {
        *out_value = iree_vm_value_make_i32((int32_t)integer_value);
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I64:
      if (loom_testbench_attr_as_i64(attr, &integer_value)) {
        *out_value = iree_vm_value_make_i64(integer_value);
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_F32:
      if (loom_testbench_attr_as_f64(attr, &floating_value)) {
        *out_value = iree_vm_value_make_f32((float)floating_value);
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_F64:
      if (loom_testbench_attr_as_f64(attr, &floating_value)) {
        *out_value = iree_vm_value_make_f64(floating_value);
        return iree_ok_status();
      }
      break;
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported scalar materialization type %s",
                          loom_scalar_type_name(scalar_type));
}

static iree_status_t loom_testbench_variant_as_nonnegative_dim(
    iree_vm_variant_t variant, iree_hal_dim_t* out_dim) {
  if (!iree_vm_variant_is_value(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dynamic dimension is not a scalar value");
  }
  iree_vm_value_t value = iree_vm_variant_value(variant);
  int64_t signed_value = 0;
  switch (value.type) {
    case IREE_VM_VALUE_TYPE_I8:
      signed_value = value.i8;
      break;
    case IREE_VM_VALUE_TYPE_I16:
      signed_value = value.i16;
      break;
    case IREE_VM_VALUE_TYPE_I32:
      signed_value = value.i32;
      break;
    case IREE_VM_VALUE_TYPE_I64:
      signed_value = value.i64;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dynamic dimension is not an integer value");
  }
  if (signed_value < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dynamic dimension is negative");
  }
  *out_dim = (iree_hal_dim_t)signed_value;
  return iree_ok_status();
}

static iree_status_t loom_testbench_resolve_shape(
    const loom_testbench_value_table_t* table, loom_type_t type,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank) {
  if (!loom_type_is_shaped(type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected shaped value type");
  }
  iree_host_size_t rank = loom_type_rank(type);
  if (rank > LOOM_TESTBENCH_MAX_SHAPE_RANK) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape rank %zu exceeds testbench limit %u", rank,
                            LOOM_TESTBENCH_MAX_SHAPE_RANK);
  }
  for (iree_host_size_t dim_index = 0; dim_index < rank; ++dim_index) {
    if (loom_type_dim_is_dynamic_at(type, dim_index)) {
      loom_value_id_t dim_value_id = loom_type_dim_value_id_at(type, dim_index);
      const iree_vm_variant_t* dim_variant = NULL;
      IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_borrow(
          table, dim_value_id, &dim_variant));
      IREE_RETURN_IF_ERROR(loom_testbench_variant_as_nonnegative_dim(
          *dim_variant, &out_shape[dim_index]));
    } else {
      int64_t static_size = loom_type_dim_static_size_at(type, dim_index);
      if (static_size < 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "static dimension is negative");
      }
      out_shape[dim_index] = (iree_hal_dim_t)static_size;
    }
  }
  *out_shape_rank = rank;
  return iree_ok_status();
}

static iree_status_t loom_testbench_scalar_type_to_hal_element_type(
    loom_scalar_type_t scalar_type, iree_hal_element_type_t* out_element_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_INDEX:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_64;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_UINT_64;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I1:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_BOOL_8;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I8:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_8;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I16:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I32:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_32;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I64:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_64;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F16:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_BF16:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F32:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F64:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported scalar type %u",
                              (unsigned)scalar_type);
  }
}

static iree_status_t loom_testbench_get_i64_value(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    int64_t* out_value) {
  const iree_vm_variant_t* variant = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_value_table_lookup_borrow(table, value_id, &variant));
  if (!iree_vm_variant_is_value(*variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value ID %u is not a scalar", (unsigned)value_id);
  }
  iree_vm_value_t value = iree_vm_variant_value(*variant);
  switch (value.type) {
    case IREE_VM_VALUE_TYPE_I8:
      *out_value = value.i8;
      return iree_ok_status();
    case IREE_VM_VALUE_TYPE_I16:
      *out_value = value.i16;
      return iree_ok_status();
    case IREE_VM_VALUE_TYPE_I32:
      *out_value = value.i32;
      return iree_ok_status();
    case IREE_VM_VALUE_TYPE_I64:
      *out_value = value.i64;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "value ID %u is not an integer scalar",
                              (unsigned)value_id);
  }
}

static uint64_t loom_testbench_random_bounded_u64(
    iree_prng_xoroshiro128_state_t* state, uint64_t bound) {
  if (bound == 0) {
    return 0;
  }
  return iree_prng_xoroshiro128starstar_next_uint64(state) % bound;
}

static double loom_testbench_random_unit_f64(
    iree_prng_xoroshiro128_state_t* state) {
  uint64_t bits = iree_prng_xoroshiro128starstar_next_uint64(state) >> 11;
  return (double)bits * (1.0 / 9007199254740992.0);
}

static bool loom_testbench_iota_i64_value(int64_t offset, int64_t step,
                                          iree_host_size_t index,
                                          int64_t* out_value) {
  if (index > (iree_host_size_t)INT64_MAX) {
    return false;
  }
  int64_t scaled_index = 0;
  return loom_checked_mul_i64((int64_t)index, step, &scaled_index) &&
         loom_checked_add_i64(offset, scaled_index, out_value);
}

#define LOOM_TESTBENCH_FILL_INT_TYPED(mapping, c_type, min_value, max_value, \
                                      expression)                            \
  do {                                                                       \
    c_type* values = (c_type*)(mapping)->contents.data;                      \
    iree_host_size_t count =                                                 \
        (mapping)->contents.data_length / sizeof(*values);                   \
    for (iree_host_size_t index = 0; index < count; ++index) {               \
      int64_t generated_value = (expression);                                \
      if (generated_value < (min_value) || generated_value > (max_value)) {  \
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                    \
                                "generated integer value is out of range");  \
      }                                                                      \
      values[index] = (c_type)generated_value;                               \
    }                                                                        \
    return iree_ok_status();                                                 \
  } while (0)

#define LOOM_TESTBENCH_FILL_IOTA_TYPED(mapping, c_type, min_value, max_value) \
  do {                                                                        \
    c_type* values = (c_type*)(mapping)->contents.data;                       \
    iree_host_size_t count =                                                  \
        (mapping)->contents.data_length / sizeof(*values);                    \
    for (iree_host_size_t index = 0; index < count; ++index) {                \
      int64_t generated_value = 0;                                            \
      if (!loom_testbench_iota_i64_value(first_value, second_value, index,    \
                                         &generated_value) ||                 \
          generated_value < (min_value) || generated_value > (max_value)) {   \
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                     \
                                "generated iota value is out of range");      \
      }                                                                       \
      values[index] = (c_type)generated_value;                                \
    }                                                                         \
    return iree_ok_status();                                                  \
  } while (0)

#define LOOM_TESTBENCH_FILL_FLOAT_TYPED(mapping, c_type, expression) \
  do {                                                               \
    c_type* values = (c_type*)(mapping)->contents.data;              \
    iree_host_size_t count =                                         \
        (mapping)->contents.data_length / sizeof(*values);           \
    for (iree_host_size_t index = 0; index < count; ++index) {       \
      values[index] = (c_type)(expression);                          \
    }                                                                \
    return iree_ok_status();                                         \
  } while (0)

#define LOOM_TESTBENCH_FILL_FLOAT16_TYPED(mapping, convert, expression) \
  do {                                                                  \
    uint16_t* values = (uint16_t*)(mapping)->contents.data;             \
    iree_host_size_t count =                                            \
        (mapping)->contents.data_length / sizeof(*values);              \
    for (iree_host_size_t index = 0; index < count; ++index) {          \
      values[index] = convert((float)(expression));                     \
    }                                                                   \
    return iree_ok_status();                                            \
  } while (0)

static iree_status_t loom_testbench_generate_integer_buffer(
    iree_hal_buffer_mapping_t* mapping, loom_testbench_generator_state_t* state,
    int64_t min_value, int64_t max_value) {
  int64_t first_value = 0;
  int64_t second_value = 0;
  if (!loom_testbench_attr_as_i64(state->first_attr, &first_value) ||
      !loom_testbench_attr_as_i64(state->second_attr, &second_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "integer generator requires i64 attributes");
  }
  if (first_value < min_value || first_value > max_value ||
      second_value < min_value || second_value > max_value) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "integer generator value is out of range");
  }

  switch (state->kind) {
    case LOOM_TESTBENCH_GENERATOR_IOTA:
      switch (state->scalar_type) {
        case LOOM_SCALAR_TYPE_I8:
          LOOM_TESTBENCH_FILL_IOTA_TYPED(mapping, int8_t, INT8_MIN, INT8_MAX);
        case LOOM_SCALAR_TYPE_I16:
          LOOM_TESTBENCH_FILL_IOTA_TYPED(mapping, int16_t, INT16_MIN,
                                         INT16_MAX);
        case LOOM_SCALAR_TYPE_I32:
          LOOM_TESTBENCH_FILL_IOTA_TYPED(mapping, int32_t, INT32_MIN,
                                         INT32_MAX);
        case LOOM_SCALAR_TYPE_INDEX:
        case LOOM_SCALAR_TYPE_OFFSET:
        case LOOM_SCALAR_TYPE_I64:
          LOOM_TESTBENCH_FILL_IOTA_TYPED(mapping, int64_t, min_value,
                                         max_value);
        default:
          break;
      }
      break;
    case LOOM_TESTBENCH_GENERATOR_FILL:
      switch (state->scalar_type) {
        case LOOM_SCALAR_TYPE_I8:
          LOOM_TESTBENCH_FILL_INT_TYPED(mapping, int8_t, INT8_MIN, INT8_MAX,
                                        first_value);
        case LOOM_SCALAR_TYPE_I16:
          LOOM_TESTBENCH_FILL_INT_TYPED(mapping, int16_t, INT16_MIN, INT16_MAX,
                                        first_value);
        case LOOM_SCALAR_TYPE_I32:
          LOOM_TESTBENCH_FILL_INT_TYPED(mapping, int32_t, INT32_MIN, INT32_MAX,
                                        first_value);
        case LOOM_SCALAR_TYPE_INDEX:
        case LOOM_SCALAR_TYPE_OFFSET:
        case LOOM_SCALAR_TYPE_I64:
          LOOM_TESTBENCH_FILL_INT_TYPED(mapping, int64_t, min_value, max_value,
                                        first_value);
        default:
          break;
      }
      break;
    case LOOM_TESTBENCH_GENERATOR_RANDOM_UNIFORM: {
      if (second_value < first_value) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "random integer upper bound is below lower");
      }
      if (second_value == INT64_MAX && first_value == INT64_MIN) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "random integer range is too wide");
      }
      uint64_t width = (uint64_t)(second_value - first_value) + 1;
      switch (state->scalar_type) {
        case LOOM_SCALAR_TYPE_I8:
          LOOM_TESTBENCH_FILL_INT_TYPED(
              mapping, int8_t, INT8_MIN, INT8_MAX,
              first_value + (int64_t)loom_testbench_random_bounded_u64(
                                &state->prng_state, width));
        case LOOM_SCALAR_TYPE_I16:
          LOOM_TESTBENCH_FILL_INT_TYPED(
              mapping, int16_t, INT16_MIN, INT16_MAX,
              first_value + (int64_t)loom_testbench_random_bounded_u64(
                                &state->prng_state, width));
        case LOOM_SCALAR_TYPE_I32:
          LOOM_TESTBENCH_FILL_INT_TYPED(
              mapping, int32_t, INT32_MIN, INT32_MAX,
              first_value + (int64_t)loom_testbench_random_bounded_u64(
                                &state->prng_state, width));
        case LOOM_SCALAR_TYPE_INDEX:
        case LOOM_SCALAR_TYPE_OFFSET:
        case LOOM_SCALAR_TYPE_I64:
          LOOM_TESTBENCH_FILL_INT_TYPED(
              mapping, int64_t, min_value, max_value,
              first_value + (int64_t)loom_testbench_random_bounded_u64(
                                &state->prng_state, width));
        default:
          break;
      }
      break;
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported integer generator type %s",
                          loom_scalar_type_name(state->scalar_type));
}

static iree_status_t loom_testbench_generate_float_buffer(
    iree_hal_buffer_mapping_t* mapping,
    loom_testbench_generator_state_t* state) {
  double first_value = 0.0;
  double second_value = 0.0;
  if (!loom_testbench_attr_as_f64(state->first_attr, &first_value) ||
      !loom_testbench_attr_as_f64(state->second_attr, &second_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "floating-point generator requires numeric attrs");
  }
  switch (state->kind) {
    case LOOM_TESTBENCH_GENERATOR_IOTA:
      switch (state->scalar_type) {
        case LOOM_SCALAR_TYPE_F16:
          LOOM_TESTBENCH_FILL_FLOAT16_TYPED(
              mapping, iree_math_f32_to_f16,
              first_value + (double)index * second_value);
        case LOOM_SCALAR_TYPE_BF16:
          LOOM_TESTBENCH_FILL_FLOAT16_TYPED(
              mapping, iree_math_f32_to_bf16,
              first_value + (double)index * second_value);
        case LOOM_SCALAR_TYPE_F32:
          LOOM_TESTBENCH_FILL_FLOAT_TYPED(
              mapping, float, first_value + (double)index * second_value);
        case LOOM_SCALAR_TYPE_F64:
          LOOM_TESTBENCH_FILL_FLOAT_TYPED(
              mapping, double, first_value + (double)index * second_value);
        default:
          break;
      }
      break;
    case LOOM_TESTBENCH_GENERATOR_FILL:
      switch (state->scalar_type) {
        case LOOM_SCALAR_TYPE_F16:
          LOOM_TESTBENCH_FILL_FLOAT16_TYPED(mapping, iree_math_f32_to_f16,
                                            first_value);
        case LOOM_SCALAR_TYPE_BF16:
          LOOM_TESTBENCH_FILL_FLOAT16_TYPED(mapping, iree_math_f32_to_bf16,
                                            first_value);
        case LOOM_SCALAR_TYPE_F32:
          LOOM_TESTBENCH_FILL_FLOAT_TYPED(mapping, float, first_value);
        case LOOM_SCALAR_TYPE_F64:
          LOOM_TESTBENCH_FILL_FLOAT_TYPED(mapping, double, first_value);
        default:
          break;
      }
      break;
    case LOOM_TESTBENCH_GENERATOR_RANDOM_UNIFORM:
      if (second_value < first_value) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "random floating-point upper bound is below "
                                "lower");
      }
      switch (state->scalar_type) {
        case LOOM_SCALAR_TYPE_F16:
          LOOM_TESTBENCH_FILL_FLOAT16_TYPED(
              mapping, iree_math_f32_to_f16,
              first_value + loom_testbench_random_unit_f64(&state->prng_state) *
                                (second_value - first_value));
        case LOOM_SCALAR_TYPE_BF16:
          LOOM_TESTBENCH_FILL_FLOAT16_TYPED(
              mapping, iree_math_f32_to_bf16,
              first_value + loom_testbench_random_unit_f64(&state->prng_state) *
                                (second_value - first_value));
        case LOOM_SCALAR_TYPE_F32:
          LOOM_TESTBENCH_FILL_FLOAT_TYPED(
              mapping, float,
              first_value + loom_testbench_random_unit_f64(&state->prng_state) *
                                (second_value - first_value));
        case LOOM_SCALAR_TYPE_F64:
          LOOM_TESTBENCH_FILL_FLOAT_TYPED(
              mapping, double,
              first_value + loom_testbench_random_unit_f64(&state->prng_state) *
                                (second_value - first_value));
        default:
          break;
      }
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported floating-point generator type %s",
                          loom_scalar_type_name(state->scalar_type));
}

static iree_status_t loom_testbench_generate_buffer(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  loom_testbench_generator_state_t* state =
      (loom_testbench_generator_state_t*)user_data;
  switch (state->scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      return loom_testbench_generate_integer_buffer(mapping, state, INT8_MIN,
                                                    INT8_MAX);
    case LOOM_SCALAR_TYPE_I16:
      return loom_testbench_generate_integer_buffer(mapping, state, INT16_MIN,
                                                    INT16_MAX);
    case LOOM_SCALAR_TYPE_I32:
      return loom_testbench_generate_integer_buffer(mapping, state, INT32_MIN,
                                                    INT32_MAX);
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_I64:
      return loom_testbench_generate_integer_buffer(mapping, state, INT64_MIN,
                                                    INT64_MAX);
    case LOOM_SCALAR_TYPE_OFFSET:
      return loom_testbench_generate_integer_buffer(mapping, state, 0,
                                                    INT64_MAX);
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return loom_testbench_generate_float_buffer(mapping, state);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported generated scalar type %s",
                              loom_scalar_type_name(state->scalar_type));
  }
}

#undef LOOM_TESTBENCH_FILL_FLOAT_TYPED
#undef LOOM_TESTBENCH_FILL_FLOAT16_TYPED
#undef LOOM_TESTBENCH_FILL_INT_TYPED
#undef LOOM_TESTBENCH_FILL_IOTA_TYPED

static iree_hal_buffer_params_t loom_testbench_default_buffer_params(void) {
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  return buffer_params;
}

static iree_hal_buffer_params_t loom_testbench_buffer_params(
    const loom_testbench_value_materializer_options_t* options) {
  if (options->buffer_params.usage == IREE_HAL_BUFFER_USAGE_NONE &&
      options->buffer_params.access == IREE_HAL_MEMORY_ACCESS_NONE &&
      options->buffer_params.type == IREE_HAL_MEMORY_TYPE_NONE) {
    return loom_testbench_default_buffer_params();
  }
  return options->buffer_params;
}

static iree_status_t loom_testbench_materialize_generated_source(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_value_source_plan_t* source,
    loom_testbench_generator_state_t* generator_state,
    loom_testbench_value_table_t* table) {
  if (!options->device_allocator) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "shaped value generation requires a HAL allocator");
  }
  if (!loom_type_is_shaped(source->type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.generate.* requires a shaped result type");
  }

  iree_hal_dim_t shape[LOOM_TESTBENCH_MAX_SHAPE_RANK] = {0};
  iree_host_size_t shape_rank = 0;
  IREE_RETURN_IF_ERROR(
      loom_testbench_resolve_shape(table, source->type, shape, &shape_rank));

  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(loom_testbench_scalar_type_to_hal_element_type(
      loom_type_element_type(source->type), &element_type));

  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_generate_buffer(
      options->device, options->device_allocator, shape_rank, shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      loom_testbench_buffer_params(options), loom_testbench_generate_buffer,
      generator_state, &buffer_view));

  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  iree_vm_variant_t variant = iree_vm_make_variant_ref_assign(buffer_view_ref);
  return loom_testbench_value_table_assign_move(table, source->value_id,
                                                &variant);
}

static iree_status_t loom_testbench_validate_buffer_view_type(
    const loom_testbench_value_table_t* table, loom_type_t expected_type,
    iree_hal_buffer_view_t* buffer_view) {
  if (!loom_type_is_shaped(expected_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file-backed value requires a shaped result type");
  }

  iree_hal_dim_t expected_shape[LOOM_TESTBENCH_MAX_SHAPE_RANK] = {0};
  iree_host_size_t expected_rank = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_resolve_shape(
      table, expected_type, expected_shape, &expected_rank));
  iree_host_size_t actual_rank = iree_hal_buffer_view_shape_rank(buffer_view);
  if (actual_rank != expected_rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY rank %zu does not match expected rank %zu",
                            actual_rank, expected_rank);
  }
  for (iree_host_size_t dim_index = 0; dim_index < expected_rank; ++dim_index) {
    if (iree_hal_buffer_view_shape_dim(buffer_view, dim_index) !=
        expected_shape[dim_index]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "NPY dimension %zu does not match expected shape",
                              dim_index);
    }
  }

  iree_hal_element_type_t expected_element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(loom_testbench_scalar_type_to_hal_element_type(
      loom_type_element_type(expected_type), &expected_element_type));
  iree_hal_element_type_t actual_element_type =
      iree_hal_buffer_view_element_type(buffer_view);
  if (actual_element_type != expected_element_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY element type 0x%08x does not match expected "
                            "element type 0x%08x",
                            actual_element_type, expected_element_type);
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_materialize_file_read_npy(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_value_source_plan_t* source,
    loom_testbench_value_table_t* table) {
  if (!options->device_allocator) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "NPY reads require a HAL allocator");
  }
  if (!options->open_read_file.fn) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no check.file.read provider is configured");
  }

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(options->open_read_file.fn(
      options->open_read_file.user_data, source->file.path, &stream));
  if (!stream) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "file provider returned NULL stream");
  }

  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_status_t status = iree_numpy_npy_load_ndarray(
      stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT,
      loom_testbench_buffer_params(options), options->device,
      options->device_allocator, &buffer_view);
  iree_io_stream_release(stream);

  if (iree_status_is_ok(status)) {
    status = loom_testbench_validate_buffer_view_type(table, source->type,
                                                      buffer_view);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(buffer_view);
    return status;
  }
  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  iree_vm_variant_t variant = iree_vm_make_variant_ref_assign(buffer_view_ref);
  return loom_testbench_value_table_assign_move(table, source->value_id,
                                                &variant);
}

static iree_status_t loom_testbench_materialize_value_source(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_value_source_plan_t* source,
    loom_testbench_value_table_t* table) {
  switch (source->kind) {
    case LOOM_TESTBENCH_VALUE_SOURCE_LITERAL: {
      if (!loom_type_is_scalar(source->type)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "check.literal requires a scalar result type");
      }
      iree_vm_value_t value = iree_vm_value_make_none();
      IREE_RETURN_IF_ERROR(loom_testbench_attr_to_vm_value(
          source->literal.value, loom_type_element_type(source->type), &value));
      return loom_testbench_value_table_set_value(table, source->value_id,
                                                  value);
    }
    case LOOM_TESTBENCH_VALUE_SOURCE_IOTA: {
      loom_testbench_generator_state_t generator_state = {
          .kind = LOOM_TESTBENCH_GENERATOR_IOTA,
          .scalar_type = loom_type_element_type(source->type),
          .first_attr = source->iota.offset,
          .second_attr = source->iota.step,
      };
      return loom_testbench_materialize_generated_source(
          options, source, &generator_state, table);
    }
    case LOOM_TESTBENCH_VALUE_SOURCE_FILL: {
      loom_testbench_generator_state_t generator_state = {
          .kind = LOOM_TESTBENCH_GENERATOR_FILL,
          .scalar_type = loom_type_element_type(source->type),
          .first_attr = source->fill.value,
          .second_attr = source->fill.value,
      };
      return loom_testbench_materialize_generated_source(
          options, source, &generator_state, table);
    }
    case LOOM_TESTBENCH_VALUE_SOURCE_RANDOM_UNIFORM: {
      int64_t seed = 0;
      IREE_RETURN_IF_ERROR(loom_testbench_get_i64_value(
          table, source->random_uniform.seed_value_id, &seed));
      loom_testbench_generator_state_t generator_state = {
          .kind = LOOM_TESTBENCH_GENERATOR_RANDOM_UNIFORM,
          .scalar_type = loom_type_element_type(source->type),
          .first_attr = source->random_uniform.lower,
          .second_attr = source->random_uniform.upper,
      };
      iree_prng_xoroshiro128_initialize((uint64_t)seed,
                                        &generator_state.prng_state);
      return loom_testbench_materialize_generated_source(
          options, source, &generator_state, table);
    }
    case LOOM_TESTBENCH_VALUE_SOURCE_FILE_READ_NPY:
      return loom_testbench_materialize_file_read_npy(options, source, table);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invalid value source plan kind %u",
                              (unsigned)source->kind);
  }
}

iree_status_t loom_testbench_materialize_case_sample(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_testbench_value_table_t* table) {
  if (sample_ordinal >= case_plan->cartesian_sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sample ordinal %zu exceeds case cartesian sample "
                            "count %zu",
                            sample_ordinal, case_plan->cartesian_sample_count);
  }
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[parameter_index];
    iree_host_size_t parameter_sample_ordinal =
        loom_testbench_case_sample_parameter_ordinal(case_plan, sample_ordinal,
                                                     parameter_index);
    loom_attribute_t attr = {0};
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        parameter, parameter_sample_ordinal, &attr));
    if (!loom_type_is_scalar(parameter->type)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "check.param.* requires a scalar result type");
    }
    iree_vm_value_t value = iree_vm_value_make_none();
    IREE_RETURN_IF_ERROR(loom_testbench_attr_to_vm_value(
        attr, loom_type_element_type(parameter->type), &value));
    IREE_RETURN_IF_ERROR(loom_testbench_value_table_set_value(
        table, parameter->value_id, value));
  }

  for (iree_host_size_t source_index = 0;
       source_index < case_plan->value_source_count; ++source_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_materialize_value_source(
        options, &case_plan->value_sources[source_index], table));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_write_single_file(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_file_write_plan_t* file_write,
    const loom_testbench_value_table_t* table) {
  if (!options->open_write_file.fn) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no check.file.write provider is configured");
  }
  iree_vm_variant_t variant = iree_vm_variant_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_retain(
      table, file_write->value_id, &variant));
  if (!iree_vm_variant_is_ref(variant)) {
    iree_vm_variant_reset(&variant);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.file.write.npy requires a buffer view "
                            "value");
  }
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_status_t status =
      iree_hal_buffer_view_check_deref(variant.ref, &buffer_view);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_validate_buffer_view_type(table, file_write->type,
                                                      buffer_view);
  }

  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = options->open_write_file.fn(options->open_write_file.user_data,
                                         file_write->path, &stream);
    if (iree_status_is_ok(status) && !stream) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "file provider returned NULL stream");
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_numpy_npy_save_ndarray(stream, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT,
                                    buffer_view, options->host_allocator);
  }
  iree_io_stream_release(stream);
  iree_vm_variant_reset(&variant);
  return status;
}

iree_status_t loom_testbench_write_case_files(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_value_table_t* table, bool case_failed) {
  for (iree_host_size_t file_write_index = 0;
       file_write_index < case_plan->file_write_count; ++file_write_index) {
    const loom_testbench_file_write_plan_t* file_write =
        &case_plan->file_writes[file_write_index];
    if (file_write->mode == LOOM_CHECK_FILE_WRITE_NPY_MODE_ON_FAILURE &&
        !case_failed) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_testbench_write_single_file(options, file_write, table));
  }
  return iree_ok_status();
}
