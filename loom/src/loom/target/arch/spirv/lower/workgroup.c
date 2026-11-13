// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/workgroup.h"

#include <stdint.h>

#include "loom/analysis/storage_layout.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/lower/workgroup_layout.h"
#include "loom/target/arch/spirv/lower/workgroup_reduce.h"
#include "loom/util/fact_table.h"

typedef enum loom_spirv_workgroup_plan_kind_e {
  LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA = 1,
  LOOM_SPIRV_WORKGROUP_PLAN_VIEW = 2,
  LOOM_SPIRV_WORKGROUP_PLAN_COMPARE = 3,
} loom_spirv_workgroup_plan_kind_t;

typedef struct loom_spirv_workgroup_alloca_plan_t {
  // Static byte length of the Workgroup allocation.
  int64_t byte_length;
  // Static byte alignment of the Workgroup allocation.
  int64_t byte_alignment;
  // Whether the allocation uses its scalar carrier's shared physical arena.
  bool packed;
} loom_spirv_workgroup_alloca_plan_t;

typedef struct loom_spirv_workgroup_view_plan_t {
  // Source storage-root value that has already lowered to low.storage.
  loom_value_id_t root_value_id;
} loom_spirv_workgroup_view_plan_t;

typedef struct loom_spirv_workgroup_compare_plan_t {
  // Exact source allocation roots; INVALID denotes a different storage domain.
  loom_value_id_t roots[2];
} loom_spirv_workgroup_compare_plan_t;

static bool loom_spirv_workgroup_i64_is_power_of_two(int64_t value) {
  return value > 0 && value <= UINT32_MAX &&
         iree_math_is_power_of_two_i64(value);
}

static iree_status_t loom_spirv_workgroup_emit_rejected(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_type_t type,
    loom_spirv_workgroup_plan_kind_t plan_kind,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_make(plan_kind, NULL);
  return loom_low_lower_emit_source_type_unsupported(context, source_op,
                                                     field_name, type);
}

static bool loom_spirv_workgroup_view_reference(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_fact_view_reference_t* out_reference) {
  *out_reference = (loom_value_fact_view_reference_t){0};
  if (!loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, value_id), out_reference)) {
    return false;
  }
  out_reference->root_value_id =
      loom_value_fact_view_reference_resolve_root_value(*out_reference,
                                                        value_id);
  return out_reference->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
}

static iree_status_t loom_spirv_workgroup_view_plan_from_facts(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_spirv_workgroup_view_plan_t* out_plan, bool* out_selected,
    bool* out_has_workgroup_facts) {
  *out_plan = (loom_spirv_workgroup_view_plan_t){0};
  *out_selected = false;
  *out_has_workgroup_facts = false;
  const loom_value_id_t result = loom_buffer_view_result(source_op);
  uint16_t array_pointer_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_spirv_resolve_workgroup_view_reg_class(
      context, result, out_has_workgroup_facts, &array_pointer_reg_class_id));
  if (!*out_has_workgroup_facts ||
      array_pointer_reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
    return iree_ok_status();
  }
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_spirv_workgroup_view_reference(
          loom_low_lower_context_fact_table(context), result, &reference)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "selected SPIR-V Workgroup view has no Workgroup reference");
  }
  out_plan->root_value_id = reference.root_value_id;
  loom_type_t array_pointer_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, array_pointer_reg_class_id, 1, &array_pointer_type));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_plan_value_type(context, result, array_pointer_type));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_spirv_select_workgroup_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  if (loom_buffer_alloca_pool(source_op) != LOOM_VALUE_ID_INVALID ||
      loom_buffer_alloca_memory_space(source_op) !=
          LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  bool lifetime_supported = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_function_storage_check_lifetime(
      context, source_op, &lifetime_supported));
  if (!lifetime_supported) {
    *out_plan =
        loom_low_lower_plan_make(LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA, NULL);
    return iree_ok_status();
  }
  loom_spirv_workgroup_alloca_plan_t plan = {0};
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (!loom_value_facts_as_non_negative_i64_maximum(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(source_op)),
          &plan.byte_length) ||
      plan.byte_length <= 0) {
    *out_plan =
        loom_low_lower_plan_make(LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA, NULL);
    return loom_low_lower_emit_function_storage_extent_unsupported(
        context, source_op, LOOM_STORAGE_SPACE_WORKGROUP,
        loom_buffer_alloca_byte_length(source_op));
  }
  if (!loom_spirv_workgroup_i64_is_power_of_two(
          loom_buffer_alloca_base_alignment(source_op))) {
    return loom_spirv_workgroup_emit_rejected(
        context, source_op, IREE_SV("workgroup_storage"),
        loom_module_value_type(loom_low_lower_context_module(context),
                               loom_buffer_alloca_result(source_op)),
        LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA, out_plan);
  }
  plan.byte_alignment = loom_buffer_alloca_base_alignment(source_op);
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_layout_record_alloca(
      context, source_op, (uint64_t)plan.byte_length,
      (uint64_t)plan.byte_alignment, &plan.packed));

  loom_spirv_workgroup_alloca_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  *plan_data = plan;
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_buffer_alloca_result(source_op),
      loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP)));
  *out_plan =
      loom_low_lower_plan_make(LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA, plan_data);
  return iree_ok_status();
}

static iree_status_t loom_spirv_select_workgroup_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  loom_spirv_workgroup_view_plan_t plan = {0};
  bool selected = false;
  bool has_workgroup_facts = false;
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_view_plan_from_facts(
      context, source_op, &plan, &selected, &has_workgroup_facts));
  if (!selected) {
    if (has_workgroup_facts) {
      return loom_spirv_workgroup_emit_rejected(
          context, source_op, IREE_SV("workgroup_view"),
          loom_module_value_type(loom_low_lower_context_module(context),
                                 loom_buffer_view_result(source_op)),
          LOOM_SPIRV_WORKGROUP_PLAN_VIEW, out_plan);
    }
    return iree_ok_status();
  }
  loom_spirv_workgroup_view_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  *plan_data = plan;
  *out_plan =
      loom_low_lower_plan_make(LOOM_SPIRV_WORKGROUP_PLAN_VIEW, plan_data);
  return iree_ok_status();
}

static iree_status_t loom_spirv_select_workgroup_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_spirv_workgroup_compare_plan_t plan = {
      .roots = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID}};
  bool has_workgroup = false;
  bool supported = true;
  const loom_value_id_t values[] = {loom_buffer_compare_address_lhs(source_op),
                                    loom_buffer_compare_address_rhs(source_op)};
  for (unsigned i = 0; i < 2; ++i) {
    loom_value_fact_buffer_reference_t reference = {0};
    if (!loom_value_facts_query_buffer_reference(
            &facts->context, loom_value_fact_table_lookup(facts, values[i]),
            &reference) ||
        reference.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      continue;
    }
    has_workgroup = true;
    // Exact origin identity is stronger than sharing an allocation. It cannot
    // substitute an unrelated byte origin introduced by same_root assertions.
    const loom_value_id_t root =
        loom_value_fact_table_query_identity(facts, values[i]);
    const loom_value_t* value = loom_module_value(module, root);
    if (loom_value_is_block_arg(value) ||
        !loom_buffer_alloca_isa(loom_value_def_op(value))) {
      supported = false;
    } else {
      plan.roots[i] = root;
    }
  }
  if (!has_workgroup) {
    return iree_ok_status();
  }
  if (!supported) {
    return loom_spirv_workgroup_emit_rejected(
        context, source_op, IREE_SV("workgroup_position"), loom_type_buffer(),
        LOOM_SPIRV_WORKGROUP_PLAN_COMPARE, out_plan);
  }
  loom_spirv_workgroup_compare_plan_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*data), (void**)&data));
  *data = plan;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, loom_buffer_compare_address_result(source_op),
      &result_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_buffer_compare_address_result(source_op), result_type));
  *out_plan = loom_low_lower_plan_make(LOOM_SPIRV_WORKGROUP_PLAN_COMPARE, data);
  return iree_ok_status();
}

iree_status_t loom_spirv_select_workgroup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_spirv_select_workgroup_alloca(context, source_op, out_plan);
    case LOOM_OP_BUFFER_COMPARE_ADDRESS:
      return loom_spirv_select_workgroup_compare(context, source_op, out_plan);
    case LOOM_OP_BUFFER_VIEW:
      return loom_spirv_select_workgroup_view(context, source_op, out_plan);
    default:
      return iree_ok_status();
  }
}

void loom_spirv_mark_workgroup_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  (void)source_op;
  switch ((loom_spirv_workgroup_plan_kind_t)plan.id) {
    case LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA:
      return;
    case LOOM_SPIRV_WORKGROUP_PLAN_COMPARE: {
      const loom_spirv_workgroup_compare_plan_t* compare = plan.target_data;
      for (unsigned i = 0; i < 2; ++i) {
        if (compare->roots[i] != LOOM_VALUE_ID_INVALID) {
          loom_low_lower_require_source_value_storage(context,
                                                      compare->roots[i]);
        }
      }
      loom_low_lower_require_source_value_storage(
          context, loom_buffer_compare_address_lhs_offset(source_op));
      loom_low_lower_require_source_value_storage(
          context, loom_buffer_compare_address_rhs_offset(source_op));
      return;
    }
    case LOOM_SPIRV_WORKGROUP_PLAN_VIEW: {
      const loom_spirv_workgroup_view_plan_t* view_plan =
          (const loom_spirv_workgroup_view_plan_t*)plan.target_data;
      loom_low_lower_require_source_value_storage(context,
                                                  view_plan->root_value_id);
      return;
    }
  }
  IREE_ASSERT_UNREACHABLE("SPIR-V Workgroup plan selected unknown op kind");
}

static bool loom_spirv_workgroup_measure_storage(
    loom_low_lower_context_t* context, uint64_t* out_byte_extent) {
  *out_byte_extent = 0;
  const iree_host_size_t storage_root_count =
      loom_spirv_workgroup_layout_storage_root_count(context);
  for (iree_host_size_t i = 0; i < storage_root_count; ++i) {
    const loom_spirv_workgroup_storage_root_requirement_t requirement =
        loom_spirv_workgroup_layout_storage_root_requirement(context, i);
    if (!loom_storage_layout_append(requirement.byte_length,
                                    requirement.byte_alignment, out_byte_extent,
                                    /*out_byte_offset=*/NULL)) {
      return false;
    }
  }

  const iree_host_size_t selected_plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < selected_plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    if (selected_plan.plan.id == LOOM_OP_KERNEL_WORKGROUP_REDUCE) {
      if (!loom_spirv_workgroup_reduce_measure_storage(selected_plan.plan,
                                                       out_byte_extent)) {
        return false;
      }
      continue;
    }
    if (selected_plan.plan.id != LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA) {
      continue;
    }
    const loom_spirv_workgroup_alloca_plan_t* alloca_plan =
        (const loom_spirv_workgroup_alloca_plan_t*)
            selected_plan.plan.target_data;
    IREE_ASSERT(alloca_plan != NULL);
    if (alloca_plan->packed) {
      continue;
    }
    if (!loom_storage_layout_append((uint64_t)alloca_plan->byte_length,
                                    (uint64_t)alloca_plan->byte_alignment,
                                    out_byte_extent,
                                    /*out_byte_offset=*/NULL)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_spirv_plan_workgroup_entry_setup(
    loom_low_lower_context_t* context) {
  uint64_t byte_extent = 0;
  if (!loom_spirv_workgroup_measure_storage(context, &byte_extent)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_lower_context_function_name(context)),
        loom_param_string(IREE_SV("workgroup")),
    };
    return loom_low_lower_emit_error_ref(
        context, loom_low_lower_context_source_function(context).op,
        LOOM_ERR_TARGET_096_REF, params, IREE_ARRAYSIZE(params));
  }
  const uint64_t byte_limit = loom_low_lower_context_bundle(context)
                                  ->snapshot->max_workgroup_storage_bytes;
  if (byte_limit != 0 && byte_extent > byte_limit) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_lower_context_function_name(context)),
        loom_param_string(loom_low_lower_context_target_key(context)),
        loom_param_u64(byte_extent),
        loom_param_u64(byte_limit),
    };
    return loom_low_lower_emit_error_ref(
        context, loom_low_lower_context_source_function(context).op,
        LOOM_ERR_TARGET_051_REF, params, IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_lower_workgroup_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_alloca_plan_t* plan) {
  if (plan->packed) {
    loom_value_id_t low_storage = LOOM_VALUE_ID_INVALID;
    loom_spirv_workgroup_layout_lookup_low_storage(
        context, loom_buffer_alloca_result(source_op), &low_storage);
    return loom_low_lower_bind_value(
        context, loom_buffer_alloca_result(source_op), low_storage);
  }
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      loom_low_lower_context_builder(context), plan->byte_length,
      plan->byte_alignment, loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP),
      source_op->location, &storage_op));
  return loom_low_lower_bind_value(
      context, loom_buffer_alloca_result(source_op),
      loom_low_storage_reserve_storage(storage_op));
}

static iree_status_t loom_spirv_lower_workgroup_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_view_plan_t* plan) {
  loom_value_id_t low_storage =
      loom_low_lower_lookup_value(context, plan->root_value_id);
  const loom_type_t array_pointer_type = loom_low_lower_value_binding_type(
      context, loom_buffer_view_result(source_op));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      loom_low_lower_context_builder(context), low_storage, /*offset=*/0,
      array_pointer_type, source_op->location, &address_op));
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   loom_low_storage_address_result(address_op));
}

// Apply the arena offset retained by storage planning to an authored byte
// position. No pointer bits or storage-layout reconstruction are required.
static iree_status_t loom_spirv_lower_workgroup_position(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t root, loom_value_id_t offset,
    loom_value_id_t* out_position) {
  *out_position = loom_low_lower_lookup_value(context, offset);
  const uint64_t bias =
      loom_spirv_workgroup_layout_root_byte_offset(context, root);
  if (!bias) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptors =
      loom_low_lower_context_descriptor_set(context);
  const loom_type_t offset_type =
      loom_low_lower_value_binding_type(context, offset);
  loom_string_id_t key;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("offset64_value"), &key));
  const loom_named_attr_t attribute = {.name_id = key,
                                       .value = loom_attr_i64((int64_t)bias)};
  const loom_low_lower_resolved_descriptor_t constant = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          descriptors, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_OFFSET64)};
  loom_op_t* emitted = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &constant, loom_make_named_attr_slice(&attribute, 1),
      offset_type, source_op->location, &emitted));
  const loom_value_id_t operands[] = {*out_position,
                                      loom_op_const_results(emitted)[0]};
  const loom_low_lower_resolved_descriptor_t add = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          descriptors, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IADD_OFFSET64)};
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &add, operands, 2, loom_named_attr_slice_empty(), &offset_type,
      1, NULL, 0, source_op->location, &emitted));
  *out_position = loom_op_const_results(emitted)[0];
  return iree_ok_status();
}

static iree_status_t loom_spirv_lower_workgroup_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_compare_plan_t* plan) {
  const loom_low_descriptor_set_t* descriptors =
      loom_low_lower_context_descriptor_set(context);
  const bool equal = loom_buffer_compare_address_predicate(source_op) ==
                     LOOM_BUFFER_COMPARE_ADDRESS_PREDICATE_EQ;
  const loom_value_id_t result = loom_buffer_compare_address_result(source_op);
  const loom_type_t result_type =
      loom_low_lower_value_binding_type(context, result);
  loom_value_id_t storage[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (unsigned i = 0; i < 2; ++i) {
    if (plan->roots[i] != LOOM_VALUE_ID_INVALID) {
      storage[i] = loom_low_lower_lookup_value(context, plan->roots[i]);
    }
  }
  loom_op_t* emitted = NULL;
  if (storage[0] == LOOM_VALUE_ID_INVALID ||
      storage[1] == LOOM_VALUE_ID_INVALID || storage[0] != storage[1]) {
    // Different logical storage objects have distinct positions. Packed roots
    // share one object and use their retained byte origins below.
    const loom_low_lower_resolved_descriptor_t descriptor = {
        .descriptor = loom_low_descriptor_set_descriptor_at(
            descriptors,
            equal ? SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_FALSE_BOOL
                  : SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_TRUE_BOOL)};
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
        context, &descriptor, loom_named_attr_slice_empty(), result_type,
        source_op->location, &emitted));
  } else {
    loom_value_id_t positions[2];
    IREE_RETURN_IF_ERROR(loom_spirv_lower_workgroup_position(
        context, source_op, plan->roots[0],
        loom_buffer_compare_address_lhs_offset(source_op), &positions[0]));
    IREE_RETURN_IF_ERROR(loom_spirv_lower_workgroup_position(
        context, source_op, plan->roots[1],
        loom_buffer_compare_address_rhs_offset(source_op), &positions[1]));
    const loom_low_lower_resolved_descriptor_t compare = {
        .descriptor = loom_low_descriptor_set_descriptor_at(
            descriptors,
            equal ? SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_I_EQUAL_OFFSET64
                  : SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_I_NOT_EQUAL_OFFSET64)};
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &compare, positions, 2, loom_named_attr_slice_empty(),
        &result_type, 1, NULL, 0, source_op->location, &emitted));
  }
  return loom_low_lower_bind_value(context, result,
                                   loom_op_const_results(emitted)[0]);
}

iree_status_t loom_spirv_lower_workgroup_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t plan) {
  switch ((loom_spirv_workgroup_plan_kind_t)plan.id) {
    case LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA:
      return loom_spirv_lower_workgroup_alloca(
          context, source_op,
          (const loom_spirv_workgroup_alloca_plan_t*)plan.target_data);
    case LOOM_SPIRV_WORKGROUP_PLAN_COMPARE:
      return loom_spirv_lower_workgroup_compare(context, source_op,
                                                plan.target_data);
    case LOOM_SPIRV_WORKGROUP_PLAN_VIEW:
      return loom_spirv_lower_workgroup_view(
          context, source_op,
          (const loom_spirv_workgroup_view_plan_t*)plan.target_data);
  }
  IREE_ASSERT_UNREACHABLE("SPIR-V Workgroup plan selected unknown op kind");
  IREE_BUILTIN_UNREACHABLE();
}
