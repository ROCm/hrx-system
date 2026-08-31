// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/workgroup.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/util/fact_table.h"

typedef enum loom_spirv_workgroup_plan_kind_e {
  LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA = 1,
  LOOM_SPIRV_WORKGROUP_PLAN_VIEW = 2,
} loom_spirv_workgroup_plan_kind_t;

typedef struct loom_spirv_workgroup_alloca_plan_t {
  // Static byte length of the Workgroup allocation.
  int64_t byte_length;
  // Static byte alignment of the Workgroup allocation.
  int64_t byte_alignment;
} loom_spirv_workgroup_alloca_plan_t;

typedef struct loom_spirv_workgroup_view_plan_t {
  // Source storage-root value that has already lowered to low.storage.
  loom_value_id_t root_value_id;
  // Register class used for the Workgroup array base pointer.
  uint16_t array_pointer_reg_class_id;
} loom_spirv_workgroup_view_plan_t;

typedef struct loom_spirv_workgroup_root_carrier_t {
  // Common bit width of every typed view over the allocation.
  uint8_t bit_width;
  // Floating-point carrier used when no access requires integer atomics.
  loom_spirv_scalar_type_t float_scalar_type;
  // Whether an access requires a same-width integer carrier.
  bool requires_integer;
  // Whether the allocation has views with incompatible bit widths.
  bool incompatible;
} loom_spirv_workgroup_root_carrier_t;

typedef struct loom_spirv_workgroup_carrier_state_t {
  // Function-local value domain covered by |scalar_types|.
  const loom_local_value_domain_t* value_domain;
  // Source facts used to select Workgroup allocations and aliases.
  const loom_value_fact_table_t* fact_table;
  // Feature set used to decide whether float atomics need integer fallback.
  loom_spirv_feature_bits_t feature_bits;
  // Selected scalar carrier indexed by function-local value ordinal.
  loom_spirv_scalar_type_t* scalar_types;
  // Number of entries allocated in |scalar_types|.
  loom_value_ordinal_t scalar_type_count;
} loom_spirv_workgroup_carrier_state_t;

static const uint8_t kLoomSpirvWorkgroupCarrierStateKey;

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

static bool loom_spirv_workgroup_view_scalar_type(
    loom_type_t view_type, loom_spirv_scalar_type_t* out_scalar_type) {
  *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  loom_spirv_value_type_t value_type = {0};
  if (!loom_spirv_value_type_from_loom_type(
          loom_type_scalar(loom_type_element_type(view_type)), &value_type) ||
      value_type.value_class != LOOM_SPIRV_VALUE_CLASS_SCALAR) {
    return false;
  }
  *out_scalar_type = value_type.scalar_type;
  return true;
}

static bool loom_spirv_workgroup_view_root_is_alloca(
    const loom_module_t* module, loom_value_id_t root_value_id) {
  if (root_value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* root = loom_module_value(module, root_value_id);
  return !loom_value_is_block_arg(root) &&
         loom_buffer_alloca_isa(loom_value_def_op(root));
}

static bool loom_spirv_workgroup_view_reference(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_fact_view_reference_t* out_reference) {
  *out_reference = (loom_value_fact_view_reference_t){0};
  return loom_value_facts_query_view_reference(
             &fact_table->context,
             loom_value_fact_table_lookup(fact_table, value_id),
             out_reference) &&
         out_reference->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
}

static loom_spirv_scalar_type_t loom_spirv_workgroup_signed_integer_carrier(
    uint8_t bit_width) {
  switch (bit_width) {
    case 8:
      return LOOM_SPIRV_SCALAR_TYPE_S8;
    case 16:
      return LOOM_SPIRV_SCALAR_TYPE_S16;
    case 32:
      return LOOM_SPIRV_SCALAR_TYPE_S32;
    case 64:
      return LOOM_SPIRV_SCALAR_TYPE_S64;
    default:
      return LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  }
}

static loom_spirv_feature_bits_t
loom_spirv_workgroup_native_float_atomic_feature(
    loom_spirv_scalar_type_t scalar_type, loom_atomic_kind_t atomic_kind) {
  switch (atomic_kind) {
    case LOOM_ATOMIC_KIND_XCHGF:
      switch (scalar_type) {
        case LOOM_SPIRV_SCALAR_TYPE_F16:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS;
        case LOOM_SPIRV_SCALAR_TYPE_F32:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS;
        case LOOM_SPIRV_SCALAR_TYPE_F64:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS;
        default:
          return 0;
      }
    case LOOM_ATOMIC_KIND_ADDF:
      switch (scalar_type) {
        case LOOM_SPIRV_SCALAR_TYPE_F16:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD;
        case LOOM_SPIRV_SCALAR_TYPE_F32:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD;
        case LOOM_SPIRV_SCALAR_TYPE_F64:
          return LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD;
        default:
          return 0;
      }
    default:
      return 0;
  }
}

static bool loom_spirv_workgroup_view_requires_integer_carrier(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_spirv_scalar_type_t scalar_type,
    loom_spirv_feature_bits_t feature_bits) {
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_view_atomic_cmpxchg_isa(user_op) &&
        loom_view_atomic_cmpxchg_view(user_op) == value_id) {
      return true;
    }

    loom_atomic_kind_t atomic_kind = LOOM_ATOMIC_KIND_COUNT_;
    if (loom_view_atomic_reduce_isa(user_op) &&
        loom_view_atomic_reduce_view(user_op) == value_id) {
      atomic_kind = loom_view_atomic_reduce_kind(user_op);
    } else if (loom_view_atomic_rmw_isa(user_op) &&
               loom_view_atomic_rmw_view(user_op) == value_id) {
      atomic_kind = loom_view_atomic_rmw_kind(user_op);
    } else {
      continue;
    }

    const loom_spirv_feature_bits_t native_feature =
        loom_spirv_workgroup_native_float_atomic_feature(scalar_type,
                                                         atomic_kind);
    if (native_feature == 0 ||
        !iree_all_bits_set(feature_bits, native_feature)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_spirv_prepare_workgroup_carriers(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    loom_spirv_feature_bits_t feature_bits, iree_arena_allocator_t* arena,
    loom_spirv_workgroup_carrier_state_t* state) {
  if (state->value_domain == value_domain && state->fact_table == fact_table &&
      state->feature_bits == feature_bits &&
      state->scalar_type_count >= value_domain->value_count) {
    return iree_ok_status();
  }

  loom_spirv_scalar_type_t* scalar_types = NULL;
  loom_spirv_workgroup_root_carrier_t* root_carriers = NULL;
  if (value_domain->value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_domain->value_count, sizeof(*scalar_types),
        (void**)&scalar_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_domain->value_count, sizeof(*root_carriers),
        (void**)&root_carriers));
    memset(scalar_types, 0, value_domain->value_count * sizeof(*scalar_types));
    memset(root_carriers, 0,
           value_domain->value_count * sizeof(*root_carriers));
  }

  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < value_domain->value_count; ++value_ordinal) {
    const loom_value_id_t value_id = value_domain->value_ids[value_ordinal];
    const loom_type_t value_type = loom_module_value_type(module, value_id);
    loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
    if (!loom_spirv_workgroup_view_scalar_type(value_type, &scalar_type)) {
      continue;
    }
    loom_value_fact_view_reference_t reference = {0};
    if (!loom_spirv_workgroup_view_reference(fact_table, value_id,
                                             &reference) ||
        !loom_spirv_workgroup_view_root_is_alloca(module,
                                                  reference.root_value_id)) {
      continue;
    }
    const loom_value_ordinal_t root_ordinal =
        loom_local_value_domain_try_ordinal(value_domain,
                                            reference.root_value_id);
    if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID) continue;

    const loom_spirv_scalar_type_descriptor_t* scalar_descriptor =
        loom_spirv_scalar_type_descriptor(scalar_type);
    if (scalar_descriptor == NULL) continue;
    loom_spirv_workgroup_root_carrier_t* root_carrier =
        &root_carriers[root_ordinal];
    if (root_carrier->bit_width == 0) {
      root_carrier->bit_width = scalar_descriptor->bit_width;
    } else if (root_carrier->bit_width != scalar_descriptor->bit_width) {
      root_carrier->incompatible = true;
      continue;
    }

    if (scalar_descriptor->kind != LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT) {
      root_carrier->requires_integer = true;
      continue;
    }
    if (root_carrier->float_scalar_type == LOOM_SPIRV_SCALAR_TYPE_UNKNOWN) {
      root_carrier->float_scalar_type = scalar_type;
    } else if (root_carrier->float_scalar_type != scalar_type) {
      root_carrier->requires_integer = true;
    }
    if (loom_spirv_workgroup_view_requires_integer_carrier(
            module, value_id, scalar_type, feature_bits)) {
      root_carrier->requires_integer = true;
    }
  }

  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < value_domain->value_count; ++value_ordinal) {
    const loom_value_id_t value_id = value_domain->value_ids[value_ordinal];
    loom_value_fact_view_reference_t reference = {0};
    if (!loom_spirv_workgroup_view_reference(fact_table, value_id,
                                             &reference) ||
        !loom_spirv_workgroup_view_root_is_alloca(module,
                                                  reference.root_value_id)) {
      continue;
    }
    const loom_value_ordinal_t root_ordinal =
        loom_local_value_domain_try_ordinal(value_domain,
                                            reference.root_value_id);
    if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID) continue;
    const loom_spirv_workgroup_root_carrier_t root_carrier =
        root_carriers[root_ordinal];
    if (root_carrier.incompatible) continue;
    scalar_types[value_ordinal] =
        root_carrier.requires_integer
            ? loom_spirv_workgroup_signed_integer_carrier(
                  root_carrier.bit_width)
            : root_carrier.float_scalar_type;
  }

  state->value_domain = value_domain;
  state->fact_table = fact_table;
  state->feature_bits = feature_bits;
  state->scalar_types = scalar_types;
  state->scalar_type_count = value_domain->value_count;
  return iree_ok_status();
}

static iree_status_t loom_spirv_resolve_workgroup_view_reg_class_from_state(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    loom_spirv_feature_bits_t feature_bits, iree_arena_allocator_t* arena,
    loom_spirv_workgroup_carrier_state_t* state,
    loom_value_id_t source_value_id, bool* out_is_workgroup,
    uint16_t* out_reg_class_id) {
  *out_is_workgroup = false;
  *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_spirv_workgroup_view_reference(fact_table, source_value_id,
                                           &reference)) {
    return iree_ok_status();
  }
  *out_is_workgroup = true;
  if (!loom_spirv_workgroup_view_root_is_alloca(module,
                                                reference.root_value_id)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_spirv_prepare_workgroup_carriers(
      module, fact_table, value_domain, feature_bits, arena, state));
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, source_value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= state->scalar_type_count) {
    return iree_ok_status();
  }
  *out_reg_class_id = loom_spirv_ptr_workgroup_array_reg_class_id(
      state->scalar_types[value_ordinal]);
  return iree_ok_status();
}

iree_status_t loom_spirv_resolve_workgroup_view_reg_class(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_workgroup, uint16_t* out_reg_class_id) {
  loom_spirv_workgroup_carrier_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kLoomSpirvWorkgroupCarrierStateKey, sizeof(*state),
      (void**)&state));
  return loom_spirv_resolve_workgroup_view_reg_class_from_state(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context),
      loom_low_lower_context_value_domain(context),
      loom_low_lower_context_bundle(context)->config->contract_feature_bits,
      loom_low_lower_context_function_arena(context), state, source_value_id,
      out_is_workgroup, out_reg_class_id);
}

iree_status_t loom_spirv_resolve_workgroup_contract_view_reg_class(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id, bool* out_is_workgroup,
    uint16_t* out_reg_class_id) {
  *out_is_workgroup = false;
  *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (environment->fact_table == NULL || environment->value_domain == NULL ||
      environment->arena == NULL) {
    return iree_ok_status();
  }
  loom_spirv_workgroup_carrier_state_t local_state = {0};
  loom_spirv_workgroup_carrier_state_t* state = &local_state;
  if (environment->target_state_allocator.fn != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_target_contract_query_get_or_allocate_target_state(
            environment, &kLoomSpirvWorkgroupCarrierStateKey, sizeof(*state),
            (void**)&state));
    if (state == NULL) return iree_ok_status();
  }
  const loom_target_bundle_t* bundle =
      loom_target_contract_query_environment_bundle(environment);
  return loom_spirv_resolve_workgroup_view_reg_class_from_state(
      environment->module, environment->fact_table, environment->value_domain,
      bundle->config->contract_feature_bits, environment->arena, state,
      source_value_id, out_is_workgroup, out_reg_class_id);
}

static iree_status_t loom_spirv_workgroup_view_plan_from_facts(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_spirv_workgroup_view_plan_t* out_plan, bool* out_selected,
    bool* out_has_workgroup_facts) {
  *out_plan = (loom_spirv_workgroup_view_plan_t){0};
  *out_selected = false;
  *out_has_workgroup_facts = false;
  const loom_value_id_t result = loom_buffer_view_result(source_op);
  IREE_RETURN_IF_ERROR(loom_spirv_resolve_workgroup_view_reg_class(
      context, result, out_has_workgroup_facts,
      &out_plan->array_pointer_reg_class_id));
  if (!*out_has_workgroup_facts ||
      out_plan->array_pointer_reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
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
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_spirv_select_workgroup_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  if (loom_buffer_alloca_memory_space(source_op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
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

  loom_spirv_workgroup_alloca_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  *plan_data = plan;
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

iree_status_t loom_spirv_select_workgroup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_spirv_select_workgroup_alloca(context, source_op, out_plan);
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

static iree_status_t loom_spirv_lower_workgroup_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_alloca_plan_t* plan) {
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
  loom_value_id_t low_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->root_value_id, &low_storage));
  loom_type_t array_pointer_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, plan->array_pointer_reg_class_id, 1, &array_pointer_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      loom_low_lower_context_builder(context), low_storage, /*offset=*/0,
      array_pointer_type, source_op->location, &address_op));
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   loom_low_storage_address_result(address_op));
}

iree_status_t loom_spirv_lower_workgroup_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t plan) {
  switch ((loom_spirv_workgroup_plan_kind_t)plan.id) {
    case LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA:
      return loom_spirv_lower_workgroup_alloca(
          context, source_op,
          (const loom_spirv_workgroup_alloca_plan_t*)plan.target_data);
    case LOOM_SPIRV_WORKGROUP_PLAN_VIEW:
      return loom_spirv_lower_workgroup_view(
          context, source_op,
          (const loom_spirv_workgroup_view_plan_t*)plan.target_data);
  }
  IREE_ASSERT_UNREACHABLE("SPIR-V Workgroup plan selected unknown op kind");
  IREE_BUILTIN_UNREACHABLE();
}
