// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/memory.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/records/target_records.h"
#include "loom/target/registers.h"

enum {
  LOOM_VM_MEMORY_MAX_PACKET_LANE_COUNT = 8,
  // An 8-lane quotient plus at most 4-, 2-, and 1-lane remainders.
  LOOM_VM_MEMORY_MAX_PACKET_COUNT = LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT /
                                        LOOM_VM_MEMORY_MAX_PACKET_LANE_COUNT +
                                    3,
  LOOM_VM_MEMORY_BYTE_SHIFT_NONE = UINT8_MAX,
};

// Callback plan IDs occupy a target-local domain outside descriptor ordinals.
static const uint64_t kLoomVmMemoryPlanIdBase = UINT64_C(0x100000200);

typedef uint8_t loom_vm_memory_plan_kind_t;
enum loom_vm_memory_plan_kind_e {
  LOOM_VM_MEMORY_PLAN_KIND_LOAD = 0,
  LOOM_VM_MEMORY_PLAN_KIND_STORE = 1,
  LOOM_VM_MEMORY_PLAN_KIND_ALIAS = 2,
  LOOM_VM_MEMORY_PLAN_KIND_ALLOCATE = 3,
  LOOM_VM_MEMORY_PLAN_KIND_COUNT_ = 4,
};

// One retained product contributing to a byte address.
typedef struct loom_vm_memory_term_t {
  // Static byte coefficient multiplied into the complete dynamic product.
  int64_t byte_stride;
  // Source SSA value supplying the leading dynamic factor.
  loom_value_id_t index;
  // First source SSA factor in the plan's packed factor array.
  uint16_t factor_start;
  // Number of source SSA factors following |index| in this product.
  uint8_t factor_count;
  // Power-of-two shift for |byte_stride|, or BYTE_SHIFT_NONE.
  uint8_t byte_shift;
} loom_vm_memory_term_t;
static_assert(sizeof(loom_vm_memory_term_t) == 16,
              "VM memory terms must remain compact");

// Function-lifetime state retained from the source-memory planner.
typedef struct loom_vm_memory_plan_t {
  // Static byte offset preceding the first transferred lane.
  int64_t static_byte_offset;
  // Source SSA value naming the underlying buffer root.
  loom_value_id_t root_value_id;
  // Number of consecutive scalar value cells transferred.
  uint16_t vector_lane_count;
  // Number of populated dynamic terms.
  uint8_t term_count;
  // Base-two logarithm of the transferred scalar byte width.
  uint8_t element_byte_log2;
  // Direct scaled-index byte stride, or zero when materialization is required.
  uint8_t direct_scale;
  // Dynamic address terms followed by their packed source SSA factor IDs.
  loom_vm_memory_term_t terms[];
} loom_vm_memory_plan_t;

static bool loom_vm_memory_source_op_supported(const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_LOAD_I8_U:
    case LOOM_OP_BUFFER_STORE_I8:
    case LOOM_OP_VIEW_LOAD:
    case LOOM_OP_VIEW_STORE:
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE:
      return true;
    default:
      return false;
  }
}

static bool loom_vm_memory_plan_is_selected(loom_low_lower_plan_t plan) {
  return plan.id >= kLoomVmMemoryPlanIdBase &&
         plan.id < kLoomVmMemoryPlanIdBase + LOOM_VM_MEMORY_PLAN_KIND_COUNT_;
}

static loom_vm_memory_plan_kind_t loom_vm_memory_plan_kind(
    loom_low_lower_plan_t plan) {
  IREE_ASSERT(loom_vm_memory_plan_is_selected(plan));
  return (loom_vm_memory_plan_kind_t)(plan.id - kLoomVmMemoryPlanIdBase);
}

static loom_low_source_memory_operation_kind_t
loom_vm_memory_plan_operation_kind(loom_low_lower_plan_t plan) {
  const loom_vm_memory_plan_kind_t plan_kind = loom_vm_memory_plan_kind(plan);
  IREE_ASSERT(plan_kind == LOOM_VM_MEMORY_PLAN_KIND_LOAD ||
              plan_kind == LOOM_VM_MEMORY_PLAN_KIND_STORE);
  return plan_kind == LOOM_VM_MEMORY_PLAN_KIND_LOAD
             ? LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD
             : LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
}

static bool loom_vm_memory_alias_values(const loom_op_t* source_op,
                                        loom_value_id_t* out_source,
                                        loom_value_id_t* out_result) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      *out_source = loom_buffer_view_buffer(source_op);
      *out_result = loom_buffer_view_result(source_op);
      return true;
    case LOOM_OP_VIEW_SUBVIEW:
      *out_source = loom_view_subview_source(source_op);
      *out_result = loom_view_subview_result(source_op);
      return true;
    default:
      return false;
  }
}

static const loom_value_id_t* loom_vm_memory_plan_factors(
    const loom_vm_memory_plan_t* plan) {
  return (const loom_value_id_t*)(plan->terms + plan->term_count);
}

static loom_value_id_t* loom_vm_memory_plan_factors_mutable(
    loom_vm_memory_plan_t* plan) {
  return (loom_value_id_t*)(plan->terms + plan->term_count);
}

static bool loom_vm_memory_element_byte_log2(uint32_t element_byte_count,
                                             uint8_t* out_log2) {
  switch (element_byte_count) {
    case 1:
      *out_log2 = 0;
      return true;
    case 2:
      *out_log2 = 1;
      return true;
    case 4:
      *out_log2 = 2;
      return true;
    case 8:
      *out_log2 = 3;
      return true;
    default:
      *out_log2 = 0;
      return false;
  }
}

static bool loom_vm_memory_payload_layout(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_memory_access_t access,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_vm_call_abi_register_layout_t* out_layout) {
  loom_value_id_t payload_value = LOOM_VALUE_ID_INVALID;
  if (operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
    if (source_op->result_count != 1) return false;
    payload_value = loom_op_const_results(source_op)[0];
  } else if (operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE) {
    payload_value = loom_memory_access_value(access);
  } else {
    return false;
  }
  if (payload_value >= module->values.count ||
      !loom_vm_call_abi_try_classify_logical_type(
          module, loom_module_value_type(module, payload_value), out_layout)) {
    return false;
  }
  return out_layout->bank == LOOM_VM_CALL_ABI_BANK_VALUE;
}

static bool loom_vm_memory_source_plan_supported(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_memory_access_t access,
    const loom_low_source_memory_access_plan_t* source_plan,
    uint8_t* out_element_byte_log2, uint16_t* out_factor_count) {
  *out_element_byte_log2 = 0;
  *out_factor_count = 0;
  if (!loom_vm_memory_source_op_supported(source_op) ||
      source_plan->root_value_id >= module->values.count ||
      !loom_type_is_buffer(
          loom_module_value_type(module, source_plan->root_value_id)) ||
      source_plan->vector_lane_count == 0 ||
      source_plan->vector_lane_count > LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT ||
      source_plan->vector_offset_kind !=
          LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_NONE ||
      !loom_vm_memory_element_byte_log2(source_plan->element_byte_count,
                                        out_element_byte_log2) ||
      (source_plan->vector_lane_count > 1 &&
       source_plan->vector_lane_byte_stride !=
           (int64_t)source_plan->element_byte_count)) {
    return false;
  }

  loom_vm_call_abi_register_layout_t payload_layout = {0};
  if (!loom_vm_memory_payload_layout(module, source_op, access,
                                     source_plan->operation_kind,
                                     &payload_layout) ||
      payload_layout.unit_count != source_plan->vector_lane_count) {
    return false;
  }
  if (loom_low_source_memory_access_dynamic_offset_has_materialized_view_base(
          source_plan) &&
      source_plan->dynamic_view_base_value_id >= module->values.count) {
    return false;
  }

  int64_t final_byte_offset = 0;
  if (!iree_checked_mul_add_i64(source_plan->static_byte_offset,
                                (int64_t)source_plan->vector_lane_count - 1,
                                source_plan->element_byte_count,
                                &final_byte_offset)) {
    return false;
  }
  (void)final_byte_offset;

  uint32_t factor_count = 0;
  for (uint8_t i = 0; i < source_plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source_plan->dynamic_terms[i];
    if (term->index >= module->values.count ||
        factor_count > UINT16_MAX - term->stride_value_count ||
        (term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE &&
         term->byte_shift >= 64)) {
      return false;
    }
    factor_count += term->stride_value_count;
    for (uint8_t j = 0; j < term->stride_value_count; ++j) {
      if (term->stride_values[j] >= module->values.count) return false;
    }
  }
  *out_factor_count = (uint16_t)factor_count;
  return true;
}

static bool loom_vm_memory_source_plan_build(
    const loom_module_t* module, const loom_view_region_table_t* view_regions,
    const loom_op_t* source_op,
    loom_low_source_memory_access_plan_t* out_source_plan,
    uint8_t* out_element_byte_log2, uint16_t* out_factor_count) {
  if (!loom_vm_memory_source_op_supported(source_op)) return false;
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(view_regions, source_op,
                                                out_source_plan, &diagnostic)) {
    return false;
  }
  const loom_memory_access_t access =
      loom_memory_access_cast(module, source_op);
  return loom_vm_memory_source_plan_supported(
      module, source_op, access, out_source_plan, out_element_byte_log2,
      out_factor_count);
}

static iree_status_t loom_vm_memory_plan_retain(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_plan,
    uint8_t element_byte_log2, uint16_t canonical_factor_count,
    loom_vm_memory_plan_t** out_plan) {
  *out_plan = NULL;
  const bool use_materialized_view_base =
      loom_low_source_memory_access_dynamic_offset_has_materialized_view_base(
          source_plan);
  const uint8_t term_count =
      use_materialized_view_base ? 1 : source_plan->dynamic_term_count;
  const uint16_t factor_count =
      use_materialized_view_base ? 0 : canonical_factor_count;

  iree_host_size_t plan_byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_vm_memory_plan_t), &plan_byte_length,
      IREE_STRUCT_FIELD_FAM(term_count, loom_vm_memory_term_t),
      IREE_STRUCT_FIELD(factor_count, loom_value_id_t, NULL)));
  loom_vm_memory_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, plan_byte_length, (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->static_byte_offset = source_plan->static_byte_offset;
  plan->root_value_id = source_plan->root_value_id;
  plan->vector_lane_count = (uint16_t)source_plan->vector_lane_count;
  plan->term_count = term_count;
  plan->element_byte_log2 = element_byte_log2;
  plan->direct_scale = 0;

  if (use_materialized_view_base) {
    plan->terms[0] = (loom_vm_memory_term_t){
        .byte_stride = 1,
        .index = source_plan->dynamic_view_base_value_id,
        .byte_shift = 0,
    };
  } else {
    loom_value_id_t* factors = loom_vm_memory_plan_factors_mutable(plan);
    uint16_t factor_ordinal = 0;
    for (uint8_t i = 0; i < term_count; ++i) {
      const loom_low_source_memory_dynamic_term_t* source_term =
          &source_plan->dynamic_terms[i];
      plan->terms[i] = (loom_vm_memory_term_t){
          .byte_stride = source_term->byte_stride,
          .index = source_term->index,
          .factor_start = factor_ordinal,
          .factor_count = source_term->stride_value_count,
          .byte_shift = source_term->byte_shift ==
                                LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE
                            ? LOOM_VM_MEMORY_BYTE_SHIFT_NONE
                            : (uint8_t)source_term->byte_shift,
      };
      for (uint8_t j = 0; j < source_term->stride_value_count; ++j) {
        factors[factor_ordinal++] = source_term->stride_values[j];
      }
    }
    IREE_ASSERT_EQ(factor_ordinal, factor_count);
  }

  if (plan->term_count == 1 && plan->terms[0].factor_count == 0 &&
      plan->terms[0].byte_stride > 0 &&
      plan->terms[0].byte_stride <= UINT8_MAX &&
      plan->static_byte_offset >= 0) {
    plan->direct_scale = (uint8_t)plan->terms[0].byte_stride;
  }
  *out_plan = plan;
  return iree_ok_status();
}

static iree_status_t loom_vm_memory_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_vm_target_bundle_is_core(
          loom_target_low_legality_bundle(context))) {
    return iree_ok_status();
  }
  loom_value_id_t alias_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t alias_result = LOOM_VALUE_ID_INVALID;
  if (loom_vm_memory_alias_values(source_op, &alias_source, &alias_result)) {
    *out_handled = true;
    return iree_ok_status();
  }
  // The Core VM profile owns one scalar invocation. Source scratch spaces
  // therefore materialize as fresh invocation-local buffers; a kernel
  // executor owns any cross-workitem sharing required by its execution ABI.
  if (source_op->kind == LOOM_OP_BUFFER_ALLOCA) {
    *out_handled = true;
    return iree_ok_status();
  }
  if (!loom_vm_memory_source_op_supported(source_op)) return iree_ok_status();
  loom_low_source_memory_access_plan_t source_plan = {0};
  uint8_t element_byte_log2 = 0;
  uint16_t factor_count = 0;
  *out_handled = loom_vm_memory_source_plan_build(
      loom_target_low_legality_module(context),
      loom_target_low_legality_view_regions(context), source_op, &source_plan,
      &element_byte_log2, &factor_count);
  return iree_ok_status();
}

const loom_target_low_legality_provider_t loom_vm_memory_low_legality_provider =
    {
        .name = IREE_SVL("vm-buffer-memory"),
        .builtin_dialect_bits = (UINT64_C(1) << LOOM_DIALECT_BUFFER) |
                                (UINT64_C(1) << LOOM_DIALECT_VIEW) |
                                (UINT64_C(1) << LOOM_DIALECT_VECTOR),
        .try_verify_op = loom_vm_memory_try_verify_op,
};

iree_status_t loom_vm_memory_try_select_op(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  loom_value_id_t alias_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t alias_result = LOOM_VALUE_ID_INVALID;
  if (loom_vm_memory_alias_values(source_op, &alias_source, &alias_result)) {
    *out_plan = loom_low_lower_plan_make(
        kLoomVmMemoryPlanIdBase + LOOM_VM_MEMORY_PLAN_KIND_ALIAS,
        /*target_data=*/NULL);
    return iree_ok_status();
  }
  if (source_op->kind == LOOM_OP_BUFFER_ALLOCA) {
    *out_plan = loom_low_lower_plan_make(
        kLoomVmMemoryPlanIdBase + LOOM_VM_MEMORY_PLAN_KIND_ALLOCATE,
        /*target_data=*/NULL);
    return iree_ok_status();
  }
  if (!loom_vm_memory_source_op_supported(source_op)) return iree_ok_status();

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  loom_low_source_memory_access_plan_t source_plan = {0};
  uint8_t element_byte_log2 = 0;
  uint16_t factor_count = 0;
  if (!loom_vm_memory_source_plan_build(loom_low_lower_context_module(context),
                                        view_regions, source_op, &source_plan,
                                        &element_byte_log2, &factor_count)) {
    return iree_ok_status();
  }

  loom_vm_memory_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_memory_plan_retain(
      context, &source_plan, element_byte_log2, factor_count, &retained_plan));
  const loom_vm_memory_plan_kind_t plan_kind =
      source_plan.operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD
          ? LOOM_VM_MEMORY_PLAN_KIND_LOAD
          : LOOM_VM_MEMORY_PLAN_KIND_STORE;
  *out_plan = loom_low_lower_plan_make(kLoomVmMemoryPlanIdBase + plan_kind,
                                       retained_plan);
  return iree_ok_status();
}

bool loom_vm_memory_mark_plan_storage_demands(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  if (!loom_vm_memory_plan_is_selected(plan)) return false;
  loom_value_id_t alias_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t alias_result = LOOM_VALUE_ID_INVALID;
  if (loom_vm_memory_alias_values(source_op, &alias_source, &alias_result)) {
    // The canonical access plan independently retains every address input.
    // Only the aliased storage root must survive when the view value itself is
    // required by another alias in the chain.
    loom_low_lower_require_source_value_storage(context, alias_source);
    return true;
  }
  if (loom_vm_memory_plan_kind(plan) == LOOM_VM_MEMORY_PLAN_KIND_ALLOCATE) {
    loom_low_lower_require_source_operands_storage(context, source_op);
    return true;
  }
  const loom_vm_memory_plan_t* memory_plan = plan.target_data;
  IREE_ASSERT(memory_plan != NULL);
  loom_low_lower_require_source_value_storage(context,
                                              memory_plan->root_value_id);
  const loom_value_id_t* factors = loom_vm_memory_plan_factors(memory_plan);
  for (uint8_t i = 0; i < memory_plan->term_count; ++i) {
    const loom_vm_memory_term_t* term = &memory_plan->terms[i];
    loom_low_lower_require_source_value_storage(context, term->index);
    for (uint8_t j = 0; j < term->factor_count; ++j) {
      loom_low_lower_require_source_value_storage(
          context, factors[term->factor_start + j]);
    }
  }
  if (loom_vm_memory_plan_operation_kind(plan) ==
      LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE) {
    const loom_memory_access_t access = loom_memory_access_cast(
        loom_low_lower_context_module(context), source_op);
    loom_low_lower_require_source_value_storage(
        context, loom_memory_access_value(access));
  }
  return true;
}

static iree_status_t loom_vm_memory_make_value_type(
    loom_low_lower_context_t* context, loom_type_t logical_type,
    uint16_t unit_count, loom_type_t* out_type) {
  return loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, unit_count, logical_type, out_type);
}

static iree_status_t loom_vm_memory_emit_binary(
    loom_low_lower_context_t* context, uint16_t descriptor_ordinal,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, /*result_count=*/1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &low_op));
  *out_result = loom_op_const_results(low_op)[0];
  return iree_ok_status();
}

static iree_status_t loom_vm_memory_emit_constant(
    loom_low_lower_context_t* context, uint64_t bits, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_result) {
  return loom_vm_inline_constant_build(loom_low_lower_context_builder(context),
                                       bits, result_type, location, out_result);
}

static iree_status_t loom_vm_memory_materialize_dynamic_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_vm_memory_plan_t* plan, loom_type_t address_type,
    loom_value_id_t* out_offset, bool* out_present) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  *out_present = false;
  const loom_value_id_t* factors = loom_vm_memory_plan_factors(plan);
  for (uint8_t i = 0; i < plan->term_count; ++i) {
    const loom_vm_memory_term_t* term = &plan->terms[i];
    loom_value_id_t term_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, term->index, &term_value));
    for (uint8_t j = 0; j < term->factor_count; ++j) {
      loom_value_id_t factor = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, factors[term->factor_start + j], &factor));
      IREE_RETURN_IF_ERROR(loom_vm_memory_emit_binary(
          context, VM_CORE_DESCRIPTOR_REF_INTEGER_MUL_I64, term_value, factor,
          address_type, source_op->location, &term_value));
    }

    if (term->byte_stride == 0) {
      IREE_RETURN_IF_ERROR(loom_vm_memory_emit_constant(
          context, 0, address_type, source_op->location, &term_value));
    } else if (term->byte_stride != 1) {
      loom_value_id_t coefficient = LOOM_VALUE_ID_INVALID;
      const bool use_shift = term->byte_shift != LOOM_VM_MEMORY_BYTE_SHIFT_NONE;
      IREE_RETURN_IF_ERROR(loom_vm_memory_emit_constant(
          context, use_shift ? term->byte_shift : (uint64_t)term->byte_stride,
          address_type, source_op->location, &coefficient));
      IREE_RETURN_IF_ERROR(loom_vm_memory_emit_binary(
          context,
          use_shift ? VM_CORE_DESCRIPTOR_REF_INTEGER_SHIFT_LEFT_I64
                    : VM_CORE_DESCRIPTOR_REF_INTEGER_MUL_I64,
          term_value, coefficient, address_type, source_op->location,
          &term_value));
    }

    if (!*out_present) {
      *out_offset = term_value;
      *out_present = true;
    } else {
      IREE_RETURN_IF_ERROR(loom_vm_memory_emit_binary(
          context, VM_CORE_DESCRIPTOR_REF_INTEGER_ADD_I64, *out_offset,
          term_value, address_type, source_op->location, out_offset));
    }
  }
  return iree_ok_status();
}

static uint8_t loom_vm_memory_packet_lane_count(uint16_t remaining_lanes) {
  if (remaining_lanes >= 8) return 8;
  if (remaining_lanes >= 4) return 4;
  if (remaining_lanes >= 2) return 2;
  return 1;
}

static uint8_t loom_vm_memory_lane_log2(uint8_t lane_count) {
  return (uint8_t)iree_math_count_trailing_zeros_u32(lane_count);
}

static iree_status_t loom_vm_memory_build_immediate_attrs(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_named_attr_t* out_attrs, iree_host_size_t attr_count) {
  IREE_ASSERT_EQ(descriptor->immediate_count, attr_count);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        loom_low_lower_context_builder(context),
        loom_low_descriptor_set_string(descriptor_set,
                                       immediate->field_name_string_offset),
        &out_attrs[i].name_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_memory_emit_allocation(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const uint8_t alignment_log2 = (uint8_t)iree_math_count_trailing_zeros_u64(
      (uint64_t)loom_buffer_alloca_base_alignment(source_op));

  loom_value_id_t low_byte_length = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_alloca_byte_length(source_op), &low_byte_length));
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, loom_buffer_alloca_result(source_op),
      &low_result_type));

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context),
          VM_CORE_DESCRIPTOR_REF_BUFFER_ALLOCATE);
  IREE_ASSERT(descriptor != NULL);
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  loom_named_attr_t alignment_attr = {0};
  IREE_RETURN_IF_ERROR(loom_vm_memory_build_immediate_attrs(
      context, descriptor, &alignment_attr, 1));
  alignment_attr.value = loom_attr_i64(alignment_log2);

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_descriptor, &low_byte_length, /*operand_count=*/1,
      loom_make_named_attr_slice(&alignment_attr, 1), &low_result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(context,
                                   loom_buffer_alloca_result(source_op),
                                   loom_op_const_results(low_op)[0]);
}

static iree_status_t loom_vm_memory_packet_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_vm_memory_plan_t* plan, loom_type_t address_type,
    loom_value_id_t dynamic_offset, bool has_dynamic_offset,
    loom_value_id_t direct_index, int64_t static_byte_offset,
    loom_value_id_t* inout_zero, loom_value_id_t* out_base,
    loom_value_id_t* out_index, uint8_t* out_scale) {
  *out_base = LOOM_VALUE_ID_INVALID;
  *out_index = LOOM_VALUE_ID_INVALID;
  *out_scale = 0;
  if (plan->direct_scale != 0) {
    IREE_ASSERT_GE(static_byte_offset, 0);
    IREE_RETURN_IF_ERROR(loom_vm_memory_emit_constant(
        context, (uint64_t)static_byte_offset, address_type,
        source_op->location, out_base));
    *out_index = direct_index;
    *out_scale = plan->direct_scale;
    return iree_ok_status();
  }

  if (*inout_zero == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_vm_memory_emit_constant(
        context, 0, address_type, source_op->location, inout_zero));
  }
  *out_index = *inout_zero;
  if (!has_dynamic_offset) {
    if (static_byte_offset == 0) {
      *out_base = *inout_zero;
      return iree_ok_status();
    }
    return loom_vm_memory_emit_constant(context, (uint64_t)static_byte_offset,
                                        address_type, source_op->location,
                                        out_base);
  }
  if (static_byte_offset == 0) {
    *out_base = dynamic_offset;
    return iree_ok_status();
  }

  loom_value_id_t static_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_memory_emit_constant(
      context, (uint64_t)static_byte_offset, address_type, source_op->location,
      &static_offset));
  return loom_vm_memory_emit_binary(
      context, VM_CORE_DESCRIPTOR_REF_INTEGER_ADD_I64, dynamic_offset,
      static_offset, address_type, source_op->location, out_base);
}

static iree_status_t loom_vm_memory_packet_type(
    loom_low_lower_context_t* context, loom_type_t logical_payload_type,
    uint8_t packet_lane_count, bool is_complete_payload,
    loom_type_t* out_type) {
  if (is_complete_payload) {
    return loom_vm_memory_make_value_type(context, logical_payload_type,
                                          packet_lane_count, out_type);
  }
  const loom_type_t fragment_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(logical_payload_type),
      loom_dim_pack_static(packet_lane_count), /*encoding_id=*/0);
  return loom_vm_memory_make_value_type(context, fragment_type,
                                        packet_lane_count, out_type);
}

static iree_status_t loom_vm_memory_store_packet_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t full_value, loom_type_t logical_payload_type,
    uint16_t unit_offset, uint8_t packet_lane_count, bool is_complete_payload,
    loom_value_id_t* out_value) {
  if (is_complete_payload) {
    *out_value = full_value;
    return iree_ok_status();
  }
  loom_type_t fragment_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_vm_memory_packet_type(
      context, logical_payload_type, packet_lane_count,
      /*is_complete_payload=*/false, &fragment_type));
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), full_value, unit_offset,
      fragment_type, source_op->location, &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

iree_status_t loom_vm_memory_emit_op(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     bool* out_handled) {
  *out_handled = loom_vm_memory_plan_is_selected(plan);
  if (!*out_handled) return iree_ok_status();
  if (loom_vm_memory_plan_kind(plan) == LOOM_VM_MEMORY_PLAN_KIND_ALLOCATE) {
    return loom_vm_memory_emit_allocation(context, source_op);
  }
  loom_value_id_t alias_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t alias_result = LOOM_VALUE_ID_INVALID;
  if (loom_vm_memory_alias_values(source_op, &alias_source, &alias_result)) {
    return loom_low_lower_bind_value_alias(context, alias_source, alias_result);
  }
  const loom_vm_memory_plan_t* memory_plan = plan.target_data;
  IREE_ASSERT(memory_plan != NULL);
  const bool is_load = loom_vm_memory_plan_operation_kind(plan) ==
                       LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_memory_access_t access =
      loom_memory_access_cast(module, source_op);
  const loom_value_id_t source_payload =
      is_load ? loom_op_const_results(source_op)[0]
              : loom_memory_access_value(access);
  const loom_type_t logical_payload_type =
      loom_module_value_type(module, source_payload);

  loom_value_id_t low_root = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, memory_plan->root_value_id, &low_root));
  loom_type_t address_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_vm_memory_make_value_type(
      context, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), /*unit_count=*/1,
      &address_type));

  loom_value_id_t direct_index = LOOM_VALUE_ID_INVALID;
  loom_value_id_t dynamic_offset = LOOM_VALUE_ID_INVALID;
  bool has_dynamic_offset = false;
  if (memory_plan->direct_scale != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, memory_plan->terms[0].index, &direct_index));
  } else {
    IREE_RETURN_IF_ERROR(loom_vm_memory_materialize_dynamic_offset(
        context, source_op, memory_plan, address_type, &dynamic_offset,
        &has_dynamic_offset));
  }

  loom_value_id_t low_store_value = LOOM_VALUE_ID_INVALID;
  if (!is_load) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_payload, &low_store_value));
  }

  const uint16_t descriptor_ordinal = is_load
                                          ? VM_CORE_DESCRIPTOR_REF_BUFFER_LOAD
                                          : VM_CORE_DESCRIPTOR_REF_BUFFER_STORE;
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  loom_named_attr_t packet_attrs[2] = {0};
  IREE_RETURN_IF_ERROR(loom_vm_memory_build_immediate_attrs(
      context, descriptor, packet_attrs, IREE_ARRAYSIZE(packet_attrs)));

  loom_value_id_t loaded_fragments[LOOM_VM_MEMORY_MAX_PACKET_COUNT];
  uint8_t loaded_fragment_count = 0;
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  uint16_t lane_ordinal = 0;
  while (lane_ordinal < memory_plan->vector_lane_count) {
    const uint8_t packet_lane_count = loom_vm_memory_packet_lane_count(
        (uint16_t)(memory_plan->vector_lane_count - lane_ordinal));
    int64_t packet_static_byte_offset = 0;
    const bool offset_in_range =
        iree_checked_mul_add_i64(memory_plan->static_byte_offset, lane_ordinal,
                                 INT64_C(1) << memory_plan->element_byte_log2,
                                 &packet_static_byte_offset);
    IREE_ASSERT(offset_in_range);
    (void)offset_in_range;

    loom_value_id_t low_base = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    uint8_t scale = 0;
    IREE_RETURN_IF_ERROR(loom_vm_memory_packet_address(
        context, source_op, memory_plan, address_type, dynamic_offset,
        has_dynamic_offset, direct_index, packet_static_byte_offset, &zero,
        &low_base, &low_index, &scale));

    packet_attrs[0].value = loom_attr_i64(scale);
    packet_attrs[1].value =
        loom_attr_i64(memory_plan->element_byte_log2 * 4 +
                      loom_vm_memory_lane_log2(packet_lane_count));
    const bool is_complete_payload =
        packet_lane_count == memory_plan->vector_lane_count;
    loom_value_id_t operands[4] = {low_root, low_base, low_index,
                                   LOOM_VALUE_ID_INVALID};
    loom_type_t result_type = loom_type_none();
    const loom_type_t* result_types = NULL;
    iree_host_size_t result_count = 0;
    if (is_load) {
      IREE_RETURN_IF_ERROR(loom_vm_memory_packet_type(
          context, logical_payload_type, packet_lane_count, is_complete_payload,
          &result_type));
      result_types = &result_type;
      result_count = 1;
    } else {
      IREE_RETURN_IF_ERROR(loom_vm_memory_store_packet_value(
          context, source_op, low_store_value, logical_payload_type,
          lane_ordinal, packet_lane_count, is_complete_payload, &operands[3]));
    }

    loom_op_t* packet_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_descriptor, operands, is_load ? 3 : 4,
        loom_make_named_attr_slice(packet_attrs, IREE_ARRAYSIZE(packet_attrs)),
        result_types, result_count, /*tied_results=*/NULL,
        /*tied_result_count=*/0, source_op->location, &packet_op));
    if (is_load) {
      IREE_ASSERT_LT(loaded_fragment_count, IREE_ARRAYSIZE(loaded_fragments));
      loaded_fragments[loaded_fragment_count++] =
          loom_op_const_results(packet_op)[0];
    }
    lane_ordinal = (uint16_t)(lane_ordinal + packet_lane_count);
  }

  if (!is_load) return iree_ok_status();
  IREE_ASSERT_GT(loaded_fragment_count, 0);
  if (loaded_fragment_count == 1) {
    return loom_low_lower_bind_value(context, source_payload,
                                     loaded_fragments[0]);
  }
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, source_payload, &low_result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), loaded_fragments,
      loaded_fragment_count, low_result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_payload,
                                   loom_low_concat_result(concat_op));
}
