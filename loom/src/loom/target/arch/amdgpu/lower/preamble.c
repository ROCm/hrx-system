// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/preamble.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/context.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"

#define LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_BITS 10u
#define LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_MASK 0x3FFu
#define LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_X_OFFSET 12u
#define LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Y_OFFSET 16u
#define LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Z_OFFSET 20u

static uint32_t loom_amdgpu_workgroup_size_dim(
    const loom_target_workgroup_size_t* size,
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return size->x;
    case LOOM_KERNEL_DIMENSION_Y:
      return size->y;
    case LOOM_KERNEL_DIMENSION_Z:
      return size->z;
    default:
      return 0;
  }
}

bool loom_amdgpu_required_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    loom_target_workgroup_size_t* out_size) {
  *out_size = (loom_target_workgroup_size_t){0};

  if (loom_kernel_def_static_workgroup_size(module, function.op, out_size)) {
    return true;
  }

  if (bundle == NULL || bundle->export_plan == NULL ||
      bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return false;
  }
  *out_size = bundle->export_plan->hal_kernel.required_workgroup_size;
  return out_size->x != 0 || out_size->y != 0 || out_size->z != 0;
}

static bool loom_amdgpu_required_workgroup_size_dim(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, loom_kernel_dimension_t dimension,
    uint32_t* out_value) {
  *out_value = 0;
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  loom_target_workgroup_size_t size = {0};
  if (!loom_amdgpu_required_workgroup_size(module, function, bundle, &size)) {
    return false;
  }
  *out_value = loom_amdgpu_workgroup_size_dim(&size, dimension);
  return *out_value != 0;
}

bool loom_amdgpu_required_flat_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t* out_flat_size) {
  *out_flat_size = 0;
  loom_target_workgroup_size_t size = {0};
  if (!loom_amdgpu_required_workgroup_size(module, function, bundle, &size) ||
      size.x == 0 || size.y == 0 || size.z == 0) {
    return false;
  }
  const uint64_t flat_size = (uint64_t)size.x * size.y * size.z;
  if (flat_size == 0 || flat_size > UINT32_MAX) {
    return false;
  }
  *out_flat_size = (uint32_t)flat_size;
  return true;
}

iree_status_t loom_amdgpu_target_wavefront_size(
    const loom_target_bundle_t* bundle, uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  if (bundle == NULL || bundle->snapshot == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU preamble lowering requires a target "
                            "snapshot");
  }
  if (bundle->snapshot->subgroup_size != 0) {
    *out_wavefront_size = bundle->snapshot->subgroup_size;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "AMDGPU preamble lowering requires a fixed "
                          "subgroup size");
}

static uint32_t loom_amdgpu_ceil_div_u32(uint32_t numerator,
                                         uint32_t denominator) {
  IREE_ASSERT_NE(denominator, 0u);
  return numerator == 0 ? 0 : 1u + (numerator - 1u) / denominator;
}

static uint32_t loom_amdgpu_u32_log2(uint32_t value) {
  IREE_ASSERT(loom_amdgpu_u32_is_power_of_two(value));
  uint32_t log2 = 0;
  while (value > 1u) {
    value >>= 1u;
    ++log2;
  }
  return log2;
}

static bool loom_amdgpu_value_facts_exact_u32(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source_value,
    uint32_t* out_value) {
  *out_value = 0;
  if (fact_table == NULL) {
    return false;
  }
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, source_value), &value) ||
      value < 0 || value > UINT32_MAX) {
    return false;
  }
  *out_value = (uint32_t)value;
  return true;
}

static bool loom_amdgpu_source_value_facts_exact_u32(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    uint32_t* out_value) {
  return loom_amdgpu_value_facts_exact_u32(
      loom_low_lower_context_fact_table(context), source_value, out_value);
}

static bool loom_amdgpu_legality_value_facts_exact_u32(
    loom_target_low_legality_context_t* context, loom_value_id_t source_value,
    uint32_t* out_value) {
  return loom_amdgpu_value_facts_exact_u32(
      loom_target_low_legality_fact_table(context), source_value, out_value);
}

static bool loom_amdgpu_source_value_has_uses(loom_low_lower_context_t* context,
                                              loom_value_id_t source_value) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_value == LOOM_VALUE_ID_INVALID ||
      source_value >= module->values.count) {
    return false;
  }
  return !loom_value_has_no_uses(loom_module_value(module, source_value));
}

static bool loom_amdgpu_can_lower_workitem_id(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  const loom_kernel_dimension_t dimension =
      loom_kernel_workitem_id_dimension(source_op);
  return dimension < LOOM_KERNEL_DIMENSION_COUNT_ &&
         loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workitem_id_result(source_op));
}

static bool loom_amdgpu_can_lower_workgroup_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_id_dimension(source_op);
  return dimension < LOOM_KERNEL_DIMENSION_COUNT_ &&
         loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workgroup_id_result(source_op));
}

static bool loom_amdgpu_can_lower_workgroup_size(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_size_dimension(source_op);
  uint32_t unused_size = 0;
  return loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workgroup_size_result(source_op)) &&
         loom_amdgpu_required_workgroup_size_dim(
             loom_low_lower_context_module(context),
             loom_low_lower_context_source_function(context),
             loom_low_lower_context_bundle(context), dimension, &unused_size);
}

static bool loom_amdgpu_can_lower_workgroup_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_count_dimension(source_op);
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  const loom_value_id_t source_result =
      loom_kernel_workgroup_count_result(source_op);
  if (!loom_amdgpu_value_is_address_scalar(context, source_result)) {
    return false;
  }

  uint32_t unused_exact_count = 0;
  if (loom_amdgpu_source_value_facts_exact_u32(context, source_result,
                                               &unused_exact_count)) {
    return true;
  }

  uint32_t workgroup_size = 0;
  return loom_amdgpu_required_workgroup_size_dim(
             loom_low_lower_context_module(context),
             loom_low_lower_context_source_function(context),
             loom_low_lower_context_bundle(context), dimension,
             &workgroup_size) &&
         loom_amdgpu_u32_is_power_of_two(workgroup_size);
}

static bool loom_amdgpu_can_lower_workitem_dispatch_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_kernel_dimension_t dimension =
      loom_kernel_workitem_dispatch_id_dimension(source_op);
  uint32_t unused_size = 0;
  return loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workitem_dispatch_id_result(source_op)) &&
         loom_amdgpu_required_workgroup_size_dim(
             loom_low_lower_context_module(context),
             loom_low_lower_context_source_function(context),
             loom_low_lower_context_bundle(context), dimension, &unused_size);
}

static iree_status_t loom_amdgpu_can_lower_subgroup_size(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &unused_wavefront_size));
  *out_selected = loom_amdgpu_value_is_address_scalar(
      context, loom_kernel_subgroup_size_result(source_op));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_can_lower_subgroup_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &unused_wavefront_size));
  uint32_t unused_flat_workgroup_size = 0;
  *out_selected =
      loom_amdgpu_value_is_address_scalar(
          context, loom_kernel_subgroup_count_result(source_op)) &&
      loom_amdgpu_required_flat_workgroup_size(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &unused_flat_workgroup_size);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_can_lower_subgroup_linear_query(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t result, bool* out_selected) {
  (void)source_op;
  *out_selected = false;
  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  uint32_t unused_flat_workgroup_size = 0;
  *out_selected =
      loom_amdgpu_value_is_address_scalar(context, result) &&
      loom_amdgpu_u32_is_power_of_two(wavefront_size) &&
      loom_amdgpu_required_flat_workgroup_size(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &unused_flat_workgroup_size);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_preamble_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  bool selected = false;
  switch (source_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID:
      selected = loom_amdgpu_can_lower_workitem_id(context, source_op);
      break;
    case LOOM_OP_KERNEL_WORKGROUP_ID:
      selected = loom_amdgpu_can_lower_workgroup_id(context, source_op);
      break;
    case LOOM_OP_KERNEL_WORKGROUP_SIZE:
      selected = loom_amdgpu_can_lower_workgroup_size(context, source_op);
      break;
    case LOOM_OP_KERNEL_WORKGROUP_COUNT:
      selected = loom_amdgpu_can_lower_workgroup_count(context, source_op);
      break;
    case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID:
      selected = loom_amdgpu_can_lower_workitem_dispatch_id(context, source_op);
      break;
    case LOOM_OP_KERNEL_SUBGROUP_SIZE: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_can_lower_subgroup_size(context, source_op, &selected));
      break;
    }
    case LOOM_OP_KERNEL_SUBGROUP_COUNT: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_can_lower_subgroup_count(context, source_op, &selected));
      break;
    }
    case LOOM_OP_KERNEL_SUBGROUP_ID: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_can_lower_subgroup_linear_query(
          context, source_op, loom_kernel_subgroup_id_result(source_op),
          &selected));
      break;
    }
    case LOOM_OP_KERNEL_SUBGROUP_LANE_ID: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_can_lower_subgroup_linear_query(
          context, source_op, loom_kernel_subgroup_lane_id_result(source_op),
          &selected));
      break;
    }
    default:
      return iree_ok_status();
  }
  if (selected) {
    *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_workitem_id_source(
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE);
    case LOOM_KERNEL_DIMENSION_Y:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Y_SOURCE);
    case LOOM_KERNEL_DIMENSION_Z:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Z_SOURCE);
    default:
      IREE_ASSERT_UNREACHABLE("unknown kernel dimension");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_string_view_t loom_amdgpu_packed_workitem_id_source(
    uint32_t dimension_count) {
  switch (dimension_count) {
    case 2:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XY_SOURCE);
    case 3:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XYZ_SOURCE);
    default:
      IREE_ASSERT_UNREACHABLE("packed workitem id requires two or three axes");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_uses_packed_workitem_id(
    loom_low_lower_context_t* context, bool* out_uses_packed_workitem_id) {
  *out_uses_packed_workitem_id = false;
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_ref(
          loom_low_lower_context_module(context),
          loom_low_lower_context_target_ref(context));
  if (processor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU preamble lowering requires an AMDGPU "
                            "processor target record");
  }
  *out_uses_packed_workitem_id =
      loom_amdgpu_processor_kernel_descriptor_has_flags(
          processor, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID);
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_workgroup_id_source(
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_X_SOURCE);
    case LOOM_KERNEL_DIMENSION_Y:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Y_SOURCE);
    case LOOM_KERNEL_DIMENSION_Z:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Z_SOURCE);
    default:
      IREE_ASSERT_UNREACHABLE("unknown kernel dimension");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_emit_workitem_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  iree_string_view_t source = loom_amdgpu_workitem_id_source(dimension);
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), vgpr_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_packed_workitem_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t dimension_count, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  iree_string_view_t source =
      loom_amdgpu_packed_workitem_id_source(dimension_count);
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), vgpr_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("packed_tid"), &value_name_id));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_name(loom_low_lower_context_module(context),
                                 *out_low_value_id, value_name_id));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_packed_workitem_id_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t packed_id, loom_kernel_dimension_t dimension,
    bool requires_masked_x, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t shifted_id = packed_id;
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      if (!requires_masked_x) {
        *out_low_value_id = packed_id;
        return iree_ok_status();
      }
      break;
    case LOOM_KERNEL_DIMENSION_Y: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_BITS, packed_id, vgpr_type,
          &shifted_id));
      break;
    }
    case LOOM_KERNEL_DIMENSION_Z: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_BITS * 2u, packed_id,
          vgpr_type, &shifted_id));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AMDGPU workitem-id dimension %u",
                              (unsigned)dimension);
  }

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, shifted_id,
      LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_MASK, vgpr_type,
      out_low_value_id);
}

static iree_string_view_t loom_amdgpu_module_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_amdgpu_lookup_live_in_by_source(
    loom_low_lower_context_t* context, iree_string_view_t expected_source,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;

  loom_op_t* low_function = loom_low_lower_context_low_function(context);
  loom_region_t* body =
      low_function ? loom_low_function_body(low_function) : NULL;
  IREE_ASSERT(body != NULL);
  IREE_ASSERT_GT(body->block_count, 0);

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_live_in_isa(op)) {
      break;
    }
    const iree_string_view_t source =
        loom_amdgpu_module_string_or_empty(module, loom_low_live_in_source(op));
    if (!iree_string_view_equal(source, expected_source)) {
      continue;
    }
    if (*out_value_id != LOOM_VALUE_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "AMDGPU preamble lowering found duplicate '%.*s' live-ins",
          (int)expected_source.size, expected_source.data);
    }
    *out_value_id = loom_low_live_in_result(op);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_packed_workitem_id_live_in(
    loom_low_lower_context_t* context, uint32_t* out_dimension_count,
    loom_value_id_t* out_value_id) {
  *out_dimension_count = 0;
  *out_value_id = LOOM_VALUE_ID_INVALID;

  loom_value_id_t xy_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_live_in_by_source(
      context, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XY_SOURCE),
      &xy_value_id));
  loom_value_id_t xyz_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_live_in_by_source(
      context,
      IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XYZ_SOURCE),
      &xyz_value_id));

  if (xy_value_id != LOOM_VALUE_ID_INVALID &&
      xyz_value_id != LOOM_VALUE_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU preamble lowering found multiple packed workitem-id live-ins");
  }
  if (xy_value_id != LOOM_VALUE_ID_INVALID) {
    *out_dimension_count = 2;
    *out_value_id = xy_value_id;
  } else if (xyz_value_id != LOOM_VALUE_ID_INVALID) {
    *out_dimension_count = 3;
    *out_value_id = xyz_value_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_workitem_id_live_in(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    loom_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_workitem_id_source(dimension), out_value_id));
  if (*out_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU preamble lowering did not emit the requested workitem-id "
        "live-in");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t packed_dimension_count, loom_value_id_t packed_workitem_id,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  if (packed_workitem_id != LOOM_VALUE_ID_INVALID) {
    if ((uint32_t)dimension >= packed_dimension_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU packed workitem-id live-in does not cover requested "
          "dimension %u",
          (unsigned)dimension);
    }
    return loom_amdgpu_emit_packed_workitem_id_extract(
        context, source_op, packed_workitem_id, dimension,
        packed_dimension_count > 1, out_low_value_id);
  }
  return loom_amdgpu_lookup_workitem_id_live_in(context, dimension,
                                                out_low_value_id);
}

static iree_status_t loom_amdgpu_lookup_workgroup_id_live_in(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    loom_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_workgroup_id_source(dimension), out_value_id));
  if (*out_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU preamble lowering did not emit the requested workgroup-id "
        "live-in");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_dispatch_ptr_live_in(
    loom_low_lower_context_t* context, loom_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_live_in_by_source(
      context, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_DISPATCH_PTR_SOURCE),
      out_value_id));
  if (*out_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU preamble lowering did not emit the requested dispatch pointer "
        "live-in");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_mark_lane_query_workitem_id_live_ins(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t optional_exact_source_result,
    const loom_op_t** first_workitem_id_ops) {
  loom_target_workgroup_size_t workgroup_size = {0};
  uint32_t unused_flat_workgroup_size = 0;
  if (!loom_amdgpu_required_workgroup_size(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &workgroup_size) ||
      !loom_amdgpu_required_flat_workgroup_size(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context),
          &unused_flat_workgroup_size) ||
      workgroup_size.x == 0 || workgroup_size.y == 0 || workgroup_size.z == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU subgroup query lowering requires a fixed workgroup size");
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_u32_is_power_of_two(wavefront_size)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU subgroup query lowering requires a power-of-two subgroup size");
  }

  uint32_t exact_result = 0;
  if (optional_exact_source_result != LOOM_VALUE_ID_INVALID &&
      loom_amdgpu_source_value_facts_exact_u32(
          context, optional_exact_source_result, &exact_result)) {
    return iree_ok_status();
  }

  if (first_workitem_id_ops[LOOM_KERNEL_DIMENSION_X] == NULL) {
    first_workitem_id_ops[LOOM_KERNEL_DIMENSION_X] = source_op;
  }
  if (workgroup_size.y > 1 &&
      first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Y] == NULL) {
    first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Y] = source_op;
  }
  if (workgroup_size.z > 1 &&
      first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Z] == NULL) {
    first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Z] = source_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_mark_subgroup_query_workitem_id_live_ins(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t** first_workitem_id_ops) {
  loom_value_id_t source_result = LOOM_VALUE_ID_INVALID;
  switch (source_op->kind) {
    case LOOM_OP_KERNEL_SUBGROUP_ID:
      source_result = loom_kernel_subgroup_id_result(source_op);
      break;
    case LOOM_OP_KERNEL_SUBGROUP_LANE_ID:
      source_result = loom_kernel_subgroup_lane_id_result(source_op);
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU subgroup query live-in marking received "
                              "unexpected op kind %u",
                              (unsigned)source_op->kind);
  }
  return loom_amdgpu_mark_lane_query_workitem_id_live_ins(
      context, source_op, source_result, first_workitem_id_ops);
}

static iree_status_t loom_amdgpu_emit_dispatch_ptr_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgprx2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgprx2_type));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(
      context, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_DISPATCH_PTR_SOURCE),
      &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), sgprx2_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  iree_string_view_t source = loom_amdgpu_workgroup_id_source(dimension);
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), sgpr_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr_scaled_add(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t accumulator, loom_value_id_t value, uint32_t scale,
    loom_type_t result_type, loom_value_id_t* out_sum) {
  *out_sum = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
      context, source_op, value, scale, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
      result_type, &scaled_value));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, accumulator,
      scaled_value, result_type, out_sum);
}

static iree_status_t loom_amdgpu_emit_workitem_dispatch_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_workgroup_id, loom_value_id_t low_workitem_id) {
  IREE_ASSERT_NE(low_workgroup_id, LOOM_VALUE_ID_INVALID);
  IREE_ASSERT_NE(low_workitem_id, LOOM_VALUE_ID_INVALID);
  const loom_kernel_dimension_t dimension =
      loom_kernel_workitem_dispatch_id_dimension(source_op);
  uint32_t workgroup_size = 0;
  if (!loom_amdgpu_required_workgroup_size_dim(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), dimension, &workgroup_size)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU workitem dispatch-id lowering requires a fixed workgroup size");
  }

  const loom_value_id_t source_result =
      loom_kernel_workitem_dispatch_id_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &result_is_vgpr));
  if (!result_is_vgpr) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU workitem dispatch-id result must lower to a VGPR");
  }

  loom_value_id_t low_scaled_workgroup_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
      context, source_op, low_workgroup_id, &low_scaled_workgroup_id));
  if (workgroup_size != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, low_scaled_workgroup_id, workgroup_size,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, result_type,
        &low_scaled_workgroup_id));
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      low_scaled_workgroup_id, low_workitem_id, result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static bool loom_amdgpu_workgroup_count_is_exact(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t* out_exact_count) {
  return loom_amdgpu_source_value_facts_exact_u32(
      context, loom_kernel_workgroup_count_result(source_op), out_exact_count);
}

static void loom_amdgpu_find_first_dynamic_workgroup_count_ops(
    loom_low_lower_context_t* context,
    const loom_op_t** out_first_workgroup_count_ops) {
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    out_first_workgroup_count_ops[i] = NULL;
  }

  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    if (selected_plan.plan.id != LOOM_OP_KERNEL_WORKGROUP_COUNT) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    if (!loom_amdgpu_source_value_has_uses(
            context, loom_kernel_workgroup_count_result(source_op))) {
      continue;
    }
    uint32_t unused_exact_count = 0;
    if (loom_amdgpu_workgroup_count_is_exact(context, source_op,
                                             &unused_exact_count)) {
      continue;
    }
    const loom_kernel_dimension_t dimension =
        loom_kernel_workgroup_count_dimension(source_op);
    IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
    if (out_first_workgroup_count_ops[dimension] == NULL) {
      out_first_workgroup_count_ops[dimension] = source_op;
    }
  }
}

iree_status_t loom_amdgpu_emit_preamble(void* user_data,
                                        loom_low_lower_context_t* context) {
  (void)user_data;
  const loom_op_t* first_workitem_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_workgroup_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_low_lower_plan_t plan = selected_plan.plan;
    switch (plan.id) {
      case LOOM_OP_KERNEL_WORKITEM_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_workitem_id_ops[dimension] == NULL) {
          first_workitem_id_ops[dimension] = source_op;
        }
        break;
      }
      case LOOM_OP_KERNEL_WORKGROUP_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workgroup_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_workgroup_id_ops[dimension] == NULL) {
          first_workgroup_id_ops[dimension] = source_op;
        }
        break;
      }
      case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_dispatch_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_workitem_id_ops[dimension] == NULL) {
          first_workitem_id_ops[dimension] = source_op;
        }
        if (first_workgroup_id_ops[dimension] == NULL) {
          first_workgroup_id_ops[dimension] = source_op;
        }
        break;
      }
      case LOOM_OP_KERNEL_SUBGROUP_ID:
      case LOOM_OP_KERNEL_SUBGROUP_LANE_ID: {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_mark_subgroup_query_workitem_id_live_ins(
                context, source_op, first_workitem_id_ops));
        break;
      }
      case LOOM_OP_VECTOR_FRAGMENT_LOAD:
      case LOOM_OP_VECTOR_FRAGMENT_STORE: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_mark_lane_query_workitem_id_live_ins(
            context, source_op, LOOM_VALUE_ID_INVALID, first_workitem_id_ops));
        break;
      }
      case LOOM_OP_KERNEL_SUBGROUP_SHUFFLE:
      case LOOM_OP_KERNEL_SUBGROUP_REDUCE:
      case LOOM_OP_KERNEL_SUBGROUP_SCAN:
      case LOOM_OP_KERNEL_WORKGROUP_REDUCE: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_mark_lane_query_workitem_id_live_ins(
            context, source_op, LOOM_VALUE_ID_INVALID, first_workitem_id_ops));
        break;
      }
      default:
        break;
    }
  }

  loom_value_id_t low_workitem_ids[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t low_workgroup_ids[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t workitem_id_dimension_count = 0;
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    if (first_workitem_id_ops[i] != NULL) {
      workitem_id_dimension_count = i + 1;
    }
  }
  bool uses_packed_workitem_id = false;
  if (workitem_id_dimension_count > 1) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_uses_packed_workitem_id(context, &uses_packed_workitem_id));
  }
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    if (first_workgroup_id_ops[i] != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_id_live_in(
          context, first_workgroup_id_ops[i], (loom_kernel_dimension_t)i,
          &low_workgroup_ids[i]));
    }
  }

  const loom_op_t* first_workgroup_count_ops[LOOM_KERNEL_DIMENSION_COUNT_];
  loom_amdgpu_find_first_dynamic_workgroup_count_ops(context,
                                                     first_workgroup_count_ops);
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    if (first_workgroup_count_ops[i] != NULL) {
      loom_value_id_t unused_low_dispatch_ptr = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_dispatch_ptr_live_in(
          context, first_workgroup_count_ops[i], &unused_low_dispatch_ptr));
      break;
    }
  }

  loom_value_id_t packed_workitem_id = LOOM_VALUE_ID_INVALID;
  if (uses_packed_workitem_id) {
    const loom_op_t* source_op = NULL;
    for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
      if (first_workitem_id_ops[i] != NULL) {
        source_op = first_workitem_id_ops[i];
        break;
      }
    }
    IREE_ASSERT(source_op != NULL);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_workitem_id_live_in(
        context, source_op, workitem_id_dimension_count, &packed_workitem_id));
  } else {
    for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
      if (first_workitem_id_ops[i] == NULL) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workitem_id_live_in(
          context, first_workitem_id_ops[i], (loom_kernel_dimension_t)i,
          &low_workitem_ids[i]));
    }
  }

  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_low_lower_plan_t plan = selected_plan.plan;
    switch (plan.id) {
      case LOOM_OP_KERNEL_WORKITEM_ID: {
        if (uses_packed_workitem_id) {
          break;
        }
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        IREE_ASSERT_NE(low_workitem_ids[dimension], LOOM_VALUE_ID_INVALID);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, loom_kernel_workitem_id_result(source_op),
            low_workitem_ids[dimension]));
        break;
      }
      case LOOM_OP_KERNEL_WORKGROUP_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workgroup_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        IREE_ASSERT_NE(low_workgroup_ids[dimension], LOOM_VALUE_ID_INVALID);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, loom_kernel_workgroup_id_result(source_op),
            low_workgroup_ids[dimension]));
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_query_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t value) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_emit_local_linear_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t packed_dimension_count, loom_value_id_t packed_workitem_id,
    const loom_target_workgroup_size_t* workgroup_size, loom_type_t result_type,
    loom_value_id_t* out_linear_id) {
  *out_linear_id = LOOM_VALUE_ID_INVALID;

  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
      context, source_op, packed_dimension_count, packed_workitem_id,
      LOOM_KERNEL_DIMENSION_X, &linear_id));

  if (workgroup_size->y > 1) {
    loom_value_id_t workitem_y = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
        context, source_op, packed_dimension_count, packed_workitem_id,
        LOOM_KERNEL_DIMENSION_Y, &workitem_y));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scaled_add(
        context, source_op, linear_id, workitem_y, workgroup_size->x,
        result_type, &linear_id));
  }

  if (workgroup_size->z > 1) {
    const uint64_t z_scale = (uint64_t)workgroup_size->x * workgroup_size->y;
    if (z_scale > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU subgroup query workgroup linearization scale overflows u32");
    }
    loom_value_id_t workitem_z = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
        context, source_op, packed_dimension_count, packed_workitem_id,
        LOOM_KERNEL_DIMENSION_Z, &workitem_z));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scaled_add(
        context, source_op, linear_id, workitem_z, (uint32_t)z_scale,
        result_type, &linear_id));
  }

  *out_linear_id = linear_id;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_query_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, uint32_t* out_wavefront_size,
    uint32_t* out_flat_workgroup_size, loom_value_id_t* out_linear_id) {
  *out_wavefront_size = 0;
  *out_flat_workgroup_size = 0;
  *out_linear_id = LOOM_VALUE_ID_INVALID;
  loom_target_workgroup_size_t workgroup_size = {0};
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_workgroup_size(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &workgroup_size) ||
      !loom_amdgpu_required_flat_workgroup_size(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &flat_workgroup_size) ||
      workgroup_size.x == 0 || workgroup_size.y == 0 || workgroup_size.z == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU subgroup query lowering requires a fixed workgroup size");
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_u32_is_power_of_two(wavefront_size)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU subgroup query lowering requires a power-of-two subgroup size");
  }

  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &result_is_vgpr));
  if (!result_is_vgpr) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU subgroup query result must lower to a VGPR");
  }

  uint32_t packed_dimension_count = 0;
  loom_value_id_t packed_workitem_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_packed_workitem_id_live_in(
      context, &packed_dimension_count, &packed_workitem_id));
  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_local_linear_workitem_id(
      context, source_op, packed_dimension_count, packed_workitem_id,
      &workgroup_size, result_type, &linear_id));
  *out_wavefront_size = wavefront_size;
  *out_flat_workgroup_size = flat_workgroup_size;
  *out_linear_id = linear_id;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_current_subgroup_lane_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_lane_id) {
  *out_lane_id = LOOM_VALUE_ID_INVALID;

  uint32_t wavefront_size = 0;
  uint32_t flat_workgroup_size = 0;
  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_query_linear_id(
      context, source_op, result_type, &wavefront_size, &flat_workgroup_size,
      &linear_id));
  if (flat_workgroup_size <= wavefront_size) {
    *out_lane_id = linear_id;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, linear_id,
        wavefront_size - 1, result_type, out_lane_id));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_current_workitem_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_linear_id) {
  uint32_t unused_wavefront_size = 0;
  uint32_t unused_flat_workgroup_size = 0;
  return loom_amdgpu_emit_subgroup_query_linear_id(
      context, source_op, result_type, &unused_wavefront_size,
      &unused_flat_workgroup_size, out_linear_id);
}

static iree_status_t loom_amdgpu_emit_subgroup_linear_query(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, bool is_lane_id) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  uint32_t exact_result = 0;
  if (loom_amdgpu_source_value_facts_exact_u32(context, source_result,
                                               &exact_result)) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, exact_result,
        result_type, &low_result));
    return loom_low_lower_bind_value(context, source_result, low_result);
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  if (is_lane_id) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
        context, source_op, result_type, &low_result));
  } else {
    uint32_t wavefront_size = 0;
    uint32_t unused_flat_workgroup_size = 0;
    loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_query_linear_id(
        context, source_op, result_type, &wavefront_size,
        &unused_flat_workgroup_size, &linear_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        loom_amdgpu_u32_log2(wavefront_size), linear_id, result_type,
        &low_result));
  }
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static uint32_t loom_amdgpu_dispatch_packet_grid_size_offset(
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_X_OFFSET;
    case LOOM_KERNEL_DIMENSION_Y:
      return LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Y_OFFSET;
    case LOOM_KERNEL_DIMENSION_Z:
      return LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Z_OFFSET;
    default:
      IREE_ASSERT_UNREACHABLE("unknown kernel dimension");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_emit_dispatch_packet_grid_size(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_kernel_dimension_t dimension,
    loom_type_t result_type, loom_value_id_t* out_grid_size) {
  *out_grid_size = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("offset"),
      loom_amdgpu_dispatch_packet_grid_size_offset(dimension), attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[] = {
      dispatch_ptr,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
      operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, &low_op));
  *out_grid_size = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_count_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_count_dimension(source_op);
  uint32_t workgroup_size = 0;
  if (!loom_amdgpu_required_workgroup_size_dim(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), dimension, &workgroup_size)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU workgroup-count lowering requires a fixed workgroup size");
  }
  if (!loom_amdgpu_u32_is_power_of_two(workgroup_size)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU workgroup-count lowering requires a power-of-two workgroup "
        "size");
  }

  const loom_value_id_t source_result =
      loom_kernel_workgroup_count_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t grid_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_dispatch_packet_grid_size(
      context, source_op, dispatch_ptr, dimension, result_type, &grid_size));

  loom_value_id_t low_result = grid_size;
  if (workgroup_size > 1) {
    // The launch-config-to-HSA packet contract produces an exact multiple:
    // grid_size = workgroup_count * workgroup_size.
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        loom_amdgpu_u32_log2(workgroup_size), result_type, &shift));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32, grid_size,
        shift, result_type, &low_result));
  }
  *out_low_result = low_result;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_entry_setup(void* user_data,
                                           loom_low_lower_context_t* context) {
  (void)user_data;
  const loom_op_t* first_workgroup_count_ops[LOOM_KERNEL_DIMENSION_COUNT_];
  loom_amdgpu_find_first_dynamic_workgroup_count_ops(context,
                                                     first_workgroup_count_ops);

  bool uses_dynamic_workgroup_count = false;
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    uses_dynamic_workgroup_count =
        uses_dynamic_workgroup_count || first_workgroup_count_ops[i] != NULL;
  }

  loom_value_id_t low_workgroup_counts[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  if (uses_dynamic_workgroup_count) {
    loom_value_id_t dispatch_ptr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_lookup_dispatch_ptr_live_in(context, &dispatch_ptr));

    for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
      if (first_workgroup_count_ops[i] == NULL) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_count_value(
          context, first_workgroup_count_ops[i], dispatch_ptr,
          &low_workgroup_counts[i]));
    }
  }

  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    if (selected_plan.plan.id != LOOM_OP_KERNEL_WORKGROUP_COUNT) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_kernel_dimension_t dimension =
        loom_kernel_workgroup_count_dimension(source_op);
    IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
    const loom_value_id_t source_result =
        loom_kernel_workgroup_count_result(source_op);
    if (!loom_amdgpu_source_value_has_uses(context, source_result)) {
      continue;
    }
    uint32_t exact_count = 0;
    if (loom_amdgpu_workgroup_count_is_exact(context, source_op,
                                             &exact_count)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_query_constant(
          context, source_op, source_result, exact_count));
    } else {
      IREE_ASSERT_NE(low_workgroup_counts[dimension], LOOM_VALUE_ID_INVALID);
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_result, low_workgroup_counts[dimension]));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_preamble_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID: {
      uint32_t packed_dimension_count = 0;
      loom_value_id_t packed_workitem_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_packed_workitem_id_live_in(
          context, &packed_dimension_count, &packed_workitem_id));
      if (packed_workitem_id != LOOM_VALUE_ID_INVALID) {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        if ((uint32_t)dimension >= packed_dimension_count) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AMDGPU packed workitem-id live-in does not cover requested "
              "dimension %u",
              (unsigned)dimension);
        }
        loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_workitem_id_extract(
            context, source_op, packed_workitem_id, dimension,
            packed_dimension_count > 1, &low_result));
        return loom_low_lower_bind_value(
            context, loom_kernel_workitem_id_result(source_op), low_result);
      }
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      return loom_low_lower_lookup_value(
          context, loom_kernel_workitem_id_result(source_op), &low_result);
    }
    case LOOM_OP_KERNEL_WORKGROUP_ID: {
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      return loom_low_lower_lookup_value(
          context, loom_kernel_workgroup_id_result(source_op), &low_result);
    }
    case LOOM_OP_KERNEL_WORKGROUP_SIZE: {
      uint32_t workgroup_size = 0;
      if (!loom_amdgpu_required_workgroup_size_dim(
              loom_low_lower_context_module(context),
              loom_low_lower_context_source_function(context),
              loom_low_lower_context_bundle(context),
              loom_kernel_workgroup_size_dimension(source_op),
              &workgroup_size)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU workgroup-size lowering requires a fixed workgroup size");
      }
      return loom_amdgpu_emit_query_constant(
          context, source_op, loom_kernel_workgroup_size_result(source_op),
          workgroup_size);
    }
    case LOOM_OP_KERNEL_WORKGROUP_COUNT: {
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      return loom_low_lower_lookup_value(
          context, loom_kernel_workgroup_count_result(source_op), &low_result);
    }
    case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID: {
      const loom_kernel_dimension_t dimension =
          loom_kernel_workitem_dispatch_id_dimension(source_op);
      IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
      loom_value_id_t low_workgroup_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workgroup_id_live_in(
          context, dimension, &low_workgroup_id));

      uint32_t packed_dimension_count = 0;
      loom_value_id_t packed_workitem_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_packed_workitem_id_live_in(
          context, &packed_dimension_count, &packed_workitem_id));
      loom_value_id_t low_workitem_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
          context, source_op, packed_dimension_count, packed_workitem_id,
          dimension, &low_workitem_id));
      return loom_amdgpu_emit_workitem_dispatch_id(
          context, source_op, low_workgroup_id, low_workitem_id);
    }
    case LOOM_OP_KERNEL_SUBGROUP_SIZE: {
      uint32_t wavefront_size = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
          loom_low_lower_context_bundle(context), &wavefront_size));
      return loom_amdgpu_emit_query_constant(
          context, source_op, loom_kernel_subgroup_size_result(source_op),
          wavefront_size);
    }
    case LOOM_OP_KERNEL_SUBGROUP_COUNT: {
      uint32_t flat_workgroup_size = 0;
      if (!loom_amdgpu_required_flat_workgroup_size(
              loom_low_lower_context_module(context),
              loom_low_lower_context_source_function(context),
              loom_low_lower_context_bundle(context), &flat_workgroup_size)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU subgroup-count lowering requires a fixed workgroup size");
      }
      uint32_t wavefront_size = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
          loom_low_lower_context_bundle(context), &wavefront_size));
      return loom_amdgpu_emit_query_constant(
          context, source_op, loom_kernel_subgroup_count_result(source_op),
          loom_amdgpu_ceil_div_u32(flat_workgroup_size, wavefront_size));
    }
    case LOOM_OP_KERNEL_SUBGROUP_ID:
      return loom_amdgpu_emit_subgroup_linear_query(
          context, source_op, loom_kernel_subgroup_id_result(source_op),
          /*is_lane_id=*/false);
    case LOOM_OP_KERNEL_SUBGROUP_LANE_ID:
      return loom_amdgpu_emit_subgroup_linear_query(
          context, source_op, loom_kernel_subgroup_lane_id_result(source_op),
          /*is_lane_id=*/true);
    default:
      IREE_ASSERT_UNREACHABLE("AMDGPU preamble plan selected unknown op kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_preamble(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  switch (op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID:
    case LOOM_OP_KERNEL_WORKGROUP_ID:
      return iree_ok_status();
    case LOOM_OP_KERNEL_WORKGROUP_SIZE: {
      uint32_t unused_workgroup_size = 0;
      if (loom_amdgpu_required_workgroup_size_dim(
              module, loom_target_low_legality_function(context), bundle,
              loom_kernel_workgroup_size_dimension(op),
              &unused_workgroup_size)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("launch.workgroup_size_fixed"));
    }
    case LOOM_OP_KERNEL_WORKGROUP_COUNT: {
      uint32_t unused_exact_count = 0;
      if (loom_amdgpu_legality_value_facts_exact_u32(
              context, loom_kernel_workgroup_count_result(op),
              &unused_exact_count)) {
        return iree_ok_status();
      }
      uint32_t workgroup_size = 0;
      if (!loom_amdgpu_required_workgroup_size_dim(
              module, loom_target_low_legality_function(context), bundle,
              loom_kernel_workgroup_count_dimension(op), &workgroup_size)) {
        return loom_amdgpu_low_legality_reject(
            context, op,
            IREE_SV("launch.workgroup_count_fixed_workgroup_size"));
      }
      if (!loom_amdgpu_u32_is_power_of_two(workgroup_size)) {
        return loom_amdgpu_low_legality_reject(
            context, op,
            IREE_SV("launch.workgroup_count_power_of_two_workgroup_size"));
      }
      return iree_ok_status();
    }
    case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID: {
      uint32_t unused_workgroup_size = 0;
      if (loom_amdgpu_required_workgroup_size_dim(
              module, loom_target_low_legality_function(context), bundle,
              loom_kernel_workitem_dispatch_id_dimension(op),
              &unused_workgroup_size)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("launch.workitem_dispatch_fixed_workgroup_size"));
    }
    case LOOM_OP_KERNEL_SUBGROUP_SIZE: {
      uint32_t unused_wavefront_size = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_target_wavefront_size(bundle, &unused_wavefront_size));
      return iree_ok_status();
    }
    case LOOM_OP_KERNEL_SUBGROUP_COUNT: {
      uint32_t unused_wavefront_size = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_target_wavefront_size(bundle, &unused_wavefront_size));
      uint32_t unused_flat_workgroup_size = 0;
      if (loom_amdgpu_required_flat_workgroup_size(
              module, loom_target_low_legality_function(context), bundle,
              &unused_flat_workgroup_size)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("launch.subgroup_count_fixed_workgroup_size"));
    }
    case LOOM_OP_KERNEL_SUBGROUP_ID:
    case LOOM_OP_KERNEL_SUBGROUP_LANE_ID: {
      uint32_t wavefront_size = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
      if (!loom_amdgpu_u32_is_power_of_two(wavefront_size)) {
        return loom_amdgpu_low_legality_reject(
            context, op, IREE_SV("launch.subgroup_size_power_of_two"));
      }
      uint32_t unused_flat_workgroup_size = 0;
      if (loom_amdgpu_required_flat_workgroup_size(
              module, loom_target_low_legality_function(context), bundle,
              &unused_flat_workgroup_size)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("launch.subgroup_index_fixed_workgroup_size"));
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}
