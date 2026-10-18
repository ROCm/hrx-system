// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/buffer_descriptor.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum loom_amdgpu_buffer_extent_base_e {
  LOOM_AMDGPU_BUFFER_EXTENT_BASE_NONE = 0,
  LOOM_AMDGPU_BUFFER_EXTENT_BASE_VALUE = 1,
  LOOM_AMDGPU_BUFFER_EXTENT_BASE_TERMS = 2,
};

typedef uint8_t loom_amdgpu_buffer_extent_plan_flags_t;
enum loom_amdgpu_buffer_extent_plan_flag_bits_e {
  // The source-view range replaces a broader ABI resource range.
  LOOM_AMDGPU_BUFFER_EXTENT_PLAN_FLAG_OVERRIDE_RESOURCE = 1u << 0,
};

struct loom_amdgpu_buffer_extent_plan_t {
  // Product of the element byte width and all static dimensions. When there
  // are no dynamic dimensions or base terms, this is the exact byte extent.
  uint32_t byte_scale;
  // Static root-relative bytes added after the dynamic footprint and base.
  uint32_t byte_offset;
  // Source base value or canonical terms selected for the dynamic origin.
  uint8_t base;
  // Number of runtime dimensions multiplied in source axis order.
  uint8_t dimension_count;
  // Descriptor-range ownership selected during target planning.
  loom_amdgpu_buffer_extent_plan_flags_t flags;
  // Source values with retained one-word SGPR carriers.
  loom_value_id_t dimensions[];
};

static iree_status_t loom_amdgpu_buffer_extent_prepare_sgpr_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_supported) {
  *out_supported = false;
  loom_type_t type = loom_type_none();
  const bool type_is_planned =
      loom_low_lower_value_binding_type_is_planned(context, source_value);
  if (type_is_planned) {
    type = loom_low_lower_value_binding_type(context, source_value);
  } else if (source_op != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_map_value(context, source_op, source_value, &type));
  }
  if (!loom_amdgpu_low_type_is_register_class(context, type,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR) ||
      loom_low_register_type_unit_count(type) != 1) {
    return iree_ok_status();
  }
  if (!type_is_planned) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_plan_value_type(context, source_value, type));
  }
  *out_supported = true;
  return iree_ok_status();
}

static bool loom_amdgpu_buffer_extent_is_planned_sgpr_u32(
    loom_low_lower_context_t* context, loom_value_id_t source_value) {
  if (!loom_low_lower_value_binding_type_is_planned(context, source_value)) {
    return false;
  }
  const loom_type_t type =
      loom_low_lower_value_binding_type(context, source_value);
  return loom_amdgpu_low_type_is_register_class(
             context, type, LOOM_AMDGPU_REG_CLASS_ID_SGPR) &&
         loom_low_register_type_unit_count(type) == 1;
}

static bool loom_amdgpu_buffer_extent_term_fits(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_dynamic_term_t* term) {
  if (term->byte_stride < 0 || term->byte_stride > UINT32_MAX ||
      !loom_value_facts_fit_unsigned_bit_count(term->byte_facts, 32) ||
      !loom_amdgpu_buffer_extent_is_planned_sgpr_u32(context, term->index)) {
    return false;
  }
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    if (!loom_amdgpu_buffer_extent_is_planned_sgpr_u32(
            context, term->stride_values[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_plan_buffer_extent_with_flags(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_buffer_extent_plan_flags_t flags,
    const loom_amdgpu_buffer_extent_plan_t** out_plan) {
  *out_plan = NULL;
  const uint16_t root_argument =
      loom_low_lower_source_memory_root_argument_index(context, source);
  if (root_argument != UINT16_MAX) {
    uint16_t argument_count = 0;
    const loom_low_lower_abi_argument_t* argument =
        &loom_low_lower_context_argument_map(context,
                                             &argument_count)[root_argument];
    if (!iree_any_bit_set(
            flags, LOOM_AMDGPU_BUFFER_EXTENT_PLAN_FLAG_OVERRIDE_RESOURCE) &&
        argument->kind == LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE &&
        iree_any_bit_set(argument->resource_build_flags,
                         LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT |
                             LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT_VALUE)) {
      return iree_ok_status();
    }
  }
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(
          &facts->context,
          loom_value_fact_table_lookup(facts, source->view_value_id),
          &reference) ||
      !loom_value_facts_fit_unsigned_bit_count(reference.base_byte_offset,
                                               32)) {
    return iree_ok_status();
  }
  int64_t static_base = 0;
  if (source->dynamic_view_base_term_count == 0 &&
      !loom_value_facts_as_exact_i64(reference.base_byte_offset,
                                     &static_base)) {
    return iree_ok_status();
  }
  loom_value_facts_t range = reference.base_byte_offset;
  loom_value_facts_addi(&range, &reference.footprint_byte_length, &range);
  if (!loom_value_facts_fit_unsigned_bit_count(range, 32)) {
    return iree_ok_status();
  }

  loom_amdgpu_buffer_extent_plan_t selected = {.flags = flags};
  loom_value_id_t dimensions[LOOM_TYPE_MAX_RANK];
  int64_t exact_range = 0;
  if (loom_value_facts_as_exact_i64(range, &exact_range)) {
    selected.byte_scale = (uint32_t)exact_range;
  } else {
    const loom_module_t* module = loom_low_lower_context_module(context);
    const loom_type_t view_type =
        loom_module_value_type(module, source->view_value_id);
    loom_value_facts_t strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
    loom_value_fact_address_layout_t layout = {0};
    if (!loom_encoding_query_type_address_layout(
            &facts->context, module, view_type, strides,
            IREE_ARRAYSIZE(strides), &layout) ||
        layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
      return iree_ok_status();
    }
    const int32_t element_bits =
        loom_scalar_type_bitwidth(loom_type_element_type(view_type));
    if (element_bits <= 0 || element_bits % 8 != 0) {
      return iree_ok_status();
    }
    int64_t scale = element_bits / 8;
    for (uint8_t axis = 0; axis < loom_type_rank(view_type); ++axis) {
      if (loom_type_dim_is_dynamic_at(view_type, axis)) {
        const loom_value_id_t dimension =
            loom_type_dim_value_id_at(view_type, axis);
        bool supported = false;
        IREE_RETURN_IF_ERROR(loom_amdgpu_buffer_extent_prepare_sgpr_u32(
            context, source_op, dimension, &supported));
        if (!supported) {
          return iree_ok_status();
        }
        dimensions[selected.dimension_count++] = dimension;
      } else {
        const int64_t dimension = loom_type_dim_static_size_at(view_type, axis);
        if (dimension < 0 || !iree_checked_mul_i64(scale, dimension, &scale) ||
            scale > UINT32_MAX) {
          return iree_ok_status();
        }
      }
    }
    selected.byte_scale = (uint32_t)scale;
    if (source->dynamic_view_base_term_count != 0) {
      static_base = source->static_view_base_byte_offset;
      bool dynamic_base_supported = false;
      if (source->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_buffer_extent_prepare_sgpr_u32(
            context, source_op, source->dynamic_view_base_value_id,
            &dynamic_base_supported));
      }
      if (dynamic_base_supported) {
        selected.base = LOOM_AMDGPU_BUFFER_EXTENT_BASE_VALUE;
        // A retained SSA base may be the complete authored expression or its
        // constant-free prefix. Charge only the static bytes it does not carry.
        static_base -= source->dynamic_view_base_value_static_byte_offset;
      } else {
        for (uint8_t i = 0; i < source->dynamic_view_base_term_count; ++i) {
          if (!loom_amdgpu_buffer_extent_term_fits(context,
                                                   &source->dynamic_terms[i])) {
            return iree_ok_status();
          }
        }
        selected.base = LOOM_AMDGPU_BUFFER_EXTENT_BASE_TERMS;
      }
    }
    if (static_base < 0 || static_base > UINT32_MAX ||
        (selected.dimension_count == 0 &&
         selected.base == LOOM_AMDGPU_BUFFER_EXTENT_BASE_NONE)) {
      return iree_ok_status();
    }
    selected.byte_offset = (uint32_t)static_base;
  }

  loom_amdgpu_buffer_extent_plan_t* retained = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context,
      sizeof(*retained) + selected.dimension_count * sizeof(*dimensions),
      (void**)&retained));
  *retained = selected;
  memcpy(retained->dimensions, dimensions,
         selected.dimension_count * sizeof(*dimensions));
  *out_plan = retained;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_plan_buffer_extent(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_buffer_extent_plan_t** out_plan) {
  return loom_amdgpu_plan_buffer_extent_with_flags(context, NULL, source, 0,
                                                   out_plan);
}

iree_status_t loom_amdgpu_plan_buffer_view_extent(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_buffer_extent_plan_t** out_plan) {
  return loom_amdgpu_plan_buffer_extent_with_flags(
      context, source_op, source,
      LOOM_AMDGPU_BUFFER_EXTENT_PLAN_FLAG_OVERRIDE_RESOURCE, out_plan);
}

static iree_status_t loom_amdgpu_emit_buffer_extent(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_buffer_extent_plan_t* plan, loom_value_id_t* out_extent) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t extent = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < plan->dimension_count; ++i) {
    loom_value_id_t dimension =
        loom_low_lower_lookup_value(context, plan->dimensions[i]);
    if (i == 0) {
      extent = dimension;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, extent,
          dimension, sgpr_type, &extent));
    }
  }
  if (plan->dimension_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        plan->byte_scale, sgpr_type, &extent));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_u32(
        context, source_op, extent, plan->byte_scale, sgpr_type, &extent));
  }
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  if (plan->base == LOOM_AMDGPU_BUFFER_EXTENT_BASE_VALUE) {
    base = loom_low_lower_lookup_value(context,
                                       source->dynamic_view_base_value_id);
  } else if (plan->base == LOOM_AMDGPU_BUFFER_EXTENT_BASE_TERMS) {
    for (uint8_t i = 0; i < source->dynamic_view_base_term_count; ++i) {
      loom_value_id_t term = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_term(
          context, source_op, &source->dynamic_terms[i], &term));
      if (i == 0) {
        base = term;
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, base,
            term, sgpr_type, &base));
      }
    }
  }
  if (plan->base != LOOM_AMDGPU_BUFFER_EXTENT_BASE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, base, extent,
        sgpr_type, &extent));
  }
  if (plan->byte_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, extent,
        plan->byte_offset, sgpr_type, &extent));
  }
  *out_extent = extent;
  return iree_ok_status();
}

typedef struct loom_amdgpu_hal_buffer_descriptor_extent_t {
  // Static descriptor range word used when dynamic_extent is absent.
  int64_t static_extent;
  // Optional SGPR descriptor range word for dynamically sized views.
  loom_value_id_t dynamic_extent;
} loom_amdgpu_hal_buffer_descriptor_extent_t;

iree_status_t loom_amdgpu_emit_hal_buffer_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_binding,
    const loom_low_source_memory_access_plan_t* source_access,
    const loom_amdgpu_buffer_extent_plan_t* plan,
    loom_value_id_t* out_low_descriptor) {
  *out_low_descriptor = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_hal_buffer_descriptor_extent_t extent = {
      .static_extent = UINT32_MAX,
      .dynamic_extent = LOOM_VALUE_ID_INVALID,
  };
  bool resource_has_extent = false;
  int64_t cache_swizzle_stride = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* binding = loom_module_value(module, low_binding);
  const loom_op_t* binding_op = loom_value_def_op(binding);
  if (binding_op != NULL && loom_low_resource_isa(binding_op)) {
    if (loom_low_resource_extent_value_is_present(binding_op)) {
      extent.dynamic_extent = loom_low_resource_extent_value(binding_op);
      resource_has_extent = true;
    } else {
      if (loom_low_resource_has_extent(binding_op)) {
        const int64_t resource_extent = loom_low_resource_extent(binding_op);
        if (resource_extent <= UINT32_MAX) {
          extent.static_extent = resource_extent;
        }
        resource_has_extent = true;
      }
    }

    if (loom_low_resource_has_cache_swizzle_stride(binding_op)) {
      cache_swizzle_stride = loom_low_resource_cache_swizzle_stride(binding_op);
    }
  }
  if (plan != NULL &&
      (!resource_has_extent ||
       iree_any_bit_set(
           plan->flags,
           LOOM_AMDGPU_BUFFER_EXTENT_PLAN_FLAG_OVERRIDE_RESOURCE))) {
    if (plan->dimension_count == 0 &&
        plan->base == LOOM_AMDGPU_BUFFER_EXTENT_BASE_NONE) {
      extent.static_extent = plan->byte_scale;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_buffer_extent(
          context, source_op, source_access, plan, &extent.dynamic_extent));
    }
  }

  loom_type_t descriptor_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 4, &descriptor_type));
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("cache_swizzle_stride"), cache_swizzle_stride, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[2] = {low_binding, extent.dynamic_extent};
  uint16_t operand_count = 1;
  uint16_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR;
  if (extent.dynamic_extent != LOOM_VALUE_ID_INVALID) {
    descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT;
    operand_count = 2;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("extent"), extent.static_extent, attrs,
        IREE_ARRAYSIZE(attrs), &attr_count));
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &descriptor_type, 1,
      &low_op));
  *out_low_descriptor = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}
