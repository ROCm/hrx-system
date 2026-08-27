// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"

iree_string_view_t loom_amdgpu_memory_space_name(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return IREE_SV("global");
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      return IREE_SV("constant");
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return IREE_SV("workgroup");
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      return IREE_SV("private");
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
      return IREE_SV("host");
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return IREE_SV("descriptor");
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return IREE_SV("generic");
  }
  return IREE_SV("invalid");
}

iree_string_view_t loom_amdgpu_memory_operation_name(
    loom_low_source_memory_operation_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD:
      return IREE_SV("load");
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE:
      return IREE_SV("store");
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH:
      return IREE_SV("prefetch");
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE:
      return IREE_SV("atomic_reduce");
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW:
      return IREE_SV("atomic_rmw");
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG:
      return IREE_SV("atomic_cmpxchg");
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_COUNT_:
      return IREE_SV("invalid");
  }
  return IREE_SV("invalid");
}

iree_string_view_t loom_amdgpu_memory_address_form_name(
    loom_amdgpu_memory_address_form_t address_form) {
  switch (address_form) {
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT:
      return IREE_SV("default");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO:
      return IREE_SV("buffer_off_zero");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR:
      return IREE_SV("ds_2addr");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR:
      return IREE_SV("global_saddr");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID:
      return IREE_SV("ds_addtid");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT:
      return IREE_SV("flat");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM:
      return IREE_SV("global_smem");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR:
      return IREE_SV("scratch_vaddr");
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_COUNT_:
      break;
  }
  return IREE_SV("invalid");
}

iree_string_view_t loom_amdgpu_memory_access_dynamic_term_kind_name(
    const loom_amdgpu_memory_access_t* access) {
  switch (access->source.dynamic_term_count) {
    case 0:
      return IREE_SV("none");
    case 1:
      break;
    default:
      return IREE_SV("multiple");
  }
  switch (access->dynamic_term_kinds[0]) {
    case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
      return IREE_SV("vaddr");
    case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
      return IREE_SV("soffset");
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_cache_policy_scope_param(
    const loom_vector_memory_cache_policy_t* policy) {
  return iree_all_bits_set(policy->build_flags,
                           LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE)
             ? loom_amdgpu_cache_scope_name(policy->cache_scope)
             : IREE_SV("<missing>");
}

static iree_string_view_t loom_amdgpu_cache_policy_temporal_param(
    const loom_vector_memory_cache_policy_t* policy) {
  return iree_all_bits_set(policy->build_flags,
                           LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL)
             ? loom_amdgpu_cache_temporal_name(policy->cache_temporal)
             : IREE_SV("<missing>");
}

static iree_string_view_t loom_amdgpu_memory_diagnostic_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static bool loom_amdgpu_memory_access_has_vector_width_diagnostic(
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  return diagnostic->vector_lane_count != 0 &&
         diagnostic->element_byte_count != 0 &&
         diagnostic->required_32bit_lane_count != 0 &&
         diagnostic->native_max_vector_lane_count != 0 &&
         diagnostic->scalarized_max_vector_lane_count != 0;
}

static iree_status_t loom_amdgpu_emit_memory_access_vector_width_error(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->export_plan->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->config->name, IREE_SV("<empty>"))),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(
          loom_amdgpu_memory_operation_name(source->operation_kind)),
      loom_param_type(diagnostic->vector_type),
      loom_param_u32(diagnostic->vector_lane_count),
      loom_param_u32(diagnostic->element_byte_count),
      loom_param_u32(diagnostic->required_32bit_lane_count),
      loom_param_u32(LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES),
      loom_param_u32(diagnostic->native_max_vector_lane_count),
      loom_param_u32(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
      loom_param_u32(diagnostic->scalarized_max_vector_lane_count),
      loom_param_string(IREE_SV("memory_access.vector_type")),
  };
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_040_REF, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_emit_memory_access_packed_footprint_error(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->export_plan->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->config->name, IREE_SV("<empty>"))),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(loom_op_name(module, op)),
      loom_param_type(diagnostic->payload_type),
      loom_param_u32(diagnostic->payload_bit_count),
      loom_param_u32(diagnostic->register_bit_count),
  };
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_019_REF, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_emit_memory_access_flat_dynamic_error(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  const loom_module_t* module = loom_target_low_legality_module(context);
  loom_low_source_memory_dynamic_term_t term = {0};
  if (diagnostic->dynamic_term_index < source->dynamic_term_count) {
    term = source->dynamic_terms[diagnostic->dynamic_term_index];
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->export_plan->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->config->name, IREE_SV("<empty>"))),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(
          loom_amdgpu_memory_operation_name(source->operation_kind)),
      loom_param_string(loom_amdgpu_memory_space_name(source->memory_space)),
      loom_param_u32(diagnostic->dynamic_term_index),
      loom_param_i64(term.byte_stride),
      loom_param_i64(term.byte_facts.range_lo),
      loom_param_i64(term.byte_facts.range_hi),
      loom_param_u32(term.byte_shift),
  };
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_020_REF, params, IREE_ARRAYSIZE(params));
}

static loom_value_facts_t loom_amdgpu_source_memory_byte_offset_facts(
    const loom_low_source_memory_access_plan_t* source) {
  return loom_low_source_memory_dynamic_offset_facts(
      source, source->static_byte_offset);
}

static iree_status_t loom_amdgpu_emit_memory_access_dynamic_offset_error(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  const loom_module_t* module = loom_target_low_legality_module(context);
  loom_low_source_memory_dynamic_term_t term = {0};
  if (diagnostic->dynamic_term_index < source->dynamic_term_count) {
    term = source->dynamic_terms[diagnostic->dynamic_term_index];
  }
  const loom_value_facts_t offset_facts =
      loom_amdgpu_source_memory_byte_offset_facts(source);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->export_plan->name, IREE_SV("<empty>"))),
      loom_param_string(loom_amdgpu_memory_diagnostic_nonempty(
          bundle->config->name, IREE_SV("<empty>"))),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(
          loom_amdgpu_memory_operation_name(source->operation_kind)),
      loom_param_string(loom_amdgpu_memory_space_name(source->memory_space)),
      loom_param_i64(offset_facts.range_lo),
      loom_param_i64(offset_facts.range_hi),
      loom_param_u32(diagnostic->dynamic_term_index),
      loom_param_i64(term.byte_stride),
      loom_param_i64(term.byte_facts.range_lo),
      loom_param_i64(term.byte_facts.range_hi),
      loom_param_u32(term.byte_shift),
      loom_param_i64(source->static_byte_offset),
      loom_param_string(IREE_SV("memory_access.dynamic_offset_range")),
  };
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_034_REF, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_emit_memory_access_rejection_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic,
    bool* out_handled) {
  *out_handled = true;
  if (iree_any_bit_set(diagnostic->rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE) &&
      loom_amdgpu_memory_access_has_vector_width_diagnostic(diagnostic)) {
    return loom_amdgpu_emit_memory_access_vector_width_error(
        context, op, source, diagnostic);
  }
  if (iree_any_bit_set(
          diagnostic->rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT)) {
    return loom_amdgpu_emit_memory_access_packed_footprint_error(context, op,
                                                                 diagnostic);
  }
  if (iree_any_bit_set(
          diagnostic->rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_FLAT_DYNAMIC_ADDRESS)) {
    return loom_amdgpu_emit_memory_access_flat_dynamic_error(
        context, op, source, diagnostic);
  }
  if (iree_any_bit_set(
          diagnostic->rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE)) {
    return loom_amdgpu_emit_memory_access_dynamic_offset_error(
        context, op, source, diagnostic);
  }
  *out_handled = false;
  return iree_ok_status();
}

typedef struct loom_amdgpu_memory_access_rejection_key_t {
  // Rejection flag tested by this row.
  loom_amdgpu_memory_access_rejection_flags_t rejection_bit;
  // Stable diagnostic constraint key returned when rejection_bit is present.
  iree_string_view_t constraint_key;
} loom_amdgpu_memory_access_rejection_key_t;

static const loom_amdgpu_memory_access_rejection_key_t
    kAmdgpuMemoryAccessRejectionKeys[] = {
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE,
            .constraint_key = IREE_SVL("memory_access.vector_type"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE,
            .constraint_key = IREE_SVL("memory_access.vector_axis_stride"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE,
            .constraint_key = IREE_SVL("memory_access.dynamic_index_source"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE,
            .constraint_key = IREE_SVL("memory_access.dynamic_stride"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE,
            .constraint_key = IREE_SVL("memory_access.scalar_dynamic_stride"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE,
            .constraint_key =
                IREE_SVL("memory_access.workgroup_dynamic_index_source"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE,
            .constraint_key =
                IREE_SVL("memory_access.global_fallback_unavailable"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS,
            .constraint_key = IREE_SVL("memory_access.global_fallback_address"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE,
            .constraint_key =
                IREE_SVL("memory_access.global_fallback_offset_range"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT,
            .constraint_key =
                IREE_SVL("memory_access.packed_register_footprint"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_FLAT_DYNAMIC_ADDRESS,
            .constraint_key = IREE_SVL("memory_access.flat_dynamic_address"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE,
            .constraint_key = IREE_SVL("memory_access.dynamic_offset_range"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET_RANGE,
            .constraint_key = IREE_SVL("memory_access.static_offset_range"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT,
            .constraint_key = IREE_SVL("memory_access.b128_dynamic_alignment"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET,
            .constraint_key = IREE_SVL("memory_access.negative_static_offset"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT,
            .constraint_key = IREE_SVL("memory_access.b128_static_alignment"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING,
            .constraint_key = IREE_SVL("memory_access.descriptor_missing"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LOW16_DESCRIPTOR_MISSING,
            .constraint_key =
                IREE_SVL("memory_access.low16_descriptor_missing"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SIGNED_I16_REPAIR_DESCRIPTOR_MISSING,
            .constraint_key =
                IREE_SVL("memory_access.signed_i16_repair_descriptor_missing"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE,
            .constraint_key =
                IREE_SVL("memory_access.descriptor_offset_immediate"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE,
            .constraint_key = IREE_SVL("memory_access.descriptor_offset_range"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE,
            .constraint_key = IREE_SVL("memory_access.memory_space"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT,
            .constraint_key = IREE_SVL("memory_access.workgroup_root"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PRIVATE_ROOT,
            .constraint_key = IREE_SVL("memory_access.private_root"),
        },
};

static iree_string_view_t loom_amdgpu_memory_access_descriptor_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access) {
  if (access->descriptor == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU memory access descriptor");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        access->descriptor->key_string_offset);
}

iree_status_t loom_amdgpu_record_memory_access_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }

  const iree_string_view_t packet_key =
      loom_amdgpu_memory_access_descriptor_key(descriptor_set, access);
  return loom_target_low_legality_record_memory_access(
      context, op, loom_amdgpu_memory_space_name(access->source.memory_space),
      loom_amdgpu_memory_operation_name(kind), packet_key,
      loom_amdgpu_memory_address_form_name(access->address_form),
      loom_amdgpu_memory_access_dynamic_term_kind_name(access),
      loom_amdgpu_memory_ds_addtid_reason_key(
          descriptor_set, loom_target_low_legality_module(context),
          loom_target_low_legality_function(context),
          loom_target_low_legality_bundle(context), access, kind),
      IREE_SV("selected"), access->source.static_byte_offset,
      access->source.element_byte_count, access->payload_register_count,
      access->source.dynamic_term_count == 1
          ? access->source.dynamic_terms[0].byte_stride
          : 0,
      access->source.vector_lane_byte_stride);
}

static iree_status_t loom_amdgpu_record_memory_cache_policy(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    iree_string_view_t decision_key, iree_string_view_t decision,
    const loom_amdgpu_memory_cache_policy_attrs_t* cache_attrs) {
  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  const bool scope_attr_present =
      cache_attrs &&
      iree_any_bit_set(cache_attrs->flags,
                       LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE);
  const bool th_attr_present =
      cache_attrs && iree_any_bit_set(cache_attrs->flags,
                                      LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH);
  const bool nt_attr_present =
      cache_attrs && iree_any_bit_set(cache_attrs->flags,
                                      LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT);
  return loom_target_low_legality_record_memory_cache_policy(
      context, op, loom_amdgpu_memory_space_name(access->source.memory_space),
      loom_amdgpu_memory_operation_name(kind),
      loom_amdgpu_cache_policy_scope_param(policy),
      loom_amdgpu_cache_policy_temporal_param(policy), decision_key, decision,
      loom_amdgpu_memory_cache_policy_encoding_key(descriptor_set),
      scope_attr_present, scope_attr_present ? cache_attrs->scope : 0,
      th_attr_present, th_attr_present ? cache_attrs->th : 0, nt_attr_present,
      nt_attr_present ? cache_attrs->nt : 0);
}

iree_status_t loom_amdgpu_record_memory_cache_policy_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_cache_policy_resolution_t resolution,
    const loom_amdgpu_memory_cache_policy_attrs_t* cache_attrs) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      resolution == LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_ABSENT) {
    return iree_ok_status();
  }

  IREE_ASSERT_NE(resolution,
                 LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED);
  const loom_low_source_memory_operation_kind_t kind =
      access->source.operation_kind;
  if (resolution == LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_DROPPED) {
    return loom_amdgpu_record_memory_cache_policy(
        context, op, descriptor_set, access, kind,
        IREE_SV("memory_cache_policy.unsupported"), IREE_SV("dropped"),
        /*cache_attrs=*/NULL);
  }
  return loom_amdgpu_record_memory_cache_policy(
      context, op, descriptor_set, access, kind,
      loom_amdgpu_memory_cache_policy_selected_key(descriptor_set),
      IREE_SV("selected"), cache_attrs);
}

iree_status_t loom_amdgpu_record_memory_cache_policy_rejection_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      !loom_amdgpu_memory_cache_policy_is_present(
          &access->source.cache_policy)) {
    return iree_ok_status();
  }

  const loom_low_source_memory_operation_kind_t kind =
      access->source.operation_kind;
  return loom_amdgpu_record_memory_cache_policy(
      context, op, descriptor_set, access, kind,
      loom_amdgpu_memory_cache_policy_rejection_key(
          descriptor_set, access, &access->source.cache_policy),
      IREE_SV("rejected"), /*cache_attrs=*/NULL);
}

iree_string_view_t loom_amdgpu_memory_access_rejection_key(
    loom_amdgpu_memory_access_rejection_flags_t rejection_bits) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuMemoryAccessRejectionKeys); ++i) {
    const loom_amdgpu_memory_access_rejection_key_t* row =
        &kAmdgpuMemoryAccessRejectionKeys[i];
    if (iree_any_bit_set(rejection_bits, row->rejection_bit)) {
      return row->constraint_key;
    }
  }
  return IREE_SV("memory_access.target_legal");
}
