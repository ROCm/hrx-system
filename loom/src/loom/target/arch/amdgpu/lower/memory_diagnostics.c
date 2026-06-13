// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"

#define LOOM_AMDGPU_LDS_BANK_COUNT 32u
#define LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES 4u

typedef struct loom_amdgpu_memory_access_bank_summary_t {
  // Distance between adjacent workitems in 32-bit LDS bank words.
  uint32_t bank_stride_words;
  // Estimated conflict degree across a 32-lane bank cycle, or zero when the
  // bank pattern is unknown from the current facts.
  uint32_t conflict_degree;
  // Stable bank-conflict classification token.
  iree_string_view_t conflict_kind;
} loom_amdgpu_memory_access_bank_summary_t;

static uint32_t loom_amdgpu_gcd_u32(uint32_t lhs, uint32_t rhs) {
  while (rhs != 0) {
    const uint32_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

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

static iree_string_view_t loom_amdgpu_memory_operation_name(
    loom_amdgpu_memory_operation_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_MEMORY_OPERATION_LOAD:
      return IREE_SV("load");
    case LOOM_AMDGPU_MEMORY_OPERATION_STORE:
      return IREE_SV("store");
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

static iree_string_view_t loom_amdgpu_source_memory_operation_name(
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
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_memory_diagnostic_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
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
          loom_amdgpu_source_memory_operation_name(source->operation_kind)),
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

iree_status_t loom_amdgpu_emit_memory_access_rejection_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic,
    bool* out_handled) {
  *out_handled = true;
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
};

static loom_amdgpu_memory_access_bank_summary_t
loom_amdgpu_memory_access_bank_summary(
    const loom_amdgpu_memory_access_t* access) {
  loom_amdgpu_memory_access_bank_summary_t summary = {
      .conflict_kind = IREE_SV("bank-pattern-unknown"),
  };
  const loom_low_source_memory_dynamic_term_t* term =
      loom_low_source_memory_access_single_dynamic_term(&access->source);
  if (access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      !term ||
      access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      term->byte_stride == 0 ||
      (term->byte_stride % LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES) != 0) {
    return summary;
  }

  summary.bank_stride_words =
      term->byte_stride / LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES;
  summary.conflict_degree = loom_amdgpu_gcd_u32(summary.bank_stride_words,
                                                LOOM_AMDGPU_LDS_BANK_COUNT);
  const uint32_t vector_footprint_bytes =
      access->packet_byte_count != 0
          ? access->packet_byte_count
          : access->source.element_byte_count *
                iree_max(access->payload_register_count, 1u);
  if (summary.conflict_degree == 1) {
    summary.conflict_kind = term->byte_stride > vector_footprint_bytes
                                ? IREE_SV("padded-bank-conflict-free")
                                : IREE_SV("bank-conflict-free");
  } else {
    summary.conflict_kind = IREE_SV("bank-conflict-risk");
  }
  return summary;
}

static iree_status_t loom_amdgpu_memory_access_descriptor_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    iree_string_view_t* out_packet_key) {
  *out_packet_key = IREE_SV("<missing>");
  if (access->descriptor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU memory access has no selected descriptor");
  }
  *out_packet_key = loom_low_descriptor_set_string(
      descriptor_set, access->descriptor->key_string_offset);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_record_memory_access_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }

  iree_string_view_t packet_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_descriptor_key(
      descriptor_set, access, &packet_key));
  const loom_amdgpu_memory_access_bank_summary_t bank_summary =
      loom_amdgpu_memory_access_bank_summary(access);
  return loom_target_low_legality_record_memory_access(
      context, op, loom_amdgpu_memory_space_name(access->source.memory_space),
      loom_amdgpu_memory_operation_name(kind), packet_key, IREE_SV("selected"),
      access->source.element_byte_count, access->payload_register_count,
      access->source.dynamic_term_count == 1
          ? access->source.dynamic_terms[0].byte_stride
          : 0,
      access->source.vector_lane_byte_stride, bank_summary.bank_stride_words,
      bank_summary.conflict_degree, bank_summary.conflict_kind);
}

static iree_status_t loom_amdgpu_record_memory_cache_policy(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind, iree_string_view_t decision_key,
    iree_string_view_t decision,
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
    loom_amdgpu_memory_operation_kind_t kind) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      !loom_amdgpu_memory_cache_policy_is_present(
          &access->source.cache_policy)) {
    return iree_ok_status();
  }

  loom_amdgpu_memory_cache_policy_attrs_t cache_attrs = {0};
  if (!loom_amdgpu_memory_cache_policy_encode(descriptor_set, access,
                                              &cache_attrs)) {
    return iree_ok_status();
  }
  return loom_amdgpu_record_memory_cache_policy(
      context, op, descriptor_set, access, kind,
      loom_amdgpu_memory_cache_policy_selected_key(descriptor_set),
      IREE_SV("selected"), &cache_attrs);
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

  const loom_amdgpu_memory_operation_kind_t kind =
      loom_amdgpu_memory_operation_kind_from_source(&access->source);
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
