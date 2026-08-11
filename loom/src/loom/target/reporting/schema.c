// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/schema.h"

#include <stdint.h>

#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"

iree_string_view_t loom_target_compile_report_artifact_kind_name(
    loom_target_compile_artifact_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE:
      return IREE_SV("hal-executable");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY:
      return IREE_SV("hal-kernel-library");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT:
      return IREE_SV("target-artifact");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_target_compile_report_source_low_selection_name(
    loom_target_compile_report_source_low_selection_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE:
      return IREE_SV("rule");
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN:
      return IREE_SV("plan");
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_host_size_t
loom_target_compile_report_source_low_memory_storage_fields_from_values(
    iree_string_view_t element_format, iree_string_view_t scale_format,
    iree_string_view_t secondary_scale_format,
    iree_string_view_t payload_packing, iree_string_view_t scale_topology,
    iree_string_view_t affine_policy, iree_string_view_t rounding_policy,
    iree_string_view_t codebook_policy, iree_string_view_t sparsity_policy,
    loom_target_compile_report_string_field_t* fields) {
  fields[0] = (loom_target_compile_report_string_field_t){
      .name = "element_format",
      .value = element_format,
  };
  fields[1] = (loom_target_compile_report_string_field_t){
      .name = "scale_format",
      .value = scale_format,
  };
  fields[2] = (loom_target_compile_report_string_field_t){
      .name = "secondary_scale_format",
      .value = secondary_scale_format,
  };
  fields[3] = (loom_target_compile_report_string_field_t){
      .name = "payload_packing",
      .value = payload_packing,
  };
  fields[4] = (loom_target_compile_report_string_field_t){
      .name = "scale_topology",
      .value = scale_topology,
  };
  fields[5] = (loom_target_compile_report_string_field_t){
      .name = "affine_policy",
      .value = affine_policy,
  };
  fields[6] = (loom_target_compile_report_string_field_t){
      .name = "rounding_policy",
      .value = rounding_policy,
  };
  fields[7] = (loom_target_compile_report_string_field_t){
      .name = "codebook_policy",
      .value = codebook_policy,
  };
  fields[8] = (loom_target_compile_report_string_field_t){
      .name = "sparsity_policy",
      .value = sparsity_policy,
  };
  return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT;
}

iree_host_size_t loom_target_compile_report_source_low_memory_storage_fields(
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_string_field_t* fields) {
  return loom_target_compile_report_source_low_memory_storage_fields_from_values(
      row->storage_element_format, row->storage_scale_format,
      row->storage_secondary_scale_format, row->storage_payload_packing,
      row->storage_scale_topology, row->storage_affine_policy,
      row->storage_rounding_policy, row->storage_codebook_policy,
      row->storage_sparsity_policy, fields);
}

iree_host_size_t
loom_target_compile_report_source_low_memory_argument_packet_storage_fields(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summary,
    loom_target_compile_report_string_field_t* fields) {
  return loom_target_compile_report_source_low_memory_storage_fields_from_values(
      summary->storage_element_format, summary->storage_scale_format,
      summary->storage_secondary_scale_format, summary->storage_payload_packing,
      summary->storage_scale_topology, summary->storage_affine_policy,
      summary->storage_rounding_policy, summary->storage_codebook_policy,
      summary->storage_sparsity_policy, fields);
}

iree_host_size_t
loom_target_compile_report_source_low_memory_strategy_storage_fields(
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        summary,
    loom_target_compile_report_string_field_t* fields) {
  return loom_target_compile_report_source_low_memory_storage_fields_from_values(
      summary->storage_element_format, summary->storage_scale_format,
      summary->storage_secondary_scale_format, summary->storage_payload_packing,
      summary->storage_scale_topology, summary->storage_affine_policy,
      summary->storage_rounding_policy, summary->storage_codebook_policy,
      summary->storage_sparsity_policy, fields);
}

iree_host_size_t loom_target_compile_report_first_non_empty_string_field(
    const loom_target_compile_report_string_field_t* fields,
    iree_host_size_t field_count) {
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (!iree_string_view_is_empty(fields[i].value)) {
      return i;
    }
  }
  return field_count;
}

bool loom_target_compile_report_memory_interval_has_range(
    const loom_target_compile_report_memory_interval_t* interval) {
  const loom_target_compile_report_memory_interval_flags_t range_flags =
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE;
  return iree_all_bits_set(interval->flags, range_flags);
}

bool loom_target_compile_report_source_low_memory_has_dynamic_evidence(
    const loom_target_compile_report_source_low_memory_summary_t* summary) {
  return summary->exact_dynamic_packet_count != 0 ||
         summary->unknown_dynamic_packet_count != 0;
}

bool loom_target_compile_report_source_low_memory_has_complete_dynamic_evidence(
    const loom_target_compile_report_source_low_memory_summary_t* summary) {
  return summary->packet_count == summary->exact_dynamic_packet_count &&
         summary->unknown_dynamic_packet_count == 0;
}

bool loom_target_compile_report_source_low_memory_has_dynamic_delta(
    const loom_target_compile_report_source_low_memory_summary_t* summary) {
  return summary->unknown_dynamic_packet_count != 0 ||
         summary->dynamic_packet_count != summary->packet_count ||
         summary->dynamic_source_byte_count != summary->source_byte_count ||
         summary->dynamic_read_byte_count != summary->read_byte_count ||
         summary->dynamic_write_byte_count != summary->write_byte_count ||
         summary->dynamic_issued_read_byte_count !=
             summary->issued_read_byte_count ||
         summary->dynamic_issued_write_byte_count !=
             summary->issued_write_byte_count ||
         summary->dynamic_issued_read_unknown_width_count !=
             summary->issued_read_unknown_width_count ||
         summary->dynamic_issued_write_unknown_width_count !=
             summary->issued_write_unknown_width_count;
}

bool loom_target_compile_report_source_low_memory_can_dispatch_scale(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload) {
  return iree_any_bit_set(
             workload->flags,
             LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT) &&
         (!loom_target_compile_report_source_low_memory_has_dynamic_evidence(
              summary) ||
          loom_target_compile_report_source_low_memory_has_complete_dynamic_evidence(
              summary));
}

bool loom_target_compile_report_source_low_memory_should_print_dynamic(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload) {
  if (!loom_target_compile_report_source_low_memory_has_dynamic_evidence(
          summary)) {
    return false;
  }
  return loom_target_compile_report_source_low_memory_has_dynamic_delta(
             summary) ||
         loom_target_compile_report_source_low_memory_can_dispatch_scale(
             summary, workload);
}
bool loom_target_compile_report_dispatch_memory_bytes(
    uint64_t read_byte_count, uint64_t write_byte_count,
    uint64_t dispatch_workitem_count,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes) {
  *out_bytes = (loom_target_compile_report_dispatch_memory_bytes_t){0};
  return iree_checked_mul_u64(read_byte_count, dispatch_workitem_count,
                              &out_bytes->read_byte_count) &&
         iree_checked_mul_u64(write_byte_count, dispatch_workitem_count,
                              &out_bytes->write_byte_count) &&
         iree_checked_add_u64(out_bytes->read_byte_count,
                              out_bytes->write_byte_count,
                              &out_bytes->total_byte_count);
}

bool loom_target_compile_report_source_low_memory_dispatch_source_bytes(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes) {
  if (!loom_target_compile_report_source_low_memory_can_dispatch_scale(
          summary, workload)) {
    return false;
  }
  const bool use_dynamic_counts =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(
          summary);
  const uint64_t read_byte_count = use_dynamic_counts
                                       ? summary->dynamic_read_byte_count
                                       : summary->read_byte_count;
  const uint64_t write_byte_count = use_dynamic_counts
                                        ? summary->dynamic_write_byte_count
                                        : summary->write_byte_count;
  return loom_target_compile_report_dispatch_memory_bytes(
      read_byte_count, write_byte_count, workload->dispatch_workitem_count,
      out_bytes);
}

bool loom_target_compile_report_source_low_memory_dispatch_issued_bytes(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes) {
  if (!loom_target_compile_report_source_low_memory_can_dispatch_scale(
          summary, workload)) {
    return false;
  }
  const bool use_dynamic_counts =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(
          summary);
  const uint64_t read_byte_count = use_dynamic_counts
                                       ? summary->dynamic_issued_read_byte_count
                                       : summary->issued_read_byte_count;
  const uint64_t write_byte_count =
      use_dynamic_counts ? summary->dynamic_issued_write_byte_count
                         : summary->issued_write_byte_count;
  return loom_target_compile_report_dispatch_memory_bytes(
      read_byte_count, write_byte_count, workload->dispatch_workitem_count,
      out_bytes);
}

iree_string_view_t
loom_target_compile_report_allocation_failure_blocking_kind_name(
    loom_target_compile_report_allocation_failure_blocking_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET:
      return IREE_SV("interval-exceeds-budget");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT:
      return IREE_SV("active-assignment");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT:
      return IREE_SV("location-constraint");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION:
      return IREE_SV("no-assignable-location");
    case LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_target_compile_report_pressure_origin_kind_name(
    loom_target_compile_report_pressure_origin_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BLOCK_ARGUMENT:
      return IREE_SV("block-argument");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONSTANT:
      return IREE_SV("constant");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_COPY:
      return IREE_SV("copy");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SLICE:
      return IREE_SV("slice");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONCAT:
      return IREE_SV("concat");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_STORAGE:
      return IREE_SV("storage");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SPILL_RELOAD:
      return IREE_SV("spill-reload");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_ALU:
      return IREE_SV("scalar-alu");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_VECTOR_ALU:
      return IREE_SV("vector-alu");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX:
      return IREE_SV("matrix");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT:
      return IREE_SV("dot");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY:
      return IREE_SV("global-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY:
      return IREE_SV("local-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_MEMORY:
      return IREE_SV("scalar-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY:
      return IREE_SV("private-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GENERIC_MEMORY:
      return IREE_SV("generic-memory");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL:
      return IREE_SV("control");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BARRIER:
      return IREE_SV("barrier");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONVERSION:
      return IREE_SV("conversion");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE:
      return IREE_SV("register-move");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CACHE:
      return IREE_SV("cache");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION:
      return IREE_SV("operation");
    case LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_target_compile_report_spill_row_kind_name(
    loom_target_compile_report_spill_row_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_PLANNED:
      return IREE_SV("planned");
    case LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_MATERIALIZED:
      return IREE_SV("materialized");
    case LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_target_compile_report_legalization_mode_name(
    loom_target_compile_report_legalization_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_EAGER:
      return IREE_SV("eager");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL:
      return IREE_SV("final");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_NONE:
    default:
      return IREE_SV("none");
  }
}

iree_string_view_t loom_target_compile_report_legalization_policy_name(
    loom_target_compile_report_legalization_policy_t policy) {
  switch (policy) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_PREFER_NATIVE:
      return IREE_SV("prefer-native");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY:
      return IREE_SV("reference-only");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REQUIRE_NATIVE:
      return IREE_SV("require-native");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_NONE:
    default:
      return IREE_SV("none");
  }
}

iree_string_view_t loom_target_compile_report_legalization_action_name(
    loom_target_compile_report_legalization_action_t action) {
  switch (action) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL:
      return IREE_SV("legal");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN:
      return IREE_SV("rewritten");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED:
      return IREE_SV("deferred");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR:
      return IREE_SV("reject-invalid-ir");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL:
      return IREE_SV("reject-unsupported-final");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_UNHANDLED:
      return IREE_SV("unhandled");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_NONE:
    default:
      return IREE_SV("none");
  }
}

iree_string_view_t loom_target_compile_report_legalization_outcome_name(
    loom_target_compile_report_legalization_outcome_t outcome) {
  switch (outcome) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_ALREADY_LEGAL:
      return IREE_SV("already-legal");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_TARGET_REWRITE:
      return IREE_SV("target-rewrite");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REFERENCE_FALLBACK:
      return IREE_SV("reference-fallback");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_DEFERRED:
      return IREE_SV("deferred");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_INVALID_IR:
      return IREE_SV("reject-invalid-ir");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_UNSUPPORTED:
      return IREE_SV("reject-unsupported");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_UNHANDLED:
      return IREE_SV("unhandled");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_NONE:
    default:
      return IREE_SV("none");
  }
}

iree_string_view_t loom_target_compile_report_contract_outcome_name(
    loom_target_compile_report_contract_outcome_t outcome) {
  switch (outcome) {
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNHANDLED:
      return IREE_SV("unhandled");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_LEGAL:
      return IREE_SV("legal");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED:
      return IREE_SV("unsupported");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_INVALID_IR:
      return IREE_SV("invalid-ir");
    case LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_NONE:
    default:
      return IREE_SV("none");
  }
}

iree_string_view_t loom_target_compile_report_capability_value_kind_name(
    loom_target_compile_report_capability_value_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL:
      return IREE_SV("bool");
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64:
      return IREE_SV("u64");
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING:
      return IREE_SV("string");
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_NONE:
    default:
      return IREE_SV("none");
  }
}

iree_string_view_t loom_target_compile_report_legalizer_strategy_name(
    loom_target_compile_report_legalizer_strategy_t strategy) {
  switch (strategy) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_TARGET:
      return IREE_SV("target");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE:
      return IREE_SV("reference");
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE:
    default:
      return IREE_SV("none");
  }
}

iree_string_view_t loom_target_compile_report_math_action_name(
    loom_target_compile_report_math_action_t action) {
  switch (action) {
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN:
      return IREE_SV("rewritten");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REJECTED:
      return IREE_SV("rejected");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_POLICY:
      return IREE_SV("missing-policy");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_RECIPE:
      return IREE_SV("missing-recipe");
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_NONE:
    default:
      return IREE_SV("none");
  }
}

const loom_target_compile_report_move_cause_descriptor_t
    loom_target_compile_report_move_cause_descriptors[] = {
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
         IREE_SVL("constant_materialization")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, IREE_SVL("low_copy")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_MOVE, IREE_SVL("low_move")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
         IREE_SVL("low_slice")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
         IREE_SVL("low_concat")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE,
         IREE_SVL("branch_edge")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION,
         IREE_SVL("operand_bank_materialization")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_CONSTRAINT_REPAIR,
         IREE_SVL("operand_constraint_repair")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_ABI_COPY, IREE_SVL("abi_copy")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_SPILL_RELOAD,
         IREE_SVL("spill_reload")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_PARTIAL_REGISTER_REPAIR,
         IREE_SVL("partial_register_repair")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_UNKNOWN, IREE_SVL("unknown")},
};
static_assert(
    IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors) ==
        LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT - 1,
    "move cause descriptor table must cover each reportable move cause");

iree_string_view_t loom_target_compile_report_type_kind_name(
    uint32_t type_kind) {
  switch (type_kind) {
    case LOOM_TYPE_NONE:
      return IREE_SV("none");
    case LOOM_TYPE_SCALAR:
      return IREE_SV("scalar");
    case LOOM_TYPE_TENSOR:
      return IREE_SV("tensor");
    case LOOM_TYPE_TILE:
      return IREE_SV("tile");
    case LOOM_TYPE_VECTOR:
      return IREE_SV("vector");
    case LOOM_TYPE_VIEW:
      return IREE_SV("view");
    case LOOM_TYPE_BUFFER:
      return IREE_SV("buffer");
    case LOOM_TYPE_FUNCTION:
      return IREE_SV("function");
    case LOOM_TYPE_ENCODING:
      return IREE_SV("encoding");
    case LOOM_TYPE_DIALECT:
      return IREE_SV("dialect");
    case LOOM_TYPE_POOL:
      return IREE_SV("pool");
    case LOOM_TYPE_REGISTER:
      return IREE_SV("register");
    case LOOM_TYPE_STORAGE:
      return IREE_SV("storage");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_target_compile_report_scalar_type_name(
    uint32_t element_type) {
  const char* name = loom_scalar_type_name((loom_scalar_type_t)element_type);
  if (name == NULL) {
    return IREE_SV("unknown");
  }
  return iree_make_cstring_view(name);
}

static void loom_target_compile_report_accumulate_move_cause(
    const loom_target_compile_report_move_cause_counts_t* counts,
    uint64_t* kind_count, uint64_t* packet_count, uint64_t* unit_count) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return;
  }
  ++*kind_count;
  *packet_count += counts->packet_count;
  *unit_count += counts->unit_count;
}

void loom_target_compile_report_move_cause_counts_totals(
    const loom_target_compile_report_move_cause_counts_t* counts,
    uint64_t* out_kind_count, uint64_t* out_packet_count,
    uint64_t* out_unit_count) {
  *out_kind_count = 0;
  *out_packet_count = 0;
  *out_unit_count = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
       ++i) {
    const loom_target_compile_report_move_cause_descriptor_t* descriptor =
        &loom_target_compile_report_move_cause_descriptors[i];
    loom_target_compile_report_accumulate_move_cause(
        &counts[descriptor->cause], out_kind_count, out_packet_count,
        out_unit_count);
  }
}

bool loom_target_compile_report_economics_has_memory(
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  return mix->memory_read_byte_count != 0 ||
         mix->memory_write_byte_count != 0 ||
         mix->memory_read_unknown_width_count != 0 ||
         mix->memory_write_unknown_width_count != 0;
}

bool loom_target_compile_report_economics_has_operations(
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  return mix->scalar_alu_count != 0 || mix->vector_alu_count != 0 ||
         mix->matrix_count != 0 || mix->mfma_count != 0 ||
         mix->smfmac_count != 0 || mix->wmma_count != 0 ||
         mix->swmmac_count != 0 || mix->dot_count != 0 ||
         mix->atomic_count != 0 || mix->branch_count != 0 ||
         mix->barrier_count != 0 || mix->control_count != 0 ||
         mix->conversion_count != 0 || mix->cache_count != 0 ||
         mix->register_move_count != 0;
}

bool loom_target_compile_report_has_economics(
    loom_target_compile_report_detail_flags_t detail_flags,
    const loom_target_compile_report_static_instruction_mix_t* mix,
    const loom_target_compile_report_workload_t* workload) {
  (void)workload;
  return iree_any_bit_set(
             detail_flags,
             LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX) &&
         (loom_target_compile_report_economics_has_memory(mix) ||
          loom_target_compile_report_economics_has_operations(mix));
}
