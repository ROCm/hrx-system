// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_format.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/util/json.h"

void loom_target_compile_report_format_options_initialize(
    loom_target_compile_report_format_options_t* out_options) {
  *out_options = (loom_target_compile_report_format_options_t){
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
}

iree_status_t loom_target_compile_report_format_mode_parse(
    iree_string_view_t value,
    loom_target_compile_report_format_mode_t* out_mode) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("details"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported compile report mode '%.*s'; expected "
                          "'none', 'summary', or 'details'",
                          (int)value.size, value.data);
}

static iree_string_view_t loom_target_compile_report_artifact_kind_name(
    loom_target_compile_artifact_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE:
      return IREE_SV("vm-archive");
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

static iree_string_view_t loom_target_compile_report_source_low_selection_name(
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

static iree_string_view_t
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

static iree_string_view_t loom_target_compile_report_pressure_origin_kind_name(
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

static iree_string_view_t loom_target_compile_report_legalization_mode_name(
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

static iree_string_view_t loom_target_compile_report_legalization_policy_name(
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

static iree_string_view_t loom_target_compile_report_legalization_action_name(
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

static iree_string_view_t loom_target_compile_report_legalization_outcome_name(
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

static iree_string_view_t loom_target_compile_report_contract_outcome_name(
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

static iree_string_view_t loom_target_compile_report_legalizer_strategy_name(
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

static iree_string_view_t loom_target_compile_report_math_action_name(
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

typedef struct loom_target_compile_report_move_cause_descriptor_t {
  // Residual move cause used as the report counter index.
  loom_target_compile_report_move_cause_t cause;
  // Stable text name emitted in compile reports.
  iree_string_view_t name;
} loom_target_compile_report_move_cause_descriptor_t;

static const loom_target_compile_report_move_cause_descriptor_t
    loom_target_compile_report_move_cause_descriptors[] = {
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
         IREE_SVL("constant_materialization")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, IREE_SVL("low_copy")},
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

static iree_string_view_t loom_target_compile_report_type_kind_name(
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
    case LOOM_TYPE_GROUP:
      return IREE_SV("group");
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

static iree_string_view_t loom_target_compile_report_scalar_type_name(
    uint32_t element_type) {
  const char* name = loom_scalar_type_name((loom_scalar_type_t)element_type);
  if (name == NULL) {
    return IREE_SV("unknown");
  }
  return iree_make_cstring_view(name);
}

static iree_string_view_t loom_target_compile_report_non_empty(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

static iree_status_t loom_target_compile_report_append_optional_u32(
    iree_string_builder_t* builder, uint32_t value) {
  if (value == UINT32_MAX) {
    return iree_string_builder_append_string(builder, IREE_SV("-"));
  }
  return iree_string_builder_append_format(builder, "%u", value);
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

static void loom_target_compile_report_move_cause_counts_totals(
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

static void loom_target_compile_report_move_cause_totals(
    const loom_target_compile_report_t* report, uint64_t* out_kind_count,
    uint64_t* out_packet_count, uint64_t* out_unit_count) {
  loom_target_compile_report_move_cause_counts_totals(
      report->move_causes, out_kind_count, out_packet_count, out_unit_count);
}

static iree_status_t loom_target_compile_report_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t name,
    iree_string_view_t value) {
  value = loom_target_compile_report_non_empty(value);
  return iree_string_builder_append_format(builder, " %.*s=%.*s",
                                           (int)name.size, name.data,
                                           (int)value.size, value.data);
}

static iree_status_t loom_target_compile_report_format_summary(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("COMPILE-REPORT: summary")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("artifact"),
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " status=%s", iree_status_code_string(report->status_code)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("function"), report->function_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("backend"), report->backend_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("bundle"), report->target_bundle_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("export"), report->target_export_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("export_symbol"), report->target_export_symbol));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("config"), report->target_config_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("lowered"), report->lowered_symbol));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " artifact_bytes=%" PRIu64, report->artifact_size));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("\n")));

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: schedule nodes=%" PRIu64 " scheduled=%" PRIu64
        " deps=%" PRIu64 " resources=%" PRIu64 " hazards=%" PRIu64
        " models=%" PRIu64 " pressure_classes=%" PRIu64
        " peak_live_units=%" PRIu64 "\n",
        report->schedule_node_count, report->scheduled_node_count,
        report->schedule_dependency_count, report->schedule_resource_use_count,
        report->schedule_hazard_gap_count, report->schedule_model_summary_count,
        report->register_pressure_summary_count,
        report->register_pressure_peak_live_units));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    const loom_target_compile_report_static_instruction_mix_t* mix =
        &report->static_instruction_mix;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: static_instruction_mix descriptors=%" PRIu64
        " unknown=%" PRIu64 " scalar_alu=%" PRIu64 " vector_alu=%" PRIu64
        " matrix=%" PRIu64 " mfma=%" PRIu64 " wmma=%" PRIu64 " dot=%" PRIu64
        " global_memory=%" PRIu64 " global_load=%" PRIu64
        " global_store=%" PRIu64 " buffer_load=%" PRIu64
        " buffer_store=%" PRIu64 " flat_memory=%" PRIu64
        " local_memory=%" PRIu64 " scalar_memory=%" PRIu64
        " generic_memory=%" PRIu64 " memory_read_unknown_width=%" PRIu64
        " memory_write_unknown_width=%" PRIu64 " memory_read_bytes=%" PRIu64
        " memory_write_bytes=%" PRIu64 " global_load_bytes=%" PRIu64
        " global_store_bytes=%" PRIu64 " buffer_load_bytes=%" PRIu64
        " buffer_store_bytes=%" PRIu64 " flat_read_bytes=%" PRIu64
        " flat_write_bytes=%" PRIu64 " local_read_bytes=%" PRIu64
        " local_write_bytes=%" PRIu64 " scalar_read_bytes=%" PRIu64
        " scalar_write_bytes=%" PRIu64 " unclassified_read_bytes=%" PRIu64
        " unclassified_write_bytes=%" PRIu64 " atomic=%" PRIu64
        " branch=%" PRIu64 " barrier=%" PRIu64 " control=%" PRIu64
        " conversion=%" PRIu64 " cache=%" PRIu64 " register_move=%" PRIu64 "\n",
        mix->descriptor_count, mix->unknown_count, mix->scalar_alu_count,
        mix->vector_alu_count, mix->matrix_count, mix->mfma_count,
        mix->wmma_count, mix->dot_count, mix->global_memory_count,
        mix->global_load_count, mix->global_store_count, mix->buffer_load_count,
        mix->buffer_store_count, mix->flat_memory_count,
        mix->local_memory_count, mix->scalar_memory_count,
        mix->generic_memory_count, mix->memory_read_unknown_width_count,
        mix->memory_write_unknown_width_count, mix->memory_read_byte_count,
        mix->memory_write_byte_count, mix->global_load_byte_count,
        mix->global_store_byte_count, mix->buffer_load_byte_count,
        mix->buffer_store_byte_count, mix->flat_read_byte_count,
        mix->flat_write_byte_count, mix->local_read_byte_count,
        mix->local_write_byte_count, mix->scalar_read_byte_count,
        mix->scalar_write_byte_count, mix->unclassified_read_byte_count,
        mix->unclassified_write_byte_count, mix->atomic_count,
        mix->branch_count, mix->barrier_count, mix->control_count,
        mix->conversion_count, mix->cache_count, mix->register_move_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: allocation assignments=%" PRIu64 " spills=%" PRIu64
        " spill_plans=%" PRIu64 " coalesced_copies=%" PRIu64
        " materialized_copies=%" PRIu64 " storage_leases=%" PRIu64
        " storage_lease_instances=%" PRIu64 " storage_release_actions=%" PRIu64
        "\n",
        report->allocation_assignment_count, report->allocation_spill_count,
        report->allocation_spill_plan_count,
        report->allocation_coalesced_copy_count,
        report->allocation_materialized_copy_count,
        report->allocation_storage_lease_count,
        report->allocation_storage_lease_instance_count,
        report->allocation_storage_release_action_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    const loom_target_compile_report_target_resources_t* resources =
        &report->target_resources;
    const iree_string_view_t scalar_register_class =
        loom_target_compile_report_non_empty(resources->scalar_register_class);
    const iree_string_view_t vector_register_class =
        loom_target_compile_report_non_empty(resources->vector_register_class);
    const iree_string_view_t limiting_resource =
        loom_target_compile_report_non_empty(resources->limiting_resource);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: target_resources scalar_register_class=%.*s "
        "scalar_registers=%" PRIu64 " scalar_pressure_peak=%" PRIu64
        " scalar_register_overhead=%" PRIu64
        " vector_register_class=%.*s "
        "vector_registers=%" PRIu64 " vector_pressure_peak=%" PRIu64
        " vector_register_overhead=%" PRIu64 " subgroup_size=%" PRIu32
        " resident_subgroups_per_simd=%" PRIu32
        " max_subgroups_per_simd=%" PRIu32 " occupancy_percent=%" PRIu32
        " limiting=%.*s\n",
        (int)scalar_register_class.size, scalar_register_class.data,
        resources->scalar_register_count,
        resources->scalar_pressure_peak_live_units,
        resources->scalar_register_overhead_units,
        (int)vector_register_class.size, vector_register_class.data,
        resources->vector_register_count,
        resources->vector_pressure_peak_live_units,
        resources->vector_register_overhead_units, resources->subgroup_size,
        resources->resident_subgroups_per_simd,
        resources->max_subgroups_per_simd, resources->occupancy_percent,
        (int)limiting_resource.size, limiting_resource.data));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    uint64_t kind_count = 0;
    uint64_t packet_count = 0;
    uint64_t unit_count = 0;
    loom_target_compile_report_move_cause_totals(report, &kind_count,
                                                 &packet_count, &unit_count);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: move_causes kinds=%" PRIu64 " packets=%" PRIu64
        " units=%" PRIu64 "\n",
        kind_count, packet_count, unit_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    const loom_target_compile_report_wait_plan_t* wait_plan =
        &report->wait_plan;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: wait_plan actions=%" PRIu64 " explicit=%" PRIu64
        " planned=%" PRIu64 " full_drains=%" PRIu64 " partial_waits=%" PRIu64
        " max_outstanding=%" PRIu64 " max_full_drain_outstanding=%" PRIu64
        " counter_rows=%" PRIhsz " action_rows=%" PRIhsz "\n",
        wait_plan->action_count, wait_plan->explicit_action_count,
        wait_plan->planned_action_count, wait_plan->full_drain_count,
        wait_plan->partial_wait_count, wait_plan->max_outstanding_before,
        wait_plan->max_full_drain_outstanding_before,
        report->wait_counter_rows.count, report->wait_action_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: emission instructions=%" PRIu64 " code_bytes=%" PRIu64
        " storage_bytes=%" PRIu64 "\n",
        report->emitted_instruction_count, report->emitted_code_byte_count,
        report->emitted_code_storage_byte_count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: math_legalization rewritten=%" PRIu64
        " rejected=%" PRIu64 " missing_policy=%" PRIu64
        " missing_recipe=%" PRIu64 " rows=%" PRIhsz "\n",
        report->math_legalization_rewritten_op_count,
        report->math_legalization_rejected_op_count,
        report->math_legalization_missing_policy_op_count,
        report->math_legalization_missing_recipe_op_count,
        report->math_legalization_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: source_low selected_ops=%" PRIu64
        " emitted_ops=%" PRIu64 " rows=%" PRIhsz "\n",
        report->source_low_selected_op_count,
        report->source_low_emitted_op_count, report->source_low_rows.count));
    if (report->source_low_memory_rows.count != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "COMPILE-REPORT: source_low_memory rows=%" PRIhsz "\n",
          report->source_low_memory_rows.count));
    }
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: target_legalization legal=%" PRIu64
        " rewritten=%" PRIu64 " target_rewritten=%" PRIu64
        " reference_rewritten=%" PRIu64 " deferred=%" PRIu64
        " invalid_ir=%" PRIu64 " unsupported=%" PRIu64 " unhandled=%" PRIu64
        " rows=%" PRIhsz "\n",
        report->target_legalization_legal_op_count,
        report->target_legalization_rewritten_op_count,
        report->target_legalization_target_rewritten_op_count,
        report->target_legalization_reference_rewritten_op_count,
        report->target_legalization_deferred_op_count,
        report->target_legalization_invalid_ir_op_count,
        report->target_legalization_unsupported_op_count,
        report->target_legalization_unhandled_op_count,
        report->target_legalization_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: memory private_bytes=%" PRIu64 " local_bytes=%" PRIu64
        "\n",
        report->private_memory_bytes, report->local_memory_bytes));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: pressure_rows count=%" PRIhsz "\n",
        report->pressure_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: pressure_origin_rows count=%" PRIhsz "\n",
        report->pressure_origin_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: schedule_band_rows count=%" PRIhsz "\n",
        report->schedule_band_rows.count));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: schedule_band_summary_rows count=%" PRIhsz "\n",
        report->schedule_band_summary_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: spill_rows count=%" PRIhsz "\n",
        report->spill_rows.count));
  }

  if (options->diagnostic_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: diagnostics count=%" PRIhsz "\n",
        options->diagnostic_count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: allocation_failure_rows count=%" PRIhsz "\n",
        report->allocation_failure_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: allocation_high_water_rows count=%" PRIhsz "\n",
        report->allocation_high_water_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: entries count=%" PRIhsz "\n",
        report->entry_rows.count));
  }

  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_entry_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->entry_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_entry_t* rows =
        (const loom_target_compile_report_entry_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_entry_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_function_name =
          loom_target_compile_report_non_empty(row->source_function_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_export_name =
          loom_target_compile_report_non_empty(row->target_export_name);
      const iree_string_view_t target_export_symbol =
          loom_target_compile_report_non_empty(row->target_export_symbol);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      uint64_t move_kind_count = 0;
      uint64_t move_packet_count = 0;
      uint64_t move_unit_count = 0;
      loom_target_compile_report_move_cause_counts_totals(
          row->move_causes, &move_kind_count, &move_packet_count,
          &move_unit_count);
      const loom_target_compile_report_wait_plan_t* wait_plan = &row->wait_plan;
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: entry[%" PRIhsz
          "] function=%.*s source=%.*s bundle=%.*s export=%.*s "
          "export_symbol=%.*s config=%.*s schedule_nodes=%" PRIu64
          " scheduled=%" PRIu64 " resource_uses=%" PRIu64
          " hazard_gaps=%" PRIu64 " model_summaries=%" PRIu64
          " pressure_summaries=%" PRIu64 " peak_live=%" PRIu64
          " assignments=%" PRIu64 " spills=%" PRIu64 " spill_plans=%" PRIu64
          " coalesced_copies=%" PRIu64 " materialized_copies=%" PRIu64
          " storage_leases=%" PRIu64 " storage_lease_instances=%" PRIu64
          " storage_release_actions=%" PRIu64 " move_kinds=%" PRIu64
          " move_packets=%" PRIu64 " move_units=%" PRIu64
          " wait_actions=%" PRIu64 " wait_explicit=%" PRIu64
          " wait_planned=%" PRIu64 " wait_full_drains=%" PRIu64
          " wait_partial=%" PRIu64 " wait_max_outstanding=%" PRIu64
          " wait_max_full_drain_outstanding=%" PRIu64 " instructions=%" PRIu64
          " code_bytes=%" PRIu64 " storage_bytes=%" PRIu64
          " private_bytes=%" PRIu64 " local_bytes=%" PRIu64,
          row_index, (int)function_name.size, function_name.data,
          (int)source_function_name.size, source_function_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_export_name.size, target_export_name.data,
          (int)target_export_symbol.size, target_export_symbol.data,
          (int)target_config_name.size, target_config_name.data,
          row->schedule_node_count, row->scheduled_node_count,
          row->schedule_resource_use_count, row->schedule_hazard_gap_count,
          row->schedule_model_summary_count,
          row->register_pressure_summary_count,
          row->register_pressure_peak_live_units,
          row->allocation_assignment_count, row->allocation_spill_count,
          row->allocation_spill_plan_count,
          row->allocation_coalesced_copy_count,
          row->allocation_materialized_copy_count,
          row->allocation_storage_lease_count,
          row->allocation_storage_lease_instance_count,
          row->allocation_storage_release_action_count, move_kind_count,
          move_packet_count, move_unit_count, wait_plan->action_count,
          wait_plan->explicit_action_count, wait_plan->planned_action_count,
          wait_plan->full_drain_count, wait_plan->partial_wait_count,
          wait_plan->max_outstanding_before,
          wait_plan->max_full_drain_outstanding_before,
          row->emitted_instruction_count, row->emitted_code_byte_count,
          row->emitted_code_storage_byte_count, row->private_memory_bytes,
          row->local_memory_bytes));
      if (iree_any_bit_set(
              row->detail_flags,
              LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
        const loom_target_compile_report_target_resources_t* resources =
            &row->target_resources;
        const iree_string_view_t scalar_register_class =
            loom_target_compile_report_non_empty(
                resources->scalar_register_class);
        const iree_string_view_t vector_register_class =
            loom_target_compile_report_non_empty(
                resources->vector_register_class);
        const iree_string_view_t limiting_resource =
            loom_target_compile_report_non_empty(resources->limiting_resource);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            " scalar_register_class=%.*s scalar_registers=%" PRIu64
            " scalar_pressure_peak=%" PRIu64
            " scalar_register_overhead=%" PRIu64
            " vector_register_class=%.*s vector_registers=%" PRIu64
            " vector_pressure_peak=%" PRIu64
            " vector_register_overhead=%" PRIu64 " subgroup_size=%" PRIu32
            " resident_subgroups_per_simd=%" PRIu32
            " max_subgroups_per_simd=%" PRIu32 " occupancy_percent=%" PRIu32
            " limiting=%.*s",
            (int)scalar_register_class.size, scalar_register_class.data,
            resources->scalar_register_count,
            resources->scalar_pressure_peak_live_units,
            resources->scalar_register_overhead_units,
            (int)vector_register_class.size, vector_register_class.data,
            resources->vector_register_count,
            resources->vector_pressure_peak_live_units,
            resources->vector_register_overhead_units, resources->subgroup_size,
            resources->resident_subgroups_per_simd,
            resources->max_subgroups_per_simd, resources->occupancy_percent,
            (int)limiting_resource.size, limiting_resource.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " pressure_rows=%" PRIhsz " pressure_origin_rows=%" PRIhsz
          " schedule_band_rows=%" PRIhsz " schedule_band_summary_rows=%" PRIhsz
          " spill_rows=%" PRIhsz " allocation_high_water_rows=%" PRIhsz
          " wait_counter_rows=%" PRIhsz " wait_action_rows=%" PRIhsz "\n",
          row->pressure_row_count, row->pressure_origin_row_count,
          row->schedule_band_row_count, row->schedule_band_summary_row_count,
          row->spill_row_count, row->allocation_high_water_row_count,
          row->wait_counter_row_count, row->wait_action_row_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_move_cause(
    iree_string_builder_t* builder,
    const loom_target_compile_report_move_cause_descriptor_t* descriptor,
    const loom_target_compile_report_move_cause_counts_t* counts) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return iree_ok_status();
  }
  return iree_string_builder_append_format(
      builder,
      "COMPILE-REPORT: move_cause[%.*s] packets=%" PRIu64 " units=%" PRIu64
      "\n",
      (int)descriptor->name.size, descriptor->name.data, counts->packet_count,
      counts->unit_count);
}

static iree_status_t loom_target_compile_report_format_move_causes(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
       ++i) {
    const loom_target_compile_report_move_cause_descriptor_t* descriptor =
        &loom_target_compile_report_move_cause_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause(
        builder, descriptor, &report->move_causes[descriptor->cause]));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_counter_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_counter_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_counter_row_t* rows =
        (const loom_target_compile_report_wait_counter_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_counter_row_t* row = &rows[i];
      const loom_target_compile_report_wait_plan_t* summary = &row->summary;
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_counter[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32 " actions=%" PRIu64
          " explicit=%" PRIu64 " planned=%" PRIu64 " full_drains=%" PRIu64
          " partial_waits=%" PRIu64 " max_outstanding=%" PRIu64
          " max_full_drain_outstanding=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          summary->action_count, summary->explicit_action_count,
          summary->planned_action_count, summary->full_drain_count,
          summary->partial_wait_count, summary->max_outstanding_before,
          summary->max_full_drain_outstanding_before));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_action_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_action_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_action_row_t* rows =
        (const loom_target_compile_report_wait_action_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_action_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
      const iree_string_view_t action_name =
          loom_target_compile_report_non_empty(row->action_name);
      const iree_string_view_t reason_name =
          loom_target_compile_report_non_empty(row->reason_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_action[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32
          " action=%.*s action_id=%" PRIu32 " reason=%.*s reason_id=%" PRIu32
          " block=%" PRIu32 " node=%" PRIu32 " ordinal=%" PRIu32
          " producer_node=",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          (int)action_name.size, action_name.data, row->action_id,
          (int)reason_name.size, reason_name.data, row->reason_id,
          row->block_index, row->node_index, row->scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->producer_node));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" producer_ordinal=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->producer_scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_operation"),
          row->producer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_descriptor_key"),
          row->producer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_semantic_tag"),
          row->producer_semantic_tag));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" consumer_node=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->consumer_node));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" consumer_ordinal=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->consumer_scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_operation"),
          row->consumer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_descriptor_key"),
          row->consumer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_semantic_tag"),
          row->consumer_semantic_tag));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " target_count=%" PRIu32 " outstanding_before=%" PRIu32 "\n",
          row->target_count, row->outstanding_before));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pressure_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->pressure_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_pressure_row_t* rows =
        (const loom_target_compile_report_pressure_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_pressure_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(
              (loom_scalar_type_t)row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_non_empty(row->peak_operation_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: pressure[%" PRIhsz
          "] function=%.*s class=%.*s type=%.*s element=%.*s "
          "peak_units=%" PRIu64 " peak_values=%" PRIu64
          " point=%u block=%.*s op=%.*s\n",
          row_index, (int)function_name.size, function_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          row->peak_live_units, row->peak_live_values, row->peak_point,
          (int)peak_block_name.size, peak_block_name.data,
          (int)peak_operation_name.size, peak_operation_name.data));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pressure_origin_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->pressure_origin_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_pressure_origin_row_t* rows =
        (const loom_target_compile_report_pressure_origin_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_pressure_origin_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_non_empty(row->peak_operation_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: pressure_origin[%" PRIhsz
          "] function=%.*s class=%.*s type=%.*s element=%.*s "
          "point=%u block=%.*s op=%.*s origin=%.*s origin_op=%.*s "
          "semantic=%.*s sample=%.*s live_units=%" PRIu64
          " live_values=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data, row->peak_point,
          (int)peak_block_name.size, peak_block_name.data,
          (int)peak_operation_name.size, peak_operation_name.data,
          (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data, row->live_units,
          row->live_values));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_schedule_band_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->schedule_band_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_schedule_band_row_t* rows =
        (const loom_target_compile_report_schedule_band_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_schedule_band_row_t* row = &rows[i];
      const loom_target_compile_report_static_instruction_mix_t* mix =
          &row->static_instruction_mix;
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: schedule_band[%" PRIhsz
          "] function=%.*s block=%.*s block_index=%" PRIu32
          " first_packet=%" PRIu64 " first_ordinal=%" PRIu32 " nodes=%" PRIu32
          " origin=%.*s"
          " origin_op=%.*s semantic=%.*s sample=%.*s"
          " descriptors=%" PRIu64 " unknown=%" PRIu64 " scalar_alu=%" PRIu64
          " vector_alu=%" PRIu64 " matrix=%" PRIu64 " mfma=%" PRIu64
          " wmma=%" PRIu64 " dot=%" PRIu64 " global_memory=%" PRIu64
          " local_memory=%" PRIu64 " scalar_memory=%" PRIu64
          " generic_memory=%" PRIu64 " atomic=%" PRIu64 " branch=%" PRIu64
          " barrier=%" PRIu64 " control=%" PRIu64 " conversion=%" PRIu64
          " cache=%" PRIu64 " register_move=%" PRIu64 " result_values=%" PRIu64
          " result_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)block_name.size, block_name.data, row->block_index,
          row->first_packet_index, row->first_scheduled_ordinal,
          row->node_count, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data,
          mix->descriptor_count, mix->unknown_count, mix->scalar_alu_count,
          mix->vector_alu_count, mix->matrix_count, mix->mfma_count,
          mix->wmma_count, mix->dot_count, mix->global_memory_count,
          mix->local_memory_count, mix->scalar_memory_count,
          mix->generic_memory_count, mix->atomic_count, mix->branch_count,
          mix->barrier_count, mix->control_count, mix->conversion_count,
          mix->cache_count, mix->register_move_count, row->result_value_count,
          row->result_unit_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->schedule_band_summary_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_schedule_band_summary_row_t* rows =
        (const loom_target_compile_report_schedule_band_summary_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_schedule_band_summary_row_t* row =
          &rows[i];
      const loom_target_compile_report_static_instruction_mix_t* mix =
          &row->static_instruction_mix;
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: schedule_band_summary[%" PRIhsz
          "] function=%.*s block=%.*s block_index=%" PRIu32
          " first_packet=%" PRIu64 " bands=%" PRIu64 " nodes=%" PRIu64
          " max_band_nodes=%" PRIu32
          " origin=%.*s"
          " origin_op=%.*s semantic=%.*s sample=%.*s"
          " descriptors=%" PRIu64 " unknown=%" PRIu64 " scalar_alu=%" PRIu64
          " vector_alu=%" PRIu64 " matrix=%" PRIu64 " mfma=%" PRIu64
          " wmma=%" PRIu64 " dot=%" PRIu64 " global_memory=%" PRIu64
          " local_memory=%" PRIu64 " scalar_memory=%" PRIu64
          " generic_memory=%" PRIu64 " atomic=%" PRIu64 " branch=%" PRIu64
          " barrier=%" PRIu64 " control=%" PRIu64 " conversion=%" PRIu64
          " cache=%" PRIu64 " register_move=%" PRIu64 " result_values=%" PRIu64
          " result_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)block_name.size, block_name.data, row->block_index,
          row->first_packet_index, row->band_count, row->node_count,
          row->max_band_node_count, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data,
          mix->descriptor_count, mix->unknown_count, mix->scalar_alu_count,
          mix->vector_alu_count, mix->matrix_count, mix->mfma_count,
          mix->wmma_count, mix->dot_count, mix->global_memory_count,
          mix->local_memory_count, mix->scalar_memory_count,
          mix->generic_memory_count, mix->atomic_count, mix->branch_count,
          mix->barrier_count, mix->control_count, mix->conversion_count,
          mix->cache_count, mix->register_move_count, row->result_value_count,
          row->result_unit_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_spill_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->spill_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_spill_row_t* rows =
        (const loom_target_compile_report_spill_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_spill_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t slot_space =
          loom_target_compile_report_non_empty(row->slot_space);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: spill[%" PRIhsz
          "] function=%.*s value=%.*s class=%.*s type=%.*s element=%.*s "
          "assignment=%u slot=%u space=%.*s bytes=%" PRIu64 " align=%" PRIu64
          " stores=%" PRIu64 " reloads=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)value_name.size, value_name.data, (int)register_class.size,
          register_class.data, (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          row->assignment_index, row->slot_index, (int)slot_space.size,
          slot_space.data, row->byte_size, row->byte_alignment,
          row->store_count, row->reload_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_allocation_failure_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->allocation_failure_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_allocation_failure_row_t* rows =
        (const loom_target_compile_report_allocation_failure_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_allocation_failure_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t failure_code =
          loom_target_compile_report_non_empty(row->failure_code);
      const iree_string_view_t blocking_kind =
          loom_target_compile_report_allocation_failure_blocking_kind_name(
              row->blocking_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t origin_block_name =
          loom_target_compile_report_non_empty(row->origin_block_name);
      const iree_string_view_t location_kind =
          loom_target_compile_report_non_empty(row->location_kind);
      const iree_string_view_t conflict_value_name =
          loom_target_compile_report_non_empty(row->conflict_value_name);
      const iree_string_view_t conflict_location_kind =
          loom_target_compile_report_non_empty(row->conflict_location_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: allocation_failure[%" PRIhsz
          "] function=%.*s value=%.*s class=%.*s code=%.*s blocking=%.*s "
          "origin=%.*s block=%.*s start=%u end=%u required_units=%u "
          "budget_units=",
          row_index, (int)function_name.size, function_name.data,
          (int)value_name.size, value_name.data, (int)register_class.size,
          register_class.data, (int)failure_code.size, failure_code.data,
          (int)blocking_kind.size, blocking_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)origin_block_name.size, origin_block_name.data, row->start_point,
          row->end_point, row->required_unit_count));
      if (row->budget_units == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->budget_units));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " peak_live_units=%u location=%.*s[", row->peak_live_units,
          (int)location_kind.size, location_kind.data));
      if (row->location_base == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->location_base));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ":%u] conflict_assignment=", row->location_count));
      if (row->conflict_assignment_index == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_assignment_index));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " conflict_value=%.*s conflict_start=",
          (int)conflict_value_name.size, conflict_value_name.data));
      if (row->conflict_start_point == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_start_point));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" conflict_end=")));
      if (row->conflict_end_point == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_end_point));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " conflict_location=%.*s[", (int)conflict_location_kind.size,
          conflict_location_kind.data));
      if (row->conflict_location_base == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_location_base));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ":%u]\n", row->conflict_location_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->allocation_high_water_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_allocation_high_water_row_t* rows =
        (const loom_target_compile_report_allocation_high_water_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_allocation_high_water_row_t* row =
          &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t location_kind =
          loom_target_compile_report_non_empty(row->location_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: allocation_high_water[%" PRIhsz
          "] function=%.*s value=%.*s class=%.*s type=%.*s element=%.*s "
          "assignment=%u origin=%.*s origin_op=%.*s semantic=%.*s start=%u "
          "end=%u required_units=%u location=%.*s[%u:%u] high_water=%" PRIu64
          " lower_free_units=%" PRIu64
          " lower_free_runs=%u "
          "lower_largest_free_run_units=%u"
          " lower_pressure_releasable_free_units=%" PRIu64
          " lower_pressure_releasable_free_runs=%u"
          " lower_pressure_releasable_largest_free_run_units=%u"
          " active_assignment_blockers=%u "
          "active_assignment_blocker_units=%" PRIu64
          " active_storage_lease_blockers=%u "
          "active_storage_lease_blocker_units=%" PRIu64
          " active_pressure_storage_lease_blockers=%u "
          "active_pressure_storage_lease_blocker_units=%" PRIu64
          " active_fallback_storage_lease_blockers=%u "
          "active_fallback_storage_lease_blocker_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)value_name.size, value_name.data, (int)register_class.size,
          register_class.data, (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          row->assignment_index, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data, row->start_point,
          row->end_point, row->required_unit_count, (int)location_kind.size,
          location_kind.data, row->location_base, row->location_count,
          row->high_water_units, row->lower_free_unit_count,
          row->lower_free_run_count, row->lower_largest_free_run_unit_count,
          row->lower_pressure_releasable_free_unit_count,
          row->lower_pressure_releasable_free_run_count,
          row->lower_pressure_releasable_largest_free_run_unit_count,
          row->active_assignment_blocker_count,
          row->active_assignment_blocker_units,
          row->active_storage_lease_blocker_count,
          row->active_storage_lease_blocker_units,
          row->active_pressure_storage_lease_blocker_count,
          row->active_pressure_storage_lease_blocker_units,
          row->active_fallback_storage_lease_blocker_count,
          row->active_fallback_storage_lease_blocker_units));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_row_t* rows =
        (const loom_target_compile_report_source_low_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t selection_name =
          loom_target_compile_report_source_low_selection_name(
              row->selection_kind);
      const iree_string_view_t plan_key = row->plan_key;
      if (row->selection_kind ==
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE) {
        if (!iree_string_view_is_empty(plan_key)) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder,
              "COMPILE-REPORT: source_low[%" PRIhsz
              "] function=%.*s source_op=%.*s selection=%.*s rule_set=%u "
              "rule=%u plan_key=%.*s descriptor=%" PRIu64 " emitted_ops=%u\n",
              row_index, (int)function_name.size, function_name.data,
              (int)source_op_name.size, source_op_name.data,
              (int)selection_name.size, selection_name.data,
              row->rule_set_index, row->rule_index, (int)plan_key.size,
              plan_key.data, row->descriptor_id, row->emitted_low_op_count));
        } else {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder,
              "COMPILE-REPORT: source_low[%" PRIhsz
              "] function=%.*s source_op=%.*s selection=%.*s rule_set=%u "
              "rule=%u descriptor=%" PRIu64 " emitted_ops=%u\n",
              row_index, (int)function_name.size, function_name.data,
              (int)source_op_name.size, source_op_name.data,
              (int)selection_name.size, selection_name.data,
              row->rule_set_index, row->rule_index, row->descriptor_id,
              row->emitted_low_op_count));
        }
        continue;
      }
      if (!iree_string_view_is_empty(plan_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: source_low[%" PRIhsz
            "] function=%.*s source_op=%.*s selection=%.*s plan=%" PRIu64
            " plan_key=%.*s descriptor=%" PRIu64 " emitted_ops=%u\n",
            row_index, (int)function_name.size, function_name.data,
            (int)source_op_name.size, source_op_name.data,
            (int)selection_name.size, selection_name.data, row->plan_id,
            (int)plan_key.size, plan_key.data, row->descriptor_id,
            row->emitted_low_op_count));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: source_low[%" PRIhsz
            "] function=%.*s source_op=%.*s selection=%.*s plan=%" PRIu64
            " descriptor=%" PRIu64 " emitted_ops=%u\n",
            row_index, (int)function_name.size, function_name.data,
            (int)source_op_name.size, source_op_name.data,
            (int)selection_name.size, selection_name.data, row->plan_id,
            row->descriptor_id, row->emitted_low_op_count));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_memory_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_row_t* rows =
        (const loom_target_compile_report_source_low_memory_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_non_empty(row->packet_key);
      const iree_string_view_t address_form =
          loom_target_compile_report_non_empty(row->address_form);
      const iree_string_view_t dynamic_term_kind =
          loom_target_compile_report_non_empty(row->dynamic_term_kind);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_non_empty(row->fallback_reason);
      const iree_string_view_t bank_conflict_kind =
          loom_target_compile_report_non_empty(row->bank_conflict_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory[%" PRIhsz
          "] function=%.*s source_op=%.*s memory_space=%.*s operation=%.*s "
          "packet=%.*s address_form=%.*s dynamic_term_kind=%.*s "
          "fallback_reason=%.*s static_offset_bytes=%" PRId64
          " element_bytes=%u "
          "vector_lanes=%u dynamic_stride_bytes=%u "
          "vector_lane_stride_bytes=%u bank_stride_words=%u "
          "bank_conflict_degree=%u bank_conflict_kind=%.*s\n",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)memory_space.size,
          memory_space.data, (int)operation_kind.size, operation_kind.data,
          (int)packet_key.size, packet_key.data, (int)address_form.size,
          address_form.data, (int)dynamic_term_kind.size,
          dynamic_term_kind.data, (int)fallback_reason.size,
          fallback_reason.data, row->static_offset_bytes,
          row->element_byte_count, row->vector_lane_count,
          row->dynamic_stride_bytes, row->vector_lane_stride_bytes,
          row->bank_stride_words, row->bank_conflict_degree,
          (int)bank_conflict_kind.size, bank_conflict_kind.data));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_math_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->math_legalization_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_math_row_t* rows =
        (const loom_target_compile_report_math_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_math_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t policy_name =
          loom_target_compile_report_non_empty(row->policy_name);
      const iree_string_view_t constraint_key =
          loom_target_compile_report_non_empty(row->constraint_key);
      const iree_string_view_t action_name =
          loom_target_compile_report_math_action_name(row->action);
      const iree_string_view_t math_op_name =
          loom_target_math_op_name((loom_target_math_op_t)row->math_op);
      const iree_string_view_t lane_domain_name =
          loom_target_math_lane_domain_name(
              (loom_target_math_lane_domain_t)row->lane_domain);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t recipe_name =
          loom_target_math_recipe_name((loom_target_math_recipe_t)row->recipe);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: math_legalization[%" PRIhsz
          "] function=%.*s source_op=%.*s action=%.*s policy=%.*s "
          "math_op=%.*s domain=%.*s element=%.*s recipe=%.*s "
          "constraint=%.*s bundle=%.*s config=%.*s source_fastmath=0x%02x "
          "recipe_fastmath=0x%02x created_ops=%" PRIu64 " erased_ops=%" PRIu64
          "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)action_name.size,
          action_name.data, (int)policy_name.size, policy_name.data,
          (int)math_op_name.size, math_op_name.data, (int)lane_domain_name.size,
          lane_domain_name.data, (int)element_type_name.size,
          element_type_name.data, (int)recipe_name.size, recipe_name.data,
          (int)constraint_key.size, constraint_key.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data,
          (uint32_t)row->source_fastmath_flags,
          (uint32_t)row->recipe_fastmath_flags, row->created_op_count,
          row->erased_op_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_legalization_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_legalization_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_legalization_row_t* rows =
        (const loom_target_compile_report_legalization_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_legalization_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t legalizer_name =
          loom_target_compile_report_non_empty(row->legalizer_name);
      const iree_string_view_t strategy_name =
          loom_target_compile_report_legalizer_strategy_name(
              row->legalizer_strategy);
      const iree_string_view_t mode_name =
          loom_target_compile_report_legalization_mode_name(row->mode);
      const iree_string_view_t policy_name =
          loom_target_compile_report_legalization_policy_name(row->policy);
      const iree_string_view_t action_name =
          loom_target_compile_report_legalization_action_name(row->action);
      const iree_string_view_t legalization_outcome_name =
          loom_target_compile_report_legalization_outcome_name(
              row->legalization_outcome);
      const iree_string_view_t outcome_name =
          loom_target_compile_report_contract_outcome_name(
              row->contract_outcome);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: target_legalization[%" PRIhsz
          "] function=%.*s source_op=%.*s mode=%.*s policy=%.*s "
          "action=%.*s outcome=%.*s contract=%.*s legalizer=%.*s strategy=%.*s "
          "bundle=%.*s config=%.*s binding=%u case=%u rule_set=%u rule=%u "
          "diagnostic=%u "
          "descriptor=%" PRIu64 " source_rejections=0x%08" PRIx32
          " source_rejection_detail=%" PRIu32 " target_rejections=0x%08" PRIx32
          " missing_features=0x%08" PRIx32 " missing_facts=0x%08" PRIx32
          " created_ops=%" PRIu64 " erased_ops=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)mode_name.size,
          mode_name.data, (int)policy_name.size, policy_name.data,
          (int)action_name.size, action_name.data,
          (int)legalization_outcome_name.size, legalization_outcome_name.data,
          (int)outcome_name.size, outcome_name.data, (int)legalizer_name.size,
          legalizer_name.data, (int)strategy_name.size, strategy_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data,
          row->binding_index, row->case_index, row->rule_set_index,
          row->rule_index, row->diagnostic_index, row->descriptor_id,
          row->source_rejection_bits, row->source_rejection_detail,
          row->target_rejection_bits, row->missing_feature_bits,
          row->missing_fact_bits, row->created_op_count, row->erased_op_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_json_begin_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name) {
  if (!*inout_first_field) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  }
  *inout_first_field = false;
  return loom_output_stream_write_format(stream, "\"%s\":", name);
}

static iree_status_t loom_target_compile_report_json_write_string_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t
loom_target_compile_report_json_write_optional_string_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  if (iree_string_view_is_empty(value)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_target_compile_report_json_write_u64_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    uint64_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIu64, value);
}

static iree_status_t loom_target_compile_report_json_write_u32_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    uint32_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  return loom_output_stream_write_format(stream, "%u", value);
}

static iree_status_t loom_target_compile_report_json_write_i64_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    int64_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  return loom_output_stream_write_format(stream, "%" PRId64, value);
}

static iree_status_t loom_target_compile_report_json_write_int_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    int value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  return loom_output_stream_write_format(stream, "%d", value);
}

static iree_status_t loom_target_compile_report_json_write_size_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    iree_host_size_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIhsz, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u16_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    uint16_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  if (value == UINT16_MAX) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%u", value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u64_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    uint64_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  if (value == UINT64_MAX) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu64, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u32_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    uint32_t value) {
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, inout_first_field, name));
  if (value == UINT32_MAX) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%u", value);
}

static iree_status_t loom_target_compile_report_format_schedule_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "node_count", report->schedule_node_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scheduled_node_count",
      report->scheduled_node_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "dependency_count",
      report->schedule_dependency_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "resource_use_count",
      report->schedule_resource_use_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "hazard_gap_count",
      report->schedule_hazard_gap_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "model_summary_count",
      report->schedule_model_summary_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "register_pressure_summary_count",
      report->register_pressure_summary_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "register_pressure_peak_live_units",
      report->register_pressure_peak_live_units));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_instruction_mix_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "descriptor_count", mix->descriptor_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "unknown_count", mix->unknown_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scalar_alu_count", mix->scalar_alu_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "vector_alu_count", mix->vector_alu_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "matrix_count", mix->matrix_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "mfma_count", mix->mfma_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "wmma_count", mix->wmma_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "dot_count", mix->dot_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "global_memory_count", mix->global_memory_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "global_load_count", mix->global_load_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "global_store_count", mix->global_store_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "buffer_load_count", mix->buffer_load_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "buffer_store_count", mix->buffer_store_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "flat_memory_count", mix->flat_memory_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "local_memory_count", mix->local_memory_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scalar_memory_count", mix->scalar_memory_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "generic_memory_count", mix->generic_memory_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "memory_read_unknown_width_count",
      mix->memory_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "memory_write_unknown_width_count",
      mix->memory_write_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "memory_read_byte_count",
      mix->memory_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "memory_write_byte_count",
      mix->memory_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "global_load_byte_count",
      mix->global_load_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "global_store_byte_count",
      mix->global_store_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "buffer_load_byte_count",
      mix->buffer_load_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "buffer_store_byte_count",
      mix->buffer_store_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "flat_read_byte_count", mix->flat_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "flat_write_byte_count",
      mix->flat_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "local_read_byte_count",
      mix->local_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "local_write_byte_count",
      mix->local_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scalar_read_byte_count",
      mix->scalar_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scalar_write_byte_count",
      mix->scalar_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "unclassified_read_byte_count",
      mix->unclassified_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "unclassified_write_byte_count",
      mix->unclassified_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "atomic_count", mix->atomic_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "branch_count", mix->branch_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "barrier_count", mix->barrier_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "control_count", mix->control_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "conversion_count", mix->conversion_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "cache_count", mix->cache_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "register_move_count", mix->register_move_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_allocation_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "assignment_count",
      report->allocation_assignment_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "spill_count", report->allocation_spill_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "spill_plan_count",
      report->allocation_spill_plan_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "coalesced_copy_count",
      report->allocation_coalesced_copy_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "materialized_copy_count",
      report->allocation_materialized_copy_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "storage_lease_count",
      report->allocation_storage_lease_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "storage_lease_instance_count",
      report->allocation_storage_lease_instance_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "storage_release_action_count",
      report->allocation_storage_release_action_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_move_cause_json(
    loom_output_stream_t* stream,
    const loom_target_compile_report_move_cause_descriptor_t* descriptor,
    const loom_target_compile_report_move_cause_counts_t* counts,
    bool* inout_first_row) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return iree_ok_status();
  }
  if (!*inout_first_row) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  }
  *inout_first_row = false;
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "cause", descriptor->name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "packet_count", counts->packet_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "unit_count", counts->unit_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_move_cause_counts_json(
    const loom_target_compile_report_move_cause_counts_t* counts,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  uint64_t kind_count = 0;
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  loom_target_compile_report_move_cause_counts_totals(
      counts, &kind_count, &packet_count, &unit_count);
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "kind_count", kind_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "packet_count", packet_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "unit_count", unit_count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "causes"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    bool first_row = true;
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
         ++i) {
      const loom_target_compile_report_move_cause_descriptor_t* descriptor =
          &loom_target_compile_report_move_cause_descriptors[i];
      IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause_json(
          stream, descriptor, &counts[descriptor->cause], &first_row));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_move_causes_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  return loom_target_compile_report_format_move_cause_counts_json(
      report->move_causes, mode, stream);
}

static iree_status_t loom_target_compile_report_format_wait_plan_json(
    const loom_target_compile_report_wait_plan_t* wait_plan,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "action_count", wait_plan->action_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "explicit_action_count",
      wait_plan->explicit_action_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "planned_action_count",
      wait_plan->planned_action_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "full_drain_count", wait_plan->full_drain_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "partial_wait_count",
      wait_plan->partial_wait_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "max_outstanding_before",
      wait_plan->max_outstanding_before));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "max_full_drain_outstanding_before",
      wait_plan->max_full_drain_outstanding_before));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_wait_counter_row_json(
    const loom_target_compile_report_wait_counter_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "counter", row->counter_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "counter_id", row->counter_id));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, &first_field, "summary"));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_plan_json(&row->summary, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_wait_counter_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->wait_counter_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_counter_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_counter_row_t* rows =
          (const loom_target_compile_report_wait_counter_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_counter_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_wait_action_row_json(
    const loom_target_compile_report_wait_action_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "counter", row->counter_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "counter_id", row->counter_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "action", row->action_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "action_id", row->action_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "reason", row->reason_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "reason_id", row->reason_id));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "block_index", row->block_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "node_index", row->node_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "scheduled_ordinal", row->scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "producer_node", row->producer_node));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "producer_scheduled_ordinal",
      row->producer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "producer_operation",
          row->producer_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "producer_descriptor_key",
          row->producer_descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "producer_semantic_tag",
          row->producer_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "consumer_node", row->consumer_node));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "consumer_scheduled_ordinal",
      row->consumer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "consumer_operation",
          row->consumer_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "consumer_descriptor_key",
          row->consumer_descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "consumer_semantic_tag",
          row->consumer_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "target_count", row->target_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "outstanding_before", row->outstanding_before));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_wait_action_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->wait_action_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_action_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_action_row_t* rows =
          (const loom_target_compile_report_wait_action_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_action_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_emission_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "instruction_count",
      report->emitted_instruction_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "code_byte_count",
      report->emitted_code_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "code_storage_byte_count",
      report->emitted_code_storage_byte_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_memory_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "private_bytes", report->private_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "local_bytes", report->local_memory_bytes));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_target_resources_json(
    const loom_target_compile_report_target_resources_t* resources,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "scalar_register_class",
          resources->scalar_register_class));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scalar_register_count",
      resources->scalar_register_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scalar_pressure_peak_live_units",
      resources->scalar_pressure_peak_live_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scalar_register_overhead_units",
      resources->scalar_register_overhead_units));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "vector_register_class",
          resources->vector_register_class));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "vector_register_count",
      resources->vector_register_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "vector_pressure_peak_live_units",
      resources->vector_pressure_peak_live_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "vector_register_overhead_units",
      resources->vector_register_overhead_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "subgroup_size", resources->subgroup_size));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "max_subgroups_per_simd",
      resources->max_subgroups_per_simd));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "resident_subgroups_per_simd",
      resources->resident_subgroups_per_simd));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "occupancy_percent", resources->occupancy_percent));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "limiting_resource",
          resources->limiting_resource));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_entry_json(
    const loom_target_compile_report_entry_t* row, iree_host_size_t row_index,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "source_function", row->source_function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_bundle", row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_snapshot", row->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_export", row->target_export_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_export_symbol",
          row->target_export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_config", row->target_config_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "detail_flags", row->detail_flags));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "schedule_node_count", row->schedule_node_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "scheduled_node_count", row->scheduled_node_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "schedule_dependency_count",
      row->schedule_dependency_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "schedule_resource_use_count",
      row->schedule_resource_use_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "schedule_hazard_gap_count",
      row->schedule_hazard_gap_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "schedule_model_summary_count",
      row->schedule_model_summary_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "register_pressure_summary_count",
      row->register_pressure_summary_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "register_pressure_peak_live_units",
      row->register_pressure_peak_live_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_assignment_count",
      row->allocation_assignment_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_spill_count",
      row->allocation_spill_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_spill_plan_count",
      row->allocation_spill_plan_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_coalesced_copy_count",
      row->allocation_coalesced_copy_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_materialized_copy_count",
      row->allocation_materialized_copy_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_storage_lease_count",
      row->allocation_storage_lease_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_storage_lease_instance_count",
      row->allocation_storage_lease_instance_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "allocation_storage_release_action_count",
      row->allocation_storage_release_action_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, &first_field, "move_causes"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause_counts_json(
      row->move_causes, mode, stream));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "wait_plan"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_wait_plan_json(
        &row->wait_plan, stream));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "instruction_count",
      row->emitted_instruction_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "code_byte_count", row->emitted_code_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "code_storage_byte_count",
      row->emitted_code_storage_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "private_memory_bytes", row->private_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "local_memory_bytes", row->local_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "pressure_row_count", row->pressure_row_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "pressure_origin_row_count",
      row->pressure_origin_row_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "schedule_band_row_count",
      row->schedule_band_row_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "schedule_band_summary_row_count",
      row->schedule_band_summary_row_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "spill_row_count", row->spill_row_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "allocation_high_water_row_count",
      row->allocation_high_water_row_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "wait_counter_row_count",
      row->wait_counter_row_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "wait_action_row_count",
      row->wait_action_row_count));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "target_resources"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_resources_json(
            &row->target_resources, stream));
  }
  if (iree_any_bit_set(
          row->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "static_instruction_mix"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->static_instruction_mix, stream));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_entries_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->entry_rows.count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, &first_field, "rows"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->entry_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_entry_t* rows =
        (const loom_target_compile_report_entry_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      if (row_index != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
      }
      IREE_RETURN_IF_ERROR(loom_target_compile_report_format_entry_json(
          &rows[i], row_index, mode, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_pressure_row_json(
    const loom_target_compile_report_pressure_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "register_class", row->register_class));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "type_kind", row->type_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "type",
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "element_type", row->element_type));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "element",
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "peak_live_units", row->peak_live_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "peak_live_values", row->peak_live_values));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "peak_point", row->peak_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "peak_block", row->peak_block_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "peak_operation", row->peak_operation_name));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_pressure_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->pressure_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->pressure_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_pressure_row_t* rows =
          (const loom_target_compile_report_pressure_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pressure_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_pressure_origin_row_json(
    const loom_target_compile_report_pressure_origin_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "register_class", row->register_class));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "type_kind", row->type_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "type",
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "element_type", row->element_type));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "element",
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "peak_point", row->peak_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "peak_block", row->peak_block_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "peak_operation", row->peak_operation_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "origin_kind", row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "origin",
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "origin_operation",
          row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "semantic_tag", row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "sample_value", row->sample_value_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "live_units", row->live_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "live_values", row->live_values));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_pressure_origin_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->pressure_origin_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->pressure_origin_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_pressure_origin_row_t* rows =
          (const loom_target_compile_report_pressure_origin_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pressure_origin_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_schedule_band_row_json(
    const loom_target_compile_report_schedule_band_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "block", row->block_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "block_index", row->block_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "first_packet_index", row->first_packet_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "first_scheduled_ordinal",
      row->first_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "node_count", row->node_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "origin_kind", row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "origin",
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "origin_operation",
          row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "semantic_tag", row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "sample_value", row->sample_value_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, &first_field, "static_instruction_mix"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
      &row->static_instruction_mix, stream));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "result_value_count", row->result_value_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "result_unit_count", row->result_unit_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_schedule_band_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->schedule_band_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->schedule_band_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_schedule_band_row_t* rows =
          (const loom_target_compile_report_schedule_band_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_schedule_band_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_row_json(
    const loom_target_compile_report_schedule_band_summary_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "block", row->block_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "block_index", row->block_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "first_packet_index", row->first_packet_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "band_count", row->band_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "node_count", row->node_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "max_band_node_count", row->max_band_node_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "origin_kind", row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "origin",
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "origin_operation",
          row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "semantic_tag", row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "sample_value", row->sample_value_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, &first_field, "static_instruction_mix"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
      &row->static_instruction_mix, stream));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "result_value_count", row->result_value_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "result_unit_count", row->result_unit_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->schedule_band_summary_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->schedule_band_summary_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_schedule_band_summary_row_t* rows =
          (const loom_target_compile_report_schedule_band_summary_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_schedule_band_summary_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_spill_row_json(
    const loom_target_compile_report_spill_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "value", row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "register_class", row->register_class));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "type_kind", row->type_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "type",
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "element_type", row->element_type));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "element",
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "assignment_index", row->assignment_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "slot_index", row->slot_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "slot_space", row->slot_space));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "byte_size", row->byte_size));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "byte_alignment", row->byte_alignment));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "store_count", row->store_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "reload_count", row->reload_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_spill_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->spill_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec = report->spill_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_spill_row_t* rows =
          (const loom_target_compile_report_spill_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(loom_target_compile_report_format_spill_row_json(
            &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_allocation_failure_row_json(
    const loom_target_compile_report_allocation_failure_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "value", row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "register_class", row->register_class));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "type_kind", row->type_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "type",
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "element_type", row->element_type));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "element",
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "failure_code", row->failure_code));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "blocking_kind",
      loom_target_compile_report_allocation_failure_blocking_kind_name(
          row->blocking_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "origin_operation",
          row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "origin_block", row->origin_block_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "start_point", row->start_point));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "end_point", row->end_point));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "required_unit_count", row->required_unit_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "budget_units", row->budget_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "peak_live_units", row->peak_live_units));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "location_kind", row->location_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "location_base", row->location_base));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "location_count", row->location_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "conflict_assignment_index",
      row->conflict_assignment_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "conflict_value", row->conflict_value_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "conflict_start_point", row->conflict_start_point));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "conflict_end_point", row->conflict_end_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "conflict_location_kind",
          row->conflict_location_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      stream, &first_field, "conflict_location_base",
      row->conflict_location_base));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "conflict_location_count",
      row->conflict_location_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_allocation_failure_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->allocation_failure_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->allocation_failure_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_allocation_failure_row_t* rows =
          (const loom_target_compile_report_allocation_failure_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_allocation_failure_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_row_json(
    const loom_target_compile_report_allocation_high_water_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "value", row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "register_class", row->register_class));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "type_kind", row->type_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "type",
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "element_type", row->element_type));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "element",
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "assignment_index", row->assignment_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "origin_kind", row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "origin",
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "origin_operation",
          row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "semantic_tag", row->semantic_tag));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "start_point", row->start_point));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "end_point", row->end_point));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "required_unit_count", row->required_unit_count));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "location_kind", row->location_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "location_base", row->location_base));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "location_count", row->location_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "high_water_units", row->high_water_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "lower_free_unit_count",
      row->lower_free_unit_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "lower_free_run_count", row->lower_free_run_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "lower_largest_free_run_unit_count",
      row->lower_largest_free_run_unit_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "lower_pressure_releasable_free_unit_count",
      row->lower_pressure_releasable_free_unit_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "lower_pressure_releasable_free_run_count",
      row->lower_pressure_releasable_free_run_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field,
      "lower_pressure_releasable_largest_free_run_unit_count",
      row->lower_pressure_releasable_largest_free_run_unit_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "active_assignment_blocker_count",
      row->active_assignment_blocker_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "active_assignment_blocker_units",
      row->active_assignment_blocker_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "active_storage_lease_blocker_count",
      row->active_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "active_storage_lease_blocker_units",
      row->active_storage_lease_blocker_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "active_pressure_storage_lease_blocker_count",
      row->active_pressure_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "active_pressure_storage_lease_blocker_units",
      row->active_pressure_storage_lease_blocker_units));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "active_fallback_storage_lease_blocker_count",
      row->active_fallback_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "active_fallback_storage_lease_blocker_units",
      row->active_fallback_storage_lease_blocker_units));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->allocation_high_water_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->allocation_high_water_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_allocation_high_water_row_t* rows =
          (const loom_target_compile_report_allocation_high_water_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_allocation_high_water_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_diagnostics_json(
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write(stream, options->diagnostic_json_objects));
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_target_compile_report_format_source_low_row_json(
    const loom_target_compile_report_source_low_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "source_op", row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "source_op_kind", row->source_op_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "selection",
      loom_target_compile_report_source_low_selection_name(
          row->selection_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      stream, &first_field, "rule_set_index", row->rule_set_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      stream, &first_field, "rule_index", row->rule_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u64_field(
      stream, &first_field, "plan_id", row->plan_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "plan_key", row->plan_key));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u64_field(
      stream, &first_field, "descriptor_id", row->descriptor_id));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "emitted_low_op_count", row->emitted_low_op_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_row_json(
    const loom_target_compile_report_source_low_memory_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "source_op", row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "source_op_kind", row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "memory_space", row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "operation", row->operation_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "packet", row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "address_form", row->address_form));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "dynamic_term_kind", row->dynamic_term_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "fallback_reason", row->fallback_reason));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_i64_field(
      stream, &first_field, "static_offset_bytes", row->static_offset_bytes));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "element_bytes", row->element_byte_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "vector_lanes", row->vector_lane_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "dynamic_stride_bytes", row->dynamic_stride_bytes));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "vector_lane_stride_bytes",
      row->vector_lane_stride_bytes));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "bank_stride_words", row->bank_stride_words));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "bank_conflict_degree", row->bank_conflict_degree));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "bank_conflict_kind", row->bank_conflict_kind));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_source_low_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "selected_op_count",
      report->source_low_selected_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "emitted_op_count",
      report->source_low_emitted_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->source_low_rows.count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "memory_count",
      report->source_low_memory_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_row_t* rows =
          (const loom_target_compile_report_source_low_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "memory_rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_row_t* rows =
          (const loom_target_compile_report_source_low_memory_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_math_row_json(
    const loom_target_compile_report_math_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "source_op", row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "source_op_kind", row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_bundle", row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_config", row->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "policy", row->policy_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "constraint_key", row->constraint_key));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "math_op",
      loom_target_math_op_name((loom_target_math_op_t)row->math_op)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "lane_domain",
      loom_target_math_lane_domain_name(
          (loom_target_math_lane_domain_t)row->lane_domain)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "element_type",
      loom_target_compile_report_scalar_type_name(
          (loom_scalar_type_t)row->element_type)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "action",
      loom_target_compile_report_math_action_name(row->action)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "recipe",
      loom_target_math_recipe_name((loom_target_math_recipe_t)row->recipe)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "source_fastmath_flags",
      row->source_fastmath_flags));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "recipe_fastmath_flags",
      row->recipe_fastmath_flags));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "created_op_count", row->created_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "erased_op_count", row->erased_op_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_math_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "rewritten_op_count",
      report->math_legalization_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "rejected_op_count",
      report->math_legalization_rejected_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "missing_policy_op_count",
      report->math_legalization_missing_policy_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "missing_recipe_op_count",
      report->math_legalization_missing_recipe_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->math_legalization_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->math_legalization_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_math_row_t* rows =
          (const loom_target_compile_report_math_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(loom_target_compile_report_format_math_row_json(
            &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_legalization_row_json(
    const loom_target_compile_report_legalization_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "index", row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "source_op", row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "source_op_kind", row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_bundle", row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_config", row->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "legalizer", row->legalizer_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "legalizer_strategy",
      loom_target_compile_report_legalizer_strategy_name(
          row->legalizer_strategy)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "mode",
      loom_target_compile_report_legalization_mode_name(row->mode)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "policy",
      loom_target_compile_report_legalization_policy_name(row->policy)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "action",
      loom_target_compile_report_legalization_action_name(row->action)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "legalization_outcome",
      loom_target_compile_report_legalization_outcome_name(
          row->legalization_outcome)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "contract_outcome",
      loom_target_compile_report_contract_outcome_name(row->contract_outcome)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      stream, &first_field, "binding_index", row->binding_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      stream, &first_field, "case_index", row->case_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      stream, &first_field, "rule_set_index", row->rule_set_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      stream, &first_field, "rule_index", row->rule_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      stream, &first_field, "diagnostic_index", row->diagnostic_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u64_field(
      stream, &first_field, "descriptor_id", row->descriptor_id));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "source_rejection_bits",
      row->source_rejection_bits));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "source_rejection_detail",
      row->source_rejection_detail));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "target_rejection_bits",
      row->target_rejection_bits));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "missing_feature_bits", row->missing_feature_bits));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "missing_fact_bits", row->missing_fact_bits));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "created_op_count", row->created_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "erased_op_count", row->erased_op_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_compile_report_format_legalization_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "legal_op_count",
      report->target_legalization_legal_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "rewritten_op_count",
      report->target_legalization_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "target_rewritten_op_count",
      report->target_legalization_target_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "reference_rewritten_op_count",
      report->target_legalization_reference_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "deferred_op_count",
      report->target_legalization_deferred_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "invalid_ir_op_count",
      report->target_legalization_invalid_ir_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "unsupported_op_count",
      report->target_legalization_unsupported_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
      stream, &first_field, "unhandled_op_count",
      report->target_legalization_unhandled_op_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
      stream, &first_field, "count", report->target_legalization_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "rows"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->target_legalization_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_legalization_row_t* rows =
          (const loom_target_compile_report_legalization_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        if (row_index != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_legalization_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t loom_target_compile_report_format_text(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_summary(report, options, builder));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_entry_rows(report, builder));
  }
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_move_causes(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_counter_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_action_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_rows(report, builder));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pressure_origin_rows(
        report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_summary_rows(report,
                                                                     builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_spill_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_failure_rows(report,
                                                                  builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_high_water_rows(report,
                                                                     builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_math_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_rows(report,
                                                                 builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_legalization_rows(report, builder));
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_format_json(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream) {
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "artifact_kind",
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_int_field(
      stream, &first_field, "status_code", (int)report->status_code));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_string_field(
      stream, &first_field, "status",
      iree_make_cstring_view(iree_status_code_string(report->status_code))));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u32_field(
      stream, &first_field, "detail_flags", report->detail_flags));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "module", report->module_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "function", report->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "backend", report->backend_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_family", report->target_family_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_key", report->target_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_bundle", report->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_snapshot",
          report->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_export", report->target_export_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_export_symbol",
          report->target_export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "target_config", report->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "lowered", report->lowered_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          stream, &first_field, "executable_format",
          report->executable_format));

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_u64_field(
        stream, &first_field, "artifact_size", report->artifact_size));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
      stream, &first_field, "entries"));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_entries_json(
      report, options->mode, stream));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "schedule"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_json(report, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "static_instruction_mix"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &report->static_instruction_mix, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "allocation"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "move_causes"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_causes_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "wait_plan"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_wait_plan_json(
        &report->wait_plan, stream));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "wait_counter_rows"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_counter_rows_json(
            report, options->mode, stream));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "wait_action_rows"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_action_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "emission"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_emission_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "memory"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_memory_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "target_resources"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_resources_json(
            &report->target_resources, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "pressure_rows"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pressure_rows_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "pressure_origin_rows"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_origin_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "schedule_band_rows"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_rows_json(
            report, options->mode, stream));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "schedule_band_summary_rows"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_summary_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "spill_rows"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_spill_rows_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "allocation_failure_rows"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_failure_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "allocation_high_water_rows"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_high_water_rows_json(
            report, options->mode, stream));
  }
  if (options->diagnostic_count != 0) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_size_field(
        stream, &first_field, "diagnostic_count", options->diagnostic_count));
    if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
      IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
          stream, &first_field, "diagnostics"));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_diagnostics_json(options, stream));
    }
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "math_legalization"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_math_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "source_low"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_source_low_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_json_begin_field(
        stream, &first_field, "target_legalization"));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_legalization_json(
        report, options->mode, stream));
  }
  return loom_output_stream_write_cstring(stream, "}");
}
