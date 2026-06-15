// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/scf/ops.h"

static iree_status_t loom_scf_emit(iree_diagnostic_emitter_t emitter,
                                   const loom_op_t* op,
                                   const loom_error_def_t* error,
                                   const loom_diagnostic_param_t* params,
                                   iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static uint32_t loom_scf_saturating_u32(iree_host_size_t value) {
  if (value > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)value;
}

static iree_status_t loom_scf_emit_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, loom_attr_kind_t actual_kind,
    loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_scf_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_scf_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_scf_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_scf_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, iree_host_size_t actual_count,
    iree_string_view_t expected_field_name, iree_host_size_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32(loom_scf_saturating_u32(actual_count)),
      loom_param_string(expected_field_name),
      loom_param_u32(loom_scf_saturating_u32(expected_count)),
  };
  return loom_scf_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_scf_emit_value_type_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t value_field_name,
    loom_value_id_t value_id, iree_string_view_t expected_field_name,
    loom_value_id_t expected_value_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(value_field_name),
      loom_param_type(loom_module_value_type(module, value_id)),
      loom_param_string(expected_field_name),
      loom_param_type(loom_module_value_type(module, expected_value_id)),
  };
  return loom_scf_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                       IREE_ARRAYSIZE(params));
}

static const loom_block_t* loom_scf_region_entry_block_or_null(
    const loom_region_t* region) {
  if (!region || region->block_count == 0) return NULL;
  return loom_region_const_entry_block(region);
}

static const loom_op_t* loom_scf_region_terminator_or_null(
    const loom_region_t* region) {
  const loom_block_t* entry = loom_scf_region_entry_block_or_null(region);
  return entry ? entry->last_op : NULL;
}

static void loom_scf_format_lookup_field_name(char* buffer,
                                              iree_host_size_t buffer_capacity,
                                              const char* prefix,
                                              iree_host_size_t index) {
  iree_snprintf(buffer, buffer_capacity, "%s[%" PRIhsz "]", prefix, index);
}

static void loom_scf_format_switch_region_field_name(
    char* buffer, iree_host_size_t buffer_capacity, iree_string_view_t prefix,
    iree_host_size_t region_index, iree_host_size_t value_index) {
  iree_snprintf(buffer, buffer_capacity, "%.*s[%" PRIhsz "].yield[%" PRIhsz "]",
                (int)prefix.size, prefix.data, region_index, value_index);
}

static iree_status_t loom_scf_emit_lookup_type_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_host_size_t value_index,
    iree_host_size_t result_index) {
  loom_value_slice_t values = loom_scf_lookup_values(op);
  loom_value_slice_t results = loom_scf_lookup_results(op);
  char value_name[32];
  char result_name[32];
  loom_scf_format_lookup_field_name(value_name, sizeof(value_name), "table",
                                    value_index);
  loom_scf_format_lookup_field_name(result_name, sizeof(result_name), "results",
                                    result_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(value_name)),
      loom_param_type(
          loom_module_value_type(module, values.values[value_index])),
      loom_param_string(iree_make_cstring_view(result_name)),
      loom_param_type(
          loom_module_value_type(module, results.values[result_index])),
  };
  return loom_scf_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                       IREE_ARRAYSIZE(params));
}

iree_status_t loom_scf_lookup_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  if (op->result_count == 0) {
    return loom_scf_emit_count_mismatch(emitter, op, IREE_SV("results"), 0,
                                        IREE_SV("at least one lookup result"),
                                        1);
  }

  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY) {
    return loom_scf_emit_attribute_kind_mismatch(
        emitter, op, IREE_SV("case_keys"), case_keys.kind, LOOM_ATTR_I64_ARRAY);
  }
  if (case_keys.count > 0 && !case_keys.i64_array) {
    return loom_scf_emit_attribute_value_constraint(
        emitter, op, IREE_SV("case_keys"), case_keys.count,
        IREE_SV("non-null i64_array storage when count is non-zero"));
  }

  for (uint16_t i = 1; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] <= case_keys.i64_array[i - 1]) {
      return loom_scf_emit_attribute_value_constraint(
          emitter, op, IREE_SV("case_keys"), case_keys.i64_array[i],
          IREE_SV("strictly increasing sorted unique case key"));
    }
  }

  loom_value_slice_t values = loom_scf_lookup_values(op);
  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  iree_host_size_t expected_value_count =
      row_count * (iree_host_size_t)op->result_count;
  if (values.count != expected_value_count) {
    return loom_scf_emit_count_mismatch(
        emitter, op, IREE_SV("table"), values.count,
        IREE_SV("(case key count + default row) * result count"),
        expected_value_count);
  }

  loom_value_slice_t results = loom_scf_lookup_results(op);
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    iree_host_size_t result_index = i % op->result_count;
    loom_type_t value_type = loom_module_value_type(module, values.values[i]);
    loom_type_t result_type =
        loom_module_value_type(module, results.values[result_index]);
    if (!loom_type_equal(value_type, result_type)) {
      return loom_scf_emit_lookup_type_mismatch(module, emitter, op, i,
                                                result_index);
    }
  }

  return iree_ok_status();
}

iree_status_t loom_scf_for_verify(const loom_module_t* module,
                                  const loom_op_t* op,
                                  iree_diagnostic_emitter_t emitter) {
  (void)module;
  bool has_unroll_factor = loom_scf_for_unroll_factor_is_present(op);
  bool has_unroll_policy = !loom_attr_is_absent(
      loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX]);
  if (!has_unroll_factor || !has_unroll_policy) {
    return iree_ok_status();
  }
  return loom_scf_emit_attribute_value_constraint(
      emitter, op, IREE_SV("unroll"), 2,
      IREE_SV("either bare unroll or unroll factor, not both"));
}

iree_status_t loom_scf_while_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t iter_args = loom_scf_while_iter_args(op);
  const loom_block_t* after_entry =
      loom_scf_region_entry_block_or_null(loom_scf_while_after(op));
  iree_host_size_t after_arg_count = after_entry ? after_entry->arg_count : 0;
  if (after_arg_count != iter_args.count) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    loom_type_t iter_arg_type =
        loom_module_value_type(module, iter_args.values[i]);
    loom_type_t after_arg_type =
        loom_module_value_type(module, loom_block_arg_id(after_entry, i));
    if (!loom_type_equal(iter_arg_type, after_arg_type)) {
      return iree_ok_status();
    }
  }

  const loom_op_t* condition =
      loom_scf_region_terminator_or_null(loom_scf_while_before(op));
  if (!condition || !loom_scf_condition_isa(condition)) {
    return iree_ok_status();
  }

  loom_value_slice_t forwarded = loom_scf_condition_forwarded(condition);
  if (forwarded.count != after_arg_count) {
    return loom_scf_emit_count_mismatch(
        emitter, condition, IREE_SV("forwarded"), forwarded.count,
        IREE_SV("after block args"), after_arg_count);
  }

  for (uint16_t i = 0; i < forwarded.count; ++i) {
    loom_value_id_t forwarded_id = forwarded.values[i];
    loom_value_id_t after_arg_id = loom_block_arg_id(after_entry, i);
    if (loom_type_equal(loom_module_value_type(module, forwarded_id),
                        loom_module_value_type(module, after_arg_id))) {
      continue;
    }
    char forwarded_name[32];
    char after_arg_name[32];
    loom_scf_format_lookup_field_name(forwarded_name, sizeof(forwarded_name),
                                      "forwarded", i);
    loom_scf_format_lookup_field_name(after_arg_name, sizeof(after_arg_name),
                                      "after", i);
    return loom_scf_emit_value_type_mismatch(
        module, emitter, condition, iree_make_cstring_view(forwarded_name),
        forwarded_id, iree_make_cstring_view(after_arg_name), after_arg_id);
  }

  return iree_ok_status();
}

iree_status_t loom_scf_if_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  if (op->result_count == 0 || loom_scf_if_else_region(op)) {
    return iree_ok_status();
  }
  return loom_scf_emit_count_mismatch(
      emitter, op, IREE_SV("else_region"), 0,
      IREE_SV("present when scf.if has results"), 1);
}

static iree_status_t loom_scf_emit_switch_type_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t yield_field_name,
    loom_value_id_t yield_value, iree_host_size_t result_index) {
  loom_value_slice_t results = loom_scf_switch_results(op);
  char result_name[32];
  loom_scf_format_lookup_field_name(result_name, sizeof(result_name), "results",
                                    result_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(yield_field_name),
      loom_param_type(loom_module_value_type(module, yield_value)),
      loom_param_string(iree_make_cstring_view(result_name)),
      loom_param_type(
          loom_module_value_type(module, results.values[result_index])),
  };
  return loom_scf_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_scf_switch_verify_region_yield(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_region_t* region,
    iree_string_view_t field_name, iree_host_size_t region_ordinal) {
  if (!region || region->block_count == 0) return iree_ok_status();
  const loom_block_t* entry = loom_region_const_entry_block(region);
  const loom_op_t* terminator = entry->last_op;
  if (!terminator || !loom_scf_yield_isa(terminator)) {
    return iree_ok_status();
  }

  loom_value_slice_t yielded_values = loom_scf_yield_values(terminator);
  if (yielded_values.count != op->result_count) {
    return loom_scf_emit_count_mismatch(emitter, op, field_name,
                                        yielded_values.count,
                                        IREE_SV("results"), op->result_count);
  }

  loom_value_slice_t results = loom_scf_switch_results(op);
  loom_type_value_remap_t yield_remap = {
      .source_values = results.values,
      .target_values = yielded_values.values,
      .count = yielded_values.count,
  };
  for (uint16_t i = 0; i < yielded_values.count; ++i) {
    loom_type_t yield_type =
        loom_module_value_type(module, yielded_values.values[i]);
    loom_type_t result_type = loom_module_value_type(module, results.values[i]);
    if (loom_type_equal_after_value_remap(result_type, yield_type,
                                          &yield_remap)) {
      continue;
    }
    char yield_name[48];
    loom_scf_format_switch_region_field_name(yield_name, sizeof(yield_name),
                                             field_name, region_ordinal, i);
    return loom_scf_emit_switch_type_mismatch(
        module, emitter, op, iree_make_cstring_view(yield_name),
        yielded_values.values[i], i);
  }
  return iree_ok_status();
}

iree_status_t loom_scf_switch_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_attribute_t case_keys = loom_scf_switch_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY) {
    return loom_scf_emit_attribute_kind_mismatch(
        emitter, op, IREE_SV("case_keys"), case_keys.kind, LOOM_ATTR_I64_ARRAY);
  }
  if (case_keys.count > 0 && !case_keys.i64_array) {
    return loom_scf_emit_attribute_value_constraint(
        emitter, op, IREE_SV("case_keys"), case_keys.count,
        IREE_SV("non-null i64_array storage when count is non-zero"));
  }

  for (uint16_t i = 1; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] <= case_keys.i64_array[i - 1]) {
      return loom_scf_emit_attribute_value_constraint(
          emitter, op, IREE_SV("case_keys"), case_keys.i64_array[i],
          IREE_SV("strictly increasing sorted unique case key"));
    }
  }

  loom_region_slice_t case_regions = loom_scf_switch_case_regions(op);
  if (case_regions.count != case_keys.count) {
    return loom_scf_emit_count_mismatch(emitter, op, IREE_SV("case_regions"),
                                        case_regions.count,
                                        IREE_SV("case_keys"), case_keys.count);
  }

  IREE_RETURN_IF_ERROR(loom_scf_switch_verify_region_yield(
      module, op, emitter, loom_scf_switch_default_region(op),
      IREE_SV("default_region"), 0));
  for (uint8_t i = 0; i < case_regions.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_switch_verify_region_yield(
        module, op, emitter, case_regions.regions[i], IREE_SV("case_regions"),
        i));
  }
  return iree_ok_status();
}
