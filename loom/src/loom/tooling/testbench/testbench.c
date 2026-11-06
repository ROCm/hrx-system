// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/testbench.h"

#include <math.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/command/composition.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/tooling/testbench/scenario/plan.h"

enum {
  LOOM_TESTBENCH_INTERNAL_INDEX_INVALID = UINT32_MAX,
};

typedef struct loom_testbench_plan_counts_t {
  // Number of check.case records in the module.
  iree_host_size_t case_count;
  // Number of check.benchmark records in the module.
  iree_host_size_t benchmark_count;
  // Number of parameter ops in all case bodies.
  iree_host_size_t parameter_count;
  // Number of value source ops in all case bodies.
  iree_host_size_t value_source_count;
  // Number of file output ops in all case bodies.
  iree_host_size_t file_write_count;
  // Number of invocation ops in all case bodies.
  iree_host_size_t invocation_count;
  // Number of kernel launch ops in all case bodies.
  iree_host_size_t kernel_launch_count;
  // Number of launch/schedule events in all case bodies.
  iree_host_size_t launch_schedule_event_count;
  // Number of expectation ops in all case bodies.
  iree_host_size_t expectation_count;
  // Maximum number of structured issues this planning pass can emit.
  iree_host_size_t issue_capacity;
} loom_testbench_plan_counts_t;

typedef struct loom_testbench_record_indices_t {
  // Index of the check.case defining this symbol, or INTERNAL_INDEX_INVALID.
  uint32_t case_index;
  // Index of the check.scenario defining this symbol, or
  // INTERNAL_INDEX_INVALID.
  uint32_t scenario_index;
} loom_testbench_record_indices_t;

void loom_testbench_plan_options_initialize(
    loom_testbench_plan_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->max_samples_per_case =
      LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE;
}

static iree_string_view_t loom_testbench_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (!symbol || symbol->name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, symbol->name_id);
}

static const loom_symbol_t* loom_testbench_symbol_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  return &module->symbols.entries[ref.symbol_id];
}

static iree_string_view_t loom_testbench_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, string_id);
}

static loom_scalar_type_t loom_testbench_value_scalar_type(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return LOOM_SCALAR_TYPE_NONE;
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return LOOM_SCALAR_TYPE_NONE;
  }
  return loom_type_element_type(type);
}

static loom_type_t loom_testbench_value_type(const loom_module_t* module,
                                             loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return (loom_type_t){0};
  }
  return loom_module_value_type(module, value_id);
}

static bool loom_testbench_scalar_type_is_integral_sample(
    loom_scalar_type_t scalar_type) {
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(scalar_type);
}

static bool loom_testbench_attr_as_i64_checked(loom_attribute_t attr,
                                               int64_t* out_value) {
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_testbench_attr_as_f64_checked(loom_attribute_t attr,
                                               double* out_value) {
  if (attr.kind == LOOM_ATTR_F64) {
    *out_value = loom_attr_as_f64(attr);
    return true;
  }
  if (attr.kind == LOOM_ATTR_I64) {
    *out_value = (double)loom_attr_as_i64(attr);
    return true;
  }
  return false;
}

static bool loom_testbench_add_i64_nonnegative_offset(int64_t base,
                                                      uint64_t offset,
                                                      int64_t* out_value) {
  if (offset > (uint64_t)INT64_MAX) {
    return false;
  }
  int64_t signed_offset = (int64_t)offset;
  if (base > INT64_MAX - signed_offset) {
    return false;
  }
  *out_value = base + signed_offset;
  return true;
}

static bool loom_testbench_next_power_of_two(int64_t value,
                                             int64_t* out_power_of_two) {
  if (value <= 0) {
    return false;
  }
  uint64_t unsigned_value = (uint64_t)value;
  uint64_t next_value = 1;
  while (next_value < unsigned_value) {
    if (next_value > (uint64_t)INT64_MAX / 2) {
      return false;
    }
    next_value *= 2;
  }
  *out_power_of_two = (int64_t)next_value;
  return true;
}

static bool loom_testbench_linear_i64_sample_count(
    int64_t lower, int64_t upper, int64_t step,
    iree_host_size_t* out_sample_count) {
  if (step <= 0 || lower > upper) {
    return false;
  }
  uint64_t distance = (uint64_t)upper - (uint64_t)lower;
  if (distance > (uint64_t)INT64_MAX) {
    return false;
  }
  uint64_t step_value = (uint64_t)step;
  uint64_t sample_count = distance / step_value + 1;
  if (sample_count > IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_sample_count = (iree_host_size_t)sample_count;
  return sample_count > 0;
}

static bool loom_testbench_linear_f64_sample_count(
    double lower, double upper, double step,
    iree_host_size_t* out_sample_count) {
  if (!(step > 0.0) || !(lower <= upper)) {
    return false;
  }
  double sample_count = floor((upper - lower) / step) + 1.0;
  if (!(sample_count >= 1.0) || sample_count > (double)IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_sample_count = (iree_host_size_t)sample_count;
  return true;
}

static bool loom_testbench_po2_sample_count(
    int64_t lower, int64_t upper, iree_host_size_t* out_sample_count) {
  int64_t value = 0;
  if (!loom_testbench_next_power_of_two(lower, &value) || upper < value) {
    return false;
  }
  iree_host_size_t sample_count = 0;
  while (value <= upper) {
    ++sample_count;
    if (value > INT64_MAX / 2) {
      break;
    }
    value *= 2;
  }
  *out_sample_count = sample_count;
  return sample_count > 0;
}

static bool loom_testbench_plan_range_parameter(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_parameter_plan_t* out_parameter) {
  loom_value_id_t value_id = loom_check_param_range_result(op);
  loom_scalar_type_t scalar_type =
      loom_testbench_value_scalar_type(module, value_id);
  if (scalar_type == LOOM_SCALAR_TYPE_NONE) {
    return false;
  }

  loom_attribute_t lower = loom_check_param_range_lower(op);
  loom_attribute_t upper = loom_check_param_range_upper(op);
  loom_attribute_t step = loom_check_param_range_step(op);
  loom_check_param_range_policy_t policy =
      (loom_check_param_range_policy_t)loom_check_param_range_policy(op);

  iree_host_size_t sample_count = 0;
  if (policy == LOOM_CHECK_PARAM_RANGE_POLICY_PO2) {
    int64_t lower_value = 0;
    int64_t upper_value = 0;
    if (!loom_testbench_scalar_type_is_integral_sample(scalar_type) ||
        !loom_testbench_attr_as_i64_checked(lower, &lower_value) ||
        !loom_testbench_attr_as_i64_checked(upper, &upper_value) ||
        !loom_testbench_po2_sample_count(lower_value, upper_value,
                                         &sample_count)) {
      return false;
    }
  } else if (policy == LOOM_CHECK_PARAM_RANGE_POLICY_LINEAR) {
    if (loom_testbench_scalar_type_is_integral_sample(scalar_type)) {
      int64_t lower_value = 0;
      int64_t upper_value = 0;
      int64_t step_value = 1;
      if (!loom_attr_is_absent(step) &&
          !loom_testbench_attr_as_i64_checked(step, &step_value)) {
        return false;
      }
      if (!loom_testbench_attr_as_i64_checked(lower, &lower_value) ||
          !loom_testbench_attr_as_i64_checked(upper, &upper_value) ||
          !loom_testbench_linear_i64_sample_count(lower_value, upper_value,
                                                  step_value, &sample_count)) {
        return false;
      }
    } else if (loom_scalar_type_is_float(scalar_type)) {
      double lower_value = 0.0;
      double upper_value = 0.0;
      double step_value = 0.0;
      if (loom_attr_is_absent(step) ||
          !loom_testbench_attr_as_f64_checked(lower, &lower_value) ||
          !loom_testbench_attr_as_f64_checked(upper, &upper_value) ||
          !loom_testbench_attr_as_f64_checked(step, &step_value) ||
          !loom_testbench_linear_f64_sample_count(lower_value, upper_value,
                                                  step_value, &sample_count)) {
        return false;
      }
    } else {
      return false;
    }
  } else {
    return false;
  }

  out_parameter->kind = LOOM_TESTBENCH_PARAMETER_RANGE;
  out_parameter->op = op;
  out_parameter->value_id = value_id;
  out_parameter->type = loom_module_value_type(module, value_id);
  out_parameter->sample_count = sample_count;
  out_parameter->range.policy = policy;
  out_parameter->range.scalar_type = scalar_type;
  out_parameter->range.lower = lower;
  out_parameter->range.upper = upper;
  out_parameter->range.step = step;
  return true;
}

static bool loom_testbench_plan_choice_parameter(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_parameter_plan_t* out_parameter) {
  loom_attribute_t values = loom_check_param_choice_values(op);
  if (values.kind != LOOM_ATTR_I64_ARRAY || values.count == 0) {
    return false;
  }

  out_parameter->kind = LOOM_TESTBENCH_PARAMETER_CHOICE;
  out_parameter->op = op;
  out_parameter->value_id = loom_check_param_choice_result(op);
  out_parameter->type = loom_module_value_type(module, out_parameter->value_id);
  out_parameter->sample_count = values.count;
  out_parameter->choice.values = values.i64_array;
  out_parameter->choice.count = values.count;
  return true;
}

static bool loom_testbench_plan_seed_parameter(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_parameter_plan_t* out_parameter) {
  int64_t count = loom_check_param_seed_count(op);
  if (count <= 0 || (uint64_t)count > IREE_HOST_SIZE_MAX) {
    return false;
  }
  int64_t last_seed = 0;
  if (!loom_testbench_add_i64_nonnegative_offset(
          loom_check_param_seed_base(op), (uint64_t)count - 1, &last_seed)) {
    return false;
  }

  out_parameter->kind = LOOM_TESTBENCH_PARAMETER_SEED;
  out_parameter->op = op;
  out_parameter->value_id = loom_check_param_seed_result(op);
  out_parameter->type = loom_module_value_type(module, out_parameter->value_id);
  out_parameter->sample_count = (iree_host_size_t)count;
  out_parameter->seed.base = loom_check_param_seed_base(op);
  out_parameter->seed.count = (iree_host_size_t)count;
  return true;
}

static bool loom_testbench_plan_parameter_name(
    const loom_module_t* module, loom_attribute_t name_attr,
    loom_testbench_parameter_plan_t* parameter) {
  parameter->name_id = LOOM_STRING_ID_INVALID;
  parameter->name = iree_string_view_empty();
  if (loom_attr_is_absent(name_attr)) {
    return true;
  }
  if (name_attr.kind != LOOM_ATTR_STRING ||
      name_attr.string_id >= module->strings.count) {
    return false;
  }

  iree_string_view_t name =
      loom_string_table_get(&module->strings, name_attr.string_id);
  if (iree_string_view_is_empty(name)) {
    return false;
  }
  parameter->name_id = name_attr.string_id;
  parameter->name = name;
  return true;
}

static bool loom_testbench_plan_parameter(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_parameter_plan_t* out_parameter) {
  memset(out_parameter, 0, sizeof(*out_parameter));
  if (loom_check_param_range_isa(op)) {
    return loom_testbench_plan_range_parameter(module, op, out_parameter) &&
           loom_testbench_plan_parameter_name(
               module, loom_check_param_range_param_name_attr(op),
               out_parameter);
  }
  if (loom_check_param_choice_isa(op)) {
    return loom_testbench_plan_choice_parameter(module, op, out_parameter) &&
           loom_testbench_plan_parameter_name(
               module, loom_check_param_choice_param_name_attr(op),
               out_parameter);
  }
  if (loom_check_param_seed_isa(op)) {
    return loom_testbench_plan_seed_parameter(module, op, out_parameter) &&
           loom_testbench_plan_parameter_name(
               module, loom_check_param_seed_param_name_attr(op),
               out_parameter);
  }
  return false;
}

static bool loom_testbench_plan_file_write(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_file_write_plan_t* out_file_write) {
  memset(out_file_write, 0, sizeof(*out_file_write));
  if (!loom_check_file_write_npy_isa(op)) {
    return false;
  }
  out_file_write->op = op;
  out_file_write->value_id = loom_check_file_write_npy_value(op);
  out_file_write->type =
      loom_testbench_value_type(module, out_file_write->value_id);
  out_file_write->path_id = loom_check_file_write_npy_path(op);
  out_file_write->path =
      loom_testbench_string_from_id(module, out_file_write->path_id);
  out_file_write->mode = loom_check_file_write_npy_has_mode(op)
                             ? loom_check_file_write_npy_mode(op)
                             : LOOM_CHECK_FILE_WRITE_NPY_MODE_ON_FAILURE;
  return out_file_write->value_id < module->values.count &&
         out_file_write->path_id < module->strings.count &&
         out_file_write->mode > 0 &&
         out_file_write->mode < LOOM_CHECK_FILE_WRITE_NPY_MODE_COUNT_;
}

static bool loom_testbench_value_ids_are_in_range(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    iree_host_size_t value_count) {
  for (iree_host_size_t value_index = 0; value_index < value_count;
       ++value_index) {
    if (value_ids[value_index] >= module->values.count) {
      return false;
    }
  }
  return true;
}

static bool loom_testbench_is_function_call_op(const loom_module_t* module,
                                               const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  return vtable && vtable->call_like &&
         vtable->call_like->kind == LOOM_CALL_LIKE_KIND_SEMANTIC;
}

static bool loom_testbench_plan_kernel_launch_invocation(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_invocation_plan_t* out_invocation) {
  if (!loom_kernel_launch_isa(op)) {
    return false;
  }

  const loom_value_slice_t workloads = loom_kernel_launch_workloads(op);
  const loom_value_slice_t arguments = loom_kernel_launch_arguments(op);
  memset(out_invocation, 0, sizeof(*out_invocation));
  out_invocation->kind = LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH;
  out_invocation->module = module;
  out_invocation->op = op;
  out_invocation->callee_ref = loom_kernel_launch_callee(op);
  out_invocation->provider_id = LOOM_STRING_ID_INVALID;
  out_invocation->provider = iree_string_view_empty();
  out_invocation->attrs = loom_named_attr_slice_empty();
  out_invocation->execution_epoch = LOOM_TESTBENCH_EXECUTION_EPOCH_INVALID;
  out_invocation->workload_value_ids = workloads.values;
  out_invocation->workload_count = workloads.count;
  out_invocation->input_value_ids = arguments.values;
  out_invocation->input_count = arguments.count;
  return loom_symbol_ref_is_valid(out_invocation->callee_ref) &&
         loom_testbench_value_ids_are_in_range(
             module, out_invocation->workload_value_ids,
             out_invocation->workload_count) &&
         loom_testbench_value_ids_are_in_range(module,
                                               out_invocation->input_value_ids,
                                               out_invocation->input_count);
}

static bool loom_testbench_plan_semantic_call_invocation(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_invocation_plan_t* out_invocation) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->call_like ||
      vtable->call_like->kind != LOOM_CALL_LIKE_KIND_SEMANTIC) {
    return false;
  }

  loom_call_like_t call = loom_call_like_cast(module, (loom_op_t*)op);
  const loom_call_like_vtable_t* call_like = call.vtable;
  loom_value_slice_t inputs = loom_call_like_operands(call);
  loom_value_slice_t results = loom_call_like_results(call);
  if (call_like->callee_attr_index >= op->attribute_count ||
      inputs.count > op->operand_count || results.count > op->result_count) {
    return false;
  }

  memset(out_invocation, 0, sizeof(*out_invocation));
  out_invocation->kind = LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL;
  out_invocation->module = module;
  out_invocation->op = op;
  out_invocation->callee_ref = loom_attr_as_symbol(
      loom_op_const_attrs(op)[call_like->callee_attr_index]);
  out_invocation->provider_id = LOOM_STRING_ID_INVALID;
  out_invocation->provider = iree_string_view_empty();
  out_invocation->attrs = loom_named_attr_slice_empty();
  out_invocation->execution_epoch = LOOM_TESTBENCH_EXECUTION_EPOCH_INVALID;
  out_invocation->input_value_ids = inputs.values;
  out_invocation->input_count = inputs.count;
  out_invocation->result_value_ids = results.values;
  out_invocation->result_count = results.count;
  return loom_symbol_ref_is_valid(out_invocation->callee_ref) &&
         loom_testbench_value_ids_are_in_range(module,
                                               out_invocation->input_value_ids,
                                               out_invocation->input_count) &&
         loom_testbench_value_ids_are_in_range(module,
                                               out_invocation->result_value_ids,
                                               out_invocation->result_count);
}

static bool loom_testbench_plan_oracle_invocation(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_invocation_plan_t* out_invocation) {
  if (!loom_check_oracle_call_isa(op)) {
    return false;
  }

  loom_value_slice_t inputs = loom_check_oracle_call_inputs(op);
  loom_value_slice_t results = loom_check_oracle_call_results(op);
  memset(out_invocation, 0, sizeof(*out_invocation));
  out_invocation->kind = LOOM_TESTBENCH_INVOCATION_ORACLE;
  out_invocation->module = module;
  out_invocation->op = op;
  out_invocation->callee_ref = loom_check_oracle_call_callee(op);
  out_invocation->provider_id = loom_check_oracle_call_provider(op);
  out_invocation->provider =
      loom_testbench_string_from_id(module, out_invocation->provider_id);
  out_invocation->attrs = loom_check_oracle_call_attrs(op);
  out_invocation->execution_epoch = LOOM_TESTBENCH_EXECUTION_EPOCH_INVALID;
  out_invocation->input_value_ids = inputs.values;
  out_invocation->input_count = inputs.count;
  out_invocation->result_value_ids = results.values;
  out_invocation->result_count = results.count;
  return loom_symbol_ref_is_valid(out_invocation->callee_ref) &&
         out_invocation->provider_id < module->strings.count &&
         !iree_string_view_is_empty(out_invocation->provider) &&
         loom_testbench_value_ids_are_in_range(module,
                                               out_invocation->input_value_ids,
                                               out_invocation->input_count) &&
         loom_testbench_value_ids_are_in_range(module,
                                               out_invocation->result_value_ids,
                                               out_invocation->result_count);
}

static bool loom_testbench_is_invocation_op(const loom_module_t* module,
                                            const loom_op_t* op) {
  return loom_check_oracle_call_isa(op) || loom_kernel_launch_isa(op) ||
         loom_testbench_is_function_call_op(module, op);
}

static bool loom_testbench_plan_invocation(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_invocation_plan_t* out_invocation) {
  if (loom_check_oracle_call_isa(op)) {
    return loom_testbench_plan_oracle_invocation(module, op, out_invocation);
  }
  if (loom_kernel_launch_isa(op)) {
    return loom_testbench_plan_kernel_launch_invocation(module, op,
                                                        out_invocation);
  }
  return loom_testbench_plan_semantic_call_invocation(module, op,
                                                      out_invocation);
}

static bool loom_testbench_is_parameter_op(const loom_op_t* op) {
  return loom_check_param_range_isa(op) || loom_check_param_choice_isa(op) ||
         loom_check_param_seed_isa(op);
}

static bool loom_testbench_is_launch_schedule_op(const loom_op_t* op) {
  return loom_command_composition_is_kernel_launch_schedule(op);
}

static loom_region_t* loom_testbench_launch_schedule_body(const loom_op_t* op) {
  return loom_op_regions(op)[0];
}

// Iterates a launch schedule tree in source order without recursion or
// transient allocations. Verified launch schedule regions contain one block
// and only launch, nested schedule, and yield operations.
typedef struct loom_testbench_launch_schedule_iterator_t {
  // Top-level schedule operation whose body bounds the walk.
  const loom_op_t* root_op;
  // Next operation to return, or NULL once the walk is complete.
  loom_op_t* next_op;
  // Schedule nesting depth of |next_op|, beginning at one.
  iree_host_size_t next_depth;
} loom_testbench_launch_schedule_iterator_t;

static void loom_testbench_launch_schedule_iterator_initialize(
    const loom_op_t* root_op,
    loom_testbench_launch_schedule_iterator_t* out_iterator) {
  loom_region_t* body = loom_testbench_launch_schedule_body(root_op);
  loom_block_t* block =
      body && body->block_count != 0 ? loom_region_entry_block(body) : NULL;
  *out_iterator = (loom_testbench_launch_schedule_iterator_t){
      .root_op = root_op,
      .next_op = block ? block->first_op : NULL,
      .next_depth = 1,
  };
}

static bool loom_testbench_launch_schedule_iterator_next(
    loom_testbench_launch_schedule_iterator_t* iterator, loom_op_t** out_op,
    iree_host_size_t* out_depth) {
  loom_op_t* op = iterator->next_op;
  if (op == NULL) {
    return false;
  }

  *out_op = op;
  *out_depth = iterator->next_depth;
  if (loom_testbench_is_launch_schedule_op(op)) {
    loom_region_t* body = loom_testbench_launch_schedule_body(op);
    loom_block_t* block =
        body && body->block_count != 0 ? loom_region_entry_block(body) : NULL;
    if (block && block->first_op) {
      iterator->next_op = block->first_op;
      ++iterator->next_depth;
      return true;
    }
  }

  while (op->next_op == NULL) {
    op = op->parent_op;
    if (op == iterator->root_op) {
      iterator->next_op = NULL;
      return true;
    }
    --iterator->next_depth;
  }
  iterator->next_op = op->next_op;
  return true;
}

static bool loom_testbench_is_supported_check_body_op(
    const loom_module_t* module, const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_CHECK_RETURN:
    case LOOM_OP_CHECK_REQUIRES:
    case LOOM_OP_CHECK_SKIP_IF:
    case LOOM_OP_CHECK_PARAM_RANGE:
    case LOOM_OP_CHECK_PARAM_CHOICE:
    case LOOM_OP_CHECK_PARAM_SEED:
    case LOOM_OP_CHECK_LITERAL:
    case LOOM_OP_CHECK_GENERATE_IOTA:
    case LOOM_OP_CHECK_GENERATE_FILL:
    case LOOM_OP_CHECK_GENERATE_RANDOM_UNIFORM:
    case LOOM_OP_CHECK_FILE_READ_NPY:
    case LOOM_OP_CHECK_FILE_WRITE_NPY:
    case LOOM_OP_CHECK_TENSOR_VIEW:
    case LOOM_OP_CHECK_ORACLE_CALL:
      return true;
    default:
      break;
  }
  return loom_testbench_is_launch_schedule_op(op) ||
         loom_testbench_is_expectation_op(op) ||
         loom_testbench_is_invocation_op(module, op);
}

static void loom_testbench_count_launch_schedule(
    const loom_op_t* schedule_op, loom_testbench_plan_counts_t* counts) {
  counts->launch_schedule_event_count += 2;
  loom_testbench_launch_schedule_iterator_t iterator;
  loom_testbench_launch_schedule_iterator_initialize(schedule_op, &iterator);
  loom_op_t* op = NULL;
  iree_host_size_t depth = 0;
  while (loom_testbench_launch_schedule_iterator_next(&iterator, &op, &depth)) {
    (void)depth;
    ++counts->issue_capacity;
    if (loom_kernel_launch_isa(op)) {
      ++counts->invocation_count;
      ++counts->kernel_launch_count;
      ++counts->launch_schedule_event_count;
    } else if (loom_testbench_is_launch_schedule_op(op)) {
      counts->launch_schedule_event_count += 2;
    }
  }
}

static void loom_testbench_count_case_body(
    const loom_module_t* module, const loom_op_t* case_op,
    loom_testbench_plan_counts_t* counts) {
  loom_region_t* body = loom_check_case_body(case_op);
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      ++counts->issue_capacity;
      if (loom_testbench_is_launch_schedule_op(op)) {
        loom_testbench_count_launch_schedule(op, counts);
        continue;
      }
      if (loom_testbench_is_parameter_op(op)) {
        ++counts->parameter_count;
      }
      if (loom_testbench_is_value_source_op(op)) {
        ++counts->value_source_count;
      }
      if (loom_check_file_write_npy_isa(op)) {
        ++counts->file_write_count;
      }
      if (loom_testbench_is_invocation_op(module, op)) {
        ++counts->invocation_count;
        if (loom_kernel_launch_isa(op)) {
          ++counts->kernel_launch_count;
          ++counts->launch_schedule_event_count;
        }
      }
      if (loom_testbench_is_expectation_op(op)) {
        ++counts->expectation_count;
      }
    }
  }
}

static void loom_testbench_count_module(const loom_module_t* module,
                                        loom_testbench_plan_counts_t* counts) {
  memset(counts, 0, sizeof(*counts));
  if (!module->body || module->body->block_count == 0) {
    return;
  }
  const loom_block_t* block = loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_check_case_isa(op)) {
      ++counts->case_count;
      loom_testbench_count_case_body(module, op, counts);
    } else if (loom_check_benchmark_isa(op)) {
      ++counts->benchmark_count;
      ++counts->issue_capacity;
      counts->issue_capacity += loom_check_benchmark_attrs(op).count;
    }
  }
}

static iree_status_t loom_testbench_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, element_size, out_ptr));
  memset(*out_ptr, 0, count * element_size);
  return iree_ok_status();
}

static void loom_testbench_append_issue(
    loom_testbench_issue_t* issues, iree_host_size_t issue_capacity,
    iree_host_size_t* inout_issue_count, loom_testbench_issue_kind_t kind,
    iree_host_size_t case_index, iree_host_size_t scenario_index,
    iree_host_size_t benchmark_index, const loom_op_t* op,
    loom_symbol_ref_t record_ref) {
  IREE_ASSERT(*inout_issue_count < issue_capacity);
  loom_testbench_issue_t* issue = &issues[(*inout_issue_count)++];
  issue->kind = kind;
  issue->case_index = case_index;
  issue->scenario_index = scenario_index;
  issue->benchmark_index = benchmark_index;
  issue->op = op;
  issue->record_ref = record_ref;
}

static void loom_testbench_multiply_sample_count(
    iree_host_size_t factor, iree_host_size_t cap,
    iree_host_size_t* inout_cartesian_count,
    iree_host_size_t* inout_sample_count, bool* inout_truncated) {
  iree_host_size_t cartesian_count = *inout_cartesian_count;
  if (cartesian_count != IREE_HOST_SIZE_MAX) {
    if (factor != 0 && cartesian_count > IREE_HOST_SIZE_MAX / factor) {
      cartesian_count = IREE_HOST_SIZE_MAX;
    } else {
      cartesian_count *= factor;
    }
  }

  iree_host_size_t sample_count = *inout_sample_count;
  if (sample_count != 0) {
    if (factor != 0 && sample_count > IREE_HOST_SIZE_MAX / factor) {
      sample_count = IREE_HOST_SIZE_MAX;
    } else {
      sample_count *= factor;
    }
  }
  if (sample_count > cap) {
    sample_count = cap;
    *inout_truncated = true;
  }
  if (cartesian_count > sample_count) {
    *inout_truncated = true;
  }

  *inout_cartesian_count = cartesian_count;
  *inout_sample_count = sample_count;
}

static bool loom_testbench_parameter_names_have_duplicate(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t parameter_index) {
  const loom_testbench_parameter_plan_t* parameter =
      &case_plan->parameters[parameter_index];
  if (iree_string_view_is_empty(parameter->name)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < parameter_index; ++i) {
    if (iree_string_view_equal(case_plan->parameters[i].name,
                               parameter->name)) {
      return true;
    }
  }
  return false;
}

typedef struct loom_testbench_launch_schedule_frame_t {
  // Next operation awaiting traversal in this schedule region.
  loom_op_t* next_op;
  // First execution epoch available to children in this region.
  iree_host_size_t base_epoch;
  // Number of execution epochs occupied by children visited so far.
  iree_host_size_t span;
  // Number of enclosing launch schedule operations.
  iree_host_size_t depth;
  // Location of the schedule operation owning this frame.
  loom_location_id_t location;
  // True when direct children share |base_epoch|.
  bool is_concurrent;
} loom_testbench_launch_schedule_frame_t;

typedef struct loom_testbench_invocation_planner_t {
  // Module whose invocation operations are being planned.
  const loom_module_t* module;
  // Module-local ordinal of |case_plan|.
  iree_host_size_t case_index;
  // Case receiving planned invocation metadata.
  loom_testbench_case_plan_t* case_plan;
  // Module-wide invocation storage.
  loom_testbench_invocation_plan_t* invocations;
  // Number of initialized entries in |invocations|.
  iree_host_size_t* invocation_count;
  // Module-wide ordered kernel launch references.
  const loom_testbench_invocation_plan_t** kernel_launches;
  // Number of initialized entries in |kernel_launches|.
  iree_host_size_t* kernel_launch_count;
  // Module-wide balanced launch/schedule events.
  loom_testbench_launch_schedule_event_t* launch_schedule_events;
  // Number of initialized entries in |launch_schedule_events|.
  iree_host_size_t* launch_schedule_event_count;
  // Dense storage identities indexed by module value ID.
  loom_testbench_storage_identity_t* value_storage_identities;
  // Module-wide planning issue storage.
  loom_testbench_issue_t* issues;
  // Number of entries available in |issues|.
  iree_host_size_t issue_capacity;
  // Number of initialized entries in |issues|.
  iree_host_size_t* issue_count;
  // Arena owning the reusable launch-schedule traversal stack.
  iree_arena_allocator_t* arena;
  // Active structured launch-schedule traversal frames.
  loom_testbench_launch_schedule_frame_t* schedule_frames;
  // Number of active entries in |schedule_frames|.
  iree_host_size_t schedule_frame_count;
  // Number of allocated entries in |schedule_frames|.
  iree_host_size_t schedule_frame_capacity;
  // Epoch assigned to the next ordered kernel launch.
  iree_host_size_t next_execution_epoch;
} loom_testbench_invocation_planner_t;

static bool loom_testbench_type_has_storage(loom_type_t type) {
  return loom_type_is_buffer(type) || loom_type_is_tensor(type) ||
         loom_type_is_view(type);
}

static void loom_testbench_set_storage_root(
    loom_value_id_t value_id,
    loom_testbench_storage_identity_t* value_storage_identities) {
  value_storage_identities[value_id] = (loom_testbench_storage_identity_t){
      .root_value_id = value_id,
      .byte_offset = 0,
  };
}

static iree_status_t loom_testbench_plan_value_source_storage(
    const loom_testbench_value_source_plan_t* source,
    loom_testbench_storage_identity_t* value_storage_identities) {
  if (!loom_testbench_type_has_storage(source->type)) {
    return iree_ok_status();
  }
  if (source->kind != LOOM_TESTBENCH_VALUE_SOURCE_TENSOR_VIEW) {
    loom_testbench_set_storage_root(source->value_id, value_storage_identities);
    return iree_ok_status();
  }

  const loom_testbench_storage_identity_t parent =
      value_storage_identities[source->tensor_view.source_value_id];
  if (parent.root_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_testbench_storage_identity_t* identity =
      &value_storage_identities[source->value_id];
  identity->root_value_id = parent.root_value_id;
  if (!iree_device_size_checked_add(parent.byte_offset,
                                    source->tensor_view.byte_offset,
                                    &identity->byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "testbench tensor view byte offset overflow");
  }
  return iree_ok_status();
}

static void loom_testbench_plan_invocation_result_storage(
    const loom_module_t* module,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_storage_identity_t* value_storage_identities) {
  const uint16_t result_offset =
      invocation->kind == LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL
          ? loom_call_like_result_offset(
                loom_call_like_const_cast(module, invocation->op))
          : 0;
  for (iree_host_size_t result_index = 0;
       result_index < invocation->result_count; ++result_index) {
    const loom_value_id_t result_value_id =
        invocation->result_value_ids[result_index];
    if (!loom_testbench_type_has_storage(
            loom_module_value_type(module, result_value_id))) {
      continue;
    }
    const loom_tied_result_t* tied_result = loom_op_find_tied_result(
        invocation->op, (uint16_t)(result_offset + result_index));
    if (tied_result) {
      const loom_value_id_t source_value_id =
          loom_op_const_operands(invocation->op)[tied_result->operand_index];
      value_storage_identities[result_value_id] =
          value_storage_identities[source_value_id];
    } else {
      loom_testbench_set_storage_root(result_value_id,
                                      value_storage_identities);
    }
  }
}

static iree_status_t loom_testbench_reserve_launch_schedule_frames(
    loom_testbench_invocation_planner_t* planner,
    iree_host_size_t additional_count) {
  const iree_host_size_t required_count =
      planner->schedule_frame_count + additional_count;
  if (required_count <= planner->schedule_frame_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(
      planner->arena, planner->schedule_frame_count, required_count,
      sizeof(*planner->schedule_frames), &planner->schedule_frame_capacity,
      (void**)&planner->schedule_frames);
}

static void loom_testbench_append_launch_schedule_event(
    loom_testbench_invocation_planner_t* planner,
    loom_testbench_launch_schedule_event_kind_t kind,
    loom_location_id_t location) {
  planner->launch_schedule_events[(*planner->launch_schedule_event_count)++] =
      (loom_testbench_launch_schedule_event_t){
          .kind = kind,
          .location = location,
      };
  ++planner->case_plan->launch_schedule_event_count;
}

static iree_status_t loom_testbench_push_launch_schedule_frame(
    loom_testbench_invocation_planner_t* planner, const loom_op_t* schedule_op,
    iree_host_size_t base_epoch, iree_host_size_t depth) {
  IREE_RETURN_IF_ERROR(
      loom_testbench_reserve_launch_schedule_frames(planner, 1));
  loom_region_t* body = loom_testbench_launch_schedule_body(schedule_op);
  loom_block_t* block =
      body && body->block_count != 0 ? loom_region_entry_block(body) : NULL;
  const bool is_concurrent =
      loom_command_composition_schedule_mode(schedule_op) ==
      LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT;
  planner->schedule_frames[planner->schedule_frame_count++] =
      (loom_testbench_launch_schedule_frame_t){
          .next_op = block ? block->first_op : NULL,
          .base_epoch = base_epoch,
          .span = 0,
          .depth = depth,
          .location = schedule_op->location,
          .is_concurrent = is_concurrent,
      };
  loom_testbench_append_launch_schedule_event(
      planner,
      is_concurrent ? LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_CONCURRENT_BEGIN
                    : LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_SERIAL_BEGIN,
      schedule_op->location);
  return iree_ok_status();
}

static iree_status_t loom_testbench_launch_schedule_child_epoch(
    const loom_testbench_launch_schedule_frame_t* frame,
    iree_host_size_t* out_epoch) {
  *out_epoch = frame->base_epoch;
  if (frame->is_concurrent) {
    return iree_ok_status();
  }
  if (!iree_host_size_checked_add(frame->base_epoch, frame->span, out_epoch)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "launch schedule execution epoch overflow");
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_commit_launch_schedule_span(
    loom_testbench_launch_schedule_frame_t* frame,
    iree_host_size_t child_span) {
  if (frame->is_concurrent) {
    frame->span = iree_max(frame->span, child_span);
    return iree_ok_status();
  }
  if (!iree_host_size_checked_add(frame->span, child_span, &frame->span)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "launch schedule execution span overflow");
  }
  return iree_ok_status();
}

static void loom_testbench_append_invocation_plan(
    loom_testbench_invocation_planner_t* planner, const loom_op_t* op,
    iree_host_size_t execution_epoch, iree_host_size_t schedule_depth) {
  loom_testbench_invocation_plan_t* invocation =
      &planner->invocations[(*planner->invocation_count)++];
  if (!loom_testbench_plan_invocation(planner->module, op, invocation)) {
    loom_testbench_append_issue(
        planner->issues, planner->issue_capacity, planner->issue_count,
        LOOM_TESTBENCH_ISSUE_INVALID_INVOCATION, planner->case_index,
        LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
        LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID, op, planner->case_plan->ref);
    return;
  }

  invocation->execution_epoch = execution_epoch;
  invocation->launch_schedule_depth = schedule_depth;
  loom_testbench_plan_invocation_result_storage(
      planner->module, invocation, planner->value_storage_identities);
  if (invocation->kind == LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
    planner->kernel_launches[(*planner->kernel_launch_count)++] = invocation;
    loom_testbench_append_launch_schedule_event(
        planner, LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_LAUNCH, op->location);
    ++planner->case_plan->kernel_launch_count;
  }
}

static iree_status_t loom_testbench_plan_launch_schedule(
    loom_testbench_invocation_planner_t* planner,
    const loom_op_t* schedule_op) {
  IREE_ASSERT_EQ(planner->schedule_frame_count, 0u);
  IREE_RETURN_IF_ERROR(loom_testbench_push_launch_schedule_frame(
      planner, schedule_op, planner->next_execution_epoch, /*depth=*/1));

  while (planner->schedule_frame_count != 0) {
    loom_testbench_launch_schedule_frame_t* frame =
        &planner->schedule_frames[planner->schedule_frame_count - 1];
    if (frame->next_op == NULL) {
      const iree_host_size_t completed_span = frame->span;
      loom_testbench_append_launch_schedule_event(
          planner, LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_END, frame->location);
      --planner->schedule_frame_count;
      if (planner->schedule_frame_count == 0) {
        if (!iree_host_size_checked_add(planner->next_execution_epoch,
                                        completed_span,
                                        &planner->next_execution_epoch)) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "testbench execution epoch overflow");
        }
      } else {
        IREE_RETURN_IF_ERROR(loom_testbench_commit_launch_schedule_span(
            &planner->schedule_frames[planner->schedule_frame_count - 1],
            completed_span));
      }
      continue;
    }

    loom_op_t* op = frame->next_op;
    frame->next_op = op->next_op;
    iree_host_size_t child_epoch = 0;
    IREE_RETURN_IF_ERROR(
        loom_testbench_launch_schedule_child_epoch(frame, &child_epoch));
    if (loom_kernel_launch_isa(op)) {
      loom_testbench_append_invocation_plan(planner, op, child_epoch,
                                            frame->depth);
      IREE_RETURN_IF_ERROR(
          loom_testbench_commit_launch_schedule_span(frame, 1));
    } else if (loom_testbench_is_launch_schedule_op(op)) {
      const iree_host_size_t child_depth = frame->depth + 1;
      IREE_RETURN_IF_ERROR(loom_testbench_push_launch_schedule_frame(
          planner, op, child_epoch, child_depth));
    } else if (!loom_kernel_launch_yield_isa(op)) {
      loom_testbench_append_issue(
          planner->issues, planner->issue_capacity, planner->issue_count,
          LOOM_TESTBENCH_ISSUE_UNSUPPORTED_CASE_BODY_OP, planner->case_index,
          LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
          LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID, op, planner->case_plan->ref);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_plan_case_body(
    const loom_module_t* module, iree_host_size_t case_index,
    iree_host_size_t max_samples_per_case,
    loom_testbench_case_plan_t* case_plan,
    loom_testbench_parameter_plan_t* parameters,
    iree_host_size_t* inout_parameter_count,
    loom_testbench_value_source_plan_t* value_sources,
    iree_host_size_t* inout_value_source_count,
    loom_testbench_file_write_plan_t* file_writes,
    iree_host_size_t* inout_file_write_count,
    loom_testbench_invocation_plan_t* invocations,
    iree_host_size_t* inout_invocation_count,
    const loom_testbench_invocation_plan_t** kernel_launches,
    iree_host_size_t* inout_kernel_launch_count,
    loom_testbench_launch_schedule_event_t* launch_schedule_events,
    iree_host_size_t* inout_launch_schedule_event_count,
    loom_testbench_storage_identity_t* value_storage_identities,
    loom_testbench_expectation_plan_t* expectations,
    iree_host_size_t* inout_expectation_count, loom_testbench_issue_t* issues,
    iree_host_size_t issue_capacity, iree_host_size_t* inout_issue_count,
    iree_arena_allocator_t* arena) {
  case_plan->parameters =
      parameters ? parameters + *inout_parameter_count : NULL;
  case_plan->value_sources =
      value_sources ? value_sources + *inout_value_source_count : NULL;
  case_plan->file_writes =
      file_writes ? file_writes + *inout_file_write_count : NULL;
  case_plan->invocations =
      invocations ? invocations + *inout_invocation_count : NULL;
  case_plan->kernel_launches =
      kernel_launches ? kernel_launches + *inout_kernel_launch_count : NULL;
  case_plan->kernel_launch_count = 0;
  case_plan->launch_schedule_events =
      launch_schedule_events
          ? launch_schedule_events + *inout_launch_schedule_event_count
          : NULL;
  case_plan->launch_schedule_event_count = 0;
  case_plan->expectations =
      expectations ? expectations + *inout_expectation_count : NULL;
  case_plan->issues = issues ? issues + *inout_issue_count : NULL;
  case_plan->cartesian_sample_count = 1;
  case_plan->sample_count = 1;

  loom_testbench_invocation_planner_t invocation_planner = {
      .module = module,
      .case_index = case_index,
      .case_plan = case_plan,
      .invocations = invocations,
      .invocation_count = inout_invocation_count,
      .kernel_launches = kernel_launches,
      .kernel_launch_count = inout_kernel_launch_count,
      .launch_schedule_events = launch_schedule_events,
      .launch_schedule_event_count = inout_launch_schedule_event_count,
      .value_storage_identities = value_storage_identities,
      .issues = issues,
      .issue_capacity = issue_capacity,
      .issue_count = inout_issue_count,
      .arena = arena,
  };

  loom_region_t* body = loom_check_case_body(case_plan->op);
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_testbench_is_supported_check_body_op(module, op)) {
        loom_testbench_append_issue(
            issues, issue_capacity, inout_issue_count,
            LOOM_TESTBENCH_ISSUE_UNSUPPORTED_CASE_BODY_OP, case_index,
            LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
            LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID, op, case_plan->ref);
        continue;
      }

      if (loom_testbench_is_launch_schedule_op(op)) {
        IREE_RETURN_IF_ERROR(
            loom_testbench_plan_launch_schedule(&invocation_planner, op));
        continue;
      }

      if (loom_testbench_is_parameter_op(op)) {
        loom_testbench_parameter_plan_t* parameter =
            &parameters[(*inout_parameter_count)++];
        if (!loom_testbench_plan_parameter(module, op, parameter)) {
          loom_testbench_append_issue(
              issues, issue_capacity, inout_issue_count,
              LOOM_TESTBENCH_ISSUE_INVALID_PARAMETER, case_index,
              LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
              LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID, op, case_plan->ref);
          continue;
        }
        loom_testbench_multiply_sample_count(
            parameter->sample_count, max_samples_per_case,
            &case_plan->cartesian_sample_count, &case_plan->sample_count,
            &case_plan->sample_count_truncated);
        continue;
      }

      if (loom_testbench_is_value_source_op(op)) {
        loom_testbench_value_source_plan_t* source =
            &value_sources[(*inout_value_source_count)++];
        if (!loom_testbench_plan_value_source(module, op, source)) {
          loom_testbench_append_issue(
              issues, issue_capacity, inout_issue_count,
              LOOM_TESTBENCH_ISSUE_INVALID_VALUE_SOURCE, case_index,
              LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
              LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID, op, case_plan->ref);
        } else {
          IREE_RETURN_IF_ERROR(loom_testbench_plan_value_source_storage(
              source, value_storage_identities));
        }
        continue;
      }

      if (loom_check_file_write_npy_isa(op)) {
        loom_testbench_file_write_plan_t* file_write =
            &file_writes[(*inout_file_write_count)++];
        if (!loom_testbench_plan_file_write(module, op, file_write)) {
          loom_testbench_append_issue(
              issues, issue_capacity, inout_issue_count,
              LOOM_TESTBENCH_ISSUE_INVALID_FILE_WRITE, case_index,
              LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
              LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID, op, case_plan->ref);
        }
        continue;
      }

      if (loom_testbench_is_invocation_op(module, op)) {
        const bool is_kernel_launch = loom_kernel_launch_isa(op);
        const iree_host_size_t execution_epoch =
            is_kernel_launch ? invocation_planner.next_execution_epoch++
                             : LOOM_TESTBENCH_EXECUTION_EPOCH_INVALID;
        loom_testbench_append_invocation_plan(
            &invocation_planner, op, execution_epoch, /*schedule_depth=*/0);
        continue;
      }

      if (loom_testbench_is_expectation_op(op)) {
        loom_testbench_expectation_plan_t* expectation =
            &expectations[(*inout_expectation_count)++];
        if (!loom_testbench_plan_expectation(module, op, expectation)) {
          loom_testbench_append_issue(
              issues, issue_capacity, inout_issue_count,
              LOOM_TESTBENCH_ISSUE_INVALID_EXPECTATION, case_index,
              LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
              LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID, op, case_plan->ref);
        }
        continue;
      }
    }
  }

  case_plan->parameter_count =
      case_plan->parameters
          ? (parameters + *inout_parameter_count) - case_plan->parameters
          : 0;
  case_plan->value_source_count =
      case_plan->value_sources ? (value_sources + *inout_value_source_count) -
                                     case_plan->value_sources
                               : 0;
  case_plan->file_write_count =
      case_plan->file_writes
          ? (file_writes + *inout_file_write_count) - case_plan->file_writes
          : 0;
  case_plan->invocation_count =
      case_plan->invocations
          ? (invocations + *inout_invocation_count) - case_plan->invocations
          : 0;
  case_plan->expectation_count =
      case_plan->expectations
          ? (expectations + *inout_expectation_count) - case_plan->expectations
          : 0;
  case_plan->issue_count =
      case_plan->issues ? (issues + *inout_issue_count) - case_plan->issues : 0;
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    if (!loom_testbench_parameter_names_have_duplicate(case_plan,
                                                       parameter_index)) {
      continue;
    }
    loom_testbench_append_issue(
        issues, issue_capacity, inout_issue_count,
        LOOM_TESTBENCH_ISSUE_DUPLICATE_PARAMETER_NAME, case_index,
        LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
        LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID,
        case_plan->parameters[parameter_index].op, case_plan->ref);
  }
  case_plan->issue_count =
      case_plan->issues ? (issues + *inout_issue_count) - case_plan->issues : 0;
  if (case_plan->issue_count != 0) {
    case_plan->sample_count = 0;
  }
  return iree_ok_status();
}

static void loom_testbench_fill_record_index_map(
    const loom_testbench_case_plan_t* cases, iree_host_size_t case_count,
    const loom_testbench_scenario_plan_t* scenarios,
    iree_host_size_t scenario_count,
    loom_testbench_record_indices_t* symbol_to_record_indices) {
  for (iree_host_size_t i = 0; i < case_count; ++i) {
    if (!loom_symbol_ref_is_valid(cases[i].ref)) {
      continue;
    }
    symbol_to_record_indices[cases[i].ref.symbol_id].case_index = (uint32_t)i;
  }
  for (iree_host_size_t i = 0; i < scenario_count; ++i) {
    if (!loom_symbol_ref_is_valid(scenarios[i].ref)) {
      continue;
    }
    symbol_to_record_indices[scenarios[i].ref.symbol_id].scenario_index =
        (uint32_t)i;
  }
}

static const loom_testbench_record_indices_t*
loom_testbench_record_indices_from_ref(
    const loom_testbench_record_indices_t* symbol_to_record_indices,
    iree_host_size_t symbol_count, loom_symbol_ref_t record_ref) {
  if (!loom_symbol_ref_is_valid(record_ref) || record_ref.module_id != 0 ||
      record_ref.symbol_id >= symbol_count) {
    return NULL;
  }
  const loom_testbench_record_indices_t* indices =
      &symbol_to_record_indices[record_ref.symbol_id];
  if (indices->case_index == LOOM_TESTBENCH_INTERNAL_INDEX_INVALID &&
      indices->scenario_index == LOOM_TESTBENCH_INTERNAL_INDEX_INVALID) {
    return NULL;
  }
  return indices;
}

static bool loom_testbench_range_parameter_sample_ordinal(
    const loom_testbench_parameter_plan_t* parameter, loom_attribute_t value,
    iree_host_size_t* out_sample_ordinal) {
  const loom_testbench_range_plan_t* range = &parameter->range;
  *out_sample_ordinal = IREE_HOST_SIZE_MAX;

  if (range->policy == LOOM_CHECK_PARAM_RANGE_POLICY_PO2) {
    int64_t target_value = 0;
    int64_t lower_value = 0;
    int64_t upper_value = 0;
    int64_t sample_value = 0;
    if (!loom_testbench_attr_as_i64_checked(value, &target_value) ||
        !loom_testbench_attr_as_i64_checked(range->lower, &lower_value) ||
        !loom_testbench_attr_as_i64_checked(range->upper, &upper_value) ||
        !loom_testbench_next_power_of_two(lower_value, &sample_value) ||
        target_value < sample_value || target_value > upper_value) {
      return false;
    }
    for (iree_host_size_t ordinal = 0; ordinal < parameter->sample_count;
         ++ordinal) {
      if (sample_value == target_value) {
        *out_sample_ordinal = ordinal;
        return true;
      }
      if (sample_value > INT64_MAX / 2) {
        break;
      }
      sample_value *= 2;
    }
    return false;
  }

  if (range->policy == LOOM_CHECK_PARAM_RANGE_POLICY_LINEAR &&
      loom_testbench_scalar_type_is_integral_sample(range->scalar_type)) {
    int64_t target_value = 0;
    int64_t lower_value = 0;
    int64_t upper_value = 0;
    int64_t step_value = 1;
    if (!loom_testbench_attr_as_i64_checked(value, &target_value) ||
        !loom_testbench_attr_as_i64_checked(range->lower, &lower_value) ||
        !loom_testbench_attr_as_i64_checked(range->upper, &upper_value) ||
        (!loom_attr_is_absent(range->step) &&
         !loom_testbench_attr_as_i64_checked(range->step, &step_value)) ||
        step_value <= 0 || target_value < lower_value ||
        target_value > upper_value) {
      return false;
    }
    int64_t offset = target_value - lower_value;
    if (offset % step_value != 0) {
      return false;
    }
    int64_t ordinal = offset / step_value;
    if (ordinal < 0 || (uint64_t)ordinal >= parameter->sample_count) {
      return false;
    }
    *out_sample_ordinal = (iree_host_size_t)ordinal;
    return true;
  }

  if (range->policy == LOOM_CHECK_PARAM_RANGE_POLICY_LINEAR &&
      loom_scalar_type_is_float(range->scalar_type)) {
    double target_value = 0.0;
    double lower_value = 0.0;
    double step_value = 0.0;
    if (!loom_testbench_attr_as_f64_checked(value, &target_value) ||
        !loom_testbench_attr_as_f64_checked(range->lower, &lower_value) ||
        !loom_testbench_attr_as_f64_checked(range->step, &step_value) ||
        !(step_value > 0.0)) {
      return false;
    }
    double ordinal_value = (target_value - lower_value) / step_value;
    double rounded_ordinal = floor(ordinal_value + 0.5);
    double ordinal_delta = fabs(ordinal_value - rounded_ordinal);
    if (!(rounded_ordinal >= 0.0) || ordinal_delta > 1e-9 ||
        rounded_ordinal > (double)IREE_HOST_SIZE_MAX) {
      return false;
    }
    iree_host_size_t ordinal = (iree_host_size_t)rounded_ordinal;
    if (ordinal >= parameter->sample_count) {
      return false;
    }
    double sample_value = lower_value + (double)ordinal * step_value;
    double sample_delta = fabs(sample_value - target_value);
    double tolerance = fabs(step_value) * 1e-9 + 1e-12;
    if (sample_delta > tolerance) {
      return false;
    }
    *out_sample_ordinal = ordinal;
    return true;
  }

  return false;
}

static bool loom_testbench_parameter_sample_ordinal_from_value(
    const loom_testbench_parameter_plan_t* parameter, loom_attribute_t value,
    iree_host_size_t* out_sample_ordinal) {
  *out_sample_ordinal = IREE_HOST_SIZE_MAX;
  switch (parameter->kind) {
    case LOOM_TESTBENCH_PARAMETER_RANGE:
      return loom_testbench_range_parameter_sample_ordinal(parameter, value,
                                                           out_sample_ordinal);
    case LOOM_TESTBENCH_PARAMETER_CHOICE: {
      int64_t target_value = 0;
      if (!loom_testbench_attr_as_i64_checked(value, &target_value)) {
        return false;
      }
      for (iree_host_size_t i = 0; i < parameter->choice.count; ++i) {
        if (parameter->choice.values[i] == target_value) {
          *out_sample_ordinal = i;
          return true;
        }
      }
      return false;
    }
    case LOOM_TESTBENCH_PARAMETER_SEED: {
      int64_t target_value = 0;
      if (!loom_testbench_attr_as_i64_checked(value, &target_value) ||
          target_value < parameter->seed.base) {
        return false;
      }
      uint64_t ordinal = (uint64_t)(target_value - parameter->seed.base);
      if (ordinal >= parameter->seed.count) {
        return false;
      }
      *out_sample_ordinal = (iree_host_size_t)ordinal;
      return true;
    }
    default:
      return false;
  }
}

static iree_host_size_t loom_testbench_case_parameter_index_by_name(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_string_id_t name_id) {
  if (name_id >= module->strings.count) {
    return IREE_HOST_SIZE_MAX;
  }
  iree_string_view_t name = loom_string_table_get(&module->strings, name_id);
  for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
    if (iree_string_view_equal(case_plan->parameters[i].name, name)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static bool loom_testbench_benchmark_assignment_has_duplicate_key(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_host_size_t assignment_index) {
  if (assignment_index >= attrs.count ||
      attrs.entries[assignment_index].name_id >= module->strings.count) {
    return false;
  }
  iree_string_view_t name = loom_string_table_get(
      &module->strings, attrs.entries[assignment_index].name_id);
  for (iree_host_size_t i = 0; i < assignment_index; ++i) {
    if (attrs.entries[i].name_id >= module->strings.count) {
      continue;
    }
    if (iree_string_view_equal(
            loom_string_table_get(&module->strings, attrs.entries[i].name_id),
            name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_testbench_plan_benchmark_assignments(
    const loom_module_t* module, iree_host_size_t benchmark_index,
    iree_host_size_t max_samples_per_case,
    loom_testbench_benchmark_plan_t* benchmark,
    const loom_testbench_case_plan_t* case_plan, iree_arena_allocator_t* arena,
    loom_testbench_issue_t* issues, iree_host_size_t issue_capacity,
    iree_host_size_t* inout_issue_count) {
  benchmark->cartesian_sample_count = 1;
  benchmark->sample_count = 1;
  benchmark->sample_count_truncated = false;
  benchmark->parameter_sample_ordinal_count = case_plan->parameter_count;

  iree_host_size_t* parameter_sample_ordinals = NULL;
  iree_host_size_t issue_count_before = *inout_issue_count;
  if (case_plan->parameter_count != 0) {
    IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
        arena, case_plan->parameter_count, sizeof(*parameter_sample_ordinals),
        (void**)&parameter_sample_ordinals));
    for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
      parameter_sample_ordinals[i] =
          LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL;
    }
    benchmark->parameter_sample_ordinals = parameter_sample_ordinals;
  }

  for (iree_host_size_t i = 0; i < benchmark->attrs.count; ++i) {
    const loom_named_attr_t* assignment = &benchmark->attrs.entries[i];
    if (loom_testbench_benchmark_assignment_has_duplicate_key(
            module, benchmark->attrs, i)) {
      loom_testbench_append_issue(
          issues, issue_capacity, inout_issue_count,
          LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_ASSIGNMENT,
          benchmark->case_index, LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
          benchmark_index, benchmark->op, benchmark->record_ref);
      continue;
    }
    iree_host_size_t parameter_index =
        loom_testbench_case_parameter_index_by_name(module, case_plan,
                                                    assignment->name_id);
    if (parameter_index == IREE_HOST_SIZE_MAX) {
      loom_testbench_append_issue(
          issues, issue_capacity, inout_issue_count,
          LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_ASSIGNMENT,
          benchmark->case_index, LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
          benchmark_index, benchmark->op, benchmark->record_ref);
      continue;
    }

    iree_host_size_t sample_ordinal = IREE_HOST_SIZE_MAX;
    if (!loom_testbench_parameter_sample_ordinal_from_value(
            &case_plan->parameters[parameter_index], assignment->value,
            &sample_ordinal)) {
      loom_testbench_append_issue(
          issues, issue_capacity, inout_issue_count,
          LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_ASSIGNMENT,
          benchmark->case_index, LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
          benchmark_index, benchmark->op, benchmark->record_ref);
      continue;
    }
    parameter_sample_ordinals[parameter_index] = sample_ordinal;
  }

  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    if (benchmark->parameter_sample_ordinals[parameter_index] !=
        LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL) {
      continue;
    }
    loom_testbench_multiply_sample_count(
        case_plan->parameters[parameter_index].sample_count,
        max_samples_per_case, &benchmark->cartesian_sample_count,
        &benchmark->sample_count, &benchmark->sample_count_truncated);
  }

  if (*inout_issue_count != issue_count_before || case_plan->issue_count != 0) {
    benchmark->sample_count = 0;
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_sample_count(
    const loom_testbench_scenario_plan_t* scenario,
    iree_host_size_t* out_sample_count) {
  iree_host_size_t trials_per_configuration = 0;
  for (iree_host_size_t i = 0; i < scenario->trial_count; ++i) {
    const iree_host_size_t trial_count = scenario->trials[i].trial_count;
    if (trial_count > IREE_HOST_SIZE_MAX - trials_per_configuration) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "scenario '%.*s' trial domain count exceeds host limits",
          (int)scenario->name.size, scenario->name.data);
    }
    trials_per_configuration += trial_count;
  }
  if (scenario->configuration_count != 0 &&
      trials_per_configuration >
          IREE_HOST_SIZE_MAX / scenario->configuration_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "scenario '%.*s' configuration and trial product exceeds host limits",
        (int)scenario->name.size, scenario->name.data);
  }
  *out_sample_count = trials_per_configuration * scenario->configuration_count;
  return iree_ok_status();
}

iree_status_t loom_testbench_plan_module(
    const loom_module_t* module, const loom_testbench_plan_options_t* options,
    iree_arena_allocator_t* arena, loom_testbench_module_plan_t* out_plan) {
  loom_testbench_plan_options_t default_options = {0};
  if (!options) {
    loom_testbench_plan_options_initialize(&default_options);
    options = &default_options;
  }
  iree_host_size_t max_samples_per_case = options->max_samples_per_case;
  if (max_samples_per_case == 0) {
    max_samples_per_case = LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE;
  }

  loom_testbench_plan_counts_t counts = {0};
  loom_testbench_count_module(module, &counts);
  loom_testbench_scenario_plan_counts_t scenario_counts = {0};
  loom_testbench_count_scenario_plans(module, &scenario_counts);
  if (scenario_counts.issue_capacity >
      IREE_HOST_SIZE_MAX - counts.issue_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "testbench planning issue count exceeds host "
                            "limits");
  }
  const iree_host_size_t issue_capacity =
      counts.issue_capacity + scenario_counts.issue_capacity;

  loom_testbench_case_plan_t* cases = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.case_count, sizeof(*cases), (void**)&cases));

  loom_testbench_benchmark_plan_t* benchmarks = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.benchmark_count, sizeof(*benchmarks), (void**)&benchmarks));

  loom_testbench_parameter_plan_t* parameters = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.parameter_count, sizeof(*parameters), (void**)&parameters));

  loom_testbench_value_source_plan_t* value_sources = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.value_source_count, sizeof(*value_sources),
      (void**)&value_sources));

  loom_testbench_file_write_plan_t* file_writes = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.file_write_count, sizeof(*file_writes),
      (void**)&file_writes));

  loom_testbench_invocation_plan_t* invocations = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.invocation_count, sizeof(*invocations),
      (void**)&invocations));

  const loom_testbench_invocation_plan_t** kernel_launches = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.kernel_launch_count, sizeof(*kernel_launches),
      (void**)&kernel_launches));

  loom_testbench_launch_schedule_event_t* launch_schedule_events = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.launch_schedule_event_count,
      sizeof(*launch_schedule_events), (void**)&launch_schedule_events));

  const iree_host_size_t value_storage_identity_count =
      counts.case_count == 0 ? 0 : module->values.count;
  loom_testbench_storage_identity_t* value_storage_identities = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, value_storage_identity_count, sizeof(*value_storage_identities),
      (void**)&value_storage_identities));
  for (iree_host_size_t i = 0; i < value_storage_identity_count; ++i) {
    value_storage_identities[i].root_value_id = LOOM_VALUE_ID_INVALID;
  }

  loom_testbench_expectation_plan_t* expectations = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, counts.expectation_count, sizeof(*expectations),
      (void**)&expectations));

  loom_testbench_issue_t* issues = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, issue_capacity, sizeof(*issues), (void**)&issues));

  loom_testbench_record_indices_t* symbol_to_record_indices = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_allocate_array(
      arena, module->symbols.count, sizeof(*symbol_to_record_indices),
      (void**)&symbol_to_record_indices));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    symbol_to_record_indices[i].case_index =
        LOOM_TESTBENCH_INTERNAL_INDEX_INVALID;
    symbol_to_record_indices[i].scenario_index =
        LOOM_TESTBENCH_INTERNAL_INDEX_INVALID;
  }

  iree_host_size_t case_count = 0;
  iree_host_size_t benchmark_count = 0;
  const loom_block_t* module_block =
      module->body && module->body->block_count != 0
          ? loom_region_const_entry_block(module->body)
          : NULL;
  const loom_op_t* op = NULL;
  if (module_block) {
    loom_block_for_each_op(module_block, op) {
      if (loom_check_case_isa(op)) {
        loom_testbench_case_plan_t* case_plan = &cases[case_count];
        case_plan->ref = loom_check_case_case_symbol(op);
        case_plan->symbol =
            loom_testbench_symbol_from_ref(module, case_plan->ref);
        case_plan->op = op;
        case_plan->name = loom_testbench_symbol_name(module, case_plan->symbol);
        case_plan->is_public =
            loom_check_case_visibility(op) == LOOM_CHECK_VISIBILITY_PUBLIC;
        ++case_count;
      } else if (loom_check_benchmark_isa(op)) {
        loom_testbench_benchmark_plan_t* benchmark =
            &benchmarks[benchmark_count];
        benchmark->ref = loom_check_benchmark_benchmark(op);
        benchmark->symbol =
            loom_testbench_symbol_from_ref(module, benchmark->ref);
        benchmark->op = op;
        benchmark->name = loom_testbench_symbol_name(module, benchmark->symbol);
        benchmark->record_ref = loom_check_benchmark_case_ref(op);
        benchmark->case_index = LOOM_TESTBENCH_CASE_INDEX_INVALID;
        benchmark->scenario_index = LOOM_TESTBENCH_SCENARIO_INDEX_INVALID;
        benchmark->attrs = loom_check_benchmark_attrs(op);
        benchmark->cartesian_sample_count = 1;
        benchmark->sample_count = 1;
        ++benchmark_count;
      }
    }
  }

  iree_host_size_t parameter_count = 0;
  iree_host_size_t value_source_count = 0;
  iree_host_size_t file_write_count = 0;
  iree_host_size_t invocation_count = 0;
  iree_host_size_t kernel_launch_count = 0;
  iree_host_size_t launch_schedule_event_count = 0;
  iree_host_size_t expectation_count = 0;
  iree_host_size_t issue_count = 0;
  for (iree_host_size_t i = 0; i < case_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_testbench_plan_case_body(
        module, i, max_samples_per_case, &cases[i], parameters,
        &parameter_count, value_sources, &value_source_count, file_writes,
        &file_write_count, invocations, &invocation_count, kernel_launches,
        &kernel_launch_count, launch_schedule_events,
        &launch_schedule_event_count, value_storage_identities, expectations,
        &expectation_count, issues, issue_capacity, &issue_count, arena));
  }

  const loom_testbench_scenario_plan_t* scenarios = NULL;
  iree_host_size_t scenario_count = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_plan_scenarios(
      module, &scenario_counts, arena, issues, issue_capacity, &issue_count,
      &scenarios, &scenario_count));

  loom_testbench_fill_record_index_map(
      cases, case_count, scenarios, scenario_count, symbol_to_record_indices);

  for (iree_host_size_t i = 0; i < benchmark_count; ++i) {
    loom_testbench_benchmark_plan_t* benchmark = &benchmarks[i];
    const loom_testbench_record_indices_t* record_indices =
        loom_testbench_record_indices_from_ref(symbol_to_record_indices,
                                               module->symbols.count,
                                               benchmark->record_ref);
    if (record_indices == NULL) {
      loom_testbench_append_issue(issues, issue_capacity, &issue_count,
                                  LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_RECORD,
                                  LOOM_TESTBENCH_CASE_INDEX_INVALID,
                                  LOOM_TESTBENCH_SCENARIO_INDEX_INVALID, i,
                                  benchmark->op, benchmark->record_ref);
      benchmark->sample_count = 0;
      continue;
    }
    if (record_indices->case_index != LOOM_TESTBENCH_INTERNAL_INDEX_INVALID) {
      benchmark->case_index = record_indices->case_index;
      IREE_RETURN_IF_ERROR(loom_testbench_plan_benchmark_assignments(
          module, i, max_samples_per_case, benchmark,
          &cases[benchmark->case_index], arena, issues, issue_capacity,
          &issue_count));
      continue;
    }

    benchmark->scenario_index = record_indices->scenario_index;
    const loom_testbench_scenario_plan_t* scenario =
        &scenarios[benchmark->scenario_index];
    IREE_RETURN_IF_ERROR(loom_testbench_scenario_sample_count(
        scenario, &benchmark->cartesian_sample_count));
    benchmark->sample_count = benchmark->cartesian_sample_count;
    if (scenario->issue_count != 0) {
      benchmark->sample_count = 0;
    }
    for (iree_host_size_t assignment_index = 0;
         assignment_index < benchmark->attrs.count; ++assignment_index) {
      loom_testbench_append_issue(
          issues, issue_capacity, &issue_count,
          LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_ASSIGNMENT,
          LOOM_TESTBENCH_CASE_INDEX_INVALID, benchmark->scenario_index, i,
          benchmark->op, benchmark->record_ref);
      benchmark->sample_count = 0;
    }
  }

  out_plan->module = module;
  out_plan->value_storage_identities = value_storage_identities;
  out_plan->value_storage_identity_count = value_storage_identity_count;
  out_plan->cases = cases;
  out_plan->case_count = case_count;
  out_plan->scenarios = scenarios;
  out_plan->scenario_count = scenario_count;
  out_plan->benchmarks = benchmarks;
  out_plan->benchmark_count = benchmark_count;
  out_plan->issues = issues;
  out_plan->issue_count = issue_count;
  return iree_ok_status();
}

loom_testbench_case_sample_t loom_testbench_benchmark_case_sample(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t ordinal) {
  return (loom_testbench_case_sample_t){
      .ordinal = ordinal,
      .assignments = benchmark_plan->parameter_sample_ordinals,
  };
}

iree_host_size_t loom_testbench_case_sample_next(
    loom_testbench_case_sample_t* remaining, iree_host_size_t sample_count) {
  if (remaining->assignments != NULL) {
    const iree_host_size_t assignment = *remaining->assignments++;
    if (assignment != LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL) {
      return assignment;
    }
  }
  const iree_host_size_t ordinal = remaining->ordinal % sample_count;
  remaining->ordinal /= sample_count;
  return ordinal;
}

int loom_testbench_case_sample_compare(
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_sample_t lhs, loom_testbench_case_sample_t rhs) {
  if (lhs.assignments == rhs.assignments) {
    return (lhs.ordinal > rhs.ordinal) - (lhs.ordinal < rhs.ordinal);
  }
  int order = 0;
  for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
    const iree_host_size_t radix = case_plan->parameters[i].sample_count;
    const iree_host_size_t lhs_ordinal =
        loom_testbench_case_sample_next(&lhs, radix);
    const iree_host_size_t rhs_ordinal =
        loom_testbench_case_sample_next(&rhs, radix);
    if (lhs_ordinal != rhs_ordinal) {
      // Later parameters are more significant, preserving Cartesian order
      // without requiring its numeric representation.
      order = lhs_ordinal < rhs_ordinal ? -1 : 1;
    }
  }
  return order;
}

bool loom_testbench_case_sample_flatten(
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_sample_t sample, iree_host_size_t* out_ordinal) {
  *out_ordinal = 0;
  if (sample.assignments == NULL) {
    *out_ordinal = sample.ordinal;
    return true;
  }
  iree_host_size_t ordinal = 0;
  iree_host_size_t stride = 1;
  bool stride_fits = true;
  for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
    const iree_host_size_t radix = case_plan->parameters[i].sample_count;
    const iree_host_size_t coordinate =
        loom_testbench_case_sample_next(&sample, radix);
    if (coordinate != 0) {
      iree_host_size_t offset = 0;
      if (!stride_fits ||
          !iree_host_size_checked_mul(coordinate, stride, &offset) ||
          !iree_host_size_checked_add(ordinal, offset, &ordinal)) {
        return false;
      }
    }
    if (stride_fits) {
      stride_fits = iree_host_size_checked_mul(stride, radix, &stride);
    }
  }
  *out_ordinal = ordinal;
  return true;
}

loom_testbench_scenario_sample_coordinate_t
loom_testbench_benchmark_sample_scenario_coordinate(
    const loom_testbench_scenario_plan_t* scenario_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t benchmark_sample_ordinal) {
  IREE_ASSERT(benchmark_plan->case_index == LOOM_TESTBENCH_CASE_INDEX_INVALID);
  IREE_ASSERT(benchmark_plan->scenario_index !=
              LOOM_TESTBENCH_SCENARIO_INDEX_INVALID);
  IREE_ASSERT(benchmark_sample_ordinal < benchmark_plan->sample_count);
  IREE_ASSERT(scenario_plan->configuration_count != 0);
  IREE_ASSERT(
      benchmark_plan->sample_count % scenario_plan->configuration_count == 0);

  const iree_host_size_t trials_per_configuration =
      benchmark_plan->sample_count / scenario_plan->configuration_count;
  loom_testbench_scenario_sample_coordinate_t coordinate = {
      .configuration_ordinal =
          benchmark_sample_ordinal / trials_per_configuration,
  };
  iree_host_size_t ordinal =
      benchmark_sample_ordinal % trials_per_configuration;
  for (iree_host_size_t trial_index = 0;
       trial_index < scenario_plan->trial_count; ++trial_index) {
    const iree_host_size_t trial_count =
        scenario_plan->trials[trial_index].trial_count;
    if (ordinal < trial_count) {
      coordinate.trial_index = trial_index;
      coordinate.trial_ordinal = ordinal;
      return coordinate;
    }
    ordinal -= trial_count;
  }
  IREE_ASSERT_UNREACHABLE("planned scenario sample outside its trial domains");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_testbench_range_sample_value(
    const loom_testbench_parameter_plan_t* parameter,
    iree_host_size_t parameter_sample_ordinal, loom_attribute_t* out_value) {
  const loom_testbench_range_plan_t* range = &parameter->range;
  if (range->policy == LOOM_CHECK_PARAM_RANGE_POLICY_PO2) {
    int64_t value = 0;
    int64_t lower = 0;
    int64_t upper = 0;
    if (!loom_testbench_attr_as_i64_checked(range->lower, &lower) ||
        !loom_testbench_attr_as_i64_checked(range->upper, &upper) ||
        !loom_testbench_next_power_of_two(lower, &value)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invalid po2 range plan");
    }
    for (iree_host_size_t i = 0; i < parameter_sample_ordinal; ++i) {
      if (value > INT64_MAX / 2 || value * 2 > upper) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "range sample ordinal out of bounds");
      }
      value *= 2;
    }
    *out_value = loom_attr_i64(value);
    return iree_ok_status();
  }

  if (range->policy == LOOM_CHECK_PARAM_RANGE_POLICY_LINEAR &&
      loom_testbench_scalar_type_is_integral_sample(range->scalar_type)) {
    int64_t lower = 0;
    int64_t step = 1;
    if (!loom_testbench_attr_as_i64_checked(range->lower, &lower)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invalid integer range plan");
    }
    if (!loom_attr_is_absent(range->step) &&
        !loom_testbench_attr_as_i64_checked(range->step, &step)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invalid integer range step");
    }
    uint64_t offset = (uint64_t)parameter_sample_ordinal * (uint64_t)step;
    int64_t value = 0;
    if (!loom_testbench_add_i64_nonnegative_offset(lower, offset, &value)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "integer range sample overflows i64");
    }
    *out_value = loom_attr_i64(value);
    return iree_ok_status();
  }

  if (range->policy == LOOM_CHECK_PARAM_RANGE_POLICY_LINEAR &&
      loom_scalar_type_is_float(range->scalar_type)) {
    double lower = 0.0;
    double step = 0.0;
    if (!loom_testbench_attr_as_f64_checked(range->lower, &lower) ||
        !loom_testbench_attr_as_f64_checked(range->step, &step)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invalid floating-point range plan");
    }
    *out_value = loom_attr_f64(lower + (double)parameter_sample_ordinal * step);
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "unsupported range plan");
}

iree_status_t loom_testbench_parameter_sample_value(
    const loom_testbench_parameter_plan_t* parameter,
    iree_host_size_t parameter_sample_ordinal, loom_attribute_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  if (parameter_sample_ordinal >= parameter->sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter sample ordinal out of bounds");
  }

  switch (parameter->kind) {
    case LOOM_TESTBENCH_PARAMETER_RANGE:
      return loom_testbench_range_sample_value(
          parameter, parameter_sample_ordinal, out_value);
    case LOOM_TESTBENCH_PARAMETER_CHOICE:
      *out_value =
          loom_attr_i64(parameter->choice.values[parameter_sample_ordinal]);
      return iree_ok_status();
    case LOOM_TESTBENCH_PARAMETER_SEED: {
      int64_t seed = 0;
      if (!loom_testbench_add_i64_nonnegative_offset(
              parameter->seed.base, (uint64_t)parameter_sample_ordinal,
              &seed)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "seed sample overflows i64");
      }
      *out_value = loom_attr_i64(seed);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invalid parameter plan");
  }
}
