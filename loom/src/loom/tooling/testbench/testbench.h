// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Production testbench case discovery and deterministic sample planning.
//
// This library is target-free: it interprets check.* harness structure and
// leaves actual compilation, oracle dispatch, and value materialization to
// later execution layers.

#ifndef LOOM_TOOLING_TESTBENCH_TESTBENCH_H_
#define LOOM_TOOLING_TESTBENCH_TESTBENCH_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ops/check/expectation.h"
#include "loom/ops/check/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TESTBENCH_CASE_INDEX_INVALID IREE_HOST_SIZE_MAX
#define LOOM_TESTBENCH_SCENARIO_INDEX_INVALID IREE_HOST_SIZE_MAX
#define LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID IREE_HOST_SIZE_MAX
#define LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL IREE_HOST_SIZE_MAX
#define LOOM_TESTBENCH_EXECUTION_EPOCH_INVALID IREE_HOST_SIZE_MAX

enum {
  // Default cap for the number of concrete samples executed per check.case.
  LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE = 4096,
};

typedef enum loom_testbench_parameter_kind_e {
  // Invalid or uninitialized parameter slot.
  LOOM_TESTBENCH_PARAMETER_NONE = 0,
  // Parameter produced by check.param.range.
  LOOM_TESTBENCH_PARAMETER_RANGE = 1,
  // Parameter produced by check.param.choice.
  LOOM_TESTBENCH_PARAMETER_CHOICE = 2,
  // Parameter produced by check.param.seed.
  LOOM_TESTBENCH_PARAMETER_SEED = 3,
} loom_testbench_parameter_kind_t;

typedef enum loom_testbench_issue_kind_e {
  // Invalid or uninitialized issue slot.
  LOOM_TESTBENCH_ISSUE_NONE = 0,
  // A check.case body op is outside the V0 executable testbench subset.
  LOOM_TESTBENCH_ISSUE_UNSUPPORTED_CASE_BODY_OP = 1,
  // A parameter op has no valid deterministic sample set.
  LOOM_TESTBENCH_ISSUE_INVALID_PARAMETER = 2,
  // A check.benchmark does not reference a discovered test record.
  LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_RECORD = 3,
  // A named parameter collides with another parameter in the same check.case.
  LOOM_TESTBENCH_ISSUE_DUPLICATE_PARAMETER_NAME = 4,
  // A check.benchmark assignment key or value does not match the referenced
  // check.case parameter domain.
  LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_ASSIGNMENT = 5,
  // A value source op cannot be planned as a deterministic input.
  LOOM_TESTBENCH_ISSUE_INVALID_VALUE_SOURCE = 6,
  // A file output op cannot be planned as a deterministic sink.
  LOOM_TESTBENCH_ISSUE_INVALID_FILE_WRITE = 7,
  // An actual or oracle invocation op cannot be planned for execution.
  LOOM_TESTBENCH_ISSUE_INVALID_INVOCATION = 8,
  // A check.expect.* op cannot be planned for evaluation.
  LOOM_TESTBENCH_ISSUE_INVALID_EXPECTATION = 9,
  // A check.scenario configuration body operation is unsupported.
  LOOM_TESTBENCH_ISSUE_UNSUPPORTED_SCENARIO_BODY_OP = 10,
  // A check.trial recipe operation is unsupported.
  LOOM_TESTBENCH_ISSUE_UNSUPPORTED_TRIAL_BODY_OP = 11,
  // A check.compare or check.invoke action cannot be planned.
  LOOM_TESTBENCH_ISSUE_INVALID_SCENARIO_ACTION = 12,
  // A check.generate call cannot be planned for recipe execution.
  LOOM_TESTBENCH_ISSUE_INVALID_TRIAL_GENERATOR = 13,
} loom_testbench_issue_kind_t;

typedef enum loom_testbench_value_source_kind_e {
  // Invalid or uninitialized value source slot.
  LOOM_TESTBENCH_VALUE_SOURCE_NONE = 0,
  // Static scalar literal from check.literal.
  LOOM_TESTBENCH_VALUE_SOURCE_LITERAL = 1,
  // Deterministic shaped iota from check.generate.iota.
  LOOM_TESTBENCH_VALUE_SOURCE_IOTA = 2,
  // Deterministic shaped fill from check.generate.fill.
  LOOM_TESTBENCH_VALUE_SOURCE_FILL = 3,
  // Deterministic random-uniform value from check.generate.random.uniform.
  LOOM_TESTBENCH_VALUE_SOURCE_RANDOM_UNIFORM = 4,
  // Fixture file input from check.file.read.npy.
  LOOM_TESTBENCH_VALUE_SOURCE_FILE_READ_NPY = 5,
  // Dense typed alias from check.tensor.view.
  LOOM_TESTBENCH_VALUE_SOURCE_TENSOR_VIEW = 6,
  // Named immutable entropy derivation from check.entropy.fork.
  LOOM_TESTBENCH_VALUE_SOURCE_ENTROPY_FORK = 7,
  // Deterministic indexed entropy word from check.entropy.read.
  LOOM_TESTBENCH_VALUE_SOURCE_ENTROPY_READ = 8,
} loom_testbench_value_source_kind_t;

typedef enum loom_testbench_invocation_kind_e {
  // Invalid or uninitialized invocation slot.
  LOOM_TESTBENCH_INVOCATION_NONE = 0,
  // Semantic call-like op that invokes an ordinary function.
  LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL = 1,
  // kernel.launch op that dispatches a device kernel.
  LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH = 2,
  // check.oracle.call op that invokes a reference provider.
  LOOM_TESTBENCH_INVOCATION_ORACLE = 3,
  // Finite command program issued with staged specializations.
  LOOM_TESTBENCH_INVOCATION_COMMAND_PROGRAM = 4,
  // Pipeline episode issued with staged specializations.
  LOOM_TESTBENCH_INVOCATION_PIPELINE = 5,
} loom_testbench_invocation_kind_t;

typedef enum loom_testbench_scenario_action_kind_e {
  // Invalid or uninitialized scenario action.
  LOOM_TESTBENCH_SCENARIO_ACTION_NONE = 0,
  // Differential target/oracle action from check.compare.
  LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE = 1,
  // Target-only action from check.invoke.
  LOOM_TESTBENCH_SCENARIO_ACTION_INVOKE = 2,
} loom_testbench_scenario_action_kind_t;

typedef enum loom_testbench_trial_recipe_step_kind_e {
  // Invalid or uninitialized recipe step.
  LOOM_TESTBENCH_TRIAL_RECIPE_STEP_NONE = 0,
  // Built-in deterministic value source.
  LOOM_TESTBENCH_TRIAL_RECIPE_STEP_VALUE_SOURCE = 1,
  // Authored ordinary-function generator invocation.
  LOOM_TESTBENCH_TRIAL_RECIPE_STEP_GENERATOR = 2,
} loom_testbench_trial_recipe_step_kind_t;

typedef struct loom_testbench_plan_options_t {
  // Maximum samples retained per case after cartesian expansion.
  iree_host_size_t max_samples_per_case;
} loom_testbench_plan_options_t;

// Initializes planning options with a bounded default sample budget.
void loom_testbench_plan_options_initialize(
    loom_testbench_plan_options_t* out_options);

typedef struct loom_testbench_range_plan_t {
  // Static sampling policy declared by check.param.range.
  loom_check_param_range_policy_t policy;
  // Scalar element type of the produced parameter value.
  loom_scalar_type_t scalar_type;
  // Inclusive lower bound attribute.
  loom_attribute_t lower;
  // Inclusive upper bound attribute.
  loom_attribute_t upper;
  // Positive step attribute, or ABSENT when the policy supplies the step.
  loom_attribute_t step;
} loom_testbench_range_plan_t;

typedef struct loom_testbench_choice_plan_t {
  // Borrowed immutable choice values from the module-owned attribute payload.
  const int64_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_testbench_choice_plan_t;

typedef struct loom_testbench_seed_plan_t {
  // First seed value.
  int64_t base;
  // Number of sequential seeds.
  iree_host_size_t count;
} loom_testbench_seed_plan_t;

typedef struct loom_testbench_literal_source_plan_t {
  // Static scalar payload.
  loom_attribute_t value;
} loom_testbench_literal_source_plan_t;

typedef struct loom_testbench_iota_source_plan_t {
  // First generated element value.
  loom_attribute_t offset;
  // Additive step between generated elements.
  loom_attribute_t step;
  // Positive element period, or zero when the sequence does not repeat.
  iree_host_size_t period;
} loom_testbench_iota_source_plan_t;

typedef struct loom_testbench_fill_source_plan_t {
  // Static scalar payload used for every element.
  loom_attribute_t value;
} loom_testbench_fill_source_plan_t;

typedef struct loom_testbench_random_uniform_source_plan_t {
  // SSA value providing the sampled seed.
  loom_value_id_t seed_value_id;
  // Inclusive lower generated value bound.
  loom_attribute_t lower;
  // Inclusive upper generated value bound.
  loom_attribute_t upper;
} loom_testbench_random_uniform_source_plan_t;

typedef struct loom_testbench_file_source_plan_t {
  // Interned module string ID for the fixture path.
  loom_string_id_t path_id;
  // Borrowed fixture path text.
  iree_string_view_t path;
} loom_testbench_file_source_plan_t;

typedef struct loom_testbench_tensor_view_source_plan_t {
  // SSA value providing the source tensor storage.
  loom_value_id_t source_value_id;
  // Static byte offset from the beginning of the source tensor.
  iree_device_size_t byte_offset;
} loom_testbench_tensor_view_source_plan_t;

typedef struct loom_testbench_entropy_fork_source_plan_t {
  // SSA value providing the parent entropy identity.
  loom_value_id_t entropy_value_id;
  // Interned name used to derive the child identity.
  loom_string_id_t name_id;
  // Borrowed fork name text.
  iree_string_view_t name;
} loom_testbench_entropy_fork_source_plan_t;

typedef struct loom_testbench_entropy_read_source_plan_t {
  // SSA value providing the entropy identity.
  loom_value_id_t entropy_value_id;
  // Borrowed dynamic ordinal value IDs in source order.
  const loom_value_id_t* ordinal_value_ids;
  // Number of entries in |ordinal_value_ids|.
  iree_host_size_t ordinal_value_count;
  // Borrowed static ordinals with INT64_MIN sentinels at dynamic positions.
  const int64_t* static_ordinals;
  // Number of entries in |static_ordinals|.
  iree_host_size_t static_ordinal_count;
} loom_testbench_entropy_read_source_plan_t;

typedef struct loom_testbench_parameter_plan_t {
  // Parameter kind and payload discriminator.
  loom_testbench_parameter_kind_t kind;
  // Operation that produced the parameter.
  const loom_op_t* op;
  // SSA value produced by the parameter op.
  loom_value_id_t value_id;
  // Type of |value_id| at planning time.
  loom_type_t type;
  // Optional case-local parameter name, or INVALID when unnamed.
  loom_string_id_t name_id;
  // Borrowed case-local parameter name text, or empty when unnamed.
  iree_string_view_t name;
  // Number of valid samples for this parameter.
  iree_host_size_t sample_count;
  union {
    // Payload for check.param.range.
    loom_testbench_range_plan_t range;
    // Payload for check.param.choice.
    loom_testbench_choice_plan_t choice;
    // Payload for check.param.seed.
    loom_testbench_seed_plan_t seed;
  };
} loom_testbench_parameter_plan_t;

typedef struct loom_testbench_value_source_plan_t {
  // Source kind and payload discriminator.
  loom_testbench_value_source_kind_t kind;
  // Operation that produced the source value.
  const loom_op_t* op;
  // SSA value produced by the source op.
  loom_value_id_t value_id;
  // Type of |value_id| at planning time.
  loom_type_t type;
  union {
    // Payload for check.literal.
    loom_testbench_literal_source_plan_t literal;
    // Payload for check.generate.iota.
    loom_testbench_iota_source_plan_t iota;
    // Payload for check.generate.fill.
    loom_testbench_fill_source_plan_t fill;
    // Payload for check.generate.random.uniform.
    loom_testbench_random_uniform_source_plan_t random_uniform;
    // Payload for check.file.read.npy.
    loom_testbench_file_source_plan_t file;
    // Payload for check.tensor.view.
    loom_testbench_tensor_view_source_plan_t tensor_view;
    // Payload for check.entropy.fork.
    loom_testbench_entropy_fork_source_plan_t entropy_fork;
    // Payload for check.entropy.read.
    loom_testbench_entropy_read_source_plan_t entropy_read;
  };
} loom_testbench_value_source_plan_t;

typedef struct loom_testbench_file_write_plan_t {
  // Operation that declared the file output.
  const loom_op_t* op;
  // SSA value written to the fixture path.
  loom_value_id_t value_id;
  // Type of |value_id| at planning time.
  loom_type_t type;
  // Interned module string ID for the output path.
  loom_string_id_t path_id;
  // Borrowed output path text.
  iree_string_view_t path;
  // Static write policy. Absent IR spelling defaults to ON_FAILURE.
  loom_check_file_write_npy_mode_t mode;
} loom_testbench_file_write_plan_t;

// Canonical authored storage identity for a shaped SSA value.
typedef struct loom_testbench_storage_identity_t {
  // Ultimate allocation value, or INVALID when no storage is represented.
  loom_value_id_t root_value_id;
  // Cumulative byte offset from |root_value_id|.
  iree_device_size_t byte_offset;
} loom_testbench_storage_identity_t;

typedef enum loom_testbench_launch_schedule_event_kind_e {
  // A kernel launch entry from the case's ordered launch list.
  LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_LAUNCH = 0,
  // Beginning of a serial launch schedule region.
  LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_SERIAL_BEGIN = 1,
  // Beginning of a concurrent launch schedule region.
  LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_CONCURRENT_BEGIN = 2,
  // End of the innermost launch schedule region.
  LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_END = 3,
} loom_testbench_launch_schedule_event_kind_t;

// One event in a flat, balanced representation of authored launch structure.
typedef struct loom_testbench_launch_schedule_event_t {
  // Event discriminator.
  loom_testbench_launch_schedule_event_kind_t kind;
  // Location of the source launch or schedule operation.
  loom_location_id_t location;
} loom_testbench_launch_schedule_event_t;

typedef struct loom_testbench_invocation_plan_t {
  // Invocation kind and provider discriminator.
  loom_testbench_invocation_kind_t kind;
  // Module owning |op|, symbol references, strings, and attributes.
  const loom_module_t* module;
  // Operation that declared the invocation.
  const loom_op_t* op;
  // Subject symbol passed to the execution or oracle provider.
  loom_symbol_ref_t callee_ref;
  // Interned provider string ID for oracle invocations, or INVALID otherwise.
  loom_string_id_t provider_id;
  // Borrowed provider name for oracle invocations, or empty otherwise.
  iree_string_view_t provider;
  // Provider-specific attributes for oracle invocations.
  loom_named_attr_slice_t attrs;
  // Execution epoch for kernel launches, or INVALID for other invocations.
  // Adjacent launches in the same epoch may execute concurrently; increasing
  // epochs require an execution and visibility edge.
  iree_host_size_t execution_epoch;
  // Number of enclosing kernel launch schedule operations. Direct check.case
  // launches have depth zero.
  iree_host_size_t launch_schedule_depth;
  // Borrowed SSA value IDs used as kernel launch workload inputs.
  const loom_value_id_t* workload_value_ids;
  // Number of entries in |workload_value_ids|.
  iree_host_size_t workload_count;
  // Borrowed SSA value IDs used as invocation inputs.
  const loom_value_id_t* input_value_ids;
  // Number of entries in |input_value_ids|.
  iree_host_size_t input_count;
  // Borrowed SSA value IDs assigned from invocation results.
  const loom_value_id_t* result_value_ids;
  // Number of entries in |result_value_ids|.
  iree_host_size_t result_count;
} loom_testbench_invocation_plan_t;

typedef struct loom_testbench_trial_recipe_step_t {
  // Step kind and payload discriminator.
  loom_testbench_trial_recipe_step_kind_t kind;
  union {
    // Plan for a built-in check value source.
    loom_testbench_value_source_plan_t value_source;
    // Plan for a check.generate ordinary-function call.
    loom_testbench_invocation_plan_t generator;
  };
} loom_testbench_trial_recipe_step_t;

typedef struct loom_testbench_approximate_expectation_plan_t {
  // Absolute tolerance.
  double absolute_tolerance;
  // Relative tolerance.
  double relative_tolerance;
  // NaN comparison policy.
  loom_check_expect_approximate_nan_t nan_policy;
  // Numeric descriptor kind used to decode the actual operand.
  uint8_t actual_numeric_kind;
  // Numeric descriptor kind used to decode the expected operand.
  uint8_t expected_numeric_kind;
} loom_testbench_approximate_expectation_plan_t;

static_assert(sizeof(loom_testbench_approximate_expectation_plan_t) == 24,
              "approximate expectation plans must remain compact");

typedef struct loom_testbench_integer_close_expectation_plan_t {
  // Inclusive exact distance accepted by the expectation.
  uint64_t maximum_distance;
  // Numeric descriptor kind used to decode the actual operand.
  uint8_t actual_numeric_kind;
  // Numeric descriptor kind used to decode the expected operand.
  uint8_t expected_numeric_kind;
} loom_testbench_integer_close_expectation_plan_t;

typedef struct loom_testbench_shape_expectation_plan_t {
  // Borrowed dynamic dimension value IDs in source order.
  const loom_value_id_t* dimension_value_ids;
  // Number of entries in |dimension_value_ids|.
  iree_host_size_t dimension_value_count;
  // Borrowed static dimensions with INT64_MIN sentinels for dynamic positions.
  const int64_t* static_dimensions;
  // Rank represented by |static_dimensions|.
  iree_host_size_t static_dimension_count;
} loom_testbench_shape_expectation_plan_t;

typedef struct loom_testbench_event_expectation_plan_t {
  // Interned provider string ID.
  loom_string_id_t provider_id;
  // Borrowed provider name.
  iree_string_view_t provider;
  // Borrowed event subset-match attributes.
  loom_named_attr_slice_t attrs;
} loom_testbench_event_expectation_plan_t;

typedef struct loom_testbench_expectation_plan_t {
  // Expectation kind and payload discriminator.
  loom_check_expectation_kind_t kind;
  // Operation that declared the expectation.
  const loom_op_t* op;
  // Primary actual value checked by the expectation.
  loom_value_id_t actual_value_id;
  // Expected value checked by binary expectations, or INVALID for shape.
  loom_value_id_t expected_value_id;
  // Type of |actual_value_id| at planning time.
  loom_type_t type;
  union {
    // Payload for check.expect.approximate.
    loom_testbench_approximate_expectation_plan_t approximate;
    // Payload for check.expect.close.signed and close.unsigned.
    loom_testbench_integer_close_expectation_plan_t integer_close;
    // Payload for check.expect.shape.
    loom_testbench_shape_expectation_plan_t shape;
    // Payload for check.expect.event<provider>.
    loom_testbench_event_expectation_plan_t event;
  };
} loom_testbench_expectation_plan_t;

static_assert(IREE_PTR_SIZE != 8 ||
                  sizeof(loom_testbench_expectation_plan_t) == 88,
              "expectation plan must retain its 64-bit layout");

typedef struct loom_testbench_issue_t {
  // Structured issue classification.
  loom_testbench_issue_kind_t kind;
  // Case ordinal in the module plan, or INVALID for module-level issues.
  iree_host_size_t case_index;
  // Scenario ordinal in the module plan, or INVALID for non-scenario issues.
  iree_host_size_t scenario_index;
  // Benchmark ordinal in the module plan, or INVALID for non-benchmark issues.
  iree_host_size_t benchmark_index;
  // Operation related to the issue.
  const loom_op_t* op;
  // Referenced case or scenario symbol when the issue involves one.
  loom_symbol_ref_t record_ref;
} loom_testbench_issue_t;

typedef struct loom_testbench_scenario_action_plan_t {
  // Action kind and execution-mode discriminator.
  loom_testbench_scenario_action_kind_t kind;
  // Defining check.compare or check.invoke operation.
  const loom_op_t* op;
  // Target invocation with target result identities for comparisons.
  loom_testbench_invocation_plan_t target;
  // Oracle invocation with expected result identities, or NONE for invoke.
  loom_testbench_invocation_plan_t oracle;
  // Explicit comparison expectations in source order.
  const loom_testbench_expectation_plan_t* expectations;
  // Number of entries in |expectations|.
  iree_host_size_t expectation_count;
  // True when an expectation consumes target device events.
  bool expects_device_events;
} loom_testbench_scenario_action_plan_t;

typedef struct loom_testbench_trial_plan_t {
  // Defining check.trial operation.
  const loom_op_t* op;
  // SSA value receiving the concrete trial ordinal.
  loom_value_id_t ordinal_value_id;
  // SSA value receiving the immutable trial entropy identity.
  loom_value_id_t entropy_value_id;
  // Number of concrete trials in this domain.
  iree_host_size_t trial_count;
  // Runtime recipe steps in source order.
  const loom_testbench_trial_recipe_step_t* recipe_steps;
  // Number of entries in |recipe_steps|.
  iree_host_size_t recipe_step_count;
  // Terminal action for every concrete trial.
  loom_testbench_scenario_action_plan_t action;
  // Issues discovered while planning this trial.
  const loom_testbench_issue_t* issues;
  // Number of entries in |issues|.
  iree_host_size_t issue_count;
} loom_testbench_trial_plan_t;

typedef struct loom_testbench_scenario_plan_t {
  // Module-local symbol reference naming this scenario.
  loom_symbol_ref_t ref;
  // Symbol table entry for |ref|.
  const loom_symbol_t* symbol;
  // Defining check.scenario operation.
  const loom_op_t* op;
  // Borrowed scenario name from the module string table.
  iree_string_view_t name;
  // True when the check.scenario is public testbench API.
  bool is_public;
  // True when the scenario declares an explicit configuration domain.
  bool is_configured;
  // Number of concrete configurations; one for an unconfigured scenario.
  iree_host_size_t configuration_count;
  // SSA value receiving the configuration ordinal, or INVALID when absent.
  loom_value_id_t configuration_ordinal_value_id;
  // SSA value receiving configuration entropy, or INVALID when absent.
  loom_value_id_t configuration_entropy_value_id;
  // Configuration-scoped source plans in source order.
  const loom_testbench_value_source_plan_t* configuration_sources;
  // Number of entries in |configuration_sources|.
  iree_host_size_t configuration_source_count;
  // Trial domains in source order.
  const loom_testbench_trial_plan_t* trials;
  // Number of entries in |trials|.
  iree_host_size_t trial_count;
  // Issues discovered while planning this scenario.
  const loom_testbench_issue_t* issues;
  // Number of entries in |issues|.
  iree_host_size_t issue_count;
} loom_testbench_scenario_plan_t;

typedef struct loom_testbench_scenario_sample_coordinate_t {
  // Scenario configuration selected by the benchmark sample.
  iree_host_size_t configuration_ordinal;
  // Scenario trial domain selected by the benchmark sample.
  iree_host_size_t trial_index;
  // Trial ordinal within the selected domain.
  iree_host_size_t trial_ordinal;
} loom_testbench_scenario_sample_coordinate_t;

typedef struct loom_testbench_case_plan_t {
  // Module-local symbol reference naming this case.
  loom_symbol_ref_t ref;
  // Symbol table entry for |ref|.
  const loom_symbol_t* symbol;
  // Defining check.case operation.
  const loom_op_t* op;
  // Borrowed case name from the module string table.
  iree_string_view_t name;
  // True when the check.case is public testbench API.
  bool is_public;
  // Parameter plans in source order.
  const loom_testbench_parameter_plan_t* parameters;
  // Number of entries in |parameters|.
  iree_host_size_t parameter_count;
  // Value source plans in source order.
  const loom_testbench_value_source_plan_t* value_sources;
  // Number of entries in |value_sources|.
  iree_host_size_t value_source_count;
  // File output plans in source order.
  const loom_testbench_file_write_plan_t* file_writes;
  // Number of entries in |file_writes|.
  iree_host_size_t file_write_count;
  // Invocation plans in source order.
  const loom_testbench_invocation_plan_t* invocations;
  // Number of entries in |invocations|.
  iree_host_size_t invocation_count;
  // Kernel launch plans in source order, referencing entries in |invocations|.
  const loom_testbench_invocation_plan_t* const* kernel_launches;
  // Number of entries in |kernel_launches|.
  iree_host_size_t kernel_launch_count;
  // Balanced launch/schedule events preserving authored execution structure.
  const loom_testbench_launch_schedule_event_t* launch_schedule_events;
  // Number of entries in |launch_schedule_events|.
  iree_host_size_t launch_schedule_event_count;
  // Expectation plans in source order.
  const loom_testbench_expectation_plan_t* expectations;
  // Number of entries in |expectations|.
  iree_host_size_t expectation_count;
  // Full cartesian sample count before applying |max_samples_per_case|.
  iree_host_size_t cartesian_sample_count;
  // Number of samples retained for execution after budget capping.
  iree_host_size_t sample_count;
  // True when |sample_count| is smaller than |cartesian_sample_count|.
  bool sample_count_truncated;
  // Issues discovered while planning this case.
  const loom_testbench_issue_t* issues;
  // Number of entries in |issues|.
  iree_host_size_t issue_count;
} loom_testbench_case_plan_t;

typedef struct loom_testbench_benchmark_plan_t {
  // Module-local symbol reference naming this benchmark record.
  loom_symbol_ref_t ref;
  // Symbol table entry for |ref|.
  const loom_symbol_t* symbol;
  // Defining check.benchmark operation.
  const loom_op_t* op;
  // Explicit symbol name used for reports and benchmark selection.
  iree_string_view_t name;
  // Referenced check.case or check.scenario record symbol.
  loom_symbol_ref_t record_ref;
  // Ordinal of the referenced case, or INVALID for a scenario benchmark.
  iree_host_size_t case_index;
  // Ordinal of the referenced scenario, or INVALID for a case benchmark.
  iree_host_size_t scenario_index;
  // Borrowed benchmark assignment dictionary.
  loom_named_attr_slice_t attrs;
  // Fixed sample ordinals in case parameter order, covering every parameter.
  // Empty for scenario benchmarks. Entries are PARAMETER_SAMPLE_ORDINAL_ALL
  // for unassigned parameters.
  const iree_host_size_t* parameter_sample_ordinals;
  // Number of entries in |parameter_sample_ordinals|.
  iree_host_size_t parameter_sample_ordinal_count;
  // Full case cartesian domain or concrete scenario action count.
  iree_host_size_t cartesian_sample_count;
  // Number of benchmark samples retained for execution. Case domains may be
  // capped by planning options; explicit scenario domains remain exhaustive.
  iree_host_size_t sample_count;
  // True when |sample_count| is smaller than |cartesian_sample_count|.
  bool sample_count_truncated;
} loom_testbench_benchmark_plan_t;

typedef struct loom_testbench_module_plan_t {
  // Module this plan was derived from.
  const loom_module_t* module;
  // Dense storage identities indexed by module value ID. Values outside
  // planned check.case bodies have INVALID roots.
  const loom_testbench_storage_identity_t* value_storage_identities;
  // Number of entries in |value_storage_identities|.
  iree_host_size_t value_storage_identity_count;
  // Discovered check.case records.
  const loom_testbench_case_plan_t* cases;
  // Number of entries in |cases|.
  iree_host_size_t case_count;
  // Discovered check.scenario records.
  const loom_testbench_scenario_plan_t* scenarios;
  // Number of entries in |scenarios|.
  iree_host_size_t scenario_count;
  // Discovered check.benchmark records.
  const loom_testbench_benchmark_plan_t* benchmarks;
  // Number of entries in |benchmarks|.
  iree_host_size_t benchmark_count;
  // Flat issue list across cases, scenarios, and benchmark records.
  const loom_testbench_issue_t* issues;
  // Number of entries in |issues|.
  iree_host_size_t issue_count;
} loom_testbench_module_plan_t;

// Plans check.case/check.scenario/check.benchmark records in |module|.
//
// IR validity problems are reported as structured issues in |out_plan|. The
// status channel is reserved for infrastructure failures such as allocation.
iree_status_t loom_testbench_plan_module(
    const loom_module_t* module, const loom_testbench_plan_options_t* options,
    iree_arena_allocator_t* arena, loom_testbench_module_plan_t* out_plan);

// Returns true when |op| is a deterministic check value source understood by
// the testbench planner.
bool loom_testbench_is_value_source_op(const loom_op_t* op);

// Plans one deterministic check value source operation.
bool loom_testbench_plan_value_source(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_value_source_plan_t* out_source);

// Returns true when |op| is an explicit check expectation.
bool loom_testbench_is_expectation_op(const loom_op_t* op);

// Plans one explicit check expectation operation.
bool loom_testbench_plan_expectation(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_expectation_plan_t* out_expectation);

// One concrete sample in a case's parameter domain. The Cartesian product may
// exceed host limits even when every parameter and the selected sweep fit.
// Retaining assignments avoids flattening that product into a lossy index.
typedef struct loom_testbench_case_sample_t {
  // Position in the reduced sweep of unassigned parameters, first varying
  // fastest. With no assignments this is the ordinary case sample ordinal.
  iree_host_size_t ordinal;
  // Borrowed parameter-local ordinals in case parameter order, or NULL when
  // every parameter is unassigned. A non-NULL array covers every parameter;
  // PARAMETER_SAMPLE_ORDINAL_ALL entries vary with |ordinal|.
  const iree_host_size_t* assignments;
} loom_testbench_case_sample_t;

// Selects a sample without allocation. The result borrows the benchmark plan's
// assignments. |ordinal| must be within the planned benchmark sample count.
loom_testbench_case_sample_t loom_testbench_benchmark_case_sample(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t ordinal);

// Consumes one parameter from a copy of a sample, returning its parameter-local
// ordinal. Call once per parameter in source order with that parameter's
// nonzero sample count. Decoding the whole sample is linear in parameter count.
iree_host_size_t loom_testbench_case_sample_next(
    loom_testbench_case_sample_t* remaining, iree_host_size_t sample_count);

// Compares resolved parameter coordinates within the same case, independent of
// how a benchmark pins them. Zero means the samples select identical inputs.
int loom_testbench_case_sample_compare(
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_sample_t lhs, loom_testbench_case_sample_t rhs);

// Derives the global Cartesian ordinal for reporting, if it fits in a host
// word. On false, |out_ordinal| is zero; the sample is still valid and usable.
// Execution and identity must remain valid when this representation cannot
// fit; sort accelerators may use it with coordinate comparison on overflow.
bool loom_testbench_case_sample_flatten(
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_sample_t sample, iree_host_size_t* out_ordinal);

// Maps a benchmark-local sample ordinal to its scenario domain coordinate.
//
// Configurations vary slowest. Within each configuration, trial domains and
// their concrete ordinals follow source order.
loom_testbench_scenario_sample_coordinate_t
loom_testbench_benchmark_sample_scenario_coordinate(
    const loom_testbench_scenario_plan_t* scenario_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t benchmark_sample_ordinal);

// Materializes one static scalar parameter sample as an attribute payload.
iree_status_t loom_testbench_parameter_sample_value(
    const loom_testbench_parameter_plan_t* parameter,
    iree_host_size_t parameter_sample_ordinal, loom_attribute_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_TESTBENCH_H_
