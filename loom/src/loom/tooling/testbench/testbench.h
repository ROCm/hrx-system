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
#include "loom/ops/check/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TESTBENCH_CASE_INDEX_INVALID IREE_HOST_SIZE_MAX
#define LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID IREE_HOST_SIZE_MAX
#define LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL IREE_HOST_SIZE_MAX

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
  // A check.benchmark does not reference a discovered check.case.
  LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_CASE = 3,
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
} loom_testbench_value_source_kind_t;

typedef enum loom_testbench_invocation_kind_e {
  // Invalid or uninitialized invocation slot.
  LOOM_TESTBENCH_INVOCATION_NONE = 0,
  // Semantic call-like op that invokes the function under test.
  LOOM_TESTBENCH_INVOCATION_ACTUAL = 1,
  // check.oracle.call op that invokes a reference provider.
  LOOM_TESTBENCH_INVOCATION_ORACLE = 2,
} loom_testbench_invocation_kind_t;

typedef enum loom_testbench_expectation_kind_e {
  // Invalid or uninitialized expectation slot.
  LOOM_TESTBENCH_EXPECTATION_NONE = 0,
  // check.expect.equal over scalar values or shaped buffer views.
  LOOM_TESTBENCH_EXPECTATION_EQUAL = 1,
  // check.expect.bitwise over scalar values or shaped buffer views.
  LOOM_TESTBENCH_EXPECTATION_BITWISE = 2,
  // check.expect.close approximate comparison.
  LOOM_TESTBENCH_EXPECTATION_CLOSE = 3,
  // check.expect.shape metadata comparison.
  LOOM_TESTBENCH_EXPECTATION_SHAPE = 4,
  // check.expect<provider> custom validation.
  LOOM_TESTBENCH_EXPECTATION_CUSTOM = 5,
} loom_testbench_expectation_kind_t;

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

typedef struct loom_testbench_invocation_plan_t {
  // Invocation kind and provider discriminator.
  loom_testbench_invocation_kind_t kind;
  // Module owning |op|, symbol references, strings, and attributes.
  const loom_module_t* module;
  // Operation that declared the invocation.
  const loom_op_t* op;
  // Function symbol passed to the execution or oracle provider.
  loom_symbol_ref_t callee_ref;
  // Interned provider string ID for oracle invocations, or INVALID otherwise.
  loom_string_id_t provider_id;
  // Borrowed provider name for oracle invocations, or empty otherwise.
  iree_string_view_t provider;
  // Provider-specific attributes for oracle invocations.
  loom_named_attr_slice_t attrs;
  // Borrowed SSA value IDs used as invocation inputs.
  const loom_value_id_t* input_value_ids;
  // Number of entries in |input_value_ids|.
  iree_host_size_t input_count;
  // Borrowed SSA value IDs assigned from invocation results.
  const loom_value_id_t* result_value_ids;
  // Number of entries in |result_value_ids|.
  iree_host_size_t result_count;
} loom_testbench_invocation_plan_t;

typedef struct loom_testbench_close_expectation_plan_t {
  // Absolute tolerance.
  double absolute_tolerance;
  // Relative tolerance.
  double relative_tolerance;
  // NaN comparison policy.
  loom_check_expect_close_nan_t nan_policy;
} loom_testbench_close_expectation_plan_t;

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

typedef struct loom_testbench_custom_expectation_plan_t {
  // Interned provider string ID.
  loom_string_id_t provider_id;
  // Borrowed provider name.
  iree_string_view_t provider;
  // Borrowed custom validation attributes.
  loom_named_attr_slice_t attrs;
} loom_testbench_custom_expectation_plan_t;

typedef struct loom_testbench_expectation_plan_t {
  // Expectation kind and payload discriminator.
  loom_testbench_expectation_kind_t kind;
  // Operation that declared the expectation.
  const loom_op_t* op;
  // Primary actual value checked by the expectation.
  loom_value_id_t actual_value_id;
  // Expected value checked by binary expectations, or INVALID for shape.
  loom_value_id_t expected_value_id;
  // Type of |actual_value_id| at planning time.
  loom_type_t type;
  union {
    // Payload for check.expect.close.
    loom_testbench_close_expectation_plan_t close;
    // Payload for check.expect.shape.
    loom_testbench_shape_expectation_plan_t shape;
    // Payload for check.expect<provider>.
    loom_testbench_custom_expectation_plan_t custom;
  };
} loom_testbench_expectation_plan_t;

typedef struct loom_testbench_issue_t {
  // Structured issue classification.
  loom_testbench_issue_kind_t kind;
  // Case ordinal in the module plan, or INVALID for module-level issues.
  iree_host_size_t case_index;
  // Benchmark ordinal in the module plan, or INVALID for non-benchmark issues.
  iree_host_size_t benchmark_index;
  // Operation related to the issue.
  const loom_op_t* op;
  // Referenced case symbol when the issue involves one.
  loom_symbol_ref_t case_ref;
} loom_testbench_issue_t;

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
  // Optional module-local symbol reference naming this benchmark policy.
  loom_symbol_ref_t ref;
  // Symbol table entry for |ref|, or NULL when the benchmark is anonymous.
  const loom_symbol_t* symbol;
  // Defining check.benchmark operation.
  const loom_op_t* op;
  // Explicit symbol name or stable case-derived anonymous name.
  iree_string_view_t name;
  // Referenced check.case symbol.
  loom_symbol_ref_t case_ref;
  // Ordinal of the referenced case in the module plan.
  iree_host_size_t case_index;
  // Borrowed benchmark assignment dictionary.
  loom_named_attr_slice_t attrs;
  // Per-parameter fixed sample ordinals selected by this benchmark. Entries are
  // LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL for unassigned parameters.
  const iree_host_size_t* parameter_sample_ordinals;
  // Number of entries in |parameter_sample_ordinals|.
  iree_host_size_t parameter_sample_ordinal_count;
  // Full cartesian sample count after benchmark assignments are applied.
  iree_host_size_t cartesian_sample_count;
  // Number of benchmark samples retained for execution after budget capping.
  iree_host_size_t sample_count;
  // True when |sample_count| is smaller than |cartesian_sample_count|.
  bool sample_count_truncated;
} loom_testbench_benchmark_plan_t;

typedef struct loom_testbench_module_plan_t {
  // Module this plan was derived from.
  const loom_module_t* module;
  // Discovered check.case records.
  const loom_testbench_case_plan_t* cases;
  // Number of entries in |cases|.
  iree_host_size_t case_count;
  // Discovered check.benchmark records.
  const loom_testbench_benchmark_plan_t* benchmarks;
  // Number of entries in |benchmarks|.
  iree_host_size_t benchmark_count;
  // Flat issue list across cases and benchmark records.
  const loom_testbench_issue_t* issues;
  // Number of entries in |issues|.
  iree_host_size_t issue_count;
} loom_testbench_module_plan_t;

// Plans check.case/check.benchmark records in |module|.
//
// IR validity problems are reported as structured issues in |out_plan|. The
// status channel is reserved for infrastructure failures such as allocation.
iree_status_t loom_testbench_plan_module(
    const loom_module_t* module, const loom_testbench_plan_options_t* options,
    iree_arena_allocator_t* arena, loom_testbench_module_plan_t* out_plan);

// Maps a case sample ordinal to a parameter-local sample ordinal.
//
// Parameters use mixed-radix expansion in source order. Parameter 0 varies
// fastest, then parameter 1, and so on. |sample_ordinal| must be less than the
// case plan's |cartesian_sample_count|.
iree_host_size_t loom_testbench_case_sample_parameter_ordinal(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, iree_host_size_t parameter_index);

// Maps a benchmark-local sample ordinal to the referenced case sample ordinal.
//
// Assigned benchmark parameters use their fixed parameter-local ordinals.
// Unassigned parameters use mixed-radix expansion in source order over the
// reduced benchmark sample domain.
iree_host_size_t loom_testbench_benchmark_sample_case_ordinal(
    const loom_testbench_case_plan_t* case_plan,
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
