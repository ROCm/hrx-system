// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Live kernel specialization classes synthesized from template decisions.

#ifndef LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_CLASSIFIER_H_
#define LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_CLASSIFIER_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/decision/class_partition.h"
#include "loom/transforms/kernel/kernel_specialization.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default maximum number of live classes collected for one kernel.
#define LOOM_KERNEL_CLASS_DEFAULT_LIMIT 128

// Default maximum affine terms transferred from ranged launch facts.
#define LOOM_KERNEL_CLASS_DEFAULT_RANGED_TERM_LIMIT 4

// Dense index into one collection's parent-linked trace table.
typedef uint32_t loom_kernel_class_trace_id_t;

// Invalid trace ID denoting the root with no accepted decisions.
#define LOOM_KERNEL_CLASS_TRACE_ID_INVALID UINT32_MAX

// Boundary projection form for one decision operand.
typedef uint8_t loom_kernel_class_projection_kind_t;
enum loom_kernel_class_projection_kind_e {
  // An exact source constant independent of kernel arguments.
  LOOM_KERNEL_CLASS_PROJECTION_CONSTANT = 0,

  // A linear expression over kernel ABI arguments.
  LOOM_KERNEL_CLASS_PROJECTION_AFFINE = 1,

  // Source facts with no supported boundary expression.
  LOOM_KERNEL_CLASS_PROJECTION_STATIC_FACTS = 2,
};

// One coefficient times a kernel ABI argument.
typedef struct loom_kernel_class_projection_term_t {
  // Signed coefficient multiplying the argument.
  int64_t coefficient;

  // Kernel ABI argument ordinal.
  uint16_t argument_ordinal;

  // Reserved bytes. Always zero.
  uint8_t reserved[6];
} loom_kernel_class_projection_term_t;

static_assert(sizeof(loom_kernel_class_projection_term_t) == 16,
              "kernel class projection terms must remain compact");

// One source SSA value projected to the kernel ABI boundary.
typedef struct loom_kernel_class_projection_t {
  // Source SSA value used for construction-time identity.
  loom_value_id_t source_value_id;

  // Projection form from loom_kernel_class_projection_kind_t.
  loom_kernel_class_projection_kind_t kind;

  // Reserved bytes. Always zero.
  uint8_t reserved[3];

  // Constant term for constant and affine projections.
  int64_t constant;

  // Borrowed affine terms, or NULL for constant/static projections.
  const loom_kernel_class_projection_term_t* terms;

  // Source-scoped scalar facts for a static projection.
  loom_value_facts_t static_facts;

  // Number of affine terms.
  uint16_t term_count;

  // Reserved bytes. Always zero.
  uint8_t trailing_reserved[6];
} loom_kernel_class_projection_t;

// Construction-time reason a decision cannot refine launch classes.
typedef uint8_t loom_kernel_class_decision_unavailable_reason_t;
enum loom_kernel_class_decision_unavailable_reason_e {
  // Every material input is available at the kernel boundary.
  LOOM_KERNEL_CLASS_DECISION_AVAILABLE = 0,

  // The application's structured or CFG path conditions are not projected.
  LOOM_KERNEL_CLASS_DECISION_PATH_CONDITION = 1,

  // At least one material scalar input has no boundary expression.
  LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT = 2,

  // At least one contextual target feature remains unresolved.
  LOOM_KERNEL_CLASS_DECISION_UNRESOLVED_TARGET = 3,
};

// Publication properties of one family or provider applicability contract.
typedef uint8_t loom_kernel_class_contract_flags_t;
enum loom_kernel_class_contract_flag_bits_e {
  // At least one scalar predicate reads an application result.
  LOOM_KERNEL_CLASS_CONTRACT_FLAG_RESULT_DEPENDENT = 1u << 0,
};

// Publication metadata for one dense provider action.
typedef struct loom_kernel_class_action_t {
  // Direct choice ordinal in the ranked decision program.
  uint32_t choice_ordinal;

  // Properties of the provider applicability contract.
  loom_kernel_class_contract_flags_t contract_flags;

  // Reserved bytes. Always zero.
  uint8_t reserved[3];
} loom_kernel_class_action_t;

static_assert(sizeof(loom_kernel_class_action_t) == 8,
              "kernel class actions must remain compact");

// Boundary projections supplying static structure in one selected body.
typedef struct loom_kernel_class_shape_t {
  // Sorted distinct projection ordinals; evaluated only for live classes.
  const uint32_t* projections;
  // Number of required projections.
  uint32_t count;
} loom_kernel_class_shape_t;

// Immutable boundary form of one template application decision.
typedef struct loom_kernel_class_decision_t {
  // Source application, retained directly without reference-analysis storage.
  const loom_op_t* application;

  // Ranked family decision model.
  const loom_template_decision_model_t* model;

  // Synthetic projection value IDs in application argument order.
  const loom_value_id_t* argument_values;

  // Synthetic projection value IDs in application result order.
  const loom_value_id_t* result_values;

  // Unique projection ordinals evaluated for this decision.
  const uint32_t* projection_ordinals;

  // Pre-resolved contextual feature outcomes in model-local order.
  const loom_decision_truth_t* feature_outcomes;

  // Publication metadata indexed by provider action ordinal.
  const loom_kernel_class_action_t* actions;

  // Structural requirements indexed by provider action, independent of the
  // header predicates used to select that provider.
  const loom_kernel_class_shape_t* action_shapes;

  // Structural requirements of every provider still possible in a generic
  // residual. Provider headers determine applicability; bodies supply
  // structure.
  loom_kernel_class_shape_t residual_shape;

  // Selection result from kernel-source facts before launch-site refinement.
  loom_decision_program_result_t generic_result;

  // Number of application arguments.
  uint16_t argument_count;

  // Number of application results.
  uint16_t result_count;

  // Number of unique projection ordinals.
  uint32_t projection_count;

  // Construction-time availability reason.
  loom_kernel_class_decision_unavailable_reason_t unavailable_reason;

  // Publication properties of the family hard requirements.
  loom_kernel_class_contract_flags_t hard_requirement_flags;

  // True if a source-feasible provider omits a generic structural requirement.
  // Used only for class-budget admission, never provider applicability or rank.
  bool can_reduce_structure;

  // Reserved byte. Always zero.
  uint8_t reserved;
} loom_kernel_class_decision_t;

// Immutable classifier owned by one materialized kernel source product.
typedef struct loom_kernel_class_classifier_t {
  // Source module retained by the owning kernel product.
  const loom_module_t* module;

  // Module-local kernel source symbol.
  loom_symbol_id_t kernel_symbol_id;

  // Projected source values sorted by ascending source SSA value ID.
  const loom_kernel_class_projection_t* projections;

  // Decisions in stable source order.
  const loom_kernel_class_decision_t* decisions;

  // Argument facts already implied by the source kernel contract, in ABI order.
  const loom_value_facts_t* argument_facts;

  // Structural values required directly by the kernel and exact helper calls.
  loom_kernel_class_shape_t shape;

  // True when any source or provider body has structural requirements.
  bool has_shape_requirements;

  // Number of projected source values.
  uint32_t projection_count;

  // Number of template application decisions.
  uint32_t decision_count;

  // Maximum provider count across decisions.
  uint32_t maximum_provider_count;

  // Kernel ABI argument count expected at every launch site.
  uint16_t kernel_argument_count;

  // Leading source arguments bound while compiling the artifact.
  uint16_t specialization_count;
} loom_kernel_class_classifier_t;

// Class collection policy.
typedef struct loom_kernel_class_collection_options_t {
  // Maximum number of live classes.
  loom_decision_class_ordinal_t class_limit;

  // Maximum affine terms transferred when any input facts are ranged.
  uint16_t ranged_transfer_term_limit;
} loom_kernel_class_collection_options_t;

// Returns the default bounded collection policy.
static inline loom_kernel_class_collection_options_t
loom_kernel_class_collection_options_default(void) {
  return (loom_kernel_class_collection_options_t){
      /*.class_limit=*/LOOM_KERNEL_CLASS_DEFAULT_LIMIT,
      /*.ranged_transfer_term_limit=*/
      LOOM_KERNEL_CLASS_DEFAULT_RANGED_TERM_LIMIT,
  };
}

// How one source decision contributed to the final collection.
typedef uint8_t loom_kernel_class_decision_state_t;
enum loom_kernel_class_decision_state_e {
  // The decision and its selected provider are present in every class trace.
  LOOM_KERNEL_CLASS_DECISION_ACCEPTED = 0,

  // A path-dependent decision retained its generic source form.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_PATH_CONDITION = 1,

  // An unprojectable decision retained its generic source form.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNPROJECTABLE_INPUT = 2,

  // A target-dependent decision retained its generic source form.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNRESOLVED_TARGET = 3,

  // A selected result-dependent action retained its generic source form.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_RESULT_DEPENDENT = 4,

  // Refining this decision would have exceeded the class limit.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT = 5,
};

// Collection result for one source decision.
typedef struct loom_kernel_class_decision_result_t {
  // State from loom_kernel_class_decision_state_t.
  loom_kernel_class_decision_state_t state;

  // Reserved bytes. Always zero.
  uint8_t reserved[3];
} loom_kernel_class_decision_result_t;

// One accepted decision outcome in a class trace.
typedef struct loom_kernel_class_trace_t {
  // Previous collected decision, or LOOM_KERNEL_CLASS_TRACE_ID_INVALID.
  // Collection reserves required decisions first, independently of source
  // order.
  loom_kernel_class_trace_id_t parent_trace_id;

  // Classifier decision ordinal.
  uint32_t decision_ordinal;

  // Selected provider action ordinal in the decision model.
  uint32_t action_ordinal;
} loom_kernel_class_trace_t;

// One projected structural fact retained through class materialization.
typedef struct loom_kernel_class_shape_value_t {
  // Source-boundary projection whose exact value distinguishes this class.
  uint32_t projection;
  // Exact scalar or unknown facts when no caller proves the value.
  loom_value_facts_t facts;
} loom_kernel_class_shape_value_t;

// One live kernel class.
typedef struct loom_kernel_class_t {
  // Final trace node describing all accepted decisions for this class.
  loom_kernel_class_trace_id_t trace_id;

  // Mandatory staged values shared by every site in this class.
  const loom_kernel_specialization_value_t* specializations;

  // Integer argument facts true at every site in this class, in ABI order.
  // Context-local extensions are not retained across publication.
  const loom_value_facts_t* argument_facts;

  // Structural facts shared by every site, independently of argument joins.
  const loom_kernel_class_shape_value_t* shape_values;

  // Number of structural facts.
  uint32_t shape_value_count;

  // Number of launch sites assigned to this class.
  iree_host_size_t member_count;
} loom_kernel_class_t;

// Transient live classes collected for one command workload.
typedef struct loom_kernel_class_collection_t {
  // Module owning literal specialization values; borrowed until
  // materialization.
  const loom_module_t* specialization_module;

  // Dense final class ordinal for every input site.
  const loom_decision_class_ordinal_t* site_classes;

  // Live classes in dense ordinal order.
  const loom_kernel_class_t* classes;

  // Parent-linked accepted decision traces.
  const loom_kernel_class_trace_t* traces;

  // Per-source-decision collection states.
  const loom_kernel_class_decision_result_t* decision_results;

  // Number of input sites.
  iree_host_size_t site_count;

  // Number of live classes.
  loom_decision_class_ordinal_t class_count;

  // Number of trace nodes.
  uint32_t trace_count;

  // Number of accepted source decisions.
  uint32_t accepted_decision_count;

  // Number of source decisions conservatively left generic.
  uint32_t skipped_decision_count;
} loom_kernel_class_collection_t;

// Builds an immutable boundary classifier for one kernel implementation body.
//
// Construction follows the kernel symbol's existing template-demand chain and
// projects its header decisions without walking IR. The structural requirement
// analysis owns one immutable body/call traversal. |kernel_facts| and
// |expression_context| must describe the same kernel function and immutable
// module snapshot. The classifier borrows |module| and the provider/model
// contracts backing |decision_models|; references, kernel facts, and symbolic
// expressions may retire after construction. Result storage belongs to |arena|;
// construction scratch returns to its block pool before this call returns.
iree_status_t loom_kernel_class_classifier_build(
    const loom_module_t* module, loom_symbol_id_t kernel_symbol_id,
    const loom_symbol_reference_table_t* references,
    const loom_template_decision_model_catalog_t* decision_models,
    const loom_value_fact_table_t* kernel_facts,
    loom_symbolic_expr_context_t* expression_context,
    const loom_template_applicability_target_t* kernel_target,
    iree_arena_allocator_t* arena,
    loom_kernel_class_classifier_t* out_classifier);

// Collects the finite live quotient of |sites| under |classifier|.
//
// Mandatory staged values, source structure, and decisions without generic
// residuals reserve their classes before optional refinement. |site_module| is
// required when the source has staged arguments. Provider evaluation consumes
// retained projections without allocation, IR traversal, or fact-table cloning.
// Admission includes the selected providers' structural variants and the
// remaining generic requirements. A decision can remain generic only when its
// kernel-source facts select a valid residual provider.
iree_status_t loom_kernel_class_classifier_collect(
    const loom_kernel_class_classifier_t* classifier,
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_kernel_class_collection_t* out_collection);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_CLASSIFIER_H_
