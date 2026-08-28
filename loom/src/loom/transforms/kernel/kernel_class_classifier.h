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

  // The application executes under a lexical condition not yet projected.
  LOOM_KERNEL_CLASS_DECISION_LEXICAL_CONDITION = 1,

  // At least one material scalar input has no boundary expression.
  LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT = 2,

  // At least one contextual target feature remains unresolved.
  LOOM_KERNEL_CLASS_DECISION_UNRESOLVED_TARGET = 3,
};

// Immutable boundary form of one template application decision.
typedef struct loom_kernel_class_decision_t {
  // Source application demand.
  const loom_template_demand_t* demand;

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

  // Reserved bytes. Always zero.
  uint8_t reserved[3];
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

  // Number of projected source values.
  uint32_t projection_count;

  // Number of template application decisions.
  uint32_t decision_count;

  // Maximum provider count across decisions.
  uint32_t maximum_provider_count;

  // Kernel ABI argument count expected at every launch site.
  uint16_t kernel_argument_count;

  // Reserved bytes. Always zero.
  uint8_t reserved[2];
} loom_kernel_class_classifier_t;

// One command-side launch fact environment.
typedef struct loom_kernel_class_site_t {
  // Existing O(1) fact table for the containing command function.
  const loom_value_fact_table_t* facts;

  // Kernel ABI argument value IDs in signature order.
  const loom_value_id_t* argument_values;
} loom_kernel_class_site_t;

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
      .class_limit = LOOM_KERNEL_CLASS_DEFAULT_LIMIT,
      .ranged_transfer_term_limit = LOOM_KERNEL_CLASS_DEFAULT_RANGED_TERM_LIMIT,
  };
}

// How one source decision contributed to the final collection.
typedef uint8_t loom_kernel_class_decision_state_t;
enum loom_kernel_class_decision_state_e {
  // The decision and its selected provider are present in every class trace.
  LOOM_KERNEL_CLASS_DECISION_ACCEPTED = 0,

  // A lexically conditional decision retained its generic source form.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_LEXICAL_CONDITION = 1,

  // An unprojectable decision retained its generic source form.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNPROJECTABLE_INPUT = 2,

  // A target-dependent decision retained its generic source form.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNRESOLVED_TARGET = 3,

  // Refining this decision would have exceeded the class limit.
  LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT = 4,
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
  // Previous trace node, or LOOM_KERNEL_CLASS_TRACE_ID_INVALID.
  loom_kernel_class_trace_id_t parent_trace_id;

  // Classifier decision ordinal.
  uint32_t decision_ordinal;

  // Selected provider action ordinal in the decision model.
  uint32_t action_ordinal;
} loom_kernel_class_trace_t;

// One live kernel class.
typedef struct loom_kernel_class_t {
  // Final trace node describing all accepted decisions for this class.
  loom_kernel_class_trace_id_t trace_id;

  // Number of launch sites assigned to this class.
  iree_host_size_t member_count;
} loom_kernel_class_t;

// Transient live classes collected for one command workload.
typedef struct loom_kernel_class_collection_t {
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
// walks no IR. |kernel_facts| and |expression_context| must describe the same
// kernel function and immutable module snapshot. The classifier borrows all
// supplied analyses and owns only storage allocated from |arena|.
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
// All collection storage is allocated from |arena| before site evaluation. The
// site loop performs no allocation, IR traversal, string work, hashing, or
// fact-table cloning. A decision can remain generic only when its kernel-source
// facts select a valid residual provider.
iree_status_t loom_kernel_class_classifier_collect(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_kernel_class_collection_t* out_collection);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_CLASSIFIER_H_
