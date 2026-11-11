// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Invocation-local classes of kernel launches under a shared specialization
// budget.

#ifndef LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_COLLECTION_H_
#define LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_COLLECTION_H_

#include "loom/transforms/kernel/kernel_class_classifier.h"
#include "loom/transforms/kernel/kernel_specialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Default maximum number of live classes collected for one kernel.
#define LOOM_KERNEL_CLASS_DEFAULT_LIMIT 128

// Default maximum affine terms transferred from ranged launch facts.
#define LOOM_KERNEL_CLASS_DEFAULT_RANGED_TERM_LIMIT 4

// Dense index into one collection's parent-linked trace table.
typedef uint32_t loom_kernel_class_trace_id_t;

// Invalid trace ID denoting the root with no accepted decisions.
#define LOOM_KERNEL_CLASS_TRACE_ID_INVALID UINT32_MAX

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

// One immutable compact plan and its already grouped nonempty launch slice.
typedef struct loom_kernel_class_group_t {
  // Immutable applicability and structural projections for this tuple.
  const loom_kernel_class_classifier_t* classifier;
  // Nonempty complete launch slice borrowing invocation facts.
  const loom_kernel_class_site_t* sites;
  // Number of launches in this group.
  iree_host_size_t site_count;
} loom_kernel_class_group_t;

// Collects disjoint groups under one shared class limit. All mandatory
// requirements reserve classes before any group spends capacity on optional
// refinement. Each output owns local class ordinals; the caller assigns its
// publication-wide ordinals after all groups succeed. A failure publishes no
// partial collection. Groups borrow immutable plans and caller facts.
iree_status_t loom_kernel_class_collect_groups(
    const loom_kernel_class_group_t* groups, uint32_t group_count,
    const loom_module_t* site_module,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena, loom_kernel_class_collection_t* collections);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_COLLECTION_H_
