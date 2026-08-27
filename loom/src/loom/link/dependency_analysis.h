// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Strict direct-library dependency analysis over a frozen link index.

#ifndef LOOM_LINK_DEPENDENCY_ANALYSIS_H_
#define LOOM_LINK_DEPENDENCY_ANALYSIS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/link/func_contract.h"
#include "loom/link/module_index.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kind of authored requirement crossing an INPUT-provider boundary.
typedef enum loom_link_dependency_requirement_kind_e {
  // An exact global function-like declaration or reference.
  LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL = 0,
  // An open template family demanded by template.apply.
  LOOM_LINK_DEPENDENCY_REQUIREMENT_TEMPLATE_FAMILY = 1,
} loom_link_dependency_requirement_kind_t;

// Relationship between a candidate and the INPUT providers being analyzed.
typedef enum loom_link_dependency_candidate_origin_e {
  // Candidate is a concrete definition in the jointly owned INPUT providers.
  LOOM_LINK_DEPENDENCY_CANDIDATE_INPUT = 0,
  // Candidate belongs to a declared direct library.
  LOOM_LINK_DEPENDENCY_CANDIDATE_DIRECT_LIBRARY = 1,
  // Candidate belongs only to the transitive library universe.
  LOOM_LINK_DEPENDENCY_CANDIDATE_TRANSITIVE_LIBRARY = 2,
} loom_link_dependency_candidate_origin_t;

// Strict ownership result for one authored requirement.
typedef enum loom_link_dependency_disposition_e {
  // One compatible INPUT definition satisfies the requirement locally.
  LOOM_LINK_DEPENDENCY_LOCAL = 0,
  // One direct library owns an exact contract, or one or more direct
  // libraries contribute providers for an open template family.
  LOOM_LINK_DEPENDENCY_DIRECT = 1,
  // No provider is currently available for an open template family. The
  // relocatable module may be specialized against providers in a later link.
  LOOM_LINK_DEPENDENCY_OPEN = 2,
  // Compatible candidates exist only in transitive libraries.
  LOOM_LINK_DEPENDENCY_TRANSITIVE_ONLY = 3,
  // No candidate exists in the indexed library universe.
  LOOM_LINK_DEPENDENCY_UNRESOLVED = 4,
  // Candidates exist at the nearest ownership tier but none are compatible.
  LOOM_LINK_DEPENDENCY_INCOMPATIBLE = 5,
  // Multiple compatible candidates exist where exact ownership must be unique.
  LOOM_LINK_DEPENDENCY_AMBIGUOUS = 6,
} loom_link_dependency_disposition_t;

// Returns true when |disposition| is valid for a relocatable library.
static inline bool loom_link_dependency_disposition_satisfied(
    loom_link_dependency_disposition_t disposition) {
  return disposition == LOOM_LINK_DEPENDENCY_LOCAL ||
         disposition == LOOM_LINK_DEPENDENCY_DIRECT ||
         disposition == LOOM_LINK_DEPENDENCY_OPEN;
}

// One exact symbol or template provider considered for a requirement.
typedef struct loom_link_dependency_candidate_t {
  // Index-wide candidate symbol ordinal.
  iree_host_size_t symbol_ordinal;
  // Provider owning symbol_ordinal.
  iree_host_size_t provider_ordinal;
  // Candidate relationship to the INPUT provider set.
  loom_link_dependency_candidate_origin_t origin;
  // True when the candidate satisfies the authored contract.
  bool compatible;
  // Exact incompatibility detail. Template providers either match their family
  // identity or are rejected while the index is built.
  loom_link_func_contract_mismatch_t mismatch;
} loom_link_dependency_candidate_t;

// One unique authored requirement and its strict ownership result.
typedef struct loom_link_dependency_requirement_t {
  // Requirement family.
  loom_link_dependency_requirement_kind_t kind;
  // Strict ownership result.
  loom_link_dependency_disposition_t disposition;
  // True when an INPUT provider exports this exact declaration as part of its
  // public library surface. Always false for template-family requirements.
  bool exported;
  // Number of authored dependency occurrences aggregated into this
  // requirement. An exported declaration may have zero occurrences.
  iree_host_size_t occurrence_count;
  // Index-wide source symbol owning the first occurrence, or INVALID for a
  // module-root occurrence.
  iree_host_size_t first_source_symbol_ordinal;
  // Root region slot on first_source_symbol_ordinal plus one, or zero for its
  // symbol contract or a module-root occurrence.
  uint8_t first_source_root_region_index_plus_one;
  // Accepted target interfaces unioned across exact-symbol occurrences.
  loom_symbol_interface_flags_t target_interfaces;
  // Required index identity.
  struct {
    // Exact target symbol ordinal, or INVALID for a template family.
    iree_host_size_t symbol_ordinal;
    // Template family ordinal, or INVALID for an exact symbol.
    loom_link_template_family_ordinal_t template_family_ordinal;
  } target;
  // Candidate slice in the enclosing analysis.
  struct {
    // First candidate ordinal.
    iree_host_size_t first;
    // Number of candidate entries.
    iree_host_size_t count;
  } candidates;
} loom_link_dependency_requirement_t;

// Direct-library ownership report allocated in a caller arena.
typedef struct loom_link_dependency_analysis_t {
  // Frozen index this analysis describes.
  const loom_link_module_index_t* index;
  // Authored exact requirements and template demands.
  struct {
    // Arena-owned requirement entries in stable identity order.
    const loom_link_dependency_requirement_t* values;
    // Number of requirement entries.
    iree_host_size_t count;
  } requirements;
  // Candidates grouped by requirement.
  struct {
    // Arena-owned candidate entries.
    const loom_link_dependency_candidate_t* values;
    // Number of candidate entries.
    iree_host_size_t count;
  } candidates;
  // Declared direct library providers.
  struct {
    // Caller-order provider ordinals copied into the analysis arena.
    const iree_host_size_t* values;
    // Number of direct providers.
    iree_host_size_t count;
  } direct_providers;
  // Direct providers participating in at least one requirement.
  struct {
    // Provider ordinals in direct-provider order.
    const iree_host_size_t* values;
    // Number of used direct providers.
    iree_host_size_t count;
  } used_direct_providers;
  // Direct providers with no authored exact or template-family use.
  struct {
    // Provider ordinals in direct-provider order.
    const iree_host_size_t* values;
    // Number of unused direct providers.
    iree_host_size_t count;
  } unused_direct_providers;
  // Total exact dependency occurrences scanned in INPUT providers. Exported
  // declarations without internal uses do not contribute to this count.
  iree_host_size_t exact_occurrence_count;
  // Total template.apply demand occurrences scanned in INPUT providers.
  iree_host_size_t template_demand_occurrence_count;
} loom_link_dependency_analysis_t;

// Inputs selecting direct ownership within the full indexed library universe.
typedef struct loom_link_dependency_analysis_options_t {
  // LIBRARY-provider ordinals declared as direct dependencies.
  const iree_host_size_t* direct_provider_ordinals;
  // Number of entries in direct_provider_ordinals.
  iree_host_size_t direct_provider_count;
} loom_link_dependency_analysis_options_t;

// Analyzes every dependency authored by INPUT providers in |index|.
//
// All LIBRARY providers participate in transitive candidate discovery. Only
// provider ordinals named by |options| may directly satisfy a requirement.
// Exact function contracts are projected from headers without reading bodies.
// Template families remain valid when no provider is present, allowing a
// relocatable library to defer architecture-specific specialization.
//
// Report storage is allocated from |arena| and remains valid until the arena is
// reset. |block_pool| and |allocator| back transient contract projection and
// retain no state after this call returns. Semantic dependency failures are
// reported as dispositions with OK status; statuses represent invalid options,
// allocation failures, and malformed source projection.
iree_status_t loom_link_dependency_analyze(
    const loom_link_module_index_t* index,
    const loom_link_dependency_analysis_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    iree_allocator_t allocator, loom_link_dependency_analysis_t* out_analysis);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_DEPENDENCY_ANALYSIS_H_
